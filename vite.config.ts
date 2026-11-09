import { copyFileSync, createReadStream, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import react from '@vitejs/plugin-react';
import { defineConfig, Plugin } from 'vite';

/**
 * Serve the ONNX runtime's WebAssembly and the model weights.
 *
 * Both are large and both live outside the source tree: the runtime comes
 * from `node_modules` and the weights are fetched with `npm run model:okarina`
 * and its siblings. Rather than copying them into the repository, they are
 * served straight from where they are, and copied into the build only when a
 * build is made.
 */

function onnxAssets(): Plugin {
  const runtimeDist = fileURLToPath(
    new URL('./node_modules/onnxruntime-web/dist', import.meta.url),
  );
  const weights = fileURLToPath(new URL('./models/onnx', import.meta.url));

  const runtimeFiles = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];

  /** The weight directories, each with the files the browser fetches. */
  const weightSets: Record<string, string[]> = {
    '': ['cqt156.onnx', 'okarina_s.onnx', 'okarina_m.onnx'],
    tsumugi: [
      'tsumugi_other_v1_5.pos32i_i32.onnx',
      'tsumugi_other_v1_5.pos32i_i32.onnx.data',
      'tsumugi_other_v1_5.pos32i_i32.json',
      'tsumugi_other_v1_5.heads.bin',
      'tsumugi_other_v1_5.heads.json',
    ],
    yourmt3: ['encoder.onnx', 'decoder_step.onnx', 'vocab.json'],
    piano: ['piano_kong.onnx', 'piano_kong.onnx.data'],
  };

  /** Every file the browser is allowed to ask for, by its path under the root. */
  const allowed = new Set<string>();
  for (const [set, names] of Object.entries(weightSets)) {
    for (const name of names) allowed.add(set ? `${set}/${name}` : name);
  }

  return {
    name: 'onnx-assets',

    configureServer(server) {
      // The WebAssembly runtime.
      server.middlewares.use('/ort', (req, res, next) => {
        const name = basename((req.url ?? '').split('?')[0]);
        const file = join(runtimeDist, name);
        if (!runtimeFiles.includes(name) || !existsSync(file)) return next();
        res.setHeader('Content-Type', name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
        res.setHeader('Cache-Control', 'max-age=3600');
        createReadStream(file).pipe(res);
      });

      // The model weights.
      server.middlewares.use('/model-onnx', (req, res, next) => {
        const path = (req.url ?? '').split('?')[0].replace(/^\/+/, '');
        if (!allowed.has(path)) return next();
        const file = join(weights, path);
        if (!existsSync(file)) return next();
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Cache-Control', 'max-age=3600');
        createReadStream(file).pipe(res);
      });
    },

    generateBundle() {
      for (const name of runtimeFiles) {
        const file = join(runtimeDist, name);
        if (!existsSync(file)) continue;
        this.emitFile({
          type: 'asset',
          fileName: `ort/${name}`,
          source: readFileSync(file),
        });
      }
    },

    /**
     * Copy the model weights into the build.
     *
     * They are copied on disk rather than emitted as assets, because they run
     * to hundreds of megabytes and holding them all in the build's memory
     * would be slow at best and fatal at worst.
     */
    async writeBundle(options) {
      const outDir = options.dir ?? 'dist';

      for (const [set, names] of Object.entries(weightSets)) {
        for (const name of names) {
          const source = join(weights, set ? join(set, name) : name);
          if (!existsSync(source)) continue;
          const target = join(outDir, 'model-onnx', set ? join(set, name) : name);
          mkdirSync(dirname(target), { recursive: true });
          copyFileSync(source, target);
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), onnxAssets()],
  resolve: {
    alias: {
      '@core': fileURLToPath(new URL('./src/core', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    host: '127.0.0.1',
  },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 3000,
    rollupOptions: {
      output: {
        manualChunks: {
          tensorflow: ['@tensorflow/tfjs'],
        },
      },
    },
  },
});
