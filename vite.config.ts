import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import react from '@vitejs/plugin-react';
import { defineConfig, Plugin } from 'vite';

/**
 * Serve the ONNX runtime's WebAssembly and the Okarina weights.
 *
 * Both are large and both live outside the source tree: the runtime comes
 * from `node_modules` and the weights are fetched with
 * `npm run model:okarina`. Rather than copying them into the repository,
 * they are served straight from where they are, and copied into the build
 * only when a build is made.
 */
function onnxAssets(): Plugin {
  const runtimeDist = fileURLToPath(
    new URL('./node_modules/onnxruntime-web/dist', import.meta.url),
  );
  const weights = fileURLToPath(new URL('./models/onnx', import.meta.url));

  const runtimeFiles = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];

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
        const name = basename((req.url ?? '').split('?')[0]);
        const file = join(weights, name);
        if (!name.endsWith('.onnx') || !existsSync(file)) return next();
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

      if (!existsSync(weights)) return;
      for (const name of ['cqt156.onnx', 'okarina_s.onnx', 'okarina_m.onnx']) {
        const file = join(weights, name);
        if (!existsSync(file)) continue;
        this.emitFile({
          type: 'asset',
          fileName: `model-onnx/${name}`,
          source: readFileSync(file),
        });
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
