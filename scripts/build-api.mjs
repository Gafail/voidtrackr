/**
 * Build the API server into one file.
 *
 *   node scripts/build-api.mjs
 *
 * The API runs on the box, which has Node and nothing else. Rather than
 * shipping the whole repository and a development toolchain, the server is
 * bundled into one file and the heavy native modules are left where they
 * are, to be installed on the box by npm.
 */

import { build } from 'esbuild';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const OUT = join(ROOT, 'dist-api');

/**
 * The packages that stay outside the bundle.
 *
 * They are left out because they carry native binaries, or because they are
 * already a runtime dependency of the project. Bundling them would mean
 * shipping megabytes of machine code that npm installs correctly anyway.
 */
const EXTERNAL = [
  'onnxruntime-node',
  '@tensorflow/tfjs',
  '@tensorflow/tfjs-node',
  '@tonejs/midi',
  'audio-decode',
];

await build({
  entryPoints: [join(ROOT, 'src', 'api', 'index.ts')],
  outfile: join(OUT, 'server.js'),
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'esm',
  external: EXTERNAL,
  sourcemap: false,
  minify: false,
  legalComments: 'none',
  logLevel: 'info',
});

/**
 * The package.json the box installs from.
 *
 * Only what the server needs at runtime. The versions match the root
 * package.json so the two cannot drift.
 */
const root = JSON.parse(
  await import('node:fs/promises').then((fs) =>
    fs.readFile(join(ROOT, 'package.json'), 'utf8'),
  ),
);

const runtime = {
  name: 'voidtrackr-api',
  version: root.version,
  private: true,
  type: 'module',
  description: 'The voidtrackr HTTP API.',
  license: root.license,
  main: 'server.js',
  scripts: { start: 'node server.js' },
  dependencies: {
    '@tensorflow/tfjs': root.dependencies['@tensorflow/tfjs'],
    '@tonejs/midi': root.dependencies['@tonejs/midi'],
    'audio-decode': root.devDependencies['audio-decode'],
    'onnxruntime-node': root.devDependencies['onnxruntime-node'],
  },
};

mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, 'package.json'), `${JSON.stringify(runtime, null, 2)}\n`);

console.log(`\nThe API is built into ${OUT}`);
console.log('It installs these on the box:');
for (const name of Object.keys(runtime.dependencies)) {
  console.log(`  ${name} ${runtime.dependencies[name]}`);
}
