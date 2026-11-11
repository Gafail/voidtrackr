/**
 * Build the API server into one file.
 *
 *   node scripts/build-api.mjs
 *
 * The API runs on the box, which has Node and nothing else. Rather than
 * shipping the whole repository and a development toolchain, the server is
 * bundled into one file.
 *
 * The bundle has no dependencies at all. The server describes the tool and
 * does not run it, so it needs no ONNX runtime, no TensorFlow.js and no
 * audio decoder — which means the box does not have to install anything,
 * and the box's memory is not a factor in what it can serve.
 */

import { build } from 'esbuild';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const OUT = join(ROOT, 'dist-api');

await build({
  entryPoints: [join(ROOT, 'src', 'api', 'index.ts')],
  outfile: join(OUT, 'server.js'),
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'esm',
  external: [],
  sourcemap: false,
  minify: false,
  legalComments: 'none',
  logLevel: 'info',
});

/**
 * The package.json the box runs from.
 *
 * It has no dependencies, because the bundle has none. It is here so that
 * Node treats the file as an ES module and so that the directory is
 * recognisable as the API.
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
  description: 'The voidtrackr HTTP API: a catalogue of the tool, not a converter.',
  license: root.license,
  main: 'server.js',
  scripts: { start: 'node server.js' },
  dependencies: {},
};

mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, 'package.json'), `${JSON.stringify(runtime, null, 2)}\n`);

console.log(`\nThe API is built into ${OUT}`);
console.log('It has no dependencies: the box installs nothing.');
