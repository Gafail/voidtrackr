/**
 * Fetch the weights of the default model.
 *
 *   node scripts/fetch-model.mjs
 *
 * Which model that is comes from `DEFAULT_MODEL` in `src/core/registry.ts`,
 * so there is one place to change when the default changes and nothing here
 * to keep in step with it. Each model has its own fetcher, because each one
 * comes from a different place and lands in a different place:
 *
 *   basic-pitch  public/model/       Spotify's own repository
 *   okarina      models/onnx/        an unofficial Harmonica reimplementation
 *   tsumugi      models/onnx/tsumugi/
 *   yourmt3      models/onnx/yourmt3/
 *   piano        models/onnx/piano/
 *
 * Pass a model name to fetch that one instead:
 *
 *   node scripts/fetch-model.mjs basic-pitch
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const HERE = dirname(fileURLToPath(import.meta.url));
const REGISTRY = resolve(HERE, '..', 'src', 'core', 'registry.ts');

/** Read the default model out of the registry. */
function defaultModel() {
  const source = readFileSync(REGISTRY, 'utf8');
  const match = source.match(/DEFAULT_MODEL:\s*ModelId\s*=\s*'([^']+)'/);
  if (!match) {
    throw new Error(
      `Could not find DEFAULT_MODEL in ${REGISTRY}. ` +
        'The default model is declared there; nothing else needs to change.',
    );
  }
  return match[1];
}

/** The fetcher for each model, and the arguments it takes. */
const FETCHERS = {
  'basic-pitch': { script: 'fetch-basic-pitch.mjs', args: [] },
  okarina: { script: 'fetch-okarina.mjs', args: [] },
  tsumugi: { script: 'fetch-onnx.mjs', args: ['tsumugi'] },
  yourmt3: { script: 'fetch-onnx.mjs', args: ['yourmt3'] },
  piano: { script: 'fetch-onnx.mjs', args: ['piano'] },
};

function main() {
  const requested = process.argv[2];
  const model = requested ?? defaultModel();
  const fetcher = FETCHERS[model];

  if (!fetcher) {
    console.error(`error: there is no fetcher for the model "${model}".`);
    console.error(`       Known models: ${Object.keys(FETCHERS).join(', ')}`);
    process.exit(1);
  }

  console.log(`Fetching the ${model} weights (the default model).\n`);

  const result = spawnSync(
    process.execPath,
    [resolve(HERE, fetcher.script), ...fetcher.args],
    { stdio: 'inherit' },
  );

  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }
  process.exit(result.status ?? 1);
}

main();
