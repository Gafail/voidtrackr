/**
 * Download the ONNX model weights.
 *
 * The list of files and their expected hashes comes from the lock file in
 * `src/core/`, so there is exactly one place to change when a model changes.
 * A file that is already present and matches its hash is left alone, so
 * running this twice costs nothing.
 *
 *   node scripts/fetch-onnx.mjs tsumugi
 *   node scripts/fetch-onnx.mjs yourmt3
 *   node scripts/fetch-onnx.mjs piano
 *
 * The weights are not in the repository — they are fetched, verified and
 * ignored by git. `models/onnx/SOURCE.txt` records where they came from.
 */

import { createWriteStream } from 'node:fs';
import { mkdir, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline as streamPipeline } from 'node:stream/promises';
import { readFileSync } from 'node:fs';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = resolve(HERE, '..', 'models', 'onnx');

/** The lock file each model is pinned by. */
const LOCKS = {
  tsumugi: 'tsumugi.lock.json',
  yourmt3: 'yourmt3.lock.json',
  piano: 'piano.lock.json',
};

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function download(url, destination) {
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`GET ${url} failed with ${response.status}`);
  }
  await mkdir(dirname(destination), { recursive: true });
  await streamPipeline(Readable.fromWeb(response.body), createWriteStream(destination));
}

async function sha256(path) {
  const { createHash } = await import('node:crypto');
  const { readFile } = await import('node:fs/promises');
  return createHash('sha256').update(await readFile(path)).digest('hex');
}

async function main() {
  const which = process.argv[2];
  const lockName = LOCKS[which];
  if (!lockName) {
    throw new Error(
      `Unknown model "${which ?? ''}". Choose one of: ${Object.keys(LOCKS).join(', ')}.`,
    );
  }

  const lockPath = resolve(HERE, '..', 'src', 'core', lockName);
  const lock = JSON.parse(readFileSync(lockPath, 'utf8'));

  await mkdir(OUTPUT_DIR, { recursive: true });

  for (const [name, info] of Object.entries(lock.files)) {
    const destination = join(OUTPUT_DIR, name);

    if (await exists(destination)) {
      const digest = await sha256(destination);
      if (digest === info.sha256) {
        console.log(`ok       ${name} (already present)`);
        continue;
      }
      console.log(`stale    ${name} (checksum mismatch, refetching)`);
    }

    const url = `${lock.source}/${name.split('/').pop()}`;
    console.log(`fetching ${url}`);
    await download(url, destination);

    const digest = await sha256(destination);
    if (digest !== info.sha256) {
      throw new Error(`${name} checksum mismatch: expected ${info.sha256}, got ${digest}`);
    }
    console.log(`ok       ${name}`);
  }

  await writeFile(
    join(OUTPUT_DIR, which, 'SOURCE.txt'),
    [
      lock.displayName,
      '',
      `Model id: ${lock.id}`,
      `Source: ${lock.source}`,
      `Homepage: ${lock.homepage}`,
      `Paper: ${lock.paper}`,
      `License: ${lock.license}`,
      `License URL: ${lock.licenseUrl}`,
      `Note: ${lock.note}`,
      '',
      ...Object.entries(lock.files).map(
        ([name, info]) => `${name.padEnd(44)} sha256 ${info.sha256}`,
      ),
      '',
    ].join('\n'),
    'utf8',
  );

  console.log(`\nModel ready in ${OUTPUT_DIR}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
