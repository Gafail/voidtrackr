/**
 * Download the Okarina ONNX models.
 *
 * Okarina is an unofficial ONNX reimplementation of Harmonica, the model
 * BandLab's audio-to-MIDI service is built on. The Harmonica weights were
 * never released; these come from the reimplementation.
 *
 *   **Licence: CC BY-NC-SA 4.0. Non-commercial use only.**
 *
 * The list of files and their expected hashes comes from
 * `src/core/okarina.lock.json`, so there is exactly one place to change when
 * the models change.
 *
 *   node scripts/fetch-okarina.mjs
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
const LOCK_PATH = resolve(HERE, '..', 'src', 'core', 'okarina.lock.json');

const lock = JSON.parse(readFileSync(LOCK_PATH, 'utf8'));

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

    const url = `${lock.source}/${name}`;
    console.log(`fetching ${url}`);
    await download(url, destination);

    const digest = await sha256(destination);
    if (digest !== info.sha256) {
      throw new Error(
        `${name} checksum mismatch: expected ${info.sha256}, got ${digest}`,
      );
    }
    console.log(`ok       ${name}`);
  }

  await writeFile(
    join(OUTPUT_DIR, 'SOURCE.txt'),
    [
      `${lock.displayName}`,
      '',
      `Model id: ${lock.id}`,
      `Source: ${lock.source}`,
      `Homepage: ${lock.homepage}`,
      `Paper: ${lock.paper}`,
      `License: ${lock.license}`,
      `Note: ${lock.note}`,
      '',
      ...Object.entries(lock.files).map(
        ([name, info]) => `${name.padEnd(24)} sha256 ${info.sha256}`,
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
