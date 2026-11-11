/**
 * Loading a model from disk.
 *
 * The browser fetches its weights over HTTP; Node reads them off disk.
 * This is the Node half, and it is used by the tests and by the scripts
 * that measure what a model costs, so that what they measure is what the
 * browser would do.
 *
 * Every model is checked against its lock file before it runs. If the
 * weights do not match, the output would differ from every previous run, so
 * it is better to stop than to produce a different file quietly.
 */

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { loadPinnedModel } from './basic-pitch';
import { ModelLock, ModelSource, TranscriptionModel } from './model';
import modelLock from './model.lock.json';
import { OkarinaModel, OkarinaSize, OKARINA_FILES } from './okarina';
import okarinaLock from './okarina.lock.json';
import { nodeOnnxRunner } from './onnx-node';
import { verifyOnnxWeights } from './onnx-weights';
import { PianoTranscriptionModel, PIANO_DIR, PIANO_FILES } from './piano';
import pianoLock from './piano.lock.json';
import { TsumugiModel, TSUMUGI_DIR, TSUMUGI_FILES } from './tsumugi';
import tsumugiLock from './tsumugi.lock.json';
import { YourMt3Model, YOURMT3_DIR, YOURMT3_FILES } from './yourmt3';
import yourmt3Lock from './yourmt3.lock.json';
import { ModelId } from './registry';

/** Where the weights are. */
export interface ModelLocation {
  /** Directory holding the ONNX weights. */
  modelDir: string;
  /** Path to a Basic Pitch model.json. */
  modelPath: string;
}

/** Read one of the ONNX weight files, relative to the weights root. */
async function readOnnxFile(root: string, path: string): Promise<ArrayBuffer> {
  const bytes = await readFile(join(root, path));
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
}

/** Check a model's weights against its lock file. */
async function verifyOnnxModel(
  root: string,
  lock: unknown,
  directory: string,
  files: string[],
  fetchCommand: string,
): Promise<string> {
  return verifyOnnxWeights(
    async (path) => {
      try {
        return await readOnnxFile(root, path);
      } catch (error) {
        // The weights are not in the repository, so a missing file is the
        // expected case on a fresh checkout. Say which command fixes it.
        const reason = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Could not read the model's ${path} (${reason}). ` +
            `Run \`${fetchCommand}\` to fetch the weights.`,
        );
      }
    },
    lock as { files: Record<string, { sha256: string }> },
    { dir: directory, files },
    fetchCommand,
  );
}

/** Check the Okarina weights against the lock file. */
async function verifyOkarinaWeights(
  directory: string,
  size: OkarinaSize,
): Promise<string> {
  const lock = okarinaLock as unknown as {
    files: Record<string, { sha256: string }>;
  };
  const digests: string[] = [];

  for (const name of [OKARINA_FILES.frontEnd, OKARINA_FILES[size]]) {
    const expected = lock.files[name]?.sha256;
    if (!expected) {
      throw new Error(
        `The lock file does not record a hash for ${name}. ` +
          'Add it to src/core/okarina.lock.json before using this model.',
      );
    }
    const bytes = await readFile(join(directory, name));
    const digest = createHash('sha256').update(bytes).digest('hex');
    if (digest !== expected) {
      throw new Error(
        `The model's ${name} does not match the pinned version.\n` +
          `  expected sha256 ${expected}\n` +
          `  got      sha256 ${digest}\n` +
          'Run `npm run model:okarina` to fetch the pinned weights.',
      );
    }
    digests.push(digest);
  }

  return createHash('sha256').update(digests.join('')).digest('hex');
}

/**
 * A fetch that reads `file://` URLs off the disk.
 *
 * TensorFlow.js fetches the model's manifest and its weight shards itself,
 * and Node's `fetch` does not understand the `file://` scheme. The native
 * TensorFlow.js backend patches that, but the native backend is a large
 * binary and the box the API runs on does not have it — so this stands in.
 */
export const fileFetch: typeof fetch = async (input, init) => {
  const url =
    typeof input === 'string'
      ? input
      : input instanceof URL
        ? input.href
        : (input as Request).url;

  if (!url.startsWith('file://')) return fetch(input as RequestInfo, init);

  const bytes = await readFile(fileURLToPath(url));
  return new Response(bytes, {
    status: 200,
    headers: { 'Content-Type': 'application/octet-stream' },
  });
};

/**
 * Read the model from disk.
 *
 * Node's `fetch` does not understand `file://` URLs, so the files are read
 * here and the URL is handed to TensorFlow.js separately.
 *
 * The fetch is brought along too, because TensorFlow.js fetches the model's
 * manifest and its weight shards itself.
 */
function nodeModelSource(modelPath: string): ModelSource {
  const directory = dirname(modelPath);

  return {
    url: pathToFileURL(modelPath).href,
    fetch: fileFetch,
    read: async (name: string) => {
      const bytes = await readFile(join(directory, name));
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ) as ArrayBuffer;
    },
  };
}

/**
 * Load the model the caller asked for.
 *
 * The weights are verified against the lock file first, so a model that has
 * been swapped out or half-downloaded is refused rather than used.
 */
export async function loadModel(
  model: ModelId,
  location: ModelLocation,
  okarinaSize: OkarinaSize = 'm',
): Promise<TranscriptionModel> {
  const modelDir = location.modelDir;

  if (model === 'okarina') {
    const fingerprint = await verifyOkarinaWeights(modelDir, okarinaSize);
    return OkarinaModel.load(nodeOnnxRunner, modelDir, okarinaSize, fingerprint);
  }

  if (model === 'tsumugi') {
    const directory = join(modelDir, TSUMUGI_DIR);
    const fingerprint = await verifyOnnxModel(
      modelDir,
      tsumugiLock,
      TSUMUGI_DIR,
      [TSUMUGI_FILES.graph, TSUMUGI_FILES.heads, TSUMUGI_FILES.headsJson],
      'npm run model:tsumugi',
    );
    return TsumugiModel.load(nodeOnnxRunner, directory, fingerprint);
  }

  if (model === 'yourmt3') {
    const directory = join(modelDir, YOURMT3_DIR);
    const fingerprint = await verifyOnnxModel(
      modelDir,
      yourmt3Lock,
      YOURMT3_DIR,
      [YOURMT3_FILES.encoder, YOURMT3_FILES.decoder, YOURMT3_FILES.vocab],
      'npm run model:yourmt3',
    );
    return YourMt3Model.load(nodeOnnxRunner, directory, fingerprint);
  }

  if (model === 'piano') {
    const directory = join(modelDir, PIANO_DIR);
    const fingerprint = await verifyOnnxModel(
      modelDir,
      pianoLock,
      PIANO_DIR,
      [PIANO_FILES.graph],
      'npm run model:piano',
    );
    return PianoTranscriptionModel.load(nodeOnnxRunner, directory, fingerprint);
  }

  return loadPinnedModel(nodeModelSource(location.modelPath), modelLock as ModelLock);
}

/** Where the weights live, when nothing says otherwise. */
export function defaultModelLocation(): ModelLocation {
  return {
    modelDir: resolve(process.cwd(), 'models', 'onnx'),
    modelPath: resolve(process.cwd(), 'public', 'model', 'model.json'),
  };
}
