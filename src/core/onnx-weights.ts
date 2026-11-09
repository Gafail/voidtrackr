/**
 * Checking a model's weight files against its lock file.
 *
 * Every ONNX model in this project ships as a set of files — a graph, often a
 * sidecar of external weights, and sometimes some metadata. They are pinned
 * by SHA-256 in a lock file, and checked before the model runs.
 *
 * The check is the same in the browser and on the command line; only the way
 * the bytes are read differs. So the reading is passed in, and everything
 * else — the hash, the comparison, the fingerprint — lives here.
 */

import { sha256Hex } from './model';

/** The part of a lock file this needs. */
export interface OnnxLock {
  files: Record<string, { sha256: string }>;
}

/** A set of files that make up one model. */
export interface OnnxWeightSet {
  /** Directory under the weights root. Empty means the root itself. */
  dir: string;
  /** File names inside that directory. */
  files: string[];
}

/**
 * Check a model's files and return its fingerprint.
 *
 * The fingerprint is the hash of the file hashes, joined in order. It goes
 * into every exported MIDI file, so a file can always be traced back to the
 * exact weights that made it.
 */
export async function verifyOnnxWeights(
  read: (path: string) => Promise<ArrayBuffer>,
  lock: OnnxLock,
  set: OnnxWeightSet,
  /** The command that fetches the weights, named in the error message. */
  fetchCommand: string,
): Promise<string> {
  const digests: string[] = [];

  for (const name of set.files) {
    const path = set.dir ? `${set.dir}/${name}` : name;
    const expected = lock.files[path]?.sha256;
    if (!expected) {
      throw new Error(
        `The lock file does not record a hash for ${path}. ` +
          'Add it to the model lock file before using this model.',
      );
    }

    const bytes = await read(path);
    const digest = await sha256Hex(bytes);
    if (digest !== expected) {
      throw new Error(
        `The model's ${path} does not match the pinned version.\n` +
          `  expected sha256 ${expected}\n` +
          `  got      sha256 ${digest}\n` +
          `Run \`${fetchCommand}\` to fetch the pinned weights.`,
      );
    }
    digests.push(digest);
  }

  return sha256Hex(digests.join(''));
}
