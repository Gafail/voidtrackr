/**
 * The files each model is made of, and how big they are.
 *
 * A model is exactly the files its lock file pins, so the lock is the
 * authority on the names and on the sizes. They are listed here as well
 * because the browser has to fetch them before the runtime asks for them:
 * the runtime fetches the weights itself and the browser says nothing at
 * all while it does, which is why the progress bar used to sit on "Loading
 * model" until the model was ready.
 *
 * The names come from the constants beside the models, so a file that is
 * renamed cannot be renamed in one place and missed in another.
 */

import modelLock from '@core/model.lock.json';
import { OKARINA_FILES } from '@core/okarina';
import okarinaLock from '@core/okarina.lock.json';
import { PIANO_DIR, PIANO_EXTERNAL_DATA, PIANO_FILES } from '@core/piano';
import pianoLock from '@core/piano.lock.json';
import { ModelId } from '@core/registry';
import { TSUMUGI_DIR, TSUMUGI_EXTERNAL_DATA, TSUMUGI_FILES } from '@core/tsumugi';
import tsumugiLock from '@core/tsumugi.lock.json';
import { YOURMT3_DIR, YOURMT3_FILES } from '@core/yourmt3';
import yourmt3Lock from '@core/yourmt3.lock.json';

/** Where the ONNX weights are served from. */
export const ONNX_ROOT = '/model-onnx/';

/** Where Basic Pitch's weights are served from. */
export const BASIC_PITCH_ROOT = '/model/';

/** The part of a lock file this needs. */
export interface LockFiles {
  files: Record<string, { sha256: string; bytes?: number }>;
}

/** One file of a model. */
export interface ModelFile {
  /** The path, relative to the model's root. */
  path: string;
  /** The size in bytes. Zero when the lock does not record one. */
  bytes: number;
}

/** The files of one model. */
export interface ModelFiles {
  /** The directory the paths are relative to. */
  root: string;
  /** Every file the browser fetches for this model. */
  files: ModelFile[];
  /** How many bytes they come to in all. */
  totalBytes: number;
  /** The command that fetches the weights, named in the error message. */
  fetchCommand: string;
}

/** The lock file each model is pinned by. */
const LOCKS: Record<ModelId, LockFiles> = {
  'basic-pitch': modelLock,
  tsumugi: tsumugiLock,
  yourmt3: yourmt3Lock,
  piano: pianoLock,
  okarina: okarinaLock,
};

/**
 * The files of each model, by the path the lock file records them under.
 *
 * These are every file the browser fetches for that model: the app reads
 * them to check them against the lock, and the runtime reads them again to
 * build the session. A file that is missing from this list is one the
 * runtime would fetch with nothing to show for it, so the list has to be
 * complete — which is what the test in `test/model-files.test.ts` checks.
 */
const MODEL_PATHS: Record<ModelId, string[]> = {
  // Basic Pitch's shard is named in the manifest, and the lock records it.
  'basic-pitch': Object.keys(modelLock.files),
  tsumugi: [
    `${TSUMUGI_DIR}/${TSUMUGI_FILES.graph}`,
    `${TSUMUGI_DIR}/${TSUMUGI_EXTERNAL_DATA}`,
    `${TSUMUGI_DIR}/${TSUMUGI_FILES.heads}`,
    `${TSUMUGI_DIR}/${TSUMUGI_FILES.headsJson}`,
  ],
  yourmt3: [
    `${YOURMT3_DIR}/${YOURMT3_FILES.encoder}`,
    `${YOURMT3_DIR}/${YOURMT3_FILES.decoder}`,
    `${YOURMT3_DIR}/${YOURMT3_FILES.vocab}`,
  ],
  piano: [`${PIANO_DIR}/${PIANO_FILES.graph}`, `${PIANO_DIR}/${PIANO_EXTERNAL_DATA}`],
  // Okarina's files sit at the root of the tree rather than in a folder.
  okarina: [OKARINA_FILES.frontEnd, OKARINA_FILES.m],
};

/** Build the file list of one model, with the sizes the lock records. */
function modelFiles(id: ModelId, root: string, fetchCommand: string): ModelFiles {
  const lock = LOCKS[id];
  const files = MODEL_PATHS[id].map((path) => ({
    path,
    bytes: lock.files[path]?.bytes ?? 0,
  }));

  return {
    root,
    files,
    totalBytes: files.reduce((sum, file) => sum + file.bytes, 0),
    fetchCommand,
  };
}

/** Every model, and the files the browser fetches for it. */
export const MODEL_FILES: Record<ModelId, ModelFiles> = {
  'basic-pitch': modelFiles('basic-pitch', BASIC_PITCH_ROOT, 'npm run model:basic-pitch'),
  tsumugi: modelFiles('tsumugi', ONNX_ROOT, 'npm run model:tsumugi'),
  yourmt3: modelFiles('yourmt3', ONNX_ROOT, 'npm run model:yourmt3'),
  piano: modelFiles('piano', ONNX_ROOT, 'npm run model:piano'),
  okarina: modelFiles('okarina', ONNX_ROOT, 'npm run model:okarina'),
};

/** The address of one of a model's files. */
export function modelFileUrl(root: string, path: string, base: string): string {
  return new URL(root + path, base).href;
}
