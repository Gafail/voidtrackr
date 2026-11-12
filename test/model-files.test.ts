/**
 * The files the browser fetches for each model.
 *
 * The app fetches a model's weights before the runtime does, so that the
 * progress bar can count the bytes while they come down. That only works if
 * the list is the same list the runtime would have fetched: a file that is
 * missing from it would be fetched by the runtime with nothing to show for
 * it, and a file on it that the lock does not record would fail the
 * download.
 */

import { stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';

import { MODEL_FILES } from '../src/app/model-files';
import { MODEL_IDS } from '../src/core/registry';
import modelLock from '../src/core/model.lock.json';
import okarinaLock from '../src/core/okarina.lock.json';
import pianoLock from '../src/core/piano.lock.json';
import tsumugiLock from '../src/core/tsumugi.lock.json';
import yourmt3Lock from '../src/core/yourmt3.lock.json';

/** The lock file each model is pinned by. */
const LOCKS: Record<string, { files: Record<string, { sha256: string; bytes?: number }> }> = {
  'basic-pitch': modelLock,
  tsumugi: tsumugiLock,
  yourmt3: yourmt3Lock,
  piano: pianoLock,
  okarina: okarinaLock,
};

/**
 * The files a lock records that the browser never asks for.
 *
 * A lock pins everything a model is made of, not only what the app loads.
 * These are the ones it does not load, and they are the only ones it is
 * allowed to leave out.
 */
const NOT_FETCHED: Record<string, string[]> = {
  // The 's' size of Okarina. The app always loads the 'm' size.
  okarina: ['okarina_s.onnx'],
  // The export's own description of the graph. The app reads the settings
  // it needs from the heads file instead.
  tsumugi: ['tsumugi/tsumugi_other_v1_5.pos32i_i32.json'],
};

/** Where a model's files are on disk. */
function onDisk(id: string, path: string): string {
  return id === 'basic-pitch' ? resolve('public/model', path) : resolve('models/onnx', path);
}

test('the browser fetches the files the lock records', () => {
  for (const id of MODEL_IDS) {
    const lock = LOCKS[id];
    const fetched = MODEL_FILES[id].files.map((file) => file.path);

    for (const path of fetched) {
      assert.ok(lock.files[path], `${id}: the lock does not record ${path}`);
    }

    const missing = Object.keys(lock.files).filter(
      (path) => !fetched.includes(path) && !(NOT_FETCHED[id] ?? []).includes(path),
    );
    assert.deepEqual(missing, [], `${id}: the browser does not fetch these files`);
  }
});

test('every file has a recorded size, so the bar has a total to count towards', () => {
  for (const id of MODEL_IDS) {
    const files = MODEL_FILES[id].files;
    assert.ok(files.length > 0, `${id}: the browser fetches nothing`);

    for (const file of files) {
      assert.ok(file.bytes > 0, `${id}: ${file.path} has no size`);
    }

    const total = files.reduce((sum, file) => sum + file.bytes, 0);
    assert.equal(MODEL_FILES[id].totalBytes, total, `${id}: the total is wrong`);
  }
});

test('the recorded size is the size of the file', async () => {
  for (const id of MODEL_IDS) {
    const lock = LOCKS[id];

    for (const file of MODEL_FILES[id].files) {
      const path = onDisk(id, file.path);
      let size: number;
      try {
        size = (await stat(path)).size;
      } catch {
        // The weights are fetched with `npm run model`, so a checkout
        // without them is normal.
        continue;
      }

      assert.equal(size, file.bytes, `${id}: ${file.path} is a different size on disk`);
      assert.equal(size, lock.files[file.path].bytes, `${id}: ${file.path} is a different size`);
    }
  }
});
