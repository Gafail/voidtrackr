/**
 * The native TensorFlow.js backend, when it is installed.
 *
 * Basic Pitch needs TensorFlow.js. The pure-JavaScript backend works
 * everywhere, but the native one is several times faster, and the command
 * line has always used it.
 *
 * Its published build has a packaging bug: it requires Node's own `util`
 * module where it meant to require TensorFlow.js's utilities, then calls a
 * function that does not exist there. Node's `util` has no
 * `isNullOrUndefined`, so the native backend throws the moment it reads a
 * tensor.
 *
 * `require("util")` hands back a singleton, so defining the missing helper
 * on it is enough to make the backend work. It has to happen before the
 * backend is imported, which is why this is a function rather than a side
 * effect at module scope.
 *
 * The web server may not have the native backend installed — it is a large
 * native module and the box it runs on is small — so the import is allowed
 * to fail. Basic Pitch then runs on the pure-JavaScript backend, which is
 * slower but produces the same notes.
 */

import nodeUtil from 'node:util';

const mutableUtil = nodeUtil as typeof nodeUtil & {
  isNullOrUndefined?: (value: unknown) => boolean;
};

/** Whether the native backend was loaded. */
let installed: boolean | undefined;

/**
 * Install the native TensorFlow.js backend if it is there.
 *
 * @returns true when the native backend is in use.
 */
export async function installTfjsNodeBackend(): Promise<boolean> {
  if (installed !== undefined) return installed;

  if (typeof mutableUtil.isNullOrUndefined !== 'function') {
    mutableUtil.isNullOrUndefined = (value: unknown) =>
      value === null || value === undefined;
  }

  try {
    await import('@tensorflow/tfjs-node');
    installed = true;
  } catch {
    // The native backend is not installed. The pure-JavaScript one will do.
    installed = false;
  }

  return installed;
}
