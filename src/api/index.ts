/**
 * The entry point of the API server.
 *
 * The settings come from the environment, so the same build runs on the
 * developer's machine and on the box:
 *
 *   VOIDTRACKR_PORT           the port to listen on (default 8787)
 *   VOIDTRACKR_HOST           the address to listen on (default 127.0.0.1)
 *   VOIDTRACKR_MODEL_DIR      where the ONNX weights are
 *   VOIDTRACKR_MODEL_PATH     where the Basic Pitch model.json is
 *   VOIDTRACKR_API_TOKEN      a token callers have to present, if there is one
 *   VOIDTRACKR_MAX_UPLOAD     the largest upload, in megabytes (default 100)
 *   VOIDTRACKR_CONCURRENCY    how many conversions at once (default 1)
 */

import { resolve } from 'node:path';

import { installTfjsNodeBackend } from './backend';
import { createApiServer, ServerOptions } from './server';

/** Read a number from the environment, with a fallback. */
function numberFromEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    console.error(`[voidtrackr] ${name} is not a number: "${raw}". Using ${fallback}.`);
    return fallback;
  }
  return value;
}

function optionsFromEnv(): ServerOptions {
  return {
    port: numberFromEnv('VOIDTRACKR_PORT', 8787),
    host: process.env.VOIDTRACKR_HOST ?? '127.0.0.1',
    modelDir:
      process.env.VOIDTRACKR_MODEL_DIR ?? resolve(process.cwd(), 'models', 'onnx'),
    modelPath:
      process.env.VOIDTRACKR_MODEL_PATH ??
      resolve(process.cwd(), 'public', 'model', 'model.json'),
    maxUploadBytes: numberFromEnv('VOIDTRACKR_MAX_UPLOAD', 100) * 1024 * 1024,
    token: process.env.VOIDTRACKR_API_TOKEN,
    concurrency: numberFromEnv('VOIDTRACKR_CONCURRENCY', 1),
  };
}

async function main(): Promise<void> {
  const options = optionsFromEnv();

  // The native TensorFlow.js backend is much faster, but it is optional: the
  // box may not have it installed, and Basic Pitch then runs on the
  // pure-JavaScript backend instead.
  const native = await installTfjsNodeBackend();

  const server = createApiServer(options);

  server.listen(options.port, options.host, () => {
    console.log(
      `[voidtrackr] the API is listening on http://${options.host}:${options.port}/api`,
    );
    console.log(`[voidtrackr] weights: ${options.modelDir}`);
    console.log(`[voidtrackr] native TensorFlow.js backend: ${native ? 'yes' : 'no'}`);
  });

  const stop = () => {
    server.close(() => process.exit(0));
  };
  process.on('SIGTERM', stop);
  process.on('SIGINT', stop);
}

await main();
