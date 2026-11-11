/**
 * The entry point of the API server.
 *
 * The settings come from the environment, so the same build runs on the
 * developer's machine and on the box:
 *
 *   VOIDTRACKR_PORT   the port to listen on (default 8787)
 *   VOIDTRACKR_HOST   the address to listen on (default 127.0.0.1)
 *
 * There is very little to set, because the server does very little. It
 * answers questions about the tool; it does not run the tool, and it never
 * receives a recording.
 */

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
  };
}

async function main(): Promise<void> {
  const options = optionsFromEnv();
  const server = createApiServer();

  server.listen(options.port, options.host, () => {
    console.log(
      `[voidtrackr] the API is listening on http://${options.host}:${options.port}/api`,
    );
    console.log('[voidtrackr] it describes the tool; it does not convert anything.');
  });

  const stop = () => {
    server.close(() => process.exit(0));
  };
  process.on('SIGTERM', stop);
  process.on('SIGINT', stop);
}

await main();
