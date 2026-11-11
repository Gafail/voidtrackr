/**
 * The HTTP API.
 *
 * The site is static and the browser does the work. This is the other half
 * of that: a machine-readable description of what the tool can do, so that
 * a script can find out what to ask for without reading the source.
 *
 *   GET http://192.168.100.109/api/models
 *
 * What it does not do is take a recording. There is no endpoint that accepts
 * one, and the server never reads a request body at all. That is the whole
 * promise of this tool: a recording goes in and a MIDI file comes out, on
 * the machine that holds the recording, and nothing is uploaded. A server
 * that accepted the audio would break it, so it does not have one.
 *
 * The conversion happens wherever the recording already is — in the
 * browser, or on the command line — and both read the model weights from
 * here, the same way the browser does.
 */

import { createServer, IncomingMessage, Server, ServerResponse } from 'node:http';

import { DEFAULT_MODEL, listModels, MODEL_IDS, MODELS } from '../core/registry';
import { PARAMETERS } from '../core/parameters';
import { PIPELINE_VERSION } from '../core/version';

import { ApiError } from './errors';
import { openApiDocument } from './openapi';

/** How the server is set up. */
export interface ServerOptions {
  /** The port to listen on. */
  port: number;
  /** The address to listen on. 127.0.0.1 keeps it off the network. */
  host: string;
}

/** Send JSON. */
function sendJson(
  response: ServerResponse,
  status: number,
  value: unknown,
  headers: Record<string, string> = {},
): void {
  const body = JSON.stringify(value, null, 2);
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store',
    ...headers,
  });
  response.end(body);
}

/** Send an error, in the shape every error comes back as. */
function sendError(response: ServerResponse, error: unknown): void {
  const apiError =
    error instanceof ApiError ? error : ApiError.internal('Something went wrong.');

  if (!(error instanceof ApiError)) {
    console.error('[voidtrackr] ', error);
  }

  const body = JSON.stringify(apiError.toBody(), null, 2);
  response.writeHead(apiError.status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store',
  });
  response.end(body);
}

/**
 * Start the API.
 *
 * The returned server is not listening yet; `listen()` is separate so that
 * a test can choose the port.
 */
export function createApiServer(): Server {
  const startedAt = Date.now();

  const handle = async (
    request: IncomingMessage,
    response: ServerResponse,
  ): Promise<void> => {
    const url = new URL(request.url ?? '/', 'http://localhost');
    const path = url.pathname.replace(/\/+$/, '') || '/';
    const method = request.method ?? 'GET';

    try {
      if (path === '/' || path === '/api') {
        if (method !== 'GET' && method !== 'HEAD') {
          throw ApiError.methodNotAllowed(method, ['GET', 'HEAD']);
        }
        const accept = request.headers.accept ?? '';
        if (accept.includes('text/html')) {
          const body = apiHomePage();
          response.writeHead(200, {
            'Content-Type': 'text/html; charset=utf-8',
            'Content-Length': Buffer.byteLength(body),
            'Cache-Control': 'no-store',
          });
          response.end(method === 'HEAD' ? undefined : body);
          return;
        }
        sendJson(response, 200, apiIndex());
        return;
      }

      if (path === '/api/health') {
        if (method !== 'GET' && method !== 'HEAD') {
          throw ApiError.methodNotAllowed(method, ['GET', 'HEAD']);
        }
        sendJson(response, 200, {
          status: 'ok',
          pipeline: PIPELINE_VERSION,
          defaultModel: DEFAULT_MODEL,
          models: listModels().map((model) => model.id),
          uptimeSeconds: Math.round((Date.now() - startedAt) / 1000),
        });
        return;
      }

      if (path === '/api/models') {
        if (method !== 'GET' && method !== 'HEAD') {
          throw ApiError.methodNotAllowed(method, ['GET', 'HEAD']);
        }
        sendJson(response, 200, {
          default: DEFAULT_MODEL,
          models: MODEL_IDS.map((id) => ({
            ...MODELS[id],
            weightsUrl: MODELS[id].weightsPath,
            weightsRoot: MODELS[id].weightsRoot,
          })),
        });
        return;
      }

      if (path === '/api/parameters') {
        if (method !== 'GET' && method !== 'HEAD') {
          throw ApiError.methodNotAllowed(method, ['GET', 'HEAD']);
        }
        sendJson(response, 200, {
          parameters: PARAMETERS,
          note:
            'These are applied by the client. The server never receives a ' +
            'recording, so it has no conversion to be configured.',
        });
        return;
      }

      if (path === '/api/openapi.json') {
        if (method !== 'GET' && method !== 'HEAD') {
          throw ApiError.methodNotAllowed(method, ['GET', 'HEAD']);
        }
        sendJson(response, 200, openApiDocument());
        return;
      }

      // The endpoint that took a recording is gone, and saying so plainly is
      // more use to somebody with an old script than a bare 404.
      if (path === '/api/convert') {
        throw new ApiError(
          'gone',
          410,
          'This server does not take recordings. The audio never leaves the ' +
            'machine it is on: convert it in the browser, or on the command ' +
            'line with `npm run cli`.',
          { see: '/api/parameters' },
        );
      }

      throw ApiError.notFound(`There is nothing at ${path}.`);
    } catch (error) {
      sendError(response, error);
    }
  };

  return createServer((request, response) => {
    void handle(request, response).catch((error) => sendError(response, error));
  });
}

/** What the API is, for a client that asks. */
function apiIndex(): Record<string, unknown> {
  return {
    name: 'voidtrackr',
    description:
      'The catalogue of what this tool can do. The work happens on the ' +
      'machine that holds the recording — the browser, or the command line.',
    pipeline: PIPELINE_VERSION,
    acceptsRecordings: false,
    endpoints: [
      { method: 'GET', path: '/api/', description: 'This index, or a page about the API.' },
      { method: 'GET', path: '/api/health', description: 'Whether the server is up.' },
      { method: 'GET', path: '/api/models', description: 'The models and their licences.' },
      {
        method: 'GET',
        path: '/api/parameters',
        description: 'The parameters a conversion takes, and their defaults.',
      },
      { method: 'GET', path: '/api/openapi.json', description: 'The OpenAPI document.' },
    ],
    openapi: '/api/openapi.json',
  };
}

/** A page about the API, for a person who opens it in a browser. */
function apiHomePage(): string {
  const rows = PARAMETERS.map(
    (parameter) =>
      `    <tr><td><code>${parameter.name}</code></td><td><code>${parameter.flag}</code></td>` +
      `<td>${parameter.default}</td><td>${parameter.description}</td></tr>`,
  ).join('\n');

  const modelRows = MODEL_IDS.map(
    (id) =>
      `    <tr><td><code>${id}</code></td><td><code>${MODELS[id].weightsPath}</code></td>` +
      `<td>${MODELS[id].license}</td></tr>`,
  ).join('\n');

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>voidtrackr — API</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; padding: 2.5rem 1.5rem; background: #07070b; color: #ecebf5;
    font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 46rem; margin: 0 auto; }
  h1 { font-weight: 600; letter-spacing: -0.02em; margin: 0 0 0.25rem; }
  h1 span { color: #8b7bf7; }
  h2 { font-weight: 500; font-size: 1rem; margin: 2rem 0 0.6rem; color: #ecebf5; }
  .kicker {
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.8rem;
    color: #74718c; margin: 0 0 1.5rem;
  }
  p { color: #a7a4bd; }
  .note {
    background: #0c0c13; border: 1px solid #232330; border-left: 2px solid #8b7bf7;
    border-radius: 8px; padding: 0.9rem 1rem; color: #a7a4bd; margin: 1.25rem 0;
  }
  .note strong { color: #ecebf5; font-weight: 600; }
  pre {
    background: #0c0c13; border: 1px solid #232330; border-radius: 8px;
    padding: 0.9rem 1rem; overflow-x: auto; color: #b7adff;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.86rem;
  }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; }
  th, td { text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid #232330; }
  th { color: #74718c; font-weight: 500; font-size: 0.78rem; text-transform: uppercase;
       letter-spacing: 0.08em; }
  td { color: #a7a4bd; }
  td code { color: #b7adff; }
  a { color: #8b7bf7; }
</style>
</head>
<body>
<main>
  <p class="kicker">// a void tool</p>
  <h1>void<span>trackr</span> API</h1>
  <p>The catalogue of what this tool can do. The work happens on your machine.</p>

  <div class="note">
    <strong>No endpoint here takes a recording.</strong> The audio never leaves
    the machine it is on — not to this server, and not anywhere else. Convert
    it in the browser, or on the command line, and read the catalogue below to
    find out what the options are.
  </div>

  <h2>The one line</h2>
  <pre>npm run cli -- song.wav -o .</pre>

  <h2>With the site's model</h2>
  <pre>npm run cli -- --server http://192.168.100.109 song.wav -o .</pre>
  <p style="font-size:0.9rem">The model is fetched from the server and checked
  against its pinned hash. The recording is not sent.</p>

  <h2>Parameters</h2>
  <table>
    <tr><th>Name</th><th>Flag</th><th>Default</th><th>What it does</th></tr>
${rows}
  </table>

  <h2>Models</h2>
  <table>
    <tr><th>Model</th><th>Weights</th><th>Licence</th></tr>
${modelRows}
  </table>

  <h2>More</h2>
  <p>
    <a href="/api/openapi.json">The OpenAPI document</a> ·
    <a href="/api/health">Health</a> ·
    <a href="/api/models">Models</a> ·
    <a href="/api/parameters">Parameters</a> ·
    <a href="/">The site</a>
  </p>
</main>
</body>
</html>
`;
}
