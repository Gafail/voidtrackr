/**
 * The HTTP API.
 *
 * The site is static and the browser does the work, but a tool that is
 * driven from a script needs an address to call. This is that address: it
 * runs on the same machine as the site, at the same origin, and it does the
 * same work through the same code — so a file it produces is byte-identical
 * to one the command line produces for the same input.
 *
 *   curl -F file=@song.wav -F model=piano http://192.168.100.109/api/convert -o song.mid
 *
 * The audio is read into memory, converted, and the MIDI is sent back. The
 * upload is never written to disk.
 */

import { createHash } from 'node:crypto';
import { createServer, IncomingMessage, Server, ServerResponse } from 'node:http';

import { keyName } from '../core/tempo';
import { DEFAULT_MODEL, listModels, MODEL_IDS, MODELS, ModelId } from '../core/registry';
import { PIPELINE_VERSION, provenanceFor } from '../core/version';
import { ONNX_THREADS } from '../core/onnx';
import { TranscriptionModel } from '../core/model';

import { installTfjsNodeBackend } from './backend';
import { convertAudio } from './convert';
import { ApiError } from './errors';
import { loadModel, ModelLocation } from './model-loader';
import { parseMultipart } from './multipart';
import { openApiDocument } from './openapi';
import { ApiParameters, parseParameters } from './params';

/** How the server is set up. */
export interface ServerOptions {
  /** The port to listen on. */
  port: number;
  /** The address to listen on. 127.0.0.1 keeps it off the network. */
  host: string;
  /** Where the ONNX weights are. */
  modelDir: string;
  /** Where the Basic Pitch model.json is. */
  modelPath: string;
  /** The largest upload the server will take, in bytes. */
  maxUploadBytes: number;
  /** A token callers have to present. Off when it is not set. */
  token?: string;
  /** How many conversions may run at once. */
  concurrency: number;
}

/** What one request is carrying. */
interface RequestContext {
  parameters: ApiParameters;
  upload?: { content: Buffer; filename?: string };
}

/** What the server is doing right now. */
interface ServerState {
  startedAt: number;
  conversions: number;
  failures: number;
  inFlight: number;
  model?: TranscriptionModel;
  modelId?: string;
  okarinaSize?: string;
}

/** Read the body, refusing anything over the limit. */
function readBody(request: IncomingMessage, limit: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;

    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      request.destroy();
      reject(error);
    };

    request.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > limit) {
        fail(
          ApiError.payloadTooLarge(
            `The upload is bigger than the ${Math.round(limit / 1048576)} MB limit.`,
            { limitBytes: limit },
          ),
        );
        return;
      }
      chunks.push(chunk);
    });

    request.on('end', () => {
      if (settled) return;
      settled = true;
      resolve(Buffer.concat(chunks));
    });

    request.on('error', (error) => fail(error));
  });
}

/** The name the reply's file should carry. */
function outputName(filename: string | undefined): string {
  const base = (filename ?? 'audio').replace(/\\/g, '/').split('/').pop() ?? 'audio';
  const withoutExtension = base.replace(/\.[^.]+$/, '') || 'audio';
  return `${withoutExtension}.mid`;
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

/** The JSON body of a conversion, for a caller that wants the notes. */
function jsonBody(
  parameters: ApiParameters,
  result: Awaited<ReturnType<typeof convertAudio>>['result'],
  midi: Uint8Array,
  provenance: string,
): Record<string, unknown> {
  return {
    model: parameters.model,
    pipeline: PIPELINE_VERSION,
    provenance,
    duration: Number(result.duration.toFixed(3)),
    tempo: result.tempo,
    key: { ...result.key, name: keyName(result.key.tonic, result.key.mode) },
    stats: result.stats,
    log: result.log,
    tracks: result.tracks.map((track) => ({
      id: track.id,
      name: track.name,
      kind: track.kind,
      notes: track.notes.map((note) => ({
        pitch: note.pitchMidi,
        startTime: Number(note.startTime.toFixed(6)),
        duration: Number(note.duration.toFixed(6)),
        amplitude: Number(note.amplitude.toFixed(6)),
      })),
    })),
    midi: {
      bytes: midi.length,
      sha256: createHash('sha256').update(midi).digest('hex'),
    },
  };
}

/**
 * Start the API.
 *
 * The returned server is not listening yet; `listen()` is separate so that
 * a test can choose the port.
 */
export function createApiServer(options: ServerOptions): Server {
  const state: ServerState = {
    startedAt: Date.now(),
    conversions: 0,
    failures: 0,
    inFlight: 0,
  };

  const location: ModelLocation = {
    modelDir: options.modelDir,
    modelPath: options.modelPath,
  };

  /**
   * Load a model, keeping the last one so a repeat request is quick.
   *
   * Only one is kept: the box this runs on has a gigabyte of memory, and
   * five models at once would not fit.
   */
  const modelFor = async (
    modelId: ModelId,
    okarinaSize: string,
  ): Promise<TranscriptionModel> => {
    if (state.model && state.modelId === modelId && state.okarinaSize === okarinaSize) {
      return state.model;
    }
    state.model?.dispose?.();
    state.model = undefined;
    state.modelId = undefined;
    state.okarinaSize = undefined;

    const model = await loadModel(modelId, location, okarinaSize as 's' | 'm' | 'l');
    state.model = model;
    state.modelId = modelId;
    state.okarinaSize = okarinaSize;
    return model;
  };

  /** Check the bearer token, if one is set. */
  const checkToken = (request: IncomingMessage): void => {
    if (!options.token) return;
    const header = request.headers.authorization ?? '';
    const given = header.startsWith('Bearer ') ? header.slice(7).trim() : '';
    if (given !== options.token) {
      throw ApiError.unauthorized('This API needs a bearer token.');
    }
  };

  /** Read the request: the parameters and the file. */
  const readRequest = async (request: IncomingMessage): Promise<RequestContext> => {
    const url = new URL(request.url ?? '/', 'http://localhost');
    const source: Record<string, string | undefined> = {};

    for (const [key, value] of url.searchParams) {
      source[key] = value;
    }

    const contentType = request.headers['content-type'];
    const body = await readBody(request, options.maxUploadBytes);

    let upload: RequestContext['upload'];

    if (contentType && contentType.toLowerCase().includes('multipart/form-data')) {
      const parts = parseMultipart(body, contentType);
      let named: RequestContext['upload'];
      let first: RequestContext['upload'];

      // Every part is read. The loop does not stop at the file: the fields
      // may come after it, and curl puts them in the order they were given,
      // so `-F file=@song.wav -F model=piano` has the file first.
      for (const part of parts) {
        if (part.filename !== undefined) {
          const candidate = { content: part.content, filename: part.filename };
          if (part.name === 'file' || part.name === 'audio') {
            named = candidate;
          }
          first ??= candidate;
          continue;
        }
        if (part.name !== '') {
          source[part.name] = part.content.toString('utf8').trim();
        }
      }

      upload = named ?? first;
    } else if (contentType && contentType.toLowerCase().startsWith('audio/')) {
      // The body is the file itself. Its name comes from the query string.
      upload = { content: body, filename: source.filename };
    } else if (body.length > 0) {
      throw ApiError.unsupportedMediaType(
        'Send the audio as multipart/form-data, or as the request body with an ' +
          'audio content type.',
        { received: contentType ?? '(none)' },
      );
    }

    const format = source.format === 'json' ? 'json' : undefined;
    const parameters = parseParameters(source, format);
    if (source.filename === undefined && upload?.filename) {
      parameters.filename = upload.filename;
    }

    return { parameters, upload };
  };

  /** Do the conversion and answer. */
  const convert = async (
    context: RequestContext,
    response: ServerResponse,
  ): Promise<void> => {
    const { parameters } = context;

    if (!context.upload || context.upload.content.length === 0) {
      throw ApiError.badRequest('No audio was sent. Put the file in a part called "file".', {
        parameter: 'file',
      });
    }

    if (state.inFlight >= options.concurrency) {
      throw ApiError.busy(
        'The server is already converting as much as it can. Try again in a moment.',
        { concurrency: options.concurrency },
      );
    }
    state.inFlight += 1;

    try {
      const data = context.upload.content.buffer.slice(
        context.upload.content.byteOffset,
        context.upload.content.byteOffset + context.upload.content.byteLength,
      ) as ArrayBuffer;

      let model: TranscriptionModel;
      try {
        model = await modelFor(parameters.model, parameters.okarinaSize);
      } catch (error) {
        throw ApiError.modelUnavailable(
          `The ${parameters.model} model is not available on this machine.`,
          { reason: error instanceof Error ? error.message : String(error) },
        );
      }

      const provenance = provenanceFor(model);

      let output;
      try {
        output = await convertAudio(data, parameters, model, provenance);
      } catch (error) {
        throw ApiError.unprocessable(
          `The audio could not be read: ${error instanceof Error ? error.message : String(error)}`,
        );
      }

      state.conversions += 1;

      const headers: Record<string, string> = {
        'X-Voidtrackr-Pipeline': PIPELINE_VERSION,
        'X-Voidtrackr-Model': parameters.model,
        'X-Voidtrackr-Notes': String(
          output.result.tracks.reduce((sum, track) => sum + track.notes.length, 0),
        ),
        'X-Voidtrackr-Tempo': output.result.tempo.bpm.toFixed(2),
        'X-Voidtrackr-Key': keyName(output.result.key.tonic, output.result.key.mode),
        'X-Voidtrackr-SHA256': createHash('sha256').update(output.midi).digest('hex'),
        'Cache-Control': 'no-store',
      };

      if (parameters.format === 'json') {
        sendJson(
          response,
          200,
          jsonBody(parameters, output.result, output.midi, provenance),
          headers,
        );
        return;
      }

      response.writeHead(200, {
        ...headers,
        'Content-Type': 'audio/midi',
        'Content-Length': output.midi.length,
        'Content-Disposition': `attachment; filename="${outputName(parameters.filename)}"`,
      });
      response.end(Buffer.from(output.midi));
    } finally {
      state.inFlight -= 1;
    }
  };

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
          threads: ONNX_THREADS,
          defaultModel: DEFAULT_MODEL,
          models: listModels().map((model) => model.id),
          conversions: state.conversions,
          failures: state.failures,
          inFlight: state.inFlight,
          concurrency: options.concurrency,
          uptimeSeconds: Math.round((Date.now() - state.startedAt) / 1000),
          nativeBackend: await installTfjsNodeBackend(),
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
            weightsUrl: `/model-onnx/${id}`,
          })),
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

      if (path === '/api/convert') {
        if (method !== 'POST') {
          throw ApiError.methodNotAllowed(method, ['POST']);
        }
        checkToken(request);
        const context = await readRequest(request);
        await convert(context, response);
        return;
      }

      throw ApiError.notFound(`There is nothing at ${path}.`);
    } catch (error) {
      if (!(error instanceof ApiError)) state.failures += 1;
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
      'Audio to MIDI, locally. The same pipeline the site and the command line use.',
    pipeline: PIPELINE_VERSION,
    endpoints: [
      { method: 'GET', path: '/api/', description: 'This index, or a page about the API.' },
      { method: 'GET', path: '/api/health', description: 'Whether the server is up.' },
      { method: 'GET', path: '/api/models', description: 'The models and their licences.' },
      { method: 'GET', path: '/api/openapi.json', description: 'The OpenAPI document.' },
      {
        method: 'POST',
        path: '/api/convert',
        description: 'Convert an audio file to MIDI.',
      },
    ],
    openapi: '/api/openapi.json',
  };
}

/** A page about the API, for a person who opens it in a browser. */
function apiHomePage(): string {
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
  <p>Audio to MIDI, on this machine. The same pipeline the site and the command line use.</p>

  <h2>One line</h2>
  <pre>curl -F file=@song.wav http://192.168.100.109/api/convert -o song.mid</pre>

  <h2>With values</h2>
  <pre>curl -F file=@song.wav -F model=piano -F pianify=true -F grid=30 \\
     http://192.168.100.109/api/convert -o song.mid</pre>

  <h2>As JSON</h2>
  <pre>curl -F file=@song.wav -F format=json http://192.168.100.109/api/convert</pre>

  <h2>Parameters</h2>
  <table>
    <tr><th>Name</th><th>Default</th><th>What it does</th></tr>
    <tr><td><code>file</code></td><td>—</td><td>The audio. Required.</td></tr>
    <tr><td><code>model</code></td><td>piano</td><td>basic-pitch, tsumugi, yourmt3, piano, okarina</td></tr>
    <tr><td><code>okarinaSize</code></td><td>m</td><td>s, m or l. Only for Okarina.</td></tr>
    <tr><td><code>drums</code></td><td>true</td><td>Look for a drum kit.</td></tr>
    <tr><td><code>tempo</code></td><td>estimated</td><td>Force a tempo, in BPM.</td></tr>
    <tr><td><code>rangeLow</code></td><td>0</td><td>Lowest frequency to keep, in Hz.</td></tr>
    <tr><td><code>rangeHigh</code></td><td>—</td><td>Highest frequency to keep, in Hz.</td></tr>
    <tr><td><code>grid</code></td><td>0</td><td>Slots per second. 0 leaves the notes alone.</td></tr>
    <tr><td><code>gridPhase</code></td><td>0</td><td>Where the first slot sits, in ms.</td></tr>
    <tr><td><code>pianify</code></td><td>false</td><td>Arrange the notes for two hands.</td></tr>
    <tr><td><code>format</code></td><td>midi</td><td>midi or json.</td></tr>
  </table>

  <h2>More</h2>
  <p>
    <a href="/api/openapi.json">The OpenAPI document</a> ·
    <a href="/api/health">Health</a> ·
    <a href="/api/models">Models</a> ·
    <a href="/">The site</a>
  </p>
</main>
</body>
</html>
`;
}
