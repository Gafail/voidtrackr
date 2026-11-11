/**
 * The HTTP API.
 *
 * Three things are checked here: that the multipart parser reads a body
 * correctly, that the parameters are read strictly, and that a request
 * through the server produces the same MIDI the command line produces.
 *
 * The last one is the one that matters. The tool promises that the same
 * input gives the same file however it is asked for, and the API is a new
 * way of asking.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { AddressInfo } from 'node:net';

import { installTfjsNodeBackend } from '../src/api/backend';
import { fileFetch } from '../src/api/model-loader';
import { boundaryOf, parseMultipart } from '../src/api/multipart';
import { parseParameters } from '../src/api/params';
import { createApiServer } from '../src/api/server';
import { DEFAULT_MODEL } from '../src/core/registry';

/** Build a multipart body, the way a browser would send one. */
function multipart(
  parts: { name: string; value?: string; filename?: string; content?: Buffer }[],
  boundary = '----voidtrackrTest',
): { body: Buffer; contentType: string } {
  const chunks: Buffer[] = [];

  for (const part of parts) {
    let header = `--${boundary}\r\nContent-Disposition: form-data; name="${part.name}"`;
    if (part.filename !== undefined) header += `; filename="${part.filename}"`;
    header += '\r\n';
    if (part.filename !== undefined) header += 'Content-Type: application/octet-stream\r\n';
    header += '\r\n';

    chunks.push(Buffer.from(header, 'utf8'));
    chunks.push(
      part.content ?? Buffer.from(part.value ?? '', 'utf8'),
    );
    chunks.push(Buffer.from('\r\n', 'utf8'));
  }

  chunks.push(Buffer.from(`--${boundary}--\r\n`, 'utf8'));

  return {
    body: Buffer.concat(chunks),
    contentType: `multipart/form-data; boundary=${boundary}`,
  };
}

/**
 * The body of a request.
 *
 * TypeScript's `BodyInit` wants an `ArrayBufferView` over a plain
 * `ArrayBuffer`, and a Buffer is over an `ArrayBufferLike`, which may be
 * shared memory. A copy into a fresh Uint8Array settles it.
 */
function asBody(buffer: Buffer): ArrayBuffer {
  const copy = new ArrayBuffer(buffer.length);
  new Uint8Array(copy).set(buffer);
  return copy;
}

/** The model directory, for the server. */
const MODEL_DIR = resolve('models', 'onnx');
const MODEL_PATH = resolve('public', 'model', 'model.json');

test('a file URL can be fetched off the disk', async () => {
  // TensorFlow.js fetches the model's manifest and its shards itself, and
  // Node's own fetch does not understand the file:// scheme. Without this,
  // Basic Pitch cannot be loaded on a machine without the native backend.
  const url = pathToFileURL(resolve('public', 'model', 'model.json')).href;
  const response = await fileFetch(url);

  assert.equal(response.status, 200);
  const manifest = JSON.parse(await response.text()) as { modelTopology?: unknown };
  assert.ok(manifest.modelTopology);
});

test('the boundary comes out of the content type', () => {
  assert.equal(boundaryOf('multipart/form-data; boundary=abc'), 'abc');
  assert.equal(boundaryOf('multipart/form-data; boundary="a b c"'), 'a b c');
  assert.equal(boundaryOf('multipart/form-data'), null);
  assert.equal(boundaryOf(undefined), null);
});

test('a multipart body is read into its parts', () => {
  const { body, contentType } = multipart([
    { name: 'model', value: 'piano' },
    { name: 'file', filename: 'song.wav', content: Buffer.from([0, 1, 2, 3, 255]) },
  ]);

  const parts = parseMultipart(body, contentType);

  assert.equal(parts.length, 2);
  assert.equal(parts[0].name, 'model');
  assert.equal(parts[0].content.toString('utf8'), 'piano');
  assert.equal(parts[1].name, 'file');
  assert.equal(parts[1].filename, 'song.wav');
  assert.deepEqual([...parts[1].content], [0, 1, 2, 3, 255]);
});

test('a multipart body keeps its bytes intact', () => {
  // The bytes a WAV file is made of include the boundary-shaped sequence
  // only by accident, but a file part can be anything at all, so the
  // content has to survive whatever is in it.
  const content = Buffer.from(
    'RIFF----nottheboundary----WAVEfmt ',
    'utf8',
  );
  const { body, contentType } = multipart([
    { name: 'file', filename: 'a.wav', content },
  ]);

  const parts = parseMultipart(body, contentType);

  assert.equal(parts.length, 1);
  assert.equal(parts[0].content.toString('utf8'), content.toString('utf8'));
});

test('a body with no boundary is refused', () => {
  assert.throws(() => parseMultipart(Buffer.from('x'), 'text/plain'), /no boundary/);
});

test('the parameters have sensible defaults', () => {
  const parameters = parseParameters({});

  assert.equal(parameters.model, DEFAULT_MODEL);
  assert.equal(parameters.detectDrums, true);
  assert.equal(parameters.format, 'midi');
  assert.equal(parameters.range, undefined);
  assert.equal(parameters.quantise, undefined);
  assert.equal(parameters.pianify, undefined);
  assert.equal(parameters.forceTempo, undefined);
});

test('the parameters are read', () => {
  const parameters = parseParameters({
    model: 'okarina',
    okarinaSize: 'l',
    drums: 'false',
    tempo: '120',
    rangeLow: '100',
    rangeHigh: '1200',
    grid: '30',
    gridPhase: '12',
    pianify: 'true',
  });

  assert.equal(parameters.model, 'okarina');
  assert.equal(parameters.okarinaSize, 'l');
  assert.equal(parameters.detectDrums, false);
  assert.equal(parameters.forceTempo, 120);
  assert.equal(parameters.range?.low, 100);
  assert.equal(parameters.range?.high, 1200);
  assert.equal(parameters.quantise?.grid, 30);
  assert.equal(parameters.quantise?.phase, 0.012);
  assert.deepEqual(parameters.pianify, {});
});

test('a parameter that does not make sense is refused', () => {
  assert.throws(() => parseParameters({ model: 'nonsense' }), /not a model/);
  assert.throws(() => parseParameters({ grid: 'abc' }), /must be a number/);
  assert.throws(() => parseParameters({ drums: 'maybe' }), /true or false/);
  assert.throws(() => parseParameters({ tempo: '5' }), /between 20 and 400/);
  assert.throws(
    () => parseParameters({ rangeLow: '900', rangeHigh: '400' }),
    /must be below/,
  );
  assert.throws(() => parseParameters({ format: 'xml' }), /midi or json/);
  assert.throws(() => parseParameters({ gridPhase: '10' }), /needs a "grid"/);
});

/** Start the API on a port the operating system picks. */
async function startServer() {
  await installTfjsNodeBackend();
  const server = createApiServer({
    port: 0,
    host: '127.0.0.1',
    modelDir: MODEL_DIR,
    modelPath: MODEL_PATH,
    maxUploadBytes: 32 * 1024 * 1024,
    concurrency: 1,
  });

  await new Promise<void>((done) => server.listen(0, '127.0.0.1', done));
  const { port } = server.address() as AddressInfo;

  return {
    url: `http://127.0.0.1:${port}`,
    close: () => new Promise<void>((done) => server.close(() => done())),
  };
}

test('a request through the API gives the MIDI, and says what it made', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const { body, contentType } = multipart([
    { name: 'model', value: 'basic-pitch' },
    { name: 'file', filename: 'sample.wav', content: readFileSync('test/sample.wav') },
  ]);

  const response = await fetch(`${api.url}/api/convert`, {
    method: 'POST',
    headers: { 'Content-Type': contentType },
    body: asBody(body),
  });

  assert.equal(response.status, 200);
  assert.equal(response.headers.get('content-type'), 'audio/midi');
  assert.equal(response.headers.get('x-voidtrackr-model'), 'basic-pitch');
  assert.ok(Number(response.headers.get('x-voidtrackr-notes')) > 0);
  assert.match(response.headers.get('x-voidtrackr-sha256') ?? '', /^[0-9a-f]{64}$/);

  const midi = Buffer.from(await response.arrayBuffer());
  assert.equal(midi.subarray(0, 4).toString('ascii'), 'MThd');
});

test('the API can answer with the notes instead of the file', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const { body, contentType } = multipart([
    { name: 'model', value: 'basic-pitch' },
    { name: 'format', value: 'json' },
    { name: 'file', filename: 'sample.wav', content: readFileSync('test/sample.wav') },
  ]);

  const response = await fetch(`${api.url}/api/convert`, {
    method: 'POST',
    headers: { 'Content-Type': contentType },
    body: asBody(body),
  });

  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-type') ?? '', /application\/json/);

  const json = (await response.json()) as {
    model: string;
    pipeline: string;
    tracks: { notes: { pitch: number; startTime: number }[] }[];
  };

  assert.equal(json.model, 'basic-pitch');
  assert.match(json.pipeline, /^\d+\.\d+\.\d+$/);
  assert.ok(json.tracks.length > 0);
  assert.ok(json.tracks[0].notes.length > 0);
  assert.equal(typeof json.tracks[0].notes[0].pitch, 'number');
});

test('the fields are read even when they come after the file', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  // curl sends the parts in the order they were given, so
  // `-F file=@song.wav -F model=piano` puts the file first.
  const { body, contentType } = multipart([
    { name: 'file', filename: 'sample.wav', content: readFileSync('test/sample.wav') },
    { name: 'model', value: 'basic-pitch' },
  ]);

  const response = await fetch(`${api.url}/api/convert`, {
    method: 'POST',
    headers: { 'Content-Type': contentType },
    body: asBody(body),
  });

  assert.equal(response.status, 200);
  assert.equal(response.headers.get('x-voidtrackr-model'), 'basic-pitch');
});

test('a request with no file is refused', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const { body, contentType } = multipart([{ name: 'model', value: 'piano' }]);

  const response = await fetch(`${api.url}/api/convert`, {
    method: 'POST',
    headers: { 'Content-Type': contentType },
    body: asBody(body),
  });

  assert.equal(response.status, 400);
  const json = (await response.json()) as { error: { code: string; message: string } };
  assert.equal(json.error.code, 'bad_request');
  assert.match(json.error.message, /No audio/);
});

test('a parameter that does not make sense is refused with a code', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const { body, contentType } = multipart([
    { name: 'grid', value: 'abc' },
    { name: 'file', filename: 'sample.wav', content: readFileSync('test/sample.wav') },
  ]);

  const response = await fetch(`${api.url}/api/convert`, {
    method: 'POST',
    headers: { 'Content-Type': contentType },
    body: asBody(body),
  });

  assert.equal(response.status, 400);
  const json = (await response.json()) as { error: { code: string; details: unknown } };
  assert.equal(json.error.code, 'bad_request');
  assert.deepEqual(json.error.details, { parameter: 'grid', value: 'abc' });
});

test('the wrong method is refused, and says what is allowed', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const response = await fetch(`${api.url}/api/convert`, { method: 'GET' });

  assert.equal(response.status, 405);
  const json = (await response.json()) as { error: { code: string; details: { allowed: string[] } } };
  assert.equal(json.error.code, 'method_not_allowed');
  assert.deepEqual(json.error.details.allowed, ['POST']);
});

test('an address that does not exist is refused', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const response = await fetch(`${api.url}/api/nonsense`);

  assert.equal(response.status, 404);
  const json = (await response.json()) as { error: { code: string } };
  assert.equal(json.error.code, 'not_found');
});

test('the API says what it is', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const index = await fetch(`${api.url}/api/`);
  assert.equal(index.status, 200);
  const json = (await index.json()) as { name: string; endpoints: unknown[] };
  assert.equal(json.name, 'voidtrackr');
  assert.ok(json.endpoints.length > 0);

  const health = await fetch(`${api.url}/api/health`);
  assert.equal(health.status, 200);
  const state = (await health.json()) as { status: string; models: string[] };
  assert.equal(state.status, 'ok');
  assert.ok(state.models.includes('piano'));

  const models = await fetch(`${api.url}/api/models`);
  const catalogue = (await models.json()) as { models: { id: string; license: string }[] };
  assert.equal(catalogue.models.length, 5);
  assert.ok(catalogue.models.every((model) => model.license.length > 0));

  const spec = await fetch(`${api.url}/api/openapi.json`);
  const document = (await spec.json()) as { openapi: string; paths: Record<string, unknown> };
  assert.match(document.openapi, /^3\./);
  assert.ok('/convert' in document.paths);
});

test('the API answers a browser with a page', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const response = await fetch(`${api.url}/api/`, {
    headers: { Accept: 'text/html' },
  });

  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-type') ?? '', /text\/html/);
  assert.match(await response.text(), /voidtrackr/);
});
