/**
 * The HTTP API.
 *
 * The API describes the tool; it does not run it. The property that matters
 * most is therefore a negative one: there is no endpoint that takes a
 * recording, and the server never reads a request body at all. That is what
 * the tests below check, and what the rest of the tests then build on.
 *
 * The other half is that a client can get the model's weights from the
 * server and convert a recording locally, which is the one-line path a
 * script uses.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile } from 'node:fs/promises';
import { createServer, request as httpRequest, Server } from 'node:http';
import { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { createApiServer } from '../src/api/server';
import { PARAMETERS } from '../src/core/parameters';
import modelLock from '../src/core/model.lock.json';
import okarinaLock from '../src/core/okarina.lock.json';
import pianoLock from '../src/core/piano.lock.json';
import { PIANO_DIR } from '../src/core/piano';
import { DEFAULT_MODEL, MODELS } from '../src/core/registry';
import tsumugiLock from '../src/core/tsumugi.lock.json';
import { TSUMUGI_DIR } from '../src/core/tsumugi';
import yourmt3Lock from '../src/core/yourmt3.lock.json';
import { YOURMT3_DIR } from '../src/core/yourmt3';
import { fetchWeights } from '../src/core/weights-fetch';

/** Start the API on a free port. */
async function startServer(): Promise<{ url: string; server: Server; close: () => void }> {
  const server = createApiServer();
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', done));
  const address = server.address() as AddressInfo;

  return {
    url: `http://127.0.0.1:${address.port}`,
    server,
    // `fetch` keeps its connections alive in a pool, and `close()` waits
    // for them. The idle ones are closed first, so it does not.
    close: () => {
      server.closeIdleConnections();
      server.close();
    },
  };
}

test('the API says what it is', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const index = await fetch(`${api.url}/api/`);
  assert.equal(index.status, 200);
  const json = (await index.json()) as {
    name: string;
    endpoints: { path: string }[];
    acceptsRecordings: boolean;
  };
  assert.equal(json.name, 'voidtrackr');
  assert.ok(json.endpoints.length > 0);
  assert.equal(json.acceptsRecordings, false);

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
  const document = (await spec.json()) as {
    openapi: string;
    paths: Record<string, unknown>;
    info: { description: string };
  };
  assert.match(document.openapi, /^3\./);
  assert.ok('/models' in document.paths);
  assert.ok('/parameters' in document.paths);
  assert.match(document.info.description, /never reads a request body/);
});

test('the API publishes the parameters a conversion takes', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const response = await fetch(`${api.url}/api/parameters`);
  assert.equal(response.status, 200);
  const body = (await response.json()) as {
    parameters: { name: string; flag: string; default: unknown }[];
  };

  const names = body.parameters.map((parameter) => parameter.name);
  assert.deepEqual(names, PARAMETERS.map((parameter) => parameter.name));

  // The names are the ones the command line takes, so a script that reads
  // this and builds a command line cannot get it wrong.
  for (const parameter of body.parameters) {
    assert.ok(parameter.flag.startsWith('--'), `${parameter.name} has no flag`);
  }
});

test('the catalogue says where each model\'s weights really are', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const response = await fetch(`${api.url}/api/models`);
  assert.equal(response.status, 200);
  const catalogue = (await response.json()) as {
    models: { id: string; weightsUrl: string; weightsRoot: string }[];
  };

  const paths = new Map(catalogue.models.map((model) => [model.id, model.weightsUrl]));

  // Basic Pitch is a TensorFlow.js model and its files are not in the ONNX
  // tree at all. Okarina's ONNX files sit at the root of that tree rather
  // than in a folder of their own. The other three are in a folder named
  // after the model.
  assert.equal(paths.get('basic-pitch'), '/model/');
  assert.equal(paths.get('okarina'), '/model-onnx/');
  assert.equal(paths.get('piano'), `/model-onnx/${PIANO_DIR}/`);
  assert.equal(paths.get('tsumugi'), `/model-onnx/${TSUMUGI_DIR}/`);
  assert.equal(paths.get('yourmt3'), `/model-onnx/${YOURMT3_DIR}/`);

  // No model may point at a path the site does not serve. This is the
  // mistake the catalogue was made to stop: a path guessed from the model's
  // id is a 404 for every model whose layout is not the obvious one.
  for (const [id, path] of paths) {
    assert.ok(path.startsWith('/'), `${id} has no absolute path`);
    assert.ok(
      path === '/model/' || path.startsWith('/model-onnx/'),
      `${id} points outside the served weight directories: ${path}`,
    );
  }
});

test('the API does not take a recording', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  // The endpoint that used to take an upload is gone, and says so.
  const response = await fetch(`${api.url}/api/convert`, {
    method: 'POST',
    body: new Uint8Array(1024),
  });

  assert.equal(response.status, 410);
  const body = (await response.json()) as { error: { code: string; message: string } };
  assert.equal(body.error.code, 'gone');
  assert.match(body.error.message, /never leaves the machine/);

  // No other address takes one either.
  for (const path of ['/api/', '/api/health', '/api/models', '/api/parameters']) {
    const posted = await fetch(`${api.url}${path}`, {
      method: 'POST',
      body: new Uint8Array(1024),
    });
    assert.equal(posted.status, 405, `${path} answered a POST`);
  }
});

test('the server never reads the body', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  // A request whose body never ends. A server that read the body would
  // wait for it; this one answers without looking.
  const answer = await new Promise<{ status: number; body: string }>((done, failed) => {
    const client = httpRequest(
      {
        host: '127.0.0.1',
        port: new URL(api.url).port,
        path: '/api/convert',
        method: 'POST',
        headers: { 'Content-Length': String(1024 * 1024) },
      },
      (response) => {
        const chunks: Buffer[] = [];
        response.on('data', (chunk: Buffer) => chunks.push(chunk));
        response.on('end', () => {
          // The request was never finished, so the socket is closed here.
          client.destroy();
          done({ status: response.statusCode ?? 0, body: Buffer.concat(chunks).toString() });
        });
      },
    );

    client.on('error', failed);
    client.write(Buffer.alloc(16));

    // If the answer has not come by now, the server is waiting for the
    // body — which is exactly what must not happen.
    setTimeout(() => {
      client.destroy();
      failed(new Error('the server waited for the body'));
    }, 5000);
  });

  assert.equal(answer.status, 410);
  assert.match(answer.body, /gone/);
});

test('an address that does not exist is refused', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const response = await fetch(`${api.url}/api/nope`);
  assert.equal(response.status, 404);
  const body = (await response.json()) as { error: { code: string; message: string } };
  assert.equal(body.error.code, 'not_found');
  assert.ok(body.error.message.length > 0);
});

test('the wrong method is refused, and says what is allowed', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const response = await fetch(`${api.url}/api/models`, { method: 'DELETE' });
  assert.equal(response.status, 405);
  const body = (await response.json()) as { error: { code: string; details: { allowed: string[] } } };
  assert.equal(body.error.code, 'method_not_allowed');
  assert.deepEqual(body.error.details.allowed, ['GET', 'HEAD']);
});

test('the API answers a browser with a page', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const response = await fetch(`${api.url}/api/`, {
    headers: { Accept: 'text/html' },
  });

  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-type') ?? '', /text\/html/);
  const page = await response.text();
  assert.match(page, /voidtrackr/);
  assert.match(page, /No endpoint here takes a recording/);
});

/**
 * A server that behaves like the site: it serves the catalogue and the
 * model's weights, and it records every request it is asked for.
 */
async function startWeightsServer(): Promise<{
  url: string;
  asked: string[];
  close: () => void;
}> {
  const asked: string[] = [];

  const server = createServer((request, response) => {
    const path = new URL(request.url ?? '/', 'http://localhost').pathname;
    asked.push(path);

    if (path === '/api/models') {
      const body = JSON.stringify({
        default: DEFAULT_MODEL,
        models: Object.values(MODELS).map((model) => ({
          id: model.id,
          weightsPath: model.weightsPath,
          weightsRoot: model.weightsRoot,
        })),
      });
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(body);
      return;
    }

    // The weights themselves, read off this machine. Basic Pitch is the
    // small one and its files are not in the ONNX tree, so it is the one
    // this server serves.
    const name = path.startsWith('/model/') ? path.slice('/model/'.length) : '';
    if (name) {
      readFile(join(process.cwd(), 'public', 'model', name)).then(
        (bytes) => {
          response.writeHead(200, { 'Content-Length': bytes.length });
          response.end(bytes);
        },
        () => {
          response.writeHead(404);
          response.end();
        },
      );
      return;
    }

    response.writeHead(404);
    response.end();
  });

  await new Promise<void>((done) => server.listen(0, '127.0.0.1', done));
  const address = server.address() as AddressInfo;

  return {
    url: `http://127.0.0.1:${address.port}`,
    asked,
    close: () => {
      server.closeIdleConnections();
      server.close();
    },
  };
}

test('the weight files hang off the root the catalogue names', async (t) => {
  const api = await startServer();
  t.after(() => api.close());

  const response = await fetch(`${api.url}/api/models`);
  const catalogue = (await response.json()) as {
    models: { id: string; weightsRoot: string }[];
  };

  // The lock file names the files; the catalogue says what to put in front
  // of them. Together they are the address of every weight file.
  const locks: Record<string, { files: Record<string, unknown> }> = {
    'basic-pitch': modelLock,
    tsumugi: tsumugiLock,
    yourmt3: yourmt3Lock,
    piano: pianoLock,
    okarina: okarinaLock,
  };

  for (const model of catalogue.models) {
    const root = model.weightsRoot;
    for (const name of Object.keys(locks[model.id].files)) {
      const address = `${root}${name}`;
      assert.ok(address.startsWith('/model'), `${model.id}: ${address}`);
    }
  }

  // The two layouts that differ, spelled out.
  const roots = new Map(catalogue.models.map((m) => [m.id, m.weightsRoot]));
  assert.equal(roots.get('piano'), '/model-onnx/');
  assert.equal(`${roots.get('piano')}${'piano/piano_kong.onnx'}`, '/model-onnx/piano/piano_kong.onnx');
  assert.equal(`${roots.get('basic-pitch')}${'model.json'}`, '/model/model.json');
  assert.equal(`${roots.get('okarina')}${'cqt156.onnx'}`, '/model-onnx/cqt156.onnx');
});

test('a client can take the model from the server', async (t) => {
  const server = await startWeightsServer();
  t.after(() => server.close());

  const cache = await mkdtemp(join(tmpdir(), 'voidtrackr-weights-'));
  const fetched = await fetchWeights({ base: server.url }, 'basic-pitch', cache);

  assert.ok(fetched.bytes > 0, 'nothing was downloaded');
  assert.equal(fetched.cached, 0);

  // The files landed where the loader looks for them.
  const manifest = await readFile(fetched.modelPath);
  assert.ok(manifest.length > 0);

  // And only the weights were asked for. A recording is never part of this.
  assert.ok(server.asked.every((path) => !path.endsWith('.wav')));
  assert.ok(server.asked.includes('/api/models'));
  assert.ok(server.asked.includes('/model/model.json'));

  // The second time, the files are already there and nothing is downloaded.
  const again = await fetchWeights({ base: server.url }, 'basic-pitch', cache);
  assert.equal(again.bytes, 0);
  assert.ok(again.cached > 0);
});

test('a client can take the weights and convert, without sending anything', async (t) => {
  const server = await startWeightsServer();
  t.after(() => server.close());

  const cache = await mkdtemp(join(tmpdir(), 'voidtrackr-weights-'));
  await fetchWeights({ base: server.url }, 'basic-pitch', cache);

  // The weights came over the wire; the recording did not. The server was
  // asked for the catalogue and the weights, and for nothing else.
  assert.deepEqual(
    server.asked.filter((path) => !path.startsWith('/model/')),
    ['/api/models'],
  );
});

test('the command line takes the model from the server and sends no recording', async (t) => {
  const server = await startWeightsServer();
  t.after(() => server.close());

  const localOut = await mkdtemp(join(tmpdir(), 'voidtrackr-out-'));
  const remoteOut = await mkdtemp(join(tmpdir(), 'voidtrackr-out-'));
  const cache = await mkdtemp(join(tmpdir(), 'voidtrackr-cache-'));

  // The command line is a program, not a library, so it is run as one.
  const run = (extra: string[], output: string) =>
    new Promise<{ code: number; stdout: string; stderr: string }>((done) => {
      execFile(
        join(process.cwd(), 'node_modules', '.bin', 'tsx'),
        [
          join(process.cwd(), 'src', 'cli', 'convert.ts'),
          ...extra,
          'test/sample.wav',
          '-o',
          output,
          '--model',
          'basic-pitch',
          '--quiet',
        ],
        { env: { ...process.env, VOIDTRACKR_CACHE: cache } },
        (error, stdout, stderr) => {
          done({
            code: error && typeof error.code === 'number' ? error.code : 0,
            stdout,
            stderr,
          });
        },
      );
    });

  // The same command with and without the server. The file it produces
  // must be the same either way: taking the model from the server is not
  // allowed to change the answer.
  const local = await run([], localOut);
  assert.equal(local.code, 0, local.stderr);

  const remote = await run(['--server', server.url], remoteOut);
  assert.equal(remote.code, 0, remote.stderr);

  const localBytes = await readFile(join(localOut, 'sample.mid'));
  const remoteBytes = await readFile(join(remoteOut, 'sample.mid'));
  assert.deepEqual(remoteBytes, localBytes);

  // The server was asked for the catalogue and the weights. It was never
  // asked for the recording, because the recording never went to it.
  assert.deepEqual(
    server.asked.filter((path) => !path.startsWith('/model/')),
    ['/api/models'],
  );
  assert.ok(server.asked.every((path) => !path.endsWith('.wav')));
});

