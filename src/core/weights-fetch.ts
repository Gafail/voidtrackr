/**
 * Fetching a model's weights from a server.
 *
 * The site holds the weights and the browser fetches them over HTTP, so a
 * client on the network can do exactly the same thing. It asks the
 * catalogue where the files are, downloads the ones the lock file names,
 * and checks each one against its pinned hash.
 *
 * What travels is the model's own weights — the same bytes the browser
 * already downloads. What does not travel is the recording: it stays on
 * the machine that has it, and the conversion happens there too.
 *
 * The files are cached, so the second run of a model does not download it
 * again.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { sha256Hex } from './model';
import modelLock from './model.lock.json';
import okarinaLock from './okarina.lock.json';
import pianoLock from './piano.lock.json';
import tsumugiLock from './tsumugi.lock.json';
import yourmt3Lock from './yourmt3.lock.json';
import { ModelId } from './registry';

/** The part of a lock file this needs. */
interface Lock {
  files: Record<string, { sha256: string }>;
}

/** The lock file that pins each model. */
const LOCKS: Record<ModelId, Lock> = {
  'basic-pitch': modelLock as Lock,
  tsumugi: tsumugiLock as Lock,
  yourmt3: yourmt3Lock as Lock,
  piano: pianoLock as Lock,
  okarina: okarinaLock as Lock,
};

/** Where a server is, and how to talk to it. */
export interface WeightsServer {
  /** The base address, such as `http://192.168.100.109`. */
  base: string;
  /** A bearer token, when the server wants one. */
  token?: string;
}

/** The weights, and where they ended up. */
export interface FetchedWeights {
  /** The directory holding the ONNX weights. */
  modelDir: string;
  /** The path to the Basic Pitch model.json. */
  modelPath: string;
  /** How many bytes came over the wire. */
  bytes: number;
  /** How many files were already cached. */
  cached: number;
}

/** One entry of the catalogue. */
interface CatalogueEntry {
  id: ModelId;
  weightsPath: string;
  weightsRoot: string;
}

/** Join a base address and a path without doubling the slash. */
function url(base: string, path: string): string {
  return `${base.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;
}

/**
 * Ask the server where the weights are.
 *
 * The catalogue is the only thing that knows the layout, and it is asked
 * rather than assumed: the paths are not the same shape for every model.
 */
async function fetchCatalogue(server: WeightsServer): Promise<CatalogueEntry[]> {
  const response = await fetch(url(server.base, '/api/models'), {
    headers: server.token ? { Authorization: `Bearer ${server.token}` } : {},
  });

  if (!response.ok) {
    throw new Error(
      `The server's catalogue could not be read (${response.status} from ${url(server.base, '/api/models')}).`,
    );
  }

  const body = (await response.json()) as { models?: CatalogueEntry[] };
  if (!Array.isArray(body.models)) {
    throw new Error("The server's catalogue did not carry a list of models.");
  }

  return body.models;
}

/**
 * Fetch one model's weights and check them.
 *
 * Every file the lock file names is fetched, because the lock file is what
 * says the model is made of. The hashes are checked against it, so a
 * truncated download or a swapped file is caught rather than used.
 */
export async function fetchWeights(
  server: WeightsServer,
  model: ModelId,
  cacheDir: string,
): Promise<FetchedWeights> {
  const catalogue = await fetchCatalogue(server);
  const entry = catalogue.find((item) => item.id === model);

  if (!entry) {
    throw new Error(
      `The server does not offer the ${model} model. It offers: ` +
        catalogue.map((item) => item.id).join(', '),
    );
  }

  const lock = LOCKS[model];
  const target = join(cacheDir, model);
  let bytes = 0;
  let cached = 0;

  for (const [name, info] of Object.entries(lock.files)) {
    const destination = join(target, name);
    const expected = info.sha256;

    const existing = await readFile(destination).catch(() => undefined);
    if (existing) {
      const digest = await sha256Hex(existing);
      if (digest === expected) {
        cached += 1;
        continue;
      }
    }

    const address = url(server.base, `${entry.weightsRoot}${name}`);
    const response = await fetch(address, {
      headers: server.token ? { Authorization: `Bearer ${server.token}` } : {},
    });

    if (!response.ok) {
      throw new Error(
        `The weights file ${name} could not be fetched (${response.status} from ${address}).`,
      );
    }

    const content = Buffer.from(await response.arrayBuffer());
    const digest = await sha256Hex(content);

    if (digest !== expected) {
      throw new Error(
        `The fetched ${name} does not match the pinned version.\n` +
          `  expected sha256 ${expected}\n` +
          `  got      sha256 ${digest}\n` +
          'The download was incomplete or the file was changed on the way.',
      );
    }

    await mkdir(dirname(destination), { recursive: true });
    await writeFile(destination, content);
    bytes += content.length;
  }

  return {
    modelDir: target,
    modelPath: join(target, 'model.json'),
    bytes,
    cached,
  };
}
