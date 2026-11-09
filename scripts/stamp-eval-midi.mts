/**
 * Add the provenance stamp to the MIDI files the evaluation wrote.
 *
 * The evaluation called the MIDI writer without the stamp, so the files in
 * the report carry the notes but not the record of which pipeline and which
 * weights made them. The tool's own exports always carry it. This brings
 * the report's files back in line.
 *
 *   npx tsx scripts/stamp-eval-midi.mts eval/midi
 */
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import nodeUtil from 'node:util';

const mutableUtil = nodeUtil as typeof nodeUtil & { isNullOrUndefined?: (value: unknown) => boolean };
if (typeof mutableUtil.isNullOrUndefined !== 'function') {
  mutableUtil.isNullOrUndefined = (value: unknown) => value === null || value === undefined;
}
// The native backend has to be in place before TensorFlow.js tries to read
// the Basic Pitch weights, because it is the one that can read a file:// URL.
await import('@tensorflow/tfjs-node');

import ToneMidi from '@tonejs/midi';

import { loadPinnedModel } from '../src/core/basic-pitch';
import modelLock from '../src/core/model.lock.json';
import { ModelLock, ModelSource } from '../src/core/model';
import { OkarinaModel, OKARINA_FILES } from '../src/core/okarina';
import okarinaLock from '../src/core/okarina.lock.json';
import { nodeOnnxRunner } from '../src/core/onnx-node';
import { PianoTranscriptionModel, PIANO_DIR, PIANO_FILES } from '../src/core/piano';
import pianoLock from '../src/core/piano.lock.json';
import { TsumugiModel, TSUMUGI_DIR, TSUMUGI_FILES } from '../src/core/tsumugi';
import tsumugiLock from '../src/core/tsumugi.lock.json';
import { YourMt3Model, YOURMT3_DIR, YOURMT3_FILES } from '../src/core/yourmt3';
import yourmt3Lock from '../src/core/yourmt3.lock.json';
import { verifyOnnxWeights } from '../src/core/onnx-weights';
import { provenanceFor } from '../src/core/version';

function midiConstructor(): typeof ToneMidi.Midi {
  const module = ToneMidi as unknown as {
    Midi?: typeof ToneMidi.Midi;
    default?: typeof ToneMidi.Midi | { Midi?: typeof ToneMidi.Midi };
  };
  if (typeof module.Midi === 'function') return module.Midi;
  if (module.default) {
    if (typeof module.default === 'function') return module.default;
    if (typeof module.default.Midi === 'function') return module.default.Midi;
  }
  throw new Error('Could not load @tonejs/midi');
}

const DIR = resolve(process.argv[2] ?? 'eval/midi');
const MODEL_DIR = resolve('models', 'onnx');

async function checkOnnx(
  lock: unknown,
  directory: string,
  files: string[],
): Promise<string> {
  return verifyOnnxWeights(
    async (path: string) => {
      const bytes = await readFile(join(MODEL_DIR, path));
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ) as ArrayBuffer;
    },
    lock as { files: Record<string, { sha256: string }> },
    { dir: directory, files },
    'npm run model',
  );
}

/** Load Okarina with the same fingerprint the command line gives it. */
async function loadOkarina(): Promise<OkarinaModel> {
  const lock = okarinaLock as unknown as { files: Record<string, { sha256: string }> };
  const digests: string[] = [];
  for (const name of [OKARINA_FILES.frontEnd, OKARINA_FILES.m]) {
    const expected = lock.files[name]?.sha256;
    if (!expected) throw new Error(`The lock file has no hash for ${name}.`);
    const bytes = await readFile(join(MODEL_DIR, name));
    const digest = createHash('sha256').update(bytes).digest('hex');
    if (digest !== expected) {
      throw new Error(`The model's ${name} does not match the pinned version.`);
    }
    digests.push(digest);
  }
  const fingerprint = createHash('sha256').update(digests.join('')).digest('hex');
  return OkarinaModel.load(nodeOnnxRunner, MODEL_DIR, 'm', fingerprint);
}

/** The Basic Pitch weights are read off disk rather than over HTTP. */
function nodeModelSource(modelPath: string): ModelSource {
  const directory = dirname(modelPath);
  return {
    url: pathToFileURL(modelPath).href,
    read: async (name: string) => {
      const bytes = await readFile(join(directory, name));
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ) as ArrayBuffer;
    },
  };
}

async function main(): Promise<void> {
  const tsumugi = await TsumugiModel.load(
    nodeOnnxRunner,
    join(MODEL_DIR, TSUMUGI_DIR),
    await checkOnnx(tsumugiLock, TSUMUGI_DIR, [
      TSUMUGI_FILES.graph,
      TSUMUGI_FILES.heads,
      TSUMUGI_FILES.headsJson,
    ]),
  );
  const piano = await PianoTranscriptionModel.load(
    nodeOnnxRunner,
    join(MODEL_DIR, PIANO_DIR),
    await checkOnnx(pianoLock, PIANO_DIR, [PIANO_FILES.graph]),
  );
  const yourmt3 = await YourMt3Model.load(
    nodeOnnxRunner,
    join(MODEL_DIR, YOURMT3_DIR),
    await checkOnnx(yourmt3Lock, YOURMT3_DIR, [
      YOURMT3_FILES.encoder,
      YOURMT3_FILES.decoder,
      YOURMT3_FILES.vocab,
    ]),
  );

  const stamps: Record<string, string> = {
    tsumugi: provenanceFor(tsumugi),
    piano: provenanceFor(piano),
    yourmt3: provenanceFor(yourmt3),
    'basic-pitch': provenanceFor(
      await loadPinnedModel(
        nodeModelSource(resolve('public/model/model.json')),
        modelLock as ModelLock,
      ),
    ),
    okarina: provenanceFor(await loadOkarina()),
  };

  const files = (await readdir(DIR)).filter((f) => f.endsWith('.mid'));
  let stamped = 0;
  let skipped = 0;

  for (const file of files) {
    const stem = file.slice(0, -4);
    const dot = stem.lastIndexOf('.');
    const model = stem.slice(dot + 1);
    const stamp = stamps[model];
    if (!stamp) {
      skipped += 1;
      continue;
    }

    const path = join(DIR, file);
    const bytes = await readFile(path);
    const midi = new (midiConstructor())(
      bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ) as ArrayBuffer,
    );

    // Already stamped? Leave it alone.
    const already = midi.header.meta.some(
      (m) => m.type === 'text' && (m as { text?: string }).text?.startsWith('voidtrackr'),
    );
    if (already) {
      skipped += 1;
      continue;
    }

    midi.header.meta.push({ text: stamp, type: 'text', ticks: 0 });
    await writeFile(path, Buffer.from(midi.toArray()));
    stamped += 1;
  }

  console.log(`stamped ${stamped} files, skipped ${skipped}`);
}

await main();
