/**
 * Measure how reproducible the pipeline actually is.
 *
 *   npm run check:reproducibility
 *
 * Compares:
 *   1. the same command run twice
 *   2. the native tfjs-node backend vs the pure-JS CPU backend
 *      (the CPU backend is what the browser falls back to)
 *   3. the note lists, not just the file bytes
 *
 * The point is to keep the claim honest. If a change to the pipeline makes
 * the output wobble, this is where it shows up.
 */

import nodeUtil from 'node:util';

const mutableUtil = nodeUtil as typeof nodeUtil & { isNullOrUndefined?: (value: unknown) => boolean };
if (typeof mutableUtil.isNullOrUndefined !== 'function') {
  mutableUtil.isNullOrUndefined = (value: unknown) => value === null || value === undefined;
}
await import('@tensorflow/tfjs-node');

import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import * as tf from '@tensorflow/tfjs';

import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { loadPinnedModel } from '../src/core/basic-pitch';
import { ModelLock, ModelSource } from '../src/core/model';
import modelLock from '../src/core/model.lock.json';
import { transcribe } from '../src/core/pipeline';
import { buildMidiFile } from '../src/core/midi';
import { provenanceFor } from '../src/core/version';

const FILE = resolve('test/sample.wav');
const MODEL_PATH = resolve('public/model/model.json');
const LOCK = modelLock as ModelLock;

/** Read the model off disk, the way the command line does. */
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

async function run(backend: string) {
  await tf.setBackend(backend);
  await tf.ready();

  const model = await loadPinnedModel(nodeModelSource(MODEL_PATH), LOCK);
  const data = await readFile(FILE);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const result = await transcribe(prepareMono(audio), { model });
  const provenance = provenanceFor(model);
  model.dispose();

  const bytes = buildMidiFile(result.tracks, result.tempo, result.key, { provenance });
  const notes = result.tracks.flatMap((t) =>
    t.notes.map(
      (n) => `${t.name}:${n.pitchMidi}@${n.startTime.toFixed(4)}+${n.duration.toFixed(4)}`,
    ),
  );

  return {
    backend: tf.getBackend(),
    midiHash: createHash('sha256').update(bytes).digest('hex').slice(0, 16),
    noteHash: createHash('sha256').update(notes.join('|')).digest('hex').slice(0, 16),
    noteCount: notes.length,
    notes,
    tempo: result.tempo.bpm,
    key: `${result.key.tonic}:${result.key.mode}`,
  };
}

const nat1 = await run('tensorflow');
const nat2 = await run('tensorflow');
const cpu1 = await run('cpu');
const cpu2 = await run('cpu');

console.log('backend   midi-hash     note-hash     notes  tempo  key');
for (const result of [nat1, nat2, cpu1, cpu2]) {
  console.log(
    `${result.backend.padEnd(9)} ${result.midiHash}  ${result.noteHash}  ${String(
      result.noteCount,
    ).padStart(4)}  ${result.tempo}  ${result.key}`,
  );
}

console.log('\nsame backend, twice:');
console.log('  midi identical:', nat1.midiHash === nat2.midiHash);
console.log('  notes identical:', nat1.noteHash === nat2.noteHash);
console.log('  cpu vs cpu midi identical:', cpu1.midiHash === cpu2.midiHash);

console.log('\ndifferent backends (native vs pure JS):');
console.log('  midi identical:', nat1.midiHash === cpu1.midiHash);
console.log('  notes identical:', nat1.noteHash === cpu1.noteHash);

if (nat1.noteHash !== cpu1.noteHash) {
  const differences: string[] = [];
  const max = Math.max(nat1.notes.length, cpu1.notes.length);
  for (let i = 0; i < max; i++) {
    if (nat1.notes[i] !== cpu1.notes[i]) {
      differences.push(
        `    node: ${nat1.notes[i] ?? '(none)'}\n    cpu : ${cpu1.notes[i] ?? '(none)'}`,
      );
    }
  }
  console.log(`  ${differences.length} differing notes:`);
  console.log(differences.slice(0, 20).join('\n'));
}
