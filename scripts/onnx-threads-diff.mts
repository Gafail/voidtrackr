/**
 * How much does the thread count change the notes?
 *
 *   npx tsx scripts/onnx-threads-diff.mts [file.wav]
 *
 * The ONNX runtime reduces in an order that depends on the thread count, so
 * the count is pinned to one. That is the reason the pin exists, but a pin
 * should be justified by a difference that matters, and the difference
 * should be measured rather than assumed.
 *
 * This transcribes the same audio at several thread counts and reports, for
 * each, the time it took and how the notes differ from the one-thread run:
 * the pitches that moved, and how far the timings moved.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import * as ort from 'onnxruntime-node';

import { decodeAudio } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { prepareMono } from '../src/core/audio';
import { PianoTranscriptionModel, PIANO_DIR, PIANO_FILES } from '../src/core/piano';
import { OnnxTensor } from '../src/core/onnx';
import { RawNote } from '../src/core/types';

/**
 * The shape of the notes, for comparing two runs.
 *
 * The times are rounded to a tenth of a millisecond, so a difference in the
 * last bits of a float does not count as a difference.
 */
function fingerprint(notes: RawNote[]): string {
  return notes
    .map(
      (note) =>
        `${note.pitchMidi}@${Math.round(note.startTime * 10000)}:${Math.round(note.duration * 10000)}`,
    )
    .sort()
    .join('\n');
}

async function run(threads: number, samples: Float32Array): Promise<RawNote[]> {
  const directory = resolve('models/onnx', PIANO_DIR);
  const raw = await ort.InferenceSession.create(resolve(directory, PIANO_FILES.graph), {
    intraOpNumThreads: threads,
    interOpNumThreads: 1,
    executionMode: 'sequential',
  });

  // The model wants the shared interface's tensors, so the raw session is
  // wrapped the way `src/core/onnx-node.ts` wraps it.
  const session = {
    inputNames: raw.inputNames,
    outputNames: raw.outputNames,
    async run(feeds: Record<string, OnnxTensor>) {
      const tensors: Record<string, ort.Tensor> = {};
      for (const [name, tensor] of Object.entries(feeds)) {
        tensors[name] = new ort.Tensor(
          tensor.type as 'float32' | 'int32' | 'int64' | 'bool',
          tensor.data as Float32Array,
          tensor.dims,
        );
      }
      const result = await raw.run(tensors);
      const output: Record<string, OnnxTensor> = {};
      for (const [name, tensor] of Object.entries(result)) {
        output[name] = {
          type: tensor.type as OnnxTensor['type'],
          data: tensor.data as Float32Array,
          dims: tensor.dims,
        };
      }
      return output;
    },
  };

  const model = new PianoTranscriptionModel(session, 'local');
  const result = await model.transcribe(samples);
  await raw.release();
  return result.parts[0].notes;
}

async function main(): Promise<void> {
  const file = process.argv[2] ?? 'test/sample.wav';

  const data = readFileSync(file);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  console.log(`${file}\n`);
  console.log('threads   seconds   notes   same as 1 thread   pitch diffs   worst timing   velocity');
  console.log('-'.repeat(94));

  const runs = new Map<number, { notes: RawNote[]; seconds: number }>();
  const counts = [1, 2, 4, 8];

  for (const threads of counts) {
    const started = Date.now();
    const notes = await run(threads, samples);
    const seconds = (Date.now() - started) / 1000;
    runs.set(threads, { notes, seconds });
  }

  const reference = runs.get(1)!;
  const referencePrint = fingerprint(reference.notes);

  for (const threads of counts) {
    const { notes, seconds } = runs.get(threads)!;
    const same = fingerprint(notes) === referencePrint;

    // How many of the reference's notes are missing, and how far did the
    // timings move for the ones that are still there?
    const byKey = new Map(
      notes.map((note) => [
        `${note.pitchMidi}@${Math.round(note.startTime * 1000)}`,
        note,
      ]),
    );
    let moved = 0;
    let worst = 0;
    for (const note of reference.notes) {
      const key = `${note.pitchMidi}@${Math.round(note.startTime * 1000)}`;
      const found = byKey.get(key);
      if (!found) {
        moved += 1;
        continue;
      }
      const difference = Math.abs(found.startTime - note.startTime);
      if (difference > 0.0001) {
        worst = Math.max(worst, difference);
        moved += 1;
      }
    }

    // The last difference there might be is the loudness. The MIDI file
    // carries it as a velocity from 0 to 127, so this asks whether the
    // difference survives that rounding — that is, whether the exported
    // bytes would differ.
    const velocities = new Map(
      notes.map((note) => [
        `${note.pitchMidi}@${Math.round(note.startTime * 1000)}`,
        Math.round(Math.max(0, Math.min(1, note.amplitude)) * 127),
      ]),
    );
    let velocityDiffs = 0;
    for (const note of reference.notes) {
      const key = `${note.pitchMidi}@${Math.round(note.startTime * 1000)}`;
      const found = velocities.get(key);
      if (found !== undefined && found !== Math.round(Math.max(0, Math.min(1, note.amplitude)) * 127)) {
        velocityDiffs += 1;
      }
    }

    console.log(
      `${String(threads).padStart(7)} ${seconds.toFixed(1).padStart(8)} ` +
        `${String(notes.length).padStart(8)} ${String(same).padStart(16)} ` +
        `${String(moved).padStart(13)} ${(worst * 1000).toFixed(3).padStart(12)} ms` +
        ` ${String(velocityDiffs).padStart(10)}`,
    );
  }

  console.log(
    '\nthe same column compares the notes at a tenth of a millisecond, which is',
  );
  console.log('the precision the MIDI file can carry anyway. The last column is',
  );
  console.log('the velocity differences that survive being written as 0 to 127.');
}

await main();
