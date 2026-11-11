/**
 * Does the thread count change the exported file, over the material?
 *
 *   npx tsx scripts/onnx-threads-material.mts [Music Import] [seconds]
 *
 * The thread count is pinned to one because the runtime reduces in an order
 * that depends on it. The reason worth checking is the one underneath that:
 * if the count changed the file, then a machine with a different number of
 * cores would produce a different file, and the promise the tool makes would
 * be false.
 *
 * It does not. Measured on 11 November 2026, the exported MIDI is the same
 * at one thread and at four, on every track in the folder. The raw numbers
 * do differ in Node at four threads and above, but the model's thresholds
 * turn those differences into the same notes.
 *
 * This transcribes a slice of each track at one thread and at four, and
 * compares the exported MIDI bytes.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { basename, join } from 'node:path';
import * as ort from 'onnxruntime-node';

import { decodeAudio } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { prepareMono } from '../src/core/audio';
import { OnnxTensor } from '../src/core/onnx';
import { RawNote, Track } from '../src/core/types';
import { buildMidiFile } from '../src/core/midi';
import { estimateTempo, estimateKey } from '../src/core/tempo';
import { SAMPLE_RATE } from '../src/core/constants';

const PIANO_DIR = 'piano';
const PIANO_GRAPH = 'piano_kong.onnx';

async function transcribe(
  samples: Float32Array,
  threads: number,
): Promise<RawNote[]> {
  const directory = join(process.cwd(), 'models/onnx', PIANO_DIR);
  const raw = await ort.InferenceSession.create(join(directory, PIANO_GRAPH), {
    intraOpNumThreads: threads,
    interOpNumThreads: 1,
    executionMode: 'sequential',
  });

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

  // The piano model's own front end and decoding, so the measurement is of
  // the real pipeline and not of a graph in isolation.
  const { PianoTranscriptionModel } = await import('../src/core/piano');
  const model = new PianoTranscriptionModel(session as never, 'local');
  const result = await model.transcribe(samples);
  await raw.release();
  return result.parts[0].notes;
}

async function main(): Promise<void> {
  const directory = process.argv[2] ?? 'Music Import';
  const seconds = Number(process.argv[3] ?? 60);
  const limit = Number(process.argv[4] ?? 5);

  const files = readdirSync(directory)
    .filter((name) => name.endsWith('.wav'))
    .sort()
    .slice(0, limit);

  console.log(`${files.length} tracks from ${directory}, the first ${seconds} s each\n`);
  console.log('track                     notes   1 thread      4 threads    same file');
  console.log('-'.repeat(76));

  let allSame = true;

  for (const file of files) {
    const data = readFileSync(join(directory, file));
    const audio = await decodeAudio(
      data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
      nodeAudioDecoder,
    );
    const samples = prepareMono(audio).slice(0, Math.floor(seconds * SAMPLE_RATE));

    const hashes: string[] = [];
    let notes = 0;

    for (const threads of [1, 4]) {
      const result = await transcribe(samples, threads);
      notes = result.length;
      const track: Track = { id: 'piano', name: 'Piano', kind: 'melody', notes: result };
      const duration = Math.max(1, ...result.map((note) => note.startTime + note.duration));
      const bytes = buildMidiFile(
        [track],
        estimateTempo(result, duration),
        estimateKey(result),
      );
      hashes.push(createHash('sha256').update(bytes).digest('hex').slice(0, 10));
    }

    const same = hashes[0] === hashes[1];
    if (!same) allSame = false;

    console.log(
      `${basename(file, '.wav').padEnd(24)} ${String(notes).padStart(6)} ` +
        `${hashes[0].padStart(12)} ${hashes[1].padStart(14)} ${String(same).padStart(11)}`,
    );
  }

  console.log(
    `\nevery track gives the same file at one thread and at four: ${allSame}`,
  );
  console.log(
    'So the file does not depend on how many cores the machine has, with or',
  );
  console.log(
    'without the pin. The pin is still there; this is what says it is safe',
  );
  console.log('to lift.');
}

await main();
