/**
 * Does the exported file change with the thread count?
 *
 *   npx tsx scripts/onnx-threads-midi.mts [file.wav]
 *
 * The runtime reduces in an order that depends on the thread count, so the
 * count is pinned to one. The promise the tool makes is that two runs of the
 * same file produce the same bytes, and this asks what the pin is actually
 * protecting: the notes, or the file.
 *
 * It builds the MIDI file at each thread count and compares the bytes, which
 * is what the reproducibility check compares.
 */

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import * as ort from 'onnxruntime-node';

import { decodeAudio } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { prepareMono } from '../src/core/audio';
import { PianoTranscriptionModel, PIANO_DIR, PIANO_FILES } from '../src/core/piano';
import { OnnxTensor } from '../src/core/onnx';
import { RawNote, Track } from '../src/core/types';
import { buildMidiFile } from '../src/core/midi';
import { estimateTempo, estimateKey } from '../src/core/tempo';

async function run(threads: number, samples: Float32Array): Promise<RawNote[]> {
  const directory = resolve('models/onnx', PIANO_DIR);
  const raw = await ort.InferenceSession.create(resolve(directory, PIANO_FILES.graph), {
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

  const counts = [1, 2, 4, 8];
  const hashes: string[] = [];

  console.log(`${file}\n`);
  console.log('threads   seconds   midi bytes   sha256');
  console.log('-'.repeat(76));

  for (const threads of counts) {
    const started = Date.now();
    const notes = await run(threads, samples);
    const seconds = (Date.now() - started) / 1000;

    const track: Track = { id: 'piano', name: 'Piano', kind: 'melody', notes };
    const duration = Math.max(1, ...notes.map((note) => note.startTime + note.duration));
    const tempo = estimateTempo(notes, duration);
    const key = estimateKey(notes);

    const bytes = buildMidiFile([track], tempo, key);
    const digest = createHash('sha256').update(bytes).digest('hex').slice(0, 16);
    hashes.push(digest);

    console.log(
      `${String(threads).padStart(7)} ${seconds.toFixed(1).padStart(8)} ` +
        `${String(bytes.length).padStart(11)}   ${digest}`,
    );
  }

  const same = hashes.every((hash) => hash === hashes[0]);
  console.log(`\nevery thread count gives the same file: ${same}`);
}

await main();
