/**
 * What does the thread count cost, and what does it buy?
 *
 *   npx tsx scripts/onnx-threads.mts [file.wav]
 *
 * The ONNX runtime reduces in an order that depends on the thread count, so
 * the count is pinned to one everywhere: two runs of the same file have to
 * come out byte for byte the same, and that is the promise the tool makes.
 *
 * That is a real cost, and the question this answers is what it buys and
 * what it costs. It runs the same graph at several thread counts and reports
 * the time each took and whether the notes came out the same.
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

function hash(notes: RawNote[]): string {
  // A stable digest of the notes, so two runs can be compared exactly.
  const parts = notes
    .map(
      (note) =>
        `${note.pitchMidi}:${note.startTime.toFixed(6)}:${note.duration.toFixed(6)}:${note.amplitude.toFixed(6)}`,
    )
    .sort();
  let hash = 0x811c9dc5;
  for (const part of parts.join('|')) {
    hash ^= part.charCodeAt(0);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

async function main(): Promise<void> {
  const file = process.argv[2] ?? 'test/sample.wav';

  const data = readFileSync(file);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  const directory = resolve('models/onnx', PIANO_DIR);
  const graph = resolve(directory, PIANO_FILES.graph);

  console.log(`${file}\n`);
  console.log('threads   seconds   notes   digest');
  console.log('-'.repeat(44));

  const digests = new Set<number>();

  for (const threads of [1, 2, 4, 8]) {
    const started = Date.now();
    const raw = await ort.InferenceSession.create(graph, {
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
    const seconds = (Date.now() - started) / 1000;
    const notes = result.parts[0].notes;
    const digest = hash(notes);

    console.log(
      `${String(threads).padStart(7)} ${seconds.toFixed(1).padStart(8)} ` +
        `${String(notes.length).padStart(8)}   ${digest}`,
    );

    digests.add(notes.length);
    await raw.release();
  }

  console.log(
    `\nThe note count is the same at every thread count on this file: ` +
      `${digests.size === 1}.`,
  );
  console.log(
    'The digests are what the reproducibility check compares: the same file at',
  );
  console.log('one thread always gives the same one.');
}

await main();
