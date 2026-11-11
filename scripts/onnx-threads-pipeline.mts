/**
 * The whole conversion, at one thread and at eight.
 *
 *   npx tsx scripts/onnx-threads-pipeline.mts [file.wav]
 *
 * The graph is not the whole conversion: the drum detector, the tempo and
 * the key all run afterwards on the main thread and none of them can be
 * spread over more than one thread. So the speedup the graph enjoys is not
 * the speedup the conversion enjoys, and this measures the conversion.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import * as ort from 'onnxruntime-node';

import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { OnnxTensor } from '../src/core/onnx';
import { RawNote, Track } from '../src/core/types';
import { buildMidiFile } from '../src/core/midi';
import { estimateTempo, estimateKey } from '../src/core/tempo';
import { SAMPLE_RATE } from '../src/core/constants';
import { analyzeDrums, hitsToNotes } from '../src/core/drums';

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

  const { PianoTranscriptionModel } = await import('../src/core/piano');
  const model = new PianoTranscriptionModel(session as never, 'local');
  const result = await model.transcribe(samples);
  await raw.release();
  return result.parts[0].notes;
}

async function main(): Promise<void> {
  const file = process.argv[2] ?? '/tmp/short.wav';

  const data = readFileSync(file);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  console.log(`${file}, ${(samples.length / SAMPLE_RATE).toFixed(1)} s of audio\n`);
  console.log('threads   model s   rest s   total s   file');
  console.log('-'.repeat(58));

  const files: string[] = [];

  for (const threads of [1, 2, 4, 8]) {
    const started = Date.now();
    const notes = await transcribe(samples, threads);
    const afterModel = Date.now();

    // The stages that run after the model and cannot be threaded.
    const tracks: Track[] = [{ id: 'piano', name: 'Piano', kind: 'melody', notes }];
    const drums = analyzeDrums(samples, SAMPLE_RATE);
    if (drums.detected && drums.hits.length > 0) {
      tracks.push({ id: 'drums', name: 'Drums', kind: 'drums', notes: hitsToNotes(drums.hits) });
    }
    const duration = Math.max(1, ...notes.map((note) => note.startTime + note.duration));
    const bytes = buildMidiFile(
      tracks,
      estimateTempo(notes, duration),
      estimateKey(notes),
    );
    const total = Date.now();

    files.push(createHash('sha256').update(bytes).digest('hex').slice(0, 12));

    console.log(
      `${String(threads).padStart(7)} ` +
        `${((afterModel - started) / 1000).toFixed(1).padStart(8)} ` +
        `${((total - afterModel) / 1000).toFixed(1).padStart(8)} ` +
        `${((total - started) / 1000).toFixed(1).padStart(8)}   ${files[files.length - 1]}`,
    );
  }

  console.log(`\nthe same file at every thread count: ${new Set(files).size === 1}`);
}

await main();
