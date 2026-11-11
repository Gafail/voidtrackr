/**
 * Does the exported file change with the thread count, for every model?
 *
 *   npx tsx scripts/onnx-threads-models.mts [file.wav]
 *
 * The runtime reduces in an order that depends on the thread count, so the
 * count is pinned to one everywhere. That is a promise about the output, and
 * it should hold for every model in the catalogue, not only the default one.
 *
 * This loads each ONNX model the way the command line does, runs it at
 * several thread counts, builds the MIDI file and compares the bytes.
 */

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import * as ort from 'onnxruntime-node';

import { decodeAudio } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { prepareMono } from '../src/core/audio';
import { OnnxTensor } from '../src/core/onnx';
import { TranscriptionModel } from '../src/core/model';
import { Track } from '../src/core/types';
import { buildMidiFile } from '../src/core/midi';
import { estimateTempo, estimateKey } from '../src/core/tempo';

import { PianoTranscriptionModel, PIANO_DIR } from '../src/core/piano';
import { OkarinaModel, OKARINA_FILES } from '../src/core/okarina';
import { TsumugiModel, TSUMUGI_DIR, TSUMUGI_FILES } from '../src/core/tsumugi';
import { YourMt3Model, YOURMT3_DIR, YOURMT3_FILES } from '../src/core/yourmt3';

/** A runner that hands out a session with the given thread count. */
function runnerFor(threads: number): {
  createSession: (path: string) => Promise<unknown>;
  read: (path: string) => Promise<ArrayBuffer>;
} {
  return {
    async createSession(path: string) {
      const raw = await ort.InferenceSession.create(path, {
        intraOpNumThreads: threads,
        interOpNumThreads: 1,
        executionMode: 'sequential',
      });

      return {
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
        async release() {
          await raw.release();
        },
      };
    },
    async read(path: string) {
      const bytes = readFileSync(path);
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ) as ArrayBuffer;
    },
  };
}

interface ModelSetup {
  name: string;
  build: (runner: ReturnType<typeof runnerFor>) => Promise<TranscriptionModel>;
}

const MODELS: ModelSetup[] = [
  {
    name: 'piano',
    build: async (runner) =>
      PianoTranscriptionModel.load(
        runner as never,
        resolve('models/onnx', PIANO_DIR),
        'test',
      ),
  },
  {
    name: 'okarina (m)',
    build: async (runner) =>
      // Okarina's weights sit at the root of the weight directory, not in a
      // subdirectory of their own: the front end and the two note heads are
      // all in `models/onnx` itself.
      OkarinaModel.load(runner as never, resolve('models/onnx'), 'm', 'test'),
  },
  {
    name: 'tsumugi',
    build: async (runner) =>
      TsumugiModel.load(runner as never, resolve('models/onnx', TSUMUGI_DIR), 'test'),
  },
  {
    name: 'yourmt3',
    build: async (runner) =>
      YourMt3Model.load(runner as never, resolve('models/onnx', YOURMT3_DIR), 'test'),
  },
];

async function main(): Promise<void> {
  const file = process.argv[2] ?? 'test/sample.wav';

  const data = readFileSync(file);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  const counts = [1, 2, 4, 8];

  console.log(`${file}\n`);

  for (const setup of MODELS) {
    const hashes: string[] = [];
    const times: number[] = [];
    let failed = '';

    for (const threads of counts) {
      const started = Date.now();
      try {
        const model = await setup.build(runnerFor(threads));
        const result = await model.transcribe(samples);
        const notes = result.parts.flatMap((part) => part.notes);

        const track: Track = { id: 'm', name: 'M', kind: 'melody', notes };
        const duration = Math.max(1, ...notes.map((note) => note.startTime + note.duration));
        const bytes = buildMidiFile(
          [track],
          estimateTempo(notes, duration),
          estimateKey(notes),
        );
        hashes.push(createHash('sha256').update(bytes).digest('hex').slice(0, 12));
        times.push((Date.now() - started) / 1000);
      } catch (error) {
        failed = (error as Error).message;
        break;
      }
    }

    if (failed) {
      console.log(`${setup.name.padEnd(12)} could not run: ${failed}`);
      continue;
    }

    const same = hashes.every((hash) => hash === hashes[0]);
    const timesText = counts.map((_, index) => `${times[index].toFixed(1)}s`).join(' ');
    console.log(
      `${setup.name.padEnd(12)} ${counts.join('/')} threads: ${timesText.padEnd(24)} ` +
        `same file: ${same}`,
    );
  }

  console.log(
    '\nThe comparison is of the exported MIDI bytes, which is what the',
  );
  console.log('reproducibility check compares.');
}

void join;
void OKARINA_FILES;
void TSUMUGI_FILES;
void YOURMT3_FILES;
await main();
