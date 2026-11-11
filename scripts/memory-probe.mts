/**
 * Where does the memory go?
 *
 * The piano model's graph is run one window at a time, but the process
 * grows with the length of the recording. This prints the resident memory
 * after every window, so it is clear whether the growth is per window or
 * one big step.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import * as ort from 'onnxruntime-node';

import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';

const SEGMENT_SAMPLES = 160000; // the piano model's window: ten seconds at 16 kHz
const HOP_SAMPLES = 80000;

/** The resident memory of this process, in megabytes. */
function rss(): number {
  const status = readFileSync('/proc/self/status', 'utf8');
  const match = /VmRSS:\s+(\d+) kB/.exec(status);
  return Number(match?.[1] ?? 0) / 1024;
}

async function main(): Promise<void> {
  const file = process.argv[2] ?? '/tmp/short.wav';
  const data = readFileSync(file);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  // The piano model wants 16 kHz, not the pipeline's 22.05 kHz.
  const samples = prepareMono(audio, 16000);

  console.log(`${file}: ${(samples.length / 16000).toFixed(1)} s`);
  console.log(`after decoding: ${rss().toFixed(0)} MB`);

  const graph = resolve('models/onnx/piano/piano_kong.onnx');
  const arena = process.argv[3] !== 'noarena';
  const session = await ort.InferenceSession.create(graph, {
    intraOpNumThreads: 4,
    interOpNumThreads: 1,
    executionMode: 'sequential',
    enableCpuMemArena: arena,
    enableMemPattern: arena,
  });
  console.log(`arena: ${arena ? 'on' : 'off'}`);
  console.log(`after loading the model: ${rss().toFixed(0)} MB`);

  const window = new Float32Array(SEGMENT_SAMPLES);
  let count = 0;

  for (let start = 0; start + SEGMENT_SAMPLES <= samples.length; start += HOP_SAMPLES) {
    window.fill(0);
    window.set(samples.subarray(start, start + SEGMENT_SAMPLES));

    const input = new ort.Tensor('float32', window, [1, SEGMENT_SAMPLES]);
    const out = await session.run({ [session.inputNames[0]]: input });

    // The same thing the pipeline does with them.
    let total = 0;
    for (const name of ['reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output']) {
      total += (out[name] as ort.Tensor).data.length;
    }

    count += 1;
    if (count <= 6 || count % 10 === 0) {
      console.log(
        `  window ${String(count).padStart(3)}: ${rss().toFixed(0)} MB ` +
          `(${total} values read)`,
      );
    }
  }

  const status = readFileSync('/proc/self/status', 'utf8');
  const hwm = Number(/VmHWM:\s+(\d+) kB/.exec(status)?.[1] ?? 0) / 1024;
  console.log(`after ${count} windows: ${rss().toFixed(0)} MB (peak ${hwm.toFixed(0)} MB)`);
  await session.release();
}

await main();
