/**
 * How much memory does one model need?
 *
 *   npx tsx scripts/memory-models.mts [model] [file.wav]
 *
 * The box the API runs on has a gigabyte of it, so this is what decides
 * which models it can serve. The peak is read from the kernel, which knows
 * the high-water mark even after the memory has been given back.
 *
 * One model per process: the high-water mark is for the whole process, so
 * measuring two models in one process gives the second one the first one's
 * peak.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { loadModel } from '../src/core/model-loader';
import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { ModelId } from '../src/core/registry';

/** The resident memory of this process, in megabytes. */
function rss(): number {
  const status = readFileSync('/proc/self/status', 'utf8');
  return Number(/VmRSS:\s+(\d+) kB/.exec(status)?.[1] ?? 0) / 1024;
}

/** The highest this process has been, in megabytes. */
function peak(): number {
  const status = readFileSync('/proc/self/status', 'utf8');
  return Number(/VmHWM:\s+(\d+) kB/.exec(status)?.[1] ?? 0) / 1024;
}

async function main(): Promise<void> {
  const model = (process.argv[2] ?? 'piano') as ModelId;
  const file = process.argv[3] ?? '/tmp/short.wav';

  const data = readFileSync(file);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  const before = rss();
  const loaded = await loadModel(model, {
    modelDir: resolve('models', 'onnx'),
    modelPath: resolve('public', 'model', 'model.json'),
  });
  const afterLoad = rss();

  const result = await loaded.transcribe(samples);
  const notes = result.parts.flatMap((part) => part.notes).length;

  console.log(
    `${model.padEnd(10)} weights ${(afterLoad - before).toFixed(0).padStart(5)} MB   ` +
      `settled ${rss().toFixed(0).padStart(5)} MB   peak ${peak().toFixed(0).padStart(5)} MB   ` +
      `${notes} notes`,
  );

  loaded.dispose?.();
}

await main();
