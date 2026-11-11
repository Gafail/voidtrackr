/**
 * The whole conversion, with the memory read at every stage.
 *
 * The piano model's graph is not the only thing that uses memory, and on a
 * box with a gigabyte of it every stage matters.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { installTfjsNodeBackend } from '../src/core/backend';
import { convertAudio } from '../src/core/convert';
import { loadModel } from '../src/core/model-loader';
import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { analyzeDrums } from '../src/core/drums';

/** The resident memory of this process, in megabytes. */
function rss(): number {
  const status = readFileSync('/proc/self/status', 'utf8');
  const match = /VmRSS:\s+(\d+) kB/.exec(status);
  return Number(match?.[1] ?? 0) / 1024;
}

/** The highest this process has been. */
function peak(): number {
  const status = readFileSync('/proc/self/status', 'utf8');
  const match = /VmHWM:\s+(\d+) kB/.exec(status);
  return Number(match?.[1] ?? 0) / 1024;
}

async function main(): Promise<void> {
  const file = process.argv[2] ?? '/tmp/short.wav';

  console.log(`start              : ${rss().toFixed(0)} MB`);
  await installTfjsNodeBackend();
  console.log(`after the backend  : ${rss().toFixed(0)} MB`);

  const data = readFileSync(file);
  const buffer = data.buffer.slice(
    data.byteOffset,
    data.byteOffset + data.byteLength,
  ) as ArrayBuffer;

  const audio = await decodeAudio(buffer, nodeAudioDecoder);
  console.log(`after decoding     : ${rss().toFixed(0)} MB`);

  const samples = prepareMono(audio);
  console.log(`after the mono mix : ${rss().toFixed(0)} MB`);

  const model = await loadModel('piano', {
    modelDir: resolve('models', 'onnx'),
    modelPath: resolve('public', 'model', 'model.json'),
  });
  console.log(`after the model    : ${rss().toFixed(0)} MB`);

  analyzeDrums(samples, 22050);
  console.log(`after the drums    : ${rss().toFixed(0)} MB`);

  const output = await convertAudio(buffer, { detectDrums: false }, model, 'test');
  console.log(`after the model run: ${rss().toFixed(0)} MB`);
  console.log(`notes: ${output.result.tracks[0].notes.length}`);
  console.log(`peak               : ${peak().toFixed(0)} MB`);
}

await main();
