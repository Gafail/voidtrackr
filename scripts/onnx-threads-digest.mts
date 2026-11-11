/**
 * The same digests, from Node, so the browser can be compared with it.
 *
 *   npx tsx scripts/onnx-threads-digest.mts
 *
 * The browser test prints a digest of each of the piano graph's four
 * outputs, for a one-second 440 Hz tone. This prints the same digests from
 * Node, so the two runtimes can be compared directly: if they agree at one
 * thread and disagree at another, the thread count is part of the output and
 * the browser and the command line cannot both be trusted at once.
 */

import { resolve } from 'node:path';
import * as ort from 'onnxruntime-node';

async function digestsFor(threads: number): Promise<{ name: string; digest: string }[]> {
  const directory = resolve('models/onnx', 'piano');
  const session = await ort.InferenceSession.create(resolve(directory, 'piano_kong.onnx'), {
    intraOpNumThreads: threads,
    interOpNumThreads: 1,
    executionMode: 'sequential',
  });

  const samples = new Float32Array(160000);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = Math.sin((2 * Math.PI * 440 * i) / 16000) * 0.5;
  }

  const result = await session.run({
    [session.inputNames[0]]: new ort.Tensor('float32', samples, [1, 160000]),
  });

  const digests: { name: string; digest: string }[] = [];
  for (const name of ['frame_output', 'reg_onset_output', 'reg_offset_output', 'velocity_output']) {
    let hash = 0x811c9dc5;
    const values = result[name].data as Float32Array;
    for (let i = 0; i < values.length; i++) {
      hash ^= Math.round(values[i] * 1e6);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    digests.push({ name: name.replace('_output', ''), digest: hash.toString(16).padStart(8, '0') });
  }

  await session.release();
  return digests;
}

console.log('node, the piano graph, a one-second 440 Hz tone:\n');

for (const threads of [1, 2, 4, 8]) {
  console.log(`threads ${threads}`);
  for (const { name, digest } of await digestsFor(threads)) {
    console.log(`  ${name} ${digest}`);
  }
  console.log('');
}
