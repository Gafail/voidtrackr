/**
 * Checks for the model seam.
 *
 * The point of the seam is that a model can be swapped without the rest of
 * the pipeline changing. These tests cover that: the catalogue is right, a
 * stand-in model can be plugged in, and the real models behave.
 */

import nodeUtil from 'node:util';

const mutableUtil = nodeUtil as typeof nodeUtil & { isNullOrUndefined?: (value: unknown) => boolean };
if (typeof mutableUtil.isNullOrUndefined !== 'function') {
  mutableUtil.isNullOrUndefined = (value: unknown) => value === null || value === undefined;
}
await import('@tensorflow/tfjs-node');

import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';

import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { ModelPart, ModelTranscription, TranscriptionModel } from '../src/core/model';
import { DEFAULT_MODEL, isModelId, listModels, MODELS } from '../src/core/registry';
import { estimateBeatGrid } from '../src/core/beat';
import { transcribe } from '../src/core/pipeline';
import { OkarinaModel } from '../src/core/okarina';
import { nodeOnnxRunner } from '../src/core/onnx-node';
import { RawNote } from '../src/core/types';

// --- The catalogue -----------------------------------------------------

test('the catalogue lists every model', () => {
  const models = listModels();
  assert.equal(models.length, Object.keys(MODELS).length);
  for (const model of models) {
    assert.ok(model.displayName.length > 0, `${model.id} has no name`);
    assert.ok(model.license.length > 0, `${model.id} has no licence`);
    assert.ok(model.summary.length > 0, `${model.id} has no summary`);
  }
});

test('the default model is one of the listed models', () => {
  assert.ok(isModelId(DEFAULT_MODEL));
  assert.ok(listModels().some((model) => model.id === DEFAULT_MODEL));
});

test('an unknown model name is not accepted', () => {
  assert.equal(isModelId('basic-pitch'), true);
  assert.equal(isModelId('not-a-model'), false);
});

// --- The seam ----------------------------------------------------------

/** A model that always reports the same notes. */
function stubModel(notes: RawNote[]): TranscriptionModel {
  return {
    id: 'stub',
    displayName: 'Stub',
    license: 'none',
    fingerprint: 'stub',
    inputSampleRate: 22050,
    async transcribe(): Promise<ModelTranscription> {
      const part: ModelPart = { id: 'stub', name: 'Stub', kind: 'melody', notes };
      return { parts: [part], log: ['stub ran'] };
    },
  };
}

test('a stand-in model can be plugged into the pipeline', async () => {
  const notes: RawNote[] = [
    { pitchMidi: 60, startTime: 0, duration: 0.5, amplitude: 0.8 },
    { pitchMidi: 64, startTime: 0.5, duration: 0.5, amplitude: 0.8 },
    { pitchMidi: 67, startTime: 1, duration: 0.5, amplitude: 0.8 },
  ];

  const result = await transcribe(new Float32Array(22050 * 2), {
    model: stubModel(notes),
    detectDrums: false,
  });

  assert.equal(result.tracks.length, 1);
  assert.equal(result.tracks[0].notes.length, 3);
  assert.equal(result.tracks[0].notes[1].startTime, 0.5);
  assert.deepEqual(result.log, ['stub ran']);
});

test('the pipeline resamples to the rate the model asks for', async () => {
  let seenRate = 0;
  let seenLength = 0;

  const model: TranscriptionModel = {
    id: 'rate-check',
    displayName: 'Rate check',
    license: 'none',
    fingerprint: 'rate-check',
    inputSampleRate: 16000,
    async transcribe(samples) {
      seenRate = samples.length;
      seenLength = 1;
      return { parts: [], log: [] };
    },
  };

  const result = await transcribe(new Float32Array(22050 * 3), { model, detectDrums: false });
  assert.equal(seenLength, 1);
  // Three seconds at 22.05 kHz becomes three seconds at 16 kHz.
  assert.ok(Math.abs(seenRate - 48000) < 100, `the model got ${seenRate} samples`);
  assert.equal(result.tracks.length, 0);
});

// --- The beat tracker --------------------------------------------------

test('the beat tracker finds a known tempo', async () => {
  const data = await readFile(resolve('test/sample.wav'));
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  const grid = estimateBeatGrid(samples, 22050);

  // The test recording is written at 120 BPM.
  assert.ok(
    Math.abs(grid.bpm - 120) < 1,
    `the tracker found ${grid.bpm} BPM, expected about 120`,
  );
  assert.ok(grid.confidence > 0.5, `confidence was only ${grid.confidence}`);
});

// --- Okarina -----------------------------------------------------------

const OKARINA_DIR = resolve('models', 'onnx');

async function okarinaWeightsPresent(): Promise<boolean> {
  try {
    await stat(resolve(OKARINA_DIR, 'cqt156.onnx'));
    await stat(resolve(OKARINA_DIR, 'okarina_m.onnx'));
    return true;
  } catch {
    return false;
  }
}

test('Okarina transcribes the test recording', async (t) => {
  if (!(await okarinaWeightsPresent())) {
    t.skip('the Okarina weights are not fetched; run npm run model:okarina');
    return;
  }

  const data = await readFile(resolve('test/sample.wav'));
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  const model = await OkarinaModel.load(nodeOnnxRunner, OKARINA_DIR, 'm');

  // Through the pipeline, so the resampling to the model's rate happens the
  // way it does in real use.
  const result = await transcribe(samples, { model, detectDrums: false });

  const notes = result.tracks.flatMap((track) => track.notes);
  assert.ok(notes.length > 0, 'Okarina found no notes at all');

  // Every note has to be inside the recording.
  for (const note of notes) {
    assert.ok(note.startTime >= 0, `a note starts before the recording`);
    assert.ok(
      note.startTime + note.duration <= audio.duration + 0.01,
      `a note runs past the end of the recording`,
    );
    assert.ok(note.pitchMidi >= 21 && note.pitchMidi <= 108, `odd pitch ${note.pitchMidi}`);
  }
});

test('Okarina gives the same answer twice', async (t) => {
  if (!(await okarinaWeightsPresent())) {
    t.skip('the Okarina weights are not fetched; run npm run model:okarina');
    return;
  }

  const data = await readFile(resolve('test/sample.wav'));
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  const model = await OkarinaModel.load(nodeOnnxRunner, OKARINA_DIR, 'm');
  const first = await model.transcribe(samples);
  const second = await model.transcribe(samples);

  const flat = (result: ModelTranscription) =>
    result.parts.flatMap((part) =>
      part.notes.map((note) => `${part.id}|${note.pitchMidi}|${note.startTime}|${note.duration}`),
    );

  assert.deepEqual(flat(first), flat(second));
});
