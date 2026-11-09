/**
 * Checks for the ONNX models.
 *
 * Three of them run through ONNX Runtime: Okarina, Tsumugi, YourMT3+ and
 * ByteDance's piano transcription model. These tests check the things that
 * matter about each one: that it finds notes, that the notes are inside the
 * recording, and that it gives the same answer twice — which is the whole
 * promise this project makes.
 *
 * A model whose weights are not fetched is skipped rather than failed, so a
 * checkout without them still runs the suite.
 */

import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';

import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { SAMPLE_RATE } from '../src/core/constants';
import { resample } from '../src/core/dsp';
import { ModelTranscription, TranscriptionModel } from '../src/core/model';
import { OkarinaModel } from '../src/core/okarina';
import { nodeOnnxRunner } from '../src/core/onnx-node';
import { PianoTranscriptionModel } from '../src/core/piano';
import { TsumugiModel } from '../src/core/tsumugi';
import { YourMt3Model } from '../src/core/yourmt3';
import { MODELS } from '../src/core/registry';

const ONNX_DIR = resolve('models/onnx');

/** Whether a model's weights are on disk. */
async function present(directory: string, file: string): Promise<boolean> {
  try {
    await stat(resolve(ONNX_DIR, directory, file));
    return true;
  } catch {
    return false;
  }
}

/** The test recording, as mono samples at the pipeline's rate. */
async function testRecording(): Promise<{ samples: Float32Array; duration: number }> {
  const data = await readFile(resolve('test/sample.wav'));
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  return { samples: prepareMono(audio), duration: audio.duration };
}

/**
 * The test recording at the rate a model asks for.
 *
 * The pipeline resamples before it hands the audio over, so a test that
 * calls the model directly has to do the same — otherwise a 16 kHz model
 * reads a 22 kHz recording as if it were slower than it is.
 */
async function recordingFor(
  model: TranscriptionModel,
): Promise<{ samples: Float32Array; duration: number }> {
  const { samples, duration } = await testRecording();
  if (model.inputSampleRate === SAMPLE_RATE) return { samples, duration };
  const resampled = resample(samples, SAMPLE_RATE, model.inputSampleRate);
  return { samples: resampled, duration: resampled.length / model.inputSampleRate };
}

/** A flat, comparable summary of a transcription. */
function flat(result: ModelTranscription): string[] {
  return result.parts.flatMap((part) =>
    part.notes.map(
      (note) => `${part.id}|${note.pitchMidi}|${note.startTime}|${note.duration}`,
    ),
  );
}

/** The checks every model has to pass. */
function checkNotes(
  result: ModelTranscription,
  duration: number,
  label: string,
): void {
  const notes = result.parts.flatMap((part) => part.notes);
  assert.ok(notes.length > 0, `${label} found no notes at all`);

  for (const note of notes) {
    assert.ok(note.startTime >= 0, `${label}: a note starts before the recording`);
    assert.ok(
      note.startTime + note.duration <= duration + 0.01,
      `${label}: a note runs past the end of the recording`,
    );
    assert.ok(
      note.pitchMidi >= 21 && note.pitchMidi <= 108,
      `${label}: odd pitch ${note.pitchMidi}`,
    );
  }
}

/** Run a model over the test recording and check it twice. */
async function checkModel(
  label: string,
  model: TranscriptionModel,
  samples: Float32Array,
  duration: number,
): Promise<void> {
  const first = await model.transcribe(samples);
  checkNotes(first, duration, label);

  const second = await model.transcribe(samples);
  assert.deepEqual(
    flat(first),
    flat(second),
    `${label} gave a different answer the second time`,
  );
}

// --- Okarina -------------------------------------------------------------

test('Okarina transcribes the test recording', async (t) => {
  if (!(await present('', 'okarina_m.onnx'))) {
    t.skip('the Okarina weights are not fetched; run npm run model:okarina');
    return;
  }

  const model = await OkarinaModel.load(nodeOnnxRunner, ONNX_DIR, 'm');
  const { samples, duration } = await recordingFor(model);
  await checkModel('Okarina', model, samples, duration);
});

// --- Tsumugi -------------------------------------------------------------

test('Tsumugi transcribes the test recording', async (t) => {
  if (!(await present('tsumugi', 'tsumugi_other_v1_5.pos32i_i32.onnx'))) {
    t.skip('the Tsumugi weights are not fetched; run npm run model:tsumugi');
    return;
  }

  const model = await TsumugiModel.load(nodeOnnxRunner, resolve(ONNX_DIR, 'tsumugi'));
  const { samples, duration } = await recordingFor(model);
  await checkModel('Tsumugi', model, samples, duration);
});

test('Tsumugi separates the instruments', async (t) => {
  if (!(await present('tsumugi', 'tsumugi_other_v1_5.pos32i_i32.onnx'))) {
    t.skip('the Tsumugi weights are not fetched; run npm run model:tsumugi');
    return;
  }

  const model = await TsumugiModel.load(nodeOnnxRunner, resolve(ONNX_DIR, 'tsumugi'));
  const { samples } = await recordingFor(model);
  const result = await model.transcribe(samples);

  // The test recording has a melody, a bass line and drums, so the model
  // should separate at least the melodic parts from the bass.
  assert.ok(result.parts.length >= 2, 'Tsumugi put everything in one part');
  for (const part of result.parts) {
    assert.ok(part.notes.length > 0, `Tsumugi produced an empty part: ${part.name}`);
  }
});

// --- YourMT3+ ------------------------------------------------------------

test('YourMT3+ transcribes the test recording', async (t) => {
  if (!(await present('yourmt3', 'encoder.onnx'))) {
    t.skip('the YourMT3+ weights are not fetched; run npm run model:yourmt3');
    return;
  }

  const model = await YourMt3Model.load(nodeOnnxRunner, resolve(ONNX_DIR, 'yourmt3'));
  const { samples } = await recordingFor(model);

  // The whole recording, not a slice. This model reads a segment as a
  // continuation of whatever came before it, so the first segment of a file
  // tends to come back empty — the same thing the upstream Python reference
  // does with the same bytes. Cutting the recording short would test that
  // quirk rather than the model.
  const first = await model.transcribe(samples);
  checkNotes(first, samples.length / model.inputSampleRate, 'YourMT3+');

  const second = await model.transcribe(samples);
  assert.deepEqual(
    flat(first),
    flat(second),
    'YourMT3+ gave a different answer the second time',
  );
});

// --- ByteDance piano transcription ----------------------------------------

test('the piano model transcribes the test recording', async (t) => {
  if (!(await present('piano', 'piano_kong.onnx'))) {
    t.skip('the piano weights are not fetched; run npm run model:piano');
    return;
  }

  const model = await PianoTranscriptionModel.load(nodeOnnxRunner, resolve(ONNX_DIR, 'piano'));
  const { samples, duration } = await recordingFor(model);
  await checkModel('the piano model', model, samples, duration);
});

// --- The licences --------------------------------------------------------

test('every model states its licence', () => {
  for (const model of Object.values(MODELS)) {
    assert.ok(model.license.length > 0, `${model.id} states no licence`);
    assert.ok(model.licenseUrl.startsWith('http'), `${model.id} has no licence URL`);
    assert.equal(
      typeof model.commercialUse,
      'boolean',
      `${model.id} does not say whether it can be used commercially`,
    );
  }
});

test('only Okarina is marked as non-commercial', () => {
  // Okarina's weights are CC BY-NC-SA 4.0. Everything else is permissive
  // about the weights themselves.
  assert.equal(MODELS.okarina.commercialUse, false);
  assert.ok(MODELS.okarina.license.includes('non-commercial'));

  for (const id of ['basic-pitch', 'tsumugi', 'yourmt3', 'piano'] as const) {
    assert.equal(MODELS[id].commercialUse, true, `${id} should be marked as usable`);
  }
});
