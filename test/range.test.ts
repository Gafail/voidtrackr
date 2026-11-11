/**
 * The frequency range: which notes it keeps.
 *
 * The range is a selection over the notes a model found, never a filter on
 * the audio, so these tests are about the selection itself. The pipeline's
 * use of it is checked in `pipeline.test.ts`.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { noteName } from '../src/core/constants';
import {
  describeRange,
  filterByRange,
  MAX_FREQUENCY,
  MIN_FREQUENCY,
  normaliseRange,
  noteInRange,
  pitchAtOrAbove,
  pitchAtOrBelow,
} from '../src/core/range';
import { RawNote } from '../src/core/types';

function note(pitchMidi: number): RawNote {
  return { pitchMidi, startTime: 0, duration: 1, amplitude: 0.8 };
}

test('a note is kept when its own fundamental is inside the range', () => {
  // A3 is 220 Hz and A4 is 440 Hz.
  assert.equal(noteInRange(57, 'melody', { low: 0, high: 440 }), true);
  assert.equal(noteInRange(57, 'melody', { low: 0, high: 219 }), false);
  assert.equal(noteInRange(69, 'melody', { low: 440, high: 880 }), true);
  assert.equal(noteInRange(69, 'melody', { low: 441, high: 880 }), false);
});

test('a range that is wide open keeps every note', () => {
  for (let pitch = 21; pitch <= 108; pitch++) {
    assert.equal(
      noteInRange(pitch, 'melody', { low: MIN_FREQUENCY, high: MAX_FREQUENCY }),
      true,
      `pitch ${pitch} was dropped by an open range`,
    );
  }
});

test('a range typed the wrong way round is read the right way round', () => {
  assert.deepEqual(normaliseRange({ low: 400, high: 100 }), { low: 100, high: 400 });
});

test('the bounds land on the notes they name', () => {
  // 400 Hz is G4, and 40 Hz is just above E1.
  assert.equal(pitchAtOrBelow(400), 67);
  assert.equal(noteName(pitchAtOrBelow(400)), 'G4');
  assert.equal(pitchAtOrAbove(40), 28);
  assert.equal(noteName(pitchAtOrAbove(40)), 'E1');
});

test('a drum is judged by the band it occupies, not by its note number', () => {
  // The kick is written as C1 but sounds between 40 and 160 Hz; the hi-hat is
  // written as F#1 but lives between 6 and 10 kHz.
  assert.equal(noteInRange(36, 'drums', { low: 0, high: 400 }), true);
  assert.equal(noteInRange(42, 'drums', { low: 0, high: 400 }), false);
  assert.equal(noteInRange(42, 'drums', { low: 0, high: 10000 }), true);

  // The snare straddles 180 and 1500 Hz, so a 400 Hz ceiling still catches
  // the bottom of it, and a 100 Hz one does not.
  assert.equal(noteInRange(38, 'drums', { low: 0, high: 400 }), true);
  assert.equal(noteInRange(38, 'drums', { low: 0, high: 100 }), false);
});

test('filtering keeps the notes inside the range and drops the rest', () => {
  const notes = [note(36), note(60), note(72), note(90)];
  const kept = filterByRange(notes, 'melody', { low: 0, high: 440 });
  assert.deepEqual(
    kept.map((keptNote) => keptNote.pitchMidi),
    [36, 60],
  );
});

test('the range reads out in hertz and in notes', () => {
  assert.equal(describeRange({ low: 0, high: 400 }), '0 Hz – 400 Hz · A0–G4');
  assert.equal(describeRange({ low: 40, high: 1200 }), '40 Hz – 1200 Hz · E1–D6');
});
