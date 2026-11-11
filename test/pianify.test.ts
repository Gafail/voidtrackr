/**
 * Arranging a transcription for two hands.
 *
 * A piano has ten fingers, two hands and a reach of about an octave. The
 * tests below are about the promises the pianify stage makes: the re-triggers
 * are folded together, each hand stays inside its limits, and the drums are
 * left alone.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  collapseRepeats,
  divideHands,
  limitRate,
  pianify,
  peakPolyphony,
} from '../src/core/pianify';
import { RawNote, Track } from '../src/core/types';

function note(pitchMidi: number, startTime: number, duration = 0.4, amplitude = 0.8): RawNote {
  return { pitchMidi, startTime, duration, amplitude };
}

function track(notes: RawNote[], kind: Track['kind'] = 'melody'): Track {
  return { id: 'piano', name: 'Piano', kind, notes };
}

/** The most notes one hand plays at the same moment. */
function peakChord(notes: RawNote[]): number {
  let peak = 0;
  for (const candidate of notes) {
    const chord = notes.filter(
      (other) => Math.abs(other.startTime - candidate.startTime) <= 0.025,
    );
    if (chord.length > peak) peak = chord.length;
  }
  return peak;
}

/** The widest reach one hand has to make. */
function widestSpan(notes: RawNote[]): number {
  let widest = 0;
  for (const candidate of notes) {
    const chord = notes.filter(
      (other) => Math.abs(other.startTime - candidate.startTime) <= 0.025,
    );
    const pitches = chord.map((entry) => entry.pitchMidi);
    const span = Math.max(...pitches) - Math.min(...pitches);
    if (span > widest) widest = span;
  }
  return widest;
}

test('the same key struck again is one note', () => {
  const { notes, collapsed } = collapseRepeats(
    [note(60, 0, 0.5), note(60, 0.51, 0.3), note(60, 1.2, 0.3)],
    0.05,
  );

  assert.equal(collapsed, 1);
  assert.equal(notes.length, 2);
  // The folded note starts when the first strike did, so the onset — the part
  // the model gets right — does not move.
  assert.equal(notes[0].startTime, 0);
  // It lasts as long as the longest of the strikes it swallowed. It does not
  // last until the last strike ended, because that made the fold a chain:
  // a key struck every forty milliseconds became one note of seconds.
  assert.ok(Math.abs(notes[0].duration - 0.5) < 1e-9);
  assert.equal(notes[1].startTime, 1.2);
});

test('a note that is struck again much later is a note of its own', () => {
  const { notes, collapsed } = collapseRepeats([note(60, 0, 0.4), note(60, 1.0, 0.4)], 0.05);

  assert.equal(collapsed, 0);
  assert.equal(notes.length, 2);
});

test('the folded note keeps the loudest of the strikes', () => {
  const { notes } = collapseRepeats(
    [note(60, 0, 0.2, 0.3), note(60, 0.21, 0.2, 0.9)],
    0.05,
  );

  assert.equal(notes.length, 1);
  assert.equal(notes[0].amplitude, 0.9);
});

test('the low notes go to the left hand and the high ones to the right', () => {
  const { left, right } = divideHands(
    [note(48, 0), note(55, 0), note(72, 0), note(79, 0)],
    60,
    5,
    12,
    12,
  );

  assert.deepEqual(left.map((entry) => entry.pitchMidi), [48, 55]);
  assert.deepEqual(right.map((entry) => entry.pitchMidi), [72, 79]);
});

test('a hand never holds more notes than it has fingers', () => {
  // Seven notes struck together. Five go to one hand and two to the other.
  const notes = [40, 44, 48, 52, 56, 60, 64].map((pitch) => note(pitch, 0, 0.4, 0.8));
  const { left, right } = divideHands(notes, 60, 5, 12, 12);

  assert.ok(peakChord(left) <= 5, `the left hand was holding ${peakChord(left)} notes`);
  assert.ok(peakChord(right) <= 5, `the right hand was holding ${peakChord(right)} notes`);
});

test('a hand never reaches further than an octave', () => {
  // A bass note and a note three octaves above it, struck together.
  const notes = [note(36, 0), note(96, 0)];
  const { left, right } = divideHands(notes, 60, 5, 12, 12);

  assert.ok(widestSpan(left) <= 12, `the left hand reached ${widestSpan(left)} semitones`);
  assert.ok(widestSpan(right) <= 12, `the right hand reached ${widestSpan(right)} semitones`);
});

test('a note neither hand can take is dropped', () => {
  // Six notes an octave apart from each other, all at once: no hand can
  // hold them, so some have to go.
  const notes = [30, 42, 54, 66, 78, 90].map((pitch) => note(pitch, 0, 0.4, 0.8));
  const { left, right, dropped } = divideHands(notes, 60, 5, 12, 0);

  assert.ok(dropped > 0, 'nothing was dropped');
  assert.ok(peakChord(left) <= 5);
  assert.ok(peakChord(right) <= 5);
  assert.ok(widestSpan(left) <= 12);
  assert.ok(widestSpan(right) <= 12);
});

test('the hands meet at middle C', () => {
  // The grand staff splits at middle C, and the stage does too: the left
  // hand plays below it and the right hand plays from it up.
  const result = pianify([track([note(48, 0), note(60, 0), note(72, 0)])]);

  assert.equal(result.stats.splitPoint, 60);
  assert.deepEqual(result.tracks[0].notes.map((entry) => entry.pitchMidi), [48]);
  assert.deepEqual(result.tracks[1].notes.map((entry) => entry.pitchMidi), [60, 72]);
});

test('the left hand does not reach into the treble', () => {
  // A wide texture, the kind a mix produces. The hands meet at middle C and
  // overlap by a third, the way they do on a grand staff, so the left hand
  // may reach a little above the split — but not into the treble. The first
  // version let it play up to 87.
  const notes = [28, 40, 52, 59, 64, 72, 80, 88].map((pitch) => note(pitch, 0));
  const result = pianify([track(notes)]);

  const left = result.tracks[0].notes;
  assert.ok(
    left.every((entry) => entry.pitchMidi < 68),
    `the left hand reached ${Math.max(...left.map((entry) => entry.pitchMidi))}`,
  );
  const right = result.tracks[1].notes;
  assert.ok(
    right.every((entry) => entry.pitchMidi >= 56),
    `the right hand dropped to ${Math.min(...right.map((entry) => entry.pitchMidi))}`,
  );
});

test('the melody of a chord is kept, not the note underneath it', () => {
  // Six notes at once in the right hand's range. The hand has five fingers,
  // so one note has to go, and it must be the lowest one: the top of a
  // chord is the melody.
  const notes = [64, 67, 71, 74, 79, 84].map((pitch) => note(pitch, 0, 0.4, 0.8));
  const { right } = divideHands(notes, 60, 5, 12, 0);

  assert.ok(right.length <= 5);
  assert.ok(
    right.some((entry) => entry.pitchMidi === 84),
    'the top of the chord was dropped',
  );
  assert.ok(
    !right.some((entry) => entry.pitchMidi === 64),
    'the note under the chord was kept instead of the melody',
  );
});

test('a note too short to play is lengthened', () => {
  // One frame of the model, ten milliseconds. It is not a note.
  const result = pianify([track([note(60, 0, 0.01)])]);

  assert.ok(
    result.tracks[1].notes[0].duration >= 0.06,
    `the note was ${result.tracks[1].notes[0].duration} seconds long`,
  );
  // The onset does not move: that is the part the model gets right.
  assert.equal(result.tracks[1].notes[0].startTime, 0);
});

test('a key struck again and again is still one note', () => {
  // Twenty strikes, forty milliseconds apart, each one a model frame long.
  // The fold is not allowed to run on: the note must not become the length
  // of the whole run.
  const notes = Array.from({ length: 20 }, (_, index) =>
    note(60, index * 0.04, 0.01),
  );
  const { notes: folded } = collapseRepeats(notes, 0.05);

  assert.equal(folded.length, 1);
  assert.ok(
    folded[0].duration <= 0.05,
    `the note became ${folded[0].duration} seconds long`,
  );
});

test('a hand does not play faster than it can', () => {
  // Twelve notes a tenth of a second apart: a dozen a second.
  const notes = Array.from({ length: 12 }, (_, index) => note(60, index * 0.1, 0.05, 0.5));
  const { notes: kept, dropped } = limitRate(notes, 10);

  assert.ok(dropped > 0, 'nothing was dropped');
  let peak = 0;
  for (const candidate of kept) {
    const window = kept.filter(
      (other) =>
        other.startTime >= candidate.startTime && other.startTime < candidate.startTime + 1,
    );
    if (window.length > peak) peak = window.length;
  }
  assert.ok(peak <= 10, `the hand played ${peak} notes in a second`);
});

test('the loud notes are the ones that are kept', () => {
  const notes = [
    note(60, 0, 0.05, 0.9),
    note(62, 0.02, 0.05, 0.2),
    note(64, 0.04, 0.05, 0.8),
  ];
  const { notes: kept } = limitRate(notes, 1);

  assert.ok(
    kept.some((entry) => entry.amplitude === 0.9),
    'the loudest note was dropped',
  );
  assert.ok(
    !kept.some((entry) => entry.amplitude === 0.2),
    'the quietest note was kept',
  );
});

test('the stage leaves the drums alone', () => {
  const drums: Track = {
    id: 'drums',
    name: 'Drums',
    kind: 'drums',
    notes: [note(36, 0, 0.1), note(36, 0.25, 0.1)],
  };
  const result = pianify([track([note(60, 0), note(72, 0)]), drums]);

  const drumTrack = result.tracks.find((entry) => entry.kind === 'drums');
  assert.ok(drumTrack, 'the drum kit went missing');
  assert.equal(drumTrack.notes.length, 2);
});

test('the stage gives two hands', () => {
  const result = pianify([track([note(48, 0), note(60, 0), note(72, 0)])]);

  assert.equal(result.tracks.length, 2);
  assert.equal(result.tracks[0].name, 'Left hand');
  assert.equal(result.tracks[1].name, 'Right hand');
});

test('the two hands can be put back in one track', () => {
  const result = pianify([track([note(48, 0), note(72, 0)])], { splitHands: false });

  assert.equal(result.tracks.length, 1);
  assert.equal(result.tracks[0].notes.length, 2);
});

test('the stage says what it did', () => {
  const result = pianify([track([note(60, 0), note(60, 0.01)])]);

  assert.equal(result.stats.collapsed, 1);
  assert.equal(result.stats.after, 1);
  assert.match(result.log, /two hands/);
});

test('a transcription of nothing is still nothing', () => {
  const result = pianify([track([])]);

  assert.equal(result.stats.after, 0);
  assert.equal(result.tracks.length, 2);
});

test('the most notes sounding at once is counted', () => {
  assert.equal(peakPolyphony([note(60, 0, 1), note(64, 0, 1)]), 2);
  // The second note starts while the first is still down, so they overlap.
  assert.equal(peakPolyphony([note(60, 0, 1), note(64, 0.5, 1)]), 2);
  // This one starts after the first has finished.
  assert.equal(peakPolyphony([note(60, 0, 0.4), note(64, 0.5, 1)]), 1);
});
