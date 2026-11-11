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
  chooseSplitPoint,
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
  // The folded note starts when the first strike did and ends when the last
  // one did, so no of the sound is lost.
  assert.equal(notes[0].startTime, 0);
  assert.ok(Math.abs(notes[0].duration - 0.81) < 1e-9);
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

test('the split point is worked out from the material', () => {
  const notes = [
    ...[36, 40, 43].map((pitch) => note(pitch, 0)),
    ...[72, 76, 79].map((pitch) => note(pitch, 0)),
  ];

  const split = chooseSplitPoint(notes, 5, 12);
  // The split has to fall between the bass and the melody, so that each
  // hand gets one of them.
  assert.ok(split > 43 && split <= 72, `the split landed at ${split}`);
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
