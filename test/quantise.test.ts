/**
 * Quantising notes to a grid.
 *
 * The grid is for things that run at a fixed rate — a game rendering 30
 * frames a second, a light rig stepping every 33 milliseconds. The notes
 * have to land on the frame they belong to, so the times are snapped to the
 * nearest slot.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  describeQuantise,
  quantiseNotes,
  quantiseTime,
  slotSeconds,
} from '../src/core/quantise';
import { RawNote } from '../src/core/types';

function note(startTime: number, duration: number): RawNote {
  return { pitchMidi: 60, startTime, duration, amplitude: 0.8 };
}

test('a slot is a frame long', () => {
  assert.equal(slotSeconds({ grid: 30 }), 1 / 30);
  assert.equal(slotSeconds({ grid: 60 }), 1 / 60);
  assert.equal(slotSeconds({ grid: 0 }), 0);
});

test('a time snaps to the nearest slot', () => {
  const grid = { grid: 30 };
  const period = 1 / 30;

  assert.equal(quantiseTime(0, grid), 0);
  // A time exactly half a slot in rounds up to the next slot.
  assert.ok(Math.abs(quantiseTime(period / 2, grid) - period) < 1e-9);
  assert.ok(Math.abs(quantiseTime(period / 2 - 0.001, grid) - 0) < 1e-9);
  assert.ok(Math.abs(quantiseTime(period / 2 + 0.001, grid) - period) < 1e-9);
  assert.ok(Math.abs(quantiseTime(1.0, grid) - Math.round(1.0 / period) * period) < 1e-9);
});

test('every snapped time sits on a slot', () => {
  const grid = { grid: 30 };
  const period = 1 / 30;
  for (const time of [0.004, 0.017, 0.05, 0.51, 1.234, 7.999]) {
    const snapped = quantiseTime(time, grid);
    const slots = snapped / period;
    assert.ok(
      Math.abs(slots - Math.round(slots)) < 1e-9,
      `${snapped} s is not a multiple of the slot`,
    );
  }
});

test('a time never moves before the recording does', () => {
  assert.equal(quantiseTime(-0.4, { grid: 30 }), 0);
});

test('the grid can be shifted', () => {
  // A phase of 10 ms puts the first slot 10 ms into the recording.
  const grid = { grid: 30, phase: 0.01 };
  assert.ok(Math.abs(quantiseTime(0.01, grid) - 0.01) < 1e-9);
  assert.ok(Math.abs(quantiseTime(0.026, grid) - 0.01) < 1e-9);
  assert.ok(Math.abs(quantiseTime(0.028, grid) - (0.01 + 1 / 30)) < 1e-9);
});

test('a note keeps its length and never ends before it really does', () => {
  const period = 1 / 30;
  const notes = quantiseNotes([note(0.01, 0.2), note(0.5, 0.05)], { grid: 30 });

  for (const snapped of notes) {
    assert.ok(snapped.duration > 0, 'a note collapsed to nothing');
    assert.ok(
      snapped.startTime + snapped.duration >= snapped.startTime,
      'a note ends before it starts',
    );
  }

  // The end is rounded up, so the note covers at least what it covered.
  assert.ok(notes[0].startTime + notes[0].duration >= 0.21);
  assert.ok(notes[1].startTime + notes[1].duration >= 0.55);
  assert.ok(notes[0].startTime % period < 1e-9);
});

test('a grid of zero leaves the notes alone', () => {
  const notes = [note(0.013, 0.4)];
  const snapped = quantiseNotes(notes, { grid: 0 });
  assert.equal(snapped[0].startTime, 0.013);
  assert.equal(snapped[0].duration, 0.4);
});

test('the grid describes itself', () => {
  assert.equal(describeQuantise({ grid: 30 }), '30 per second · 33.3 ms a slot');
  assert.equal(describeQuantise({ grid: 0 }), 'off');
});
