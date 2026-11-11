/**
 * How long are the notes, and how many does one hand hold at once?
 *
 *   npx tsx scripts/pianify-duration.mts [eval/midi]
 *
 * Two things could make the arranged notes look wrong on the piano roll, and
 * only one of them has been measured so far.
 *
 * The first is the length of a note. The model reports notes of one frame,
 * ten milliseconds, which is a click rather than a note, and the stage
 * lengthens them.
 *
 * The second is how many notes one hand is holding. The stage limits the
 * notes a hand *strikes* at once to five, but a note that lasts two seconds
 * is still down when the next chord comes. This measures the notes that are
 * sounding at the same moment, per hand, which is the number a player would
 * have to be holding down.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

import { pianify } from '../src/core/pianify';
import { RawNote, Track } from '../src/core/types';

const Midi = ToneMidi.Midi;

interface Note {
  pitch: number;
  start: number;
  end: number;
  amplitude: number;
}

function readNotes(path: string): Note[] {
  const midi = new Midi(readFileSync(path));
  const notes: Note[] = [];
  for (const track of midi.tracks) {
    if (/drum/i.test(track.name ?? '')) continue;
    for (const note of track.notes) {
      notes.push({
        pitch: note.midi,
        start: note.time,
        end: note.time + note.duration,
        amplitude: note.velocity ?? 0.5,
      });
    }
  }
  return notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

function toRaw(notes: Note[]): RawNote[] {
  return notes.map((note) => ({
    pitchMidi: note.pitch,
    startTime: note.start,
    duration: Math.max(0.01, note.end - note.start),
    amplitude: note.amplitude,
  }));
}

/** The most notes sounding at the same moment. */
function peakSounding(notes: RawNote[]): number {
  const events: { time: number; delta: number }[] = [];
  for (const note of notes) {
    events.push({ time: note.startTime, delta: 1 });
    events.push({ time: note.startTime + note.duration, delta: -1 });
  }
  events.sort((a, b) => a.time - b.time || a.delta - b.delta);
  let current = 0;
  let peak = 0;
  for (const event of events) {
    current += event.delta;
    if (current > peak) peak = current;
  }
  return peak;
}

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

let leftPeak = 0;
let rightPeak = 0;
let under60 = 0;
let under120 = 0;
let over1s = 0;
let total = 0;
let leftTotal = 0;
let rightTotal = 0;

console.log('file                       notes   dur p50   p95   max   left hold  right hold');
console.log('-'.repeat(74));

for (const file of files) {
  const source = readNotes(join(directory, file));
  if (source.length === 0) continue;

  const track: Track = {
    id: 'piano',
    name: 'Piano',
    kind: 'melody',
    notes: toRaw(source),
  };

  const result = pianify([track]);
  const left = result.tracks.find((entry) => entry.id === 'left-hand')?.notes ?? [];
  const right = result.tracks.find((entry) => entry.id === 'right-hand')?.notes ?? [];
  const all = [...left, ...right];

  const durations = all.map((note) => note.duration);
  const leftHold = peakSounding(left);
  const rightHold = peakSounding(right);

  console.log(
    `${file.replace('.piano.mid', '').padEnd(26)} ${String(all.length).padStart(5)} ` +
      `${quantile(durations, 0.5).toFixed(2).padStart(7)} ` +
      `${quantile(durations, 0.95).toFixed(2).padStart(6)} ` +
      `${Math.max(...durations).toFixed(2).padStart(6)} ` +
      `${String(leftHold).padStart(11)} ${String(rightHold).padStart(11)}`,
  );

  leftPeak = Math.max(leftPeak, leftHold);
  rightPeak = Math.max(rightPeak, rightHold);
  leftTotal += left.length;
  rightTotal += right.length;
  total += all.length;
  under60 += durations.filter((duration) => duration < 0.06).length;
  under120 += durations.filter((duration) => duration < 0.12).length;
  over1s += durations.filter((duration) => duration > 1).length;
}

console.log('-'.repeat(74));
console.log(
  `${'all'.padEnd(26)} ${String(total).padStart(5)} ` +
    `${String(leftTotal).padStart(11)} ${String(rightTotal).padStart(11)} notes`,
);
console.log(
  `\nshortest notes: ${under60} under 60 ms, ${under120} under 120 ms, ${over1s} over a second.`,
);
console.log(
  `most notes one hand holds down at once: ${leftPeak} in the left, ${rightPeak} in the right.`,
);
console.log(
  'A hand has five fingers, so more than five held down at once means the',
);
console.log('sustain is carrying notes the hand has let go of — the pedal.');
