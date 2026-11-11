/**
 * The old ordering against the new one.
 *
 *   npx tsx scripts/pianify-order.mts [eval/midi]
 *
 * The first version of the stage put the notes of a chord into the hands in
 * the order the model produced them, which is lowest first. The right hand
 * was therefore given the bottom of the treble, and when it ran out of
 * fingers the melody on top of the chord was the note that went.
 *
 * This runs both orderings over the same material and reports what each one
 * keeps: the top of each chord, which is the melody, and the bottom of the
 * left hand's part, which is the bass.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

import { pianify, collapseRepeats, lengthenNotes, limitRate } from '../src/core/pianify';
import { RawNote, Track } from '../src/core/types';

const Midi = ToneMidi.Midi;

const CLUSTER = 0.025;

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

/** The old division: lowest first, one note at a time, in time order. */
function oldDivide(
  notes: RawNote[],
  splitPoint: number,
  maxPolyphony: number,
  maxSpan: number,
): { left: RawNote[]; right: RawNote[] } {
  const hands: RawNote[][] = [[], []];

  const sorted = [...notes].sort((a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi);

  for (const note of sorted) {
    const preferred = note.pitchMidi < splitPoint ? 0 : 1;
    const near = hands[preferred].filter(
      (other) => Math.abs(other.startTime - note.startTime) <= CLUSTER,
    );
    const pitches = [...near, note].map((entry) => entry.pitchMidi);
    if (pitches.length > maxPolyphony) continue;
    if (Math.max(...pitches) - Math.min(...pitches) > maxSpan) continue;
    hands[preferred].push(note);
  }

  return { left: hands[0], right: hands[1] };
}

function chords(notes: Note[]): Note[][] {
  const groups: Note[][] = [];
  let current: Note[] = [];
  for (const note of notes) {
    if (current.length === 0) {
      current.push(note);
      continue;
    }
    const previous = current[current.length - 1];
    if (note.start - previous.start <= CLUSTER) {
      current.push(note);
      continue;
    }
    groups.push(current);
    current = [note];
  }
  if (current.length > 0) groups.push(current);
  return groups;
}

function kept(arranged: RawNote[], pitch: number, start: number): boolean {
  return arranged.some(
    (note) => note.pitchMidi === pitch && Math.abs(note.startTime - start) <= 0.05,
  );
}

interface Counts {
  top: number;
  topKept: number;
  bass: number;
  bassKept: number;
  notes: number;
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

const oldCounts: Counts = { top: 0, topKept: 0, bass: 0, bassKept: 0, notes: 0 };
const newCounts: Counts = { top: 0, topKept: 0, bass: 0, bassKept: 0, notes: 0 };

for (const file of files) {
  const source = readNotes(join(directory, file));
  if (source.length === 0) continue;

  const raw = toRaw(source);
  const track: Track = { id: 'piano', name: 'Piano', kind: 'melody', notes: raw };

  const prepared = lengthenNotes(collapseRepeats(raw, 0.05).notes, 0.06);

  const oldHands = oldDivide(prepared, 60, 5, 12);
  // The same rate limit on both sides, or the comparison is not a comparison.
  const oldAll = [
    ...limitRate(oldHands.left, 10).notes,
    ...limitRate(oldHands.right, 10).notes,
  ];

  const newResult = pianify([track]);
  const newAll = newResult.tracks.flatMap((entry) => entry.notes);

  for (const [arranged, counts] of [
    [oldAll, oldCounts],
    [newAll, newCounts],
  ] as const) {
    for (const chord of chords(source)) {
      const highest = chord[chord.length - 1];
      const lowest = chord[0];
      counts.top += 1;
      if (kept(arranged, highest.pitch, highest.start)) counts.topKept += 1;
      if (lowest.pitch < 60) {
        counts.bass += 1;
        if (kept(arranged, lowest.pitch, lowest.start)) counts.bassKept += 1;
      }
    }
    counts.notes += arranged.length;
  }
}

const percent = (part: number, whole: number) =>
  `${((part / Math.max(1, whole)) * 100).toFixed(1)}%`;

console.log(`${files.length} files from ${directory}\n`);
console.log('ordering        melody kept    bass kept    notes kept');
console.log('-'.repeat(56));
console.log(
  `the old one   ${percent(oldCounts.topKept, oldCounts.top).padStart(9)} ` +
    `${percent(oldCounts.bassKept, oldCounts.bass).padStart(14)} ` +
    `${String(oldCounts.notes).padStart(14)}`,
);
console.log(
  `the new one   ${percent(newCounts.topKept, newCounts.top).padStart(9)} ` +
    `${percent(newCounts.bassKept, newCounts.bass).padStart(14)} ` +
    `${String(newCounts.notes).padStart(14)}`,
);
console.log(
  `\nthe top of a chord is the melody; the bottom of the left hand's part is the bass.`,
);
