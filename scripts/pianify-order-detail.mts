/**
 * Why does the melody go missing?
 *
 *   npx tsx scripts/pianify-order-detail.mts [eval/midi]
 *
 * The arrangement was changed so that the right hand is given the top of a
 * chord first and the left hand the bottom, on the theory that the top of a
 * chord is the melody and it should not be the note that gets dropped.
 *
 * The first measurement said otherwise, so this breaks it down: how wide the
 * chords are, whether the melody is in the treble or the bass, and what each
 * ordering keeps.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

import { collapseRepeats, lengthenNotes } from '../src/core/pianify';
import { RawNote } from '../src/core/types';

const Midi = ToneMidi.Midi;

const CLUSTER = 0.025;

interface Note {
  pitch: number;
  start: number;
  end: number;
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
    amplitude: 0.5,
  }));
}

function chords(notes: RawNote[]): RawNote[][] {
  const groups: RawNote[][] = [];
  let current: RawNote[] = [];
  for (const note of notes) {
    if (current.length === 0) {
      current.push(note);
      continue;
    }
    const previous = current[current.length - 1];
    if (note.startTime - previous.startTime <= CLUSTER) {
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

/** The old ordering: lowest first, one note at a time, in time order. */
function oldDivide(notes: RawNote[], splitPoint: number): RawNote[] {
  const hands: RawNote[][] = [[], []];
  const sorted = [...notes].sort(
    (a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi,
  );
  for (const note of sorted) {
    const preferred = note.pitchMidi < splitPoint ? 0 : 1;
    const near = hands[preferred].filter(
      (other) => Math.abs(other.startTime - note.startTime) <= CLUSTER,
    );
    const pitches = [...near, note].map((entry) => entry.pitchMidi);
    if (pitches.length > 5) continue;
    if (Math.max(...pitches) - Math.min(...pitches) > 12) continue;
    hands[preferred].push(note);
  }
  return [...hands[0], ...hands[1]];
}

/** The new ordering: the right hand takes the top of the chord first. */
function newDivide(notes: RawNote[], splitPoint: number): RawNote[] {
  const hands: RawNote[][] = [[], []];
  for (const chord of chords(notes)) {
    const high = chord
      .filter((note) => note.pitchMidi >= splitPoint)
      .sort((a, b) => b.pitchMidi - a.pitchMidi);
    const low = chord
      .filter((note) => note.pitchMidi < splitPoint)
      .sort((a, b) => a.pitchMidi - b.pitchMidi);
    for (const [hand, list] of [
      [1, high],
      [0, low],
    ] as const) {
      for (const note of list) {
        const near = hands[hand].filter(
          (other) => Math.abs(other.startTime - note.startTime) <= CLUSTER,
        );
        const pitches = [...near, note].map((entry) => entry.pitchMidi);
        if (pitches.length > 5) continue;
        if (Math.max(...pitches) - Math.min(...pitches) > 12) continue;
        hands[hand].push(note);
      }
    }
  }
  return [...hands[0], ...hands[1]];
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

interface Counts {
  chords: number;
  oldTop: number;
  newTop: number;
  oldBottom: number;
  newBottom: number;
}

const widths = new Map<number, Counts>();
let trebleChords = 0;
let trebleOld = 0;
let trebleNew = 0;
let bassChords = 0;
let bassOld = 0;
let bassNew = 0;

for (const file of files) {
  const source = readNotes(join(directory, file));
  if (source.length === 0) continue;
  const raw = lengthenNotes(collapseRepeats(toRaw(source), 0.05).notes, 0.06);

  const oldArranged = oldDivide(raw, 60);
  const newArranged = newDivide(raw, 60);

  for (const chord of chords(raw)) {
    const width = Math.min(9, chord.length);
    const counts =
      widths.get(width) ??
      { chords: 0, oldTop: 0, newTop: 0, oldBottom: 0, newBottom: 0 };
    widths.set(width, counts);

    const highest = chord[chord.length - 1];
    const lowest = chord[0];

    counts.chords += 1;
    if (kept(oldArranged, highest.pitchMidi, highest.startTime)) counts.oldTop += 1;
    if (kept(newArranged, highest.pitchMidi, highest.startTime)) counts.newTop += 1;
    if (kept(oldArranged, lowest.pitchMidi, lowest.startTime)) counts.oldBottom += 1;
    if (kept(newArranged, lowest.pitchMidi, lowest.startTime)) counts.newBottom += 1;

    if (highest.pitchMidi >= 60) {
      trebleChords += 1;
      if (kept(oldArranged, highest.pitchMidi, highest.startTime)) trebleOld += 1;
      if (kept(newArranged, highest.pitchMidi, highest.startTime)) trebleNew += 1;
    } else {
      bassChords += 1;
      if (kept(oldArranged, highest.pitchMidi, highest.startTime)) bassOld += 1;
      if (kept(newArranged, highest.pitchMidi, highest.startTime)) bassNew += 1;
    }
  }
}

console.log('chord width   chords    top kept (old)   top kept (new)');
console.log('-'.repeat(62));
for (const width of [...widths.keys()].sort((a, b) => a - b)) {
  const counts = widths.get(width)!;
  const pct = (part: number) => `${((part / Math.max(1, counts.chords)) * 100).toFixed(0)}%`.padStart(6);
  console.log(
    `${String(width).padStart(10)} ${String(counts.chords).padStart(7)} ` +
      `${pct(counts.oldTop)} (${counts.oldTop})      ${pct(counts.newTop)} (${counts.newTop})`,
  );
}

console.log('\nthe top note of the chord, by where it sits:');
console.log(
  `  in the treble: ${trebleOld}/${trebleChords} kept by the old one, ` +
    `${trebleNew}/${trebleChords} by the new one`,
);
console.log(
  `  in the bass:   ${bassOld}/${bassChords} kept by the old one, ` +
    `${bassNew}/${bassChords} by the new one`,
);
