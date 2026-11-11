/**
 * Does the arrangement keep the melody?
 *
 *   npx tsx scripts/pianify-melody-keep.mts [eval/midi]
 *
 * The first version of the stage put the notes of a chord into the hands in
 * the order they came out of the model, lowest first. The right hand was
 * therefore given the bottom of the treble and the melody on top of it was
 * the note that got dropped when the hand was full. The user could see it:
 * the "one hand" notes looked wrong.
 *
 * This measures it directly. For every chord in the transcription it asks:
 * was the top note of that chord kept, and was the bottom note of the left
 * hand's part kept? A melody is the top of a chord, and the bass is the
 * bottom, so those are the two notes a piano arrangement cannot lose.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

import { pianify } from '../src/core/pianify';
import { RawNote, Track } from '../src/core/types';

const Midi = ToneMidi.Midi;

/** How close together two notes have to start to be one chord. */
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

/** The chords: notes that start within a few tens of milliseconds of each other. */
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

/** Say whether a note of this pitch, at this time, is in the arrangement. */
function kept(
  arranged: Note[],
  pitch: number,
  start: number,
  tolerance = 0.05,
): boolean {
  return arranged.some(
    (note) => note.pitch === pitch && Math.abs(note.start - start) <= tolerance,
  );
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

interface Counts {
  top: number;
  topKept: number;
  bass: number;
  bassKept: number;
  chords: number;
  notesIn: number;
  notesOut: number;
}

const total: Counts = {
  top: 0,
  topKept: 0,
  bass: 0,
  bassKept: 0,
  chords: 0,
  notesIn: 0,
  notesOut: 0,
};

console.log('file                        chords   melody kept     bass kept   notes in/out');
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
  const arranged = result.tracks.flatMap((entry) =>
    entry.notes.map((note) => ({
      pitch: note.pitchMidi,
      start: note.startTime,
      end: note.startTime + note.duration,
      amplitude: note.amplitude,
    })),
  );

  let top = 0;
  let topKept = 0;
  let bass = 0;
  let bassKept = 0;

  for (const chord of chords(source)) {
    // The top of the chord, and the bottom of the part the left hand plays.
    const highest = chord[chord.length - 1];
    const lowest = chord[0];

    top += 1;
    if (kept(arranged, highest.pitch, highest.start)) topKept += 1;

    if (lowest.pitch < 60) {
      bass += 1;
      if (kept(arranged, lowest.pitch, lowest.start)) bassKept += 1;
    }
  }

  console.log(
    `${file.replace('.piano.mid', '').padEnd(26)} ${String(chords(source).length).padStart(6)} ` +
      `${String(topKept).padStart(8)}/${String(top).padEnd(5)} ` +
      `${String(bassKept).padStart(8)}/${String(bass).padEnd(5)} ` +
      `${String(source.length).padStart(6)}/${String(arranged.length).padEnd(6)}`,
  );

  total.top += top;
  total.topKept += topKept;
  total.bass += bass;
  total.bassKept += bassKept;
  total.chords += chords(source).length;
  total.notesIn += source.length;
  total.notesOut += arranged.length;
}

console.log('-'.repeat(74));
console.log(
  `${'all'.padEnd(26)} ${String(total.chords).padStart(6)} ` +
    `${String(total.topKept).padStart(8)}/${String(total.top).padEnd(5)} ` +
    `${String(total.bassKept).padStart(8)}/${String(total.bass).padEnd(5)} ` +
    `${String(total.notesIn).padStart(6)}/${String(total.notesOut).padEnd(6)}`,
);

console.log(
  `\nmelody kept: ${((total.topKept / Math.max(1, total.top)) * 100).toFixed(1)} % ` +
    `of the top notes of ${total.chords} chords.`,
);
console.log(
  `bass kept:   ${((total.bassKept / Math.max(1, total.bass)) * 100).toFixed(1)} % ` +
    `of the bottom notes of the left hand's part.`,
);
console.log(
  `notes:       ${total.notesIn} in, ${total.notesOut} out ` +
    `(${((total.notesOut / total.notesIn) * 100).toFixed(0)} % kept).`,
);
