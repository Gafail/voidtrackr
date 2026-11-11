/**
 * How far do the hands actually cross?
 *
 *   npx tsx scripts/pianify-crossing.mts [eval/midi]
 *
 * A note the left hand cannot take may be played by the right hand instead,
 * and the other way round. How far it may cross is a question of taste and
 * it has to be answered with something a pianist would recognise.
 *
 * The split is at middle C. The grand staff splits there, and the hands sit
 * on either side of it. In real playing they overlap a little: the right
 * hand sometimes comes down below the split, the left hand sometimes reaches
 * up over it. But the left hand reaching into the treble is the thing that
 * made the first version look wrong.
 *
 * This reports, for each distance, how much the hands overlap and how many
 * notes actually change hands.
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
        end: note.time + (note.duration ?? 0.1),
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

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

console.log(`${files.length} files from ${directory}\n`);
console.log(
  'crossing   left above C   right below C   notes changed hands   notes',
);
console.log('-'.repeat(70));

for (const distance of [0, 2, 4, 5, 7, 12]) {
  let leftAbove = 0;
  let rightBelow = 0;
  let moved = 0;
  let notes = 0;

  for (const file of files) {
    const source = readNotes(join(directory, file));
    if (source.length === 0) continue;

    const track: Track = {
      id: 'piano',
      name: 'Piano',
      kind: 'melody',
      notes: toRaw(source),
    };

    const result = pianify([track], { moveRange: distance });
    const left = result.tracks.find((entry) => entry.id === 'left-hand')?.notes ?? [];
    const right = result.tracks.find((entry) => entry.id === 'right-hand')?.notes ?? [];

    leftAbove += left.filter((note) => note.pitchMidi >= 60).length;
    rightBelow += right.filter((note) => note.pitchMidi < 60).length;
    moved += result.stats.moved;
    notes += left.length + right.length;
  }

  const percent = (part: number, whole: number) =>
    `${((part / Math.max(1, whole)) * 100).toFixed(1)}%`.padStart(8);

  console.log(
    `${String(distance).padStart(8)} ${percent(leftAbove, notes)} ${percent(rightBelow, notes)} ` +
      `${String(moved).padStart(20)} ${String(Math.round(notes / files.length)).padStart(6)}`,
  );
}

console.log(
  '\nleft above C: notes the left hand plays above the split, which is the',
);
console.log('treble. Right below C: the right hand in the bass.');
console.log('a third is the usual overlap on a grand staff.');
