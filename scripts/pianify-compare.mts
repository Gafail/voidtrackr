/**
 * The shipped version against the corrected one.
 *
 *   npx tsx scripts/pianify-compare.mts [eval/midi]
 *
 * This runs the stage as it was shipped in 0.6.0 and the stage as it is now,
 * over the same material, and reports what each one produces. The point is
 * to check the claim that the shipped one put the notes in the wrong hands
 * and that the corrected one does not.
 *
 * The old stage is read out of the git history, so it is the code that was
 * actually shipped rather than a reconstruction of it.
 */

import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import ToneMidi from '@tonejs/midi';

import { pianify } from '../src/core/pianify';
import { RawNote, Track } from '../src/core/types';

const Midi = ToneMidi.Midi;

const CLUSTER = 0.025;

interface Note {
  pitch: number;
  start: number;
  end: number;
  amplitude: number;
}

/**
 * Pull the 0.6.0 stage out of the git history and make it importable.
 *
 * It is written next to a copy of the types it was written against, so the
 * two versions cannot be confused with each other.
 */
async function loadOldStage(): Promise<(tracks: Track[]) => { tracks: Track[] }> {
  const directory = mkdtempSync(join(tmpdir(), 'pianify-v1-'));
  writeFileSync(
    join(directory, 'types.ts'),
    execFileSync('git', ['show', 'ba2fb5b:src/core/types.ts'], { encoding: 'utf8' }),
  );
  writeFileSync(
    join(directory, 'pianify.ts'),
    execFileSync('git', ['show', 'ba2fb5b:src/core/pianify.ts'], { encoding: 'utf8' }),
  );
  writeFileSync(
    join(directory, 'entry.mts'),
    `export { pianify } from './pianify.ts';\n`,
  );

  const module = await import(pathToFileURL(join(directory, 'entry.mts')).href);
  return module.pianify;
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

interface Summary {
  notes: number;
  ticks: number;
  leftMax: number;
  leftMean: number;
  rightMin: number;
  rightMean: number;
  melodyKept: number;
  melodyTotal: number;
  bassKept: number;
  bassTotal: number;
}

/**
 * Summarise the arrangement.
 *
 * The hands are read from the tracks, not from the pitches: a note's pitch
 * says nothing about which hand it was given to, and working it out from the
 * pitch would make the register measure agree with itself.
 */
function summarise(hands: Track[], source: RawNote[]): Summary {
  const left = (hands.find((entry) => entry.id === 'left-hand') ?? { notes: [] }).notes;
  const right = (hands.find((entry) => entry.id === 'right-hand') ?? { notes: [] }).notes;
  const arranged = [...left, ...right];

  let melodyKept = 0;
  let melodyTotal = 0;
  let bassKept = 0;
  let bassTotal = 0;

  for (const chord of chords(source)) {
    const highest = chord[chord.length - 1];
    const lowest = chord[0];
    melodyTotal += 1;
    if (kept(arranged, highest.pitchMidi, highest.startTime)) melodyKept += 1;
    if (lowest.pitchMidi < 60) {
      bassTotal += 1;
      if (kept(arranged, lowest.pitchMidi, lowest.startTime)) bassKept += 1;
    }
  }

  const mean = (notes: RawNote[]) =>
    notes.length === 0
      ? 0
      : Math.round(notes.reduce((sum, note) => sum + note.pitchMidi, 0) / notes.length);

  return {
    notes: arranged.length,
    ticks: arranged.filter((note) => note.duration <= 0.02).length,
    leftMax: left.length === 0 ? 0 : Math.max(...left.map((note) => note.pitchMidi)),
    leftMean: mean(left),
    rightMin: right.length === 0 ? 0 : Math.min(...right.map((note) => note.pitchMidi)),
    rightMean: mean(right),
    melodyKept,
    melodyTotal,
    bassKept,
    bassTotal,
  };
}

const oldPianify = await loadOldStage();

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

const oldSummary: Summary[] = [];
const newSummary: Summary[] = [];

for (const file of files) {
  const source = readNotes(join(directory, file));
  if (source.length === 0) continue;
  const raw = toRaw(source);
  const track: Track = { id: 'piano', name: 'Piano', kind: 'melody', notes: raw };

  const oldResult = oldPianify([track]);
  const newResult = pianify([track]);

  oldSummary.push(summarise(oldResult.tracks, raw));
  newSummary.push(summarise(newResult.tracks, raw));
}

const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);
const worst = (values: number[]) => Math.max(...values);
const best = (values: number[]) => Math.min(...values);
const percent = (part: number, whole: number) =>
  ((part / Math.max(1, whole)) * 100).toFixed(1);

console.log(`${files.length} files from ${directory}\n`);

const rows: [string, (s: Summary[]) => string][] = [
  ['notes', (s) => String(Math.round(sum(s.map((entry) => entry.notes)) / s.length))],
  ['ticks (20 ms or less)', (s) => `${percent(sum(s.map((e) => e.ticks)), sum(s.map((e) => e.notes)))}%`],
  ['left hand, highest note', (s) => String(worst(s.map((entry) => entry.leftMax)))],
  ['left hand, average note', (s) => String(Math.round(sum(s.map((e) => e.leftMean)) / s.length))],
  ['right hand, lowest note', (s) => String(best(s.map((entry) => entry.rightMin)))],
  ['right hand, average note', (s) => String(Math.round(sum(s.map((e) => e.rightMean)) / s.length))],
  ['melody kept', (s) => `${percent(sum(s.map((e) => e.melodyKept)), sum(s.map((e) => e.melodyTotal)))}%`],
  ['bass kept', (s) => `${percent(sum(s.map((e) => e.bassKept)), sum(s.map((e) => e.bassTotal)))}%`],
];

console.log('measure                  the shipped 0.6.0    the corrected one');
console.log('-'.repeat(58));
for (const [name, measure] of rows) {
  console.log(`${name.padEnd(22)} ${measure(oldSummary).padStart(12)} ${measure(newSummary).padStart(20)}`);
}
console.log(
  '\nthe left hand should stay under middle C and the right hand over it.',
);
console.log('the melody is the top note of a chord; the bass is the bottom of the left hand\'s part.');
