/**
 * Where is the music?
 *
 *   npx tsx scripts/pianify-pitch.mts [eval/midi]
 *
 * The arrangement has to decide where the left hand stops and the right hand
 * starts. The first version worked that out by balancing the load, which put
 * the split in the middle of the texture and left the left hand playing the
 * treble.
 *
 * This prints the pitch distribution of every file: where the bass sits, how
 * much of the music is below middle C, and how far the top reaches. It is the
 * measurement the split rule is built on.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

const Midi = ToneMidi.Midi;

interface Note {
  pitch: number;
  start: number;
}

function readNotes(path: string): Note[] {
  const midi = new Midi(readFileSync(path));
  const notes: Note[] = [];
  for (const track of midi.tracks) {
    if (/drum/i.test(track.name ?? '')) continue;
    for (const note of track.notes) {
      notes.push({ pitch: note.midi, start: note.time });
    }
  }
  return notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
}

/** The lowest note at each onset: the bass line. */
function bassLine(notes: Note[]): number[] {
  const lowest = new Map<number, number>();
  for (const note of notes) {
    const key = Math.round(note.start / 0.03);
    const current = lowest.get(key);
    if (current === undefined || note.pitch < current) lowest.set(key, note.pitch);
  }
  return [...lowest.values()];
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

console.log(
  'file                       notes  min   p5   p25  med    p75  max   below60  bass  bassMed',
);
console.log('-'.repeat(94));

let noBass = 0;

for (const file of files) {
  const notes = readNotes(join(directory, file));
  if (notes.length === 0) continue;
  const pitches = notes.map((note) => note.pitch);
  const bass = bassLine(notes);
  const below = pitches.filter((pitch) => pitch < 60).length;

  console.log(
    `${file.replace('.piano.mid', '').padEnd(26)} ${String(notes.length).padStart(5)} ` +
      `${String(Math.min(...pitches)).padStart(4)} ` +
      `${String(quantile(pitches, 0.05)).padStart(4)} ` +
      `${String(quantile(pitches, 0.25)).padStart(5)} ` +
      `${String(quantile(pitches, 0.5)).padStart(4)} ` +
      `${String(quantile(pitches, 0.75)).padStart(5)} ` +
      `${String(Math.max(...pitches)).padStart(4)} ` +
      `${((below / pitches.length) * 100).toFixed(0).padStart(6)}% ` +
      `${String(quantile(bass, 0.5)).padStart(5)} ` +
      `${String(quantile(bass, 0.25)).padStart(7)}`,
  );

  if (below / pitches.length < 0.05) noBass += 1;
}

console.log(
  `\n${files.length} files, ${noBass} of them with almost nothing below middle C.`,
);
console.log(
  'below60 is the share of notes under middle C; bass is the median of the',
);
console.log('lowest note at each onset, and bassMed is the lower quartile of that.');
