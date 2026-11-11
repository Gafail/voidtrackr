/**
 * Which notes are worth keeping?
 *
 *   npx tsx scripts/pianify-value.mts eval/midi
 *
 * The pianify stage has to drop notes: a hand cannot hold ten at once, and
 * a key cannot be struck twice in ten milliseconds. Which notes go is the
 * whole question.
 *
 * Two candidate answers:
 *
 *   1. The model's own loudness. The piano model has a velocity head, so
 *      every note carries a 0..1 number for how hard the key was struck. A
 *      note the model was unsure about ought to be quieter.
 *
 *   2. The other models' opinion. Five models have transcribed this same
 *      material. A note that the piano model and Okarina both found is more
 *      likely to be a real note than one only the piano model found.
 *
 * This measures both: for every note the piano model reported, it says
 * whether the other models found the same pitch at the same time, and then
 * compares the loudness of the two groups. If loudness separates them, the
 * stage can use it. If it does not, the stage has to ask for a second model,
 * and the cost of that has to be weighed.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
// The package is CommonJS, so under ESM the class is on the default export.
import ToneMidi from '@tonejs/midi';

const Midi = ToneMidi.Midi;

interface Note {
  pitch: number;
  start: number;
  end: number;
  velocity: number;
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
        velocity: note.velocity ?? 0.8,
      });
    }
  }
  return notes;
}

/** Is there a note on the same pitch within `tolerance` seconds? */
function corroborated(notes: Note[], note: Note, tolerance: number): boolean {
  for (const other of notes) {
    if (other.pitch !== note.pitch) continue;
    if (Math.abs(other.start - note.start) <= tolerance) return true;
  }
  return false;
}

/** Mean and the share below a number. */
function summarise(values: number[]): { mean: number; median: number; low: number } {
  if (values.length === 0) return { mean: 0, median: 0, low: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const median = sorted[Math.floor(sorted.length / 2)];
  const low = values.filter((value) => value < 0.25).length / values.length;
  return { mean, median, low };
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory).filter((name) => name.endsWith('.piano.mid'));

const tolerance = 0.05;

const agreed: number[] = [];
const solo: number[] = [];
const agreedChains: number[] = [];
const soloChains: number[] = [];

for (const file of files) {
  const stem = file.replace(/\.piano\.mid$/, '');
  const piano = readNotes(join(directory, file));

  const others: Note[] = [];
  for (const other of files) {
    void other;
  }
  for (const name of readdirSync(directory)) {
    if (!name.startsWith(stem + '.') || name.endsWith('.piano.mid')) continue;
    others.push(...readNotes(join(directory, name)));
  }
  if (others.length === 0) continue;

  // Which notes are a re-trigger: the same key struck again within 60 ms.
  // The index has to be the index into the whole file, so the notes are
  // looked up by their place in the sorted list.
  const chains = new Set<number>();
  const byPitch = new Map<number, { note: Note; index: number }[]>();
  piano.forEach((note, index) => {
    const list = byPitch.get(note.pitch);
    if (list) list.push({ note, index });
    else byPitch.set(note.pitch, [{ note, index }]);
  });
  for (const list of byPitch.values()) {
    for (let i = 1; i < list.length; i++) {
      const gap = Math.max(0, list[i].note.start - list[i - 1].note.end);
      if (gap <= 0.06) chains.add(list[i].index);
    }
  }

  piano.forEach((note, index) => {
    const isChain = chains.has(index);
    if (corroborated(others, note, tolerance)) {
      agreed.push(note.velocity);
      if (isChain) agreedChains.push(note.velocity);
    } else {
      solo.push(note.velocity);
      if (isChain) soloChains.push(note.velocity);
    }
  });
}

console.log(`files: ${files.length}   tolerance: ${(tolerance * 1000).toFixed(0)} ms\n`);

const a = summarise(agreed);
const s = summarise(solo);
console.log('loudness (velocity head), 0..1');
console.log(
  `  corroborated by another model : ${agreed.length} notes   mean ${a.mean.toFixed(3)}   median ${a.median.toFixed(3)}   under 0.25: ${(a.low * 100).toFixed(1)}%`,
);
console.log(
  `  found by the piano model only : ${solo.length} notes   mean ${s.mean.toFixed(3)}   median ${s.median.toFixed(3)}   under 0.25: ${(s.low * 100).toFixed(1)}%`,
);

const chainTotal = agreedChains.length + soloChains.length;
console.log(
  `\nre-triggers (the same key within 60 ms): ${chainTotal} of ${agreed.length + solo.length} notes ` +
    `(${(chainTotal / Math.max(1, agreed.length + solo.length) * 100).toFixed(1)}%)`,
);
console.log(
  `  of those, corroborated: ${agreedChains.length} (${(agreedChains.length / Math.max(1, chainTotal) * 100).toFixed(1)}%)`,
);
console.log(
  `  of those, piano model only: ${soloChains.length} (${(soloChains.length / Math.max(1, chainTotal) * 100).toFixed(1)}%)`,
);

const share = solo.length / Math.max(1, agreed.length + solo.length);
console.log(
  `\nSo ${(share * 100).toFixed(1)}% of the piano model's notes are not in any other model's output.`,
);
console.log(
  'The question is whether those are noise, or simply the notes the other models missed.',
);
