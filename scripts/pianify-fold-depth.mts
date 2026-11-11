/**
 * How deep does the fold go?
 *
 *   npx tsx scripts/pianify-fold-depth.mts [eval/midi]
 *
 * The stage folds two notes on the same key into one when they are close
 * together. The fold is a chain: a merged note grows as long as the next
 * strike is within the gap of the merged note, which is longer than the gap
 * of the note before it. A key struck every forty milliseconds for a minute
 * becomes one note a minute long.
 *
 * The output has notes of fifteen seconds in it, and the raw model has none
 * over six. This measures how many strikes end up inside one folded note,
 * and what the gap between two strikes on the same key actually looks like.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

const Midi = ToneMidi.Midi;

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
      notes.push({ pitch: note.midi, start: note.time, end: note.time + note.duration });
    }
  }
  return notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

// The gap between one strike on a key and the next, when they are close.
const gaps: number[] = [];
for (const file of files) {
  const source = readNotes(join(directory, file));
  const byPitch = new Map<number, Note[]>();
  for (const note of source) {
    const list = byPitch.get(note.pitch);
    if (list) list.push(note);
    else byPitch.set(note.pitch, [note]);
  }

  for (const list of byPitch.values()) {
    for (let i = 1; i < list.length; i++) {
      const gap = list[i].start - list[i - 1].end;
      if (gap >= 0 && gap < 0.2) gaps.push(gap);
    }
  }
}

gaps.sort((a, b) => a - b);

console.log(`${files.length} files from ${directory}\n`);
console.log('the gap between one strike on a key and the next:');
for (const q of [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]) {
  const value = gaps[Math.min(gaps.length - 1, Math.floor(gaps.length * q))];
  console.log(`  p${String(Math.round(q * 100)).padStart(2)}  ${(value * 1000).toFixed(0)} ms`);
}
console.log(`  ${gaps.length} gaps under 200 ms`);

// How many strikes a chain of the fold swallows, for a key struck at a steady
// rate.
console.log('\na key struck steadily, each note 10 ms long:');
for (const gapMs of [20, 30, 40, 60]) {
  const gap = gapMs / 1000;
  let depth = 1;
  let end = 0.01;
  for (let i = 1; i < 100; i++) {
    const nextStart = i * (gap + 0.01);
    if (nextStart <= end + 0.05) {
      depth += 1;
      end = nextStart + 0.01;
    } else break;
  }
  const seconds = (depth - 1) * (gap + 0.01);
  console.log(
    `  every ${String(gapMs).padStart(2)} ms -> the fold swallows ${depth} strikes ` +
      `and makes a note of ${seconds.toFixed(2)} s`,
  );
}
