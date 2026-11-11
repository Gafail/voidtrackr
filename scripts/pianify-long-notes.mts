/**
 * What is inside the long notes?
 *
 *   npx tsx scripts/pianify-long-notes.mts [eval/midi]
 *
 * The output has notes of seventeen seconds in it. The raw model has none
 * over six. This finds the longest notes and prints the strikes that were
 * folded into them, so the shape of the problem is visible rather than
 * guessed at.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

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

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

interface Chain {
  pitch: number;
  strikes: Note[];
}

const chains: Chain[] = [];

for (const file of files) {
  const source = readNotes(join(directory, file));
  const byPitch = new Map<number, Note[]>();
  for (const note of source) {
    const list = byPitch.get(note.pitch);
    if (list) list.push(note);
    else byPitch.set(note.pitch, [note]);
  }

  for (const [pitch, list] of byPitch) {
    let current: Note[] = [list[0]];
    for (let i = 1; i < list.length; i++) {
      const next = list[i];
      const previous = current[current.length - 1];
      if (next.start - previous.end <= 0.05) {
        current.push(next);
      } else {
        chains.push({ pitch, strikes: current });
        current = [next];
      }
    }
    chains.push({ pitch, strikes: current });
  }
}

chains.sort((a, b) => {
  const spanA = a.strikes[a.strikes.length - 1].end - a.strikes[0].start;
  const spanB = b.strikes[b.strikes.length - 1].end - b.strikes[0].start;
  return spanB - spanA;
});

console.log(`${files.length} files from ${directory}\n`);
console.log('the longest chains of strikes on one key:');
console.log('pitch   strikes   span      first strike        last strike');
console.log('-'.repeat(72));

for (const chain of chains.slice(0, 12)) {
  const first = chain.strikes[0];
  const last = chain.strikes[chain.strikes.length - 1];
  const span = last.end - first.start;
  console.log(
    `${String(chain.pitch).padStart(5)} ${String(chain.strikes.length).padStart(8)} ` +
      `${span.toFixed(2).padStart(8)} s   ` +
      `${first.start.toFixed(2).padStart(7)}s ${(first.end - first.start).toFixed(2).padStart(6)}s   ` +
      `${last.start.toFixed(2).padStart(7)}s ${(last.end - last.start).toFixed(2).padStart(6)}s`,
  );
}

const long = chains.filter(
  (chain) => chain.strikes[chain.strikes.length - 1].end - chain.strikes[0].start > 3,
);
const longWithMany = long.filter((chain) => chain.strikes.length > 3);
console.log(
  `\n${long.length} chains last over three seconds, ${longWithMany.length} of them from more than three strikes.`,
);

const strikes = long.reduce((sum, chain) => sum + chain.strikes.length, 0);
console.log(`${strikes} strikes are inside those chains.`);

if (long.length > 0) {
  const durations = long.flatMap((chain) =>
    chain.strikes.map((note) => note.end - note.start),
  );
  durations.sort((a, b) => a - b);
  const median = durations[Math.floor(durations.length / 2)];
  console.log(
    `the strikes inside them last a median of ${median.toFixed(2)} s — they are one model frame each.`,
  );
}
