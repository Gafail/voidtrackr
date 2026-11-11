/**
 * What should the arrangement do with the notes?
 *
 *   npx tsx scripts/pianify-split.mts [eval/midi]
 *
 * The pianify stage divides the notes between two hands. The first version
 * chose where the hands meet by balancing the load, and let a note be
 * handed to the other hand when its own was full. Both turned out to be
 * wrong in the same way: the left hand ended up playing the treble, and a
 * quarter of the notes were one model frame long — a tenth of a second,
 * which is a click and not a note.
 *
 * This sweeps the choices: how the chords are divided, whether a note may be
 * handed to the other hand, and the shortest note a hand is allowed to
 * play.
 *
 * What matters is the register each hand ends up in and how long the notes
 * are. A left hand that reaches into the treble and a quarter of the notes
 * being ticks is not a piano arrangement, however well it satisfies the
 * finger limits.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

import { pianify } from '../src/core/pianify';
import { Track } from '../src/core/types';

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

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
}

/** Match reference notes to estimated notes, one to one. */
function score(reference: Note[], estimate: Note[], tolerance: number): number {
  const used = new Set<number>();
  let matched = 0;
  for (const ref of reference) {
    let best = -1;
    let bestDistance = Infinity;
    for (let i = 0; i < estimate.length; i++) {
      if (used.has(i) || estimate[i].pitch !== ref.pitch) continue;
      const distance = Math.abs(estimate[i].start - ref.start);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    }
    if (best >= 0 && bestDistance <= tolerance) {
      used.add(best);
      matched += 1;
    }
  }
  const precision = reference.length === 0 ? 0 : matched / reference.length;
  const recall = estimate.length === 0 ? 0 : matched / estimate.length;
  return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
}

interface Configuration {
  name: string;
  /** How the split point is worked out. */
  rule: 'load' | 'quantile' | 'bass' | 'middleC';
  /** The quantile, for the quantile rule. */
  quantile: number;
  /** How far above the bass the split sits, for the bass rule. */
  offset: number;
  handover: number;
  minDuration: number;
}

const configurations: Configuration[] = [
  { name: 'the first version', rule: 'load', quantile: 0.5, offset: 0, handover: 12, minDuration: 0 },
  { name: 'load, no handover, 60 ms', rule: 'load', quantile: 0.5, offset: 0, handover: 0, minDuration: 0.06 },
  { name: 'p40, no handover, 60 ms', rule: 'quantile', quantile: 0.4, offset: 0, handover: 0, minDuration: 0.06 },
  { name: 'p30, no handover, 60 ms', rule: 'quantile', quantile: 0.3, offset: 0, handover: 0, minDuration: 0.06 },
  { name: 'p25, no handover, 60 ms', rule: 'quantile', quantile: 0.25, offset: 0, handover: 0, minDuration: 0.06 },
  { name: 'bass + 0, no handover', rule: 'bass', quantile: 0.5, offset: 0, handover: 0, minDuration: 0.06 },
  { name: 'bass + 3, no handover', rule: 'bass', quantile: 0.5, offset: 3, handover: 0, minDuration: 0.06 },
  { name: 'bass + 7, no handover', rule: 'bass', quantile: 0.5, offset: 7, handover: 0, minDuration: 0.06 },
  { name: 'bass + 12, no handover', rule: 'bass', quantile: 0.5, offset: 12, handover: 0, minDuration: 0.06 },
  { name: 'middle C, no handover', rule: 'middleC', quantile: 0.5, offset: 0, handover: 0, minDuration: 0.06 },
  { name: 'middle C or the bass', rule: 'bass', quantile: 0.5, offset: 0, handover: 0, minDuration: 0.06 },
];

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

console.log(`${files.length} files from ${directory}\n`);
console.log(
  'configuration                  left p95  left max  right p5  ticks%  short%  notes    L/R   F1',
);
console.log('-'.repeat(88));

for (const configuration of configurations) {
  let leftP95 = 0;
  let leftMax = 0;
  let rightP5 = 100;
  let ticks = 0;
  let short = 0;
  let notes = 0;
  let leftCount = 0;
  let rightCount = 0;
  let f1 = 0;

  for (const file of files) {
    const source = readNotes(join(directory, file));
    const reference = readNotes(join(directory, file.replace('.piano.mid', '.okarina.mid')));

    const track: Track = {
      id: 'piano',
      name: 'Piano',
      kind: 'melody',
      notes: source.map((note) => ({
        pitchMidi: note.pitch,
        startTime: note.start,
        duration: Math.max(0.01, note.end - note.start),
        amplitude: note.amplitude,
      })),
    };

    let splitPoint: number | undefined;
    if (configuration.rule === 'quantile') {
      splitPoint = quantile(source.map((n) => n.pitch), configuration.quantile);
    } else if (configuration.rule === 'bass') {
      // The median of the lowest note at each onset: where the bass sits.
      const bass = source
        .filter((note) => {
          const same = source.filter(
            (other) =>
              other.start >= note.start - 0.03 &&
              other.start <= note.start + 0.03,
          );
          return same.every((other) => other.pitch >= note.pitch);
        })
        .map((note) => note.pitch);
      // The middle C, or the bass when the bass is lower than it.
      splitPoint = Math.min(60, quantile(bass, 0.5) + configuration.offset);
    } else if (configuration.rule === 'middleC') {
      splitPoint = 60;
    }

    const result = pianify([track], {
      mergeGap: 0.05,
      maxPolyphony: 5,
      maxSpan: 12,
      maxRate: 10,
      moveRange: configuration.handover,
      minDuration: configuration.minDuration,
      splitPoint,
    });

    const left = result.tracks[0].notes;
    const right = result.tracks[1].notes;
    leftCount += left.length;
    rightCount += right.length;
    const all = [...left, ...right];
    notes += all.length;

    leftP95 = Math.max(leftP95, quantile(left.map((n) => n.pitchMidi), 0.95));
    leftMax = Math.max(leftMax, ...left.map((n) => n.pitchMidi));
    rightP5 = Math.min(rightP5, quantile(right.map((n) => n.pitchMidi), 0.05));

    ticks += all.filter((n) => n.duration <= 0.02).length;
    short += all.filter((n) => n.duration <= 0.06).length;

    const estimate = all.map((n) => ({
      pitch: n.pitchMidi,
      start: n.startTime,
      end: n.startTime + n.duration,
      amplitude: n.amplitude,
    }));
    f1 += score(reference, estimate, 0.05);
  }

  console.log(
    `${configuration.name.padEnd(28)} ${String(leftP95).padStart(7)} ${String(leftMax).padStart(10)} ` +
      `${String(rightP5).padStart(9)} ` +
      `${((ticks / notes) * 100).toFixed(1).padStart(5)}% ` +
      `${((short / notes) * 100).toFixed(1).padStart(6)}% ` +
      `${String(Math.round(notes / files.length)).padStart(7)} ` +
      `${String(Math.round(leftCount / files.length)).padStart(5)}/${String(Math.round(rightCount / files.length)).padStart(4)} ` +
      `${((f1 / files.length) * 100).toFixed(1).padStart(5)}%`,
  );
}

console.log(
  '\nticks are notes of 20 ms or less, one model frame or two; short are 60 ms or less.',
);
console.log('A piano note is at least a tenth of a second, and the left hand stays below the treble.');
