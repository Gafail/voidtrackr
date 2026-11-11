/**
 * How should a re-trigger be folded?
 *
 *   npx tsx scripts/pianify-merge-sweep.mts [eval/midi]
 *
 * Two notes on the same key, close together, are one note: a key cannot be
 * struck again that fast. The stage folds them.
 *
 * Two things about the fold are in question.
 *
 * The first is the gap. The stage folds anything within 50 ms, and the
 * material is full of keys struck every thirty or forty milliseconds, so
 * long chains of strikes fold into a single note — the output has notes of
 * seventeen seconds in it, which no piano has.
 *
 * The second is how long the folded note is. It can last until the last
 * strike ends, which is what the stage does now, or as long as the longest
 * of the strikes it swallowed.
 *
 * This sweeps both and reports what each choice buys and costs.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

import { RawNote } from '../src/core/types';

const Midi = ToneMidi.Midi;

/** Two strikes on the same key closer than this are one note. */
const RETRIGGER = 0.06;

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

/**
 * Fold the re-triggers.
 *
 * `mode` is 'span' or 'longest'. The gap is measured from the end of the
 * previous strike, so the fold is a pairwise operation and cannot chain.
 */
function fold(
  notes: RawNote[],
  mergeGap: number,
  mode: 'span' | 'longest',
): { notes: RawNote[]; folded: number } {
  const byPitch = new Map<number, RawNote[]>();
  for (const note of notes) {
    const list = byPitch.get(note.pitchMidi);
    if (list) list.push(note);
    else byPitch.set(note.pitchMidi, [note]);
  }

  const out: RawNote[] = [];
  let folded = 0;

  for (const list of byPitch.values()) {
    list.sort((a, b) => a.startTime - b.startTime);

    let current: RawNote = { ...list[0] };
    let previousEnd = current.startTime + current.duration;

    for (let i = 1; i < list.length; i++) {
      const next = list[i];
      // The gap is measured from the strike before, so a fold cannot run on
      // and on: each strike is judged on its own.
      if (next.startTime <= previousEnd + mergeGap) {
        const end = Math.max(previousEnd, next.startTime + next.duration);
        current = {
          ...current,
          duration:
            mode === 'span'
              ? end - current.startTime
              : Math.max(current.duration, next.duration),
          amplitude: Math.max(current.amplitude, next.amplitude),
        };
        previousEnd = Math.max(previousEnd, next.startTime + next.duration);
        folded += 1;
      } else {
        out.push(current);
        current = { ...next };
        previousEnd = next.startTime + next.duration;
      }
    }
    out.push(current);
  }

  return { notes: out.sort((a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi), folded };
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

/** The most notes sounding at the same moment. */
function peakSounding(notes: RawNote[]): number {
  const events: { time: number; delta: number }[] = [];
  for (const note of notes) {
    events.push({ time: note.startTime, delta: 1 });
    events.push({ time: note.startTime + note.duration, delta: -1 });
  }
  events.sort((a, b) => a.time - b.time || a.delta - b.delta);
  let current = 0;
  let peak = 0;
  for (const event of events) {
    current += event.delta;
    if (current > peak) peak = current;
  }
  return peak;
}

/** How often the same key is struck again straight away. */
function reTriggers(notes: RawNote[]): number {
  const byPitch = new Map<number, RawNote[]>();
  for (const note of notes) {
    const list = byPitch.get(note.pitchMidi);
    if (list) list.push(note);
    else byPitch.set(note.pitchMidi, [note]);
  }
  let count = 0;
  for (const list of byPitch.values()) {
    for (let i = 1; i < list.length; i++) {
      const gap = list[i].startTime - (list[i - 1].startTime + list[i - 1].duration);
      if (gap >= 0 && gap <= RETRIGGER) count += 1;
    }
  }
  return count;
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

const sources = files
  .map((file) => ({
    file,
    raw: toRaw(readNotes(join(directory, file))),
    reference: readNotes(join(directory, file.replace('.piano.mid', '.okarina.mid'))),
  }))
  .filter((entry) => entry.raw.length > 0);

console.log(`${sources.length} files from ${directory}\n`);
console.log(
  'gap    folding      notes  folded  retrig  longest  over 1 s  held   F1',
);
console.log('-'.repeat(72));

for (const mode of ['span', 'longest'] as const) {
  for (const mergeGap of [0.01, 0.02, 0.03, 0.05, 0.08]) {
    let notes = 0;
    let folded = 0;
    let retriggers = 0;
    let longest = 0;
    let over1 = 0;
    let held = 0;
    let f1 = 0;

    for (const entry of sources) {
      const result = fold(entry.raw, mergeGap, mode);
      notes += result.notes.length;
      folded += result.folded;
      retriggers += reTriggers(result.notes);
      longest = Math.max(longest, ...result.notes.map((note) => note.duration));
      over1 += result.notes.filter((note) => note.duration > 1).length;
      held = Math.max(held, peakSounding(result.notes));

      f1 += score(
        entry.reference,
        result.notes.map((note) => ({
          pitch: note.pitchMidi,
          start: note.startTime,
          end: note.startTime + note.duration,
          amplitude: note.amplitude,
        })),
        0.05,
      );
    }

    console.log(
      `${(mergeGap * 1000).toFixed(0).padStart(3)} ms ${mode.padEnd(10)} ` +
        `${String(Math.round(notes / sources.length)).padStart(5)} ${String(folded).padStart(7)} ` +
        `${String(retriggers).padStart(7)} ${longest.toFixed(2).padStart(8)} ` +
        `${String(over1).padStart(9)} ${String(held).padStart(6)} ` +
        `${((f1 / sources.length) * 100).toFixed(1).padStart(5)}%`,
    );
  }
  console.log('');
}

console.log('retrig: the same key struck again within 60 ms, which no hand can do.');
console.log('the F1 is against Okarina, the best model on this material.');
