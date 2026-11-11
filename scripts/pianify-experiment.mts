/**
 * Does the pianify stage actually make the music playable?
 *
 *   npx tsx scripts/pianify-experiment.mts [eval/midi]
 *
 * This runs the stage over the material that has already been transcribed and
 * scores the result two ways:
 *
 *   - **Playability.** How many notes sound at once, how wide the chords are,
 *     how fast the hands move, how often a key is struck twice in a row.
 *     These are the numbers the stage exists to move.
 *
 *   - **Accuracy.** The F1 of the result against Okarina's transcription of
 *     the same audio, at the same 50 ms tolerance the rest of the project
 *     uses. Okarina is not ground truth — it is the best model here, at
 *     F1 85.3 % against the published Harmonica output — but it is the
 *     closest thing to a right answer this project has, and the stage does
 *     not use it, so the comparison is not circular.
 *
 * The point is the trade-off. A stage that made the output playable by
 * throwing everything away would score well on one and badly on the other.
 */

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
// The package is CommonJS, so under ESM the class is on the default export.
import ToneMidi from '@tonejs/midi';

import { pianify, PianifyOptions } from '../src/core/pianify';
import { RawNote, Track } from '../src/core/types';

const Midi = ToneMidi.Midi;

/** The gap below which two notes on one key are the same note. */
const RETRIGGER = 0.06;

/** How close together two notes have to start to be one chord. */
const CLUSTER = 0.025;

interface Note {
  pitch: number;
  start: number;
  end: number;
  amplitude: number;
}

function readNotes(path: string, withDrums = false): Note[] {
  const midi = new Midi(readFileSync(path));
  const notes: Note[] = [];
  for (const track of midi.tracks) {
    if (!withDrums && /drum/i.test(track.name ?? '')) continue;
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

function fromRaw(notes: RawNote[]): Note[] {
  return notes.map((note) => ({
    pitch: note.pitchMidi,
    start: note.startTime,
    end: note.startTime + note.duration,
    amplitude: note.amplitude,
  }));
}

/** Match reference notes to estimated notes, one to one. */
function score(
  reference: Note[],
  estimate: Note[],
  tolerance: number,
): { precision: number; recall: number; f1: number } {
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
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { precision, recall, f1 };
}

/** The numbers the stage exists to move. */
function playability(hands: Note[][]) {
  const notes = hands
    .flat()
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);

  if (notes.length === 0) {
    return {
      notes: 0,
      perSecond: 0,
      retriggers: 0,
      peakFingers: 0,
      peakSounding: 0,
      widestHand: 0,
      peakHandRate: 0,
      peakRate: 0,
    };
  }

  const duration = Math.max(...notes.map((note) => note.end));

  // The same key struck again.
  const byPitch = new Map<number, Note[]>();
  for (const note of notes) {
    const list = byPitch.get(note.pitch);
    if (list) list.push(note);
    else byPitch.set(note.pitch, [note]);
  }
  let retriggers = 0;
  for (const list of byPitch.values()) {
    for (let i = 1; i < list.length; i++) {
      const gap = Math.max(0, list[i].start - list[i - 1].end);
      if (gap <= RETRIGGER) retriggers += 1;
    }
  }

  // Per hand: the most notes struck together, how far the hand has to reach,
  // and how fast it plays. These are the finger limits.
  let peakFingers = 0;
  let widestHand = 0;
  let peakHandRate = 0;
  for (const hand of hands) {
    if (hand.length === 0) continue;
    const ordered = [...hand].sort((a, b) => a.start - b.start || a.pitch - b.pitch);

    // Chords: the notes the hand is playing at the same moment. This is the
    // same rule the stage uses, so the measurement and the constraint are
    // looking at the same thing. The window is walked with two pointers,
    // because the notes are in time order and the window only moves one way.
    let low = 0;
    let high = 0;
    for (const note of ordered) {
      while (low < ordered.length && ordered[low].start < note.start - CLUSTER) low += 1;
      while (high < ordered.length && ordered[high].start <= note.start + CLUSTER) high += 1;
      peakFingers = Math.max(peakFingers, high - low);
      let lowest = Infinity;
      let highest = -Infinity;
      for (let i = low; i < high; i++) {
        if (ordered[i].pitch < lowest) lowest = ordered[i].pitch;
        if (ordered[i].pitch > highest) highest = ordered[i].pitch;
      }
      widestHand = Math.max(widestHand, highest - lowest);
    }

    // The most notes the hand plays in one second.
    let head = 0;
    for (let i = 0; i < ordered.length; i++) {
      while (ordered[i].start - ordered[head].start > 1) head += 1;
      peakHandRate = Math.max(peakHandRate, i - head + 1);
    }
  }

  // The most notes sounding at once, across the whole instrument. With the
  // pedal down this can be more than ten; without it, it cannot.
  const events: { time: number; delta: number }[] = [];
  for (const note of notes) {
    events.push({ time: note.start, delta: 1 });
    events.push({ time: note.end, delta: -1 });
  }
  events.sort((a, b) => a.time - b.time || a.delta - b.delta);
  let sounding = 0;
  let peakSounding = 0;
  for (const event of events) {
    sounding += event.delta;
    if (sounding > peakSounding) peakSounding = sounding;
  }

  // The most notes in any half second.
  let peakRate = 0;
  let head = 0;
  for (let i = 0; i < notes.length; i++) {
    while (notes[i].start - notes[head].start > 0.5) head += 1;
    peakRate = Math.max(peakRate, i - head + 1);
  }

  return {
    notes: notes.length,
    perSecond: notes.length / Math.max(1, duration),
    retriggers,
    peakFingers,
    peakSounding,
    widestHand,
    peakHandRate,
    peakRate,
  };
}

interface Configuration {
  name: string;
  options: PianifyOptions | null;
}

const configurations: Configuration[] = [
  { name: 'the model, untouched', options: null },
  { name: 'collapse only', options: { mergeGap: 0.05, maxPolyphony: 99, maxSpan: 99, maxRate: 0 } },
  { name: 'collapse + two hands', options: { mergeGap: 0.05, maxPolyphony: 5, maxSpan: 12, maxRate: 0 } },
  { name: 'collapse + hands + 12/s', options: { mergeGap: 0.05, maxPolyphony: 5, maxSpan: 12, maxRate: 12 } },
  { name: 'collapse + hands + 10/s', options: { mergeGap: 0.05, maxPolyphony: 5, maxSpan: 12, maxRate: 10 } },
  { name: 'collapse + hands + 8/s', options: { mergeGap: 0.05, maxPolyphony: 5, maxSpan: 12, maxRate: 8 } },
  { name: 'merge 30ms + hands + 10/s', options: { mergeGap: 0.03, maxPolyphony: 5, maxSpan: 12, maxRate: 10 } },
  { name: 'merge 100ms + hands + 10/s', options: { mergeGap: 0.1, maxPolyphony: 5, maxSpan: 12, maxRate: 10 } },
  { name: 'span 16 + hands + 10/s', options: { mergeGap: 0.05, maxPolyphony: 5, maxSpan: 16, maxRate: 10 } },
  { name: 'poly 10 + hands + 10/s', options: { mergeGap: 0.05, maxPolyphony: 10, maxSpan: 12, maxRate: 10 } },
  { name: 'no handover + 10/s', options: { mergeGap: 0.05, maxPolyphony: 5, maxSpan: 12, maxRate: 10, moveRange: 0 } },
  { name: 'handover 24 + 10/s', options: { mergeGap: 0.05, maxPolyphony: 5, maxSpan: 12, maxRate: 10, moveRange: 24 } },
  { name: 'handover 36 + 10/s', options: { mergeGap: 0.05, maxPolyphony: 5, maxSpan: 12, maxRate: 10, moveRange: 36 } },
];

const directory = process.argv[2] ?? 'eval/midi';
const outDirectory = '/tmp/pianify-eval';
mkdirSync(outDirectory, { recursive: true });

const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

if (files.length === 0) {
  console.error(`No .piano.mid files in ${directory}`);
  process.exit(1);
}

console.log(`${files.length} files from ${directory}\n`);

const header =
  'configuration                  notes   /s  retrig  fingers  reach  hand/s  sound    prec   recall      F1';
console.log(header);
console.log('-'.repeat(header.length));

for (const configuration of configurations) {
  let notes = 0;
  let perSecond = 0;
  let retriggers = 0;
  let peakFingers = 0;
  let widestHand = 0;
  let peakHandRate = 0;
  let peakSounding = 0;
  let f1 = 0;
  let precision = 0;
  let recall = 0;
  let seconds = 0;

  for (const file of files) {
    const stem = file.replace(/\.piano\.mid$/, '');
    const source = readNotes(join(directory, file));
    const reference = readNotes(join(directory, `${stem}.okarina.mid`));

    let hands: Note[][];
    if (configuration.options === null) {
      // Before the stage, everything is in one track: one hand's worth of
      // notes, which is exactly the thing that is unplayable.
      hands = [source];
    } else {
      const track: Track = {
        id: 'piano',
        name: 'Piano',
        kind: 'melody',
        notes: toRaw(source),
      };
      const result = pianify([track], configuration.options);
      hands = result.tracks.map((t) => fromRaw(t.notes));

      if (configuration.name === 'collapse + hands + 10/s') {
        const bytes = new Midi();
        for (const t of result.tracks) {
          const midiTrack = bytes.addTrack();
          midiTrack.name = t.name;
          for (const note of t.notes) {
            midiTrack.addNote({
              midi: note.pitchMidi,
              time: note.startTime,
              duration: Math.max(0.01, note.duration),
              velocity: Math.max(0.01, Math.min(1, note.amplitude)),
            });
          }
        }
        writeFileSync(join(outDirectory, `${stem}.pianify.mid`), bytes.toArray());
      }
    }

    const stats = playability(hands);
    const duration = Math.max(1, ...source.map((note) => note.end));
    seconds += duration;

    notes += stats.notes;
    perSecond += stats.perSecond;
    retriggers += stats.retriggers;
    peakFingers = Math.max(peakFingers, stats.peakFingers);
    widestHand = Math.max(widestHand, stats.widestHand);
    peakHandRate = Math.max(peakHandRate, stats.peakHandRate);
    peakSounding = Math.max(peakSounding, stats.peakSounding);
    const scored = score(reference, hands.flat(), 0.05);
    f1 += scored.f1;
    precision += scored.precision;
    recall += scored.recall;
  }

  const files_ = files.length;
  console.log(
    `${configuration.name.padEnd(28)} ${String(Math.round(notes / files_)).padStart(5)} ` +
      `${(notes / seconds).toFixed(1).padStart(5)} ` +
      `${String(Math.round(retriggers / files_)).padStart(6)} ` +
      `${String(peakFingers).padStart(8)} ` +
      `${String(widestHand).padStart(6)} ` +
      `${String(peakHandRate).padStart(7)} ` +
      `${String(peakSounding).padStart(6)} ` +
      `${((precision / files_) * 100).toFixed(1).padStart(6)}% ` +
      `${((recall / files_) * 100).toFixed(1).padStart(6)}% ` +
      `${((f1 / files_) * 100).toFixed(1).padStart(6)}%`,
  );
}

console.log(
  '\nnotes and rates are averages per file; the rest are the worst over all the files.',
);
console.log(
  'fingers: the most notes one hand strikes at once. reach: the widest span one hand covers.',
);
console.log(`The arranged files are in ${outDirectory}.`);
void basename;
