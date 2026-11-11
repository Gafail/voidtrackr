/**
 * Is there a melody in the transcription?
 *
 *   npx tsx scripts/pianify-melody.mts [eval/midi]
 *
 * The pianify stage puts the notes above the split in the right hand, and
 * that is not a melody: in a dense mix it is the top of every chord, which
 * jumps about and reads as noise. The fix is to trace one line through the
 * notes and play that in the right hand.
 *
 * The trace is a Viterbi over the onsets. At each onset one note is chosen,
 * and the path is rewarded for being smooth — a melody moves by step far
 * more often than it leaps — and for landing on notes the model was loud and
 * sustained about.
 *
 * The measurement is the share of the line that moves by step. A melody is
 * mostly steps; a scatter of chord tones is not. The baseline for comparison
 * is "the highest note at each onset", which is what the right hand plays
 * today.
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

/** Notes this close together are one onset. */
const ONSET = 0.03;

/** What one semitone of leap costs. */
const PITCH_COST = 1;

/** What it costs to skip an onset entirely. */
const REST_COST = 9;

/** How much a second of silence between two notes costs. */
const GAP_COST = 1.5;

/** How much being loud and sustained is worth. */
const SALIENCE_COST = 4;

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

/** Group the notes into onsets. */
function onsets(notes: Note[]): Note[][] {
  const groups: Note[][] = [];
  let current: Note[] = [];
  let anchor = 0;

  for (const note of notes) {
    if (current.length === 0) {
      current.push(note);
      anchor = note.start;
      continue;
    }
    if (note.start - anchor <= ONSET) {
      current.push(note);
      continue;
    }
    groups.push(current);
    current = [note];
    anchor = note.start;
  }
  if (current.length > 0) groups.push(current);
  return groups;
}

/** How loud and how long a note is, as one number. */
function salience(note: Note): number {
  const sustain = Math.min(1, (note.end - note.start) / 0.3);
  return note.amplitude * sustain;
}

/**
 * Trace one line through the notes.
 *
 * At every onset the trace chooses a note. The path that costs the least
 * wins, and the cost is made of how far the line moves in pitch, how long it
 * waits, how loud and sustained the note it lands on is, and a charge for
 * skipping an onset altogether.
 */
function trace(groups: Note[][]): Note[] {
  // The states are the candidates of the onset, plus one state for resting.
  // The costs and the back-pointers are kept per onset.
  const restCost: number[] = [];
  const restBack: number[] = [];
  const backs: number[][] = [];
  let costs: number[] = [];

  for (let index = 0; index < groups.length; index++) {
    const group = groups[index];
    const previous = groups[index - 1];

    // Resting at this onset is either resting again, or resting after a note.
    let bestRest = (restCost[index - 1] ?? 0) + REST_COST;
    let bestRestBack = -1;
    for (let k = 0; k < costs.length; k++) {
      if (costs[k] + REST_COST < bestRest) {
        bestRest = costs[k] + REST_COST;
        bestRestBack = k;
      }
    }
    restCost[index] = bestRest;
    restBack[index] = bestRestBack;

    const rowCost: number[] = [];
    const rowBack: number[] = [];

    for (let c = 0; c < group.length; c++) {
      const note = group[c];
      const unary = -SALIENCE_COST * salience(note);

      // From the rest.
      let cost = (restCost[index - 1] ?? 0) + unary;
      let back = -1;

      // From a note chosen at the previous onset.
      for (let k = 0; k < costs.length; k++) {
        const previousNote = previous[k];
        const gap = Math.max(0, note.start - previousNote.end);
        const total =
          costs[k] +
          PITCH_COST * Math.abs(note.pitch - previousNote.pitch) +
          GAP_COST * gap +
          unary;
        if (total < cost) {
          cost = total;
          back = k;
        }
      }

      rowCost.push(cost);
      rowBack.push(back);
    }

    costs = rowCost;
    backs[index] = rowBack;
  }

  // Find the cheapest place to finish.
  let state = -1;
  let best = restCost[groups.length - 1] ?? Infinity;
  for (let k = 0; k < costs.length; k++) {
    if (costs[k] < best) {
      best = costs[k];
      state = k;
    }
  }

  // Walk the path back.
  const line: Note[] = [];
  for (let index = groups.length - 1; index >= 0; index--) {
    if (state === -1) {
      state = restBack[index];
      continue;
    }
    const note = groups[index][state];
    if (note) line.unshift(note);
    state = backs[index]?.[state] ?? -1;
  }

  return line;
}

/** The share of the line that moves by step. */
function stepShare(line: Note[]): number {
  if (line.length < 2) return 0;
  let steps = 0;
  for (let i = 1; i < line.length; i++) {
    if (Math.abs(line[i].pitch - line[i - 1].pitch) <= 2) steps += 1;
  }
  return steps / (line.length - 1);
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

console.log(`${files.length} files from ${directory}\n`);
console.log('file                          traced            step%   top%   med   range');
console.log('-'.repeat(70));

let totalSteps = 0;
let totalTop = 0;
let count = 0;

for (const file of files) {
  const notes = readNotes(join(directory, file));
  const groups = onsets(notes);
  const line = trace(groups);

  // The baseline: the highest note at each onset.
  const top = groups.map((group) => group[group.length - 1]);

  const steps = stepShare(line);
  const topSteps = stepShare(top);

  // How often the traced note is the highest one at its onset.
  const topShare =
    line.filter((note) => {
      const group = groups.find((g) => g.includes(note));
      return group ? group[group.length - 1] === note : false;
    }).length / Math.max(1, line.length);

  const pitches = line.map((n) => n.pitch).sort((a, b) => a - b);
  const med = pitches[Math.floor(pitches.length / 2)] ?? 0;

  console.log(
    `${file.replace('.piano.mid', '').padEnd(28)} ${String(line.length).padStart(4)} notes   ` +
      `${steps.toFixed(1).padStart(5)}   ${(topShare * 100).toFixed(0).padStart(4)}   ` +
      `${String(med).padStart(4)}   ${pitches[0]}-${pitches[pitches.length - 1]}   ` +
      `(the top line: ${topSteps.toFixed(1)}% steps)`,
  );

  totalSteps += steps;
  totalTop += topSteps;
  count += 1;
}

console.log(
  `\nThe traced line moves by step ${(totalSteps / count).toFixed(1)} % of the time; ` +
    `the top-of-the-chord line that the right hand plays today moves by step ` +
    `${(totalTop / count).toFixed(1)} % of the time.`,
);
