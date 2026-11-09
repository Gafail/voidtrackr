/**
 * Score one MIDI file against another, note by note.
 *
 *   node scripts/evaluate-midi.mjs <reference.mid> <estimate.mid> [toleranceMs]
 *
 * This is the measurement the project was missing: the only way to know
 * whether a change to the pipeline helped is to score its output against a
 * known-good transcription.
 *
 * A note counts as matched when the pitch is the same and the onsets are
 * within the tolerance. Matching is one-to-one, so a reference note cannot
 * be matched twice.
 *
 * The numbers are the standard automatic music transcription metrics:
 * precision, recall and F1 over note onsets, plus the mean absolute
 * deviation of the onsets that did match.
 */

import { readFileSync } from 'node:fs';
import * as ToneMidi from '@tonejs/midi';

const Midi = ToneMidi.Midi ?? ToneMidi.default?.Midi ?? ToneMidi.default;

/** Read every note out of a MIDI file, flattened across tracks. */
function readNotes(path) {
  const midi = new Midi(readFileSync(path));
  const notes = [];
  for (const track of midi.tracks) {
    for (const note of track.notes) {
      notes.push({
        track: track.name,
        pitch: note.midi,
        start: note.time,
        end: note.time + note.duration,
        velocity: note.velocity,
      });
    }
  }
  return {
    notes,
    tempo: midi.header.tempos[0]?.bpm ?? 0,
    duration: Math.max(0, ...notes.map((note) => note.end)),
    tracks: midi.tracks.map((track) => `${track.name}:${track.notes.length}`).join(', '),
  };
}

/**
 * Match reference notes to estimated notes, one to one.
 *
 * For each reference note the closest unmatched estimate with the same
 * pitch is taken, if it is within the tolerance.
 */
function match(reference, estimate, tolerance) {
  const used = new Set();
  const pairs = [];

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
      pairs.push({ ref, estimate: estimate[best], distance: bestDistance });
    }
  }

  return { pairs, unmatched: estimate.length - pairs.length };
}

function score(reference, estimate, tolerance) {
  const { pairs, unmatched } = match(reference, estimate, tolerance);
  const precision = reference.length === 0 ? 0 : pairs.length / reference.length;
  const recall = estimate.length === 0 ? 0 : pairs.length / estimate.length;
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  const meanDeviation =
    pairs.length === 0
      ? 0
      : pairs.reduce((total, pair) => total + pair.distance, 0) / pairs.length;

  // Which pitches were missed, and which were invented.
  const missed = new Map();
  const invented = new Map();
  const matchedPitches = new Set(pairs.map((pair) => pair.ref.pitch));
  for (const note of reference) {
    if (!matchedPitches.has(note.pitch)) missed.set(note.pitch, (missed.get(note.pitch) ?? 0) + 1);
  }
  const estimatedPitches = new Set(estimate.map((note) => note.pitch));
  for (const note of estimate) {
    if (!estimatedPitches.has(note.pitch)) continue;
  }
  for (const pair of pairs) {
    matchedPitches.delete(pair.ref.pitch);
  }
  for (const [pitch, count] of matchedPitches) {
    missed.set(pitch, (missed.get(pitch) ?? 0) + count);
  }
  const estimatedOnly = new Set();
  for (const note of estimate) {
    if (!reference.some((ref) => ref.pitch === note.pitch)) {
      estimatedOnly.add(note.pitch);
    }
  }
  for (const pitch of estimatedOnly) invented.set(pitch, 1);

  return { pairs, unmatched, precision, recall, f1, meanDeviation, missed, invented };
}

const [referencePath, estimatePath, toleranceArg] = process.argv.slice(2);
if (!referencePath || !estimatePath) {
  console.error('Usage: node scripts/evaluate-midi.mjs <reference.mid> <estimate.mid> [toleranceMs]');
  process.exit(1);
}
const tolerance = (Number(toleranceArg ?? 50)) / 1000;

const reference = readNotes(referencePath);
const estimate = readNotes(estimatePath);

console.log(`reference : ${referencePath}`);
console.log(`  ${reference.tracks || '(no tracks)'} | ${reference.notes.length} notes | ${reference.tempo} BPM`);
console.log(`estimate  : ${estimatePath}`);
console.log(`  ${estimate.tracks || '(no tracks)'} | ${estimate.notes.length} notes | ${estimate.tempo} BPM`);
console.log(`tolerance : ${(tolerance * 1000).toFixed(0)} ms\n`);

for (const window of [0.02, 0.05, 0.1]) {
  const result = score(reference.notes, estimate.notes, window);
  console.log(
    `onset ±${(window * 1000).toFixed(0).padStart(3)} ms  ` +
      `precision ${(result.precision * 100).toFixed(1).padStart(5)}%  ` +
      `recall ${(result.recall * 100).toFixed(1).padStart(5)}%  ` +
      `F1 ${(result.f1 * 100).toFixed(1).padStart(5)}%  ` +
      `matched ${String(result.pairs.length).padStart(4)}  ` +
      `mean deviation ${(result.meanDeviation * 1000).toFixed(1).padStart(5)} ms`,
  );
}

const result = score(reference.notes, estimate.notes, tolerance);
if (result.missed.size > 0) {
  const missed = [...result.missed.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([pitch, count]) => `${pitch}×${count}`)
    .join(' ');
  console.log(`\npitches missing from the estimate: ${missed}`);
}
if (result.invented.size > 0) {
  const invented = [...result.invented.keys()].sort((a, b) => a - b).slice(0, 10).join(' ');
  console.log(`pitches only in the estimate: ${invented}`);
}

/**
 * Look for a constant shift between the two files.
 *
 * Two transcriptions of the same audio often disagree by a fixed amount,
 * because the models report their frames at slightly different points in
 * the window. If a single shift lifts the score a lot, the difference is a
 * calibration problem rather than a difference in what was heard.
 */
if (process.argv.includes('--search-offset')) {
  console.log('\nglobal offset search:');
  let best = { shift: 0, f1: 0 };
  const row = [];
  for (let shift = -0.3; shift <= 0.3001; shift += 0.01) {
    const shifted = estimate.notes.map((note) => ({
      ...note,
      start: note.start + shift,
      end: note.end + shift,
    }));
    const trial = score(reference.notes, shifted, tolerance);
    row.push({ shift, f1: trial.f1 });
    if (trial.f1 > best.f1) best = { shift, f1: trial.f1 };
  }
  for (const { shift, f1 } of row) {
    if (Math.abs(shift % 0.05) < 0.006) {
      const bar = '#'.repeat(Math.round(f1 * 40));
      console.log(`  ${(shift * 1000).toFixed(0).padStart(5)} ms  ${(f1 * 100).toFixed(1).padStart(5)}%  ${bar}`);
    }
  }
  console.log(
    `\nbest shift ${(best.shift * 1000).toFixed(0)} ms lifts F1 to ${(best.f1 * 100).toFixed(1)}% ` +
      `(from ${(result.f1 * 100).toFixed(1)}%)`,
  );
}
