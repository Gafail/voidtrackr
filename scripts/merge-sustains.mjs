/**
 * Join notes that are really one note.
 *
 *   node scripts/merge-sustains.mjs <input.mid> <output.mid> [gapMs]
 *
 * A sustain that the model reports in two pieces shows up as the same pitch
 * twice, a few tens of milliseconds apart, with nothing in between. Joining
 * those back together removes clutter from the piano roll without changing
 * which pitches are there or when they start.
 *
 * The gap is measured from the end of the first note to the start of the
 * next. Anything longer than the threshold is left alone.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import * as ToneMidi from '@tonejs/midi';

const Midi = ToneMidi.Midi ?? ToneMidi.default?.Midi ?? ToneMidi.default;

const [inputPath, outputPath, gapArg] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error('Usage: node scripts/merge-sustains.mjs <input.mid> <output.mid> [gapMs]');
  process.exit(1);
}
const gap = (Number(gapArg ?? 60)) / 1000;

const midi = new Midi(readFileSync(inputPath));

for (const track of midi.tracks) {
  // One list per pitch, so the merge only ever joins a note to a note of the
  // same pitch.
  const byPitch = new Map();
  for (const note of track.notes) {
    if (!byPitch.has(note.midi)) byPitch.set(note.midi, []);
    byPitch.get(note.midi).push(note);
  }

  const merged = [];
  for (const [, notes] of byPitch) {
    notes.sort((a, b) => a.time - b.time);
    // The note objects are kept and edited in place: `duration` is a setter
    // over the note's tick count, so a plain object would lose the change.
    let current = notes[0];
    for (let i = 1; i < notes.length; i++) {
      const next = notes[i];
      // The gap is from the end of the current note to the start of the next.
      const silence = next.time - (current.time + current.duration);
      if (next.time >= current.time && silence <= gap) {
        // The same pitch again, soon after: one sustain, reported twice.
        const end = Math.max(current.time + current.duration, next.time + next.duration);
        current.duration = end - current.time;
        current.velocity = Math.max(current.velocity, next.velocity);
      } else {
        merged.push(current);
        current = next;
      }
    }
    merged.push(current);
  }

  merged.sort((a, b) => a.time - b.time);
  track.notes = merged;
}

writeFileSync(outputPath, Buffer.from(midi.toArray()));
const total = midi.tracks.reduce((sum, track) => sum + track.notes.length, 0);
console.log(`${inputPath} -> ${outputPath}  (${total} notes, gap ${gap * 1000} ms)`);
