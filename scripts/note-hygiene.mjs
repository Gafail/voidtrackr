/**
 * What is actually wrong with a set of transcriptions?
 *
 *   node scripts/note-hygiene.mjs eval/midi/*.piano.mid
 *
 * The band-split idea is meant to give clearer tracks. Before trying it, it
 * is worth knowing what "unclear" actually looks like in the output: notes
 * that are cut in half, notes that overlap their own kind, notes that are
 * really harmonics reported an octave away, and notes too short to be
 * played.
 *
 * This counts those things, so the alternative — cleaning the output up —
 * can be judged on the same footing as the alternative of transcribing the
 * audio again in pieces.
 */

import { readFileSync } from 'node:fs';
import * as ToneMidi from '@tonejs/midi';

const Midi = ToneMidi.Midi ?? ToneMidi.default?.Midi ?? ToneMidi.default;

function readNotes(path) {
  const midi = new Midi(readFileSync(path));
  const notes = [];
  for (const track of midi.tracks) {
    for (const note of track.notes) {
      notes.push({
        pitch: note.midi,
        start: note.time,
        end: note.time + note.duration,
        duration: note.duration,
      });
    }
  }
  return notes;
}

/** Do two notes overlap in time? */
function overlaps(a, b) {
  return a.start < b.end - 1e-6 && b.start < a.end - 1e-6;
}

function analyse(path) {
  const notes = readNotes(path);
  if (notes.length === 0) return { path, count: 0 };

  let tooShort = 0;
  let overlapping = 0;
  let octavePairs = 0;
  let adjacent = 0;

  for (let i = 0; i < notes.length; i++) {
    const a = notes[i];

    if (a.duration < 0.05) tooShort += 1;

    for (let j = 0; j < notes.length; j++) {
      if (i === j) continue;
      const b = notes[j];

      // The same pitch sounding twice at once: a note counted twice.
      if (b.pitch === a.pitch && overlaps(a, b)) {
        overlapping += 1;
        break;
      }

      // The same pitch struck again a moment later: one note cut in two.
      if (
        b.pitch === a.pitch &&
        !overlaps(a, b) &&
        b.start >= a.start &&
        b.start - a.end < 0.06 &&
        b.start - a.end > -0.06
      ) {
        adjacent += 1;
        break;
      }

      // A note and the same note an octave away, starting together: the
      // harmonic reported as a note of its own.
      if (
        (b.pitch === a.pitch + 12 || b.pitch === a.pitch - 12) &&
        Math.abs(b.start - a.start) < 0.03
      ) {
        octavePairs += 1;
        break;
      }
    }
  }

  const duration = Math.max(...notes.map((note) => note.end));

  return {
    path,
    count: notes.length,
    perSecond: notes.length / Math.max(1, duration),
    tooShort,
    overlapping,
    adjacent,
    octavePairs,
  };
}

const paths = process.argv.slice(2);
if (paths.length === 0) {
  console.error('Usage: node scripts/note-hygiene.mjs <midi files...>');
  process.exit(1);
}

console.log('file                                    notes   /s   short  overlap  cut  octave');
for (const path of paths) {
  const r = analyse(path);
  if (r.count === 0) {
    console.log(`${path}  (no notes)`);
    continue;
  }
  console.log(
    `${path.replace(/^.*[/\\]/, '').padEnd(38)} ${String(r.count).padStart(5)} ` +
      `${r.perSecond.toFixed(1).padStart(5)} ` +
      `${String(r.tooShort).padStart(6)} ` +
      `${String(r.overlapping).padStart(8)} ` +
      `${String(r.adjacent).padStart(4)} ` +
      `${String(r.octavePairs).padStart(6)}`,
  );
}
