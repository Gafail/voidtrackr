/**
 * How far may a note cross to the other hand?
 *
 *   npx tsx scripts/pianify-handover.mts [eval/midi]
 *
 * A note the left hand cannot take may be played by the right hand instead,
 * and the other way round. How far it may cross is a question of taste, and
 * the shipped version answered it badly: it allowed a note to cross an
 * octave, so the left hand reached up to MIDI 87 and the right hand came
 * down to 46. The hands were playing each other's music.
 *
 * At the other end, a handover of nothing means a note one semitone from the
 * split is dropped rather than played, which is also a waste.
 *
 * This sweeps the distance and reports what each one buys and costs: the
 * register each hand ends up in, the melody that survives, and the accuracy
 * against Okarina's transcription of the same audio.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import ToneMidi from '@tonejs/midi';

import { pianify } from '../src/core/pianify';
import { RawNote, Track } from '../src/core/types';

const Midi = ToneMidi.Midi;

const CLUSTER = 0.025;

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

function toRaw(notes: Note[]): RawNote[] {
  return notes.map((note) => ({
    pitchMidi: note.pitch,
    startTime: note.start,
    duration: Math.max(0.01, note.end - note.start),
    amplitude: note.amplitude,
  }));
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

function chords(notes: RawNote[]): RawNote[][] {
  const groups: RawNote[][] = [];
  let current: RawNote[] = [];
  for (const note of notes) {
    if (current.length === 0) {
      current.push(note);
      continue;
    }
    const previous = current[current.length - 1];
    if (note.startTime - previous.startTime <= CLUSTER) {
      current.push(note);
      continue;
    }
    groups.push(current);
    current = [note];
  }
  if (current.length > 0) groups.push(current);
  return groups;
}

function kept(arranged: RawNote[], pitch: number, start: number): boolean {
  return arranged.some(
    (note) => note.pitchMidi === pitch && Math.abs(note.startTime - start) <= 0.05,
  );
}

const directory = process.argv[2] ?? 'eval/midi';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.piano.mid'))
  .sort();

const distances = [0, 1, 2, 3, 4, 5, 7, 12];

console.log(`${files.length} files from ${directory}\n`);
console.log(
  'crossing   left max  right min   notes    L/R     melody   bass      F1',
);
console.log('-'.repeat(74));

for (const distance of distances) {
  let leftMax = 0;
  let rightMin = 200;
  let notes = 0;
  let leftCount = 0;
  let rightCount = 0;
  let melodyKept = 0;
  let melodyTotal = 0;
  let bassKept = 0;
  let bassTotal = 0;
  let f1 = 0;

  for (const file of files) {
    const source = readNotes(join(directory, file));
    if (source.length === 0) continue;
    const reference = readNotes(join(directory, file.replace('.piano.mid', '.okarina.mid')));

    const raw = toRaw(source);
    const track: Track = { id: 'piano', name: 'Piano', kind: 'melody', notes: raw };

    const result = pianify([track], { moveRange: distance });
    const left = result.tracks.find((entry) => entry.id === 'left-hand')?.notes ?? [];
    const right = result.tracks.find((entry) => entry.id === 'right-hand')?.notes ?? [];
    const arranged = [...left, ...right];

    leftCount += left.length;
    rightCount += right.length;
    notes += arranged.length;

    if (left.length > 0) leftMax = Math.max(leftMax, ...left.map((note) => note.pitchMidi));
    if (right.length > 0) rightMin = Math.min(rightMin, ...right.map((note) => note.pitchMidi));

    for (const chord of chords(raw)) {
      const highest = chord[chord.length - 1];
      const lowest = chord[0];
      melodyTotal += 1;
      if (kept(arranged, highest.pitchMidi, highest.startTime)) melodyKept += 1;
      if (lowest.pitchMidi < 60) {
        bassTotal += 1;
        if (kept(arranged, lowest.pitchMidi, lowest.startTime)) bassKept += 1;
      }
    }

    f1 += score(
      reference,
      arranged.map((note) => ({
        pitch: note.pitchMidi,
        start: note.startTime,
        end: note.startTime + note.duration,
        amplitude: note.amplitude,
      })),
      0.05,
    );
  }

  const percent = (part: number, whole: number) =>
    `${((part / Math.max(1, whole)) * 100).toFixed(1)}%`.padStart(6);

  console.log(
    `${String(distance).padStart(8)} ${String(leftMax).padStart(10)} ${String(rightMin).padStart(10)} ` +
      `${String(Math.round(notes / files.length)).padStart(7)} ` +
      `${String(Math.round(leftCount / files.length)).padStart(4)}/${String(Math.round(rightCount / files.length)).padEnd(4)} ` +
      `${percent(melodyKept, melodyTotal)} ${percent(bassKept, bassTotal)} ` +
      `${((f1 / files.length) * 100).toFixed(1).padStart(6)}%`,
  );
}

console.log(
  '\nThe crossing is how far, in semitones, a note may be played by the other hand.',
);
console.log('An octave was too far: the left hand reached into the treble.');
