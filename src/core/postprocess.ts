/**
 * Note cleanup.
 *
 * The raw output of the model is noisy in ways that are hard to see until you
 * look at the notes: a single held note can be split in two by vibrato, a
 * chord can arrive with its notes a few frames apart, and the second
 * harmonic of a strong note can show up as a note of its own.
 *
 * Each step below fixes one of those problems. They are separate functions
 * rather than one big pass so they can be tuned, switched off, or tested in
 * isolation.
 */

import { ModelOutput, PITCH_BINS } from './model';
import { modelFrameToTime } from './constants';
import { RawNote } from './types';

/** A note while it is still expressed in model frames. */
export interface FrameNote {
  /** Index into the model's pitch bins (0 = A0). */
  pitchIndex: number;
  startFrame: number;
  durationFrames: number;
  amplitude: number;
  /** Strongest onset around the start of the note. */
  onsetPeak: number;
}

/**
 * Model frame rate, in frames per second.
 *
 * Only used to turn a duration into a number of frames. Absolute positions
 * always go through `modelFrameToTime`, because the frames of a window do
 * not sit a fixed distance apart in time.
 */
const FPS = 22050 / 256;

/** Notes shorter than this need a strong attack to survive. */
const FULL_LENGTH_FRAMES = 8;

/** Onset strength required for a short note to be kept. */
const SHORT_NOTE_ONSET = 0.7;

/** Below this activation a pitch counts as released. */
const RELEASE_FLOOR = 0.15;

/** How many sub-floor frames may be bridged when extending a note. */
const RELEASE_MAX_DIP = 4;

/** Maximum gap, in frames, for two notes of the same pitch to be joined. */
const JOIN_MAX_GAP = 4;

/** Harmonic intervals that are treated as overtones of a lower note. */
const OVERTONE_INTERVALS = [12, 19, 24, 28, 31, 36];

/** Chord notes that start within this many seconds are treated as one chord. */
const CHORD_WINDOW = 0.06;

/** A run is at least this many notes. */
const RUN_MIN_NOTES = 4;

/** Gaps between the notes of a run. */
const RUN_MIN_GAP = 0.005;
const RUN_MAX_GAP = 0.15;

/** Largest step, in semitones, between the notes of a run. */
const RUN_MAX_STEP = 7;

/** Convert frame notes into timed notes. */
export function toTimedNotes(notes: FrameNote[]): RawNote[] {
  return notes
    .map((note) => ({
      pitchMidi: note.pitchIndex + 21,
      startTime: modelFrameToTime(note.startFrame),
      duration:
        modelFrameToTime(note.startFrame + note.durationFrames) - modelFrameToTime(note.startFrame),
      amplitude: note.amplitude,
    }))
    .sort((a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi);
}

/** Read a frame's activation for one pitch. */
function activation(output: ModelOutput, frame: number, pitchIndex: number): number {
  if (frame < 0 || frame >= output.nFrames) return 0;
  return output.frames[frame * PITCH_BINS + pitchIndex];
}

/** Read a frame's onset strength for one pitch. */
function onset(output: ModelOutput, frame: number, pitchIndex: number): number {
  if (frame < 0 || frame >= output.nFrames) return 0;
  return output.onsets[frame * PITCH_BINS + pitchIndex];
}

/** The strongest onset within +/- `radius` frames of a note's start. */
export function onsetPeakAt(output: ModelOutput, frame: number, pitchIndex: number, radius = 2): number {
  let peak = 0;
  for (let f = frame - radius; f <= frame + radius; f++) {
    peak = Math.max(peak, onset(output, f, pitchIndex));
  }
  return peak;
}

/**
 * Decide whether a note really starts with a new attack.
 *
 * Vibrato and tremolo inside a held note create onset peaks that would
 * otherwise split it. A genuine re-attack looks different: the pitch either
 * faded away before the new hit, or it rose clearly out of a dip.
 */
export function isReattack(output: ModelOutput, pitchIndex: number, frame: number): boolean {
  if (onsetPeakAt(output, frame, pitchIndex, 2) >= 0.75) return true;

  let before = 0;
  for (let f = frame - 15; f < frame - 4; f++) before = Math.max(before, activation(output, f, pitchIndex));
  if (before <= 0) return false;

  let dip = Infinity;
  for (let f = frame - 5; f <= frame; f++) dip = Math.min(dip, activation(output, f, pitchIndex));

  let after = 0;
  for (let f = frame; f <= frame + 6; f++) after = Math.max(after, activation(output, f, pitchIndex));

  if (dip < before * 0.6) return true;
  return after - dip >= 0.3 && dip < before;
}

/**
 * Move each note's start back to where the pitch actually began.
 *
 * The decoder starts a note where the onset head fired, which is often a
 * little after the sound started. Walking back down the rising edge of the
 * activation finds the real beginning.
 *
 * The walk stops at the end of the previous note of the same pitch, so a
 * repeated note that is still sounding is never swallowed by its predecessor.
 */
export function startAtFirstAttack(notes: FrameNote[], output: ModelOutput): FrameNote[] {
  const byPitch = new Map<number, FrameNote[]>();
  for (const note of notes) {
    const list = byPitch.get(note.pitchIndex) ?? [];
    list.push(note);
    byPitch.set(note.pitchIndex, list);
  }

  for (const list of byPitch.values()) {
    list.sort((a, b) => a.startFrame - b.startFrame);
    let previousEnd = 0;

    for (const note of list) {
      // Only walk back over a rise that is actually a rise. The model's
      // activation never reaches zero, so "not falling" on its own would
      // drag the start back to the beginning of the file.
      const peak = peakActivation(output, note.pitchIndex, note.startFrame);
      const threshold = Math.max(0.1, peak * 0.15);
      const limit = Math.max(previousEnd, note.startFrame - 20);
      let start = note.startFrame;

      for (let f = note.startFrame - 1; f >= limit; f--) {
        const value = activation(output, f, note.pitchIndex);
        // Below this the pitch is silent, so the note starts after this frame.
        if (value < threshold) break;
        // Activation falling as we go back means we have left this note.
        if (value > activation(output, f + 1, note.pitchIndex)) break;
        start = f;
      }

      if (start < note.startFrame) {
        note.durationFrames += note.startFrame - start;
        note.startFrame = start;
      }
      previousEnd = note.startFrame + note.durationFrames;
    }
  }

  return notes;
}

/** The highest activation a pitch reaches around a frame. */
function peakActivation(output: ModelOutput, pitchIndex: number, frame: number): number {
  let peak = 0;
  for (let f = Math.max(0, frame - 8); f <= frame; f++) {
    peak = Math.max(peak, activation(output, f, pitchIndex));
  }
  return peak;
}

/**
 * Join notes of the same pitch that are really one long note.
 *
 * Two notes of the same pitch separated by a tiny gap belong together unless
 * there is a fresh attack between them.
 */
export function joinSustainedNotes(notes: FrameNote[], output: ModelOutput): FrameNote[] {
  const sorted = [...notes].sort((a, b) => a.startFrame - b.startFrame || a.pitchIndex - b.pitchIndex);
  const result: FrameNote[] = [];

  for (const note of sorted) {
    const previous = result[result.length - 1];
    const gap =
      previous && previous.pitchIndex === note.pitchIndex
        ? note.startFrame - (previous.startFrame + previous.durationFrames)
        : Infinity;

    // Two notes of the same pitch that touch, or nearly touch, are one note
    // unless the second one was really struck again.
    if (gap >= 0 && gap <= JOIN_MAX_GAP && !isReattack(output, note.pitchIndex, note.startFrame)) {
      const end = note.startFrame + note.durationFrames;
      previous.durationFrames = Math.max(previous.startFrame + previous.durationFrames, end) - previous.startFrame;
      previous.amplitude = Math.max(previous.amplitude, note.amplitude);
      previous.onsetPeak = Math.max(previous.onsetPeak, note.onsetPeak);
      continue;
    }
    result.push({ ...note });
  }

  return result;
}

/** Drop very short notes unless they were struck hard. */
export function gateShortNotes(notes: FrameNote[]): FrameNote[] {
  return notes.filter(
    (note) => note.durationFrames >= FULL_LENGTH_FRAMES || note.onsetPeak >= SHORT_NOTE_ONSET,
  );
}

/**
 * Extend a note while its pitch is still audible.
 *
 * The model's confidence fades faster than the note does, so a held note
 * often ends too early. The extension stops as soon as the pitch really has
 * gone quiet.
 */
export function extendReleases(notes: FrameNote[], output: ModelOutput): FrameNote[] {
  const byPitch = new Map<number, FrameNote[]>();
  for (const note of notes) {
    const list = byPitch.get(note.pitchIndex) ?? [];
    list.push(note);
    byPitch.set(note.pitchIndex, list);
  }

  for (const list of byPitch.values()) {
    list.sort((a, b) => a.startFrame - b.startFrame);
    for (let i = 0; i < list.length; i++) {
      const note = list[i];
      const next = list[i + 1];
      const limit = next ? next.startFrame : output.nFrames;

      let end = note.startFrame + note.durationFrames;
      let dip = 0;
      for (let f = end; f < limit; f++) {
        if (onset(output, f, note.pitchIndex) >= 0.5) break;
        if (activation(output, f, note.pitchIndex) >= RELEASE_FLOOR) {
          end = f + 1;
          dip = 0;
        } else {
          dip += 1;
          if (dip > RELEASE_MAX_DIP) break;
        }
      }
      note.durationFrames = Math.max(note.durationFrames, end - note.startFrame);
    }
  }

  return notes;
}

/**
 * Recover notes that swelled in without a detectable attack.
 *
 * Soft legato lines (a breathy lead, some synths) have no onset peak, so the
 * decoder never starts a note for them. A swell is only accepted when some
 * other pitch was clearly loud just before it and faded during the rise,
 * which is what "the line moved on" looks like.
 */
export function rescueSwells(notes: FrameNote[], output: ModelOutput): FrameNote[] {
  const PEAK = 0.28;
  const RISE = 0.15;
  const FLOOR = 0.16;
  const LOOKBACK = 15;
  const MIN_FRAMES = 8;
  const MAX_PITCH = 84;
  const HANDOFF_LEVEL = 0.3;
  const HANDOFF_DIP = 0.2;

  const recovered: FrameNote[] = [];

  for (let pitchIndex = 0; pitchIndex < PITCH_BINS; pitchIndex++) {
    const midi = pitchIndex + 21;
    if (midi > MAX_PITCH) continue;

    let f = 1;
    while (f < output.nFrames - 1) {
      const value = activation(output, f, pitchIndex);
      if (
        value < PEAK ||
        value <= activation(output, f - 1, pitchIndex) ||
        value <= activation(output, f + 1, pitchIndex)
      ) {
        f += 1;
        continue;
      }

      // Walk back down the rising flank.
      let start = f;
      while (
        start > 0 &&
        activation(output, start - 1, pitchIndex) <= activation(output, start, pitchIndex) &&
        activation(output, start - 1, pitchIndex) >= FLOOR * 0.5
      ) {
        start -= 1;
      }

      // Walk forward while the pitch is still sounding.
      const hold = PEAK * 0.6;
      let end = f;
      while (end + 1 < output.nFrames && activation(output, end + 1, pitchIndex) >= hold) {
        end += 1;
      }

      let base = Infinity;
      for (let g = Math.max(0, start - LOOKBACK); g <= start; g++) {
        base = Math.min(base, activation(output, g, pitchIndex));
      }

      const neighbourLoud =
        activation(output, f, pitchIndex - 1) > value || activation(output, f, pitchIndex + 1) > value;

      const handoff = hasHandoff(output, pitchIndex, start, f, HANDOFF_LEVEL, HANDOFF_DIP);

      if (
        base <= FLOOR &&
        value - base >= RISE &&
        end - start + 1 >= MIN_FRAMES &&
        !neighbourLoud &&
        handoff
      ) {
        let sum = 0;
        for (let g = start; g <= end; g++) sum += activation(output, g, pitchIndex);
        recovered.push({
          pitchIndex,
          startFrame: start,
          durationFrames: end - start + 1,
          amplitude: sum / (end - start + 1),
          onsetPeak: onsetPeakAt(output, start, pitchIndex),
        });
        f = end + 1;
      } else {
        f += 1;
      }
    }
  }

  const existing = new Set(
    notes.map((note) => `${note.pitchIndex}:${note.startFrame}`),
  );
  const added = recovered.filter((note) => {
    if (existing.has(`${note.pitchIndex}:${note.startFrame}`)) return false;
    // A swell that overlaps, or that runs straight into a note we already
    // have, is that note's attack rather than a separate note.
    const nearby = notes.some(
      (other) =>
        other.pitchIndex === note.pitchIndex &&
        note.startFrame < other.startFrame + other.durationFrames + SWELL_PROXIMITY &&
        other.startFrame < note.startFrame + note.durationFrames + SWELL_PROXIMITY,
    );
    return !nearby;
  });
  return [...notes, ...added];
}

/** How close a decoded note has to be for a swell to be treated as its attack. */
const SWELL_PROXIMITY = 12;

/**
 * Was some other pitch loud before this rise and quiet during it?
 */
function hasHandoff(
  output: ModelOutput,
  pitchIndex: number,
  start: number,
  peak: number,
  level: number,
  dip: number,
): boolean {
  for (let other = 0; other < PITCH_BINS; other++) {
    if (Math.abs(other - pitchIndex) > 12) continue;

    let loudBefore = 0;
    for (let f = Math.max(0, start - 12); f < start; f++) {
      loudBefore = Math.max(loudBefore, activation(output, f, other));
    }
    if (loudBefore < level) continue;

    let during = Infinity;
    for (let f = start; f <= peak; f++) during = Math.min(during, activation(output, f, other));
    if (during <= loudBefore * (1 - dip)) return true;
  }
  return false;
}

/**
 * Remove notes that are really overtones of a lower note.
 *
 * A strong note's second harmonic often shows up as a weaker note an octave
 * above, sounding through most of the original.
 */
export function removeOvertones(notes: FrameNote[]): FrameNote[] {
  const result: FrameNote[] = [];

  for (const note of notes) {
    let isOvertone = false;

    for (const lower of notes) {
      if (lower === note) continue;
      const interval = note.pitchIndex - lower.pitchIndex;
      if (!OVERTONE_INTERVALS.includes(interval)) continue;

      const overlap = overlapFrames(note, lower);
      if (overlap < 0.8 * note.durationFrames) continue;
      if (note.amplitude >= 0.6 * lower.amplitude) continue;

      isOvertone = true;
      break;
    }

    if (!isOvertone) result.push(note);
  }

  return result;
}

/** How many frames two notes of the same pitch overlap. */
function overlapFrames(a: FrameNote, b: FrameNote): number {
  const start = Math.max(a.startFrame, b.startFrame);
  const end = Math.min(a.startFrame + a.durationFrames, b.startFrame + b.durationFrames);
  return Math.max(0, end - start);
}

/**
 * Find runs: quick scalar passages that should not be treated as chords.
 */
export function detectRuns(notes: FrameNote[]): Set<FrameNote> {
  const runs = new Set<FrameNote>();
  const sorted = [...notes].sort((a, b) => a.startFrame - b.startFrame);

  for (const direction of [1, -1]) {
    let best: FrameNote[] = [];
    for (const note of sorted) {
      const previous = best[best.length - 1];
      if (
        previous &&
        note.startFrame > previous.startFrame &&
        (note.startFrame - previous.startFrame) / FPS >= RUN_MIN_GAP &&
        (note.startFrame - previous.startFrame) / FPS <= RUN_MAX_GAP &&
        Math.abs(note.pitchIndex - previous.pitchIndex) >= 1 &&
        Math.abs(note.pitchIndex - previous.pitchIndex) <= RUN_MAX_STEP &&
        Math.sign(note.pitchIndex - previous.pitchIndex) === direction
      ) {
        best.push(note);
      } else {
        if (best.length >= RUN_MIN_NOTES) for (const n of best) runs.add(n);
        best = [note];
      }
    }
    if (best.length >= RUN_MIN_NOTES) for (const n of best) runs.add(n);
  }

  return runs;
}

/**
 * Pull the notes of a chord onto the same start time.
 *
 * The model reports the notes of a chord a few frames apart, which makes the
 * chord sound like an arpeggio when it is played back.
 */
export function alignChordOnsets(notes: FrameNote[], runs: Set<FrameNote>): FrameNote[] {
  const windowFrames = CHORD_WINDOW * FPS;
  const result = notes.map((note) => ({ ...note }));
  const sorted = [...result].sort((a, b) => a.startFrame - b.startFrame);

  let group: FrameNote[] = [];
  const flush = () => {
    if (group.length < 2) {
      group = [];
      return;
    }
    const starts = group.map((n) => n.startFrame).sort((a, b) => a - b);
    const median = starts[Math.floor(starts.length / 2)];
    for (const note of group) {
      if (runs.has(note)) continue;
      const end = note.startFrame + note.durationFrames;
      note.startFrame = median;
      note.durationFrames = Math.max(0.02 * FPS, end - median);
    }
    group = [];
  };

  for (const note of sorted) {
    if (group.length === 0) {
      group = [note];
      continue;
    }
    const anchor = group[0].startFrame;
    if (note.startFrame - anchor <= windowFrames) {
      group.push(note);
    } else {
      flush();
      group = [note];
    }
  }
  flush();

  return result;
}

/**
 * Repair octave errors in the low register.
 *
 * The model sometimes places a bass note an octave below where it belongs,
 * usually when the real fundamental is masked. When two notes an octave
 * apart sound at the same time, the louder one is nearly always the right
 * one, so the quieter one is dropped.
 */
export function repairOctaveErrors(notes: FrameNote[]): FrameNote[] {
  const result: FrameNote[] = [];

  for (const note of notes) {
    let dominated = false;
    for (const other of notes) {
      if (other === note) continue;
      if (other.pitchIndex - note.pitchIndex !== 12) continue;
      const overlap = overlapFrames(note, other);
      if (overlap < 0.5 * Math.min(note.durationFrames, other.durationFrames)) continue;
      if (other.amplitude > note.amplitude) {
        dominated = true;
        break;
      }
    }
    if (!dominated) result.push(note);
  }

  return result;
}

/**
 * The full cleanup chain.
 */
export function cleanNotes(notes: FrameNote[], output: ModelOutput): FrameNote[] {
  let working = notes;
  working = startAtFirstAttack(working, output);
  working = joinSustainedNotes(working, output);
  working = gateShortNotes(working);
  working = extendReleases(working, output);
  working = rescueSwells(working, output);
  working = removeOvertones(working);
  const runs = detectRuns(working);
  working = alignChordOnsets(working, runs);
  return working;
}
