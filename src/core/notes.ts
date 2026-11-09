/**
 * Turning raw model output into discrete notes.
 *
 * The model produces three matrices:
 *   - `frames`   – per-frame probability that each pitch is sounding
 *   - `onsets`   – per-frame probability that each pitch was *struck*
 *   - `contours` – a finer-grained pitch trajectory (3 bins per semitone)
 *
 * A note is a strike (an onset peak) followed by a stretch of frames where the
 * pitch keeps sounding. This module implements that extraction, including the
 * optional "Melodia" fallback that recovers notes the onset head missed, and
 * the pitch bend curves read out of the contour head.
 *
 * The extraction works in model frames rather than seconds, because the
 * cleanup chain in `postprocess.ts` needs to look at neighbouring frames.
 */

import {
  ANNOTATIONS_BASE_FREQUENCY,
  CONTOURS_BINS_PER_SEMITONE,
  MIDI_OFFSET,
  MODEL_FRAME_SECONDS,
  N_FREQ_BINS_CONTOURS,
  modelFrameToTime,
  midiToHz,
  timeToModelFrame,
} from './constants';
import { ModelOutput, PITCH_BINS } from './model';
import { RawNote } from './types';
import { FrameNote } from './postprocess';

export interface NoteExtractionOptions {
  /** Onset strength required to start a note, 0..1. */
  onsetThreshold?: number;
  /** Frame probability below which a pitch counts as silent. `null` = auto. */
  frameThreshold?: number | null;
  /** Notes shorter than this many frames are discarded. */
  minNoteLengthFrames?: number;
  /** Use the frame derivative to synthesise extra onsets. */
  inferOnsets?: boolean;
  /** Ignore pitches above this frequency. */
  maxFreq?: number | null;
  /** Ignore pitches below this frequency. */
  minFreq?: number | null;
  /**
   * Recover notes from sustained frames that had no onset.
   *
   * Off by default: across varied recordings the greedy leftover-energy
   * harvest adds more wrong notes than it recovers.
   */
  melodiaTrick?: boolean;
  /** How many consecutive silent frames may be bridged before a note ends. */
  energyTolerance?: number;
}

const DEFAULTS: Required<NoteExtractionOptions> = {
  onsetThreshold: 0.5,
  frameThreshold: 0.3,
  minNoteLengthFrames: 4,
  inferOnsets: true,
  melodiaTrick: false,
  energyTolerance: 11,
  maxFreq: null,
  minFreq: null,
};

/** Read `data[row * bins + col]`. */
function at(data: Float32Array, row: number, col: number, bins: number): number {
  return data[row * bins + col];
}

/** Mean and standard deviation of every value in the frame matrix. */
function meanStdDev(data: Float32Array): [number, number] {
  let sum = 0;
  let sumSq = 0;
  const count = data.length;
  for (let i = 0; i < count; i++) {
    sum += data[i];
    sumSq += data[i] * data[i];
  }
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  return [mean, Math.sqrt(variance)];
}

/** Zero out pitch bins outside the requested frequency range. */
function constrainFrequency(
  onsets: Float32Array,
  frames: Float32Array,
  nFrames: number,
  maxFreq: number | null,
  minFreq: number | null,
): void {
  const bin = (hz: number) => Math.round(12 * (Math.log2(hz) - Math.log2(440)) + 69) - MIDI_OFFSET;
  if (maxFreq !== null) {
    const idx = Math.max(0, Math.min(PITCH_BINS, bin(maxFreq)));
    for (let r = 0; r < nFrames; r++) {
      const base = r * PITCH_BINS;
      for (let c = idx; c < PITCH_BINS; c++) {
        onsets[base + c] = 0;
        frames[base + c] = 0;
      }
    }
  }
  if (minFreq !== null) {
    const idx = Math.max(0, Math.min(PITCH_BINS, bin(minFreq)));
    for (let r = 0; r < nFrames; r++) {
      const base = r * PITCH_BINS;
      for (let c = 0; c < idx; c++) {
        onsets[base + c] = 0;
        frames[base + c] = 0;
      }
    }
  }
}

/**
 * Synthesise onsets from the frame matrix.
 *
 * A pitch that jumps up sharply between two frames was almost certainly
 * struck there, even if the onset head was not confident about it.
 */
function inferOnsetsFromFrames(onsets: Float32Array, frames: Float32Array, nFrames: number): Float32Array {
  const nDiff = 2;
  const diff = new Float32Array(nFrames * PITCH_BINS);

  for (let r = 0; r < nFrames; r++) {
    for (let c = 0; c < PITCH_BINS; c++) {
      const now = at(frames, r, c, PITCH_BINS);
      const d1 = r >= 1 ? at(frames, r - 1, c, PITCH_BINS) : 0;
      const d2 = r >= 2 ? at(frames, r - 2, c, PITCH_BINS) : 0;
      diff[r * PITCH_BINS + c] = Math.max(0, Math.min(now - d1, now - d2));
    }
  }
  // The first frames have no history, so they cannot show a rise.
  for (let r = 0; r < Math.min(nDiff, nFrames); r++) {
    diff.fill(0, r * PITCH_BINS, (r + 1) * PITCH_BINS);
  }

  // Scale the derivative to the onset head's range so the two can be compared.
  let onsetMax = 0;
  let diffMax = 0;
  for (let i = 0; i < onsets.length; i++) {
    if (onsets[i] > onsetMax) onsetMax = onsets[i];
    if (diff[i] > diffMax) diffMax = diff[i];
  }
  if (diffMax > 0) {
    const scale = onsetMax / diffMax;
    for (let i = 0; i < diff.length; i++) diff[i] *= scale;
  }

  const combined = new Float32Array(onsets.length);
  for (let i = 0; i < combined.length; i++) {
    combined[i] = Math.max(onsets[i], diff[i]);
  }
  return combined;
}

/**
 * Keep only values that are strictly greater than their immediate neighbours
 * in time.
 */
function relativeMaxima(data: Float32Array, nFrames: number, bins: number): number[] {
  const peaks: number[] = [];
  for (let c = 0; c < bins; c++) {
    for (let r = 0; r < nFrames; r++) {
      const value = data[r * bins + c];
      let isPeak = true;
      for (let rr = Math.max(0, r - 1); isPeak && rr <= Math.min(nFrames - 1, r + 1); rr++) {
        if (rr !== r && !(value > data[rr * bins + c])) {
          isPeak = false;
          break;
        }
      }
      if (isPeak) peaks.push(r * bins + c);
    }
  }
  return peaks;
}

/**
 * Extract notes from the model output, in model frames.
 */
export function extractFrameNotes(output: ModelOutput, options: NoteExtractionOptions = {}): FrameNote[] {
  const opts = { ...DEFAULTS, ...options };
  const { nFrames } = output;
  if (nFrames === 0) return [];

  const frames = Float32Array.from(output.frames);
  const onsets = Float32Array.from(output.onsets);

  constrainFrequency(onsets, frames, nFrames, opts.maxFreq, opts.minFreq);

  let frameThreshold = opts.frameThreshold;
  if (frameThreshold === null) {
    const [mean, std] = meanStdDev(frames);
    frameThreshold = mean + std;
  }

  const inferred = opts.inferOnsets
    ? inferOnsetsFromFrames(onsets, frames, nFrames)
    : Float32Array.from(onsets);

  // Only strict local maxima can start a note.
  const peaks = relativeMaxima(inferred, nFrames, PITCH_BINS);
  const peakThreshold = new Float32Array(nFrames * PITCH_BINS);
  for (const index of peaks) peakThreshold[index] = inferred[index];

  const candidates: { row: number; col: number }[] = [];
  for (let i = 0; i < peakThreshold.length; i++) {
    if (peakThreshold[i] > opts.onsetThreshold) {
      candidates.push({ row: Math.floor(i / PITCH_BINS), col: i % PITCH_BINS });
    }
  }
  // Newest first: the energy bookkeeping below depends on this order.
  candidates.reverse();

  const remaining = Float32Array.from(frames);
  const notes: FrameNote[] = [];

  const zeroWithNeighbours = (row: number, col: number) => {
    const base = row * PITCH_BINS;
    remaining[base + col] = 0;
    if (col < MAX_PITCH_INDEX) remaining[base + col + 1] = 0;
    if (col > 0) remaining[base + col - 1] = 0;
  };

  const onsetPeakAround = (row: number, col: number) => {
    let peak = 0;
    for (let f = row - 2; f <= row + 2; f++) {
      if (f < 0 || f >= nFrames) continue;
      peak = Math.max(peak, onsets[f * PITCH_BINS + col]);
    }
    return peak;
  };

  for (const candidate of candidates) {
    const { row: startRow, col } = candidate;
    if (startRow >= nFrames - 1) continue;

    let i = startRow + 1;
    let quiet = 0;
    while (i < nFrames - 1 && quiet < opts.energyTolerance) {
      if (remaining[i * PITCH_BINS + col] < frameThreshold) quiet += 1;
      else quiet = 0;
      i += 1;
    }
    i -= quiet;

    if (i - startRow <= opts.minNoteLengthFrames) continue;

    for (let j = startRow; j < i; j++) zeroWithNeighbours(j, col);

    let sum = 0;
    for (let j = startRow; j < i; j++) sum += frames[j * PITCH_BINS + col];
    const amplitude = sum / (i - startRow);

    notes.push({
      pitchIndex: col,
      startFrame: startRow,
      durationFrames: i - startRow,
      amplitude,
      onsetPeak: onsetPeakAround(startRow, col),
    });
  }

  if (opts.melodiaTrick) {
    // Recover notes that were sustained but never struck: look for the
    // strongest remaining pitch energy and grow a note around it.
    for (;;) {
      let bestValue = 0;
      let bestRow = -1;
      let bestCol = -1;
      for (let r = 0; r < nFrames; r++) {
        const base = r * PITCH_BINS;
        for (let c = 0; c < PITCH_BINS; c++) {
          if (remaining[base + c] > bestValue) {
            bestValue = remaining[base + c];
            bestRow = r;
            bestCol = c;
          }
        }
      }
      if (bestRow < 0 || bestValue <= frameThreshold) break;

      remaining[bestRow * PITCH_BINS + bestCol] = 0;

      let i = bestRow + 1;
      let quiet = 0;
      while (i < nFrames - 1 && quiet < opts.energyTolerance) {
        if (remaining[i * PITCH_BINS + bestCol] < frameThreshold) quiet += 1;
        else quiet = 0;
        zeroWithNeighbours(i, bestCol);
        i += 1;
      }
      const endRow = i - 1 - quiet;

      let j = bestRow - 1;
      quiet = 0;
      while (j > 0 && quiet < opts.energyTolerance) {
        if (remaining[j * PITCH_BINS + bestCol] < frameThreshold) quiet += 1;
        else quiet = 0;
        zeroWithNeighbours(j, bestCol);
        j -= 1;
      }
      const startRow = j + 1 + quiet;

      if (endRow - startRow <= opts.minNoteLengthFrames) continue;

      let sum = 0;
      for (let r = startRow; r < endRow; r++) sum += frames[r * PITCH_BINS + bestCol];
      const amplitude = sum / (endRow - startRow);

      notes.push({
        pitchIndex: bestCol,
        startFrame: startRow,
        durationFrames: endRow - startRow,
        amplitude,
        onsetPeak: onsetPeakAround(startRow, bestCol),
      });
    }
  }

  return notes;
}

/** Highest pitch bin the model reports. */
const MAX_PITCH_INDEX = 87;

/**
 * Extract notes and convert them to timed notes with pitch bends.
 */
export function extractNotes(output: ModelOutput, options: NoteExtractionOptions = {}): RawNote[] {
  const frameNotes = extractFrameNotes(output, options);
  const notes: RawNote[] = frameNotes.map((note) => ({
    pitchMidi: note.pitchIndex + MIDI_OFFSET,
    startTime: modelFrameToTime(note.startFrame),
    duration: modelFrameToTime(note.startFrame + note.durationFrames) - modelFrameToTime(note.startFrame),
    amplitude: note.amplitude,
  }));

  attachPitchBends(output.contours, output.nFrames, notes);
  sortNotes(notes);
  return notes;
}

/** Sort notes by start time, then pitch. */
export function sortNotes(notes: RawNote[]): void {
  notes.sort((a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi);
}

/**
 * Read a pitch bend curve for each note out of the contour head.
 *
 * The contour head resolves pitch three times finer than the frame head, so
 * it can describe a note that slides between two semitones. The value for
 * each frame is the contour bin that best matches the note's pitch, offset so
 * that 0 means "in tune".
 */
export function attachPitchBends(contours: Float32Array, nFrames: number, notes: RawNote[]): void {
  if (contours.length === 0) return;

  const nBinsTolerance = 25;
  const windowLength = nBinsTolerance * 2 + 1;
  const gaussian = new Float32Array(windowLength);
  for (let n = 0; n < windowLength; n++) {
    gaussian[n] = Math.exp((-1 * (n - (windowLength - 1) / 2) ** 2) / (2 * 5 ** 2));
  }

  for (const note of notes) {
    const bin = Math.floor(
      Math.round(
        12 * CONTOURS_BINS_PER_SEMITONE * Math.log2(midiToHz(note.pitchMidi) / ANNOTATIONS_BASE_FREQUENCY),
      ),
    );
    const start = Math.max(bin - nBinsTolerance, 0);
    const end = Math.min(N_FREQ_BINS_CONTOURS, bin + nBinsTolerance + 1);

    const gaussianStart = Math.max(0, nBinsTolerance - bin);
    const gaussianEnd = windowLength - Math.max(0, bin - (N_FREQ_BINS_CONTOURS - nBinsTolerance - 1));
    const weights = gaussian.subarray(gaussianStart, Math.max(gaussianStart, gaussianEnd));

    const shift = nBinsTolerance - Math.max(0, nBinsTolerance - bin);

    const frameStart = Math.max(0, timeToModelFrame(note.startTime));
    const frameEnd = Math.min(
      nFrames,
      frameStart + Math.max(1, Math.round(note.duration / MODEL_FRAME_SECONDS)),
    );
    const frameCount = Math.max(1, frameEnd - frameStart);

    const bends = new Array<number>(frameCount);
    for (let f = 0; f < frameCount; f++) {
      const row = Math.min(nFrames - 1, frameStart + f);
      const base = row * N_FREQ_BINS_CONTOURS;
      let best = -Infinity;
      let bestIdx = start;
      for (let c = start; c < end; c++) {
        const weight = weights[c - start] ?? 0;
        const value = contours[base + c] * weight;
        if (value > best) {
          best = value;
          bestIdx = c;
        }
      }
      bends[f] = bestIdx - shift;
    }
    note.pitchBends = bends;
  }
}
