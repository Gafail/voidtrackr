/**
 * Beat tracking.
 *
 * Two things need a beat grid. The first is measurement: to say whether a
 * transcription lines up with the music, the grid has to come from the
 * recording rather than from the notes, or the comparison is circular.
 *
 * The second is the output itself. Almost all the music this tool is used on
 * was sequenced to a grid, so a note that is a few milliseconds off is
 * almost certainly meant to be exactly on it. Snapping the onsets to the
 * grid removes the quantisation error that comes from the model's frame
 * rate, without changing which notes are there.
 *
 * The grid is found from the recording's own onset envelope: the flux of
 * the spectrum tells us when things happen, and the spacing between the
 * peaks tells us the tempo.
 */

import { binRange, normalize, spectrogram, Spectrogram } from './dsp';

/** Size of the analysis window, in samples. */
const WINDOW = 1024;

/** Hop between analysis windows, in samples. */
const HOP = 256;

/** Lowest frequency the onset envelope looks at. */
const LOW_HZ = 40;

/** Highest frequency the onset envelope looks at. */
const HIGH_HZ = 8000;

export interface BeatGrid {
  /** Beats per minute. */
  bpm: number;
  /**
   * Where the first beat falls, in seconds.
   *
   * Grid positions are `phase + k * 60 / bpm`.
   */
  phase: number;
  /** 0..1. How sure the tracker is. */
  confidence: number;
  /** Seconds between grid positions. One beat is 60 / bpm. */
  period: number;
}

/**
 * Find the beat grid of a recording.
 *
 * The onset envelope is built from the flux of the spectrum, then
 * autocorrelated to find the spacing that repeats. The phase is then chosen
 * so the grid sits on the strongest onsets.
 */
export function estimateBeatGrid(
  samples: Float32Array,
  sampleRate: number,
): BeatGrid {
  const spec = spectrogram(samples, {
    size: WINDOW,
    hop: HOP,
    sampleRate,
  });

  if (spec.frames < 16) {
    return { bpm: 120, phase: 0, confidence: 0, period: 0.5 };
  }

  const [lo, hi] = binRange(spec, LOW_HZ, HIGH_HZ);
  const envelope = onsetEnvelope(spec, lo, hi);
  const fps = spec.fps;
  const period = findPeriod(envelope, fps);
  const bpm = 60 / period;

  // The phase that puts the most onset energy on the grid.
  const phase = findPhase(envelope, fps, period);

  // Confidence: how much stronger the chosen period is than the rest.
  const confidence = periodConfidence(envelope, fps, period);

  return { bpm, phase, confidence, period };
}

/**
 * The onset envelope: how much energy appears at each frame.
 *
 * Positive differences only, because a note starting is what matters here,
 * not a note stopping.
 */
function onsetEnvelope(spec: Spectrogram, lo: number, hi: number): Float32Array {
  const envelope = new Float32Array(spec.frames);

  for (let f = 1; f < spec.frames; f++) {
    let sum = 0;
    for (let b = lo; b < hi; b++) {
      const current = spec.mags[f * spec.bins + b];
      const previous = spec.mags[(f - 1) * spec.bins + b];
      const difference = current - previous;
      if (difference > 0) sum += difference;
    }
    envelope[f] = sum;
  }

  return normalize(envelope);
}

/**
 * Find the spacing at which the onsets repeat.
 *
 * The envelope is autocorrelated and the strongest lag wins. The search is
 * limited to a musical range, and a lag that is a multiple of a shorter
 * period is penalised so a half-time tempo does not win over the real one.
 */
function findPeriod(envelope: Float32Array, fps: number): number {
  const minBpm = 60;
  const maxBpm = 200;

  const minLag = Math.max(2, Math.floor((60 / maxBpm) * fps));
  const maxLag = Math.min(envelope.length - 1, Math.ceil((60 / minBpm) * fps));

  let bestLag = Math.round((60 / 120) * fps);
  let bestScore = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    let count = 0;
    for (let f = lag; f < envelope.length; f++) {
      sum += envelope[f] * envelope[f - lag];
      count += 1;
    }
    if (count === 0) continue;

    // Normalise by the overlap so long lags are not favoured for having
    // fewer terms.
    let score = sum / count;

    // Prefer the shortest lag that scores nearly as well: a lag that is a
    // multiple of a shorter one describes the same rhythm at half the
    // tempo.
    if (lag % 2 === 0) {
      const half = score * 1.15;
      let halfSum = 0;
      let halfCount = 0;
      for (let f = lag / 2; f < envelope.length; f++) {
        halfSum += envelope[f] * envelope[f - lag / 2];
        halfCount += 1;
      }
      if (halfCount > 0 && halfSum / halfCount >= half) score *= 0.9;
    }

    // A small preference for tempos near 120 BPM, where most music sits.
    const bpm = (60 * fps) / lag;
    score *= 1 - 0.12 * Math.abs(Math.log2(bpm / 120));

    if (score > bestScore) {
      bestScore = score;
      bestLag = lag;
    }
  }

  return bestLag / fps;
}

/**
 * Find where in the cycle the beats fall.
 *
 * The phase is searched in small steps, and for each one the grid is laid
 * over the envelope and the energy on it is summed.
 */
function findPhase(envelope: Float32Array, fps: number, period: number): number {
  const steps = 48;
  const stepSize = period / steps;

  let bestPhase = 0;
  let bestScore = -Infinity;

  for (let s = 0; s < steps; s++) {
    const phase = s * stepSize;
    let score = 0;
    for (let t = phase; t < envelope.length; t += period) {
      // Take the strongest of the two nearest frames, so a beat that falls
      // between two frames is not missed.
      const frame = Math.round(t * fps);
      const value = envelope[Math.min(envelope.length - 1, Math.max(0, frame))];
      score += value;
    }
    if (score > bestScore) {
      bestScore = score;
      bestPhase = phase;
    }
  }

  return bestPhase;
}

/**
 * How sure the tracker is, 0..1.
 *
 * Compares the strength of the chosen period with the strongest period
 * overall. A recording with no clear pulse gives a low number.
 */
function periodConfidence(envelope: Float32Array, fps: number, period: number): number {
  const lag = Math.max(2, Math.round(period * fps));
  const chosen = correlation(envelope, lag);

  let best = 0;
  for (let other = 2; other < Math.min(envelope.length, lag * 4); other++) {
    best = Math.max(best, correlation(envelope, other));
  }
  if (best <= 0) return 0;

  return Math.max(0, Math.min(1, chosen / best));
}

/** Normalised autocorrelation at one lag. */
function correlation(envelope: Float32Array, lag: number): number {
  let sum = 0;
  let count = 0;
  for (let f = lag; f < envelope.length; f++) {
    sum += envelope[f] * envelope[f - lag];
    count += 1;
  }
  if (count === 0) return 0;

  let energy = 0;
  for (let f = 0; f < envelope.length; f++) energy += envelope[f] * envelope[f];
  if (energy <= 0) return 0;

  return sum / Math.sqrt(energy * count);
}

/**
 * Snap a time to the nearest grid position.
 *
 * Returns the original time when the grid is further away than the
 * tolerance, so a note that clearly sits off the beat is left alone.
 */
export function snapToGrid(time: number, grid: BeatGrid, tolerance: number): number {
  if (!Number.isFinite(grid.period) || grid.period <= 0) return time;

  const relative = time - grid.phase;
  const steps = Math.round(relative / grid.period);
  const snapped = grid.phase + steps * grid.period;

  if (Math.abs(snapped - time) > tolerance) return time;
  return snapped;
}

/**
 * How far a time is from the nearest grid position, in seconds.
 */
export function distanceToGrid(time: number, grid: BeatGrid): number {
  if (!Number.isFinite(grid.period) || grid.period <= 0) return Infinity;
  const relative = time - grid.phase;
  const steps = Math.round(relative / grid.period);
  return Math.abs(time - (grid.phase + steps * grid.period));
}

/** The frequency range of the spectrogram, as bin indices. */
