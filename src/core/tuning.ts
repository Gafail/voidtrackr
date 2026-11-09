/**
 * Tuning estimation.
 *
 * The model was trained on A440 equal temperament. A recording that is a
 * quarter tone flat halves the activation of every note, so it is worth
 * measuring how far off the recording is and correcting it before the model
 * ever sees the audio.
 *
 * The correction is a resampling: shifting every frequency by `cents` is the
 * same as playing the recording `cents` cents sharp, which we do by changing
 * the playback rate and interpolating.
 */

import { fft } from './dsp';
import { SAMPLE_RATE } from './constants';

export interface TuningEstimate {
  /** Deviation from A440, in cents. Positive means the recording is sharp. */
  cents: number;
  /** How consistent the measurement is, 0..1. */
  strength: number;
}

/** Number of seconds of audio used for the measurement. */
const ANALYSIS_SECONDS = 60;

/**
 * Measure how far the recording is from A440.
 *
 * For each frame we find the strongest spectral peaks, work out how far each
 * one is from the nearest equal-tempered note, and average those deviations
 * as a circular mean so that +50 cents and -50 cents are the same thing.
 */
export function estimateTuning(samples: Float32Array, sampleRate = SAMPLE_RATE): TuningEstimate {
  const size = 4096;
  const hop = 2048;
  const loHz = 80;
  const hiHz = 4000;

  const limit = Math.min(samples.length, Math.floor(ANALYSIS_SECONDS * sampleRate));
  if (limit < size) return { cents: 0, strength: 0 };

  const loBin = Math.max(1, Math.ceil((loHz * size) / sampleRate));
  const hiBin = Math.min(size / 2 - 1, Math.floor((hiHz * size) / sampleRate));

  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size);

  const re = new Float32Array(size);
  const im = new Float32Array(size);

  let sx = 0;
  let sy = 0;
  let total = 0;

  for (let start = 0; start + size <= limit; start += hop) {
    for (let i = 0; i < size; i++) {
      re[i] = samples[start + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);

    const magnitudes = new Float32Array(size / 2);
    let top = -Infinity;
    for (let k = 0; k < size / 2; k++) {
      magnitudes[k] = Math.hypot(re[k], im[k]);
      if (magnitudes[k] > top) top = magnitudes[k];
    }
    if (!(top > 0)) continue;

    for (let k = loBin; k <= hiBin; k++) {
      if (!(magnitudes[k] > magnitudes[k - 1] && magnitudes[k] >= magnitudes[k + 1])) continue;
      if (magnitudes[k] <= top - 3.5) continue;

      // Parabolic interpolation on the log magnitude gives a much better
      // frequency estimate than the bin centre.
      const a = Math.log(magnitudes[k - 1] + 1e-12);
      const b = Math.log(magnitudes[k] + 1e-12);
      const c = Math.log(magnitudes[k + 1] + 1e-12);
      const denominator = a - 2 * b + c;
      const shift = Math.abs(denominator) > 1e-12 ? (0.5 * (a - c)) / denominator : 0;
      const frequency = (k + shift) * (sampleRate / size);

      const deviation = 1200 * Math.log2(frequency / 440);
      // Wrap to +/- 50 cents so a note a semitone away does not drag the
      // measurement.
      const wrapped = (deviation / 100 - Math.round(deviation / 100)) * 100;
      const weight = Math.exp(magnitudes[k] - top);

      const angle = (wrapped / 100) * 2 * Math.PI;
      sx += weight * Math.cos(angle);
      sy += weight * Math.sin(angle);
      total += weight;
    }
  }

  if (total <= 0) return { cents: 0, strength: 0 };

  const strength = Math.hypot(sx, sy) / total;
  const cents = (Math.atan2(sy, sx) / (2 * Math.PI)) * 100;
  return { cents, strength };
}

/** Below this deviation we assume the recording is in tune. */
const MIN_CENTS = 20;

/** Below this consistency we do not trust the measurement. */
const MIN_STRENGTH = 0.06;

/**
 * Return a copy of the audio tuned to A440, plus a function that maps a time
 * in the corrected audio back to the original.
 *
 * If the recording is already in tune the original array is returned
 * unchanged and the mapping is the identity.
 */
export function correctTuning(
  samples: Float32Array,
  sampleRate = SAMPLE_RATE,
): {
  samples: Float32Array;
  toSource: (time: number) => number;
  applied: boolean;
  cents: number;
} {
  const { cents, strength } = estimateTuning(samples, sampleRate);
  if (Math.abs(cents) < MIN_CENTS || strength < MIN_STRENGTH) {
    return { samples, toSource: (time) => time, applied: false, cents: 0 };
  }

  // Playing back faster raises the pitch, so to lower a sharp recording we
  // slow it down: the resampling ratio is 2^(-cents/1200).
  const ratio = Math.pow(2, -cents / 1200);
  const corrected = resampleWithRatio(samples, ratio);
  return { samples: corrected, toSource: (time) => time * ratio, applied: true, cents };
}

/** Resample by an arbitrary ratio with cubic interpolation. */
function resampleWithRatio(samples: Float32Array, ratio: number): Float32Array {
  const length = Math.ceil(samples.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const t = pos - idx;
    const p0 = samples[idx - 1] ?? samples[0];
    const p1 = samples[idx] ?? samples[0];
    const p2 = samples[idx + 1] ?? samples[0];
    const p3 = samples[idx + 2] ?? p2;
    out[i] =
      0.5 *
      (2 * p1 +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
  }
  return out;
}

/**
 * Energy above a cutoff, relative to the whole signal, in dB.
 *
 * Used to detect recordings that contain nothing but bass.
 */
export function highFrequencyRatioDb(
  samples: Float32Array,
  cutoff: number,
  sampleRate = SAMPLE_RATE,
): number {
  const size = 1024;
  const hop = 512;
  if (samples.length < size) return 0;

  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size);

  const re = new Float32Array(size);
  const im = new Float32Array(size);
  const cutoffBin = Math.floor((cutoff * size) / sampleRate);

  let highEnergy = 0;
  let totalEnergy = 0;

  for (let start = 0; start + size <= samples.length; start += hop) {
    for (let i = 0; i < size; i++) {
      re[i] = samples[start + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k < size / 2; k++) {
      const power = re[k] * re[k] + im[k] * im[k];
      totalEnergy += power;
      if (k >= cutoffBin) highEnergy += power;
    }
  }

  if (totalEnergy <= 0) return -Infinity;
  return 10 * Math.log10(highEnergy / totalEnergy + 1e-12);
}
