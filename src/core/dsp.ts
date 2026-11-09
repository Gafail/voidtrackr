/**
 * Signal processing helpers shared by the drum detector, the tempo tracker
 * and the band-splitting stage.
 *
 * Everything works on plain `Float32Array`s so the same code runs in the
 * browser and in Node.
 */

import { SAMPLE_RATE } from './constants';

/** In-place iterative radix-2 FFT. `re` and `im` must have the same power-of-two length. */
export function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  if (n <= 1) return;

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const step = (-2 * Math.PI) / len;
    const wr = Math.cos(step);
    const wi = Math.sin(step);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k;
        const b = a + half;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

/** Magnitude spectrogram of a mono signal. */
export interface Spectrogram {
  /** Magnitudes, row-major `[frames * bins]`. */
  mags: Float32Array;
  frames: number;
  bins: number;
  /** Frames per second. */
  fps: number;
  /** Frequency of one FFT bin, in Hz. */
  binHz: number;
}

export interface SpectrogramOptions {
  /** FFT size. */
  size?: number;
  /** Hop size in samples. */
  hop?: number;
  /** Sample rate of the input. */
  sampleRate?: number;
}

export function spectrogram(samples: Float32Array, options: SpectrogramOptions = {}): Spectrogram {
  const size = options.size ?? 1024;
  const hop = options.hop ?? 256;
  const rate = options.sampleRate ?? SAMPLE_RATE;

  const bins = size / 2;
  const frames = Math.max(0, Math.floor((samples.length - size) / hop) + 1);
  const mags = new Float32Array(frames * bins);

  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size);

  const re = new Float32Array(size);
  const im = new Float32Array(size);

  for (let f = 0; f < frames; f++) {
    const offset = f * hop;
    for (let i = 0; i < size; i++) {
      const s = offset + i;
      re[i] = s >= 0 && s < samples.length ? samples[s] * window[i] : 0;
      im[i] = 0;
    }
    fft(re, im);
    const base = f * bins;
    for (let k = 0; k < bins; k++) {
      mags[base + k] = Math.hypot(re[k], im[k]);
    }
  }

  return { mags, frames, bins, fps: rate / hop, binHz: rate / size };
}

/** FFT bin range covering a frequency band. */
export function binRange(spec: Spectrogram, lo: number, hi: number): [number, number] {
  return [
    Math.max(1, Math.floor(lo / spec.binHz)),
    Math.min(spec.bins - 1, Math.floor(hi / spec.binHz)),
  ];
}

/**
 * Level of each band over time, using one band per semitone.
 *
 * Narrow bands at the low end are merged so the same frequency is not counted
 * twice.
 */
export interface BandLevels {
  levels: Float32Array;
  count: number;
  frames: number;
}

export function bandLevels(spec: Spectrogram, lo: number, hi: number): BandLevels {
  const edges: number[] = [];
  for (let f = lo; f <= hi * 1.0001; f *= Math.pow(2, 1 / 12)) edges.push(f);

  const bands: [number, number][] = [];
  let start = Math.floor(edges[0] / spec.binHz);
  for (let e = 1; e < edges.length; e++) {
    const end = Math.floor(edges[e] / spec.binHz);
    if (end > start) {
      bands.push([start, Math.min(end, spec.bins)]);
      start = end;
    }
  }

  const levels = new Float32Array(spec.frames * bands.length);
  for (let f = 0; f < spec.frames; f++) {
    const base = f * spec.bins;
    for (let b = 0; b < bands.length; b++) {
      let sum = 0;
      const [s, e] = bands[b];
      for (let k = s; k < e; k++) {
        const m = spec.mags[base + k];
        sum += m * m;
      }
      levels[f * bands.length + b] = Math.sqrt(sum / Math.max(1, e - s));
    }
  }

  return { levels, count: bands.length, frames: spec.frames };
}

/**
 * Onset strength per band: how much each band has risen over the last few
 * frames. A strike makes a band jump; a slow swell does not.
 *
 * `floor` is the share of a band's own loud level that counts as silence. It
 * stops faint leakage into an otherwise empty band (a snare bleeding into
 * the kick band) from registering as a large relative jump.
 */
export function bandFlux(
  spec: Spectrogram,
  lo: number,
  hi: number,
  lag = 4,
  floor = 0,
): Float32Array {
  const bands = bandLevels(spec, lo, hi);
  const out = new Float32Array(spec.frames);

  // Per-band silence threshold, taken from that band's own loud moments.
  const floors = new Float32Array(bands.count);
  for (let k = 0; k < bands.count; k++) {
    const column = new Float32Array(spec.frames);
    for (let f = 0; f < spec.frames; f++) column[f] = bands.levels[f * bands.count + k];
    column.sort();
    floors[k] = floor * (column[Math.floor(spec.frames * 0.99)] || 0);
  }

  for (let t = lag; t < spec.frames; t++) {
    let sum = 0;
    for (let k = 0; k < bands.count; k++) {
      let previous = floors[k];
      for (let l = 1; l <= lag; l++) {
        previous = Math.max(previous, bands.levels[(t - l) * bands.count + k]);
      }
      const current = Math.max(floors[k], bands.levels[t * bands.count + k]);
      const rise = Math.log(1 + 1000 * current) - Math.log(1 + 1000 * previous);
      if (rise > 0) sum += rise;
    }
    out[t] = sum / bands.count;
  }
  return out;
}

/** Full-spectrum flux: a broadband onset strength curve. */
export function spectralFlux(spec: Spectrogram): Float32Array {
  const out = new Float32Array(spec.frames);
  for (let t = 1; t < spec.frames; t++) {
    let sum = 0;
    for (let k = 1; k < spec.bins; k++) {
      const current = spec.mags[t * spec.bins + k];
      const previous = spec.mags[(t - 1) * spec.bins + k];
      const diff = current - previous;
      if (diff > 0) sum += diff;
    }
    out[t] = sum;
  }
  return out;
}

/** Divide every value by the 99.5th percentile, so the loudest moment is ~1. */
export function normalize(values: Float32Array): Float32Array {
  if (values.length === 0) return values;
  const sorted = Float32Array.from(values).sort();
  const top = sorted[Math.floor(values.length * 0.995)] || 1;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = values[i] / top;
  return out;
}

/** Median of an array of numbers. */
export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1];
}

/**
 * Spectral flatness of one frame over a bin range, 0..1.
 *
 * Flatness is low for a pure tone and close to 1 for noise, which is how we
 * tell a hi-hat from a bright pitched note.
 */
export function flatness(spec: Spectrogram, frame: number, range: [number, number]): number {
  const lo = Math.max(0, Math.min(spec.frames - 1, frame));
  const base = lo * spec.bins;
  let logSum = 0;
  let sum = 0;
  const n = Math.max(1, range[1] - range[0] + 1);
  for (let k = range[0]; k <= range[1]; k++) {
    const power = spec.mags[base + k] * spec.mags[base + k] + 1e-12;
    logSum += Math.log(power);
    sum += power;
  }
  return Math.exp(logSum / n) / (sum / n);
}

/**
 * Share of a frame's energy inside a band, in dB.
 *
 * A real hi-hat carries a noticeable part of the sound; the attack click of a
 * dark pitched note is broadband but tiny next to the note itself.
 */
export function bandShare(
  spec: Spectrogram,
  frame: number,
  band: [number, number],
  full: [number, number],
): number {
  const base = Math.max(0, Math.min(spec.frames - 1, frame)) * spec.bins;
  let inBand = 0;
  let total = 1e-12;
  for (let k = full[0]; k <= full[1]; k++) {
    const power = spec.mags[base + k] * spec.mags[base + k];
    total += power;
    if (k >= band[0] && k <= band[1]) inBand += power;
  }
  return (10 * Math.log(inBand / total + 1e-12)) / Math.LN10;
}

/** Energy of one frame inside a bin range. */
export function bandPower(spec: Spectrogram, frame: number, range: [number, number]): number {
  const base = Math.max(0, Math.min(spec.frames - 1, frame)) * spec.bins;
  let sum = 0;
  for (let k = range[0]; k <= range[1]; k++) sum += spec.mags[base + k] * spec.mags[base + k];
  return sum;
}

/**
 * How far a band falls in the 60 ms after a hit, in dB.
 *
 * A hi-hat or cymbal decays; a sung "s" or "sh" holds or grows. A negative
 * number means the band dropped.
 */
export function hatDecay(
  spec: Spectrogram,
  frame: number,
  range: [number, number],
  lagSeconds = 0.06,
): number | null {
  const lag = Math.round(lagSeconds * spec.fps);
  if (frame + lag >= spec.frames) return null;
  let peak = 0;
  for (let j = 0; j < 3; j++) peak = Math.max(peak, bandPower(spec, frame + j, range));
  return (10 * Math.log((bandPower(spec, frame + lag, range) + 1e-12) / (peak + 1e-12))) / Math.LN10;
}


/**
 * Pick peaks that stand out from their surroundings.
 *
 * A frame qualifies when it is a local maximum, is above `delta`, and is
 * `mult` times the local average.
 */
export function pickPeaks(
  curve: Float32Array,
  fps: number,
  delta: number,
  mult: number,
  options: { windowSeconds?: number; minGapSeconds?: number } = {},
): number[] {
  const windowSeconds = options.windowSeconds ?? 0.5;
  const minGapSeconds = options.minGapSeconds ?? 0.06;

  const half = Math.round((windowSeconds * fps) / 2);
  const gap = Math.round(minGapSeconds * fps);
  const peaks: number[] = [];

  const n = curve.length;
  let sum = 0;
  let from = 0;
  let to = -1;
  let last = -Infinity;

  for (let t = 1; t < n - 1; t++) {
    while (to < Math.min(n - 1, t + half)) sum += curve[++to];
    while (from < t - half) sum -= curve[from++];
    if (!(curve[t] >= curve[t - 1] && curve[t] > curve[t + 1])) continue;
    const localMean = sum / Math.max(1, to - from + 1);
    if (curve[t] > mult * localMean + delta && t - last >= gap) {
      peaks.push(t);
      last = t;
    }
  }
  return peaks;
}

/** Highest autocorrelation of a signal at a given lag, normalised to 0..1. */
export function periodicity(curve: Float32Array, minLag: number, maxLag: number): number {
  const n = curve.length;
  if (n === 0) return 0;

  let mean = 0;
  for (let i = 0; i < n; i++) mean += curve[i] / n;

  let energy = 0;
  for (let i = 0; i < n; i++) energy += (curve[i] - mean) * (curve[i] - mean);
  if (!(energy > 0)) return 0;

  // Lags must be whole samples, otherwise indexing the signal returns
  // undefined and the result becomes NaN.
  const first = Math.max(1, Math.round(minLag));
  const last = Math.min(Math.round(maxLag), n - 1);

  let best = 0;
  for (let lag = first; lag <= last; lag++) {
    let s = 0;
    for (let i = lag; i < n; i++) s += (curve[i] - mean) * (curve[i - lag] - mean);
    best = Math.max(best, s / energy);
  }
  return best;
}

/** Biquad filter coefficients for a 2nd-order Butterworth-style response. */
export interface Biquad {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

/** Low-pass filter coefficients. */
export function lowpassCoefficients(cutoff: number, sampleRate: number, q = 0.7071): Biquad {
  const w0 = (2 * Math.PI * cutoff) / sampleRate;
  const alpha = Math.sin(w0) / (2 * q);
  const cos = Math.cos(w0);
  const a0 = 1 + alpha;
  return {
    b0: ((1 - cos) / 2) / a0,
    b1: (1 - cos) / a0,
    b2: ((1 - cos) / 2) / a0,
    a1: (-2 * cos) / a0,
    a2: (1 - alpha) / a0,
  };
}

/** High-pass filter coefficients. */
export function highpassCoefficients(cutoff: number, sampleRate: number, q = 0.7071): Biquad {
  const w0 = (2 * Math.PI * cutoff) / sampleRate;
  const alpha = Math.sin(w0) / (2 * q);
  const cos = Math.cos(w0);
  const a0 = 1 + alpha;
  return {
    b0: ((1 + cos) / 2) / a0,
    b1: (-(1 + cos)) / a0,
    b2: ((1 + cos) / 2) / a0,
    a1: (-2 * cos) / a0,
    a2: (1 - alpha) / a0,
  };
}

/** Apply a biquad filter in place, keeping the tail of the signal. */
export function applyBiquad(samples: Float32Array, coeffs: Biquad): Float32Array {
  const out = new Float32Array(samples.length);
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < samples.length; i++) {
    const x0 = samples[i];
    const y0 = coeffs.b0 * x0 + coeffs.b1 * x1 + coeffs.b2 * x2 - coeffs.a1 * y1 - coeffs.a2 * y2;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
    out[i] = y0;
  }
  return out;
}

/** Reverse a signal in place and return it. */
function reversed(samples: Float32Array): Float32Array {
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) out[i] = samples[samples.length - 1 - i];
  return out;
}

/**
 * Apply a biquad filter forwards and then backwards.
 *
 * A causal filter shifts every transient later, which moves note attacks
 * around by a few milliseconds. Filtering in both directions cancels the
 * phase shift, so the split keeps the attacks where they were.
 */
export function applyBiquadZeroPhase(samples: Float32Array, coeffs: Biquad): Float32Array {
  return reversed(applyBiquad(reversed(applyBiquad(samples, coeffs)), coeffs));
}

/**
 * Split a signal into two bands with a complementary crossover.
 *
 * The low-pass and high-pass are each applied forwards and backwards, so
 * neither shifts the attacks and their sum stays close to the original.
 */
export function splitBands(
  samples: Float32Array,
  cutoff: number,
  sampleRate: number = SAMPLE_RATE,
): { low: Float32Array; high: Float32Array } {
  const lowpass = lowpassCoefficients(cutoff, sampleRate);
  const highpass = highpassCoefficients(cutoff, sampleRate);
  const low = applyBiquadZeroPhase(applyBiquadZeroPhase(samples, lowpass), lowpass);
  const high = applyBiquadZeroPhase(applyBiquadZeroPhase(samples, highpass), highpass);
  return { low, high };
}

/** Downmix to mono by averaging channels. */
export function toMono(channels: Float32Array[]): Float32Array {
  if (channels.length === 0) return new Float32Array(0);
  if (channels.length === 1) return channels[0];
  const length = channels[0].length;
  const out = new Float32Array(length);
  for (const channel of channels) {
    for (let i = 0; i < length; i++) out[i] += channel[i];
  }
  for (let i = 0; i < length; i++) out[i] /= channels.length;
  return out;
}

/**
 * Resample to a target rate using cubic interpolation.
 *
 * Good enough for analysis, and it avoids depending on the browser's
 * `OfflineAudioContext` in the CLI.
 */
export function resample(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return samples;
  const ratio = to / from;
  const length = Math.round(samples.length * ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const pos = i / ratio;
    const idx = Math.floor(pos);
    const t = pos - idx;
    const p0 = samples[idx - 1] ?? samples[0];
    const p1 = samples[idx] ?? samples[0];
    const p2 = samples[idx + 1] ?? samples[0];
    const p3 = samples[idx + 2] ?? p2;
    // Catmull-Rom spline.
    out[i] =
      0.5 *
      (2 * p1 +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
  }
  return out;
}
