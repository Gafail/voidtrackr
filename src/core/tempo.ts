/**
 * Tempo and key estimation.
 *
 * Both work from the transcribed notes rather than from the audio, which
 * keeps them cheap and makes them easy to reason about when the note
 * extraction changes.
 */

import { RawNote } from './types';
import { NOTE_NAMES } from './constants';

/** BPM range we are willing to report. */
const MIN_BPM = 50;
const MAX_BPM = 200;

/** Result of tempo estimation. */
export interface TempoEstimate {
  bpm: number;
  confidence: number;
  adjusted: boolean;
}

/**
 * Estimate the tempo from note onsets.
 *
 * Onsets are turned into a click train and correlated with itself. The
 * strongest lag is not always the beat: a passage with eighth notes has a
 * peak at half the beat as well, and at twice the beat. So every candidate
 * tempo is scored with a comb filter that also rewards its multiples, and a
 * broad prior pulls the answer towards the range songs usually sit in.
 */
export function estimateTempo(notes: RawNote[], duration: number): TempoEstimate {
  if (notes.length < 2 || duration <= 0) {
    return { bpm: 120, confidence: 0, adjusted: false };
  }

  const resolution = 0.005;
  const frames = Math.ceil(duration / resolution) + 1;
  const envelope = new Float32Array(frames);

  for (const note of notes) {
    const index = Math.min(frames - 1, Math.max(0, Math.round(note.startTime / resolution)));
    // Longer, louder notes contribute more to the click train.
    const weight = 0.4 + 0.6 * Math.min(1, note.duration * 4) + 0.3 * note.amplitude;
    envelope[index] = Math.min(1, envelope[index] + weight);
  }

  // Remove the mean so the correlation measures "is there a pulse", not
  // "is it loud".
  let mean = 0;
  for (let i = 0; i < frames; i++) mean += envelope[i] / frames;
  const centred = new Float32Array(frames);
  for (let i = 0; i < frames; i++) centred[i] = envelope[i] - mean;

  let energy = 0;
  for (let i = 0; i < frames; i++) energy += centred[i] * centred[i];
  if (!(energy > 0)) return { bpm: 120, confidence: 0, adjusted: false };

  const maxLag = Math.min(frames - 1, Math.ceil(60 / MIN_BPM / resolution));
  const correlation = new Float32Array(maxLag + 1);
  for (let lag = 1; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = lag; i < frames; i++) sum += centred[i] * centred[i - lag];
    correlation[lag] = sum / energy;
  }

  /** How well a tempo explains the click train, counting its multiples. */
  const combScore = (bpm: number, multiples: number): number => {
    if (!(bpm > 0)) return 0;
    const lag = 60 / bpm / resolution;
    let sum = 0;
    let count = 0;
    for (let k = 1; k <= multiples; k++) {
      const index = Math.round(k * lag);
      if (index < 1 || index > maxLag) break;
      sum += correlation[index];
      count += 1;
    }
    return count > 0 ? sum / count : 0;
  };

  /** Songs sit near 120 BPM more often than anywhere else. */
  const prior = (bpm: number) => Math.exp(-0.5 * Math.pow(Math.log2(bpm / 120), 2));

  const score = (bpm: number) =>
    (combScore(bpm, 4) + 0.5 * combScore(bpm * 2, 4)) * prior(bpm);

  let bestBpm = 120;
  let bestScore = -Infinity;
  for (let bpm = MIN_BPM; bpm <= MAX_BPM; bpm += 0.25) {
    const value = score(bpm);
    if (value > bestScore) {
      bestScore = value;
      bestBpm = bpm;
    }
  }

  // Refine around the winner.
  for (let bpm = bestBpm - 0.25; bpm <= bestBpm + 0.25; bpm += 0.01) {
    if (bpm < MIN_BPM || bpm > MAX_BPM) continue;
    const value = score(bpm);
    if (value > bestScore) {
      bestScore = value;
      bestBpm = bpm;
    }
  }

  let bpm = bestBpm;
  let adjusted = false;

  // Fold into a musical range: 60-200 BPM covers most pop music, but a
  // simple pulse at 80 BPM and the same pulse at 160 BPM are the same rhythm.
  while (bpm < 75) {
    bpm *= 2;
    adjusted = true;
  }
  while (bpm > 175) {
    bpm /= 2;
    adjusted = true;
  }

  // How much better is the winner than the average candidate? A clear
  // winner means we trust the number.
  let average = 0;
  let samples = 0;
  for (let candidate = MIN_BPM; candidate <= MAX_BPM; candidate += 1) {
    average += score(candidate);
    samples += 1;
  }
  average /= Math.max(1, samples);
  const confidence = Math.max(0, Math.min(1, (bestScore - average) * 4));

  return { bpm: Math.round(bpm * 10) / 10, confidence, adjusted };
}

/**
 * Estimate the key from the pitch content.
 *
 * Uses the Krumhansl-Schmuckler key profiles: the chroma of the piece is
 * correlated against all 24 rotations of the major and minor profiles.
 */
export function estimateKey(notes: RawNote[]): {
  tonic: number;
  mode: 'major' | 'minor';
  confidence: number;
} {
  const chroma = new Float64Array(12);
  for (const note of notes) {
    const pitchClass = ((Math.round(note.pitchMidi) % 12) + 12) % 12;
    // Longer notes carry more weight; very short ones are often ornaments.
    const weight = Math.max(0.05, Math.min(1, note.duration * 2)) * (0.4 + 0.6 * note.amplitude);
    chroma[pitchClass] += weight;
  }

  const total = chroma.reduce((a, b) => a + b, 0);
  if (total <= 0) return { tonic: 0, mode: 'major', confidence: 0 };
  for (let i = 0; i < 12; i++) chroma[i] /= total;

  const MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const MINOR = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

  let best: { tonic: number; mode: 'major' | 'minor'; score: number } = {
    tonic: 0,
    mode: 'major',
    score: -Infinity,
  };
  const scores: number[] = [];

  for (const [profile, mode] of [
    [MAJOR, 'major'],
    [MINOR, 'minor'],
  ] as const) {
    for (let tonic = 0; tonic < 12; tonic++) {
      const score = correlate(chroma, profile, tonic);
      scores.push(score);
      if (score > best.score) best = { tonic, mode, score };
    }
  }

  const average = scores.reduce((a, b) => a + b, 0) / Math.max(1, scores.length);
  const confidence = Math.max(0, Math.min(1, (best.score - average) * 3));

  return { tonic: best.tonic, mode: best.mode, confidence };
}

/** Pearson correlation between a chroma vector and a rotated key profile. */
function correlate(chroma: Float64Array, profile: number[], tonic: number): number {
  let sumA = 0;
  let sumB = 0;
  for (let i = 0; i < 12; i++) {
    sumA += chroma[i];
    sumB += profile[(i + tonic) % 12];
  }
  const meanA = sumA / 12;
  const meanB = sumB / 12;

  let cov = 0;
  let varA = 0;
  let varB = 0;
  for (let i = 0; i < 12; i++) {
    const a = chroma[i] - meanA;
    const b = profile[(i + tonic) % 12] - meanB;
    cov += a * b;
    varA += a * a;
    varB += b * b;
  }
  return varA > 0 && varB > 0 ? cov / Math.sqrt(varA * varB) : 0;
}

/** Human readable key name, e.g. "A minor". */
export function keyName(tonic: number, mode: 'major' | 'minor'): string {
  return `${NOTE_NAMES[tonic]} ${mode}`;
}
