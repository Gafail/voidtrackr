/**
 * Drum detection.
 *
 * There is no learned drum model here: kicks, snares and hi-hats are found
 * with per-band spectral flux. Each drum lives in a fairly distinct part of
 * the spectrum, and a strike shows up as a sharp rise in that band.
 *
 * The hard part is deciding whether a clip has drums at all. The flux
 * detector compares a band against itself, so it happily finds "hits" in a
 * solo piano piece. So we score each ten-second window on several
 * independent signals and only trust the result when they agree.
 */

import { RawNote } from './types';
import {
  bandFlux,
  bandShare,
  binRange,
  flatness,
  hatDecay,
  median,
  normalize,
  periodicity,
  pickPeaks,
  spectrogram,
  Spectrogram,
} from './dsp';

export type DrumKind = 'kick' | 'snare' | 'hat';

export interface DrumHit {
  kind: DrumKind;
  /** MIDI note number (36 = kick, 38 = snare, 42 = closed hi-hat). */
  pitch: number;
  time: number;
  /** How strong the hit was, 0..1. */
  strength: number;
}

export interface DrumAnalysis {
  detected: boolean;
  /** Overall score; above zero means drums were found. */
  score: number;
  hits: DrumHit[];
  /** Per-window breakdown, useful for tuning the detector. */
  windows: {
    start: number;
    end: number;
    score: number;
  }[];
}

interface DrumBand {
  kind: DrumKind;
  pitch: number;
  lo: number;
  hi: number;
  /** Share of the band's loud level treated as silence when picking hits. */
  floor: number;
  /** Absolute onset strength required. */
  delta: number;
  /** How far above the local average a hit must be. */
  mult: number;
}

const BANDS: DrumBand[] = [
  { kind: 'kick', pitch: 36, lo: 40, hi: 160, floor: 0.5, delta: 0.3, mult: 2 },
  { kind: 'snare', pitch: 38, lo: 180, hi: 1500, floor: 0.3, delta: 0.3, mult: 2.5 },
  { kind: 'hat', pitch: 42, lo: 6000, hi: 10000, floor: 0.1, delta: 0.1, mult: 2 },
];

const SPEC_SIZE = 1024;
const SPEC_HOP = 220;
const WINDOW_SECONDS = 10;
/** A snare within this distance of a kick is treated as spill from the kick. */
const KICK_SPILL_SECONDS = 0.03;

/** Analyse a mono 22.05 kHz signal for drum hits. */
export function analyzeDrums(samples: Float32Array, sampleRate: number): DrumAnalysis {
  if (samples.length < sampleRate) {
    return { detected: false, score: -Infinity, hits: [], windows: [] };
  }

  const spec = spectrogram(samples, { size: SPEC_SIZE, hop: SPEC_HOP, sampleRate });
  if (spec.frames < spec.fps) {
    return { detected: false, score: -Infinity, hits: [], windows: [] };
  }

  // Unfloored curves are used to judge whether drums are present; floored
  // curves are used to pick the actual hits.
  const rawFlux: Record<string, Float32Array> = {};
  for (const band of BANDS) rawFlux[band.kind] = bandFlux(spec, band.lo, band.hi);

  const combined = new Float32Array(spec.frames);
  for (const band of BANDS) {
    const curve = normalize(rawFlux[band.kind]);
    for (let i = 0; i < combined.length; i++) combined[i] += curve[i];
  }

  const hatNorm = normalize(rawFlux.hat);
  const windows: { start: number; end: number; score: number }[] = [];

  const windowFrames = Math.round(WINDOW_SECONDS * spec.fps);
  for (let start = 0; start < spec.frames; start += windowFrames) {
    const end = Math.min(spec.frames, start + windowFrames);
    // A short tail joins the previous window rather than being judged alone.
    if (windows.length && end - start < windowFrames / 2) break;
    const score = scoreWindow(spec, combined, hatNorm, rawFlux.hat, start, end);
    windows.push({ start: start / spec.fps, end: end / spec.fps, score });
  }

  const score = median(windows.map((w) => w.score));

  const hits: DrumHit[] = [];
  for (const band of BANDS) {
    const curve = normalize(bandFlux(spec, band.lo, band.hi, 4, band.floor));
    for (const frame of pickPeaks(curve, spec.fps, band.delta, band.mult)) {
      hits.push({
        kind: band.kind,
        pitch: band.pitch,
        time: frame / spec.fps,
        strength: Math.min(1, curve[frame]),
      });
    }
  }

  // A kick also lights up the snare band, which was the main source of
  // phantom snares. Drop snares that sit right on top of a kick.
  const kicks = hits.filter((h) => h.kind === 'kick').map((h) => h.time);
  const filtered = hits.filter(
    (h) =>
      h.kind !== 'snare' ||
      h.strength >= 0.6 ||
      !kicks.some((t) => Math.abs(t - h.time) <= KICK_SPILL_SECONDS),
  );

  filtered.sort((a, b) => a.time - b.time || a.pitch - b.pitch);

  return { detected: score > 0, score, hits: filtered, windows };
}

/**
 * Score one window of audio for the presence of a drum kit.
 *
 * The terms are margins past a midpoint between drummed and drumless music,
 * each scaled by how much that feature normally varies. The final -0.5 leans
 * towards "no drums": a phantom drum track under a piano piece is a worse
 * mistake than a missing drum track.
 */
function scoreWindow(
  spec: Spectrogram,
  combined: Float32Array,
  hatNorm: Float32Array,
  hatFlux: Float32Array,
  start: number,
  end: number,
): number {
  const hatHits = pickPeaks(hatNorm.subarray(start, end), spec.fps, 0.1, 1.5);
  if (hatHits.length === 0) {
    // No hi-hat candidates at all: score on pulse alone.
    const pulse = periodicity(combined.subarray(start, end), 0.25 * spec.fps, 1.5 * spec.fps);
    return (pulse - 0.38) / 0.15 - 0.5;
  }

  const noiseRange = binRange(spec, 4000, 10000);
  const fullRange = binRange(spec, 40, 10000);

  const hatNoise = median(
    hatHits.map((t) => flatness(spec, start + t, noiseRange)),
  );
  const hatStrength = median(
    hatHits.map((t) => hatFlux[start + t]),
  );
  const hatShare = median(
    hatHits.map((t) => bandShare(spec, start + t, noiseRange, fullRange)),
  );
  const decay = median(
    hatHits
      .map((t) => hatDecay(spec, start + t, noiseRange))
      .filter((d): d is number => d !== null),
  );
  const pulse = periodicity(combined.subarray(start, end), 0.25 * spec.fps, 1.5 * spec.fps);

  // Hats that are far below the rest of the sound are attack clicks on a
  // dark pitched note, not a drum kit.
  if (hatShare < -40) return -5;
  // No steady pulse, and "hats" that do not decay: this is a voice's
  // sibilants, which are loud, noisy and bright enough to look like hats.
  if (pulse < 0.25 && decay > 0) return -5;

  return (
    (hatNoise - 0.27) / 0.1 +
    (pulse - 0.38) / 0.15 +
    (hatStrength - 0.5) / 0.4 -
    0.5
  );
}

/** Convert drum hits into short MIDI notes on the General MIDI drum map. */
export function hitsToNotes(hits: DrumHit[]): RawNote[] {
  return hits.map((hit) => ({
    pitchMidi: hit.pitch,
    startTime: hit.time,
    duration: 0.1,
    amplitude: (40 + 80 * hit.strength) / 127,
  }));
}
