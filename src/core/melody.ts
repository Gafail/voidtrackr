/**
 * Melody tracing.
 *
 * Counting note attacks is a poor way to find the melody: a held vocal note
 * that gets chopped into pieces by vibrato would look like several notes.
 * Instead we trace a continuous line through the activation matrix with a
 * Viterbi decoder, where the states are "this pitch is sounding" or "this
 * pitch is resting".
 *
 * The decoder is rewarded for staying on one pitch and for moving by small
 * steps, and it is charged extra for jumping far, so the path it finds is a
 * singable line rather than a collection of the loudest moments.
 *
 * One detail drives the whole design: the model's frame head never returns
 * zero. Even in silence every pitch bin sits at a small baseline value, so
 * "no pitch here" and "this very quiet pitch here" look almost the same.
 * Comparing raw values therefore never lets the decoder say "rest". The
 * baseline is measured and subtracted first, which makes the absence of a
 * pitch genuinely unlikely and lets the decoder breathe.
 */

import { MIDI_OFFSET, modelFrameToTime, timeToModelFrame } from './constants';
import { ModelOutput, PITCH_BINS } from './model';
import { RawNote } from './types';

/** The rest state, placed after the 88 pitch states. */
const REST = PITCH_BINS;

/** Frames between decoder steps. 2 frames is about 23 ms. */
const STRIDE = 2;

/** Minimum length of a traced note, in seconds. */
const MIN_NOTE_SECONDS = 0.08;

/** Probability floor used when a pitch is completely absent. */
const ABSENT_PENALTY = 25;

/** Pseudo-count for "nothing is playing". */
const REST_MASS = 0.15;

/** How quickly the rest mass disappears as music comes in. */
const REST_SCALE = 0.25;

export interface MelodyTraceOptions {
  /** How strongly the decoder is pulled towards the middle of the range. */
  centrePull?: number;
  /** Extra cost for sitting below MIDI 48 when the texture is mixed. */
  lowRegisterPenalty?: number;
  /** Onset strength needed to split a held note. */
  reattackThreshold?: number;
  /** Baseline of the frame head, estimated from the audio when omitted. */
  baseline?: number;
}

export interface MelodyTraceResult {
  notes: RawNote[];
  /** The register the tracer believes the melody lives in. */
  centre: number;
  /** True when the source texture looked mixed rather than monophonic. */
  mixedTexture: boolean;
  /** The measured baseline of the frame head. */
  baseline: number;
}

/**
 * Trace the most likely melodic line through the model output.
 */
export function traceMelody(
  output: ModelOutput,
  options: MelodyTraceOptions = {},
): MelodyTraceResult {
  const centrePull = options.centrePull ?? 0.035;
  const lowRegisterPenalty = options.lowRegisterPenalty ?? 0.22;
  const reattackThreshold = options.reattackThreshold ?? 0.65;

  const { frames, onsets, nFrames } = output;
  if (nFrames === 0) {
    return { notes: [], centre: 60, mixedTexture: false, baseline: 0 };
  }

  const baseline = options.baseline ?? estimateBaseline(frames);

  // --- Where does the music live? -------------------------------------
  const weights = new Float64Array(PITCH_BINS);
  let total = 0;
  let middle = 0;
  for (let f = 0; f < nFrames; f++) {
    for (let k = 0; k < PITCH_BINS; k++) {
      const value = Math.max(0, frames[f * PITCH_BINS + k] - baseline);
      weights[k] += value;
      total += value;
      const midi = k + MIDI_OFFSET;
      if (midi >= 48 && midi <= 84) middle += value;
    }
  }

  const mixedTexture = total > 0 && middle > 0.3 * total;
  const centre = findCentre(weights, mixedTexture ? middle / total : 1);

  // --- Set up the decoder ---------------------------------------------
  const stepFrames: number[] = [];
  for (let f = 0; f < nFrames; f += STRIDE) stepFrames.push(f);
  const steps = stepFrames.length;
  const states = PITCH_BINS + 1;

  const costs = new Float32Array(steps * states);
  const back = new Uint8Array(steps * states);

  const pitchCost = new Float32Array(states);
  const presence = new Float64Array(PITCH_BINS);

  for (let t = 0; t < steps; t++) {
    const f = stepFrames[t];
    const base = f * PITCH_BINS;

    // How much of the frame is music, and how much is silence?
    let sum = 0;
    let maxActivation = 0;
    for (let k = 0; k < PITCH_BINS; k++) {
      const value = Math.max(0, frames[base + k] - baseline);
      presence[k] = value;
      sum += value;
      if (value > maxActivation) maxActivation = value;
    }

    const restMass = REST_MASS * Math.exp(-sum / REST_SCALE);
    const normaliser = sum + restMass;

    for (let k = 0; k < PITCH_BINS; k++) {
      const midi = k + MIDI_OFFSET;
      let penalty = centrePull * Math.abs(midi - centre);
      if (mixedTexture && midi < 48) penalty += lowRegisterPenalty;
      pitchCost[k] =
        (presence[k] > 0 ? -Math.log(presence[k] / normaliser) : ABSENT_PENALTY) + penalty;
    }
    const restCost = -Math.log(restMass / normaliser);

    if (t === 0) {
      for (let s = 0; s < states; s++) {
        costs[s] = s === REST ? restCost : pitchCost[s];
      }
      continue;
    }

    const previousBase = (t - 1) * states;
    const currentBase = t * states;

    for (let s = 0; s < states; s++) {
      let best = Infinity;
      let bestPrev = 0;

      for (let p = 0; p < states; p++) {
        const transition = transitionCost(p, s, onsets, base);
        const total = costs[previousBase + p] + transition;
        if (total < best) {
          best = total;
          bestPrev = p;
        }
      }

      costs[currentBase + s] = best + (s === REST ? restCost : pitchCost[s]);
      back[currentBase + s] = bestPrev;
    }

    // Normalise so the numbers do not grow without bound.
    let rowMax = -Infinity;
    for (let s = 0; s < states; s++) {
      if (costs[currentBase + s] > rowMax) rowMax = costs[currentBase + s];
    }
    for (let s = 0; s < states; s++) costs[currentBase + s] -= rowMax;
  }

  // --- Trace back ------------------------------------------------------
  const lastBase = (steps - 1) * states;
  let state = 0;
  let bestLast = Infinity;
  for (let s = 0; s < states; s++) {
    if (costs[lastBase + s] < bestLast) {
      bestLast = costs[lastBase + s];
      state = s;
    }
  }

  const path = new Uint8Array(steps);
  for (let t = steps - 1; t >= 0; t--) {
    path[t] = state;
    state = back[t * states + state];
  }

  // --- Turn the path into notes ---------------------------------------
  const notes: RawNote[] = [];

  let runStart = 0;
  let runState = path[0];

  const flushRun = (endStep: number) => {
    if (runState === REST) return;
    const startFrame = stepFrames[runStart];
    const endFrame = stepFrames[Math.min(endStep, steps - 1)] + STRIDE;
    const startTime = modelFrameToTime(startFrame);
    const duration = modelFrameToTime(endFrame) - startTime;
    if (duration < MIN_NOTE_SECONDS) return;

    let sum = 0;
    let count = 0;
    for (let f = startFrame; f < Math.min(endFrame, nFrames); f++) {
      sum += frames[f * PITCH_BINS + runState];
      count += 1;
    }
    notes.push({
      pitchMidi: runState + MIDI_OFFSET,
      startTime,
      duration,
      amplitude: count > 0 ? sum / count : 0.5,
    });
  };

  for (let t = 1; t < steps; t++) {
    if (path[t] !== runState) {
      flushRun(t);
      runStart = t;
      runState = path[t];
    }
  }
  flushRun(steps);

  // Re-articulation: a held note that is struck again should become two
  // notes. Loudness wobble on its own is not enough.
  const rearticulated: RawNote[] = [];
  for (const note of notes) {
    const pitchIndex = note.pitchMidi - MIDI_OFFSET;
    const startFrame = timeToModelFrame(note.startTime);
    const endFrame = startFrame + Math.round(note.duration * (22050 / 256));

    let split = -1;
    for (let f = startFrame + 3; f < endFrame - 1; f++) {
      if ((f - startFrame) / (22050 / 256) < 0.09) continue;
      const value = onsets[f * PITCH_BINS + pitchIndex];
      if (value < reattackThreshold) continue;
      let isPeak = true;
      for (let j = Math.max(0, f - 2); j <= Math.min(nFrames - 1, f + 2); j++) {
        if (j !== f && value <= onsets[j * PITCH_BINS + pitchIndex]) {
          isPeak = false;
          break;
        }
      }
      if (isPeak) {
        split = f;
        break;
      }
    }

    if (split < 0) {
      rearticulated.push(note);
      continue;
    }

    const splitTime = modelFrameToTime(split);
    const firstDuration = splitTime - note.startTime;
    if (firstDuration >= MIN_NOTE_SECONDS) {
      rearticulated.push({ ...note, duration: firstDuration });
    }
    rearticulated.push({
      ...note,
      startTime: splitTime,
      duration: Math.max(MIN_NOTE_SECONDS, note.startTime + note.duration - splitTime),
    });
  }

  rearticulated.sort((a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi);
  return { notes: rearticulated, centre, mixedTexture, baseline };
}

/**
 * Cost of moving from one state to the next.
 *
 * Staying on a pitch is nearly free, moving by a step is cheap, and jumping
 * an octave or more is expensive. An onset at the destination makes the move
 * cheaper, because something clearly started there.
 */
function transitionCost(from: number, to: number, onsets: Float32Array, frameBase: number): number {
  if (from === REST && to === REST) return 0;
  if (from === REST || to === REST) return 1.5;

  const step = Math.abs(to - from);
  let cost = 0.3 + 1.0 * Math.min(1, step / 12);
  const onset = onsets[frameBase + to] ?? 0;
  cost *= 1 - Math.min(0.6, onset * 0.65);
  return cost;
}

/**
 * The baseline the frame head reports when nothing is playing.
 *
 * Estimated from a low percentile of every value, so it is not dragged up by
 * the music itself.
 */
export function estimateBaseline(frames: Float32Array): number {
  if (frames.length === 0) return 0;

  // Sample the values rather than sorting all of them.
  const sampleSize = Math.min(20000, frames.length);
  const stride = Math.max(1, Math.floor(frames.length / sampleSize));
  const sample: number[] = [];
  for (let i = 0; i < frames.length; i += stride) sample.push(frames[i]);
  sample.sort((a, b) => a - b);

  const percentile = sample[Math.floor(sample.length * 0.05)] ?? 0;
  // The baseline cannot exceed this, or genuinely quiet music would be
  // erased along with the noise.
  return Math.min(percentile, 0.35);
}

/** The pitch where the cumulative weight crosses `target` of the total. */
function findCentre(weights: Float64Array, share: number): number {
  let total = 0;
  for (const w of weights) total += w;
  if (total <= 0) return 60;

  const target = 0.7 * share * total;
  let cumulative = 0;
  for (let k = 0; k < weights.length; k++) {
    cumulative += weights[k];
    if (cumulative >= target) return k + MIDI_OFFSET;
  }
  return 60;
}

/**
 * Traced notes that the attack-based decoder missed.
 *
 * The decoder needs a clear onset to start a note; the tracer does not. Any
 * traced note that is long enough and loud enough, and does not overlap an
 * existing note at the same pitch, is worth keeping.
 */
export function unmatchedMelody(melody: RawNote[], decoded: RawNote[]): RawNote[] {
  const result: RawNote[] = [];
  for (const note of melody) {
    if (note.duration < 0.2 || note.amplitude < 0.28) continue;
    const overlaps = decoded.some(
      (other) =>
        other.pitchMidi === note.pitchMidi &&
        note.startTime < other.startTime + other.duration &&
        other.startTime < note.startTime + note.duration,
    );
    if (!overlaps) result.push(note);
  }
  return result;
}
