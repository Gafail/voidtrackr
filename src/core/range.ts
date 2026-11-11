/**
 * Choosing which notes to keep.
 *
 * A frequency range selects the notes that live inside it: the low register,
 * the top end, or a band in between. It is applied to the notes a model has
 * already found, never to the audio — and that is deliberate.
 *
 * Cutting the audio was measured, and it is much worse. A model reads a note
 * from its whole harmonic series, so a low-pass filter takes away the evidence
 * that places the fundamental. On this project's own material, transcribing a
 * recording low-passed at 400 Hz and keeping the notes below 400 Hz recovered
 * 14 of Okarina's low notes; transcribing the whole mix and keeping the same
 * notes recovered 142. Ten times as many, from the same model, in one pass.
 *
 * So the range is a selection over the finished notes. The audio is never
 * touched, the model always hears everything, and what is kept is exactly
 * what was found.
 */

import { hzToMidi, midiToHz, noteName } from './constants';
import { drumBand } from './drums';
import { RawNote, TrackKind } from './types';

/** A range of the spectrum, in Hz. */
export interface FrequencyRange {
  /** Lower bound in Hz. Zero means "no lower bound". */
  low: number;
  /** Upper bound in Hz. */
  high: number;
}

/** The lowest frequency a range can be given, in Hz. */
export const MIN_FREQUENCY = 0;

/** The highest frequency a range can be given, in Hz — the top of hearing. */
export const MAX_FREQUENCY = 20000;

/**
 * Put a range's ends in order.
 *
 * A range typed the wrong way round is read the right way round rather than
 * quietly matching nothing.
 */
export function normaliseRange(range: FrequencyRange): FrequencyRange {
  const low = Math.max(MIN_FREQUENCY, Math.min(range.low, range.high));
  const high = Math.min(MAX_FREQUENCY, Math.max(range.low, range.high));
  return { low, high };
}

/** The lowest MIDI note whose fundamental reaches a frequency. 40 Hz is E1. */
export function pitchAtOrAbove(hz: number): number {
  if (hz <= MIN_FREQUENCY) return 0;
  return Math.ceil(hzToMidi(hz));
}

/** The highest MIDI note whose fundamental stays under a frequency. 400 Hz is G4. */
export function pitchAtOrBelow(hz: number): number {
  if (hz >= MAX_FREQUENCY) return 127;
  return Math.floor(hzToMidi(hz));
}

/**
 * Does a note survive the range?
 *
 * A drum is judged by the band it actually occupies rather than by its note
 * number: the kick is written as C1 on the General MIDI map, but it sounds
 * at about 60 Hz, and the hi-hat is written as F#1 while it lives at 6 kHz.
 * A drum is kept when any part of its band falls inside the range.
 */
export function noteInRange(
  pitchMidi: number,
  kind: TrackKind,
  range: FrequencyRange,
): boolean {
  const { low, high } = normaliseRange(range);

  if (kind === 'drums') {
    const band = drumBand(pitchMidi);
    if (!band) return false;
    return band.hi >= low && band.lo <= high;
  }

  const frequency = midiToHz(pitchMidi);
  return frequency >= low && frequency <= high;
}

/** Keep only the notes inside the range. */
export function filterByRange(
  notes: RawNote[],
  kind: TrackKind,
  range: FrequencyRange,
): RawNote[] {
  return notes.filter((note) => noteInRange(note.pitchMidi, kind, range));
}

/**
 * Describe a range for the user, in both of the languages they think in.
 *
 * "40 Hz – 1200 Hz · E1–D6" says the same thing to a game programmer and to
 * a musician. The notes are clamped to the piano keyboard, because that is
 * the only range the models ever report.
 */
export function describeRange(range: FrequencyRange): string {
  const { low, high } = normaliseRange(range);
  const from = low <= MIN_FREQUENCY ? '0 Hz' : `${Math.round(low)} Hz`;
  const to = high >= MAX_FREQUENCY ? '20 kHz' : `${Math.round(high)} Hz`;
  const lowest = Math.max(21, pitchAtOrAbove(low));
  const highest = Math.min(108, pitchAtOrBelow(high));
  const notes = highest < lowest ? 'no notes' : `${noteName(lowest)}–${noteName(highest)}`;
  return `${from} – ${to} · ${notes}`;
}
