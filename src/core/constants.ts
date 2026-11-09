/**
 * Constants that pin us to the Basic Pitch model contract.
 *
 * The model is a fixed-window graph model: it consumes exactly
 * `AUDIO_N_SAMPLES` samples of mono 22.05 kHz audio and emits 172 annotation
 * frames (86 fps). Anything that changes these numbers will silently shift
 * every note in the output, so they live in one place.
 */

/** Sample rate the model was trained at. */
export const SAMPLE_RATE = 22050;

/** Hop size of the model's internal CQT, in samples. */
export const FFT_HOP = 256;

/** Annotation frames per second emitted by the model. */
export const ANNOTATIONS_FPS = Math.floor(SAMPLE_RATE / FFT_HOP);

/** Analysis window of the model, in seconds. */
export const AUDIO_WINDOW_LENGTH_SECONDS = 2;

/** Number of samples in one model window. */
export const AUDIO_N_SAMPLES = SAMPLE_RATE * AUDIO_WINDOW_LENGTH_SECONDS - FFT_HOP;

/** Annotation frames produced for one model window. */
export const ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH_SECONDS;

/**
 * Number of overlapping frames between consecutive windows. The model needs
 * this context on both sides of the window; the edges are trimmed afterwards.
 */
export const N_OVERLAPPING_FRAMES = 30;

/** Half of the overlap, trimmed from each edge of a window. */
export const N_OVERLAP_OVER_2 = Math.floor(N_OVERLAPPING_FRAMES / 2);

/** Overlap expressed in samples. */
export const OVERLAP_LENGTH_FRAMES = N_OVERLAPPING_FRAMES * FFT_HOP;

/** Distance between the start of consecutive windows, in samples. */
export const HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LENGTH_FRAMES;

/** Lowest MIDI number the model's pitch bins represent (A0 = 21). */
export const MIDI_OFFSET = 21;

/** Highest usable pitch bin index. */
export const MAX_FREQ_IDX = 87;

/** Contour resolution: bins per semitone. */
export const CONTOURS_BINS_PER_SEMITONE = 3;

/** Lowest frequency the contour head covers, in Hz. */
export const ANNOTATIONS_BASE_FREQUENCY = 27.5;

/** Number of semitones the contour head covers. */
export const ANNOTATIONS_N_SEMITONES = 88;

/** Total contour bins. */
export const N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE;

/**
 * The model's window is centred on its input, so annotation frame `f` is
 * reported slightly before the sample it corresponds to. This offset
 * converts frame indices to wall-clock seconds.
 */
export const WINDOW_OFFSET =
  (FFT_HOP / SAMPLE_RATE) * (ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP) + 0.0018;

/** Seconds of audio covered by one model frame, before the window correction. */
export const MODEL_FRAME_SECONDS = FFT_HOP / SAMPLE_RATE;

/**
 * Number of frames one analysis window contributes.
 *
 * A window produces {@link ANNOT_N_FRAMES} frames, of which
 * {@link N_OVERLAP_OVER_2} are trimmed from each side because they overlap
 * the neighbouring window.
 */
export const FRAMES_PER_WINDOW = ANNOT_N_FRAMES - 2 * N_OVERLAP_OVER_2;

/**
 * Convert a model frame index to a time in seconds.
 *
 * Consecutive windows overlap, so the frames of a window do not sit one
 * frame apart in time: the window advances the source by {@link HOP_SIZE}
 * samples while contributing {@link FRAMES_PER_WINDOW} frames. The
 * correction term below takes that up, and it is what keeps a long file from
 * drifting out of sync. Leave it out and the notes run fast by about
 * 5 ms for every second of audio.
 *
 * This is the same mapping the reference implementation uses, which is why
 * our output lines up with it note for note.
 */
export function modelFrameToTime(frame: number): number {
  return frame * MODEL_FRAME_SECONDS - WINDOW_OFFSET * Math.floor(frame / ANNOT_N_FRAMES);
}

/**
 * Convert a time in seconds back to the nearest model frame index.
 *
 * This is the inverse of {@link modelFrameToTime}. Because the correction
 * term is piecewise constant, the frame index cannot simply be divided out:
 * the window the frame falls in has to be found first, and that depends on
 * the answer. The window the time suggests is tried, along with its
 * neighbours, and whichever frame lands closest to the time wins.
 */
export function timeToModelFrame(time: number): number {
  const windowSeconds = ANNOT_N_FRAMES * MODEL_FRAME_SECONDS;
  const guess = Math.floor(time / windowSeconds);

  let best = 0;
  let bestError = Infinity;

  for (let window = Math.max(0, guess - 2); window <= guess + 2; window++) {
    const frame = Math.round((time + WINDOW_OFFSET * window) / MODEL_FRAME_SECONDS);
    const error = Math.abs(modelFrameToTime(frame) - time);
    if (error < bestError) {
      bestError = error;
      best = frame;
    }
  }

  return best;
}

/** MIDI note number -> frequency in Hz. */
export function midiToHz(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/** Frequency in Hz -> (fractional) MIDI note number. */
export function hzToMidi(hz: number): number {
  return 12 * (Math.log2(hz) - Math.log2(440)) + 69;
}

/** Note names, used for the key detection output and the editor ruler. */
export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/** Human readable note name for a MIDI number, e.g. 60 -> "C4". */
export function noteName(midi: number): string {
  const rounded = Math.round(midi);
  return `${NOTE_NAMES[((rounded % 12) + 12) % 12]}${Math.floor(rounded / 12) - 1}`;
}
