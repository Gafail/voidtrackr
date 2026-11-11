/**
 * The conversion pipeline.
 *
 * Everything the tool does lives here as a sequence of explicit stages, so
 * each stage can be inspected, replaced, or measured on its own.
 *
 *   decode -> tune -> model -> drums -> tempo/key -> MIDI
 *
 * The model stage is the only one that differs between models: each model
 * turns the signal into notes grouped into parts, and everything after that
 * — the drum detector, the tempo estimate, the key estimate and the MIDI
 * writer — is the same whichever model was used.
 */

import { ModelPart, TranscriptionModel } from './model';
import { SAMPLE_RATE } from './constants';
import { analyzeDrums, hitsToNotes } from './drums';
import { resample } from './dsp';
import { sortNotes } from './notes';
import { describeRange, filterByRange, FrequencyRange } from './range';
import { correctTuning } from './tuning';
import { estimateKey, estimateTempo } from './tempo';
import { describeQuantise, quantiseNotes, QuantiseOptions } from './quantise';
import { pianify, PianifyOptions } from './pianify';
import { KeyInfo, ProgressInfo, TempoInfo, Track, TranscriptionResult } from './types';

export interface TranscribeOptions {
  /** The model to transcribe with. */
  model: TranscriptionModel;
  /** Progress callback. */
  onProgress?: (info: ProgressInfo) => void;
  /** Look for drums. On by default. */
  detectDrums?: boolean;
  /** Force the tempo instead of estimating it. */
  forceTempo?: number;
  /** Force the key instead of estimating it. */
  forceKey?: { tonic: number; mode: 'major' | 'minor' };
  /**
   * Keep only the notes that live inside this frequency range.
   *
   * The range is applied to the notes the model found, not to the audio, so
   * the model still hears the whole recording. The tempo and the key are
   * estimated from the whole recording as well, because they belong to the
   * song rather than to the part of it that was kept.
   */
  range?: FrequencyRange;
  /**
   * Snap the notes to a fixed grid, for something that runs at a fixed rate.
   *
   * A game rendering 30 frames a second can only show a note on one of its
   * own frames, so the file is better off agreeing with the frame grid than
   * being more precise than the screen. The tempo and the key are estimated
   * before this, from the notes as the model found them.
   */
  quantise?: QuantiseOptions;
  /**
   * Arrange the notes for two hands, the way a player would.
   *
   * A transcription has no hands in it: it can report twenty notes sounding
   * at once and the same key struck twice in a row. This stage folds the
   * re-triggers together, divides the chords between a left and a right
   * hand, and drops what neither hand could hold. See `pianify.ts`.
   *
   * It runs on the notes the model found, before the range, because the
   * arrangement is a decision about the whole piece, and before the grid, so
   * the grid places the arranged notes.
   */
  pianify?: PianifyOptions;
}

/**
 * Convert a mono signal into a set of MIDI tracks.
 */
export async function transcribe(
  samples: Float32Array,
  options: TranscribeOptions,
): Promise<TranscriptionResult> {
  const report = (value: number, label: string) => options.onProgress?.({ value, label });
  const log: string[] = [];
  const duration = samples.length / SAMPLE_RATE;

  report(0.02, 'Checking tuning');

  // --- Tuning ---------------------------------------------------------
  const tuning = correctTuning(samples);
  const working = tuning.samples;
  if (tuning.applied) {
    log.push(
      `Recording was ${Math.abs(tuning.cents).toFixed(0)} cents ${
        tuning.cents > 0 ? 'sharp' : 'flat'
      }; corrected to A440.`,
    );
  }
  const toSource = tuning.toSource;

  // --- Model ----------------------------------------------------------
  report(0.05, 'Transcribing');
  // Each model works at its own rate. Basic Pitch wants 22.05 kHz, which is
  // what the pipeline already uses; Okarina wants 16 kHz.
  const modelInput =
    options.model.inputSampleRate === SAMPLE_RATE
      ? working
      : resample(working, SAMPLE_RATE, options.model.inputSampleRate);
  const modelResult = await options.model.transcribe(modelInput, {
    onProgress: (value) => report(0.05 + value * 0.7, 'Transcribing'),
  });
  log.push(...modelResult.log);

  const tracks: Track[] = modelResult.parts.map((part: ModelPart) => ({
    id: part.id,
    name: part.name,
    kind: part.kind,
    notes: part.notes,
  }));

  const modelNoteCount = tracks.reduce((sum, track) => sum + track.notes.length, 0);

  // --- Drums ----------------------------------------------------------
  let drumsDetected = false;
  let drumHits = 0;
  // Some models find the drums themselves. Adding a second drum track on top
  // of theirs would double every hit, so the detector only runs when the
  // model did not.
  const modelHasDrums = modelResult.parts.some((part) => part.kind === 'drums');
  if (options.detectDrums !== false && !modelHasDrums) {
    report(0.8, 'Looking for drums');
    const drums = analyzeDrums(samples, SAMPLE_RATE);
    drumsDetected = drums.detected;
    if (drumsDetected && drums.hits.length > 0) {
      drumHits = drums.hits.length;
      tracks.push({
        id: 'drums',
        name: 'Drums',
        kind: 'drums',
        notes: hitsToNotes(drums.hits),
      });
      log.push(`Detected ${drums.hits.length} drum hits.`);
    } else {
      log.push('No drum kit detected.');
    }
  }

  // --- Tempo and key --------------------------------------------------
  report(0.92, 'Estimating tempo and key');
  const allNotes = tracks.flatMap((track) => track.notes);
  const tempo: TempoInfo = options.forceTempo
    ? { bpm: options.forceTempo, confidence: 1, adjusted: false }
    : estimateTempo(allNotes, duration);
  const key: KeyInfo = options.forceKey
    ? { ...options.forceKey, confidence: 1 }
    : estimateKey(allNotes);

  // --- Map the times back onto the original audio ---------------------
  if (tuning.applied) {
    for (const track of tracks) {
      for (const note of track.notes) note.startTime = toSource(note.startTime);
    }
  }

  // --- The two hands ----------------------------------------------------
  //
  // This runs on the notes the model found, before the range, because the
  // arrangement is a decision about the whole piece: the split between the
  // hands and the notes that are dropped depend on the music as a whole. The
  // range then chooses from the arranged notes.
  //
  // The tempo and the key were estimated earlier, from the notes as the
  // model found them: they belong to the song, not to the arrangement.
  if (options.pianify) {
    const arranged = pianify(tracks, options.pianify);
    tracks.length = 0;
    tracks.push(...arranged.tracks);
    log.push(arranged.log);
  }

  // --- The frequency range ---------------------------------------------
  //
  // This runs on the notes, not on the audio. Filtering the audio instead
  // would be much worse: a model reads a note from its harmonic series, and
  // cutting the audio takes that evidence away. See `range.ts`.
  if (options.range) {
    const before = tracks.reduce((sum, track) => sum + track.notes.length, 0);
    for (const track of tracks) {
      track.notes = filterByRange(track.notes, track.kind, options.range);
    }
    const after = tracks.reduce((sum, track) => sum + track.notes.length, 0);
    log.push(
      `Kept ${describeRange(options.range)}: ${after} of ${before} notes.`,
    );
  }

  for (const track of tracks) sortNotes(track.notes);

  // --- The grid ---------------------------------------------------------
  //
  // This runs after the range, so only the notes that are kept are moved.
  // The tempo and the key were estimated before both, from the notes as the
  // model found them: the grid is for the thing watching the file, not for
  // the song.
  if (options.quantise && options.quantise.grid > 0) {
    for (const track of tracks) {
      track.notes = quantiseNotes(track.notes, options.quantise);
    }
    log.push(`Snapped the notes to a grid of ${describeQuantise(options.quantise)}.`);
  }

  report(1, 'Done');

  return {
    tracks,
    tempo,
    key,
    duration,
    stats: {
      modelNotes: modelNoteCount,
      finalNotes: tracks.reduce((sum, track) => sum + track.notes.length, 0),
      drumHits,
      drumsDetected,
    },
    log,
  };
}
