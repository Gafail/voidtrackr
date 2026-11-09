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
import { correctTuning } from './tuning';
import { estimateKey, estimateTempo } from './tempo';
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
  if (options.detectDrums !== false) {
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

  for (const track of tracks) sortNotes(track.notes);

  report(1, 'Done');

  return {
    tracks,
    tempo,
    key,
    duration,
    stats: {
      modelNotes: modelNoteCount,
      finalNotes: allNotes.length,
      drumHits,
      drumsDetected,
    },
    log,
  };
}
