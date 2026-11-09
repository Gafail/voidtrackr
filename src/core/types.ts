/**
 * Shared types for the audio -> MIDI pipeline.
 *
 * The pipeline is split into three stages so they can be swapped or improved
 * independently:
 *
 *   1. `RawNotes`  – straight out of the model, one entry per detected note.
 *   2. `Track`     – notes grouped into musical roles (melody / chords / bass).
 *   3. `MidiFile`  – serialisable MIDI bytes ready for download.
 */

export interface RawNote {
  /** MIDI note number (60 = middle C). */
  pitchMidi: number;
  /** Start time in seconds from the beginning of the file. */
  startTime: number;
  /** Duration in seconds. */
  duration: number;
  /** Model amplitude, 0..1. */
  amplitude: number;
  /**
   * Optional pitch bend curve, one value per frame (in semitones, -1..1),
   * describing how the pitch moved while the note was sounding.
   */
  pitchBends?: number[];
}

export type TrackKind = 'melody' | 'chords' | 'bass' | 'drums' | 'extra';

export interface Track {
  id: string;
  name: string;
  kind: TrackKind;
  notes: RawNote[];
  /** Per-track gain applied when the track is rendered to audio. */
  gain?: number;
  muted?: boolean;
}

export interface TempoInfo {
  /** Beats per minute. */
  bpm: number;
  /** Confidence in the estimate, 0..1. */
  confidence: number;
  /** True when the tempo was doubled/halved to land in a musical range. */
  adjusted: boolean;
}

export interface KeyInfo {
  /** 0..11, C = 0. */
  tonic: number;
  /** 'major' | 'minor' */
  mode: 'major' | 'minor';
  /** 0..1 */
  confidence: number;
}

export interface TranscriptionResult {
  tracks: Track[];
  tempo: TempoInfo;
  key: KeyInfo;
  /** Length of the source audio in seconds. */
  duration: number;
  /** Notes that were not confidently assigned to a musical role. */
  stats: {
    modelNotes: number;
    finalNotes: number;
    drumHits: number;
    drumsDetected: boolean;
  };
  /** Human-readable notes about what the pipeline did, for the UI. */
  log: string[];
}

export interface ProgressInfo {
  /** 0..1 */
  value: number;
  label: string;
}
