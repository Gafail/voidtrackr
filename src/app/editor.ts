/**
 * The editor's own view of a transcription.
 *
 * The core pipeline works with plain notes; the editor needs stable ids so
 * it can track a note through edits, and a per-track mute state.
 */

import { RawNote, Track, TrackKind } from '@core/types';

export interface EditorNote {
  /** Stable id, used for selection. */
  id: string;
  trackId: string;
  /** MIDI note number. */
  pitch: number;
  /** Start time in seconds. */
  start: number;
  /** Duration in seconds. */
  duration: number;
  /** Velocity, 0..1. */
  velocity: number;
}

export interface EditorTrack {
  id: string;
  name: string;
  kind: TrackKind;
  notes: EditorNote[];
  muted: boolean;
  color: string;
}

/** Colour per kind of track, so parts are easy to tell apart. */
export const TRACK_COLORS: Record<TrackKind, string> = {
  melody: '#5b9be8',
  chords: '#a78bfa',
  bass: '#e8a05c',
  drums: '#3fb980',
  extra: '#8683a0',
};

let counter = 0;

/** A fresh id. Ids only need to be unique within a session. */
export function nextId(prefix = 'n'): string {
  counter += 1;
  return `${prefix}${counter}`;
}

/** Convert the pipeline's tracks into the editor's tracks. */
export function toEditorTracks(tracks: Track[]): EditorTrack[] {
  return tracks.map((track) => ({
    id: track.id,
    name: track.name,
    kind: track.kind,
    muted: track.muted ?? false,
    color: TRACK_COLORS[track.kind] ?? TRACK_COLORS.extra,
    notes: track.notes.map((note) => ({
      id: nextId(),
      trackId: track.id,
      pitch: note.pitchMidi,
      start: note.startTime,
      duration: note.duration,
      velocity: note.amplitude,
    })),
  }));
}

/** Convert the editor's tracks back into the pipeline's tracks. */
export function fromEditorTracks(tracks: EditorTrack[]): Track[] {
  return tracks.map((track) => ({
    id: track.id,
    name: track.name,
    kind: track.kind,
    muted: track.muted,
    notes: track.notes.map(
      (note): RawNote => ({
        pitchMidi: note.pitch,
        startTime: note.start,
        duration: note.duration,
        amplitude: note.velocity,
      }),
    ),
  }));
}

/** The lowest and highest note in a set of tracks. */
export function pitchRange(tracks: EditorTrack[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const track of tracks) {
    for (const note of track.notes) {
      if (note.pitch < min) min = note.pitch;
      if (note.pitch > max) max = note.pitch;
    }
  }
  if (!Number.isFinite(min)) return { min: 48, max: 72 };
  return { min: Math.floor(min), max: Math.ceil(max) };
}

/** The total length of the material, in seconds. */
export function contentDuration(tracks: EditorTrack[]): number {
  let end = 0;
  for (const track of tracks) {
    for (const note of track.notes) end = Math.max(end, note.start + note.duration);
  }
  return end;
}
