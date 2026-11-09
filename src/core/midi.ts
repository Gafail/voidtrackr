/**
 * Multi-track MIDI output.
 *
 * The file is written as a Standard MIDI File, format 1: track 0 is the
 * conductor (tempo, time signature, key signature) and every musical part
 * gets its own track, so a DAW can show them as separate instrument lanes.
 */

import * as ToneMidi from '@tonejs/midi';
import { KeyInfo, RawNote, TempoInfo, Track, TrackKind } from './types';

/**
 * `@tonejs/midi` ships as CommonJS, so Node's ESM loader exposes it as a
 * default export while bundlers expose the named export. Resolve both.
 */
function midiConstructor(): typeof ToneMidi.Midi {
  const module = ToneMidi as unknown as {
    Midi?: typeof ToneMidi.Midi;
    default?: typeof ToneMidi.Midi | { Midi?: typeof ToneMidi.Midi };
  };
  if (typeof module.Midi === 'function') return module.Midi;
  if (module.default) {
    if (typeof module.default === 'function') return module.default;
    if (typeof module.default.Midi === 'function') return module.default.Midi;
  }
  throw new Error('Could not load @tonejs/midi');
}

/** MIDI channel assigned to each kind of track. */
const CHANNELS: Record<TrackKind, number> = {
  melody: 0,
  chords: 1,
  bass: 2,
  extra: 3,
  drums: 9,
};

/** General MIDI program assigned to each kind of track. */
const PROGRAMS: Record<TrackKind, number> = {
  melody: 0, // Acoustic Grand Piano
  chords: 0,
  bass: 33, // Electric Bass, finger
  extra: 0,
  drums: 0,
};

/**
 * Major key names by tonic, spelled with flats where that is the usual
 * choice. These are the spellings a MIDI key signature event understands.
 */
const MAJOR_KEY_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];

/**
 * Key names by signed key number, from seven flats to seven sharps.
 *
 * A MIDI key signature event holds a number from -7 to 7, where 0 is C
 * major, so the position in this list minus seven is that number.
 */
const KEY_SIGNATURE_NAMES = [
  'Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F',
  'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
];

/** The signed key number of a major key, as a MIDI key signature wants it. */
function signedKeyNumber(name: string): number {
  const index = KEY_SIGNATURE_NAMES.indexOf(name);
  if (index < 0) throw new Error(`Unknown key name: ${name}`);
  return index - 7;
}

export interface MidiExportOptions {
  /** Beats per bar for the time signature. */
  beatsPerBar?: number;
  /** Beat unit for the time signature (4 = quarter note). */
  beatUnit?: number;
  /** MIDI track names. */
  includeNames?: boolean;
  /**
   * A stamp describing the pipeline version and the model that produced the
   * file. It is written as a text meta event in the header track, so the
   * file carries its own provenance.
   */
  provenance?: string;
}

/** Convert a track's notes into a MIDI track on its own channel. */
export function buildMidiFile(
  tracks: Track[],
  tempo: TempoInfo,
  key: KeyInfo,
  options: MidiExportOptions = {},
): Uint8Array {
  const beatsPerBar = options.beatsPerBar ?? 4;
  const beatUnit = options.beatUnit ?? 4;
  const includeNames = options.includeNames ?? true;

  const midi = new (midiConstructor())();

  // The library writes the header as the first track, so the tempo, the time
  // signature and the key all live there.
  midi.header.name = 'voidtrackr';
  midi.header.setTempo(tempo.bpm);
  midi.header.timeSignatures.push({
    ticks: 0,
    timeSignature: [beatsPerBar, beatUnit],
    measures: 0,
  });

  // The key signature event is rewritten afterwards, because the library
  // writes the wrong values for it. See `writeKeySignature`.
  const keyName = MAJOR_KEY_NAMES[key.mode === 'minor' ? (key.tonic + 3) % 12 : key.tonic];
  midi.header.keySignatures.push({
    ticks: 0,
    key: keyName,
    scale: key.mode,
  });

  // The stamp goes in the header track as a text meta event, which every DAW
  // reads and ignores.
  if (options.provenance) {
    midi.header.meta.push({ text: options.provenance, type: 'text', ticks: 0 });
  }

  for (const track of tracks) {
    if (track.notes.length === 0) continue;
    const midiTrack = midi.addTrack();
    midiTrack.name = track.name;
    midiTrack.channel = CHANNELS[track.kind] ?? 0;
    if (track.kind !== 'drums') {
      midiTrack.instrument.number = PROGRAMS[track.kind] ?? 0;
    }

    for (const note of track.notes) {
      midiTrack.addNote({
        midi: note.pitchMidi,
        time: Math.max(0, note.startTime),
        duration: Math.max(0.01, note.duration),
        velocity: Math.max(0.01, Math.min(1, note.amplitude)),
      });
    }
  }

  if (!includeNames) {
    for (const track of midi.tracks) track.name = '';
  }

  const bytes = midi.toArray();
  writeKeySignature(bytes, signedKeyNumber(keyName), key.mode === 'minor' ? 1 : 0);
  return bytes;
}

/**
 * Rewrite the key signature event in the header track.
 *
 * `@tonejs/midi` writes the key signature wrong in two ways. It treats the
 * position of the key name in its table of names as if it were the signed key
 * number, so C major comes out as fourteen sharps instead of none. And it
 * writes the mode as a string, which always lands in the file as zero, so a
 * minor key is written as if it were major.
 *
 * The event is a fixed five bytes, `FF 59 02 sf mi`, so the two bytes that
 * matter can simply be overwritten once the file has been written. This walks
 * the header track rather than searching for the byte pattern, so it cannot
 * hit anything else.
 */
function writeKeySignature(bytes: Uint8Array, signedKey: number, mode: 0 | 1): void {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (bytes.length < 14) return;
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== 'MThd') return;

  const headerLength = view.getUint32(4, false);
  const trackStart = 8 + headerLength;
  if (trackStart + 8 > bytes.length) return;
  if (String.fromCharCode(...bytes.subarray(trackStart, trackStart + 4)) !== 'MTrk') return;

  const trackLength = view.getUint32(trackStart + 4, false);
  let offset = trackStart + 8;
  const end = Math.min(offset + trackLength, bytes.length);

  while (offset < end) {
    // Delta time: a variable length number.
    offset = readVarInt(view, offset).offset;

    if (offset >= end) break;
    const status = bytes[offset];
    offset += 1;

    if (status === 0xff) {
      // Meta event: a type byte, then the length, then the data.
      const type = bytes[offset];
      offset += 1;
      const length = readVarInt(view, offset);
      offset = length.offset;

      if (type === 0x59) {
        bytes[offset] = signedKey & 0xff;
        bytes[offset + 1] = mode;
        return;
      }
      offset += length.value;
    } else if (status === 0xf0 || status === 0xf7) {
      // System exclusive: a length, then the data.
      const length = readVarInt(view, offset);
      offset = length.offset + length.value;
    } else if ((status & 0xf0) === 0xc0 || (status & 0xf0) === 0xd0) {
      offset += 1; // one data byte
    } else {
      offset += 2; // two data bytes
    }
  }
}

/**
 * Read a variable length number.
 *
 * Each byte carries seven bits, and the top bit says whether another byte
 * follows.
 */
function readVarInt(view: DataView, offset: number): { value: number; offset: number } {
  let value = 0;
  let byte: number;
  do {
    byte = view.getUint8(offset);
    offset += 1;
    value = (value << 7) | (byte & 0x7f);
  } while (byte & 0x80);
  return { value, offset };
}

/**
 * Copy the bytes into a fresh ArrayBuffer.
 *
 * The MIDI writer hands back a view over a buffer that may be shared, and
 * `Blob` only accepts a plain `ArrayBuffer`.
 */
export function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return copy.buffer;
}

/** Write the MIDI file to a Blob for download in the browser. */
export function midiToBlob(bytes: Uint8Array): Blob {
  return new Blob([toArrayBuffer(bytes)], { type: 'audio/midi' });
}

/**
 * Flatten all tracks into a single note list, for tools that want one line
 * of music rather than separate parts.
 */
export function flattenTracks(tracks: Track[]): RawNote[] {
  return tracks.flatMap((track) => track.notes);
}
