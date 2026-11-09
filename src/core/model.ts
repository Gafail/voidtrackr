/**
 * The contract every transcription model has to meet.
 *
 * A model is handed mono audio and hands back notes, grouped into the parts
 * it thinks they belong to. Everything downstream of that — tempo, key, drum
 * detection, the MIDI writer — is the same whichever model produced the
 * notes, so a new model can be dropped in without touching the rest of the
 * pipeline.
 *
 * The files in this module are free of DOM and Node specifics, and of
 * TensorFlow, so both the browser and the command line can load it.
 */

import {
  ANNOT_N_FRAMES,
  CONTOURS_BINS_PER_SEMITONE,
  MIDI_OFFSET,
  SAMPLE_RATE,
} from './constants';
import { RawNote, TrackKind } from './types';

/** Number of pitch bins the models report. */
export const PITCH_BINS = 88;

/**
 * What a model has to produce.
 *
 * Each model is free to work in its own way: Basic Pitch reasons about
 * frames of pitch probabilities, Okarina about onsets and sustains per part.
 * They all end up here, as notes with a start time and a duration.
 */
export interface ModelPart {
  /** Stable identifier, used in the UI and in the MIDI file. */
  id: string;
  /** Name shown to the user. */
  name: string;
  /** The musical role of this part. */
  kind: TrackKind;
  /** The notes, in seconds from the beginning of the file. */
  notes: RawNote[];
}

/** What a model reports back. */
export interface ModelTranscription {
  /** The parts the model found, in the order they should be shown. */
  parts: ModelPart[];
  /** Human-readable notes about what the model did, for the UI. */
  log: string[];
}

export interface ModelOptions {
  /** Called with a 0..1 progress value and a short label. */
  onProgress?: (value: number, label: string) => void;
  /** Label used in progress messages. */
  label?: string;
}

/**
 * A transcription model.
 *
 * This is the seam where a different model can be dropped in. Anything that
 * can turn a signal into notes can replace the current one without the rest
 * of the pipeline changing: the drum detector, the tempo estimator and the
 * key estimator all work on the result, not on the model.
 */
export interface TranscriptionModel {
  /** Stable identifier, e.g. `basic-pitch-icassp2022`. */
  readonly id: string;
  /** Name shown to the user. */
  readonly displayName: string;
  /** Licence the weights are under. */
  readonly license: string;
  /** Hash of the weights, so an output can be traced to them. */
  readonly fingerprint: string;
  /** The sample rate the model wants its input at. */
  readonly inputSampleRate: number;
  /** Turn a mono signal into notes. */
  transcribe(samples: Float32Array, options?: ModelOptions): Promise<ModelTranscription>;
  /** Release whatever the model is holding on to. */
  dispose?(): void;
}

/**
 * The shape a model has to present for the rest of the pipeline to work.
 *
 * Recorded in the lock file so a model that does not fit is rejected up
 * front, rather than producing silently wrong notes.
 */
export interface ModelContract {
  /** Sample rate the model expects. */
  sampleRate: number;
  /** Number of pitch bins per frame. */
  pitchBins: number;
  /** MIDI number of the lowest pitch bin. */
  lowestMidi: number;
  /** Annotation frames produced for one analysis window. */
  framesPerWindow: number;
  /** Contour resolution, in bins per semitone. */
  contourBinsPerSemitone: number;
}

/** The pinned model definition, as recorded in `model.lock.json`. */
export interface ModelLock {
  id: string;
  displayName: string;
  license: string;
  source: string;
  files: Record<string, { sha256: string }>;
  contract: ModelContract;
}

/**
 * Where a model's files come from.
 *
 * The browser can fetch them over HTTP; Node has to read them off disk
 * because its `fetch` does not understand `file://` URLs. Keeping the two
 * apart here means the verification logic is shared, not duplicated.
 */
export interface ModelSource {
  /** The URL the runtime loads the graph model from. */
  url: string;
  /**
   * Read one of the model's files. `name` is a path relative to the model
   * directory, such as `model.json` or `group1-shard1of1.bin`.
   */
  read(name: string): Promise<ArrayBuffer>;
}

/**
 * Check every file of a model against the lock file.
 *
 * Returns the model's fingerprint, which is the hash of its manifest. Any
 * model loader can use this, so a replacement model gets the same checking
 * for free.
 */
export async function verifyModelFiles(
  source: ModelSource,
  lock: ModelLock,
): Promise<string> {
  assertContractMatches(lock);

  const manifestBytes = await source.read('model.json');
  const digest = await sha256Hex(manifestBytes);
  assertModelMatches(digest, lock, 'model.json');

  // The manifest lists the weight shards it needs. Checking them too means a
  // swapped-out shard is caught, not just a rewritten manifest.
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as {
    weightsManifest?: { paths?: string[] }[];
  };
  const shards = (manifest.weightsManifest ?? []).flatMap((group) => group.paths ?? []);

  for (const shard of shards) {
    const shardDigest = await sha256Hex(await source.read(shard));
    assertModelMatches(shardDigest, lock, shard);
  }

  return digest;
}

/**
 * Compare a file's hash with the one recorded in the lock file.
 *
 * Throws with both hashes so the mismatch can be diagnosed without guessing.
 */
export function assertModelMatches(
  digest: string,
  lock: ModelLock,
  name: string,
): void {
  const expected = lock.files[name]?.sha256;
  if (!expected) {
    throw new Error(
      `The lock file does not record a hash for ${name}. ` +
        'Add it to model.lock.json before using this model.',
    );
  }
  if (digest !== expected) {
    throw new Error(
      `The model's ${name} does not match the pinned version.\n` +
        `  expected sha256 ${expected}\n` +
        `  got      sha256 ${digest}\n` +
        'Run `npm run model` to fetch the pinned weights.',
    );
  }
}

/**
 * Check that a model speaks the contract the pipeline is written against.
 *
 * Swapping in a model with a different window or a different pitch range
 * would shift every note in the output, so it is refused rather than
 * half-supported.
 */
export function assertContractMatches(lock: ModelLock): void {
  const contract = lock.contract;
  if (!contract) return;

  const expected: Array<[string, number, number]> = [
    ['sampleRate', contract.sampleRate, SAMPLE_RATE],
    ['pitchBins', contract.pitchBins, PITCH_BINS],
    ['lowestMidi', contract.lowestMidi, MIDI_OFFSET],
    ['framesPerWindow', contract.framesPerWindow, ANNOT_N_FRAMES],
    ['contourBinsPerSemitone', contract.contourBinsPerSemitone, CONTOURS_BINS_PER_SEMITONE],
  ];

  for (const [name, actual, wanted] of expected) {
    if (actual !== wanted) {
      throw new Error(
        `The pinned model does not fit this pipeline: ${name} is ${actual}, ` +
          `but the pipeline is built for ${wanted}. ` +
          'Adjust the pipeline or pin a different model.',
      );
    }
  }
}

/** SHA-256 of some bytes or a string, as hex. */
export async function sha256Hex(data: ArrayBuffer | Uint8Array | string): Promise<string> {
  const source = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
  // Copy into a plain ArrayBuffer, so the digest never sees a view onto
  // shared memory.
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Flattened model output.
 *
 * `frames`, `onsets` and `contours` are row-major `[nFrames * bins]` buffers.
 */
export interface ModelOutput {
  frames: Float32Array;
  onsets: Float32Array;
  contours: Float32Array;
  nFrames: number;
}

export interface EvaluateOptions {
  /** Called with a 0..1 progress value and a short label. */
  onProgress?: (value: number, label: string) => void;
  /** Label used in progress messages. */
  label?: string;
}
