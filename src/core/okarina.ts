/**
 * The Okarina model.
 *
 * Okarina is an unofficial ONNX reimplementation of Harmonica, the model
 * BandLab's audio-to-MIDI service is built on. Harmonica's own weights were
 * never released; these come from the reimplementation.
 *
 *   **Licence: CC BY-NC-SA 4.0. Non-commercial use only.**
 *
 * It works in two steps. A parameter-free CQT front end turns 16 kHz audio
 * into a magnitude spectrogram of 156 frames by 264 bins. A note head then
 * turns that into three matrices, one per part (vocal, bass, other), each
 * with an onset, a sustain and an offset value per frame and pitch:
 *
 *   - *onset* fires on the frame where a note is struck;
 *   - *sustain* says how strongly the pitch is sounding;
 *   - *offset* fires on the frame where the note is released.
 *
 * The offset head matters. Deciding where a note ends from the sustain value
 * alone is coarse, because the sustain decays over several frames; the offset
 * head puts the end on the right frame.
 *
 * The frame timebase below was measured against tone bursts at known times,
 * not assumed.
 */

import { ModelOptions, ModelPart, ModelTranscription, TranscriptionModel } from './model';
import { OnnxRunner, OnnxSession, OnnxTensor } from './onnx';
import { RawNote } from './types';

/** The front end wants 16 kHz mono. */
export const OKARINA_SAMPLE_RATE = 16000;

/** Samples the CQT front end consumes per window. */
const FRONTEND_SAMPLES = 109869;

/** Frames the front end produces per window. */
const FRONTEND_FRAMES = 156;

/** Contour bins the front end produces per frame. */
const FRONTEND_BINS = 264;

/** Pitch bins the note head produces per frame. */
const PITCHES = 88;

/**
 * Seconds between frames.
 *
 * The note head's CQT has a hop of 512 samples at 16 kHz, which is exactly
 * 32 ms. Measured by fitting a line through the onset peaks of tone bursts
 * at known times; every burst then lands on its true time to the millisecond.
 */
const FRAME_SEC = 0.032;

/**
 * Seconds of leading context before the first frame.
 *
 * Also measured, the same way. The frames of consecutive windows are
 * contiguous: the first frame of a window is the frame after the last one
 * of the window before it.
 */
const FRAME_OFFSET = 0.976;

/** The parts the note head separates, in the order it reports them. */
export const OKARINA_PARTS = ['vocal', 'bass', 'other'] as const;

/** Which of our track kinds each part becomes. */
const PART_KINDS = ['melody', 'bass', 'chords'] as const;

/** Names shown to the user for each part. */
const PART_NAMES = ['Melody', 'Bass', 'Chords'] as const;

/** File names of the pinned weights. */
export const OKARINA_FILES = {
  frontEnd: 'cqt156.onnx',
  s: 'okarina_s.onnx',
  m: 'okarina_m.onnx',
  l: 'okarina_l.onnx',
} as const;

export type OkarinaSize = 's' | 'm' | 'l';

export interface OkarinaOptions {
  /** Onset strength needed to start a note, 0..1. */
  onsetThreshold?: number;
  /** Offset strength needed to end a note, 0..1. */
  offsetThreshold?: number;
  /** Sustain level below which a pitch counts as silent, 0..1. */
  sustainFloor?: number;
  /** Notes shorter than this are dropped, in seconds. */
  minDuration?: number;
}

const DEFAULTS: Required<OkarinaOptions> = {
  onsetThreshold: 0.3,
  offsetThreshold: 0.3,
  sustainFloor: 0.15,
  minDuration: 0.05,
};

/** One part's output for the whole recording. */
interface PartMatrices {
  onset: Float32Array;
  sustain: Float32Array;
  offset: Float32Array;
}

/**
 * Okarina, as a model the pipeline can use.
 */
export class OkarinaModel implements TranscriptionModel {
  readonly id = 'okarina-mazun-2026';
  readonly displayName = 'Okarina (Harmonica-class)';
  readonly license = 'CC BY-NC-SA 4.0';
  readonly inputSampleRate = OKARINA_SAMPLE_RATE;

  constructor(
    private readonly frontEnd: OnnxSession,
    private readonly noteHead: OnnxSession,
    /** Hash of the weights this instance was loaded from. */
    public readonly fingerprint: string = 'unknown',
    private readonly options: OkarinaOptions = {},
  ) {}

  /**
   * Load the two graphs.
   *
   * `path` is the directory the weights live in. The caller is responsible
   * for checking them against the lock file first.
   */
  static async load(
    runner: OnnxRunner,
    path: string,
    size: OkarinaSize = 'm',
    fingerprint = 'unknown',
  ): Promise<OkarinaModel> {
    const frontEnd = await runner.createSession(joinPath(path, OKARINA_FILES.frontEnd));
    const noteHead = await runner.createSession(joinPath(path, OKARINA_FILES[size]));
    return new OkarinaModel(frontEnd, noteHead, fingerprint);
  }

  async transcribe(samples: Float32Array, options: ModelOptions = {}): Promise<ModelTranscription> {
    const report = (value: number, label: string) => options.onProgress?.(value, label);
    const settings = { ...DEFAULTS, ...this.options };

    report(0.05, 'Analysing the spectrum');
    const matrices = await this.runFrontEnd(samples, report);

    report(0.55, 'Reading notes');
    const parts = this.decode(matrices, settings, samples.length / OKARINA_SAMPLE_RATE);

    return {
      parts,
      log: [
        `Okarina found ${parts.reduce((sum, part) => sum + part.notes.length, 0)} notes ` +
          `in ${parts.length} parts.`,
      ],
    };
  }

  /**
   * Run the CQT front end and the note head over the whole recording.
   *
   * The front end consumes 6.87 s of audio but reports 5 s of frames; the
   * extra is context for the note head. Consecutive windows therefore start
   * exactly one frame block apart, and the frames are simply concatenated.
   */
  private async runFrontEnd(
    samples: Float32Array,
    report: (value: number, label: string) => void,
  ): Promise<PartMatrices[]> {
    const hopSamples = Math.round(FRAME_SEC * FRONTEND_FRAMES * OKARINA_SAMPLE_RATE);
    const windowCount = Math.max(1, Math.ceil(samples.length / hopSamples));
    const frames = windowCount * FRONTEND_FRAMES;

    const parts: PartMatrices[] = OKARINA_PARTS.map(() => ({
      onset: new Float32Array(frames * PITCHES),
      sustain: new Float32Array(frames * PITCHES),
      offset: new Float32Array(frames * PITCHES),
    }));

    const window = new Float32Array(FRONTEND_SAMPLES);

    for (let k = 0; k < windowCount; k++) {
      const start = k * hopSamples;
      window.fill(0);
      window.set(samples.subarray(start, start + FRONTEND_SAMPLES));

      const cqtOut = await this.frontEnd.run({
        [this.frontEnd.inputNames[0]]: {
          type: 'float32',
          data: window,
          dims: [1, FRONTEND_SAMPLES],
        },
      });
      const cqt = cqtOut[this.frontEnd.outputNames[0]] as OnnxTensor;

      const out = await this.noteHead.run({
        cqt_mag: {
          type: 'float32',
          data: cqt.data as Float32Array,
          dims: [1, FRONTEND_FRAMES, FRONTEND_BINS],
        },
      });

      // The note head reports one part at a time, laid out as [part, frame, pitch].
      for (let part = 0; part < OKARINA_PARTS.length; part++) {
        const target = parts[part];
        for (const [name, buffer] of [
          ['onset', target.onset],
          ['sustain', target.sustain],
          ['offset', target.offset],
        ] as const) {
          const tensor = out[name] as OnnxTensor;
          const values = tensor.data as Float32Array;
          for (let fr = 0; fr < FRONTEND_FRAMES; fr++) {
            const dst = (k * FRONTEND_FRAMES + fr) * PITCHES;
            const src = (part * FRONTEND_FRAMES + fr) * PITCHES;
            buffer.set(values.subarray(src, src + PITCHES), dst);
          }
        }
      }

      report(0.05 + (0.5 * (k + 1)) / windowCount, 'Analysing the spectrum');
    }

    return parts;
  }

  /**
   * Turn the matrices into notes.
   *
   * A note runs from the frame its onset fires to the frame its offset fires.
   * The sustain value in between gives the note its loudness, so the output
   * has real dynamics rather than a flat velocity.
   */
  private decode(
    parts: PartMatrices[],
    settings: Required<OkarinaOptions>,
    audioDuration: number,
  ): ModelPart[] {
    // The frames only describe the part of the recording that was analysed.
    const usableFrames = Math.max(
      0,
      Math.min(
        parts[0].onset.length / PITCHES,
        Math.ceil((audioDuration - FRAME_OFFSET) / FRAME_SEC),
      ),
    );

    const result: ModelPart[] = [];

    for (let part = 0; part < parts.length; part++) {
      const { onset, sustain, offset } = parts[part];
      const notes: RawNote[] = [];

      for (let pitch = 0; pitch < PITCHES; pitch++) {
        let open = false;
        let startFrame = 0;

        // A note cannot run past the end of the recording, even when the last
        // frame does.
        const clampEnd = (frame: number) =>
          Math.min(frame * FRAME_SEC + FRAME_OFFSET, audioDuration);

        for (let f = 0; f < usableFrames; f++) {
          const at = f * PITCHES + pitch;
          const onsetValue = onset[at];
          const offsetValue = offset[at];
          const sustainValue = sustain[at];

          // A note ends when its offset fires, or when the pitch falls silent
          // and stays that way.
          if (open) {
            const silent = sustainValue < settings.sustainFloor;
            if (
              offsetValue >= settings.offsetThreshold ||
              (silent &&
                f + 1 < usableFrames &&
                sustain[(f + 1) * PITCHES + pitch] < settings.sustainFloor)
            ) {
              const duration = clampEnd(f) - clampEnd(startFrame);
              if (duration >= settings.minDuration) {
                notes.push({
                  pitchMidi: pitch + 21,
                  startTime: clampEnd(startFrame),
                  duration,
                  amplitude: meanSustain(sustain, startFrame, f, pitch),
                });
              }
              open = false;
            }
          }

          // A note starts when its onset fires.
          if (!open && onsetValue >= settings.onsetThreshold) {
            open = true;
            startFrame = f;
          }
        }

        // A note still sounding at the end of the recording.
        if (open) {
          const duration = clampEnd(usableFrames) - clampEnd(startFrame);
          if (duration >= settings.minDuration) {
            notes.push({
              pitchMidi: pitch + 21,
              startTime: clampEnd(startFrame),
              duration,
              amplitude: meanSustain(sustain, startFrame, usableFrames - 1, pitch),
            });
          }
        }
      }

      notes.sort((a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi);

      if (notes.length === 0) continue;
      result.push({
        id: OKARINA_PARTS[part],
        name: PART_NAMES[part],
        kind: PART_KINDS[part],
        notes,
      });
    }

    return result;
  }
}

/** Mean sustain over a range of frames, which becomes the note's loudness. */
function meanSustain(sustain: Float32Array, from: number, to: number, pitch: number): number {
  let sum = 0;
  let count = 0;
  for (let f = from; f <= to; f++) {
    sum += sustain[f * PITCHES + pitch];
    count += 1;
  }
  return count > 0 ? Math.min(1, Math.max(0.05, sum / count)) : 0.5;
}

/** Join a directory and a file name, for both Node and the browser. */
function joinPath(directory: string, name: string): string {
  if (directory.endsWith('/') || directory.endsWith('\\')) return directory + name;
  return `${directory}/${name}`;
}
