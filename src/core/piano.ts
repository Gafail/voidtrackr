/**
 * The high-resolution piano transcription model.
 *
 * ByteDance's piano model — Kong, Li, Song, Wan and Wang, "High-resolution
 * Piano Transcription with Pedals by Regressing Onsets and Offsets Times",
 * arXiv 2010.01815 — as an unofficial ONNX export.
 *
 *   **Licence: CC BY 4.0.** The weights come from Zenodo record 4034264.
 *
 * This is the only model here whose front end lives inside the graph. The
 * graph takes ten seconds of 16 kHz mono audio and computes its own log-mel
 * spectrogram, so there is no filterbank, no FFT and no windowing on this
 * side of the seam: the waveform goes in, four matrices come out.
 *
 * The matrices are one value per frame per pitch, and they say different
 * things:
 *
 *   - *reg onset* — how strong an onset is on that frame, and how far inside
 *     the frame it falls;
 *   - *reg offset* — the same for the end of a note;
 *   - *frame* — whether the pitch is sounding at all;
 *   - *velocity* — how hard the key was struck, sampled at the onset frame.
 *
 * The two regression heads are what make this model worth having. Instead of
 * snapping a note to a 10 ms grid, it says where inside the frame the note
 * really starts and ends, so the timing lands within a few milliseconds.
 *
 * The decoding below follows the released code
 * (github.com/bytedance/piano_transcription) closely, quirks included. Where
 * upstream has a bug or an oddity, the comment says so and says what was done
 * about it.
 */

import { ModelOptions, ModelPart, ModelTranscription, TranscriptionModel } from './model';
import { asFloat32, OnnxRunner, OnnxSession, OnnxTensor } from './onnx';
import { RawNote } from './types';

/** The sample rate the model wants its input at. */
export const PIANO_SAMPLE_RATE = 16000;

/** Directory the weights live in, under the ONNX weights root. */
export const PIANO_DIR = 'piano';

/** The pinned weight files. */
export const PIANO_FILES = {
  graph: 'piano_kong.onnx',
} as const;

/**
 * The graph's weights live in a sidecar file.
 *
 * The graph refers to it by this name, and the browser has to mount it
 * under that name because the runtime fetches the graph over HTTP and loses
 * the directory it came from.
 */
export const PIANO_EXTERNAL_DATA = 'piano_kong.onnx.data';

/** Samples in one analysis window: ten seconds at 16 kHz. */
const SEGMENT_SAMPLES = 160000;

/** How far the window moves each time: five seconds, half a window. */
const HOP_SAMPLES = 80000;

/**
 * Frames one window produces.
 *
 * The centred front end inside the graph reports one frame more than the
 * window holds, so the last frame of every window is dropped before the
 * windows are stitched together.
 */
const SEGMENT_FRAMES = 1001;

/** Frames one window keeps after that extra frame is dropped. */
const USABLE_FRAMES = 1000;

/** Frames the model reports per second. */
const FRAMES_PER_SECOND = 100;

/** Pitches: A0 to C8. */
const PITCHES = 88;

/** MIDI number of the lowest pitch, A0. */
const LOWEST_MIDI = 21;

/**
 * Onset strength needed to start a note.
 *
 * Upstream's released code uses 0.3 here, and 0.1 for the frame threshold.
 * The paper quotes 0.3 for the frame threshold as well; the code is what
 * produced the published numbers, so the code wins.
 */
const ONSET_THRESHOLD = 0.3;

/** Offset strength needed to end a note. */
const OFFSET_THRESHOLD = 0.3;

/** Sustain level below which a pitch counts as silent. */
const FRAME_THRESHOLD = 0.1;

/** How many frames on each side of a peak have to be falling for it to count. */
const ONSET_NEIGHBOUR = 2;
const OFFSET_NEIGHBOUR = 4;

/**
 * A note that has not ended after this many frames is closed anyway.
 *
 * Six seconds at 100 frames per second. Upstream's guard against a note that
 * never ends; it is kept as it is.
 */
const MAX_NOTE_FRAMES = 600;

/** The model's output matrices, one value per frame per pitch. */
interface PianoMatrices {
  /** Regression of the onset time, per frame and pitch. */
  onset: Float32Array;
  /** Regression of the offset time, per frame and pitch. */
  offset: Float32Array;
  /** Whether the pitch is sounding, per frame and pitch. */
  frame: Float32Array;
  /** How hard the key was struck, per frame and pitch. */
  velocity: Float32Array;
}

/** Where one window's frames land in the stitched output. */
interface WindowLayout {
  /** First frame of the window to keep. */
  from: number;
  /** Last frame of the window to keep, exclusive. */
  to: number;
  /** Where the kept frames start in the stitched output. */
  at: number;
}

/** A note as the decoder finds it, still in frames. */
interface DetectedNote {
  /** First frame of the note. */
  begin: number;
  /** Last frame of the note. */
  end: number;
  /** How far inside the begin frame the note really starts, in frames. */
  onsetShift: number;
  /** How far inside the end frame the note really ends, in frames. */
  offsetShift: number;
  /** How hard the key was struck, 0..1. */
  velocity: number;
}

/**
 * The piano model, as a model the pipeline can use.
 */
export class PianoTranscriptionModel implements TranscriptionModel {
  readonly id = 'piano-transcription-kong-2020';
  readonly displayName = 'Piano Transcription (Kong et al.)';
  readonly license = 'CC BY 4.0';
  readonly inputSampleRate = PIANO_SAMPLE_RATE;

  constructor(
    private readonly session: OnnxSession,
    /** Hash of the weights this instance was loaded from. */
    public readonly fingerprint: string = 'unknown',
  ) {}

  /**
   * Load the graph.
   *
   * `path` is the directory the weights live in. The caller is responsible
   * for checking them against the lock file first.
   */
  static async load(
    runner: OnnxRunner,
    path: string,
    fingerprint = 'unknown',
  ): Promise<PianoTranscriptionModel> {
    const session = await runner.createSession(joinPath(path, PIANO_FILES.graph), [
      {
        location: PIANO_EXTERNAL_DATA,
        url: joinPath(path, PIANO_EXTERNAL_DATA),
      },
    ]);
    return new PianoTranscriptionModel(session, fingerprint);
  }

  async transcribe(samples: Float32Array, options: ModelOptions = {}): Promise<ModelTranscription> {
    const report = (value: number, label: string) => options.onProgress?.(value, label);

    report(0.05, 'Running the piano model');
    const matrices = await this.runGraph(samples, report);

    report(0.9, 'Reading notes');
    const notes = this.decode(matrices, samples.length / PIANO_SAMPLE_RATE);

    const parts: ModelPart[] = [
      {
        id: PIANO_DIR,
        name: 'Piano',
        kind: 'melody',
        notes,
      },
    ];

    return {
      parts,
      log: [`Piano transcription found ${notes.length} notes.`],
    };
  }

  /**
   * Run the graph over the whole recording and stitch the windows together.
   *
   * Upstream pads the recording with silence to a whole number of windows,
   * then takes a window every half a window. Each window reports one frame
   * more than it holds, and that extra frame is dropped; what is left of the
   * windows is stitched so that the frames run continuously from the start of
   * the recording to its end.
   *
   * Upstream keeps that extra frame when the whole recording fits in one
   * window. Here it is dropped in every case, which costs one frame — ten
   * milliseconds — at the very end of the recording.
   */
  private async runGraph(
    samples: Float32Array,
    report: (value: number, label: string) => void,
  ): Promise<PianoMatrices> {
    const paddedLength = Math.ceil(samples.length / SEGMENT_SAMPLES) * SEGMENT_SAMPLES;
    const layouts = planWindows(countWindows(paddedLength));

    const matrices: PianoMatrices = {
      onset: new Float32Array(totalFrames(layouts) * PITCHES),
      offset: new Float32Array(totalFrames(layouts) * PITCHES),
      frame: new Float32Array(totalFrames(layouts) * PITCHES),
      velocity: new Float32Array(totalFrames(layouts) * PITCHES),
    };

    if (layouts.length === 0) return matrices;

    const padded = new Float32Array(paddedLength);
    padded.set(samples);

    const window = new Float32Array(SEGMENT_SAMPLES);

    for (let s = 0; s < layouts.length; s++) {
      const layout = layouts[s];
      const start = s * HOP_SAMPLES;

      window.fill(0);
      window.set(padded.subarray(start, start + SEGMENT_SAMPLES));

      const out = await this.session.run({
        [this.session.inputNames[0]]: {
          type: 'float32',
          data: window,
          dims: [1, SEGMENT_SAMPLES],
        },
      });

      const sources = [
        ['reg_onset_output', matrices.onset],
        ['reg_offset_output', matrices.offset],
        ['frame_output', matrices.frame],
        ['velocity_output', matrices.velocity],
      ] as const;

      for (const [name, target] of sources) {
        const tensor = out[name] as OnnxTensor | undefined;
        if (!tensor) {
          throw new Error(`The piano graph did not produce "${name}".`);
        }
        const values = asFloat32(tensor);
        if (values.length < SEGMENT_FRAMES * PITCHES) {
          throw new Error(
            `The piano graph returned ${values.length} values for "${name}", ` +
              `too few for ${SEGMENT_FRAMES} frames of ${PITCHES} pitches.`,
          );
        }
        for (let f = layout.from; f < layout.to; f++) {
          const src = f * PITCHES;
          const dst = (layout.at + (f - layout.from)) * PITCHES;
          target.set(values.subarray(src, src + PITCHES), dst);
        }
      }

      report(0.05 + (0.8 * (s + 1)) / layouts.length, 'Running the piano model');
    }

    return matrices;
  }

  /**
   * Turn the stitched matrices into notes.
   *
   * Every pitch is decoded on its own, because a note is a run of frames on
   * one pitch and the model's heads do not talk across pitches.
   *
   * Only the part of the recording that is really audio is decoded. The rest
   * of the last window is the silence the recording was padded with, and the
   * model can still fire an onset in it — a note the recording does not
   * contain.
   */
  private decode(matrices: PianoMatrices, audioDuration: number): RawNote[] {
    const frames = matrices.onset.length / PITCHES;
    const usable = Math.max(
      0,
      Math.min(frames, Math.ceil(audioDuration * FRAMES_PER_SECOND)),
    );

    const notes: RawNote[] = [];

    // The columns are copied out so the inner loops walk memory in order
    // rather than striding across the whole matrix for every pitch.
    const onsetColumn = new Float32Array(usable);
    const offsetColumn = new Float32Array(usable);
    const frameColumn = new Float32Array(usable);
    const velocityColumn = new Float32Array(usable);

    for (let pitch = 0; pitch < PITCHES; pitch++) {
      for (let f = 0; f < usable; f++) {
        const at = f * PITCHES + pitch;
        onsetColumn[f] = matrices.onset[at];
        offsetColumn[f] = matrices.offset[at];
        frameColumn[f] = matrices.frame[at];
        velocityColumn[f] = matrices.velocity[at];
      }

      const detected = decodePitch({
        onset: onsetColumn,
        offset: offsetColumn,
        frame: frameColumn,
        velocity: velocityColumn,
        frames: usable,
      });

      for (const note of detected) {
        // The onset shift can be negative, and a note on the very first frame
        // would then start before the recording does.
        const startTime = Math.max(
          0,
          (note.begin + note.onsetShift) / FRAMES_PER_SECOND,
        );
        // A note cannot run past the end of the recording. The offset
        // regression can put the end a frame or two beyond it, because the
        // window it came from was padded with silence.
        const endTime = Math.min(
          audioDuration,
          (note.end + note.offsetShift) / FRAMES_PER_SECOND,
        );

        notes.push({
          pitchMidi: pitch + LOWEST_MIDI,
          startTime,
          duration: Math.max(0, endTime - startTime),
          // Upstream turns this into a MIDI velocity by multiplying by 128
          // and truncating. The pipeline carries loudness as 0..1, so the
          // number is kept as it is.
          amplitude: Math.min(1, Math.max(0, note.velocity)),
        });
      }
    }

    notes.sort((a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi);

    return notes;
  }
}

/** How many windows the recording needs. */
function countWindows(totalSamples: number): number {
  let count = 0;
  for (let pointer = 0; pointer + SEGMENT_SAMPLES <= totalSamples; pointer += HOP_SAMPLES) {
    count += 1;
  }
  return count;
}

/**
 * Say which frames of each window are kept.
 *
 * The windows overlap by half, so the frames that overlap are dropped and
 * what is left runs continuously: the first window keeps its first three
 * quarters, a window in the middle keeps the half in the middle, and the last
 * window keeps its last three quarters. Upstream does the same thing, as
 * fractions of the window's frames.
 */
function planWindows(windowCount: number): WindowLayout[] {
  const layouts: WindowLayout[] = [];
  let at = 0;

  for (let s = 0; s < windowCount; s++) {
    let from: number;
    let to: number;

    if (windowCount === 1) {
      // One window covers the whole recording, so all of it is kept.
      from = 0;
      to = USABLE_FRAMES;
    } else if (s === 0) {
      from = 0;
      to = Math.round(USABLE_FRAMES * 0.75);
    } else if (s === windowCount - 1) {
      from = Math.round(USABLE_FRAMES * 0.25);
      to = USABLE_FRAMES;
    } else {
      from = Math.round(USABLE_FRAMES * 0.25);
      to = Math.round(USABLE_FRAMES * 0.75);
    }

    layouts.push({ from, to, at });
    at += to - from;
  }

  return layouts;
}

/** How many frames the stitched output holds. */
function totalFrames(layouts: WindowLayout[]): number {
  if (layouts.length === 0) return 0;
  const last = layouts[layouts.length - 1];
  return last.at + (last.to - last.from);
}

/**
 * Decode one pitch.
 *
 * This is upstream's `note_detection_with_onset_offset_regress`, with the
 * regression heads binarised first.
 */
function decodePitch(columns: {
  onset: Float32Array;
  offset: Float32Array;
  frame: Float32Array;
  velocity: Float32Array;
  frames: number;
}): DetectedNote[] {
  const { onset, offset, frame, velocity, frames } = columns;

  const onsetBinary = new Uint8Array(frames);
  const onsetShift = new Float32Array(frames);
  const offsetBinary = new Uint8Array(frames);
  const offsetShift = new Float32Array(frames);

  binarise(onset, frames, ONSET_THRESHOLD, ONSET_NEIGHBOUR, onsetBinary, onsetShift);
  binarise(offset, frames, OFFSET_THRESHOLD, OFFSET_NEIGHBOUR, offsetBinary, offsetShift);

  const notes: DetectedNote[] = [];

  let bgn: number | null = null;
  let frameDisappear: number | null = null;
  let offsetOccur: number | null = null;

  for (let i = 0; i < frames; i++) {
    if (onsetBinary[i] === 1) {
      // Upstream writes `if bgn:`, which is false for frame 0, so every note
      // that starts on the very first frame is thrown away. The check here is
      // for null, so those notes are kept.
      if (bgn !== null) {
        // Consecutive onsets: the pedal was not released, but the same pitch
        // was struck again. The earlier note ends on the frame before.
        const fin = Math.max(i - 1, 0);
        notes.push({
          begin: bgn,
          end: fin,
          onsetShift: onsetShift[bgn],
          offsetShift: 0,
          velocity: velocity[bgn],
        });
        frameDisappear = null;
        offsetOccur = null;
      }
      bgn = i;
    }

    if (bgn !== null && i > bgn) {
      // The pitch has gone quiet.
      if (frame[i] <= FRAME_THRESHOLD && frameDisappear === null) {
        frameDisappear = i;
      }

      // The offset head says the note ended here.
      if (offsetBinary[i] === 1 && offsetOccur === null) {
        offsetOccur = i;
      }

      if (frameDisappear !== null) {
        // Two ways to end a note: the offset head, or the pitch falling
        // silent. Whichever the model is more sure about wins.
        let fin: number;
        if (offsetOccur !== null && offsetOccur - bgn > frameDisappear - offsetOccur) {
          fin = offsetOccur;
        } else {
          fin = frameDisappear;
        }
        notes.push({
          begin: bgn,
          end: fin,
          onsetShift: onsetShift[bgn],
          offsetShift: offsetShift[fin],
          velocity: velocity[bgn],
        });
        bgn = null;
        frameDisappear = null;
        offsetOccur = null;
      }

      if (bgn !== null && (i - bgn >= MAX_NOTE_FRAMES || i === frames - 1)) {
        // No end was found: close the note anyway.
        const fin = i;
        notes.push({
          begin: bgn,
          end: fin,
          onsetShift: onsetShift[bgn],
          offsetShift: offsetShift[fin],
          velocity: velocity[bgn],
        });
        bgn = null;
        frameDisappear = null;
        offsetOccur = null;
      }
    }
  }

  return notes;
}

/**
 * Turn a regression column into peaks.
 *
 * A frame counts as an onset or an offset when the regression is over the
 * threshold and the values around it fall away on both sides — that is, the
 * frame is the top of a hill rather than part of a slope.
 *
 * The shift is how far inside the frame the event really happened, worked out
 * from the two neighbours of the peak.
 */
function binarise(
  x: Float32Array,
  frames: number,
  threshold: number,
  neighbour: number,
  binary: Uint8Array,
  shift: Float32Array,
): void {
  for (let n = neighbour; n < frames - neighbour; n++) {
    // Upstream tests `x[n] > threshold`, so a value that is not a number
    // falls out here. Testing it the other way round would let a NaN
    // through as if it were a peak.
    if (!(x[n] > threshold)) continue;
    if (!isMonotonicNeighbour(x, n, neighbour)) continue;

    binary[n] = 1;

    const left = x[n - 1];
    const right = x[n + 1];

    // Upstream's two branches differ in the denominator: the peak is measured
    // against whichever neighbour is the larger one. The numerators are the
    // same in both branches.
    const denominator = left > right ? x[n] - right : x[n] - left;
    const numerator = right - left;

    // Upstream divides without guarding against a zero denominator, which can
    // happen when two frames have exactly the same value. A shift that is not
    // a number would poison the note's time, so it is left at zero.
    const value = denominator === 0 ? 0 : numerator / denominator / 2;
    shift[n] = Number.isFinite(value) ? value : 0;
  }
}

/**
 * Say whether the values around `n` fall away on both sides.
 *
 * Upstream's `is_monotonic_neighbour`. The window is `neighbour` frames to
 * either side of `n`, and every step away from the peak has to be flat or
 * falling.
 */
function isMonotonicNeighbour(x: Float32Array, n: number, neighbour: number): boolean {
  let monotonic = true;
  for (let i = 0; i < neighbour; i++) {
    if (x[n - i] < x[n - i - 1]) monotonic = false;
    if (x[n + i] < x[n + i + 1]) monotonic = false;
  }
  return monotonic;
}

/** Join a directory and a file name, for both Node and the browser. */
function joinPath(directory: string, name: string): string {
  if (directory.endsWith('/') || directory.endsWith('\\')) return directory + name;
  return `${directory}/${name}`;
}
