/**
 * The Basic Pitch model, and the note extraction built on top of it.
 *
 * Basic Pitch is a fixed-window analyser: it consumes 43 844 samples of mono
 * 22.05 kHz audio and returns 172 annotation frames of pitch state, onset
 * strength and pitch contour. To cover a whole file we slide that window
 * across the audio and stitch the frames back together.
 *
 * The pipeline runs the model twice over the same audio:
 *
 *   - the *upper* pass sees a high-passed copy, which removes the bass from
 *     the melody's way and avoids the model reporting a bass note an octave
 *     up;
 *   - the *bass* pass sees the whole mix but only the low register is kept
 *     from it.
 *
 * Everything here is deliberately free of DOM/Node specifics so the same code
 * runs in the browser and in the CLI.
 */

import * as tf from '@tensorflow/tfjs';

import {
  ANNOT_N_FRAMES,
  AUDIO_N_SAMPLES,
  HOP_SIZE,
  MIDI_OFFSET,
  N_OVERLAP_OVER_2,
  N_FREQ_BINS_CONTOURS,
  OVERLAP_LENGTH_FRAMES,
  SAMPLE_RATE,
} from './constants';
import { splitBands } from './dsp';
import { traceMelody, unmatchedMelody } from './melody';
import {
  EvaluateOptions,
  ModelLock,
  ModelOptions,
  ModelOutput,
  ModelPart,
  ModelSource,
  ModelTranscription,
  PITCH_BINS,
  TranscriptionModel,
  verifyModelFiles,
} from './model';
import { extractFrameNotes, sortNotes } from './notes';
import { cleanNotes, FrameNote, repairOctaveErrors, toTimedNotes } from './postprocess';
import { RawNote } from './types';

/** Output tensor names inside the graph model. */
const TENSOR_FRAMES = 'Identity_1';
const TENSOR_ONSETS = 'Identity_2';
const TENSOR_CONTOURS = 'Identity';

/** Crossover between the bass register and the upper register. */
const BASS_SPLIT_HZ = 200;

/** Highest MIDI note treated as bass. */
const BASS_MAX_PITCH = 54;

/** Lowest MIDI note treated as part of the upper register. */
const UPPER_MIN_PITCH = 55;

/** A note belongs to the melody if the traced line covers this much of it. */
const MELODY_OVERLAP_RATIO = 0.5;

/**
 * The TensorFlow.js graph model, and the windowing around it.
 *
 * Kept apart from the note extraction so the extraction can be tested
 * without loading any weights.
 */
export class BasicPitchGraph {
  private constructor(private readonly model: tf.GraphModel) {}

  /** Load the graph model. `modelUrl` may point at a local directory served by
   * the dev server, a file:// URL, or a remote host.
   *
   *  `fetch` is used to get the manifest and the weight shards. It is needed
   *  for a `file://` URL on a Node without the native TensorFlow.js backend,
   *  because Node's own fetch does not understand that scheme. */
  static async load(modelUrl: string, fetchFunc?: typeof fetch): Promise<BasicPitchGraph> {
    const model = await tf.loadGraphModel(modelUrl, { fetchFunc });
    return new BasicPitchGraph(model);
  }

  /**
   * Run the model over the whole clip.
   *
   * The audio is windowed with a small overlap on both sides so the model has
   * context around every analysis point; the frames produced from that
   * overlap are trimmed back off afterwards.
   *
   * The last window is padded with zeros rather than dropped, so the very
   * end of the recording is analysed too.
   */
  async evaluate(samples: Float32Array, options: EvaluateOptions = {}): Promise<ModelOutput> {
    const label = options.label ?? 'Transcribing';
    const onProgress = options.onProgress ?? (() => {});

    const lead = OVERLAP_LENGTH_FRAMES / 2;
    const paddedLength = lead + samples.length;
    if (paddedLength === 0) {
      return {
        frames: new Float32Array(0),
        onsets: new Float32Array(0),
        contours: new Float32Array(0),
        nFrames: 0,
      };
    }

    // Windows that fit entirely inside the signal.
    const fullWindows =
      paddedLength < AUDIO_N_SAMPLES
        ? 0
        : Math.floor((paddedLength - AUDIO_N_SAMPLES) / HOP_SIZE) + 1;

    // One more window, zero-padded, so the tail of the recording is covered.
    const nInputFrames = fullWindows + (fullWindows * HOP_SIZE < paddedLength ? 1 : 0);

    const framesPerWindow = ANNOT_N_FRAMES - 2 * N_OVERLAP_OVER_2;

    // How many frames are needed to cover the whole recording.
    //
    // A window advances the source by HOP_SIZE samples and contributes
    // framesPerWindow frames, so the frames do not sit a frame apart in time:
    // they cover HOP_SIZE / framesPerWindow samples each, which is a little
    // less than the model's 256-sample hop. Counting the frames at the model's
    // own 86 fps instead would end the output about one second early on a
    // three-minute song.
    const wanted = Math.ceil((samples.length * framesPerWindow) / HOP_SIZE);
    const totalOutFrames = Math.min(nInputFrames * framesPerWindow, Math.max(0, wanted));

    const frames = new Float32Array(totalOutFrames * PITCH_BINS);
    const onsets = new Float32Array(totalOutFrames * PITCH_BINS);
    const contours = new Float32Array(totalOutFrames * N_FREQ_BINS_CONTOURS);

    // Reused window buffer so we do not allocate per frame.
    const window = new Float32Array(AUDIO_N_SAMPLES);
    let outFrame = 0;

    for (let i = 0; i < nInputFrames && outFrame < totalOutFrames; i++) {
      const start = i * HOP_SIZE;
      window.fill(0);
      // Copy the window out of the padded signal.
      const available = Math.max(0, Math.min(AUDIO_N_SAMPLES, paddedLength - start));
      for (let j = 0; j < available; j++) {
        const src = start + j - lead;
        window[j] = src >= 0 && src < samples.length ? samples[src] : 0;
      }

      const input = tf.tensor3d(window, [1, AUDIO_N_SAMPLES, 1]);
      const result = this.model.execute(input, [
        TENSOR_FRAMES,
        TENSOR_ONSETS,
        TENSOR_CONTOURS,
      ]) as tf.Tensor[];
      const f = result[0] as tf.Tensor3D;
      const o = result[1] as tf.Tensor3D;
      const c = result[2] as tf.Tensor3D;

      const fData = await f.data();
      const oData = await o.data();
      const cData = await c.data();

      // Trim the overlap from both edges of the window.
      const rows = Math.min(framesPerWindow, totalOutFrames - outFrame);
      for (let r = 0; r < rows; r++) {
        const srcRow = r + N_OVERLAP_OVER_2;
        const dst = outFrame + r;
        frames.set(fData.subarray(srcRow * PITCH_BINS, (srcRow + 1) * PITCH_BINS), dst * PITCH_BINS);
        onsets.set(oData.subarray(srcRow * PITCH_BINS, (srcRow + 1) * PITCH_BINS), dst * PITCH_BINS);
        contours.set(
          cData.subarray(srcRow * N_FREQ_BINS_CONTOURS, (srcRow + 1) * N_FREQ_BINS_CONTOURS),
          dst * N_FREQ_BINS_CONTOURS,
        );
      }
      outFrame += rows;

      input.dispose();
      f.dispose();
      o.dispose();
      c.dispose();

      onProgress((i + 1) / nInputFrames, label);
    }

    return { frames, onsets, contours, nFrames: outFrame };
  }

  dispose(): void {
    this.model.dispose();
  }
}

/**
 * Turn the model's two passes into musical parts.
 *
 * The register filter is applied before and after the cleanup, because the
 * cleanup can add notes of its own.
 */
export async function transcribeBasicPitch(
  samples: Float32Array,
  graph: BasicPitchGraph,
  options: ModelOptions = {},
): Promise<ModelTranscription> {
  const report = (value: number, label: string) => options.onProgress?.(value, label);

  report(0.05, 'Separating registers');
  const { high } = splitBands(samples, BASS_SPLIT_HZ, SAMPLE_RATE);

  report(0.1, 'Transcribing upper notes');
  const upperOutput = await graph.evaluate(high, {
    label: 'Transcribing upper notes',
    onProgress: (value) => report(0.1 + value * 0.35, 'Transcribing upper notes'),
  });

  // The whole mix is used here and the register is filtered afterwards.
  // Low-passing first was tried and made the bass worse: the model relies on
  // the bass note's upper harmonics to place the fundamental, and removing
  // them introduced octave errors.
  report(0.46, 'Transcribing bass notes');
  const bassOutput = await graph.evaluate(samples, {
    label: 'Transcribing bass notes',
    onProgress: (value) => report(0.46 + value * 0.3, 'Transcribing bass notes'),
  });

  report(0.78, 'Reading notes');
  const inUpper = (note: FrameNote) => note.pitchIndex + MIDI_OFFSET >= UPPER_MIN_PITCH;
  const inBass = (note: FrameNote) => note.pitchIndex + MIDI_OFFSET <= BASS_MAX_PITCH;

  const upperFrameNotes = extractFrameNotes(upperOutput, {
    onsetThreshold: 0.5,
    frameThreshold: 0.3,
    minNoteLengthFrames: 4,
    melodiaTrick: false,
  }).filter(inUpper);

  const bassFrameNotes = extractFrameNotes(bassOutput, {
    onsetThreshold: 0.5,
    frameThreshold: 0.3,
    minNoteLengthFrames: 4,
    melodiaTrick: false,
  }).filter(inBass);

  report(0.83, 'Cleaning up notes');
  const cleanedUpper = cleanNotes(upperFrameNotes, upperOutput).filter(inUpper);
  const cleanedBass = repairOctaveErrors(cleanNotes(bassFrameNotes, bassOutput).filter(inBass));

  report(0.89, 'Following the melody');
  const trace = traceMelody(upperOutput);
  const melodyNotes = trace.notes;

  report(0.93, 'Arranging parts');
  const upperNotes = toTimedNotes(cleanedUpper);
  const bassNotes = toTimedNotes(cleanedBass);

  const parts = arrangeParts(upperNotes, bassNotes, melodyNotes);

  return { parts, log: [] };
}

/**
 * Split the notes into musical parts.
 *
 * Bass is anything in the low register. Above that, a note belongs to the
 * melody when the traced line runs along it, and to the chord part
 * otherwise.
 */
export function arrangeParts(
  upperNotes: RawNote[],
  bassNotes: RawNote[],
  melodyNotes: RawNote[],
): ModelPart[] {
  const melody: RawNote[] = [];
  const chords: RawNote[] = [];

  for (const note of upperNotes) {
    const supported = melodyNotes.some(
      (melodyNote) =>
        melodyNote.pitchMidi === note.pitchMidi &&
        overlapSeconds(melodyNote, note) >= note.duration * MELODY_OVERLAP_RATIO,
    );
    if (supported) melody.push(note);
    else chords.push(note);
  }

  // The tracer can find sustained notes that never had a detectable attack.
  // Those fill the gaps in the melody rather than forming a new part.
  const extra = unmatchedMelody(melodyNotes, upperNotes);
  melody.push(...extra);

  sortNotes(melody);
  sortNotes(chords);
  sortNotes(bassNotes);

  const parts: ModelPart[] = [];
  if (melody.length > 0) parts.push({ id: 'melody', name: 'Melody', kind: 'melody', notes: melody });
  if (chords.length > 0) parts.push({ id: 'chords', name: 'Chords', kind: 'chords', notes: chords });
  if (bassNotes.length > 0) parts.push({ id: 'bass', name: 'Bass', kind: 'bass', notes: bassNotes });
  return parts;
}

/** How long two notes overlap, in seconds. */
function overlapSeconds(a: RawNote, b: RawNote): number {
  const start = Math.max(a.startTime, b.startTime);
  const end = Math.min(a.startTime + a.duration, b.startTime + b.duration);
  return Math.max(0, end - start);
}

/**
 * Load the pinned model, checking every file against the lock file.
 *
 * A mismatch means the weights on disk are not the ones the pipeline was
 * tuned against, so this refuses to run rather than quietly producing
 * different output.
 */
export async function loadPinnedModel(
  source: ModelSource,
  lock: ModelLock,
): Promise<BasicPitchModel> {
  const digest = await verifyModelFiles(source, lock);
  const graph = await BasicPitchGraph.load(source.url, source.fetch);
  return new BasicPitchModel(graph, digest);
}

/**
 * Basic Pitch, as a model the pipeline can use.
 *
 * The weights are checked against the lock file before anything runs. If they
 * do not match, the output would differ from every previous run, so it is
 * better to stop than to produce a different file quietly.
 */
export class BasicPitchModel implements TranscriptionModel {
  readonly id = 'basic-pitch-icassp2022';
  readonly displayName = 'Basic Pitch (Spotify)';
  readonly license = 'Apache-2.0';
  readonly inputSampleRate = SAMPLE_RATE;

  constructor(
    private readonly graph: BasicPitchGraph,
    /** Hash of the weights this instance was loaded from. */
    public readonly fingerprint: string = 'unknown',
  ) {}

  /** Record the weights' fingerprint on an existing instance. */
  withFingerprint(fingerprint: string): BasicPitchModel {
    return new BasicPitchModel(this.graph, fingerprint);
  }

  async transcribe(samples: Float32Array, options?: ModelOptions): Promise<ModelTranscription> {
    return transcribeBasicPitch(samples, this.graph, options);
  }

  dispose(): void {
    this.graph.dispose();
  }
}

/** The sample rate every part of the pipeline works at. */
export { SAMPLE_RATE };
