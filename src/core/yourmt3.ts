/**
 * The YourMT3+ model.
 *
 * YourMT3+ is a multi-instrument audio-to-MIDI model by Sungkyun Chang,
 * Emmanouel Benetos, Holger Kirchhoff and Simon Dixon — "YourMT3+: Multi-instrument
 * Music Transcription with Enhanced Transformer Architectures and Cross-dataset
 * Stem Augmentation", MLSP 2024, arXiv:2407.04822. The weights are the
 * authors' own, released under Apache-2.0; the authors confirmed that licence
 * in issue #12 and issue #22 of the project's GitHub repository.
 *
 *   **Licence: Apache-2.0 for the weights.**
 *
 *   The training mix includes MAESTRO, which is CC BY-NC-SA 4.0, and which the
 *   authors state they cannot license on the dataset owners' behalf. Whether
 *   the model may be used commercially is therefore the user's own assessment,
 *   not something this file can settle.
 *
 * What is here is an *unofficial* ONNX export of those weights, made by
 * somebody else. It is not an official release of the model.
 *
 * The model works in three stages, and only the middle one is a graph:
 *
 *   1. The audio is cut into non-overlapping segments of 32767 samples at
 *      16 kHz — about 2.05 seconds each — and each segment becomes a log
 *      spectrogram of 110 frames by 1024 bins. This happens here, in
 *      double precision, because the encoder's mixture-of-experts routing
 *      switches experts when the input differs in the last bits, and a
 *      float32 transform would change the notes.
 *   2. The encoder turns the spectrogram into the cross-attention keys and
 *      values of eight decoder layers, one row per instrument channel.
 *   3. The decoder is then run one token at a time. Thirteen channels are
 *      decoded in parallel, greedily, and the greedy loop lives outside the
 *      graph: the graph is a single step, with the self-attention cache
 *      passed in and out.
 *
 * The tokens are a small event language — `shift`, `pitch`, `velocity`,
 * `program`, `drum`, `tie` — and turning them into notes is a direct port of
 * the upstream `event2note.py`. The rules are not obvious, so they are
 * spelled out where they are implemented.
 *
 * The model is large — 181 MB of weights — and slow on a CPU: roughly twice
 * as long as the audio it is transcribing. It reports progress as it goes.
 */

import { ModelOptions, ModelPart, ModelTranscription, TranscriptionModel } from './model';
import { OnnxRunner, OnnxSession, OnnxTensor, asFloat32, asInt64 } from './onnx';
import { RawNote, TrackKind } from './types';

/** The directory the weights live in, under the ONNX model root. */
export const YOURMT3_DIR = 'yourmt3';

/** The files of the pinned weights. */
export const YOURMT3_FILES = {
  encoder: 'encoder.onnx',
  decoder: 'decoder_step.onnx',
  vocab: 'vocab.json',
} as const;

/** The front end wants 16 kHz mono. */
export const YOURMT3_SAMPLE_RATE = 16000;

/** Samples per segment: 32767, which is 2.048 seconds at 16 kHz. */
const SEGMENT_SAMPLES = 32767;

/** Instrument channels decoded in parallel. */
const CHANNELS = 13;

/** The longest token sequence a channel may produce. */
const MAX_TOKENS = 256;

/** Decoder layers, and so the number of key and value caches. */
const LAYERS = 8;

/** Attention heads per layer. */
const HEADS = 6;

/** The width of one head. */
const D_KV = 64;

/** Frames in the spectrogram of one segment. */
const ENC_FRAMES = 110;

/** Transform size of the front end's FFT. */
const N_FFT = 2048;

/** Hop of the front end's FFT, in samples. */
const HOP = 300;

/** Reflect padding either side of a segment, in samples. */
const PAD = 1024;

/** Bins in the spectrogram of one frame: the DC bin is dropped. */
const FRAME_BINS = N_FFT / 2;

/** Ticks per second in the model's timebase. One tick is 10 ms. */
const TPS = 100;

/** The token that ends a channel's sequence. */
const EOS = 1;

/** The shortest note the model will report, in seconds. */
const MINIMUM_OFFSET_SEC = 0.01;

/** The longest note the model will report, in seconds. */
const MAXIMUM_NOTE_SEC = 10;

/** The velocity written into the MIDI file. The model itself only says 0 or 1. */
const MIDI_VELOCITY = 0.8;

/** The starting size of the self-attention cache. It doubles when it fills. */
const CACHE_INITIAL = 64;

/** One event of the model's token language. */
export interface VocabEvent {
  /** `PAD`, `EOS`, `UNK`, `shift`, `pitch`, `velocity`, `tie`, `program` or `drum`. */
  type: string;
  /** The event's value: a tick count, a pitch, a velocity, a program, a drum. */
  value: number;
}

/** What a model program becomes in the output. */
export interface ProgramInfo {
  /** The output MIDI program number. */
  program: number;
  /** The output instrument name. */
  name: string;
}

/**
 * The model's vocabulary, read from `vocab.json` when it is loaded.
 *
 * The event table is a flat list: the index of an entry is the token id, so
 * the offsets are not hard-coded here.
 */
export interface Vocab {
  /** The name of the checkpoint this export came from. */
  model: string;
  /** The sample rate the model wants. */
  sampleRate: number;
  /** Samples per segment. */
  segment: number;
  /** Number of instrument channels. */
  channels: number;
  /** The longest token sequence a channel may produce. */
  maxTokens: number;
  /** Number of decoder layers. */
  layers: number;
  /** Attention heads per layer. */
  heads: number;
  /** The width of one head. */
  dKv: number;
  /** Frames in the spectrogram of one segment. */
  encFrames: number;
  /** The number of distinct tokens. */
  vocabSize: number;
  /** The token that ends a sequence. */
  eos: number;
  /** The token that pads a sequence. */
  pad: number;
  /** Ticks per second. */
  tps: number;
  /** Token id to event. The index in this list is the token id. */
  events: VocabEvent[];
  /** Model program to output program and name. */
  programs: Map<number, ProgramInfo>;
}

/** A note event: an onset or an offset, before notes are assembled. */
interface NoteEvent {
  /** Drums have no program of their own; they are always 128. */
  isDrum: boolean;
  /** The model program. */
  program: number;
  /** The time in seconds, or null for the tied notes of the preamble. */
  time: number | null;
  /** 1 for an onset, 0 for an offset. */
  velocity: number;
  /** The MIDI note number. */
  pitch: number;
}

/** A note, once the note events have been assembled into one. */
interface Note {
  isDrum: boolean;
  program: number;
  /** The start of the note, in seconds. */
  onset: number;
  /** The end of the note, in seconds. */
  offset: number;
  pitch: number;
  velocity: number;
}

/** A model program and a pitch, which is what the tie preamble lists. */
type ProgramPitch = [number, number];

/** What one segment's tokens decode into. */
interface SegmentEvents {
  /** The note events of the segment. */
  noteEvents: NoteEvent[];
  /** The notes tied over from the previous segment. */
  tieNoteEvents: NoteEvent[];
  /** The programs and pitches still sounding at the end of the segment. */
  lastActivity: ProgramPitch[];
}

/**
 * YourMT3+, as a model the pipeline can use.
 */
export class YourMt3Model implements TranscriptionModel {
  readonly id = 'yourmt3-plus-mimbres-2024';
  readonly displayName = 'YourMT3+';
  readonly license = 'Apache-2.0';
  readonly inputSampleRate = YOURMT3_SAMPLE_RATE;

  constructor(
    private readonly encoder: OnnxSession,
    private readonly decoder: OnnxSession,
    private readonly vocab: Vocab,
    /** Hash of the weights this instance was loaded from. */
    public readonly fingerprint: string = 'unknown',
  ) {}

  /**
   * Load the two graphs and the vocabulary.
   *
   * `path` is the directory the weights live in. The caller is responsible
   * for checking them against the lock file first.
   */
  static async load(
    runner: OnnxRunner,
    path: string,
    fingerprint = 'unknown',
  ): Promise<YourMt3Model> {
    // The vocabulary sits next to the graphs and is not in them, so the
    // runtime has to be able to read files. Both the browser and Node can.
    const read = runner.read;
    if (!read) {
      throw new Error(
        'This runtime cannot read the files next to the model, so the YourMT3 ' +
          'vocabulary cannot be loaded.',
      );
    }
    const bytes = await read.call(runner, joinPath(path, YOURMT3_FILES.vocab));
    const vocab = parseVocab(JSON.parse(new TextDecoder().decode(bytes)) as unknown);

    const encoder = await runner.createSession(joinPath(path, YOURMT3_FILES.encoder));
    const decoder = await runner.createSession(joinPath(path, YOURMT3_FILES.decoder));
    return new YourMt3Model(encoder, decoder, vocab, fingerprint);
  }

  async transcribe(samples: Float32Array, options: ModelOptions = {}): Promise<ModelTranscription> {
    const report = (value: number, label: string) => options.onProgress?.(value, label);

    report(0.02, 'Slicing the audio');
    const segments = sliceSegments(samples);
    const window = hannWindow(N_FFT);
    const fft = new DoubleFft(N_FFT);

    // The tokens of every channel, segment by segment.
    const tokensByChannel: number[][][] = [];
    for (let channel = 0; channel < CHANNELS; channel++) tokensByChannel.push([]);

    for (let s = 0; s < segments.length; s++) {
      report(0.05 + (0.8 * s) / segments.length, 'Analysing the spectrum');
      const spec = segmentSpectrogram(segments[s], window, fft);

      report(0.05 + (0.8 * s) / segments.length, 'Reading notes');
      const rows = await this.decodeSegment(spec);
      for (let row = 0; row < rows.length && row < CHANNELS; row++) {
        tokensByChannel[row].push(rows[row]);
      }

      report(0.05 + (0.8 * (s + 1)) / segments.length, 'Reading notes');
    }

    report(0.9, 'Building the tracks');
    const parts = this.buildParts(tokensByChannel, segments.length);

    return {
      parts,
      log: [
        `YourMT3+ found ${parts.reduce((sum, part) => sum + part.notes.length, 0)} notes ` +
          `in ${parts.length} parts.`,
      ],
    };
  }

  /**
   * Run the encoder and then the decoder over one segment.
   *
   * The encoder returns the cross-attention keys and values of every layer,
   * already expanded to one row per channel. The decoder then runs one step
   * at a time over all thirteen rows at once, which is what makes the whole
   * thing affordable: a step costs about as much as one row does, so the
   * channels are nearly free.
   */
  private async decodeSegment(spec: Float32Array): Promise<number[][]> {
    const encoderOutput = await this.encoder.run({
      spec: { type: 'float32', data: spec, dims: [1, ENC_FRAMES, FRAME_BINS] },
    });

    const ck: OnnxTensor[] = [];
    const cv: OnnxTensor[] = [];
    for (let layer = 0; layer < LAYERS; layer++) {
      const keys = encoderOutput[`ck_${layer}`];
      const values = encoderOutput[`cv_${layer}`];
      if (!keys || !values) {
        throw new Error(`The YourMT3 encoder did not return the layer ${layer} caches.`);
      }
      ck.push(keys);
      cv.push(values);
    }

    // One row per instrument channel.
    const rows = ck[0].dims[0];
    if (rows !== CHANNELS) {
      throw new Error(
        `The YourMT3 encoder returned ${rows} rows, but ${CHANNELS} instrument ` +
          'channels were expected.',
      );
    }

    // The self-attention cache. It starts at 64 slots and doubles when the
    // decoder needs more room than that.
    let capacity = CACHE_INITIAL;
    let keys: Float32Array[] = [];
    let values: Float32Array[] = [];
    for (let layer = 0; layer < LAYERS; layer++) {
      keys.push(new Float32Array(rows * HEADS * D_KV * capacity));
      values.push(new Float32Array(rows * HEADS * capacity * D_KV));
    }

    // The token each row is fed at the next step. It starts as the
    // beginning-of-sequence marker.
    let ids: BigInt64Array = new BigInt64Array(rows);
    const alive = Array.from({ length: rows }, (_, index) => index);
    const tokens: number[][] = Array.from({ length: rows }, () => []);
    // The number of rows still in the batch. It shrinks when a channel ends.
    let live = rows;

    for (let pos = 0; pos < MAX_TOKENS; pos++) {
      if (pos >= capacity) {
        capacity *= 2;
        for (let layer = 0; layer < LAYERS; layer++) {
          keys[layer] = growKeyCache(keys[layer], live, capacity / 2, capacity);
          values[layer] = growValueCache(values[layer], live, capacity / 2, capacity);
        }
      }

      const feeds: Record<string, OnnxTensor> = {
        ids: { type: 'int64', data: ids, dims: [live] },
        pos: { type: 'int64', data: new BigInt64Array([BigInt(pos)]), dims: [1] },
      };
      for (let layer = 0; layer < LAYERS; layer++) {
        feeds[`pk_${layer}`] = {
          type: 'float32',
          data: keys[layer],
          dims: [live, HEADS, D_KV, capacity],
        };
        feeds[`pv_${layer}`] = {
          type: 'float32',
          data: values[layer],
          dims: [live, HEADS, capacity, D_KV],
        };
        feeds[`ck_${layer}`] = ck[layer];
        feeds[`cv_${layer}`] = cv[layer];
      }

      const output = await this.decoder.run(feeds);

      const nextId = asInt64(output.next_id);
      for (let layer = 0; layer < LAYERS; layer++) {
        const newKeys = output[`nk_${layer}`];
        const newValues = output[`nv_${layer}`];
        if (!newKeys || !newValues) {
          throw new Error(`The YourMT3 decoder did not return the layer ${layer} caches.`);
        }
        writeKeyCache(keys[layer], asFloat32(newKeys), live, capacity, pos);
        writeValueCache(values[layer], asFloat32(newValues), live, capacity, pos);
      }

      // Collect the tokens. The first step is the one after the
      // beginning-of-sequence marker, and an end-of-sequence token there is
      // not an ending — upstream does not treat it as one, and neither do we.
      const finished: boolean[] = [];
      for (let j = 0; j < alive.length; j++) {
        const id = Number(nextId[j]);
        tokens[alive[j]].push(id);
        finished.push(pos >= 1 && id === EOS);
      }

      if (finished.every(Boolean)) break;

      const next = new BigInt64Array(alive.length);
      for (let j = 0; j < alive.length; j++) next[j] = nextId[j];
      ids = next;

      // A channel that has finished is dropped from the batch. Upstream does
      // the same: the dropped row's output was thrown away anyway, so the
      // rows that survive are not affected by the dropping.
      if (finished.some(Boolean)) {
        const keep = finished.map((done, index) => (done ? -1 : index)).filter((index) => index >= 0);
        const kept = keep.map((index) => alive[index]);
        for (let layer = 0; layer < LAYERS; layer++) {
          keys[layer] = selectRows(keys[layer], keep, HEADS * D_KV * capacity);
          values[layer] = selectRows(values[layer], keep, HEADS * capacity * D_KV);
          ck[layer] = selectTensorRows(ck[layer], keep);
          cv[layer] = selectTensorRows(cv[layer], keep);
        }
        ids = selectInt64Rows(ids, keep);
        alive.length = 0;
        alive.push(...kept);
        live = kept.length;
      }
    }

    return tokens;
  }

  /**
   * Turn the tokens into the parts the pipeline shows.
   *
   * Every channel is decoded on its own and then the channels are mixed,
   * which is what upstream does: the channels are separate instrument
   * groups, and a note that starts in one segment and ends in the next is
   * carried across the boundary by the tie preamble.
   */
  private buildParts(tokensByChannel: number[][][], segmentCount: number): ModelPart[] {
    const startTimes: number[] = [];
    for (let s = 0; s < segmentCount; s++) {
      startTimes.push((s * SEGMENT_SAMPLES) / YOURMT3_SAMPLE_RATE);
    }

    const perChannel: Note[][] = [];
    for (let channel = 0; channel < CHANNELS; channel++) {
      const segments: SegmentEvents[] = [];
      for (let s = 0; s < segmentCount; s++) {
        const tokens = tokensByChannel[channel][s] ?? [];
        const events = tokens.map((id) => this.vocab.events[id] ?? { type: 'UNK', value: 0 });
        segments.push(eventToNoteEvents(events, startTimes[s], this.vocab.tps));
      }
      perChannel.push(mergeSegments(segments, startTimes).notes);
    }

    return groupIntoParts(mixNotes(perChannel), this.vocab.programs);
  }
}

/**
 * Read the vocabulary.
 *
 * The event table is a flat list whose index is the token id, and the
 * program table is keyed by the model's own program numbers. Both are read
 * at run time; nothing here is hard-coded.
 */
function parseVocab(raw: unknown): Vocab {
  if (typeof raw !== 'object' || raw === null) {
    throw new Error('The YourMT3 vocabulary is not a JSON object.');
  }
  const json = raw as Record<string, unknown>;

  if (!Array.isArray(json.events)) {
    throw new Error('The YourMT3 vocabulary has no event table.');
  }
  const events: VocabEvent[] = [];
  for (const entry of json.events) {
    if (!Array.isArray(entry) || typeof entry[0] !== 'string' || typeof entry[1] !== 'number') {
      throw new Error('The YourMT3 event table is malformed.');
    }
    events.push({ type: entry[0], value: entry[1] });
  }

  if (typeof json.programs !== 'object' || json.programs === null) {
    throw new Error('The YourMT3 vocabulary has no program table.');
  }
  const programs = new Map<number, ProgramInfo>();
  for (const [key, value] of Object.entries(json.programs as Record<string, unknown>)) {
    if (!Array.isArray(value) || typeof value[0] !== 'number' || typeof value[1] !== 'string') {
      throw new Error('The YourMT3 program table is malformed.');
    }
    programs.set(Number(key), { program: value[0], name: value[1] });
  }

  const vocab: Vocab = {
    model: typeof json.model === 'string' ? json.model : 'unknown',
    sampleRate: numberOr(json.sample_rate, YOURMT3_SAMPLE_RATE),
    segment: numberOr(json.segment, SEGMENT_SAMPLES),
    channels: numberOr(json.channels, CHANNELS),
    maxTokens: numberOr(json.max_tokens, MAX_TOKENS),
    layers: numberOr(json.layers, LAYERS),
    heads: numberOr(json.heads, HEADS),
    dKv: numberOr(json.d_kv, D_KV),
    encFrames: numberOr(json.enc_frames, ENC_FRAMES),
    vocabSize: numberOr(json.vocab_size, events.length),
    eos: numberOr(json.eos, EOS),
    pad: numberOr(json.pad, 0),
    tps: numberOr(json.tps, TPS),
    events,
    programs,
  };

  // The exported graphs were built for one set of shapes. A vocabulary that
  // says otherwise belongs to a different export, and running it would give
  // quietly wrong notes rather than an error.
  expect(vocab.segment, SEGMENT_SAMPLES, 'the segment length');
  expect(vocab.channels, CHANNELS, 'the number of channels');
  expect(vocab.layers, LAYERS, 'the number of layers');
  expect(vocab.heads, HEADS, 'the number of heads');
  expect(vocab.dKv, D_KV, 'the head width');
  expect(vocab.encFrames, ENC_FRAMES, 'the number of frames');
  expect(vocab.vocabSize, events.length, 'the size of the vocabulary');
  expect(vocab.eos, EOS, 'the end-of-sequence token');
  expect(vocab.tps, TPS, 'the tick rate');

  return vocab;
}

/** A number from the vocabulary, or the default. */
function numberOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** Complain if the vocabulary does not match the exported graphs. */
function expect(actual: number, wanted: number, what: string): void {
  if (actual !== wanted) {
    throw new Error(
      `The YourMT3 vocabulary says ${what} is ${actual}, but the exported graphs ` +
        `were built for ${wanted}. These weights do not belong together.`,
    );
  }
}

/**
 * Cut the audio into the segments the model works on.
 *
 * The segments do not overlap, and the last one is padded with silence.
 * There is always at least one segment, so an empty recording still goes
 * through the model.
 */
function sliceSegments(samples: Float32Array): Float32Array[] {
  const count = Math.max(1, Math.ceil(samples.length / SEGMENT_SAMPLES));
  const segments: Float32Array[] = [];
  for (let s = 0; s < count; s++) {
    const slice = samples.subarray(s * SEGMENT_SAMPLES, (s + 1) * SEGMENT_SAMPLES);
    if (slice.length === SEGMENT_SAMPLES) {
      segments.push(slice);
      continue;
    }
    const padded = new Float32Array(SEGMENT_SAMPLES);
    padded.set(slice);
    segments.push(padded);
  }
  return segments;
}

/** The periodic Hann window the front end uses. */
function hannWindow(size: number): Float64Array {
  const window = new Float64Array(size);
  for (let n = 0; n < size; n++) {
    window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / size);
  }
  return window;
}

/**
 * The log spectrogram of one segment: `ENC_FRAMES` frames of `FRAME_BINS`
 * bins, as float32.
 *
 * This is a port of the upstream `spectrogram_np`: the segment is padded by
 * reflection, cut into frames of 2048 samples every 300, transformed,
 * turned into a logarithm, and normalised by the segment's own range.
 *
 * The transform is in double precision. The model's mixture-of-experts
 * routing is sensitive to the last bits of its input, so a float32
 * transform would change which notes come out.
 */
function segmentSpectrogram(
  segment: Float32Array,
  window: Float64Array,
  fft: DoubleFft,
): Float32Array {
  const length = segment.length;
  const paddedLength = length + 2 * PAD;
  const padded = new Float64Array(paddedLength);

  // Reflection padding, numpy's `mode="reflect"`. The left side mirrors the
  // beginning of the segment about its first sample, and the right side
  // mirrors the end about the gap before its last sample. Neither side
  // repeats the sample it is reflecting about.
  for (let i = 0; i < PAD; i++) padded[i] = segment[PAD - i];
  for (let i = 0; i < length; i++) padded[PAD + i] = segment[i];
  for (let i = 0; i < PAD; i++) padded[PAD + length + i] = segment[length - 2 - i];

  const frames = 1 + Math.floor((paddedLength - N_FFT) / HOP);
  const frame = new Float64Array(N_FFT);
  const magnitudes = new Float64Array(FRAME_BINS + 1);
  const log = new Float64Array(frames * FRAME_BINS);

  let lowest = Infinity;
  let highest = -Infinity;

  for (let f = 0; f < frames; f++) {
    const offset = f * HOP;
    for (let n = 0; n < N_FFT; n++) frame[n] = padded[offset + n] * window[n];
    fft.magnitudes(frame, magnitudes);

    // The DC bin is dropped, so there are 1024 bins rather than 1025.
    for (let k = 1; k <= FRAME_BINS; k++) {
      const value = Math.log(magnitudes[k] + 1e-5);
      log[f * FRAME_BINS + (k - 1)] = value;
      if (value < lowest) lowest = value;
      if (value > highest) highest = value;
    }
  }

  // Each segment is normalised by its own range.
  const spec = new Float32Array(frames * FRAME_BINS);
  const range = highest - lowest + 0.008;
  for (let i = 0; i < log.length; i++) {
    spec[i] = (log[i] - lowest) / range;
  }
  return spec;
}

/**
 * A radix-2 complex FFT over doubles, with the twiddles worked out once.
 *
 * The project has an FFT in `dsp.ts`, but it works on float32 buffers, and
 * this one has to be in double precision, so it is written out again here.
 */
class DoubleFft {
  private readonly re: Float64Array;
  private readonly im: Float64Array;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly reversed: Uint32Array;

  constructor(private readonly size: number) {
    this.re = new Float64Array(size);
    this.im = new Float64Array(size);

    // The twiddle table holds one turn of the circle, so a stage of length
    // `len` reads it with a stride of `size / len`.
    this.cos = new Float64Array(size / 2);
    this.sin = new Float64Array(size / 2);
    for (let k = 0; k < size / 2; k++) {
      const angle = (-2 * Math.PI * k) / size;
      this.cos[k] = Math.cos(angle);
      this.sin[k] = Math.sin(angle);
    }

    // The bit-reversal permutation, worked out once.
    const bits = Math.round(Math.log2(size));
    this.reversed = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      let value = 0;
      for (let bit = 0; bit < bits; bit++) {
        if (i & (1 << bit)) value |= 1 << (bits - 1 - bit);
      }
      this.reversed[i] = value;
    }
  }

  /**
   * Transform a real frame and write the magnitudes of its bins into `out`,
   * which must hold `size / 2 + 1` values.
   */
  magnitudes(frame: Float64Array, out: Float64Array): void {
    const { re, im, cos, sin, reversed, size } = this;
    re.set(frame);
    im.fill(0);

    for (let i = 0; i < size; i++) {
      const j = reversed[i];
      if (j > i) {
        const tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        const ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }

    for (let len = 2; len <= size; len <<= 1) {
      const half = len >> 1;
      const stride = size / len;
      for (let i = 0; i < size; i += len) {
        for (let k = 0; k < half; k++) {
          const t = k * stride;
          const wr = cos[t];
          const wi = sin[t];
          const a = i + k;
          const b = a + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }

    for (let k = 0; k <= size / 2; k++) {
      out[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    }
  }
}

/**
 * Double the key cache, keeping what was in it.
 *
 * The layout is `[row, head, width, slot]`, so growing the last axis means
 * copying each block of slots into a wider one.
 */
function growKeyCache(
  source: Float32Array,
  rows: number,
  was: number,
  now: number,
): Float32Array {
  const grown = new Float32Array(rows * HEADS * D_KV * now);
  for (let row = 0; row < rows; row++) {
    for (let head = 0; head < HEADS; head++) {
      for (let d = 0; d < D_KV; d++) {
        const from = ((row * HEADS + head) * D_KV + d) * was;
        const to = ((row * HEADS + head) * D_KV + d) * now;
        grown.set(source.subarray(from, from + was), to);
      }
    }
  }
  return grown;
}

/**
 * Double the value cache, keeping what was in it.
 *
 * The layout is `[row, head, slot, width]`, so growing the third axis means
 * copying each block of slots into a taller one.
 */
function growValueCache(
  source: Float32Array,
  rows: number,
  was: number,
  now: number,
): Float32Array {
  const grown = new Float32Array(rows * HEADS * now * D_KV);
  for (let row = 0; row < rows; row++) {
    for (let head = 0; head < HEADS; head++) {
      const from = (row * HEADS + head) * was * D_KV;
      const to = (row * HEADS + head) * now * D_KV;
      grown.set(source.subarray(from, from + was * D_KV), to);
    }
  }
  return grown;
}

/** Write the step's new keys into the cache, at position `pos`. */
function writeKeyCache(
  cache: Float32Array,
  values: Float32Array,
  rows: number,
  capacity: number,
  pos: number,
): void {
  for (let row = 0; row < rows; row++) {
    for (let head = 0; head < HEADS; head++) {
      const source = (row * HEADS + head) * D_KV;
      for (let d = 0; d < D_KV; d++) {
        cache[((row * HEADS + head) * D_KV + d) * capacity + pos] = values[source + d];
      }
    }
  }
}

/** Write the step's new values into the cache, at position `pos`. */
function writeValueCache(
  cache: Float32Array,
  values: Float32Array,
  rows: number,
  capacity: number,
  pos: number,
): void {
  for (let row = 0; row < rows; row++) {
    for (let head = 0; head < HEADS; head++) {
      const source = (row * HEADS + head) * D_KV;
      const target = (row * HEADS + head) * capacity * D_KV + pos * D_KV;
      for (let d = 0; d < D_KV; d++) cache[target + d] = values[source + d];
    }
  }
}

/** Keep only the given rows of a flat cache. */
function selectRows(
  source: Float32Array,
  keep: number[],
  perRow: number,
): Float32Array {
  const out = new Float32Array(keep.length * perRow);
  for (let i = 0; i < keep.length; i++) {
    out.set(source.subarray(keep[i] * perRow, (keep[i] + 1) * perRow), i * perRow);
  }
  return out;
}

/** Keep only the given rows of a tensor. */
function selectTensorRows(tensor: OnnxTensor, keep: number[]): OnnxTensor {
  const rows = tensor.dims[0];
  const perRow = tensor.data.length / rows;
  const out = new Float32Array(keep.length * perRow);
  for (let i = 0; i < keep.length; i++) {
    const start = keep[i] * perRow;
    out.set((tensor.data as Float32Array).subarray(start, start + perRow), i * perRow);
  }
  return { type: 'float32', data: out, dims: [keep.length, ...tensor.dims.slice(1)] };
}

/** Keep only the given rows of a list of token ids. */
function selectInt64Rows(source: BigInt64Array, keep: number[]): BigInt64Array {
  const out = new BigInt64Array(keep.length);
  for (let i = 0; i < keep.length; i++) out[i] = source[keep[i]];
  return out;
}

/**
 * Turn one segment's events into note events.
 *
 * A port of upstream `event2note_event`. The rules are:
 *
 *   - The sequence starts with a preamble that says which notes are tied over
 *     from the previous segment. The preamble ends at the `tie` token. If
 *     there is no `tie` token the segment is discarded, because there is no
 *     way to tell what was still sounding.
 *   - `shift v` moves the clock to `v` ticks after the start of the segment.
 *     It is absolute, not cumulative: two shifts do not add up.
 *   - `velocity 1` is a note-on and `velocity 0` is a note-off. The paper
 *     has this the other way round; the released code is what counts, and it
 *     is the code that produced the weights' training targets.
 *   - A `drum` event is a hit, and has no offset.
 */
function eventToNoteEvents(events: VocabEvent[], startTime: number, tps: number): SegmentEvents {
  const noteEvents: NoteEvent[] = [];
  const tieNoteEvents: NoteEvent[] = [];
  const lastActivity: ProgramPitch[] = [];

  // The preamble. It ends at the first `tie`, or at the first `shift` — in
  // which case there is no tie and the segment is dropped.
  let tieIndex = -1;
  let programState: number | null = null;
  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    if (event.type === 'tie') {
      tieIndex = i;
      break;
    }
    if (event.type === 'shift') break;
    if (event.type === 'program') {
      programState = event.value;
    } else if (event.type === 'pitch') {
      if (programState === null) continue;
      tieNoteEvents.push({
        isDrum: false,
        program: programState,
        time: null,
        velocity: 1,
        pitch: event.value,
      });
      lastActivity.push([programState, event.value]);
    }
  }

  if (tieIndex < 0) {
    // No tie: the segment cannot be placed, so it is thrown away.
    return { noteEvents: [], tieNoteEvents: [], lastActivity: [] };
  }

  const startTick = Math.round(startTime * tps);
  let tickState = startTick;
  let velocityState: number | null = null;

  for (let i = tieIndex + 1; i < events.length; i++) {
    const event = events[i];
    switch (event.type) {
      case 'shift':
        if (event.value <= 0 || event.value > 1000) break;
        tickState = startTick + event.value;
        break;
      case 'drum':
        noteEvents.push({
          isDrum: true,
          program: 128,
          time: tickState / tps,
          velocity: 1,
          pitch: event.value,
        });
        break;
      case 'program':
        programState = event.value;
        break;
      case 'velocity':
        velocityState = event.value;
        break;
      case 'pitch': {
        if (programState === null || velocityState === null) break;
        if (velocityState > 0) {
          lastActivity.push([programState, event.value]);
        } else {
          const index = lastActivity.findIndex(
            (entry) => entry[0] === programState && entry[1] === event.value,
          );
          if (index < 0) break;
          lastActivity.splice(index, 1);
        }
        noteEvents.push({
          isDrum: false,
          program: programState,
          time: tickState / tps,
          velocity: velocityState,
          pitch: event.value,
        });
        break;
      }
      case 'EOS':
        i = events.length;
        break;
      case 'tie':
        // A second tie is an error upstream; the event is skipped.
        break;
      case 'PAD':
      case 'UNK':
        break;
      default:
        break;
    }
  }

  noteEvents.sort(
    (a, b) =>
      (a.time ?? 0) - (b.time ?? 0) ||
      Number(a.isDrum) - Number(b.isDrum) ||
      a.program - b.program ||
      a.velocity - b.velocity ||
      a.pitch - b.pitch,
  );
  tieNoteEvents.sort(
    (a, b) =>
      Number(a.isDrum) - Number(b.isDrum) || a.program - b.program || a.pitch - b.pitch,
  );

  return { noteEvents, tieNoteEvents, lastActivity };
}

/**
 * Turn note events into notes.
 *
 * A port of upstream `note_event2note`. Active notes are kept by pitch and
 * program. A note-on closes whatever was sounding on that pitch and opens a
 * new note; a note-off closes it. A drum hit is a note of the minimum
 * length. A note still sounding at the end gets the minimum length too.
 */
function noteEventsToNotes(
  noteEvents: NoteEvent[],
  tieNoteEvents: NoteEvent[] | null,
): Note[] {
  const notes: Note[] = [];
  const active = new Map<string, NoteEvent>();

  if (tieNoteEvents) {
    for (const event of tieNoteEvents) {
      active.set(`${event.pitch}|${event.program}`, event);
    }
  }

  // Velocity comes before program here, so a note-off at the same instant
  // and the same pitch is handled before the note-on that follows it.
  const sorted = [...noteEvents].sort(
    (a, b) =>
      (a.time ?? 0) - (b.time ?? 0) ||
      Number(a.isDrum) - Number(b.isDrum) ||
      a.pitch - b.pitch ||
      a.velocity - b.velocity ||
      a.program - b.program,
  );

  for (const event of sorted) {
    if (event.time === null) continue;

    if (event.isDrum) {
      if (event.velocity === 1) {
        notes.push({
          isDrum: true,
          program: 128,
          onset: event.time,
          offset: event.time + MINIMUM_OFFSET_SEC,
          pitch: event.pitch,
          velocity: 1,
        });
      }
      continue;
    }

    const key = `${event.pitch}|${event.program}`;
    if (event.velocity === 1) {
      const open = active.get(key);
      if (open) {
        active.delete(key);
        // A tied note has no onset of its own. Upstream builds a note with
        // no onset and then drops it, which is the same as not building it.
        if (open.time !== null) {
          notes.push({
            isDrum: false,
            program: open.program,
            onset: open.time,
            offset: event.time,
            pitch: open.pitch,
            velocity: open.velocity,
          });
        }
      }
      active.set(key, event);
    } else if (event.velocity === 0) {
      const open = active.get(key);
      if (open) {
        active.delete(key);
        if (open.time !== null) {
          notes.push({
            isDrum: false,
            program: open.program,
            onset: open.time,
            offset: event.time,
            pitch: open.pitch,
            velocity: open.velocity,
          });
        }
      }
    }
  }

  // Whatever is still sounding gets the minimum length.
  for (const event of active.values()) {
    if (event.velocity !== 1) continue;
    if (event.time === null) continue;
    notes.push({
      isDrum: false,
      program: event.program,
      onset: event.time,
      offset: event.time + MINIMUM_OFFSET_SEC,
      pitch: event.pitch,
      velocity: 1,
    });
  }

  // A note longer than ten seconds is a mistake, and is cut short.
  for (const note of notes) {
    if (note.offset - note.onset > MAXIMUM_NOTE_SEC) {
      note.offset = note.onset + MINIMUM_OFFSET_SEC;
    }
  }

  notes.sort(
    (a, b) =>
      a.onset - b.onset ||
      Number(a.isDrum) - Number(b.isDrum) ||
      a.program - b.program ||
      a.velocity - b.velocity ||
      a.pitch - b.pitch,
  );

  // Upstream checks the offsets before trimming the overlaps here, and the
  // other way round when the channels are mixed. The order matters: a note
  // that is clipped by the next one can end up with no length at all.
  return trimOverlappingNotes(validateNotes(notes));
}

/**
 * Merge the segments of one channel into a single run of notes.
 *
 * A port of upstream `merge_zipped_note_events_and_ties_to_notes`. A note
 * that was still sounding at the end of the previous segment has to be
 * closed, unless the current segment's preamble says it is tied over — in
 * which case it keeps sounding, and the note-off that ends it comes later.
 */
function mergeSegments(segments: SegmentEvents[], startTimes: number[]): { notes: Note[] } {
  const merged: NoteEvent[] = [];
  let previousActivity: ProgramPitch[] | null = null;

  for (let s = 0; s < segments.length; s++) {
    const segment = segments[s];
    if (previousActivity !== null) {
      const tied = new Set(segment.tieNoteEvents.map((event) => `${event.program}|${event.pitch}`));
      for (const [program, pitch] of previousActivity) {
        if (tied.has(`${program}|${pitch}`)) continue;
        // The note was still sounding but is not tied, so it ends now.
        merged.push({
          isDrum: false,
          program,
          time: startTimes[s],
          velocity: 0,
          pitch,
        });
      }
    }
    merged.push(...segment.noteEvents);
    previousActivity = segment.lastActivity;
  }

  return { notes: noteEventsToNotes(merged, null) };
}

/**
 * Mix the channels into one set of notes.
 *
 * A port of upstream `mix_notes`. The notes are sorted, the overlapping
 * ones are trimmed, and only then are the offsets checked — upstream does it
 * in that order, and it matters, because trimming can leave a note with no
 * length at all.
 */
function mixNotes(perChannel: Note[][]): Note[] {
  const mixed = perChannel.flat();
  mixed.sort(
    (a, b) =>
      a.onset - b.onset ||
      Number(a.isDrum) - Number(b.isDrum) ||
      a.program - b.program ||
      a.velocity - b.velocity ||
      a.pitch - b.pitch ||
      a.offset - b.offset,
  );
  return validateNotes(trimOverlappingNotes(mixed));
}

/**
 * Trim the notes that overlap on the same pitch and program.
 *
 * A port of upstream `trim_overlapping_notes`. Two notes of the same pitch
 * cannot sound at once, so the earlier one is cut short where the later one
 * starts, and a note with no length left is dropped.
 */
function trimOverlappingNotes(notes: Note[]): Note[] {
  if (notes.length <= 1) return notes;

  const groups = new Map<string, Note[]>();
  for (const note of notes) {
    const key = `${note.pitch}|${note.program}|${note.isDrum}`;
    const group = groups.get(key);
    if (group) group.push(note);
    else groups.set(key, [note]);
  }

  const trimmed: Note[] = [];
  for (const group of groups.values()) {
    const sorted = [...group].sort((a, b) => a.onset - b.onset);
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i - 1].offset > sorted[i].onset) {
        sorted[i - 1].offset = sorted[i].onset;
      }
    }
    for (const note of sorted) {
      if (note.onset < note.offset) trimmed.push(note);
    }
  }

  trimmed.sort(
    (a, b) =>
      a.onset - b.onset ||
      Number(a.isDrum) - Number(b.isDrum) ||
      a.program - b.program ||
      a.velocity - b.velocity ||
      a.pitch - b.pitch,
  );
  return trimmed;
}

/**
 * Check the notes and fix the ones that cannot be right.
 *
 * A port of upstream `validate_notes`. A note that ends before it starts is
 * pushed forward, and a note that is too short is stretched to the minimum
 * length. Drums are left alone: a drum hit is meant to be short.
 */
function validateNotes(notes: Note[]): Note[] {
  const valid: Note[] = [];
  for (const note of notes) {
    if (note.onset > note.offset) {
      note.offset = Math.max(note.offset, note.onset + MINIMUM_OFFSET_SEC);
    } else if (!note.isDrum && note.offset - note.onset < MINIMUM_OFFSET_SEC) {
      note.offset = note.onset + MINIMUM_OFFSET_SEC;
    }
    valid.push(note);
  }
  return valid;
}

/**
 * Group the notes into the parts the pipeline shows.
 *
 * The model reports its own fine-grained programs; the vocabulary maps them
 * to the output programs, and the output programs are what the tracks are
 * named after.
 */
function groupIntoParts(notes: Note[], programs: Map<number, ProgramInfo>): ModelPart[] {
  const groups = new Map<number, { name: string; notes: RawNote[] }>();

  for (const note of notes) {
    const info = note.isDrum
      ? { program: 128, name: 'Drums' }
      : programs.get(note.program) ?? { program: note.program, name: `Prg. ${note.program}` };

    let group = groups.get(info.program);
    if (!group) {
      group = { name: info.name, notes: [] };
      groups.set(info.program, group);
    }
    group.notes.push({
      pitchMidi: note.pitch,
      startTime: note.onset,
      duration: note.offset - note.onset,
      amplitude: MIDI_VELOCITY,
    });
  }

  const parts: { program: number; part: ModelPart }[] = [];
  for (const [program, group] of groups) {
    parts.push({
      program,
      part: {
        id: program === 128 ? 'drums' : `program-${program}`,
        name: group.name,
        kind: kindForProgram(program, group.name),
        notes: group.notes,
      },
    });
  }

  // By program number, with the drums last.
  parts.sort((a, b) => Number(a.program === 128) - Number(b.program === 128) || a.program - b.program);
  return parts.map((entry) => entry.part);
}

/**
 * The musical role of a track.
 *
 * The drums are drums, the bass is the bass, the parts that carry a tune
 * are the melody, and everything else is chords.
 */
function kindForProgram(program: number, name: string): TrackKind {
  if (program === 128) return 'drums';
  if (name === 'Bass') return 'bass';
  if (
    name === 'Singing Voice' ||
    name === 'Singing Voice (chorus)' ||
    name === 'Synth Lead' ||
    name === 'Brass' ||
    name === 'Reed' ||
    name === 'Pipe'
  ) {
    return 'melody';
  }
  return 'chords';
}

/** Join a directory and a file name, for both Node and the browser. */
function joinPath(directory: string, name: string): string {
  if (directory.endsWith('/') || directory.endsWith('\\')) return directory + name;
  return `${directory}/${name}`;
}
