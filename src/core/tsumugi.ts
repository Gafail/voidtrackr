/**
 * The Tsumugi model.
 *
 * Tsumugi extends TransKun's Neural Semi-CRF to every instrument, and is
 * aimed squarely at synthesised and anime music — the closest published
 * match to the material this tool is usually pointed at.
 *
 *   **Licence: MIT.** The weights are MIT, not just the code.
 *
 * It works in three layers.
 *
 * **The graph.** A single ONNX graph takes the raw waveform — eight seconds
 * of stereo audio at 22 050 Hz — and computes its own CQT front end, so
 * there is no spectrogram to write here. It then produces, for every
 * (track, begin, end) triple with a positive score, one candidate; a
 * diagonal score per frame; and two feature maps, one per pitch, which the
 * host heads below need.
 *
 * A *track* is a pitch and a slot: there are three slots per pitch, so a
 * sustained note and a repeated note on the same pitch can both exist.
 * Track index is `pitch * 3 + slot`, and the pitch is the MIDI number
 * minus 21.
 *
 * **The decode.** A backward Viterbi over the candidate list picks, for
 * each track, the best chain of intervals. Two transitions are possible:
 * *skip* to the next frame, at no cost, or an *interval* to a later frame,
 * at the candidate's score. A positive diagonal score adds itself at every
 * frame the chain visits, which is how a single-frame note is expressed.
 *
 * **The heads.** Two small heads live on the host rather than in the graph,
 * because they depend on the exact feature layout and there is only one
 * file of their weights. The boundary head refines where a note starts and
 * ends, within the frame; the instrument head says what is playing.
 */

import { ModelOptions, ModelPart, ModelTranscription, TranscriptionModel } from './model';
import { asFloat32, asInt32, asUint8, OnnxRunner, OnnxSession, OnnxTensor } from './onnx';
import { RawNote, TrackKind } from './types';

/** The sample rate the model wants its input at. */
export const TSUMUGI_SAMPLE_RATE = 22050;

/** Directory the weights live in, under the ONNX weights root. */
export const TSUMUGI_DIR = 'tsumugi';

/** The pinned weight files. */
export const TSUMUGI_FILES = {
  graph: 'tsumugi_other_v1_5.pos32i_i32.onnx',
  heads: 'tsumugi_other_v1_5.heads.bin',
  headsJson: 'tsumugi_other_v1_5.heads.json',
} as const;

/**
 * The graph's weights live in a sidecar file.
 *
 * The graph refers to it by this name, and the browser has to mount it
 * under that name because the runtime fetches the graph over HTTP and loses
 * the directory it came from.
 */
export const TSUMUGI_EXTERNAL_DATA = 'tsumugi_other_v1_5.pos32i_i32.onnx.data';

/** One decoded interval, before it becomes a note. */
interface Interval {
  /** First frame. */
  begin: number;
  /** Last frame, exclusive of nothing — the note is sounding at it too. */
  end: number;
}

/** A note as the model understands it, in samples. */
interface ModelNote {
  /** MIDI note number. */
  pitch: number;
  /** Which of the three slots for this pitch. */
  slot: number;
  /** Where it starts, in samples from the beginning of the recording. */
  startSample: number;
  /** Where it ends, in samples. */
  endSample: number;
  /** Whether the model saw an onset here, as opposed to a continuation. */
  hasOnset: boolean;
  /** Whether the model saw an offset here. */
  hasOffset: boolean;
  /** The instrument class it thinks is playing. */
  instrument: number;
  /** How sure it is about that instrument, 0..1. */
  instrumentScore: number;
}

/** The metadata that comes with the head weights. */
interface HeadsMeta {
  sample_rate: number;
  hop_length: number;
  num_pitches: number;
  min_midi_pitch: number;
  num_pitch_slots: number;
  num_instrument_classes: number;
  feature_dim: number;
  layer_norm_eps: number;
  window_ms: number;
  stride_ms: number;
  merge_gap_ms: number | null;
  merge_onset_ms: number;
  silence_gate_rms_dbfs: number;
  use_boundary_head: boolean;
  velocity: number;
  min_midi_note_ms: number;
  max_midi_melodic_instruments: number;
  collapse_crash_cymbals: boolean;
  drum_pitch_aliases: Record<string, number>;
  instrument_classes: string[];
  instrument_programs: number[];
  drum_class_ids: number[];
  tensors: Record<string, { offset: number; shape: number[] }>;
  floats: number;
}

/** The head weights, unpacked from the binary file. */
interface Heads {
  meta: HeadsMeta;
  slotEmbedding: Float32Array;
  boundaryW1: Float32Array;
  boundaryB1: Float32Array;
  boundaryW2: Float32Array;
  boundaryB2: Float32Array;
  instrumentLnW: Float32Array;
  instrumentLnB: Float32Array;
  instrumentW1: Float32Array;
  instrumentB1: Float32Array;
  instrumentW2: Float32Array;
  instrumentB2: Float32Array;
}

/**
 * Tsumugi, as a model the pipeline can use.
 */
export class TsumugiModel implements TranscriptionModel {
  readonly id = 'tsumugi-anime-song-v1.5';
  readonly displayName = 'Tsumugi';
  readonly license = 'MIT';
  readonly inputSampleRate = TSUMUGI_SAMPLE_RATE;

  constructor(
    private readonly session: OnnxSession,
    private readonly heads: Heads,
    /** Hash of the weights this instance was loaded from. */
    public readonly fingerprint: string = 'unknown',
  ) {}

  /**
   * Load the graph and the head weights.
   *
   * `path` is the directory the weights live in. The caller is responsible
   * for checking them against the lock file first.
   */
  static async load(
    runner: OnnxRunner,
    path: string,
    fingerprint = 'unknown',
  ): Promise<TsumugiModel> {
    const read = runner.read?.bind(runner);
    if (!read) {
      throw new Error('This runtime cannot read the Tsumugi head weights.');
    }

    const [session, headsJson, headsBin] = await Promise.all([
      runner.createSession(joinPath(path, TSUMUGI_FILES.graph), [
        {
          location: TSUMUGI_EXTERNAL_DATA,
          url: joinPath(path, TSUMUGI_EXTERNAL_DATA),
        },
      ]),
      read(joinPath(path, TSUMUGI_FILES.headsJson)).then((bytes) =>
        JSON.parse(new TextDecoder().decode(bytes)) as HeadsMeta,
      ),
      read(joinPath(path, TSUMUGI_FILES.heads)),
    ]);

    return new TsumugiModel(session, unpackHeads(headsJson, headsBin), fingerprint);
  }

  async transcribe(samples: Float32Array, options: ModelOptions = {}): Promise<ModelTranscription> {
    const report = (value: number, label: string) => options.onProgress?.(value, label);
    const meta = this.heads.meta;

    const total = samples.length;
    const hop = meta.hop_length;
    const windowSamples = Math.round((meta.window_ms * meta.sample_rate) / 1000);
    const strideSamples = Math.round((meta.stride_ms * meta.sample_rate) / 1000);

    const starts = windowStarts(total, windowSamples, strideSamples);

    // Notes are collected per track, because the stitching rules work on a
    // track's own history.
    const byTrack = new Map<number, ModelNote[]>();
    const lastClosedFrame = new Map<number, number>();

    for (let index = 0; index < starts.length; index++) {
      const windowStart = starts[index];

      report(0.05 + (0.85 * index) / starts.length, 'Reading notes');

      // A window that is silent carries no information, and decoding it
      // only produces noise.
      if (isSilent(samples, windowStart, windowSamples, meta.silence_gate_rms_dbfs)) {
        continue;
      }

      const validAudio = Math.max(0, Math.min(windowSamples, total - windowStart));
      const validFrames = Math.ceil(validAudio / hop);

      const window = new Float32Array(windowSamples);
      window.set(samples.subarray(windowStart, windowStart + validAudio));

      // The graph wants stereo. The pipeline works in mono, so the same
      // signal goes to both channels and the front end sees what a mono
      // mix would give it.
      const stereo = new Float32Array(2 * windowSamples);
      stereo.set(window, 0);
      stereo.set(window, windowSamples);

      const out = await this.session.run({
        waveform: { type: 'float32', data: stereo, dims: [1, 2, windowSamples] },
        valid_audio_frames: {
          type: 'int64',
          data: BigInt64Array.from([BigInt(validAudio)]),
          dims: [1],
        },
      });

      const notes = this.decodeWindow(out, windowStart, validAudio, validFrames, lastClosedFrame);
      for (const [track, windowNotes] of notes) {
        const list = byTrack.get(track) ?? [];
        for (const note of windowNotes) {
          const previous = list[list.length - 1];
          if (previous && note.startSample < previous.endSample) {
            // The two overlap. A new onset starts a fresh note and replaces
            // the old one; a continuation extends it.
            if (note.hasOnset) {
              list[list.length - 1] = note;
            } else {
              previous.endSample = Math.max(previous.endSample, note.endSample);
              previous.hasOffset = note.hasOffset;
            }
          } else if (note.hasOnset) {
            list.push(note);
          }
        }
        byTrack.set(track, list);
      }
    }

    report(0.92, 'Tidying up');

    const all = [...byTrack.values()].flat();
    const finished = finaliseNotes(all, this.heads, total);

    const parts = this.toParts(finished);

    return {
      parts,
      log: [
        `Tsumugi found ${finished.length} notes in ${parts.length} parts.`,
      ],
    };
  }

  /**
   * Decode one window into notes.
   *
   * The graph gives candidates, a diagonal score per frame, and the feature
   * maps the heads need. The Viterbi picks the intervals, and the heads
   * turn them into notes with a start, an end and an instrument.
   */
  private decodeWindow(
    out: Record<string, OnnxTensor>,
    windowStart: number,
    validAudio: number,
    validFrames: number,
    lastClosedFrame: Map<number, number>,
  ): Map<number, ModelNote[]> {
    const meta = this.heads.meta;

    // The graph says how many frames of the window carry audio. Upstream runs
    // its Viterbi over exactly those frames and no further: the frames past
    // them are the zero padding at the end of the window, and the model still
    // gives them scores, so a decoder that walks over them can pick an
    // interval that runs into the padding.
    const mask = asUint8(out.frame_valid_mask);
    let frames = 0;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) frames += 1;
    }
    if (frames <= 0) frames = validFrames;

    const candScore = asFloat32(out.cand_score);
    const candIdx = asInt32(out.cand_idx);
    const diag = asFloat32(out.diag_score);
    const activePitches = asInt32(out.active_pitches);
    const pitchFeatures = asFloat32(out.pitch_features);
    const instrumentFeatures = asFloat32(out.instrument_pitch_features);

    // Candidates, grouped by track and then by the frame they start on.
    const byTrack = new Map<number, Map<number, { end: number; score: number }[]>>();
    for (let row = 0; row < candIdx.length / 4; row++) {
      const track = candIdx[row * 4 + 1];
      const begin = candIdx[row * 4 + 2];
      const end = candIdx[row * 4 + 3];
      const score = candScore[row];
      if (end <= begin) continue;

      let starts = byTrack.get(track);
      if (!starts) {
        starts = new Map();
        byTrack.set(track, starts);
      }
      const list = starts.get(begin);
      if (list) list.push({ end, score });
      else starts.set(begin, [{ end, score }]);
    }

    const windowModelStart = Math.round(windowStart / meta.hop_length);
    const notes = new Map<number, ModelNote[]>();

    for (const [track, starts] of byTrack) {
      // The next window starts where this one's last closed note ended, so a
      // note that runs over the seam is not cut in two.
      const lastClosed = lastClosedFrame.get(track);
      const forced =
        lastClosed === undefined
          ? 0
          : Math.max(0, Math.min(lastClosed - windowModelStart, validFrames - 1));

      const intervals = viterbi(starts, diag, track, frames, validFrames, forced);

      const trackNotes: ModelNote[] = [];
      for (const interval of intervals) {
        const note = this.toNote(
          interval,
          track,
          windowStart,
          validAudio,
          activePitches,
          pitchFeatures,
          instrumentFeatures,
        );
        if (!note) continue;
        trackNotes.push(note);
        if (note.hasOffset) {
          lastClosedFrame.set(track, windowModelStart + interval.end);
        }
      }
      notes.set(track, trackNotes);
    }

    return notes;
  }

  /**
   * Turn one interval into a note.
   *
   * The boundary head says where inside the frame the note starts and ends,
   * and whether there was an onset and an offset at all. The instrument head
   * says what is playing.
   */
  private toNote(
    interval: Interval,
    track: number,
    windowStart: number,
    validAudio: number,
    activePitches: Int32Array,
    pitchFeatures: Float32Array,
    instrumentFeatures: Float32Array,
  ): ModelNote | null {
    const meta = this.heads.meta;
    const hop = meta.hop_length;
    const pitchIndex = Math.floor(track / meta.num_pitch_slots);
    const slot = track % meta.num_pitch_slots;

    const pitch = meta.min_midi_pitch + pitchIndex;

    // The features are per pitch; the slot is added on top of them.
    const activeIndex = indexOf(activePitches, pitchIndex);
    const feature = (source: Float32Array, frame: number): Float32Array => {
      const out = new Float32Array(meta.feature_dim);
      if (activeIndex < 0) return out;
      const base =
        (frame * activePitches.length + activeIndex) * meta.feature_dim;
      for (let i = 0; i < meta.feature_dim; i++) {
        out[i] = source[base + i] + this.heads.slotEmbedding[slot * meta.feature_dim + i];
      }
      return out;
    };

    const beginFeatures = feature(pitchFeatures, interval.begin);
    const endFeatures = feature(pitchFeatures, interval.end);

    let hasOnset: boolean;
    let hasOffset: boolean;
    let onsetOffset: number;
    let offsetOffset: number;

    if (meta.use_boundary_head) {
      const boundary = this.boundaryHead(beginFeatures, endFeatures);
      hasOnset = boundary.hasOnset;
      hasOffset = boundary.hasOffset;
      onsetOffset = boundary.onsetOffset;
      offsetOffset = boundary.offsetOffset;
    } else {
      // Without the boundary head, upstream says a note has an offset when
      // it ends before the last frame that carries audio.
      const validModelFrames = Math.ceil(validAudio / hop);
      hasOnset = interval.begin > 0;
      hasOffset = interval.end < validModelFrames - 1;
      onsetOffset = 0;
      offsetOffset = 1;
    }

    const instrument = this.instrumentHead(
      interval,
      feature(instrumentFeatures, interval.begin),
      feature(instrumentFeatures, interval.end),
      instrumentFeatures,
      activePitches,
      activeIndex,
      slot,
    );

    const startSample = windowStart + Math.round((interval.begin + onsetOffset) * hop);
    const endSample =
      windowStart + Math.min(validAudio, Math.round((interval.end + offsetOffset) * hop));

    const totalLimit = windowStart + validAudio;
    if (startSample >= totalLimit) return null;

    return {
      pitch,
      slot,
      startSample: Math.max(0, startSample),
      endSample: Math.max(startSample + 1, endSample),
      hasOnset,
      hasOffset,
      instrument: instrument.id,
      instrumentScore: instrument.score,
    };
  }

  /**
   * The boundary head: where inside the frame the note starts and ends.
   *
   * Two linear layers with a GELU between them, on the two endpoint
   * features and their product.
   */
  private boundaryHead(
    begin: Float32Array,
    end: Float32Array,
  ): { hasOnset: boolean; hasOffset: boolean; onsetOffset: number; offsetOffset: number } {
    const heads = this.heads;
    const dim = begin.length;

    const input = new Float32Array(dim * 3);
    input.set(begin, 0);
    input.set(end, dim);
    for (let i = 0; i < dim; i++) input[dim * 2 + i] = begin[i] * end[i];

    const hidden = new Float32Array(256);
    for (let j = 0; j < 256; j++) {
      let sum = heads.boundaryB1[j];
      const row = j * (dim * 3);
      for (let i = 0; i < dim * 3; i++) sum += heads.boundaryW1[row + i] * input[i];
      hidden[j] = gelu(sum);
    }

    const out = new Float32Array(4);
    for (let j = 0; j < 4; j++) {
      let sum = heads.boundaryB2[j];
      const row = j * 256;
      for (let i = 0; i < 256; i++) sum += heads.boundaryW2[row + i] * hidden[i];
      out[j] = sum;
    }

    return {
      hasOnset: out[0] > 0,
      hasOffset: out[1] > 0,
      onsetOffset: rescale(out[2]),
      offsetOffset: rescale(out[3]),
    };
  }

  /**
   * The instrument head: which of the 36 classes is playing.
   *
   * A layer norm, two linear layers and a GELU, on the two endpoint
   * features, their product, the mean over the interval and the interval's
   * length.
   */
  private instrumentHead(
    interval: Interval,
    begin: Float32Array,
    end: Float32Array,
    source: Float32Array,
    activePitches: Int32Array,
    activeIndex: number,
    slot: number,
  ): { id: number; score: number } {
    const heads = this.heads;
    const meta = heads.meta;
    const dim = meta.feature_dim;

    // The mean of the features across the interval, inclusive.
    const mean = new Float32Array(dim);
    if (activeIndex >= 0) {
      const count = interval.end - interval.begin + 1;
      for (let frame = interval.begin; frame <= interval.end; frame++) {
        const base = (frame * activePitches.length + activeIndex) * dim;
        for (let i = 0; i < dim; i++) {
          mean[i] += source[base + i] + heads.slotEmbedding[slot * dim + i];
        }
      }
      for (let i = 0; i < dim; i++) mean[i] /= count;
    }

    const input = new Float32Array(dim * 4 + 1);
    input.set(begin, 0);
    input.set(end, dim);
    input.set(mean, dim * 2);
    for (let i = 0; i < dim; i++) input[dim * 3 + i] = begin[i] * end[i];
    input[dim * 4] = (interval.end - interval.begin + 1) / 345;

    // Layer norm over the whole 1025-element vector.
    const normed = new Float32Array(input.length);
    let sum = 0;
    for (let i = 0; i < input.length; i++) sum += input[i];
    const meanValue = sum / input.length;
    let variance = 0;
    for (let i = 0; i < input.length; i++) {
      const d = input[i] - meanValue;
      variance += d * d;
    }
    variance /= input.length;
    const scale = 1 / Math.sqrt(variance + meta.layer_norm_eps);
    for (let i = 0; i < input.length; i++) {
      normed[i] = (input[i] - meanValue) * scale * heads.instrumentLnW[i] + heads.instrumentLnB[i];
    }

    const hidden = new Float32Array(256);
    for (let j = 0; j < 256; j++) {
      let total = heads.instrumentB1[j];
      const row = j * input.length;
      for (let i = 0; i < input.length; i++) total += heads.instrumentW1[row + i] * normed[i];
      hidden[j] = gelu(total);
    }

    let best = 0;
    let bestScore = -1;
    for (let j = 0; j < meta.num_instrument_classes; j++) {
      let total = heads.instrumentB2[j];
      const row = j * 256;
      for (let i = 0; i < 256; i++) total += heads.instrumentW2[row + i] * hidden[i];
      const probability = sigmoid(total);
      if (probability > bestScore) {
        bestScore = probability;
        best = j;
      }
    }

    return { id: best, score: bestScore };
  }

  /** Group the finished notes into one part per instrument. */
  private toParts(notes: ModelNote[]): ModelPart[] {
    const meta = this.heads.meta;
    const drumClasses = new Set(meta.drum_class_ids);

    const groups = new Map<number, RawNote[]>();
    for (const note of notes) {
      const list = groups.get(note.instrument);
      const raw: RawNote = {
        pitchMidi: note.pitch,
        startTime: note.startSample / meta.sample_rate,
        duration: (note.endSample - note.startSample) / meta.sample_rate,
        amplitude: 0.8,
      };
      if (list) list.push(raw);
      else groups.set(note.instrument, [raw]);
    }

    const parts: ModelPart[] = [];
    const sorted = [...groups.entries()].sort((a, b) => {
      const aDrum = drumClasses.has(a[0]) ? 1 : 0;
      const bDrum = drumClasses.has(b[0]) ? 1 : 0;
      if (aDrum !== bDrum) return aDrum - bDrum;
      return a[0] - b[0];
    });

    for (const [instrument, raw] of sorted) {
      raw.sort((a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi);
      const name = titleCase(meta.instrument_classes[instrument] ?? `Instrument ${instrument}`);
      parts.push({
        id: drumClasses.has(instrument) ? 'drums' : `instrument-${instrument}`,
        name,
        kind: trackKind(meta.instrument_classes[instrument] ?? '', drumClasses.has(instrument)),
        notes: raw,
      });
    }

    return parts;
  }
}

/** The track a note belongs to, for the stitching rules. */
function trackOf(note: ModelNote): number {
  return note.pitch * 3 + note.slot;
}

/**
 * The window starts.
 *
 * Windows overlap by half, and the last one is moved so that it ends on
 * the last sample of the recording.
 */
function windowStarts(total: number, windowSamples: number, strideSamples: number): number[] {
  if (total <= 0) return [0];
  const starts: number[] = [];
  const limit = Math.max(1, total - windowSamples + 1);
  for (let start = 0; start < limit; start += strideSamples) starts.push(start);
  if (total > windowSamples) {
    const last = total - windowSamples;
    if (starts[starts.length - 1] !== last) starts.push(last);
  }
  return starts;
}

/** Whether a window is quiet enough to skip. */
function isSilent(
  samples: Float32Array,
  windowStart: number,
  windowSamples: number,
  silenceGateDbfs: number,
): boolean {
  const gate = Math.pow(10, silenceGateDbfs / 20);
  const end = Math.min(samples.length, windowStart + windowSamples);
  if (windowStart >= end) return true;

  // The gate is an RMS over the whole window, zero padding included, and
  // without subtracting the mean: `sqrt(mean(x^2))`. A window that runs off
  // the end of the recording is padded with zeros, so its level drops the
  // same way upstream's padding drops it.
  let energy = 0;
  for (let i = windowStart; i < end; i++) {
    energy += samples[i] * samples[i];
  }
  const rms = Math.sqrt(energy / windowSamples);
  return rms < gate;
}

/**
 * The backward Viterbi over a track's candidates.
 *
 * At each frame the chain either skips to the next frame for free, or takes
 * an interval to a later frame and pays the candidate's score. A positive
 * diagonal score is added at every frame the chain stands on.
 */
function viterbi(
  starts: Map<number, { end: number; score: number }[]>,
  diag: Float32Array,
  track: number,
  frames: number,
  validFrames: number,
  forcedStart: number,
): Interval[] {
  const tracks = 264;
  const q = new Float32Array(frames);
  const ptr = new Int32Array(frames).fill(-1);

  const diagonalAt = (frame: number) => diag[frame * tracks + track];

  const last = frames - 1;
  q[last] = diagonalAt(last) > 0 ? diagonalAt(last) : 0;

  for (let t = last - 1; t >= 0; t--) {
    let best = Number.NEGATIVE_INFINITY;
    let bestEnd = -1;
    const candidates = starts.get(t);
    if (candidates) {
      for (const candidate of candidates) {
        if (candidate.end >= frames) continue;
        const value = q[candidate.end] + candidate.score;
        if (value > best) {
          best = value;
          bestEnd = candidate.end;
        }
      }
    }

    const skip = q[t + 1];
    // A tie goes to the skip, as it does upstream.
    const useInterval = best > skip;
    ptr[t] = useInterval ? bestEnd : -1;
    const diagonal = diagonalAt(t);
    q[t] = (useInterval ? best : skip) + (diagonal > 0 ? diagonal : 0);
  }

  const intervals: Interval[] = [];
  let pos = forcedStart;
  while (pos < last) {
    if (validFrames > 0 && pos >= validFrames) break;
    if (diagonalAt(pos) > 0) intervals.push({ begin: pos, end: pos });
    const next = ptr[pos];
    if (next < 0) {
      pos += 1;
    } else {
      intervals.push({ begin: pos, end: next });
      pos = next;
    }
  }
  if (validFrames > 0 && last < validFrames && diagonalAt(last) > 0) {
    intervals.push({ begin: last, end: last });
  }

  return intervals;
}

/**
 * Finish the notes: close the last one on each track, merge the ones that
 * belong together, drop the ones that never ended, and apply the drum
 * pitch aliases and the minimum length.
 */
function finaliseNotes(notes: ModelNote[], heads: Heads, totalSamples: number): ModelNote[] {
  const meta = heads.meta;
  const sampleRate = meta.sample_rate;
  const minSeparation = Math.max(1, Math.round(sampleRate * 0.005));
  const minNote = Math.round((meta.min_midi_note_ms * sampleRate) / 1000);
  const mergeGap =
    meta.merge_gap_ms === null ? meta.hop_length : Math.round((meta.merge_gap_ms * sampleRate) / 1000);
  const mergeOnset = Math.round((meta.merge_onset_ms * sampleRate) / 1000);

  // Group by track, so the merge rules and the last-note rule apply to a
  // track's own history.
  const byTrack = new Map<number, ModelNote[]>();
  for (const note of notes) {
    const key = trackOf(note);
    const list = byTrack.get(key);
    if (list) list.push(note);
    else byTrack.set(key, [note]);
  }

  const merged: ModelNote[] = [];
  for (const list of byTrack.values()) {
    list.sort((a, b) => a.startSample - b.startSample || a.endSample - b.endSample);

    // The last note of a track is force-closed.
    if (list.length > 0) list[list.length - 1].hasOffset = true;

    let current: ModelNote | null = null;
    for (const note of list) {
      const mergeByGap =
        current !== null && !note.hasOnset && note.startSample <= current.endSample + mergeGap;
      const mergeByOnset =
        current !== null && Math.abs(note.startSample - current.startSample) <= mergeOnset;

      if (current && (mergeByGap || mergeByOnset)) {
        current.endSample = Math.max(current.endSample, note.endSample);
        // A gap merge takes the continuation's offset as it is; an onset
        // merge keeps whichever of the two has one.
        current.hasOffset = mergeByGap
          ? note.hasOffset
          : current.hasOffset || note.hasOffset;
        if (note.instrumentScore > current.instrumentScore) {
          current.instrument = note.instrument;
          current.instrumentScore = note.instrumentScore;
        }
      } else {
        if (current) merged.push(current);
        current = note;
      }
    }
    if (current) merged.push(current);
  }

  // A note that never ended is not a note.
  const closed = merged.filter((note) => note.hasOffset);

  // Drum pitch aliases, so a crash and a ride do not double up.
  const drumClasses = new Set(meta.drum_class_ids);
  if (meta.collapse_crash_cymbals) {
    for (const note of closed) {
      if (!drumClasses.has(note.instrument)) continue;
      const alias = meta.drum_pitch_aliases[String(note.pitch)];
      if (alias !== undefined) note.pitch = alias;
    }
  }

  // Trim overlaps per instrument and pitch, the way a MIDI file has to,
  // because a note-off cannot address one of two notes on the same pitch.
  const groups = new Map<string, ModelNote[]>();
  for (const note of closed) {
    const key = `${note.instrument}:${note.pitch}`;
    const list = groups.get(key);
    if (list) list.push(note);
    else groups.set(key, [note]);
  }

  const trimmed: ModelNote[] = [];
  for (const list of groups.values()) {
    list.sort((a, b) => a.startSample - b.startSample);
    for (let i = 0; i < list.length; i++) {
      const note = list[i];
      const next = list[i + 1];
      if (next && note.endSample > next.startSample - minSeparation) {
        note.endSample = next.startSample - minSeparation;
        if (note.endSample - note.startSample < minSeparation) continue;
      }
      if (note.endSample - note.startSample < minNote) {
        note.endSample = note.startSample + minNote;
        if (next && note.endSample > next.startSample - minSeparation) {
          note.endSample = next.startSample - minSeparation;
        }
      }
      trimmed.push(note);
    }
  }

  trimmed.sort(
    (a, b) =>
      a.startSample - b.startSample ||
      a.instrument - b.instrument ||
      a.pitch - b.pitch ||
      a.slot - b.slot ||
      a.endSample - b.endSample,
  );

  return trimmed.filter((note) => note.startSample < totalSamples);
}

/** Which musical role an instrument class plays. */
function trackKind(instrumentClass: string, isDrum: boolean): TrackKind {
  if (isDrum) return 'drums';
  if (instrumentClass.includes('bass')) return 'bass';
  if (
    instrumentClass.includes('melody') ||
    instrumentClass.includes('vocal') ||
    instrumentClass.includes('choir') ||
    instrumentClass.includes('lead')
  ) {
    return 'melody';
  }
  return 'chords';
}

/** Turn `synth_pad` into `Synth Pad`. */
function titleCase(name: string): string {
  return name
    .split('_')
    .filter((word) => word.length > 0)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(' ');
}

/** The index of a value in an ascending array, or -1. */
function indexOf(values: Int32Array, target: number): number {
  for (let i = 0; i < values.length; i++) {
    if (values[i] === target) return i;
    if (values[i] > target) return -1;
  }
  return -1;
}

/** Unpack the head weights from the binary file. */
function unpackHeads(meta: HeadsMeta, bytes: ArrayBuffer): Heads {
  const all = new Float32Array(bytes);
  const view = (name: string): Float32Array => {
    const entry = meta.tensors[name];
    if (!entry) throw new Error(`The head weights do not contain ${name}.`);
    const start = entry.offset;
    const length = entry.shape.reduce((a, b) => a * b, 1);
    return all.subarray(start, start + length);
  };

  return {
    meta,
    slotEmbedding: view('slot_embedding'),
    boundaryW1: view('boundary.w1'),
    boundaryB1: view('boundary.b1'),
    boundaryW2: view('boundary.w2'),
    boundaryB2: view('boundary.b2'),
    instrumentLnW: view('instrument.ln_w'),
    instrumentLnB: view('instrument.ln_b'),
    instrumentW1: view('instrument.w1'),
    instrumentB1: view('instrument.b1'),
    instrumentW2: view('instrument.w2'),
    instrumentB2: view('instrument.b2'),
  };
}

/** The GELU activation, as PyTorch's default does it. */
function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

/**
 * The error function.
 *
 * Abramowitz and Stegun's approximation, which is accurate to about
 * 1.5e-7 — far below anything the heads can notice.
 */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    0.254829592 * t -
    0.284496736 * t * t +
    1.421413741 * t * t * t -
    1.453152027 * t * t * t * t +
    1.061405429 * t * t * t * t * t;
  return sign * (1 - poly * Math.exp(-ax * ax));
}

/** The logistic function. */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Turn a boundary logit into a position inside the frame.
 *
 * The head's output is a probability; upstream rescales it so that the
 * extremes still reach the ends of the frame.
 */
function rescale(logit: number): number {
  const value = (sigmoid(logit) - 0.005) / 0.99;
  return Math.min(1, Math.max(0, value));
}

/** Join a directory and a file name, for both Node and the browser. */
function joinPath(directory: string, name: string): string {
  if (directory.endsWith('/') || directory.endsWith('\\')) return directory + name;
  return `${directory}/${name}`;
}
