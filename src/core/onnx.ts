/**
 * A small, runtime-agnostic interface for ONNX inference.
 *
 * The pipeline must run in the browser and in Node, and the two ONNX
 * runtimes are different packages with different APIs. Rather than pull
 * either of them into the shared code, the environment provides an adapter
 * that speaks this interface.
 *
 * Everything here is plain data, so `src/core` stays free of Node and DOM
 * specifics.
 */

/** The element types the models use. */
export type OnnxDataType = 'float32' | 'int32' | 'int64' | 'bool';

/** A tensor handed to, or returned from, an ONNX graph. */
export interface OnnxTensor {
  /** The element type of the values. */
  type: OnnxDataType;
  /** The values: a Float32Array, an Int32Array, a BigInt64Array or a Uint8Array. */
  data: Float32Array | Int32Array | BigInt64Array | Uint8Array;
  dims: readonly number[];
}

/**
 * Read a tensor's values as floats.
 *
 * The graphs return float32 for everything they predict, and integers only
 * for indices and token ids, so this is the common case.
 */
export function asFloat32(tensor: OnnxTensor): Float32Array {
  return tensor.data as Float32Array;
}

/** Read a tensor's values as 32-bit integers. */
export function asInt32(tensor: OnnxTensor): Int32Array {
  return tensor.data as Int32Array;
}

/** Read a tensor's values as 64-bit integers. */
export function asInt64(tensor: OnnxTensor): BigInt64Array {
  return tensor.data as BigInt64Array;
}

/**
 * Read a tensor's values as bytes.
 *
 * A boolean tensor comes back as bytes, one per value, where zero is false
 * and anything else is true.
 */
export function asUint8(tensor: OnnxTensor): Uint8Array {
  return tensor.data as Uint8Array;
}

/** An inference session. */
export interface OnnxSession {
  /** Names of the graph's inputs. */
  readonly inputNames: readonly string[];
  /** Names of the graph's outputs. */
  readonly outputNames: readonly string[];
  /** Run the graph. */
  run(feeds: Record<string, OnnxTensor>): Promise<Record<string, OnnxTensor>>;
}

/**
 * A file a graph needs that is not part of the graph itself.
 *
 * A large ONNX model sometimes keeps its weights in a sidecar file next to
 * the graph, and the graph refers to it by name. The browser has to be told
 * about the file explicitly, because the runtime fetches the graph over
 * HTTP and loses the directory it came from.
 */
export interface ExternalDataFile {
  /** The name the graph refers to the file by. */
  location: string;
  /** Where the file is — a URL in the browser, a path on disk in Node. */
  url: string;
}

/** Creates sessions. */
export interface OnnxRunner {
  /**
   * Load a graph.
   *
   * `externalData` lists the files the graph refers to but does not contain.
   * Node finds them on disk by itself; the browser has to be told.
   */
  createSession(path: string, externalData?: ExternalDataFile[]): Promise<OnnxSession>;
  /**
   * Read a file that sits next to the model.
   *
   * Some models ship extra files — a vocabulary, a set of head weights — that
   * the graph itself does not contain. The browser fetches them over HTTP and
   * Node reads them off disk, so the reading is left to the environment.
   */
  read?(path: string): Promise<ArrayBuffer>;
}

/**
 * How many threads a model may use.
 *
 * This was one, for a reason that turned out to be smaller than it looked.
 * ONNX Runtime reduces in an order that depends on the thread count, so with
 * more than one thread the raw output of a graph can differ in its last bits.
 * The exported file did not: measured on 11 November 2026 over every model in
 * the catalogue, in the browser and on the command line, and over the user's
 * own tracks, the MIDI was byte identical at 1, 2, 4 and 8 threads. The
 * differences are in the last bits of a float and the model's thresholds turn
 * them into the same notes. See `docs/threading.md`.
 *
 * Four is the number, and not eight, because eight is not always better:
 * measured per model, Okarina is slower at eight than at four, and Tsumugi
 * and YourMT3+ both turn round at eight. Only the piano model — the default
 * — keeps going.
 */
export const ONNX_THREADS = 4;

/**
 * Session options shared by both runtimes.
 *
 * The operations still run one after another, so the graph's order does not
 * change; the parallelism is inside an operation, which is where the time
 * goes.
 *
 * The memory arena is turned off. It is a pool that keeps allocations
 * around for the next run, which is faster but holds on to far more memory
 * than it needs: measured on 12 November 2026 with the piano model, the
 * arena on meant a peak of 1.15 GB and the arena off meant 392 MB, for the
 * same notes and a conversion that was not measurably slower. The box the
 * API runs on has a gigabyte of memory, so the difference between those two
 * numbers is the difference between the API working there and not.
 */
export const DETERMINISTIC_SESSION_OPTIONS = {
  intraOpNumThreads: ONNX_THREADS,
  interOpNumThreads: 1,
  executionMode: 'sequential',
  enableCpuMemArena: false,
  enableMemPattern: false,
} as const;
