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
export type OnnxDataType = 'float32' | 'int32' | 'int64';

/** A tensor handed to, or returned from, an ONNX graph. */
export interface OnnxTensor {
  /** The element type of the values. */
  type: OnnxDataType;
  /** The values: a Float32Array, an Int32Array or a BigInt64Array. */
  data: Float32Array | Int32Array | BigInt64Array;
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
 * Session options that keep the output reproducible.
 *
 * ONNX Runtime reduces in an order that depends on the thread count, so the
 * count is pinned. Otherwise two runs of the same file could differ in the
 * last bits, which is exactly what this project promises not to do.
 */
export const DETERMINISTIC_SESSION_OPTIONS = {
  intraOpNumThreads: 1,
  interOpNumThreads: 1,
  executionMode: 'sequential',
} as const;
