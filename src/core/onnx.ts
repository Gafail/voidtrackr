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

/** A float32 tensor. */
export interface OnnxTensor {
  data: Float32Array;
  dims: readonly number[];
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

/** Creates sessions. */
export interface OnnxRunner {
  createSession(path: string): Promise<OnnxSession>;
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
