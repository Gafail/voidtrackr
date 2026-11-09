/**
 * The browser ONNX runtime, behind the shared interface.
 *
 * Only the app uses this. The command line has its own adapter, so neither
 * runtime leaks into the shared pipeline code.
 *
 * The runtime is imported lazily: it is a large WebAssembly bundle, and
 * most sessions never touch it because Basic Pitch does not need it.
 */

import { DETERMINISTIC_SESSION_OPTIONS, OnnxRunner, OnnxTensor } from './onnx';

/** Where the WebAssembly runtime is served from. */
const WASM_PATH = '/ort/';

/**
 * Load the runtime and return a runner.
 *
 * The WebAssembly build is used rather than the WebGPU one: it works
 * everywhere, and it is the one that keeps the output identical between
 * runs, because it does not reorder its reductions the way the GPU path
 * does.
 */
export async function createWebOnnxRunner(): Promise<OnnxRunner> {
  const ort = await import('onnxruntime-web/wasm');

  ort.env.wasm.wasmPaths = WASM_PATH;
  ort.env.wasm.numThreads = 1;

  return {
    async createSession(path: string) {
      const session = await ort.InferenceSession.create(path, {
        ...DETERMINISTIC_SESSION_OPTIONS,
      });

      return {
        inputNames: session.inputNames,
        outputNames: session.outputNames,
        async run(feeds: Record<string, OnnxTensor>) {
          const tensors: Record<string, InstanceType<typeof ort.Tensor>> = {};
          for (const [name, tensor] of Object.entries(feeds)) {
            tensors[name] = new ort.Tensor('float32', tensor.data, tensor.dims);
          }
          const result = await session.run(tensors);
          const output: Record<string, OnnxTensor> = {};
          for (const [name, tensor] of Object.entries(result)) {
            output[name] = { data: tensor.data as Float32Array, dims: tensor.dims };
          }
          return output;
        },
      };
    },
  };
}
