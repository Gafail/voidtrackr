/**
 * The browser ONNX runtime, behind the shared interface.
 *
 * Only the app uses this. The tests and the measurement scripts have their
 * own adapter, so neither runtime leaks into the shared pipeline code.
 *
 * The runtime is imported lazily: it is a large WebAssembly bundle, and
 * most sessions never touch it because Basic Pitch does not need it.
 */

import {
  DETERMINISTIC_SESSION_OPTIONS,
  ExternalDataFile,
  ONNX_THREADS,
  OnnxDataType,
  OnnxRunner,
  OnnxTensor,
} from './onnx';

/** Where the WebAssembly runtime is served from. */
const WASM_PATH = '/ort/';

/** The address of a file, as the page sees it. */
function absolute(path: string): string {
  return new URL(path, window.location.href).href;
}

/**
 * Load the runtime and return a runner.
 *
 * The WebAssembly build is used rather than the WebGPU one: it works
 * everywhere, and it is the one that keeps the output identical between
 * runs, because it does not reorder its reductions the way the GPU path
 * does.
 *
 * `weights` holds the files the app has already fetched, by their absolute
 * address. The runtime fetches the weights itself, and it does it without
 * saying how far it has got — which is why the progress bar used to sit
 * still through the whole of the load. When the app has already read a
 * file, the runtime is handed the bytes rather than being sent to fetch
 * them again.
 *
 * The address is the absolute one, because the runtime is given a path
 * relative to the page and resolves it for itself.
 */
export async function createWebOnnxRunner(
  weights?: Map<string, Uint8Array>,
): Promise<OnnxRunner> {
  const ort = await import('onnxruntime-web/wasm');

  ort.env.wasm.wasmPaths = WASM_PATH;
  // The runtime reads this once, when its WebAssembly proxy starts, so it
  // has to be set before the first session is created. It is also ignored
  // unless the page is cross-origin isolated: without `SharedArrayBuffer`
  // the runtime puts the count back to one and says so in the console.
  // The isolation headers are in `ops/nginx/voidtrackr-headers.conf`.
  ort.env.wasm.numThreads = ONNX_THREADS;

  return {
    async createSession(path: string, externalData?: ExternalDataFile[]) {
      const options = {
        ...DETERMINISTIC_SESSION_OPTIONS,
        // The runtime does not know which directory the graph came from, so
        // it cannot find the sidecar file the graph refers to. Mounting it
        // under the name the graph uses fixes that.
        externalData: externalData?.map((file) => ({
          path: file.location,
          data: weights?.get(absolute(file.url)) ?? file.url,
        })),
      };

      const bytes = weights?.get(absolute(path));
      const session = bytes
        ? await ort.InferenceSession.create(bytes, options)
        : await ort.InferenceSession.create(path, options);

      return {
        inputNames: session.inputNames,
        outputNames: session.outputNames,
        async run(feeds: Record<string, OnnxTensor>) {
          const tensors: Record<string, InstanceType<typeof ort.Tensor>> = {};
          for (const [name, tensor] of Object.entries(feeds)) {
            tensors[name] = toOrtTensor(ort, tensor);
          }
          const result = await session.run(tensors);
          const output: Record<string, OnnxTensor> = {};
          for (const [name, tensor] of Object.entries(result)) {
            output[name] = {
              type: tensor.type as OnnxDataType,
              data: tensor.data as Float32Array | Int32Array | BigInt64Array,
              dims: tensor.dims,
            };
          }
          return output;
        },
      };
    },

    /** Fetch a file that sits next to the model, such as a vocabulary. */
    async read(path: string): Promise<ArrayBuffer> {
      const bytes = weights?.get(absolute(path));
      if (bytes) {
        return bytes.buffer.slice(
          bytes.byteOffset,
          bytes.byteOffset + bytes.byteLength,
        ) as ArrayBuffer;
      }

      const response = await fetch(absolute(path));
      if (!response.ok) {
        throw new Error(`Could not read the model file ${path} (${response.status})`);
      }
      return response.arrayBuffer();
    },
  };
}

/**
 * Build a runtime tensor.
 *
 * The runtime wants a different array type for each element type, and 64-bit
 * integers are BigInt rather than number, so the cases are spelled out rather
 * than cast into one.
 */
function toOrtTensor(
  ort: typeof import('onnxruntime-web/wasm'),
  tensor: OnnxTensor,
): InstanceType<typeof ort.Tensor> {
  switch (tensor.type) {
    case 'float32':
      return new ort.Tensor('float32', tensor.data as Float32Array, tensor.dims);
    case 'int32':
      return new ort.Tensor('int32', tensor.data as Int32Array, tensor.dims);
    case 'int64':
      return new ort.Tensor('int64', tensor.data as BigInt64Array, tensor.dims);
    case 'bool':
      // The runtime only ever hands booleans back, so this case is never
      // reached from a feed.
      return new ort.Tensor('bool', tensor.data as Uint8Array, tensor.dims);
  }
}
