/**
 * The Node ONNX runtime, behind the shared interface.
 *
 * Only the command line uses this. The browser has its own adapter, so
 * neither runtime leaks into the shared pipeline code.
 */

import * as ort from 'onnxruntime-node';
import { readFile } from 'node:fs/promises';

import {
  DETERMINISTIC_SESSION_OPTIONS,
  OnnxDataType,
  OnnxRunner,
  OnnxTensor,
} from './onnx';

export const nodeOnnxRunner: OnnxRunner = {
  /**
   * Load a graph.
   *
   * The external data is not passed on: Node's runtime finds the sidecar
   * file by itself, because it loads the graph from a path and keeps the
   * directory it came from.
   */
  async createSession(path: string) {
    const session = await ort.InferenceSession.create(path, {
      ...DETERMINISTIC_SESSION_OPTIONS,
    });

    return {
      inputNames: session.inputNames,
      outputNames: session.outputNames,
      async run(feeds: Record<string, OnnxTensor>) {
        const tensors: Record<string, ort.Tensor> = {};
        for (const [name, tensor] of Object.entries(feeds)) {
          tensors[name] = toOrtTensor(tensor);
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

  /** Read a file that sits next to the model, such as a vocabulary. */
  async read(path: string): Promise<ArrayBuffer> {
    const bytes = await readFile(path);
    return bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    ) as ArrayBuffer;
  },
};

/**
 * Build a runtime tensor.
 *
 * The runtime wants a different array type for each element type, and 64-bit
 * integers are BigInt rather than number, so the cases are spelled out rather
 * than cast into one.
 */
function toOrtTensor(tensor: OnnxTensor): ort.Tensor {
  switch (tensor.type) {
    case 'float32':
      return new ort.Tensor('float32', tensor.data as Float32Array, tensor.dims);
    case 'int32':
      return new ort.Tensor('int32', tensor.data as Int32Array, tensor.dims);
    case 'int64':
      return new ort.Tensor('int64', tensor.data as BigInt64Array, tensor.dims);
  }
}
