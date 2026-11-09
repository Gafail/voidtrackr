/**
 * The Node ONNX runtime, behind the shared interface.
 *
 * Only the command line uses this. The browser has its own adapter, so
 * neither runtime leaks into the shared pipeline code.
 */

import * as ort from 'onnxruntime-node';

import { DETERMINISTIC_SESSION_OPTIONS, OnnxRunner, OnnxTensor } from './onnx';

export const nodeOnnxRunner: OnnxRunner = {
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
          tensors[name] = new ort.Tensor('float32', tensor.data, tensor.dims);
        }
        const result = await session.run(tensors);
        const output: Record<string, OnnxTensor> = {};
        for (const [name, tensor] of Object.entries(result)) {
          output[name] = { data: tensor.data as Float32Array, dims: tensor.dims };
        }
        return output;
      },    };
  },
};
