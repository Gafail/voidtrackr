/**
 * One conversion, from audio to MIDI.
 *
 * The command line and the API both call this. That is the point: the tool
 * promises that the same input gives the same file whichever way it is
 * asked for, and the only way to keep that promise is to have one code
 * path rather than two that agree today.
 */

import { decodeAudio, prepareMono } from './audio';
import { nodeAudioDecoder } from './audio-node';
import { buildMidiFile } from './midi';
import { transcribe } from './pipeline';
import { FrequencyRange } from './range';
import { QuantiseOptions } from './quantise';
import { PianifyOptions } from './pianify';
import { TranscriptionModel } from './model';
import { ProgressInfo, TranscriptionResult } from './types';

/** What a caller asks for. */
export interface ConvertRequest {
  /** Keep only the notes inside this frequency range. */
  range?: FrequencyRange;
  /** Snap the notes to a grid of this many slots per second. */
  quantise?: QuantiseOptions;
  /** Arrange the notes for two hands. */
  pianify?: PianifyOptions;
  /** Look for a drum kit. */
  detectDrums: boolean;
  /** Force a tempo instead of estimating one. */
  forceTempo?: number;
  /** Told how far along the conversion is. */
  onProgress?: (progress: ProgressInfo) => void;
}

/** What a conversion produced. */
export interface ConvertOutput {
  /** The notes, grouped into tracks. */
  result: TranscriptionResult;
  /** The MIDI file. */
  midi: Uint8Array;
}

/**
 * Convert one recording.
 *
 * @param data       the audio file, exactly as it was uploaded.
 * @param request    what to do with it.
 * @param model      the model to use, already loaded.
 * @param provenance the stamp to write into the MIDI file.
 */
export async function convertAudio(
  data: ArrayBuffer,
  request: ConvertRequest,
  model: TranscriptionModel,
  provenance: string,
): Promise<ConvertOutput> {
  const audio = await decodeAudio(data, nodeAudioDecoder);
  const samples = prepareMono(audio);

  const result = await transcribe(samples, {
    model,
    detectDrums: request.detectDrums,
    forceTempo: request.forceTempo,
    range: request.range,
    quantise: request.quantise,
    pianify: request.pianify,
    onProgress: request.onProgress,
  });

  const midi = buildMidiFile(result.tracks, result.tempo, result.key, { provenance });

  return { result, midi };
}
