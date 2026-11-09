/**
 * The Node audio decoder.
 *
 * This module is only imported by the CLI. Keeping it separate means the
 * browser build never pulls in a Node-only package.
 */

import { AudioDecoder, DecodedAudio, readChannels } from './audio';

/**
 * Decode audio using a pure-JavaScript decoder.
 *
 * This handles the formats the browser handles without needing `ffmpeg` or
 * any other system binary, which keeps the CLI portable.
 */
export const nodeAudioDecoder: AudioDecoder = async (data: ArrayBuffer): Promise<DecodedAudio> => {
  // `audio-decode` v3 is CommonJS, so the function arrives as the default
  // export under Node's ESM loader.
  const module = await import('audio-decode');
  const decode =
    (module as unknown as { default?: (data: ArrayBuffer) => Promise<unknown> }).default ??
    (module as unknown as { audioDecode?: (data: ArrayBuffer) => Promise<unknown> }).audioDecode;
  if (!decode) throw new Error('Could not load the audio decoder');

  const buffer = await decode(data);
  const channels = readChannels(buffer);
  const sampleRate = (buffer as { sampleRate: number }).sampleRate;

  return {
    channels,
    sampleRate,
    duration: channels.length > 0 ? channels[0].length / sampleRate : 0,
  };
};
