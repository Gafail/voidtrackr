/**
 * Audio decoding and resampling.
 *
 * The browser and Node use different decoders, so they are wrapped in one
 * interface. Everything downstream of this file works on a mono
 * `Float32Array` at exactly {@link SAMPLE_RATE}.
 */

import { SAMPLE_RATE } from './constants';
import { resample, toMono } from './dsp';
import { detectSampleRate } from './sample-rate';

export interface DecodedAudio {
  channels: Float32Array[];
  sampleRate: number;
  duration: number;
}

/** True when we are running in a browser with the Web Audio API. */
export function hasWebAudio(): boolean {
  return typeof window !== 'undefined' && typeof window.AudioContext !== 'undefined';
}

/**
 * Decode an encoded audio file.
 *
 * In the browser this uses `AudioContext.decodeAudioData`, which handles
 * whatever the browser supports. Outside the browser a decoder has to be
 * supplied, because there is no Web Audio API to fall back on. The CLI
 * passes one in from `audio-node.ts`.
 */
export async function decodeAudio(
  data: ArrayBuffer,
  decoder?: AudioDecoder,
): Promise<DecodedAudio> {
  if (hasWebAudio()) {
    // Decode at the file's own rate when we can find it. The browser would
    // otherwise resample to the sound card's preferred rate, which means the
    // same file decodes differently on different machines.
    const rate = detectSampleRate(data) ?? undefined;
    const context = createContext(rate);
    try {
      const buffer = await context.decodeAudioData(data.slice(0));
      const channels: Float32Array[] = [];
      for (let c = 0; c < buffer.numberOfChannels; c++) {
        channels.push(Float32Array.from(buffer.getChannelData(c)));
      }
      return {
        channels,
        sampleRate: buffer.sampleRate,
        duration: buffer.duration,
      };
    } finally {
      void context.close();
    }
  }

  if (!decoder) {
    throw new Error(
      'No audio decoder available. In the browser this uses the Web Audio API; ' +
        'elsewhere pass a decoder explicitly.',
    );
  }

  return decoder(data);
}

/**
 * Create an AudioContext, preferring a particular sample rate.
 *
 * Not every device supports every rate, so a context that refuses the
 * request falls back to the default.
 */
function createContext(sampleRate?: number): AudioContext {
  if (sampleRate) {
    try {
      return new AudioContext({ sampleRate });
    } catch {
      // Fall through to the default context.
    }
  }
  return new AudioContext();
}

/** A function that turns encoded audio into channels of samples. */
export type AudioDecoder = (data: ArrayBuffer) => Promise<DecodedAudio>;

/**
 * Read the channel data out of whatever a decoder returned.
 *
 * `audio-decode` v3 gives `{ channelData, sampleRate }`; older versions give
 * a Web Audio style `AudioBuffer`.
 */
export function readChannels(buffer: unknown): Float32Array[] {
  const candidate = buffer as {
    channelData?: Float32Array[];
    numberOfChannels?: number;
    getChannelData?: (index: number) => Float32Array;
  };

  if (Array.isArray(candidate.channelData)) {
    return candidate.channelData.map((channel) => Float32Array.from(channel));
  }

  if (typeof candidate.getChannelData === 'function' && candidate.numberOfChannels) {
    const channels: Float32Array[] = [];
    for (let c = 0; c < candidate.numberOfChannels; c++) {
      channels.push(Float32Array.from(candidate.getChannelData(c)));
    }
    return channels;
  }

  throw new Error('The audio decoder returned an unexpected result');
}

/**
 * Fold to mono.
 *
 * Averaging is the right default, but if the two channels are strongly
 * anti-correlated the average would cancel most of the signal. That happens
 * with widened stereo, so in that case the strongest channel is used instead.
 */
export function foldToMono(channels: Float32Array[]): Float32Array {
  if (channels.length <= 1) return channels[0] ?? new Float32Array(0);

  let strongest = 0;
  let strongestEnergy = -1;
  const energies = channels.map((channel) => {
    let sum = 0;
    for (let i = 0; i < channel.length; i++) sum += channel[i] * channel[i];
    if (sum > strongestEnergy) {
      strongestEnergy = sum;
      strongest = channels.indexOf(channel);
    }
    return sum;
  });

  if (channels.length >= 2 && energies[0] > 0 && energies[1] > 0) {
    let cross = 0;
    const length = Math.min(channels[0].length, channels[1].length);
    for (let i = 0; i < length; i++) cross += channels[0][i] * channels[1][i];
    const correlation = cross / Math.sqrt(energies[0] * energies[1]);
    if (correlation < -0.15) {
      return channels[strongest];
    }
  }

  return toMono(channels);
}

/** Fold to mono and resample to the model's sample rate. */
export function prepareMono(audio: DecodedAudio, targetRate: number = SAMPLE_RATE): Float32Array {
  const mono = foldToMono(audio.channels);
  return resample(mono, audio.sampleRate, targetRate);
}
