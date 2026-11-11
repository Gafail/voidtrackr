/**
 * How much of a recording's energy sits in each frequency band, and how much
 * of it belongs to notes whose own fundamental is in that band.
 *
 *   npx tsx scripts/band-energy.mts "Music Import/1 - You Say When.wav"
 *
 * The bands are the ones the band-split proposal asks for. The point of the
 * second number is that energy in a band is not the same as notes in a band:
 * a note in the low register puts its upper harmonics in the middle and high
 * bands, and those harmonics are what the model reads as pitch.
 */

import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';

import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { SAMPLE_RATE } from '../src/core/constants';
import { spectrogram } from '../src/core/dsp';

const BANDS = [
  { name: '0-400', lo: 0, hi: 400 },
  { name: '400-1200', lo: 400, hi: 1200 },
  { name: '1200-2000', lo: 1200, hi: 2000 },
  { name: '2000+', lo: 2000, hi: SAMPLE_RATE / 2 },
];

/** The frequency of a MIDI note. */
function frequencyOf(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/** The MIDI notes whose own fundamental lies inside a band. */
function notesInBand(lo: number, hi: number): number[] {
  const notes: number[] = [];
  for (let midi = 21; midi <= 108; midi++) {
    const freq = frequencyOf(midi);
    if (freq >= lo && freq < hi) notes.push(midi);
  }
  return notes;
}

/**
 * The MIDI notes that have *any* harmonic inside a band.
 *
 * These are the notes whose energy the band carries, whether the model reads
 * it as a note in its own right or as part of a lower note.
 */
function notesWithHarmonicInBand(lo: number, hi: number): number[] {
  const notes: number[] = [];
  for (let midi = 21; midi <= 108; midi++) {
    const fundamental = frequencyOf(midi);
    for (let harmonic = 1; harmonic <= 16; harmonic++) {
      const freq = fundamental * harmonic;
      if (freq >= lo && freq < hi) {
        notes.push(midi);
        break;
      }
    }
  }
  return notes;
}

/** A short range label, e.g. "A0..67 (28..392 Hz)". */
function label(notes: number[]): string {
  if (notes.length === 0) return 'none';
  const first = notes[0];
  const last = notes[notes.length - 1];
  return `${first}..${last} (${frequencyOf(first).toFixed(0)}..${frequencyOf(last).toFixed(0)} Hz)`;
}

async function main(): Promise<void> {
  const input = process.argv[2];
  if (!input) {
    console.error('Usage: npx tsx scripts/band-energy.mts <audio file>');
    process.exit(1);
  }

  const data = await readFile(input);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);
  const spec = spectrogram(samples, { size: 2048, hop: 512, sampleRate: SAMPLE_RATE });

  const energy = BANDS.map(() => 0);
  let total = 0;

  for (let frame = 0; frame < spec.frames; frame++) {
    const base = frame * spec.bins;
    for (let bin = 1; bin < spec.bins; bin++) {
      const freq = bin * spec.binHz;
      if (freq > SAMPLE_RATE / 2) break;
      const power = spec.mags[base + bin] ** 2;
      total += power;
      for (let b = 0; b < BANDS.length; b++) {
        if (freq >= BANDS[b].lo && freq < BANDS[b].hi) {
          energy[b] += power;
          break;
        }
      }
    }
  }

  console.log(`${basename(input)}  ${(samples.length / SAMPLE_RATE).toFixed(1)}s\n`);
  console.log('band           energy    notes whose fundamental    notes with a harmonic');
  console.log('                        is inside the band          inside the band');
  for (let b = 0; b < BANDS.length; b++) {
    const band = BANDS[b];
    const share = (energy[b] / Math.max(1e-12, total)) * 100;
    console.log(
      `${band.name.padEnd(13)} ${share.toFixed(1).padStart(5)}%   ` +
        `${label(notesInBand(band.lo, band.hi)).padEnd(24)} ${label(
          notesWithHarmonicInBand(band.lo, band.hi),
        )}`,
    );
  }

  console.log(
    '\nThe third column is the notes a band could legitimately hold on its own.\n' +
      'The fourth is the notes whose energy it actually carries: a band above\n' +
      'about 1200 Hz is mostly harmonics of notes that live far below it.',
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
