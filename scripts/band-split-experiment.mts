/**
 * Would it help to transcribe the audio in three frequency bands and combine
 * the results?
 *
 *   npx tsx scripts/band-split-experiment.mts "Music Import/1 - You Say When.wav" [outDir]
 *
 * The idea under test is that a model given less to listen to has less to be
 * confused by: run it over 0-400 Hz, then 400-1200 Hz, then 1200-2000 Hz,
 * and put the three note lists together.
 *
 * This script measures that. It runs the piano model over the whole mix and
 * over each band on its own, writes a MIDI file for every run, and merges
 * the three band runs two ways:
 *
 *   - **naive** — every note from every band, as it comes;
 *   - **gated** — only the notes whose own fundamental frequency falls
 *     inside the band that produced them.
 *
 * Each output is then scored against a reference transcription:
 *
 *   node scripts/evaluate-midi.mjs reference.mid estimate.mid
 */

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { basename, extname, join, resolve } from 'node:path';

import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { SAMPLE_RATE } from '../src/core/constants';
import { resample, splitBands } from '../src/core/dsp';
import { buildMidiFile } from '../src/core/midi';
import { nodeOnnxRunner } from '../src/core/onnx-node';
import { PianoTranscriptionModel, PIANO_DIR, PIANO_SAMPLE_RATE } from '../src/core/piano';
import { estimateKey, estimateTempo } from '../src/core/tempo';
import { correctTuning } from '../src/core/tuning';
import { RawNote } from '../src/core/types';

/** The bands the proposal asks for. */
const BANDS = [
  { name: 'low', lo: 0, hi: 400 },
  { name: 'mid', lo: 400, hi: 1200 },
  { name: 'high', lo: 1200, hi: 2000 },
] as const;

/** Where the weights live. */
const MODEL_DIR = resolve('models', 'onnx');

/** The frequency of a MIDI note. */
function frequencyOf(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/**
 * Cut one band out of a signal.
 *
 * The filters are applied forwards and backwards, so the attacks do not move.
 */
function cutBand(
  samples: Float32Array,
  lo: number,
  hi: number,
  sampleRate: number,
): Float32Array {
  const above = lo > 0 ? splitBands(samples, lo, sampleRate).high : samples;
  return splitBands(above, hi, sampleRate).low;
}

/** Run the piano model over one signal. */
async function transcribe(
  model: PianoTranscriptionModel,
  samples: Float32Array,
): Promise<RawNote[]> {
  const result = await model.transcribe(samples);
  return result.parts.flatMap((part) => part.notes);
}

/** Write a note list to a MIDI file. */
async function writeMidi(path: string, notes: RawNote[], duration: number): Promise<void> {
  const tempo = estimateTempo(notes, duration);
  const key = estimateKey(notes);
  const bytes = buildMidiFile(
    [{ id: 'piano', name: 'Piano', kind: 'melody', notes }],
    tempo,
    key,
  );
  await writeFile(path, bytes);
}

/** Describe the notes a run produced. */
function describe(notes: RawNote[]): string {
  if (notes.length === 0) return 'no notes';
  const pitches = notes.map((note) => note.pitchMidi).sort((a, b) => a - b);
  const median = pitches[Math.floor(pitches.length / 2)];
  return (
    `${notes.length} notes, pitches ${pitches[0]}..${pitches[pitches.length - 1]} ` +
    `(median ${median}, ${frequencyOf(median).toFixed(0)} Hz)`
  );
}

/** How many notes fall outside the band that produced them. */
function outsideBand(notes: RawNote[], lo: number, hi: number): number {
  return notes.filter((note) => {
    const freq = frequencyOf(note.pitchMidi);
    return freq < lo || freq >= hi;
  }).length;
}

async function main(): Promise<void> {
  const input = process.argv[2];
  if (!input) {
    console.error(
      'Usage: npx tsx scripts/band-split-experiment.mts <audio file> [out dir]',
    );
    process.exit(1);
  }
  const outDir = resolve(process.argv[3] ?? join('eval', 'band-split'));
  await mkdir(outDir, { recursive: true });

  const stem = basename(input, extname(input)).replace(/[^a-z0-9]+/gi, '-');

  // Decode, fold to mono, correct the tuning and drop to the model's rate —
  // the same steps the pipeline takes, so the full-mix run here is the same
  // transcription the eval produced.
  const data = await readFile(input);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const mono = prepareMono(audio);
  const tuning = correctTuning(mono);
  const atModelRate = resample(tuning.samples, SAMPLE_RATE, PIANO_SAMPLE_RATE);
  const duration = atModelRate.length / PIANO_SAMPLE_RATE;

  console.log(`${basename(input)}  ${duration.toFixed(1)}s  @ ${PIANO_SAMPLE_RATE} Hz`);
  if (tuning.applied) {
    console.log(`  tuning: ${tuning.cents.toFixed(0)} cents, corrected`);
  }

  const model = await PianoTranscriptionModel.load(nodeOnnxRunner, join(MODEL_DIR, PIANO_DIR));

  // --- The full mix, the baseline --------------------------------------
  console.log('\nfull mix');
  const full = await transcribe(model, atModelRate);
  console.log(`  ${describe(full)}`);
  await writeMidi(join(outDir, `${stem}.full.mid`), full, duration);

  // --- Each band on its own --------------------------------------------
  const perBand: { name: string; notes: RawNote[]; lo: number; hi: number }[] = [];
  for (const band of BANDS) {
    console.log(`\nband ${band.lo}-${band.hi} Hz`);
    const cut = cutBand(atModelRate, band.lo, band.hi, PIANO_SAMPLE_RATE);
    const notes = await transcribe(model, cut);
    const outside = outsideBand(notes, band.lo, band.hi);
    console.log(`  ${describe(notes)}`);
    console.log(
      `  ${outside} of them (${((outside / Math.max(1, notes.length)) * 100).toFixed(0)}%) ` +
        `have a fundamental outside ${band.lo}-${band.hi} Hz`,
    );
    await writeMidi(join(outDir, `${stem}.${band.name}.mid`), notes, duration);
    perBand.push({ name: band.name, notes, lo: band.lo, hi: band.hi });
  }

  // --- Merging the bands -----------------------------------------------
  const naive = perBand.flatMap((band) => band.notes);
  console.log(`\nmerged, every note kept: ${describe(naive)}`);
  await writeMidi(join(outDir, `${stem}.merged-naive.mid`), naive, duration);

  const gated = perBand.flatMap((band) =>
    band.notes.filter((note) => {
      const freq = frequencyOf(note.pitchMidi);
      return freq >= band.lo && freq < band.hi;
    }),
  );
  console.log(`\nmerged, gated by the band's own range: ${describe(gated)}`);
  await writeMidi(join(outDir, `${stem}.merged-gated.mid`), gated, duration);

  // How much the bands agree with each other: the same note turning up in
  // more than one run is the merge problem in one number.
  console.log('\nsame note in more than one band:');
  for (let i = 0; i < perBand.length; i++) {
    for (let j = i + 1; j < perBand.length; j++) {
      const a = perBand[i];
      const b = perBand[j];
      let same = 0;
      for (const note of a.notes) {
        if (
          b.notes.some(
            (other) => other.pitchMidi === note.pitchMidi && Math.abs(other.startTime - note.startTime) < 0.05,
          )
        ) {
          same += 1;
        }
      }
      console.log(`  ${a.name} and ${b.name}: ${same} notes`);
    }
  }

  console.log(`\nMIDI files written to ${outDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
