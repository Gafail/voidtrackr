/**
 * Generate a synthetic test recording so the pipeline can be checked without
 * shipping copyrighted audio.
 *
 *   node scripts/make-test-wav.mjs [output.wav]
 *
 * The signal is a simple arrangement: a melody on top, a bass line, and a
 * four-on-the-floor kick with hats, at 120 BPM in C major.
 */

import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const SAMPLE_RATE = 44100;
const BPM = 120;
const BEAT = 60 / BPM;

const OUTPUT = resolve(process.argv[2] ?? 'test/sample.wav');

/** MIDI note -> frequency. */
const hz = (midi) => 440 * Math.pow(2, (midi - 69) / 12);

/**
 * Write one note into the buffer.
 *
 * `kind` changes the timbre so the different parts are distinguishable:
 * a saw-ish tone for the melody, a sine for the bass, a pitch-dropping
 * sine for the kick, and filtered noise for the hats.
 */
function addNote(buffer, startSeconds, durationSeconds, midi, gain, kind) {
  const start = Math.floor(startSeconds * SAMPLE_RATE);
  const length = Math.floor(durationSeconds * SAMPLE_RATE);
  const frequency = hz(midi);

  for (let i = 0; i < length; i++) {
    const index = start + i;
    if (index < 0 || index >= buffer.length) continue;

    const t = i / SAMPLE_RATE;
    const progress = i / length;

    // Simple attack/decay envelope.
    const attack = Math.min(1, t / 0.01);
    const release = Math.min(1, (durationSeconds - t) / 0.05);
    const envelope = attack * Math.max(0, release);

    let value = 0;
    if (kind === 'bass') {
      value = Math.sin(2 * Math.PI * frequency * t);
    } else if (kind === 'kick') {
      const sweep = frequency * (1 + 6 * Math.exp(-t * 35));
      value = Math.sin(2 * Math.PI * sweep * t) * Math.exp(-t * 9);
    } else if (kind === 'snare') {
      // A tone with a noisy top, the way a real snare sits in the spectrum.
      const noise = (Math.random() * 2 - 1) * Math.exp(-t * 40);
      const body = Math.sin(2 * Math.PI * 190 * t) * Math.exp(-t * 25);
      value = noise * 0.8 + body * 0.5;
    } else if (kind === 'hat') {
      value = (Math.random() * 2 - 1) * Math.exp(-t * 70);
    } else {
      // A couple of harmonics with a soft sawtooth character.
      value =
        Math.sin(2 * Math.PI * frequency * t) +
        0.4 * Math.sin(4 * Math.PI * frequency * t) +
        0.2 * Math.sin(6 * Math.PI * frequency * t);
    }

    buffer[index] += value * envelope * gain;
  }
}

/** A deterministic pseudo-random source, so the output is reproducible. */
let seed = 12345;
Math.random = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};

const bars = 4;
const totalSeconds = bars * 4 * BEAT;
const buffer = new Float32Array(Math.floor(totalSeconds * SAMPLE_RATE));

// Melody: C4 E4 G4 A4 | C5 G4 E4 D4, one note per beat.
const melody = [
  [60, 64, 67, 69],
  [72, 67, 64, 62],
  [60, 64, 67, 72],
  [74, 71, 69, 67],
];
for (let bar = 0; bar < bars; bar++) {
  for (let beat = 0; beat < 4; beat++) {
    const start = (bar * 4 + beat) * BEAT;
    addNote(buffer, start, BEAT * 0.9, melody[bar][beat], 0.28, 'lead');
  }
}

// Bass: root notes on beats 1 and 3. The roots sit clear of the kick's
// pitch so the test measures transcription rather than how the two parts
// compete for the same register.
const roots = [43, 43, 48, 50];
for (let bar = 0; bar < bars; bar++) {
  for (const beat of [0, 2]) {
    const start = (bar * 4 + beat) * BEAT;
    addNote(buffer, start, BEAT * 1.8, roots[bar], 0.3, 'bass');
  }
}

// Drums: kick on every beat, snare on 2 and 4, hat on every eighth.
for (let bar = 0; bar < bars; bar++) {
  for (let beat = 0; beat < 4; beat++) {
    const at = (bar * 4 + beat) * BEAT;
    addNote(buffer, at, 0.25, 36, 0.5, 'kick');
    if (beat % 2 === 1) addNote(buffer, at, 0.2, 38, 0.34, 'snare');
    addNote(buffer, at + BEAT / 2, 0.12, 42, 0.2, 'hat');
  }
}

// Normalise and quantise to 16 bit.
let peak = 0;
for (const sample of buffer) peak = Math.max(peak, Math.abs(sample));
const scale = peak > 0 ? 0.85 / peak : 1;

const dataSize = buffer.length * 2;
const out = Buffer.alloc(44 + dataSize);
out.write('RIFF', 0);
out.writeUInt32LE(36 + dataSize, 4);
out.write('WAVE', 8);
out.write('fmt ', 12);
out.writeUInt32LE(16, 16);
out.writeUInt16LE(1, 20);
out.writeUInt16LE(1, 22);
out.writeUInt32LE(SAMPLE_RATE, 24);
out.writeUInt32LE(SAMPLE_RATE * 2, 28);
out.writeUInt16LE(2, 32);
out.writeUInt16LE(16, 34);
out.write('data', 36);
out.writeUInt32LE(dataSize, 40);
for (let i = 0; i < buffer.length; i++) {
  const clamped = Math.max(-1, Math.min(1, buffer[i] * scale));
  out.writeInt16LE(Math.round(clamped * 32767), 44 + i * 2);
}

await writeFile(OUTPUT, out);
console.log(`Wrote ${OUTPUT} (${totalSeconds.toFixed(1)}s, ${BPM} BPM)`);
