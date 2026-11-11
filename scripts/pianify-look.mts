/**
 * Look at the arrangement.
 *
 *   npx tsx scripts/pianify-look.mts [file.wav] [from] [to]
 *
 * Prints the arrangement as a text piano roll, so the shape of it can be
 * seen without a browser: the left hand in one mark, the right hand in
 * another, one column every fifty milliseconds.
 *
 * It is there because the numbers said the arrangement was within its
 * limits and it still did not look like a piano piece. A roll makes the
 * difference obvious in a way a table of averages does not.
 */

import { readFileSync } from 'node:fs';
import { basename } from 'node:path';

import { decodeAudio } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { prepareMono } from '../src/core/audio';
import { nodeOnnxRunner } from '../src/core/onnx-node';
import { PianoTranscriptionModel, PIANO_DIR, PIANO_FILES } from '../src/core/piano';
import { pianify } from '../src/core/pianify';
import { Track } from '../src/core/types';

const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const name = (pitch: number) => NAMES[pitch % 12] + (Math.floor(pitch / 12) - 1);

/** One column is this long. */
const COLUMN = 0.05;

/** The part of the recording to show. */
const FROM = Number(process.argv[3] ?? 0);
const TO = Number(process.argv[4] ?? 12);

async function main(): Promise<void> {
  const file = process.argv[2] ?? 'test/sample.wav';

  const data = readFileSync(file);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  const session = await nodeOnnxRunner.createSession(
    `../models/onnx/${PIANO_DIR}/${PIANO_FILES.graph}`.replace('../', ''),
    [{ location: `${PIANO_FILES.graph}.data`, url: `${PIANO_FILES.graph}.data` }],
  ).catch(async () => {
    const { join, resolve } = await import('node:path');
    const dir = resolve('models/onnx', PIANO_DIR);
    return nodeOnnxRunner.createSession(join(dir, PIANO_FILES.graph), [
      { location: `${PIANO_FILES.graph}.data`, url: join(dir, `${PIANO_FILES.graph}.data`) },
    ]);
  });
  const model = new PianoTranscriptionModel(session, 'local');
  const result = await model.transcribe(samples);

  const track: Track = {
    id: 'piano',
    name: 'Piano',
    kind: 'melody',
    notes: result.parts[0].notes,
  };

  // The configurations worth looking at side by side.
  const configurations = [
    { name: 'the shipped 0.6.0', options: { splitPoint: 74, moveRange: 12, minDuration: 0 } },
    { name: 'split at middle C', options: { splitPoint: 60, moveRange: 0, minDuration: 0.06 } },
    { name: 'middle C, a fourth across', options: { splitPoint: 60, moveRange: 4, minDuration: 0.06 } },
    { name: 'middle C, an octave across', options: { splitPoint: 60, moveRange: 12, minDuration: 0.06 } },
  ];

  for (const configuration of configurations) {
    const arranged = pianify([track], configuration.options);
    const left = arranged.tracks[0].notes;
    const right = arranged.tracks[1].notes;

    const all = [...left, ...right];
    if (all.length === 0) continue;

    const lowest = Math.min(...all.map((note) => note.pitchMidi));
    const highest = Math.max(...all.map((note) => note.pitchMidi));
    const columns = Math.ceil((TO - FROM) / COLUMN);

    console.log(`\n${configuration.name}`);
    console.log(
      `  left ${left.length} notes, right ${right.length} notes, ` +
        `${columns} columns of ${COLUMN * 1000} ms\n`,
    );

    const rows: string[][] = [];
    for (let pitch = highest; pitch >= lowest; pitch--) {
      const row = new Array<string>(columns).fill(' ');
      for (const note of left) {
        if (note.pitchMidi !== pitch) continue;
        const from = Math.round((note.startTime - FROM) / COLUMN);
        const to = Math.max(from + 1, Math.round((note.startTime + note.duration - FROM) / COLUMN));
        for (let i = from; i < Math.min(to, columns); i++) if (i >= 0) row[i] = 'o';
      }
      for (const note of right) {
        if (note.pitchMidi !== pitch) continue;
        const from = Math.round((note.startTime - FROM) / COLUMN);
        const to = Math.max(from + 1, Math.round((note.startTime + note.duration - FROM) / COLUMN));
        for (let i = from; i < Math.min(to, columns); i++) if (i >= 0) row[i] = '#';
      }
      rows.push([name(pitch).padEnd(3), ...row]);
    }

    for (const row of rows) {
      console.log(`  ${row[0]} ${row.slice(1).join('')}`);
    }
  }

  console.log('\n  o = left hand, # = right hand');
  void basename;
}

await main();
