/**
 * Compare two MIDI files note by note.
 *
 *   node scripts/compare-midi.mjs <a.mid> <b.mid>
 *
 * Used to check that two runs of the tool agree: the command line against the
 * browser, one model against another, or the same file before and after a
 * change to the pipeline.
 *
 * The comparison ignores the order notes are stored in and only looks at
 * pitch, start time, duration and velocity, so two files that carry the same
 * music in a different order still count as a match.
 */

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import * as ToneMidi from '@tonejs/midi';

const Midi = ToneMidi.Midi ?? ToneMidi.default?.Midi ?? ToneMidi.default;

function describe(path) {
  const raw = readFileSync(path);
  const midi = new Midi(raw);

  const notes = [];
  for (const track of midi.tracks) {
    for (const note of track.notes) {
      notes.push(
        `${track.name}|${note.midi}|${note.time.toFixed(4)}|${note.duration.toFixed(4)}|${Math.round(
          note.velocity,
        )}`,
      );
    }
  }

  const sorted = [...notes].sort();
  return {
    bytes: raw.length,
    hash: createHash('sha256').update(raw).digest('hex'),
    noteHash: createHash('sha256').update(sorted.join('|')).digest('hex'),
    tempo: midi.header.tempos[0]?.bpm,
    key: midi.header.keySignatures[0]
      ? `${midi.header.keySignatures[0].key} ${midi.header.keySignatures[0].scale}`
      : 'none',
    meta: midi.header.meta.map((event) => event.text),
    tracks: midi.tracks.map((track) => `${track.name}:${track.notes.length}`).join(', '),
    notes: sorted,
    count: notes.length,
  };
}

const [firstPath, secondPath] = process.argv.slice(2);
if (!firstPath || !secondPath) {
  console.error('Usage: node scripts/compare-midi.mjs <a.mid> <b.mid>');
  process.exit(1);
}

const first = describe(firstPath);
const second = describe(secondPath);

console.log(`${firstPath}`);
console.log(`  ${first.tracks} | ${first.count} notes | ${first.tempo} BPM | ${first.key} | ${first.bytes} bytes`);
console.log(`${secondPath}`);
console.log(`  ${second.tracks} | ${second.count} notes | ${second.tempo} BPM | ${second.key} | ${second.bytes} bytes`);

console.log('\nfile bytes identical :', first.hash === second.hash);
console.log('note sets identical  :', first.noteHash === second.noteHash);

if (first.meta.length || second.meta.length) {
  console.log('\nprovenance:');
  for (const stamp of new Set([...first.meta, ...second.meta])) console.log(`  ${stamp}`);
}

const firstSet = new Set(first.notes);
const secondSet = new Set(second.notes);
const onlyFirst = first.notes.filter((note) => !secondSet.has(note));
const onlySecond = second.notes.filter((note) => !firstSet.has(note));

if (onlyFirst.length) {
  console.log(`\nonly in the first file (${onlyFirst.length}):`);
  console.log('  ' + onlyFirst.slice(0, 20).join('\n  '));
}
if (onlySecond.length) {
  console.log(`\nonly in the second file (${onlySecond.length}):`);
  console.log('  ' + onlySecond.slice(0, 20).join('\n  '));
}

process.exit(first.noteHash === second.noteHash ? 0 : 1);
