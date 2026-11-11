/**
 * How far is a transcription from something a pair of hands could play?
 *
 *   node scripts/pianify-stats.mjs eval/midi/*.piano.mid
 *   node scripts/pianify-stats.mjs --json eval/midi/*.piano.mid
 *
 * A piano is a machine with ten fingers, two hands and a reach of about an
 * octave. A transcription can be perfectly accurate and still be unplayable:
 * too many notes at once, notes on the same key a few milliseconds apart, a
 * chord that reaches further than a hand can, or a flurry faster than a wrist
 * can move.
 *
 * This counts those things. It is the measurement the pianify stage has to be
 * judged on, and the thing to compare against afterwards.
 *
 * The numbers:
 *
 *   retrig   — the same key struck again within 60 ms of the last one ending.
 *              A key cannot be re-struck that fast, so this is one note, not
 *              two.
 *   overlap  — the same key sounding at the same time as itself: a note
 *              counted twice.
 *   poly     — how many notes sound at once. Ten is the ceiling, and a hand
 *              can only reach five of them.
 *   cluster  — notes that start together, within 25 ms. This is what a chord
 *              is, and it is what gets spread across two hands.
 *   span     — how far apart the lowest and highest note of a cluster are.
 *              More than an octave and one hand cannot hold it.
 *   rate     — the most notes in any half second. A fast hand plays about
 *              ten a second; the whole piano manages perhaps twenty.
 */

import { readFileSync } from 'node:fs';
import * as ToneMidi from '@tonejs/midi';

const Midi = ToneMidi.Midi ?? ToneMidi.default?.Midi ?? ToneMidi.default;

/** Notes that start together, this close, are one chord. */
const CLUSTER_SECONDS = 0.025;

/** A key struck again within this long is the same note, not a new one. */
const RETRIGGER_SECONDS = 0.06;

/** The widest reach one hand is given. */
const HAND_SPAN = 12;

/** The most notes one hand is given to hold. */
const HAND_POLYPHONY = 5;

function readNotes(path) {
  const midi = new Midi(readFileSync(path));
  const notes = [];
  for (const track of midi.tracks) {
    // The drum kit is not played by fingers on keys, so it is left out: a
    // drum hit is a separate thing from a note on the keyboard.
    if (/drum/i.test(track.name ?? '')) continue;
    for (const note of track.notes) {
      notes.push({
        pitch: note.midi,
        start: note.time,
        end: note.time + note.duration,
        velocity: note.velocity ?? 0.8,
      });
    }
  }
  notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  return notes;
}

/** Group notes that start together into chords. */
function cluster(notes) {
  const groups = [];
  let current = [];
  let anchor = null;

  for (const note of notes) {
    if (anchor === null || note.start - anchor <= CLUSTER_SECONDS) {
      current.push(note);
      anchor = current.length === 1 ? note.start : anchor;
      continue;
    }
    groups.push(current);
    current = [note];
    anchor = note.start;
  }
  if (current.length > 0) groups.push(current);
  return groups;
}

/** The most notes sounding at the same time. */
function peakPolyphony(notes) {
  const events = [];
  for (const note of notes) {
    events.push({ time: note.start, delta: 1 });
    events.push({ time: note.end, delta: -1 });
  }
  // Endings come before beginnings at the same instant, so a note that ends
  // exactly when the next begins does not count as two at once.
  events.sort((a, b) => a.time - b.time || a.delta - b.delta);

  let current = 0;
  let peak = 0;
  const samples = [];
  for (const event of events) {
    current += event.delta;
    if (current > peak) peak = current;
    samples.push(current);
  }
  samples.sort((a, b) => a - b);
  const p95 = samples[Math.floor(samples.length * 0.95)] ?? 0;
  return { peak, p95 };
}

/** The most notes in any window of `window` seconds. */
function peakRate(notes, window) {
  if (notes.length === 0) return 0;
  let peak = 0;
  let head = 0;
  for (let i = 0; i < notes.length; i++) {
    while (notes[i].start - notes[head].start > window) head++;
    const count = i - head + 1;
    if (count > peak) peak = count;
  }
  return peak;
}

function analyse(path) {
  const notes = readNotes(path);
  if (notes.length === 0) return { path, notes: 0 };

  const duration = Math.max(...notes.map((note) => note.end));

  // --- The same key twice ------------------------------------------------
  let retriggers = 0;
  let retriggerNotes = 0;
  let overlaps = 0;
  const gaps = [];
  const byPitch = new Map();
  for (const note of notes) {
    const list = byPitch.get(note.pitch);
    if (list) list.push(note);
    else byPitch.set(note.pitch, [note]);
  }
  for (const list of byPitch.values()) {
    for (let i = 1; i < list.length; i++) {
      const previous = list[i - 1];
      const gap = list[i].start - previous.end;
      if (gap < -1e-6) {
        overlaps += 1;
        continue;
      }
      if (gap <= RETRIGGER_SECONDS) {
        retriggers += 1;
        retriggerNotes += 1;
        gaps.push(gap);
      }
    }
  }
  gaps.sort((a, b) => a - b);

  // --- Chords ------------------------------------------------------------
  const groups = cluster(notes);
  let spanViolations = 0;
  let polyViolations = 0;
  let clusterSize = 0;
  let widestSpan = 0;
  for (const group of groups) {
    clusterSize += group.length;
    const pitches = group.map((note) => note.pitch);
    const span = Math.max(...pitches) - Math.min(...pitches);
    if (span > widestSpan) widestSpan = span;
    if (span > HAND_SPAN) spanViolations += 1;
    if (group.length > HAND_POLYPHONY) polyViolations += 1;
  }

  const polyphony = peakPolyphony(notes);

  return {
    path,
    notes: notes.length,
    duration,
    perSecond: notes.length / Math.max(1, duration),
    retriggers,
    retriggerNotes,
    retriggerShare: retriggerNotes / notes.length,
    overlaps,
    peakPolyphony: polyphony.peak,
    p95Polyphony: polyphony.p95,
    clusters: groups.length,
    meanCluster: clusterSize / Math.max(1, groups.length),
    maxCluster: Math.max(...groups.map((g) => g.length)),
    spanViolations,
    polyViolations,
    widestSpan,
    peakRateHalf: peakRate(notes, 0.5),
    peakRateSecond: peakRate(notes, 1),
    gapMedian: gaps.length > 0 ? gaps[Math.floor(gaps.length / 2)] : 0,
    gapP90: gaps.length > 0 ? gaps[Math.floor(gaps.length * 0.9)] : 0,
  };
}

const args = process.argv.slice(2);
const asJson = args.includes('--json');
const paths = args.filter((arg) => !arg.startsWith('--'));
if (paths.length === 0) {
  console.error('Usage: node scripts/pianify-stats.mjs [--json] <midi files...>');
  process.exit(1);
}

const results = paths.map(analyse);

if (asJson) {
  console.log(JSON.stringify(results, null, 2));
  process.exit(0);
}

console.log(
  'file                              notes    /s  retrig  over  poly  p95  cl(mean/max)  span>12  poly>5  rate/0.5s',
);
for (const r of results) {
  if (r.notes === 0) {
    console.log(`${r.path.replace(/^.*[/\\]/, '')}  (no notes)`);
    continue;
  }
  console.log(
    `${r.path.replace(/^.*[/\\]/, '').padEnd(34)} ${String(r.notes).padStart(5)} ` +
      `${r.perSecond.toFixed(1).padStart(5)} ` +
      `${String(r.retriggers).padStart(6)} ` +
      `${String(r.overlaps).padStart(5)} ` +
      `${String(r.peakPolyphony).padStart(4)} ` +
      `${String(r.p95Polyphony).padStart(4)} ` +
      `${`${r.meanCluster.toFixed(1)}/${r.maxCluster}`.padStart(11)} ` +
      `${String(r.spanViolations).padStart(7)} ` +
      `${String(r.polyViolations).padStart(6)} ` +
      `${String(r.peakRateHalf).padStart(8)}`,
  );
}

// The totals, over everything that was measured.
const total = results.reduce(
  (sum, r) => ({
    notes: sum.notes + (r.notes ?? 0),
    retriggers: sum.retriggers + (r.retriggers ?? 0),
    overlaps: sum.overlaps + (r.overlaps ?? 0),
    span: sum.span + (r.spanViolations ?? 0),
    poly: sum.poly + (r.polyViolations ?? 0),
    clusters: sum.clusters + (r.clusters ?? 0),
    meanCluster:
      sum.meanCluster * sum.clusters + (r.meanCluster ?? 0) * (r.clusters ?? 0),
    peak: Math.max(sum.peak, r.peakPolyphony ?? 0),
    rate: Math.max(sum.rate, r.peakRateHalf ?? 0),
    duration: sum.duration + (r.duration ?? 0),
  }),
  {
    notes: 0,
    retriggers: 0,
    overlaps: 0,
    span: 0,
    poly: 0,
    clusters: 0,
    meanCluster: 0,
    peak: 0,
    rate: 0,
    duration: 0,
  },
);

console.log(
  `\n${results.length} files: ${total.notes} notes over ${total.duration.toFixed(0)} s ` +
    `(${((total.notes / Math.max(1, total.duration)) * 1).toFixed(1)}/s), ` +
    `${total.retriggers} re-triggers (${((total.retriggers / Math.max(1, total.notes)) * 100).toFixed(1)}% of notes), ` +
    `${total.overlaps} overlaps, ${total.span} chords too wide for a hand, ` +
    `${total.poly} chords too big for a hand, ` +
    `worst polyphony ${total.peak}, worst half-second ${total.rate} notes, ` +
    `mean chord ${(total.meanCluster / Math.max(1, total.clusters)).toFixed(2)} notes.`,
);
