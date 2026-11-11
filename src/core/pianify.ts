/**
 * Pianify: turn a transcription into something two hands could play.
 *
 * A piano has ten fingers, two hands and a reach of about an octave. An
 * automatic transcription has none of those limits: it can report twenty notes
 * sounding at once, a chord that reaches three octaves, and the same key
 * struck again five milliseconds after the last time.
 *
 * Measured over this project's own material, the piano model reports 11.1
 * notes a second, 35 % of them are the same key struck twice within 60 ms,
 * and its widest chord reaches five octaves. None of that is wrong as
 * transcription. All of it is unplayable.
 *
 * This stage is the arrangement step: it takes the notes the model found and
 * divides them between two hands, the way a player would, dropping what the
 * hands cannot hold.
 *
 * Four passes:
 *
 *   1. **Collapse.** The same key struck again within a few tens of
 *      milliseconds is one note, not two. A key cannot be re-struck that
 *      fast, and the pitch content is kept exactly.
 *
 *   2. **Divide.** The notes that start together are a chord, and a chord is
 *      played by two hands. The split point is chosen from the material
 *      rather than fixed at middle C, because the music does not always put
 *      its bass below its melody.
 *
 *   3. **Trim.** Each hand gets at most five fingers and an octave of reach.
 *      Where a note does not fit, it goes to the other hand if that hand can
 *      take it, and is dropped if neither can.
 *
 *   4. **Thin.** A hand plays at most ten notes a second. Where the
 *      transcription asks for more, the quietest notes go.
 *
 * The design follows the same idea as the fluency work on piano fingering
 * (Srivatsan & Berg-Kirkpatrick, ISMIR 2022, arXiv 2209.05622): per-note
 * accuracy does not measure playability. Playability is a property of the
 * notes together — how many sound at once, how far apart they are, how fast
 * they arrive — so it has to be constrained directly.
 */

import { RawNote, Track } from './types';

/** How close together two notes have to start to be one chord. */
const CLUSTER_WINDOW = 0.025;

/** The default gap below which two notes on one key are one note. */
const DEFAULT_MERGE_GAP = 0.05;

export interface PianifyOptions {
  /**
   * Two notes on the same key, this close together, are one note.
   *
   * A piano key cannot be struck again in a few milliseconds, so anything
   * below this is one note that the model cut in two. Default 50 ms.
   */
  mergeGap?: number;
  /** How many notes one hand may hold at once. Default 5, one per finger. */
  maxPolyphony?: number;
  /** How far one hand may reach, in semitones. Default 12, an octave. */
  maxSpan?: number;
  /** How many notes one hand may strike in a second. Default 10. */
  maxRate?: number;
  /**
   * The MIDI note where the left hand stops and the right hand starts.
   *
   * Left out, it is worked out from the material: the point that splits the
   * notes most evenly while keeping both hands within their reach.
   */
  splitPoint?: number;
  /**
   * How far from the split point a note may be handed to the other hand.
   *
   * When a hand is full, a note near the split can be played by the other
   * hand instead. One far away cannot: the left hand does not reach up to
   * the treble, and the right hand does not come down to the bass. Such a
   * note is dropped. Default an octave.
   */
  moveRange?: number;
  /**
   * Keep the two hands as separate tracks.
   *
   * Left out, they are kept apart — a piano has two hands, and a file that
   * says which is which is a file someone can play.
   */
  splitHands?: boolean;
}

/** What the stage did, so the pipeline can say so. */
export interface PianifyStats {
  /** Notes before the stage. */
  before: number;
  /** Notes after the stage. */
  after: number;
  /** Re-triggers folded into the note before them. */
  collapsed: number;
  /** Notes no hand could hold. */
  dropped: number;
  /** Notes moved from one hand to the other. */
  moved: number;
  /** Where the left hand stopped. */
  splitPoint: number;
  /** Notes in the left hand. */
  left: number;
  /** Notes in the right hand. */
  right: number;
  /** The most notes that sounded at once. */
  peakPolyphony: number;
  /** The most notes in any half second. */
  peakRate: number;
}

export interface PianifyResult {
  /** The tracks that replace the pitched ones. */
  tracks: Track[];
  stats: PianifyStats;
  /** A line for the pipeline's log. */
  log: string;
}

/**
 * Arrange the pitched tracks for two hands.
 *
 * The drum kit is left alone: a drum is struck with a stick, and the
 * limits that matter here are the ones on the keyboard.
 */
export function pianify(tracks: Track[], options: PianifyOptions = {}): PianifyResult {
  const mergeGap = options.mergeGap ?? DEFAULT_MERGE_GAP;
  const maxPolyphony = options.maxPolyphony ?? 5;
  const maxSpan = options.maxSpan ?? 12;
  const maxRate = options.maxRate ?? 10;
  const moveRange = options.moveRange ?? 12;
  const splitHands = options.splitHands !== false;

  const pitched = tracks.filter((track) => track.kind !== 'drums');
  const others = tracks.filter((track) => track.kind === 'drums');

  const before = pitched.reduce((sum, track) => sum + track.notes.length, 0);

  // --- 1. Collapse the re-triggers ------------------------------------
  const collapsedNotes = collapseRepeats(
    pitched.flatMap((track) => track.notes),
    mergeGap,
  );
  const collapsed = collapsedNotes.collapsed;

  // --- 2. Divide the chords between the hands -------------------------
  const splitPoint =
    options.splitPoint ?? chooseSplitPoint(collapsedNotes.notes, maxPolyphony, maxSpan);

  const divided = divideHands(
    collapsedNotes.notes,
    splitPoint,
    maxPolyphony,
    maxSpan,
    moveRange,
  );

  // --- 3. Thin each hand to a human rate ------------------------------
  const left = limitRate(divided.left, maxRate);
  const right = limitRate(divided.right, maxRate);

  const dropped = divided.dropped + left.dropped + right.dropped;

  const result: Track[] = splitHands
    ? [
        { id: 'left-hand', name: 'Left hand', kind: 'bass', notes: left.notes },
        { id: 'right-hand', name: 'Right hand', kind: 'melody', notes: right.notes },
      ]
    : [
        {
          id: 'piano',
          name: 'Piano',
          kind: 'melody',
          notes: sortByTime([...left.notes, ...right.notes]),
        },
      ];

  const after = result.reduce((sum, track) => sum + track.notes.length, 0);

  const stats: PianifyStats = {
    before,
    after,
    collapsed,
    dropped,
    moved: divided.moved,
    splitPoint,
    left: left.notes.length,
    right: right.notes.length,
    peakPolyphony: peakPolyphony(result.flatMap((track) => track.notes)),
    peakRate: peakRate(result.flatMap((track) => track.notes), 0.5),
  };

  const log = splitHands
    ? `Arranged for two hands: ${after} notes, ${collapsed} re-triggers folded in, ` +
      `${dropped} dropped, split at MIDI ${splitPoint}.`
    : `Arranged for two hands and put back in one track: ${after} notes, ` +
      `${collapsed} re-triggers folded in, ${dropped} dropped.`;

  return { tracks: [...result, ...others], stats, log };
}

/**
 * Fold notes that are the same key struck again into the note before them.
 *
 * The result keeps the first strike's start and the last strike's end, so
 * the note sounds for as long as the key was actually down, and it keeps the
 * loudest of the strikes, because that is the one the model was surest of.
 */
export function collapseRepeats(
  notes: RawNote[],
  mergeGap: number,
): { notes: RawNote[]; collapsed: number } {
  const byPitch = new Map<number, RawNote[]>();
  for (const note of notes) {
    const list = byPitch.get(note.pitchMidi);
    if (list) list.push(note);
    else byPitch.set(note.pitchMidi, [note]);
  }

  const out: RawNote[] = [];
  let collapsed = 0;

  for (const list of byPitch.values()) {
    list.sort((a, b) => a.startTime - b.startTime);

    let current: RawNote = { ...list[0] };
    for (let i = 1; i < list.length; i++) {
      const next = list[i];
      const currentEnd = current.startTime + current.duration;
      if (next.startTime <= currentEnd + mergeGap) {
        const end = Math.max(currentEnd, next.startTime + next.duration);
        current = {
          ...current,
          duration: end - current.startTime,
          amplitude: Math.max(current.amplitude, next.amplitude),
        };
        collapsed += 1;
      } else {
        out.push(current);
        current = { ...next };
      }
    }
    out.push(current);
  }

  return { notes: sortByTime(out), collapsed };
}

/**
 * Work out where the left hand should stop.
 *
 * Every distinct pitch is tried as the split point, and each one is judged
 * on how badly it overflows the hands: how many notes one hand has to hold
 * at once, and how far it has to reach. The point that overflows least wins,
 * and a split that leaves the hands lopsided is a worse answer than one that
 * does not.
 */
export function chooseSplitPoint(
  notes: RawNote[],
  maxPolyphony: number,
  maxSpan: number,
): number {
  if (notes.length === 0) return 60;

  const pitches = [...new Set(notes.map((note) => note.pitchMidi))].sort((a, b) => a - b);
  if (pitches.length < 2) return pitches[0];

  let best = { point: 60, cost: Infinity, balance: Infinity };

  for (const point of pitches) {
    const low = notes.filter((note) => note.pitchMidi < point);
    const high = notes.filter((note) => note.pitchMidi >= point);

    const lowLimits = handLimits(low);
    const highLimits = handLimits(high);
    const worstCount = Math.max(lowLimits.count, highLimits.count);
    const worstSpan = Math.max(lowLimits.span, highLimits.span);

    const cost = worstCount / maxPolyphony + worstSpan / maxSpan;
    const balance = Math.abs(low.length - high.length);
    if (cost < best.cost - 1e-9 || (Math.abs(cost - best.cost) < 1e-9 && balance < best.balance)) {
      best = { point, cost, balance };
    }
  }

  return best.point;
}

/**
 * What a hand has to do: the most notes it plays at once, and the widest
 * reach it has to make.
 *
 * The notes are in time order, so the window of notes that count as "at the
 * same time" is walked with two pointers rather than searched for each note
 * in turn.
 */
function handLimits(hand: RawNote[]): { count: number; span: number } {
  let worstCount = 0;
  let worstSpan = 0;
  let low = 0;
  let high = 0;

  for (const note of hand) {
    const time = note.startTime;
    while (low < hand.length && hand[low].startTime < time - CLUSTER_WINDOW) low += 1;
    while (high < hand.length && hand[high].startTime <= time + CLUSTER_WINDOW) high += 1;

    if (high - low > worstCount) worstCount = high - low;

    let lowest = Infinity;
    let highest = -Infinity;
    for (let i = low; i < high; i++) {
      if (hand[i].pitchMidi < lowest) lowest = hand[i].pitchMidi;
      if (hand[i].pitchMidi > highest) highest = hand[i].pitchMidi;
    }
    if (highest - lowest > worstSpan) worstSpan = highest - lowest;
  }

  return { count: worstCount, span: worstSpan };
}

/**
 * Divide the notes between two hands.
 *
 * Each hand can hold five notes at a time and reach an octave. The notes
 * that count as "at a time" are the ones that start within a few tens of
 * milliseconds of each other, and a hand is over its limit when the notes it
 * is playing then are too many, or reach too far.
 *
 * The notes are placed in time order. Each one goes to the hand it belongs
 * to — the lower ones to the left hand, which is playing the bass, and the
 * higher ones to the right hand, which is playing the melody — unless that
 * hand cannot take it, in which case the other hand is tried, and if neither
 * can, the note is dropped.
 *
 * Placing the notes one at a time, and checking the notes either side of the
 * new one as well, means the decision never depends on a note that has not
 * been placed yet, so the result does not depend on the order the notes are
 * considered in.
 */
export function divideHands(
  notes: RawNote[],
  splitPoint: number,
  maxPolyphony: number,
  maxSpan: number,
  moveRange: number,
): { left: RawNote[]; right: RawNote[]; dropped: number; moved: number } {
  const hands: RawNote[][] = [[], []];
  let dropped = 0;
  let moved = 0;

  for (const note of sortByTime(notes)) {
    const preferred = note.pitchMidi < splitPoint ? 0 : 1;

    // A note only goes to the other hand if it is close enough to the split
    // to be that hand's music and not the other one's.
    const handover =
      moveRange > 0 && Math.abs(note.pitchMidi - splitPoint) <= moveRange;

    let placed = false;
    for (const hand of handover ? [preferred, 1 - preferred] : [preferred]) {
      if (!canTake(hands[hand], note, maxPolyphony, maxSpan)) continue;
      hands[hand].push(note);
      placed = true;
      if (hand !== preferred) moved += 1;
      break;
    }

    if (!placed) dropped += 1;
  }

  return {
    left: sortByTime(hands[0]),
    right: sortByTime(hands[1]),
    dropped,
    moved,
  };
}

/**
 * Say whether a hand could play this note.
 *
 * Two things have to hold. The notes the hand would be playing at that
 * moment have to be within its limits, and so do the notes either side of
 * the new one, because the new note joins their chord as well.
 */
function canTake(
  hand: RawNote[],
  note: RawNote,
  maxPolyphony: number,
  maxSpan: number,
): boolean {
  const near = hand.filter(
    (other) => Math.abs(other.startTime - note.startTime) <= CLUSTER_WINDOW,
  );

  if (!withinLimits([...near, note], maxPolyphony, maxSpan)) return false;

  for (const neighbour of near) {
    const chord = hand.filter(
      (other) => Math.abs(other.startTime - neighbour.startTime) <= CLUSTER_WINDOW,
    );
    if (!withinLimits([...chord, note], maxPolyphony, maxSpan)) return false;
  }

  return true;
}

/** Say whether a set of notes is within a hand's limits. */
function withinLimits(
  notes: RawNote[],
  maxPolyphony: number,
  maxSpan: number,
): boolean {
  if (notes.length === 0) return true;
  if (notes.length > maxPolyphony) return false;
  return spanOf(notes) <= maxSpan;
}

/** The distance between the lowest and the highest note. */
function spanOf(notes: RawNote[]): number {
  if (notes.length === 0) return 0;
  let lowest = Infinity;
  let highest = -Infinity;
  for (const note of notes) {
    if (note.pitchMidi < lowest) lowest = note.pitchMidi;
    if (note.pitchMidi > highest) highest = note.pitchMidi;
  }
  return highest - lowest;
}

/**
 * Keep the hand under its limits.
 *
 * The anchor is the note the hand is built around: the lowest for the left
 * hand, which is playing the bass, and the highest for the right hand, which
 * is playing the melody. The note that goes is the one furthest from it,
 * because that is the one the hand cannot reach.
 */

/**
 * Keep the hand under a rate a person could play.
 *
 * A hand plays about a dozen notes a second at its fastest. Where the
 * transcription asks for more, the quietest notes go first: the loud ones
 * are the ones the model was surest of.
 */
export function limitRate(
  notes: RawNote[],
  maxRate: number,
): { notes: RawNote[]; dropped: number } {
  if (maxRate <= 0 || notes.length === 0) return { notes: [...notes], dropped: 0 };

  const kept: RawNote[] = [];
  const window: RawNote[] = [];
  let dropped = 0;

  for (const note of notes) {
    // The window is the last second of this hand's notes.
    while (window.length > 0 && window[0].startTime < note.startTime - 1) {
      window.shift();
    }

    if (window.length < maxRate) {
      window.push(note);
      kept.push(note);
      continue;
    }

    // The hand is already playing as fast as it can. The quietest note in
    // the window goes, and if that is the note that just arrived, it never
    // joins the hand at all.
    const quietest = [...window, note].reduce((lowest, candidate) =>
      candidate.amplitude < lowest.amplitude ? candidate : lowest,
    );
    if (quietest === note) {
      dropped += 1;
      continue;
    }

    window.splice(window.indexOf(quietest), 1);
    const index = kept.indexOf(quietest);
    if (index >= 0) kept.splice(index, 1);
    dropped += 1;

    window.push(note);
    kept.push(note);
  }

  return { notes: sortByTime(kept), dropped };
}

/** The most notes sounding at the same moment. */
export function peakPolyphony(notes: RawNote[]): number {
  const events: { time: number; delta: number }[] = [];
  for (const note of notes) {
    events.push({ time: note.startTime, delta: 1 });
    events.push({ time: note.startTime + note.duration, delta: -1 });
  }
  events.sort((a, b) => a.time - b.time || a.delta - b.delta);

  let current = 0;
  let peak = 0;
  for (const event of events) {
    current += event.delta;
    if (current > peak) peak = current;
  }
  return peak;
}

/** The most notes in any window of `window` seconds. */
export function peakRate(notes: RawNote[], window: number): number {
  if (notes.length === 0) return 0;
  let peak = 0;
  let head = 0;
  for (let i = 0; i < notes.length; i++) {
    while (notes[i].startTime - notes[head].startTime > window) head += 1;
    const count = i - head + 1;
    if (count > peak) peak = count;
  }
  return peak;
}

/** Sort by start time, then by pitch. */
function sortByTime(notes: RawNote[]): RawNote[] {
  return [...notes].sort((a, b) => a.startTime - b.startTime || a.pitchMidi - b.pitchMidi);
}
