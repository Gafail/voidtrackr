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
 *      played by two hands. The hands meet at middle C, which is where the
 *      grand staff splits, and the notes either side of it go to the hand
 *      they belong to.
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

/**
 * Where the left hand stops and the right hand starts.
 *
 * Middle C is the answer because it is the answer for a piano, not because it
 * fitted this material best. The grand staff splits there: the bass clef goes
 * to the left hand and the treble clef to the right.
 *
 * The first version worked the split out from the material instead, choosing
 * the point that balanced the load between the hands. It was measured, and it
 * was wrong in a way that is obvious on the piano roll: the median of this
 * material's texture sits at MIDI 57 to 76, well above middle C, so a split
 * that balances the load lands in the treble. The left hand ended up playing
 * the melody up to MIDI 87 and the right hand played the bass down to 46.
 */
const DEFAULT_SPLIT_POINT = 60;

/** The shortest note a hand can play, in seconds. */
const DEFAULT_MIN_DURATION = 0.06;

/**
 * How far a note may cross to the other hand, in semitones.
 *
 * The hands meet at middle C and they overlap a little, the way they do on a
 * grand staff: the right hand sometimes comes down below the split, the left
 * hand sometimes reaches up over it. A third is the usual overlap.
 *
 * The first version allowed a note to cross an octave, and that is what put
 * the melody into the left hand's track and the bass into the right hand's.
 * At a third, about one note in fifty changes hands.
 */
const DEFAULT_MOVE_RANGE = 4;

export interface PianifyOptions {
  /**
   * The shortest note a hand can play, in seconds.
   *
   * The piano model works in frames of ten milliseconds, so a quarter of the
   * notes it reports are one frame long: a tenth of a second, which is a
   * click and not a note. The fastest staccato on a real piano is about
   * this long, so a note shorter than it is lengthened rather than kept as
   * it is. Default 0.06, sixty milliseconds.
   */
  minDuration?: number;
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
   * Left out, it is middle C, which is where the grand staff splits.
   */
  splitPoint?: number;
  /**
   * How far from the split point a note may be handed to the other hand.
   *
   * When a hand is full, a note near the split can be played by the other
   * hand instead. One far away cannot: the left hand does not reach up to
   * the treble, and the right hand does not come down to the bass. Such a
   * note is dropped.
   *
   * Default a third, which is the usual overlap on a grand staff.
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
  const moveRange = options.moveRange ?? DEFAULT_MOVE_RANGE;
  const splitHands = options.splitHands !== false;
  const minDuration = options.minDuration ?? DEFAULT_MIN_DURATION;

  const pitched = tracks.filter((track) => track.kind !== 'drums');
  const others = tracks.filter((track) => track.kind === 'drums');

  const before = pitched.reduce((sum, track) => sum + track.notes.length, 0);

  // --- 1. Collapse the re-triggers ------------------------------------
  const collapsedNotes = collapseRepeats(
    pitched.flatMap((track) => track.notes),
    mergeGap,
  );
  const collapsed = collapsedNotes.collapsed;

  // --- 2. Lengthen the notes that are too short to play -----------------
  //
  // A note of ten milliseconds is one frame of the model, not a note. It is
  // lengthened rather than dropped: the onset is the part the model gets
  // right, and a key that was struck does not stop sounding the instant the
  // frame ends.
  const lengthened = lengthenNotes(collapsedNotes.notes, minDuration);

  // --- 3. Divide the chords between the hands -------------------------
  const splitPoint = options.splitPoint ?? DEFAULT_SPLIT_POINT;

  const divided = divideHands(
    lengthened,
    splitPoint,
    maxPolyphony,
    maxSpan,
    moveRange,
  );

  // --- 4. Thin each hand to a human rate ------------------------------
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
 * The result keeps the first strike's start, so the onset — the part the
 * model gets right — does not move, and it keeps the loudest of the strikes,
 * because that is the one the model was surest of.
 *
 * How long the folded note is, is the part that matters. The first version
 * made it last until the last strike ended, and the fold was a chain: the
 * merged note grew, so the next strike was measured against the longer note
 * and folded in as well. A key struck every forty milliseconds for ten
 * seconds became one note of ten seconds, and the output had notes of
 * seventeen seconds in it. No piano has a note like that.
 *
 * So the fold is measured from the strike before it, and the folded note
 * lasts as long as the longest of the strikes it swallowed. A note is never
 * longer than the model said some single strike was.
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
    // The end of the strike before, not of the folded note and not the
    // longest of them: the fold is a pairwise operation and cannot run on.
    let previousEnd = current.startTime + current.duration;

    for (let i = 1; i < list.length; i++) {
      const next = list[i];
      if (next.startTime <= previousEnd + mergeGap) {
        current = {
          ...current,
          duration: Math.max(current.duration, next.duration),
          amplitude: Math.max(current.amplitude, next.amplitude),
        };
        previousEnd = next.startTime + next.duration;
        collapsed += 1;
      } else {
        out.push(current);
        current = { ...next };
        previousEnd = next.startTime + next.duration;
      }
    }
    out.push(current);
  }

  return { notes: sortByTime(out), collapsed };
}

/**
 * Lengthen the notes that are too short to be played.
 *
 * The onset is kept where it is — it is the part the model gets right — and
 * only the end moves, so the note lasts at least as long as a hand can
 * play it.
 */
export function lengthenNotes(notes: RawNote[], minDuration: number): RawNote[] {
  return notes.map((note) =>
    note.duration >= minDuration
      ? note
      : { ...note, duration: minDuration },
  );
}

/**
 * Divide the notes between two hands.
 *
 * The notes that start together are a chord, and a chord is what a hand holds.
 * The chords are taken one at a time, because a hand's limits are about the
 * notes it is playing at one moment.
 *
 * The low notes go to the left hand and the high ones to the right hand.
 *
 * Within a hand the order matters. The right hand is given the top of the
 * chord first and the left hand the bottom, because those are the notes the
 * music is about: the melody on top, the bass underneath. When a hand is
 * full, the note it gives up is then the one furthest from the line it is
 * playing — the low note of a right-hand chord, rather than the melody above
 * it. The first version did the opposite, and it threw the melody away.
 *
 * A note that its own hand cannot take may be played by the other hand, but
 * only if it is within `moveRange` semitones of the split: the left hand
 * does not reach up into the treble and the right hand does not come down
 * to the bass.
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

  for (const chord of groupOnsets(sortByTime(notes), CLUSTER_WINDOW)) {
    const high = chord
      .filter((note) => note.pitchMidi >= splitPoint)
      .sort((a, b) => b.pitchMidi - a.pitchMidi);
    const low = chord
      .filter((note) => note.pitchMidi < splitPoint)
      .sort((a, b) => a.pitchMidi - b.pitchMidi);

    // The right hand first: its melody notes are the ones worth a place, and
    // the left hand's leftovers should not be able to take them.
    for (const note of high) {
      const hand = place(hands, note, 1, splitPoint, maxPolyphony, maxSpan, moveRange);
      if (hand < 0) dropped += 1;
      else if (hand !== 1) moved += 1;
    }

    for (const note of low) {
      const hand = place(hands, note, 0, splitPoint, maxPolyphony, maxSpan, moveRange);
      if (hand < 0) dropped += 1;
      else if (hand !== 0) moved += 1;
    }
  }

  return {
    left: sortByTime(hands[0]),
    right: sortByTime(hands[1]),
    dropped,
    moved,
  };
}

/**
 * Put a note in a hand, or in the other one if it is close enough to the
 * split and that hand can take it.
 *
 * Returns the hand the note went to, or -1 if neither could take it.
 */
function place(
  hands: RawNote[][],
  note: RawNote,
  preferred: number,
  splitPoint: number,
  maxPolyphony: number,
  maxSpan: number,
  moveRange: number,
): number {
  if (canTake(hands[preferred], note, maxPolyphony, maxSpan)) {
    hands[preferred].push(note);
    return preferred;
  }

  if (moveRange > 0 && Math.abs(note.pitchMidi - splitPoint) <= moveRange) {
    const other = 1 - preferred;
    if (canTake(hands[other], note, maxPolyphony, maxSpan)) {
      hands[other].push(note);
      return other;
    }
  }

  return -1;
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

/**
 * Group the notes that start together into chords.
 *
 * A chord is the notes that begin within a few tens of milliseconds of each
 * other, which is what a hand has to hold at once. The grouping chains: two
 * notes thirty milliseconds apart are in the same chord even if there is
 * another note between them, because a player does not lift the hand in the
 * middle of a chord.
 *
 * The window is the same one the limits are measured with, so that the code
 * and the measurement are looking at the same chords.
 */
export function groupOnsets(notes: RawNote[], window: number): RawNote[][] {
  const groups: RawNote[][] = [];
  let current: RawNote[] = [];

  for (const note of notes) {
    if (current.length === 0) {
      current.push(note);
      continue;
    }
    const previous = current[current.length - 1];
    if (note.startTime - previous.startTime <= window) {
      current.push(note);
      continue;
    }
    groups.push(current);
    current = [note];
  }
  if (current.length > 0) groups.push(current);

  return groups;
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
