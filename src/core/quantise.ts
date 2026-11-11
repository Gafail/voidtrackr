/**
 * Quantising notes to a grid.
 *
 * Some things run at a fixed rate: a game renders 30 frames a second, a
 * light rig steps every 33 milliseconds, a display refreshes at 60 Hz. A
 * note that starts between two frames cannot be shown on the frame it
 * actually belongs to — the frame either has already been drawn or has not
 * been rendered yet, so the note lands one frame early or late no matter
 * what the MIDI says.
 *
 * Snapping the onsets to the frame grid makes the file agree with the
 * screen. The error is at most half a frame, and the quantised time is
 * exactly the frame the thing on the screen will use.
 *
 * This is a different thing from quantising to the beat. The beat grid
 * needs a tempo and a phase, and the tracker in `beat.ts` is not wired in
 * because the model's own onsets are better than the grid it would snap
 * them to. A frame grid needs neither: it is a fixed number of slots per
 * second, counted from the start of the file.
 *
 * One detail about the file itself: a MIDI file stores ticks, not seconds,
 * so the grid is written to the nearest tick. At 480 ticks a quarter and
 * 120 BPM a tick is 1.04 ms, so a phase lands within half of that of where
 * it was asked for, and the spacing between the slots is exact.
 */

/** How the notes are snapped to the grid. */
export interface QuantiseOptions {
  /**
   * How many slots there are in a second.
   *
   * 30 for a game running at 30 frames a second. A slot is 33.3 ms.
   */
  grid: number;
  /**
   * Where the first slot sits, in seconds.
   *
   * Zero means the first slot is at the start of the recording, which is
   * right when the thing showing the notes starts with the recording and
   * counts its frames from the same moment.
   */
  phase?: number;
}

/** The length of one slot, in seconds. */
export function slotSeconds(options: QuantiseOptions): number {
  const grid = Math.max(0, options.grid);
  if (grid === 0) return 0;
  return 1 / grid;
}

/**
 * Snap a time to the nearest slot.
 *
 * Nearest, not down: a note that is 4 ms before a frame belongs to that
 * frame, and rounding it down would put it on the previous one.
 */
export function quantiseTime(time: number, options: QuantiseOptions): number {
  const period = slotSeconds(options);
  if (period === 0) return Math.max(0, time);

  const phase = options.phase ?? 0;
  const slots = Math.round((time - phase) / period);
  return Math.max(0, phase + slots * period);
}

/** A note's place on the grid. */
export interface QuantisedSpan {
  /** Where the note starts, on a slot. */
  start: number;
  /** How long it lasts. */
  duration: number;
}

/**
 * Snap a note's start and end to the grid.
 *
 * The onset goes to the nearest slot. The offset goes to the next slot at
 * or after it, so a note never ends before it really does, and a note that
 * would collapse to nothing is given a slot of its own.
 */
export function quantiseSpan(
  start: number,
  duration: number,
  options: QuantiseOptions,
): QuantisedSpan {
  const period = slotSeconds(options);
  if (period === 0) return { start: Math.max(0, start), duration };

  const phase = options.phase ?? 0;
  const from = quantiseTime(start, options);

  // Round the end up, so the note covers at least what it covered.
  const slots = Math.ceil((start + duration - phase) / period);
  let to = phase + slots * period;
  if (to <= from) to = from + period;

  return { start: from, duration: to - from };
}

/**
 * Snap a set of notes to the grid.
 *
 * The notes carry a start time and a length; the result carries the same
 * shape, so the editor's own notes can be snapped the same way.
 */
export function quantiseNotes<T extends { startTime: number; duration: number }>(
  notes: T[],
  options: QuantiseOptions,
): T[] {
  if (slotSeconds(options) === 0) return notes;
  return notes.map((note) => {
    const span = quantiseSpan(note.startTime, note.duration, options);
    return { ...note, startTime: span.start, duration: span.duration };
  });
}

/**
 * Describe the grid for the user.
 *
 * "30 per second · 33.3 ms a slot" says what the number means without making
 * them work it out.
 */
export function describeQuantise(options: QuantiseOptions): string {
  if (!options.grid || options.grid <= 0) return 'off';
  const period = slotSeconds(options);
  const phase = options.phase ?? 0;
  const shifted = phase === 0 ? '' : ` · ${(phase * 1000).toFixed(0)} ms offset`;
  return `${options.grid} per second · ${(period * 1000).toFixed(1)} ms a slot${shifted}`;
}
