/**
 * The piano roll.
 *
 * A canvas that shows every track's notes on a time and pitch grid, and lets
 * you select, move, resize, add and delete them.
 *
 * The view is scrolled and zoomed internally rather than by making the canvas
 * larger, so zooming stays smooth and the drawing stays a single pass.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { noteName } from '@core/constants';
import { EditorNote, EditorTrack } from './editor';

/** Width of the pitch label gutter. */
const GUTTER = 56;
/** Height of the time ruler. */
const RULER = 26;
/** How close to a note's right edge counts as grabbing the edge. */
const RESIZE_HANDLE = 6;

interface View {
  /** Horizontal scroll, in pixels. */
  scrollX: number;
  /** Vertical scroll, in pixels. */
  scrollY: number;
  /** Horizontal zoom, in pixels per second. */
  pixelsPerSecond: number;
  /** Vertical zoom, in pixels per semitone. */
  rowHeight: number;
}

export interface PianoRollProps {
  tracks: EditorTrack[];
  selected: Set<string>;
  tempo: number;
  playhead: number;
  onSelect: (ids: Set<string>) => void;
  onChange: (tracks: EditorTrack[]) => void;
  onEditStart: () => void;
  onEditEnd: () => void;
}

interface DragState {
  mode: 'move' | 'resize' | 'marquee';
  /** The note the drag started on. */
  noteId?: string;
  /** Where the pointer was when the drag started. */
  originX: number;
  originY: number;
  /** Initial positions of every note being dragged. */
  originals: Map<string, { start: number; duration: number; pitch: number }>;
  /** Marquee rectangle, in canvas pixels. */
  rect?: { x: number; y: number; width: number; height: number };
}

/**
 * Read a colour out of the stylesheet.
 *
 * A canvas cannot be styled with CSS, so the tokens are read off the document
 * root instead. That way the roll keeps the same colours as the rest of the
 * page, and follows the theme when the theme changes.
 */
function token(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export function PianoRoll({
  tracks,
  selected,
  tempo,
  playhead,
  onSelect,
  onChange,
  onEditStart,
  onEditEnd,
}: PianoRollProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragState | null>(null);

  const [view, setView] = useState<View>({
    scrollX: 0,
    scrollY: 0,
    pixelsPerSecond: 90,
    rowHeight: 14,
  });
  const [size, setSize] = useState({ width: 900, height: 420 });

  /**
   * Bumped when the colour scheme changes, so the canvas repaints itself.
   *
   * The canvas is painted by hand, so it does not follow the stylesheet on its
   * own — it has to be told that the tokens have changed underneath it.
   */
  const [themeTick, setThemeTick] = useState(0);

  useEffect(() => {
    const query = window.matchMedia('(prefers-color-scheme: light)');
    const onChange = () => setThemeTick((value) => value + 1);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  // --- Layout helpers --------------------------------------------------

  const allNotes = tracks.flatMap((track) => track.notes);
  const maxPitch = allNotes.length ? Math.max(...allNotes.map((n) => n.pitch)) + 2 : 72;

  const timeToX = useCallback(
    (time: number) => GUTTER + time * view.pixelsPerSecond - view.scrollX,
    [view.pixelsPerSecond, view.scrollX],
  );
  const pitchToY = useCallback(
    (pitch: number) => RULER + (maxPitch - pitch) * view.rowHeight - view.scrollY,
    [maxPitch, view.rowHeight, view.scrollY],
  );
  const xToTime = useCallback(
    (x: number) => (x - GUTTER + view.scrollX) / view.pixelsPerSecond,
    [view.pixelsPerSecond, view.scrollX],
  );
  const yToPitch = useCallback(
    (y: number) => maxPitch - (y - RULER + view.scrollY) / view.rowHeight,
    [maxPitch, view.rowHeight, view.scrollY],
  );

  /** The note under a canvas position, if any. */
  const noteAt = useCallback(
    (x: number, y: number): { note: EditorNote; track: EditorTrack; rightEdge: boolean } | null => {
      for (const track of tracks) {
        for (const note of track.notes) {
          const noteX = timeToX(note.start);
          const noteY = pitchToY(note.pitch);
          const noteW = Math.max(4, note.duration * view.pixelsPerSecond);
          const noteH = view.rowHeight - 2;
          if (x >= noteX && x <= noteX + noteW && y >= noteY && y <= noteY + noteH) {
            return {
              note,
              track,
              rightEdge: x >= noteX + noteW - RESIZE_HANDLE,
            };
          }
        }
      }
      return null;
    },
    [tracks, timeToX, pitchToY, view.pixelsPerSecond, view.rowHeight],
  );

  // --- Drawing ---------------------------------------------------------

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    canvas.width = size.width * ratio;
    canvas.height = size.height * ratio;
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);

    context.clearRect(0, 0, size.width, size.height);

    // The palette, read from the stylesheet so the roll follows the theme.
    const surface = token('--void-800', '#0c0c13');
    const surfaceAlt = token('--void-900', '#07070b');
    const line = token('--line', '#232330');
    const lineSoft = token('--line-soft', '#1b1b26');
    const ink3 = token('--ink-3', '#74718c');
    const rulerSurface = token('--void-700', '#12121b');
    const accent = token('--accent', '#8b7bf7');
    const accentInk = token('--accent-ink', '#b7adff');
    const live = token('--live', '#4fd48b');

    context.fillStyle = surface;
    context.fillRect(0, 0, size.width, size.height);

    const firstPitch = Math.floor(yToPitch(size.height));
    const lastPitch = Math.ceil(yToPitch(0));

    // Rows, shaded for the black keys. The highest pitch is at the top, so
    // the loop counts down.
    for (let pitch = lastPitch; pitch >= firstPitch; pitch--) {
      const y = pitchToY(pitch);
      if (y < RULER - view.rowHeight || y > size.height + view.rowHeight) continue;
      const isBlack = [1, 3, 6, 8, 10].includes(((pitch % 12) + 12) % 12);
      context.fillStyle = isBlack ? surfaceAlt : surface;
      context.fillRect(GUTTER, y, size.width - GUTTER, view.rowHeight);
    }

    // Beat and bar lines.
    const secondsPerBeat = 60 / tempo;
    const firstBeat = Math.floor(xToTime(GUTTER) / secondsPerBeat);
    const lastBeat = Math.ceil(xToTime(size.width) / secondsPerBeat);
    for (let beat = firstBeat; beat <= lastBeat; beat++) {
      const x = timeToX(beat * secondsPerBeat);
      if (x < GUTTER) continue;
      const isBar = beat % 4 === 0;
      context.strokeStyle = isBar ? line : lineSoft;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(Math.round(x) + 0.5, RULER);
      context.lineTo(Math.round(x) + 0.5, size.height);
      context.stroke();
    }

    // Pitch labels.
    context.fillStyle = ink3;
    context.font = '11px ui-sans-serif, system-ui, sans-serif';
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    for (let pitch = lastPitch; pitch >= firstPitch; pitch--) {
      const y = pitchToY(pitch);
      if (y < RULER || y > size.height) continue;
      context.fillStyle = ink3;
      context.fillText(noteName(pitch), GUTTER - 6, y + view.rowHeight / 2);
    }

    // Ruler.
    context.fillStyle = rulerSurface;
    context.fillRect(0, 0, size.width, RULER);
    context.strokeStyle = lineSoft;
    context.beginPath();
    context.moveTo(0, RULER + 0.5);
    context.lineTo(size.width, RULER + 0.5);
    context.stroke();

    context.fillStyle = ink3;
    context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.textAlign = 'left';
    for (let beat = Math.floor(xToTime(GUTTER) / secondsPerBeat); beat <= lastBeat; beat++) {
      const x = timeToX(beat * secondsPerBeat);
      if (x < GUTTER) continue;
      const label = beat % 4 === 0 ? `${beat / 4 + 1}` : '';
      if (label) context.fillText(label, x + 3, 13);
    }

    // Notes.
    for (const track of tracks) {
      if (track.muted) continue;
      for (const note of track.notes) {
        const x = timeToX(note.start);
        const w = Math.max(4, note.duration * view.pixelsPerSecond);
        if (x + w < GUTTER || x > size.width) continue;
        const y = pitchToY(note.pitch);
        const h = Math.max(3, view.rowHeight - 2);

        context.fillStyle = track.color;
        context.globalAlpha = 0.35 + 0.65 * note.velocity;
        roundRect(context, x, y, w, h, 3);
        context.fill();
        context.globalAlpha = 1;

        if (selected.has(note.id)) {
          context.strokeStyle = accent;
          context.lineWidth = 2;
          roundRect(context, x + 1, y + 1, w - 2, h - 2, 3);
          context.stroke();
        }
      }
    }

    // Marquee.
    const drag = dragRef.current;
    if (drag?.mode === 'marquee' && drag.rect) {
      context.strokeStyle = accentInk;
      context.globalAlpha = 0.16;
      context.fillStyle = accent;
      context.lineWidth = 1;
      context.fillRect(drag.rect.x, drag.rect.y, drag.rect.width, drag.rect.height);
      context.globalAlpha = 1;
      context.strokeRect(drag.rect.x + 0.5, drag.rect.y + 0.5, drag.rect.width, drag.rect.height);
    }

    // Playhead.
    const playX = timeToX(playhead);
    if (playX >= GUTTER && playX <= size.width) {
      context.strokeStyle = live;
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(playX, RULER);
      context.lineTo(playX, size.height);
      context.stroke();
    }

    // Gutter edge.
    context.strokeStyle = lineSoft;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(GUTTER + 0.5, 0);
    context.lineTo(GUTTER + 0.5, size.height);
    context.stroke();
  }, [tracks, selected, view, size, tempo, playhead, timeToX, pitchToY, xToTime, yToPitch, themeTick]);

  // --- Sizing ----------------------------------------------------------

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      setSize({
        width: container.clientWidth,
        height: container.clientHeight,
      });
    });
    observer.observe(container);
    setSize({ width: container.clientWidth, height: container.clientHeight });
    return () => observer.disconnect();
  }, []);

  // --- Interaction -----------------------------------------------------

  const pointerPosition = (event: React.MouseEvent): { x: number; y: number } => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const handleMouseDown = (event: React.MouseEvent) => {
    if (event.button !== 0) return;
    const { x, y } = pointerPosition(event);
    const hit = noteAt(x, y);

    if (!hit) {
      if (!event.shiftKey) onSelect(new Set());
      dragRef.current = {
        mode: 'marquee',
        originX: x,
        originY: y,
        originals: new Map(),
        rect: { x, y, width: 0, height: 0 },
      };
      return;
    }

    const next = new Set(selected);
    if (event.shiftKey) {
      if (next.has(hit.note.id)) next.delete(hit.note.id);
      else next.add(hit.note.id);
      onSelect(next);
    } else if (!next.has(hit.note.id)) {
      next.clear();
      next.add(hit.note.id);
      onSelect(next);
    }

    // Remember where everything started, so the drag is relative.
    const originals = new Map<string, { start: number; duration: number; pitch: number }>();
    for (const track of tracks) {
      for (const note of track.notes) {
        if (next.has(note.id)) {
          originals.set(note.id, { start: note.start, duration: note.duration, pitch: note.pitch });
        }
      }
    }

    dragRef.current = {
      mode: hit.rightEdge ? 'resize' : 'move',
      noteId: hit.note.id,
      originX: x,
      originY: y,
      originals,
    };
    onEditStart();
  };

  const handleMouseMove = (event: React.MouseEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    const { x, y } = pointerPosition(event);

    if (drag.mode === 'marquee') {
      const rect = {
        x: Math.min(drag.originX, x),
        y: Math.min(drag.originY, y),
        width: Math.abs(x - drag.originX),
        height: Math.abs(y - drag.originY),
      };
      drag.rect = rect;

      const inside = new Set<string>();
      for (const track of tracks) {
        for (const note of track.notes) {
          const noteX = timeToX(note.start);
          const noteY = pitchToY(note.pitch);
          const noteW = Math.max(4, note.duration * view.pixelsPerSecond);
          const noteH = view.rowHeight - 2;
          if (
            noteX < rect.x + rect.width &&
            noteX + noteW > rect.x &&
            noteY < rect.y + rect.height &&
            noteY + noteH > rect.y
          ) {
            inside.add(note.id);
          }
        }
      }
      onSelect(inside);
      return;
    }

    const deltaTime = (x - drag.originX) / view.pixelsPerSecond;
    const deltaPitch = Math.round((drag.originY - y) / view.rowHeight);

    onChange(
      tracks.map((track) => ({
        ...track,
        notes: track.notes.map((note) => {
          const original = drag.originals.get(note.id);
          if (!original) return note;

          if (drag.mode === 'resize' && note.id === drag.noteId) {
            const duration = Math.max(0.05, original.duration + deltaTime);
            return { ...note, duration };
          }

          const start = Math.max(0, original.start + deltaTime);
          const pitch = Math.max(0, Math.min(127, original.pitch + deltaPitch));
          return { ...note, start, pitch };
        }),
      })),
    );
  };

  const handleMouseUp = () => {
    if (dragRef.current && dragRef.current.mode !== 'marquee') onEditEnd();
    dragRef.current = null;
  };

  const handleDoubleClick = (event: React.MouseEvent) => {
    const { x, y } = pointerPosition(event);
    if (x < GUTTER || y < RULER) return;
    if (noteAt(x, y)) return;

    const track = tracks[0];
    if (!track) return;

    const pitch = Math.max(0, Math.min(127, Math.round(yToPitch(y))));
    const start = Math.max(0, xToTime(x));
    const note: EditorNote = {
      id: `new-${Math.random().toString(36).slice(2)}`,
      trackId: track.id,
      pitch,
      start,
      duration: 0.4,
      velocity: 0.8,
    };

    onEditStart();
    onChange(
      tracks.map((t) => (t.id === track.id ? { ...t, notes: [...t.notes, note] } : t)),
    );
    onSelect(new Set([note.id]));
  };

  const handleWheel = (event: React.WheelEvent) => {
    if (event.ctrlKey || event.metaKey) {
      const { x } = pointerPosition(event as unknown as React.MouseEvent);
      const timeAtPointer = xToTime(x);
      const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
      setView((previous) => {
        const pixelsPerSecond = Math.max(8, Math.min(900, previous.pixelsPerSecond * factor));
        return {
          ...previous,
          pixelsPerSecond,
          scrollX: Math.max(0, GUTTER + timeAtPointer * pixelsPerSecond - x),
        };
      });
      return;
    }

    const dx = event.shiftKey ? event.deltaY : event.deltaX;
    const dy = event.shiftKey ? 0 : event.deltaY;
    setView((previous) => ({
      ...previous,
      scrollX: Math.max(0, previous.scrollX + dx),
      scrollY: Math.max(0, previous.scrollY + dy),
    }));
  };

  return (
    <div className="piano-roll" ref={containerRef}>
      <canvas
        ref={canvasRef}
        className="piano-roll-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
      />
    </div>
  );
}

/** A rounded rectangle path. */
function roundRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): void {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}
