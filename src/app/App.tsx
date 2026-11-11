/**
 * The application shell: drop a file, transcribe it, then edit and export
 * the result.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { decodeAudio, prepareMono } from '@core/audio';
import { loadPinnedModel } from '@core/basic-pitch';
import { ModelLock, ModelSource, sha256Hex, TranscriptionModel } from '@core/model';
import modelLock from '@core/model.lock.json';
import { OkarinaModel, OKARINA_FILES } from '@core/okarina';
import okarinaLock from '@core/okarina.lock.json';
import { createWebOnnxRunner } from '@core/onnx-web';
import { OnnxRunner, ONNX_THREADS } from '@core/onnx';
import { verifyOnnxWeights } from '@core/onnx-weights';
import { PianoTranscriptionModel, PIANO_DIR, PIANO_FILES } from '@core/piano';
import pianoLock from '@core/piano.lock.json';
import { TsumugiModel, TSUMUGI_DIR, TSUMUGI_FILES } from '@core/tsumugi';
import tsumugiLock from '@core/tsumugi.lock.json';
import { YourMt3Model, YOURMT3_DIR, YOURMT3_FILES } from '@core/yourmt3';
import yourmt3Lock from '@core/yourmt3.lock.json';
import { DEFAULT_MODEL, listModels, ModelId, MODELS } from '@core/registry';
import { transcribe } from '@core/pipeline';
import { buildMidiFile, toArrayBuffer } from '@core/midi';
import { provenanceFor } from '@core/version';
import { NOTE_NAMES } from '@core/constants';
import {
  describeRange,
  FrequencyRange,
  MAX_FREQUENCY,
  MIN_FREQUENCY,
  noteInRange,
} from '@core/range';
import { describeQuantise, quantiseSpan } from '@core/quantise';
import { KeyInfo, TempoInfo } from '@core/types';
import { PianoRoll } from './PianoRoll';
import { EditorTrack, arrangeTracks, fromEditorTracks, toEditorTracks } from './editor';
import { Player } from './playback';

const MODEL_URL = '/model/model.json';

/** Where the ONNX weights are served from. */
const ONNX_URL = '/model-onnx/';

/**
 * How many threads the model will actually use.
 *
 * The runtime can only use more than one when the page is cross-origin
 * isolated, which needs `SharedArrayBuffer`. The web server sends the two
 * headers that ask for it, but a page served over plain HTTP at a LAN
 * address is not a secure context, so the browser will not grant it unless
 * the browser is told to treat the address as secure. When it is not, the
 * runtime prints "Falling back to single-threading" and runs on one thread.
 *
 * So the count is read rather than assumed: the header says what the browser
 * is allowing, and the header is shown so that a browser without the flag
 * says so rather than quietly running slowly.
 */
const THREADS = (() => {
  if (typeof self === 'undefined' || !self.crossOriginIsolated) return 1;
  const cores = navigator.hardwareConcurrency || 1;
  return Math.max(1, Math.min(ONNX_THREADS, cores));
})();

/** What to tell the reader about the thread count. */
const THREADS_HINT =
  'The model is running on one thread. The browser will only allow more when the ' +
  'page is cross-origin isolated, which needs the browser to be started with ' +
  '--unsafely-treat-insecure-origin-as-secure=' +
  window.location.origin;

/**
 * Read the model over HTTP.
 *
 * The weights are checked against the lock file before the model runs, so a
 * swapped-out or partially downloaded model is refused rather than used.
 */
const modelSource: ModelSource = {
  url: MODEL_URL,
  read: async (name: string) => {
    const base = new URL(MODEL_URL, window.location.href);
    const response = await fetch(new URL(name, base).href);
    if (!response.ok) {
      throw new Error(`Could not read the model file ${name} (${response.status})`);
    }
    return response.arrayBuffer();
  },
};

/**
 * Load the Okarina model, checking its weights first.
 *
 * The weights are large, so they are fetched once and hashed. A mismatch
 * means the output would differ from every previous run, so it is better to
 * stop than to produce a different file quietly.
 */
async function loadOkarinaModel(): Promise<TranscriptionModel> {
  const lock = okarinaLock as unknown as { files: Record<string, { sha256: string }> };
  const digests: string[] = [];

  for (const name of [OKARINA_FILES.frontEnd, OKARINA_FILES.m]) {
    const expected = lock.files[name]?.sha256;
    const response = await fetch(new URL(name, new URL(ONNX_URL, window.location.href)).href);
    if (!response.ok) {
      throw new Error(
        `Could not read the Okarina weights ${name} (${response.status}). ` +
          'Run `npm run model:okarina` to fetch them.',
      );
    }
    const bytes = await response.arrayBuffer();
    const digest = await sha256Hex(bytes);
    if (digest !== expected) {
      throw new Error(
        `The Okarina weights ${name} do not match the pinned version.\n` +
          `  expected sha256 ${expected}\n` +
          `  got      sha256 ${digest}\n` +
          'Run `npm run model:okarina` to fetch the pinned weights.',
      );
    }
    digests.push(digest);
  }

  const fingerprint = await sha256Hex(digests.join(''));
  const runner = await createWebOnnxRunner();
  return OkarinaModel.load(runner, ONNX_URL, 'm', fingerprint);
}

/**
 * Read one of the ONNX weight files, relative to the weights root.
 *
 * The command that fetches the weights is carried in the error, because a
 * missing file is the common case: the default model needs its weights
 * fetched before the app will do anything with a recording.
 */
function onnxFileReader(fetchCommand: string): (path: string) => Promise<ArrayBuffer> {
  return async (path: string): Promise<ArrayBuffer> => {
    const response = await fetch(
      new URL(path, new URL(ONNX_URL, window.location.href)).href,
    );
    if (!response.ok) {
      throw new Error(
        `Could not read the model file ${path} (${response.status}). ` +
          `Run \`${fetchCommand}\` to fetch the weights.`,
      );
    }
    return response.arrayBuffer();
  };
}

/**
 * Load one of the ONNX models, checking its weights first.
 *
 * The same check runs whichever model it is: the files are fetched, hashed
 * against the lock file, and only then is the model allowed to run.
 */
async function loadOnnxModel(
  lock: unknown,
  directory: string,
  files: string[],
  fetchCommand: string,
  create: (
    runner: OnnxRunner,
    directory: string,
    fingerprint: string,
  ) => Promise<TranscriptionModel>,
): Promise<TranscriptionModel> {
  const fingerprint = await verifyOnnxWeights(
    onnxFileReader(fetchCommand),
    lock as { files: Record<string, { sha256: string }> },
    { dir: directory, files },
    fetchCommand,
  );
  const runner = await createWebOnnxRunner();
  return create(runner, `${ONNX_URL}${directory}/`, fingerprint);
}

/** How many edits can be undone. */
const HISTORY_LIMIT = 60;

type Status = 'idle' | 'working' | 'done' | 'error';
type Theme = 'light' | 'dark';

/** What the reader's system is asking for. */
function systemTheme(): Theme {
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

/**
 * The theme, and the button that changes it.
 *
 * The reader's system decides until they choose otherwise. The choice is
 * written onto the document so the stylesheet can follow it, and so the
 * piano roll — which paints itself rather than being styled — reads the
 * right palette.
 *
 * Nothing is stored: no localStorage, no cookies. A reload goes back to the
 * system's preference, which is the house rule.
 */
function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(systemTheme);
  const chosen = useRef(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // The system is followed until the reader picks for themselves.
  useEffect(() => {
    const query = window.matchMedia('(prefers-color-scheme: light)');
    const onChange = () => {
      if (!chosen.current) setTheme(query.matches ? 'light' : 'dark');
    };
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  const toggle = useCallback(() => {
    chosen.current = true;
    const next = theme === 'dark' ? 'light' : 'dark';
    // The document is changed before the state, so a redraw never happens
    // against the old palette.
    document.documentElement.dataset.theme = next;
    setTheme(next);
  }, [theme]);

  return { theme, toggle };
}

export function App() {
  const [status, setStatus] = useState<Status>('idle');
  const [progress, setProgress] = useState({ value: 0, label: '' });
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const [tracks, setTracks] = useState<EditorTrack[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [tempo, setTempo] = useState(120);
  const [key, setKey] = useState<KeyInfo>({ tonic: 0, mode: 'major', confidence: 0 });
  const [duration, setDuration] = useState(0);

  const [history, setHistory] = useState<{ past: EditorTrack[][]; future: EditorTrack[][] }>({
    past: [],
    future: [],
  });

  const [playhead, setPlayhead] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [modelId, setModelId] = useState<ModelId>(DEFAULT_MODEL);

  /**
   * The range of the spectrum to keep, in Hz.
   *
   * It starts wide open: everything from the bottom of the keyboard to the
   * top of hearing. Narrowing it hides the notes outside it, and the file
   * that is exported carries only what is left.
   */
  const [range, setRange] = useState<FrequencyRange>({
    low: MIN_FREQUENCY,
    high: MAX_FREQUENCY,
  });

  /**
   * The grid the notes are snapped to, in slots per second.
   *
   * Zero means the notes are left where the model put them. Thirty lines
   * them up with a screen that renders thirty frames a second, which is
   * what a game that reacts to the music needs: a note that starts between
   * two frames cannot be shown on the frame it belongs to, so the file is
   * better off agreeing with the frames than being more precise than the
   * screen.
   */
  const [grid, setGrid] = useState(0);

  /** Where the first slot sits, in milliseconds. */
  const [phase, setPhase] = useState(0);

  /**
   * Whether the notes have been arranged for two hands.
   *
   * The arrangement folds the re-triggers together, divides the chords
   * between a left and a right hand, and drops what neither hand could
   * hold. It is applied to the notes as they stand, so it can be turned on
   * and off without transcribing again: a piano run takes minutes.
   */
  const [arranged, setArranged] = useState(false);

  /** The notes as they were before the arrangement, so it can be undone. */
  const arrangedRef = useRef<EditorTrack[] | null>(null);

  /**
   * The tracks as they are shown, played and exported: only the notes inside
   * the range, and snapped to the grid if one is set.
   *
   * The notes outside the range are not thrown away — they stay in the
   * editor's state, so widening the range brings them back. That matters
   * here: a transcription takes minutes, and the range is the thing worth
   * experimenting with.
   */
  const visibleTracks = useMemo(() => {
    const kept = tracks.map((track) => ({
      ...track,
      notes: track.notes.filter((note) => noteInRange(note.pitch, track.kind, range)),
    }));
    if (grid <= 0) return kept;
    const settings = { grid, phase: phase / 1000 };
    return kept.map((track) => ({
      ...track,
      notes: track.notes.map((note) => {
        const span = quantiseSpan(note.start, note.duration, settings);
        return { ...note, start: span.start, duration: span.duration };
      }),
    }));
  }, [tracks, range, grid, phase]);

  /**
   * Take an edit made on the visible tracks back into the full set.
   *
   * A note that was hidden could not have been moved or deleted, so it stays
   * as it was. A note that was visible takes the edited version, and a
   * visible note that is no longer in the edited list was deleted.
   */
  const applyVisibleChange = useCallback(
    (visible: EditorTrack[]) => {
      setTracks((previous) =>
        previous.map((track) => {
          const edited = visible.find((candidate) => candidate.id === track.id);
          if (!edited) return track;
          const byId = new Map(edited.notes.map((note) => [note.id, note]));
          return {
            ...track,
            notes: track.notes.flatMap((note) => {
              const changed = byId.get(note.id);
              if (changed) return [changed];
              return noteInRange(note.pitch, track.kind, range) ? [] : [note];
            }),
          };
        }),
      );
    },
    [range],
  );

  const { theme, toggle: toggleTheme } = useTheme();

  const modelCacheRef = useRef<Map<ModelId, TranscriptionModel>>(new Map());
  const playerRef = useRef<Player | null>(null);
  const dragSnapshotRef = useRef<EditorTrack[] | null>(null);

  // --- Model -----------------------------------------------------------

  /**
   * Load the chosen model.
   *
   * The weights are checked against the lock file before the model runs, so
   * a swapped-out or partially downloaded model is refused rather than used.
   * Each model is cached, so switching back and forth does not refetch it.
   */
  const loadModel = useCallback(async (id: ModelId): Promise<TranscriptionModel> => {
    const cached = modelCacheRef.current.get(id);
    if (cached) return cached;

    let model: TranscriptionModel;
    switch (id) {
      case 'okarina':
        model = await loadOkarinaModel();
        break;
      case 'tsumugi':
        model = await loadOnnxModel(
          tsumugiLock,
          TSUMUGI_DIR,
          [TSUMUGI_FILES.graph, TSUMUGI_FILES.heads, TSUMUGI_FILES.headsJson],
          'npm run model:tsumugi',
          (runner, directory, fingerprint) =>
            TsumugiModel.load(runner, directory, fingerprint),
        );
        break;
      case 'yourmt3':
        model = await loadOnnxModel(
          yourmt3Lock,
          YOURMT3_DIR,
          [YOURMT3_FILES.encoder, YOURMT3_FILES.decoder, YOURMT3_FILES.vocab],
          'npm run model:yourmt3',
          (runner, directory, fingerprint) =>
            YourMt3Model.load(runner, directory, fingerprint),
        );
        break;
      case 'piano':
        model = await loadOnnxModel(
          pianoLock,
          PIANO_DIR,
          [PIANO_FILES.graph],
          'npm run model:piano',
          (runner, directory, fingerprint) =>
            PianoTranscriptionModel.load(runner, directory, fingerprint),
        );
        break;
      default:
        model = await loadPinnedModel(modelSource, modelLock as ModelLock);
    }

    modelCacheRef.current.set(id, model);
    return model;
  }, []);

  // --- Player ----------------------------------------------------------

  useEffect(() => {
    const player = new Player();
    playerRef.current = player;
    player.onEnded = () => {
      setIsPlaying(false);
      setPlayhead(0);
    };
    return () => player.dispose();
  }, []);

  useEffect(() => {
    playerRef.current?.setTracks(visibleTracks);
  }, [visibleTracks]);

  useEffect(() => {
    if (!isPlaying) return;
    let frame = 0;
    const tick = () => {
      const player = playerRef.current;
      if (player) setPlayhead(player.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [isPlaying]);

  // --- Conversion ------------------------------------------------------

  const handleFile = useCallback(
    async (file: File) => {
      setStatus('working');
      setError(null);
      setFileName(file.name);
      setProgress({ value: 0, label: 'Decoding audio' });

      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(file));

      try {
        const buffer = await file.arrayBuffer();
        setProgress({ value: 0.02, label: 'Decoding audio' });
        const audio = await decodeAudio(buffer);
        const samples = prepareMono(audio);

        setProgress({ value: 0.05, label: 'Loading model' });
        const model = await loadModel(modelId);

        const result = await transcribe(samples, {
          model,
          onProgress: setProgress,
        });

        setTracks(toEditorTracks(result.tracks));
        setArranged(false);
        arrangedRef.current = null;
        setTempo(result.tempo.bpm);
        setKey(result.key);
        setDuration(result.duration);
        setLog(result.log);
        setSelected(new Set());
        setHistory({ past: [], future: [] });
        setPlayhead(0);
        setStatus('done');
      } catch (caught) {
        console.error(caught);
        setError((caught as Error).message || 'Something went wrong during the conversion.');
        setStatus('error');
      }
    },
    [audioUrl, loadModel, modelId],
  );

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) void handleFile(file);
  };

  // --- Editing ---------------------------------------------------------

  /** A deep enough copy to keep as an undo snapshot. */
  const copyTracks = (source: EditorTrack[]): EditorTrack[] =>
    source.map((track) => ({
      ...track,
      notes: track.notes.map((note) => ({ ...note })),
    }));

  /**
   * Arrange the notes for two hands, or put them back as they were.
   *
   * The arrangement is a change to the notes themselves rather than a view
   * over them, because it changes what the tracks are: two hands instead of
   * one. The notes it started from are kept, so the switch can be undone.
   */
  const toggleArranged = useCallback(() => {
    if (arranged) {
      const before = arrangedRef.current;
      arrangedRef.current = null;
      setArranged(false);
      if (before) setTracks(before);
    } else {
      arrangedRef.current = copyTracks(tracks);
      setTracks(arrangeTracks(tracks));
      setArranged(true);
    }
    setSelected(new Set());
  }, [arranged, tracks]);

  /**
   * Called once when an edit starts, so undo has the state before it.
   *
   * The snapshot is taken here rather than on every change, because a drag
   * produces a change on every mouse move.
   */
  const onEditStart = useCallback(() => {
    if (!dragSnapshotRef.current) {
      dragSnapshotRef.current = copyTracks(tracks);
    }
  }, [tracks]);

  /** Called when an edit ends, which is when it becomes undoable. */
  const onEditEnd = useCallback(() => {
    const snapshot = dragSnapshotRef.current;
    dragSnapshotRef.current = null;
    if (!snapshot) return;
    setHistory((previous) => {
      const past = [...previous.past, snapshot];
      return {
        past: past.length > HISTORY_LIMIT ? past.slice(past.length - HISTORY_LIMIT) : past,
        future: [],
      };
    });
  }, []);

  const undo = useCallback(() => {
    if (history.past.length === 0) return;
    const last = history.past[history.past.length - 1];
    setHistory({
      past: history.past.slice(0, -1),
      future: [...history.future, copyTracks(tracks)],
    });
    setTracks(last);
    setSelected(new Set());
  }, [history, tracks]);

  const redo = useCallback(() => {
    if (history.future.length === 0) return;
    const last = history.future[history.future.length - 1];
    setHistory({
      past: [...history.past, copyTracks(tracks)],
      future: history.future.slice(0, -1),
    });
    setTracks(last);
    setSelected(new Set());
  }, [history, tracks]);

  const deleteSelected = useCallback(() => {
    if (selected.size === 0) return;
    onEditStart();
    onEditEnd();
    setTracks((previous) =>
      previous.map((track) => ({
        ...track,
        notes: track.notes.filter((note) => !selected.has(note.id)),
      })),
    );
    setSelected(new Set());
  }, [selected, onEditStart, onEditEnd]);

  const toggleMute = (trackId: string) => {
    setTracks((previous) =>
      previous.map((track) =>
        track.id === trackId ? { ...track, muted: !track.muted } : track,
      ),
    );
  };

  // --- Transport -------------------------------------------------------

  const togglePlay = useCallback(() => {
    const player = playerRef.current;
    if (!player) return;
    if (player.isPlaying) {
      player.stop();
      setIsPlaying(false);
      setPlayhead(0);
    } else {
      void player.play(playhead);
      setIsPlaying(true);
    }
  }, [playhead]);

  const stop = useCallback(() => {
    playerRef.current?.stop();
    setIsPlaying(false);
    setPlayhead(0);
  }, []);

  // --- Keyboard --------------------------------------------------------

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }

      if ((event.key === 'Delete' || event.key === 'Backspace') && selected.size > 0) {
        event.preventDefault();
        deleteSelected();
        return;
      }

      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
        return;
      }

      if (event.code === 'Space') {
        event.preventDefault();
        togglePlay();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [selected, deleteSelected, undo, redo, togglePlay]);

  // --- Export ----------------------------------------------------------

  const downloadMidi = () => {
    // The model that produced the notes, so the stamp names the right one.
    const model = modelCacheRef.current.get(modelId);
    const bytes = buildMidiFile(
      fromEditorTracks(visibleTracks),
      { bpm: tempo, confidence: 1, adjusted: false } satisfies TempoInfo,
      key,
      // The stamp records which pipeline and model produced the file, so an
      // exported MIDI can always be traced back.
      { provenance: model ? provenanceFor(model) : undefined },
    );
    const blob = new Blob([toArrayBuffer(bytes)], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${(fileName ?? 'audio').replace(/\.[^.]+$/, '')}.mid`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  };

  const noteCount = useMemo(
    () => visibleTracks.reduce((total, track) => total + track.notes.length, 0),
    [visibleTracks],
  );

  /** How many notes the arrangement left. The drum kit is not part of it. */
  const arrangedNotes = useMemo(
    () =>
      tracks
        .filter((track) => track.kind !== 'drums')
        .reduce((total, track) => total + track.notes.length, 0),
    [tracks],
  );

  // --- Render ----------------------------------------------------------

  return (
    <div className="app">
      <div className="void-bg" aria-hidden="true" />
      <header className="app-header">
        <div className="brand">
          <p className="kicker">
            <span className="kicker-slash">//</span> a void tool
          </p>
          <h1 className="wordmark">
            <span className="wordmark-mark" aria-hidden="true" />
            <span className="wordmark-text">voidtrackr</span>
          </h1>
          <p className="subtitle">
            A recording in, a MIDI file out. Nothing is uploaded, and the audio
            never leaves this machine.
          </p>
        </div>
        <div className="header-actions">
          <p
            className={`threads${THREADS > 1 ? '' : ' threads-single'}`}
            title={THREADS > 1 ? `The model is using ${THREADS} threads.` : THREADS_HINT}
          >
            {THREADS} {THREADS === 1 ? 'thread' : 'threads'}
          </p>
          <button
            className="theme-toggle"
            type="button"
            onClick={toggleTheme}
            aria-label={
              theme === 'dark' ? 'Switch to the light theme' : 'Switch to the dark theme'
            }
            aria-pressed={theme === 'dark'}
            title={theme === 'dark' ? 'Switch to the light theme' : 'Switch to the dark theme'}
          >
            <svg
              className="theme-icon theme-icon-dark"
              viewBox="0 0 24 24"
              aria-hidden="true"
              focusable="false"
            >
              <path d="M20.5 14.3A8.5 8.5 0 1 1 9.7 3.5a6.8 6.8 0 0 0 10.8 10.8Z" />
            </svg>
            <svg
              className="theme-icon theme-icon-light"
              viewBox="0 0 24 24"
              aria-hidden="true"
              focusable="false"
            >
              <circle cx="12" cy="12" r="4.2" />
              <path d="M12 2.5v2.2M12 19.3v2.2M21.5 12h-2.2M4.7 12H2.5M18.7 5.3l-1.6 1.6M6.9 17.1l-1.6 1.6M18.7 18.7l-1.6-1.6M6.9 6.9 5.3 5.3" />
            </svg>
          </button>
          <label className="field model-picker">
            <span>Model</span>
            <select
              value={modelId}
              onChange={(event) => setModelId(event.target.value as ModelId)}
              disabled={status === 'working'}
            >
              {listModels().map((model) => (
                <option key={model.id} value={model.id}>
                  {model.displayName} — {model.license}
                </option>
              ))}
            </select>
            <em className={MODELS[modelId].commercialUse ? 'licence-ok' : 'licence-restricted'}>
              {MODELS[modelId].commercialUse
                ? `${MODELS[modelId].license} — commercial use permitted`
                : `${MODELS[modelId].license} — do not sell the output`}
            </em>
          </label>
          <label className="field range-picker">
            <span>Range</span>
            <input
              type="number"
              min={MIN_FREQUENCY}
              max={MAX_FREQUENCY}
              step={10}
              value={range.low}
              onChange={(event) =>
                setRange((previous) => ({
                  ...previous,
                  low: Math.max(MIN_FREQUENCY, Number(event.target.value) || 0),
                }))
              }
              disabled={status === 'working'}
              aria-label="Lowest frequency to keep, in hertz"
            />
            <span className="dash">–</span>
            <input
              type="number"
              min={MIN_FREQUENCY}
              max={MAX_FREQUENCY}
              step={10}
              value={range.high}
              onChange={(event) =>
                setRange((previous) => ({
                  ...previous,
                  high: Math.min(MAX_FREQUENCY, Number(event.target.value) || MAX_FREQUENCY),
                }))
              }
              disabled={status === 'working'}
              aria-label="Highest frequency to keep, in hertz"
            />
            <em>{describeRange(range)}</em>
          </label>
          <label className="field pianify-picker">
            <span>Pianify</span>
            <input
              type="checkbox"
              checked={arranged}
              onChange={toggleArranged}
              disabled={status === 'working' || tracks.length === 0}
              aria-label="Arrange the notes for two hands"
              title="Fold the re-triggers together, divide the chords between a left and a right hand, and drop what neither hand could hold."
            />
            <em>{arranged ? `${arrangedNotes} notes` : 'two hands'}</em>
          </label>
          <label className="field grid-picker">
            <span>Grid</span>
            <input
              type="number"
              min={0}
              max={480}
              step={1}
              value={grid}
              onChange={(event) => setGrid(Math.max(0, Number(event.target.value) || 0))}
              disabled={status === 'working'}
              aria-label="Snap the notes to a grid of this many slots per second. Zero leaves them alone."
              title="Slots per second. 30 lines the notes up with a 30 FPS screen. Zero leaves them alone."
            />
            <span className="dash">@</span>
            <input
              type="number"
              min={0}
              max={1000}
              step={1}
              value={phase}
              onChange={(event) => setPhase(Math.max(0, Number(event.target.value) || 0))}
              disabled={status === 'working' || grid <= 0}
              aria-label="Where the first slot sits, in milliseconds"
              title="Where the first slot sits, in milliseconds"
            />
            <em>
              {grid > 0
                ? `${describeQuantise({ grid, phase: phase / 1000 })}`
                : 'off — the notes stay where the model put them'}
            </em>
          </label>
          {status === 'done' && (
            <>
              {audioUrl && (
                <audio className="original-audio" src={audioUrl} controls preload="metadata" />
              )}
              <button className="button primary" onClick={downloadMidi}>
                Download MIDI
              </button>
            </>
          )}
        </div>
      </header>

      {status !== 'done' && (
        <div
          className={`dropzone${status === 'working' ? ' busy' : ''}`}
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          {status === 'working' ? (
            <>
              <h2>{fileName}</h2>
              <div className="progress">
                <div
                  className="progress-bar"
                  style={{ width: `${Math.round(progress.value * 100)}%` }}
                />
              </div>
              <p className="progress-label">
                {progress.label} · {Math.round(progress.value * 100)}%
              </p>
            </>
          ) : (
            <>
              <h2>Drop an audio file here</h2>
              <p>WAV, MP3, M4A, OGG or FLAC. Or</p>
              <label className="button">
                Choose a file
                <input
                  type="file"
                  accept="audio/*"
                  hidden
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void handleFile(file);
                  }}
                />
              </label>
              {status === 'error' && <p className="error">{error}</p>}
            </>
          )}
        </div>
      )}

      {status === 'done' && (
        <main className="workspace">
          <aside className="sidebar">
            <div className="panel">
              <h3>Song</h3>
              <label className="field">
                <span>Tempo</span>
                <input
                  type="number"
                  min={30}
                  max={300}
                  step={0.1}
                  value={tempo}
                  onChange={(event) => setTempo(Number(event.target.value) || 120)}
                />
                <em>BPM</em>
              </label>
              <label className="field">
                <span>Key</span>
                <select
                  value={`${key.tonic}:${key.mode}`}
                  onChange={(event) => {
                    const [tonic, mode] = event.target.value.split(':');
                    setKey({ tonic: Number(tonic), mode: mode as 'major' | 'minor', confidence: 1 });
                  }}
                >
                  {NOTE_NAMES.flatMap((name, tonic) =>
                    (['major', 'minor'] as const).map((mode) => (
                      <option key={`${tonic}:${mode}`} value={`${tonic}:${mode}`}>
                        {name} {mode}
                      </option>
                    )),
                  )}
                </select>
              </label>
              <dl className="stats">
                <div>
                  <dt>Length</dt>
                  <dd>{duration.toFixed(1)}s</dd>
                </div>
                <div>
                  <dt>Notes</dt>
                  <dd>{noteCount}</dd>
                </div>
                <div>
                  <dt>Tracks</dt>
                  <dd>{tracks.length}</dd>
                </div>
              </dl>
            </div>

            <div className="panel">
              <h3>Model</h3>
              <p className="model-name">{MODELS[modelId].displayName}</p>
              <p className="model-licence">
                <a
                  href={MODELS[modelId].licenseUrl}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  {MODELS[modelId].license}
                </a>
              </p>
              <p className={MODELS[modelId].commercialUse ? 'licence-ok' : 'licence-restricted'}>
                {MODELS[modelId].commercialUse
                  ? 'Commercial use permitted by the model licence.'
                  : 'Non-commercial use only. Do not sell what this model produces.'}
              </p>
              <p className="model-summary">{MODELS[modelId].summary}</p>
            </div>

            <div className="panel">
              <h3>Tracks</h3>
              <ul className="track-list">
                {visibleTracks.map((track) => (
                  <li key={track.id}>
                    <span className="swatch" style={{ background: track.color }} />
                    <span className="track-name">{track.name}</span>
                    <span className="track-count">{track.notes.length}</span>
                    <button
                      className={`chip${track.muted ? ' off' : ''}`}
                      onClick={() => toggleMute(track.id)}
                      title={track.muted ? 'Unmute' : 'Mute'}
                    >
                      {track.muted ? 'M' : '●'}
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            {log.length > 0 && (
              <div className="panel">
                <h3>Notes</h3>
                <ul className="log">
                  {log.map((line, index) => (
                    <li key={index}>{line}</li>
                  ))}
                </ul>
              </div>
            )}
          </aside>

          <section className="editor">
            <div className="editor-toolbar">
              <button className="button" onClick={togglePlay}>
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <button className="button" onClick={stop}>Stop</button>
              <input
                className="seek"
                type="range"
                min={0}
                max={Math.max(0.1, duration)}
                step={0.01}
                value={Math.min(playhead, duration)}
                onChange={(event) => {
                  const time = Number(event.target.value);
                  setPlayhead(time);
                  playerRef.current?.seek(time);
                }}
              />
              <span className="time">
                {playhead.toFixed(2)}s / {duration.toFixed(2)}s
              </span>
              <span className="spacer" />
              <span className="hint">
                {selected.size > 0
                  ? `${selected.size} selected`
                  : 'Click to select · drag to move · drag the edge to resize'}
              </span>
              <button
                className="button"
                onClick={deleteSelected}
                disabled={selected.size === 0}
              >
                Delete
              </button>
              <button className="button" onClick={undo} disabled={history.past.length === 0}>
                Undo
              </button>
              <button className="button" onClick={redo} disabled={history.future.length === 0}>
                Redo
              </button>
            </div>

            <PianoRoll
              tracks={visibleTracks}
              selected={selected}
              tempo={tempo}
              playhead={playhead}
              theme={theme}
              onSelect={setSelected}
              onChange={applyVisibleChange}
              onEditStart={onEditStart}
              onEditEnd={onEditEnd}
            />
          </section>
        </main>
      )}
    </div>
  );
}
