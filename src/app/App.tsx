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
import { DEFAULT_MODEL, listModels, ModelId, MODELS } from '@core/registry';
import { transcribe } from '@core/pipeline';
import { buildMidiFile, toArrayBuffer } from '@core/midi';
import { provenanceFor } from '@core/version';
import { NOTE_NAMES } from '@core/constants';
import { KeyInfo, TempoInfo } from '@core/types';
import { PianoRoll } from './PianoRoll';
import { EditorTrack, fromEditorTracks, toEditorTracks } from './editor';
import { Player } from './playback';

const MODEL_URL = '/model/model.json';

/** Where the Okarina weights are served from. */
const OKARINA_URL = '/model-onnx/';

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
    const response = await fetch(new URL(name, new URL(OKARINA_URL, window.location.href)).href);
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
  return OkarinaModel.load(runner, OKARINA_URL, 'm', fingerprint);
}

/** How many edits can be undone. */
const HISTORY_LIMIT = 60;

type Status = 'idle' | 'working' | 'done' | 'error';

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
    if (id === 'okarina') {
      model = await loadOkarinaModel();
    } else {
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
    playerRef.current?.setTracks(tracks);
  }, [tracks]);

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
      fromEditorTracks(tracks),
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
    () => tracks.reduce((total, track) => total + track.notes.length, 0),
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
          <label className="field model-picker">
            <span>Model</span>
            <select
              value={modelId}
              onChange={(event) => setModelId(event.target.value as ModelId)}
              disabled={status === 'working'}
            >
              {listModels().map((model) => (
                <option key={model.id} value={model.id}>
                  {model.displayName}
                </option>
              ))}
            </select>
            <em>{MODELS[modelId].license}</em>
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
              <h3>Tracks</h3>
              <ul className="track-list">
                {tracks.map((track) => (
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
              tracks={tracks}
              selected={selected}
              tempo={tempo}
              playhead={playhead}
              onSelect={setSelected}
              onChange={setTracks}
              onEditStart={onEditStart}
              onEditEnd={onEditEnd}
            />
          </section>
        </main>
      )}
    </div>
  );
}
