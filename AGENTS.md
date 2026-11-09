# voidtrackr — AI Handoff

> Turn a recording into editable multi-track MIDI, entirely on your own
> machine. A browser app with a piano-roll editor, and a command-line
> converter for batch work.

**Status:** local-only — no deployment yet
**Domain:** none (not deployed)
**Brand identity:** "A recording in, a MIDI file out. Nothing is uploaded." — deep violet-black surfaces, one violet accent, mono for anything that identifies the tool.

## Project purpose

voidtrackr exists for one job: take a finished recording and give back MIDI you
can actually work with. It transcribes audio into notes grouped into parts,
estimates tempo and key, adds a drum track, and hands you a piano roll to fix
up whatever the model got wrong. It runs fully offline — no account, no upload,
no telemetry — because the audio is the user's own material and it should never
have to leave the machine it is on.

## Status

**Local-only. Not deployed.** The app runs from `npm run dev` on
`127.0.0.1:5173`; the CLI runs from `npm run cli`. There is no server, no
container, no domain and no credentials, so there is no `CREDENTIALS.md` for
this project.

The tool is complete and verified: 27 tests pass, the type check is clean, the
production build succeeds, and the browser and the command line produce
byte-identical MIDI for both models.

The one open product decision: **Okarina is the better model on this user's
material** (F1 85.3 % against published Harmonica output, versus 30.4 % for the
Basic Pitch pipeline) but it is licensed **CC BY-NC-SA 4.0 — non-commercial
only**. The default is still Basic Pitch (Apache-2.0) because it is
permissively licensed and works straight after `npm run model`. Flipping the
default is one line: `DEFAULT_MODEL` in `src/core/registry.ts`.

## Architecture

- **Frontend:** React 18 + TypeScript + Vite 6. Canvas piano roll, no UI
  framework. `src/app/`.
- **Backend:** none. There is no server process, no API and no database. The
  browser does the work; the CLI does the same work in Node.
- **Models:** two, behind one interface. Basic Pitch (TensorFlow.js, Apache-2.0)
  and Okarina (ONNX Runtime, CC BY-NC-SA). Both are pinned by SHA-256.
- **Persistence:** none. Files are read from disk or dropped into the page,
  and the only output is a downloaded `.mid`.
- **Audio decoding:** the Web Audio API in the browser; a pure-JavaScript
  decoder (`src/core/audio-node.ts`) in Node, so the CLI needs no `ffmpeg`.
- **Model seam:** `TranscriptionModel` in `src/core/model.ts` — a model returns
  *notes grouped into parts*, not a matrix. Everything downstream (drums,
  tempo, key, MIDI writing) is shared between models.
- **Deploy pipeline:** none. `npm run build` produces a static `dist/`. It has
  never been published anywhere.
- **Key files:** `src/core/pipeline.ts` (the shared finaliser), `src/core/model.ts`
  (the seam), `src/core/version.ts` (provenance), `src/app/App.tsx` (the shell).

## File map

```
src/core/     the pipeline — no DOM, no Node specifics, runs in both
  model.ts        the TranscriptionModel interface and the lock helpers
  registry.ts     the model catalogue and DEFAULT_MODEL
  pipeline.ts     the shared finaliser: tune → model → drums → tempo/key → MIDI
  basic-pitch.ts  the Basic Pitch model and its pinned weight loader
  okarina.ts      the Okarina model: CQT front end, offset-head decoding
  onnx.ts         the runtime-agnostic ONNX interface
  onnx-node.ts    ONNX Runtime for Node
  onnx-web.ts     ONNX Runtime for the browser (wasm, lazily imported)
  audio.ts        decoding in the browser
  audio-node.ts   decoding in Node
  sample-rate.ts  reading the real sample rate out of the file's header
  dsp.ts          FFT, spectrogram, autocorrelation
  constants.ts    the shared numbers, including the frame→time conversion
  melody.ts       the Viterbi melody tracer
  postprocess.ts  frame matrix → notes
  drums.ts        the per-band spectral flux drum detector
  tempo.ts        comb-filter tempo estimation and key detection
  tuning.ts       A440 tuning correction
  beat.ts         the model-independent beat grid tracker (not wired in)
  notes.ts        note naming and the shared note helpers
  types.ts        the shared data types
  midi.ts         the MIDI writer, including the key-signature fix
  version.ts      PIPELINE_VERSION, TOOL_NAME and the provenance stamp
  model.lock.json     the pinned Basic Pitch weights
  okarina.lock.json   the pinned Okarina weights

src/app/      the React app
  App.tsx        the shell: drop, transcribe, edit, export
  PianoRoll.tsx  the canvas piano roll
  editor.ts      editor state and the track colours
  playback.ts    the Web Audio player
  styles.css     the Void theme — tokens, layout, components

src/cli/      convert.ts — the command line converter

scripts/
  fetch-model.mjs        fetch and verify the Basic Pitch weights
  fetch-okarina.mjs      fetch and verify the Okarina weights
  make-test-wav.mjs      generate the synthetic test signal
  check-reproducibility.mts   repeat runs and both TF backends
  compare-midi.mjs       diff two MIDI files
  evaluate-midi.mjs      score one MIDI file against another
  evaluate-folder.mts    transcribe a folder, write eval/report.json + eval/index.html

docs/model-research.md   the survey of every model, and what was measured
test/                    pipeline.test.ts, models.test.ts
eval/                    generated folder report (gitignored)
models/onnx/             the Okarina weights (gitignored)
Music Import/            the user's own songs (gitignored)
AGENTS.md                this file
```

## How to deploy

There is nothing deployed, so there is nothing to push. To get a local build:

```bash
npm install --no-audit --no-fund
npm run model              # fetch and verify the Basic Pitch weights
npm run model:okarina      # fetch and verify the Okarina weights (~21 MB)
npm run build              # typecheck + vite build → dist/
npm run preview            # serve dist/ on 127.0.0.1:4173
```

`dist/` is about 52 MB, because the ONNX runtime's WebAssembly (15 MB) and the
Okarina weights (21 MB) are copied into it. Without them it is about 3 MB, and
only Basic Pitch is available in the browser.

Before calling any change done:

```bash
npm run typecheck
npm test                              # 27 tests
npm run check:reproducibility
```

## How to debug

- **The app says it cannot read the model file** → the weights are not there.
  Run `npm run model` (Basic Pitch) or `npm run model:okarina` (Okarina).
- **"The Okarina weights do not match the pinned version"** → the download was
  truncated or the file was changed. Re-run `npm run model:okarina`; do not
  bypass the check, because a different weight file means different output.
- **"Could not read the Okarina weights … (404)" in the browser** → the dev
  server is not serving `/model-onnx/`. That route is added by the `onnxAssets`
  plugin in `vite.config.ts`; check that `models/onnx/` exists.
- **The browser and the CLI disagree** → they should not. Check the sample
  rate first (`src/core/sample-rate.ts`), then the ONNX thread count.
- **Notes drift later in a long recording** → a frame index was converted to
  seconds without `modelFrameToTime`. See "Known gotchas".
- **The same file gives different notes on different machines** → the audio was
  decoded at the sound card's rate rather than the file's own rate.
- **A model change altered the output unexpectedly** → check
  `PIPELINE_VERSION` in `src/core/version.ts`; it is stamped into every
  exported file, so the stamp and the notes should move together.
- **`npm run cli` crashes on import** → the broken `@tensorflow/tfjs-node`
  4.21/4.22 build; see "Known gotchas".
- **To compare two MIDI files** → `node scripts/compare-midi.mjs a.mid b.mid`.
  To score one against a reference → `node scripts/evaluate-midi.mjs ref.mid est.mid`.

## Known gotchas

- **A model frame is not a moment in the recording.** The model's windows
  overlap, so frames inside a window sit closer together in time than the
  audio they cover. `modelFrameToTime` in `src/core/constants.ts` corrects for
  that, and *every* place that turns a frame index into a second must use it.
  Until 0.2.0 two places did not (`toTimedNotes` in `src/core/postprocess.ts`
  and the tracer in `src/core/melody.ts`), and every note drifted about 5 ms
  later for every second of audio — a three-minute song ended up roughly half
  a second out of step with its own beginning. There is a regression test
  (`keeps the timing steady over a long recording`); do not add a
  frame-to-seconds conversion without one.
- **`timeToModelFrame` must stay the inverse of `modelFrameToTime`.** It
  searches for the window the time falls in and picks the closest frame inside
  it. A naive `time / MODEL_FRAME_SECONDS` is wrong by up to the window
  overlap.
- **`@tonejs/midi` writes the key signature wrong.** It writes the position of
  the key name in its table as if it were the signed key number, so C major
  comes out as fourteen sharps, and it writes the mode as a string, so minor
  keys are written as major. `src/core/midi.ts` rewrites those two bytes after
  the file is written. Do not remove that.
- **`@tensorflow/tfjs-node` 4.21 and 4.22 ship a broken build.** Its published
  code requires Node's own `util` module where it meant to require
  TensorFlow.js's utilities, and then calls a function that does not exist
  there. `src/cli/convert.ts` adds the missing function before the backend
  loads.
- The native backend registers itself as `tensorflow`, not `node`.
- **ONNX Runtime reduces in an order that depends on the thread count.** Both
  `src/core/onnx-node.ts` and `src/core/onnx-web.ts` pin `intraOpNumThreads`
  to 1, or two runs of the same file could differ in the last bits.
- **Okarina's timebase is measured, never assumed.** The frame period is
  exactly 0.032 s and the leading context exactly 0.976 s, fitted by least
  squares against tone bursts at known times. The earlier constants
  (0.03204 / 0.9705) were wrong and accumulated about 20 ms of timing error.
- **Okarina decodes notes from the offset head**, not from a sustain
  threshold — the sustain decays over several frames, so threshold-decided
  ends were badly over-long. This alone took F1 from 70.1 % to 81.5 %.
- **Do not use `pkill -f "vite --port 5173"` to stop the dev server.** The
  pattern matches the shell running the command. Use `ps aux | grep "[v]ite"`
  and kill by PID.
- **The piano roll paints itself**, so it does not follow the stylesheet.
  It reads the CSS tokens with `token()` in `src/app/PianoRoll.tsx` and
  redraws on a `prefers-color-scheme` change. A new colour has to be added
  there as well as in `styles.css`.

## Brand identity

- **Name:** `voidtrackr` — lowercase, no separators, starts with `void`, under
  twelve characters. The wordmark in the app is mono and lowercase with a
  violet dot mark.
- **Tagline:** "A recording in, a MIDI file out. Nothing is uploaded."
- **Voice:** calm, plain, a little understated. State what the thing does and
  what it will not do. No exclamation marks, no growth language, no
  "revolutionary".
- **Promise:** the audio never leaves the machine. No accounts, no tracking,
  no cookies, no analytics, no telemetry.
- **Palette:** the Void tokens, shared with TheVoidCo's own site —
  `--void-900 #07070b`, `--void-800 #0c0c13`, `--void-700 #12121b`,
  `--line #232330`, `--ink #ecebf5`, `--ink-2 #a7a4bd`, `--ink-3 #74718c`,
  `--accent #8b7bf7`, `--accent-ink #b7adff`. Light theme overrides the same
  variables; nothing else in the stylesheet knows which theme is active.
- **Type:** `--font-sans` for prose, `--font-mono` for anything that
  identifies the tool — the wordmark, the kicker, panel headings, numbers.
- **Background:** the `.void-bg` radial gradients and starfield, drifting
  slowly. It respects `prefers-reduced-motion`.

## Live operations state

Nothing is deployed. No server, no container, no domain, no tunnel, no
credentials.

- **Current pipeline version:** 0.3.0 (`src/core/version.ts`) — the rename to
  voidtrackr; the notes are unchanged from 0.2.0.
- **Models in the catalogue:** Basic Pitch (Apache-2.0, the default),
  Okarina (CC BY-NC-SA 4.0, non-commercial).
- **Default model:** Basic Pitch, `DEFAULT_MODEL` in `src/core/registry.ts`.
- **Pinned weights:** `src/core/model.lock.json` (Basic Pitch) and
  `src/core/okarina.lock.json` (Okarina). Both are checked before the model
  runs, and a mismatch stops the conversion.
- **Registry:** TheVoidCo's root `README.md` holds the pointer row for
  `voidtrackr`; TheVoidCo holds **no copy of this source**. The folder is
  `~/Desktop/voidtrackr/`.
- **Git:** this folder is not a git repository. There is no CI and no remote.
