# voidtrackr — AI Handoff

> Turn a recording into editable multi-track MIDI, entirely on your own
> machine. A browser app with a piano-roll editor, and a command-line
> converter for batch work.

**Status:** live on the LAN — `http://192.168.100.109/`
**Domain:** none (LAN-only, no tunnel, no public DNS)
**Brand identity:** "A recording in, a MIDI file out. Nothing is uploaded." — deep violet-black surfaces, one violet accent, mono for anything that identifies the tool.

## Project purpose

voidtrackr exists for one job: take a finished recording and give back MIDI you
can actually work with. It transcribes audio into notes grouped into parts,
estimates tempo and key, adds a drum track, and hands you a piano roll to fix
up whatever the model got wrong. It runs fully offline — no account, no upload,
no telemetry — because the audio is the user's own material and it should never
have to leave the machine it is on.

## Status

**Live on the LAN, not deployed publicly.** The app is served as static files
from a Proxmox LXC at `192.168.100.109`, and the CLI runs from `npm run cli`.
There is no domain and no tunnel, so the site is reachable only from the LAN.

The source lives in its own repository, `https://github.com/Gafail/voidtrackr`,
which is public. The folder is `~/Desktop/AudiotoMidi/` — the name on disk is
not the name of the tool, and the user asked for it to stay that way.

The tool is complete and verified: 41 tests pass, the type check is clean, the
production build succeeds, and the browser and the command line produce
byte-identical MIDI for all five models.

The one open product decision: **Okarina is the best model on this user's
material** (F1 85.3 % against published Harmonica output, versus 30.4 % for the
Basic Pitch pipeline) but it is licensed **CC BY-NC-SA 4.0 — non-commercial
only**. The default is still Basic Pitch (Apache-2.0) because it is
permissively licensed and works straight after `npm run model`. Flipping the
default is one line: `DEFAULT_MODEL` in `src/core/registry.ts`.

Measured over the user's own 19 tracks, **ByteDance's piano model (CC BY 4.0)
is the best model that can be sold**: it shares 63.9 % of Okarina's notes,
against 29.6 % for YourMT3+ and 23.4 % for Tsumugi. The numbers are in
`docs/model-research.md` section 6, and the report is `eval/index.html`.

## Architecture

- **Frontend:** React 18 + TypeScript + Vite 6. Canvas piano roll, no UI
  framework. `src/app/`.
- **Backend:** none. There is no server process, no API and no database. The
  browser does the work; the CLI does the same work in Node.
- **Models:** five, behind one interface. Basic Pitch (TensorFlow.js,
  Apache-2.0), Tsumugi (ONNX, MIT), YourMT3+ (ONNX, Apache-2.0), ByteDance's
  piano model (ONNX, CC BY 4.0) and Okarina (ONNX, CC BY-NC-SA). All are
  pinned by SHA-256.
- **Persistence:** none. Files are read from disk or dropped into the page,
  and the only output is a downloaded `.mid`.
- **Audio decoding:** the Web Audio API in the browser; a pure-JavaScript
  decoder (`src/core/audio-node.ts`) in Node, so the CLI needs no `ffmpeg`.
- **Model seam:** `TranscriptionModel` in `src/core/model.ts` — a model returns
  *notes grouped into parts*, not a matrix. Everything downstream (drums,
  tempo, key, MIDI writing) is shared between models.
- **Deploy pipeline:** `npm run build` → a static `dist/`, rsynced to the
  container's webroot by `deploy.sh`. nginx serves the files; there is no
  runtime on the box.
- **Key files:** `src/core/pipeline.ts` (the shared finaliser), `src/core/model.ts`
  (the seam), `src/core/version.ts` (provenance), `src/app/App.tsx` (the shell),
  `deploy.sh` (the LAN deploy), `ops/nginx/` (the vhost and headers).

## File map

```
src/core/     the pipeline — no DOM, no Node specifics, runs in both
  model.ts        the TranscriptionModel interface and the lock helpers
  registry.ts     the model catalogue and DEFAULT_MODEL
  pipeline.ts     the shared finaliser: tune → model → drums → tempo/key → MIDI
  basic-pitch.ts  the Basic Pitch model and its pinned weight loader
  okarina.ts      the Okarina model: CQT front end, offset-head decoding
  tsumugi.ts      the Tsumugi model: semi-CRF decode, boundary and instrument heads
  yourmt3.ts      the YourMT3+ model: spectrogram, token decode, event-to-note
  piano.ts        the ByteDance piano model: windowing and onset/offset pairing
  onnx.ts         the runtime-agnostic ONNX interface
  onnx-node.ts    ONNX Runtime for Node
  onnx-web.ts     ONNX Runtime for the browser (wasm, lazily imported)
  onnx-weights.ts the shared weight check against a lock file
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
  sha256.ts       the pure-JavaScript SHA-256 fallback
  midi.ts         the MIDI writer, including the key-signature fix
  version.ts      PIPELINE_VERSION, TOOL_NAME and the provenance stamp
  model.lock.json     the pinned Basic Pitch weights
  okarina.lock.json   the pinned Okarina weights
  tsumugi.lock.json   the pinned Tsumugi weights
  yourmt3.lock.json   the pinned YourMT3+ weights
  piano.lock.json     the pinned ByteDance piano weights

src/app/      the React app
  App.tsx        the shell: drop, transcribe, edit, export
  PianoRoll.tsx  the canvas piano roll
  editor.ts      editor state and the track colours
  playback.ts    the Web Audio player
  styles.css     the Void theme — tokens, layout, components

src/cli/      convert.ts — the command line converter

ops/nginx/    the deployment configuration
  voidtrackr.conf          the site vhost
  voidtrackr-headers.conf  the security headers, with this site's CSP

scripts/
  fetch-model.mjs        fetch and verify the Basic Pitch weights
  fetch-okarina.mjs      fetch and verify the Okarina weights
  fetch-onnx.mjs        fetch and verify the Tsumugi, YourMT3+ and piano weights
  make-test-wav.mjs      generate the synthetic test signal
  check-reproducibility.mts   repeat runs and both TF backends
  compare-midi.mjs       diff two MIDI files
  evaluate-midi.mjs      score one MIDI file against another
  evaluate-folder.mts    transcribe a folder, write eval/report.json + eval/index.html
  stamp-eval-midi.mts   add the provenance stamp to the evaluation's MIDI files

docs/model-research.md   the survey of every model, and what was measured
test/                    pipeline.test.ts, models.test.ts, sha256.test.ts, onnx-models.test.ts
public/                  404.html and robots.txt, copied into the build
eval/                    generated folder report (gitignored)
models/onnx/             the Okarina weights (gitignored)
AGENTS.md                this file
deploy.sh                build, upload and reload the LAN host
```

## How to deploy

The site is static: nginx serves the built files and there is no runtime on the
box.

```bash
npm install --no-audit --no-fund
npm run model              # fetch and verify the Basic Pitch weights
npm run model:okarina      # fetch and verify the Okarina weights (~21 MB)
npm run model:tsumugi      # fetch and verify the Tsumugi weights (~74 MB)
npm run model:yourmt3      # fetch and verify the YourMT3+ weights (~181 MB)
npm run model:piano        # fetch and verify the piano weights (~100 MB)
npm run build              # typecheck + vite build → dist/
./deploy.sh 192.168.100.109   # build, rsync, install the vhost, reload nginx
```

`deploy.sh` finds the umbrella repo (for the shared deploy key) by looking at
the siblings of this folder. It refuses to run against the placeholder address
`192.0.2.0`, which is what the checked-in vhost carries instead of a real one.

The host is registered in the umbrella's `ops/vm.sh`, so it can also be reached
directly:

```bash
cd ~/Desktop/TheVoidCo && ./ops/vm.sh voidtrackr 'hostname; nginx -t'
```

Before calling any change done:

```bash
npm run typecheck
npm test                              # 41 tests
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
- **The site loads but the model never appears** → check the browser console
  first. `crypto.subtle` is undefined outside a secure context, and the
  Content-Security-Policy may be blocking WebAssembly. Both have bitten; see
  "Known gotchas".
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
- **`crypto.subtle` only exists in a secure context.** HTTPS, or localhost.
  Served over plain HTTP on a LAN it is not there, and the model check died
  with "cannot read properties of undefined (reading 'digest')" — the app
  loaded and then silently did nothing. `src/core/sha256.ts` is the fallback;
  it gives the same digest, so the fingerprint does not move. There is a test
  that takes the platform's digest away and checks the answer does not change.
- **nginx has no MIME type for `.mjs`, and a browser will not run a module
  script without one.** Ubuntu's `mime.types` has `application/javascript js;`
  but nothing for `.mjs`, so the ONNX runtime's glue module was served as
  `application/octet-stream`. The browser refused it with "Expected a
  JavaScript-or-Wasm module script", ONNX reported "no available backend
  found", and the app sat there doing nothing with no visible error. The
  `/ort/` location in `ops/nginx/voidtrackr.conf` carries its own `types`
  block. Check the `Content-Type` of anything the browser loads as a module.
- **A strict CSP blocks WebAssembly.** The browser treats
  `WebAssembly.compile` as script evaluation, so `script-src 'self'` stops the
  Okarina model loading. The policy needs `'wasm-unsafe-eval'`. And a site
  cannot send two Content-Security-Policy headers and expect the looser one
  to win: the browser enforces every policy it receives. That is why
  `ops/nginx/voidtrackr-headers.conf` exists instead of including the
  umbrella's shared snippet.
- **`'wasm-unsafe-streaming'` is a Firefox source expression only.** Chrome
  logs "invalid source … It will be ignored" for it on every page load.
- **The theme is one set of tokens, not two stylesheets.** `styles.css` uses
  `light-dark()`, which reads the used value of `color-scheme`. The system
  decides until the reader picks otherwise, and `data-theme` on the document
  overrides it. The piano roll paints itself, so the theme is passed to it as
  a prop and it repaints when it changes — the CSS cannot reach a canvas.
- **`@tonejs/midi` writes the key signature wrong.** It writes the position of
  the key name in its table as if it were the signed key number, so C major
  comes out as fourteen sharps, and it writes the mode as a string, so minor
  keys are written as major. `src/core/midi.ts` rewrites those two bytes after
  the file is written. Do not remove that.
- **Two ports deliberately depart from upstream, and they are marked in the
  code.** ByteDance's `note_detection_with_onset_offset_regress` writes
  `if bgn:`, which is false for frame 0, so every note starting on the very
  first frame is thrown away. `src/core/piano.ts` tests for `null` instead
  and keeps those notes. The comment at the call site says so. Do not
  "fix" it back to upstream's behaviour without knowing what you are
  changing. The other departure is the Tsumugi port, which follows upstream
  exactly — the silence gate is an RMS over the whole window rather than a
  standard deviation, the Viterbi runs over the frames the graph marks as
  valid rather than all 345 of them, and a gap merge takes the
  continuation's offset rather than keeping either one.
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
- **The build ships the ONNX runtime's wasm twice.** Once at `/ort/` (served
  from `node_modules`, and the path the app asks for) and once at `/assets/`
  (emitted by Vite because the runtime imports it). The `/assets/` copy is
  dead weight — about 15 MB — and can be excluded if that is ever worth doing.

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
  `--accent #8b7bf7`, `--accent-ink #b7adff`. The stylesheet holds one set of
  tokens and lets `light-dark()` pick between the two palettes; a theme
  toggle in the header overrides the reader's system preference, and nothing
  is stored.
- **Type:** `--font-sans` for prose, `--font-mono` for anything that
  identifies the tool — the wordmark, the kicker, panel headings, numbers.
- **Background:** the `.void-bg` radial gradients and starfield, drifting
  slowly. It respects `prefers-reduced-motion`.

## Live operations state

- **Site:** `http://192.168.100.109/` — a Proxmox LXC named `voidtrackr`,
  running nginx and nothing else. The webroot is `/var/www/voidtrackr` and the
  vhost is `/etc/nginx/sites-available/voidtrackr.conf`.
- **Deploy:** `./deploy.sh 192.168.100.109` from this folder. It builds,
  rsyncs, installs the vhost and headers, disables the template's own base
  site, validates with `nginx -t`, reloads, and smoke tests.
- **Host registration:** `ops/vm.sh` in the umbrella repo, under `voidtrackr`,
  with the shared deploy key. `./ops/vm.sh voidtrackr 'hostname'`.
- **SSH:** the container's host keys were regenerated on provisioning — the
  clone had inherited the template's identity. The old ones are in
  `/root/.trash-removed/` on the box.
- **Current pipeline version:** 0.3.1 (`src/core/version.ts`) — the Tsumugi
  and Piano ports brought back in line with upstream at the edges of a
  recording. The notes of most recordings are unchanged from 0.2.0.
- **Models in the catalogue:** Basic Pitch (Apache-2.0, the default),
  Tsumugi (MIT), YourMT3+ (Apache-2.0), Piano Transcription (CC BY 4.0),
  Okarina (CC BY-NC-SA 4.0, non-commercial).
- **Default model:** Basic Pitch, `DEFAULT_MODEL` in `src/core/registry.ts`.
- **The licence is shown in the app**, next to the model picker, because it
  decides whether the output can be sold. `commercialUse` in the registry is
  the model's own grant, not a legal opinion: MAESTRO is in the training mix
  of most of them and is CC BY-NC-SA 4.0.
- **Pinned weights:** `src/core/model.lock.json` (Basic Pitch),
  `src/core/okarina.lock.json`, `src/core/tsumugi.lock.json`,
  `src/core/yourmt3.lock.json` and `src/core/piano.lock.json`. All are
  checked before the model runs, and a mismatch stops the conversion.
- **Repository:** `https://github.com/Gafail/voidtrackr` — public, branch
  `main`. TheVoidCo holds **no copy of this source**; its root `README.md`
  holds the pointer row and its site renders it.
- **Git:** this folder is its own repository. There is no CI and no remote
  other than origin.
