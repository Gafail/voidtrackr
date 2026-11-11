# voidtrackr — AI Handoff

> Turn a recording into editable multi-track MIDI, entirely on your own
> machine. A web app with a piano-roll editor. The browser does all of it,
> and nothing is ever uploaded.

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
from a Proxmox LXC at `192.168.100.109`. There is no domain and no tunnel, so
the site is reachable only from the LAN.

**It is a web app and nothing else.** There is no HTTP API and no command
line: the browser is the only way to use it. An API was built and then
removed — it took a recording, which is the one thing that breaks the
promise; and a command line was never asked for in the first place.

The source lives in its own repository, `https://github.com/Gafail/voidtrackr`,
which is public. The folder is `~/Desktop/AudiotoMidi/` — the name on disk is
not the name of the tool, and the user asked for it to stay that way.

The tool is complete and verified: 89 tests pass, the type check is clean, the
production build succeeds, and the browser and the Node path the tests use
produce byte-identical MIDI for all five models.

**Okarina is the best model on this user's material** (F1 85.3 % against
published Harmonica output, versus 30.4 % for the Basic Pitch pipeline) but it
is licensed **CC BY-NC-SA 4.0 — non-commercial only**, so it cannot be the
default for anything that will be sold.

Measured over the user's own 19 tracks, **ByteDance's piano model (CC BY 4.0)
is the best model that can be sold**: it shares 63.9 % of Okarina's notes,
against 29.6 % for YourMT3+ and 23.4 % for Tsumugi, and it lands its onsets
within 0.1 points of Okarina's on the beat. It is the default as of 0.4.0.
The numbers are in `docs/model-research.md` section 6, and the report is
`eval/index.html`.

Flipping the default is one line: `DEFAULT_MODEL` in `src/core/registry.ts`.
`npm run model` follows it, so the fetch script never has to be told which
model the default is.

**The output is arranged for two hands by an optional stage.** The piano
model's transcription is accurate and unplayable at the same time: the same
key struck again within 60 ms is 35 % of all its notes, a chord can reach
three octaves, and the busiest half second holds 41 notes. The pianify stage
folds the re-triggers together, divides the chords between a left and a right
hand at middle C, and drops what neither hand could hold. It keeps half the
notes and the notes it keeps are a seventh more likely to be real. The
research, the design and the cost are in `docs/pianify.md`.

## Architecture

- **Frontend:** React 18 + TypeScript + Vite 6. Canvas piano roll, no UI
  framework. `src/app/`.
- **Backend:** none. There is no server process, no API and no database. The
  browser does all of the work: it decodes the audio, runs the model and
  writes the MIDI file, on the machine that opened the page. The box serves
  static files and nothing else.
- **Models:** five, behind one interface. Basic Pitch (TensorFlow.js,
  Apache-2.0), Tsumugi (ONNX, MIT), YourMT3+ (ONNX, Apache-2.0), ByteDance's
  piano model (ONNX, CC BY 4.0) and Okarina (ONNX, CC BY-NC-SA). All are
  pinned by SHA-256.
- **Persistence:** none. Files are dropped into the page, and the only output
  is a downloaded `.mid`. Nothing is ever uploaded: there is no endpoint, no
  service and no process that could accept one.
- **Audio decoding:** the Web Audio API in the browser. The pure-JavaScript
  decoder (`src/core/audio-node.ts`) is for the tests and the measurement
  scripts, so nothing needs `ffmpeg`.
- **Model seam:** `TranscriptionModel` in `src/core/model.ts` — a model returns
  *notes grouped into parts*, not a matrix. Everything downstream (drums,
  tempo, key, MIDI writing) is shared between models.
- **Frequency range:** two bounds in hertz that choose which notes are kept,
  in the app's header. It selects *notes*
  after the model has run; the audio is never filtered, because filtering the
  audio was measured and it is much worse. See "Known gotchas".
- **Deploy pipeline:** `npm run build` → a static `dist/`, rsynced to the
  container's webroot by `deploy.sh`. nginx serves the files; there is no
  runtime on the box.
- **Key files:** `src/core/pipeline.ts` (the shared finaliser), `src/core/model.ts`
  (the seam), `src/core/version.ts` (provenance), `src/app/App.tsx` (the shell),
  `deploy.sh` (the LAN deploy), `ops/nginx/` (the vhost and headers).

## File map

```
src/core/     the pipeline — no DOM, no Node specifics
  convert.ts     one conversion, from audio to MIDI — the Node path the
                 tests and the measurement scripts use
  model-loader.ts  loading a model from disk, and the file:// fetch
  backend.ts     the native TensorFlow.js backend, when it is installed
  model.ts        the TranscriptionModel interface and the lock helpers
  registry.ts     the model catalogue and DEFAULT_MODEL
  pipeline.ts     the shared finaliser: tune → model → drums → tempo/key → MIDI
  basic-pitch.ts  the Basic Pitch model and its pinned weight loader
  okarina.ts      the Okarina model: CQT front end, offset-head decoding
  tsumugi.ts      the Tsumugi model: semi-CRF decode, boundary and instrument heads
  yourmt3.ts      the YourMT3+ model: spectrogram, token decode, event-to-note
  piano.ts        the ByteDance piano model: windowing and onset/offset pairing
  onnx.ts         the runtime-agnostic ONNX interface
  onnx-node.ts    ONNX Runtime for Node — the tests and the scripts
  onnx-web.ts     ONNX Runtime for the browser (wasm, lazily imported)
  onnx-weights.ts the shared weight check against a lock file
  audio.ts        decoding in the browser
  audio-node.ts   decoding in Node — the tests and the scripts
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
  range.ts        the frequency range: which notes to keep
  quantise.ts     the grid: snapping notes to a fixed rate
  pianify.ts      the arrangement for two hands: collapse, divide, trim, thin
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


ops/nginx/    the deployment configuration
  voidtrackr.conf          the site vhost
  voidtrackr-headers.conf  the security headers, with this site's CSP

scripts/
  memory-models.mts  how much memory one model needs
  memory-probe.mts   the memory, window by window
  memory-stages.mts  the memory, stage by stage
  fetch-model.mjs        fetch the default model's weights, whichever that is
  fetch-basic-pitch.mjs  fetch and verify the Basic Pitch weights
  fetch-okarina.mjs      fetch and verify the Okarina weights
  fetch-onnx.mjs        fetch and verify the Tsumugi, YourMT3+ and piano weights
  make-test-wav.mjs      generate the synthetic test signal
  check-reproducibility.mts   repeat runs and both TF backends
  compare-midi.mjs       diff two MIDI files
  evaluate-midi.mjs      score one MIDI file against another
  evaluate-folder.mts    transcribe a folder, write eval/report.json + eval/index.html
  stamp-eval-midi.mts   add the provenance stamp to the evaluation's MIDI files
  pianify-stats.mjs     how far a transcription is from something playable
  pianify-value.mts     whether the model's loudness says a note is real
  pianify-experiment.mts  the stage, measured: playability against accuracy
  pianify-compare.mts   the shipped stage against the corrected one
  pianify-pitch.mts     where the music sits, per file
  pianify-split.mts     where the hands should meet
  pianify-crossing.mts  how far a note may cross to the other hand
  pianify-melody-keep.mts  whether the melody of a chord survives
  pianify-order.mts     the old note order against the new one
  pianify-order-detail.mts  the same, by chord width
  pianify-duration.mts  how long the notes are, and how many a hand holds
  pianify-merge-sweep.mts  the fold: the gap and how long a folded note is
  pianify-fold-depth.mts  the gap between one strike on a key and the next
  pianify-long-notes.mts  what is inside the long notes
  pianify-look.mts      the arrangement as a text piano roll
  onnx-threads.mts      the graph alone, at each thread count
  onnx-threads-pipeline.mts  the whole conversion, at each thread count
  onnx-threads-models.mts    the exported file, for every model
  onnx-threads-material.mts  the exported file, over the user's own tracks
  onnx-threads-midi.mts  the exported file, note by note
  onnx-threads-diff.mts  the notes that differ, if any do
  onnx-threads-digest.mts  the raw output of the graph, at each thread count

docs/model-research.md   the survey of every model, and what was measured
docs/pianify.md          the arrangement stage: the research, the design, the cost
docs/threading.md        the thread count: why it is pinned, and what it buys
test/                    pipeline.test.ts, models.test.ts, sha256.test.ts, onnx-models.test.ts
                         range.test.ts, quantise.test.ts, pianify.test.ts
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
npm run model              # fetch and verify the default model's weights
npm run model:piano        # fetch and verify the piano weights (~100 MB)
npm run model:basic-pitch  # fetch and verify the Basic Pitch weights (~900 KB)
npm run model:okarina      # fetch and verify the Okarina weights (~21 MB)
npm run model:tsumugi      # fetch and verify the Tsumugi weights (~74 MB)
npm run model:yourmt3      # fetch and verify the YourMT3+ weights (~181 MB)
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
npm test                              # 89 tests
npm run check:reproducibility
```

## How to debug

- **The app says it cannot read the model file** → the weights are not there.
  Run `npm run model` for the default model, or `npm run model:<name>` for a
  specific one. The error names the command.
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
- **The Node path crashes on import** → the broken `@tensorflow/tfjs-node`
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
- **Never filter the audio by frequency band before the model.** This was
  measured on the user's own material and it is much worse than one pass over
  the whole mix: transcribing a recording low-passed at 400 Hz and keeping the
  notes below 400 Hz recovered 14 of Okarina's low notes, where transcribing
  the whole mix and keeping the same notes recovered 142. A model reads a note
  from its whole harmonic series, and a band-limited recording takes that
  evidence away — a model handed only harmonics invents notes, which is how
  the 1200–2000 Hz run came back with 2466 notes across the whole keyboard,
  63 % of them outside the band that produced them. The tool's frequency range
  therefore selects *notes* after the model has run (`src/core/range.ts`), and
  never touches the audio. The numbers are in `docs/model-research.md`,
  section 6.
- **`@tensorflow/tfjs-node` 4.21 and 4.22 ship a broken build.** Its published
  code requires Node's own `util` module where it meant to require
  TensorFlow.js's utilities, and then calls a function that does not exist
  there. `src/core/backend.ts` adds the missing function before the backend
  loads.
- The native backend registers itself as `tensorflow`, not `node`.
- **ONNX Runtime reduces in an order that depends on the thread count, and
  this used to be pinned to one.** It is four now, in both runtimes, from
  `ONNX_THREADS` in `src/core/onnx.ts`. The pin was justified by the raw
  output of a graph differing in its last bits, and that part is true: in
  Node the piano graph's raw output *does* differ at four threads and above
  (one and two agree; four and eight agree with each other but not with one
  and two). The **exported MIDI does not**: it is byte-identical at 1/2/4/8
  threads for every model in the catalogue, in the browser and in the Node
  path, and over the user's own tracks. The browser at any thread count and
  the Node path at any thread count both produce the same file. The
  differences are in the last bits of a float and the model's thresholds turn
  them into the same notes. `docs/threading.md` has the full record.
- **The browser cannot thread on plain HTTP at a LAN address, whatever the
  headers say.** The runtime needs `SharedArrayBuffer`, which needs
  cross-origin isolation, which needs `Cross-Origin-Opener-Policy: same-origin`
  and `Cross-Origin-Embedder-Policy: require-corp` — both now in
  `ops/nginx/voidtrackr-headers.conf`. But cross-origin isolation also needs
  a **secure context**, and a page served over plain HTTP at a LAN address is
  not one. Measured: with the headers on, `http://192.168.100.109/` still
  reports `isSecureContext: false`, `crossOriginIsolated: false` and no
  `SharedArrayBuffer`, and the runtime falls back to one thread on its own
  (`ort.mjs:1784-1795`, "Falling back to single-threading"). The browser has
  to be started with
  `--unsafely-treat-insecure-origin-as-secure=http://192.168.100.109`, or
  the site has to be served over https with a certificate the client trusts.
  The app reads the count the browser is actually allowing and shows it in
  the header, so a browser without the flag says "1 thread" rather than
  running slowly in silence.
- **Eight threads is not always the best number.** Measured per model: the
  piano model keeps improving to eight threads, but Okarina is *slower* at
  eight than at four, and Tsumugi and YourMT3+ both turn round at eight.
  Four is the number to pick.
- **The ONNX runtime's memory arena has to be off.** The arena is a pool
  that keeps allocations for the next run. It is faster, and it holds far
  more memory than it needs: measured on 12 November 2026, the piano model
  peaked at **1.15 GB** with the arena on and **661 MB** with it off. The
  notes are byte-identical and the conversion is about 5 % slower.
  `DETERMINISTIC_SESSION_OPTIONS` in `src/core/onnx.ts` turns it off.
- **What each model needs, measured** (one model per process, because the
  high-water mark is for the whole process): Okarina 460 MB, piano 661 MB,
  YourMT3+ 1056 MB, Tsumugi 1183 MB. This is what the **client** needs, not
  the box: the model runs wherever the recording is, so a client that wants
  Tsumugi needs a gigabyte and a half to itself. The box only holds the
  weights as files, so it serves all five to every client, whatever the
  box has.
- **TensorFlow.js cannot fetch a `file://` URL without the native backend.**
  It fetches the model's manifest and its weight shards itself, and Node's
  own `fetch` does not understand that scheme. The native backend patches
  it; the Node path brings its own (`fileFetch` in
  `src/core/model-loader.ts`). Without it, Basic Pitch fails with "fetch
  failed".
- **A model's weights are not at `/model-onnx/<id>`.** The layout is not
  guessable from the model's id: Basic Pitch is a TensorFlow.js model and
  its files are in `/model/`, and Okarina's ONNX files sit at the root of
  `/model-onnx/` rather than in a folder of their own. The paths are in
  the lock files, and the folder names are constants next to the models
  (`PIANO_DIR`, `TSUMUGI_DIR`, `YOURMT3_DIR` in `src/core/`). Change one
  of those and the lock file and the site's paths have to change with it.
- **There must never be an endpoint that takes a recording.** This is the
  rule the whole project rests on: a recording goes in and a MIDI file
  comes out, on the machine that holds the recording, and nothing is
  uploaded. An upload endpoint was built and then removed, because it was
  the one thing that broke the promise, and "it is on our own LAN" is not
  a reason to break it for anyone else on the network. The tool is a
  static site now: nginx serves files and there is no process behind it.
  **If a function cannot be done without sending the recording to the
  box, the function does not get built.**
- **Okarina's timebase is measured, never assumed.** The frame period is
  exactly 0.032 s and the leading context exactly 0.976 s, fitted by least
  squares against tone bursts at known times. The earlier constants
  (0.03204 / 0.9705) were wrong and accumulated about 20 ms of timing error.
- **Okarina decodes notes from the offset head**, not from a sustain
  threshold — the sustain decays over several frames, so threshold-decided
  ends were badly over-long. This alone took F1 from 70.1 % to 81.5 %.
- **The grid is not the beat, and the two must not be confused.**
  `src/core/quantise.ts` snaps the notes to a fixed number of slots per
  second, for something that runs at a fixed rate — a game rendering 30
  frames a second. It is deliberately *not* the beat grid in `beat.ts`,
  which is still unwired: the beat grid needs a tempo and a phase, it is
  only accurate to about 12 ms, and the model's own onsets are better than
  it, so snapping to the beat would replace a good time with a worse one.
  Snapping to a frame grid is the opposite case — the frame is what the
  note will be drawn on, so agreeing with it is the more accurate answer.
  The grid never touches the tempo or the key, which are estimated from the
  notes as the model found them. A MIDI file stores ticks rather than
  seconds, so a phase lands on the nearest tick — 1.04 ms at 480 PPQ and
  120 BPM — and the spacing between the slots is exact.
- **Playability is a property of the notes together, not of one note.**
  This is the whole idea behind `src/core/pianify.ts`, and it is the idea
  behind the checklist models for piano fingering (Srivatsan &
  Berg-Kirkpatrick, ISMIR 2022, arXiv 2209.05622): per-note accuracy does
  not measure whether a piece can be played. A note that is right can still
  be one of eleven sounding at once. The limits the stage enforces are five
  notes a hand at a time, twelve semitones of reach and ten notes a second
  a hand, and changing any of them changes the output.
- **"At the same time" has to mean the same thing in the code and in the
  measurement.** The stage works in a window of 25 milliseconds either side
  of a note, and `scripts/pianify-experiment.mts` measures the same window.
  An earlier version clustered the notes of the whole piece and split each
  cluster, which let a hand hide a note outside its reach behind the other
  hand's note, and the repair that moved notes between the hands ping-ponged
  — 852 moves in a file with 790 notes in it. Both were fixed by placing the
  notes one at a time against a symmetric window, which cannot oscillate.
  If you change the window, change it in `pianify.ts` and in the experiment
  together, or the numbers will not mean anything.
- **The loudness head is a weak signal, and it is only used as a tie.**
  Over the user's 19 tracks, the notes another model also found have a mean
  velocity of 0.689 and the ones only the piano model found have 0.662. That
  is not enough to tell a real note from a false one, so the stage uses it
  to break ties and nothing more. The alternative — agreement between
  models — would be a better noise detector and would double the time of
  every conversion.
- **The split between the hands is middle C, and it is not a parameter to
  tune.** The first version worked it out from the material, choosing the
  pitch that left the hands least overstretched. That rule is wrong on a mix:
  the median of the texture sits at MIDI 57 to 76, well above middle C, so a
  split that balances the load lands in the treble. The left hand then played
  the melody up to MIDI 87 and the right hand played the bass down to 46 — the
  two hands playing each other's music. Middle C is where the grand staff
  splits, and it is the answer because it is the answer for a piano, not
  because it fitted this material best. `scripts/pianify-pitch.mts` prints
  where each file's music sits; `scripts/pianify-split.mts` sweeps the rules.
- **The fold of a re-trigger must not chain.** The first version made a folded
  note last until the *last* strike ended, and measured the gap against the
  *merged* note, so a key struck every forty milliseconds for ten seconds
  became one note of ten seconds. The output had notes of seventeen seconds in
  it and the raw model has none over six. The fold is pairwise now — each
  strike is judged against the strike before it — and a folded note lasts as
  long as the longest strike it swallowed, so it can never be longer than the
  model said some single note was. `scripts/pianify-merge-sweep.mts` sweeps
  the gap and the two rules; `scripts/pianify-long-notes.mts` shows what is
  inside the long ones.
- **Within a chord, the melody is placed first.** The order the notes of a
  chord are offered to a hand decides which one is dropped when the hand is
  full. Ascending by pitch — the order the model produces them in — hands the
  right hand the bottom of the treble and drops the melody on top of it. The
  right hand is given the top of the chord first and the left hand the
  bottom, because those are the notes the music is about. Measured over the
  user's 19 tracks, this raises the share of chords whose melody survives
  from 49.7 % to 52.6 % at the same note count.
- **The arrangement changes the notes, so it is not a view over them.**
  Unlike the range and the grid, the pianify stage changes what the tracks
  are: two hands instead of one. The app therefore keeps the notes it
  started from and puts them back when the switch is turned off, which
  means edits made while the arrangement is on are lost when it is turned
  off. Do not "fix" that by making it a view; the merge-back works on note
  ids, and the arrangement makes new notes.
- **The arrangement does not touch the sustain.** Up to nineteen notes can
  still sound at once, because a piano with the pedal down can do that. A
  file without pedal cannot, and trimming the durations to fit would be a
  bigger change than this stage is.
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
  running nginx and nothing else. The webroot is `/var/www/voidtrackr` and
  the vhost is `/etc/nginx/sites-available/voidtrackr.conf`. There is no
  service behind it, no process and no runtime: it serves files.
- **The box is small, and that is fine.** Two cores and a gigabyte of memory
  at the time of writing; the container is being cut to **1 core and 256 MB**,
  which is the user's manual job on the Proxmox side and is not done by this
  project. It does not need to be bigger, because the box only holds files:
  the model's weights (367 MB) and the site. Measured after the API was
  removed, the whole stack is **53 MB resident** — journald 27 MB,
  networkd 26 MB, systemd 15 MB, nginx under 10 MB — and streaming the 96 MB
  piano weight file moves that by about 2 MB, because the file goes through
  the reclaimable page cache and not through nginx's heap. The model runs in
  the browser, so the machine that opens the page needs the memory, not the
  box.
- **Deploy:** `./deploy.sh 192.168.100.109` from this folder. It builds,
  rsyncs, takes the old API off the box if it is still there, installs the
  vhost and headers, disables the template's own base site, validates with
  `nginx -t`, reloads, and smoke tests the pages and the weights.
- **Host registration:** `ops/vm.sh` in the umbrella repo, under `voidtrackr`,
  with the shared deploy key. `./ops/vm.sh voidtrackr 'hostname'`.
- **SSH:** the container's host keys were regenerated on provisioning — the
  clone had inherited the template's identity. The old ones are in
  `/root/.trash-removed/` on the box.
- **Never touch Proxmox.** Changing the container's memory, its CPU or its
  network is the user's manual job. Ask; do not do it.
- **Current pipeline version:** 0.8.0 (`src/core/version.ts`) — 0.8.0 turned the
  model threads on, at four. The count was one, and the reason it gave was that
  the runtime reduces in an order that depends on the thread count — true of
  the raw output of a graph, false of the exported file, which is identical at
  1, 2, 4 and 8 threads for every model. A conversion takes about half the
  time again: 44 seconds to 30 in the browser, and the notes do not change.
  The browser needs
  `--unsafely-treat-insecure-origin-as-secure=http://192.168.100.109`, because
  plain HTTP at a LAN address is not a secure context and the isolation
  headers alone do nothing there; the app shows the count the browser is
  actually allowing, in the header.
- **The pianify stage** was added in 0.6.0 and corrected in 0.7.0. The stage
  as it shipped put the
  notes in the wrong hands: it worked the split point out from the material,
  which on a mix lands in the treble, so the left hand played the melody up to
  MIDI 87 and the right hand played the bass down to 46. The split is at
  middle C now, the fold of a re-trigger no longer chains (it made notes of
  seventeen seconds out of strikes of ten milliseconds), notes shorter than
  sixty milliseconds are lengthened, and the notes of a chord are placed
  melody first so the top of a chord survives. The stage runs after the
  frequency range and the grid, and before the grid. A conversion that does not
  ask for it produces the same notes as 0.5.0. The default model is still
  ByteDance's piano model, which was chosen after all five were measured
  over the user's own 19 tracks.
- **Models in the catalogue:** Piano Transcription (CC BY 4.0, the default),
  Basic Pitch (Apache-2.0), Tsumugi (MIT), YourMT3+ (Apache-2.0),
  Okarina (CC BY-NC-SA 4.0, non-commercial).
- **Default model:** Piano Transcription, `DEFAULT_MODEL` in
  `src/core/registry.ts`. It won the measurement over the user's own 19
  tracks: 63.9 % of Okarina's notes, against 29.6 % for YourMT3+ and 23.4 %
  for Tsumugi. `npm run model` fetches whichever model this names.
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
