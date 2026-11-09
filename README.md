# voidtrackr

**A recording in, a MIDI file out. Nothing is uploaded, and the audio never
leaves your machine.**

voidtrackr turns a finished recording into editable multi-track MIDI. It runs
entirely locally — in the browser, or on the command line for batch work.
There is no account, no upload endpoint, no server and no telemetry: the audio
is decoded, transcribed and written back out on the machine that opened it.

```bash
npm install
npm run model     # fetch and verify the Basic Pitch weights (~900 KB)
npm run dev       # http://127.0.0.1:5173
```

---

## What it does

- **Transcribes** a recording into notes, split into melody, bass and chords.
- **Adds a drum track** from the per-band spectral flux in the mix, when there
  are drums to find.
- **Estimates tempo and key**, both editable afterwards.
- **Gives you a piano roll** to move, resize, add and delete notes in.
- **Exports multi-track MIDI** — SMF format 1, one track per part.

A single-shot transcription is missing most of that, so it is added around the
model rather than inside it: a separate bass line, a melody line traced through
sustained notes, chord and drum tracks, tempo and key detection, and an editor
for whatever the model got wrong.

## The models

Two, behind one interface, so which one runs is a choice rather than a fact
about the program.

| | Basic Pitch | Okarina |
|---|---|---|
| By | Spotify | an unofficial ONNX port of Harmonica |
| Licence | Apache-2.0 | **CC BY-NC-SA 4.0 — non-commercial only** |
| Weights | ~900 KB | ~21 MB |
| Runs on | TensorFlow.js | ONNX Runtime (WebAssembly in the browser) |
| F1 vs published Harmonica output | 30.4 % | **85.3 %** |

Both are pinned by SHA-256 and checked before they run.

**Okarina is the better model on dense material** — it finds substantially
more notes and separates them into vocal, bass and other. Its licence is
non-commercial, so it is opt-in rather than the default. The default is Basic
Pitch, which is permissively licensed and works straight after `npm run model`.
To switch it, change `DEFAULT_MODEL` in `src/core/registry.ts`.

There is a survey of the alternatives, and of what was measured, in
[`docs/model-research.md`](docs/model-research.md).

## The guarantee

**The same recording produces the same file every time**, as long as the
pipeline and the model do not change. That is a property of the design rather
than something to hope for:

- **The model is pinned.** `src/core/model.lock.json` and
  `src/core/okarina.lock.json` record the SHA-256 of every weight file. The
  weights are hashed and compared before anything runs; a mismatch stops the
  conversion instead of quietly producing something different.
- **Every file carries its provenance.** The pipeline version and the model
  fingerprint are written into the exported MIDI as a text meta event:

  ```
  voidtrackr 0.3.0 | model basic-pitch-icassp2022 | sha256:1ed1aaee…
  ```

- **The output does not depend on the machine.** The sample rate is read out of
  the file's own header, so the same recording gives the same notes on any
  computer.
- **The browser and the command line agree.** The browser and the CLI produce
  byte-identical MIDI for the same input, on both models.
- **A test guards it.** `npm test` converts the same recording twice and
  compares the bytes; `npm run check:reproducibility` also compares the native
  and pure-JavaScript TensorFlow backends.

To compare any two files:

```bash
node scripts/compare-midi.mjs out/song.mid other/song.mid
```

## How it works

```
audio file
   │
   ├─ decode ────────────► mono 22.05 kHz float samples
   ├─ tune ──────────────► measured against A440, resampled if it is off
   │
   ├─ model ─────────────► Basic Pitch or Okarina ─► notes in three parts
   │
   ├─ drums ─────────────► kick, snare and hi-hat from spectral flux
   ├─ tempo & key ───────► comb-filter tempo, Krumhansl-Schmuckler key
   │
   └─ MIDI ──────────────► SMF format 1, one track per part
```

The model sits behind a seam that returns **notes grouped into parts**, not a
matrix of activations. That matters: everything after it — the drum detector,
the tempo estimate, the key estimate, the MIDI writer — is shared between
models, and a model that reasons about onsets and offsets can be dropped in
without touching the rest.

`src/core/` holds the pipeline and is free of both DOM and Node specifics, so
the same code runs in the browser and on the command line. The only difference
is how audio is decoded: the browser uses the Web Audio API, Node uses a
pure-JavaScript decoder, so the CLI needs no `ffmpeg`.

## Installing

Requires Node.js 20 or newer.

```bash
npm install --no-audit --no-fund
npm run model              # Basic Pitch weights, verified against the lock file
npm run model:okarina      # Okarina weights, verified against the lock file (~21 MB)
```

Both fetch from the upstream source. Neither is redistributed here, and the
licence of each travels with it: Basic Pitch is Apache-2.0, Okarina is
CC BY-NC-SA 4.0 and is for non-commercial use only.

## Using it

### In the browser

`npm run dev`, then open <http://127.0.0.1:5173>.

Drop a WAV, MP3, M4A, OGG or FLAC onto the page. It transcribes, and you get a
piano roll. Click to select, drag to move, drag an edge to resize, and use
**Delete** to remove the selection. **Space** plays and pauses, and
<kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Z</kbd> undoes. The model picker is in the
header, so you can change model before dropping a file in.

### On the command line

```bash
npm run cli -- song.mp3 -o out/
npm run cli -- song.mp3 -o out/ --model okarina
npm run cli -- --list-models
```

Full reference: `npm run cli -- --help`.

## Commands

| | |
|---|---|
| `npm run dev` | the app on `127.0.0.1:5173` |
| `npm run build` | typecheck and build for production into `dist/` |
| `npm run preview` | serve `dist/` on `127.0.0.1:4173` |
| `npm run model` | fetch and verify the Basic Pitch weights |
| `npm run model:okarina` | fetch and verify the Okarina weights |
| `npm test` | the test suite |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run check:reproducibility` | compare repeat runs and both backends |
| `npm run cli -- <input> -o <dir>` | convert on the command line |
| `npm run evaluate -- <folder> --out eval` | transcribe a folder and write an HTML report |

A production build is about 52 MB, because the ONNX runtime's WebAssembly
(15 MB) and the Okarina weights (21 MB) are copied into it so the browser can
use that model. Without them it is about 3 MB, and only Basic Pitch is
available.

## Deploying

The app is a static site, so it can be served by any web server. See
[`ops/nginx/`](ops/nginx) for a worked nginx vhost and
[`deploy.sh`](deploy.sh) for a script that builds, uploads and reloads it.

Two things will bite you:

- **A strict Content-Security-Policy blocks WebAssembly.** The Okarina model
  runs through WebAssembly, and the browser treats `WebAssembly.compile` as
  script evaluation. The policy needs `'wasm-unsafe-eval'` in `script-src`.
- **`crypto.subtle` needs a secure context.** It only exists over HTTPS or on
  localhost. Served over plain HTTP on a LAN it is not there, and the model
  check falls back to a built-in SHA-256 — the same digest, so the output does
  not change.

## Working on it

```bash
npm run typecheck
npm test
npm run check:reproducibility
```

`AGENTS.md` is the handoff document: the architecture, the file map, the traps
that have bitten before, and how the pieces fit. Read it before changing
anything.

Two traps worth repeating here:

- **A model frame is not a moment in the recording.** The model's windows
  overlap, so frames inside a window sit closer together in time than the
  audio they cover. `modelFrameToTime` in `src/core/constants.ts` corrects for
  that, and *every* place that turns a frame index into a second must use it.
  There is a regression test for it; do not add a frame-to-seconds conversion
  without one.
- **`@tonejs/midi` writes the key signature wrong** — C major comes out as
  fourteen sharps, and minor keys are written as major. `src/core/midi.ts`
  rewrites those two bytes after the file is written. Do not remove that.

### Adding a model

A model has to be able to turn a signal into notes; everything else follows
from that.

1. Implement `TranscriptionModel` in `src/core/model.ts`: an identifier, a
   display name, a licence, a fingerprint, the sample rate the model wants,
   and a `transcribe` method that returns notes grouped into parts.
2. Add it to the catalogue in `src/core/registry.ts`, so it can be chosen in
   the app and on the command line.
3. Pin its weights by hash and check them before it runs.
4. Run `npm test` and `npm run check:reproducibility`.

## Licence

The code is MIT licensed — see [`LICENSE`](LICENSE).

The model weights are **not** part of this repository and are not
redistributed with it. They are fetched separately and carry their own terms:

- **Basic Pitch** — © Spotify AB, [Apache-2.0](https://github.com/spotify/basic-pitch-ts/blob/main/LICENSE)
- **Okarina** — © the Harmonica authors, [CC BY-NC-SA 4.0](https://huggingface.co/mazun/rekord). **Non-commercial use only.**

Other dependencies: [TensorFlow.js](https://github.com/tensorflow/tfjs-js)
(Apache-2.0), [@tonejs/midi](https://github.com/Tonejs/Midi) (MIT),
[onnxruntime-web](https://github.com/microsoft/onnxruntime) (MIT).

The feature set was modelled on
[convert.town/audio-to-midi](https://convert.town/audio-to-midi). This is an
independent implementation; no code was taken from that site.
