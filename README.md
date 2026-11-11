# voidtrackr

**A recording in, a MIDI file out. Nothing is uploaded, and the audio never
leaves your machine.**

voidtrackr turns a finished recording into editable multi-track MIDI. It runs
entirely locally — in the browser, or on the command line for batch work.
There is no account, no upload endpoint, no server and no telemetry: the audio
is decoded, transcribed and written back out on the machine that opened it.

```bash
npm install
npm run model     # fetch and verify the default model's weights (~100 MB)
npm run dev       # http://127.0.0.1:5173
```

---

## What it does

- **Transcribes** a recording into notes, split into melody, bass and chords.
- **Adds a drum track** from the per-band spectral flux in the mix, when there
  are drums to find.
- **Estimates tempo and key**, both editable afterwards.
- **Gives you a piano roll** to move, resize, add and delete notes in.
- **Lets you keep only the part of the spectrum you want** — a range in hertz,
  from nothing at all to everything the keyboard holds.
- **Snaps the notes to a grid** — so they land on the frames of a screen that
  renders at a fixed rate.
- **Arranges the notes for two hands** — the re-triggers folded together, the
  chords divided between a left and a right hand, and nothing a hand could
  not hold.
- **Exports multi-track MIDI** — SMF format 1, one track per part.

A single-shot transcription is missing most of that, so it is added around the
model rather than inside it: a separate bass line, a melody line traced through
sustained notes, chord and drum tracks, tempo and key detection, and an editor
for whatever the model got wrong.

## The models

Five, behind one interface, so which one runs is a choice rather than a fact
about the program. **The licence is part of the model**, and the app states it
next to the picker — it decides whether the output can be sold.

| | Basic Pitch | Tsumugi | YourMT3+ | Piano Transcription | Okarina |
|---|---|---|---|---|---|
| By | Spotify | anime-song | Mimbres et al. | ByteDance | an unofficial ONNX port of Harmonica |
| Licence | Apache-2.0 | **MIT** | **Apache-2.0** | **CC BY 4.0** | **CC BY-NC-SA 4.0 — non-commercial only** |
| Commercial use | yes | yes | yes | yes | **no** |
| Weights | ~900 KB | ~74 MB | ~181 MB | ~100 MB | ~21 MB |
| Runs on | TensorFlow.js | ONNX Runtime (WebAssembly in the browser) | | | |
| What it is good at | being small and quick | synthesised and anime-style music; instrument-agnostic | many instruments at once, with drums | the exact onset and offset of a piano note | dense mixes; vocal, bass and other |

All of them are pinned by SHA-256 and checked before they run.

**The default is the Piano Transcription model**, because it was measured over
your own 19 tracks and won. It shares **63.9 %** of Okarina's notes — more than
double any other permissive model — and its onsets land on the beat within 0.1
points of Okarina's. It finds fewer notes than any model except Okarina itself,
which is what precision looks like. Its licence is **CC BY 4.0**, so what it
produces can be sold.

To switch the default, change `DEFAULT_MODEL` in `src/core/registry.ts`.

**YourMT3+** is second (29.6 % against Okarina, and the best eighth-note
accuracy of the five). It is the strongest published model on multi-instrument
material (F1 0.817 on Slakh, where Basic Pitch scores 0.465), but it is the
slowest and wants a couple of gigabytes of memory.

**Tsumugi** has the cleanest licence of the three — MIT weights, not just MIT
code — and it was built for synthesised music, but on your material it was the
weakest of the three: it found as many notes as Basic Pitch with only 23.4 %
of them confirmed by Okarina. It stays in the app as a third opinion.

**Basic Pitch** is the smallest and quickest of the five, and it is the one
with no legal question attached to it at all — which is why it was the default
until the measurement came in. Pick it with `--model basic-pitch`.

**Okarina is not for selling.** CC BY-NC-SA 4.0 means non-commercial use only.
It stays in the app because on dense material it is still the best of the
five, but anything it produces should not be sold.

There is a survey of the alternatives, and of what was measured, in
[`docs/model-research.md`](docs/model-research.md). It includes the licence
table for every model considered, and the two that are one email away from
being usable.

## The guarantee

**The same recording produces the same file every time**, as long as the
pipeline and the model do not change. That is a property of the design rather
than something to hope for:

- **The model is pinned.** Every model has a lock file in `src/core/` —
  `model.lock.json`, `piano.lock.json`, `tsumugi.lock.json`, `yourmt3.lock.json`
  and `okarina.lock.json` — recording the SHA-256 of every weight file. The
  weights are hashed and compared before anything runs; a mismatch stops the
  conversion instead of quietly producing something different.
- **Every file carries its provenance.** The pipeline version and the model
  fingerprint are written into the exported MIDI as a text meta event:

  ```
  voidtrackr 0.4.0 | model piano-transcription-kong-2020 | sha256:27c96172…
  ```

- **The output does not depend on the machine.** The sample rate is read out of
  the file's own header, so the same recording gives the same notes on any
  computer.
- **The browser and the command line agree.** The browser and the CLI produce
  byte-identical MIDI for the same input, on all five models.
- **A test guards it.** `npm test` converts the same recording twice and
  compares the bytes; `npm run check:reproducibility` also compares the native
  and pure-JavaScript TensorFlow backends, and checks the default model twice
  over.

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
   ├─ model ─────────────► Basic Pitch, Tsumugi, YourMT3+, Piano or Okarina
   │                         ─► notes grouped into parts
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
npm run model              # the default model's weights, verified against the lock file
npm run model:piano        # Piano Transcription weights (~100 MB)
npm run model:basic-pitch  # Basic Pitch weights (~900 KB)
npm run model:tsumugi      # Tsumugi weights (~74 MB)
npm run model:yourmt3      # YourMT3+ weights (~181 MB)
npm run model:okarina      # Okarina weights (~21 MB)
```

Each fetches from the upstream source and is checked against its lock file.
None is redistributed here, and the licence of each travels with it: Basic
Pitch is Apache-2.0, Tsumugi is MIT, YourMT3+ is Apache-2.0, Piano
Transcription is CC BY 4.0, and Okarina is CC BY-NC-SA 4.0 and is for
non-commercial use only.

`npm run model` fetches whichever model is the default, so after a fresh
checkout that is all you need. The others are only fetched when you ask for
them.

## Using it

### In the browser

`npm run dev`, then open <http://127.0.0.1:5173>.

Drop a WAV, MP3, M4A, OGG or FLAC onto the page. It transcribes, and you get a
piano roll. Click to select, drag to move, drag an edge to resize, and use
**Delete** to remove the selection. **Space** plays and pauses, and
<kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Z</kbd> undoes. The model picker is in the
header, so you can change model before dropping a file in.

The **Range** boxes in the header choose which part of the spectrum to keep —
say 0–400 Hz for the bass end of a track. They act on the notes the model
found, not on the audio, so the model always hears the whole recording and the
notes you keep are exactly the notes that were found. Nothing is thrown away:
widen the range and the rest comes back.

The **Grid** boxes snap the notes to a fixed grid of slots per second. Set it
to 30 and every onset lands on a multiple of 33.3 ms, which is exactly where
a game rendering 30 frames a second will draw them — a note that starts
between two frames cannot be shown on the frame it belongs to, so the file is
better off agreeing with the frames than being more precise than the screen.
The second box shifts the grid, in milliseconds, for when the screen's frames
do not start with the recording.

The **Pianify** switch arranges the notes for two hands. A transcription has
no hands in it: the same key can be struck twice in a row, a chord can reach
three octaves, and a half second can hold forty notes. Pianify folds the
re-triggers together, divides the chords between a left and a right hand at
middle C, and drops what neither hand could hold — five notes a hand, an
octave of reach, ten notes a second. It is applied to the notes as they
stand, so it can be turned on and off without transcribing again. What it
costs and what it buys is in [`docs/pianify.md`](docs/pianify.md).

The header shows how many threads the model is using. It says **1** unless
the page is cross-origin isolated, which a browser will only allow on
`localhost` or over https. To get four threads on a LAN address, start the
browser with

```
--unsafely-treat-insecure-origin-as-secure=http://192.168.100.109
```

or serve the site over https with a certificate the machine trusts. The
command line needs neither: it uses four threads on its own.

### On the command line

```bash
npm run cli -- song.mp3 -o out/                  # the default model
npm run cli -- song.mp3 -o out/ --model basic-pitch
npm run cli -- song.mp3 -o out/ --model tsumugi
npm run cli -- song.mp3 -o out/ --model yourmt3
npm run cli -- song.mp3 -o out/ --model piano
npm run cli -- song.mp3 -o out/ --model okarina
npm run cli -- song.mp3 -o out/ --range 400      # everything below 400 Hz
npm run cli -- song.mp3 -o out/ --range 100-1200 # between 100 and 1200 Hz
npm run cli -- song.mp3 -o out/ --quantize 30    # on a 30 FPS grid
npm run cli -- song.mp3 -o out/ --pianify        # arranged for two hands
npm run cli -- --list-models
```

Full reference: `npm run cli -- --help`.

### Over HTTP

The site has an address you can call. The box runs a small service
alongside it, at the same origin, that describes what the tool can do: the
models and their licences, the parameters and their defaults, and where
each model's weights are served from.

```bash
curl http://192.168.100.109/api/models
curl http://192.168.100.109/api/parameters
```

There is a page about it at `/api/`, an OpenAPI document at
`/api/openapi.json`, and the whole thing is in
[`docs/api.md`](docs/api.md).

**It does not take a recording.** There is no endpoint that accepts one and
the server never reads a request body, so the audio cannot leave the
machine it is on — not through the API, not through anything else. A
script converts with the command line, optionally taking the model from
the site:

```bash
npm run cli -- --server http://192.168.100.109 song.wav -o .
```

That fetches the model's weights over HTTP, checks each file against its
pinned hash, and converts. The recording is read from disk and stays
there. The file it produces is byte-identical to one made with the weights
already on disk, and there is a test that runs both and compares them.

## Commands

| | |
|---|---|
| `npm run dev` | the app on `127.0.0.1:5173` |
| `npm run build` | typecheck and build for production into `dist/` |
| `npm run build:api` | bundle the API server into `dist-api/` |
| `npm run api` | run the API server on `127.0.0.1:8787` |
| `npm run preview` | serve `dist/` on `127.0.0.1:4173` |
| `npm run model` | fetch and verify the default model's weights |
| `npm run model:piano` | fetch and verify the Piano Transcription weights |
| `npm run model:basic-pitch` | fetch and verify the Basic Pitch weights |
| `npm run model:tsumugi` | fetch and verify the Tsumugi weights |
| `npm run model:yourmt3` | fetch and verify the YourMT3+ weights |
| `npm run model:okarina` | fetch and verify the Okarina weights |
| `npm test` | the test suite |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run check:reproducibility` | compare repeat runs and both backends |
| `npm run cli -- <input> -o <dir>` | convert on the command line |
| `npm run evaluate -- <folder> --out eval` | transcribe a folder and write an HTML report |
| `npm run evaluate -- --out eval --summary` | print the aggregate from the saved results, without transcribing again |

A production build is about 380 MB, because the ONNX runtime's WebAssembly
(15 MB) and every model's weights are copied into it so the browser can use
them. The weights are fetched when a model is chosen, not when the page
loads, so the page itself is still small.

## Deploying

The app is a static site, so it can be served by any web server. See
[`ops/nginx/`](ops/nginx) for a worked nginx vhost and
[`deploy.sh`](deploy.sh) for a script that builds, uploads and reloads it.

Two things will bite you:

- **A strict Content-Security-Policy blocks WebAssembly.** Four of the five
  models run through WebAssembly, and the browser treats `WebAssembly.compile` as
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
- **Tsumugi** — © the anime-song authors, [MIT](https://huggingface.co/anime-song/instrument_agnostic_amt)
- **YourMT3+** — © Chang, Benetos, Kirchhoff and Dixon, [Apache-2.0](https://huggingface.co/mimbres/YourMT3). The training mix includes MAESTRO, which is CC BY-NC-SA 4.0.
- **Piano Transcription** — © Kong, Li, Song, Wan and Wang, [CC BY 4.0](https://zenodo.org/records/4034264). Trained on MAESTRO, which is CC BY-NC-SA 4.0.
- **Okarina** — © the Harmonica authors, [CC BY-NC-SA 4.0](https://huggingface.co/mazun/rekord). **Non-commercial use only.**

The four permissive licences cover the *weights*. Whether a model trained on
MAESTRO can be used commercially is a separate question that the model cards
do not settle — the authors of YourMT3+ say so themselves. If you are
selling, get a lawyer's opinion rather than a licence table.

Other dependencies: [TensorFlow.js](https://github.com/tensorflow/tfjs-js)
(Apache-2.0), [@tonejs/midi](https://github.com/Tonejs/Midi) (MIT),
[onnxruntime-web](https://github.com/microsoft/onnxruntime) (MIT).

The feature set was modelled on
[convert.town/audio-to-midi](https://convert.town/audio-to-midi). This is an
independent implementation; no code was taken from that site.
