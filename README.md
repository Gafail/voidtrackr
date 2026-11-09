# voidtrackr

Turn a recording into editable multi-track MIDI — in the browser, or on the
command line for batch work. Everything runs on your own machine.

**A recording in, a MIDI file out. Nothing is uploaded. Nothing is ever sent
anywhere.**

> **Working on this project?** [`AGENTS.md`](AGENTS.md) is the handoff
> document: what the project is, how it is put together, how to run it, and the
> gotchas that have bitten before. Read that first.

It is built around Spotify's **Basic Pitch** model (Apache-2.0) and adds the
parts a single-shot transcription is missing: a separate bass line, a melody
line traced through sustained notes, chord and drum tracks, tempo and key
detection, and a piano-roll editor to fix up whatever the model got wrong.

A second model, **Okarina** (CC BY-NC-SA, non-commercial), is available as an
alternative. It finds substantially more notes and separates them into vocal,
bass and other. See [`docs/model-research.md`](docs/model-research.md) for the
measurements.

```bash
npm install
npm run model     # download the Basic Pitch weights (~900 KB)
npm run dev       # http://127.0.0.1:5173
```

## What you get

| | |
|---|---|
| **Browser app** | Drag in a WAV/MP3/M4A/OGG/FLAC, watch it transcribe, edit the notes on a piano roll, download a multi-track `.mid`. Nothing is uploaded. |
| **CLI** | `npm run cli -- song.mp3 -o out/` — same pipeline, no browser, for scripts and batch jobs. |
| **Two models** | Basic Pitch by default, Okarina on request. Pick one in the app's header or with `--model`. |
| **Tests** | `npm test` checks the pipeline against a synthetic recording with a known melody, bass line, tempo and key, and checks that two runs produce the same bytes. |
| **Reproducibility check** | `npm run check:reproducibility` compares repeat runs and the native and pure-JavaScript TensorFlow backends. |
| **Folder report** | `npm run evaluate -- "Music Import"` transcribes a whole folder with every model and writes an HTML report with piano rolls. |

## Reproducible output

The same recording produces the same file every time, as long as the pipeline
and the model do not change. That is a property of the design rather than
something to hope for:

- **The model is pinned.** `src/core/model.lock.json` records the SHA-256 of
  every model file, and the shape the pipeline expects a model to have. The
  weights are hashed and compared before anything runs; a mismatch stops the
  conversion instead of quietly producing something different.
- **Every file carries its provenance.** The pipeline version and the model's
  fingerprint are written into the exported MIDI as a text meta event, so a
  file can always be traced back to the code and weights that made it:

  ```
  voidtrackr 0.3.0 | model basic-pitch-icassp2022 | sha256:1ed1aaee…
  ```

- **The output does not depend on the machine.** The browser used to decode
  at the sound card's preferred rate, so the same file gave different notes on
  different computers. It now reads the rate out of the file's own header and
  decodes at that rate.
- **Floating point is stable.** The native TensorFlow backend and the pure
  JavaScript one give bit-identical results here, so the command line and the
  browser agree byte for byte.
- **A test guards it.** `npm test` converts the same recording twice and
  compares the bytes.

To check any two files against each other:

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

Each stage is a separate module in `src/core/`, so any of them can be
replaced, measured, or improved on its own.

### The model

Both models turn a signal into notes grouped into parts. That is the seam:
everything after it — the drum detector, the tempo estimate, the key estimate
and the MIDI writer — is the same whichever model was used.

**Basic Pitch** is a small convolutional network that takes a two-second
window of mono 22.05 kHz audio and reports, for every frame, how likely each
of the 88 piano pitches is sounding, how likely each was just struck, and
where the pitch sits between two semitones. The pipeline runs it twice: once
over a high-passed copy for the melody, once over the whole mix for the bass.

The weights come from
[`spotify/basic-pitch-ts`](https://github.com/spotify/basic-pitch-ts). They are
pinned by hash in `src/core/model.lock.json`, checked when they are downloaded
and again when they are loaded, and they are the same files every Basic Pitch
deployment uses.

**Okarina** is an unofficial ONNX reimplementation of
[Harmonica](https://arxiv.org/abs/2609.04640), the model behind BandLab's
audio-to-MIDI service. It works in two steps: a parameter-free CQT front end
turns 16 kHz audio into a spectrogram, and a note head turns that into onset,
sustain and offset values per frame, per pitch, for three parts (vocal, bass,
other). The offset head is what makes it better than Basic Pitch at placing
note ends — deciding where a note stops from the sustain value alone is
coarse, because the sustain decays over several frames.

Its licence is **CC BY-NC-SA 4.0, so it is for non-commercial use only**. The
weights are pinned in `src/core/okarina.lock.json` and are fetched separately:

```bash
npm run model:okarina     # ~21 MB
```

**On your own music, Okarina is the better model.** Over the 19 tracks in
`Music Import`, Basic Pitch produces about 42 % more notes, and only 40.6 %
of them are confirmed by Okarina, against 57.6 % of Okarina's notes being
confirmed by Basic Pitch. On a track with published reference outputs, Okarina
scores F1 85.3 % against the published Harmonica output where the Basic Pitch
pipeline scores 30.4 %.

The default is still Basic Pitch, because it is permissively licensed and
works straight after `npm run model`. To make Okarina the default, change
`DEFAULT_MODEL` in `src/core/registry.ts`.

### Upgrading the model

A model has to be able to turn a signal into notes; everything else follows
from that. To add one:

1. Implement `TranscriptionModel` in `src/core/model.ts`: an identifier, a
   display name, a licence, a fingerprint, the sample rate the model wants,
   and a `transcribe` method that returns notes grouped into parts.
2. Add it to the catalogue in `src/core/registry.ts`, so it can be chosen in
   the app and on the command line.
3. Pin its weights by hash and check them before it runs.
4. Run `npm test` and `npm run check:reproducibility`.

The pipeline version lives in `src/core/version.ts`. Bump it whenever the
output changes, so the stamp inside exported files keeps telling the truth.

**There is a survey of the alternatives in
[`docs/model-research.md`](docs/model-research.md).**

### Why two passes

The model struggles when a bass note and a melody note compete. So the
melody is transcribed from a copy with everything below 200 Hz removed, and
the bass from the whole mix. The register filter is applied both before and
after the cleanup, because the cleanup can add notes of its own.

### The melody trace

Counting note attacks finds the melody badly: a held vocal note chopped up by
vibrato looks like several notes. Instead the melody is traced through the
frame matrix with a Viterbi decoder whose states are "this pitch is sounding"
or "this pitch is resting".

The detail that makes it work is that **the model never returns zero**. Even
in silence every pitch bin sits at a small baseline value, so "no pitch here"
and "this very quiet pitch here" look almost the same, and the decoder can
never say "rest". Measuring that baseline and subtracting it first makes the
absence of a pitch genuinely unlikely.

### The drum track

There is no drum model. Kicks, snares and hi-hats are found with per-band
spectral flux: each drum lives in a distinct part of the spectrum, and a
strike shows up as a sharp rise in that band.

The hard part is deciding whether a clip has drums at all, because the flux
detector compares a band against itself and happily finds "hits" in a solo
piano piece. So each ten-second window is scored on several independent
signals — a steady pulse, hi-hats that are broadband noise rather than bright
pitched notes, and hi-hats that decay — and a drum track is only added when
they agree.

## Layout

```
src/core/     the pipeline, free of DOM and Node specifics
src/app/      the React app: editor, piano roll, playback
src/cli/      the command line converter
scripts/      model download, test signal generation, reproducibility checks
docs/         the model survey in docs/model-research.md
test/         end-to-end checks
```

`src/core` runs unchanged in the browser and in Node. The only difference is
how audio is decoded: the browser uses the Web Audio API, Node uses a
pure-JavaScript decoder (`src/core/audio-node.ts`) so the CLI needs no
`ffmpeg`.

## Commands

```bash
npm run dev                      # the app on 127.0.0.1:5173
npm run build                    # typecheck and build for production
npm run model                    # fetch the pinned weights and check them
npm run model:okarina            # fetch the Okarina weights (non-commercial)
npm test                         # the test suite
npm run cli -- song.mp3 -o out/  # convert on the command line
npm run cli -- song.mp3 -o out/ --model okarina
npm run cli -- --list-models     # what models are available
npm run check:reproducibility    # compare repeat runs and both backends
npm run evaluate -- "Music Import" --out eval   # transcribe a folder, write a report
node scripts/compare-midi.mjs a.mid b.mid   # compare two MIDI files
node scripts/evaluate-midi.mjs ref.mid est.mid   # score one against another
```

A production build is about 52 MB, because the ONNX runtime's WebAssembly
(15 MB) and the Okarina weights (21 MB) are copied into it so the browser can
use that model. Without them it is about 3 MB.

## Working on it

The pipeline is tuned against one synthetic recording, which is enough to
catch regressions but nowhere near enough to call the output good. These are
the places where the most is left to do, roughly in order of how much they
matter:

1. **Evaluation against a ground truth.** `npm run evaluate` compares the
   models over a folder of recordings, but there is nothing to score them
   against — only each other. Hand-played MIDI, or the stems the songs were
   generated from, would turn the comparison into a score.

2. **The bass line.** Bass is where the model is weakest, and the current
   approach just filters the model's output by register. A dedicated tracker
   — level envelope for attacks, YIN for pitch — would handle repeated notes
   and low fundamentals much better. `src/core/dsp.ts` already has the FFT and
   autocorrelation pieces needed for it.

3. **Kick and bass competing.** A kick sits in the same register as a bass
   note, and the two get confused. The drum detector knows where the kicks
   are; the transcription could be told to ignore those moments.

4. **A third model.** Two are implemented. The next candidate is TransKun
   V2: MIT licensed, ONNX weights available, and its decoder already exists
   in JavaScript, so it is mostly plumbing. The catch is that it was
   reported to under-report on dense mixes.

5. **The drum presence test.** The midpoints in the scoring function were
   chosen by hand. A set of real recordings with and without drums would let
   them be measured instead.

6. **Tuning drift.** Out-of-tune recordings are corrected by a constant
   offset. A recording that drifts while it plays is not corrected.

7. **Beat-grid quantisation.** The model reports onsets on a frame grid, so
   an onset can only land to within ±5.8 ms and usually lands on the frame
   boundary. The beat tracker in `src/core/beat.ts` finds the pulse of a
   recording from the audio itself, and snapping the onsets to it would
   remove the quantisation error. It has not been done, because the grid is
   only accurate to about 12 ms — worse than the model — and it would need a
   tracker that follows the phase rather than laying down a constant grid.

8. **Chords.** Chords are currently "everything that is not the melody and
   not the bass". Grouping the notes into actual chords, and dropping notes
   that are not part of one, would clean the chord track up.

## Known issues

- Repeated notes of the same pitch in the bass register can be reported late,
  because the model sees one continuous tone rather than two attacks.
- A dense mix with overlapping instruments is genuinely hard; the output gets
  rough rather than wrong.
- The drum detector prefers a phantom drum track under a piano piece over a
  missing one, so quiet drum parts are often missed.

## Notes for anyone changing this

- **A model frame is not a moment in the recording.** The model's windows
  overlap, so the frames inside a window sit closer together in time than the
  audio they cover. `modelFrameToTime` in `src/core/constants.ts` corrects
  for that, and *every* place that turns a frame index into a second has to
  use it. Until 0.2.0 two of them did not — `toTimedNotes` in
  `src/core/postprocess.ts` and the tracer in `src/core/melody.ts` — and every
  note drifted about 5 ms later for every second of audio, so a three-minute
  song ended up roughly half a second out of step with its own beginning.
  There is a test for this now (`keeps the timing steady over a long
  recording`); do not add a frame-to-seconds conversion without it.
- **`@tonejs/midi` writes the key signature wrong.** It writes the position of
  the key name in its table as if it were the signed key number, so C major
  comes out as fourteen sharps, and it writes the mode as a string, so minor
  keys are written as major. `src/core/midi.ts` rewrites those two bytes after
  the file is written; do not remove that.
- **`@tensorflow/tfjs-node` 4.21 and 4.22 ship a broken build.** Its published
  code requires Node's own `util` module where it meant to require TensorFlow.js's
  utilities, and then calls a function that does not exist there.
  `src/cli/convert.ts` adds the missing function before the backend loads.
- The native backend registers itself as `tensorflow`, not `node`.
- **ONNX Runtime reduces in an order that depends on the thread count.**
  `src/core/onnx-node.ts` and `src/core/onnx-web.ts` pin `intraOpNumThreads`,
  or two runs of the same file could differ in the last bits.

## Credits and licensing

- **Basic Pitch** — model and reference implementation, © Spotify AB,
  [Apache-2.0](https://github.com/spotify/basic-pitch-ts/blob/main/LICENSE).
- **TensorFlow.js** — Apache-2.0.
- **@tonejs/midi** — MIT, used for reading and writing MIDI files.
- **Okarina** — the optional comparison model, an unofficial ONNX
  reimplementation of Harmonica (© the Harmonica authors),
  [CC BY-NC-SA 4.0](https://huggingface.co/mazun/rekord). **Non-commercial.**
  It is fetched separately and is not part of the shipped app.
- The feature set was modelled on
  [convert.town/audio-to-midi](https://convert.town/audio-to-midi). The code
  here is an independent implementation; no code was taken from that site.

The code in this repository is MIT licensed.

## Licence and identity

A standalone Void* project. TheVoidCo keeps a registry row that points at this
folder; it holds no copy of the source. Local-first, no accounts, no tracking,
no telemetry.
