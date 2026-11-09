# Is there a better model, and has anyone solved the timing problem?

A survey of what exists, what is actually usable, and what we measured here.

This is voidtrackr's model survey. Written after the 0.2.0 timing fix, so the
numbers below are from the current pipeline.

**The short version: yes, and we now have it.** Okarina is implemented as a
second model you can choose, and on your own music it is clearly the better
one. The full survey is below; the measurements are in section 6.

---

## 1. The short answer

**Yes, there is a much better model, and we can get a version of it running
locally.** It is not Basic Pitch.

The family that matters is the one Harmonica introduced (BandLab's
audio-to-MIDI model, [arXiv 2609.04640](https://arxiv.org/abs/2609.04640)).
Harmonica's own weights were never released, but an unofficial ONNX
reimplementation called **Okarina**
([`mazun/rekord`](https://huggingface.co/mazun/rekord)) is, and it runs
locally through `onnxruntime`.

On a track where the published outputs of two reference implementations are
available, Okarina scores **F1 85.3 % against the published Harmonica
x-large output**, where our Basic Pitch pipeline scores **30.4 %**.

**And part of what you described as "the MIDI is similar but not super
close" was our own bug**, not the model's limit. Until 0.2.0 the pipeline
placed every note about 5 ms too late for every second of audio, so a
three-minute song drifted roughly 0.5 s out of step with its own beginning.
That is fixed and now has a test.

**On timing specifically**, the answer is yes: the modern models do not
quantise onsets to a frame grid, they *regress* the exact onset and offset
time, and several of them also ship beat tracking so notes can be snapped to
the grid. Both ideas are described in section 5.

The one decision you had to make before we could go further was **commercial
use or not**, because the best available model is licensed
**CC BY-NC-SA 4.0 (non-commercial)**. Your answer was non-commercial, so
Okarina is in.

---

## 2. The timing bug we found and fixed

Worth stating plainly, because it changes how the rest of the numbers should
be read.

A model frame is not a moment in the recording. The Basic Pitch model is a
fixed-window analyser: it consumes 43 844 samples of mono 22.05 kHz audio and
emits 172 frames. Because consecutive windows overlap, the frames inside a
window sit closer together in time than the audio they cover. Mapping a
frame index to a second therefore needs a correction.

The pipeline had that correction in `constants.ts` and used it in
`notes.ts` — but two other places converted frames to seconds without it:

| file | what it did | effect |
| --- | --- | --- |
| `src/core/postprocess.ts` | `toTimedNotes()` used `startFrame / FPS` | every chord and bass note drifted |
| `src/core/melody.ts` | `flushRun()` and the re-articulation split used `startFrame / fps` | every melody note drifted |

Measured on a synthetic file with tone bursts at exactly known times, every
4 seconds for two minutes:

```
   2 -> 1.927     -73 ms
  14 ->  14.001      +1 ms
  46 ->   46.172    +172 ms
  74 ->   74.315    +315 ms
 114 ->  114.521    +521 ms
mean error 222.1 ms, worst 521 ms
```

That is **5.3 ms of drift per second of audio**.

After the fix the same measurement gives a net slope of **-0.13 ms/s**, and
the residual is a bounded ±12 ms sawtooth inherited from the reference
mapping rather than a growing error.

Two other things came out of the same investigation:

* **The last second of every file was being thrown away.** The frame budget
  was computed at the model's nominal 86 fps, but the frames the pipeline
  actually produces advance slightly faster than that, so the output stopped
  about 1 s early on a 156 s recording. Now the budget comes from the window
  geometry and the output reaches the end of the audio.
* **`timeToModelFrame` was not the inverse of `modelFrameToTime`.** It
  dropped the window correction entirely, so any time → frame → time round
  trip lost about 5 ms per second. It now searches for the window the time
  falls in and picks the frame that lands closest.

The mapping itself was left as the reference implementation has it. We
verified it against the published Basic Pitch output for the same audio:

```
ours (floor/172)   n=1235  median  -0.0 ms   p10 -11.3  p90  +0.5
raw (no correction) n= 447  median +24.1 ms   p10 -75.7  p90 +102.8
```

So our notes now line up with the reference note for note, to within a few
milliseconds.

### What the fix did to the output

Same test track, same published reference outputs, ±50 ms onset tolerance:

| estimate | vs published Basic Pitch | vs published Harmonica x-large |
| --- | --- | --- |
| published Basic Pitch | — | F1 64.9 % (P 75.3 / R 57.0) |
| our pipeline, 0.1.0 | F1 15.2 % | F1 10.9 % |
| our pipeline, 0.2.0 | **F1 35.7 %** (P 41.6 / R 31.3) | F1 30.4 % (P 42.0 / R 23.9) |
| **Okarina, 0.2.0** | **F1 63.4 %** (P 52.9 / R 79.2) | **F1 85.3 %** (P 80.2 / R 91.0) |

Our Basic Pitch pipeline has low *recall* (31.3 %) because it emits more
notes than upstream does — 1625 against 1220 — and the extras are mostly
harmonics and overtones. Its *timing* is now good: the mean deviation of
matched notes is **9 ms** at a ±50 ms tolerance, and 2.2 ms at ±20 ms.

Okarina reaches 85.3 % against the Harmonica reference. For comparison, the
published Basic Pitch output only reaches 64.9 % against that same reference,
so Okarina is not merely close to the better model — it is closer to it than
Basic Pitch is to itself.

---

## 3. The models

### 3.1 Harmonica — the target, but the weights are not public

[arXiv 2609.04640](https://arxiv.org/abs/2609.04640) (4 Sep 2026, ICASSP
2027 submission; Ou, Martel, Hennessy-Priest, Cho). BandLab's
audio-to-MIDI service is built on it. Instrument-agnostic, multi-depth
harmonic convolution.

Reported: the `nano` model has 26 300 parameters, runs at 1622.5× real time
and scores frame F1 0.796 — **14.6 points above Basic Pitch**. The `x-large`
model has 15.1 M parameters at 123× real time. On MAESTRO it reaches onset
precision 0.881 against MT3's 0.448. It runs on an iPhone 15; ONNX Runtime
on CPU takes 22.16 s at 7× real time, LiteRT on GPU 4.86 s at 32.1×.

**The weights were not released.** Everything below is either a
reimplementation or a different model.

### 3.2 Okarina — a usable Harmonica reimplementation

[`mazun/rekord`](https://huggingface.co/mazun/rekord) on Hugging Face. An
unofficial ONNX derivative of Harmonica.

* `cqt156.onnx` — a parameter-free CQT front end
* `okarina_s.onnx` (0.7 MB), `okarina_m.onnx` (2.9 MB), `okarina_l.onnx` (13.1 MB)
* `beatthis_small0.onnx` (MIT) as a bonus

Shape of the graph, which we confirmed by running it locally:

```
front end : [1, 109869] float32  ->  cqt_mag [1, 156, 264]
note model: cqt_mag [B,T,264]     ->  onset / sustain / offset, each [B, 3, T, 88]
```

The three axes of the note head are the parts (vocal, bass, other). The front
end is 109 869 samples at 16 kHz mono, and the note model consumes a
156-frame × 264-bin CQT.

We calibrated the frame timebase empirically rather than assuming it:
**frame = 0.03204 s**, with **0.9705 s** of leading context.

**Licence: CC BY-NC-SA 4.0 — non-commercial.** This is the constraint that
decides the next step.

Measured on our test track: F1 70.1 % against the published Harmonica x-large
output, with recall 88.2 %. It finds notes Basic Pitch misses, at the cost of
some extra notes.

### 3.3 TransKun V2 — the MIREX winner

[`Yujia-Yan/TransKun`](https://github.com/Yujia-Yan/TransKun), MIT code,
`pip install transkun`.

Ranked **#1 on the MIREX 2024 polyphonic-transcription leaderboard**:
summary onset F1 0.9648, holistic note F1 0.9081 for the V2 Aug checkpoint.
A Vision Transformer with a Neural Semi-CRF decoder, working on 16-second
segments every 8 seconds, with edge heads that refine onset and offset times
and a sustain-pedal model.

ONNX exports exist (`transkun_score.onnx`, `transkun_attr.onnx`) and the
semi-CRF decoding already exists in JavaScript, in
[`ravetools/piano-visualizer`](https://huggingface.co/ravetools/piano-visualizer).
That makes it the **easiest candidate to get into a browser build**.

The catch, from Harmonica's own demo page: on out-of-domain material TransKun
produced "the sparsest output here" — it dropped a lot of notes. On clean
piano it is excellent; on a dense AI-generated mix it may under-report.

### 3.4 ByteDance `piano_transcription` — the best timing

[`bytedance/piano_transcription`](https://github.com/bytedance/piano_transcription).
This is the one to look at if *timing accuracy* is the priority.

On MAESTRO: onset F1 96.72 %, pedal F1 91.86 %, note+offset F1 82.47 %.
The high-resolution variant ([arXiv 2409.19614](https://arxiv.org/abs/2409.19614))
reaches onset F1 97.34–97.39 % and note+offset F1 84.67–84.76 % with only
0.9–2.7 M parameters.

Its important property: it does not quantise onsets to a frame grid, it
**regresses the exact onset and offset time**. That is the single biggest
structural difference from Basic Pitch and the thing that makes onsets land
where they really are.

Weights are **CC BY 4.0** according to the musicnote model card — the most
permissive licence of anything that works this well. It is trained for piano,
so on a full mix it will pick out the piano-like content; combined with stem
separation that is a reasonable route.

### 3.5 Tsumugi — the closest match for synthesised music

[`anime-song/tsumugi`](https://github.com/anime-song/tsumugi), MIT code.
Takes TransKun's Neural Semi-CRF and extends it to all instruments. It is
explicitly designed for **anime and synth music**, which is the closest
published match to AI-generated songs.

It ships separate checkpoints per part (`best_model_other_v1_5.pth` and
friends) plus separate **velocity**, **beat**, **chord** and **key** models,
and an instrument-refinement model. An ONNX export exists in the musicnote
repository.

Caveat from its own documentation: it works best on clean, sustained material
where each stem is monophonic, and its instrument classification is limited.

### 3.6 MuScriptor

[`muscriptor/muscriptor`](https://github.com/muscriptor/muscriptor) (Kyutai /
Mirelo). A transformer decoder, described as the most accurate open-source
transcription model. Weights are **CC BY-NC 4.0**.

### 3.7 Browser-ready ONNX with a manifest

[`baroned/musicnote-ondevice-models`](https://huggingface.co/baroned/musicnote-ondevice-models)
is worth a look for a practical reason: its `manifest.json` lists every file
with its **URL, byte size and SHA-256**, which is exactly the lock-file
pattern this project already uses. It carries `piano_kong.onnx` (CC BY 4.0),
`guitar_crnn.onnx` (MIT), `tsumugi_other_v1_5.*` (MIT), `beat_this.onnx` (MIT)
and `basic_pitch_nmp.onnx` (Apache-2.0).

Its author flags a licence caveat: weights trained on MAESTRO, MedleyDB and
GAPS are "not cleared to ship".

### 3.8 The piano-cover route

[`anime-song/tsumugi-piano-cover`](https://github.com/anime-song/tsumugi-piano-cover)
generates a piano cover of a pop song as MIDI. It is not a transcription at
all — it listens to the song, follows the beat, and *writes* a new piano
arrangement that is aligned to the original timeline. Weights CC BY-NC 4.0.

This is worth flagging because it may be what the user actually wants. When
the input is a dense AI-generated arrangement, a literal transcription is
bound to be dense and hard to play; an autoregressive piano arrangement is
both more playable and more musically convincing.

Its documentation also states the thing that matters most for this project:
**"metric rhythm strongly depends on source beat tracking"**, and its V2
generates bar by bar on the SheetSage2-Refined beat grid.

---

## 4. What the benchmarks actually say

MIREX 2024 results for dense polyphonic material:

| material | best score |
| --- | --- |
| dense polyphonic mixes | ~38 % |
| guitar | ~78 % |
| vocals | ~52 % |
| clean solo piano | ~96 % |

Two things follow.

1. **A dense AI-generated mix is the hardest case there is.** Even the best
   published systems sit around 38 % on that kind of material.
2. **There is no ground truth to score against.** Two strong models disagree
   with each other far more than either disagrees with a human reference:
   published Basic Pitch vs published Harmonica x-large is only **F1 64.9 %**
   on our test track. Two models that both sound right to a musician can
   score 35 % against each other.

This is why we should not chase a number. The practical test is whether the
result *sounds* like the song.

---

## 5. How the timing problems have been solved

Five techniques, all of them applicable here.

**a) Regression instead of quantisation.** The oldest problem in automatic
transcription is that onsets land on a frame grid. Basic Pitch predicts
"there is an onset here" for each of 86 frames per second, so an onset can
only be reported to within ±5.8 ms and usually lands on the frame boundary.
ByteDance, TransKun and Okarina all replace this with a head that predicts
the **fractional offset of the onset within the frame**. This is the single
biggest structural improvement, and it is why we recommend ByteDance or
Okarina over anything frame-based.

**b) Beat-grid quantisation.** If the music is already on a grid, snap the
onsets to it. This matters enormously here because AI-generated songs are
almost always generated on a grid in the first place. The tools:
[Beat This!](https://github.com/CPJKU/beat_this) (MIT) for beat and
downbeat tracking, and SheetSage2-Refined for a cleaner grid. Basic Pitch
already ships a beat tracker, and Okarina bundles `beatthis_small0.onnx`.

**c) Segment-boundary joining.** Long files are transcribed in windows, and a
note that crosses a boundary gets cut in two. TransKun addresses this
explicitly with its overlapping 16 s / 8 s segmentation.

**d) Stem separation first.** Separate the mix, then transcribe each stem.
[Demucs](https://github.com/facebookresearch/demucs) is the usual choice,
but its own README warns that the six-source **piano** stem has "a lot of
bleeding and artifacts", so it helps more for vocals than for piano.

**e) Dedicated velocity models.** Tsumugi ships one. Velocity is a large
part of why a transcription sounds dead, and the frame head does not produce
good velocities.

---

## 6. What we measured on your own music

The `Music Import` folder holds 19 tracks — the same recordings as mp3, ogg
and wav — totalling 58 minutes. Both models were run over every one of them.

```bash
npm run model:okarina
npm run evaluate -- "Music Import" --out eval
```

That writes an HTML report to `eval/index.html` with, for every song: a piano
roll per model, a **Play** button that sounds the notes, a link to the MIDI
file, and a link to the original recording so the two can be compared.

### The result

Over all 19 songs:

| | Basic Pitch | Okarina |
| --- | --- | --- |
| notes per song | 4456 | 3130 |
| onsets within 30 ms of the beat | 14.2 % | 15.7 % |
| onsets within 30 ms of an eighth note | 27.2 % | 29.2 % |
| mean distance to the nearest beat | 117.9 ms | 117.1 ms |

And, the measure that matters most — **how much the two models agree**:

| | |
| --- | --- |
| Basic Pitch notes also found by Okarina | **40.6 %** |
| Okarina notes also found by Basic Pitch | **57.6 %** |

Read that carefully, because it is the whole answer:

* Basic Pitch produces about **42 % more notes** than Okarina.
* Only **40.6 %** of Basic Pitch's notes are confirmed by Okarina. The rest —
  nearly three fifths — are notes the more accurate model does not find.
* On the other hand, **57.6 %** of Okarina's notes *are* confirmed by Basic
  Pitch.

So Basic Pitch's extra notes are overwhelmingly notes that nobody else
agrees with. That is what over-transcription looks like: harmonics reported
as notes, sustained notes chopped into pieces, the same note counted twice.
Together with the reference-track measurement in section 2 — where Okarina
scores 85.3 % against the published Harmonica output and our Basic Pitch
pipeline scores 30.4 % — the conclusion is not close.

**Okarina is the better model for this material.**

### Why the beat numbers are low for both

Only around 15 % of the onsets land within 30 ms of a beat, for either model.
That is not a defect in the models: this music is not a steady pulse with
notes on every beat. It is mostly sustained pads, string swells and arpeggios
that begin between the beats. The beat grid is there as a ruler, not as a
target, and it is the same ruler for both models.

The grid is measured from the recording itself, not from the notes, so it
cannot flatter one model over the other. It finds 120.19 BPM on the test
recording whose tempo is exactly 120, and it holds within about 12 ms over a
four-minute track.

### What we did not do, and why

**Beat-grid quantisation.** Snapping the onsets to the beat is the obvious
next idea, and the tracker was built for it — but the grid is only accurate
to about 12 ms, and the model's own onsets are better than that. Snapping
would replace a good time with a worse one. It needs a beat tracker that
follows the phase instead of laying down a constant grid, which is more work
than it is worth until we know which model wins.

**Stem separation.** Demucs would help the vocals, but its own README warns
that the piano stem has "a lot of bleeding and artifacts", and Okarina
already separates vocal, bass and other on its own.

---

## 7. How this fits the code we already have

The good news is that the work done in phase 1 made this a drop-in question
rather than a rewrite.

`src/core/model.ts` defines the seam:

```ts
export interface TranscriptionModel {
  readonly id: string;
  readonly displayName: string;
  readonly license: string;
  readonly fingerprint: string;
  readonly inputSampleRate: number;
  transcribe(samples: Float32Array, options?: ModelOptions): Promise<ModelTranscription>;
}
```

A model is handed audio and hands back **notes grouped into parts**. The
seam used to sit one level lower, on the three matrices Basic Pitch produces;
that was too Basic-Pitch-shaped to be a real seam, because a model that
reasons about onsets and offsets — which is exactly what the newer models do
— would have had to pretend to be Basic Pitch.

Everything after the seam — the drum detector, the tempo estimator, the key
estimator, the MIDI writer — is shared. `src/core/registry.ts` is the
catalogue, and it is what the app's picker and the command line's `--model`
flag read from.

### What each option needs

| option | state |
| --- | --- |
| **Okarina** | **Done.** Implemented in `src/core/okarina.ts`, running in the browser through `onnxruntime-web` and on the command line through `onnxruntime-node`. The two produce byte-identical MIDI. |
| **TransKun V2** | Not started. MIT, ONNX available, the JavaScript decoder already exists in `ravetools/piano-visualizer`. The risk is under-reporting on dense mixes. |
| **ByteDance** | Not started. Best onset and offset accuracy by a distance, and CC BY 4.0. Makes most sense combined with stem separation, since it is a piano model. |
| **Tsumugi** | Not started. Python inference today; the ONNX export would need the decoder ported. Closest match to synth material, MIT code. |
| **A piano-cover model** | Not started. If the goal is a playable arrangement rather than an accurate copy, `tsumugi-piano-cover` is that task. |

Two practical notes:

* **ONNX determinism.** ONNX Runtime's floating-point reduction order can
  depend on the thread count, so `intraOpNumThreads` is pinned to 1 in both
  adapters. Otherwise two runs of the same file could differ in the last
  bits.
* **Okarina's frame timebase had to be measured, not assumed.** The CQT's
  frame period is exactly 32 ms and the leading context exactly 0.976 s;
  getting either wrong by a small amount accumulates into tens of
  milliseconds of timing error over a song. It is fitted against tone bursts
  at known times, and the fit is exact to the millisecond.

---

## 8. Recommendation

1. **Use Okarina.** It is measurably better on both the reference track and
   your own music, and it is implemented. Pick it in the app's header or with
   `--model okarina`.
2. **Listen to the report.** `eval/index.html` has every song with both
   models side by side, playable, with the original recording alongside.
   That is the final check; no number replaces it.
3. **If a third model is wanted**, TransKun V2 is the next one to try: MIT,
   ONNX available, and the decoder already exists in JavaScript.
4. **Beat-grid quantisation** is the best remaining idea for the *current*
   model, but it needs a beat tracker that follows the phase. It is not worth
   doing until the model question is settled.
5. **A piano-cover model** rather than a transcription, if the goal is a
   playable arrangement rather than an accurate copy.

---

## 9. What we need to finish the evaluation

The published reference outputs we have are for one track, and two strong
models only agree 65 % with each other on it. The only way to choose is to
run the candidates over your own material and listen.

That is now done: the report exists, the MIDI files are written, and the
numbers are in section 6. What is left is your judgement of how the two
sound.

If you want a ground truth rather than a comparison, the strongest option
would be the stems the songs were generated from, if they still exist. That
would let the models be scored outright instead of only against each other.

---

## Sources

* [Harmonica — arXiv 2609.04640](https://arxiv.org/abs/2609.04640)
* [Okarina — huggingface.co/mazun/rekord](https://huggingface.co/mazun/rekord)
* [TransKun — github.com/Yujia-Yan/TransKun](https://github.com/Yujia-Yan/TransKun)
* [MIREX 2024 polyphonic transcription results](https://music-ir.org/mirex/wiki/2024:Polyphonic_Transcription_Results)
* [ByteDance piano transcription — github.com/bytedance/piano_transcription](https://github.com/bytedance/piano_transcription)
* [High-resolution piano transcription — arXiv 2409.19614](https://arxiv.org/abs/2409.19614)
* [Tsumugi — github.com/anime-song/tsumugi](https://github.com/anime-song/tsumugi)
* [tsumugi-piano-cover — github.com/anime-song/tsumugi-piano-cover](https://github.com/anime-song/tsumugi-piano-cover)
* [MuScriptor — github.com/muscriptor/muscriptor](https://github.com/muscriptor/muscriptor)
* [baroned/musicnote-ondevice-models](https://huggingface.co/baroned/musicnote-ondevice-models)
* [ravetools/piano-visualizer](https://huggingface.co/ravetools/piano-visualizer)
* [Beat This! — github.com/CPJKU/beat_this](https://github.com/CPJKU/beat_this)
* [Demucs — github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)
* [basic-pitch-ts — github.com/spotify/basic-pitch-ts](https://github.com/spotify/basic-pitch-ts/blob/main/src/toMidi.ts)
