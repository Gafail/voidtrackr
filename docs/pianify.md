# Pianify

Turning a transcription into something a pair of hands could play.

## The question

The piano model is the best of the five here on this project's own material,
and it is still not music a person could play. The question this document
answers is whether there is a known conversion that turns that output into
something piano-playable — not literally playable, but more musical: fewer
notes, a clear top line, and a texture a hand can hold.

The answer is no, there is no such conversion you can point at. The pieces
exist, and they are the right pieces, but none of them does this job. What
follows is what the literature says, what was measured, and what the stage
that was built actually does.

## What the literature has

Four papers, all verified on arXiv.

**Shibata, Nakamura & Yoshii, "Non-Local Musical Statistics as Guides for
Audio-to-Score Piano Transcription", Information Sciences 566 (2021) 262,
[arXiv:2008.12710](https://arxiv.org/abs/2008.12710).** The closest thing to
an answer. It integrates a DNN multipitch detector with statistical rhythm
quantisation, and it uses *non-local* musical statistics — priors over pitch
and rhythm, some of them learned from separated hand parts — to correct the
global properties of the transcription: its tempo scale, its metre, where
the bar lines are. It reports a 7.1 % transcription error rate and an 85.6 %
downbeat F-measure, and it says the transcriptions it makes "can be
partially used for music performance". It is the direct ancestor of the idea
that a transcription can be *repaired* by musical knowledge rather than only
by listening harder.

**Srivatsan & Berg-Kirkpatrick, "Checklist Models for Improved Output
Fluency in Piano Fingering Prediction", ISMIR 2022,
[arXiv:2209.05622](https://arxiv.org/abs/2209.05622).** The most useful idea
in this whole survey, and it is a negative one: per-note labelling precision
does *not* measure playability. The paper defines fluency as statistics over
*adjacent* finger pairs — how hard the transition from one note to the next
would be for a hand — and trains a model to minimise them, with
reinforcement learning on top. Human experts confirm the performability gain.
The lesson for this project is that playability is a property of the notes
*together*, and has to be constrained directly rather than hoped for.

**Penner & Hindle, "Statistical Models for Automatic Fingering-Annotated
Piano Sheet Music Transcription", September 2026,
[arXiv:2609.28787](https://arxiv.org/abs/2609.28787).** The full pipeline, in
one paper: piano audio to transcription to hand separation to fingering.
Hand separation reaches 90.8 % and joint hand-and-finger prediction 56.6 %
on the PIG dataset. This is what "pianify" would look like if the weights
were available.

**Ramoneda et al., "Combining piano performance dimensions for score
difficulty classification", 2023,
[arXiv:2306.08480](https://arxiv.org/abs/2306.08480).** The CIPI dataset
("Can I Play It?"), with Henle difficulty labels, classified at 39.47 %
balanced accuracy over nine levels. Useful as a *target*: a stage could be
told "make this playable by a beginner" and be scored on it. Nothing in this
project does that yet.

### Why none of them is the answer

Every one of them needs either piano audio or a score. Hand separation,
fingering prediction and difficulty classification are all trained on
performances of real pianos, with the notes already known. This project's
input is the opposite: a noisy transcription of a mix, where nothing is
known. Running a piano model over the mix gets closer to the piano-audio
domain, but it does not get there, and these papers do not offer a way to
apply their models to the output of another model.

So the conversion has to be built, from the same physical and musical
constraints the papers use.

## What is actually wrong with the output

Measured over the 19 tracks in `Music Import/`, transcribed with the piano
model (`scripts/pianify-stats.mjs`, `scripts/pianify-value.mts`):

| | the model | a piano |
|---|---|---|
| notes a second | 11.1 | about 6, across both hands |
| the same key struck again within 60 ms | 35.4 % of all notes | almost never |
| notes struck at one moment | up to 19 | 10, and 5 per hand |
| the widest chord | 75 semitones | 12 per hand |
| notes in the busiest second | up to 60 | 10 per hand |

Three of those are not "noisy" in the sense of wrong notes. They are a
transcription of music that has no hands in it. A key that is struck again
five milliseconds later is one note that the model cut in two; a chord that
reaches three octaves is a chord that is really several events; and forty-one
notes in half a second is a texture that no player could plan.

The fourth one *is* about wrong notes: 36.5 % of the piano model's notes are
found by no other model. But the loudness head does not separate them: the
notes another model also found have a mean velocity of 0.689 and the ones
only the piano model found have 0.662. Loudness is a weak signal — good
enough to break a tie, not enough to decide on its own. The other way to
tell noise from music is to run a second model and keep the notes they agree
on, which is a real option and much more expensive: it doubles the
transcription time.

## The stage

`src/core/pianify.ts`. Four passes, in this order.

**1. Collapse.** The same key struck again within 50 ms is one note, not two.
The folded note starts when the first strike did and ends when the last one
did, so none of the sound is lost, and it keeps the loudest of the strikes.
A key cannot be re-struck that fast, so this is not a judgement call about
the music — it is the instrument.

**2. Divide.** The notes are split between a left and a right hand. The split
point is not fixed at middle C: it is worked out from the material, as the
pitch that leaves both hands least overstretched. Below the split is the
left hand, which is playing the bass; above it is the right hand, which is
playing the melody.

**3. Trim.** Each hand gets five notes at a time and an octave of reach.
Notes that start together count as at the same moment. A note that arrives
when its hand is full goes to the other hand if that hand can take it and
the note is close enough to the split to be that hand's music; otherwise it
is dropped.

**4. Thin.** A hand plays at most ten notes a second. Where the
transcription asks for more, the quietest notes go.

The rules are the constraints a piano has: ten fingers, five per hand, an
octave of reach, and a finger that cannot move faster than about ten times a
second. They are the same kind of constraint the checklist models use — not
per note, but over the notes together.

## What it costs

`scripts/pianify-experiment.mts` runs the stage over the material and scores
the result twice: playability, and F1 against Okarina's transcription of the
same audio at 50 ms. Okarina is not ground truth — it is the best model here,
at F1 85.3 % against the published Harmonica output — but it is the closest
thing to a right answer this project has, and the stage never looks at it.

| | the model | arranged |
|---|---|---|
| notes per file | 2001 | 1047 |
| notes a second | 11.1 | 5.8 |
| the same key struck again | 708 | 34 |
| notes struck at one moment, one hand | up to 11 | 5 |
| the widest chord, one hand | 75 semitones | 12 |
| notes in the busiest second, one hand | up to 60 | 10 |
| of the model's notes, how many are real | 24.7 % | **29.8 %** |
| of Okarina's notes, how many are found | 32.3 % | 19.9 % |

The last two lines are the point. The stage keeps half the notes, and the
notes it keeps are a fifth more likely to be right than the ones it threw
away. It is not deleting at random.

The cost is in the line above it: 12.4 points of the reference are no longer
found. That is a real loss and it is not hidden. A transcription that is
supposed to be *right* and a transcription that is supposed to be *playable*
are different products, and this stage trades the first for the second.

The numbers move with the settings, and the script sweeps them:

| configuration | notes | notes/s | F1 |
|---|---|---|---|
| untouched | 2001 | 11.1 | 26.6 % |
| collapse only | 1349 | 7.5 | 24.3 % |
| collapse, two hands | 1264 | 7.0 | 24.2 % |
| collapse, two hands, 12 a second | 1131 | 6.3 | 23.8 % |
| collapse, two hands, 10 a second | 1047 | 5.8 | 23.3 % |
| collapse, two hands, 8 a second | 925 | 5.1 | 22.3 % |

Ten a second is about as fast as a hand plays. Eight costs a point of F1 for
nothing, and twelve keeps a few more notes but sits at the very edge of what
a hand can do, so the default is ten.

## What it does not do

- **It does not write a melody.** The right hand plays the notes above the
  split, which is usually the melody and sometimes is not. Choosing the
  melody properly needs a salience model, and nothing here has one.
- **It does not re-voice chords.** A chord of six notes becomes two hands'
  worth of notes, not a triad in a good inversion.
- **It does not snap to the beat.** The model's own onsets are better than
  the beat grid, and the grid this project has is the frame grid, which
  comes after.
- **It does not touch the sustain.** Nineteen notes can still sound at once,
  because a piano with the pedal down can do that. A file without pedal
  cannot, and trimming the durations to fit would be a bigger change than
  this stage is.
- **It does not use a second model.** Agreement between models would be a
  better noise detector than loudness. It would also double the time of
  every conversion, which for a piano run is minutes.

## Where it sits

    model -> drums -> tempo and key -> pianify -> range -> grid

The tempo and the key are estimated from the notes as the model found them,
because they belong to the song rather than to the arrangement. The
arrangement comes before the range, because it is a decision about the whole
piece: the split between the hands and the notes that are dropped depend on
the music as a whole. The grid comes last because a game rendering thirty
frames a second wants the arranged notes on its frames.

In the app the arrangement is a switch in the header. It is applied to the
notes as they stand, so it can be turned on and off without transcribing
again. It is a change to the notes rather than a view over them, because it
changes what the tracks are: two hands instead of one.

## References

- [arXiv:2008.12710](https://arxiv.org/abs/2008.12710) — Shibata, Nakamura
  & Yoshii, non-local musical statistics as guides for audio-to-score piano
  transcription.
- [arXiv:2209.05622](https://arxiv.org/abs/2209.05622) — Srivatsan &
  Berg-Kirkpatrick, checklist models for output fluency in piano fingering
  prediction.
- [arXiv:2609.28787](https://arxiv.org/abs/2609.28787) — Penner & Hindle,
  statistical models for automatic fingering-annotated piano sheet music
  transcription.
- [arXiv:2306.08480](https://arxiv.org/abs/2306.08480) — Ramoneda et al.,
  combining piano performance dimensions for score difficulty classification.
