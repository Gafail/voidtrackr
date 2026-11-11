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

**1. Collapse.** The same key struck again within 50 ms is one note, not two. A
key cannot be re-struck that fast, so this is not a judgement call about the
music — it is the instrument. The folded note starts when the first strike did,
so the onset does not move, and it lasts as long as the longest of the strikes
it swallowed.

That last part matters more than it sounds. The first version made the folded
note last until the *last* strike ended, and the fold was a chain: the merged
note grew, so the next strike was measured against the longer note and folded
in as well. A key struck every forty milliseconds for ten seconds became one
note of ten seconds, and the output had notes of **seventeen seconds** in it.
No piano has a note like that, and it is what a person looking at the piano
roll would call unreasonable. The fold is now pairwise — each strike is judged
against the strike before it — and the longest note in the output is the
longest the model itself reported.

**2. Divide.** The notes are split between a left and a right hand. The split
is at **middle C**, which is where the grand staff splits: below it is the
left hand, playing the bass, and from it up is the right hand, playing the
melody. The hands overlap by a third, the way they do on a real keyboard.

The split used to be worked out from the material — the point that left both
hands least overstretched. It was measured, and it was wrong in a way that is
obvious on the piano roll. This project's material is a mix, and the median of
its texture sits at MIDI 57 to 76, well above middle C, so a split that
balances the load lands in the treble. The left hand ended up playing the
melody, up to **MIDI 87**, and the right hand played the bass, down to **46**.
The two hands were the wrong way round.

**3. Trim.** Each hand gets five notes at a time and an octave of reach. Notes
that start together count as at the same moment. Within a chord the right hand
is given the top of it first and the left hand the bottom, because those are
the notes the music is about: the melody on top, the bass underneath. When a
hand is full, the note it gives up is then the one furthest from the line it
is playing — the low note of a right-hand chord, rather than the melody above
it. A note that arrives when its hand is full goes to the other hand if that
hand can take it and the note is close enough to the split to be that hand's
music; otherwise it is dropped.

**4. Thin.** A hand plays at most ten notes a second. Where the transcription
asks for more, the quietest notes go.

Before the hands are divided, the notes that are too short to play are
lengthened. A quarter of the model's notes are one frame long — ten
milliseconds, which is a click and not a note — and the fastest staccato on a
real piano is about sixty milliseconds. The onset is kept where it is, because
that is the part the model gets right, and only the end moves.

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
| notes per file | 2001 | 1008 |
| notes a second | 11.1 | 5.6 |
| the same key struck again | 708 | 58 |
| notes struck at one moment, one hand | up to 11 | 5 |
| the widest chord, one hand | 75 semitones | 12 |
| notes in the busiest second, one hand | up to 60 | 10 |
| the longest note | 6.0 s | 6.0 s |
| of the model's notes, how many are real | 24.7 % | **28.4 %** |
| of Okarina's notes, how many are found | 32.3 % | 18.1 % |

The last two lines are the point. The stage keeps half the notes, and the
notes it keeps are a seventh more likely to be right than the ones it threw
away. It is not deleting at random.

The cost is in the line above it: 14.2 points of the reference are no longer
found. That is a real loss and it is not hidden. A transcription that is
supposed to be *right* and a transcription that is supposed to be *playable*
are different products, and this stage trades the first for the second.

The numbers move with the settings, and the script sweeps them:

| configuration | notes | notes/s | F1 |
|---|---|---|---|
| untouched | 2001 | 11.1 | 26.6 % |
| collapse only | 1349 | 7.5 | 24.3 % |
| collapse, two hands | 1235 | 6.8 | 22.6 % |
| collapse, two hands, 12 a second | 1095 | 6.1 | 22.1 % |
| collapse, two hands, 10 a second | 1008 | 5.6 | 21.6 % |
| collapse, two hands, 8 a second | 886 | 4.9 | 20.4 % |

Ten a second is about as fast as a hand plays. Eight costs a point of F1 for
nothing, and twelve keeps a few more notes but sits at the very edge of what
a hand can do, so the default is ten.

### What the correction changed

The stage shipped in 0.6.0 arranged the notes into the wrong hands, and
looking at the piano roll is how it was found. `scripts/pianify-compare.mts`
runs the shipped stage and the corrected one over the same material — the
shipped one is read out of the git history, so it is the code that was
actually shipped:

| | shipped 0.6.0 | corrected |
|---|---|---|
| the highest note the left hand plays | **87** | 64 |
| the lowest note the right hand plays | **46** | 56 |
| notes of 20 ms or less | **27.0 %** | 0.0 % |
| the longest note | **17.31 s** | 6.00 s |
| the melody of a chord kept | 53.8 % | 53.9 % |
| the bass kept | 56.3 % | 56.1 % |
| notes | 1047 | 1008 |

The left hand reaching MIDI 87 and the right hand down to 46 is the two hands
playing each other's music. The quarter of the notes that were one model frame
long is a click rather than a note. And the longest note is now the longest
the model itself reported, rather than one the fold invented.

## What it does not do

- **It does not write a melody.** The right hand plays the notes above the
  split, which is usually the melody and sometimes is not. Choosing the
  melody properly needs a salience model, and nothing here has one.
- **It does not re-voice chords.** A chord of six notes becomes two hands'
  worth of notes, not a triad in a good inversion.
- **It does not snap to the beat.** The model's own onsets are better than
  the beat grid, and the grid this project has is the frame grid, which
  comes after.
- **It does not touch the sustain.** Up to sixteen notes can still sound at
  once, because a piano with the pedal down can do that. A file without pedal
  cannot, and trimming the durations to fit would be a bigger change than
  this stage is.
- **It does not use a second model.** Agreement between models would be a
  better noise detector than loudness. It would also double the time of
  every conversion, which for a piano run is minutes.

### The melody

It is worth saying what the stage found about the melody, because it is a
negative result. The right hand is meant to be playing it, and the melody of a
chord is its top note — so the question is whether the stage keeps it.

Measured over the material (`scripts/pianify-melody-keep.mts`), the top note
of a chord survives in **53.9 %** of the chords. That is not a failure of the
stage; it is the same number the model itself would score if you asked it to
name the melody, because this material is a mix of four instruments and there
is no melody in it for a model to find.

A melody tracer was tried and measured (`scripts/pianify-melody.mts`): a
Viterbi trace over the onsets moves by a step 0.8 % of the time against 0.3 %
for the top-of-chord baseline, which is not smooth enough to be worth
shipping. The source is a mix, not a piano, and the hand-separation models in
the literature all need piano audio or a score. So the stage does not try to
find the melody and put it in the right hand — it keeps the top of each chord,
which is the melody when there is one.

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

## The measurements

Everything above was measured, and the scripts are in `scripts/`. They are
kept because a number that cannot be reproduced is a claim, not a fact.

| script | what it answers |
|---|---|
| `pianify-stats.mjs` | what the model's output looks like before the stage |
| `pianify-value.mts` | whether the loudness head tells real notes from noise |
| `pianify-experiment.mts` | playability and accuracy, and the sweep of the settings |
| `pianify-compare.mts` | the shipped stage against the corrected one |
| `pianify-pitch.mts` | where the music sits, per file |
| `pianify-split.mts` | where the hands should meet |
| `pianify-crossing.mts` | how far a note may cross to the other hand |
| `pianify-melody-keep.mts` | whether the melody of a chord survives |
| `pianify-order.mts` | the old note order against the new one |
| `pianify-order-detail.mts` | the same, by chord width |
| `pianify-duration.mts` | how long the notes are, and how many a hand holds |
| `pianify-merge-sweep.mts` | the fold: the gap and how long a folded note is |
| `pianify-long-notes.mts` | what is inside the long notes |
| `pianify-fold-depth.mts` | the gap between one strike on a key and the next |
| `pianify-look.mts` | the arrangement as a text piano roll |

`pianify-look.mts` is the one that settles an argument. It prints the
arrangement as a piano roll in the terminal — the left hand in one mark, the
right hand in another, one column every fifty milliseconds — and it is how the
wrong-handed arrangement of 0.6.0 was seen to be wrong. A table of averages
says the hands were within their limits; the roll shows that the left hand
was playing the treble.

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
