# A licence request for the Harmonica reproduction

**What this is:** a draft you can post as a GitHub issue on
<https://github.com/z-huang/harmonica>. The author, Zion Huang, has no public
email address on GitHub, and the repository has issues enabled with none open,
so an issue is the natural route. It also has the advantage of leaving the
grant in public view, which is what a licence is for.

Nothing here has been sent. It is yours to edit, shorten or drop.

---

## The draft

**Title:** Licence for the bundled Harmonica weights

**Body:**

Hi Zion,

Your reproduction of Harmonica looks excellent — the numbers in your README
sit just under the paper's and still comfortably ahead of everything else
released, and bundling the weights for all five sizes in the wheel makes it
the only way to actually run the model.

One thing stops me from using it in a tool I am building: there is no licence
file in the repository, no licence classifier on the PyPI metadata, and
`license: null` on the project page. As far as I can tell, the weights and
the code are therefore all-rights-reserved, and I cannot ship them or build
anything commercial on top of them.

Would you consider adding a licence? MIT or Apache-2.0 would do for my case,
and either would be more than enough for this project — the code is a
reimplementation from the paper, so there is no upstream code to conflict
with.

A few specifics, in case they help you decide:

- The package as it stands reproduces Harmonica faithfully and adds a velocity
  head the paper does not have. I would credit it as "an independent
  reproduction of Harmonica" and link back to the repository and the paper
  wherever the model appears — in the app, in the README, and in the
  provenance stamp written into every exported MIDI file.
- I would not redistribute the weights myself. The tool would fetch them with
  a pinned URL and a SHA-256 check, the same way it already handles every
  other model it uses, so your copy stays the only one.
- If a licence is not what you want, a short note saying what *is* allowed
  would help just as much. "Non-commercial only" or "ask me first" are both
  answers I can work with; silence is the one thing I cannot.

If there is a reason for the absence that I have not thought of — you may be
waiting on something from the original authors, or there may be a
training-data question I cannot see — I would rather know than assume.

Thanks for the work either way.

---

## Why this is worth an email

Nothing publicly released beats Harmonica. Okarina, the Harmonica-class
reimplementation this project already ships, is CC BY-NC-SA 4.0 and therefore
non-commercial. Every other Harmonica derivative is either non-commercial or
unlicensed in the same way. The permissive models that do exist — Tsumugi
(MIT), YourMT3+ (Apache-2.0), ByteDance's piano transcription (CC BY 4.0) —
are all weaker on this material.

So this one file is worth more than any amount of further engineering.

## If they ask what it would be used for

A local-first audio-to-MIDI converter that runs entirely on the user's own
machine: audio in, multi-track MIDI out, with a piano-roll editor. No
accounts, no uploads, no telemetry. The model runs on the user's own
hardware, and the weights are fetched from the author's own repository and
verified against a pinned hash — the author keeps sole control of what is
distributed.

## Notes on the route

- The repository is <https://github.com/z-huang/harmonica>. Issues are enabled
  and there are none open, so one will be seen.
- The author is Zion Huang, NTU CSIE, Taipei. GitHub lists no email address,
  so an issue is the route that will reach them.
- The PyPI project is <https://pypi.org/project/harmonica-amt/>. Its metadata
  also shows `license: null` and no licence classifiers, so a licence would
  need adding there too — worth mentioning in the issue.
