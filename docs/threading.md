# Is the site threading correctly, or is it single-threaded?

Written on 11 November 2026, after the question was asked and measured rather
than guessed at.

**The short version: the site is single-threaded, on purpose, and the reason
it gives for that purpose is weaker than it claims.** The thread count does
not change the exported file — not for any model, not in the browser, not on
your own tracks — and turning threads on makes a conversion about twice as
fast. What it costs is a set of headers on the web server that cross-origin
isolate the site, and a thinner margin than the current comment suggests.

---

## 1. The short answer

The site runs the model on **one thread**. It is not broken, and it is not
hanging on a lock: it is pinned in three separate places, and the third is
enforced by the browser itself.

| layer | where | what it does |
|---|---|---|
| the runtime's own setting | `src/core/onnx-web.ts:34` | `ort.env.wasm.numThreads = 1` — the browser only |
| the session options | `src/core/onnx.ts:106` | `intraOpNumThreads: 1, interOpNumThreads: 1, executionMode: 'sequential'` — the browser and the command line |
| the platform | the web server | no COOP or COEP headers, so there is no `SharedArrayBuffer`, so the runtime is not allowed to thread at all |

The third layer is the interesting one. Even if the first two were removed,
the site as it is served today would still run on one thread, because
cross-origin isolation is what the browser requires before it will hand out
`SharedArrayBuffer`.

The runtime says so itself. In `node_modules/onnxruntime-web/dist/ort.mjs`
the WebAssembly proxy checks whether isolation is on, and if it is not:

```js
const multiThreadSupported = isMultiThreadSupported();
if (numThreads > 1 && !multiThreadSupported) {
  if (typeof self !== "undefined" && !self.crossOriginIsolated) {
    console.warn(
      "env.wasm.numThreads is set to " + numThreads + ", but this will not work unless you enable crossOriginIsolated mode."
    );
  }
  console.warn(
    "WebAssembly multi-threading is not supported in the current environment. Falling back to single-threading."
  );
  flags.numThreads = numThreads = 1;
}
```

That is `ort.mjs:1784-1795`. The fallback is not ours; it is the runtime
giving up on its own.

## 2. What the live site actually does

Measured against `http://192.168.100.109/` on 11 November 2026, with a
24-second clip, on a machine with twelve cores.

| what was asked | what it said |
|---|---|
| `crossOriginIsolated` | `false` |
| `SharedArrayBuffer` | `false` |
| `navigator.hardwareConcurrency` | `12` |
| Workers created during the whole transcription | **0** |
| CPU used by the browser process tree | **1.1 to 1.2 cores**, steady |
| the whole conversion | **43.5 s** cold, **44.6 s** warm |

The CPU figure is the one that settles it. The tree never went above one and
a half cores, and it sat at one and a tenth for the entire run, which is one
thread saturated. If the graph had been spread over four or eight threads,
the tree would have been burning four or eight cores.

The 94 MB of model weights are not part of that time: they download from the
LAN in 1.6 seconds.

The page stays responsive during the conversion — a click landed eight
seconds in and took 53 ms — because the model runs one window at a time and
yields between them. That is a separate matter from threading and it works
either way.

## 3. Why it was pinned

The comment in `src/core/onnx.ts` says:

> ONNX Runtime reduces in an order that depends on the thread count, so the
> count is pinned. Otherwise two runs of the same file could differ in the
> last bits, which is exactly what this project promises not to do.

That reason is **true of the raw numbers and false of the file**, and the
difference matters.

### The raw numbers do move

In Node, the raw output of the piano graph differs at four threads and
above. One thread and two threads agree with each other; four and eight
agree with each other; the two groups do not agree with one another.

In the browser, the WebAssembly build does not have this problem: the raw
output digests were identical at 1, 2, 4 and 8 threads.

### The file does not move

The exported MIDI is byte-for-byte identical at every thread count, in every
place it was checked.

| what was compared | result |
|---|---|
| Node, the piano model, exported MIDI | identical at 1 / 2 / 4 / 8 |
| Node, Okarina, exported MIDI | identical at 1 / 2 / 4 / 8 |
| Node, Tsumugi, exported MIDI | identical at 1 / 2 / 4 / 8 |
| Node, YourMT3+, exported MIDI | identical at 1 / 2 / 4 / 8 |
| Browser, exported MIDI | identical at 1 / 2 / 4 / 8 |
| Browser at 1 / 2 / 4 / 8 threads against the command line | **identical** — sha256 `7b4930d4efb7df8f0c9e813677b78300a5a9d907ea0c390e7dff8ef1e25f10cd` |
| The command line itself, at 1 / 2 / 4 / 8 threads | **identical** — the same sha256, `7b4930d4…` |
| Your own tracks — the first five, 45 seconds each | identical at 1 thread and at 4 |

The last two are the ones that matter. The promise the tool makes is that
the browser and the command line produce the same file, and that the file
does not change from one machine to another. Both hold with threads on.

The reason is that the differences are in the last bits of a float, and the
model's own thresholds turn those into the same notes. The piano model
regresses an onset and an offset and compares them against a threshold; a
float that is different by one part in ten million crosses the threshold
only if the note is sitting exactly on it, and when it does, the difference
is one note out of hundreds.

**So the pin is protecting a margin that is much larger than the
difference.** It is not protecting nothing — but the comment overstates the
danger.

## 4. What threads would buy

Measured with the isolation headers in place, which is the only way to let
the browser use more than one thread.

### The whole conversion, in the browser

| threads | the conversion | the notes |
|---|---|---|
| 1 | 45.7 s | Piano 474, Drums 72 |
| 2 | 33.0 s | Piano 474, Drums 72 |
| 4 | 30.4 s | Piano 474, Drums 72 |
| 8 | **25.2 s** | Piano 474, Drums 72 |

**1.8 times faster**, and the notes are the same at every count.

### The whole conversion, on the command line

| threads | the model | the rest | the whole thing |
|---|---|---|---|
| 1 | 21.9 s | 0.1 s | 22.0 s |
| 2 | 12.7 s | 0.1 s | 12.8 s |
| 4 | 12.6 s | 0.2 s | 12.8 s |
| 8 | 10.4 s | 0.2 s | **10.6 s** |

**Just over twice as fast.** The drum detector, the tempo and the key take a
fifth of a second between them, so almost everything is the model.

### The speedup is not linear, and eight threads is not always the best

Per model, on the same clip, in Node:

| model | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| piano | 33.6 s | 30.0 s | 17.4 s | **14.0 s** |
| okarina | 3.4 s | 2.7 s | **2.5 s** | 4.2 s |
| tsumugi | 30.2 s | 22.1 s | **20.5 s** | 22.2 s |
| yourmt3 | 37.2 s | 28.7 s | **27.1 s** | 29.6 s |

Okarina is *slower* at eight threads than at four. Tsumugi and YourMT3+ both
turn round at eight. Only the piano model — the default — keeps going.

**Four threads is the safe number.** Eight is faster for the default model
and slower for three of the other four.

## 5. What it would cost

### The web server has to isolate the site

For the browser to use more than one thread it needs `SharedArrayBuffer`,
which needs cross-origin isolation, which needs two headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The site sends neither today. `ops/nginx/voidtrackr-headers.conf` would
grow them.

What that costs:

- `window.opener` is severed. The site does not use it and does not need
  to.
- Every subresource must be same-origin or carry `Cross-Origin-Resource-
  Policy`. Everything the site loads is its own — there is no CDN, no
  third-party asset — so nothing breaks.
- The site can no longer be embedded in a cross-origin frame. The
  Content-Security-Policy already says `frame-src 'none'`, so this changes
  nothing.
- The `/ort/` runtime and the model weights are served from the same origin
  and need no change.

**The one that needs thought is this:** if the browser threads and the
command line does not, the two could stop agreeing, because the Node runtime
does produce different raw numbers at four threads and above. That is why
the honest version of this change turns threads on in **both** places, and
then checks that the two still produce the same file. The check is in
section 3 above, and it passes: the browser at 1, 2, 4 and 8 threads all
produce `7b4930d4…`, which is what the command line produces.

### The margin is thinner than the comment says

The raw numbers do move in Node at four threads and above. The exported file
does not, in any test, over any material measured. But the safety the pin
buys is real, and it is the kind of safety that only shows up when it is
missing: a note that lands exactly on a threshold, on a machine with a
different core count, in a version of the runtime that reduces differently
again.

## 6. The three ways this can go

**A. Leave it as it is.** One thread. The conversion takes 44 seconds in the
browser and 22 on the command line. Nothing changes, nothing can break, and
the promise that the same file comes out of the browser and the command line
does not depend on the runtime's reduction order at all.

**B. Turn threads on in both, and isolate the site.** Two headers on nginx,
one line in each runtime adapter. The conversion takes 25 seconds in the
browser and 12 on the command line, at four threads or eight. The site
becomes cross-origin isolated, which costs nothing it currently uses. The
file does not change: this was measured in the browser at four thread counts
and on the command line at four thread counts, over your own tracks as well
as the test clip.

**C. Turn threads on in the command line only.** The command line gets
nearly twice as fast and the site is untouched. The risk is that the
command line and the browser stop agreeing, because they would be reducing
in different orders. That risk is already closed by measurement: the command
line at 1, 2, 4 and 8 threads produces `7b4930d4…`, which is what the
browser produces at 1, 2, 4 and 8 threads. So C is safe today, and it is
the cheapest of the three.

**My recommendation is B, at four threads.** The reason the pin exists is
reproducibility, and B keeps the reproducibility by construction rather than
by luck: both sides thread the same way, the file is checked to be the same
on both sides, and the number of threads is chosen from what was measured
rather than from what the machine happens to have. Four threads rather than
eight because three of the five models are slower at eight.

But this is your call, and the measurements above are what the call should
be made on. Nothing has been changed: the site is exactly as it was.

## 7. The measurements

| script | what it answers |
|---|---|
| `scripts/onnx-threads.mts` | the graph alone, at each thread count, with a digest of the notes |
| `scripts/onnx-threads-pipeline.mts` | the whole conversion, at each thread count, and the hash of the file |
| `scripts/onnx-threads-models.mts` | the exported file, for every model in the catalogue |
| `scripts/onnx-threads-material.mts` | the exported file, over your own tracks |
| `scripts/onnx-threads-midi.mts` | the exported file, note by note |
| `scripts/onnx-threads-diff.mts` | the notes that differ, if any do |
| `scripts/onnx-threads-digest.mts` | the raw output of the graph, at each thread count |

The browser numbers came from the site itself, driven through a proxy that
adds the two isolation headers, because the site does not send them.

The isolated measurements were made through the development server, because
that is where the proxy could point without rebuilding and redeploying the
site. It is the same source, the same weights and the same graph; the only
difference is that the development server serves unbundled modules. The two
agree at one thread — the live site takes 43.5 to 44.6 seconds and the
development server 45.7 — so the comparison is sound.

To measure a thread count other than one in the browser, the runtime's own
setting and the session options have to be changed. That was done as a
temporary change, measured, and reverted: the shipped code still says
`ort.env.wasm.numThreads = 1`.
