# Is the site threading correctly, or is it single-threaded?

Written on 11 November 2026, after the question was asked and measured rather
than guessed at.

**The short version: the site was single-threaded, on purpose, and the reason
it gave for that purpose was weaker than it claims.** The thread count did not
change the exported file — not for any model, not in the browser, not on your
own tracks — and turning the threads on makes a conversion about twice as
fast. It is now on, at four threads.

---

## 1. The short answer

The site ran the model on **one thread**. It was not broken, and it was not
hanging on a lock: it was pinned in three separate places, and the third was
enforced by the browser itself.

| layer | where | what it did |
|---|---|---|
| the runtime's own setting | `src/core/onnx-web.ts:34` | `ort.env.wasm.numThreads = 1` — the browser only |
| the session options | `src/core/onnx.ts:106` | `intraOpNumThreads: 1, interOpNumThreads: 1, executionMode: 'sequential'` — the browser and the command line |
| the platform | the web server | no COOP or COEP headers, so there was no `SharedArrayBuffer`, so the runtime was not allowed to thread at all |

The third layer is the interesting one. Even if the first two were removed,
the site as it was served would still have run on one thread, because cross-
origin isolation is what the browser requires before it will hand out
`SharedArrayBuffer`.

The runtime said so itself. In `node_modules/onnxruntime-web/dist/ort.mjs`
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

That is `ort.mjs:1784-1795`. The fallback was not ours; it was the runtime
giving up on its own.

## 2. What the live site actually did

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

The page stayed responsive during the conversion — a click landed eight
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

## 5. What it costs

### The web server has to isolate the site

For the browser to use more than one thread it needs `SharedArrayBuffer`,
which needs cross-origin isolation, which needs two headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Both are now in `ops/nginx/voidtrackr-headers.conf`.

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

**The one that needed thought was this:** if the browser threaded and the
command line did not, the two could have stopped agreeing, because the Node
runtime does produce different raw numbers at four threads and above. That is
why the threads were turned on in **both** places, and the two were then
checked against each other. The check is in section 3 above, and it passes:
the browser at 1, 2, 4 and 8 threads all produced `7b4930d4…`, which is what
the command line produced.

### The margin is thinner than the comment said

The raw numbers do move in Node at four threads and above. The exported file
does not, in any test, over any material measured. But the safety the pin
bought was real, and it is the kind of safety that only shows up when it is
missing: a note that lands exactly on a threshold, on a machine with a
different core count, in a version of the runtime that reduces differently
again.

## 6. What was done

**The threads are on, at four.** `ONNX_THREADS` in `src/core/onnx.ts` is four,
both runtimes use it, and the two isolation headers are in
`ops/nginx/voidtrackr-headers.conf`. The command line was the easy half: Node
can thread without anything else changing, and the conversion went from 22.0
seconds to 12.4.

The browser was the awkward half, and the reason is worth writing down.

### The headers alone are not enough

Cross-origin isolation needs a secure context, and a page served over plain
HTTP at a LAN address is not one. Measured, with the isolation headers on:

| how the site is reached | `isSecureContext` | `crossOriginIsolated` | `SharedArrayBuffer` |
|---|---|---|---|
| `http://192.168.100.193/` | **false** | **false** | **false** |
| `http://127.0.0.1:9310/` | true | true | true |
| `https://192.168.100.193:9443/`, trusted certificate | true | true | true |

So adding the headers to nginx changes nothing for a client that opens the
site at its LAN address over plain HTTP. The browser will not isolate the
page, the runtime will not thread, and the conversion stays at 44 seconds.

There are two ways out, and only one of them is set up.

**The one that is set up: the browser is told to treat the address as
secure.** Chrome and Chromium take a flag:

```
--unsafely-treat-insecure-origin-as-secure=http://192.168.100.109
```

With it, the page reports `crossOriginIsolated: true`, `SharedArrayBuffer`
is present, and the runtime uses four threads. It is a flag on the browser
that opens the site, set once in the shortcut, and nothing else changes: the
URL stays the same, there is no certificate and no warning.

**The one that is not: https with a certificate the client trusts.** That
makes the origin secure without asking anything of the browser, and it works
in every browser on every machine. It needs a certificate installed in the
trust store of each client, and the URL changes to `https://`. It is the
better long-term answer, and it is the one to do if the site is ever opened
from more than one machine.

**The site says which of these is in force.** The header shows the thread
count it is actually using — one or four — read from the browser rather than
assumed, so a browser without the flag says so instead of quietly running
slowly.

### The count is four, not eight

Measured per model, on the same clip, in Node:

| model | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| piano | 33.6 s | 30.0 s | 17.4 s | **14.0 s** |
| okarina | 3.4 s | 2.7 s | **2.5 s** | 4.2 s |
| tsumugi | 30.2 s | 22.1 s | **20.5 s** | 22.2 s |
| yourmt3 | 37.2 s | 28.7 s | **27.1 s** | 29.6 s |

Okarina is *slower* at eight threads than at four. Tsumugi and YourMT3+ both
turn round at eight. Only the piano model — the default — keeps going, and
it is the model that is the default for a reason: it is the one that can be
sold.

Four is the number, and it is the number that does not go backwards for
anything in the catalogue.

### The file did not change

After the change, the command line at four threads produces the same notes as
it did at one: 474 piano notes, 72 drum hits, the same tempo, the same key.
Only the provenance stamp moved, 0.7.0 to 0.8.0, because the pipeline is a
different version.

And the browser and the command line still agree. The live site was measured
both ways after the change, with and without the flag:

| | `crossOriginIsolated` | the header says | workers | cores | the file |
|---|---|---|---|---|---|
| without the flag | false | "1 thread" | 0 | 1.25 | `8bad2f2d5439c4a1…` |
| with the flag | true | "4 threads" | **3** | **3.96** | `8bad2f2d5439c4a1…` |

The cores are the proof: one and a quarter without the flag, four with it,
and three workers created to run the graph. The two exported files are the
same bytes, and they are the same bytes as the command line's.

The margin is thinner than the old comment implied, and it is thinner in one
place only: the Node runtime's raw graph output differs at four threads and
above. The exported file does not, in any test, over any material measured.

### A note on the timings

The absolute numbers move with what the machine is doing. Measured on a quiet
machine, the browser conversion went from 44.6 seconds to 30.4, and the
command line from 22.0 to 12.8. Measured while the desktop was busy, the same
two conversions read 83.5 and 41.1 seconds, and 61.7 and 16.3.

The ratios are the durable part, and the core counts are the proof: one and a
quarter cores becomes four.

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
