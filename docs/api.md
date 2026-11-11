# The API

Written on 12 November 2026.

**The short version: the site has an address you can call.** The browser
does the work and nothing is uploaded, but a tool that is driven from a
script needs somewhere to send a file. So the box runs a small service
alongside the site, at the same origin, and it answers at `/api/`.

```bash
curl -F file=@song.wav http://192.168.100.109/api/convert -o song.mid
```

That is the whole thing: one line, and the MIDI file comes back.

---

## 1. What it is, and what it is not

The site is still static and the browser still does the work when you use
it. The API is a **second way in**, for scripts, and it is the only part of
the tool where an audio file leaves the browser.

That is worth being plain about, because the site's promise is "the audio
never leaves the machine it is on":

- **In the browser**, the audio is decoded and transcribed on the machine
  you opened the site on. It is never sent anywhere. This has not changed.
- **Through the API**, the audio is sent to the box at
  `192.168.100.109`, and the box does the work. The box is on your own
  network, and it is the same machine that serves the site, so the audio
  does not leave the house — but it does leave the browser.

The API is opt-in. Nothing in the site calls it.

## 2. The endpoints

| method | path | what it does |
|---|---|---|
| `GET` | `/api/` | an index of the endpoints, or a page about them |
| `GET` | `/api/health` | whether the server is up, and what it has been doing |
| `GET` | `/api/models` | the models, their licences, and where their weights are |
| `GET` | `/api/openapi.json` | the OpenAPI 3.1 document |
| `POST` | `/api/convert` | convert an audio file to MIDI |

### `GET /api/`

Answers with JSON, or with a page about the API if the caller asks for
HTML. The page has the one-liners and the parameter table on it.

### `GET /api/health`

```json
{
  "status": "ok",
  "pipeline": "0.8.0",
  "threads": 4,
  "defaultModel": "piano",
  "models": ["basic-pitch", "tsumugi", "yourmt3", "piano", "okarina"],
  "conversions": 12,
  "failures": 0,
  "inFlight": 0,
  "concurrency": 1,
  "uptimeSeconds": 4210,
  "nativeBackend": false
}
```

`nativeBackend` says whether the fast TensorFlow.js backend is installed.
It is not, on the box: the native module is large and the box has a
gigabyte of memory, so Basic Pitch runs on the pure-JavaScript backend
there. The other four models do not use TensorFlow.js at all.

### `GET /api/models`

The catalogue, with the licence on each entry. The licence decides whether
the output can be sold, so it is stated rather than buried.

Each entry also carries `weightsUrl`, the path the site serves that model's
weights from. It is worth reading rather than guessing, because the paths
are not all the same shape:

| model | `weightsUrl` |
|---|---|
| `basic-pitch` | `/model/` — a TensorFlow.js model, not an ONNX one |
| `okarina` | `/model-onnx/` — its files sit at the root of the tree |
| `piano` | `/model-onnx/piano/` |
| `tsumugi` | `/model-onnx/tsumugi/` |
| `yourmt3` | `/model-onnx/yourmt3/` |

### `POST /api/convert`

The one that does the work. See below.

## 3. Converting a file

### The request

The audio goes in a multipart part called `file`. The values go in the
query string or as fields of the same body, whichever is easier:

```bash
# The smallest useful request.
curl -F file=@song.wav http://192.168.100.109/api/convert -o song.mid

# With values, as fields.
curl -F file=@song.wav -F model=piano -F pianify=true -F grid=30 \
     http://192.168.100.109/api/convert -o song.mid

# With values, in the query string.
curl -F file=@song.wav "http://192.168.100.109/api/convert?model=okarina&format=json"
```

The body may also be the audio itself, with an audio content type:

```bash
curl --data-binary @song.wav -H 'Content-Type: audio/wav' \
     http://192.168.100.109/api/convert -o song.mid
```

The fields may come before or after the file. `curl` sends the parts in the
order they were given, so `-F file=@song.wav -F model=piano` puts the file
first, and that is read correctly.

### The parameters

| name | default | what it does |
|---|---|---|
| `file` | — | the audio. Required. |
| `model` | `piano` | `basic-pitch`, `tsumugi`, `yourmt3`, `piano` or `okarina` |
| `okarinaSize` | `m` | `s`, `m` or `l`. Only for Okarina. |
| `drums` | `true` | look for a drum kit |
| `tempo` | estimated | force a tempo, in beats per minute |
| `rangeLow` | `0` | lowest frequency to keep, in hertz |
| `rangeHigh` | — | highest frequency to keep, in hertz |
| `grid` | `0` | slots per second. Zero leaves the notes alone. |
| `gridPhase` | `0` | where the first slot sits, in milliseconds |
| `pianify` | `false` | arrange the notes for two hands |
| `format` | `midi` | `midi` or `json` |

These are the same controls the site has, with the same names and the same
meanings, so a value set in the browser means the same thing here.

A parameter that is present but wrong is an error rather than a shrug.
`grid=abc` gets a 400 that says so; it does not get a file that quietly
ignored it.

### The reply

`format=midi` — the default — sends the MIDI file:

```
Content-Type: audio/midi
Content-Disposition: attachment; filename="song.mid"
X-Voidtrackr-Pipeline: 0.8.0
X-Voidtrackr-Model: piano
X-Voidtrackr-Notes: 546
X-Voidtrackr-Tempo: 104.10
X-Voidtrackr-Key: Db major
X-Voidtrackr-SHA256: 8bad2f2d5439c4a1…
```

`format=json` sends the notes instead, for a caller that wants to work with
them rather than play them:

```json
{
  "model": "piano",
  "pipeline": "0.8.0",
  "provenance": "voidtrackr 0.8.0 | model piano-transcription-kong-2020 | sha256:27c96172…",
  "duration": 24.0,
  "tempo": { "bpm": 104.1, "confidence": 0.8, "adjusted": false },
  "key": { "tonic": 1, "mode": "major", "name": "Db major" },
  "stats": { "modelNotes": 608, "finalNotes": 546, "drumHits": 72, "drumsDetected": true },
  "log": ["Piano transcription found 474 notes.", "Detected 72 drum hits."],
  "tracks": [
    {
      "id": "piano",
      "name": "Piano",
      "kind": "melody",
      "notes": [{ "pitch": 60, "startTime": 0.0, "duration": 0.42, "amplitude": 0.8 }]
    }
  ],
  "midi": { "bytes": 4630, "sha256": "8bad2f2d…" }
}
```

## 4. Errors

Every failure comes back as the same shape, with a code a program can
branch on and a status code that matches what went wrong:

```json
{
  "error": {
    "code": "bad_request",
    "message": "\"grid\" must be a number, not \"abc\".",
    "details": { "parameter": "grid", "value": "abc" }
  }
}
```

| code | status | when |
|---|---|---|
| `bad_request` | 400 | a parameter is wrong, or the file is missing |
| `unauthorized` | 401 | a token is set and the one given is wrong |
| `not_found` | 404 | there is nothing at that address |
| `method_not_allowed` | 405 | that method is not allowed there |
| `payload_too_large` | 413 | the upload is over the limit |
| `unsupported_media_type` | 415 | the body is neither multipart nor audio |
| `unprocessable_entity` | 422 | the audio could not be read |
| `busy` | 503 | the server is already converting as much as it can |
| `model_unavailable` | 503 | that model is not installed here |
| `internal_error` | 500 | something went wrong that the caller cannot fix |

## 5. The limits, and why they are there

The box has two cores and a gigabyte of memory.

- **One conversion at a time.** A second request while one is running gets
  a 503 rather than a queue, because two models in memory on a
  gigabyte-sized box is how the box falls over. `VOIDTRACKR_CONCURRENCY`
  raises it.
- **A hundred megabytes per upload.** `VOIDTRACKR_MAX_UPLOAD`, in
  megabytes. nginx has to be told the same number, or it refuses the
  upload before the API sees it.
- **A token, off by default.** `VOIDTRACKR_API_TOKEN` turns it on, and then
  callers have to send `Authorization: Bearer <token>`. It is off because
  the box is on a private network and the site itself is open; it is there
  for anyone who wants it.

### How much memory each model needs

Measured on 12 November 2026, one model per process, on the piano model's
worst case for the recording length. The peak is the kernel's own
high-water mark.

| model | weights | settled | peak | on this box |
|---|---|---|---|---|
| Okarina (m) | 70 MB | 272 MB | **460 MB** | yes |
| piano | 218 MB | 469 MB | **661 MB** | yes, with the arena off |
| YourMT3+ | 671 MB | 713 MB | 1056 MB | **no** |
| Tsumugi | 119 MB | 1039 MB | 1183 MB | **no** |

Tsumugi and YourMT3+ need more than the box has. A request for one of them
will fail rather than produce a file, and giving the box more memory is the
only way to serve them.

### The memory arena

The runtime keeps a pool of allocations for the next run. It is faster, and
it holds on to far more memory than it needs: with the arena **on**, the
piano model peaked at **1.15 GB** and the conversion was killed by the
kernel's out-of-memory killer on a box with a gigabyte. With the arena
**off**, the same conversion peaks at **661 MB**, the notes are
**byte-identical**, and it is about 5 % slower — 9.4 seconds against 8.9 on
a twelve-core machine.

So the arena is off. `DETERMINISTIC_SESSION_OPTIONS` in `src/core/onnx.ts`
turns it off for both runtimes, and the comment there says why.

### The native TensorFlow.js backend

The box does not have it. It is a large native binary and the box does not
have the memory for it, so Basic Pitch runs on the pure-JavaScript backend
there. The two produce **byte-identical** files — that is what the
reproducibility check compares — so the command line and the API still
agree even though they are using different backends.

Without the native backend, TensorFlow.js cannot fetch a `file://` URL,
because it fetches the model's manifest and its weight shards itself. The
API brings its own fetch for that (`fileFetch` in
`src/api/model-loader.ts`), so Basic Pitch works on the box.

## 6. How it runs

- **The server** is `src/api/`, bundled into one file by
  `scripts/build-api.mjs`. It listens on `127.0.0.1:8787` and nothing else
  can reach it.
- **nginx** proxies `/api/` to it. The site's own vhost carries the
  `location /api/` block.
- **systemd** runs it as `voidtrackr-api.service`, from `/opt/voidtrackr/api`.
- **The weights** are the ones the site already serves, at
  `/var/www/voidtrackr/model-onnx`, so the box holds one copy of them
  rather than two.
- **The upload is never written to disk.** It is read into memory,
  converted, and the MIDI is sent back.

### The settings

| variable | default | what it does |
|---|---|---|
| `VOIDTRACKR_PORT` | `8787` | the port to listen on |
| `VOIDTRACKR_HOST` | `127.0.0.1` | the address to listen on |
| `VOIDTRACKR_MODEL_DIR` | `models/onnx` | where the ONNX weights are |
| `VOIDTRACKR_MODEL_PATH` | `public/model/model.json` | where the Basic Pitch model is |
| `VOIDTRACKR_API_TOKEN` | — | a token callers have to present |
| `VOIDTRACKR_MAX_UPLOAD` | `100` | the largest upload, in megabytes |
| `VOIDTRACKR_CONCURRENCY` | `1` | how many conversions at once |

## 7. The same file, whichever way it is asked for

The tool promises that the same input gives the same file however it is
asked for. The API is a new way of asking, so it had to be checked.

The conversion lives in `src/api/convert.ts`, and the command line calls it
as well. There is one code path, not two that agree today.

Measured:

| what | the file |
|---|---|
| the command line, Basic Pitch, `test/sample.wav` | `1014 bytes` |
| the API, Basic Pitch, `test/sample.wav` | the same bytes |
| the command line, the piano model, a 24 second clip | `4630 bytes` |
| the API, the piano model, the same clip | the same bytes |

`scripts/compare-midi.mjs` is what compares them:

```bash
node scripts/compare-midi.mjs <the cli output>.mid <the api output>.mid
```

The browser and the command line were already byte-identical. Now the API
is in the same set.

## 8. The measurements

| script | what it answers |
|---|---|
| `scripts/build-api.mjs` | builds the server into one file, and writes the package.json the box installs from |
| `test/api.test.ts` | the parser, the parameters, and a request end to end |

The tests cover the parts that are easy to get wrong: a multipart body
with binary content in it, the fields arriving after the file, a parameter
that is not a number, the wrong method, an address that does not exist, and
a conversion that has to come back as a MIDI file with the right notes in
it.
