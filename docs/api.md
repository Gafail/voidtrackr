# The API

> A recording in, a MIDI file out. Nothing is uploaded.

## 1. What it is, and what it is not

The API is the address of the tool: a small service on the same box as the
site that says what the tool can do. It lists the models and the licences
they carry, the parameters a conversion takes and their defaults, and where
each model's weights are served from.

**It does not take a recording.** There is no endpoint that accepts one,
the server never reads a request body, and there is no code path that could.
An upload endpoint was built and then removed: it was the one thing in the
project that broke the promise, and the promise is the point.

So the recording never leaves the machine it is on. It is converted there —
in the browser, or on the command line — and the only thing that travels
over the network is the model's own weights, which are the tool's, not
yours.

### What a script does instead

```bash
# The one line. The recording is read from disk and stays there.
npm run cli -- song.wav -o .

# With the site's model, fetched over HTTP and checked against its hash.
npm run cli -- --server http://192.168.100.109 song.wav -o .
```

The second one calls the site for the catalogue, downloads the model's
weights, checks each file against the pinned SHA-256, and converts. The
recording is not involved in any of that.

## 2. The endpoints

| | | |
|---|---|---|
| `GET` | `/api/` | the index, or a page about the API |
| `GET` | `/api/health` | whether the server is up |
| `GET` | `/api/models` | the models, their licences and their weights |
| `GET` | `/api/parameters` | the parameters a conversion takes |
| `GET` | `/api/openapi.json` | the OpenAPI 3.1 document |

All of them are `GET`. There is nothing that writes, and nothing that
takes a file.

### `GET /api/`

The index. It answers with JSON, or with a page when the caller asks for
HTML with `Accept: text/html`.

```bash
curl http://192.168.100.109/api/
```

### `GET /api/health`

```json
{
  "status": "ok",
  "pipeline": "0.8.0",
  "defaultModel": "piano",
  "models": ["basic-pitch", "tsumugi", "yourmt3", "piano", "okarina"],
  "uptimeSeconds": 3011
}
```

### `GET /api/models`

The catalogue, with the licence on each entry. The licence decides whether
the output can be sold, so it is stated rather than buried.

Each entry also carries `weightsUrl`, the path the site serves that model's
weights from, and `weightsRoot`, the directory the lock file's file names
are relative to. They are worth reading rather than guessing, because the
paths are not all the same shape:

| model | `weightsUrl` | `weightsRoot` |
|---|---|---|
| `basic-pitch` | `/model/` — a TensorFlow.js model, not an ONNX one | `/model/` |
| `okarina` | `/model-onnx/` — its files sit at the root of the tree | `/model-onnx/` |
| `piano` | `/model-onnx/piano/` | `/model-onnx/` |
| `tsumugi` | `/model-onnx/tsumugi/` | `/model-onnx/` |
| `yourmt3` | `/model-onnx/yourmt3/` | `/model-onnx/` |

The address of a weight file is `weightsRoot` and the file's name in the
lock file, joined. The piano model's graph is `piano/piano_kong.onnx` in
`piano.lock.json`, and its address is `/model-onnx/piano/piano_kong.onnx`.

### `GET /api/parameters`

The parameters a conversion takes, with their defaults and the command line
flag that sets each one. The names are the ones the site uses, so a value
set in one means the same thing in the others.

```bash
curl http://192.168.100.109/api/parameters
```

## 3. The parameters

| Name | Flag | Default | What it does |
|---|---|---|---|
| `model` | `--model` | `piano` | Which model does the work. |
| `okarinaSize` | `--okarina-size` | `m` | Which Okarina model. Only for Okarina. |
| `drums` | `--no-drums` | `true` | Look for a drum kit. |
| `tempo` | `--tempo` | estimated | Force a tempo, in BPM. |
| `rangeLow` | `--range-low` | `0` | Lowest frequency to keep, in Hz. |
| `rangeHigh` | `--range-high` | no limit | Highest frequency to keep, in Hz. |
| `grid` | `--grid` | `0` | Slots per second. 0 leaves the notes alone. |
| `gridPhase` | `--grid-phase` | `0` | Where the first slot sits, in ms. |
| `pianify` | `--pianify` | `false` | Arrange the notes for two hands. |
| `format` | `--json` | `midi` | midi or json. |

## 4. Errors

Every error comes back as the same JSON shape, with a machine-readable code
and an HTTP status that matches what went wrong. A client should be able
to branch on the code without reading the message.

```json
{
  "error": {
    "code": "gone",
    "message": "This server does not take recordings. ..."
  }
}
```

| Code | Status | What it means |
|---|---|---|
| `not_found` | 404 | Nothing at that address. |
| `method_not_allowed` | 405 | That address exists, but not for that method. |
| `gone` | 410 | The endpoint that took recordings is gone. |
| `internal_error` | 500 | Something went wrong that the caller cannot fix. |

`POST /api/convert` answers `410`, not `404`. The endpoint used to be
there, and somebody with an old script deserves to be told what happened
rather than a bare "nothing here".

## 5. What it runs on, and what it needs

The API has **no dependencies**. It is bundled into one file, the box
installs nothing, and the service asks for 256 MB of memory it will never
use.

That is a consequence of the rule above. The server used to convert, which
meant it needed the ONNX runtime, TensorFlow.js and an audio decoder on
the box, and it meant the box's memory decided which models it could serve:

| model | Peak memory, one model per process |
|---|---|
| Okarina | 460 MB |
| Piano Transcription | 661 MB |
| YourMT3+ | 1056 MB |
| Tsumugi | 1183 MB |

On a container with a gigabyte, Tsumugi and YourMT3+ did not fit and the
API could not serve them at all. Now that the box only holds files, every
model is available to every client, and the box's memory does not enter
into it.

The weights themselves are still on the box — they are 367 MB — because
that is where the browser has always fetched them from, and a client on the
network fetches them from the same place.

### The settings

| | |
|---|---|
| `VOIDTRACKR_HOST` | `127.0.0.1` — nginx is the only thing that should reach it |
| `VOIDTRACKR_PORT` | `8787` |

There is no token, no upload limit and no concurrency setting, because
there is nothing to upload and nothing to run. The API is read-only and
holds nothing in memory.

## 6. How it runs

```
nginx  ──▶  /            the static site
     │  ──▶  /model/     Basic Pitch's weights
     │  ──▶  /model-onnx/  the ONNX weights
     │  ──▶  /api/       the API ──▶ 127.0.0.1:8787
```

The API listens on the loopback address only. nginx is the only thing that
can reach it, and it is the only thing that should.

## 7. The same file, whichever way it is asked for

The point of the arrangement is that a recording converted in the browser
and the same recording converted on the command line come out as the same
bytes. The browser and the command line share the conversion code in
`src/core/convert.ts`; there is no second implementation to drift.

Taking the model's weights from the server does not change that either.
The command line with `--server` and the command line without it produce
byte-identical files, and there is a test that runs both and compares them.

## 8. The measurements

- The API answers in a few milliseconds. It does no work.
- The weights are 367 MB in total: Basic Pitch 0.9 MB, piano 96 MB,
  Tsumugi 74 MB, YourMT3+ 181 MB, Okarina 21 MB.
- A client that fetches a model caches it, so the second run of a model
  downloads nothing.
- The browser fetches the same weights from the same place and converts
  locally. It never asks this API for anything — it does not need to.
