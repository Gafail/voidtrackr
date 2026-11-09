/**
 * Transcribe a folder of recordings with every model and report how they did.
 *
 *   npx tsx scripts/evaluate-folder.mts "Music Import" --out eval
 *
 * There is no ground truth for this kind of music, so the report does not
 * pretend to give a score. It gives the things that can be measured without
 * one:
 *
 *   - how many notes each model found;
 *   - how well the onsets line up with the beat grid of the recording,
 *     which comes from the audio itself, so every model is measured against
 *     the same ruler;
 *   - how much the models agree with each other;
 *   - and the MIDI files themselves, so the result can be listened to.
 *
 * The report is written as a self-contained HTML page.
 */

import nodeUtil from 'node:util';

const mutableUtil = nodeUtil as typeof nodeUtil & { isNullOrUndefined?: (value: unknown) => boolean };
if (typeof mutableUtil.isNullOrUndefined !== 'function') {
  mutableUtil.isNullOrUndefined = (value: unknown) => value === null || value === undefined;
}
await import('@tensorflow/tfjs-node');

import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { basename, extname, join, relative, resolve } from 'node:path';

import { decodeAudio, prepareMono } from '../src/core/audio';
import { nodeAudioDecoder } from '../src/core/audio-node';
import { loadPinnedModel } from '../src/core/basic-pitch';
import { ModelLock, ModelSource, TranscriptionModel } from '../src/core/model';
import modelLock from '../src/core/model.lock.json';
import { OkarinaModel, OkarinaSize } from '../src/core/okarina';
import okarinaLock from '../src/core/okarina.lock.json';
import { nodeOnnxRunner } from '../src/core/onnx-node';
import { verifyOnnxWeights } from '../src/core/onnx-weights';
import { PianoTranscriptionModel, PIANO_DIR, PIANO_FILES } from '../src/core/piano';
import pianoLock from '../src/core/piano.lock.json';
import { TsumugiModel, TSUMUGI_DIR, TSUMUGI_FILES } from '../src/core/tsumugi';
import tsumugiLock from '../src/core/tsumugi.lock.json';
import { YourMt3Model, YOURMT3_DIR, YOURMT3_FILES } from '../src/core/yourmt3';
import yourmt3Lock from '../src/core/yourmt3.lock.json';
import { transcribe } from '../src/core/pipeline';
import { buildMidiFile } from '../src/core/midi';
import { BeatGrid, distanceToGrid, estimateBeatGrid } from '../src/core/beat';
import { RawNote, Track } from '../src/core/types';
import { pathToFileURL } from 'node:url';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';

const MODEL_DIR = resolve('models', 'onnx');
const MODEL_PATH = resolve('public/model/model.json');

/** How close an onset has to be to the grid to count as on it. */
const GRID_TOLERANCE = 0.03;

/** How close two notes have to be to count as the same note. */
const MATCH_TOLERANCE = 0.05;

interface Options {
  folder: string;
  out: string;
  models: string[];
  okarinaSize: OkarinaSize;
  /** Rebuild the page from the saved results, without transcribing again. */
  reportOnly: boolean;
  /** Print the aggregate from the saved results, without transcribing again. */
  summaryOnly: boolean;
}

function parseArgs(argv: string[]): Options {
  const options: Options = {
    folder: '',
    out: 'eval',
    models: ['basic-pitch', 'tsumugi', 'piano', 'okarina', 'yourmt3'],
    okarinaSize: 'm',
    reportOnly: false,
    summaryOnly: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--out') options.out = argv[++i];
    else if (arg === '--models') options.models = argv[++i].split(',');
    else if (arg === '--okarina-size') options.okarinaSize = argv[++i] as OkarinaSize;
    else if (arg === '--report-only') options.reportOnly = true;
    else if (arg === '--summary') options.summaryOnly = true;
    else if (!arg.startsWith('-')) options.folder = arg;
  }
  if (!options.folder && !options.reportOnly && !options.summaryOnly) {
    console.error(
      'Usage: tsx scripts/evaluate-folder.mts <folder> [--out dir] [--models a,b]\n' +
        '       tsx scripts/evaluate-folder.mts --out dir --summary\n' +
        '       tsx scripts/evaluate-folder.mts --out dir --report-only',
    );
    process.exit(1);
  }
  return options;
}

/** One song's worth of results. */
interface SongResult {
  name: string;
  file: string;
  /** A link to the original recording, relative to the report. */
  audio: string;
  duration: number;
  grid: BeatGrid;
  models: Record<string, ModelResult>;
}

interface ModelResult {
  notes: number;
  density: number;
  pitches: number;
  medianPitch: number;
  meanDuration: number;
  /** Fraction of the recording covered by at least one note. */
  coverage: number;
  /** Fraction of onsets within the tolerance of a beat. */
  onBeat: number;
  /** Fraction of onsets within the tolerance of an eighth note. */
  onEighth: number;
  /** Mean distance from an onset to the nearest beat, in milliseconds. */
  beatError: number;
  /** The MIDI file that was written. */
  midi: string;
  /** Notes, for the report's piano roll and playback. */
  noteData: Array<[number, number, number]>;
  tracks: { name: string; notes: number }[];
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));

  // Rebuilding the page from the saved results, without transcribing again.
  if (options.reportOnly || options.summaryOnly) {
    const saved = JSON.parse(await readFile(join(options.out, 'report.json'), 'utf8')) as {
      results: SongResult[];
    };
    // The agreement is recomputed, so the page always reflects the notes
    // that are actually in the results.
    if (options.summaryOnly) {
      printSummary(saved.results);
      return;
    }
    await writeFile(join(options.out, 'index.html'), buildReport(saved.results));
    console.log(`report: ${resolve(options.out, 'index.html')}`);
    return;
  }

  const songs = await findSongs(options.folder);

  console.log(`${songs.length} songs in ${options.folder}`);
  console.log(`models: ${options.models.join(', ')}\n`);

  const models = await loadModels(options);

  await mkdir(join(options.out, 'midi'), { recursive: true });

  const results: SongResult[] = [];

  for (const song of songs) {
    const data = await readFile(song.file);
    const audio = await decodeAudio(
      data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
      nodeAudioDecoder,
    );
    const samples = prepareMono(audio);

    // The beat grid comes from the audio, so it is the same ruler for every
    // model.
    const grid = estimateBeatGrid(samples, 22050);

    const songResult: SongResult = {
      name: song.name,
      file: song.file,
      audio: relative(options.out, song.audio),
      duration: audio.duration,
      grid,
      models: {},
    };

    for (const [id, model] of Object.entries(models)) {
      const result = await transcribe(samples, { model, detectDrums: true });
      const notes = result.tracks.flatMap((track) => track.notes);
      const bytes = buildMidiFile(result.tracks, result.tempo, result.key);
      const midiName = `${song.slug}.${id}.mid`;
      await writeFile(join(options.out, 'midi', midiName), Buffer.from(bytes));

      songResult.models[id] = measure(result.tracks, notes, audio.duration, grid, midiName);
    }

    results.push(songResult);

    const summary = Object.entries(songResult.models)
      .map(([id, m]) => `${id}: ${m.notes} notes, ${(m.onBeat * 100).toFixed(0)}% on beat`)
      .join('   ');
    console.log(
      `${song.name.padEnd(30)} ${grid.bpm.toFixed(1).padStart(6)} BPM  ${summary}`,
    );

    // The results are written after every song, not just at the end.
    //
    // A folder of recordings takes hours to go through, and the run can be
    // interrupted. If the results only appeared at the very end, an
    // interrupted run would have nothing to show. The page is rebuilt from
    // this file with `--report-only`, so nothing is lost.
    await writeFile(
      join(options.out, 'report.json'),
      JSON.stringify(
        { generated: new Date().toISOString(), results, agreements: computeAgreements(results) },
        null,
        2,
      ),
    );
  }

  // Cross-model agreement, in both directions.
  const agreements = computeAgreements(results);

  await writeFile(
    join(options.out, 'report.json'),
    JSON.stringify({ generated: new Date().toISOString(), results, agreements }, null, 2),
  );
  await writeFile(join(options.out, 'index.html'), buildReport(results, agreements));

  console.log('\n--- summary ---');
  printSummary(results);
  console.log(`\nreport: ${resolve(options.out, 'index.html')}`);
}

/**
 * Print the aggregate over every song.
 *
 * The same numbers are printed whether the run has just finished or the
 * results were read back off disk, so a run that was interrupted can still
 * be reported on.
 */
function printSummary(results: SongResult[]): void {
  const ids = Object.keys(results[0]?.models ?? {});
  const agreements = computeAgreements(results);

  for (const id of ids) {
    const all = results.map((r) => r.models[id]);
    const avg = (pick: (m: ModelResult) => number) =>
      all.reduce((sum, m) => sum + pick(m), 0) / all.length;
    console.log(
      `${id.padEnd(14)} ${avg((m) => m.notes).toFixed(0).padStart(6)} notes   ` +
        `${(avg((m) => m.onBeat) * 100).toFixed(1).padStart(5)}% on beat   ` +
        `${(avg((m) => m.onEighth) * 100).toFixed(1).padStart(5)}% on 8th   ` +
        `${avg((m) => m.beatError).toFixed(1).padStart(5)} ms beat error`,
    );
  }
  for (const [key, value] of Object.entries(agreements)) {
    console.log(`agreement ${key.padEnd(28)} ${(value * 100).toFixed(1)}%`);
  }
}

/** Find one file per song, preferring the lossless one. */
async function findSongs(
  folder: string,
): Promise<Array<{ name: string; slug: string; file: string; audio: string }>> {
  const files = await readdir(folder);
  const bySong = new Map<string, Record<string, string>>();

  for (const file of files) {
    const ext = extname(file).toLowerCase();
    if (!['.mp3', '.ogg', '.wav'].includes(ext)) continue;
    const stem = basename(file, ext).replace(/\.(mp3|ogg|wav)$/i, '');
    const format = ext.slice(1);
    bySong.set(stem, { ...(bySong.get(stem) ?? {}), [format]: join(folder, file) });
  }

  const songs: Array<{ name: string; slug: string; file: string; audio: string }> = [];
  for (const [stem, formats] of [...bySong].sort()) {
    const file = formats.wav ?? formats.mp3 ?? formats.ogg;
    if (!file) continue;
    songs.push({
      name: stem,
      slug: stem.replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-|-$/g, ''),
      file,
      // The mp3 is a fraction of the size of the wav, which matters when
      // the browser has to load it.
      audio: formats.mp3 ?? formats.ogg ?? formats.wav ?? file,
    });
  }
  return songs;
}

async function loadModels(options: Options): Promise<Record<string, TranscriptionModel>> {
  const models: Record<string, TranscriptionModel> = {};

  if (options.models.includes('basic-pitch')) {
    models['basic-pitch'] = await loadPinnedModel(nodeModelSource(MODEL_PATH), modelLock as ModelLock);
  }

  if (options.models.includes('okarina')) {
    const lock = okarinaLock as unknown as { files: Record<string, { sha256: string }> };
    for (const name of ['cqt156.onnx', `okarina_${options.okarinaSize}.onnx`]) {
      const bytes = await readFile(join(MODEL_DIR, name));
      const digest = createHash('sha256').update(bytes).digest('hex');
      if (digest !== lock.files[name]?.sha256) {
        throw new Error(`${name} does not match the pinned version.`);
      }
    }
    models['okarina'] = await OkarinaModel.load(
      nodeOnnxRunner,
      MODEL_DIR,
      options.okarinaSize,
    );
  }

  if (options.models.includes('tsumugi')) {
    const fingerprint = await checkOnnx(
      tsumugiLock,
      TSUMUGI_DIR,
      [TSUMUGI_FILES.graph, TSUMUGI_FILES.heads, TSUMUGI_FILES.headsJson],
      'npm run model:tsumugi',
    );
    models['tsumugi'] = await TsumugiModel.load(
      nodeOnnxRunner,
      join(MODEL_DIR, TSUMUGI_DIR),
      fingerprint,
    );
  }

  if (options.models.includes('piano')) {
    const fingerprint = await checkOnnx(
      pianoLock,
      PIANO_DIR,
      [PIANO_FILES.graph],
      'npm run model:piano',
    );
    models['piano'] = await PianoTranscriptionModel.load(
      nodeOnnxRunner,
      join(MODEL_DIR, PIANO_DIR),
      fingerprint,
    );
  }

  if (options.models.includes('yourmt3')) {
    const fingerprint = await checkOnnx(
      yourmt3Lock,
      YOURMT3_DIR,
      [YOURMT3_FILES.encoder, YOURMT3_FILES.decoder, YOURMT3_FILES.vocab],
      'npm run model:yourmt3',
    );
    models['yourmt3'] = await YourMt3Model.load(
      nodeOnnxRunner,
      join(MODEL_DIR, YOURMT3_DIR),
      fingerprint,
    );
  }

  return models;
}

/** Check a model's weights against its lock file and return the fingerprint. */
async function checkOnnx(
  lock: unknown,
  directory: string,
  files: string[],
  fetchCommand: string,
): Promise<string> {
  return verifyOnnxWeights(
    async (path: string) => {
      const bytes = await readFile(join(MODEL_DIR, path));
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ) as ArrayBuffer;
    },
    lock as { files: Record<string, { sha256: string }> },
    { dir: directory, files },
    fetchCommand,
  );
}

function nodeModelSource(modelPath: string): ModelSource {
  const directory = dirname(modelPath);
  return {
    url: pathToFileURL(modelPath).href,
    read: async (name: string) => {
      const bytes = await readFile(join(directory, name));
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ) as ArrayBuffer;
    },
  };
}

/** Work out the numbers for one model's output on one song. */
function measure(
  tracks: Track[],
  notes: RawNote[],
  duration: number,
  grid: BeatGrid,
  midiName: string,
): ModelResult {
  const pitches = notes.map((n) => n.pitchMidi).sort((a, b) => a - b);
  const onBeat = countOnGrid(notes.map((n) => n.startTime), grid, 1);
  const onEighth = countOnGrid(notes.map((n) => n.startTime), grid, 2);

  let beatError = 0;
  for (const note of notes) {
    beatError += distanceToGrid(note.startTime, grid) * 1000;
  }

  const covered = new Float32Array(Math.ceil(duration * 100));
  for (const note of notes) {
    const from = Math.max(0, Math.floor(note.startTime * 100));
    const to = Math.min(covered.length, Math.ceil((note.startTime + note.duration) * 100));
    for (let i = from; i < to; i++) covered[i] = 1;
  }
  let coveredCount = 0;
  for (const value of covered) coveredCount += value;

  return {
    notes: notes.length,
    density: notes.length / Math.max(1, duration),
    pitches: new Set(pitches).size,
    medianPitch: pitches.length ? pitches[Math.floor(pitches.length / 2)] : 0,
    meanDuration: notes.length
      ? notes.reduce((sum, n) => sum + n.duration, 0) / notes.length
      : 0,
    coverage: coveredCount / covered.length,
    onBeat: onBeat,
    onEighth: onEighth,
    beatError: notes.length ? beatError / notes.length : 0,
    midi: `midi/${midiName}`,
    noteData: notes.map((n) => [
      Math.round(n.startTime * 1000),
      Math.round(n.duration * 1000),
      n.pitchMidi,
    ]),
    tracks: tracks.map((t) => ({ name: t.name, notes: t.notes.length })),
  };
}

/** Fraction of times within the tolerance of a grid position. */
function countOnGrid(times: number[], grid: BeatGrid, division: number): number {
  if (times.length === 0) return 0;
  const scaled: BeatGrid = { ...grid, period: grid.period / division };
  let hits = 0;
  for (const time of times) {
    if (distanceToGrid(time, scaled) <= GRID_TOLERANCE) hits += 1;
  }
  return hits / times.length;
}

/**
 * What fraction of one model's notes have a note of the same pitch at the
 * same time in the other model's output.
 */
function agreement(
  left: Array<[number, number, number]>,
  right: Array<[number, number, number]>,
): number {
  if (left.length === 0 || right.length === 0) return 0;

  const byPitch = new Map<number, number[]>();
  for (const [start, , pitch] of right) {
    if (!byPitch.has(pitch)) byPitch.set(pitch, []);
    byPitch.get(pitch)!.push(start / 1000);
  }
  for (const times of byPitch.values()) times.sort((a, b) => a - b);

  let matched = 0;
  for (const [start, , pitch] of left) {
    const times = byPitch.get(pitch);
    if (!times) continue;
    const t = start / 1000;
    if (times.some((candidate) => Math.abs(candidate - t) <= MATCH_TOLERANCE)) matched += 1;
  }
  return matched / left.length;
}

/**
 * How much the models agree, in both directions.
 *
 * The direction matters. If one model finds many more notes than the other
 * and the extra ones are not matched, they are probably wrong:
 * over-transcription shows up as notes nobody else agrees with.
 */
function computeAgreements(results: SongResult[]): Record<string, number> {
  const out: Record<string, number> = {};
  const ids = Object.keys(results[0]?.models ?? {});

  for (let a = 0; a < ids.length; a++) {
    for (let b = a + 1; b < ids.length; b++) {
      for (const [key, leftId, rightId] of [
        [`${ids[a]} notes also found by ${ids[b]}`, ids[a], ids[b]],
        [`${ids[b]} notes also found by ${ids[a]}`, ids[b], ids[a]],
      ]) {
        let total = 0;
        let matched = 0;
        for (const song of results) {
          const left = song.models[leftId];
          const right = song.models[rightId];
          if (!left || !right) continue;
          total += agreement(left.noteData, right.noteData);
          matched += 1;
        }
        out[key] = matched > 0 ? total / matched : 0;
      }
    }
  }

  return out;
}

/** Build the HTML report. */
function buildReport(
  results: SongResult[],
  agreements?: Record<string, number>,
): string {
  const ids = Object.keys(results[0]?.models ?? []);
  const agree = agreements ?? computeAgreements(results);
  const avg = (id: string, pick: (m: ModelResult) => number) => {
    const all = results.map((r) => r.models[id]).filter(Boolean);
    return all.length ? all.reduce((sum, m) => sum + pick(m), 0) / all.length : 0;
  };

  const summaryRows = ids
    .map(
      (id) => `<tr>
        <td class="name">${id}</td>
        <td>${avg(id, (m) => m.notes).toFixed(0)}</td>
        <td>${avg(id, (m) => m.density).toFixed(2)}</td>
        <td>${avg(id, (m) => m.pitches).toFixed(0)}</td>
        <td>${avg(id, (m) => m.medianPitch).toFixed(0)}</td>
        <td>${avg(id, (m) => m.meanDuration).toFixed(2)}</td>
        <td>${(avg(id, (m) => m.onBeat) * 100).toFixed(1)}%</td>
        <td>${(avg(id, (m) => m.onEighth) * 100).toFixed(1)}%</td>
        <td>${avg(id, (m) => m.beatError).toFixed(1)} ms</td>
      </tr>`,
    )
    .join('');

  const songRows = results
    .map((song) => {
      const cells = ids
        .map((id) => {
          const m = song.models[id];
          return `<td>
            <a href="${m.midi}">${m.notes} notes</a>
            <span class="dim">${(m.onBeat * 100).toFixed(0)}% on beat</span>
          </td>`;
        })
        .join('');
      return `<tr>
        <td class="name">${song.name}</td>
        <td>${song.duration.toFixed(0)}s</td>
        <td>${song.grid.bpm.toFixed(1)}</td>
        <td>${song.grid.confidence.toFixed(2)}</td>
        ${cells}
      </tr>`;
    })
    .join('');

  const rolls = results
    .map((song) => {
      const canvases: string = ids
        .map(
          (id) => `<div class="roll">
            <div class="rollhead">
              <span>${id}</span>
              <button class="play" data-notes='${JSON.stringify(song.models[id].noteData)}'>Play</button>
              <a href="${song.models[id].midi}">MIDI</a>
            </div>
            <canvas data-notes='${JSON.stringify(song.models[id].noteData)}' width="900" height="150"></canvas>
            <canvas class="detail" data-notes='${JSON.stringify(song.models[id].noteData)}' data-span="15" width="900" height="150"></canvas>
          </div>`,
        )
        .join('');
      return `<section>
        <h2>${song.name}</h2>
        <p class="dim">${song.duration.toFixed(1)}s &middot; ${song.grid.bpm.toFixed(1)} BPM (confidence ${song.grid.confidence.toFixed(2)}) &middot;
          <a href="${song.audio}">original recording</a></p>
        ${canvases}
      </section>`;
    })
    .join('');

  const agreementRows = Object.entries(agree)
    .map(([key, value]) => `<tr><td class="name">${key}</td><td>${(value * 100).toFixed(1)}%</td></tr>`)
    .join('');

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Transcription report</title>
<style>
  :root { color-scheme: dark; }
  body { font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 24px 32px 64px; background: #12131a; color: #e6e7ef; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  h2 { font-size: 15px; margin: 32px 0 4px; font-weight: 600; }
  p.dim, .dim { color: #8b8fa3; font-size: 12px; }
  table { border-collapse: collapse; margin: 12px 0 24px; font-size: 13px; width: 100%; max-width: 1100px; }
  th, td { border-bottom: 1px solid #262936; padding: 6px 12px 6px 0; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #8b8fa3; font-weight: 500; font-size: 12px; }
  td.name { color: #cfd2e0; }
  a { color: #7fd1ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  td span.dim { display: block; }
  .roll { margin: 8px 0 16px; }
  .rollhead { font-size: 12px; color: #8b8fa3; margin-bottom: 2px; display: flex; gap: 10px; align-items: center; }
  .rollhead span { min-width: 90px; }
  button.play { background: #1f2536; color: #cfd2e0; border: 1px solid #39405a; border-radius: 5px; padding: 2px 10px; font: inherit; font-size: 12px; cursor: pointer; }
  button.play:hover { background: #2a3247; }
  canvas { width: 900px; max-width: 100%; background: #171923; border: 1px solid #262936; border-radius: 4px; display: block; }
  .note { color: #8b8fa3; max-width: 900px; }
</style>
</head>
<body>
<h1>Transcription report</h1>
<p class="note">
  Every model was run over the same recordings. The beat grid comes from the audio itself,
  so the "on beat" column measures the same thing for every model: how many of its onsets
  land on the pulse of the music. Nothing here is a correctness score — listen to the MIDI
  files to judge.
</p>

<h2>Summary</h2>
<table>
  <tr><th>model</th><th>notes</th><th>notes/s</th><th>pitches</th><th>median pitch</th><th>mean length</th><th>on beat</th><th>on 8th</th><th>beat error</th></tr>
  ${summaryRows}
</table>

${
  agreementRows
    ? `<h2>How much the models agree</h2>
<table>
  <tr><th>pair</th><th>notes matched</th></tr>
  ${agreementRows}
</table>`
    : ''
}

<h2>Every song</h2>
<table>
  <tr><th>song</th><th>length</th><th>BPM</th><th>grid confidence</th>${ids.map((id) => `<th>${id}</th>`).join('')}</tr>
  ${songRows}
</table>

<h2>Piano rolls</h2>
${rolls}

<script>
(function () {
  var AUDIO = new (window.AudioContext || window.webkitAudioContext)();

  function draw(canvas) {
    var notes = JSON.parse(canvas.dataset.notes);
    if (!notes.length) return;
    var ctx = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // A detail canvas shows the first few seconds at full width; the wide
    // one shows the whole song.
    var span = canvas.dataset.span ? Number(canvas.dataset.span) * 1000 : 0;

    var lo = 127, hi = 0, end = 0;
    for (var i = 0; i < notes.length; i++) {
      lo = Math.min(lo, notes[i][2]); hi = Math.max(hi, notes[i][2]);
      end = Math.max(end, notes[i][0] + notes[i][1]);
    }
    if (span) end = span;

    var rows = Math.max(1, hi - lo + 1);
    var rowH = h / rows;
    var scale = w / Math.max(1, end);

    for (var j = 0; j < notes.length; j++) {
      var x = notes[j][0] * scale;
      if (span && notes[j][0] > span) continue;
      var len = Math.max(1, notes[j][1] * scale);
      var y = h - (notes[j][2] - lo + 1) * rowH;
      ctx.fillStyle = notes[j][2] < 55 ? '#6f7ea8' : '#7fd1ff';
      ctx.fillRect(x, y + 0.5, len, Math.max(1, rowH - 1));
    }
  }

  document.querySelectorAll('canvas').forEach(draw);

  // Play the notes, so the models can be compared by ear.
  //
  // A plain decaying sine with a second harmonic. It is not a piano, but it
  // is enough to hear whether the notes are in the right place and whether
  // there are too many of them.
  function play(notes) {
    if (!notes.length) return;
    var now = AUDIO.currentTime + 0.05;
    var master = AUDIO.createGain();
    master.gain.value = 0.25;
    master.connect(AUDIO.destination);

    for (var i = 0; i < notes.length; i++) {
      var start = notes[i][0] / 1000;
      var length = Math.max(0.04, notes[i][1] / 1000);
      var freq = 440 * Math.pow(2, (notes[i][2] - 69) / 12);

      var osc = AUDIO.createOscillator();
      osc.type = 'triangle';
      osc.frequency.value = freq;

      var gain = AUDIO.createGain();
      gain.gain.setValueAtTime(0, now + start);
      gain.gain.linearRampToValueAtTime(0.6, now + start + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0008, now + start + length + 0.18);

      osc.connect(gain);
      gain.connect(master);
      osc.start(now + start);
      osc.stop(now + start + length + 0.25);
    }
  }

  document.querySelectorAll('button.play').forEach(function (button) {
    button.addEventListener('click', function () {
      if (AUDIO.state === 'suspended') AUDIO.resume();
      play(JSON.parse(button.dataset.notes));
    });
  });
})();
</script>
</body>
</html>`;
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
