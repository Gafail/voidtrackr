/**
 * Command line converter.
 *
 *   npm run cli -- song.mp3 -o out/
 *   npm run cli -- songs/*.wav --no-drums --json
 *   npm run cli -- song.mp3 -o out/ --model okarina
 *
 * Uses the native TensorFlow.js backend so it is fast enough to run over a
 * folder of recordings.
 */

import nodeUtil from 'node:util';

/**
 * Work around a packaging bug in `@tensorflow/tfjs-node`.
 *
 * Its published build does `require("util")` where it meant to require
 * TensorFlow.js's own utilities, then calls `util.isNullOrUndefined(...)`.
 * Node's `util` module has no such function, so the native backend throws
 * the moment it reads a tensor.
 *
 * `require("util")` hands back a singleton, so defining the missing helper
 * on it here is enough to make the backend work. This has to happen before
 * the backend is used, but not before it is imported, so the import below
 * is dynamic.
 */
const mutableUtil = nodeUtil as typeof nodeUtil & { isNullOrUndefined?: (value: unknown) => boolean };
if (typeof mutableUtil.isNullOrUndefined !== 'function') {
  mutableUtil.isNullOrUndefined = (value: unknown) => value === null || value === undefined;
}

await import('@tensorflow/tfjs-node');

import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { decodeAudio, prepareMono } from '../core/audio';
import { nodeAudioDecoder } from '../core/audio-node';
import { loadPinnedModel } from '../core/basic-pitch';
import { ModelLock, ModelSource, TranscriptionModel } from '../core/model';
import modelLock from '../core/model.lock.json';
import { OkarinaModel, OkarinaSize, OKARINA_FILES } from '../core/okarina';
import okarinaLock from '../core/okarina.lock.json';
import { nodeOnnxRunner } from '../core/onnx-node';
import { verifyOnnxWeights } from '../core/onnx-weights';
import { PianoTranscriptionModel, PIANO_DIR, PIANO_FILES } from '../core/piano';
import pianoLock from '../core/piano.lock.json';
import { FrequencyRange } from '../core/range';
import { QuantiseOptions } from '../core/quantise';
import { PianifyOptions } from '../core/pianify';
import { TsumugiModel, TSUMUGI_DIR, TSUMUGI_FILES } from '../core/tsumugi';
import tsumugiLock from '../core/tsumugi.lock.json';
import { YourMt3Model, YOURMT3_DIR, YOURMT3_FILES } from '../core/yourmt3';
import yourmt3Lock from '../core/yourmt3.lock.json';
import { transcribe } from '../core/pipeline';
import { buildMidiFile } from '../core/midi';
import { keyName } from '../core/tempo';
import { DEFAULT_MODEL, listModels, ModelId, MODELS } from '../core/registry';
import { provenanceFor } from '../core/version';
import { TranscriptionResult } from '../core/types';

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL_PATH = resolve(HERE, '..', '..', 'public', 'model', 'model.json');
const DEFAULT_ONNX_DIR = resolve(HERE, '..', '..', 'models', 'onnx');

interface CliOptions {
  inputs: string[];
  outputDir: string;
  /** Which model to use. */
  model: ModelId;
  /** Path to a Basic Pitch model.json. */
  modelPath: string;
  /** Directory holding the ONNX weights. */
  modelDir: string;
  /** Which Okarina model to use. */
  okarinaSize: OkarinaSize;
  drums: boolean;
  tempo?: number;
  /** Keep only the notes inside this frequency range. */
  range?: FrequencyRange;
  /** Snap the notes to a grid of this many slots per second. */
  quantise?: QuantiseOptions;
  /** Arrange the notes for two hands. */
  pianify?: PianifyOptions;
  json: boolean;
  quiet: boolean;
}

/**
 * Read a frequency range from the command line.
 *
 * `--range 400` means "everything below 400 Hz", and `--range 100-1200`
 * means "between 100 Hz and 1200 Hz". The bounds are the notes' own
 * fundamentals, so 400 Hz is G4.
 */
function parseRange(value: string): FrequencyRange {
  const parts = value.split('-').map((part) => part.trim());
  if (parts.length === 1) {
    const high = Number(parts[0]);
    if (!Number.isFinite(high)) throw new Error(`Could not read the range "${value}".`);
    return { low: 0, high };
  }
  if (parts.length !== 2) {
    throw new Error(`Could not read the range "${value}". Use LOW-HIGH, in Hz.`);
  }
  const low = Number(parts[0]);
  const high = Number(parts[1]);
  if (!Number.isFinite(low) || !Number.isFinite(high)) {
    throw new Error(`Could not read the range "${value}". Use LOW-HIGH, in Hz.`);
  }
  return { low, high };
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    inputs: [],
    outputDir: '.',
    model: DEFAULT_MODEL,
    modelPath: DEFAULT_MODEL_PATH,
    modelDir: DEFAULT_ONNX_DIR,
    okarinaSize: 'm',
    drums: true,
    json: false,
    quiet: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case '-o':
      case '--out':
        options.outputDir = argv[++i] ?? options.outputDir;
        break;
      case '--model': {
        const value = argv[++i] ?? '';
        if (!(value in MODELS)) {
          console.error(`Unknown model: ${value}`);
          console.error(`Available models: ${Object.keys(MODELS).join(', ')}`);
          process.exit(1);
        }
        options.model = value as ModelId;
        break;
      }
      case '--model-path':
        options.modelPath = argv[++i] ?? options.modelPath;
        break;
      case '--model-dir':
        options.modelDir = argv[++i] ?? options.modelDir;
        break;
      case '--okarina-size':
        options.okarinaSize = (argv[++i] as OkarinaSize) ?? options.okarinaSize;
        break;
      case '--no-drums':
        options.drums = false;
        break;
      case '--tempo':
        options.tempo = Number(argv[++i]);
        break;
      case '--range': {
        const value = argv[++i] ?? '';
        try {
          options.range = parseRange(value);
        } catch (error) {
          console.error((error as Error).message);
          process.exit(1);
        }
        break;
      }
      case '--quantize':
      case '--quantise': {
        const value = Number(argv[++i]);
        if (!Number.isFinite(value) || value <= 0) {
          console.error(
            `Could not read the grid "${argv[i]}". Use the number of slots per second, such as 30.`,
          );
          process.exit(1);
        }
        options.quantise = { grid: value, phase: 0 };
        break;
      }
      case '--pianify': {
        options.pianify = {};
        break;
      }
      case '--no-pianify': {
        options.pianify = undefined;
        break;
      }
      case '--phase': {
        const value = Number(argv[++i]);
        if (!Number.isFinite(value)) {
          console.error(`Could not read the phase "${argv[i]}". Use milliseconds.`);
          process.exit(1);
        }
        options.quantise = { grid: options.quantise?.grid ?? 30, phase: value / 1000 };
        break;
      }
      case '--json':
        options.json = true;
        break;
      case '-q':
      case '--quiet':
        options.quiet = true;
        break;
      case '-l':
      case '--list-models':
        for (const model of listModels()) {
          console.log(
            `${model.id.padEnd(12)} ${model.displayName.padEnd(12)} ${model.license.padEnd(32)} ${model.summary}`,
          );
        }
        process.exit(0);
        break;
      case '-h':
      case '--help':
        printHelp();
        process.exit(0);
        break;
      default:
        if (arg.startsWith('-')) {
          console.error(`Unknown option: ${arg}`);
          printHelp();
          process.exit(1);
        }
        options.inputs.push(arg);
    }
  }

  return options;
}

function printHelp(): void {
  console.log(`
voidtrackr — audio to MIDI, locally

  npm run cli -- <input...> [options]

Options:
  -o, --out <dir>           Directory for the .mid files (default: current directory)
      --model <name>        Which model to use: ${Object.keys(MODELS).join(' or ')} (default: ${DEFAULT_MODEL})
      --model-path <path>   Path to a Basic Pitch model.json
      --model-dir <dir>     Directory holding the ONNX weights
      --okarina-size <s|m|l> Which Okarina model to use (default m)
      --no-drums            Skip drum detection
      --tempo <bpm>         Force a tempo instead of estimating one
      --range <low>-<high>  Keep only the notes inside this frequency range, in Hz.
                            --range 400 means everything below 400 Hz.
      --quantize <hz>        Snap the notes to a grid of this many slots per second.
                            --quantize 30 lines the notes up with a 30 FPS screen.
      --phase <ms>           Where the first slot sits, in milliseconds.
      --pianify              Arrange the notes for two hands: fold the re-triggers
                            together, divide the chords between the hands, and drop
                            what neither hand could hold.
      --json                Print results as JSON
  -l, --list-models         List the models and their licences
  -q, --quiet               Only print errors
`);
}

/**
 * Load the model the user asked for.
 *
 * Every model is checked against its lock file before it runs. If the
 * weights do not match, the output would differ from every previous run, so
 * it is better to stop than to produce a different file quietly.
 */
async function loadModel(options: CliOptions): Promise<TranscriptionModel> {
  if (options.model === 'okarina') {
    const fingerprint = await verifyOkarinaWeights(options.modelDir, options.okarinaSize);
    return OkarinaModel.load(
      nodeOnnxRunner,
      options.modelDir,
      options.okarinaSize,
      fingerprint,
    );
  }

  if (options.model === 'tsumugi') {
    const directory = join(options.modelDir, TSUMUGI_DIR);
    const fingerprint = await verifyOnnxModel(
      options.modelDir,
      tsumugiLock,
      TSUMUGI_DIR,
      [TSUMUGI_FILES.graph, TSUMUGI_FILES.heads, TSUMUGI_FILES.headsJson],
      'npm run model:tsumugi',
    );
    return TsumugiModel.load(nodeOnnxRunner, directory, fingerprint);
  }

  if (options.model === 'yourmt3') {
    const directory = join(options.modelDir, YOURMT3_DIR);
    const fingerprint = await verifyOnnxModel(
      options.modelDir,
      yourmt3Lock,
      YOURMT3_DIR,
      [YOURMT3_FILES.encoder, YOURMT3_FILES.decoder, YOURMT3_FILES.vocab],
      'npm run model:yourmt3',
    );
    return YourMt3Model.load(nodeOnnxRunner, directory, fingerprint);
  }

  if (options.model === 'piano') {
    const directory = join(options.modelDir, PIANO_DIR);
    const fingerprint = await verifyOnnxModel(
      options.modelDir,
      pianoLock,
      PIANO_DIR,
      [PIANO_FILES.graph],
      'npm run model:piano',
    );
    return PianoTranscriptionModel.load(nodeOnnxRunner, directory, fingerprint);
  }

  return loadPinnedModel(nodeModelSource(options.modelPath), modelLock as ModelLock);
}

/** Read one of the ONNX weight files, relative to the weights root. */
async function readOnnxFile(root: string, path: string): Promise<ArrayBuffer> {
  const bytes = await readFile(join(root, path));
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
}

/** Check a model's weights against its lock file. */
async function verifyOnnxModel(
  root: string,
  lock: unknown,
  directory: string,
  files: string[],
  fetchCommand: string,
): Promise<string> {
  return verifyOnnxWeights(
    async (path) => {
      try {
        return await readOnnxFile(root, path);
      } catch (error) {
        // The weights are not in the repository, so a missing file is the
        // expected case on a fresh checkout. Say which command fixes it.
        const reason = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Could not read the model's ${path} (${reason}). ` +
            `Run \`${fetchCommand}\` to fetch the weights.`,
        );
      }
    },
    lock as { files: Record<string, { sha256: string }> },
    { dir: directory, files },
    fetchCommand,
  );
}

/** Check the Okarina weights against the lock file. */
async function verifyOkarinaWeights(
  directory: string,
  size: OkarinaSize,
): Promise<string> {
  const lock = okarinaLock as unknown as {
    files: Record<string, { sha256: string }>;
  };
  const digests: string[] = [];

  for (const name of [OKARINA_FILES.frontEnd, OKARINA_FILES[size]]) {
    const expected = lock.files[name]?.sha256;
    if (!expected) {
      throw new Error(
        `The lock file does not record a hash for ${name}. ` +
          'Add it to src/core/okarina.lock.json before using this model.',
      );
    }
    const bytes = await readFile(join(directory, name));
    const digest = createHash('sha256').update(bytes).digest('hex');
    if (digest !== expected) {
      throw new Error(
        `The model's ${name} does not match the pinned version.\n` +
          `  expected sha256 ${expected}\n` +
          `  got      sha256 ${digest}\n` +
          'Run `npm run model:okarina` to fetch the pinned weights.',
      );
    }
    digests.push(digest);
  }

  return createHash('sha256').update(digests.join('')).digest('hex');
}

/** Convert one file and write the MIDI output. */
async function convertFile(
  filePath: string,
  options: CliOptions,
  model: TranscriptionModel,
  provenance: string,
): Promise<{ result: TranscriptionResult; outputPath: string }> {
  const data = await readFile(filePath);
  const audio = await decodeAudio(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    nodeAudioDecoder,
  );
  const samples = prepareMono(audio);

  const result = await transcribe(samples, {
    model,
    detectDrums: options.drums,
    forceTempo: Number.isFinite(options.tempo) ? options.tempo : undefined,
    range: options.range,
    quantise: options.quantise,
    pianify: options.pianify,
    onProgress: options.quiet
      ? undefined
      : (info) => {
          process.stderr.write(`\r  ${info.label.padEnd(30)} ${Math.round(info.value * 100)}%`);
        },
  });

  if (!options.quiet) process.stderr.write('\r' + ' '.repeat(50) + '\r');

  const bytes = buildMidiFile(result.tracks, result.tempo, result.key, { provenance });
  await mkdir(options.outputDir, { recursive: true });
  const outputPath = join(
    options.outputDir,
    `${basename(filePath, extname(filePath))}.mid`,
  );
  await writeFile(outputPath, bytes);

  return { result, outputPath };
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));

  if (options.inputs.length === 0) {
    printHelp();
    process.exit(1);
  }

  const model = await loadModel(options);
  const provenance = provenanceFor(model);

  if (!options.quiet) {
    console.log(
      `Model ${model.displayName} (${model.id}, sha256:${model.fingerprint.slice(0, 16)}…)`,
    );
  }

  const summaries: unknown[] = [];

  for (const input of options.inputs) {
    const path = resolve(input);
    if (!options.quiet) console.log(`\n${basename(path)}`);

    try {
      const { result, outputPath } = await convertFile(path, options, model, provenance);

      const summary = {
        input: path,
        output: outputPath,
        model: model.id,
        duration: Number(result.duration.toFixed(2)),
        tempo: result.tempo.bpm,
        key: keyName(result.key.tonic, result.key.mode),
        provenance,
        tracks: result.tracks.map((track) => ({
          name: track.name,
          notes: track.notes.length,
        })),
        stats: result.stats,
        log: result.log,
      };
      summaries.push(summary);

      if (options.json) continue;

      console.log(`  ${result.duration.toFixed(1)}s  ${result.tempo.bpm} BPM  ${summary.key}`);
      for (const track of result.tracks) {
        console.log(`    ${track.name.padEnd(10)} ${String(track.notes.length).padStart(5)} notes`);
      }
      for (const line of result.log) console.log(`    · ${line}`);
      console.log(`  -> ${outputPath}`);
    } catch (error) {
      console.error(`  failed: ${(error as Error).message}`);
      process.exitCode = 1;
    }
  }

  if (options.json) {
    console.log(JSON.stringify(summaries, null, 2));
  }

  model.dispose?.();
}

/**
 * Read the model from disk.
 *
 * Node's `fetch` does not understand `file://` URLs, so the files are read
 * here and the URL is handed to TensorFlow.js separately.
 */
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

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
