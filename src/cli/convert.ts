/**
 * Command line converter.
 *
 *   npm run cli -- song.mp3 -o out/
 *   npm run cli -- songs/*.wav --no-drums --json
 *   npm run cli -- song.mp3 -o out/ --model okarina
 *   npm run cli -- --server http://192.168.100.109 song.mp3 -o out/
 *
 * The recording is read from disk and stays there. Nothing is uploaded, and
 * this command never talks to a server about the audio — only about the
 * model's weights, when it is told to fetch them with `--server`.
 *
 * The conversion itself lives in `src/core/convert.ts`, which the browser
 * uses as well. That is deliberate: a file produced here and a file
 * produced in the browser have to be byte-identical, and the only way to be
 * sure of that is to have one code path.
 */

import { installTfjsNodeBackend } from '../core/backend';
import { convertAudio } from '../core/convert';
import { loadModel } from '../core/model-loader';

await installTfjsNodeBackend();

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { FrequencyRange } from '../core/range';
import { QuantiseOptions } from '../core/quantise';
import { PianifyOptions } from '../core/pianify';
import { OkarinaSize } from '../core/okarina';
import { TranscriptionModel } from '../core/model';
import { keyName } from '../core/tempo';
import { DEFAULT_MODEL, listModels, ModelId, MODELS } from '../core/registry';
import { PARAMETERS } from '../core/parameters';
import { provenanceFor } from '../core/version';
import { TranscriptionResult } from '../core/types';
import { fetchWeights } from '../core/weights-fetch';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const DEFAULT_MODEL_PATH = resolve(HERE, '..', '..', 'public', 'model', 'model.json');
const DEFAULT_ONNX_DIR = resolve(HERE, '..', '..', 'models', 'onnx');

/** Where the weights fetched from a server are kept. */
const WEIGHTS_CACHE = resolve(
  process.env.VOIDTRACKR_CACHE ?? join(homedir(), '.cache', 'voidtrackr', 'weights'),
);

interface CliOptions {
  inputs: string[];
  outputDir: string;
  /** Which model to use. */
  model: ModelId;
  /** Path to a Basic Pitch model.json. */
  modelPath: string;
  /** Directory holding the ONNX weights. */
  modelDir: string;
  /** Where to fetch the weights from, if not the local disk. */
  server?: string;
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
 * fundamentals, so 400 Hz is G4. The two bounds may also be given on their
 * own, as `--range-low` and `--range-high`, which are the names the API
 * publishes.
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

  /** The bounds given one at a time, if any. */
  let rangeLow: number | undefined;
  let rangeHigh: number | undefined;

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
      case '--server':
        options.server = argv[++i] ?? options.server;
        break;
      case '--okarina-size':
      case '--okarinaSize':
        options.okarinaSize = (argv[++i] as OkarinaSize) ?? options.okarinaSize;
        break;
      case '--no-drums':
        options.drums = false;
        break;
      case '--drums':
        options.drums = true;
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
      case '--range-low':
      case '--rangeLow':
        rangeLow = Number(argv[++i]);
        break;
      case '--range-high':
      case '--rangeHigh':
        rangeHigh = Number(argv[++i]);
        break;
      case '--grid':
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
      case '--grid-phase':
      case '--gridPhase':
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
      case '--format': {
        const value = argv[++i] ?? '';
        if (value !== 'midi' && value !== 'json') {
          console.error(`Could not read the format "${value}". Use midi or json.`);
          process.exit(1);
        }
        options.json = value === 'json';
        break;
      }
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

  // The bounds given on their own are joined with whatever `--range` set.
  if (rangeLow !== undefined || rangeHigh !== undefined) {
    const base = options.range ?? { low: 0, high: Number.POSITIVE_INFINITY };
    const low = rangeLow ?? base.low;
    const high = rangeHigh ?? base.high;
    if (!Number.isFinite(low) || !Number.isFinite(high) || low >= high) {
      console.error('The range is not readable. The low bound must be below the high one.');
      process.exit(1);
    }
    options.range = { low, high };
  }

  return options;
}

function printHelp(): void {
  const parameters = PARAMETERS.map((parameter) => {
    const flag = parameter.flag.padEnd(18);
    const fallback = String(parameter.default).padEnd(12);
    const unit = parameter.unit ? ` ${parameter.unit}` : '';
    return `  ${flag} ${fallback} ${parameter.description}${unit}`;
  }).join('\n');

  console.log(`
voidtrackr — audio to MIDI, locally

  npm run cli -- <input...> [options]

The recording is read from disk and stays there. Nothing is uploaded, to a
server or anywhere else.

Options:
  -o, --out <dir>           Directory for the .mid files (default: current directory)
      --model <name>        Which model to use: ${Object.keys(MODELS).join(' or ')} (default: ${DEFAULT_MODEL})
      --server <url>        Fetch the model's weights from a server, such as
                            http://192.168.100.109. The recording is not sent.
      --model-path <path>   Path to a Basic Pitch model.json
      --model-dir <dir>     Directory holding the ONNX weights
      --json                Print results as JSON
  -l, --list-models         List the models and their licences
  -q, --quiet               Only print errors

Parameters:
${parameters}

The same names are published at /api/parameters on the server.
`);
}

/** Convert one file and write the MIDI output. */
async function convertFile(
  filePath: string,
  options: CliOptions,
  model: TranscriptionModel,
  provenance: string,
): Promise<{ result: TranscriptionResult; outputPath: string }> {
  const data = await readFile(filePath);
  const buffer = data.buffer.slice(
    data.byteOffset,
    data.byteOffset + data.byteLength,
  ) as ArrayBuffer;

  const { result, midi } = await convertAudio(
    buffer,
    {
      detectDrums: options.drums,
      forceTempo: Number.isFinite(options.tempo) ? options.tempo : undefined,
      range: options.range,
      quantise: options.quantise,
      pianify: options.pianify,
      onProgress: options.quiet
        ? undefined
        : (info) => {
            process.stderr.write(
              `\r  ${info.label.padEnd(30)} ${Math.round(info.value * 100)}%`,
            );
          },
    },
    model,
    provenance,
  );

  if (!options.quiet) process.stderr.write('\r' + ' '.repeat(50) + '\r');

  await mkdir(options.outputDir, { recursive: true });
  const outputPath = join(
    options.outputDir,
    `${basename(filePath, extname(filePath))}.mid`,
  );
  await writeFile(outputPath, midi);

  return { result, outputPath };
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));

  if (options.inputs.length === 0) {
    printHelp();
    process.exit(1);
  }

  let modelDir = options.modelDir;
  let modelPath = options.modelPath;

  // The weights may come from a server rather than from disk. What comes
  // over the wire is the model's own files, checked against the pinned
  // hashes. The recording is not involved: it is read from disk below and
  // never leaves the machine.
  if (options.server) {
    const fetched = await fetchWeights(
      { base: options.server },
      options.model,
      WEIGHTS_CACHE,
    );
    modelDir = fetched.modelDir;
    modelPath = fetched.modelPath;

    if (!options.quiet) {
      console.log(
        `Weights from ${options.server}: ${fetched.cached} already cached, ` +
          `${fetched.bytes} bytes downloaded.`,
      );
    }
  }

  const model = await loadModel(
    options.model,
    { modelDir, modelPath },
    options.okarinaSize,
  );
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

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
