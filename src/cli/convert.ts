/**
 * Command line converter.
 *
 *   npm run cli -- song.mp3 -o out/
 *   npm run cli -- songs/*.wav --no-drums --json
 *   npm run cli -- song.mp3 -o out/ --model okarina
 *
 * The conversion itself lives in `src/api/convert.ts`, which the HTTP API
 * calls as well. That is deliberate: a file produced here and a file
 * produced by the API have to be byte-identical, and the only way to be
 * sure of that is to have one code path.
 */

import { installTfjsNodeBackend } from '../api/backend';
import { convertAudio } from '../api/convert';
import { loadModel } from '../api/model-loader';

await installTfjsNodeBackend();

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { basename, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { FrequencyRange } from '../core/range';
import { QuantiseOptions } from '../core/quantise';
import { PianifyOptions } from '../core/pianify';
import { OkarinaSize } from '../core/okarina';
import { TranscriptionModel } from '../core/model';
import { keyName } from '../core/tempo';
import { DEFAULT_MODEL, listModels, ModelId, MODELS } from '../core/registry';
import { provenanceFor } from '../core/version';
import { TranscriptionResult } from '../core/types';

const HERE = fileURLToPath(new URL('.', import.meta.url));
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

  const model = await loadModel(options.model, {
    modelDir: options.modelDir,
    modelPath: options.modelPath,
  }, options.okarinaSize);
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
