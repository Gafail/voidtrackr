/**
 * The catalogue of models this tool can use.
 *
 * The model sits behind an interface, so which one runs is a choice rather
 * than a fact about the program. Each entry here describes a model: what it
 * is called, where the weights come from, what licence they are under, and
 * what it is good at.
 *
 * The licence is not decoration. It decides whether the output of this tool
 * can be sold, so every entry states it plainly and the app shows it.
 *
 * The registry only holds the descriptions. Loading is left to the
 * environment, because the browser fetches the weights over HTTP and Node
 * reads them off disk.
 */

/** The identifiers a model can be chosen by. */
export const MODEL_IDS = ['basic-pitch', 'tsumugi', 'yourmt3', 'piano', 'okarina'] as const;

export type ModelId = (typeof MODEL_IDS)[number];

export interface ModelDescriptor {
  id: ModelId;
  /** Name shown to the user. */
  displayName: string;
  /** Licence the weights are under, as the model card states it. */
  license: string;
  /** Where the licence is written down. */
  licenseUrl: string;
  /**
   * Whether the weights may be used commercially.
   *
   * This is the model author's own grant, not a legal opinion: a model
   * trained on data with non-commercial terms carries that caveat even when
   * the weights themselves are permissively licensed.
   */
  commercialUse: boolean;
  /** One line on what the model is like. */
  summary: string;
  /** Where the weights come from. */
  source: string;
  /** How the weights are fetched. */
  fetchCommand: string;
  /** What to expect from it, for the user. */
  notes: string;
}

export const MODELS: Record<ModelId, ModelDescriptor> = {
  'basic-pitch': {
    id: 'basic-pitch',
    displayName: 'Basic Pitch',
    license: 'Apache-2.0',
    licenseUrl: 'https://github.com/spotify/basic-pitch-ts/blob/main/LICENSE',
    commercialUse: true,
    summary:
      "Spotify's small convolutional model. Quick, and the reference every other model is measured against.",
    source: 'https://raw.githubusercontent.com/spotify/basic-pitch-ts/main/model',
    fetchCommand: 'npm run model',
    notes:
      'Finds fewer notes than the newer models and puts onsets on a frame grid. ' +
      'It is the default because it is small, fast and permissively licensed.',
  },
  tsumugi: {
    id: 'tsumugi',
    displayName: 'Tsumugi',
    license: 'MIT',
    licenseUrl: 'https://huggingface.co/anime-song/instrument_agnostic_amt',
    commercialUse: true,
    summary:
      'An instrument-agnostic model built for synthesised and anime music — the closest published match to this material.',
    source: 'https://huggingface.co/baroned/musicnote-ondevice-models',
    fetchCommand: 'npm run model:tsumugi',
    notes:
      'The weights are MIT, not just the code. It extends the Neural Semi-CRF that ' +
      'TransKun introduced. Its training mix includes MAESTRO, which is CC BY-NC-SA, ' +
      'so commercial use is your own assessment to make.',
  },
  yourmt3: {
    id: 'yourmt3',
    displayName: 'YourMT3+',
    license: 'Apache-2.0',
    licenseUrl: 'https://huggingface.co/mimbres/YourMT3',
    commercialUse: true,
    summary:
      'A multi-instrument model that separates the mix into piano, bass, drums, vocals and more.',
    source: 'https://huggingface.co/ONLY4DarkKnight/YourMT3-plus-onnx',
    fetchCommand: 'npm run model:yourmt3',
    notes:
      'The strongest published numbers on multi-instrument material: F1 0.817 on ' +
      'Slakh against Basic Pitch at 0.465. It is also the slowest — roughly a fifth ' +
      'of real time on the CPU, and it wants a couple of gigabytes of memory. ' +
      'The authors state that MAESTRO in their training mix cannot be licensed by them.',
  },
  piano: {
    id: 'piano',
    displayName: 'Piano Transcription',
    license: 'CC BY 4.0',
    licenseUrl: 'https://zenodo.org/records/4034264',
    commercialUse: true,
    summary:
      "ByteDance's high-resolution piano model. The best onset and offset timing of anything here.",
    source: 'https://huggingface.co/baroned/musicnote-ondevice-models',
    fetchCommand: 'npm run model:piano',
    notes:
      'It regresses the exact onset and offset time instead of snapping to a frame ' +
      'grid, which is why its timing is so much better. It is a piano model, so on a ' +
      'full mix it picks out the piano-like content.',
  },
  okarina: {
    id: 'okarina',
    displayName: 'Okarina',
    license: 'CC BY-NC-SA 4.0 — non-commercial only',
    licenseUrl: 'https://huggingface.co/mazun/rekord',
    commercialUse: false,
    summary:
      "An ONNX reimplementation of Harmonica, the model behind BandLab's audio-to-MIDI service.",
    source: 'https://huggingface.co/mazun/rekord/resolve/main',
    fetchCommand: 'npm run model:okarina',
    notes:
      'Finds substantially more notes and separates them into vocal, bass and other. ' +
      'Its onsets land on a 32 ms grid, so snapping them to the beat helps. ' +
      'The weights are licensed for non-commercial use only, so it stays opt-in.',
  },
};

/**
 * The model used when none is chosen.
 *
 * It was Basic Pitch until 0.4.0, because Basic Pitch needs no question asked
 * about it: it is a megabyte, it is Apache-2.0, and it works the moment its
 * weights are fetched. It is also the weakest of the models here.
 *
 * The piano model replaced it after being measured over the user's own 19
 * tracks, where it shared 63.9 % of Okarina's notes against 29.6 % for
 * YourMT3+ and 23.4 % for Tsumugi. Its licence is CC BY 4.0, so what it
 * produces can be sold. See `docs/model-research.md`, section 6.
 */
export const DEFAULT_MODEL: ModelId = 'piano';

/** Every model, in the order they should be offered. */
export function listModels(): ModelDescriptor[] {
  return MODEL_IDS.map((id) => MODELS[id]);
}

/** Check that a string names a model this tool knows about. */
export function isModelId(value: string): value is ModelId {
  return (MODEL_IDS as readonly string[]).includes(value);
}
