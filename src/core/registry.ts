/**
 * The catalogue of models this tool can use.
 *
 * The model sits behind an interface, so which one runs is a choice rather
 * than a fact about the program. Each entry here describes a model: what it
 * is called, where the weights come from, what licence they are under, and
 * what it is good at.
 *
 * The registry only holds the descriptions. Loading is left to the
 * environment, because the browser fetches the weights over HTTP and Node
 * reads them off disk.
 */

/** The identifiers a model can be chosen by. */
export const MODEL_IDS = ['basic-pitch', 'okarina'] as const;

export type ModelId = (typeof MODEL_IDS)[number];

export interface ModelDescriptor {
  id: ModelId;
  /** Name shown to the user. */
  displayName: string;
  /** Licence the weights are under. */
  license: string;
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
    summary: "Spotify's small convolutional model. Quick, and the reference every other model is measured against.",
    source: 'https://raw.githubusercontent.com/spotify/basic-pitch-ts/main/model',
    fetchCommand: 'npm run model',
    notes:
      'Finds fewer notes than the newer models and puts onsets on a frame grid. ' +
      'It is the default because it is small, fast and permissively licensed.',
  },
  okarina: {
    id: 'okarina',
    displayName: 'Okarina',
    license: 'CC BY-NC-SA 4.0 — non-commercial',
    summary:
      "An ONNX reimplementation of Harmonica, the model behind BandLab's audio-to-MIDI service.",
    source: 'https://huggingface.co/mazun/rekord/resolve/main',
    fetchCommand: 'npm run model:okarina',
    notes:
      'Finds substantially more notes and separates them into vocal, bass and other. ' +
      'Its onsets land on a 32 ms grid, so snapping them to the beat helps. ' +
      'The weights are licensed for non-commercial use only.',
  },
};

/** The model used when none is chosen. */
export const DEFAULT_MODEL: ModelId = 'basic-pitch';

/** Every model, in the order they should be offered. */
export function listModels(): ModelDescriptor[] {
  return MODEL_IDS.map((id) => MODELS[id]);
}

/** Check that a string names a model this tool knows about. */
export function isModelId(value: string): value is ModelId {
  return (MODEL_IDS as readonly string[]).includes(value);
}
