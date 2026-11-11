/**
 * The parameters of a conversion.
 *
 * These are the names the app's controls use, the names the command line
 * uses and the names the API describes. One table, so that a value set in
 * one place means the same thing in the others — a script that reads
 * `/api/parameters` and builds a command line cannot get it wrong.
 *
 * Nothing here is a request. The server never receives a recording, so it
 * has no parameters of its own to parse; it publishes the ones a client
 * applies locally, and the client applies them wherever the recording
 * already is.
 */

/** What kind of value a parameter takes. */
export type ParameterType = 'model' | 'enum' | 'boolean' | 'number';

/** One parameter of a conversion. */
export interface ParameterSpec {
  /** The name, as the app and the API call it. */
  name: string;
  /** What kind of value it takes. */
  type: ParameterType;
  /** What it is when it is not set. */
  default: string | number | boolean;
  /** What it does, in one line. */
  description: string;
  /** The values it may take, when it may only take a few. */
  allowed?: readonly string[];
  /** The command line flag that sets it. */
  flag: string;
  /** The unit, when it has one. */
  unit?: string;
}

/**
 * Every parameter, in the order they are offered.
 *
 * `file` is not here, because a file is not a parameter: it is the
 * recording itself, and the recording never goes anywhere.
 */
export const PARAMETERS: readonly ParameterSpec[] = [
  {
    name: 'model',
    type: 'model',
    default: 'piano',
    description: 'Which model does the work.',
    allowed: ['basic-pitch', 'tsumugi', 'yourmt3', 'piano', 'okarina'],
    flag: '--model',
  },
  {
    name: 'okarinaSize',
    type: 'enum',
    default: 'm',
    description: 'Which Okarina model to use. Only for Okarina.',
    allowed: ['s', 'm', 'l'],
    flag: '--okarina-size',
  },
  {
    name: 'drums',
    type: 'boolean',
    default: true,
    description: 'Look for a drum kit.',
    flag: '--no-drums',
  },
  {
    name: 'tempo',
    type: 'number',
    default: 'estimated',
    description: 'Force a tempo instead of estimating one.',
    flag: '--tempo',
    unit: 'BPM',
  },
  {
    name: 'rangeLow',
    type: 'number',
    default: 0,
    description: 'Lowest frequency to keep.',
    flag: '--range-low',
    unit: 'Hz',
  },
  {
    name: 'rangeHigh',
    type: 'number',
    default: 'no limit',
    description: 'Highest frequency to keep.',
    flag: '--range-high',
    unit: 'Hz',
  },
  {
    name: 'grid',
    type: 'number',
    default: 0,
    description:
      'Snap the notes to a grid of this many slots per second. 0 leaves the notes alone.',
    flag: '--grid',
  },
  {
    name: 'gridPhase',
    type: 'number',
    default: 0,
    description: 'Where the first slot sits.',
    flag: '--grid-phase',
    unit: 'ms',
  },
  {
    name: 'pianify',
    type: 'boolean',
    default: false,
    description:
      'Arrange the notes for two hands: fold the re-triggers together, divide the chords, and drop what neither hand could hold.',
    flag: '--pianify',
  },
  {
    name: 'format',
    type: 'enum',
    default: 'midi',
    description: 'What to write out.',
    allowed: ['midi', 'json'],
    flag: '--json',
  },
];

/** The names of every parameter, in order. */
export const PARAMETER_NAMES: readonly string[] = PARAMETERS.map(
  (parameter) => parameter.name,
);

/** Look a parameter up by name. */
export function parameterNamed(name: string): ParameterSpec | undefined {
  return PARAMETERS.find((parameter) => parameter.name === name);
}
