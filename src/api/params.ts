/**
 * The parameters of a conversion request.
 *
 * The names here are the same ones the app uses, so that a value set in the
 * browser means the same thing when it is sent to the API. Everything has a
 * default, so the smallest useful request is just a file.
 *
 * A parameter that is present but wrong is an error, not a shrug. It is
 * better for a caller to be told that `grid=abc` is not a number than to
 * get a file that quietly ignored it.
 */

import { FrequencyRange } from '../core/range';
import { QuantiseOptions } from '../core/quantise';
import { PianifyOptions } from '../core/pianify';
import { OkarinaSize } from '../core/okarina';
import { DEFAULT_MODEL, MODEL_IDS, ModelId } from '../core/registry';
import { ConvertRequest } from './convert';
import { ApiError } from './errors';

/** Everything a conversion needs. */
export interface ApiParameters extends ConvertRequest {
  /** Which model to use. */
  model: ModelId;
  /** Which Okarina model to use. */
  okarinaSize: OkarinaSize;
  /** How to answer. */
  format: 'midi' | 'json';
  /** The name of the uploaded file, for the reply's filename. */
  filename?: string;
}

/** The names of the accepted parameters, for the documentation. */
export const PARAMETER_NAMES = [
  'file',
  'model',
  'okarinaSize',
  'drums',
  'tempo',
  'rangeLow',
  'rangeHigh',
  'grid',
  'gridPhase',
  'pianify',
  'format',
] as const;

/** Where the values came from: the query string, or a form field. */
export type ParameterSource = Record<string, string | undefined>;

/** Read a boolean. Anything that is not obviously false is true. */
function readBoolean(value: string, name: string): boolean {
  const lower = value.trim().toLowerCase();
  if (['1', 'true', 'yes', 'on', ''].includes(lower)) return true;
  if (['0', 'false', 'no', 'off'].includes(lower)) return false;
  throw ApiError.badRequest(
    `"${name}" must be true or false, not "${value}".`,
    { parameter: name, value },
  );
}

/** Read a number. */
function readNumber(value: string, name: string): number {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw ApiError.badRequest(
      `"${name}" must be a number, not "${value}".`,
      { parameter: name, value },
    );
  }
  return number;
}

/**
 * Read the parameters of a request.
 *
 * @param source  the values, from the query string and the form.
 * @param format  the format the caller asked for, which overrides the field.
 */
export function parseParameters(
  source: ParameterSource,
  format?: 'midi' | 'json',
): ApiParameters {
  const parameters: ApiParameters = {
    model: DEFAULT_MODEL,
    okarinaSize: 'm',
    detectDrums: true,
    format: format ?? 'midi',
  };

  if (source.model !== undefined && source.model !== '') {
    if (!(MODEL_IDS as readonly string[]).includes(source.model)) {
      throw ApiError.badRequest(
        `"${source.model}" is not a model this tool has.`,
        { parameter: 'model', value: source.model, allowed: [...MODEL_IDS] },
      );
    }
    parameters.model = source.model as ModelId;
  }

  if (source.okarinaSize !== undefined && source.okarinaSize !== '') {
    if (!['s', 'm', 'l'].includes(source.okarinaSize)) {
      throw ApiError.badRequest(
        `"okarinaSize" must be s, m or l, not "${source.okarinaSize}".`,
        { parameter: 'okarinaSize', value: source.okarinaSize },
      );
    }
    parameters.okarinaSize = source.okarinaSize as OkarinaSize;
  }

  if (source.drums !== undefined && source.drums !== '') {
    parameters.detectDrums = readBoolean(source.drums, 'drums');
  }

  if (source.tempo !== undefined && source.tempo !== '') {
    const tempo = readNumber(source.tempo, 'tempo');
    if (tempo < 20 || tempo > 400) {
      throw ApiError.badRequest(
        '"tempo" must be between 20 and 400 beats per minute.',
        { parameter: 'tempo', value: tempo },
      );
    }
    parameters.forceTempo = tempo;
  }

  const low = source.rangeLow !== undefined && source.rangeLow !== ''
    ? readNumber(source.rangeLow, 'rangeLow')
    : undefined;
  const high = source.rangeHigh !== undefined && source.rangeHigh !== ''
    ? readNumber(source.rangeHigh, 'rangeHigh')
    : undefined;

  if (low !== undefined || high !== undefined) {
    const range: FrequencyRange = {
      low: Math.max(0, low ?? 0),
      high: high ?? Number.POSITIVE_INFINITY,
    };
    if (range.low >= range.high) {
      throw ApiError.badRequest(
        '"rangeLow" must be below "rangeHigh".',
        { parameter: 'rangeLow', value: range.low, rangeHigh: range.high },
      );
    }
    parameters.range = range;
  }

  if (source.grid !== undefined && source.grid !== '') {
    const grid = readNumber(source.grid, 'grid');
    if (grid < 0) {
      throw ApiError.badRequest('"grid" cannot be negative.', {
        parameter: 'grid',
        value: grid,
      });
    }
    const phase = source.gridPhase !== undefined && source.gridPhase !== ''
      ? readNumber(source.gridPhase, 'gridPhase')
      : 0;
    const quantise: QuantiseOptions = { grid, phase: phase / 1000 };
    parameters.quantise = grid > 0 ? quantise : undefined;
  } else if (source.gridPhase !== undefined && source.gridPhase !== '') {
    throw ApiError.badRequest(
      '"gridPhase" needs a "grid" to be a phase of.',
      { parameter: 'gridPhase' },
    );
  }

  if (source.pianify !== undefined && source.pianify !== '') {
    if (readBoolean(source.pianify, 'pianify')) {
      parameters.pianify = {} satisfies PianifyOptions;
    }
  }

  if (source.format !== undefined && source.format !== '') {
    if (source.format !== 'midi' && source.format !== 'json') {
      throw ApiError.badRequest(
        `"format" must be midi or json, not "${source.format}".`,
        { parameter: 'format', value: source.format },
      );
    }
    parameters.format = source.format;
  }

  if (source.filename !== undefined && source.filename !== '') {
    parameters.filename = source.filename;
  }

  return parameters;
}
