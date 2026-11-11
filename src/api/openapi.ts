/**
 * The OpenAPI document the API serves.
 *
 * It is built from the same numbers the code uses — the model list, the
 * parameter names, the version — so it cannot drift away from what the
 * server actually does.
 */

import { MODEL_IDS, DEFAULT_MODEL, MODELS } from '../core/registry';
import { PIPELINE_VERSION } from '../core/version';
import { PARAMETER_NAMES } from './params';

/** The document, as a plain object. */
export function openApiDocument(): Record<string, unknown> {
  const modelParameter = {
    name: 'model',
    in: 'query',
    required: false,
    description: 'Which model to use.',
    schema: {
      type: 'string',
      enum: [...MODEL_IDS],
      default: DEFAULT_MODEL,
    },
  };

  const scalarParameters = [
    {
      name: 'okarinaSize',
      in: 'query',
      required: false,
      description: 'Which Okarina model to use.',
      schema: { type: 'string', enum: ['s', 'm', 'l'], default: 'm' },
    },
    {
      name: 'drums',
      in: 'query',
      required: false,
      description: 'Look for a drum kit.',
      schema: { type: 'boolean', default: true },
    },
    {
      name: 'tempo',
      in: 'query',
      required: false,
      description: 'Force a tempo, in beats per minute.',
      schema: { type: 'number', minimum: 20, maximum: 400 },
    },
    {
      name: 'rangeLow',
      in: 'query',
      required: false,
      description: 'Lowest frequency to keep, in hertz.',
      schema: { type: 'number', minimum: 0, default: 0 },
    },
    {
      name: 'rangeHigh',
      in: 'query',
      required: false,
      description: 'Highest frequency to keep, in hertz.',
      schema: { type: 'number', minimum: 0 },
    },
    {
      name: 'grid',
      in: 'query',
      required: false,
      description: 'Slots per second. Zero leaves the notes where the model put them.',
      schema: { type: 'number', minimum: 0, default: 0 },
    },
    {
      name: 'gridPhase',
      in: 'query',
      required: false,
      description: 'Where the first slot sits, in milliseconds.',
      schema: { type: 'number', default: 0 },
    },
    {
      name: 'pianify',
      in: 'query',
      required: false,
      description: 'Arrange the notes for two hands.',
      schema: { type: 'boolean', default: false },
    },
    {
      name: 'format',
      in: 'query',
      required: false,
      description: 'MIDI, or the notes as JSON.',
      schema: { type: 'string', enum: ['midi', 'json'], default: 'midi' },
    },
  ];

  const errorSchema = {
    type: 'object',
    properties: {
      error: {
        type: 'object',
        required: ['code', 'message'],
        properties: {
          code: {
            type: 'string',
            enum: [
              'bad_request',
              'unsupported_media_type',
              'not_found',
              'method_not_allowed',
              'payload_too_large',
              'unprocessable_entity',
              'model_unavailable',
              'unauthorized',
              'busy',
              'internal_error',
            ],
          },
          message: { type: 'string' },
          details: { type: 'object', additionalProperties: true },
        },
      },
    },
  };

  const responses = (description: string) => ({
    '400': { description: 'The request does not make sense.', content: { 'application/json': { schema: errorSchema } } },
    '401': { description: 'The token is missing or wrong.', content: { 'application/json': { schema: errorSchema } } },
    '404': { description: 'There is nothing at that address.', content: { 'application/json': { schema: errorSchema } } },
    '405': { description: 'That method is not allowed here.', content: { 'application/json': { schema: errorSchema } } },
    '413': { description: 'The upload is too large.', content: { 'application/json': { schema: errorSchema } } },
    '415': { description: 'The media type is not one the API takes.', content: { 'application/json': { schema: errorSchema } } },
    '422': { description: 'The audio could not be read.', content: { 'application/json': { schema: errorSchema } } },
    '503': { description: 'The server is busy, or a model is not installed.', content: { 'application/json': { schema: errorSchema } } },
    default: { description, content: { 'application/json': { schema: errorSchema } } },
  });

  return {
    openapi: '3.1.0',
    info: {
      title: 'voidtrackr',
      version: PIPELINE_VERSION,
      summary: 'Audio to MIDI, on your own machine.',
      description:
        'Convert a recording to multi-track MIDI. The same pipeline the browser ' +
        'site and the command line use, so the file a caller gets is byte ' +
        'identical to the one they would get from either of them.\n\n' +
        'The audio is read into memory, converted, and the MIDI is sent back. ' +
        'Nothing is written to disk.',
      license: { name: 'MIT', identifier: 'MIT' },
    },
    servers: [{ url: '/api', description: 'This site' }],
    tags: [
      { name: 'convert', description: 'Turning audio into MIDI.' },
      { name: 'the api itself', description: 'What this API is and what it can do.' },
    ],
    paths: {
      '/convert': {
        post: {
          tags: ['convert'],
          summary: 'Convert an audio file to MIDI.',
          description:
            'Send the recording in a multipart part called `file`, or as the ' +
            'request body with an audio content type. The reply is the MIDI ' +
            'file, unless `format=json` asks for the notes instead.',
          operationId: 'convert',
          parameters: [modelParameter, ...scalarParameters],
          requestBody: {
            required: true,
            content: {
              'multipart/form-data': {
                schema: {
                  type: 'object',
                  required: ['file'],
                  properties: {
                    file: { type: 'string', format: 'binary' },
                    ...Object.fromEntries(
                      scalarParameters.map((parameter) => [
                        parameter.name,
                        { type: parameter.schema.type },
                      ]),
                    ),
                  },
                },
              },
              'audio/wav': { schema: { type: 'string', format: 'binary' } },
              'audio/mpeg': { schema: { type: 'string', format: 'binary' } },
              'audio/x-wav': { schema: { type: 'string', format: 'binary' } },
            },
          },
          responses: {
            '200': {
              description: 'The MIDI file.',
              headers: {
                'X-Voidtrackr-Pipeline': { description: 'The pipeline version.', schema: { type: 'string' } },
                'X-Voidtrackr-Model': { description: 'The model that was used.', schema: { type: 'string' } },
                'X-Voidtrackr-Notes': { description: 'How many notes it found.', schema: { type: 'integer' } },
                'X-Voidtrackr-Tempo': { description: 'The tempo, in beats per minute.', schema: { type: 'string' } },
                'X-Voidtrackr-Key': { description: 'The key it found.', schema: { type: 'string' } },
                'X-Voidtrackr-SHA256': { description: 'The hash of the MIDI file.', schema: { type: 'string' } },
              },
              content: {
                'audio/midi': { schema: { type: 'string', format: 'binary' } },
                'application/json': { schema: { type: 'object' } },
              },
            },
            ...responses('Something went wrong.'),
          },
        },
      },
      '/health': {
        get: {
          tags: ['the api itself'],
          summary: 'Whether the server is up.',
          operationId: 'health',
          responses: {
            '200': {
              description: 'The server is up.',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      status: { type: 'string' },
                      pipeline: { type: 'string' },
                      threads: { type: 'integer' },
                      defaultModel: { type: 'string' },
                      models: { type: 'array', items: { type: 'string' } },
                      conversions: { type: 'integer' },
                      failures: { type: 'integer' },
                      uptimeSeconds: { type: 'integer' },
                    },
                  },
                },
              },
            },
          },
        },
      },
      '/models': {
        get: {
          tags: ['the api itself'],
          summary: 'The models and their licences.',
          description:
            'The licence decides whether the output can be sold, so it is ' +
            'stated plainly rather than buried.',
          operationId: 'models',
          responses: {
            '200': {
              description: 'The catalogue.',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      default: { type: 'string' },
                      models: {
                        type: 'array',
                        items: {
                          type: 'object',
                          properties: {
                            id: { type: 'string', enum: [...MODEL_IDS] },
                            displayName: { type: 'string' },
                            license: { type: 'string' },
                            commercialUse: { type: 'boolean' },
                            summary: { type: 'string' },
                          },
                        },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
      '/openapi.json': {
        get: {
          tags: ['the api itself'],
          summary: 'This document.',
          operationId: 'openapi',
          responses: {
            '200': {
              description: 'The OpenAPI document.',
              content: { 'application/json': { schema: { type: 'object' } } },
            },
          },
        },
      },
      '/': {
        get: {
          tags: ['the api itself'],
          summary: 'What this API is.',
          operationId: 'index',
          responses: {
            '200': {
              description: 'An index of the endpoints, or a page about them.',
              content: {
                'application/json': { schema: { type: 'object' } },
                'text/html': { schema: { type: 'string' } },
              },
            },
          },
        },
      },
    },
    'x-parameters': [...PARAMETER_NAMES],
    'x-models': MODEL_IDS.map((id) => ({
      id,
      displayName: MODELS[id].displayName,
      license: MODELS[id].license,
      commercialUse: MODELS[id].commercialUse,
    })),
  };
}
