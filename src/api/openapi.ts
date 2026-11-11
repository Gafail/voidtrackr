/**
 * The OpenAPI document the API serves.
 *
 * It is built from the same numbers the code uses — the model list, the
 * parameter table, the version — so it cannot drift away from what the
 * server actually does.
 *
 * There is no conversion endpoint in it, because there is none in the
 * server. The document says so in its description, so that a client
 * reading it learns the rule rather than only finding out by trying.
 */

import { MODEL_IDS, DEFAULT_MODEL, MODELS } from '../core/registry';
import { PARAMETERS } from '../core/parameters';
import { PIPELINE_VERSION } from '../core/version';

/** The document, as a plain object. */
export function openApiDocument(): Record<string, unknown> {
  const parameterList = PARAMETERS.map((parameter) => ({
    name: parameter.name,
    in: 'query',
    required: false,
    description: parameter.description,
    schema: {
      type: parameter.type === 'model' ? 'string' : parameter.type,
      ...(parameter.allowed ? { enum: [...parameter.allowed] } : {}),
      default: parameter.default,
    },
  }));

  const modelList = MODEL_IDS.map((id) => ({
    id,
    displayName: MODELS[id].displayName,
    license: MODELS[id].license,
    commercialUse: MODELS[id].commercialUse,
    weightsUrl: MODELS[id].weightsPath,
    weightsRoot: MODELS[id].weightsRoot,
    fetchCommand: MODELS[id].fetchCommand,
  }));

  return {
    openapi: '3.1.0',
    info: {
      title: 'voidtrackr',
      version: PIPELINE_VERSION,
      summary: 'The catalogue of what voidtrackr can do.',
      description:
        'This API describes the tool; it does not run it. No endpoint here ' +
        'accepts a recording, and the server never reads a request body: a ' +
        'recording is converted on the machine that holds it, in the browser ' +
        'or on the command line. The model weights are served from this ' +
        'origin, which is what a client downloads.',
    },
    servers: [{ url: '/api', description: 'This server.' }],
    paths: {
      '/': {
        get: {
          summary: 'The index of the API, or a page about it.',
          description:
            'Answers with JSON, or with a page when the caller asks for HTML.',
          responses: {
            '200': {
              description: 'The index.',
              content: {
                'application/json': { schema: { type: 'object' } },
                'text/html': { schema: { type: 'string' } },
              },
            },
          },
        },
      },
      '/health': {
        get: {
          summary: 'Whether the server is up.',
          responses: {
            '200': {
              description: 'The state of the server.',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      status: { type: 'string' },
                      pipeline: { type: 'string' },
                      defaultModel: { type: 'string' },
                      models: { type: 'array', items: { type: 'string' } },
                      uptimeSeconds: { type: 'number' },
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
          summary: 'The models, their licences and where their weights are.',
          description:
            'The licence on each entry decides whether the output can be ' +
            'sold, so it is stated rather than buried.',
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
                            weightsUrl: { type: 'string' },
                            weightsRoot: { type: 'string' },
                            fetchCommand: { type: 'string' },
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
      '/parameters': {
        get: {
          summary: 'The parameters a conversion takes, and their defaults.',
          description:
            'These are applied by the client. The names are the ones the ' +
            'site uses and the ones the command line takes, so a value set ' +
            'in one means the same thing in the others.',
          responses: {
            '200': {
              description: 'The parameter table.',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      parameters: {
                        type: 'array',
                        items: {
                          type: 'object',
                          properties: {
                            name: { type: 'string' },
                            type: {
                              type: 'string',
                              enum: ['model', 'enum', 'boolean', 'number'],
                            },
                            default: {},
                            description: { type: 'string' },
                            allowed: {
                              type: 'array',
                              items: { type: 'string' },
                            },
                            flag: { type: 'string' },
                            unit: { type: 'string' },
                          },
                        },
                      },
                      note: { type: 'string' },
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
          summary: 'This document.',
          responses: {
            '200': {
              description: 'The OpenAPI document.',
              content: { 'application/json': { schema: { type: 'object' } } },
            },
          },
        },
      },
    },
    components: {
      parameters: {
        conversionParameters: {
          description:
            'The parameters a conversion takes. They are applied by the ' +
            'client, on the machine that holds the recording.',
          in: 'query',
          name: 'parameters',
          schema: { type: 'object' },
          example: { model: DEFAULT_MODEL, drums: true },
        },
      },
    },
    'x-voidtrackr': {
      acceptsRecordings: false,
      parameters: parameterList,
      models: modelList,
      note:
        'No endpoint accepts a recording. The conversion runs on the client, ' +
        'and the model weights are fetched from this origin.',
    },
  };
}
