/**
 * The API's errors.
 *
 * Every failure the caller can cause comes back as the same JSON shape, with
 * a machine-readable code and an HTTP status that matches what went wrong.
 * A client should be able to branch on the code without reading the
 * message.
 */

/** The codes the API answers with. */
export type ApiErrorCode =
  | 'bad_request'
  | 'unsupported_media_type'
  | 'not_found'
  | 'method_not_allowed'
  | 'payload_too_large'
  | 'unprocessable_entity'
  | 'model_unavailable'
  | 'unauthorized'
  | 'busy'
  | 'internal_error';

/** The body every error comes back as. */
export interface ApiErrorBody {
  error: {
    code: ApiErrorCode;
    message: string;
    details?: Record<string, unknown>;
  };
}

/** An error that carries its own status code and machine-readable code. */
export class ApiError extends Error {
  readonly code: ApiErrorCode;
  readonly status: number;
  readonly details?: Record<string, unknown>;

  constructor(
    code: ApiErrorCode,
    status: number,
    message: string,
    details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.details = details;
  }

  /** The body to send back. */
  toBody(): ApiErrorBody {
    return {
      error: {
        code: this.code,
        message: this.message,
        ...(this.details ? { details: this.details } : {}),
      },
    };
  }

  /** A bad request: the caller sent something that does not make sense. */
  static badRequest(message: string, details?: Record<string, unknown>): ApiError {
    return new ApiError('bad_request', 400, message, details);
  }

  /** The media type is not one the API can take. */
  static unsupportedMediaType(
    message: string,
    details?: Record<string, unknown>,
  ): ApiError {
    return new ApiError('unsupported_media_type', 415, message, details);
  }

  /** Nothing is at that address. */
  static notFound(message: string): ApiError {
    return new ApiError('not_found', 404, message);
  }

  /** That address exists, but not for that method. */
  static methodNotAllowed(method: string, allowed: string[]): ApiError {
    return new ApiError(
      'method_not_allowed',
      405,
      `${method} is not allowed here.`,
      { allowed },
    );
  }

  /** The upload is bigger than the server will take. */
  static payloadTooLarge(message: string, details?: Record<string, unknown>): ApiError {
    return new ApiError('payload_too_large', 413, message, details);
  }

  /** The audio could not be read, or the request was otherwise well-formed
   *  but could not be carried out. */
  static unprocessable(message: string, details?: Record<string, unknown>): ApiError {
    return new ApiError('unprocessable_entity', 422, message, details);
  }

  /** The model asked for is not on this machine. */
  static modelUnavailable(message: string, details?: Record<string, unknown>): ApiError {
    return new ApiError('model_unavailable', 503, message, details);
  }

  /** The token is missing or wrong. */
  static unauthorized(message: string): ApiError {
    return new ApiError('unauthorized', 401, message);
  }

  /** The server is already working on as much as it can. */
  static busy(message: string, details?: Record<string, unknown>): ApiError {
    return new ApiError('busy', 503, message, details);
  }

  /** Something went wrong that the caller cannot fix. */
  static internal(message: string): ApiError {
    return new ApiError('internal_error', 500, message);
  }
}
