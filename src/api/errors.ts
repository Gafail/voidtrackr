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
  | 'not_found'
  | 'method_not_allowed'
  | 'gone'
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

  /** Something went wrong that the caller cannot fix. */
  static internal(message: string): ApiError {
    return new ApiError('internal_error', 500, message);
  }
}
