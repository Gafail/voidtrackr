/**
 * A parser for `multipart/form-data`.
 *
 * The API takes a file, and the way a browser sends a file is a multipart
 * body. This parses one, without a dependency, because the format is small
 * and the rules that matter are few:
 *
 *   - the boundary comes from the Content-Type header and is not allowed to
 *     appear in the content itself, so the parts can be found by scanning
 *     for it;
 *   - each part is a block of headers, a blank line, and then content that
 *     runs until the next boundary;
 *   - the content is bytes, not text, so everything is done on Buffers and
 *     nothing is decoded until it has to be.
 *
 * The parts are returned in order, and each one carries its own headers, so
 * a caller can tell a file part from a field by its filename.
 */

import { ApiError } from './errors';

/** One part of a multipart body. */
export interface MultipartPart {
  /** The part's name, from its Content-Disposition header. */
  name: string;
  /** The filename, if the part is a file. */
  filename?: string;
  /** The part's content type, if it declared one. */
  contentType?: string;
  /** The part's headers, lower-cased names. */
  headers: Record<string, string>;
  /** The part's content. */
  content: Buffer;
}

/** The boundary, taken out of a Content-Type header. */
export function boundaryOf(contentType: string | undefined): string | null {
  if (!contentType) return null;
  const match = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType);
  if (!match) return null;
  const boundary = (match[1] ?? match[2] ?? '').trim();
  return boundary.length > 0 ? boundary : null;
}

/** Read the `name` and `filename` out of a Content-Disposition header. */
function readDisposition(header: string): { name?: string; filename?: string } {
  const name = /name="([^"]*)"/i.exec(header)?.[1];
  const filename = /filename="([^"]*)"/i.exec(header)?.[1];
  return { name, filename };
}

/**
 * Parse a multipart body.
 *
 * @param body        the bytes.
 * @param contentType the request's Content-Type header.
 */
export function parseMultipart(
  body: Buffer,
  contentType: string | undefined,
): MultipartPart[] {
  const boundary = boundaryOf(contentType);
  if (!boundary) {
    throw ApiError.badRequest(
      'The multipart body has no boundary, so it cannot be read.',
      { parameter: 'Content-Type' },
    );
  }

  const delimiter = Buffer.from(`--${boundary}`);
  const parts: MultipartPart[] = [];

  // The body starts with the delimiter, possibly after a preamble.
  let at = body.indexOf(delimiter);
  if (at === -1) {
    throw ApiError.badRequest('The multipart body has no parts in it.');
  }

  while (at !== -1) {
    // Step over the delimiter. What follows is either "--" (the end) or
    // the part's headers.
    let cursor = at + delimiter.length;

    // A trailing "--" marks the end of the body.
    if (body[cursor] === 0x2d && body[cursor + 1] === 0x2d) break;

    // The delimiter is followed by CRLF, and so is the end of the headers.
    if (body[cursor] === 0x0d && body[cursor + 1] === 0x0a) cursor += 2;
    else if (body[cursor] === 0x0a) cursor += 1;

    const headerEnd = body.indexOf('\r\n\r\n', cursor, 'utf8');
    if (headerEnd === -1) {
      throw ApiError.badRequest('A multipart part has no end to its headers.');
    }

    const headerText = body.slice(cursor, headerEnd).toString('utf8');
    const contentStart = headerEnd + 4;

    // The content runs until the CRLF that precedes the next delimiter.
    const next = body.indexOf(delimiter, contentStart);
    if (next === -1) {
      throw ApiError.badRequest('A multipart part is not closed.');
    }

    let contentEnd = next;
    // The CRLF before the delimiter belongs to the delimiter, not the
    // content, so it is taken back off.
    if (contentEnd >= 2 && body[contentEnd - 2] === 0x0d && body[contentEnd - 1] === 0x0a) {
      contentEnd -= 2;
    } else if (contentEnd >= 1 && body[contentEnd - 1] === 0x0a) {
      contentEnd -= 1;
    }

    const headers: Record<string, string> = {};
    for (const line of headerText.split('\r\n')) {
      const colon = line.indexOf(':');
      if (colon === -1) continue;
      headers[line.slice(0, colon).trim().toLowerCase()] = line.slice(colon + 1).trim();
    }

    const disposition = readDisposition(headers['content-disposition'] ?? '');
    parts.push({
      name: disposition.name ?? '',
      filename: disposition.filename,
      contentType: headers['content-type'],
      headers,
      content: body.slice(contentStart, contentEnd),
    });

    at = next;
  }

  return parts;
}
