/**
 * Downloading a file with a running count of the bytes.
 *
 * The model's weights are large — the piano model's are 95 MB, and 94 MB of
 * that is one sidecar file — and the browser says nothing at all while it
 * fetches them. Counting the bytes as they arrive is what makes the
 * progress bar move instead of sitting still.
 */

/** How far a download has got. */
export interface DownloadProgress {
  /** The bytes read so far. */
  loaded: number;
  /** The size of the file, when the server says. Zero when it does not. */
  total: number;
}

/**
 * Read a URL, reporting the bytes as they arrive.
 *
 * The bytes are kept, because they are what the model is built from: the
 * runtime is handed them rather than being sent to fetch them a second
 * time. The browser's cache will not do that job on its own — measured on
 * 12 November 2026, the 980 kB piano graph was served from the cache on the
 * second request and the 94 MB sidecar was not, so the sidecar crossed the
 * network twice.
 */
export async function downloadFile(
  url: string,
  onProgress: (progress: DownloadProgress) => void,
): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`the server answered ${response.status}`);
  }

  const body = response.body;
  if (!body) {
    // No stream to read: the whole file arrives at once, so there is
    // nothing to count on the way.
    const bytes = new Uint8Array(await response.arrayBuffer());
    onProgress({ loaded: bytes.byteLength, total: bytes.byteLength });
    return bytes;
  }

  const announced = Number(response.headers.get('Content-Length') ?? 0);
  const reader = body.getReader();

  // The size is known, so the bytes go straight into one buffer. Collecting
  // chunks and joining them afterwards would hold two copies of a 94 MB
  // file at once.
  const sized = announced > 0 ? new Uint8Array(announced) : null;
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done || !value) break;
    if (sized) {
      if (loaded + value.byteLength > sized.length) {
        throw new Error('the server sent more bytes than it said');
      }
      sized.set(value, loaded);
    } else {
      chunks.push(value);
    }
    loaded += value.byteLength;
    onProgress({ loaded, total: announced });
  }

  if (announced > 0 && loaded !== announced) {
    // A truncated file would fail the hash check with a misleading message,
    // so say what actually went wrong.
    throw new Error(`the server sent ${loaded} bytes and said ${announced}`);
  }

  if (sized) {
    onProgress({ loaded, total: announced });
    return sized;
  }

  const joined = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  onProgress({ loaded, total: loaded });
  return joined;
}

/**
 * A size in bytes, as something a person can read.
 *
 * The weights are large enough that "42.3 MB" says more than "44354560".
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} kB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
