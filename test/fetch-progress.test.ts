/**
 * Counting the bytes of a download.
 *
 * The progress bar is only worth having if the numbers it shows are the
 * numbers that came down, so this checks the counting against a real
 * server rather than against a mock.
 */

import { createServer } from 'node:http';
import test from 'node:test';
import assert from 'node:assert/strict';

import { downloadFile, formatBytes } from '../src/app/fetch-progress';

/** Serve one body, and run something against it. */
async function withServer(
  body: Buffer,
  status: number,
  run: (url: string) => Promise<void>,
): Promise<void> {
  const server = createServer((_request, response) => {
    response.writeHead(status, { 'Content-Length': String(body.byteLength) });
    response.end(body);
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;

  try {
    await run(`http://127.0.0.1:${port}/weights`);
  } finally {
    server.closeIdleConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

test('the bytes are counted as they arrive, and the file comes back whole', async () => {
  const body = Buffer.alloc(512 * 1024, 7);
  const loaded: number[] = [];
  const totals: number[] = [];

  await withServer(body, 200, async (url) => {
    const bytes = await downloadFile(url, (progress) => {
      loaded.push(progress.loaded);
      totals.push(progress.total);
    });

    assert.equal(bytes.byteLength, body.byteLength, 'the file came back short');
    assert.deepEqual(
      Array.from(bytes.slice(0, 16)),
      Array.from(body.slice(0, 16)),
      'the bytes are not the ones that were sent',
    );
  });

  assert.ok(loaded.length > 0, 'the progress was never reported');
  assert.equal(loaded.at(-1), body.byteLength, 'the count did not reach the end');
  assert.equal(totals.at(-1), body.byteLength, 'the total was not reported');

  // The count only ever goes up.
  for (let i = 1; i < loaded.length; i++) {
    assert.ok(loaded[i] >= loaded[i - 1], 'the count went backwards');
  }
});

test('a file that is not there says so', async () => {
  await withServer(Buffer.from('not found'), 404, async (url) => {
    await assert.rejects(downloadFile(url, () => {}), /the server answered 404/);
  });
});

test('a size is written the way a person would read it', () => {
  assert.equal(formatBytes(0), '0 B');
  assert.equal(formatBytes(999), '999 B');
  assert.equal(formatBytes(174537), '170 kB');
  assert.equal(formatBytes(98705408), '94.1 MB');
  assert.equal(formatBytes(3 * 1024 * 1024 * 1024), '3.00 GB');
});
