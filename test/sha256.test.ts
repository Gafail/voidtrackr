/**
 * Checks for the hashing the model lock depends on.
 *
 * The interesting case is the one that only shows up once the tool is
 * actually served somewhere: `crypto.subtle` only exists in a secure
 * context, and plain HTTP on a LAN is not one. The fallback has to give
 * exactly the same digest, because the fingerprint written into an exported
 * MIDI file must not depend on how the page was served.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { sha256Hex } from '../src/core/model';
import { sha256Bytes, sha256HexFallback } from '../src/core/sha256';

/** The digest as hex, straight from the fallback. */
function fallback(data: Uint8Array | string): string {
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  return sha256HexFallback(bytes);
}

test('the fallback matches the known SHA-256 vectors', () => {
  assert.equal(
    fallback(''),
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
  );
  assert.equal(
    fallback('abc'),
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
  );
  assert.equal(
    fallback('The quick brown fox jumps over the lazy dog'),
    'd7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592',
  );
});

test('the fallback handles the block boundaries', async () => {
  // 55, 56, 63 and 64 bytes are the lengths where the padding rules change.
  for (const length of [0, 1, 54, 55, 56, 63, 64, 65, 119, 120, 1000]) {
    const bytes = new Uint8Array(length);
    for (let i = 0; i < length; i++) bytes[i] = (i * 37) & 0xff;

    const viaFallback = sha256HexFallback(bytes);
    const viaPlatform = Array.from(new Uint8Array(await platformDigest(bytes)))
      .map((byte) => byte.toString(16).padStart(2, '0'))
      .join('');

    assert.equal(viaFallback, viaPlatform, `length ${length}`);
  }
});

test('sha256Hex falls back when crypto.subtle is not available', async () => {
  const data = new TextEncoder().encode('voidtrackr');

  // What the platform gives, with the platform present.
  const withPlatform = await sha256Hex(data);

  // Now take the platform away, the way a page served over plain HTTP on a
  // LAN sees it, and check the answer does not move.
  const original = globalThis.crypto;
  Object.defineProperty(globalThis, 'crypto', {
    value: { ...original, subtle: undefined },
    configurable: true,
    writable: true,
  });

  try {
    const withoutPlatform = await sha256Hex(data);
    assert.equal(withoutPlatform, withPlatform);
    assert.equal(withoutPlatform, fallback(data));
  } finally {
    Object.defineProperty(globalThis, 'crypto', {
      value: original,
      configurable: true,
      writable: true,
    });
  }
});

test('the fallback digests a real model shard the same way', async () => {
  const shard = await readFile(
    resolve(import.meta.dirname, '..', 'public', 'model', 'group1-shard1of1.bin'),
  ).catch(() => null);

  // The weights are fetched separately, so the test skips itself when they
  // are not on disk rather than failing.
  if (!shard) return;

  const bytes = new Uint8Array(shard);
  assert.equal(sha256HexFallback(bytes), await sha256Hex(bytes));
});

/** The platform's own digest, for comparison. */
async function platformDigest(bytes: Uint8Array): Promise<ArrayBuffer> {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return globalThis.crypto.subtle.digest('SHA-256', buffer);
}

// Keep the import of sha256Bytes meaningful: it is the primitive the
// fallback is built on, and a broken one would show up as a wrong digest.
test('sha256Bytes returns 32 bytes', () => {
  assert.equal(sha256Bytes(new Uint8Array(0)).length, 32);
  assert.equal(sha256Bytes(new Uint8Array(1000)).length, 32);
});
