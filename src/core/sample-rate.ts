/**
 * Reading a file's own sample rate from its container header.
 *
 * This matters for reproducibility. The Web Audio API always decodes to the
 * rate of its `AudioContext`, which is whatever the machine's sound card
 * prefers. If the file is 44.1 kHz and the context is 48 kHz, the browser
 * resamples once before we ever see the samples, and again on the way down to
 * the model's 22.05 kHz. Anywhere else the file is decoded once and resampled
 * once.
 *
 * Decoding at the file's own rate removes that extra step, so the same file
 * produces the same samples in the browser and on the command line.
 */

/** The sample rate to assume when the header cannot be read. */
export const DEFAULT_SAMPLE_RATE = 44100;

/**
 * Find the sample rate of an encoded audio file.
 *
 * Returns null when the format is not one of the ones handled here, in which
 * case the caller should fall back to the default.
 */
export function detectSampleRate(data: ArrayBuffer): number | null {
  const view = new DataView(data);
  const bytes = new Uint8Array(data);

  // RIFF/WAVE: the "fmt " chunk holds the rate at a fixed offset.
  if (readAscii(bytes, 0, 4) === 'RIFF' && readAscii(bytes, 8, 4) === 'WAVE') {
    const rate = readWaveSampleRate(view);
    if (rate) return rate;
  }

  // AIFF: the COMM chunk holds an 80-bit extended float.
  if (readAscii(bytes, 0, 4) === 'FORM' && readAscii(bytes, 8, 4) === 'AIFF') {
    const rate = readAiffSampleRate(view);
    if (rate) return rate;
  }

  // FLAC: STREAMINFO inside the first metadata block.
  if (readAscii(bytes, 0, 4) === 'fLaC') {
    const rate = readFlacSampleRate(view);
    if (rate) return rate;
  }

  // ISO base media (M4A): the sample entry inside stsd.
  if (readAscii(bytes, 4, 4) === 'ftyp') {
    const rate = readMp4SampleRate(view, data.byteLength);
    if (rate) return rate;
  }

  // MP3: the first frame header carries the rate.
  if (readAscii(bytes, 0, 3) === 'ID3') {
    const rate = readMp3SampleRate(view, bytes.length);
    if (rate) return rate;
  } else {
    const rate = readMp3SampleRate(view, bytes.length);
    if (rate) return rate;
  }

  return null;
}

/** Read four ASCII characters. */
function readAscii(bytes: Uint8Array, offset: number, length: number): string {
  let out = '';
  for (let i = 0; i < length; i++) {
    const code = bytes[offset + i];
    if (code === undefined) return '';
    out += String.fromCharCode(code);
  }
  return out;
}

/** The sample rate of a RIFF/WAVE file. */
function readWaveSampleRate(view: DataView): number | null {
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const id = readAscii(new Uint8Array(view.buffer, view.byteOffset + offset, 4), 0, 4);
    const size = view.getUint32(offset + 4, true);
    if (id === 'fmt ' && offset + 16 <= view.byteLength) {
      const rate = view.getUint32(offset + 12, true);
      return rate > 0 ? rate : null;
    }
    offset += 8 + size + (size % 2);
  }
  return null;
}

/** The sample rate of an AIFF file. */
function readAiffSampleRate(view: DataView): number | null {
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const id = readAscii(new Uint8Array(view.buffer, view.byteOffset + offset, 4), 0, 4);
    const size = view.getUint32(offset + 4, false);
    if (id === 'COMM' && offset + 10 <= view.byteLength) {
      return extendedFloat80(view, offset + 8);
    }
    offset += 8 + size + (size % 2);
  }
  return null;
}

/** Decode an 80-bit IEEE extended float, as used by AIFF. */
function extendedFloat80(view: DataView, offset: number): number | null {
  const exponent = view.getUint16(offset, false);
  const mantissaHigh = view.getUint32(offset + 2, false);
  const mantissaLow = view.getUint32(offset + 6, false);
  const sign = exponent & 0x8000 ? -1 : 1;
  const biased = exponent & 0x7fff;
  if (biased === 0 && mantissaHigh === 0 && mantissaLow === 0) return 0;
  if (biased === 0x7fff) return null;

  const exponentValue = biased - 16383;
  const mantissa = mantissaHigh * 2 ** 32 + mantissaLow;
  return sign * mantissa * 2 ** (exponentValue - 63);
}

/** The sample rate of a FLAC file. */
function readFlacSampleRate(view: DataView): number | null {
  if (view.byteLength < 42) return null;
  // The first metadata block starts at offset 4: type in one byte, length in
  // three, then the STREAMINFO block.
  const blockType = view.getUint8(4);
  if (blockType !== 0) return null;
  // 20 bits of sample rate, 3 bits of channels, 5 bits of bits per sample.
  const rate = (view.getUint8(10) << 12) | (view.getUint8(11) << 4) | (view.getUint8(12) >> 4);
  return rate > 0 ? rate : null;
}

/** The sample rate of an MP4/M4A file. */
function readMp4SampleRate(view: DataView, length: number): number | null {
  // Walk the box tree looking for the audio sample entry.
  const findBox = (start: number, end: number, path: string[]): number | null => {
    let offset = start;
    while (offset + 8 <= end) {
      let size = view.getUint32(offset, false);
      const type = readAscii(new Uint8Array(view.buffer, view.byteOffset + offset + 4, 4), 0, 4);
      let headerSize = 8;
      if (size === 1) {
        if (offset + 16 > end) return null;
        size = Number(view.getBigUint64(offset + 8, false));
        headerSize = 16;
      } else if (size === 0) {
        size = end - offset;
      }
      if (size < headerSize) return null;

      const bodyStart = offset + headerSize;
      const bodyEnd = Math.min(offset + size, end);

      if (type === 'stsd') {
        const rate = readMp4SampleEntry(view, bodyStart, bodyEnd);
        if (rate) return rate;
      } else if (['moov', 'trak', 'mdia', 'minf', 'stbl'].includes(type)) {
        const found = findBox(bodyStart, bodyEnd, [...path, type]);
        if (found) return found;
      }

      offset += size;
    }
    return null;
  };

  return findBox(0, length, []);
}

/** The sample rate inside an MP4 audio sample entry. */
function readMp4SampleEntry(view: DataView, start: number, end: number): number | null {
  if (start + 8 > end) return null;
  // version/flags (4), entry count (4), then the entry itself.
  const entryStart = start + 8;
  if (entryStart + 28 > end) return null;
  // The sample rate sits 24 bytes into the entry, after the size and format.
  const rate = view.getUint32(entryStart + 24, false);
  return rate > 0 ? rate : null;
}

/** The sample rate from the first MP3 frame header. */
function readMp3SampleRate(view: DataView, length: number): number | null {
  const RATES = [11025, 12000, 8000, null, 22050, 24000, 16000, null, 44100, 48000, 32000, null];
  const bytes = new Uint8Array(view.buffer, view.byteOffset, length);

  let offset = 0;
  // Skip an ID3 tag if there is one.
  if (readAscii(bytes, 0, 3) === 'ID3' && length > 10) {
    const size =
      ((bytes[6] & 0x7f) << 21) | ((bytes[7] & 0x7f) << 14) | ((bytes[8] & 0x7f) << 7) | (bytes[9] & 0x7f);
    offset = 10 + size;
  }

  while (offset + 4 <= length) {
    if (bytes[offset] === 0xff && (bytes[offset + 1] & 0xe0) === 0xe0) {
      const versionBits = (bytes[offset + 1] >> 3) & 0x03;
      const layerBits = (bytes[offset + 1] >> 1) & 0x03;
      const rateIndex = (bytes[offset + 2] >> 2) & 0x03;
      if (versionBits !== 1 && layerBits !== 0 && rateIndex !== 3) {
        const rate = RATES[rateIndex];
        if (rate) return rate;
      }
    }
    offset += 1;
  }
  return null;
}
