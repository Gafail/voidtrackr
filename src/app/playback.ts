/**
 * Playback.
 *
 * A small synthesiser built on the Web Audio API: pitched notes are
 * oscillators with an envelope, and the drum track is generated from noise
 * and sweeps. No samples are needed, which keeps the app self-contained.
 *
 * Notes are scheduled a short time ahead of the playhead, the standard way
 * to keep timing solid in the browser.
 */

import { EditorNote, EditorTrack } from './editor';

const LOOKAHEAD_MS = 25;
const SCHEDULE_AHEAD = 0.15;

/** How long the release of a note lasts, in seconds. */
const RELEASE = 0.06;

export class Player {
  private context: AudioContext | null = null;
  private master: GainNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;

  private tracks: EditorTrack[] = [];

  private playing = false;
  private startedAt = 0;
  private offset = 0;
  private scheduledUntil = 0;
  private timer: number | null = null;
  private frame: number | null = null;

  onEnded: (() => void) | null = null;

  /** Prepare the audio graph. Safe to call before the user has interacted. */
  private ensureContext(): AudioContext {
    if (!this.context) {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.context = new Ctor();
      this.compressor = this.context.createDynamicsCompressor();
      this.compressor.threshold.value = -14;
      this.compressor.ratio.value = 4;
      this.master = this.context.createGain();
      this.master.gain.value = 0.8;
      this.master.connect(this.compressor);
      this.compressor.connect(this.context.destination);
    }
    if (this.context.state === 'suspended') void this.context.resume();
    return this.context;
  }

  setTracks(tracks: EditorTrack[]): void {
    this.tracks = tracks;
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** The current playhead position, in seconds. */
  get currentTime(): number {
    if (!this.playing || !this.context) return this.offset;
    return this.offset + (this.context.currentTime - this.startedAt);
  }

  async play(from?: number): Promise<void> {
    const context = this.ensureContext();
    const start = from ?? this.offset;

    this.offset = Math.max(0, start);
    this.startedAt = context.currentTime;
    this.scheduledUntil = this.offset;
    this.playing = true;

    this.tick();
    this.timer = window.setInterval(() => this.tick(), LOOKAHEAD_MS);
    this.frame = requestAnimationFrame(() => this.checkEnded());
  }

  stop(): void {
    this.playing = false;
    if (this.timer !== null) {
      window.clearInterval(this.timer);
      this.timer = null;
    }
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }
    this.offset = 0;
    this.scheduledUntil = 0;
  }

  /** Move the playhead without playing. */
  seek(time: number): void {
    if (this.playing) {
      this.stop();
      void this.play(time);
      return;
    }
    this.offset = Math.max(0, time);
  }

  dispose(): void {
    this.stop();
    void this.context?.close();
    this.context = null;
  }

  private checkEnded(): void {
    if (!this.playing) return;
    const end = this.contentEnd();
    if (this.currentTime >= end) {
      this.stop();
      this.onEnded?.();
      return;
    }
    this.frame = requestAnimationFrame(() => this.checkEnded());
  }

  private contentEnd(): number {
    let end = 0;
    for (const track of this.tracks) {
      for (const note of track.notes) end = Math.max(end, note.start + note.duration);
    }
    return end + 0.2;
  }

  /** Schedule everything that falls inside the next slice of time. */
  private tick(): void {
    if (!this.playing || !this.context) return;

    const now = this.currentTime;
    const until = now + SCHEDULE_AHEAD;

    for (const track of this.tracks) {
      if (track.muted) continue;
      for (const note of track.notes) {
        if (note.start < this.scheduledUntil || note.start >= until) continue;
        const when = this.startedAt + (note.start - this.offset);
        this.scheduleNote(note, track, when);
      }
    }

    this.scheduledUntil = until;
  }

  private scheduleNote(note: EditorNote, track: EditorTrack, when: number): void {
    const context = this.ensureContext();
    if (!this.master) return;

    if (track.kind === 'drums') {
      this.scheduleDrum(note, when);
      return;
    }

    const gain = context.createGain();
    const filter = context.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = track.kind === 'bass' ? 900 : 3200;

    const oscillator = context.createOscillator();
    oscillator.type = track.kind === 'bass' ? 'sine' : 'triangle';
    oscillator.frequency.value = 440 * Math.pow(2, (note.pitch - 69) / 12);

    // A second oscillator a little detuned gives the tone some width.
    const shimmer = context.createOscillator();
    shimmer.type = 'sine';
    shimmer.frequency.value = oscillator.frequency.value * 2;
    const shimmerGain = context.createGain();
    shimmerGain.gain.value = track.kind === 'bass' ? 0 : 0.18;

    const peak = Math.max(0.02, note.velocity) * (track.kind === 'chords' ? 0.45 : 0.6);
    const attack = 0.008;
    const releaseStart = when + Math.max(attack, note.duration);

    gain.gain.setValueAtTime(0.0001, when);
    gain.gain.exponentialRampToValueAtTime(peak, when + attack);
    gain.gain.setValueAtTime(peak, releaseStart);
    gain.gain.exponentialRampToValueAtTime(0.0001, releaseStart + RELEASE);

    oscillator.connect(gain);
    shimmer.connect(shimmerGain);
    shimmerGain.connect(gain);
    gain.connect(filter);
    filter.connect(this.master);

    oscillator.start(when);
    shimmer.start(when);
    oscillator.stop(releaseStart + RELEASE + 0.02);
    shimmer.stop(releaseStart + RELEASE + 0.02);
  }

  private scheduleDrum(note: EditorNote, when: number): void {
    const context = this.ensureContext();
    if (!this.master) return;

    if (note.pitch === 36) {
      // Kick: a fast pitch drop with a long decay.
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(130, when);
      oscillator.frequency.exponentialRampToValueAtTime(45, when + 0.12);
      gain.gain.setValueAtTime(0.0001, when);
      gain.gain.exponentialRampToValueAtTime(Math.max(0.05, note.velocity), when + 0.005);
      gain.gain.exponentialRampToValueAtTime(0.0001, when + 0.28);
      oscillator.connect(gain);
      gain.connect(this.master);
      oscillator.start(when);
      oscillator.stop(when + 0.3);
      return;
    }

    // Snare and hi-hat are filtered noise.
    const length = 0.22;
    const buffer = context.createBuffer(1, Math.ceil(context.sampleRate * length), context.sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < channel.length; i++) {
      channel[i] = Math.random() * 2 - 1;
    }

    const source = context.createBufferSource();
    source.buffer = buffer;

    const filter = context.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = note.pitch === 38 ? 1800 : 8000;
    filter.Q.value = note.pitch === 38 ? 0.9 : 1.4;

    const gain = context.createGain();
    const level = Math.max(0.03, note.velocity) * (note.pitch === 38 ? 0.5 : 0.28);
    gain.gain.setValueAtTime(level, when);
    gain.gain.exponentialRampToValueAtTime(0.0001, when + (note.pitch === 38 ? 0.18 : 0.06));

    source.connect(filter);
    filter.connect(gain);
    gain.connect(this.master);
    source.start(when);
    source.stop(when + length);

    // A snare also has a tone in it.
    if (note.pitch === 38) {
      const oscillator = context.createOscillator();
      const toneGain = context.createGain();
      oscillator.type = 'triangle';
      oscillator.frequency.value = 190;
      toneGain.gain.setValueAtTime(level * 0.5, when);
      toneGain.gain.exponentialRampToValueAtTime(0.0001, when + 0.12);
      oscillator.connect(toneGain);
      toneGain.connect(this.master);
      oscillator.start(when);
      oscillator.stop(when + 0.14);
    }
  }
}
