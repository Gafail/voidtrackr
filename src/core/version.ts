/**
 * Identity of the pipeline.
 *
 * Every exported MIDI file carries this version and the fingerprint of the
 * model that produced it, so a file can always be traced back to the exact
 * code and weights that made it.
 *
 * This is what makes the output reproducible in a practical sense: the same
 * input, with the same pipeline version and the same model, gives the same
 * file. If either changes, the stamp changes too, and the difference is
 * visible in the file itself rather than being a surprise later.
 */

/**
 * Bumped whenever the pipeline's output could change.
 *
 * 0.2.0 fixed the frame-to-time mapping. The notes of a long recording used
 * to drift about 5 ms for every second of audio, so a three-minute song ended
 * up roughly half a second out of step with its own beginning. The notes
 * also stopped about a second before the end of the recording.
 *
 * 0.3.0 changed nothing about the notes. The tool renamed itself to
 * voidtrackr, and the name is part of the provenance stamp, so the bytes of
 * every exported file changed even though the transcription did not.
 *
 * 0.3.1 brought the Tsumugi and Piano ports back in line with the code they
 * were ported from, in the places where the two could disagree about the
 * edges of a recording: the silence gate measured the wrong thing, the
 * Viterbi walked into the zero padding at the end of a window, a merge rule
 * kept notes that upstream drops, and a note could start before the
 * recording did. The notes of most recordings do not change; the ones that
 * do are at the quiet ends and the very end of a file.
 *
 * 0.4.0 changed the default model from Basic Pitch to ByteDance's piano
 * model, after measuring all five over the user's own 19 tracks. The piano
 * model shared 63.9 % of Okarina's notes where Basic Pitch shared 40.6 %,
 * and its licence allows the output to be sold. A conversion that does not
 * name a model now produces different notes, and the stamp says so.
 *
 * 0.5.0 added the frequency range: a pair of bounds in hertz that choose
 * which notes are kept. A conversion that does not ask for a range produces
 * the same notes as before, so only the stamp changes. The range is applied
 * to the notes rather than to the audio, because filtering the audio was
 * measured and it is much worse — see `docs/model-research.md`, section 6.
 *
 * 0.6.0 added the pianify stage: the notes are arranged for two hands, the
 * re-triggers are folded together, the chords are divided between a left and
 * a right hand, and what neither hand could hold is dropped. A conversion
 * that does not ask for it produces the same notes as before, so only the
 * stamp changes. See `docs/pianify.md`.
 *
 * 0.7.0 corrected the arrangement. The 0.6.0 stage put the notes in the
 * wrong hands: it worked the split point out from the material, which on a
 * mix lands in the treble, so the left hand played the melody up to MIDI 87
 * and the right hand played the bass down to 46. The split is at middle C
 * now, where the grand staff splits. The fold of a re-trigger was also a
 * chain, which made notes of seventeen seconds out of strikes of ten
 * milliseconds; it is pairwise now, and a folded note is never longer than
 * the longest strike it swallowed. Notes shorter than sixty milliseconds are
 * lengthened, and the notes of a chord are placed melody first, so the top
 * of a chord is the note that survives. A conversion that does not ask for
 * the arrangement produces the same notes as before, so only the stamp
 * changes.
 */
export const PIPELINE_VERSION = '0.7.0';

/** Where the weights came from, for the record. */
export const MODEL_SOURCE =
  'https://raw.githubusercontent.com/spotify/basic-pitch-ts/main/model';

/** What produced a given output. */
export interface Provenance {
  /** Version of the pipeline code. */
  pipeline: string;
  /** Identifier of the model. */
  model: string;
  /** Hash of the model's weights. */
  fingerprint: string;
}

/** The name this tool stamps its output with. */
export const TOOL_NAME = 'voidtrackr';

/**
 * Build the stamp that is written into every exported MIDI file.
 *
 * The fingerprint is the full SHA-256 of the model's manifest, so the file
 * can be matched against `model.lock.json` without guessing.
 */
export function provenanceStamp(provenance: Provenance): string {
  return (
    `${TOOL_NAME} ${provenance.pipeline}` +
    ` | model ${provenance.model}` +
    ` | sha256:${provenance.fingerprint}`
  );
}

/**
 * Build the stamp for a loaded model.
 *
 * The pipeline version comes from here, so it is the one place to change when
 * the pipeline's behaviour changes.
 */
export function provenanceFor(model: { id: string; fingerprint: string }): string {
  return provenanceStamp({
    pipeline: PIPELINE_VERSION,
    model: model.id,
    fingerprint: model.fingerprint,
  });
}
