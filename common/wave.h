/* wave.h - a software sampler over a KSB bank.
 *
 * Reads a KSB bank and mixes it, which is what makes a bank sound the
 * same on a machine with no MIDI hardware at all and on one whose
 * synthesiser is whatever the mapper happens to point at: this plays
 * the bytes, so it plays the same thing either way.
 *
 * IT DOES NOT TOUCH THE SOUND CARD. wave_render() fills a buffer and
 * that is all. So the whole synth is testable with no device: render
 * into memory, measure it.
 *
 * THE BUFFER IS ALSO THE CLOCK. A caller renders N frames and then
 * advances the sequence by N/rate seconds. That is exact, where a
 * sleep-and-measure loop is not - Sleep(1) returns after anywhere from
 * 1 to 15 ms and the error accumulates into something that sounds like
 * a bad tempo map.
 *
 * EXPANSION HAPPENS BEFORE THE FIRST NOTE, one of two ways. With a single
 * program, the reader forces that program on every channel but the drum
 * kit and ksb_unpack decodes just its samples, so a mid-piece program
 * change does not switch instruments. With the whole bank expanded
 * (ksb_unpack_all), program changes select real instruments.
 *
 * BANK SELECT IS PART OF THE PATCH. CC0 and CC32 are kept per channel and
 * a note-on resolves (MSB, LSB, program) through ksb_zone_at, so a bank
 * carrying a file's own instruments plays them; a bank that does not carry
 * the selected pair falls back to its GM set. The drum channel keeps the
 * program-128 kit in bank 0 and ignores a bank select of its own.
 *
 * C89.
 */
#ifndef LZ_WAVE_H
#define LZ_WAVE_H

#include "ksb.h"

/* Voice pool. The mixer walks this many entries every block, so it is a
 * compile-time size. 192 rather than 128 because a note-on starts one
 * voice per matching REGION, and a bank that layers velocity splits and
 * stacked zones spends several voices on one key: on the onestop
 * arrangement the 128-entry pool dropped 15 notes in the dense
 * passages, and 192 dropped none. A tighter target can override the
 * macro when compiling. */
#ifndef WAVE_VOICES
#define WAVE_VOICES 192
#endif

/* `bank` must be expanded (ksb_unpack or ksb_unpack_all) and stay alive
 * until wave_close. `rate` is the output rate in frames a second. Returns 0
 * on success. */
int  wave_open(const KsbBank *bank, long rate);
void wave_close(void);

/* One channel message: status in the low byte, then the two data bytes,
   which is the shape common/smf.h's LZSmfEvent carries. Anything this
   synth has no answer for is dropped - a sampler with no expression
   follows no expression. Returns 0. */
int  wave_message(unsigned long msg);

/* Mix `frames` STEREO frames, interleaved left then right, so the buffer
   takes 2 * frames shorts. Always fills the whole buffer, with silence if
   nothing is sounding. Stereo because a DLS's pan, reverb and chorus are
   stereo effects, and the caller rather than the mixer decides the channel
   count. */
void wave_render(short *out, long frames);

/* Voices in use, and notes dropped because all of them were. Read at the end
   of a piece: `busy` must be zero, because a voice that is never freed is
   invisible until the polyphony runs out. */
int  wave_busy(void);
long wave_stolen(void);
/* Notes with no zone in the bank - the drum channel, mostly. */
long wave_unvoiced(void);

#endif /* LZ_WAVE_H */
