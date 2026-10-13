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
 * The instrument is decided before the first note. The reader forces one
 * GM program on every channel but the drum kit, and this expands exactly
 * one instrument into PCM while the bank is opening, so a mid-piece
 * program change is not something this can answer for.
 *
 * C89.
 */
#ifndef LZ_WAVE_H
#define LZ_WAVE_H

#include "ksb.h"

#define WAVE_VOICES 32

/* `bank` must be expanded (ksb_unpack) and stay alive until wave_close.
 * `rate` is the output rate in frames a second. Returns 0 on success. */
int  wave_open(const KsbBank *bank, long rate);
void wave_close(void);

/* One channel message: status in the low byte, then the two data bytes,
   which is the shape common/smf.h's LZSmfEvent carries. Anything this
   synth has no answer for is dropped - a sampler with no expression
   follows no expression. Returns 0. */
int  wave_message(unsigned long msg);

/* Mix `frames` mono samples. Always fills the whole buffer, with silence if
   nothing is sounding. */
void wave_render(short *out, long frames);

/* Voices in use, and notes dropped because all of them were. Read at the end
   of a piece: `busy` must be zero, because a voice that is never freed is
   invisible until the polyphony runs out. */
int  wave_busy(void);
long wave_stolen(void);
/* Notes with no zone in the bank - the drum channel, mostly. */
long wave_unvoiced(void);

#endif /* LZ_WAVE_H */
