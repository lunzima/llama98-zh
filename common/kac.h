/* kac.h -- the KAC decoder, as this tree uses it.
 *
 * A KAC image is one sample: a whole file, header and all, whose own
 * TotalSamples is the frame count the bank's record carries. The reader
 * hands this a pointer into the bank and a length and gets 16-bit PCM
 * back, which is why the bank stores the image's length rather than its
 * frame count.
 *
 * The handle is supplied by the caller and is small (200 bytes stereo,
 * and the bank is mono); nothing here allocates, and there is no
 * decoder state outside this struct, so several samples can be open at
 * once without sharing anything.
 *
 * This interface reads one thing: a whole image of known length. There is no
 * encoder in this tree -- the images are produced when the bank is built, by
 * the codec's own binary -- and nothing hands this a partial image.
 */
#ifndef LZ_KAC_H
#define LZ_KAC_H

#include <stddef.h>

#include "kac_int.h"

typedef struct {
    uint32_t       sample_rate;
    uint32_t       total;
    uint32_t       loop_start, loop_end;
    uint32_t       data_bytes;
    uint32_t       index_stride, index_entries;
    uint8_t        version, flags, channels, tier;
    unsigned       w, n, shift;
    const uint8_t *data;
    const uint8_t *init_snap;
    const uint8_t *loop_snap;
    const uint8_t *idx_off;
    const uint8_t *idx_snap;
    const int16_t *dqt;
    kac_ch         ch[KAC_MAX_CHANNELS];
    uint32_t       pos;
    int            loop_en;

    /* CHUNKED: the current block and the one LoopStart falls in. blk_off is
     * the data-area offset of the block's first slice; its tier byte sits at
     * blk_off - 1. */
    int      chunked;
    uint32_t blk_start, blk_len, blk_off;
    uint32_t loop_blk_start, loop_blk_off;
} kacdec_t;

void kacdec_init(kacdec_t *restrict d);

/* Parse and validate a whole image. The buffer must outlive the handle:
 * the decoder reads slices straight out of it and keeps no copy. Returns 0
 * or a negative KAC_E_*. A failed parse leaves the handle inert rather
 * than half of a previous use, so it decodes nothing and can be reopened
 * without being re-initialised. */
int kacdec_open(kacdec_t *restrict d, const uint8_t *restrict buf, size_t len);

/* Decode up to `frames` frames into interleaved output and return how many
 * were produced: the request is clamped to what is left, so a fixed block
 * size works and 0 marks the end. */
int32_t kacdec_read(kacdec_t *restrict d, int16_t *restrict out, uint32_t frames);

void kacdec_close(kacdec_t *restrict d);

uint32_t kacdec_sample_rate(const kacdec_t *d);
uint32_t kacdec_total(const kacdec_t *d);
unsigned kacdec_channels(const kacdec_t *d);
unsigned kacdec_tier(const kacdec_t *d);

/* The loop region and the seek index are features the file's own header
 * declares, so the decoder of a file honours them. Nothing in this tree calls
 * either: a bank sample's loop points live in the bank's own record and the
 * player wraps there. A decoder that could not honour an image's own LOOP flag
 * would not be a decoder of this format, so they are here regardless.
 *
 * kacdec_set_loop(1) on an image with no loop region does not wrap, and every
 * read then returns KAC_E_NOLOOP; nothing checks that when the mode is turned
 * on, so this is where it surfaces. kacdec_seek needs the seek index and
 * returns KAC_E_INDEX without one. */
void kacdec_set_loop(kacdec_t *restrict d, int enable);
int  kacdec_seek(kacdec_t *restrict d, uint32_t sample);

#endif /* LZ_KAC_H */
