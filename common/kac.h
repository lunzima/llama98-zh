/* kac.h -- the KAC decoder, as this tree uses it.
 *
 * An image is decoded whole. The handle is supplied by the caller; nothing
 * here allocates, and there is no decoder state outside the struct, so several
 * samples can be open at once without sharing anything.
 */
#ifndef LZ_KAC_H
#define LZ_KAC_H

#include <stddef.h>

#include "kac_int.h"

typedef struct {
    /* per-sample state */
    kac_ch         ch[KAC_MAX_CHANNELS];
    const uint8_t *data;
    const int16_t *dqt;
    unsigned       w, n, shift;
    uint32_t       pos;
    uint32_t       total;
    uint32_t       loop_end;
    int            loop_en;
    uint8_t        channels;

    /* current block: tier byte, samples covered, byte base and form */
    uint32_t blk_start, blk_len, blk_off;
    int      ms;
    uint8_t  tier;

    /* header and stream state */
    uint32_t       sample_rate;
    uint32_t       loop_start;
    uint32_t       data_bytes;
    uint32_t       index_stride, index_entries;
    uint8_t        version, flags;
    const uint8_t *init_snap;
    const uint8_t *loop_snap;
    const uint8_t *idx_off;
    const uint8_t *idx_snap;
    uint32_t       loop_blk_start, loop_blk_off;
    int            stream;
} kacdec_t;

void kacdec_init(kacdec_t *restrict d);

/* Open a complete file image; a failed open leaves the handle inert. */
int kacdec_open(kacdec_t *restrict d, const uint8_t *restrict buf, size_t len);

/* Open a stream preamble and load its INIT_STATE. SEEK and LOOP are refused. */
int kacdec_stream_open(kacdec_t *restrict d, const uint8_t *restrict head, size_t len);

/* Supply the next chunk (tier byte plus slices). The buffer is borrowed until
 * the next feed; feeding before the previous chunk is consumed is refused. */
int kacdec_stream_feed(kacdec_t *restrict d, const uint8_t *restrict blk, size_t len);

/* Decode up to `frames`; returns frames produced, 0 at end. Loop mode wraps at
 * LoopEnd and reports KAC_E_NOLOOP if the file has no loop region. A stream
 * returns 0 when the fed chunks are exhausted. */
int32_t kacdec_read(kacdec_t *restrict d, int16_t *restrict out, uint32_t frames);

/* Enable loop wrapping; a source without a loop region reports KAC_E_NOLOOP
 * from kacdec_read(). */
void kacdec_set_loop(kacdec_t *restrict d, int enable);
int  kacdec_seek(kacdec_t *restrict d, uint32_t sample);
void kacdec_close(kacdec_t *restrict d);

uint32_t kacdec_sample_rate(const kacdec_t *d);
uint32_t kacdec_total(const kacdec_t *d);
unsigned kacdec_channels(const kacdec_t *d);
unsigned kacdec_tier(const kacdec_t *d);

#endif /* LZ_KAC_H */
