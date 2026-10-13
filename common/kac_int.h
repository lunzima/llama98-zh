/* kac_int.h -- the KAC decoder's own material: the tier table, prediction,
 * step, saturation, the snapshot layout and the little endian reads.
 *
 * Everything here is a read or a state update. The byte and snapshot
 * writers the format defines are not part of a reader, so they are not
 * here. The width types come from ../src/lz_int.h, which is this tree's
 * one authority for them and also answers `restrict` where the compiler
 * has no such keyword. The functions are `static` without `inline`: the
 * floor has no `inline` keyword, and gcc inlines a one-translation-unit
 * static at -O2 without being asked.
 */

#ifndef LZ_KAC_INT_H
#define LZ_KAC_INT_H

#include "../src/lz_int.h"

/* Arithmetic right shift of negative values is implementation-defined in C99,
 * not undefined: the prediction and the weight update rely on it. */
typedef char kac_assert_arith_shift[((-1 >> 1) == -1) ? 1 : -1];
typedef char kac_assert_i32[(sizeof(int32_t) == 4) ? 1 : -1];

#include "kac_dqt.h"

#define KAC_DQ_T1 kac_dq_t1
#define KAC_DQ_T2 kac_dq_t2
#define KAC_DQ_T3 kac_dq_t3
#define KAC_DQ_T4 kac_dq_t4
#define KAC_DQ_T5 kac_dq_t5
#define KAC_DQ_T6 kac_dq_t6

/* One tier: the bit width w of its amplitudes and the samples per slice n.
 * There is no id field: the array is indexed by tier - 1 everywhere, so the
 * position IS the identity, and the byte written to the stream is the tier
 * number `i + 1` rather than a field read back out. */
typedef struct {
    unsigned w, n;
} kac_tier;

/* Tier ids the format admits. T6 is the w = 1 tier: 4 + 60*1 == 64, so a
 * slice is still one 8 byte word with no carry, and 64/60 bit/sample takes
 * the ladder below T1's floor. Its id is the highest while its rate is the
 * lowest, so the array is not sorted by rate and every walk of the ladder has
 * to go through kac_rate_order instead of counting up the ids. */
#define KAC_TIER_MIN 1
#define KAC_TIER_MAX 6

static const kac_tier kac_tiers[6] = {{2, 30}, {3, 20}, {4, 15}, {5, 12}, {6, 10}, {1, 60}};

#define KAC_HDR_SIZE 36u
#define KAC_SNAP_SIZE 16u
#define KAC_SLICE_SIZE 8u
#define KAC_FLAG_LOOP 0x01u
#define KAC_FLAG_INIT 0x02u
#define KAC_FLAG_SEEK 0x04u
/* bit3 CHUNKED: the data area is `[tier byte][this block's slices]...`, one
 * block per KAC_CHUNK_SAMPLES samples, and Tier carries block 0's tier. With
 * the bit clear the data area is plain slices at the header's Tier. */
#define KAC_FLAG_CHUNKED 0x08u
#define KAC_FLAG_RESERVED 0xF0u
#define KAC_VERSION 1u

/* Block grid of the CHUNKED data area; the last block takes the remainder.
 *
 * 960 samples is a whole slice at every tier (960 = 30*32 = 20*48 = 15*64 =
 * 12*80 = 10*96), so a block boundary is always a slice boundary whatever tier
 * the block carries. That property alone does *not* single 960 out -- 120, 480
 * and 1920 all share it. What fixes the number is rate control: one tier byte
 * per block costs 8/960 bit/sample, so the block must be long enough to
 * amortise it and short enough for the tier to follow the material.
 */
#define KAC_CHUNK_SAMPLES 960u

/* Default initial state; both ends must use it when INIT_STATE is absent.
 * Weight 16383 rather than 16384 keeps the worst-case accumulator sum below
 * INT32_MAX (4 * 16384 * 32768 == 2^31). */
#define KAC_INIT_W0 0
#define KAC_INIT_W1 0
#define KAC_INIT_W2 (-8192)
#define KAC_INIT_W3 16383
#define KAC_W_LIM 16383

#define KAC_MAX_CHANNELS 2

/* TotalSamples ceiling. It is what keeps DataBytes, the index area and the whole
 * file inside signed 32-bit, so a 32-bit target can hold the file and no 64-bit
 * arithmetic is needed anywhere. An encoder must not produce a file above it and
 * a decoder must reject one. */
#define KAC_MAX_TOTAL (1u << 28)

/* Negative return codes shared by both ends. */
enum {
    KAC_OK        = 0,
    KAC_E_ARG     = -1,
    KAC_E_MAGIC   = -2,
    KAC_E_VERSION = -3,
    KAC_E_FLAGS   = -4,
    KAC_E_CHANNEL = -5,
    KAC_E_TIER    = -6,
    KAC_E_TOTAL   = -7,
    KAC_E_LOOP    = -8,
    KAC_E_DATA    = -9,
    KAC_E_SIZE    = -10,
    KAC_E_INDEX   = -11,
    KAC_E_RANGE   = -12,
    KAC_E_SHORT   = -13, /* reserved, never returned: a well formed file cannot
                          * run out of data, and the read path clamps instead */
    KAC_E_NOLOOP = -14
};

typedef struct {
    int32_t w[4];
    int32_t h[4];
} kac_ch;

static LZ_MAYBE_UNUSED int32_t kac_sat_w(int32_t v)
{
    return v > KAC_W_LIM ? KAC_W_LIM : (v < -KAC_W_LIM ? -KAC_W_LIM : v);
}

static LZ_MAYBE_UNUSED int32_t kac_clamp16(int32_t s)
{
    return s > 32767 ? 32767 : (s < -32768 ? -32768 : s);
}

static LZ_MAYBE_UNUSED int32_t kac_predict(const kac_ch *restrict c)
{
    return (c->w[0] * c->h[0] + c->w[1] * c->h[1] + c->w[2] * c->h[2] + c->w[3] * c->h[3]) >> 13;
}

static LZ_MAYBE_UNUSED int32_t kac_step(kac_ch *restrict c, int32_t residual)
{
    int32_t pred = kac_predict(c);
    int32_t s    = kac_clamp16(pred + residual);
    int32_t d    = residual >> 4;
    int32_t s0   = (c->h[0] < 0) ? -d : d;
    int32_t s1   = (c->h[1] < 0) ? -d : d;
    int32_t s2   = (c->h[2] < 0) ? -d : d;
    int32_t s3   = (c->h[3] < 0) ? -d : d;
    c->w[0]      = kac_sat_w(c->w[0] + s0);
    c->w[1]      = kac_sat_w(c->w[1] + s1);
    c->w[2]      = kac_sat_w(c->w[2] + s2);
    c->w[3]      = kac_sat_w(c->w[3] + s3);
    c->h[0]      = c->h[1];
    c->h[1]      = c->h[2];
    c->h[2]      = c->h[3];
    c->h[3]      = s;
    return s;
}

static LZ_MAYBE_UNUSED void kac_ch_init(kac_ch *restrict c)
{
    c->w[0] = KAC_INIT_W0;
    c->w[1] = KAC_INIT_W1;
    c->w[2] = KAC_INIT_W2;
    c->w[3] = KAC_INIT_W3;
    c->h[0] = 0;
    c->h[1] = 0;
    c->h[2] = 0;
    c->h[3] = 0;
}

static LZ_MAYBE_UNUSED const int16_t *kac_pick_dqt(unsigned tier)
{
    switch (tier) {
    case 1u:
        return &KAC_DQ_T1[0][0];
    case 2u:
        return &KAC_DQ_T2[0][0];
    case 3u:
        return &KAC_DQ_T3[0][0];
    case 4u:
        return &KAC_DQ_T4[0][0];
    case 5u:
        return &KAC_DQ_T5[0][0];
    default:
        return &KAC_DQ_T6[0][0];
    }
}

typedef struct {
    int16_t h[4];
    int16_t w[4];
} kac_snap;

static LZ_MAYBE_UNUSED uint16_t kac_get16(const uint8_t *restrict p)
{
    return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8));
}

static LZ_MAYBE_UNUSED uint32_t kac_get32(const uint8_t *restrict p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static LZ_MAYBE_UNUSED uint64_t kac_get64(const uint8_t *restrict p)
{
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

/* Weights from a snapshot are file data and may be out of range, so each one
 * is saturated on load; history is within int16 already. The four calls are
 * written out one by one so that their count stays visible in the disassembly. */
static LZ_MAYBE_UNUSED void kac_snap_load(kac_ch *restrict c, const uint8_t *restrict p)
{
    c->h[0] = (int32_t)(int16_t)kac_get16(p + 0);
    c->h[1] = (int32_t)(int16_t)kac_get16(p + 2);
    c->h[2] = (int32_t)(int16_t)kac_get16(p + 4);
    c->h[3] = (int32_t)(int16_t)kac_get16(p + 6);
    c->w[0] = kac_sat_w((int32_t)(int16_t)kac_get16(p + 8));
    c->w[1] = kac_sat_w((int32_t)(int16_t)kac_get16(p + 10));
    c->w[2] = kac_sat_w((int32_t)(int16_t)kac_get16(p + 12));
    c->w[3] = kac_sat_w((int32_t)(int16_t)kac_get16(p + 14));
}

#endif /* LZ_KAC_INT_H */
