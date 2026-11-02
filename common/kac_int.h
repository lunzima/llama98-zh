/* kac_int.h -- the KAC decoder's own material: tables, prediction, step,
 * saturation, the snapshot layout and the little endian reads. */
#ifndef LZ_KAC_INT_H
#define LZ_KAC_INT_H

#include "../src/lz_int.h"

/* Arithmetic right shift of negative values is implementation-defined in C99,
 * not undefined: the prediction and the weight update rely on it. */
typedef char kac_assert_arith_shift[((-1 >> 1) == -1) ? 1 : -1];
typedef char kac_assert_i32[(sizeof(int32_t) == 4) ? 1 : -1];

#include "kac_dqt.h"

/* One tier: amplitude width w and slice length n. The array index is the tier
 * identity; the stream carries tier number i + 1. */
typedef struct {
    unsigned w, n;
} kac_tier;

/* Format tiers T1..T7. For every full slice, 64 bits over n samples make the
 * rate (64/60)*w. T7 leaves four spare bits and is the top of the ladder. */
#define KAC_TIER_MIN 1
#define KAC_TIER_MAX 7

/* Tiers a run may pin: T1 and T7 exist for the format and harnesses, but each
 * is only useful where a block's own level chooses it. */
#define KAC_TIER_FIXED_MIN 2
#define KAC_TIER_FIXED_MAX 6

/* Slice rankings split a block into the weakest KAC_BAND_PCT, the middle, and
 * the strongest share. The middle is the choice reading; T1 and T7 are gated on
 * the weak and strong ends. A file block is at most 960 samples and a slice is
 * at least 8, so KAC_BAND_KMAX covers 20 % of 120 slices. */
#define KAC_BAND_PCT 20u

/* Share at each end used by the floor and top-tier gates. */
#define KAC_GATE_PCT 5u

#define KAC_BAND_KMAX 24u

/* The widest the residual path search ever runs: paths kept per sample and the
 * longest trace walked back. A handle uses its quality level, or the brute
 * level when one is asked for. */
#define KAC_BEAM_WIDTH 12u
#define KAC_MAX_SLICE 60u
#define KAC_BEAM_NODES (KAC_MAX_SLICE * KAC_BEAM_WIDTH * 2u + 1u)

/* Encoder search effort: level 0 is hardest. The table in kac_init_params()
 * gives beam width, candidate count, form search and tier-reading kind; levels
 * 1..9 estimate tiers with one approximate pass. */
#define KAC_QUALITY_MAX 10

/* Default effort. Level 0 measures each tier by real coding; this level reads
 * tiers from the approximate pass and keeps the form measurement. */
#define KAC_QUALITY_DEFAULT 3

/* Raise a real-coded reading taken with no path search to what a run's search
 * would deliver, in milli-dB. Measured mean, rounded to the nearest 25; the
 * spread across tiers is not corrected. */
#define KAC_MEASURE_FIX_MDB 650

/* Per-tier correction for approximate readings, in milli-dB: measured gap to
 * a real coding plus rung slack, rounded to 50. One table per candidate rule;
 * the rules are listed at est_cand_limit(). */
static const int32_t kac_est_fixfull_mdb[KAC_TIER_MAX] = {1500, 1500, 1500, 1500, 1500, 1500, 1500};
static const int32_t kac_est_fixt12_mdb[KAC_TIER_MAX]  = {-800, 1700, 1850, 1650, 1650, 1650, 1650};
static const int32_t kac_est_fixtop3_mdb[KAC_TIER_MAX] = {2350, 2300, 1650, 1650, 1650, 1650, 1650};
static const int32_t kac_est_fixone_mdb[KAC_TIER_MAX]  = {5800, 4000, 3000, 2500, 2100, 1900, 1800};

/* Rung slack included in each entry above; form comparison subtracts it. */
#define KAC_EST_FIX_AIM 1500

/* Required mid/side advantage over left/right, in milli-dB, per tier (index 0
 * is unused). The margin grows with tier because the available advantage
 * shrinks; it stays well below a tier step, so it cannot choose the tier. */
static const int32_t kac_ms_margin_mdb[KAC_TIER_MAX + 1] = {0, 0, 0, 250, 500, 750, 1000, 1250};

/* Try a second scalefactor when it is within this percentage of the winner's
 * error; the residual search can change which one wins. */
#define KAC_SF_NEIGHBOR_PCT 25u

/* Sentinel for a tier the window did not measure. No finite reading reaches it. */
#define KAC_UNMEASURED INT32_MIN

static const kac_tier kac_tiers[KAC_TIER_MAX] = {{1, 60}, {2, 30}, {3, 20}, {4, 15},
                                                 {5, 12}, {6, 10}, {7, 8}};

#define KAC_HDR_SIZE 36u
#define KAC_SNAP_SIZE 16u
#define KAC_SLICE_SIZE 8u
#define KAC_FLAG_LOOP 0x01u
#define KAC_FLAG_INIT 0x02u
#define KAC_FLAG_SEEK 0x04u
/* The file may carry mid/side blocks. A decoder that cannot decode them must
 * reject the file; per-block form is bit 3 of each tier byte, and the header
 * Tier field carries block 0's form. */
#define KAC_FLAG_MS 0x08u
#define KAC_TIER_MASK 0x07u
/* Bits 4..7 of a tier byte carry nothing and must be zero: the same rule the
 * reserved Flags bits follow, so a byte with them set is rejected. */
#define KAC_TIER_SPARE 0xF0u
#define KAC_BLOCK_MS 0x08u
#define KAC_FLAG_RESERVED 0xF0u
#define KAC_VERSION 1u

/* Maximum block or stream-chunk samples; also the shared buffer length. */
#define KAC_CHUNK_SAMPLES 960u

/* Least sample count divisible by every tier's slice length; per-block runs use
 * multiples of this. */
#define KAC_CHUNK_LCM 120u

/* Samples in a file block: 20 ms rounded to KAC_CHUNK_LCM, capped at 20.5 ms and
 * KAC_CHUNK_SAMPLES; 120 is the floor below about 5.9 kHz. */
static LZ_MAYBE_UNUSED unsigned kac_block_for_rate(unsigned rate)
{
    uint64_t target = (uint64_t)rate * 20u / 1000u;
    uint64_t cap    = target + (uint64_t)rate / 2000u;
    uint64_t k      = (target + KAC_CHUNK_LCM / 2u) / KAC_CHUNK_LCM;
    uint64_t kmax   = KAC_CHUNK_SAMPLES / KAC_CHUNK_LCM;

    if (k < 1u)
        k = 1u;
    if (k > kmax)
        k = kmax;
    while (k > 1u && k * KAC_CHUNK_LCM > cap)
        k--;
    if (k * KAC_CHUNK_LCM > cap)
        return KAC_CHUNK_LCM;
    return (unsigned)(k * KAC_CHUNK_LCM);
}

/* Worst-case bytes for one block: tier byte plus a slice per 8 samples per
 * channel (T7 is the narrowest slice). */
#define KAC_BLOCK_BYTES (1u + (KAC_CHUNK_SAMPLES / 8u) * KAC_MAX_CHANNELS * KAC_SLICE_SIZE)

/* Default initial state, used by both ends when INIT_STATE is absent. Weight
 * 16383 keeps the worst-case sum below INT32_MAX (4 * 16384 * 32768 == 2^31). */
#define KAC_INIT_W0 0
#define KAC_INIT_W1 0
#define KAC_INIT_W2 (-8192)
#define KAC_INIT_W3 16383
#define KAC_W_LIM 16383

#define KAC_MAX_CHANNELS 2

/* TotalSamples ceiling: keeps DataBytes, the index and the whole file inside
 * signed 32-bit. Encoders must not exceed it; decoders must reject it. */
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
    KAC_E_NOLOOP  = -14
};

typedef struct {
    int32_t w[4];
    int32_t h[4];
} kac_ch;

/* The "this reading has no level" sentinel, in milli-dB. */
#define KAC_MDB_MAX 99000

/* The highest meaningful target: 14-bit PCM's audio-band signal-to-noise
 * ratio, about 85 dB. Readings can sit above this, up to the no-error sentinel;
 * targets and the aim stop here. */
#define KAC_AIM_MAX_MDB 85000

/* 10*log10(1 + i/256) in milli-dB. Linear interpolation is within 0.5
 * milli-dB; the encoder uses it so its measurements match the decoder bit for
 * bit without a math library. */
static const int16_t kac_log_tab[257] = {
    0,    17,   34,   51,   67,   84,   101,  117,  134,  150,  166,  183,  199,  215,  231,  247,
    263,  279,  295,  311,  327,  342,  358,  374,  389,  405,  420,  435,  451,  466,  481,  496,
    512,  527,  542,  557,  571,  586,  601,  616,  631,  645,  660,  674,  689,  703,  718,  732,
    746,  761,  775,  789,  803,  817,  831,  845,  859,  873,  887,  901,  914,  928,  942,  956,
    969,  983,  996,  1010, 1023, 1036, 1050, 1063, 1076, 1090, 1103, 1116, 1129, 1142, 1155, 1168,
    1181, 1194, 1207, 1220, 1232, 1245, 1258, 1271, 1283, 1296, 1308, 1321, 1333, 1346, 1358, 1371,
    1383, 1395, 1408, 1420, 1432, 1444, 1456, 1469, 1481, 1493, 1505, 1517, 1529, 1541, 1552, 1564,
    1576, 1588, 1600, 1611, 1623, 1635, 1646, 1658, 1669, 1681, 1693, 1704, 1715, 1727, 1738, 1750,
    1761, 1772, 1783, 1795, 1806, 1817, 1828, 1839, 1850, 1862, 1873, 1884, 1895, 1906, 1916, 1927,
    1938, 1949, 1960, 1971, 1981, 1992, 2003, 2014, 2024, 2035, 2045, 2056, 2067, 2077, 2088, 2098,
    2109, 2119, 2129, 2140, 2150, 2160, 2171, 2181, 2191, 2201, 2212, 2222, 2232, 2242, 2252, 2262,
    2272, 2282, 2292, 2302, 2312, 2322, 2332, 2342, 2352, 2362, 2372, 2382, 2391, 2401, 2411, 2421,
    2430, 2440, 2450, 2459, 2469, 2479, 2488, 2498, 2507, 2517, 2526, 2536, 2545, 2555, 2564, 2573,
    2583, 2592, 2601, 2611, 2620, 2629, 2639, 2648, 2657, 2666, 2675, 2685, 2694, 2703, 2712, 2721,
    2730, 2739, 2748, 2757, 2766, 2775, 2784, 2793, 2802, 2811, 2820, 2828, 2837, 2846, 2855, 2864,
    2872, 2881, 2890, 2899, 2907, 2916, 2925, 2933, 2942, 2951, 2959, 2968, 2976, 2985, 2993, 3002,
    3010};

/* 10*log10(v) in milli-dB, for any v up to 2^63. A caller needing 20*log10(v)
 * doubles the result; a quotient is a difference of logs, so no overflow-prone
 * multiply happens. Returns KAC_MDB_MAX for v == 0. */
static LZ_MAYBE_UNUSED int32_t kac_log10_mdb(uint64_t v)
{
    int32_t r;
    int     e = 0;
    if (v == 0)
        return KAC_MDB_MAX;
    while (v >= ((uint64_t)1 << 32)) {
        v >>= 1;
        e++;
    }
    while (v < ((uint64_t)1 << 31)) {
        v <<= 1;
        e--;
    }
    {
        uint32_t off = (uint32_t)(v - ((uint64_t)1 << 31)); /* [0, 2^31) */
        int      i   = (int)(off >> 23);                    /* [0, 256) */
        uint32_t f   = off & ((1u << 23) - 1u);
        int32_t  d   = (int32_t)kac_log_tab[i + 1] - (int32_t)kac_log_tab[i];
        r            = (int32_t)kac_log_tab[i] + (int32_t)(((int64_t)d * f) >> 23);
    }
    /* v was scaled into [2^31, 2^32), so v == m * 2^(e + 31) with m in [1, 2)
     * and kac_log_tab holds 10*log10(m). The 31 belongs to the exponent. */
    e += 31;
    /* e*10*log10(2) in milli-dB, to the nearest unit: 30103/10 == 3010.3 */
    return r + (e * 30103 + (e < 0 ? -5 : 5)) / 10;
}

static LZ_MAYBE_UNUSED int32_t kac_sat_w(int32_t v)
{
    return v > KAC_W_LIM ? KAC_W_LIM : (v < -KAC_W_LIM ? -KAC_W_LIM : v);
}

/* Narrowest slice length: buffer bounds must use it, not the current tier. */
static LZ_MAYBE_UNUSED unsigned kac_narrowest_n(void)
{
    unsigned t, n = (unsigned)kac_tiers[0].n;
    for (t = 1u; t < (unsigned)KAC_TIER_MAX; t++)
        if ((unsigned)kac_tiers[t].n < n)
            n = (unsigned)kac_tiers[t].n;
    return n;
}

static LZ_MAYBE_UNUSED int32_t kac_clamp16(int32_t s)
{
    return s > 32767 ? 32767 : (s < -32768 ? -32768 : s);
}

/* Mid/side pair conversion, in place. Both channels are halved in 16-bit range;
 * merging can exceed the range by one count, so it clamps. */
static LZ_MAYBE_UNUSED void kac_ms_split(int16_t *restrict a, int16_t *restrict b, unsigned len)
{
    unsigned i;
    for (i = 0; i < len; i++) {
        int32_t l = a[i], r = b[i];
        a[i] = (int16_t)((l + r) >> 1);
        b[i] = (int16_t)((l - r) >> 1);
    }
}

static LZ_MAYBE_UNUSED void kac_ms_merge(int16_t *restrict a, int16_t *restrict b, unsigned len)
{
    unsigned i;
    for (i = 0; i < len; i++) {
        int32_t m = a[i], s = b[i];
        a[i] = (int16_t)kac_clamp16(m + s);
        b[i] = (int16_t)kac_clamp16(m - s);
    }
}

static LZ_MAYBE_UNUSED int32_t kac_predict(const kac_ch *restrict c)
{
    return (c->w[0] * c->h[0] + c->w[1] * c->h[1] + c->w[2] * c->h[2] + c->w[3] * c->h[3]) >> 13;
}

/* Weight update from a precomputed prediction. The shift floors, so close the
 * one-sided dead zone; a zero residual must not adapt the filter. */
static LZ_MAYBE_UNUSED int32_t kac_step_pred(kac_ch *restrict c, int32_t residual, int32_t pred)
{
    int32_t s  = kac_clamp16(pred + residual);
    int32_t d  = residual >> 4;
    int32_t s0, s1, s2, s3;
    d += (d == 0 && residual != 0);
    s0 = (c->h[0] < 0) ? -d : d;
    s1 = (c->h[1] < 0) ? -d : d;
    s2 = (c->h[2] < 0) ? -d : d;
    s3 = (c->h[3] < 0) ? -d : d;
    c->w[0]    = kac_sat_w(c->w[0] + s0);
    c->w[1]    = kac_sat_w(c->w[1] + s1);
    c->w[2]    = kac_sat_w(c->w[2] + s2);
    c->w[3]    = kac_sat_w(c->w[3] + s3);
    c->h[0]    = c->h[1];
    c->h[1]    = c->h[2];
    c->h[2]    = c->h[3];
    c->h[3]    = s;
    return s;
}

/* The decoder's sample loop computes no prediction of its own, so it takes the
 * step that derives one. */
static LZ_MAYBE_UNUSED int32_t kac_step(kac_ch *restrict c, int32_t residual)
{
    return kac_step_pred(c, residual, kac_predict(c));
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
        return &kac_dq_t1[0][0];
    case 2u:
        return &kac_dq_t2[0][0];
    case 3u:
        return &kac_dq_t3[0][0];
    case 4u:
        return &kac_dq_t4[0][0];
    case 5u:
        return &kac_dq_t5[0][0];
    case 6u:
        return &kac_dq_t6[0][0];
    default:
        return &kac_dq_t7[0][0];
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

static LZ_MAYBE_UNUSED void kac_put16(uint8_t *restrict p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static LZ_MAYBE_UNUSED void kac_put32(uint8_t *restrict p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static LZ_MAYBE_UNUSED uint64_t kac_get64(const uint8_t *restrict p)
{
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

static LZ_MAYBE_UNUSED void kac_put64(uint8_t *restrict p, uint64_t v)
{
    int i;
    for (i = 0; i < 8; i++)
        p[i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
}

static LZ_MAYBE_UNUSED void kac_snap_write(uint8_t *restrict p, const kac_snap *restrict s)
{
    int i;
    for (i = 0; i < 4; i++)
        kac_put16(p + 2 * i, (uint16_t)s->h[i]);
    for (i = 0; i < 4; i++)
        kac_put16(p + 8 + 2 * i, (uint16_t)s->w[i]);
}

static LZ_MAYBE_UNUSED void kac_snap_store(uint8_t *restrict p, const kac_ch *restrict c)
{
    kac_snap s;
    int      i;
    for (i = 0; i < 4; i++)
        s.h[i] = (int16_t)c->h[i];
    for (i = 0; i < 4; i++)
        s.w[i] = (int16_t)c->w[i];
    kac_snap_write(p, &s);
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
