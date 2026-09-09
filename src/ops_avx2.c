/* AVX2 row kernels for the matmul/dot-product family. gcc-only: this
   file is never compiled by Watcom. Task 7's Makefile rule will only
   ever pass -mavx2 to this translation unit, never -mfma - see the
   design doc's "core tension and solution" section on why FMA is
   excluded even though -march=x86-64-v3 nominally bundles it in. This
   tier is never a member of kernel_detect()'s AUTO cascade; reachable
   only via an explicit --kernel avx2.

   Every accumulation below is INTEGER (int8/int16 products summed in
   int32), never floating point, so there is no FMA-vs-two-step
   rounding question for this file the way rule two worries about for
   the float epilogue elsewhere: within the documented per-group bound
   (|w<<8| <= 32512, |x| <= 127, 32-element group caps ~1.3e8, far
   under INT32_MAX), int32 addition is exact and fully associative.
   Bit-identity with the SSE2 tier (tests/test_ops.c, training-side
   repo) holds because both compute the same exact integer sum - not
   because this file replicates SSE2's exact intermediate register
   grouping, which it deliberately does not.

   This first pass does not implement the SSE2 tier's g_pair
   (two-token, shared-weight-load) optimization from
   src/ops_kernel_dot_sse2.h - each token is processed independently.
   Documented, deferred follow-up; not a silent omission. */
#if defined(LZ_AVX2_TU) && !defined(__WATCOMC__)

#include <immintrin.h>
#include "ops_avx2.h"

/* Reduce one __m256i of 8 x int32 partial sums to a single scalar.
   Shared tail for every part32_*_avx2 below - none of them do a
   horizontal reduction any other way, so factoring it once here means
   the reduction sequence can only drift in one place, not four. */
static int32_t fold8_to_scalar_avx2(__m256i s) {
    __m128i lo = _mm256_castsi256_si128(s);
    __m128i hi = _mm256_extracti128_si256(s, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    return _mm_cvtsi128_si32(sum);
}

/* 32-element partial sum, one int32 result. w<<8 convention matches
   part32_x16 (src/ops_kernel_dot_sse2.h) exactly - the shared row-end
   epilogue elsewhere divides by the same 256 regardless of which
   kernel tier produced acc32[], so this file must reproduce that
   convention, not the register sequence that produces it.

   _mm256_cvtepu8_epi16 zero-extends a 128-bit (16-byte) load to a
   256-bit (16 x int16) result IN ELEMENT ORDER - unlike
   _mm256_unpacklo/hi_epi8, which interleave within each 128-bit lane
   independently and would silently pair the wrong weight byte with
   the wrong activation if used here alongside a plain contiguous
   activation load. Concretely: with wb0 holding weight bytes 0-31
   across its two 128-bit lanes, _mm256_unpacklo_epi8(z, wb0) would
   zero-extend bytes {0-7, 16-23} into its 16 int16 lanes, in that
   order - lane 0's low half interleaved with lane 1's low half, NOT
   bytes 0-15 the way a caller reasoning "unpacklo takes the low half"
   would expect. Paired against a plain contiguous _mm256_loadu_si256
   of activations 0-15, that multiplies weight byte 20 against
   activation element 12 instead of element 20 - finite, plausible,
   wrong. Recovering the contiguous order from the naive unpack would
   need an extra cross-lane step (_mm256_permute4x64_epi64 on wb0
   before unpacking, or permutevar8x32 after) that _mm256_cvtepu8_epi16
   needs not, because it is never lane-restricted to begin with.

   Unaligned loads throughout (_mm_loadu_si128/_mm256_loadu_si256),
   unlike part32_x16's aligned _mm_load_si128 for both operands: this
   file has no established guarantee that either the weight row
   (c->w4, from the model blob) or a 32-element activation group
   offset lands on a 32-byte boundary - src/ops_matmul.c's g_xw
   alignment comment establishes 16-byte, not 32-byte, as what the
   SSE2 tier actually relies on. An unaligned load costs nothing extra
   when the pointer happens to be more strictly aligned, so loadu is
   the conservative choice here, not an unverified aligned claim. */
static int32_t part32_x32_avx2(const int8_t *w, const int16_t *x) {
    __m128i wb0 = _mm_loadu_si128((const __m128i *)w);
    __m128i wb1 = _mm_loadu_si128((const __m128i *)(w + 16));
    __m256i w0 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb0), 8);
    __m256i w1 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb1), 8);
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 16));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(w0, x0),
                                  _mm256_madd_epi16(w1, x1));
    return fold8_to_scalar_avx2(s);
}

/* ---- Q8 row kernel -------------------------------------------------------
   SSE2 twin: row_q8_sse2_intrin, src/ops_sse2.c. No Watcom twin -
   AVX2 has none, unlike every other tier in this codebase. No pairing
   either: g_pair (src/ops_sse2.c) is the SSE2 tier's two-token,
   shared-weight-load trick, not implemented here - see this file's
   header comment. Each token is processed independently. */
void row_q8_avx2_intrin(const lz_row_ctx *c) {
    const int8_t *wr = (const int8_t *)c->w4;
    int tk, g;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (g = 0; g < c->nb; g++)
                acc[g] = part32_x32_avx2(wr + (size_t)g * 32,
                                          xwt + (size_t)g * 32);
        }
    }
}

/* Q4_1 32-element partial sum. Nibble layout (model.h, LZ_FMT_Q4_1): a
   32-element sub-block is 16 bytes; byte j's LOW nibble is element j,
   its HIGH nibble is element j+16 - same convention part32_q41
   (src/ops_kernel_dot_sse2.h) decodes, reproduced here bit for bit.
   There is no zero-point work in this function: LZTensor.zero (the
   per-group min) is an affine term applied later by the epilogue
   (ops_matmul.c's epi_q41 family via the hoisted xg[]/zq term), never
   inside the row kernel - this function's job is only the same raw
   unsigned-nibble dot product part32_q41 computes, nothing more.

   `lo`/`hi` each hold all 16 elements of their half in one 16-byte
   register, in order - unlike part32_x32_avx2's two full 32-byte
   weight loads, a nibble sub-block is only 16 bytes total, so one
   _mm256_cvtepu8_epi16 per half is enough; no second widen call and
   no unpacklo/hi_epi8 trap to avoid, since the mask/shift already
   produced one byte per element in the right order before the widen. */
static int32_t part32_q41_avx2(const unsigned char *w, const int16_t *x) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m128i b  = _mm_loadu_si128((const __m128i *)w);
    __m128i lo = _mm_and_si128(b, m0f);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
    __m256i lo16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8);
    __m256i hi16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8);
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 16));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(lo16, x0),
                                  _mm256_madd_epi16(hi16, x1));
    return fold8_to_scalar_avx2(s);
}

/* ---- Q4_1 row kernel ------------------------------------------------
   SSE2 twin: row_q41_sse2_intrin, src/ops_sse2.c. No Watcom twin, no
   pairing - see this file's header comment; each token is processed
   independently, same as row_q8_avx2_intrin. */
void row_q41_avx2_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = part32_q41_avx2(wn + (size_t)s * 16,
                                          xwt + (size_t)s * 32);
        }
    }
}

/* Q6_1 32-element partial sum: two-plane format, model.h's LZ_FMT_Q6_1.
   `w4` is the SAME 4-bit-nibble plane as Q4_1 (elements 0-15 in low
   nibbles, 16-31 in high nibbles); `w2` is a second, independent 2-bit
   plane, 8 bytes per 32-element sub-block, byte j's bits 0-1/2-3/4-5/6-7
   holding elements j/j+8/j+16/j+24 respectively. A 6-bit code is
   q = lo_nibble | (2bit_field << 4); this function reads BOTH c->w4 AND
   c->w2, unlike every other format in this file, which reads only
   c->w4 - see row_q61_sse2_intrin (src/ops_sse2.c) for the same
   two-pointer read on the reference tier.

   Reassembly (m0f/m03 masks, the two shifts-by-2/4/6, `_mm_or_si128`
   with the nibble) is copied instruction-for-instruction from
   part32_q61's 128-bit domain, unchanged - that step produces one byte
   per element, in order, before anything is widened, so it carries no
   AVX2-specific risk. Only the final byte-to-int16 widen swaps
   part32_q61's unpacklo/hi_epi8 pair for a single in-order
   _mm256_cvtepu8_epi16 per half, same as part32_q41_avx2 above. No
   zero-point work here either, for the same reason as Q4_1: `zero` is
   an epilogue-side affine term, not part of this raw dot product. */
static int32_t part32_q61_avx2(const unsigned char *w4,
                                const unsigned char *w2, const int16_t *x) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i m03 = _mm_set1_epi8(0x03);
    __m128i b  = _mm_loadu_si128((const __m128i *)w4);
    __m128i t  = _mm_loadl_epi64((const __m128i *)w2);
    __m128i g0 = _mm_and_si128(t, m03);
    __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03);
    __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03);
    __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03);
    __m128i lo = _mm_or_si128(_mm_and_si128(b, m0f),
                              _mm_slli_epi16(_mm_unpacklo_epi64(g0, g1), 4));
    __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b, 4), m0f),
                              _mm_slli_epi16(_mm_unpacklo_epi64(g2, g3), 4));
    __m256i lo16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8);
    __m256i hi16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8);
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 16));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(lo16, x0),
                                  _mm256_madd_epi16(hi16, x1));
    return fold8_to_scalar_avx2(s);
}

/* ---- Q6_1 row kernel ------------------------------------------------
   SSE2 twin: row_q61_sse2_intrin, src/ops_sse2.c. No Watcom twin, no
   pairing - see this file's header comment; each token is processed
   independently, same as row_q8_avx2_intrin. Reads both c->w4 (4-bit
   plane) and c->w2 (2-bit plane), each advanced at its own stride (16
   and 8 bytes per 32-element sub-block). */
void row_q61_avx2_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *w2 = (const unsigned char *)c->w2;
    int tk, s;
    (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = part32_q61_avx2(wn + (size_t)s * 16,
                                          w2 + (size_t)s * 8,
                                          xwt + (size_t)s * 32);
        }
    }
}

/* Q16_0 32-element partial sum: weight is already int16 (model.h,
   LZ_FMT_Q16_0), so there is no unpack step at all - the simplest of
   the four formats in this file, same conclusion part32_q16's own
   comment (src/ops_kernel_dot_sse2.h) draws for the SSE2 tier. Two
   plain 256-bit loads per operand, two pmaddwd, one add, one fold. No
   x<<8 scale here either: Q16_0's row-end epilogue uses post=1.0, not
   the 1/256 the other three formats share (see part32_q16's comment). */
static int32_t part32_q16_avx2(const int16_t *w, const int16_t *x) {
    __m256i w0 = _mm256_loadu_si256((const __m256i *)w);
    __m256i w1 = _mm256_loadu_si256((const __m256i *)(w + 16));
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 16));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(w0, x0),
                                  _mm256_madd_epi16(w1, x1));
    return fold8_to_scalar_avx2(s);
}

/* ---- Q16_0 row kernel -------------------------------------------------
   SSE2 twin: row_q16_sse2_intrin, src/ops_sse2.c. No Watcom twin, no
   pairing - see this file's header comment; each token is processed
   independently, same as row_q8_avx2_intrin. */
void row_q16_avx2_intrin(const lz_row_ctx *c) {
    const int16_t *wr = (const int16_t *)c->w4;
    int tk, g;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (g = 0; g < c->nb; g++)
                acc[g] = part32_q16_avx2(wr + (size_t)g * 32,
                                          xwt + (size_t)g * 32);
        }
    }
}

/* ---- Q8 group-scale amax, AVX2 tier -------------------------------------
   SSE2 reference: src/ops_sse2.c's lz_amax32_sse2. Bit-pattern integer
   max is exact and associative (see this file's own top-of-plan
   licence note) - this does not need to reproduce the SSE2 tree shape,
   only the correct maximum. */
unsigned lz_amax32_avx2(const float *x, int n) {
    __m256i acc = _mm256_setzero_si256();
    __m256i msk = _mm256_set1_epi32(0x7FFFFFFF);
    int k;
    for (k = 0; k + 7 < n; k += 8) {
        __m256i v = _mm256_and_si256(
            _mm256_loadu_si256((const __m256i *)(const void *)(x + k)), msk);
        __m256i gt = _mm256_cmpgt_epi32(v, acc);
        acc = _mm256_or_si256(_mm256_and_si256(v, gt),
                              _mm256_andnot_si256(gt, acc));
    }
    {
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i gt128 = _mm_cmpgt_epi32(hi, lo);
        __m128i m4 = _mm_or_si128(_mm_and_si128(hi, gt128),
                                  _mm_andnot_si128(gt128, lo));
        union { __m128i m; unsigned u[4]; } o;
        unsigned a, b;
        o.m = m4;
        a = o.u[0] > o.u[1] ? o.u[0] : o.u[1];
        b = o.u[2] > o.u[3] ? o.u[2] : o.u[3];
        return a > b ? a : b;
    }
}

/* ---- RMSNorm output loop / softmax max-scan / softmax scale, AVX2 tier -
   SSE references: src/ops_kernel_norm.h's lz_rmsnorm_out_sse/lz_vmax_sse/
   lz_vscale_sse. n4 counts groups of 4 floats and is not guaranteed
   even - each function below processes pairs of groups (8 floats) at a
   time and falls back to a scalar tail for a possible odd leftover
   group, rather than mixing in a 128-bit SSE path for four floats. */
void lz_rmsnorm_out_avx2(float *o, const float *x, const float *w,
                         int n4, const float *k2) {
    __m256 vi = _mm256_set1_ps(k2[0]);
    __m256 one = _mm256_set1_ps(k2[1]);
    int b, n8 = n4 & ~1;
    for (b = 0; b < n8; b += 2) {
        __m256 vx = _mm256_loadu_ps(x + b * 4);
        __m256 vw = _mm256_loadu_ps(w + b * 4);
        _mm256_storeu_ps(o + b * 4,
            _mm256_mul_ps(_mm256_mul_ps(vx, vi), _mm256_add_ps(one, vw)));
    }
    if (b < n4) {
        int i;
        for (i = b * 4; i < n4 * 4; i++)
            o[i] = x[i] * k2[0] * (1.0f + w[i]);
    }
}

void lz_vmax_avx2(const float *x, int n4, float *pmax) {
    __m256 acc = _mm256_set1_ps(*pmax);
    int b, n8 = n4 & ~1;
    for (b = 0; b < n8; b += 2)
        acc = _mm256_max_ps(acc, _mm256_loadu_ps(x + b * 4));
    {
        /* Vector-domain tree, not a union+scalar-loop reduction: the
           loop shape compiled to a genuine stack spill + 7-iteration
           scalar read-back (confirmed by disassembly at -O2 -mavx2),
           unlike lz_amax32_avx2's own tail three functions up and
           lz_vmax_sse's ternary-tree C twin (ops_kernel_norm.h), both
           of which gcc keeps entirely in registers. This mirrors
           Watcom's own lz_vmax_sse_asm (ops_mmx_sse.c): maxps twice,
           movhlps, shufps twice, one scalar store out - no memory
           traffic between the first load and the final result. */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 m  = _mm_max_ps(lo, hi);
        __m128 t  = _mm_movehl_ps(m, m);
        int i;
        m = _mm_max_ps(m, t);
        t = _mm_shuffle_ps(m, m, 1);
        m = _mm_max_ps(m, t);
        for (i = n8 * 4; i < n4 * 4; i++)
            m = _mm_max_ss(m, _mm_load_ss(x + i));
        _mm_store_ss(pmax, m);
    }
}

void lz_vscale_avx2(float *x, int n4, const float *pk) {
    __m256 vk = _mm256_set1_ps(*pk);
    int b, n8 = n4 & ~1;
    for (b = 0; b < n8; b += 2)
        _mm256_storeu_ps(x + b * 4, _mm256_mul_ps(_mm256_loadu_ps(x + b * 4), vk));
    if (b < n4) {
        int i;
        for (i = b * 4; i < n4 * 4; i++) x[i] *= *pk;
    }
}

/* ---- matmul epilogue's int32 x int16 -> int64 MAC, AVX2 tier -----------
   SSE2 reference: src/ops_sse2.c's lz_epi_mac_i16_sse2 - read that
   function's comment for the split-multiply technique this widens.
   Exact integer accumulation (see this file's own top-of-plan licence
   note) - the accumulator pairing below does not need to match the
   SSE2 tier's, only sum the same eight exact products somewhere.

   Overflow bound, derived from the actual call site (ops_epi.c:1323,
   epi_q41_join), not from the row kernels' ~1.3e8 dot-product cap that
   an earlier draft of this comment cited by mistake - that number
   belongs to a different accumulator. Here `a` is g_epi_zg[tk]: a sum
   of up to LZ_EPI_MAX_R (16) products each bounded 133,165,088
   (ops_epi.c:604-612's own derivation), so |a| <= 16 * 133,165,088 =
   2,130,641,408 (~2.13e9, deliberately near INT32_MAX by construction;
   measured max on kmr20 is 76,280,004). With |m| <= 32767 and up to
   LZ_EPI_MAX_NG (64) terms per call, |sum| <= 64 * 2,130,641,408 *
   32767 ~= 4.47e15, a ~2064x margin under INT64_MAX (~9.2e18).
   Processing 8 elements per iteration instead of 4 changes nothing
   about this: the per-lane arithmetic (ah/als/bb/m32/pa/pl) is
   unaffected by vector width, only how many lanes run between int64
   accumulator folds. */
lz_i64 lz_epi_mac_i16_avx2(const int32_t *a, const int16_t *m, int n) {
    const __m256i lo16 = _mm256_set1_epi32(0x0000FFFF);
    const __m256i one  = _mm256_set1_epi32(1);
    const __m128i zero128 = _mm_setzero_si128();
    __m128i acc_hi = zero128, acc_als = zero128;
    lz_i64 t[2], s_hi, s_als;
    int g = 0;

    for (; g + 7 < n; g += 8) {
        __m256i av    = _mm256_loadu_si256((const __m256i *)(const void *)(a + g));
        __m128i mv128 = _mm_loadu_si128((const __m128i *)(const void *)(m + g));
        /* Zero-extend 8 x int16 to 8 x int32 IN ELEMENT ORDER - the
           AVX2-safe equivalent of the SSE2 tier's unpacklo_epi16-with-
           zero, same reason row_q8_avx2_intrin uses cvtepu8_epi16
           instead of unpacklo/hi_epi8: a raw 256-bit unpack would
           interleave WITHIN each 128-bit half rather than across the
           whole register. */
        __m256i m32 = _mm256_cvtepu16_epi32(mv128);
        __m256i ah  = _mm256_srli_epi32(av, 16);
        __m256i als = _mm256_and_si256(av, lo16);
        __m256i bb  = _mm256_and_si256(_mm256_srli_epi32(av, 15), one);
        __m256i pa  = _mm256_add_epi32(_mm256_madd_epi16(ah, m32),
                                       _mm256_madd_epi16(bb, m32));
        __m256i pl  = _mm256_madd_epi16(als, m32);
        /* Split to 128-bit halves before sign-extending to int64 with
           the SAME 128-bit unpacklo/hi sequence the SSE2 tier uses
           (safe there: no cross-256-bit-lane concern once already
           split). Which half feeds which accumulator add does not
           affect the final sum - an exact integer total, any grouping
           of the same eight products gives the same int64. */
        {
            __m128i pa_lo = _mm256_castsi256_si128(pa);
            __m128i pa_hi = _mm256_extracti128_si256(pa, 1);
            __m128i pl_lo = _mm256_castsi256_si128(pl);
            __m128i pl_hi = _mm256_extracti128_si256(pl, 1);
            acc_hi = _mm_add_epi64(acc_hi,
                     _mm_unpacklo_epi32(pa_lo, _mm_srai_epi32(pa_lo, 31)));
            acc_hi = _mm_add_epi64(acc_hi,
                     _mm_unpackhi_epi32(pa_lo, _mm_srai_epi32(pa_lo, 31)));
            acc_hi = _mm_add_epi64(acc_hi,
                     _mm_unpacklo_epi32(pa_hi, _mm_srai_epi32(pa_hi, 31)));
            acc_hi = _mm_add_epi64(acc_hi,
                     _mm_unpackhi_epi32(pa_hi, _mm_srai_epi32(pa_hi, 31)));
            acc_als = _mm_add_epi64(acc_als,
                     _mm_unpacklo_epi32(pl_lo, _mm_srai_epi32(pl_lo, 31)));
            acc_als = _mm_add_epi64(acc_als,
                     _mm_unpackhi_epi32(pl_lo, _mm_srai_epi32(pl_lo, 31)));
            acc_als = _mm_add_epi64(acc_als,
                     _mm_unpacklo_epi32(pl_hi, _mm_srai_epi32(pl_hi, 31)));
            acc_als = _mm_add_epi64(acc_als,
                     _mm_unpackhi_epi32(pl_hi, _mm_srai_epi32(pl_hi, 31)));
        }
    }
    _mm_storeu_si128((__m128i *)(void *)t, acc_hi);  s_hi  = t[0] + t[1];
    _mm_storeu_si128((__m128i *)(void *)t, acc_als); s_als = t[0] + t[1];
    s_hi = (s_hi << 16) + s_als;
    for (; g < n; g++) s_hi += (lz_i64)a[g] * (lz_i64)m[g];
    return s_hi;
}

/* ---- FWHT butterfly stage, AVX2 tier -------------------------------
   SSE2 reference: src/ops_sse2.c's lz_fwht_stage_sse2. Integer add/sub
   butterfly, exact by construction (this file's top-of-plan licence
   note) - no reduction, so widening to 8-at-a-time changes nothing
   about correctness. len is always a power of two; when it is exactly
   4 the 8-wide loop below does zero iterations and the whole stage
   falls through to the 4-wide tail, same shape as the SSE2 tier
   falling through to MMX at len == 2. */
void lz_fwht_stage_avx2(int32_t *y, int n, int len) {
    int i, j;
    for (i = 0; i < n; i += (len << 1)) {
        int32_t *a = y + i;
        int32_t *b = y + i + len;
        for (j = 0; j + 7 < len; j += 8) {
            __m256i u = _mm256_loadu_si256((const __m256i *)(const void *)(a + j));
            __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(b + j));
            _mm256_storeu_si256((__m256i *)(void *)(a + j), _mm256_add_epi32(u, v));
            _mm256_storeu_si256((__m256i *)(void *)(b + j), _mm256_sub_epi32(u, v));
        }
        for (; j < len; j += 4) {
            __m128i u = _mm_loadu_si128((const __m128i *)(const void *)(a + j));
            __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(b + j));
            _mm_storeu_si128((__m128i *)(void *)(a + j), _mm_add_epi32(u, v));
            _mm_storeu_si128((__m128i *)(void *)(b + j), _mm_sub_epi32(u, v));
        }
    }
}

/* ---- RoPE, AVX2 tier -----------------------------------------------------
   SSE2 twin: lz_rope_sse2, src/ops_sse2.c. Same shape - head outer loop,
   i inner loop - just 8-wide instead of 4-wide, and the same scalar tail
   for i + 8 > half. Each (i, h) rotation is an independent IEEE mul/add
   with no reduction, so this is bit-identical to the scalar path by the
   same per-element argument every Phase 3 element-wise kernel relies on;
   only the loop order/width changes.

   _mm256_shuffle_ps operates within each 128-bit half independently, not
   across the full register: shuffle(lo,hi,0x88) on [c0,s0,c1,s1,c2,s2,c3,s3]
   / [c4,s4,c5,s5,c6,s6,c7,s7] gives [c0,c1,c4,c5, c2,c3,c6,c7], not the
   contiguous [c0..c7]. permute4x64_epi64 with 0xD8 (index pattern
   [0,2,1,3] over the four 64-bit lanes) swaps the middle two chunks back
   into place; done through an int bitcast since permute4x64 has no float
   form. */
void lz_rope_avx2(float *v, int n_heads, int head_dim, int rotary_dim,
                  int pos, const float *cs) {
    int half = rotary_dim / 2;
    int h, i;
    const float *row = cs + (size_t)pos * half * 2;
    for (h = 0; h < n_heads; h++) {
        float *base = v + (size_t)h * head_dim;
        for (i = 0; i + 8 <= half; i += 8) {
            __m256 lo = _mm256_loadu_ps(row + i * 2);
            __m256 hi = _mm256_loadu_ps(row + i * 2 + 8);
            __m256 cshuf = _mm256_shuffle_ps(lo, hi, 0x88);
            __m256 sshuf = _mm256_shuffle_ps(lo, hi, 0xDD);
            __m256 c = _mm256_castsi256_ps(
                _mm256_permute4x64_epi64(_mm256_castps_si256(cshuf), 0xD8));
            __m256 s = _mm256_castsi256_ps(
                _mm256_permute4x64_epi64(_mm256_castps_si256(sshuf), 0xD8));
            __m256 x0 = _mm256_loadu_ps(base + i);
            __m256 x1 = _mm256_loadu_ps(base + i + half);
            __m256 n0 = _mm256_sub_ps(_mm256_mul_ps(x0, c), _mm256_mul_ps(x1, s));
            __m256 n1 = _mm256_add_ps(_mm256_mul_ps(x1, c), _mm256_mul_ps(x0, s));
            _mm256_storeu_ps(base + i, n0);
            _mm256_storeu_ps(base + i + half, n1);
        }
        for (; i < half; i++) {
            float c0 = row[i * 2], s0 = row[i * 2 + 1];
            float x0 = base[i], x1 = base[i + half];
            base[i]        = x0 * c0 - x1 * s0;
            base[i + half] = x1 * c0 + x0 * s0;
        }
    }
}

/* lz_rope_avx2_compiled_in is deliberately NOT defined here: it belongs
   in the dispatch translation unit (src/ops_rope.c), the same place
   ops_epi.c's lz_epi_avx2_compiled_in and fwht.c's
   lz_fwht_avx2_compiled_in live. Defined in this file it could only
   ever return 1 - LZ_ROPE_AVX2_EXTERN comes from this file's own
   header under the same LZ_AVX2_TU guard that wraps this whole
   translation unit - so it would answer nothing. In ops_rope.c the
   macro is present only if that file's #include "ops_avx2.h" is, which
   is the question the test asks. */

#endif /* LZ_AVX2_TU && !__WATCOMC__ */
