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

   g_pair (two tokens, one shared weight load) is implemented in every
   row kernel below - q8 and q16 inline, q41/q61/t2 through the
   part32_*_avx2_2 macro that shares the format's decode. The weight
   stream is the larger of the two and does not change between tokens;
   on the Q8 kernel at nt=64 that is 0.036 -> 0.030 ns/element. */
#if defined(LZ_AVX2_TU) && !defined(__WATCOMC__)

#include <immintrin.h>
#include <string.h>   /* memcpy, for lz_wsum_pair_avx2's coef dword */
#include "ops_avx2.h"
#include "ops_sched.h"  /* g_pair - the two-token pairing knob the row
                           kernels read, same as the SSE2 tier's */
#include "ops_kernel_shared.h"  /* LZ_PF_DIST, LZ_PF_*, LZ_PFI_* - the
                           prefetch discipline the row kernels below share
                           with the SSE1/SSE2/MMX ones. */

/* One prefetch per group at the format's byte stride, one for one with
   the Watcom SSE2 and gcc MMX rows (ops_kernel_dot_shared.h's LZ_Q8_ACC
   and siblings). The gcc SSE2 rows drop c->pf_end, so this tier is
   strictly wider than those. Guarded like the macros: prefetchnta
   cannot fault but PII's dummy load is a real access. The mode is
   resolved once per call rather than per group, as LZ_Q8_PFSEL does by
   expanding its loop four ways. */
#define LZ_AVX2_PF(P_, M_) do { \
        if ((M_) == LZ_PF_AMD)       LZ_PFI_AMD(P_); \
        else if ((M_) == LZ_PF_LOAD) LZ_PFI_LOAD(P_); \
        else if ((M_) == LZ_PF_NONE) LZ_PFI_NONE(P_); \
        else                         LZ_PFI_NTA(P_); \
    } while (0)

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
   unlike part32_x16's aligned _mm_load_si128 for both operands, and the
   reason survives this tier's own alignment work: the activation side
   (c->xw = g_xw) and the weight planes are now DECLARED 32-byte aligned
   (LZ_XW_ALIGN in src/ops_matmul.c, aligned_malloc in src/model.c), but
   a 256-bit load still cannot use the aligned form on all of them -
   Q4_1's weight group is 16 bytes and Q6_1's two planes are 16 and 8,
   so those advance by less than 32 and land 16-aligned on every other
   group whatever the base. loadu on a 32-aligned address never crosses
   a cache line, so the declaration is what buys the performance; the
   encoding is left unaligned so that a broken invariant is a correctness
   failure somewhere else rather than a #GP here. */
/* The same, stopping one step short: the eight int32 partials, not
   their total. Split out so the group-of-four path below can fold four
   groups with ONE transposed reduction instead of four separate ones -
   the same amortization the SSE2 tier's LZ_Q8_GROUP4/dot128 makes at
   128 bits, and the only part of this kernel that is per-group fixed
   cost rather than per-element work. */
/* Macros, not functions, and that is a correctness requirement rather
   than a style choice: __m256i is 32-byte aligned, so returning one by
   value makes gcc materialize a 32-byte temporary - and Windows x64
   guarantees only 16-byte RSP, which gcc does not realign for. Measured
   with these as functions: #GP on `vmovdqa %ymm0,(%rax)` into a
   16-aligned slot. -mpreferred-stack-boundary cannot help; this MinGW
   caps it at 4. As macros the values stay in registers.

   Every argument is parenthesised at the call site: the bodies cast
   first and add second, so a bare `x + 16` steps by the cast's type. */
/* The same, for TWO tokens, sharing the one weight decode. Both
   operands are macro arguments and both are parenthesised for the
   reason the block above gives. This is what makes the paired
   group-of-four path possible: the SSE2 tier's part32_x16_2 is the
   128-bit form of the same idea (src/ops_kernel_dot_sse2.h), and its
   gcc row kernel already runs the two axes together. */
#define LZ_P32_Q8_VEC2(w, XA_, XB_, OA_, OB_) do { \
        __m128i wb0_ = _mm_loadu_si128((const __m128i *)(const void *)(w)); \
        __m128i wb1_ = _mm_loadu_si128((const __m128i *)(const void *)((w) + 16)); \
        __m256i w0_ = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb0_), 8); \
        __m256i w1_ = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb1_), 8); \
        OA_ = _mm256_add_epi32(\
            _mm256_madd_epi16(w0_, _mm256_loadu_si256((const __m256i *)(const void *)(XA_))), \
            _mm256_madd_epi16(w1_, _mm256_loadu_si256((const __m256i *)(const void *)((XA_) + 16)))); \
        OB_ = _mm256_add_epi32(\
            _mm256_madd_epi16(w0_, _mm256_loadu_si256((const __m256i *)(const void *)(XB_))), \
            _mm256_madd_epi16(w1_, _mm256_loadu_si256((const __m256i *)(const void *)((XB_) + 16)))); \
    } while (0)

#define LZ_P32_Q8_VEC(w, x, OUT_) do { \
        __m128i wb0 = _mm_loadu_si128((const __m128i *)(w)); \
        __m128i wb1 = _mm_loadu_si128((const __m128i *)((w) + 16)); \
        __m256i w0 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb0), 8); \
        __m256i w1 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb1), 8); \
        __m256i x0 = _mm256_loadu_si256((const __m256i *)(x)); \
        __m256i x1 = _mm256_loadu_si256((const __m256i *)((x) + 16)); \
        OUT_ = _mm256_add_epi32(_mm256_madd_epi16(w0, x0), \
                                _mm256_madd_epi16(w1, x1)); \
    } while (0)

static int32_t part32_x32_avx2(const int8_t *w, const int16_t *x) {
    __m256i s_;
    LZ_P32_Q8_VEC(w, x, s_);
    return fold8_to_scalar_avx2(s_);
}

/* Four groups' partials to four scalars, in group order, in one
   transposed reduction. hadd is lane-local: after the first two, a_
   holds the low four lanes' half sums of groups 0 and 1 and b_ the
   same for groups 2 and 3, each with lane 0 covering elements 0-3 and
   lane 1 covering 4-7. The third hadd interleaves those so that the
   lane-wise add that follows pairs each group's two half sums, leaving
   the four totals side by side in lane 0. Nothing is
   reassociated: each group's eight products are still summed into that
   group's own scalar, and integer addition is exact, so the four
   values are the same ones four fold8 calls would have produced. */
#define LZ_FOLD4(V0_, V1_, V2_, V3_, OUT4_) do { \
        __m256i a_ = _mm256_hadd_epi32(V0_, V1_); \
        __m256i b_ = _mm256_hadd_epi32(V2_, V3_); \
        __m256i c_ = _mm256_hadd_epi32(a_, b_); \
        __m256i d_ = _mm256_add_epi32(c_, _mm256_permute2x128_si256(c_, c_, 0x01)); \
        _mm_storeu_si128((__m128i *)(void *)(OUT4_), _mm256_castsi256_si128(d_)); \
    } while (0)


/* 128 elements (four 32-element groups) at once. Only the four folds are
   amortized - the weight decode and the madds are per group either way -
   which is exactly what the SSE2 tier found worth doing at its own
   width. */
static void part128_q8_avx2(const int8_t *w, const int16_t *x,
                            int32_t *acc4) {
    __m256i v0, v1, v2, v3;
    LZ_P32_Q8_VEC(w,      x,      v0);
    LZ_P32_Q8_VEC(w + 32, x + 32, v1);
    LZ_P32_Q8_VEC(w + 64, x + 64, v2);
    LZ_P32_Q8_VEC(w + 96, x + 96, v3);
    LZ_FOLD4(v0, v1, v2, v3, acc4);
}

/* A/B for the 128-element paths, the SSE2 side's LZ_G128_GCC in shape
   and purpose: -DLZ_AVX2_G4=0 falls back to per-32 for the whole tier.
   They sit inside the pair test and inside `nb MOD 4 == 0`, so no
   runtime knob reaches them.

   A value in the loop conditions rather than #if blocks: at 0 the quad
   loop does not run, its index stays 0, and the per-32 tail starts
   there. */
#ifndef LZ_AVX2_G4
#define LZ_AVX2_G4 1
#endif /* LZ_AVX2_G4 */

/* ---- Q8 row kernel -------------------------------------------------------
   SSE2 twin: row_q8_sse2_intrin, src/ops_sse2.c. No Watcom twin -
   AVX2 has none, unlike every other tier in this codebase. */
void row_q8_avx2_intrin(const lz_row_ctx *c) {
    const int8_t *wr = (const int8_t *)c->w4;
    const int8_t *wend = (const int8_t *)c->pf_end;
    const int pfm = lz_prefetch_mode();
    int tk, g;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* g_pair: two tokens share one weight load, the same knob
               the SSE2 tier's dot32_x16_mmx_2 reads. The decode below -
               two 16-byte loads, two vpmovzxbw, two shifts - is paid
               once and serves both, because the weight stream is the
               larger of the two and does not change between tokens. */
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                /* Four groups by two tokens: one weight decode serves the
                   pair and one transposed fold serves each token. nb MOD 4
                   left over runs in the tail below. */
                for (g = 0; LZ_AVX2_G4 && g + 3 < c->nb; g += 4) {
                    __m256i a0, a1, a2, a3, b0, b1, b2, b3;
                    const int8_t *wp = wr + (size_t)g * 32;
                    /* One per FOUR groups: the same distance, mode and
                       guard the single-token group loop and the per-32
                       tail in this kernel use. Without it a paired q8 row
                       - which is all quads when nb MOD 4 == 0 - issued no
                       prefetch at all. */
                    { const int8_t *lz_pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                      if (lz_pf_ < wend) LZ_AVX2_PF(lz_pf_, pfm); }
                    const int16_t *xp = xwt + (size_t)g * 32;
                    const int16_t *xq = xw2 + (size_t)g * 32;
                    LZ_P32_Q8_VEC2(wp,      xp,      xq,      a0, b0);
                    LZ_P32_Q8_VEC2(wp + 32, xp + 32, xq + 32, a1, b1);
                    LZ_P32_Q8_VEC2(wp + 64, xp + 64, xq + 64, a2, b2);
                    LZ_P32_Q8_VEC2(wp + 96, xp + 96, xq + 96, a3, b3);
                    LZ_FOLD4(a0, a1, a2, a3, acc + g);
                    LZ_FOLD4(b0, b1, b2, b3, accb + g);
                }
                for (; g < c->nb; g++) {
                    const int8_t *w = wr + (size_t)g * 32;
                    /* Once per PAIR, not once per token: this is a hint
                       about the weight row, and the second token reads
                       the bytes the first just pulled in - the same
                       sentence ops_matmul.c's SSE2 paired loop carries. */
                    const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                    if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                    __m256i w0 = _mm256_slli_epi16(
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)w)), 8);
                    __m256i w1 = _mm256_slli_epi16(
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)(w + 16))), 8);
                    const int16_t *xa = xwt + (size_t)g * 32;
                    const int16_t *xb = xw2 + (size_t)g * 32;
                    acc[g] = fold8_to_scalar_avx2(_mm256_add_epi32(
                        _mm256_madd_epi16(w0, _mm256_loadu_si256(
                            (const __m256i *)(const void *)xa)),
                        _mm256_madd_epi16(w1, _mm256_loadu_si256(
                            (const __m256i *)(const void *)(xa + 16)))));
                    accb[g] = fold8_to_scalar_avx2(_mm256_add_epi32(
                        _mm256_madd_epi16(w0, _mm256_loadu_si256(
                            (const __m256i *)(const void *)xb)),
                        _mm256_madd_epi16(w1, _mm256_loadu_si256(
                            (const __m256i *)(const void *)(xb + 16)))));
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            if (LZ_AVX2_G4 && (c->nb & 3) == 0) {
                /* Groups of four, one prefetch per four - the same shape
                   and the same distance LZ_Q8_GROUP4 uses, so the tier
                   issues FEWER prefetches here than the per-32 path,
                   never more. */
                for (g = 0; g < c->nb; g += 4) {
                    const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                    if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                    part128_q8_avx2(wr + (size_t)g * 32,
                                    xwt + (size_t)g * 32, acc + g);
                }
                continue;
            }
            for (g = 0; g < c->nb; g++) {
                const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                acc[g] = part32_x32_avx2(wr + (size_t)g * 32,
                                          xwt + (size_t)g * 32);
            }
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
#define LZ_P32_Q41_VEC(w, x, OUT_) do { \
        const __m128i m0f = _mm_set1_epi8(0x0F); \
        __m128i b  = _mm_loadu_si128((const __m128i *)(w)); \
        __m128i lo = _mm_and_si128(b, m0f); \
        __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f); \
        __m256i lo16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8); \
        __m256i hi16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8); \
        __m256i x0 = _mm256_loadu_si256((const __m256i *)(x)); \
        __m256i x1 = _mm256_loadu_si256((const __m256i *)((x) + 16)); \
        __m256i s = _mm256_add_epi32(_mm256_madd_epi16(lo16, x0), \
                                      _mm256_madd_epi16(hi16, x1)); \
        OUT_ = s; \
    } while (0)

static int32_t part32_q41_avx2(const unsigned char *w, const int16_t *x) {
    __m256i s_;
    LZ_P32_Q41_VEC(w, x, s_);
    return fold8_to_scalar_avx2(s_);
}

/* 128 elements (four groups) at once: the four folds become one
   transposed reduction, exactly as the Q8 path documents. */
static void part128_q41_avx2(const unsigned char *w, const int16_t *x, int32_t *acc4) {
    /* SIXTEEN bytes of weight per group, not thirty-two: a Q4_1 sub-block
       is one nibble per element, so four groups are 64 bytes of plane
       where Q8_0's are 128. The activation side is 32 int16 a group
       whichever format it belongs to. */
    __m256i v0, v1, v2, v3;
    LZ_P32_Q41_VEC(w,      x,      v0);
    LZ_P32_Q41_VEC(w + 16, x + 32, v1);
    LZ_P32_Q41_VEC(w + 32, x + 64, v2);
    LZ_P32_Q41_VEC(w + 48, x + 96, v3);
    LZ_FOLD4(v0, v1, v2, v3, acc4);
}

/* ---- Q4_1 row kernel ------------------------------------------------
   SSE2 twin: row_q41_sse2_intrin, src/ops_sse2.c. */
/* The same sub-block against two tokens, sharing the nibble decode: the
   mask, the shift and the two widens are the larger half of this
   format's work and none of them depends on the token. */
#define LZ_P32_Q41_VEC2(w, xa, xb, OA_, OB_) do { \
        const __m128i m0f = _mm_set1_epi8(0x0F); \
        __m128i b  = _mm_loadu_si128((const __m128i *)(const void *)(w)); \
        __m128i lo = _mm_and_si128(b, m0f); \
        __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f); \
        __m256i lo16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8); \
        __m256i hi16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8); \
        OA_ = (_mm256_add_epi32( \
            _mm256_madd_epi16(lo16, _mm256_loadu_si256((const __m256i *)(const void *)(xa))), \
            _mm256_madd_epi16(hi16, _mm256_loadu_si256((const __m256i *)(const void *)((xa) + 16))))); \
        OB_ = (_mm256_add_epi32( \
            _mm256_madd_epi16(lo16, _mm256_loadu_si256((const __m256i *)(const void *)(xb))), \
            _mm256_madd_epi16(hi16, _mm256_loadu_si256((const __m256i *)(const void *)((xb) + 16))))); \
    } while (0)

static void part32_q41_avx2_2(const unsigned char *w,
                              const int16_t *xa, const int16_t *xb,
                              int32_t *pa, int32_t *pb) {
    __m256i va_, vb_;
    LZ_P32_Q41_VEC2(w, xa, xb, va_, vb_);
    *pa = fold8_to_scalar_avx2(va_);
    *pb = fold8_to_scalar_avx2(vb_);
}


void row_q41_avx2_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *wend = (const unsigned char *)c->pf_end;
    const int pfm = lz_prefetch_mode();
    int tk, s;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                /* Four groups by two tokens: one weight decode serves the
                   pair and one transposed fold serves each token. nb MOD 4
                   left over runs in the tail below. */
                for (s = 0; LZ_AVX2_G4 && s + 3 < c->nb; s += 4) {
                    /* Eight live vectors, so gcc spills a pair to the
                       stack (unaligned vmovdqu, nothing faults). Measured
                       at nt=2 nb=32: 42.0 ns/row here against 60.3 with
                       the pair disabled, which does not spill. */
                    __m256i a0, a1, a2, a3, b0, b1, b2, b3;
                    /* One per FOUR groups - see the q8 kernel's note. */
                    { const unsigned char *lz_pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16;
                      if (lz_pf_ < wend) LZ_AVX2_PF(lz_pf_, pfm); }
                    LZ_P32_Q41_VEC2(wn + (size_t)s * 16, xwt + (size_t)s * 32, xw2 + (size_t)s * 32, a0, b0);
                    LZ_P32_Q41_VEC2(wn + (size_t)s * 16 + 16, xwt + (size_t)s * 32 + 32, xw2 + (size_t)s * 32 + 32, a1, b1);
                    LZ_P32_Q41_VEC2(wn + (size_t)s * 16 + 32, xwt + (size_t)s * 32 + 64, xw2 + (size_t)s * 32 + 64, a2, b2);
                    LZ_P32_Q41_VEC2(wn + (size_t)s * 16 + 48, xwt + (size_t)s * 32 + 96, xw2 + (size_t)s * 32 + 96, a3, b3);
                    LZ_FOLD4(a0, a1, a2, a3, acc + s);
                    LZ_FOLD4(b0, b1, b2, b3, accb + s);
                }
                for (; s < c->nb; s++) {
                    const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16;
                    if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                    part32_q41_avx2_2(wn + (size_t)s * 16,
                                      xwt + (size_t)s * 32,
                                      xw2 + (size_t)s * 32,
                                      &acc[s], &accb[s]);
                }
                tk++;
                xwt += c->in_dim;
                acc += c->nb;
                continue;
            }
            if (LZ_AVX2_G4 && (c->nb & 3) == 0) {
                for (s = 0; s < c->nb; s += 4) {
                    const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16;
                    if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                    part128_q41_avx2(wn + (size_t)s * 16,
                                     xwt + (size_t)s * 32, acc + s);
                }
                continue;
            }
            for (s = 0; s < c->nb; s++) {
                const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16;
                if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                acc[s] = part32_q41_avx2(wn + (size_t)s * 16,
                                          xwt + (size_t)s * 32);
            }
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
#define LZ_P32_Q61_VEC(w4, w2, x, OUT_) do { \
        const __m128i m0f = _mm_set1_epi8(0x0F); \
        const __m128i m03 = _mm_set1_epi8(0x03); \
        __m128i b  = _mm_loadu_si128((const __m128i *)(w4)); \
        __m128i t  = _mm_loadl_epi64((const __m128i *)(w2)); \
        __m128i g0 = _mm_and_si128(t, m03); \
        __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03); \
        __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03); \
        __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03); \
        __m128i lo = _mm_or_si128(_mm_and_si128(b, m0f), \
                                  _mm_slli_epi16(_mm_unpacklo_epi64(g0, g1), 4)); \
        __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b, 4), m0f), \
                                  _mm_slli_epi16(_mm_unpacklo_epi64(g2, g3), 4)); \
        __m256i lo16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8); \
        __m256i hi16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8); \
        __m256i x0 = _mm256_loadu_si256((const __m256i *)(x)); \
        __m256i x1 = _mm256_loadu_si256((const __m256i *)((x) + 16)); \
        __m256i s = _mm256_add_epi32(_mm256_madd_epi16(lo16, x0), \
                                      _mm256_madd_epi16(hi16, x1)); \
        OUT_ = s; \
    } while (0)

static int32_t part32_q61_avx2(const unsigned char *w4, const unsigned char *w2, const int16_t *x) {
    __m256i s_;
    LZ_P32_Q61_VEC(w4, w2, x, s_);
    return fold8_to_scalar_avx2(s_);
}

static void part128_q61_avx2(const unsigned char *w4, const unsigned char *w2, const int16_t *x, int32_t *acc4) {
    __m256i v0, v1, v2, v3;
    LZ_P32_Q61_VEC(w4,      w2,      x,      v0);
    LZ_P32_Q61_VEC(w4 + 16, w2 + 8,  x + 32, v1);
    LZ_P32_Q61_VEC(w4 + 32, w2 + 16, x + 64, v2);
    LZ_P32_Q61_VEC(w4 + 48, w2 + 24, x + 96, v3);
    LZ_FOLD4(v0, v1, v2, v3, acc4);
}

/* The same sub-block against two tokens. This format's decode is the
   most expensive of the four (two planes, four masks, three shifts and
   two 64-bit merges before the widens), so it is the one that gains the
   most from being paid once for two tokens. */
#define LZ_P32_Q61_VEC2(w4, w2, xa, xb, OA_, OB_) do { \
        const __m128i m0f = _mm_set1_epi8(0x0F); \
        const __m128i m03 = _mm_set1_epi8(0x03); \
        __m128i b  = _mm_loadu_si128((const __m128i *)(const void *)(w4)); \
        __m128i t  = _mm_loadl_epi64((const __m128i *)(const void *)(w2)); \
        __m128i g0 = _mm_and_si128(t, m03); \
        __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03); \
        __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03); \
        __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03); \
        __m128i lo = _mm_or_si128(_mm_and_si128(b, m0f), \
                                  _mm_slli_epi16(_mm_unpacklo_epi64(g0, g1), 4)); \
        __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b, 4), m0f), \
                                  _mm_slli_epi16(_mm_unpacklo_epi64(g2, g3), 4)); \
        __m256i lo16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8); \
        __m256i hi16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8); \
        OA_ = (_mm256_add_epi32( \
            _mm256_madd_epi16(lo16, _mm256_loadu_si256((const __m256i *)(const void *)(xa))), \
            _mm256_madd_epi16(hi16, _mm256_loadu_si256((const __m256i *)(const void *)((xa) + 16))))); \
        OB_ = (_mm256_add_epi32( \
            _mm256_madd_epi16(lo16, _mm256_loadu_si256((const __m256i *)(const void *)(xb))), \
            _mm256_madd_epi16(hi16, _mm256_loadu_si256((const __m256i *)(const void *)((xb) + 16))))); \
    } while (0)

static void part32_q61_avx2_2(const unsigned char *w4,
                              const unsigned char *w2,
                              const int16_t *xa, const int16_t *xb,
                              int32_t *pa, int32_t *pb) {
    __m256i va_, vb_;
    LZ_P32_Q61_VEC2(w4, w2, xa, xb, va_, vb_);
    *pa = fold8_to_scalar_avx2(va_);
    *pb = fold8_to_scalar_avx2(vb_);
}


/* ---- Q6_1 row kernel ------------------------------------------------
   SSE2 twin: row_q61_sse2_intrin, src/ops_sse2.c. Reads both c->w4 (4-bit
   plane) and c->w2 (2-bit plane), each advanced at its own stride (16
   and 8 bytes per 32-element sub-block). */
void row_q61_avx2_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *w2 = (const unsigned char *)c->w2;
    const unsigned char *wend  = (const unsigned char *)c->pf_end;
    const unsigned char *wend2 = (const unsigned char *)c->pf_end2;
    const int pfm = lz_prefetch_mode();
    int tk, s;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                /* Four groups by two tokens. The SSE2 twin's paired loop is
                   per-32, so this shape is this tier's alone. */
                for (s = 0; LZ_AVX2_G4 && s + 3 < c->nb; s += 4) {
                    __m256i a0, a1, a2, a3, b0, b1, b2, b3;
                    const unsigned char *pf_  = wn + (size_t)(s + LZ_PF_DIST) * 16;
                    const unsigned char *pf2_ = w2 + (size_t)(s + LZ_PF_DIST) * 8;
                    if (pf_  < wend)  LZ_AVX2_PF(pf_,  pfm);
                    if (pf2_ < wend2) LZ_AVX2_PF(pf2_, pfm);
                    LZ_P32_Q61_VEC2(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                    xwt + (size_t)s * 32, xw2 + (size_t)s * 32, a0, b0);
                    LZ_P32_Q61_VEC2(wn + (size_t)(s + 1) * 16, w2 + (size_t)(s + 1) * 8,
                                    xwt + (size_t)(s + 1) * 32, xw2 + (size_t)(s + 1) * 32, a1, b1);
                    LZ_P32_Q61_VEC2(wn + (size_t)(s + 2) * 16, w2 + (size_t)(s + 2) * 8,
                                    xwt + (size_t)(s + 2) * 32, xw2 + (size_t)(s + 2) * 32, a2, b2);
                    LZ_P32_Q61_VEC2(wn + (size_t)(s + 3) * 16, w2 + (size_t)(s + 3) * 8,
                                    xwt + (size_t)(s + 3) * 32, xw2 + (size_t)(s + 3) * 32, a3, b3);
                    LZ_FOLD4(a0, a1, a2, a3, acc + s);
                    LZ_FOLD4(b0, b1, b2, b3, accb + s);
                }
                for (; s < c->nb; s++) {
                    /* BOTH planes, matching LZ_Q61_PF: the 2-bit plane is
                       a real, narrower hot stream, not belt-and-braces.
                       Once per pair, on the same reasoning as the other
                       formats' paired loops. */
                    const unsigned char *pf_  = wn + (size_t)(s + LZ_PF_DIST) * 16;
                    const unsigned char *pf2_ = w2 + (size_t)(s + LZ_PF_DIST) * 8;
                    if (pf_  < wend)  LZ_AVX2_PF(pf_,  pfm);
                    if (pf2_ < wend2) LZ_AVX2_PF(pf2_, pfm);
                    part32_q61_avx2_2(wn + (size_t)s * 16,
                                      w2 + (size_t)s * 8,
                                      xwt + (size_t)s * 32,
                                      xw2 + (size_t)s * 32,
                                      &acc[s], &accb[s]);
                }
                tk++;
                xwt += c->in_dim;
                acc += c->nb;
                continue;
            }
            if (LZ_AVX2_G4 && (c->nb & 3) == 0) {
                for (s = 0; s < c->nb; s += 4) {
                    const unsigned char *pf_  = wn + (size_t)(s + LZ_PF_DIST) * 16;
                    const unsigned char *pf2_ = w2 + (size_t)(s + LZ_PF_DIST) * 8;
                    if (pf_  < wend)  LZ_AVX2_PF(pf_,  pfm);
                    if (pf2_ < wend2) LZ_AVX2_PF(pf2_, pfm);
                    part128_q61_avx2(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                     xwt + (size_t)s * 32, acc + s);
                }
                continue;
            }
            for (s = 0; s < c->nb; s++) {
                const unsigned char *pf_  = wn + (size_t)(s + LZ_PF_DIST) * 16;
                const unsigned char *pf2_ = w2 + (size_t)(s + LZ_PF_DIST) * 8;
                if (pf_  < wend)  LZ_AVX2_PF(pf_,  pfm);
                if (pf2_ < wend2) LZ_AVX2_PF(pf2_, pfm);
                acc[s] = part32_q61_avx2(wn + (size_t)s * 16,
                                          w2 + (size_t)s * 8,
                                          xwt + (size_t)s * 32);
            }
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
/* The same, for TWO tokens: one pair of weight loads serves both. */
#define LZ_P32_Q16_VEC2(w, XA_, XB_, OA_, OB_) do { \
        __m256i w0_ = _mm256_loadu_si256((const __m256i *)(const void *)(w)); \
        __m256i w1_ = _mm256_loadu_si256((const __m256i *)(const void *)((w) + 16)); \
        OA_ = _mm256_add_epi32(\
            _mm256_madd_epi16(w0_, _mm256_loadu_si256((const __m256i *)(const void *)(XA_))), \
            _mm256_madd_epi16(w1_, _mm256_loadu_si256((const __m256i *)(const void *)((XA_) + 16)))); \
        OB_ = _mm256_add_epi32(\
            _mm256_madd_epi16(w0_, _mm256_loadu_si256((const __m256i *)(const void *)(XB_))), \
            _mm256_madd_epi16(w1_, _mm256_loadu_si256((const __m256i *)(const void *)((XB_) + 16)))); \
    } while (0)

#define LZ_P32_Q16_VEC(w, x, OUT_) do { \
        __m256i w0 = _mm256_loadu_si256((const __m256i *)(w)); \
        __m256i w1 = _mm256_loadu_si256((const __m256i *)((w) + 16)); \
        __m256i x0 = _mm256_loadu_si256((const __m256i *)(x)); \
        __m256i x1 = _mm256_loadu_si256((const __m256i *)((x) + 16)); \
        __m256i s = _mm256_add_epi32(_mm256_madd_epi16(w0, x0), \
                                      _mm256_madd_epi16(w1, x1)); \
        OUT_ = s; \
    } while (0)

static int32_t part32_q16_avx2(const int16_t *w, const int16_t *x) {
    __m256i s_;
    LZ_P32_Q16_VEC(w, x, s_);
    return fold8_to_scalar_avx2(s_);
}

/* 128 elements (four groups) at once: the four folds become one
   transposed reduction, exactly as the Q8 path documents. */
static void part128_q16_avx2(const int16_t *w, const int16_t *x, int32_t *acc4) {
    __m256i v0, v1, v2, v3;
    LZ_P32_Q16_VEC(w,      x,      v0);
    LZ_P32_Q16_VEC(w + 32, x + 32, v1);
    LZ_P32_Q16_VEC(w + 64, x + 64, v2);
    LZ_P32_Q16_VEC(w + 96, x + 96, v3);
    LZ_FOLD4(v0, v1, v2, v3, acc4);
}

/* ---- Q16_0 row kernel -------------------------------------------------
   SSE2 twin: row_q16_sse2_intrin, src/ops_sse2.c. No Watcom twin. Pairs
   like the rest - inline, because this format's weight side is one pair
   of plain 256-bit loads with no decode to share, so a helper would only
   add a call. */
void row_q16_avx2_intrin(const lz_row_ctx *c) {
    const int16_t *wr = (const int16_t *)c->w4;
    const int16_t *wend = (const int16_t *)c->pf_end;
    const int pfm = lz_prefetch_mode();
    int tk, g;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Two tokens, one weight load - row_q8_avx2_intrin's pairing,
               and here the weight side is the ONLY decode there is: two
               plain 256-bit loads serve both tokens. */
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                /* Four groups by two tokens: one weight decode serves the
                   pair and one transposed fold serves each token. nb MOD 4
                   left over runs in the tail below. */
                for (g = 0; LZ_AVX2_G4 && g + 3 < c->nb; g += 4) {
                    __m256i a0, a1, a2, a3, b0, b1, b2, b3;
                    const int16_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                    if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                    LZ_P32_Q16_VEC2(wr + (size_t)g * 32,
                                    xwt + (size_t)g * 32, xw2 + (size_t)g * 32, a0, b0);
                    LZ_P32_Q16_VEC2(wr + (size_t)(g + 1) * 32,
                                    xwt + (size_t)(g + 1) * 32, xw2 + (size_t)(g + 1) * 32, a1, b1);
                    LZ_P32_Q16_VEC2(wr + (size_t)(g + 2) * 32,
                                    xwt + (size_t)(g + 2) * 32, xw2 + (size_t)(g + 2) * 32, a2, b2);
                    LZ_P32_Q16_VEC2(wr + (size_t)(g + 3) * 32,
                                    xwt + (size_t)(g + 3) * 32, xw2 + (size_t)(g + 3) * 32, a3, b3);
                    LZ_FOLD4(a0, a1, a2, a3, acc + g);
                    LZ_FOLD4(b0, b1, b2, b3, accb + g);
                }
                for (; g < c->nb; g++) {
                    const int16_t *w = wr + (size_t)g * 32;
                    const int16_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                    if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                    __m256i w0 = _mm256_loadu_si256((const __m256i *)(const void *)w);
                    __m256i w1 = _mm256_loadu_si256((const __m256i *)(const void *)(w + 16));
                    const int16_t *xa = xwt + (size_t)g * 32;
                    const int16_t *xb = xw2 + (size_t)g * 32;
                    acc[g] = fold8_to_scalar_avx2(_mm256_add_epi32(
                        _mm256_madd_epi16(w0, _mm256_loadu_si256(
                            (const __m256i *)(const void *)xa)),
                        _mm256_madd_epi16(w1, _mm256_loadu_si256(
                            (const __m256i *)(const void *)(xa + 16)))));
                    accb[g] = fold8_to_scalar_avx2(_mm256_add_epi32(
                        _mm256_madd_epi16(w0, _mm256_loadu_si256(
                            (const __m256i *)(const void *)xb)),
                        _mm256_madd_epi16(w1, _mm256_loadu_si256(
                            (const __m256i *)(const void *)(xb + 16)))));
                }
                tk++;
                xwt += c->in_dim;
                acc += c->nb;
                continue;
            }
            if (LZ_AVX2_G4 && (c->nb & 3) == 0) {
                for (g = 0; g < c->nb; g += 4) {
                    const int16_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                    if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                    part128_q16_avx2(wr + (size_t)g * 32,
                                     xwt + (size_t)g * 32, acc + g);
                }
                continue;
            }
            for (g = 0; g < c->nb; g++) {
                const int16_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                if (pf_ < wend) LZ_AVX2_PF(pf_, pfm);
                acc[g] = part32_q16_avx2(wr + (size_t)g * 32,
                                          xwt + (size_t)g * 32);
            }
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
        /* max, not the cmpgt/and/andnot/or select: masking the sign bit
           makes v non-negative and acc starts at zero and only ever
           takes v's value, so both operands are in [0, 2^31) and the
           signed maximum IS the maximum. Four instructions to one, and
           the loop-carried chain through acc drops from four to one.
           The tail below already leans on the same range: it compares
           o.u[] as UNSIGNED. */
        acc = _mm256_max_epi32(acc, v);
    }
    {
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i m4 = _mm_max_epi32(hi, lo);
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

   Overflow bound, derived from the actual call site (ops_epi.c:1351,
   epi_q41_join), not from the row kernels' ~1.3e8 dot-product cap that
   an earlier draft of this comment cited by mistake - that number
   belongs to a different accumulator. Here `a` is g_epi_zg[tk]: a sum
   of up to LZ_EPI_MAX_R (16) products each bounded 133,165,088
   (ops_epi.c:632-637's own derivation), so |a| <= 16 * 133,165,088 =
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
    /* Two accumulator pairs per term rather than one: chaining each
       iteration's four add_epi64 onto a single register makes the
       loop-carried dependency four adds deep, and splitting it 2+2 makes
       it two. Free to do - the total is an exact int64 and any grouping
       of the same products gives the same sum. */
    __m128i acc_hi0 = zero128, acc_hi1 = zero128;
    __m128i acc_als0 = zero128, acc_als1 = zero128;
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
            acc_hi0 = _mm_add_epi64(acc_hi0,
                      _mm_unpacklo_epi32(pa_lo, _mm_srai_epi32(pa_lo, 31)));
            acc_hi1 = _mm_add_epi64(acc_hi1,
                      _mm_unpackhi_epi32(pa_lo, _mm_srai_epi32(pa_lo, 31)));
            acc_hi0 = _mm_add_epi64(acc_hi0,
                      _mm_unpacklo_epi32(pa_hi, _mm_srai_epi32(pa_hi, 31)));
            acc_hi1 = _mm_add_epi64(acc_hi1,
                      _mm_unpackhi_epi32(pa_hi, _mm_srai_epi32(pa_hi, 31)));
            acc_als0 = _mm_add_epi64(acc_als0,
                      _mm_unpacklo_epi32(pl_lo, _mm_srai_epi32(pl_lo, 31)));
            acc_als1 = _mm_add_epi64(acc_als1,
                      _mm_unpackhi_epi32(pl_lo, _mm_srai_epi32(pl_lo, 31)));
            acc_als0 = _mm_add_epi64(acc_als0,
                      _mm_unpacklo_epi32(pl_hi, _mm_srai_epi32(pl_hi, 31)));
            acc_als1 = _mm_add_epi64(acc_als1,
                      _mm_unpackhi_epi32(pl_hi, _mm_srai_epi32(pl_hi, 31)));
        }
    }
    _mm_storeu_si128((__m128i *)(void *)t, _mm_add_epi64(acc_hi0, acc_hi1));
    s_hi  = t[0] + t[1];
    _mm_storeu_si128((__m128i *)(void *)t, _mm_add_epi64(acc_als0, acc_als1));
    s_als = t[0] + t[1];
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

/* ---- FLOAT Hadamard stage, AVX2 tier -----------------------------------
   SSE1 twin: lz_fwht_stage_f32_sse (src/ops_mmx_sse.c). Same butterfly,
   one add and one sub per element pair, 8-wide instead of 4-wide, same
   4-wide tail for a len that is not a multiple of 8 (len is a power of
   two, so only len == 4 reaches it).

   Bit-identity is structural, not an argument about lane order: there
   is no reduction here at all - each output element is one IEEE add or
   sub of the same two inputs the 4-wide body would have used, and
   IEEE 754 round-to-nearest is width-independent. No constant enters
   either, which is the same reason fwht.c's own comment gives for the
   transform being bit-identical between wcc386 and gcc. */
void lz_fwht_stage_f32_avx2(float *y, int n, int len) {
    int i, j;
    for (i = 0; i < n; i += (len << 1)) {
        float *a = y + i;
        float *b = y + i + len;
        for (j = 0; j + 7 < len; j += 8) {
            __m256 u = _mm256_loadu_ps(a + j);
            __m256 v = _mm256_loadu_ps(b + j);
            _mm256_storeu_ps(a + j, _mm256_add_ps(u, v));
            _mm256_storeu_ps(b + j, _mm256_sub_ps(u, v));
        }
        for (; j < len; j += 4) {
            __m128 u = _mm_loadu_ps(a + j);
            __m128 v = _mm_loadu_ps(b + j);
            _mm_storeu_ps(a + j, _mm_add_ps(u, v));
            _mm_storeu_ps(b + j, _mm_sub_ps(u, v));
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

/* ---- GDN/KDA fixed-point pass 2, mul32, AVX2 tier ----------------------
   SSE2 twin: lz_p2_mul32_sse2, src/ops_sse2.c. Same algorithm, 16
   elements per iteration instead of 8 (for(q<2) instead of for(q<4)).

   hi/lo sign-extend: cvtepi8_epi16 replaces the twin's two per-half
   idioms (dup-and-shift for lo; zero-extend-into-the-high-byte for hi),
   and hw is then shifted left 8 the way the twin's own comment reads
   "hi * 256" off its unpack. The shift is bit-identical to that multiply
   because (int8_t)hi[k] is in [-128,127], so *256 is in [-32768,32512],
   inside int16 with no wraparound - the same bound the SSE2 comment
   relies on.

   The trap: unpacklo/unpackhi_epi16 read each 128-bit lane
   independently. H and d hold 16 elements (lane0 = elements 0-7, lane1 =
   elements 8-15). unpacklo(H,d) interleaves each lane's LOW 4, giving
   elements {0,1,2,3, 8,9,10,11} - not the contiguous {0..7} the
   algorithm needs. Measured on an iota-input probe: without the permute
   unpacklo yields 0 1 2 3 8 9 10 11, with 0xD8 it yields 0 1 2 3 4 5 6
   7. Fix: permute4x64_epi64(_, 0xD8) on both H and d BEFORE unpacking.
   0xD8 reorders the four 64-bit (4-element) chunks [0-3,4-7,8-11,12-15]
   to [0-3,8-11,4-7,12-15]; after that each lane's low-4/high-4 line up
   with contiguous elements, so unpacklo/unpackhi after the permute give
   back {0..7} and {8..15} respectively. */
void lz_p2_mul32_avx2(const int8_t *hi, const int8_t *lo,
                      const int16_t *dq, lz_p2_blk *blk) {
    __m256i acc1 = _mm256_setzero_si256(), acc2 = _mm256_setzero_si256();
    __m256i mul  = _mm256_broadcastsi128_si256(
        _mm_load_si128((const __m128i *)(const void *)blk->mul));
    int q;

    for (q = 0; q < 2; q++) {
        __m256i hw, lw, H, d, Hp, dp, p0, p1, s0, s1, a0, a1;

        hw = _mm256_slli_epi16(
            _mm256_cvtepi8_epi16(_mm_loadu_si128(
                (const __m128i *)(const void *)(hi + q * 16))), 8);
        lw = _mm256_cvtepi8_epi16(_mm_loadu_si128(
            (const __m128i *)(const void *)(lo + q * 16)));
        H  = _mm256_add_epi16(hw, lw);
        d  = _mm256_loadu_si256((const __m256i *)(const void *)(dq + q * 16));

        Hp = _mm256_permute4x64_epi64(H, 0xD8);
        dp = _mm256_permute4x64_epi64(d, 0xD8);
        p0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(Hp, dp), mul);
        p1 = _mm256_madd_epi16(_mm256_unpackhi_epi16(Hp, dp), mul);
        _mm256_storeu_si256((__m256i *)(void *)(blk->a + q * 16),     p0);
        _mm256_storeu_si256((__m256i *)(void *)(blk->a + q * 16 + 8), p1);

        s0 = _mm256_srai_epi32(p0, 31);
        a0 = _mm256_sub_epi32(_mm256_xor_si256(p0, s0), s0);   /* |A| */
        s1 = _mm256_srai_epi32(p1, 31);
        a1 = _mm256_sub_epi32(_mm256_xor_si256(p1, s1), s1);
        /* pmaxsd, not the cmpgt/and/andnot/or idiom: every operand here
           is an ABSOLUTE value, so both sides are non-negative and the
           signed maximum is the maximum. One instruction against four,
           and it shortens the loop-carried chain through acc1/acc2 from
           about three cycles an iteration to one. The idiom it replaces
           is what a portable max looks like; this is what the value
           range licenses. */
        acc1 = _mm256_max_epi32(acc1, a0);
        acc2 = _mm256_max_epi32(acc2, a1);
    }
    /* Fold the two 8-lane accumulators down to the 4 lanes the caller
       reads (blk->amax[4]), the same "different tiers hand back the same
       shape" contract lz_p2_mul32_sse2's own comment states. Lane j of
       the result is the max over the eight elements whose index is j mod
       4, which is exactly the partition the SSE2 twin's final
       cmpgt-fold leaves behind - the caller maxes the four lanes, so any
       partition works and this one matches. */
    {
        __m128i lo1 = _mm256_castsi256_si128(acc1);
        __m128i hi1 = _mm256_extracti128_si256(acc1, 1);
        __m128i lo2 = _mm256_castsi256_si128(acc2);
        __m128i hi2 = _mm256_extracti128_si256(acc2, 1);
        __m128i g1  = _mm_max_epi32(lo1, hi1);
        __m128i g2  = _mm_max_epi32(lo2, hi2);
        __m128i gg  = _mm_max_epi32(g1, g2);
        _mm_store_si128((__m128i *)(void *)blk->amax, gg);
    }
}

/* ---- GDN/KDA fixed-point pass 2, split32, AVX2 tier --------------------
   SSE2 twin: lz_p2_split32_sse2, src/ops_sse2.c. Same algorithm over all
   32 elements in one pass - four __m256i covering elements 0-7 / 8-15 /
   16-23 / 24-31 - instead of the twin's two o-iterations of four
   __m128i.

   Two chained narrowing packs, each needing its own
   permute4x64_epi64(_, 0xD8) fixup, for the same reason mul32's unpack
   does: packs_epi32/packs_epi16 interleave 128-bit lanes rather than
   running across the whole register. Measured element orders off the raw
   pack results: packs_epi32(A0, A1) gives 16-bit lanes holding
   {0-3, 8-11, 4-7, 12-15}, and packs_epi16(H0, H1) gives byte lanes
   holding {0-7, 16-23, 8-15, 24-31}. 0xD8 selects the four 64-bit
   chunks in the order 0,2,1,3, which restores contiguous element order
   in both cases. The rounding/shift/clamp steps between the two packs
   are per-lane and need no fixup.

   k128 and kclp come from blk, one broadcast load each, the same source
   the MMX and SSE2 tiers read: this kernel has no constants of its own,
   so a caller that writes different values gets the same arithmetic from
   those tiers. (The SSE1 tier reads k128 too, but clamps with kmin and a
   single pmaxsw instead of kclp.) */
void lz_p2_split32_avx2(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m256i rnd  = _mm256_broadcastsi128_si256(
        _mm_load_si128((const __m128i *)(const void *)blk->rnd));
    __m128i cnt  = _mm_load_si128((const __m128i *)(const void *)blk->cnt);
    __m256i k128 = _mm256_broadcastsi128_si256(
        _mm_load_si128((const __m128i *)(const void *)blk->k128));
    __m256i kclp = _mm256_broadcastsi128_si256(
        _mm_load_si128((const __m128i *)(const void *)blk->kclp));
    __m256i A0, A1, A2, A3;
    __m256i HN0, HN1, H0, H1, L0, L1, T;

    A0 = _mm256_loadu_si256((const __m256i *)(const void *)(blk->a + 0));
    A1 = _mm256_loadu_si256((const __m256i *)(const void *)(blk->a + 8));
    A2 = _mm256_loadu_si256((const __m256i *)(const void *)(blk->a + 16));
    A3 = _mm256_loadu_si256((const __m256i *)(const void *)(blk->a + 24));
    A0 = _mm256_sra_epi32(_mm256_add_epi32(A0, rnd), cnt);
    A1 = _mm256_sra_epi32(_mm256_add_epi32(A1, rnd), cnt);
    A2 = _mm256_sra_epi32(_mm256_add_epi32(A2, rnd), cnt);
    A3 = _mm256_sra_epi32(_mm256_add_epi32(A3, rnd), cnt);

    HN0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(A0, A1), 0xD8);
    HN1 = _mm256_permute4x64_epi64(_mm256_packs_epi32(A2, A3), 0xD8);

    H0 = _mm256_srai_epi16(_mm256_add_epi16(HN0, k128), 8);
    H1 = _mm256_srai_epi16(_mm256_add_epi16(HN1, k128), 8);
    L0 = _mm256_sub_epi16(HN0, _mm256_slli_epi16(H0, 8));
    L1 = _mm256_sub_epi16(HN1, _mm256_slli_epi16(H1, 8));
    L0 = _mm256_adds_epi16(_mm256_subs_epi16(L0, kclp), kclp);  /* max(l, -127) */
    L1 = _mm256_adds_epi16(_mm256_subs_epi16(L1, kclp), kclp);

    T = _mm256_permute4x64_epi64(_mm256_packs_epi16(H0, H1), 0xD8);
    _mm256_storeu_si256((__m256i *)(void *)oh, T);
    T = _mm256_permute4x64_epi64(_mm256_packs_epi16(L0, L1), 0xD8);
    _mm256_storeu_si256((__m256i *)(void *)ol, T);
}

/* ---- Q8 activation rounding, AVX2 tier -------------------------------
   SSE2 twin: lz_q8round32_simd, src/ops_sse2.c. That kernel
   processes 32 elements as two 16-wide iterations (b<2); this processes
   all 32 in one 32-wide pass, so both narrowing packs it chains
   (packs_epi32 then packs_epi16) need a permute4x64(_,0xD8) fixup - same
   derivation as GDN split32's two-pack chain, but simpler: one iteration,
   not two, so there's no o-loop to interact with the permute placement.

   cvtps_epi32 follows MXCSR's rounding mode; q8_round follows
   lz_fastfp() instead - truncation toward zero when it is on, the magic
   add's round-to-nearest-even when off. The two spell the same rule only
   when the MXCSR mode agrees with lz_fastfp(), which is the shipping
   pairing; when they disagree they are two different rules, not one rule
   written twice.

   The clamp is [-127, +127], the same contract lz_quantize_q8's scalar
   tail and lz_gdn_quantize_2p's both state in so many words, and the
   same one lz_q8round32_simd's p127/m127 hold. It is not a tunable: the
   tiers owe each other bit-identity, and since inv is 127/amax the
   largest element of a group lands on +127 often enough that any other
   ceiling diverges on nearly every group. */
void lz_q8round32_avx2(const float *x, int8_t *o, const float *pinv) {
    __m256 vi = _mm256_set1_ps(*pinv);
    __m256i p127 = _mm256_set1_epi16(127), m127 = _mm256_set1_epi16(-127);
    __m256i a0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x),      vi));
    __m256i a1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + 8),  vi));
    __m256i a2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + 16), vi));
    __m256i a3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + 24), vi));
    __m256i l = _mm256_permute4x64_epi64(_mm256_packs_epi32(a0, a1), 0xD8);
    __m256i h = _mm256_permute4x64_epi64(_mm256_packs_epi32(a2, a3), 0xD8);
    __m256i t;
    l = _mm256_max_epi16(_mm256_min_epi16(l, p127), m127);
    h = _mm256_max_epi16(_mm256_min_epi16(h, p127), m127);
    t = _mm256_permute4x64_epi64(_mm256_packs_epi16(l, h), 0xD8);
    _mm256_storeu_si256((__m256i *)(void *)o, t);
}

/* ---- Q15 table interpolation, AVX2 tier --------------------------------
   SSE2 twin: lz_lerp_q15_simd (src/ops_sse2.c); precondition and
   contract are lz_lerp_q15_mmx's (src/ops_mmx.h): |b - a| <= 32767 and
   frac in [0, 32767], which is what makes the SSE2 twin's 16-bit
   mullo/mulhi pair reconstruct the exact 32-bit product. This body
   computes that same product with a single 32-bit multiply and shifts
   it the same way, so it is the same arithmetic, not a wider one - and
   `a` stays int32 because it reaches 32768, exactly as the MMX
   comment says. No saturation can be reached on either path, which is
   the one place these two could have differed. */
void lz_lerp_q15_avx2(const int32_t *a, const int32_t *b,
                      const int32_t *frac, int32_t *out, int n) {
    int k;
    for (k = 0; k + 7 < n; k += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + k));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + k));
        __m256i vf = _mm256_loadu_si256((const __m256i *)(const void *)(frac + k));
        __m256i d = _mm256_sub_epi32(vb, va);
        __m256i p = _mm256_srai_epi32(_mm256_mullo_epi32(d, vf), 15);
        _mm256_storeu_si256((__m256i *)(void *)(out + k),
                            _mm256_add_epi32(p, va));
    }
    for (; k < n; k++)
        out[k] = a[k] + (((b[k] - a[k]) * frac[k]) >> 15);
}

/* ---- lz_exp_fixed's Q20 Taylor, AVX2 tier ------------------------------
   SSE2 twin: lz_exp_q20_simd (src/ops_sse2.c), whose exp_mul_shr helper
   this mirrors lane for lane, 8 wide instead of 4. Read that one for
   why the multiplies are UNSIGNED and the shifts LOGICAL: every operand
   here is non-negative (s in [0, 2^15], tab and cq positive), so the
   unsigned 64-bit lane arithmetic IS the signed arithmetic the scalar
   tail spells out. Widening the register changes no value.

   The one thing that has to be re-derived for 256 bits is the odd-lane
   pass: srli_si256/slli_si256 are the per-128-bit-lane byte shifts, so
   they bring element 1 (and 5) down into lane 0's low half and put the
   product back in its high half - the same swap the SSE2 helper does,
   just done independently in each half, which is exactly what leaves
   the result in element order here. */
static __m256i exp_mul_shr_avx2(__m256i v, __m256i w, __m256i rnd, int sh) {
    __m256i e = _mm256_srli_epi64(_mm256_add_epi64(_mm256_mul_epu32(v, w), rnd), sh);
    __m256i o = _mm256_srli_epi64(
        _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_si256(v, 4),
                                          _mm256_srli_si256(w, 4)), rnd), sh);
    return _mm256_or_si256(e, _mm256_slli_si256(o, 4));
}

void lz_exp_q20_avx2(const int32_t *tab, const int32_t *s,
                     int32_t *prod, int n) {
    const __m256i kln2   = _mm256_set1_epi32(726817);
    const __m256i kln2sq = _mm256_set1_epi32(251906);
    const __m256i r20    = _mm256_set1_epi64x((long long)1 << 19);
    const __m256i r40q   = _mm256_set1_epi64x(((long long)1 << 39)
                                            + ((long long)1 << 60));
    const __m256i zero   = _mm256_setzero_si256();
    int k;

    for (k = 0; k + 7 < n; k += 8) {
        __m256i sv = _mm256_loadu_si256((const __m256i *)(const void *)(s + k));
        __m256i tv = _mm256_loadu_si256((const __m256i *)(const void *)(tab + k));
        __m256i ss = exp_mul_shr_avx2(sv, sv, zero, 0);   /* s*s, exact in int32 */
        __m256i t1 = exp_mul_shr_avx2(sv, kln2,   r20,  20);
        __m256i t2 = exp_mul_shr_avx2(ss, kln2sq, r40q, 40);  /* carries the 2^20 */
        __m256i cq = _mm256_add_epi32(t1, t2);
        _mm256_storeu_si256((__m256i *)(void *)(prod + k),
                            exp_mul_shr_avx2(tv, cq, r20, 20));
    }
    for (; k < n; k++) {
        int32_t sk = s[k], cq;
        cq = (1 << 20)
           + (int32_t)((((lz_i64)726817 * sk) + (1 << 19)) >> 20)
           + (int32_t)((((lz_i64)251906 * ((lz_i64)sk * sk))
                        + (LZ_I64_C(1) << 39)) >> 40);
        prod[k] = (int32_t)((((lz_i64)tab[k] * cq) + (1 << 19)) >> 20);
    }
}

/* ---- lz_matmul's F32-weight row, AVX2 tier -----------------------------
   SSE1 twin: lz_matmul_row_sse (src/ops_mmx_sse.c), which carries the
   argument this inherits: the caller's eight scalars are two 128-bit
   accumulators because lane k accumulates exactly the sequence scalar
   a_k does, in the same order - not a re-association. ONE 256-bit
   accumulator is those same eight lanes, so the argument is unchanged
   and the body is simply wider.

   mul then add, never fused: -mfma is deliberately not passed to this
   TU (Makefile's ENG_AVX2 rule), and a fused form would round once
   where the scalar loop rounds twice. The return value is the SSE
   twin's - the count consumed, so the caller's tail loop starts where
   this stopped. */
int lz_matmul_row_avx2(const float *row, const float *x, int in_dim,
                       float *acc8) {
    __m256 a = _mm256_setzero_ps();
    int j = 0, ng = in_dim & ~7;
    for (; j < ng; j += 8)
        a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(row + j),
                                           _mm256_loadu_ps(x + j)));
    _mm256_storeu_ps(acc8, a);
    return ng;
}

/* ---- Attention scoring's single-group dot, AVX2 tier -------------------
   SSE2 twin: lz_dot32_x16_sse2 (src/ops_sse2.c), and this is the same
   kind of REUSE that file's twin is: the body is part32_x32_avx2 above,
   which already folds its four int32 partials to the one scalar this
   caller wants. Same w<<8 convention and therefore the same x256 the
   attention path cancels with a single 1/256. No second body, and no
   lane-order question the row kernel has not already answered. */
int32_t lz_dot32_x16_avx2(const int8_t *w, const int16_t *x) {
    return part32_x32_avx2(w, x);
}

/* ---- T2 (ternary) row kernel ------------------------------------------
   SSE2 twin: part32_t2 / row_t2_sse2_intrin (src/ops_kernel_dot_sse2.h,
   src/ops_sse2.c). The 8-byte weight plane decodes to 32 two-bit codes
   exactly as that header documents: byte j's bits [1:0] are element j,
   [3:2] element j+8, [5:4] j+16, [7:6] j+24, so the two 64-bit merges
   give codes 0..15 and 16..31 in order.

   cvtepu8_epi16 then widens each 16-byte half to 16 int16 in element
   order, which is the same value the SSE2 twin builds with
   unpacklo_epi8(zero, code) - the code in the HIGH byte, i.e. code*256,
   the x256 fold epi_q41 cancels with one 1/256. Codes are 0..2 and read
   unsigned, so there is nothing to sign-extend.

   The lane grouping differs from the SSE2 twin's on purpose and costs
   nothing: every product is an int32 integer, the row kernel's output
   is their total, and integer addition is exact and associative, so any
   grouping gives the same acc[s] - the licence this file's header
   states. */
#define LZ_P32_T2_VEC(w2, x, OUT_) do { \
        const __m128i m03 = _mm_set1_epi8(0x03); \
        __m128i t  = _mm_loadl_epi64((const __m128i *)(const void *)(w2)); \
        __m128i g0 = _mm_and_si128(t, m03); \
        __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03); \
        __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03); \
        __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03); \
        __m128i lo = _mm_unpacklo_epi64(g0, g1);     /* codes  0..15 */ \
        __m128i hi = _mm_unpacklo_epi64(g2, g3);     /* codes 16..31 */ \
        __m256i w0 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8); \
        __m256i w1 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8); \
        __m256i x0 = _mm256_loadu_si256((const __m256i *)(const void *)(x)); \
        __m256i x1 = _mm256_loadu_si256((const __m256i *)(const void *)((x) + 16)); \
        __m256i s = _mm256_add_epi32(_mm256_madd_epi16(w0, x0), \
                                     _mm256_madd_epi16(w1, x1)); \
        OUT_ = s; \
    } while (0)

static int32_t part32_t2_avx2(const unsigned char *w2, const int16_t *x) {
    __m256i s_;
    LZ_P32_T2_VEC(w2, x, s_);
    return fold8_to_scalar_avx2(s_);
}

static void part128_t2_avx2(const unsigned char *w2, const int16_t *x, int32_t *acc4) {
    __m256i v0, v1, v2, v3;
    LZ_P32_T2_VEC(w2,      x,      v0);
    LZ_P32_T2_VEC(w2 + 8,  x + 32, v1);
    LZ_P32_T2_VEC(w2 + 16, x + 64, v2);
    LZ_P32_T2_VEC(w2 + 24, x + 96, v3);
    LZ_FOLD4(v0, v1, v2, v3, acc4);
}

/* The same sub-block against two tokens. The 2-bit decode is the whole
   weight side of this format (four masks, three shifts, two merges,
   two widens, two shifts), so sharing it is worth more here than the
   two extra activation loads cost. */
#define LZ_P32_T2_VEC2(w2, xa, xb, OA_, OB_) do { \
        const __m128i m03 = _mm_set1_epi8(0x03); \
        __m128i t  = _mm_loadl_epi64((const __m128i *)(const void *)(w2)); \
        __m128i g0 = _mm_and_si128(t, m03); \
        __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03); \
        __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03); \
        __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03); \
        __m128i lo = _mm_unpacklo_epi64(g0, g1); \
        __m128i hi = _mm_unpacklo_epi64(g2, g3); \
        __m256i w0 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8); \
        __m256i w1 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8); \
        OA_ = (_mm256_add_epi32( \
            _mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i *)(const void *)(xa))), \
            _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i *)(const void *)((xa) + 16))))); \
        OB_ = (_mm256_add_epi32( \
            _mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i *)(const void *)(xb))), \
            _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i *)(const void *)((xb) + 16))))); \
    } while (0)

static void part32_t2_avx2_2(const unsigned char *w2,
                             const int16_t *xa, const int16_t *xb,
                             int32_t *pa, int32_t *pb) {
    __m256i va_, vb_;
    LZ_P32_T2_VEC2(w2, xa, xb, va_, vb_);
    *pa = fold8_to_scalar_avx2(va_);
    *pb = fold8_to_scalar_avx2(vb_);
}


void row_t2_avx2_intrin(const lz_row_ctx *c) {
    const unsigned char *p2 = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                /* Four groups by two tokens. The SSE2 twin's paired loop is
                   per-32, so this shape is this tier's alone. */
                for (s = 0; LZ_AVX2_G4 && s + 3 < c->nb; s += 4) {
                    __m256i a0, a1, a2, a3, b0, b1, b2, b3;
                    LZ_P32_T2_VEC2(p2 + (size_t)s * 8,
                                   xwt + (size_t)s * 32, xw2 + (size_t)s * 32, a0, b0);
                    LZ_P32_T2_VEC2(p2 + (size_t)(s + 1) * 8,
                                   xwt + (size_t)(s + 1) * 32, xw2 + (size_t)(s + 1) * 32, a1, b1);
                    LZ_P32_T2_VEC2(p2 + (size_t)(s + 2) * 8,
                                   xwt + (size_t)(s + 2) * 32, xw2 + (size_t)(s + 2) * 32, a2, b2);
                    LZ_P32_T2_VEC2(p2 + (size_t)(s + 3) * 8,
                                   xwt + (size_t)(s + 3) * 32, xw2 + (size_t)(s + 3) * 32, a3, b3);
                    LZ_FOLD4(a0, a1, a2, a3, acc + s);
                    LZ_FOLD4(b0, b1, b2, b3, accb + s);
                }
                for (; s < c->nb; s++)
                    part32_t2_avx2_2(p2 + (size_t)s * 8,
                                     xwt + (size_t)s * 32,
                                     xw2 + (size_t)s * 32,
                                     &acc[s], &accb[s]);
                tk++;
                xwt += c->in_dim;
                acc += c->nb;
                continue;
            }
            if (LZ_AVX2_G4 && (c->nb & 3) == 0) {
                /* No prefetch here, group-of-four or not: the SSE2 tier's
                   T2 row kernel does not prefetch either, and this route
                   is not allowed to exceed what the existing operators
                   issue. */
                for (s = 0; s < c->nb; s += 4)
                    part128_t2_avx2(p2 + (size_t)s * 8,
                                    xwt + (size_t)s * 32, acc + s);
                continue;
            }
            for (s = 0; s < c->nb; s++)
                acc[s] = part32_t2_avx2(p2 + (size_t)s * 8,
                                        xwt + (size_t)s * 32);
        }
    }
}

/* ---- Attention wsum's row-pair kernel, AVX2 tier ----------------------
   SSE2 twin: lz_wsum_pair_sse2 (src/ops_sse2.c), whose contract this
   reproduces lane for lane: acc32[i] += rowA[i]*coef[0] +
   rowB[i]*coef[1] over the 32 elements of one group.

   The difficulty here is the LANE MAP, not the arithmetic. 256-bit
   unpack works INSIDE each 128-bit lane, so a naive unpacklo/unpackhi
   pair does not leave the products in element order - the hazard
   part32_x32_avx2's comment describes for the row kernels, seen from
   the other side. The two madds come out as

     lo = [e0 e1 e2 e3 | e8  e9  e10 e11]
     hi = [e4 e5 e6 e7 | e12 e13 e14 e15]

   and one permute2x128 each restores the two contiguous halves (0x20
   takes the low lane of both operands, 0x31 the high lane of both), so
   acc32 is written in the element order the SSE2 body writes it in.
   Nothing is reassociated: every lane's sum is formed from the same two
   products as the SSE2 lane at the same index, and integer addition is
   exact, so the two tiers agree bit for bit by construction. */
void lz_wsum_pair_avx2(const int8_t *rowA, const int8_t *rowB,
                       const int16_t *coef, int32_t *acc32) {
    int32_t cw;
    __m256i c;
    int off;
    memcpy(&cw, coef, 4);           /* (ckA, ckB), one dword */
    c = _mm256_set1_epi32(cw);
    for (off = 0; off < 32; off += 16) {
        __m128i ra = _mm_loadu_si128((const __m128i *)(const void *)(rowA + off));
        __m128i rb = _mm_loadu_si128((const __m128i *)(const void *)(rowB + off));
        __m256i a = _mm256_cvtepi8_epi16(ra);
        __m256i b = _mm256_cvtepi8_epi16(rb);
        __m256i lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), c);
        __m256i hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), c);
        __m256i o0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        __m256i o1 = _mm256_permute2x128_si256(lo, hi, 0x31);
        int32_t *acc = acc32 + off;
        _mm256_storeu_si256((__m256i *)(void *)acc,
            _mm256_add_epi32(
                _mm256_loadu_si256((const __m256i *)(const void *)acc), o0));
        _mm256_storeu_si256((__m256i *)(void *)(acc + 8),
            _mm256_add_epi32(
                _mm256_loadu_si256((const __m256i *)(const void *)(acc + 8)), o1));
    }
}

/* ---- norm_ss_fixed's element loop, AVX2 tier --------------------------
   SSE2 twin: lz_norm_ss_sse2 (src/ops_sse2.c), 8 elements a pass against
   this one's 16. Read that one for the contract: whole groups, qout
   optional, the sum of the CLAMPED int16 squares accumulated in int64,
   and a return value the caller continues from.

   THE PACK NEEDS A LANE FIXUP THE SSE2 TWIN DOES NOT. _mm_packs_epi32
   narrows within each 128-bit lane, so packs(a, b) leaves the 64-bit
   chunks as [a0-3, b0-3, a4-7, b4-7] - elements 0-3, 8-11, 4-7, 12-15 -
   while the SSE2 body's _mm_packs_epi32(a, b) gives elements 0-7 in
   order. permute4x64(_, 0xD8) takes the chunk order [0,2,1,3], which is
   the same fixup lz_q8round32_avx2's first pack chains. Without it the
   STORED qout would be misordered; the accumulator would not care, since
   the sum of all lanes is order-free.

   The clamp is the SSE2 twin's: packs_epi32 saturates int32 to
   [-32768, 32767], so only the low end needs an instruction, and
   max_epi16 against -32767 is it. */
int lz_norm_ss_avx2(const float *x, int n, float sc, short *qout,
                    lz_i64 *acc) {
    __m256i vacc = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256();
    const __m256i lo = _mm256_set1_epi16((short)-32767);
    const __m256 vsc = _mm256_set1_ps(sc);
    union { __m256i v; lz_i64 q[4]; } out;
    int i, ng = n & ~15;
    for (i = 0; i < ng; i += 16) {
        __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_loadu_ps(x + i), vsc));
        __m256i b = _mm256_cvtps_epi32(_mm256_mul_ps(
            _mm256_loadu_ps(x + i + 8), vsc));
        __m256i w = _mm256_permute4x64_epi64(
            _mm256_packs_epi32(a, b), 0xD8);
        __m256i p;
        w = _mm256_max_epi16(w, lo);
        if (qout) _mm256_storeu_si256((__m256i *)(void *)(qout + i), w);
        p = _mm256_madd_epi16(w, w);
        vacc = _mm256_add_epi64(vacc, _mm256_unpacklo_epi32(p, zero));
        vacc = _mm256_add_epi64(vacc, _mm256_unpackhi_epi32(p, zero));
    }
    out.v = vacc;
    *acc = out.q[0] + out.q[1] + out.q[2] + out.q[3];
    return ng;
}

/* ---- int32 -> float chunk fold, AVX2 tier ------------------------------
   SSE2 twin: lz_i32f_acc32_simd (src/ops_sse2.c), eight passes of four
   against this one's four passes of eight: accf[d] += (float)acc32[d]
   for d in 0..31, the same correctly rounded conversion (cvtepi32_ps)
   and the same per-element add. No lane-order question exists here -
   nothing is unpacked, interleaved or reduced - so widening 4 lanes to
   8 changes no value. */
void lz_i32f_acc32_avx2(float *accf, const int32_t *acc32) {
    int b;
    for (b = 0; b < 4; b++) {
        __m256i vi = _mm256_loadu_si256((const __m256i *)(const void *)
                                        (acc32 + b * 8));
        _mm256_storeu_ps(accf + b * 8,
                         _mm256_add_ps(_mm256_loadu_ps(accf + b * 8),
                                       _mm256_cvtepi32_ps(vi)));
    }
}

/* lz_rope_avx2_compiled_in lives in ops_rope.c, not here - see the
   comment on its definition for why. */

#endif /* LZ_AVX2_TU && !__WATCOMC__ */
