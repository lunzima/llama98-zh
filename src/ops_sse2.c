/* Every gcc/clang SSE2-intrinsics kernel, pulled out of src/ops.c and
   its included headers into the one translation unit built with
   -msse2. Code structure, not a correctness fix (task #22, per the
   user: "SSE2 next stage should also be separated for code clarity") -
   %xmm does not alias the x87 stack the way %mm does, so nothing here
   is implicated in the bug src/ops_mmx.c's split exists to prevent. It
   still gets its own TU: the ISA-floor gate needs a clean per-file
   boundary the same way src/ops_mmx.c/src/ops_mmx_sse.c do, and the
   32-bit x87 target gains a real SSE2 tier from it (see
   build/x87/build_engine.sh's header) - before this split, ops.c's own
   -mno-sse2 meant the SSE2 row kernels were only ever reachable on the
   x86-64 dev build, where ops.c happened to be compiled with -msse2
   itself.

   -march=i686, not i586: real SSE2 hardware in the target family (P4
   Northwood, Pentium M) is i686+ - SSE2 debuted with NetBurst, after
   Pentium Pro. Same honesty argument as src/ops_mmx_sse.c's header:
   this file's own -march floor costs nothing because no SSE2-capable
   machine is below it anyway. Measured (see ops_kernel_dot_sse2.h and
   this file's own build) that -march=i686 -mmmx -msse -msse2 together
   raise nothing beyond ordinary SSE2 codegen - no AVX, no FMA, nothing
   unrequested - the same empirical check that caught -msse quietly
   enabling CMOV in src/ops_mmx.c, done again for this flag set on its
   own object.

   Watcom is unaffected and never compiles this file: its own #pragma
   aux SSE2 kernels live in ops.c's included headers exactly as
   before. */
#include <stdint.h>
#include <string.h>

#include "ops_sse2.h"

#if defined(LZ_SSE2_TU) && !defined(__WATCOMC__)

#include <emmintrin.h>

#include "ops_kernel_dot_sse2.h"

/* ---- Q8 row kernel ------------------------------------------------------
   Watcom twin: row_q8_sse2_asm, src/ops.c. */
void row_q8_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                        const int16_t *xw, int nb, int nt, int in_dim,
                        const void *pf_end, const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    int tk, g;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (g = 0; g + 3 < nb; g += 4) {
            _mm_storeu_si128((__m128i *)(void *)(acc + g),
                fold4_sse2(
                part32_x16(wr + (size_t)(g + 0) * 32, xwt + (size_t)(g + 0) * 32),
                part32_x16(wr + (size_t)(g + 1) * 32, xwt + (size_t)(g + 1) * 32),
                part32_x16(wr + (size_t)(g + 2) * 32, xwt + (size_t)(g + 2) * 32),
                part32_x16(wr + (size_t)(g + 3) * 32, xwt + (size_t)(g + 3) * 32)));
        }
        for (; g < nb; g++) {           /* tail when nb % 4 != 0 */
            __m128i p = part32_x16(wr + (size_t)g * 32, xwt + (size_t)g * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[g] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- Q4_1 row kernel -----------------------------------------------------
   Watcom twin: row_q41_sse2_asm, src/ops.c. */
void row_q41_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    int tk, s;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s + 3 < nb; s += 4) {
            _mm_storeu_si128((__m128i *)(void *)(acc + s),
                fold4_sse2(
                part32_q41(wn + (size_t)(s + 0) * 16, xwt + (size_t)(s + 0) * 32),
                part32_q41(wn + (size_t)(s + 1) * 16, xwt + (size_t)(s + 1) * 32),
                part32_q41(wn + (size_t)(s + 2) * 16, xwt + (size_t)(s + 2) * 32),
                part32_q41(wn + (size_t)(s + 3) * 16, xwt + (size_t)(s + 3) * 32)));
        }
        for (; s < nb; s++) {
            __m128i p = part32_q41(wn + (size_t)s * 16, xwt + (size_t)s * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[s] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- T2 row kernel --------------------------------------------------------
   Watcom twin: row_t2_sse2_asm, src/ops.c. No pairing (unmeasured for
   this format, same as the other tiers). */
void row_t2_sse2_intrin(int32_t *acc32, const void *w4, const void *w2v,
                        const int16_t *xw, int nb, int nt, int in_dim,
                        const void *pf_end, const void *pf_end2) {
    const unsigned char *p2 = (const unsigned char *)w4;
    int tk, s;
    (void)w2v; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++) {
            __m128i p = part32_t2(p2 + (size_t)s * 8, xwt + (size_t)s * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[s] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- Q6_1 row kernel -----------------------------------------------------
   Watcom twin: row_q61_sse2_asm, src/ops.c. */
void row_q61_sse2_intrin(int32_t *acc32, const void *w4, const void *w2v,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *w2 = (const unsigned char *)w2v;
    int tk, s;
    (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++) {
            __m128i p = part32_q61(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                   xwt + (size_t)s * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[s] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- Q16_0 row kernel ----------------------------------------------------
   Watcom twin: row_q16_sse2_asm, src/ops.c. */
void row_q16_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end, const void *pf_end2) {
    const int16_t *wr = (const int16_t *)w4;
    int tk, g;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (g = 0; g + 3 < nb; g += 4) {
            _mm_storeu_si128((__m128i *)(void *)(acc + g),
                fold4_sse2(
                part32_q16(wr + (size_t)(g + 0) * 32, xwt + (size_t)(g + 0) * 32),
                part32_q16(wr + (size_t)(g + 1) * 32, xwt + (size_t)(g + 1) * 32),
                part32_q16(wr + (size_t)(g + 2) * 32, xwt + (size_t)(g + 2) * 32),
                part32_q16(wr + (size_t)(g + 3) * 32, xwt + (size_t)(g + 3) * 32)));
        }
        for (; g < nb; g++) {           /* tail when nb % 4 != 0 */
            __m128i p = part32_q16(wr + (size_t)g * 32, xwt + (size_t)g * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[g] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- GDN/KDA fixed-point pass 2, SSE2 tier -------------------------------
   Watcom twin: lz_p2_mul32_sse2_asm / lz_p2_split32_sse2_asm,
   src/ops_kernel_p2.h. Instruction for instruction the MMX pair in
   src/ops_mmx.c, on xmm - see that pair's comments for the algorithm;
   this is the same operations twice as wide, not a different one.
   Alignment is load-bearing here (movdqa, and the shift-count operand
   FAULTS on a misaligned address) - ops.c's p2_blk() provides it by
   hand since Watcom's -zp4 caps what the type system can promise. */
void lz_p2_mul32_sse2(const int8_t *hi, const int8_t *lo,
                      const int16_t *dq, lz_p2_blk *blk) {
    __m128i zero = _mm_setzero_si128();
    __m128i acc1 = zero, acc2 = zero;
    __m128i mul  = _mm_load_si128((const __m128i *)(const void *)blk->mul);
    __m128i m;
    int q;

    for (q = 0; q < 4; q++) {
        __m128i hw, lw, H, d, p0, p1, s0, s1, a0, a1;

        hw = _mm_loadl_epi64((const __m128i *)(const void *)(hi + q * 8));
        hw = _mm_unpacklo_epi8(zero, hw);                    /* hi * 256 */
        lw = _mm_loadl_epi64((const __m128i *)(const void *)(lo + q * 8));
        lw = _mm_unpacklo_epi8(lw, lw);
        lw = _mm_srai_epi16(lw, 8);                          /* lo, signed */
        H  = _mm_add_epi16(hw, lw);
        d  = _mm_loadu_si128((const __m128i *)(const void *)(dq + q * 8));
        p0 = _mm_madd_epi16(_mm_unpacklo_epi16(H, d), mul);
        p1 = _mm_madd_epi16(_mm_unpackhi_epi16(H, d), mul);
        _mm_store_si128((__m128i *)(void *)(blk->a + q * 8),     p0);
        _mm_store_si128((__m128i *)(void *)(blk->a + q * 8 + 4), p1);
        s0 = _mm_srai_epi32(p0, 31);
        a0 = _mm_sub_epi32(_mm_xor_si128(p0, s0), s0);       /* |A| */
        s1 = _mm_srai_epi32(p1, 31);
        a1 = _mm_sub_epi32(_mm_xor_si128(p1, s1), s1);
        m    = _mm_cmpgt_epi32(acc1, a0);
        acc1 = _mm_or_si128(_mm_and_si128(acc1, m), _mm_andnot_si128(m, a0));
        m    = _mm_cmpgt_epi32(acc2, a1);
        acc2 = _mm_or_si128(_mm_and_si128(acc2, m), _mm_andnot_si128(m, a1));
    }
    /* Fold to the four lanes the caller reads. MMX leaves two 2-lane
       accumulators there and SSE2 one 4-lane one; either way the caller
       folds four int32, so the two tiers hand back the same shape. */
    m    = _mm_cmpgt_epi32(acc1, acc2);
    acc1 = _mm_or_si128(_mm_and_si128(acc1, m), _mm_andnot_si128(m, acc2));
    _mm_store_si128((__m128i *)(void *)blk->amax, acc1);
}

void lz_p2_split32_sse2(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m128i rnd  = _mm_load_si128((const __m128i *)(const void *)blk->rnd);
    __m128i cnt  = _mm_load_si128((const __m128i *)(const void *)blk->cnt);
    __m128i k128 = _mm_load_si128((const __m128i *)(const void *)blk->k128);
    __m128i kclp = _mm_load_si128((const __m128i *)(const void *)blk->kclp);
    int o;

    for (o = 0; o < 2; o++) {
        __m128i a0, a1, a2, a3, hn0, hn1, h0, h1, l0, l1, t;

        a0 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16));
        a1 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16 + 4));
        a2 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16 + 8));
        a3 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16 + 12));
        a0 = _mm_sra_epi32(_mm_add_epi32(a0, rnd), cnt);
        a1 = _mm_sra_epi32(_mm_add_epi32(a1, rnd), cnt);
        a2 = _mm_sra_epi32(_mm_add_epi32(a2, rnd), cnt);
        a3 = _mm_sra_epi32(_mm_add_epi32(a3, rnd), cnt);
        hn0 = _mm_packs_epi32(a0, a1);
        hn1 = _mm_packs_epi32(a2, a3);
        h0 = _mm_srai_epi16(_mm_add_epi16(hn0, k128), 8);
        h1 = _mm_srai_epi16(_mm_add_epi16(hn1, k128), 8);
        l0 = _mm_sub_epi16(hn0, _mm_slli_epi16(h0, 8));
        l1 = _mm_sub_epi16(hn1, _mm_slli_epi16(h1, 8));
        l0 = _mm_adds_epi16(_mm_subs_epi16(l0, kclp), kclp); /* max(l, -127) */
        l1 = _mm_adds_epi16(_mm_subs_epi16(l1, kclp), kclp);
        t = _mm_packs_epi16(h0, h1);
        _mm_storeu_si128((__m128i *)(void *)(oh + o * 16), t);
        t = _mm_packs_epi16(l0, l1);
        _mm_storeu_si128((__m128i *)(void *)(ol + o * 16), t);
    }
}

/* ---- RoPE -----------------------------------------------------------------
   4-wide float RoPE. Each (i, h) pair is an independent IEEE mul/add,
   so per-element results are bit-identical to the scalar path in
   src/ops.c's lz_rope; only the loop order changes (head outer, i
   inner) to stream one head's two rotated halves contiguously. The
   cos/sin row is interleaved [c0,s0,c1,s1,...], so one 8-float load +
   two shuffles give the c and s vectors. half % 4 tail falls back to
   scalar. */
void lz_rope_sse2(float *v, int n_heads, int head_dim,
                  int rotary_dim, int pos, const float *cs) {
    int half = rotary_dim / 2;
    int h, i;
    const float *row = cs + (size_t)pos * half * 2;
    for (h = 0; h < n_heads; h++) {
        float *base = v + (size_t)h * head_dim;
        for (i = 0; i + 4 <= half; i += 4) {
            __m128 lo = _mm_loadu_ps(row + i * 2);
            __m128 hi = _mm_loadu_ps(row + i * 2 + 4);
            __m128 c = _mm_shuffle_ps(lo, hi, 0x88); /* [c0,c1,c2,c3] */
            __m128 s = _mm_shuffle_ps(lo, hi, 0xDD); /* [s0,s1,s2,s3] */
            __m128 x0 = _mm_loadu_ps(base + i);
            __m128 x1 = _mm_loadu_ps(base + i + half);
            __m128 n0 = _mm_sub_ps(_mm_mul_ps(x0, c), _mm_mul_ps(x1, s));
            __m128 n1 = _mm_add_ps(_mm_mul_ps(x1, c), _mm_mul_ps(x0, s));
            _mm_storeu_ps(base + i, n0);
            _mm_storeu_ps(base + i + half, n1);
        }
        for (; i < half; i++) {
            float c0 = row[i * 2], s0 = row[i * 2 + 1];
            float x0 = base[i], x1 = base[i + half];
            base[i]        = x0 * c0 - x1 * s0;
            base[i + half] = x1 * c0 + x0 * s0;
        }
    }
}

/* ---- Q8 group-scale amax --------------------------------------------------
   Watcom twin: lz_amax32_sse2_asm, src/ops_kernel_amax.h. Four lanes,
   no MMX register touched (pshufd is SSE2), so no emms owed. See that
   file's header for what is being vectorized and why the compare-and-
   blend below is exact, not approximate. */
unsigned lz_amax32_sse2(const float *x, int n) {
    __m128i acc = _mm_setzero_si128();
    __m128i msk = _mm_set1_epi32(0x7FFFFFFF);
    int k;
    for (k = 0; k + 3 < n; k += 4) {
        __m128i v = _mm_and_si128(
            _mm_loadu_si128((const __m128i *)(const void *)(x + k)), msk);
        __m128i gt = _mm_cmpgt_epi32(v, acc);
        acc = _mm_or_si128(_mm_and_si128(v, gt), _mm_andnot_si128(gt, acc));
    }
    {
        union { __m128i m; unsigned u[4]; } o;
        unsigned a, b;
        o.m = acc;
        a = o.u[0] > o.u[1] ? o.u[0] : o.u[1];
        b = o.u[2] > o.u[3] ? o.u[2] : o.u[3];
        return a > b ? a : b;
    }
}

/* ---- Q8 activation rounding, SSE2 tier -----------------------------------
   Watcom twin: lz_q8round32_sse2_asm, src/ops_kernel_q8round.h.
   cvtps2dq converts 4 floats at once - no MMX register touched, unlike
   the SSE1 tier's cvtps2pi (src/ops_mmx_sse.c). */
void lz_q8round32_simd(const float *x, int8_t *o, const float *pinv) {
    __m128 vi = _mm_set1_ps(*pinv);
    __m128i p127 = _mm_set1_epi16(127), m127 = _mm_set1_epi16(-127);
    int b;
    for (b = 0; b < 2; b++) {
        const float *s = x + b * 16;
        __m128i a0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(s), vi));
        __m128i a1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(s + 4), vi));
        __m128i a2 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(s + 8), vi));
        __m128i a3 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(s + 12), vi));
        __m128i l = _mm_packs_epi32(a0, a1);
        __m128i h = _mm_packs_epi32(a2, a3);
        l = _mm_max_epi16(_mm_min_epi16(l, p127), m127);
        h = _mm_max_epi16(_mm_min_epi16(h, p127), m127);
        _mm_storeu_si128((__m128i *)(void *)(o + b * 16), _mm_packs_epi16(l, h));
    }
}

#endif /* LZ_SSE2_TU && !__WATCOMC__ */
