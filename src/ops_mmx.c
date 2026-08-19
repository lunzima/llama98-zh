/* Every gcc/clang function that writes a %mm register - the MMX
   register file, which aliases the x87 stack. The ONLY translation
   unit built with -mmmx (and -msse, for the SSE1-on-%mm cell), and it
   must stay that way: on a 32-bit x87 target, gcc and clang both use
   MMX for an 8-byte block store when -mmmx is on even with no MMX
   intrinsic anywhere nearby in the source, and neither issues emms
   before the x87 code that follows - which leaves the tag word full and
   makes the next fld overflow to real indefinite (0xFFC00000). Moving
   every %mm-touching function out of ops.c is what makes "ops.c is
   never built with -mmmx" a safe thing to do; see src/ops_mmx.h for the
   macro that tells ops.c's dispatch code these kernels still exist.

   Watcom is unaffected and untouched: it never compiles this file (not
   in $(ENG), not in any watcom-* target's source list) and keeps
   compiling its own #pragma aux MMX/SSE1 kernels straight out of ops.c
   and the ops_kernel_*.h headers, exactly as before this split.

   Kernels do NOT emit emms themselves (rule 6 item 6) - the caller
   emits one after its loop. That contract does not change here; only
   which translation unit the code compiling to %mm instructions lives
   in changes. */
#include <stdint.h>
#include <string.h>

#include "ops_mmx.h"

#if defined(LZ_MMX_TU) && !defined(__WATCOMC__)

#include "mmx_compat.h"
#include <mmintrin.h>

/* ---- wsum: attention scoring's weighted-sum dot, LZ_WSUM_PAIR's gcc
   twin. Watcom twin: lz_wsum_pair_asm, src/ops.c ~3960-4041.
   Sign-extend two int8 rows to int16, interleave so one pmaddwd covers a
   (rowA,rowB) pair against (ckA,ckB), accumulate into acc32 in place.
   Emits no emms; the driver does one per chunk, and that chunk's
   row-pair loop contains no x87 (rule 6 item 6). */
void lz_wsum_pair_mmx(const int8_t *rowA, const int8_t *rowB,
                      const int16_t *coef, int32_t *acc32) {
    __m64 c, ra, rb, a0, b0, p, q, acc;
    int off;
    memcpy(&c, coef, 8);
    for (off = 0; off < 32; off += 8) {
        memcpy(&ra, rowA + off, 8);
        memcpy(&rb, rowB + off, 8);

        a0 = _mm_srai_pi16(_mm_unpacklo_pi8(ra, ra), 8);
        b0 = _mm_srai_pi16(_mm_unpacklo_pi8(rb, rb), 8);
        p = _mm_unpacklo_pi16(a0, b0);        /* [a0,b0,a1,b1] */
        q = _mm_unpackhi_pi16(a0, b0);        /* [a2,b2,a3,b3] */
        p = _mm_madd_pi16(p, c);
        q = _mm_madd_pi16(q, c);
        memcpy(&acc, acc32 + off, 8);
        acc = _mm_add_pi32(acc, p);
        memcpy(acc32 + off, &acc, 8);
        memcpy(&acc, acc32 + off + 2, 8);
        acc = _mm_add_pi32(acc, q);
        memcpy(acc32 + off + 2, &acc, 8);

        a0 = _mm_srai_pi16(_mm_unpackhi_pi8(ra, ra), 8);
        b0 = _mm_srai_pi16(_mm_unpackhi_pi8(rb, rb), 8);
        p = _mm_unpacklo_pi16(a0, b0);
        q = _mm_unpackhi_pi16(a0, b0);
        p = _mm_madd_pi16(p, c);
        q = _mm_madd_pi16(q, c);
        memcpy(&acc, acc32 + off + 4, 8);
        acc = _mm_add_pi32(acc, p);
        memcpy(acc32 + off + 4, &acc, 8);
        memcpy(&acc, acc32 + off + 6, 8);
        acc = _mm_add_pi32(acc, q);
        memcpy(acc32 + off + 6, &acc, 8);
    }
}

/* ---- GDN/KDA fixed-point pass 2. Watcom twins:
   lz_p2_mul32_mmx_asm / lz_p2_split32_mmx_asm / lz_p2_split32_sse_asm,
   src/ops_kernel_p2.h. Full algorithm description (what each field of
   lz_p2_blk means, the ISA inventory, why SSE1 has exactly one
   instruction of content here) lives with that file - it is unaffected
   by this split and stays the authority on the "what" and "why". */

/* 16-byte-aligned scratch, by hand: neither compiler can be asked for
   it here (Watcom is built with -zp4, capping member alignment at 4;
   C89/C99 have no alignment specifier), and SSE2's movdqa faults on a
   misaligned address. One buffer for the whole engine - the kernels
   below are called back to back inside one group and nothing
   re-enters between them. */
lz_p2_blk *p2_blk(void) {
    static char raw[sizeof(lz_p2_blk) + 15];
    size_t off = (size_t)((uintptr_t)(void *)raw & 15u);
    return (lz_p2_blk *)(void *)(raw + ((16u - off) & 15u));
}

/* memcpy for every load and store, not a cast to __m64*: the state
   planes belong to the caller and nothing here may assume their
   alignment. */
void lz_p2_mul32_mmx(const int8_t *hi, const int8_t *lo,
                     const int16_t *dq, lz_p2_blk *blk) {
    __m64 zero = _mm_setzero_si64();
    __m64 acc1 = _mm_setzero_si64();
    __m64 acc2 = _mm_setzero_si64();
    __m64 mul;
    int q;

    memcpy(&mul, blk->mul, 8);
    for (q = 0; q < 8; q++) {
        __m64 hw, lw, H, d, p0, p1, s0, s1, a0, a1, m;
        int32_t hv, lv;

        memcpy(&hv, hi + q * 4, 4);
        memcpy(&lv, lo + q * 4, 4);
        hw = _mm_unpacklo_pi8(zero, _mm_cvtsi32_si64(hv));   /* hi * 256 */
        lw = _mm_cvtsi32_si64(lv);
        lw = _mm_unpacklo_pi8(lw, lw);
        lw = _mm_srai_pi16(lw, 8);                           /* lo, signed */
        H  = _mm_add_pi16(hw, lw);
        memcpy(&d, dq + q * 4, 8);
        p0 = _mm_madd_pi16(_mm_unpacklo_pi16(H, d), mul);
        p1 = _mm_madd_pi16(_mm_unpackhi_pi16(H, d), mul);
        memcpy(blk->a + q * 4,     &p0, 8);
        memcpy(blk->a + q * 4 + 2, &p1, 8);
        s0 = _mm_srai_pi32(p0, 31);
        a0 = _mm_sub_pi32(_mm_xor_si64(p0, s0), s0);         /* |A| */
        s1 = _mm_srai_pi32(p1, 31);
        a1 = _mm_sub_pi32(_mm_xor_si64(p1, s1), s1);
        m    = _mm_cmpgt_pi32(acc1, a0);
        acc1 = _mm_or_si64(_mm_and_si64(acc1, m), _mm_andnot_si64(m, a0));
        m    = _mm_cmpgt_pi32(acc2, a1);
        acc2 = _mm_or_si64(_mm_and_si64(acc2, m), _mm_andnot_si64(m, a1));
    }
    memcpy(blk->amax,     &acc1, 8);
    memcpy(blk->amax + 2, &acc2, 8);
}

void lz_p2_split32_mmx(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m64 rnd, cnt, k128, kclp;
    int o;

    memcpy(&rnd,  blk->rnd,  8);
    memcpy(&cnt,  blk->cnt,  8);
    memcpy(&k128, blk->k128, 8);
    memcpy(&kclp, blk->kclp, 8);
    for (o = 0; o < 4; o++) {
        __m64 a0, a1, a2, a3, hn0, hn1, h0, h1, l0, l1, t;

        memcpy(&a0, blk->a + o * 8,     8);
        memcpy(&a1, blk->a + o * 8 + 2, 8);
        memcpy(&a2, blk->a + o * 8 + 4, 8);
        memcpy(&a3, blk->a + o * 8 + 6, 8);
        a0 = _mm_sra_pi32(_mm_add_pi32(a0, rnd), cnt);
        a1 = _mm_sra_pi32(_mm_add_pi32(a1, rnd), cnt);
        a2 = _mm_sra_pi32(_mm_add_pi32(a2, rnd), cnt);
        a3 = _mm_sra_pi32(_mm_add_pi32(a3, rnd), cnt);
        hn0 = _mm_packs_pi32(a0, a1);
        hn1 = _mm_packs_pi32(a2, a3);
        h0 = _mm_srai_pi16(_mm_add_pi16(hn0, k128), 8);
        h1 = _mm_srai_pi16(_mm_add_pi16(hn1, k128), 8);
        l0 = _mm_sub_pi16(hn0, _mm_slli_pi16(h0, 8));
        l1 = _mm_sub_pi16(hn1, _mm_slli_pi16(h1, 8));
        l0 = _mm_adds_pi16(_mm_subs_pi16(l0, kclp), kclp);   /* max(l, -127) */
        l1 = _mm_adds_pi16(_mm_subs_pi16(l1, kclp), kclp);
        t = _mm_packs_pi16(h0, h1);
        memcpy(oh + o * 8, &t, 8);
        t = _mm_packs_pi16(l0, l1);
        memcpy(ol + o * 8, &t, 8);
    }
}

/* lz_p2_split32_sse moved to src/ops_mmx_sse.c: it needs genuine SSE1,
   and gcc treats -msse as license to assume i686 (CMOV included) for
   the WHOLE translation unit, not just the SSE-touching functions in
   it - measured, see that file's header comment. This file stays
   MMX-only (-march=i586 -mmmx, no -msse) so its plain-MMX functions,
   reachable on Socket-7 parts with no CMOV, cannot pick one up. */

/* ---- GDN/KDA pass 1, 4-lane table contraction. Watcom twin:
   the #pragma aux declaration of this same name in src/ops.c (search
   LZ_GDN_HAVE_MMX). Sign-extend two int8 rows to int16, interleave so
   one pmaddwd consumes a (rowA,rowB) pair against a (cA,cB) coefficient
   pair, four accumulators for u/w x the low/high halves of the 4-lane
   window. */
void lz_gdn1_x4_asm(const int8_t *rowA, int32_t vd, const int16_t *ctab,
                    int32_t kpairs, int32_t *out8) {
    __m64 accU0 = _mm_setzero_si64(), accU1 = _mm_setzero_si64();
    __m64 accW0 = _mm_setzero_si64(), accW1 = _mm_setzero_si64();
    const int8_t *rA = rowA;
    int32_t kp;
    for (kp = 0; kp < kpairs; kp++) {
        __m64 a, b, pl, ph, t, cu, cw;
        int32_t a32, b32;
        memcpy(&a32, rA, 4);
        memcpy(&b32, rA + vd, 4);
        a = _mm_cvtsi32_si64(a32);
        b = _mm_cvtsi32_si64(b32);
        a = _mm_unpacklo_pi8(a, a);
        a = _mm_srai_pi16(a, 8);           /* 4 int16, sign-extended */
        b = _mm_unpacklo_pi8(b, b);
        b = _mm_srai_pi16(b, 8);
        pl = _mm_unpacklo_pi16(a, b);      /* [a0,b0,a1,b1] */
        ph = _mm_unpackhi_pi16(a, b);      /* [a2,b2,a3,b3] */
        memcpy(&cu, ctab + (size_t)kp * 8,     8);
        memcpy(&cw, ctab + (size_t)kp * 8 + 4, 8);
        t = _mm_madd_pi16(pl, cu); accU0 = _mm_add_pi32(accU0, t);
        t = _mm_madd_pi16(pl, cw); accW0 = _mm_add_pi32(accW0, t);
        t = _mm_madd_pi16(ph, cu); accU1 = _mm_add_pi32(accU1, t);
        t = _mm_madd_pi16(ph, cw); accW1 = _mm_add_pi32(accW1, t);
        rA += (size_t)vd * 2;
    }
    memcpy(out8 + 0, &accU0, 8);
    memcpy(out8 + 2, &accU1, 8);
    memcpy(out8 + 4, &accW0, 8);
    memcpy(out8 + 6, &accW1, 8);
    _mm_empty();
}

/* ---- Q8 group-scale amax. Watcom twin: lz_amax32_mmx_asm,
   src/ops_kernel_amax.h. Two lanes of uint32 in one MMX register; the
   mask is built in-register (pcmpeqd then psrld 1) rather than loaded,
   so the routine touches no memory but its input. Leaves MMX state
   dirty - the caller emits one emms after the group loop. */
unsigned lz_amax32_mmx(const float *x, int n) {
    __m64 acc = _mm_setzero_si64();
    __m64 msk = _mm_set1_pi32(0x7FFFFFFF);
    int k;
    for (k = 0; k + 1 < n; k += 2) {
        __m64 v = _mm_and_si64(*(const __m64 *)(const void *)(x + k), msk);
        __m64 gt = _mm_cmpgt_pi32(v, acc);
        acc = _mm_or_si64(_mm_and_si64(v, gt), _mm_andnot_si64(gt, acc));
    }
    {
        union { __m64 m; unsigned u[2]; } o;
        o.m = acc;
        return o.u[0] > o.u[1] ? o.u[0] : o.u[1];
    }
}

/* lz_q8round32_sse moved to src/ops_mmx_sse.c, same reason as
   lz_p2_split32_sse above. */

#include "ops_kernel_dot_mmx.h"

#endif /* LZ_MMX_TU && !__WATCOMC__ */
