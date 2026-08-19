/* The two functions in src/ops_mmx.c's family that need genuine SSE1,
   not just MMX - split out because gcc's -march=i586 stops meaning
   "no CMOV" the moment -msse is also on the command line (measured:
   gcc treats -msse as license to assume at least Pentium III / i686,
   and will use cmov even in an unrelated plain-C comparison elsewhere
   in the same translation unit - lz_amax32_mmx in ops_mmx.c had one
   until this split, with nothing to do with SSE). Real SSE1 hardware
   is i686+ regardless (SSE starts at Pentium III; no Socket-7 part has
   it), so -march=i686 here is honest, not a compromise - it is
   ops_mmx.c that must stay at -march=i586 so its purely-MMX functions
   (reachable on Pentium MMX/K6/Cyrix 6x86/WinChip, none of which have
   CMOV) do not pick up a stray one.

   Same LZ_MMX_TU/__WATCOMC__ contract as ops_mmx.c: Watcom is
   unaffected and never compiles this file. */
#include <stdint.h>
#include <string.h>

#include "ops_mmx.h"

#if defined(LZ_MMX_TU) && !defined(__WATCOMC__)

#include "mmx_compat.h"
#include <mmintrin.h>
#include <xmmintrin.h>

/* GDN/KDA fixed-point pass 2, SSE1 cell. Watcom twin:
   lz_p2_split32_sse_asm, src/ops_kernel_p2.h. Named "sse" for the ISA a
   machine must HAVE to run it - it still operates on %mm (SSE1's whole
   addition to the integer side of the MMX register file is used here
   exactly once: pmaxsw does the low plane's clamp to -127 directly,
   replacing the psubsw/paddsw pair). */
void lz_p2_split32_sse(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m64 rnd, cnt, k128, kmin;
    int o;

    memcpy(&rnd,  blk->rnd,  8);
    memcpy(&cnt,  blk->cnt,  8);
    memcpy(&k128, blk->k128, 8);
    memcpy(&kmin, blk->kmin, 8);
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
        l0 = _mm_max_pi16(l0, kmin);      /* the whole SSE1 tier, twice */
        l1 = _mm_max_pi16(l1, kmin);
        t = _mm_packs_pi16(h0, h1);
        memcpy(oh + o * 8, &t, 8);
        t = _mm_packs_pi16(l0, l1);
        memcpy(ol + o * 8, &t, 8);
    }
}

/* Q8 activation rounding, SSE1 tier. Watcom twin: lz_q8round32_sse_asm,
   src/ops_kernel_q8round.h. No cvtps2dq, only cvtps2pi - converting 2
   floats at a time into MMX registers. Multiply in xmm, convert and
   pack in mm, using movhlps to slide the high 2 floats down. Writes
   MMX registers, so the caller must emit one emms after the group loop
   (rule 6.6). */
void lz_q8round32_sse(const float *x, int8_t *o, const float *pinv) {
    __m128 vi = _mm_set1_ps(*pinv);
    __m64 p127 = _mm_set1_pi16(127), m127 = _mm_set1_pi16(-127);
    int b;
    for (b = 0; b < 4; b++) {
        const float *sp = x + b * 8;
        __m128 f0 = _mm_mul_ps(_mm_loadu_ps(sp), vi);
        __m128 f1 = _mm_mul_ps(_mm_loadu_ps(sp + 4), vi);
        __m64 a0 = _mm_cvtps_pi32(f0);
        __m64 a1 = _mm_cvtps_pi32(_mm_movehl_ps(f0, f0));
        __m64 a2 = _mm_cvtps_pi32(f1);
        __m64 a3 = _mm_cvtps_pi32(_mm_movehl_ps(f1, f1));
        __m64 l = _mm_packs_pi32(a0, a1);
        __m64 h = _mm_packs_pi32(a2, a3);
        __m64 r;
        l = _mm_max_pi16(_mm_min_pi16(l, p127), m127);
        h = _mm_max_pi16(_mm_min_pi16(h, p127), m127);
        r = _mm_packs_pi16(l, h);
        memcpy(o + b * 8, &r, 8);
    }
}

#endif /* LZ_MMX_TU && !__WATCOMC__ */
