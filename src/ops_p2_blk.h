/* The GDN/KDA fixed-point pass-2 scratch block. Shared by every SIMD
   tier of that operator (MMX/SSE1 in src/ops_mmx.c, SSE2 in
   src/ops_sse2.c, and Watcom's #pragma aux twins in ops.c's included
   headers), so it lives in its own header rather than in either ISA
   header - src/ops_mmx.h and src/ops_sse2.h are independent of each
   other and both need the type without including one another. */
#ifndef LZ_OPS_P2_BLK_H
#define LZ_OPS_P2_BLK_H

#include "lz_int.h"   /* <stdint.h> is not on the language floor */

typedef struct {
    int16_t mul[8];      /*   0  in   m1,m2 repeated 4x (pmaddwd operand) */
    int32_t amax[4];     /*  16  out  |A| accumulator lanes, caller folds */
    int32_t rnd[4];      /*  32  in   1 << (sh-1), or 0 when sh = 0       */
    int32_t cnt[4];      /*  48  in   sh; the shift count is the low 64   */
    int16_t k128[8];     /*  64  in   128, the split's rounding addend    */
    int16_t kclp[8];     /*  80  in   32641, the lo clamp for MMX/SSE2    */
    int32_t a[32];       /*  96  mid  A, written by one kernel, read by   */
    int16_t kmin[8];     /* 224  in   -127, the same clamp for SSE1       */
} lz_p2_blk;             /*           240 bytes total                     */

#endif /* LZ_OPS_P2_BLK_H */
