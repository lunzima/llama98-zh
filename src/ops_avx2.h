/* Row-kernel prototypes for the gcc-only AVX2 tier. Guarded the same
   way src/ops_sse2.h guards LZ_ROW_SSE2_EXTERN: LZ_AVX2_TU is a
   build-level macro (-D, set only when src/ops_avx2.c is part of this
   link), never derived from a predefined compiler macro. AVX2 is
   gcc-only by design (see the 2026-09-08 design doc) - this file is
   never included under __WATCOMC__, and there is deliberately no
   *_avx2_asm counterpart to any of these. */
#ifndef LZ_OPS_AVX2_H
#define LZ_OPS_AVX2_H

#if defined(LZ_AVX2_TU) && !defined(__WATCOMC__)
#define LZ_ROW_AVX2_EXTERN 1

#include "ops_matmul.h"

void row_q8_avx2_intrin(const lz_row_ctx *c);
void row_q41_avx2_intrin(const lz_row_ctx *c);
void row_q61_avx2_intrin(const lz_row_ctx *c);
void row_q16_avx2_intrin(const lz_row_ctx *c);

#endif /* LZ_AVX2_TU && !__WATCOMC__ */

#endif /* LZ_OPS_AVX2_H */
