/* Row-kernel prototypes for the gcc-only AVX2 tier. Guarded the same
   way src/ops_sse2.h guards LZ_ROW_SSE2_EXTERN: LZ_AVX2_TU is a
   build-level macro (-D, set only when src/ops_avx2.c is part of this
   link), never derived from a predefined compiler macro. AVX2 is
   gcc-only by design (see the AVX2 tier design doc) - this file is
   never included under __WATCOMC__, and there is deliberately no
   *_avx2_asm counterpart to any of these. */
#ifndef LZ_OPS_AVX2_H
#define LZ_OPS_AVX2_H

#if defined(LZ_AVX2_TU) && !defined(__WATCOMC__)
#define LZ_ROW_AVX2_EXTERN 1

#include "ops_matmul.h"
#include "ops_p2_blk.h"   /* lz_p2_blk - the pass-2 scratch block */

void row_q8_avx2_intrin(const lz_row_ctx *c);
void row_q41_avx2_intrin(const lz_row_ctx *c);
void row_q61_avx2_intrin(const lz_row_ctx *c);
void row_q16_avx2_intrin(const lz_row_ctx *c);

#define LZ_HAVE_AMAX_AVX2 1
unsigned lz_amax32_avx2(const float *x, int n);

#define LZ_HAVE_NORM_AVX2 1
void lz_rmsnorm_out_avx2(float *o, const float *x, const float *w,
                         int n4, const float *k2);
void lz_vmax_avx2(const float *x, int n4, float *pmax);
void lz_vscale_avx2(float *x, int n4, const float *pk);

#define LZ_HAVE_EPI_AVX2 1
#define LZ_EPI_AVX2_EXTERN 1
lz_i64 lz_epi_mac_i16_avx2(const int32_t *a, const int16_t *m, int n);

#define LZ_FWHT_AVX2_EXTERN 1
void lz_fwht_stage_avx2(int32_t *y, int n, int len);

#define LZ_ROPE_AVX2_EXTERN 1
void lz_rope_avx2(float *v, int n_heads, int head_dim, int rotary_dim,
                  int pos, const float *cs);

#define LZ_HAVE_P2_MUL32_AVX2 1
void lz_p2_mul32_avx2(const int8_t *hi, const int8_t *lo,
                      const int16_t *dq, lz_p2_blk *blk);

#define LZ_HAVE_P2_SPLIT32_AVX2 1
void lz_p2_split32_avx2(const lz_p2_blk *blk, int8_t *oh, int8_t *ol);

#define LZ_HAVE_Q8R_AVX2 1
void lz_q8round32_avx2(const float *x, int8_t *o, const float *pinv);
/* lz_q8round32_avx2_compiled_in_quant/_gdn are declared in
   src/ops_quant.h and src/ops_gdn.h, not here - this operator has TWO
   dispatch translation units and each proves its own #include of this
   header, so neither answer can live in the file that defines the macro
   unconditionally. */

/* lz_rope_avx2_compiled_in is declared in src/ops.h, not here: its
   definition is unconditional (in src/ops_rope.c) and must have a
   prototype in scope on every build, the way
   lz_epi_avx2_compiled_in/lz_fwht_avx2_compiled_in do. */

#endif /* LZ_AVX2_TU && !__WATCOMC__ */

#endif /* LZ_OPS_AVX2_H */
