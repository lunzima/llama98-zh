/* ops_matmul.h - declarations for ops_matmul.c's functions and tables.

   The public matmul API (lz_matmul, lz_matmul_w, lz_matmul_xq,
   lz_matmul_xq_nt) stays in ops.h/ops.c. This header covers the
   internal impl functions and dispatch tables that cross the TU
   boundary, plus the epilogue functions that stay in ops.c but are
   called from here. */
#ifndef OPS_MATMUL_H
#define OPS_MATMUL_H

#include "ops.h"

/* ---- dispatch tables (defined in ops_matmul.c) ----------------------- */
typedef void (*lz_rowfn)(int32_t *acc32, const void *w4, const void *w2,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end,
                         const void *pf_end2);

extern const lz_rowfn LZ_ROW_Q8[LZ_ROW_N];
extern const lz_rowfn LZ_ROW_Q41[LZ_ROW_N];
extern const lz_rowfn LZ_ROW_T2[LZ_ROW_N];
extern const lz_rowfn LZ_ROW_Q61[LZ_ROW_N];
extern const lz_rowfn LZ_ROW_Q16[LZ_ROW_N];

/* ---- matmul impls (defined in ops_matmul.c) -------------------------- */
void matmul_q8_impl(float *o, const int8_t *xq, const float *xqs,
                    const LZTensor *w, int in_dim, int out_dim, int nt);
void matmul_q41_impl(float *o, const int8_t *xq, const float *xqs,
                     const LZTensor *w, int in_dim, int out_dim, int nt);
void matmul_q61_impl(float *o, const int8_t *xq, const float *xqs,
                     const LZTensor *w, int in_dim, int out_dim, int nt);
void matmul_t2_impl(float *o, const int8_t *xq, const float *xqs,
                    const LZTensor *w, int in_dim, int out_dim, int nt);
void matmul_q16_impl(float *o, const int8_t *xq, const float *xqs,
                     const LZTensor *w, int in_dim, int out_dim, int nt);

/* Scalar reference (the contract every SIMD tier is judged against). */
void matmul_scalar_ref(float *o, const int8_t *xq, const float *xqs,
                       const LZTensor *w, int in_dim, int out_dim, int nt);

/* ---- epilogue functions (defined in ops.c, called from here) --------- */
float epi_q8(const int32_t *accb, const float *xsb, const float *ws,
             int ng, int r, float post);
float epi_q41(const int32_t *accb, const float *xsb, const float *xgb,
              const float *wd, const float *wm, int ng, int r);
float epi_fixed(const int32_t *accb, const LZTensor *w, int row,
                int tk, int ng, int r, float post);
/* int-pipeline milestone 1 (docs/int-pipeline-project.md #2.1): epi_q41's
   dot/scale term via the shared epi_fixed exit, zsum unchanged in float.
   `ng`/`r` are epi_q41's own convention (ng = weight-group count, r =
   32-sub-blocks per group), matching epi_q41 exactly - NOT epi_fixed's
   (sub-block-count, r) convention, which this function converts to
   internally. */
float epi_q41_fixed(const int32_t *accb, const LZTensor *w, int row,
                    int tk, const float *xgb, const float *wm,
                    int ng, int r);
void epi_pack_act(const float *xqs, int nb, int nt);
/* Exponent alignment primitive (docs/int-pipeline-project.md batch 1
   #1) - epi_align_i16's epilogue-shaped wrapper. Not called from any
   forward.c site yet; see ops.c's own comment on why. */
int32_t epi_fixed_align_i16(const int32_t *accb, const LZTensor *w, int row,
                             int tk, int ng, int r, int target_e, int bound);

#endif /* OPS_MATMUL_H */
