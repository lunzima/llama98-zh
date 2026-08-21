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
/* int-pipeline milestone 2 (docs/int-pipeline-project.md #2.1): both of
   epi_q41's terms joined in the integer domain, one float materialized
   per output element. `ng`/`r` are epi_q41's own convention (ng =
   weight-group count, r = 32-sub-blocks per group), matching epi_q41
   exactly - NOT epi_fixed's (sub-block-count, r) convention, which this
   function converts to internally. `zneg` selects T2's zero coefficient
   (-scale, read from the sq/sexp plane) over Q4_1/Q6_1's stored min
   (the zq/zexp plane). Takes no float array at all: the activation half
   of the zero term arrives through epi_zero_act_int below. */
float epi_q41_fixed(const int32_t *accb, const LZTensor *w, int row,
                    int tk, int ng, int r, int zneg);
/* Both run once per token, before the row loop, on the fixed path only.
   epi_zero_act_int is the integer replacement for ops_matmul.c's float
   epi_zero_act, not an addition to it - exactly one of the two runs. */
void epi_pack_act(const float *xqs, int nb, int nt);
void epi_zero_act_int(const int8_t *xq, int in_dim, int ng, int r, int nt);
/* Exponent alignment primitive (docs/int-pipeline-project.md batch 1
   #1) - epi_align_i16's epilogue-shaped wrapper. Not called from any
   forward.c site yet; see ops.c's own comment on why. */
int32_t epi_fixed_align_i16(const int32_t *accb, const LZTensor *w, int row,
                             int tk, int ng, int r, int target_e, int bound);

#endif /* OPS_MATMUL_H */
