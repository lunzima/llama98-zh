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

/* Int-accumulator output path for the Q8_0 matmul: hands out the raw
   per-32-input-group int32 accumulator and the two per-group scale
   factors (activation xsb, weight ws) that epi_q8 combines, so a
   caller can quantize in the int domain without the float dequant
   round-trip. o_acc is [tk * (out_dim*nb) + i*nb + g] (nb = in_dim/32)
   and carries the kernels' 256-scaled sub-block sum - the same value
   epi_q8's post=1/256 cancels, so a caller reproducing the float output
   must apply that post exactly like epi_q8 does. o_xsb is [tk*nb + g];
   o_ws is [i*(in_dim/w->gs) + g128]. Both return 1 when written, 0 when
   the tensor is on the fixed-epi tier (use the float path instead). The
   scalar-ref is the contract; the impl routes to it under --kernel ref. */
int matmul_q8_int_impl(int32_t *o_acc, float *o_xsb, float *o_ws,
                       const int8_t *xq, const float *xqs,
                       const LZTensor *w, int in_dim, int out_dim, int nt);
int matmul_q8_int_scalar_ref(int32_t *o_acc, float *o_xsb, float *o_ws,
                             const int8_t *xq, const float *xqs,
                             const LZTensor *w, int in_dim, int out_dim,
                             int nt);

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
void epi_pack_act(const float *xqs, int nb, int nt);

#endif /* OPS_MATMUL_H */
