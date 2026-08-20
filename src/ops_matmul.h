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

/* Scheme A dequant for the int-accumulator output (task 2 of
   matmul-int-output, docs/matmul-int-output-design.md). Given
   matmul_q8_int_impl's o_acc/o_xsb/o_ws, quantize each 32-input-group
   of o_acc in the int domain and reconstruct the float output
     o'[i] = sum_g q[i][g] * s[i][g]
     q[i][g] = round(o_acc[i][g] * 127 / amax_g),  amax_g = max_i |o_acc[i][g]|
     s[i][g] = amax_g * o_xsb[g] * o_ws[i][g/r] / (256*127),  r = w->gs/32
   r is the weight-scale stride (one ws per r sub-blocks, the same
   stride epi_q8 walks). Zero amax -> nothing to add for that group.
   Value-changing by design: this validates the per-input-group
   quantization granularity, not a dequant-equivalent. Writes
   o[tk*out_dim+i]; same shapes the int impl fills. */
void matmul_q8_int_dequant_q8(float *o, const int32_t *o_acc,
                              const float *o_xsb, const float *o_ws,
                              const LZTensor *w, int in_dim, int out_dim,
                              int nt);

/* Int-domain downstream consumption (task 3 of matmul-int-output,
   docs/matmul-int-output-design.md). Given matmul_q8_int_impl's
   o_acc/o_xsb/o_ws for the producer w1 (Q8_0, in_dim -> out_dim),
   quantize each 32-input-group in the int domain (scheme A), fold the
   per-(row, group) scale S[i][g] = amax_g * xsb[g] * ws1[i][g/r1] /
   (256*127) to Q16, and carry the int-domain weighted activation
   q_weighted[i] = sum_g q[i][g] * A_int[g] * ws1_int[i][g/r1] (int64).
   The fold is BLOCK-normalized (A by the token's max_g |A[g]|, ws1 by
   the tensor's max |ws1|) so the 2^-16 rounding is a uniform ~1.5e-5
   RELATIVE error - a fixed Q16 against 1 would make it ~1% on the
   small scales the o_proj actually has. The per-group float dequant
   (lz_i32f + fmul + fadd per group) is gone; the downstream matmul runs
   on q_weighted in the int domain:
     acc2[j][k] = sum_{i in w2-group k} q_weighted[i] * w2_q[i][j]
     o2[j] = 2^(A_exp+W_exp-32) * sum_k acc2[j][k] * ws2[j][k]
             (w2 Q8_0, out_dim -> out2)
   where 2^A_exp and 2^W_exp are power-of-two ceilings of the token's
   max|A[g]| and the tensor's max|ws1| - a power-of-two normalizer, not a
   float/float divide, so the fold is byte-identical on x87 and SSE (a
   raw divide double-rounds on PC=24 and flips whole integers at the
   round boundary). Mirroring epi_q8's per-weight-group scale structure
   with a UNIFORM activation scale - the activation quantization is
   eliminated, not re-applied. The producer weight scale ws1 rides inside
   the Q16 fold, exactly as the probe report's note 2 requires (s must
   still be multiplied by ws[i][g/r] at the consumer side).
   w2 == NULL: the producer's own output instead - o[tk*out_dim+i] =
   q_weighted[tk][i] * 2^(A_exp+W_exp-32) - for the boundary where the
   producer feeds a residual add rather than a matmul (the o_proj call
   site today). Same fold; one int64->float conversion per element.
   Returns 1 when o was written, 0 when declined (producer or w2 not a
   Q8_0 the int path serves - fixed-epi tier or shape rejected, or the
   int64 accumulator shape overflows - keep the float path). Scalar
   reference only; the SIMD tier is a later iron-law-2 step. */
int matmul_q8_int_consume(float *o, const int32_t *o_acc,
                          const float *o_xsb, const float *o_ws,
                          const LZTensor *w1, const LZTensor *w2,
                          int in_dim, int out_dim, int out2, int nt);

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
