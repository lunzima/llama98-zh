/* ops_matmul.h - declarations for ops_matmul.c's functions and tables.

   The public matmul API (lz_matmul, lz_matmul_w, lz_matmul_xq,
   lz_matmul_xq_nt) stays in ops.h/ops.c. This header covers the
   internal impl functions and dispatch tables that cross the TU
   boundary, plus the epilogue functions that stay in ops.c but are
   called from here. */
#ifndef OPS_MATMUL_H
#define OPS_MATMUL_H

#include "ops.h"
#include "ops_kernel_shared.h"  /* LZ_ROW_N: the table sizes below */

/* ---- dispatch tables (defined in ops_matmul.c) ----------------------- */
typedef struct {
    int32_t *acc32;
    const void *w4;
    const void *w2;
    const int16_t *xw;
    int nb;
    int nt;
    int in_dim;
    const void *pf_end;
    const void *pf_end2;
} lz_row_ctx;
typedef void (*lz_rowfn)(const lz_row_ctx *c);

extern const lz_rowfn LZ_ROW_Q8[LZ_ROW_N];
extern const lz_rowfn LZ_ROW_Q41[LZ_ROW_N];
extern const lz_rowfn LZ_ROW_T2[LZ_ROW_N];
extern const lz_rowfn LZ_ROW_Q61[LZ_ROW_N];
extern const lz_rowfn LZ_ROW_Q16[LZ_ROW_N];

/* ---- matmul exit descriptor -----------------------------------------
   Where matmul_q41_impl / matmul_t2_impl / matmul_q61_impl /
   matmul_q16_impl put their epilogue results. Exactly one of `o`
   (float, what every lz_matmul_xq_nt caller wants) and `oi` (int16 at
   `target_e`, clamped to +-`bound` - lz_matmul_xq_nt_i16) is set.

   NOT per format, even though the two epilogue families behind it are:
   the Q4_1/Q6_1/T2 impls reach the int16 exit through epi_q41_align_i16
   and Q16_0 reaches it through epi_fixed_align_i16, but what a CALLER
   asks for is the same question either way. A second, identical struct
   per family is the shape this project has twice watched drift.

   `probe`/`ok` exist so that "can this tensor take the int16 exit?" has
   ONE answer computed in ONE place. That place is the impl's own
   fixed-epilogue and row-kernel guard; lz_matmul_xq_i16_ok reaches it
   by running the impl with `probe` set, which returns the moment the
   guard has been evaluated and before any activation is read. A
   separately written predicate would be a second copy of that guard,
   and it would drift the first time either side is touched. */
typedef struct {
    float *o;
    short *oi;
    int    target_e;
    int    bound;
    int    probe;               /* in:  decide only, compute nothing */
    int    ok;                  /* out: 1 if the requested exit is available */
} LZMatOut;

/* ---- matmul impls (defined in ops_matmul.c) -------------------------- */
void matmul_q8_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                    const LZTensor *w, int in_dim, int out_dim, int nt);
void matmul_q41_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                     const LZTensor *w, int in_dim, int out_dim, int nt);
void matmul_q61_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                     const LZTensor *w, int in_dim, int out_dim, int nt);
void matmul_t2_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                    const LZTensor *w, int in_dim, int out_dim, int nt);
void matmul_q16_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                     const LZTensor *w, int in_dim, int out_dim, int nt);

/* Scalar reference (the contract every SIMD tier is judged against). */
void matmul_scalar_ref(float *o, const int8_t *xq, const float *xqs,
                       const LZTensor *w, int in_dim, int out_dim, int nt);

/* ---- epilogue functions (defined in ops.c, called from here) --------- */
/* The row kernels' product scaling for the format `w` carries, as a
   power of two: -8 for Q8_0 (its kernels return 256x the true product),
   0 for Q16_0 (int16 weights, nothing folded in). Both epilogues take
   their factor from here - the float one as pow2f() of it, the fixed
   one folded into its exponent on both exits - so the two cannot
   disagree and a scaling the kernels do not produce cannot be written
   down. LZ_EPI_POST_NA for any other format, which sends the whole
   matmul to matmul_scalar_ref - the contract, and the only path here
   with no `post` concept at all. */
#define LZ_EPI_POST_NA 64
int lz_epi_post_e(const LZTensor *w);
float epi_q8(const int32_t *accb, const float *xsb, const float *ws,
             int ng, int r, float post);
float epi_q41(const int32_t *accb, const float *xsb, const float *xgb,
              const float *wd, const float *wm, int ng, int r);
/* `pe` is lz_epi_post_e(w), hoisted to the caller's row loop - see the
   note at the definition for why it is not read inside. */
float epi_fixed(const int32_t *accb, const LZTensor *w, int row,
                int tk, int ng, int r, int pe);
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
/* int-pipeline milestone 3: the same integer join (epi_q41_join, shared
   with epi_q41_fixed so the reduction order cannot fork), exited as an
   int16 already folded to the consumer's `target_e` and clamped to
   +-`bound`. No float is materialized, so nothing is billed. */
int32_t epi_q41_align_i16(const int32_t *accb, const LZTensor *w, int row,
                          int tk, int ng, int r, int zneg,
                          int target_e, int bound);
/* Both run once per token, before the row loop, on the fixed path only.
   epi_zero_act_int is the integer replacement for ops_matmul.c's float
   epi_zero_act, not an addition to it - exactly one of the two runs. */
void epi_pack_act(const float *xqs, int nb, int nt);
void epi_zero_act_int(const int8_t *xq, int in_dim, int ng, int r, int nt);
/* Exponent alignment primitive (docs/int-pipeline-project.md batch 1
   #1) - epi_align_i16's wrapper for the symmetric formats, i.e. the
   epi_fixed / lz_epi_prep family. matmul_q16_impl's int16 exit is its
   live caller (int-pipeline milestone 5); Q8_0 still has none. */
int32_t epi_fixed_align_i16(const int32_t *accb, const LZTensor *w, int row,
                             int tk, int ng, int r, int target_e, int bound);

#endif /* OPS_MATMUL_H */
