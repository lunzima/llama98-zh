#ifndef LZ_OPS_T2_SCALAR_H
#define LZ_OPS_T2_SCALAR_H
#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#ifdef __cplusplus
extern "C" {
#endif
/* Scalar ternary (LZ_FMT_T2) dot product, one 32-wide block. The bit-exact
   C reference the SIMD tiers must reproduce: dot32_t2_mmx (gcc, MMX
   intrinsics), the Watcom twins lz_dot32_t2_asm / lz_dot32_t2_sse2_asm,
   and the planned ARMv5TE asm (the scalar reference is the
   contract).

   STRIDE-8 layout, same decode as lz_t_f32 and lz_t_row_f32 in
   src/ops.c: weight[k] for
   k in 0..31 lives in byte k&7 at nibble k>>3 - byte j holds weights
   j, j+8, j+16, j+24. Code = weight+1, read UNSIGNED, in 0..3.

   The return value is the SIMD kernels' RAW int32: 256 * sum(code*x).
   Two things are deliberately NOT here, because the shared float
   epilogue (epi_q41, used by matmul_t2_impl) handles both exactly:
     - the -1 that turns a code into a value lives in the epilogue's
       hoisted zero term (wm = -scale): sum w*x = d*[sum code*x - sum x];
     - the x256 fold from punpcklbw(0, byte) is cancelled by the
       epilogue's 1/256.
   Both are powers of two, so no rounding changes at any step and this
   scalar agrees with matmul_scalar_ref's T2 branch bit-for-bit at the
   float output while matching the SIMD leaf kernels bit-for-bit at the
   int32 output. int32 accumulation is exact: |x| <= 32767, code <= 3,
   32 terms, worst case below 2^30. */
int32_t lz_dot32_t2_scalar(const uint8_t *restrict w2,
                           const int16_t *restrict x);
#ifdef __cplusplus
}
#endif
#endif
