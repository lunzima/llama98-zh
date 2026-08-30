#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include "ops_t2_scalar.h"

/* Scalar ternary dot product, one 32-wide block (one 8-byte plane).
   Weights {-1,0,1} packed 2-bit, code=value+1, STRIDE-8 layout:
   weight[k] (k in 0..31) lives in byte k&7, nibble k>>3 - the same
   layout lz_t_f32 and lz_t_row_f32 use in src/ops.c
   (`(b2[k & 7] >> (2 * (k >> 3))) & 3`, the same expression Q6_1's low
   plane reads through lz_q61_get), and the same one
   dot32_t2_mmx unpacks, so this is the exact
   scalar reference both SIMD paths must reproduce bit-for-bit (
   the scalar reference is the contract).

   Returns 256 * sum(code*x), code read unsigned in 0..3 - the SIMD
   kernels' raw int32. The -1 (code -> value) and the 1/256 (x256 fold)
   are both cancelled by the shared epilogue epi_q41: the fold by its
   1/256, the minus by its wm = -scale zero term (sum w*x = d*[sum code*x
   - sum x]). Both are powers of two, so no rounding changes at any step
   and this equals matmul_scalar_ref's T2 branch at the float output.
   No group scale here - that is the epilogue's job. */
int32_t lz_dot32_t2_scalar(const uint8_t *restrict w2,
                           const int16_t *restrict x) {
    int32_t acc = 0;
    int i;
    for (i = 0; i < 32; i++) {
        int c = (w2[i & 7] >> (2 * (i >> 3))) & 3;   /* code = weight+1, 0..3 */
        acc += (int32_t)c * (int32_t)x[i];
    }
    return acc << 8;   /* the SIMD kernels' x256 fold */
}
