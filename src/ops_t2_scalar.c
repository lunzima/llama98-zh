#include <stdint.h>
#include "ops_t2_scalar.h"

/* Scalar ternary dot product, one 32-wide block (one 8-byte plane).
   Weights {-1,0,1} packed 2-bit, code=value+1, STRIDE-8 layout:
   weight[k] (k in 0..31) lives in byte k&7, nibble k>>3 - the same
   layout src/ops.c:1104/1177 and dot32_t2_mmx use, so this is the exact
   scalar reference both SIMD paths must reproduce bit-for-bit (rule 2:
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
