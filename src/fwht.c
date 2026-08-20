#include "fwht.h"

/* Sylvester recursion, in place, mirroring lz_fwht (float) in ops.c.
   Every stage is a butterfly of one add and one sub - no multiply,
   sqrt or divide in the transform body, so it is exact in int32 up to
   the additions themselves. The caller folds the 1/sqrt(n) energy
   factor into the downstream activation-quantization scale, keeping
   this a pure integer kernel for the ARMv5TE target (no FPU, no SIMD). */
void lz_fwht_i32(int32_t *restrict y, const int32_t *restrict x, int n) {
    int len, i, j;
    int32_t *tmp = y;          /* caller provides output; butterflies in place */
    for (i = 0; i < n; i++) tmp[i] = x[i];
    for (len = 1; len < n; len <<= 1) {
        for (i = 0; i < n; i += (len << 1)) {
            for (j = 0; j < len; j++) {
                int32_t u = tmp[i + j];
                int32_t v = tmp[i + j + len];
                tmp[i + j]       = u + v;
                tmp[i + j + len] = u - v;
            }
        }
    }
}
