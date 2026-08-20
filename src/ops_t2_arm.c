#include <stdint.h>
#include "ops_t2_arm.h"

/* ARMv5TE ternary dot, one 32-wide block (one 8-byte plane), UNROLLED
   stride-8 decode: byte j holds weights j, j+8, j+16, j+24 (nibbles
   0,1,2,3). Same 256*sum(code*x) contract as lz_dot32_t2_scalar - code
   read unsigned in 0..3, the -1 and 1/256 both live in epi_q41. Unrolled
   so the per-element byte/nibble address arithmetic folds to compile-time
   constants; gcc then emits one smlabb per element (DSP multiply-accumulate,
   single-cycle on ARM926). This is the scalar-C tier of the ARM table
   (rule 8b); the hand-written asm tier is deferred pending measurement. */
int32_t lz_dot32_t2_arm(const uint8_t *restrict w2, const int16_t *restrict x) {
    int32_t acc = 0;
    int j;
    for (j = 0; j < 8; j++) {
        uint8_t p = w2[j];
        acc += ((p >> 0) & 3) * (int32_t)x[j + 0];
        acc += ((p >> 2) & 3) * (int32_t)x[j + 8];
        acc += ((p >> 4) & 3) * (int32_t)x[j + 16];
        acc += ((p >> 6) & 3) * (int32_t)x[j + 24];
    }
    return acc << 8;
}
