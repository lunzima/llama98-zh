#ifndef LZ_OPS_T2_ARM_H
#define LZ_OPS_T2_ARM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Unrolled scalar ternary (LZ_FMT_T2) dot product for ARMv5TE, one
   32-wide block. Same STRIDE-8 decode and same 256*sum(code*x) contract
   as lz_dot32_t2_scalar (rule 2: the scalar reference is the contract):
   code read UNSIGNED in 0..3, and the -1 (code->value) plus the 1/256
   (x256 fold) are both handled exactly by the shared float epilogue
   epi_q41, as documented in ops_t2_scalar.h.

   This is the scalar-C tier of the ARM table (rule 8b); the hand-written
   asm tier is deferred pending measurement. See src/ops_t2_arm.c for the
   unrolled byte/nibble decode that folds the per-element address
   arithmetic to compile-time constants and lets gcc emit one smlabb per
   element. */
int32_t lz_dot32_t2_arm(const uint8_t *restrict w2,
                        const int16_t *restrict x);
#ifdef __cplusplus
}
#endif
#endif
