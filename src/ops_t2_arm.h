#ifndef LZ_OPS_T2_ARM_H
#define LZ_OPS_T2_ARM_H
#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#ifdef __cplusplus
extern "C" {
#endif
/* Unrolled scalar ternary (LZ_FMT_T2) dot product for ARMv5TE, one
   32-wide block. Same STRIDE-8 decode and same 256*sum(code*x) contract
   as lz_dot32_t2_scalar (the scalar reference is the contract):
   code read UNSIGNED in 0..3, and the -1 (code->value) plus the 1/256
   (x256 fold) are both handled exactly by the shared float epilogue
   epi_q41, as documented in ops_t2_scalar.h.

   This is the C tier of the ARM table; the hand-written asm
   tier is lz_dot32_t2_arm_asm below. Both ship and both are selectable:
   `--kernel arm-c` and `--kernel arm-asm`. See src/ops_t2_arm.c for the
   unrolled byte/nibble decode that folds the per-element address
   arithmetic to compile-time constants and lets gcc emit one smlabb per
   element. */
int32_t lz_dot32_t2_arm(const uint8_t *restrict w2,
                        const int16_t *restrict x);

#if defined(__arm__) && defined(__GNUC__)
#define LZ_T2_ARM_ASM_EXTERN 1

/* Hand-written ARMv5TE tier, same contract and same bit-exact result as
   lz_dot32_t2_arm and lz_dot32_t2_scalar over the whole code domain 0..3.

   EXTRA PRECONDITION over the two C tiers: x must be 4-byte aligned,
   because this one reads it with LDM. matmul_t2_impl's g_xw carries an
   aligned(4) attribute for that, and every block pointer it hands over
   is g_xw plus a multiple of 64 bytes (in_dim is a multiple of 32 by the
   T2 dispatch guard, blocks are 32 int16 apart).

   Static instruction counts, which is what transfers to the AK7802
   (QEMU models no cache or bus, so no wall clock from it means
   anything): 13 instructions per 8 elements = 1.625/element against the
   C tier's 17 per 4 = 4.25; 64 instructions per whole call against 143. */
int32_t lz_dot32_t2_arm_asm(const uint8_t *w2, const int16_t *x);
#endif /* __arm__ && __GNUC__ */

#ifdef __cplusplus
}
#endif
#endif
