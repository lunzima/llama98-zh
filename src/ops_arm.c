#include <stdint.h>
#include "ops_arm.h"

/* ARMv5TE row leaf kernels, non-ternary formats. The contract each one
   owes matmul_scalar_ref_one, and the alignment the asm tiers require,
   are in src/ops_arm.h. */

/* ---- Q16_0: int16 weights, int16 activations, no fold ---------------- */

/* C tier. Unrolled by four with two accumulators: gcc emits one ldrh
   per operand and one smlabb per element whatever the shape, so the
   unroll buys the loop overhead back and the second accumulator breaks
   the multiply-accumulate dependency chain. Exact in int32
   (32767*127*32 = 1.33e8), so the split changes no bit. */
int32_t lz_dot32_q16_arm(const int16_t *restrict w, const int16_t *restrict x) {
    int32_t a0 = 0, a1 = 0;
    int j;
    for (j = 0; j < 32; j += 4) {
        a0 += (int32_t)w[j + 0] * (int32_t)x[j + 0];
        a1 += (int32_t)w[j + 1] * (int32_t)x[j + 1];
        a0 += (int32_t)w[j + 2] * (int32_t)x[j + 2];
        a1 += (int32_t)w[j + 3] * (int32_t)x[j + 3];
    }
    return a0 + a1;
}

#if defined(LZ_ARM_ASM_EXTERN)
/* Hand-written tier. TWO THINGS C CANNOT REACH HERE:

   1. SMLA<x><y> selects WHICH HALFWORD of each operand it multiplies.
      C can only say "int16 * int16", so gcc emits smlabb and nothing
      else, and must load every element into a bottom halfword of its
      own. Using BB and TT lets ONE loaded word feed two accumulates -
      half the loads and no extract at all.
   2. LDM pulls 4 words (8 int16) in one instruction. gcc emits one
      ldrh per element because int16_t promises only 2-byte alignment;
      the 4-byte alignment this kernel needs is a contract stated in the
      header and met by the g_xw / weight-row offsets.

   Not used, deliberately: PLD. Prefetch is its own orthogonal tier
   (--prefetch, rule 9) and does not belong inside a leaf kernel.

   Register map: r4-r7 eight weights, r8-r11 eight activations, r12 the
   second accumulator. */
#define LZ_Q16_ARM_BLK                                        \
    "ldmia  %[wp]!, {r4, r5, r6, r7}\n\t"                     \
    "ldmia  %[xp]!, {r8, r9, r10, r11}\n\t"                   \
    "smlabb %[a0], r4, r8,  %[a0]\n\t"                        \
    "smlatt r12,   r4, r8,  r12\n\t"                          \
    "smlabb %[a0], r5, r9,  %[a0]\n\t"                        \
    "smlatt r12,   r5, r9,  r12\n\t"                          \
    "smlabb %[a0], r6, r10, %[a0]\n\t"                        \
    "smlatt r12,   r6, r10, r12\n\t"                          \
    "smlabb %[a0], r7, r11, %[a0]\n\t"                        \
    "smlatt r12,   r7, r11, r12\n\t"

int32_t lz_dot32_q16_arm_asm(const int16_t *w, const int16_t *x) {
    const int16_t *wp = w;
    const int16_t *xp = x;
    int32_t a0;
    __asm__(
        "mov    %[a0], #0\n\t"
        "mov    r12, #0\n\t"
        LZ_Q16_ARM_BLK LZ_Q16_ARM_BLK LZ_Q16_ARM_BLK LZ_Q16_ARM_BLK
        "add    %[a0], %[a0], r12\n\t"
        : [a0] "=&r"(a0), [wp] "+r"(wp), [xp] "+r"(xp)
        :
        : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "cc");
    return a0;
}
#endif /* LZ_ARM_ASM_EXTERN */
