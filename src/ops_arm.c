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

/* ---- Q8_0: int8 weights, int16 activations, x256 fold ---------------- */

/* C tier. Same shape as q16_0's, plus the fold the x86 kernels get for
   free from punpcklbw(0, byte); one shift at the end rather than 32
   scaled products, which is the same number - the fold is a power of
   two and int32 addition is associative. */
int32_t lz_dot32_q8_arm(const int8_t *restrict w, const int16_t *restrict x) {
    int32_t a0 = 0, a1 = 0;
    int j;
    for (j = 0; j < 32; j += 4) {
        a0 += (int32_t)w[j + 0] * (int32_t)x[j + 0];
        a1 += (int32_t)w[j + 1] * (int32_t)x[j + 1];
        a0 += (int32_t)w[j + 2] * (int32_t)x[j + 2];
        a1 += (int32_t)w[j + 3] * (int32_t)x[j + 3];
    }
    return (a0 + a1) << 8;
}

#if defined(LZ_ARM_ASM_EXTERN)
/* Hand-written tier. THREE THINGS C CANNOT REACH HERE, and the third
   is the one worth reading:

   1. SMLA<x><y>'s halfword selection and 2. LDM, both as for q16_0
      above.
   3. THE x256 FOLD IS FREE, and it is the same trick MMX plays. A
      byte placed in the TOP half of a halfword IS that byte times 256
      and is already sign-extended, because bit 15 of the halfword is
      bit 7 of the byte. `and t, 0xff00ff00, W, lsl #8` therefore lifts
      bytes 0 and 2 of a weight word into two halfword lanes, scaled,
      in ONE instruction - which is what punpcklbw(0, w) does on MMX
      and what SXTB (armv6, and rejected by the assembler here) would
      otherwise be needed for. The plain `and` with no shift lifts
      bytes 1 and 3. Two instructions per four weights, no unpack.

   Register map: r3 the 0xff00ff00 mask, r4-r5 eight weights, r6-r7
   four activations, r10/r11 the scaled weight pairs, r12 the second
   accumulator. */
#define LZ_Q8_ARM_HALF(WR)                                    \
    "ldmia  %[xp]!, {r6, r7}\n\t"                             \
    "and    r10, r3, " WR ", lsl #8\n\t"                      \
    "and    r11, r3, " WR "\n\t"                              \
    "smlabb %[a0], r10, r6, %[a0]\n\t"                        \
    "smlabt r12,   r11, r6, r12\n\t"                          \
    "smlatb %[a0], r10, r7, %[a0]\n\t"                        \
    "smlatt r12,   r11, r7, r12\n\t"

#define LZ_Q8_ARM_BLK                                         \
    "ldmia  %[wp]!, {r4, r5}\n\t"                             \
    LZ_Q8_ARM_HALF("r4") LZ_Q8_ARM_HALF("r5")

int32_t lz_dot32_q8_arm_asm(const int8_t *w, const int16_t *x) {
    const int8_t *wp = w;
    const int16_t *xp = x;
    int32_t a0;
    __asm__(
        "mov    r3, #0xff00\n\t"
        "orr    r3, r3, r3, lsl #16\n\t"
        "mov    %[a0], #0\n\t"
        "mov    r12, #0\n\t"
        LZ_Q8_ARM_BLK LZ_Q8_ARM_BLK LZ_Q8_ARM_BLK LZ_Q8_ARM_BLK
        "add    %[a0], %[a0], r12\n\t"
        : [a0] "=&r"(a0), [wp] "+r"(wp), [xp] "+r"(xp)
        :
        : "r3", "r4", "r5", "r6", "r7", "r10", "r11", "r12", "cc");
    return a0;
}
#endif /* LZ_ARM_ASM_EXTERN */
