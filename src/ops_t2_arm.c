#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include "ops_t2_arm.h"

/* ARMv5TE ternary dot, one 32-wide block (one 8-byte plane), UNROLLED
   stride-8 decode: byte j holds weights j, j+8, j+16, j+24 (nibbles
   0,1,2,3). Same 256*sum(code*x) contract as lz_dot32_t2_scalar - code
   read unsigned in 0..3, the -1 and 1/256 both live in epi_q41. Unrolled
   so the per-element byte/nibble address arithmetic folds to compile-time
   constants; gcc then emits one smlabb per element (DSP multiply-accumulate,
   single-cycle on ARM926). This is the C tier of the ARM table;
   the hand-written asm tier is lz_dot32_t2_arm_asm below. */
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

#if defined(LZ_T2_ARM_ASM_EXTERN)
/* Hand-written ARMv5TE tier. Bit-identical to
   the two C tiers over the whole 0..3 code domain, so it owes them
   identity and not a tolerance.

   THREE THINGS C CANNOT REACH HERE, which is why this cell is not empty:

   1. SMLA<x><y> selects WHICH HALFWORD of each operand it multiplies.
      C can only say "int16 * int16", so gcc emits smlabb and nothing
      else, and it must load every element separately to put it in a
      bottom halfword. Using all four of BB/BT/TB/TT lets one loaded
      word feed two MACs and one extracted register feed two codes.
   2. The 0x00030003 mask. Codes for elements j and j+2 of a plane sit
      exactly 16 bits apart in the packed byte word (bit 2q+8j), which is
      exactly the distance SMLA?B and SMLA?T address. One AND therefore
      lifts TWO codes - 0.5 instructions per element where the C tier
      spends one shift plus one mask each.
   3. LDM pulls 8 int16 in one instruction. gcc emits one ldrh per
      element because int16_t promises only 2-byte alignment; the
      alignment this kernel needs is a contract stated in the header and
      met by matmul_t2_impl's g_xw.

   Not used, deliberately: PLD. Prefetch is its own orthogonal tier
   (--prefetch) and does not belong inside a leaf kernel.

   Two accumulators, merged at the end. Exact in int32 - |x| <= 32767,
   code <= 3, 32 terms, worst case below 2^22 - so the split changes no
   bit, unlike a float reduction.

   Register map: r8/r9 the two packed code words, r10 the 0x00030003
   mask, r4-r7 one plane's 8 activations, r3/r11 extracted code pairs,
   r12 the second accumulator. */
/* The activation LDM is separate from the plane body so plane 0's copy
   can be hoisted above the two accumulator inits - a four-register LDM
   is latency 7 on arm926ej-s and the two `and`s after it do not cover
   that, which armv5telint -a costed at 2 stall cycles. Planes 2/4/6
   cannot move: their LDM writes r7 and the preceding plane's last
   `smlatt` still reads it, a real WAR. */
#define LZ_T2_ARM_XLOAD "ldmia  %[xp]!, {r4, r5, r6, r7}\n\t"
#define LZ_T2_ARM_PLANE(SH)                                   \
    "and    r3,  r10, r8, lsr #" #SH "\n\t"                   \
    "and    r11, r10, r8, lsr #" #SH "+8\n\t"                 \
    "smlabb %[a0], r3,  r4, %[a0]\n\t"                        \
    "smlabt r12,   r11, r4, r12\n\t"                          \
    "smlatb %[a0], r3,  r5, %[a0]\n\t"                        \
    "smlatt r12,   r11, r5, r12\n\t"                          \
    "and    r3,  r10, r9, lsr #" #SH "\n\t"                   \
    "and    r11, r10, r9, lsr #" #SH "+8\n\t"                 \
    "smlabb %[a0], r3,  r6, %[a0]\n\t"                        \
    "smlabt r12,   r11, r6, r12\n\t"                          \
    "smlatb %[a0], r3,  r7, %[a0]\n\t"                        \
    "smlatt r12,   r11, r7, r12\n\t"

int32_t lz_dot32_t2_arm_asm(const uint8_t *w2, const int16_t *x) {
    const int16_t *xp = x;
    int32_t a0;
    __asm__(
        "ldr    r8,  [%[wp]]\n\t"
        "ldr    r9,  [%[wp], #4]\n\t"
        "mov    r10, #3\n\t"
        "orr    r10, r10, r10, lsl #16\n\t"
        LZ_T2_ARM_XLOAD                 /* plane 0's activations, early */
        "mov    %[a0], #0\n\t"
        "mov    r12, #0\n\t"
        /* one plane per macro: plane q holds x[8q..8q+7] and its codes
           sit at bit 2q+8j of the packed word */
        LZ_T2_ARM_PLANE(0)
        LZ_T2_ARM_XLOAD LZ_T2_ARM_PLANE(2)
        LZ_T2_ARM_XLOAD LZ_T2_ARM_PLANE(4)
        LZ_T2_ARM_XLOAD LZ_T2_ARM_PLANE(6)
        "add    %[a0], %[a0], r12\n\t"
        "mov    %[a0], %[a0], lsl #8\n\t"   /* the x256 fold, as in C */
        : [a0] "=&r"(a0), [xp] "+r"(xp)
        : [wp] "r"(w2)
        : "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12",
          "cc");
    return a0;
}
#endif /* LZ_T2_ARM_ASM_EXTERN */
