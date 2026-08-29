#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include "ops.h"       /* LZ_SOFTFP_FAST's default - this TU's RTZ tail
                          reads it and neither lz_int.h nor ops_arm.h
                          provides it (the command-line -D used to). */
#include "ops_arm.h"

/* THE ARMv5TE KERNELS, non-ternary formats - NOT only the row leaves:
   eight of the seventeen functions below are not row kernels at all.
   The dot32 leaves for Q16_0/Q8_0/Q4_1/Q6_1
   come first; after them come the attention pair sum, the Hadamard
   stages, the fixed-point norm, the F32 matmul row, the int16 epilogue
   MAC and the GDN recurrence - lz_wsum_pair_arm_asm says "Not a row
   kernel" in its own header. src/ops_arm.h documents each, and carries
   the contract each owes matmul_scalar_ref_one plus the alignment the
   asm tiers require. Ternary has its own file, src/ops_t2_arm.c: its
   macros share nothing with these (a 0x00030003 mask over a stride-8
   plane against Q6_1's 0x0f000f00 + 0x30003000 merge), and it has its
   own feature gate, LZ_T2_ARM_ASM_EXTERN. */

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
   (--prefetch) and does not belong inside a leaf kernel.

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
#define LZ_Q8_ARM_WLOAD "ldmia  %[wp]!, {r4, r5}\n\t"
#define LZ_Q8_ARM_TAIL  "smlatt r12,   r11, r7, r12\n\t"
#define LZ_Q8_ARM_HALF_HEAD(WR)                               \
    "ldmia  %[xp]!, {r6, r7}\n\t"                             \
    "and    r10, r3, " WR ", lsl #8\n\t"                      \
    "and    r11, r3, " WR "\n\t"                              \
    "smlabb %[a0], r10, r6, %[a0]\n\t"                        \
    "smlabt r12,   r11, r6, r12\n\t"                          \
    "smlatb %[a0], r10, r7, %[a0]\n\t"
#define LZ_Q8_ARM_HALF(WR) LZ_Q8_ARM_HALF_HEAD(WR) LZ_Q8_ARM_TAIL

/* SOFTWARE PIPELINED BY ONE INSTRUCTION, which is why the group macro
   comes in two shapes instead of one. armv5telint -a scored the
   straight version at 5 stall cycles on arm926ej-s and named all four:
   the `ldmia %[wp]!` that starts a group issues immediately after the
   previous group's last `smlatt`, and an LDM of two registers is
   latency 4 there against the MAC's 3, so the first `and` of the next
   group waits.

   The load is therefore hoisted ABOVE that final MAC. It is legal
   because the MAC reads r11, r7 and r12 while the load writes r4, r5
   and %[wp]: the weights for group k are fully consumed by the two
   `and`s at the top of each half, long before the half's last
   accumulate. Dependence-checked and symbolically verified
   (arm_sched + arm_equiv); 4 of the 5 cycles.

   The last group has no successor to load for, hence _LAST. */
#define LZ_Q8_ARM_BLK_PIPE                                    \
    LZ_Q8_ARM_HALF("r4") LZ_Q8_ARM_HALF_HEAD("r5")            \
    LZ_Q8_ARM_WLOAD LZ_Q8_ARM_TAIL
#define LZ_Q8_ARM_BLK_LAST                                    \
    LZ_Q8_ARM_HALF("r4") LZ_Q8_ARM_HALF("r5")

int32_t lz_dot32_q8_arm_asm(const int8_t *w, const int16_t *x) {
    const int8_t *wp = w;
    const int16_t *xp = x;
    int32_t a0;
    __asm__(
        "mov    r3, #0xff00\n\t"
        "orr    r3, r3, r3, lsl #16\n\t"
        "mov    %[a0], #0\n\t"
        LZ_Q8_ARM_WLOAD                 /* group 0's weights, one early */
        "mov    r12, #0\n\t"
        LZ_Q8_ARM_BLK_PIPE LZ_Q8_ARM_BLK_PIPE LZ_Q8_ARM_BLK_PIPE
        LZ_Q8_ARM_BLK_LAST
        "add    %[a0], %[a0], r12\n\t"
        : [a0] "=&r"(a0), [wp] "+r"(wp), [xp] "+r"(xp)
        :
        : "r3", "r4", "r5", "r6", "r7", "r10", "r11", "r12", "cc");
    return a0;
}
#endif /* LZ_ARM_ASM_EXTERN */

/* ---- Q4_1: one 4-bit plane, x256 fold -------------------------------- */

/* C tier. 16 bytes per 32 elements, low nibble for element k, high
   nibble for element k+16 - matmul_scalar_ref_one's q4 branch, two
   bytes per iteration so gcc has something to interleave. Codes are
   UNSIGNED 0..15; the per-group min is epi_q41's zero term. */
int32_t lz_dot32_q41_arm(const unsigned char *restrict p,
                         const int16_t *restrict x) {
    int32_t a0 = 0, a1 = 0;
    int k;
    for (k = 0; k < 16; k += 2) {
        a0 += (int32_t)(p[k] & 15) * (int32_t)x[k];
        a1 += (int32_t)(p[k] >> 4) * (int32_t)x[k + 16];
        a0 += (int32_t)(p[k + 1] & 15) * (int32_t)x[k + 1];
        a1 += (int32_t)(p[k + 1] >> 4) * (int32_t)x[k + 17];
    }
    return (a0 + a1) << 8;
}

#if defined(LZ_ARM_ASM_EXTERN)
/* Hand-written tier. The unpack and the x256 fold are ONE instruction
   each, which is the whole cell:

     and r10, r3, W, lsl #8    {lonib(b0)*256, lonib(b2)*256}
     and r11, r3, W            {lonib(b1)*256, lonib(b3)*256}
     and r10, r3, W, lsl #4    {hinib(b0)*256, hinib(b2)*256}
     and r11, r3, W, lsr #4    {hinib(b1)*256, hinib(b3)*256}

   with r3 = 0x0f000f00. The barrel shifter moves the nibble and the
   mask selects it in the same cycle, and landing it at bit 8 of the
   halfword IS the x256 that punpcklbw(0, .) gives the MMX kernel - so
   four weights come out of one word, scaled and in multiply position,
   in one instruction per pair. gcc cannot get there: from the C above
   it emits ldrb, and, lsr and one smlabb per element.

   Elements k..k+3 and k+16..k+19 share that word (low and high
   nibbles), so the activations arrive as two streams - xp over the
   first half of the block, yp over the second - and both are LDM.

   Register map: r3 the mask, r4 the weight word, r6/r7 activations,
   r10/r11 the scaled code pairs, r12 the second accumulator. */
/* Pipelined by one instruction, exactly as LZ_Q8_ARM_BLK_PIPE is and
   for the same reason: the `ldr r4` that opens a group issues right
   after the previous group's final MAC and nothing covers its latency.
   Hoisting it above that MAC is legal because the MAC reads r11, r7,
   r12 while the load writes r4 and %[wp], and r4 is fully consumed by
   the two `and`s at the top of each half. */
#define LZ_Q41_ARM_WLOAD "ldr    r4, [%[wp]], #4\n\t"
#define LZ_Q41_ARM_TAIL  "smlatt r12,   r11, r7, r12\n\t"
#define LZ_Q41_ARM_BODY                                       \
    "ldmia  %[xp]!, {r6, r7}\n\t"                             \
    "and    r10, r3, r4, lsl #8\n\t"                          \
    "and    r11, r3, r4\n\t"                                  \
    "smlabb %[a0], r10, r6, %[a0]\n\t"                        \
    "smlabt r12,   r11, r6, r12\n\t"                          \
    "smlatb %[a0], r10, r7, %[a0]\n\t"                        \
    LZ_Q41_ARM_TAIL                                           \
    "ldmia  %[yp]!, {r6, r7}\n\t"                             \
    "and    r10, r3, r4, lsl #4\n\t"                          \
    "and    r11, r3, r4, lsr #4\n\t"                          \
    "smlabb %[a0], r10, r6, %[a0]\n\t"                        \
    "smlabt r12,   r11, r6, r12\n\t"                          \
    "smlatb %[a0], r10, r7, %[a0]\n\t"
#define LZ_Q41_ARM_BLK_PIPE LZ_Q41_ARM_BODY LZ_Q41_ARM_WLOAD LZ_Q41_ARM_TAIL
#define LZ_Q41_ARM_BLK_LAST LZ_Q41_ARM_BODY LZ_Q41_ARM_TAIL

int32_t lz_dot32_q41_arm_asm(const unsigned char *p, const int16_t *x) {
    const unsigned char *wp = p;
    const int16_t *xp = x;
    const int16_t *yp = x + 16;
    int32_t a0;
    __asm__(
        "mov    r3, #0xf00\n\t"
        "orr    r3, r3, r3, lsl #16\n\t"
        "mov    %[a0], #0\n\t"
        LZ_Q41_ARM_WLOAD                /* group 0's weights, one early */
        "mov    r12, #0\n\t"
        LZ_Q41_ARM_BLK_PIPE LZ_Q41_ARM_BLK_PIPE LZ_Q41_ARM_BLK_PIPE
        LZ_Q41_ARM_BLK_LAST
        "add    %[a0], %[a0], r12\n\t"
        : [a0] "=&r"(a0), [wp] "+r"(wp), [xp] "+r"(xp), [yp] "+r"(yp)
        :
        : "r3", "r4", "r6", "r7", "r10", "r11", "r12", "cc");
    return a0;
}
#endif /* LZ_ARM_ASM_EXTERN */

/* ---- Q6_1: 4-bit plane + 2-bit plane, x256 fold ---------------------- */

/* C tier, written in the reference's shape ON PURPOSE: two plane
   accumulators merged as alo + (ahi << 4). The asm below merges the
   planes per element instead, so keeping this one structurally like
   matmul_scalar_ref_one makes it an independent check of that and not
   a second copy of the same idea.

   The 2-bit plane is hand-unrolled over its stride-8 layout for the
   same reason lz_dot32_t2_arm is: with a variable shift amount gcc
   emits the address arithmetic per element and the loop costs 17
   instructions per 4 products. */
int32_t lz_dot32_q61_arm(const unsigned char *restrict b4,
                         const unsigned char *restrict b2,
                         const int16_t *restrict x) {
    int32_t alo = 0, ahi = 0;
    int k, j;
    for (k = 0; k < 16; k += 2) {
        alo += (int32_t)(b4[k] & 15) * (int32_t)x[k];
        alo += (int32_t)(b4[k] >> 4) * (int32_t)x[k + 16];
        alo += (int32_t)(b4[k + 1] & 15) * (int32_t)x[k + 1];
        alo += (int32_t)(b4[k + 1] >> 4) * (int32_t)x[k + 17];
    }
    for (j = 0; j < 8; j++) {
        unsigned char p = b2[j];
        ahi += (int32_t)((p >> 0) & 3) * (int32_t)x[j + 0];
        ahi += (int32_t)((p >> 2) & 3) * (int32_t)x[j + 8];
        ahi += (int32_t)((p >> 4) & 3) * (int32_t)x[j + 16];
        ahi += (int32_t)((p >> 6) & 3) * (int32_t)x[j + 24];
    }
    return (alo + (ahi << 4)) << 8;
}

#if defined(LZ_ARM_ASM_EXTERN)
/* Hand-written tier. ONE multiply-accumulate per element, not two.

   The reference sums the planes separately and merges at the end. The
   6-bit weight is nibble + 16*code, so sum(lo*x) + 16*sum(hi*x) =
   sum((lo + 16*hi)*x) - exact in integers, and still exact mod 2^32
   where the sum wraps, because two's-complement addition is
   associative. This owes the reference identity, not a tolerance, and
   the leaf gate checks it over the whole single-code domain of BOTH
   planes.

   What makes that worth doing is that ARMv5TE assembles the 6-bit
   weight, scaled, in three instructions per PAIR of elements:

     and r11, r9,  r4, lsl #8      nibbles at bit 8   (value x256)
     and Rs,  r10, B,  lsl #12     codes   at bit 12  (value x4096)
     orr r11, r11, Rs              6-bit weight x256, two lanes

   r9 = 0x0f000f00 and r10 = 0x30003000. The two fields are disjoint
   inside each halfword (bits 8-11 and 12-13), so the orr cannot carry
   between them and the maximum lane value is 63*256 = 16128, safely
   positive as an int16. The barrel shifter supplies the plane's own
   shift for free in both ands - which is the whole reason the 2-bit
   plane's four sub-planes cost nothing extra here, only a different
   immediate.

   Elements k..k+3 come from the b4 word's low nibbles and k+16..k+19
   from its high nibbles, so the block walks both halves of the
   activation vector; the second pair is read at a fixed offset from
   the post-increment xp rather than through a second pointer, which
   is what keeps the whole kernel inside the register file.

   Register map: r4 the b4 word, r5/r8 the two b2 words, r6/r7
   activations, r9/r10 the two masks, r11 the assembled weight pair,
   r12 the second accumulator, and %[w2p] as scratch once the two b2
   words are loaded out of it. */
#define LZ_Q61_ARM_HALF(BR, CSH0, CSH1, NSH0, NSH1)           \
    LZ_Q61_ARM_HALF_HEAD(BR, CSH0, CSH1, NSH0, NSH1)          \
    LZ_Q61_ARM_TAIL

/* One b4 word: 4 low-nibble elements (4m..4m+3) and 4 high-nibble
   ones (16+4m..16+4m+3). Both take their 2-bit codes from the same b2
   word - byte index (4m)&7 is the same for the two - at shifts 2*(m>>1)
   and 4 + 2*(m>>1). */
/* Pipelined by one instruction, as LZ_Q8_ARM_BLK_PIPE is. The final
   `smlatt r12, r11, r7, r12` of LZ_Q61_ARM_HALF reads r11, r7 and r12,
   so the next group's `ldr r4` may issue above it - r4 is consumed by
   the `and`s inside each half, not by the half's last accumulate.
   Split out here rather than hidden in the macro because the tail of
   the second HALF is the instruction being stepped over. */
#define LZ_Q61_ARM_WLOAD "ldr    r4, [%[wp]], #4\n\t"
#define LZ_Q61_ARM_TAIL  "smlatt r12,   r11, r7, r12\n\t"
#define LZ_Q61_ARM_HALF_HEAD(BR, CSH0, CSH1, NSH0, NSH1)      \
    "and    r11, r9,  r4, " NSH0 "\n\t"                       \
    "and    %[w2p], r10, " BR ", " CSH0 "\n\t"                \
    "orr    r11, r11, %[w2p]\n\t"                             \
    "smlabb %[a0], r11, r6, %[a0]\n\t"                        \
    "smlatb r12,   r11, r7, r12\n\t"                          \
    "and    r11, r9,  r4, " NSH1 "\n\t"                       \
    "and    %[w2p], r10, " BR ", " CSH1 "\n\t"                \
    "orr    r11, r11, %[w2p]\n\t"                             \
    "smlabt %[a0], r11, r6, %[a0]\n\t"
#define LZ_Q61_ARM_BODY(BR, LSH0, LSH1, HSH0, HSH1)           \
    "ldmia  %[xp]!, {r6, r7}\n\t"                             \
    LZ_Q61_ARM_HALF(BR, LSH0, LSH1, "lsl #8", "lsl #0")       \
    "ldr    r6, [%[xp], #24]\n\t"                             \
    "ldr    r7, [%[xp], #28]\n\t"                             \
    LZ_Q61_ARM_HALF_HEAD(BR, HSH0, HSH1, "lsl #4", "lsr #4")
#define LZ_Q61_ARM_BLK_PIPE(BR, LSH0, LSH1, HSH0, HSH1)       \
    LZ_Q61_ARM_BODY(BR, LSH0, LSH1, HSH0, HSH1)               \
    LZ_Q61_ARM_WLOAD LZ_Q61_ARM_TAIL
#define LZ_Q61_ARM_BLK_LAST(BR, LSH0, LSH1, HSH0, HSH1)       \
    LZ_Q61_ARM_BODY(BR, LSH0, LSH1, HSH0, HSH1) LZ_Q61_ARM_TAIL

int32_t lz_dot32_q61_arm_asm(const unsigned char *b4, const unsigned char *b2,
                             const int16_t *x) {
    const unsigned char *wp = b4;
    const unsigned char *w2p = b2;
    const int16_t *xp = x;
    int32_t a0;
    __asm__(
        "mov    r9,  #0xf00\n\t"
        "orr    r9,  r9, r9, lsl #16\n\t"
        "mov    r10, #0x3000\n\t"
        "orr    r10, r10, r10, lsl #16\n\t"
        "ldr    r5, [%[w2p]]\n\t"
        "ldr    r8, [%[w2p], #4]\n\t"
        "mov    %[a0], #0\n\t"
        LZ_Q61_ARM_WLOAD                /* group 0's b4 word, one early */
        "mov    r12, #0\n\t"
        /* m=0 and m=1 sit in sub-plane 0 (codes at bit 0) and 2
           (bit 4); m=2 and m=3 in sub-plane 1 (bit 2) and 3 (bit 6).
           The b2 word alternates because byte (4m)&7 does. */
        LZ_Q61_ARM_BLK_PIPE("r5", "lsl #12", "lsl #4", "lsl #8", "lsl #0")
        LZ_Q61_ARM_BLK_PIPE("r8", "lsl #12", "lsl #4", "lsl #8", "lsl #0")
        LZ_Q61_ARM_BLK_PIPE("r5", "lsl #10", "lsl #2", "lsl #6", "lsr #2")
        LZ_Q61_ARM_BLK_LAST("r8", "lsl #10", "lsl #2", "lsl #6", "lsr #2")
        "add    %[a0], %[a0], r12\n\t"
        : [a0] "=&r"(a0), [wp] "+r"(wp), [xp] "+r"(xp), [w2p] "+r"(w2p)
        :
        : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "cc");
    return a0;
}
#endif /* LZ_ARM_ASM_EXTERN */

/* ---- attention weighted-sum row pair --------------------------------- */

/* Not a row kernel: it accumulates a pair of int8 rows into a live
   32-lane int32 accumulator, which is the SSM recurrence's shape rather
   than matmul's (docs/arm-asm-audit.md F2). It sits here because the
   byte-into-the-top-of-a-halfword trick below is the same one q8_0's
   leaf uses, and two files would be two copies of it.

   THE ONE THING C CANNOT REACH. q8_0's leaf can afford bytes scaled by
   256 because its caller cancels the fold; this accumulator cannot -
   its bound (512 rows x 4,161,409) already fills int32. ARMv5TE has no
   SXTB, so gcc emits one ldrsb per element and one 32x32 mul. SMLAW<y>
   closes the gap: it takes the TOP 32 bits of a 32x16 product, i.e. the
   product shifted right by 16, so feeding it a coefficient pre-scaled
   by 256 and a byte sitting at bit 8 of a halfword yields
   (ck*256 * b*256) >> 16 = ck*b EXACTLY - the two 256s cancel in the
   shift, and the byte needed no sign extension because bit 15 of the
   halfword IS bit 7 of the byte. Two `and`s therefore unpack four
   weights and four SMLAW instructions consume them: 17 instructions per
   8 products against the 9 per 2 gcc emits.

   Bit-identity is by construction: the same integer products land in
   the same lanes, only the instruction forming them differs.

   EXTRA PRECONDITION, as for the row kernels: rowA, rowB and acc32 must
   be 4-byte aligned. The rows are the V cache plus a multiple of 32
   (attn_kv_dim is a multiple of 32 - forward.c sizes the scale plane as
   attn_kv_dim/32) and acc32 is an int32_t array.

   Register map: r5 the 0xff00ff00 mask, r6/r7 the unpacked weight
   pairs, r8-r11 four live accumulators. coef[0..1] arrive pre-shifted
   from C: two instructions per PAIR against 64 products. */
#if defined(LZ_ARM_ASM_EXTERN)
/* BOTH ROW WORDS ARE LOADED UP FRONT, into r6 and r4, rather than both
   into r6 one after the other. That costs one more clobbered register
   and NOTHING else - same instruction count, same instructions - and
   removes every load-use stall in the kernel: armv5telint -a scored the
   one-register version at 24 stall cycles on arm926ej-s, 16 of them
   `and` waiting on `ldr`, and this at 8.

   Reusing r6 for the second row forced its load to sit between the two
   MAC groups, where the `and` that unpacks it is the very next
   instruction and a load is latency 3. With a register of its own it
   issues six instructions before its first consumer.

   Worth contrasting with lz_dot32_q16_arm_asm, which also shows stalls
   no reorder can remove: those are `ldm`, and gcc gives every register
   of a multi-register transfer the same latency, so the number there is
   an upper bound. These are `ldr` - one register, one arrival, nothing
   flattened - so the model means what it says. */
#define LZ_WSUM_ARM_BLK                                       \
    "ldmia  %[ac], {r8, r9, r10, r11}\n\t"                    \
    "ldr    r6, [%[pa]], #4\n\t"                              \
    "ldr    r4, [%[pb]], #4\n\t"                              \
    "and    r7, r5, r6, lsl #8\n\t"                           \
    "and    r6, r5, r6\n\t"                                   \
    "smlawb r8,  %[ka], r7, r8\n\t"                           \
    "smlawb r9,  %[ka], r6, r9\n\t"                           \
    "smlawt r10, %[ka], r7, r10\n\t"                          \
    "smlawt r11, %[ka], r6, r11\n\t"                          \
    "and    r7, r5, r4, lsl #8\n\t"                           \
    "and    r4, r5, r4\n\t"                                   \
    "smlawb r8,  %[kb], r7, r8\n\t"                           \
    "smlawb r9,  %[kb], r4, r9\n\t"                           \
    "smlawt r10, %[kb], r7, r10\n\t"                          \
    "smlawt r11, %[kb], r4, r11\n\t"                          \
    "stmia  %[ac]!, {r8, r9, r10, r11}\n\t"

void lz_wsum_pair_arm_asm(const int8_t *rowA, const int8_t *rowB,
                          const int16_t *coef, int32_t *acc32) {
    const int8_t *pa = rowA;
    const int8_t *pb = rowB;
    int32_t *ac = acc32;
    int32_t ka = (int32_t)coef[0] << 8;
    int32_t kb = (int32_t)coef[1] << 8;
    /* __volatile__, unlike the four row leaves above. They return a
       value the caller uses, which keeps them alive; this one's only
       effect is the store through acc32, and a plain __asm__ whose
       outputs are dead is DELETED - gcc emitted a bare `bx lr` for this
       function until the leaf gate caught it. The memory clobber alone
       does not stop that. */
    __asm__ __volatile__(
        "mov    r5, #0xff00\n\t"
        "orr    r5, r5, r5, lsl #16\n\t"
        LZ_WSUM_ARM_BLK LZ_WSUM_ARM_BLK LZ_WSUM_ARM_BLK LZ_WSUM_ARM_BLK
        LZ_WSUM_ARM_BLK LZ_WSUM_ARM_BLK LZ_WSUM_ARM_BLK LZ_WSUM_ARM_BLK
        : [pa] "+r"(pa), [pb] "+r"(pb), [ac] "+r"(ac)
        : [ka] "r"(ka), [kb] "r"(kb)
        : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "cc",
          "memory");
}
#endif /* LZ_ARM_ASM_EXTERN */

/* ---- the SubLN Hadamard's transform body ------------------------------ */

#if defined(LZ_ARM_ASM_EXTERN)
/* ONE INSTRUCTION IS THE WHOLE TRICK, and it is not a DSP instruction:

     add  r4, r4, r5            @ r4 = u + v
     sub  r5, r4, r5, lsl #1    @ r5 = (u+v) - 2v = u - v

   The butterfly writes both outputs with NO temporary register and no
   third instruction, because the barrel shifter is free on the second
   operand. C cannot express it - `u+v` and `u-v` are two reads of u,
   so gcc keeps u live and spends a move or a third register. That is
   what turns 8 registers of LDM into 4 butterflies with 8 arithmetic
   instructions instead of 12.

   The bound is what makes the doubling safe: fwht.h states
   |y| <= n*32767 ~= 1.68e7, so 2v stays under 2^26. Saturating QADD/
   QSUB are NOT usable here for the same reason they are not needed -
   the transform wants wraparound semantics, and the C it must match
   has them.

   Three shapes, because the pair spacing changes with the stage:
     len=1  the two halves are ADJACENT, so one ldm of 8 words is four
            butterflies - and it reads from x while writing to y, which
            is how the entry memcpy disappears entirely.
     len=2  pairs are (0,2) and (1,3) inside each group of 4; one ldm
            of 8 covers two groups.
     len>=4 the halves are len words apart, so two ldm and two stm.
            stmdb after a writeback ldmia puts them back where they came
            from without a second pointer.

   Register map is the same in all three: r4-r11 the eight words in
   flight, nothing else. */

#define LZ_FWHT_BFLY(A, B)                                        \
    "add    " A ", " A ", " B "\n\t"                              \
    "sub    " B ", " A ", " B ", lsl #1\n\t"

/* len=1 fused with the copy: y[2k] = x[2k]+x[2k+1], y[2k+1] = diff. */
static void fwht_arm_stage1(int32_t *y, const int32_t *x, int n) {
    const int32_t *xp = x;
    int32_t *yp = y;
    int cnt = n >> 3;
    __asm__ __volatile__(
        "1:\n\t"
        "ldmia  %[xp]!, {r4, r5, r6, r7, r8, r9, r10, r11}\n\t"
        LZ_FWHT_BFLY("r4", "r5")
        LZ_FWHT_BFLY("r6", "r7")
        LZ_FWHT_BFLY("r8", "r9")
        LZ_FWHT_BFLY("r10", "r11")
        "stmia  %[yp]!, {r4, r5, r6, r7, r8, r9, r10, r11}\n\t"
        "subs   %[cnt], %[cnt], #1\n\t"
        "bne    1b\n\t"
        : [xp] "+r"(xp), [yp] "+r"(yp), [cnt] "+r"(cnt)
        :
        : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "cc",
          "memory");
}

/* len=2: inside each group of four, the pairs are (0,2) and (1,3). */
static void fwht_arm_stage2(int32_t *y, int n) {
    int32_t *p = y;
    int cnt = n >> 3;
    __asm__ __volatile__(
        "1:\n\t"
        "ldmia  %[p], {r4, r5, r6, r7, r8, r9, r10, r11}\n\t"
        LZ_FWHT_BFLY("r4", "r6")
        LZ_FWHT_BFLY("r5", "r7")
        LZ_FWHT_BFLY("r8", "r10")
        LZ_FWHT_BFLY("r9", "r11")
        "stmia  %[p]!, {r4, r5, r6, r7, r8, r9, r10, r11}\n\t"
        "subs   %[cnt], %[cnt], #1\n\t"
        "bne    1b\n\t"
        : [p] "+r"(p), [cnt] "+r"(cnt)
        :
        : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "cc",
          "memory");
}

/* len>=4: the WHOLE stage, group loop included. The halves are len words
   apart, so this is the one shape that needs two pointers and two
   stores; stmdb after a writeback ldmia puts the words back where they
   came from without keeping a third.

   The group loop is here rather than in C because in C it cost NINE
   instructions per group - gcc ran out of registers outside the r4-r11
   clobber and reloaded the stage constants from the stack every group.
   Here it is five, and the trick that makes it fit is deriving the
   inner count from lenb instead of carrying it: `mov cnt, lenb, lsr #4`
   is len/4, one instruction, one register saved.

   Advancing both pointers by lenb after the inner loop lands them
   exactly on the next group. p1 ends the inner loop at y+i+len and p2
   at y+i+2len; +lenb each gives y+i+2len and y+i+3len, which are the
   next group's two halves. So the group step needs no multiply and no
   separate index - and `cmp p1, end` is the whole termination test. */
static void fwht_arm_stagen(int32_t *y, int n, int len) {
    int32_t *p1 = y;
    int32_t *p2 = y + len;
    int lenb = len << 2;
    const int32_t *end = y + n;
    __asm__ __volatile__(
        "2:\n\t"
        "mov    r12, %[lenb], lsr #4\n\t"       /* len/4 butterflies-by-4 */
        "1:\n\t"
        "ldmia  %[p1]!, {r4, r5, r6, r7}\n\t"
        "ldmia  %[p2]!, {r8, r9, r10, r11}\n\t"
        LZ_FWHT_BFLY("r4", "r8")
        LZ_FWHT_BFLY("r5", "r9")
        LZ_FWHT_BFLY("r6", "r10")
        LZ_FWHT_BFLY("r7", "r11")
        "stmdb  %[p1], {r4, r5, r6, r7}\n\t"
        "stmdb  %[p2], {r8, r9, r10, r11}\n\t"
        "subs   r12, r12, #1\n\t"
        "bne    1b\n\t"
        "add    %[p1], %[p1], %[lenb]\n\t"
        "add    %[p2], %[p2], %[lenb]\n\t"
        "cmp    %[p1], %[end]\n\t"
        "blo    2b\n\t"
        : [p1] "+r"(p1), [p2] "+r"(p2)
        : [lenb] "r"(lenb), [end] "r"(end)
        : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "cc",
          "memory");
}

void lz_fwht_i32_arm_asm(int32_t *y, const int32_t *x, int n) {
    int len;
    fwht_arm_stage1(y, x, n);
    fwht_arm_stage2(y, n);
    for (len = 4; len < n; len <<= 1)
        fwht_arm_stagen(y, n, len);
}
#endif /* LZ_ARM_ASM_EXTERN */

/* ---- norm_ss_fixed's element loop ------------------------------------ */

#if defined(LZ_ARM_ASM_EXTERN)
/* THREE THINGS C CANNOT REACH HERE, and one thing the SHAPE buys that
   no single instruction does:

   1. THE SCALE IS AN EXPONENT READ, NOT A MULTIPLY. sc is pow2f's
      return value, an exact power of two, so q8_round(x[i] * sc) is
      q8_round with k added to the exponent it was going to read
      anyway - the same number, minus one `bl __aeabi_fmul` (28
      instructions). The C says `float * float` and gcc has to honour
      that.
   2. THE MAGIC ADD IS A SHIFT AND A ROUNDING DECISION. Same mechanism
      as lz_q8_round_pow2_arm_asm in src/ops_arm_prim.h - CLZ is not
      even needed, because the shift count comes out of the exponent
      field directly - against a full `bl __aeabi_fadd` (48).
   3. SMLALBB IS THE ACCUMULATE. `acc += (lz_i64)(v * v)` is a 32x32
      `mul` plus `adds`/`adc` from gcc, because the C says int. After
      the clamp |v| <= 32767, so it is a 16x16 multiply into a 64-bit
      accumulator: ONE instruction, and on ARM9E the halfword multiply
      issues in a single cycle where the 32x32 does not.

   And the shape: THE CLAMP RUNS BEFORE THE SIGN, not after. The C
   clamps v to [-32767, 32767] with two compares and two moves; that
   window is symmetric, so clamping the MAGNITUDE and then negating is
   the same number in half the instructions. Which is only true because
   the sign has not been applied yet - the rounding above works on the
   magnitude and defers the negate, exactly as q8_round's primitive
   does.

   Bit-identity is by construction and by precondition, both stated in
   src/ops_arm.h: the same integer rounding decision, the same clamp
   window, the same products summed in the same order.

   Register map: r4/r5 the 64-bit accumulator, r6 the float word, r7
   the shift count then the dropped bits, r8 the mantissa, r9 the
   rounded value, r10 = 150 - k, r11 = 32767. */
#if LZ_SOFTFP_FAST
/* Round toward zero: the dropped bits are dropped, no round-up and no
   tie-to-even. Same class of miss as lz_q8_round_pow2_arm_asm was - the
   norm's sum-of-squares kernel rounds q8_round(x[i]*sc) on the bit
   pattern, and the magic-add path got the RTZ truncation while this
   inline kernel kept round-half-to-even. The C reference (norm_ss_fixed)
   reaches q8_round via lz_q8_round_bits, which now truncates; this must
   match, or the ARM norm's ss diverges from x86's MXCSR-truncated magic
   add on every value whose dropped bits exceed half. */
#define LZ_NORMSS_ARM_BODY(STORE)                             \
    "ldr    r6, [%[xp]], #4\n\t"                              \
    "mov    r7, r6, lsl #1\n\t"                               \
    "mov    r7, r7, lsr #24\n\t"                              \
    "sub    r7, r10, r7\n\t"                                  \
    "bic    r8, r6, #0xFF000000\n\t"                          \
    "orr    r8, r8, #0x800000\n\t"                            \
    "mov    r9, r8, lsr r7\n\t"                               \
    "cmp    r9, r11\n\t"                                      \
    "movgt  r9, r11\n\t"                                      \
    "cmp    r6, #0\n\t"                                       \
    "rsbmi  r9, r9, #0\n\t"                                   \
    STORE                                                     \
    "smlalbb r4, r5, r9, r9\n\t"                              \
    "subs   %[cnt], %[cnt], #1\n\t"
#else
#define LZ_NORMSS_ARM_BODY(STORE)                             \
    "ldr    r6, [%[xp]], #4\n\t"                              \
    "mov    r7, r6, lsl #1\n\t"                               \
    "mov    r7, r7, lsr #24\n\t"                              \
    "sub    r7, r10, r7\n\t"                                  \
    "bic    r8, r6, #0xFF000000\n\t"                          \
    "orr    r8, r8, #0x800000\n\t"                            \
    "mov    r9, r8, lsr r7\n\t"                               \
    "rsb    r7, r7, #32\n\t"                                  \
    "mov    r7, r8, lsl r7\n\t"                               \
    "cmp    r7, #0x80000000\n\t"                              \
    "addhi  r9, r9, #1\n\t"                                   \
    "andeq  r7, r9, #1\n\t"                                   \
    "addeq  r9, r9, r7\n\t"                                   \
    "cmp    r9, r11\n\t"                                      \
    "movgt  r9, r11\n\t"                                      \
    "cmp    r6, #0\n\t"                                       \
    "rsbmi  r9, r9, #0\n\t"                                   \
    STORE                                                     \
    "smlalbb r4, r5, r9, r9\n\t"                              \
    "subs   %[cnt], %[cnt], #1\n\t"
#endif /* LZ_SOFTFP_FAST */

void lz_norm_ss_fixed_arm_asm(const float *x, int n, int k, short *qout,
                              lz_i64 *acc) {
    const float *xp = x;
    short *qp = qout;
    int cnt = n;
    /* __volatile__, for lz_wsum_pair_arm_asm's reason: this returns
       nothing, its outputs are the stores through qout and acc, and a
       plain __asm__ whose outputs are dead is DELETED. An accumulator
       loop is the worst case for that - there is no returned value to
       keep it alive at all. */
    __asm__ __volatile__(
        "mov    r4, #0\n\t"
        "mov    r5, #0\n\t"
        "rsb    r10, %[k], #150\n\t"
        "mov    r11, #0x7F00\n\t"
        "orr    r11, r11, #0xFF\n\t"
        "cmp    %[qp], #0\n\t"
        "beq    2f\n\t"
        "1:\n\t"
        LZ_NORMSS_ARM_BODY("strh   r9, [%[qp]], #2\n\t")
        "bne    1b\n\t"
        "b      3f\n\t"
        "2:\n\t"
        LZ_NORMSS_ARM_BODY("")
        "bne    2b\n\t"
        "3:\n\t"
        "stmia  %[ac], {r4, r5}\n\t"
        : [xp] "+r"(xp), [qp] "+r"(qp), [cnt] "+r"(cnt)
        : [k] "r"(k), [ac] "r"(acc)
        : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "cc",
          "memory");
}

/* lz_matmul's F32 row, eight accumulators at a time (audit G2's f32mm
   cell). On a target with no FPU each of those MACs is `bl __aeabi_fmul`
   plus `bl __addsf3`, 74.0 instructions measured; this fuses them and
   costs 61.0 in the core (.prof/arm_chain_count.sh mode 26 against 20).

   WHAT THE FUSION TAKES, and it is not what the audit first priced.
   Both cores are libgcc's own: significands pre-scaled by 2^4 so UMULL's
   high word IS the product's, SBC folding the normalisation step into
   the bias, signed significands so one ADDS with an inline ASR is both
   effective operations, and ADC/BICEQ rounding. What is left over is the
   two call/return pairs, __addsf3's scan of an operand this code just
   produced, and - only in a loop - the accumulator's scan, because after
   the first element the accumulator is this kernel's own output. It does
   NOT skip the product's rounding: that would be an FMA, a different
   function, and build/arm/parity_gate.sh would go red.

   DECLINES BY THE GROUP, never inside one. The C tail this returns to
   folds leftovers into a0 alone, so resuming mid-group would change the
   summation order and stop being bit-identical. Each group therefore
   snapshots the eight accumulators onto the stack before touching them
   and restores that snapshot on the way out - PUSH/POP rather than a
   scratch pointer, which is what keeps the register budget at thirteen.
   Declined: either factor zero/subnormal/inf/NaN, a product exponent
   outside the normal range BEFORE rounding, an accumulator that is
   subnormal/inf/NaN, and a sum that would be subnormal or overflow.
   A zero accumulator is NOT declined - it is every row's first element.

   n < 8 returns 0 and the C does the row, the same convention
   lz_fwht_i32_arm_asm uses for the blocks its loops cannot cover. */
#define LZ_F32MM_PROD                                            \
        "mov    ip, #255\n\t"                                    \
        "ands   r7, ip, r4, lsr #23\n\t"                         \
        "andnes r9, ip, r5, lsr #23\n\t"                         \
        "teqne  r7, ip\n\t"                                      \
        "teqne  r9, ip\n\t"                                      \
        "beq    7f\n\t"                                          \
        "add    r7, r7, r9\n\t"                                  \
        "eor    lr, r4, r5\n\t"                                  \
        "mov    r9, #0x8000000\n\t"                              \
        "mov    r4, r4, lsl #9\n\t"                              \
        "orr    r4, r9, r4, lsr #5\n\t"                          \
        "mov    r5, r5, lsl #9\n\t"                              \
        "orr    r5, r9, r5, lsr #5\n\t"                          \
        "umull  r10, r9, r4, r5\n\t"                             \
        "and    lr, lr, #0x80000000\n\t"                         \
        "cmp    r9, #0x800000\n\t"                               \
        "movcc  r9, r9, lsl #1\n\t"                              \
        "orrcc  r9, r9, r10, lsr #31\n\t"                        \
        "movcc  r10, r10, lsl #1\n\t"                            \
        "orr    lr, lr, r9\n\t"                                  \
        "sbc    r7, r7, #127\n\t"                                \
        "cmp    r7, #253\n\t"                                    \
        "bhi    7f\n\t"                                          \
        "cmp    r10, #0x80000000\n\t"                            \
        "adc    lr, lr, r7, lsl #23\n\t"                         \
        "biceq  lr, lr, #1\n\t"

int lz_matmul_row_arm_asm(const float *w, const float *x, int n,
                          float *acc8) {
    const float *wp = w;
    const float *xp = x;
    int ng0 = n >> 3;
    int ng = ng0;
    __asm__ __volatile__(
        /* GROUP ZERO IS PEELED, and it is not an unrolling trick: this
           kernel INITIALISES acc8 rather than accumulating into it (the
           SSE twin's `xorps xmm0, xmm0` is the same contract), so the
           first group has no addend and skips the adder entirely -
           eight of the row's elements at a third of the cost. The C it
           reproduces starts each accumulator at 0.0f and 0.0f + p is p
           for every p this can produce; p is never a zero, because a
           zero factor declines.

           It also removes the accumulator's special-case scan from
           every LATER group: after this, acc8[k] is always something
           this kernel wrote, which is normal or exactly +0 - never
           subnormal, inf or NaN, because those decline. Only the zero
           test survives. */
        "mov    r4, #0\n\t"                /* acc8 is written even when */
        "mov    r5, #0\n\t"                /* nothing is consumed - the */
        "mov    r6, #0\n\t"                /* caller reads it either way */
        "mov    r7, #0\n\t"
        "mov    r9, #0\n\t"
        "mov    r10, #0\n\t"
        "mov    r11, #0\n\t"
        "mov    ip, #0\n\t"
        "cmp    %[ng], #0\n\t"
        "bne    0f\n\t"
        "stmia  %[ac], {r4, r5, r6, r7, r9, r10, r11, ip}\n\t"
        "b      8f\n\t"
        "0:\n\t"
        "sub    %[ng], %[ng], #1\n\t"
        /* Zeros are group zero's snapshot, so the shared rollback below
           serves it too: a decline here restores zeros, which is what
           "consumed nothing" has to leave behind. */
        "push   {r4, r5, r6, r7, r9, r10, r11, ip}\n\t"
        "mov    r8, #0\n\t"
        "11:\n\t"
        "ldr    r4, [%[wp]], #4\n\t"
        "ldr    r5, [%[xp]], #4\n\t"
        LZ_F32MM_PROD
        "str    lr, [%[ac], r8, lsl #2]\n\t"
        "add    r8, r8, #1\n\t"
        "cmp    r8, #8\n\t"
        "bne    11b\n\t"
        "add    sp, sp, #32\n\t"           /* group zero committed */
        "1:\n\t"
        "cmp    %[ng], #0\n\t"
        "beq    8f\n\t"
        "sub    %[ng], %[ng], #1\n\t"
        "ldmia  %[ac], {r4, r5, r6, r7, r9, r10, r11, ip}\n\t"
        /* r8 zeroed between the LDM and the PUSH that consumes it: an
           eight-register LDM is latency 12 on arm926ej-s and the push
           reads every one of them, so it interlocks. r8 is not in
           either list. One cycle, and the only one an adjacent swap
           can reach in this kernel. */
        "mov    r8, #0\n\t"
        "push   {r4, r5, r6, r7, r9, r10, r11, ip}\n\t"
        "2:\n\t"
        "ldr    r4, [%[wp]], #4\n\t"
        "ldr    r5, [%[xp]], #4\n\t"
        "ldr    r6, [%[ac], r8, lsl #2]\n\t"
        LZ_F32MM_PROD
        /* ---- accumulator: this kernel's own output, so normal or +0 */
        "movs   r4, r6, lsl #1\n\t"
        "moveq  r7, lr\n\t"
        "beq    6f\n\t"                    /* acc is zero: take the product */
        "mov    r5, lr, lsl #1\n\t"
        "teq    r4, r5\n\t"
        "bne    5f\n\t"
        "eors   r7, r6, lr\n\t"
        "movmi  r7, #0\n\t"
        "bmi    6f\n\t"                    /* exact cancellation: +0 */
        "5:\n\t"
        "mov    r7, r4, lsr #24\n\t"
        "rsbs   r9, r7, r5, lsr #24\n\t"
        "addgt  r7, r7, r9\n\t"
        "eorgt  lr, r6, lr\n\t"
        "eorgt  r6, lr, r6\n\t"
        "eorgt  lr, r6, lr\n\t"
        "rsblt  r9, r9, #0\n\t"
        "cmp    r9, #25\n\t"
        "movhi  r7, r6\n\t"
        "bhi    6f\n\t"                    /* too far apart to interact */
        "tst    r6, #0x80000000\n\t"
        "orr    r6, r6, #0x800000\n\t"
        "bic    r6, r6, #0xFF000000\n\t"
        "rsbne  r6, r6, #0\n\t"
        "tst    lr, #0x80000000\n\t"
        "orr    lr, lr, #0x800000\n\t"
        "bic    lr, lr, #0xFF000000\n\t"
        "rsbne  lr, lr, #0\n\t"
        "sub    r7, r7, #1\n\t"
        "adds   r6, r6, lr, asr r9\n\t"
        "rsb    r9, r9, #32\n\t"
        "mov    lr, lr, lsl r9\n\t"
        "and    r9, r6, #0x80000000\n\t"
        "bpl    4f\n\t"
        "rsbs   lr, lr, #0\n\t"
        "rsc    r6, r6, #0\n\t"
        "4:\n\t"
        "cmp    r6, #0x800000\n\t"
        "bcc    9f\n\t"
        "cmp    r6, #0x1000000\n\t"
        "bcc    3f\n\t"
        "movs   r6, r6, lsr #1\n\t"
        "rrx    lr, lr\n\t"
        "add    r7, r7, #1\n\t"
        "cmp    r7, #254\n\t"
        "bcs    7f\n\t"                    /* would overflow to inf */
        "3:\n\t"
        "cmp    lr, #0x80000000\n\t"
        "adc    r6, r6, r7, lsl #23\n\t"
        "biceq  r6, r6, #1\n\t"
        "orr    r7, r6, r9\n\t"
        "b      6f\n\t"
        "9:\n\t"                           /* cancellation, one bit first */
        "lsls   lr, lr, #1\n\t"
        "adc    r6, r6, r6\n\t"
        "subs   r7, r7, #1\n\t"
        "cmpcs  r6, #0x800000\n\t"
        "bcs    3b\n\t"
        "clz    ip, r6\n\t"
        "sub    ip, ip, #8\n\t"
        "subs   r7, r7, ip\n\t"
        "blt    7f\n\t"                    /* subnormal result */
        "mov    r6, r6, lsl ip\n\t"
        "add    r6, r6, r7, lsl #23\n\t"
        "orr    r7, r6, r9\n\t"
        "6:\n\t"
        "str    r7, [%[ac], r8, lsl #2]\n\t"
        "add    r8, r8, #1\n\t"
        "cmp    r8, #8\n\t"
        "bne    2b\n\t"
        "add    sp, sp, #32\n\t"           /* group committed */
        "b      1b\n\t"
        "7:\n\t"
        "pop    {r4, r5, r6, r7, r9, r10, r11, ip}\n\t"
        "stmia  %[ac], {r4, r5, r6, r7, r9, r10, r11, ip}\n\t"
        "add    %[ng], %[ng], #1\n\t"      /* this group did not commit */
        "8:\n\t"
        : [wp] "+r"(wp), [xp] "+r"(xp), [ng] "+r"(ng)
        : [ac] "r"(acc8)
        : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "ip", "lr",
          "cc", "memory");
    return (ng0 - ng) * 8;
}

/* ---- the epilogue MAC: int32 x int16 -> int64 ------------------------ */

/* SMLAL IS THE WHOLE KERNEL. The x86 tiers reach this shape by
   splitting the int32 into 16-bit halves, because neither MMX nor SSE2
   has a signed 32x32 multiply; ARMv5TE has SMLAL - signed 32x32 with a
   64-bit accumulate - so the split, the carry term and the sign
   correction all disappear and one instruction per element does it.
   Exact by construction: the accumulator is the full 64-bit product
   sum, so there is nothing to round and no order to preserve.

   gcc already finds SMLAL for this loop, so the gain is not the
   multiply - it is the operand traffic around it. gcc's inner loop is
   ldr / ldrsh / cmp / smlal / bne, five instructions an element. Here
   LDMIA fetches four a-words in one, which leaves 11 per four
   elements - 2.75 against 5.00.

   The m loads are interleaved between the multiplies rather than issued
   as a block, and r8 is deliberately reused for the fourth: its first
   value is consumed by the time the reload issues, and on an in-order
   core the reload then covers the SMLAL result latency instead of
   stalling behind it. It also leaves r11 to the register allocator,
   which matters - five operands and eight clobbers is already most of
   the file.

   n <= 0 returns 0, matching the C body's empty loop, and the test is
   per CALL. The tail is up to three elements in C; the shipping call
   passes one group count per weight row, which every packed geometry
   makes a multiple of four.

   Alignment is what the pointer types already promise: 4 bytes for the
   int32 LDMIA, 2 for the LDRSH halfwords. Unlike the row kernels above
   this adds no precondition of its own. */
lz_i64 lz_epi_mac_i16_arm_asm(const int32_t *a, const int16_t *m, int n) {
    const int32_t *ap = a;
    const int16_t *mp = m;
    int blk = n >> 2;
    int32_t lo, hi;
    lz_i64 s;
    int g, tail;

    if (n <= 0) return 0;
    __asm__(
        "mov    %[lo], #0\n\t"
        "mov    %[hi], #0\n\t"
        "cmp    %[blk], #0\n\t"
        "beq    2f\n\t"
        "1:\n\t"
        /* The first multiplier halfword ahead of the four-register LDM:
           armv5telint -a costed the other order at one extra stall
           cycle, because `smlal ... r4, r8` waits on the LDM's r4 and
           the ldrsh in between is what covers it. The two are
           independent (r8/%[mp] against r4-r7/%[ap]). */
        "ldrsh  r8,  [%[mp]], #2\n\t"
        "ldmia  %[ap]!, {r4, r5, r6, r7}\n\t"
        "ldrsh  r9,  [%[mp]], #2\n\t"
        "ldrsh  r10, [%[mp]], #2\n\t"
        "smlal  %[lo], %[hi], r4, r8\n\t"
        "ldrsh  r8,  [%[mp]], #2\n\t"
        "smlal  %[lo], %[hi], r5, r9\n\t"
        "smlal  %[lo], %[hi], r6, r10\n\t"
        "smlal  %[lo], %[hi], r7, r8\n\t"
        "subs   %[blk], %[blk], #1\n\t"
        "bne    1b\n\t"
        "2:\n\t"
        : [lo] "=&r"(lo), [hi] "=&r"(hi), [ap] "+r"(ap), [mp] "+r"(mp),
          [blk] "+r"(blk)
        :
        : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "cc", "memory");

    /* One 64-bit two's-complement value in two registers. The low word
       widens UNSIGNED or its top bit is subtracted a second time. */
    s = ((lz_i64)hi << 32) + (lz_i64)(uint32_t)lo;
    tail = n & 3;
    for (g = 0; g < tail; g++) s += (lz_i64)ap[g] * (lz_i64)mp[g];
    return s;
}

/* ---- GDN/KDA pass-1, 4-lane table contraction ------------------------ */

/* The GDN recurrence's pass-1 contraction, the ARM twin of
   lz_gdn1_x4_asm (src/ops_mmx.c). One 4-lane sub-window:

     au[l] = sum_kp  rowA[(2*kp)*vd + l] * tab[kp][0]
                   + rowA[(2*kp+1)*vd + l] * tab[kp][1]
     aw[l] = sum_kp  rowA[(2*kp)*vd + l] * tab[kp][4]
                   + rowA[(2*kp+1)*vd + l] * tab[kp][5]

   THE LANE-OUTER STRUCTURE is what the audit's D2 row lands on after the
   LDM-packed-int8 route turned out negative: eight accumulators do not
   fit 13 GPRs once rowA/vd/ctab/kpairs are live, so two accumulators
   stay in registers per lane and the coefficient pair is reloaded for
   each of the four lanes. 12 instructions per lane-pair against gcc's
   16, and the optimistic bound the audit states - not the 4-5 its first
   pass hoped for.

   WHY SMLABB/SMLABT AND NOT MUL/MLA. Inputs are int8 widened to int16,
   coefficients int16, so every product is a 16x16 halfword multiply.
   ARM926EJ-S does SMLAxy in one cycle and MUL/MLA in two to five (early
   termination on Rs; a 16-bit coefficient costs three). gcc fuses the
   narrow load into the halfword multiply only while each byte feeds one
   multiply - CSE the load and both operands widen to plain int. Hand
   writing is the only way to keep both the single load and the halfword
   MAC; from C the compiler gives one or the other. Each coefficient word
   is one ldr feeding SMLABB (low half) and SMLABT (high half), the same
   halfword-selection the matmul row kernels use.

   BIT-IDENTITY IS BY CONSTRUCTION, not by matched order: the products
   are 16x16 exact and the int32 accumulation wraps mod 2^32, so any
   summation order agrees with the C reference and with the x86 twins -
   the same argument gdn_sum_gg's own comment makes. kpairs == 0 writes
   eight zeros, matching the empty-loop C body and the MMX kernel's
   test/jz. */
void lz_gdn1_x4_arm_asm(const int8_t *rowA, int32_t vd, const int16_t *ctab,
                        int32_t kpairs, int32_t *out8) {
    int32_t au, aw, lane, k, a, b, co;
    const int8_t *ra;
    const int16_t *cb;
    __asm__ volatile(
        "mov    %[lane], #0\n\t"           /* lane = 0 */
        "1:\n\t"
        "mov    %[au], #0\n\t"             /* au = 0 */
        "mov    %[aw], #0\n\t"             /* aw = 0 */
        "add    %[ra], %[rowA], %[lane]\n\t" /* ra = rowA + lane */
        "mov    %[cb], %[ctab]\n\t"        /* cb = ctab (reset per lane) */
        "mov    %[k], %[kpairs]\n\t"       /* k = kpairs */
        "cmp    %[k], #0\n\t"
        "beq    3f\n\t"
        "2:\n\t"
        /* Scheduled for arm926ej-s, not written in dataflow order.
           armv5telint -a said 6 stall cycles here and named both:
           `ldr co` moved above `ldrsb b` so the first smlabb is not
           waiting on a load issued one slot earlier, and `add ra`
           moved between the two aw MACs, which are a serial
           accumulator chain (smlabt reads the aw smlabb just wrote).
           Two cycles, dependence-checked and symbolically verified. */
        "ldrsb  %[a], [%[ra]]\n\t"         /* a = ra[0] */
        "ldr    %[co], [%[cb]]\n\t"        /* co = ckA | ckB */
        "ldrsb  %[b], [%[ra], %[vd]]\n\t"  /* b = ra[vd] */
        "smlabb %[au], %[a], %[co], %[au]\n\t" /* au += a*ckA */
        "smlabt %[au], %[b], %[co], %[au]\n\t" /* au += b*ckB */
        "ldr    %[co], [%[cb], #8]\n\t"    /* co = cqA | cqB */
        "smlabb %[aw], %[a], %[co], %[aw]\n\t" /* aw += a*cqA */
        "add    %[ra], %[ra], %[vd], lsl #1\n\t" /* ra += 2*vd */
        "smlabt %[aw], %[b], %[co], %[aw]\n\t" /* aw += b*cqB */
        "add    %[cb], %[cb], #16\n\t"     /* cb += 8 int16 */
        "subs   %[k], %[k], #1\n\t"
        "bne    2b\n\t"
        "3:\n\t"
        "str    %[au], [%[out], %[lane], lsl #2]\n\t" /* out8[lane] = au */
        "add    %[co], %[out], #16\n\t"    /* co = &out8[4] */
        "str    %[aw], [%[co], %[lane], lsl #2]\n\t" /* out8[4+lane] = aw */
        "add    %[lane], %[lane], #1\n\t"
        "cmp    %[lane], #4\n\t"
        "bne    1b\n\t"
        : [au] "=&r"(au), [aw] "=&r"(aw), [lane] "=&r"(lane),
          [ra] "=&r"(ra), [cb] "=&r"(cb), [k] "=&r"(k),
          [a] "=&r"(a), [b] "=&r"(b), [co] "=&r"(co)
        : [rowA] "r"(rowA), [vd] "r"(vd), [ctab] "r"(ctab),
          [kpairs] "r"(kpairs), [out] "r"(out8)
        : "cc", "memory");
}
#endif /* LZ_ARM_ASM_EXTERN */
