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
#define LZ_Q41_ARM_BLK                                        \
    "ldr    r4, [%[wp]], #4\n\t"                              \
    "ldmia  %[xp]!, {r6, r7}\n\t"                             \
    "and    r10, r3, r4, lsl #8\n\t"                          \
    "and    r11, r3, r4\n\t"                                  \
    "smlabb %[a0], r10, r6, %[a0]\n\t"                        \
    "smlabt r12,   r11, r6, r12\n\t"                          \
    "smlatb %[a0], r10, r7, %[a0]\n\t"                        \
    "smlatt r12,   r11, r7, r12\n\t"                          \
    "ldmia  %[yp]!, {r6, r7}\n\t"                             \
    "and    r10, r3, r4, lsl #4\n\t"                          \
    "and    r11, r3, r4, lsr #4\n\t"                          \
    "smlabb %[a0], r10, r6, %[a0]\n\t"                        \
    "smlabt r12,   r11, r6, r12\n\t"                          \
    "smlatb %[a0], r10, r7, %[a0]\n\t"                        \
    "smlatt r12,   r11, r7, r12\n\t"

int32_t lz_dot32_q41_arm_asm(const unsigned char *p, const int16_t *x) {
    const unsigned char *wp = p;
    const int16_t *xp = x;
    const int16_t *yp = x + 16;
    int32_t a0;
    __asm__(
        "mov    r3, #0xf00\n\t"
        "orr    r3, r3, r3, lsl #16\n\t"
        "mov    %[a0], #0\n\t"
        "mov    r12, #0\n\t"
        LZ_Q41_ARM_BLK LZ_Q41_ARM_BLK LZ_Q41_ARM_BLK LZ_Q41_ARM_BLK
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
    "and    r11, r9,  r4, " NSH0 "\n\t"                       \
    "and    %[w2p], r10, " BR ", " CSH0 "\n\t"                \
    "orr    r11, r11, %[w2p]\n\t"                             \
    "smlabb %[a0], r11, r6, %[a0]\n\t"                        \
    "smlatb r12,   r11, r7, r12\n\t"                          \
    "and    r11, r9,  r4, " NSH1 "\n\t"                       \
    "and    %[w2p], r10, " BR ", " CSH1 "\n\t"                \
    "orr    r11, r11, %[w2p]\n\t"                             \
    "smlabt %[a0], r11, r6, %[a0]\n\t"                        \
    "smlatt r12,   r11, r7, r12\n\t"

/* One b4 word: 4 low-nibble elements (4m..4m+3) and 4 high-nibble
   ones (16+4m..16+4m+3). Both take their 2-bit codes from the same b2
   word - byte index (4m)&7 is the same for the two - at shifts 2*(m>>1)
   and 4 + 2*(m>>1). */
#define LZ_Q61_ARM_BLK(BR, LSH0, LSH1, HSH0, HSH1)            \
    "ldr    r4, [%[wp]], #4\n\t"                              \
    "ldmia  %[xp]!, {r6, r7}\n\t"                             \
    LZ_Q61_ARM_HALF(BR, LSH0, LSH1, "lsl #8", "lsl #0")       \
    "ldr    r6, [%[xp], #24]\n\t"                             \
    "ldr    r7, [%[xp], #28]\n\t"                             \
    LZ_Q61_ARM_HALF(BR, HSH0, HSH1, "lsl #4", "lsr #4")

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
        "mov    r12, #0\n\t"
        /* m=0 and m=1 sit in sub-plane 0 (codes at bit 0) and 2
           (bit 4); m=2 and m=3 in sub-plane 1 (bit 2) and 3 (bit 6).
           The b2 word alternates because byte (4m)&7 does. */
        LZ_Q61_ARM_BLK("r5", "lsl #12", "lsl #4", "lsl #8", "lsl #0")
        LZ_Q61_ARM_BLK("r8", "lsl #12", "lsl #4", "lsl #8", "lsl #0")
        LZ_Q61_ARM_BLK("r5", "lsl #10", "lsl #2", "lsl #6", "lsr #2")
        LZ_Q61_ARM_BLK("r8", "lsl #10", "lsl #2", "lsl #6", "lsr #2")
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
#define LZ_WSUM_ARM_BLK                                       \
    "ldmia  %[ac], {r8, r9, r10, r11}\n\t"                    \
    "ldr    r6, [%[pa]], #4\n\t"                              \
    "and    r7, r5, r6, lsl #8\n\t"                           \
    "and    r6, r5, r6\n\t"                                   \
    "smlawb r8,  %[ka], r7, r8\n\t"                           \
    "smlawb r9,  %[ka], r6, r9\n\t"                           \
    "smlawt r10, %[ka], r7, r10\n\t"                          \
    "smlawt r11, %[ka], r6, r11\n\t"                          \
    "ldr    r6, [%[pb]], #4\n\t"                              \
    "and    r7, r5, r6, lsl #8\n\t"                           \
    "and    r6, r5, r6\n\t"                                   \
    "smlawb r8,  %[kb], r7, r8\n\t"                           \
    "smlawb r9,  %[kb], r6, r9\n\t"                           \
    "smlawt r10, %[kb], r7, r10\n\t"                          \
    "smlawt r11, %[kb], r6, r11\n\t"                          \
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
        : "r5", "r6", "r7", "r8", "r9", "r10", "r11", "cc", "memory");
}
#endif /* LZ_ARM_ASM_EXTERN */
