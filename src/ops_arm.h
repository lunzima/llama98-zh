#ifndef LZ_OPS_ARM_H
#define LZ_OPS_ARM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ARMv5TE row leaf kernels for the quantized weight formats other than
   ternary. Ternary landed first and kept its own file
   (src/ops_t2_arm.c); everything else shares this one, because Q6_1 is
   Q4_1's nibble plane plus T2's 2-bit plane and the three would
   otherwise carry three copies of the same halfword-selection macro.

   Two tiers per format, both selectable (rule 8): the C kernels here
   are `--kernel arm-c`, the hand-written ones `--kernel arm-asm`. The
   C tier is not decoration - it is the control arm the assembly is
   compared against on the same code path, which `--kernel ref` cannot
   be (that one leaves the row kernels entirely, for matmul_scalar_ref).

   THE CONTRACT IS matmul_scalar_ref_one's PER-FORMAT BRANCH
   (src/ops_matmul.c), bit for bit, and these return what the x86 leaf
   kernels return, not what that branch accumulates:

     q8_0   256 * sum(w*x)      the x256 fold punpcklbw(0,byte) gives
                                on MMX; epi_q8's post = 1/256 cancels it
     q4_1   256 * sum(code*x)   codes read UNSIGNED 0..15; the per-group
                                min is epi_q41's zero term, not here
     q6_1   256 * (alo + 16*ahi) where alo is the 4-bit plane and ahi
                                the 2-bit plane - see the note below
     q16_0  sum(w*x)            int16 weights, no fold, epi_q8 post = 1

   All four accumulate in int32, which is EXACT for the activations the
   engine feeds (int8 widened to int16 by matmul_*_impl's g_xw): the
   largest bound is q16_0's 32767*127*32 = 1.33e8. Splitting into two
   accumulators is therefore not an association-order change the way a
   float reduction would be - two's-complement addition is associative,
   so it agrees with the reference even where the sum wraps.

   Q6_1 AND THE `alo + (ahi << 4)` SPLIT. The reference computes the two
   planes into separate accumulators and merges them at the end. The
   assembly instead builds the 6-bit weight `nibble + 16*code` in one
   halfword and issues ONE multiply-accumulate per element. That is the
   same number by distribution - sum(lo*x) + 16*sum(hi*x) =
   sum((lo+16*hi)*x) - exact in integers and still exact mod 2^32 when
   the sum wraps, so it owes the reference identity and not a
   tolerance. Verified over the whole single-code domain (every position
   in both planes, both planes alone) plus 2.4M random cases.

   EXTRA PRECONDITION over the C tiers, for the asm ones only: every
   pointer must be 4-byte aligned, because they read with LDM/LDR.
   Weight rows come from malloc and every block offset is a multiple of
   16 bytes or more (in_dim is a multiple of 32 by the matmul dispatch
   guards); matmul_*_impl's g_xw carries an aligned(4) attribute. */

int32_t lz_dot32_q16_arm(const int16_t *restrict w, const int16_t *restrict x);
int32_t lz_dot32_q8_arm(const int8_t *restrict w, const int16_t *restrict x);
int32_t lz_dot32_q41_arm(const unsigned char *restrict p,
                         const int16_t *restrict x);

#if defined(__arm__) && defined(__GNUC__)
#define LZ_ARM_ASM_EXTERN 1

/* Hand-written ARMv5TE tier. Static instruction counts, which is what
   transfers to the AK7802 (rule 3 - QEMU models no cache or bus, so no
   wall clock from it means anything); the C figure is the dynamic count
   of the loop gcc actually emits at -O2 -march=armv5te:

     q16_0   47 against 135   1.47 vs 4.22 instructions per element
     q8_0    69 against 135   2.16 vs 4.22
     q4_1    69 against 152   2.16 vs 4.75
 */
int32_t lz_dot32_q16_arm_asm(const int16_t *w, const int16_t *x);
int32_t lz_dot32_q8_arm_asm(const int8_t *w, const int16_t *x);
int32_t lz_dot32_q41_arm_asm(const unsigned char *p, const int16_t *x);
#endif /* __arm__ && __GNUC__ */

#ifdef __cplusplus
}
#endif
#endif
