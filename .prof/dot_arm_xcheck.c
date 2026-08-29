/* The Q16_0 / Q4_1 / Q6_1 ARMv5TE dot kernels against their C tiers.
 *
 * WHY THESE THREE. build/kernel_probe_census.sh grew an ARM half, and it
 * reports lz_dot32_q16_arm_asm, lz_dot32_q41_arm_asm and
 * lz_dot32_q61_arm_asm as reached by no ARM-cross-building driver.
 * build/arm/parity_gate.sh does run --kernel arm-asm over a real model,
 * so they are almost certainly executed there - but "almost certainly"
 * is what the census exists to replace, and the Q8_0 sibling has a
 * direct check while these three have only an argument.
 *
 * .prof/t2_arm_xcheck.c is the same file for the ternary kernel, and the
 * shape here is deliberately its shape.
 *
 * WHAT THE CONTRACT IS. src/ops_arm.h: each asm tier owes its C tier
 * BIT-IDENTITY, not closeness - the same integer products in the same
 * lanes. So a value mutation would not test anything the identity does
 * not already pin; what is NOT structural is which halfword of which
 * packed byte each SMLA<x><y> multiplies, and that is what the mutations
 * below move.
 *
 * WHAT WOULD HIDE A DEFECT, and is therefore avoided:
 *   - weights whose nibbles or 2-bit fields are equal: Q4_1 packs two
 *     weights per byte and Q6_1 three fields across two planes, and a
 *     kernel reading the wrong half agrees whenever the halves match.
 *   - activations that repeat: any lane permutation agrees.
 *   - a sign-free sweep: the products are signed and the accumulator is
 *     int32, so a kernel that dropped a sign extension agrees on
 *     non-negative data.
 *
 * Exit: 0 all three agree, 1 otherwise. Seed fixed at 1145141919.
 */
#include <stdio.h>
#include "lz_int.h"
#include "ops_arm.h"

static unsigned long rs = 1145141919UL;
static unsigned long nextu(void) { rs = rs * 1103515245UL + 12345UL; return rs; }

int main(void) {
    int16_t w16[32], x[32];
    unsigned char p4[16], b4[16], b2[8];
    long bad = 0, n = 0, moved = 0;
    int pass, i;

#if !defined(LZ_ARM_ASM_EXTERN)
    printf("dot-arm-xcheck: FAIL - built without LZ_ARM_ASM_EXTERN, so"
           " the asm tiers\n  are not in this binary and nothing was"
           " compared\n");
    return 1;
#else
    for (pass = 0; pass < 20000; pass++) {
        int32_t a, b;
        for (i = 0; i < 32; i++) {
            unsigned long u = nextu();
            /* int16 across the whole signed range, distinct per lane. */
            w16[i] = (int16_t)((int)(u % 65535u) - 32767);
            x[i]   = (int16_t)((int)((u >> 11) % 65535u) - 32767);
        }
        for (i = 0; i < 16; i++) {
            unsigned long u = nextu();
            /* The two nibbles of each byte differ, so reading the wrong
               half is visible - see the header. */
            p4[i] = (unsigned char)((u & 0x0Fu) | (((u >> 5) & 0x0Fu) << 4));
            if ((p4[i] & 0x0Fu) == (p4[i] >> 4)) p4[i] ^= 0x10u;
            b4[i] = (unsigned char)(u >> 9);
        }
        for (i = 0; i < 8; i++) b2[i] = (unsigned char)(nextu() >> 13);

        a = lz_dot32_q16_arm_asm(w16, x);
        b = lz_dot32_q16_arm(w16, x);
        if (a != b) {
            if (bad < 4)
                printf("  q16_0 pass %d: asm %ld c %ld\n",
                       pass, (long)a, (long)b);
            bad++;
        }
        if (a) moved++;
        n++;

        a = lz_dot32_q41_arm_asm(p4, x);
        b = lz_dot32_q41_arm(p4, x);
        if (a != b) {
            if (bad < 4)
                printf("  q4_1 pass %d: asm %ld c %ld\n",
                       pass, (long)a, (long)b);
            bad++;
        }
        if (a) moved++;
        n++;

        a = lz_dot32_q61_arm_asm(b4, b2, x);
        b = lz_dot32_q61_arm(b4, b2, x);
        if (a != b) {
            if (bad < 4)
                printf("  q6_1 pass %d: asm %ld c %ld\n",
                       pass, (long)a, (long)b);
            bad++;
        }
        if (a) moved++;
        n++;
    }

    /* Sensitivity: three kernels that all returned 0 would agree, and
       print the same green as three that computed. */
    printf("  %ld of %ld result(s) non-zero\n", moved, n);
    if (moved * 2 < n) {
        printf("  more than half the results were zero; the sweep is not"
               " exercising these\n");
        bad++;
    }
    if (bad) {
        printf("dot-arm-xcheck: FAIL - %ld mismatch(es) over %ld"
               " comparison(s)\n", bad, n);
        return 1;
    }
    printf("dot-arm-xcheck: PASS (%ld comparison(s), three formats)\n", n);
    return 0;
#endif /* LZ_ARM_ASM_EXTERN */
}
