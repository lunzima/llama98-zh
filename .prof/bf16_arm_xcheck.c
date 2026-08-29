/* lz_bf16's ARM fast paths against its own C core.
 *
 * WHY THIS FILE. build/kernel_probe_census.sh's ARM half reports
 * bf16_addsub_fast_asm and bf16_mul_fast_asm as reached by no
 * ARM-cross-building driver, and unlike the dot kernels there is no
 * argument that some end-to-end gate covers them: the Makefile says of
 * lz_bf16.c that NOTHING CALLS IT TODAY, so no forward pass reaches
 * either. They are in the build to keep their portability honest, and
 * that is exactly the state in which an arithmetic defect waits.
 *
 * THE REFERENCE IS THE FILE'S OWN C CORE, not an oracle this file
 * invents. src/ops_arm_prim.h states the rule these follow: "Every one
 * owes its C tier bit-identity, not closeness. Where a case is not
 * provably identical the primitive DECLINES (returns 0) and the caller
 * runs the C." So the comparison is between a build where the fast path
 * runs and a build where it always declines - same source, same inputs,
 * and the C core answers both times in the second. The driver makes the
 * declining build; this file only has to be deterministic.
 *
 * WHAT THE SWEEP DRIVES, chosen for what a bf16 fast path can get wrong:
 *   - both signs, and both orders of magnitude, so a sign or an
 *     exchange error shows.
 *   - exponents at the ends: 0, 1, 254, 255 reach zero, subnormal,
 *     overflow and NaN/inf, which are the cases a fast path declines
 *     and therefore the cases where declining wrongly is invisible to a
 *     sweep that stays in the middle.
 *   - mantissas whose low bits force a round-to-even tie, since the
 *     store is the only rounding these do.
 *
 * Exit: 0 pass, 1 fail. Seed fixed at 1145141919.
 */
#include <stdio.h>
#include "lz_int.h"
#include "lz_bf16.h"

static unsigned long rs = 1145141919UL;
static unsigned long nextu(void) { rs = rs * 1103515245UL + 12345UL; return rs; }

/* A bf16 bit pattern in the low 16 bits, built so the exponent and the
   mantissa are swept independently rather than uniformly at random -
   a uniform draw almost never lands on an exponent end. */
static uint32_t bpat(int i) {
    static const int EXP[] = { 0, 1, 2, 126, 127, 128, 253, 254, 255 };
    unsigned long u = nextu();
    int e = ((i & 1) ? EXP[(u >> 3) % 9] : (int)((u >> 5) % 256));
    unsigned m = (unsigned)((u >> 13) & 0x7Fu);
    if ((i & 3) == 0) m |= 1u;          /* force the tie bit sometimes */
    return ((u & 1u) << 15) | ((unsigned)e << 7) | m;
}

int main(void) {
    unsigned long h = 2166136261UL;
    long n = 0, nz = 0;
    int pass;

    for (pass = 0; pass < 200000; pass++) {
        uint32_t a = bpat(pass), b = bpat(pass + 1), r;
        int op = pass % 3;
        if (op == 0)      r = lz_bf16_add(a, b);
        else if (op == 1) r = lz_bf16_sub(a, b);
        else              r = lz_bf16_mul(a, b);
        h ^= r; h *= 16777619UL; h &= 0xFFFFFFFFUL;
        if (r) nz++;
        n++;
    }
    /* Sensitivity: an implementation that returned 0 for everything
       would hash consistently and agree with itself across both builds. */
    printf("  %ld of %ld result(s) non-zero\n", nz, n);
    if (nz * 4 < n) {
        printf("  too many zeros; the sweep is not exercising these\n");
        return 1;
    }
    printf("bf16-arm-xcheck: hash %08lX over %ld op(s)\n", h, n);
    return 0;
}
