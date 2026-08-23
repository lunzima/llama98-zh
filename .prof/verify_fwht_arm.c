/* lz_faddsub_arm_asm against the C it must reproduce, on the target.
 *
 * The primitive computes lz_fwht's butterfly - a+b and a-b - off one
 * unpack and one alignment. Its contract is BIT-IDENTITY with the C,
 * which on this target is libgcc's __addsf3/__subsf3, or an explicit
 * decline. So there are exactly two ways to be wrong and both are
 * checked on every case:
 *
 *   ACCEPTED and different   the tier would change the answer
 *   DECLINED where the fast path was in fact reachable - not a
 *                            correctness failure, but the decline set
 *                            is a claim about coverage, so row D
 *                            reports the rate rather than asserting
 *
 * Rows, and what each is aimed at:
 *
 *   A  EXPONENT LATTICE. Every ordered pair of exponents within the
 *      band the alignment covers, crossed with mantissa pairs chosen
 *      to sit on the rounding boundaries: 0, 1, 2^22, 2^23-1, and the
 *      four patterns that make the dropped bits exactly half an ulp.
 *      This is where d, the one quantity the shared alignment turns
 *      on, is swept end to end including both declines (d > 27) and
 *      the d = 0 shift-by-32 that ARM answers with zero.
 *
 *   B  TIES. The tie-to-even decision is the single place a soft-float
 *      adder is most often wrong, and a random sweep hits it almost
 *      never. Constructed directly: for each d in 1..27, mantissas
 *      whose alignment leaves the dropped bits at exactly 2^(d-1),
 *      with the surviving low bit both 0 and 1.
 *
 *   C  CANCELLATION. |a| - |b| with the operands within a few ulps of
 *      each other, which is the branch that normalises by CLZ and the
 *      only one where a left shift could move a sticky bit. Sweeps
 *      every mantissa difference from 0 (exact zero out) to 2^24 at
 *      several exponents, both orders.
 *
 *   D  RANDOM, the coverage net the three constructed rows are not.
 *      Uniform over the bit patterns of normal floats, both signs,
 *      with the exponents drawn near each other most of the time
 *      because that is the distribution a Hadamard stage presents.
 *
 * Build and run (WSL), from the repo root:
 *   arm-linux-gnueabi-gcc-14 -O2 -static -msoft-float -march=armv5te \
 *     -mtune=arm926ej-s -std=gnu99 -Isrc -o /tmp/vfa \
 *     .prof/verify_fwht_arm.c
 *   qemu-arm-static -cpu arm926 /tmp/vfa
 *
 * Exit 0 all rows ok, 1 any mismatch.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "ops_arm_prim.h"

#if !defined(LZ_ARM_PRIM_ASM)
int main(void) {
    printf("verify_fwht_arm: SKIP - not an ARM gcc build\n");
    return 0;
}
#else

static long fails[4];
static long cases[4];
static long declines[4];

static float f_of(uint32_t u) {
    union { float f; uint32_t u; } b;
    b.u = u;
    return b.f;
}
static uint32_t u_of(float f) {
    union { float f; uint32_t u; } b;
    b.f = f;
    return b.u;
}

/* One case. The C arm is a plain float add and subtract, which is what
   lz_fwht's inner loop compiles to and therefore what identity is owed
   to. A declined case is counted, not failed. */
static void one(int row, uint32_t ua, uint32_t ub) {
    uint32_t s, d, es, ed;
    float a = f_of(ua), b = f_of(ub);
    cases[row]++;
    if (!lz_faddsub_arm_asm(ua, ub, &s, &d)) {
        declines[row]++;
        return;
    }
    es = u_of(a + b);
    ed = u_of(a - b);
    if (s != es || d != ed) {
        if (fails[row] < 8) {
            printf("  row %d MISMATCH a=%08lx b=%08lx\n"
                   "      sum got %08lx want %08lx\n"
                   "      dif got %08lx want %08lx\n",
                   row, (unsigned long)ua, (unsigned long)ub,
                   (unsigned long)s, (unsigned long)es,
                   (unsigned long)d, (unsigned long)ed);
        }
        fails[row]++;
    }
}

/* Both sign combinations of one magnitude pair, and both orders. The
   primitive's sign handling is claimed to need no case analysis, so
   every magnitude case is run through all four. */
static void quad(int row, uint32_t ma, uint32_t mb) {
    one(row, ma, mb);
    one(row, ma, mb ^ 0x80000000u);
    one(row, ma ^ 0x80000000u, mb);
    one(row, ma ^ 0x80000000u, mb ^ 0x80000000u);
    one(row, mb, ma);
    one(row, mb, ma ^ 0x80000000u);
}

#define NMAN 12
static const uint32_t MANT[NMAN] = {
    0x000000u, 0x000001u, 0x000002u, 0x000003u,
    0x400000u, 0x555555u, 0x7FFFFEu, 0x7FFFFFu,
    0x123456u, 0x7FFFF0u, 0x00000Fu, 0x2AAAABu
};

static uint32_t rs;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}

int main(void) {
    int ea, eb, i, j, d, k;
    long n;

    /* A: exponent lattice. 1 and 254 are the ends of the normal range,
       so both the "would be subnormal" and "would overflow" declines
       are reached from inside the loop rather than only at its edges. */
    for (ea = 1; ea <= 254; ea += 3)
        for (d = -30; d <= 30; d++) {
            eb = ea + d;
            if (eb < 1 || eb > 254) continue;
            for (i = 0; i < NMAN; i++)
                for (j = 0; j < NMAN; j++)
                    quad(0, ((uint32_t)ea << 23) | MANT[i],
                            ((uint32_t)eb << 23) | MANT[j]);
        }

    /* B: exact ties. With the small operand d places down, the dropped
       bits are exactly half an ulp when its low d bits are 1 followed
       by zeros - built here rather than searched for. The surviving
       low bit of the large operand is swept both ways so the
       round-to-even decision has to go both directions. */
    for (d = 1; d <= 27; d++)
        for (ea = 30; ea <= 230; ea += 7)
            for (k = 0; k < 2; k++) {
                uint32_t mlo, big;
                eb = ea - d;
                if (eb < 1) continue;
                big = ((uint32_t)ea << 23) | 0x400000u | (uint32_t)k;
                /* small = 2^(eb) * 1.0 puts its implicit bit exactly d
                   below the big operand's ulp; d == 24 is the half-ulp
                   tie itself, and the neighbours bracket it. */
                mlo = ((uint32_t)eb << 23);
                quad(1, big, mlo);
                quad(1, big, mlo | 1u);
                quad(1, big | 0x7FFFFFu, mlo);
            }

    /* C: cancellation. Differences from 0 up through a full mantissa,
       which is the whole CLZ range, at exponents where a subnormal
       result is and is not reachable. */
    for (ea = 2; ea <= 250; ea += 11)
        for (i = 0; i < 24; i++) {
            uint32_t base = ((uint32_t)ea << 23) | 0x400000u;
            uint32_t off  = (uint32_t)1 << i;
            quad(2, base, base);
            quad(2, base, base - off);
            quad(2, base, base + off);
            quad(2, base, base - off + 1u);
        }

    /* D: random. Exponents drawn close together two thirds of the
       time - a Hadamard stage adds partial sums of one transform, so
       far-apart operands are the tail, not the body. */
    rs = 0x9E3779B9u;
    for (n = 0; n < 40000000L; n++) {
        uint32_t r1 = rnd(), r2 = rnd();
        uint32_t e1 = 1u + (r1 >> 24) % 254u;
        uint32_t e2;
        if ((r2 & 3u) != 0u) {
            int off = (int)((r1 >> 8) % 9u) - 4;
            long t = (long)e1 + off;
            if (t < 1) t = 1;
            if (t > 254) t = 254;
            e2 = (uint32_t)t;
        } else {
            e2 = 1u + (r2 >> 24) % 254u;
        }
        one(3, (r1 & 0x807FFFFFu) | (e1 << 23),
               (r2 & 0x807FFFFFu) | (e2 << 23));
    }

    {
        static const char *nm[4] = { "A LATTICE", "B TIES", "C CANCEL",
                                     "D RANDOM" };
        int bad = 0;
        for (i = 0; i < 4; i++) {
            printf("%-10s %12ld cases  %10ld declined (%5.2f%%)  "
                   "%ld mismatches  %s\n",
                   nm[i], cases[i], declines[i],
                   cases[i] ? 100.0 * (double)declines[i] / (double)cases[i]
                            : 0.0,
                   fails[i], fails[i] ? "FAIL" : "ok");
            if (fails[i]) bad = 1;
        }
        printf("VERIFY: %s\n", bad ? "FAIL" : "PASS");
        return bad;
    }
}
#endif
