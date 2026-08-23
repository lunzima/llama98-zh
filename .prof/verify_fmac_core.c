/* spw_fmac_core against the C it has to reproduce, on the target.
 *
 * The core is f32mm armA's arithmetic with no decline wrapper: it
 * assumes every operand and the result are normal, which a leaf kernel
 * would enforce with a scan the core does not carry. So this gate has
 * to state that domain and stay inside it - a mismatch outside it says
 * nothing, and a sweep that never leaves it proves nothing about the
 * scan. The DOMAIN rows below are what fix the boundary; the VALUE rows
 * check the arithmetic within it.
 *
 * THE CONTRACT IS fl(acc + fl(w*x)), two roundings, NOT an FMA. The C
 * arm is written as two statements with a volatile between them so no
 * compiler can contract it, which is the same hazard -ffp-contract=off
 * exists for in the engine.
 *
 * Rows:
 *   A  EXPONENT LATTICE. w and x over the exponent range, acc placed
 *      relative to the product rather than absolutely - what the adder
 *      turns on is the DIFFERENCE, and drawing acc independently would
 *      spend the whole sweep on the far-apart shortcut.
 *   B  TIES. Constructed so the bits below the product's ulp are
 *      exactly half, both directions of round-to-even. A random sweep
 *      reaches this almost never and it is where ADC/BICEQ live.
 *   C  CANCELLATION, including EXACT cancellation (acc == -w*x), which
 *      is the case CLZ cannot answer and the one the teq/bne guard was
 *      added for. Sweeps every magnitude of difference from 0 up.
 *   D  RANDOM inside the domain.
 *
 * Build and run (WSL), from the repo root:
 *   arm-linux-gnueabi-gcc-14 -O2 -static -msoft-float -march=armv5te \
 *     -mtune=arm926ej-s -std=gnu99 -I.prof -o /tmp/vfm \
 *     .prof/verify_fmac_core.c
 *   qemu-arm-static -cpu arm926 /tmp/vfm
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "spw_fmac_core.h"

#if !defined(__arm__) || !defined(__GNUC__)
int main(void) {
    printf("verify_fmac_core: SKIP - not an ARM gcc build\n");
    return 0;
}
#else

static long fails[4], cases[4], skipped[4];

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
static int normal(uint32_t u) {
    uint32_t e = (u >> 23) & 0xFFu;
    return e != 0u && e != 0xFFu;
}

/* The stated domain. Everything here is what a leaf kernel's scan would
   have to enforce before entering the core, so the list IS the scan's
   specification and not a convenience. */
static int in_domain(uint32_t uw, uint32_t ux, uint32_t ua, uint32_t *pp) {
    volatile float p;
    uint32_t up;
    int ep2;
    if (!normal(uw) || !normal(ux) || !normal(ua)) return 0;
    /* THE PRODUCT'S EXPONENT BEFORE ROUNDING, not after. float x float
       is exact in double, so frexp gives the true one, and the two
       differ: 0x36800000 * 0x097FFFFF has an unrounded exponent of 0
       and rounds UP to 0x00800000, which is normal. The core's own test
       is on the unrounded value (SBC leaves biased-1, then `cmp #253`),
       so a domain written on the rounded product is one case too wide -
       and that case is exactly what the 18 mismatches on the first run
       of this gate were. */
    ep2 = 0;
    (void)frexp((double)f_of(uw) * (double)f_of(ux), &ep2);
    if (ep2 + 126 < 1 || ep2 + 126 > 254) return 0;
    p = f_of(uw) * f_of(ux);
    up = u_of(p);
    if (!normal(up)) return 0;              /* product over/underflowed */
    *pp = up;
    {
        volatile float s = f_of(ua) + p;
        uint32_t us = u_of(s);
        if (us != 0u && us != 0x80000000u && !normal(us)) return 0;
    }
    return 1;
}

static void one(int row, uint32_t ua, uint32_t uw, uint32_t ux) {
    uint32_t up, got, want;
    volatile float p, s;
    if (!in_domain(uw, ux, ua, &up)) { skipped[row]++; return; }
    cases[row]++;
    /* volatile between the two operations: this must be two roundings,
       and a contraction into an FMA would silently make the reference
       a different function from the one the core implements. */
    p = f_of(uw) * f_of(ux);
    s = f_of(ua) + p;
    want = u_of(s);
    got = spw_fmac_core(ua, uw, ux);
    /* -0.0 and +0.0: the core answers exact cancellation with +0, which
       is what round-to-nearest gives; the C agrees, so this is not a
       tolerance, only a guard against comparing the two zeros' bits. */
    if (got != want && !(got == 0u && want == 0x80000000u)) {
        if (fails[row] < 8)
            printf("  row %d acc=%08lx w=%08lx x=%08lx  got %08lx want %08lx\n",
                   row, (unsigned long)ua, (unsigned long)uw,
                   (unsigned long)ux, (unsigned long)got,
                   (unsigned long)want);
        fails[row]++;
    }
}

#define NMAN 10
static const uint32_t MANT[NMAN] = {
    0x000000u, 0x000001u, 0x000003u, 0x400000u, 0x555555u,
    0x7FFFFEu, 0x7FFFFFu, 0x123456u, 0x00000Fu, 0x2AAAABu
};

static uint32_t rs = 0x12345678u;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}

int main(void) {
    int ew, ex, i, j, d, k, s;
    long n;

    /* A: w and x over the exponent range; acc placed at the product's
       exponent plus an offset, because the adder turns on the
       difference and not on either exponent alone. */
    for (ew = 1; ew <= 254; ew += 9)
        for (ex = 1; ex <= 254; ex += 17)
            for (i = 0; i < NMAN; i++)
                for (j = 0; j < NMAN; j++) {
                    uint32_t uw = ((uint32_t)ew << 23) | MANT[i];
                    uint32_t ux = ((uint32_t)ex << 23) | MANT[j];
                    volatile float p = f_of(uw) * f_of(ux);
                    int ep = (int)((u_of(p) >> 23) & 0xFFu);
                    for (d = -28; d <= 28; d += 4) {
                        int ea = ep + d;
                        if (ea < 1 || ea > 254) continue;
                        for (s = 0; s < 2; s++) {
                            uint32_t ua = ((uint32_t)ea << 23) | MANT[j]
                                        | ((uint32_t)s << 31);
                            one(0, ua, uw, ux);
                            one(0, ua, uw ^ 0x80000000u, ux);
                        }
                    }
                }

    /* B: ties. acc one ulp-family below the product so the bits that
       fall off are exactly half, with the surviving low bit both ways. */
    for (ew = 40; ew <= 210; ew += 3)
        for (d = 1; d <= 25; d++)
            for (k = 0; k < 2; k++) {
                uint32_t uw = ((uint32_t)ew << 23) | 0x400000u | (uint32_t)k;
                uint32_t ux = 0x3F800000u;              /* 1.0 */
                int ea = ew - d;
                uint32_t ua;
                if (ea < 1) continue;
                ua = (uint32_t)ea << 23;
                one(1, ua, uw, ux);
                one(1, ua | 1u, uw, ux);
                one(1, ua | 0x80000000u, uw, ux);
                one(1, ua, uw, ux ^ 0x80000000u);
            }

    /* C: cancellation, exact one included - acc = -(w*x) to the bit is
       the case CLZ cannot answer. */
    for (ew = 2; ew <= 250; ew += 7)
        for (i = 0; i < NMAN; i++) {
            uint32_t uw = ((uint32_t)ew << 23) | MANT[i];
            uint32_t ux = 0x40490FDBu;                  /* pi */
            volatile float p = f_of(uw) * f_of(ux);
            uint32_t up = u_of(p);
            one(2, up ^ 0x80000000u, uw, ux);           /* exact */
            one(2, up, uw, ux);
            for (j = 0; j < 24; j++) {
                one(2, (up ^ 0x80000000u) - ((uint32_t)1 << j), uw, ux);
                one(2, (up ^ 0x80000000u) + ((uint32_t)1 << j), uw, ux);
            }
        }

    /* D: random inside the domain. */
    for (n = 0; n < 20000000L; n++) {
        uint32_t r1 = rnd(), r2 = rnd(), r3 = rnd();
        uint32_t ew2 = 60u + (r1 >> 24) % 130u;
        uint32_t ex2 = 60u + (r2 >> 24) % 130u;
        uint32_t ea2 = 60u + (r3 >> 24) % 130u;
        one(3, (r3 & 0x807FFFFFu) | (ea2 << 23),
               (r1 & 0x807FFFFFu) | (ew2 << 23),
               (r2 & 0x807FFFFFu) | (ex2 << 23));
    }

    {
        static const char *nm[4] = { "A LATTICE", "B TIES", "C CANCEL",
                                     "D RANDOM" };
        int bad = 0;
        for (i = 0; i < 4; i++) {
            printf("%-10s %12ld in-domain  %10ld outside  %ld mismatches  %s\n",
                   nm[i], cases[i], skipped[i], fails[i],
                   fails[i] ? "FAIL" : "ok");
            if (fails[i]) bad = 1;
        }
        printf("VERIFY: %s\n", bad ? "FAIL" : "PASS");
        return bad;
    }
}
#endif
