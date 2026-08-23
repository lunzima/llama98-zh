/* Is an integer-reciprocal float divide correct, before asking whether
 * it is fast? docs/arm-asm-audit.md G4.
 *
 * __aeabi_fdiv costs 117 instructions where a multiply costs 30, and the
 * engine issues 2,251 divides a token. Its core is a restoring loop, 4
 * quotient bits per 12 instructions - generic, which libgcc must be. All
 * 2,251 here are `constant / amax`, need a 24-bit quotient, and run on a
 * core with UMULL.
 *
 * THE TABLE IS COMPUTED, NOT WRITTEN. A 256-entry reciprocal seed is the
 * one part of this that could be silently wrong - one bad entry is a
 * wrong answer on one divisor in 256, with no symptom. So this file
 * builds it from its own definition at startup and the sweep below
 * covers every divisor, which together are what a hardcoded table would
 * need a separate producer and a separate check for. The shipped form
 * would emit the same array from the same loop.
 *
 * THE ALGORITHM, and every step is exact integer arithmetic:
 *   mb in [2^23, 2^24) is the divisor's significand, ma in [mb, 2*mb)
 *   the dividend's after the one-bit normalisation.
 *   R  = floor(2^47 / mb)                       24 bits, from the table
 *        plus one Newton step plus a correction
 *   Q0 = (ma * R) >> 24                         one UMULL
 *   rem = (ma << 23) - Q0*mb                    exact
 *   while (rem >= mb) { Q0++; rem -= mb; }      at most twice
 *   round: 2*rem against mb, ties to even
 * Q0 then lands in [2^23, 2^24) and the exponent is a subtraction.
 *
 * WHY THE SEED IS A TABLE AND THE REFINEMENT IS INTEGER: a float Newton
 * step is two fmul and an fsub, 103 instructions, so any approach that
 * touches soft-float has already lost to the 117 it is replacing.
 *
 * Build and run (host or ARM):
 *   gcc -O2 -std=gnu99 -o /tmp/fdr .prof/fdiv_recip.c -lm
 *   arm-linux-gnueabi-gcc-14 -O2 -static -msoft-float -march=armv5te \
 *       -std=gnu99 -o /tmp/fdr .prof/fdiv_recip.c -lm
 * Exit 0 all rows ok, 1 any mismatch.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* Seed table: T[i] approximates floor(2^47 / mb) for the mb whose bits
   22..15 are i. Built from the midpoint of each bucket, which is what
   keeps the one Newton step below enough. */
#define RTAB_BITS 12
#define RTAB_N    (1 << RTAB_BITS)
#define RTAB_SH   (23 - RTAB_BITS)
static uint32_t RTAB[RTAB_N];

/* 12 bits of index, not 8, and the first version's 8 is why the fixup
   below was a LOOP. A seed good to k bits plus one Newton step is good
   to 2k, and the quotient needs 24 - so 8 bits left the step two bits
   short and the correction had to iterate, which measured 211 against
   __aeabi_fdiv's 123. 12 bits costs 16KB once per process. */
static void build_rtab(void) {
    int i;
    for (i = 0; i < RTAB_N; i++) {
        /* THE BUCKET'S LARGEST mb, not its midpoint, so the seed is an
           UNDER-estimate for every mb in the bucket. The midpoint form
           this replaced is over-estimate for the bucket's small half,
           and then `e = N - mb*r` underflows as unsigned, the Newton
           step adds a garbage term, and the correction loop below walks
           back from there. It still returned the right answer - the
           host sweep passed - at 655,457 instructions a call on ARM
           against __aeabi_fdiv's 123. A value gate and a cost gate
           catch different defects; this one was invisible to the first
           and unmissable to the second. */
        uint64_t mb = 0x800000u + ((uint64_t)i << RTAB_SH)
                    + ((1u << RTAB_SH) - 1u);
        RTAB[i] = (uint32_t)(((uint64_t)1 << 47) / mb);
    }
}

/* floor(2^47 / mb) exactly, for mb in [2^23, 2^24). */
static uint32_t recip47(uint32_t mb) {
    uint64_t N = (uint64_t)1 << 47;
    uint32_t r = RTAB[(mb >> RTAB_SH) & (RTAB_N - 1)];
    /* One Newton step in integers: r += r*(2^47 - mb*r) / 2^47, with the
       product taken in 64 bits. */
    {
        uint64_t e = N - (uint64_t)mb * r;      /* fits: |e| stays small */
        r += (uint32_t)(((uint64_t)r * e) >> 47);
    }
    /* Correction to exact. The Newton step lands within a couple, so
       this is a bounded fixup and not a loop in disguise - the sweep
       reports the worst iteration count it ever needed. */
    while ((uint64_t)mb * (r + 1u) <= N) r++;
    while ((uint64_t)mb * r > N) r--;
    return r;
}

static long fixups_max;

static uint32_t fdiv_fast(uint32_t ua, uint32_t ub, int *declined) {
    uint32_t ea = (ua >> 23) & 255u, eb = (ub >> 23) & 255u;
    uint32_t ma, mb, sg, R, Q0;
    uint64_t N, rem;
    int e, n;
    *declined = 0;
    if (ea == 0u || ea == 255u || eb == 0u || eb == 255u) {
        *declined = 1;
        return 0;
    }
    ma = (ua & 0x7FFFFFu) | 0x800000u;
    mb = (ub & 0x7FFFFFu) | 0x800000u;
    sg = (ua ^ ub) & 0x80000000u;
    /* +127, not +126: ma/mb lands in [1,2) after the normalisation
       below, so the true exponent is ea-eb and the biased one is
       ea-eb+127. The pack at the end subtracts the other one, because
       Q0 carries the implicit bit. Getting this wrong gives the right
       mantissa at half the value, which is exactly what the first run
       of this file reported - 16.7M mismatches all with matching low
       bits. */
    e = (int)ea - (int)eb + 127;
    if (ma < mb) { ma <<= 1; e--; }
    if (e < 1 || e > 254) { *declined = 1; return 0; }
    R = recip47(mb);
    Q0 = (uint32_t)(((uint64_t)ma * R) >> 24);
    N = (uint64_t)ma << 23;
    rem = N - (uint64_t)Q0 * mb;
    n = 0;
    while (rem >= mb) { Q0++; rem -= mb; n++; }
    if (n > fixups_max) fixups_max = n;
    /* Round half to even on the exact remainder. */
    if (rem * 2u > mb || (rem * 2u == mb && (Q0 & 1u))) Q0++;
    /* Q0 can reach 2^24; adding it to (e-1)<<23 lets that carry into the
       exponent by itself, the same trick the other primitives use. */
    return sg + ((uint32_t)(e - 1) << 23) + Q0;
}

#ifdef FDIV_RECIP_NO_MAIN
/* Included by .prof/arm_prim_count.c mode 31 to PRICE what this file
   verifies. One copy, for the reason .prof/spw_fmac_core.h is one copy:
   a probe measuring one body while a gate checks another is how both
   stay green at once. */
#else

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

static long fails[3], cases[3], decl[3];

static void one(int row, uint32_t ua, uint32_t ub) {
    int d;
    uint32_t got = fdiv_fast(ua, ub, &d);
    uint32_t want;
    volatile float q;
    if (d) { decl[row]++; return; }
    q = f_of(ua) / f_of(ub);
    want = u_of(q);
    cases[row]++;
    if (got != want) {
        if (fails[row] < 6)
            printf("  row %d a=%08lx b=%08lx got %08lx want %08lx\n", row,
                   (unsigned long)ua, (unsigned long)ub,
                   (unsigned long)got, (unsigned long)want);
        fails[row]++;
    }
}

static uint32_t rs = 0x5EED1234u;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}

int main(void) {
    uint32_t m, i;
    long n;
    int r;
    build_rtab();

    /* A: THE TWO NUMERATORS THE ENGINE ACTUALLY DIVIDES BY, against
       EVERY divisor significand - all 2^23 of them, at one exponent.
       The exponent only shifts the result, so sweeping mantissas is the
       whole domain for a fixed pair of exponents. */
    for (m = 0; m < 0x800000u; m++) {
        one(0, u_of(127.0f), 0x3F800000u | m);
        one(0, u_of(32767.0f), 0x3F800000u | m);
    }

    /* B: exponent coverage. Every (ea, eb) pair on a stride, with a few
       mantissas, so the exponent arithmetic and its declines are swept
       even though row A holds them fixed. */
    for (i = 1; i <= 254u; i += 3)
        for (m = 1; m <= 254u; m += 7) {
            static const uint32_t MM[4] = { 0u, 1u, 0x400000u, 0x7FFFFFu };
            int j, k;
            for (j = 0; j < 4; j++)
                for (k = 0; k < 4; k++) {
                    one(1, (i << 23) | MM[j], (m << 23) | MM[k]);
                    one(1, 0x80000000u | (i << 23) | MM[j], (m << 23) | MM[k]);
                }
        }

    /* C: random pairs. */
    for (n = 0; n < 20000000L; n++) {
        uint32_t a = rnd(), b = rnd();
        uint32_t ea = 1u + (a >> 24) % 254u, eb = 1u + (b >> 24) % 254u;
        one(2, (a & 0x807FFFFFu) | (ea << 23),
               (b & 0x807FFFFFu) | (eb << 23));
    }

    {
        static const char *nm[3] = { "A CONST NUM", "B EXPONENTS",
                                     "C RANDOM" };
        int bad = 0;
        for (r = 0; r < 3; r++) {
            printf("%-12s %12ld cases  %10ld declined  %ld mismatches  %s\n",
                   nm[r], cases[r], decl[r], fails[r],
                   fails[r] ? "FAIL" : "ok");
            if (fails[r]) bad = 1;
        }
        printf("worst reciprocal fixup iterations: %ld\n", fixups_max);
        printf("VERIFY: %s\n", bad ? "FAIL" : "PASS");
        return bad;
    }
}
#endif /* FDIV_RECIP_NO_MAIN */
