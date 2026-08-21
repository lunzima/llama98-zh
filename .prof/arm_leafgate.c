/* Bit-identity gate for the ARMv5TE operator primitives.
 *
 * Runs the SAME entry point (lz_i32f / q8_round / lz_exp_fixed) twice
 * per input with g_kernel flipped between LZ_KERNEL_ARM (the shared C)
 * and LZ_KERNEL_ARM_ASM (src/ops_arm_prim.h), and compares the raw
 * bits. Going through the real entry point rather than calling the
 * primitive directly is deliberate: it gates the wiring too, so a tier
 * branch that never fires shows up as "asm never differs" under the
 * mutation runs.
 *
 * Sweeps, in this order (each prints its own count):
 *   exhaustive over the small domain, where both tiers must be exact
 *   exhaustive over every rounding decision pattern (all low bits, a
 *     handful of high parts, every binade)
 *   the tie inputs - exact half-integers and the float ties - which is
 *     where round-half-to-even is the only thing that can differ
 *   uniform random over the whole 32-bit input space
 *
 * Exit 0 = identical everywhere, 1 = at least one mismatch (the first
 * 10 are printed with both bit patterns).
 *
 * Not in the engine tree's test suite by design (iron law 0): this is a
 * cross-build driver that needs QEMU, and it lives here with the rest
 * of the ARM instrumentation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ops.h"
#include "ops_quant.h"
#include "ops_kernel_shared.h"

static uint32_t g_bad;
static long g_n;
/* Divides every sweep bound. 1 is the gate; a large value is the
 * mutation harness, which only needs to reach the first mismatch. */
static long g_div = 1;
static long dv(long n) { n /= g_div; return n < 1 ? 1 : n; }

static uint32_t f2u(float f) { union { float f; uint32_t u; } b; b.f = f; return b.u; }
static float u2f(uint32_t u) { union { float f; uint32_t u; } b; b.u = u; return b.f; }

/* xorshift32 - one line, no libc rand() whose sequence differs between
 * builds and would make a failure unreproducible. */
static uint32_t g_rs = 0x9E3779B9u;
static uint32_t rnd(void) {
    g_rs ^= g_rs << 13; g_rs ^= g_rs >> 17; g_rs ^= g_rs << 5;
    return g_rs;
}

static void chk_i32f(int32_t v) {
    uint32_t a, b;
    g_kernel = LZ_KERNEL_ARM;      a = f2u(lz_i32f(v));
    g_kernel = LZ_KERNEL_ARM_ASM;  b = f2u(lz_i32f(v));
    g_n++;
    if (a != b) {
        if (g_bad < 10)
            printf("MISMATCH i32f v=%ld (0x%08lX) c=0x%08lX asm=0x%08lX\n",
                   (long)v, (unsigned long)(uint32_t)v,
                   (unsigned long)a, (unsigned long)b);
        g_bad++;
    }
}

static void chk_q8r(uint32_t u) {
    int a, b;
    float f = u2f(u);
    g_kernel = LZ_KERNEL_ARM;      a = q8_round(f);
    g_kernel = LZ_KERNEL_ARM_ASM;  b = q8_round(f);
    g_n++;
    if (a != b) {
        if (g_bad < 10)
            printf("MISMATCH q8r bits=0x%08lX (%.9g) c=%d asm=%d\n",
                   (unsigned long)u, (double)f, a, b);
        g_bad++;
    }
}

static void chk_exp(uint32_t u) {
    uint32_t a, b;
    float f = u2f(u);
    g_kernel = LZ_KERNEL_ARM;      a = f2u(lz_exp_fixed(f));
    g_kernel = LZ_KERNEL_ARM_ASM;  b = f2u(lz_exp_fixed(f));
    g_n++;
    if (a != b) {
        if (g_bad < 10)
            printf("MISMATCH exp bits=0x%08lX (%.9g) c=0x%08lX asm=0x%08lX\n",
                   (unsigned long)u, (double)f,
                   (unsigned long)a, (unsigned long)b);
        g_bad++;
    }
}

/* Rolling checksum of the arm-asm arm's own output. A gate that only
 * compares two arms cannot tell "both right" from "both running the
 * same code": this number changes when the assembly changes, so a
 * mutation that leaves the comparison green but moves this one is a
 * mutation the comparison could not see, not a mutation nothing ran. */
static uint32_t g_sum;
static void mix(uint32_t v) { g_sum = g_sum * 1000003u + v; }

static void phase(const char *name) {
    printf("  %-28s n=%-12ld bad=%-8lu sum=0x%08lX\n", name, g_n,
           (unsigned long)g_bad, (unsigned long)g_sum);
    g_n = 0;
}

/* ---- lz_i32f ---------------------------------------------------------- */
static void sweep_i32f(void) {
    long i;
    int p, s, h;
    printf("lz_i32f:\n");
    /* Exhaustive |v| <= 2^25: every value both tiers must convert
       exactly, plus the first two binades that round. */
    for (i = -dv(1L << 25); i <= dv(1L << 25); i++) chk_i32f((int32_t)i);
    phase("exhaustive |v| <= 2^25");

    /* Every rounding decision, every binade: the decision reads the
       dropped bits and the kept LSB, so all low bits x several high
       parts x both signs is the whole decision space. */
    for (p = 24; p <= 30; p++) {
        int nlow = 1 << (p - 23);
        for (h = 0; h < dv(2048); h++) {
            uint32_t hi = ((rnd() & 0x7FFFFFu) | 0x800000u) << (p - 23);
            int lo;
            for (lo = 0; lo < nlow; lo++) {
                int32_t v = (int32_t)((hi & ~(uint32_t)(nlow - 1)) | (uint32_t)lo);
                for (s = 0; s < 2; s++) chk_i32f(s ? -v : v);
            }
        }
    }
    phase("all dropped-bit patterns");

    /* INT32_MIN, the powers of two and their neighbours. */
    chk_i32f(0); chk_i32f((int32_t)0x80000000u); chk_i32f(0x7FFFFFFF);
    for (p = 0; p < 31; p++) {
        int32_t v = (int32_t)1 << p;
        chk_i32f(v); chk_i32f(v - 1); chk_i32f(v + 1);
        chk_i32f(-v); chk_i32f(-v - 1); chk_i32f(-v + 1);
    }
    phase("powers of two and neighbours");

    for (i = 0; i < dv(20000000L); i++) chk_i32f((int32_t)rnd());
    phase("uniform random int32");
}

/* ---- q8_round --------------------------------------------------------- */
static void sweep_q8r(void) {
    long i;
    int e, s, h;
    printf("q8_round:\n");
    /* Every exact half-integer in the contract's domain, both signs:
       the only inputs where round-half-to-even can disagree with any
       other rule at all. Also the quarter points either side. */
    for (i = 0; i < dv(1L << 22); i++) {
        chk_q8r(f2u((float)i + 0.5f));
        chk_q8r(f2u(-((float)i + 0.5f)));
    }
    phase("every half-integer");
    for (i = 0; i < dv(300000L); i++) {
        chk_q8r(f2u((float)i + 0.25f));
        chk_q8r(f2u((float)i + 0.75f));
        chk_q8r(f2u(-((float)i + 0.25f)));
        chk_q8r(f2u(-((float)i + 0.75f)));
    }
    phase("quarter points");

    /* All low mantissa bits x several high parts x every exponent that
       straddles the domain boundary, both signs. */
    for (e = 100; e <= 160; e++) {
        for (h = 0; h < dv(6); h++) {
            uint32_t hi = rnd() & 0x7F0000u;
            long lo;
            for (lo = 0; lo < dv(65536); lo++) {
                uint32_t m = hi | (uint32_t)lo;
                for (s = 0; s < 2; s++)
                    chk_q8r(((uint32_t)s << 31) | ((uint32_t)e << 23) | m);
            }
        }
    }
    phase("all low bits x exponents");

    /* Zero, subnormals, inf, nan, and the domain edge 2^22 itself. */
    chk_q8r(0); chk_q8r(0x80000000u);
    chk_q8r(1); chk_q8r(0x800001u); chk_q8r(0x7FFFFFu);
    chk_q8r(0x7F800000u); chk_q8r(0xFF800000u);
    chk_q8r(0x7FC00000u); chk_q8r(0x7F800001u);
    chk_q8r(f2u(4194304.0f)); chk_q8r(f2u(-4194304.0f));
    chk_q8r(f2u(4194303.5f)); chk_q8r(f2u(-4194303.5f));
    phase("zero/subnormal/inf/nan/edge");

    for (i = 0; i < dv(20000000L); i++) chk_q8r(rnd());
    phase("uniform random bit patterns");
}

/* ---- lz_exp_fixed ----------------------------------------------------- */
static void sweep_exp(void) {
    long i;
    int e, s, h;
    printf("lz_exp_fixed:\n");
    /* The guard edges and the exactly-representable arguments where
       y32 lands on an integer - the tie the floor step has to get
       right. LZ_EXP_LOG2E32 is not reachable from here, so the sweep
       walks x rather than y32 and the exponent sweep below covers the
       integer landings densely. */
    chk_exp(f2u(88.0f)); chk_exp(f2u(88.0f + 1e-5f)); chk_exp(f2u(87.9999f));
    chk_exp(f2u(-87.3f)); chk_exp(f2u(-87.30001f)); chk_exp(f2u(-87.29999f));
    chk_exp(0); chk_exp(0x80000000u); chk_exp(1); chk_exp(0x807FFFFFu);
    chk_exp(0x7F800000u); chk_exp(0xFF800000u); chk_exp(0x7FC00000u);
    phase("guards, zero, subnormal, inf, nan");

    /* All low mantissa bits x several high parts x every exponent in
       and around the live range |x| < 88, both signs. */
    for (e = 100; e <= 140; e++) {
        for (h = 0; h < dv(6); h++) {
            uint32_t hi = rnd() & 0x7F0000u;
            long lo;
            for (lo = 0; lo < dv(65536); lo++) {
                uint32_t m = hi | (uint32_t)lo;
                for (s = 0; s < 2; s++)
                    chk_exp(((uint32_t)s << 31) | ((uint32_t)e << 23) | m);
            }
        }
    }
    phase("all low bits x exponents");

    /* x on and around the values that put y32 exactly on an integer:
       x = k/LZ_EXP_LOG2E32. Walked as floats either side. */
    /* NOT divided by g_div: this is the only sweep that reaches the
       floor step's tie (y32 exactly on an integer), and it is cheap. */
    for (i = -4000; i <= 4000; i++) {
        float x = (float)i / (1.44269504f * 32.0f);
        uint32_t u = f2u(x);
        int d;
        for (d = -4; d <= 4; d++) chk_exp(u + (uint32_t)d);
    }
    phase("y32 near integers");

    for (i = 0; i < dv(4000000L); i++) {
        /* Uniform in the live range, plus raw random patterns so the
           declined paths are exercised too. */
        float x = ((float)(int32_t)rnd() / 2147483648.0f) * 88.0f;
        chk_exp(f2u(x));
        chk_exp(rnd());
    }
    phase("uniform in range + random patterns");
}

/* ---- sigmoid_q15 / sigmoid_q15_t -------------------------------------- */
static void chk_sigt(uint32_t u) {
    int32_t a, b;
    float f = u2f(u);
    g_kernel = LZ_KERNEL_ARM;      a = sigmoid_q15_t(f);
    g_kernel = LZ_KERNEL_ARM_ASM;  b = sigmoid_q15_t(f);
    g_n++;
    mix((uint32_t)b);
    if (a != b) {
        if (g_bad < 10)
            printf("MISMATCH sigt bits=0x%08lX (%.9g) c=%ld asm=%ld\n",
                   (unsigned long)u, (double)f, (long)a, (long)b);
        g_bad++;
    }
}

static void chk_sigx(uint32_t u) {
    int32_t a, b;
    float f = u2f(u);
    g_kernel = LZ_KERNEL_ARM;      a = sigmoid_q15(f);
    g_kernel = LZ_KERNEL_ARM_ASM;  b = sigmoid_q15(f);
    g_n++;
    mix((uint32_t)b);
    if (a != b) {
        if (g_bad < 10)
            printf("MISMATCH sig bits=0x%08lX (%.9g) c=%ld asm=%ld\n",
                   (unsigned long)u, (double)f, (long)a, (long)b);
        g_bad++;
    }
}

static void sweep_sig(void) {
    long i, k, ix;
    int e, s, h, sh;
    printf("sigmoid_q15_t:\n");
    /* EVERY (idx, frac) PAIR. idx + k*2^-15 needs 9 + 15 = 24 bits and
       is therefore an exact float for every idx < 512, so this walks
       the entire semantic domain of the table core: all 385 table
       pairs against all 32768 interpolation points. It is the only
       sweep that can answer for the SMULBB halfword bound, which is a
       fact about the table's values and not about the input float.
       g_div strides the inner loop rather than truncating it, so a
       divided run still touches every table pair. */
    for (ix = 0; ix < 384; ix++)
        for (k = 0; k < 32768; k += g_div)
            chk_sigt(f2u((float)ix + (float)k * (1.0f / 32768.0f)));
    phase("every (idx, frac) pair");

    /* The clamps, their neighbours, and the shapes that are not a
       positive normal in [0, 384). */
    chk_sigt(f2u(0.0f)); chk_sigt(f2u(-0.0f));
    chk_sigt(f2u(384.0f)); chk_sigt(f2u(383.99997f));
    chk_sigt(f2u(384.00003f)); chk_sigt(f2u(-1e-30f));
    chk_sigt(1); chk_sigt(0x807FFFFFu); chk_sigt(0x00800000u);
    chk_sigt(0x7F800000u); chk_sigt(0xFF800000u);
    chk_sigt(0x7FC00000u); chk_sigt(0x7F800001u); chk_sigt(0xFFC00000u);
    for (i = 0; i < 4000; i++) chk_sigt(f2u((float)i * 0.125f));
    phase("clamps, zero, subnormal, inf, nan");

    for (i = 0; i < dv(4000000L); i++) {
        chk_sigt(f2u(((float)(rnd() >> 8) / 16777216.0f) * 400.0f));
        chk_sigt(rnd());
    }
    phase("uniform in range + random patterns");

    printf("sigmoid_q15:\n");
    /* THE TIE, FOR EVERY SHIFT COUNT. The add's rounding reads the
       bits of x below the sum's ulp; the number of them is 130 - e (or
       131 - e on the branch where the sum lands in [16,24)), so the
       tie is a different bit of the mantissa for every exponent and a
       random sweep reaches it with probability 2^-sh. Constructed
       here instead: the dropped field set to exactly its half point
       and to one either side, over both branch shapes, both signs and
       several high mantissa parts. */
    for (e = 106; e <= 131; e++) {
        int shs[2];
        int j;
        shs[0] = 130 - e; shs[1] = 131 - e;
        for (j = 0; j < 2; j++) {
            sh = shs[j];
            if (sh < 1 || sh > 24) continue;
            for (h = 0; h < 64; h++) {
                uint32_t hi = rnd() & 0x7FFFFFu;
                uint32_t mask = (sh >= 24) ? 0x7FFFFFu
                                           : ((1u << sh) - 1u);
                uint32_t tie = 1u << (sh - 1);
                uint32_t base = hi & ~mask;
                int d;
                for (d = -1; d <= 1; d++) {
                    uint32_t m = (base | (tie + (uint32_t)d)) & 0x7FFFFFu;
                    for (s = 0; s < 2; s++)
                        chk_sigx(((uint32_t)s << 31) |
                                 ((uint32_t)e << 23) | m);
                }
                for (s = 0; s < 2; s++) {
                    chk_sigx(((uint32_t)s << 31) | ((uint32_t)e << 23) |
                             (base & 0x7FFFFFu));
                    chk_sigx(((uint32_t)s << 31) | ((uint32_t)e << 23) |
                             ((base | mask) & 0x7FFFFFu));
                }
            }
        }
    }
    phase("the add's tie, every shift count");

    /* All low mantissa bits x several high parts x every exponent that
       reaches the table at all, both signs. */
    for (e = 100; e <= 132; e++) {
        for (h = 0; h < dv(6); h++) {
            uint32_t hi = rnd() & 0x7F0000u;
            long lo;
            for (lo = 0; lo < dv(65536); lo++) {
                uint32_t m = hi | (uint32_t)lo;
                for (s = 0; s < 2; s++)
                    chk_sigx(((uint32_t)s << 31) | ((uint32_t)e << 23) | m);
            }
        }
    }
    phase("all low bits x exponents");

    /* The binade walls the case split is built on - +-4, +-8, +-12 -
       and the tie at 12 - 2^-20 that drives idx to 384. */
    for (i = -4096; i <= 4096; i++) {
        chk_sigx(f2u(4.0f) + (uint32_t)i);
        chk_sigx(f2u(8.0f) + (uint32_t)i);
        chk_sigx(f2u(12.0f) + (uint32_t)i);
        chk_sigx(0x80000000u + f2u(4.0f) + (uint32_t)i);
        chk_sigx(0x80000000u + f2u(8.0f) + (uint32_t)i);
        chk_sigx(0x80000000u + f2u(12.0f) + (uint32_t)i);
        chk_sigx((uint32_t)i);
        chk_sigx(0x80000000u + (uint32_t)i);
    }
    chk_sigx(f2u(0.0f)); chk_sigx(f2u(-0.0f));
    chk_sigx(0x7F800000u); chk_sigx(0xFF800000u);
    chk_sigx(0x7FC00000u); chk_sigx(0x7F800001u); chk_sigx(0xFFC00000u);
    chk_sigx(1); chk_sigx(0x807FFFFFu);
    phase("binade walls, zero, subnormal, inf, nan");

    for (i = 0; i < dv(6000000L); i++) {
        float x = ((float)(int32_t)rnd() / 2147483648.0f) * 13.0f;
        chk_sigx(f2u(x));
        chk_sigx(rnd());
    }
    phase("uniform in range + random patterns");
}

/* ---- norm_ss_fixed ----------------------------------------------------
 * Driven through the PRODUCT's entry point: the amax scan, the
 * precondition band and the element loop are one contract and gating
 * the loop alone would not see a precondition that is wrong.
 *
 * The Q15 image is compared over the WHOLE buffer, canary included, so
 * a loop that walks off the end of qout is a mismatch and not a silent
 * pass. The buffers are twice the longest row for the same reason: a
 * mutation that doubles the store stride has to reach the comparison,
 * not fault before it.
 */
#define NS_MAXN 4096
#define NS_BUF  (2 * NS_MAXN + 16)
static float ns_x[NS_MAXN];
static short ns_q1[NS_BUF], ns_q2[NS_BUF];

static void chk_norm(int n, int want_q) {
    uint32_t a, b;
    int e1 = -12345, e2 = -12345;
    float r1, r2;
    int i;
    for (i = 0; i < NS_BUF; i++) { ns_q1[i] = 0x5A5A; ns_q2[i] = 0x5A5A; }
    g_kernel = LZ_KERNEL_ARM;
    r1 = norm_ss_fixed(ns_x, n, want_q ? ns_q1 : (short *)0, &e1);
    g_kernel = LZ_KERNEL_ARM_ASM;
    r2 = norm_ss_fixed(ns_x, n, want_q ? ns_q2 : (short *)0, &e2);
    g_n++;
    a = f2u(r1); b = f2u(r2);
    mix(b); mix((uint32_t)e2);
    if (a != b || e1 != e2) {
        if (g_bad < 10)
            printf("MISMATCH normss n=%d q=%d ss c=0x%08lX asm=0x%08lX "
                   "e c=%d asm=%d\n", n, want_q, (unsigned long)a,
                   (unsigned long)b, e1, e2);
        g_bad++;
        return;
    }
    if (!want_q) return;
    for (i = 0; i < NS_BUF; i++) {
        if (i < 8 || i == n - 1 || i == n) mix((uint32_t)(uint16_t)ns_q2[i]);
        if (ns_q1[i] != ns_q2[i]) {
            if (g_bad < 10)
                printf("MISMATCH normss n=%d qout[%d] c=%d asm=%d\n",
                       n, i, (int)ns_q1[i], (int)ns_q2[i]);
            g_bad++;
            return;
        }
    }
}

/* Row shapes. `kind` picks the distribution; every one of them is fed
 * to both arms, so a shape the assembly declines is a shape the C
 * covers - what the gate proves is that the two agree, and the mutation
 * runs are what prove the assembly ran at all. */
static void ns_fill(int n, int kind) {
    int i;
    float scale;
    for (i = 0; i < n; i++) ns_x[i] = 0.0f;
    switch (kind) {
    case 0:                                  /* activation-like */
        scale = (float)(rnd() >> 8) / 16777216.0f * 8.0f + 0.01f;
        for (i = 0; i < n; i++)
            ns_x[i] = ((float)(int32_t)rnd() / 2147483648.0f) * scale;
        break;
    case 1:                                  /* wide dynamic range */
        for (i = 0; i < n; i++) {
            int sh = (int)(rnd() % 60u) - 30;
            float v = ((float)(int32_t)rnd() / 2147483648.0f);
            ns_x[i] = v * u2f((uint32_t)(127 + sh) << 23);
        }
        break;
    case 2:                                  /* raw bit patterns */
        for (i = 0; i < n; i++) ns_x[i] = u2f(rnd());
        break;
    case 3:                                  /* subnormals and zeros */
        for (i = 0; i < n; i++)
            ns_x[i] = (rnd() & 1u) ? 0.0f : u2f(rnd() & 0x807FFFFFu);
        ns_x[rnd() % (unsigned)n] = ((float)(int32_t)rnd()
                                     / 2147483648.0f) * 3.0f;
        break;
    case 4:                                  /* the +-32768 clamp */
        for (i = 0; i < n; i++)
            ns_x[i] = ((float)(int32_t)rnd() / 2147483648.0f) * 2.0f;
        /* mantissa just under 2 makes amax*2^(14-expof) reach 32768,
           which is the only input the clamp fires on. */
        ns_x[rnd() % (unsigned)n] = u2f(0x407FFFFFu);
        ns_x[rnd() % (unsigned)n] = u2f(0xC07FFFFFu);
        break;
    case 5:                                  /* extreme amax: declines */
        for (i = 0; i < n; i++)
            ns_x[i] = ((float)(int32_t)rnd() / 2147483648.0f) * 1e-3f;
        ns_x[rnd() % (unsigned)n] = u2f(((rnd() & 1u) ? 0x7Fu : 0x0Au)
                                        << 23);
        break;
    default:                                 /* inf / nan / all zero */
        for (i = 0; i < n; i++)
            ns_x[i] = ((float)(int32_t)rnd() / 2147483648.0f);
        if (kind == 6) ns_x[rnd() % (unsigned)n] = u2f(0x7F800000u);
        if (kind == 7) ns_x[rnd() % (unsigned)n] = u2f(0xFFC00000u);
        if (kind == 8) for (i = 0; i < n; i++) ns_x[i] = 0.0f;
        break;
    }
}

static void sweep_norm(void) {
    long r;
    int n, kind, q;
    printf("norm_ss_fixed:\n");
    /* Every short row, every shape, both qout modes: the loop is a
       do-while, so n=1 is its degenerate case, and the qout branch is
       taken outside the loop. */
    for (n = 1; n <= 40; n++)
        for (kind = 0; kind <= 8; kind++)
            for (q = 0; q < 2; q++) { ns_fill(n, kind); chk_norm(n, q); }
    phase("n = 1..40, every row shape");

    /* The real geometry (512 / 1536 hidden, 4096 the declared cap) and
       the lengths either side of it. */
    for (r = 0; r < dv(200); r++) {
        static const int ns[] = { 63, 64, 65, 127, 128, 255, 256, 511,
                                  512, 513, 1535, 1536, 1537, 4095, 4096 };
        for (n = 0; n < (int)(sizeof(ns) / sizeof(ns[0])); n++)
            for (kind = 0; kind <= 8; kind++) {
                ns_fill(ns[n], kind);
                chk_norm(ns[n], (int)(rnd() & 1u));
            }
    }
    phase("shipping row lengths");

    for (r = 0; r < dv(4000); r++) {
        n = (int)(rnd() % (unsigned)NS_MAXN) + 1;
        kind = (int)(rnd() % 9u);
        ns_fill(n, kind);
        chk_norm(n, (int)(rnd() & 1u));
    }
    phase("random lengths and shapes");
}

/* ---- attention scoring / weighted sum --------------------------------
 * Driven through the PRODUCT's entry points, not the leaves: the leaves
 * are static in ops_gdn.c, and going through lz_attn_score_q8 /
 * lz_attn_wsum_q8 gates the tier wiring and the ring-buffer walk at the
 * same time. att for the weighted sum is whatever the scoring pass just
 * produced, so the coefficient distribution is the real one.
 */
#define AT_MAXT   700
#define AT_MAXKVD 256
static float   at_q[AT_MAXKVD], at_o1[AT_MAXKVD], at_o2[AT_MAXKVD];
static int8_t  at_c[AT_MAXT * AT_MAXKVD];
static float   at_s[AT_MAXT * (AT_MAXKVD / 32)];
static float   at_a1[AT_MAXT], at_a2[AT_MAXT], at_cb[AT_MAXT];
static int16_t at_cq[AT_MAXT];

static void chk_attn(int hd, int kvd, int T, int sink, int ring) {
    int i, bad0 = (int)g_bad;
    for (i = 0; i < kvd; i++)
        at_q[i] = ((float)(int32_t)rnd() / 2147483648.0f) * 3.0f;
    for (i = 0; i < T * kvd; i++)
        at_c[i] = (int8_t)((int)(rnd() % 255u) - 127);
    for (i = 0; i < T * (kvd / 32); i++)
        at_s[i] = ((float)(rnd() >> 8) / 16777216.0f) * 0.05f + 1e-4f;

    g_kernel = LZ_KERNEL_ARM;
    lz_attn_score_q8(at_a1, at_q, hd, at_c, at_s, kvd, T - 1, 0.125f,
                     sink, ring);
    g_kernel = LZ_KERNEL_ARM_ASM;
    lz_attn_score_q8(at_a2, at_q, hd, at_c, at_s, kvd, T - 1, 0.125f,
                     sink, ring);
    for (i = 0; i < T; i++) {
        g_n++;
        mix(f2u(at_a2[i]));
        if (f2u(at_a1[i]) != f2u(at_a2[i])) {
            if (g_bad < 10)
                printf("MISMATCH score hd=%d kvd=%d T=%d t=%d "
                       "c=0x%08lX asm=0x%08lX\n", hd, kvd, T, i,
                       (unsigned long)f2u(at_a1[i]),
                       (unsigned long)f2u(at_a2[i]));
            g_bad++;
        }
    }
    if ((int)g_bad != bad0) return;    /* att must agree before wsum means anything */

    g_kernel = LZ_KERNEL_ARM;
    lz_attn_wsum_q8(at_o1, at_a1, hd, at_c, at_s, kvd, T - 1,
                    at_cb, at_cq, sink, ring);
    g_kernel = LZ_KERNEL_ARM_ASM;
    lz_attn_wsum_q8(at_o2, at_a1, hd, at_c, at_s, kvd, T - 1,
                    at_cb, at_cq, sink, ring);
    for (i = 0; i < hd; i++) {
        g_n++;
        mix(f2u(at_o2[i]));
        if (f2u(at_o1[i]) != f2u(at_o2[i])) {
            if (g_bad < 10)
                printf("MISMATCH wsum hd=%d kvd=%d T=%d d=%d "
                       "c=0x%08lX asm=0x%08lX\n", hd, kvd, T, i,
                       (unsigned long)f2u(at_o1[i]),
                       (unsigned long)f2u(at_o2[i]));
            g_bad++;
        }
    }
}

static void sweep_attn(void) {
    long r;
    int T;
    printf("attention:\n");
    /* Every T from 1 to 40: the odd-leftover row and the two-row pair
       loop alternate here, and T=1 is the degenerate row. */
    for (T = 1; T <= 40; T++) chk_attn(64, 64, T, 0, 0);
    phase("T = 1..40, one group pair");

    /* Across LZ_WSUM_CHUNK (512): the second walk restarts its int32
       accumulator there and folds through lz_i32f, so a kernel that
       leaves a lane dirty only shows up on the far side. */
    for (T = 505; T <= 520; T++) chk_attn(64, 128, T, 0, 0);
    chk_attn(64, 128, 600, 0, 0);
    chk_attn(32, 32, 513, 0, 0);
    phase("T across the 512 chunk");

    /* Ringed cache: sink + ring makes the slot walk revisit rows, so
       the two tiers must agree on which row each t reads. */
    for (T = 1; T <= 40; T++) chk_attn(64, 64, T, 4, 16);
    chk_attn(128, 128, 600, 8, 256);
    phase("ringed slot walk");

    for (r = 0; r < dv(400); r++) {
        int hds[3];
        int h;
        hds[0] = 32; hds[1] = 64; hds[2] = 128;
        h = hds[rnd() % 3u];
        T = (int)(rnd() % 640u) + 1;
        chk_attn(h, h, T, 0, 0);
    }
    phase("random head widths and T");
}

int main(int argc, char **argv) {
    const char *what = (argc > 1) ? argv[1] : "all";
    /* xorshift32 has one fixed point and it is zero, so a seed of 0
       feeds every sweep a stream of zeros - which quantizes to an
       all-zero query, an all-zero dot and an all-zero att, and both
       arms then agree on nothing. Caught by the checksum above. */
    if (argc > 2) g_rs = (uint32_t)strtoul(argv[2], NULL, 0);
    if (!g_rs) g_rs = 0x9E3779B9u;
    if (argc > 3) g_div = strtol(argv[3], NULL, 0);
    lz_kernel_select(LZ_KERNEL_ARM_ASM);
    if (lz_kernel_name()[0] != 'a') {
        printf("FAIL: this build has no ARM tiers (kernel=%s)\n",
               lz_kernel_name());
        return 2;
    }
    if (!strcmp(what, "i32f") || !strcmp(what, "all")) sweep_i32f();
    if (!strcmp(what, "q8r")  || !strcmp(what, "all")) sweep_q8r();
    if (!strcmp(what, "exp")  || !strcmp(what, "all")) sweep_exp();
    if (!strcmp(what, "sig")  || !strcmp(what, "all")) sweep_sig();
    if (!strcmp(what, "norm") || !strcmp(what, "all")) sweep_norm();
    if (!strcmp(what, "attn") || !strcmp(what, "all")) sweep_attn();
    printf("%s: %lu mismatches\n", g_bad ? "FAIL" : "PASS",
           (unsigned long)g_bad);
    return g_bad ? 1 : 0;
}
