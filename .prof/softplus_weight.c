/* What lz_softplus actually costs on a machine with no FPU.
 *
 * Its own comment prices the x87 control-word save/restore at ~100
 * cycles and concludes "not worth writing our own log". That reasoning
 * is about a machine that HAS an FPU. On a soft-float target the
 * control word does not exist, the #else branch runs, and the cost is
 * two libm DOUBLE transcendentals per call - which the f32 census
 * cannot see at all (it counts neither libm nor doubles).
 *
 * Four arms, same input stream, so only the ratio is read - QEMU does
 * not model cycles. Arm C is the shape a fixed-point softplus would
 * have to beat: the engine's own lz_exp table lookup, which is what
 * lz_exp_fixed is built on.
 *
 * THE NUMBER THAT CARRIES is the instruction count, not the seconds;
 * the seconds are only usable as a ratio because TCG's cost per guest
 * instruction is roughly flat, and the two agree here to 2%. Measured,
 * qemu 8.2.2 -cpu arm926, per call, harness subtracted:
 *
 *   arm                        insns/call    ratio      wall (12.8M calls)
 *   1 libm double (was shipped)      3604     1.00x      38.582 s
 *   2 libm float                     2130     1.69x      25.939 s
 *   3 table+poly SHAPE PROBE          953     3.78x      10.214 s
 *   4 lz_softplus_poly, SHIPPED      1389     2.59x      15.218 s
 *
 * so the real thing is 2.59x, not the shape probe's 3.78x, and the
 * difference between those two IS the accuracy: arm 3's acc comes out
 * 5.85e7 against every other arm's 5.82e7. Per call the shipped body
 * saves 2215 instructions, and the live GDN call site runs 144 of them
 * a token. The per-call figures include the harness's own accumulate
 * and the indirect call, identical in every arm, so the ratios are if
 * anything understated.
 *
 *   build/run (WSL):
 *     arm-linux-gnueabi-gcc-14 -O2 -static -msoft-float \
 *         -o /tmp/spw .prof/softplus_weight.c -lm
 *     qemu-arm-static -d nochain -D /tmp/spw.log /tmp/spw
 *   or count instructions with:
 *     qemu-arm-static -one-insn-per-tb -d in_asm ... | wc -l
 * Simplest portable count is the plugin-free approach: run each arm in
 * its own process and compare wall instructions via -d nochain traces,
 * or just time them - the ratio is what matters and it is enormous.
 *
 * ARM 4, -DSPW_REAL, is the SHIPPED implementation, added when it was
 * written: src/ops_quant.c compiled in, not transcribed, so this file
 * prices lz_softplus_poly and not a copy of it. Arm 3 stays as the
 * shape probe it always was - deliberately inaccurate, and the reason
 * arm 4 costs 1.46x more than it is that arm 4 is correct (a degree-8
 * minimax log1p and the engine's own exp, against arm 3's 3-term
 * polynomials). Build it with the engine's own ARM flags:
 *
 *     arm-linux-gnueabi-gcc-14 -O2 -static -msoft-float -DSPW_REAL \
 *       -march=armv5te -mtune=arm926ej-s -DLZ_NO_FPU=1 \
 *       -ffp-contract=off -Isrc -o /tmp/spwr .prof/softplus_weight.c \
 *       src/ops_sched.c src/err.c src/compat.c src/cpucheck.c \
 *       src/ops_arm.c -lm
 *     qemu-arm-static -cpu arm926 /tmp/spwr
 *
 * ops_arm.c is in that list and not in the x86 verifiers' because
 * ops_quant.c's norm_ss_fixed reaches an ARM primitive that only exists
 * on this target.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef SPW_REAL
#include "ops_quant.h"
#include "ops_quant.c"
#endif

#define N 200000

static float in[64];

/* Arm A: the shipping no-FPU path, verbatim from ops.c's #else. */
static float sp_libm(float x) {
    if (x > 20.0f) return x;
    return (float)log1p(exp((double)x));
}

/* Arm B: single-precision libm, the cheapest drop-in that keeps the
   same shape. Not equivalent - here to price the double-ness alone. */
static float sp_libmf(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

/* Arm C: a table+Taylor exp of the kind lz_exp_fixed already uses,
   composed the way a fixed-point softplus would be. Deliberately NOT
   accurate - it exists to price the SHAPE, not to be a candidate. */
static float sp_tablish(float x) {
    int i;
    float y, f;
    if (x > 20.0f) return x;
    y = x * 1.4426950408889634f;          /* log2(e) */
    i = (int)(y < 0.0f ? y - 0.5f : y + 0.5f);
    f = y - (float)i;
    /* 2^f by a 3-term poly, then ldexp by exponent-field add */
    y = 1.0f + f * (0.6931472f + f * (0.2402265f + f * 0.0555041f));
    {
        union { float f; unsigned u; } b;
        b.f = y;
        b.u += (unsigned)i << 23;
        y = b.f;
    }
    /* log1p(y) by the same shape: strip the exponent, poly the mantissa */
    {
        union { float f; unsigned u; } b;
        int e2;
        float m;
        y += 1.0f;
        b.f = y;
        e2 = (int)((b.u >> 23) & 0xFF) - 127;
        b.u = (b.u & 0x807FFFFFu) | (127u << 23);
        m = b.f - 1.0f;
        y = (float)e2 * 0.6931472f
          + m * (1.0f + m * (-0.5f + m * 0.3333333f));
    }
    return y;
}

static int g_rounds = N;

static double run(float (*fn)(float), const char *name) {
    clock_t t0, t1;
    volatile float acc = 0.0f;
    int r, k;
    t0 = clock();
    for (r = 0; r < g_rounds; r++)
        for (k = 0; k < 64; k++) acc += fn(in[k]);
    t1 = clock();
    printf("%-10s %10.3f s   acc=%g\n", name,
           (double)(t1 - t0) / CLOCKS_PER_SEC, (double)acc);
    return (double)(t1 - t0) / CLOCKS_PER_SEC;
}

/* argv: [rounds [arm]]. Both exist for the INSTRUCTION COUNT, which is
   the number that carries across machines - a QEMU wall clock does not,
   and the ratios above are only usable because TCG's cost per guest
   instruction is roughly flat. This build has no TCG plugin, so the
   count comes from a one-instruction-per-TB execution trace, which
   needs one arm at a time and very few rounds:
     for a in 0 1 2 3 4; do
       qemu-arm-static -one-insn-per-tb -d exec -D /tmp/i$a /tmp/spwr 1 $a
       grep -c '^Trace' /tmp/i$a
     done
   arm 0 is the empty harness, so (count(a) - count(0)) / 64 is arm a's
   instructions per call. */
int main(int argc, char **argv) {
    int k, arm = -1;
    double a = 0.0, b = 0.0, c = 0.0;
    /* The argument stream, unchanged since the first run so the arms
       stay comparable across commits. It sits inside the range the live
       GDN call site actually presents, measured -15.10..11.29, and like
       that range it never reaches the shortcut. */
    for (k = 0; k < 64; k++) in[k] = -8.0f + (float)k * 0.35f;
    if (argc > 1) g_rounds = atoi(argv[1]);
    if (argc > 2) arm = atoi(argv[2]);
    if (arm == 0) return 0;                       /* the empty harness */
    if (arm < 0 || arm == 1) a = run(sp_libm,    "libm-dbl");
    if (arm < 0 || arm == 2) b = run(sp_libmf,   "libm-flt");
    if (arm < 0 || arm == 3) c = run(sp_tablish, "table");
#ifdef SPW_REAL
    {
        double d = 0.0;
        if (arm < 0 || arm == 4) d = run(lz_softplus_poly, "shipped");
        if (arm < 0) {
            printf("\ncalls per arm: %d\n", g_rounds * 64);
            if (c > 0.0) printf("libm-dbl / table   = %.2fx\n", a / c);
            if (b > 0.0) printf("libm-dbl / libm-flt = %.2fx\n", a / b);
            if (d > 0.0) printf("libm-dbl / shipped = %.2fx\n", a / d);
        }
        return 0;
    }
#else
    if (arm < 0) {
        printf("\ncalls per arm: %d\n", g_rounds * 64);
        if (c > 0.0) printf("libm-dbl / table = %.2fx\n", a / c);
        if (b > 0.0) printf("libm-dbl / libm-flt = %.2fx\n", a / b);
    }
    return 0;
#endif
}
