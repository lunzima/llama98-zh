/* What lz_softplus actually costs on a machine with no FPU.
 *
 * Its own comment prices the x87 control-word save/restore at ~100
 * cycles and concludes "not worth writing our own log". That reasoning
 * is about a machine that HAS an FPU. On a soft-float target the
 * control word does not exist, the #else branch runs, and the cost is
 * two libm DOUBLE transcendentals per call - which the f32 census
 * cannot see at all (it counts neither libm nor doubles).
 *
 * Three arms, same input stream, so only the ratio is read - QEMU does
 * not model cycles. Arm C is the shape a fixed-point softplus would
 * have to beat: the engine's own lz_exp table lookup, which is what
 * lz_exp_fixed is built on.
 *
 *   build/run (WSL):
 *     arm-linux-gnueabi-gcc-14 -O2 -static -msoft-float \
 *         -o /tmp/spw .prof/softplus_weight.c -lm
 *     qemu-arm-static -d nochain -D /tmp/spw.log /tmp/spw
 *   or count instructions with:
 *     qemu-arm-static -one-insn-per-tb -d in_asm ... | wc -l
 * Simplest portable count is the plugin-free approach: run each arm in
 * its own process and compare wall instructions via -d nochain traces,
 * or just time them - the ratio is what matters and it is enormous. */
#include <stdio.h>
#include <math.h>
#include <time.h>

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

static double run(float (*fn)(float), const char *name) {
    clock_t t0, t1;
    volatile float acc = 0.0f;
    int r, k;
    t0 = clock();
    for (r = 0; r < N; r++)
        for (k = 0; k < 64; k++) acc += fn(in[k]);
    t1 = clock();
    printf("%-10s %10.3f s   acc=%g\n", name,
           (double)(t1 - t0) / CLOCKS_PER_SEC, (double)acc);
    return (double)(t1 - t0) / CLOCKS_PER_SEC;
}

int main(void) {
    int k;
    double a, b, c;
    /* The real argument range: a + dt_bias, mostly small and negative,
       with a tail that the x > 20 shortcut catches. */
    for (k = 0; k < 64; k++) in[k] = -8.0f + (float)k * 0.35f;
    a = run(sp_libm,    "libm-dbl");
    b = run(sp_libmf,   "libm-flt");
    c = run(sp_tablish, "table");
    printf("\ncalls per arm: %d\n", N * 64);
    if (c > 0.0) printf("libm-dbl / table = %.2fx\n", a / c);
    if (b > 0.0) printf("libm-dbl / libm-flt = %.2fx\n", a / b);
    return 0;
}
