/* What a float DIVIDE costs on ARMv5TE, and what the two standard ways
 * out of one cost instead.
 *
 * The f32 census counts a divide as ONE operation, the same as a
 * multiply. On a soft-float target that is not close to true, and every
 * "reduce the divides" decision in this tree has so far been made
 * without the number - which is the shape M5 already retired once.
 *
 * ARMS
 *   0  control: the loop, the loads, the sink, no float operation
 *   1  one multiply                      (__aeabi_fmul)
 *   2  one divide                        (__aeabi_fdiv)
 *   3  RECIPROCAL HOISTED: one divide outside the loop, one multiply in
 *      - the shape that replaces N divides with 1 divide and N
 *      multiplies. Measured as the per-iteration cost of the multiply,
 *      so 3 against 2 is the saving per element and 3 against 1 is what
 *      the hoist leaves behind.
 *   4  divide by a POWER OF TWO, written as a divide - the case a
 *      compiler may or may not strength-reduce, and this tree has a
 *      rule about (ops.h rule 6 item 1: Watcom reduces it, gcc does
 *      not, and they land 1 ULP apart, so the multiply is written out).
 *      Here to price what that rule buys.
 *
 * The divisor is volatile in arms 2 and 3 so neither folds; arm 4's is a
 * literal, which is the whole point of that arm.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef DIV_MODE
#define DIV_MODE 0
#endif

#define N 64

static volatile float g_x[N];
static volatile float g_d = 3.7f;
volatile int32_t g_sink;

int main(int argc, char **argv) {
    long n = (argc > 1) ? atol(argv[1]) : 256;
    union { float f; int32_t i; } u;
    long i;
    int k;
#if DIV_MODE == 3
    float rcp;
#endif

    for (k = 0; k < N; k++) g_x[k] = 1.0f + (float)k * 0.37f;
#if DIV_MODE == 3
    rcp = 1.0f / g_d;          /* once, outside - that is the point */
#endif

    for (i = 0; i < n; i++) {
        int j = (int)(i & (N - 1));
#if DIV_MODE == 1
        u.f = g_x[j] * g_d;                g_sink = u.i;
#elif DIV_MODE == 2
        u.f = g_x[j] / g_d;                g_sink = u.i;
#elif DIV_MODE == 3
        u.f = g_x[j] * rcp;                g_sink = u.i;
#elif DIV_MODE == 4
        u.f = g_x[j] / 256.0f;             g_sink = u.i;
#else
        u.f = g_x[j];                      g_sink = u.i;
#endif
    }
    printf("%ld\n", (long)g_sink);
    return 0;
}
