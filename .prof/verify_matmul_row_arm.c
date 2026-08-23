/* lz_matmul_row_arm_asm against the loop it replaces, whole rows.
 *
 * The unit under test is NOT the per-element MAC - .prof/verify_fmac_core.c
 * already sweeps that arithmetic. What is only testable here is the
 * kernel's own machinery: the group snapshot, the rollback on a decline,
 * and the resume index it hands back. So every case runs the FULL row
 * both ways and compares the eight accumulators bit for bit:
 *
 *   reference   a[k] = 0; for j in steps of 8: a[k] += w[j+k]*x[j+k]
 *   subject     a[k] = 0; j = kernel(...); then the same C loop from j
 *
 * Identical accumulators therefore mean the kernel either did the work
 * correctly or declined in a way the C finished correctly - and a
 * rollback that lost or double-counted a group cannot pass, which a
 * per-element gate could never see.
 *
 * TWO ROUNDINGS, NOT AN FMA. The reference multiplies into a volatile
 * before adding, because a contraction there would make the reference a
 * different function from the one the kernel implements - the same
 * hazard -ffp-contract=off exists for in the engine build.
 *
 * Rows:
 *   A  ALL NORMAL, every n from 0 to 40 plus a long row. The kernel
 *      must consume every whole group and leave the tail; n < 8 must
 *      return 0 rather than touching anything.
 *   B  ONE DECLINE TRIGGER, walked to every position of a 32-element
 *      row, one trigger kind at a time. This is the row that exercises
 *      snapshot and rollback: the trigger's group must come back
 *      untouched and the C must redo it.
 *   C  CANCELLATION rows, where a[k] passes through zero - the case the
 *      core's teq/bne guard exists for, reached through the kernel.
 *   D  RANDOM rows with a sprinkling of triggers.
 *
 * Build and run (WSL), from the repo root:
 *   arm-linux-gnueabi-gcc-14 -O2 -static -msoft-float -march=armv5te \
 *     -mtune=arm926ej-s -std=gnu99 -Isrc -o /tmp/vmr \
 *     .prof/verify_matmul_row_arm.c src/ops_arm.c
 *   qemu-arm-static -cpu arm926 /tmp/vmr
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ops_arm.h"

#if !defined(LZ_ARM_ASM_EXTERN)
int main(void) {
    printf("verify_matmul_row_arm: SKIP - not an ARM gcc build\n");
    return 0;
}
#else

#define MAXN 512
static long fails[4], cases[4], declines[4];

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

/* lz_matmul's own loop, verbatim in shape: eight accumulators, stride
   eight, product rounded before the add. */
static void ref_row(const float *w, const float *x, int n, float *a) {
    int j, k;
    for (k = 0; k < 8; k++) a[k] = 0.0f;
    for (j = 0; j + 7 < n; j += 8)
        for (k = 0; k < 8; k++) {
            volatile float p = w[j + k] * x[j + k];
            a[k] = a[k] + p;
        }
}

static void sub_row(const float *w, const float *x, int n, float *a,
                    int *consumed) {
    int j, k;
    for (k = 0; k < 8; k++) a[k] = 0.0f;
    j = lz_matmul_row_arm_asm(w, x, n, a);
    *consumed = j;
    for (; j + 7 < n; j += 8)
        for (k = 0; k < 8; k++) {
            volatile float p = w[j + k] * x[j + k];
            a[k] = a[k] + p;
        }
}

static void run(int row, const float *w, const float *x, int n,
                const char *what) {
    float ra[8], sa[8];
    int consumed = 0, k;
    cases[row]++;
    ref_row(w, x, n, ra);
    sub_row(w, x, n, sa, &consumed);
    if (consumed < (n & ~7)) declines[row]++;
    if ((consumed & 7) != 0 || consumed > (n & ~7)) {
        if (fails[row] < 8)
            printf("  row %d %s n=%d: bad resume index %d\n",
                   row, what, n, consumed);
        fails[row]++;
        return;
    }
    for (k = 0; k < 8; k++)
        if (u_of(ra[k]) != u_of(sa[k])) {
            if (fails[row] < 8)
                printf("  row %d %s n=%d lane %d: got %08lx want %08lx "
                       "(consumed %d)\n", row, what, n, k,
                       (unsigned long)u_of(sa[k]),
                       (unsigned long)u_of(ra[k]), consumed);
            fails[row]++;
            return;
        }
}

static uint32_t rs = 0xC0FFEEu;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}
/* A normal float with a bounded exponent, so ordinary rows do not spend
   themselves on the product-overflow decline. */
static float rnorm(void) {
    uint32_t r = rnd();
    uint32_t e = 100u + (r >> 25) % 55u;
    return f_of((r & 0x807FFFFFu) | (e << 23));
}

#define NTRIG 5
static const uint32_t TRIG[NTRIG] = {
    0x00000000u,   /* zero */
    0x00400000u,   /* subnormal */
    0x7F800000u,   /* inf */
    0x7FC00000u,   /* nan */
    0x7F000000u    /* huge: the product overflows */
};
static const char *TRIGN[NTRIG] = { "zero", "subnormal", "inf", "nan",
                                    "overflow" };

int main(void) {
    static float w[MAXN], x[MAXN];
    int i, n, p, t;
    long r;

    /* A: all-normal rows, every short n plus a long one. */
    for (i = 0; i < MAXN; i++) { w[i] = rnorm(); x[i] = rnorm(); }
    for (n = 0; n <= 40; n++) run(0, w, x, n, "normal");
    run(0, w, x, 256, "normal");
    run(0, w, x, MAXN, "normal");

    /* B: one trigger, walked to every position. The group holding it
       must come back untouched. */
    for (t = 0; t < NTRIG; t++)
        for (p = 0; p < 32; p++) {
            float sw, sx;
            for (i = 0; i < 32; i++) { w[i] = rnorm(); x[i] = rnorm(); }
            sw = w[p]; sx = x[p];
            w[p] = f_of(TRIG[t]);
            run(1, w, x, 32, TRIGN[t]);
            w[p] = sw;
            x[p] = f_of(TRIG[t]);
            run(1, w, x, 32, TRIGN[t]);
            x[p] = sx;
        }

    /* C: rows built so each accumulator passes through exact zero -
       every eighth element cancels the running sum exactly. */
    for (n = 8; n <= 128; n += 8) {
        for (i = 0; i < n; i++) { w[i] = rnorm(); x[i] = rnorm(); }
        for (i = 0; i + 15 < n; i += 16) {
            int k;
            for (k = 0; k < 8; k++) {
                volatile float pr = w[i + k] * x[i + k];
                w[i + 8 + k] = -pr;
                x[i + 8 + k] = 1.0f;
            }
        }
        run(2, w, x, n, "cancel");
    }

    /* D: random rows, triggers sprinkled at about one element in forty. */
    for (r = 0; r < 20000L; r++) {
        n = 8 + (int)(rnd() % (MAXN - 8));
        for (i = 0; i < n; i++) {
            w[i] = rnorm();
            x[i] = rnorm();
            if ((rnd() % 40u) == 0u) w[i] = f_of(TRIG[rnd() % NTRIG]);
            if ((rnd() % 40u) == 0u) x[i] = f_of(TRIG[rnd() % NTRIG]);
        }
        run(3, w, x, n, "random");
    }

    {
        static const char *nm[4] = { "A NORMAL", "B TRIGGER", "C CANCEL",
                                     "D RANDOM" };
        int bad = 0;
        for (i = 0; i < 4; i++) {
            printf("%-10s %8ld rows  %8ld with a decline  %ld mismatches  %s\n",
                   nm[i], cases[i], declines[i], fails[i],
                   fails[i] ? "FAIL" : "ok");
            if (fails[i]) bad = 1;
        }
        printf("VERIFY: %s\n", bad ? "FAIL" : "PASS");
        return bad;
    }
}
#endif
