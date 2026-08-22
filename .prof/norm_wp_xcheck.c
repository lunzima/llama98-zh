/* Cross-check for lz_rmsnorm's cached (1 + w) table, src/ops_norm.c.
 *
 * THE CLAIM. The scalar output loop reads a table of fl(1.0f + w[i])
 * built once per weight pointer instead of adding 1 per element per
 * token. That is bit-identical, not close: the table holds the same
 * value the inline expression computes and float addition is
 * deterministic. On a full pool build_wp returns NULL and the inline
 * form runs, so ONE BINARY CONTAINS BOTH ARMS.
 *
 * HOW BOTH ARMS RUN IN ONE PROCESS, without a build flag: the pool is a
 * bump allocator with LZ_WP_SLOTS entries and no eviction, so feeding it
 * more distinct weight POINTERS than it has slots makes the first
 * NW_HIT rows take the table and the rest take the inline loop. Every
 * row here holds IDENTICAL weight data, so all NW outputs must be
 * identical to each other; a difference is the two arms disagreeing.
 *
 * WHY THE MUTATION MATTERS MORE THAN THE PASS. A green run is also what
 * a build where build_wp is never called would produce. Break the table
 * (norm_wp_xcheck.sh drops the `+ 1.0f`) and the count of differing rows
 * becomes the number of rows that took the INLINE path - so the
 * mutation does not only prove the check is sensitive, it MEASURES the
 * split and shows the cached arm was really used.
 *
 * --kernel ref, deliberately: LZ_NORMOUT_TAB's vector cells would take
 * the row before the scalar loop sees it on an x86 build with SSE, and
 * the table is for the scalar loop. ref is the tier the ARM build runs
 * here anyway - its own normout cells are all NULL.
 *
 * Build:
 *   gcc -O2 -std=c99 -I../src -o norm_wp_xcheck norm_wp_xcheck.c \
 *       ../src/ops.c ../src/ops_quant.c ../src/ops_sched.c \
 *       ../src/ops_norm.c ../src/ops_gdn.c ../src/ops_matmul.c \
 *       ../src/ops_t2_scalar.c ../src/fwht.c ../src/cpucheck.c \
 *       ../src/err.c -lm
 * Exit 0 = every row identical, 1 = not. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ops.h"

/* More rows than the pool has slots, so the tail is forced inline. The
   count is deliberately NOT read from LZ_WP_SLOTS: this file must not
   need a private view into the thing it is checking, and the mutation
   run reports the actual split anyway. */
#define NW 80
#define NN 512

static float w_rows[NW][NN];
static float xv[NN];
static float out[NW][NN];
/* A SECOND weight set for the Q15 arm, and the reason is a fixture bug
   this file had on its first run: w_rows carries a 1e7 outlier - good
   for the float table, where it is exactly where the add's rounding is
   not the identity - and that outlier sets the Q15 scale
   (14 - expof(amax)) for the WHOLE row, at which 1+w and w both
   quantize to 0. The mutation went green against a table it had really
   broken. The Q15 arm therefore gets weights inside a normal range,
   where one quantization step is visible. */
static float wq_rows[NW][NN];
static int32_t iout[NW][NN];
static float ideq[NW];

int main(void) {
    uint32_t st = 0x9E3779B9u;
    int r, i, bad = 0, first = -1;
    lz_kernel_select(LZ_KERNEL_REF);
    for (i = 0; i < NN; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        xv[i] = ((float)(int32_t)st / 2147483648.0f) * 4.0f;
    }
    /* A weight row near 1 with a few values far from it: w = -1 makes
       the table entry exactly 0, and a large |w| is where the add's
       rounding is not the identity. Both are the entries a table that
       silently held w instead of 1+w would still look plausible on. */
    for (i = 0; i < NN; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        w_rows[0][i] = ((float)(int32_t)st / 2147483648.0f) * 0.5f;
    }
    w_rows[0][0] = -1.0f;
    w_rows[0][1] = 1e7f;
    w_rows[0][2] = -1e-9f;
    w_rows[0][3] = 0.0f;
    for (i = 0; i < NN; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        wq_rows[0][i] = ((float)(int32_t)st / 2147483648.0f) * 0.5f;
    }
    wq_rows[0][0] = -1.0f;      /* 1+w == 0, the entry `w` alone is -1 at */
    wq_rows[0][1] = 0.5f;
    for (r = 1; r < NW; r++) {
        memcpy(w_rows[r], w_rows[0], sizeof(w_rows[0]));
        memcpy(wq_rows[r], wq_rows[0], sizeof(wq_rows[0]));
    }

    for (r = 0; r < NW; r++)
        lz_rmsnorm(out[r], xv, w_rows[r], NN, 1e-6f);

    for (r = 1; r < NW; r++)
        if (memcmp(out[r], out[0], sizeof(out[0])) != 0) {
            if (first < 0) first = r;
            bad++;
        }
    printf("norm (1+w) table: %d rows, %d differ from row 0", NW, bad);
    if (first >= 0) printf(", first at row %d", first);
    printf("\n");
    if (bad) {
        for (i = 0; i < NN; i++)
            if (out[first][i] != out[0][i]) {
                printf("  row %d elem %d: %.9g vs %.9g (w=%.9g)\n",
                       first, i, (double)out[first][i], (double)out[0][i],
                       (double)w_rows[0][i]);
                break;
            }
        printf("FAIL\n");
        return 1;
    }

    /* The SAME check for the Q15 table (build_qw, feeding
       lz_rmsnorm_out_fixed_int). That path is not on any call chain this
       model reaches - every site that would take it is inside a SubLN
       branch - so it has no end-to-end gate at all, and the cache it
       carries is the same pointer-keyed shape with the same
       cached-vs-inline split. Checking it here costs twenty lines and is
       the difference between "unused" and "unchecked".

       Needs the fixed tier: without it lz_rmsnorm_int's float fallback
       skips the Q15 output loop entirely and this would compare that
       fallback to itself. */
    lz_norm_select("fixed");
    for (r = 0; r < NW; r++)
        lz_rmsnorm_int(iout[r], xv, wq_rows[r], NN, 1e-6f, &ideq[r]);
    first = -1;
    for (r = 1; r < NW; r++)
        if (memcmp(iout[r], iout[0], sizeof(iout[0])) != 0 ||
            ideq[r] != ideq[0]) {
            if (first < 0) first = r;
            bad++;
        }
    printf("norm Q15 (1+w) table: %d rows, %d differ from row 0", NW, bad);
    if (first >= 0) printf(", first at row %d", first);
    printf("  [deq=%.9g t0=%ld]\n", (double)ideq[0], (long)iout[0][0]);
    /* deq == 0 is lz_rmsnorm_out_fixed_int's LZ_NORM_FIXED=0 stub, or a
       norm_ss_fixed that declined: either way the Q15 loop did not run
       and the rows above are identical for a reason that has nothing to
       do with the table. */
    if (ideq[0] == 0.0f) {
        printf("FAIL: the Q15 output loop did not run (deq is 0) - the "
               "rows above compare nothing\n");
        return 1;
    }
    if (bad) { printf("FAIL\n"); return 1; }

    printf("PASS\n");
    return 0;
}
