/* Cross-check for lz_rmsnorm's cached (1 + w) table, src/ops_norm.c.
 *
 * THE CLAIM, in two halves.
 *
 * (1) THE ARMS AGREE. The scalar output loop reads a table of fl(1.0f +
 * w[i]) built once per weight ROW instead of adding 1 per element per
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
 * (2) THE KEY IS THE ROW, NOT THE ADDRESS. This is the half a fixture
 * with one address per tensor cannot see. lz_t_f32 returns t->f for an
 * F32 tensor but widens every other format into the caller's scratch,
 * so a checkpoint whose norm weights are stored narrow presents ONE
 * address for all of them; under a key on the address the second tensor
 * of a given length is handed the first one's table. The reuse fixtures
 * below re-present a single address whose row differs ONLY IN THE
 * MIDDLE, and require the result to match the same row put through a
 * fresh address. An address key and an endpoint-sample key are both
 * served the stale table there; only the full row compare is not.
 *
 * WHY THE MUTATIONS MATTER MORE THAN THE PASS. A green run is also what
 * a build where the table is never consulted would produce. The fill
 * mutation (norm_wp_xcheck.sh drops the `+ 1.0f`) makes the cached rows
 * compute x*inv*w, so the count of differing rows becomes the number of
 * rows that took the INLINE path - it does not only prove the check is
 * sensitive, it MEASURES the split. The key mutations (an endpoint
 * sample in place of the full compare) must turn the reuse fixture red,
 * which is what says that fixture was measuring the key.
 *
 * --kernel ref, deliberately: LZ_NORMOUT_TAB's vector cells would take
 * the row before the scalar loop sees it on an x86 build with SSE, and
 * the table is for the scalar loop. ref is the tier the ARM build runs
 * here anyway - its own normout cells are all NULL.
 *
 * TWO Q15 CONSUMERS, TWO INVOCATIONS. build_qw serves two tables now -
 * 1+w for lz_rmsnorm_out_fixed_int and plain w for the gated loop
 * (lz_rmsnorm_gated_int, which is the one that actually runs on these
 * models). They share one bump-allocated pool with no eviction, so
 * whichever arm goes first takes every slot and the second one gets no
 * cached rows at all - a check that compares one arm to itself and
 * cannot fail. So the arm is chosen by argv[1] and the driver runs the
 * binary twice:
 *     norm_wp_xcheck            wp table + the 1+w Q15 table
 *     norm_wp_xcheck gated      wp table + the plain-w Q15 table
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
#include "model.h"   /* LZTensor: the alias fixture builds real ones */

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
/* The gated arm's own gate vector. Held constant across rows for the
   same reason the weights are: the only thing allowed to differ between
   rows is which build_qw arm served them. */
static float gv[NN];

/* The reuse fixture, one pair per pool. `reuse` is a row whose ADDRESS
   already carries a table - it is w_rows[0] / wq_rows[0], filled first
   above - and `fresh` is an address no call has used, so it cannot hit
   anything and its result is the inline arm by construction. Both are
   given the same contents, so any difference between the two outputs is
   the pool having served the reuse address a table built from the row
   that was there BEFORE. The perturbation is confined to the middle
   third: element 0 and n-1 are left equal to the original row, which is
   what makes an endpoint sample indistinguishable from the full compare
   except by this fixture. */
static float w_reuse[NN], w_fresh[NN];
static float out_reuse[NN], out_fresh[NN];
static float wq_reuse[NN], wq_fresh[NN];
static int32_t iq_reuse[NN], iq_fresh[NN];
static float dq_reuse, dq_fresh;

/* The scratch the ENGINE hands out. lz_t_f32 returns t->f only for
   LZ_FMT_F32; every other format is widened or dequantized INTO the
   caller's scratch and that pointer is what comes back. Two tensors of the
   same length therefore arrive at lz_rmsnorm as ONE address, which is the
   case a fixture with one pointer per row cannot produce. TWO FORMATS,
   because there are two ways to reach the same `return scratch` and one
   would leave the other unmeasured: BF16 is storage and widens with a
   shift (no scale), Q8_0 is a real quantization and dequantizes with a
   per-group scale. */
static float g_sc1[NN], g_sc2[NN];
static int8_t g_bf_a[NN * 2], g_bf_b[NN * 2];
static int8_t g_q8_a[NN], g_q8_b[NN];
static float g_q8sa[NN / 32];
static float alias_a[NN], alias_b[NN], alias_ref[NN];

static void bf16_put(int8_t *p, int i, float v) {
    union { float f; uint32_t u; } bit;
    bit.f = v;
    p[i * 2]     = (int8_t)((bit.u >> 16) & 0xFFu);
    p[i * 2 + 1] = (int8_t)((bit.u >> 24) & 0xFFu);
}

/* 1 if the second tensor's output is not what its own row gives. The
   reference is that same row through a SECOND scratch: a different address
   cannot hit the first one's slot, so its output is the inline arm by
   construction and nothing here recomputes the norm. */
static int alias_case(const LZTensor *ta, const LZTensor *tb) {
    lz_t_f32(ta, g_sc1);
    lz_rmsnorm(alias_a, xv, g_sc1, NN, 1e-6f);
    lz_t_f32(tb, g_sc1);                 /* same scratch, a different row */
    lz_rmsnorm(alias_b, xv, g_sc1, NN, 1e-6f);
    lz_t_f32(tb, g_sc2);                 /* same row, another scratch */
    lz_rmsnorm(alias_ref, xv, g_sc2, NN, 1e-6f);
    return memcmp(alias_b, alias_ref, sizeof alias_b) != 0;
}

extern long long lz_debug_norm_qw;

int main(int argc, char **argv) {
    uint32_t st = 0x9E3779B9u;
    int gated = (argc > 1 && argv[1][0] == 'g');
    int r, i, bad = 0, bad_key = 0, first = -1;
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
    for (i = 0; i < NN; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        gv[i] = ((float)(int32_t)st / 2147483648.0f) * 6.0f;
    }
    for (r = 1; r < NW; r++) {
        memcpy(w_rows[r], w_rows[0], sizeof(w_rows[0]));
        memcpy(wq_rows[r], wq_rows[0], sizeof(wq_rows[0]));
    }

    /* The engine's OWN borrow, FIRST - before anything fills the pool.
       The first call below has to FILL a slot for g_sc1: if the pool is
       already full the lookup misses, both calls take the inline arm, and
       the fixture is green for a reason that has nothing to do with the
       key. That is the whole reason this block sits above the 80 rows. */
    {
        LZTensor ta, tb;
        int i2, bad_alias = 0;

        memset(&ta, 0, sizeof ta);
        memset(&tb, 0, sizeof tb);
        ta.dtype = LZ_FMT_BF16;
        ta.n = NN;
        ta.gs = 0;
        ta.f = NULL;
        ta.q = g_bf_a;
        ta.scale = NULL;
        tb = ta;
        tb.q = g_bf_b;
        /* b agrees with a at both ENDS and differs in between, so a key
           that samples the row - at either end, or anywhere - is served
           the wrong table here too. A fixture the weaker key also passes
           would be evidence about nothing. */
        for (i2 = 0; i2 < NN; i2++) {
            int edge = (i2 == 0 || i2 == NN - 1);
            bf16_put(g_bf_a, i2, w_rows[0][i2]);
            bf16_put(g_bf_b, i2, edge ? w_rows[0][i2]
                                      : w_rows[0][i2] * 0.5f + 0.25f);
        }
        if (alias_case(&ta, &tb)) bad_alias = 1;
        printf("norm key: BF16 borrow through one scratch -> %s\n",
               bad_alias ? "DIFFER" : "same");
        bad_key |= bad_alias;

        bad_alias = 0;
        memset(&ta, 0, sizeof ta);
        ta.dtype = LZ_FMT_Q8_0;
        ta.n = NN;
        ta.gs = 32;
        ta.f = NULL;
        ta.q = g_q8_a;
        ta.scale = g_q8sa;
        tb = ta;
        tb.q = g_q8_b;            /* ONE scale array: the endpoints of a
                                     dequantized row are code*scale, and a
                                     separate scale would move them. */
        for (i2 = 0; i2 < NN; i2++) {
            int edge = (i2 == 0 || i2 == NN - 1);
            g_q8_a[i2] = (int8_t)((i2 * 37) % 255 - 127);
            g_q8_b[i2] = edge ? g_q8_a[i2]
                              : (int8_t)((i2 * 11) % 255 - 127) / 2;
        }
        for (i2 = 0; i2 < NN / 32; i2++)
            g_q8sa[i2] = 0.00390625f * (float)(i2 + 1);
        if (alias_case(&ta, &tb)) bad_alias = 1;
        printf("norm key: Q8_0 borrow through one scratch -> %s\n",
               bad_alias ? "DIFFER" : "same");
        bad_key |= bad_alias;
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

    /* (2) the key, float table. w_rows[0] carries a table built from the
       row that was in it a moment ago; that ONE address is re-presented
       with a different row, and the same row is also put through an
       address no call has used. The fresh address cannot hit anything,
       so its output is the inline arm by construction - which is the
       reference, not a second guess at the arithmetic. */
    for (i = 0; i < NN; i++) w_fresh[i] = w_rows[0][i];
    for (i = NN / 4; i < NN / 2; i++) w_fresh[i] += 4.0f;
    memcpy(w_reuse, w_fresh, sizeof w_reuse);
    memcpy(w_rows[0], w_reuse, sizeof w_reuse);
    lz_rmsnorm(out_reuse, xv, w_rows[0], NN, 1e-6f);
    lz_rmsnorm(out_fresh, xv, w_fresh, NN, 1e-6f);
    if (memcmp(out_reuse, out_fresh, sizeof out_reuse) != 0) {
        bad_key = 1;
        for (i = 0; i < NN; i++)
            if (out_reuse[i] != out_fresh[i]) {
                printf("  key: one address, two rows, elem %d: %.9g vs "
                       "%.9g\n", i, (double)out_reuse[i],
                       (double)out_fresh[i]);
                break;
            }
    }
    printf("norm (1+w) key: one address, two rows -> %s\n",
           bad_key ? "DIFFER" : "same");

    /* The SAME check for the Q15 table (build_qw, feeding
       lz_rmsnorm_out_fixed_int). That path is not on any call chain this
       model reaches - every site that would take it is inside a SubLN
       branch - so it has no end-to-end gate at all, and the cache it
       carries has the same row-keyed shape and the same cached-vs-inline
       split, so both halves of the fixture apply to it. Checking it here
       costs twenty lines and is the difference between "unused" and
       "unchecked".

       Needs the fixed tier: without it lz_rmsnorm_int's float fallback
       skips the Q15 output loop entirely and this would compare that
       fallback to itself.

       Through the ONE float axis. lz_norm_select was one of seven
       per-tier selectors and is gone; lz_norm_mode() derives from
       lz_float_ref_g now, and "all" is the fixed end of it. */
    lz_float_ref_select("all");
    if (gated)
        for (r = 0; r < NW; r++)
            lz_rmsnorm_gated_int(iout[r], xv, gv, wq_rows[r], NN, 1e-6f,
                                 &ideq[r]);
    else
        for (r = 0; r < NW; r++)
            lz_rmsnorm_int(iout[r], xv, wq_rows[r], NN, 1e-6f, &ideq[r]);
    first = -1;
    for (r = 1; r < NW; r++)
        if (memcmp(iout[r], iout[0], sizeof(iout[0])) != 0 ||
            ideq[r] != ideq[0]) {
            if (first < 0) first = r;
            bad++;
        }
    printf("norm Q15 %s table: %d rows, %d differ from row 0",
           gated ? "plain-w (gated)" : "(1+w)", NW, bad);
    if (first >= 0) printf(", first at row %d", first);
    printf("  [deq=%.9g t0=%ld]\n", (double)ideq[0], (long)iout[0][0]);
    /* deq == 0 is the LZ_NORM_FIXED=0 stub, or a norm_ss_fixed that
       declined: either way the Q15 loop did not run and the rows above
       are identical for a reason that has nothing to do with the table. */
    if (ideq[0] == 0.0f) {
        printf("FAIL: the Q15 output loop did not run (deq is 0) - the "
               "rows above compare nothing\n");
        return 1;
    }
    /* The gated arm has a positive control the (1+w) arm does not: with
       the pool already carrying the wp table's rows, "0 differ" would
       also be what a build that never consulted the table produces. The
       counter says how many elements really came from it. */
    if (gated && lz_debug_norm_qw == 0) {
        printf("FAIL: the gated loop never read the table (0 elements) - "
               "the rows above compare one arm to itself\n");
        return 1;
    }
    if (gated)
        printf("  gated cached elements: %lld of %d (%d rows of %d)\n",
               lz_debug_norm_qw, NW * NN,
               (int)(lz_debug_norm_qw / NN), NW);

    /* (2) the key, Q15 table - the same fixture against build_qw. The
       gated invocation runs the plain-w table and this one the (1+w)
       table, so the two invocations cover both add1 values; whichever
       runs, wq_rows[0] carries a table and the reuse address is its
       own. Compared on the ints AND on deq, because a stale ew changes
       the returned scale even where the codes survive. */
    for (i = 0; i < NN; i++) wq_fresh[i] = wq_rows[0][i];
    for (i = NN / 4; i < NN / 2; i++) wq_fresh[i] += 4.0f;
    memcpy(wq_reuse, wq_fresh, sizeof wq_reuse);
    memcpy(wq_rows[0], wq_reuse, sizeof wq_reuse);
    if (gated)
        lz_rmsnorm_gated_int(iq_reuse, xv, gv, wq_rows[0], NN, 1e-6f,
                             &dq_reuse);
    else
        lz_rmsnorm_int(iq_reuse, xv, wq_rows[0], NN, 1e-6f, &dq_reuse);
    if (gated)
        lz_rmsnorm_gated_int(iq_fresh, xv, gv, wq_fresh, NN, 1e-6f,
                             &dq_fresh);
    else
        lz_rmsnorm_int(iq_fresh, xv, wq_fresh, NN, 1e-6f, &dq_fresh);
    {
        /* Its OWN verdict, not the running one: the float fixture above
           sets the same flag, and printing that here would label this
           line with the other fixture's result. */
        int mem_diff = memcmp(iq_reuse, iq_fresh, sizeof iq_reuse) != 0;
        int q15_diff = mem_diff || dq_reuse != dq_fresh;
        if (q15_diff) bad_key = 1;
        printf("norm Q15 %s key: one address, two rows -> %s"
               "  [ints %s deq %.9g vs %.9g]\n",
               gated ? "plain-w (gated)" : "(1+w)",
               q15_diff ? "DIFFER" : "same",
               mem_diff ? "DIFFER" : "same",
               (double)dq_reuse, (double)dq_fresh);
    }

    if (bad || bad_key) { printf("FAIL\n"); return 1; }

    printf("PASS\n");
    return 0;
}
