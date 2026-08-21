/* What dynamic range do the conv1d inputs actually have?
 *
 * The fixed conv tier needs a history exponent `es`, and the history
 * starts at zero so there is nothing to derive one from at allocation.
 * Either the scale has to travel with the state (what ssm_state_s does
 * for the recurrence, and it costs a renormalization pass), or a
 * constant is good enough. This measures which.
 *
 * A constant es is good enough only if, over real tokens:
 *   - the range is narrow enough that one exponent covers it without
 *     clipping the tail, and
 *   - it does not drift as the sequence grows, since the history holds
 *     values from earlier steps at the same scale.
 *
 * Both are reported. The second is the one a single-token probe would
 * miss entirely.
 *
 * Instrumentation is LZ_TAP, defined before forward.c is included -
 * the mechanism the file's own header describes for exactly this. The
 * taps wanted are kq/kk/kv (KDA conv inputs, before the convolution)
 * and ssm's own conv input where the model is GDN.
 *
 * Build:  .prof/build.sh wconvrange
 * Usage:  wconvrange <model-dir> [ntok]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Buckets by power of two, so "which exponent covers it" is read
   directly off the histogram rather than inferred from a min/max pair
   that one outlier can move. */
#define NB 40
static long hist_all[NB];
static long hist_early[NB], hist_late[NB];
static double gmax = 0.0;
static long n_seen = 0;
static int cur_step = 0, split_step = 0;

static void tap_range(const char *tag, const float *p, int n) {
    int i;
    if (!(tag[0] == 'k' && (tag[1] == 'q' || tag[1] == 'k' || tag[1] == 'v')
          && tag[2] == '\0'))
        return;                       /* only the three conv inputs */
    for (i = 0; i < n; i++) {
        double a = fabs((double)p[i]);
        int e;
        if (a > gmax) gmax = a;
        n_seen++;
        if (a == 0.0) { e = 0; }
        else {
            e = (int)floor(log(a) / log(2.0)) + 20;   /* 2^-20 -> bucket 0 */
            if (e < 0) e = 0;
            if (e >= NB) e = NB - 1;
        }
        hist_all[e]++;
        if (cur_step < split_step) hist_early[e]++; else hist_late[e]++;
    }
}

#define LZ_TAP(tag, li, ptr, n) tap_range((tag), (ptr), (n))

#include "ops.h"
#include "model.h"
#include "forward.h"
#include "ops.c"
#include "err.c"
#include "compat.c"
#include "lfn.c"
#include "json.c"
#include "safetensors.c"
#include "model.c"
#include "forward.c"

static int tok_at(int i) {
    static const int SEED[8] = { 3, 17, 5, 291, 42, 8, 1024, 77 };
    return SEED[i & 7];
}

/* Highest bucket holding at least `frac` of the mass, counting down.
   frac must be > 0: at 0 the first comparison is trivially true and this
   returns NB-1 whatever the data holds - which it did, reporting a
   "largest bucket" of 2^19 next to a measured max of 22.3. Use
   top_bucket() for the extreme instead of passing 0 here. */
static int pct_bucket(const long *h, double frac, long total) {
    long acc = 0;
    int i;
    if (frac <= 0.0 || total <= 0) return -1;
    for (i = NB - 1; i >= 0; i--) {
        acc += h[i];
        if ((double)acc / (double)total >= frac) return i;
    }
    return 0;
}

/* Highest NON-EMPTY bucket. */
static int top_bucket(const long *h) {
    int i;
    for (i = NB - 1; i >= 0; i--) if (h[i]) return i;
    return -1;
}

int main(int argc, char **argv) {
    LZModel m;
    LZRunState st;
    char err[512];
    int i, ntok = 64;

    if (argc < 2) { printf("usage: wconvrange <model-dir> [ntok]\n"); return 2; }
    if (argc > 2) ntok = atoi(argv[2]);
    if (ntok < 8) ntok = 8;
    split_step = ntok / 2;

    if (lz_open(&m, argv[1], err, sizeof(err)) != 0) {
        printf("open failed: %s\n", err); return 1;
    }
    if (lz_read_weights(&m, err, sizeof(err)) != 0) {
        printf("weights failed: %s\n", err); return 1;
    }
    if (lz_state_alloc(&st, &m, ntok + 8, 0, err, sizeof(err)) != 0) {
        printf("state failed: %s\n", err); return 1;
    }
    for (i = 0; i < ntok; i++) {
        cur_step = i;
        lz_forward(&m, &st, tok_at(i), i);
    }

    printf("conv1d input range over %d tokens, %ld samples\n", ntok, n_seen);
    printf("max |x| = %.4f\n\n", gmax);

    {
        int b999 = pct_bucket(hist_all, 0.001, n_seen);   /* top 0.1%% */
        int bmax = top_bucket(hist_all);
        int be = pct_bucket(hist_early, 0.001,
                            n_seen ? n_seen / 2 : 1);
        int bl = pct_bucket(hist_late, 0.001, n_seen ? n_seen / 2 : 1);
        printf("bucket = floor(log2|x|) + 20\n");
        printf("  top 0.1%% of |x| at or above 2^%d\n", b999 - 20);
        printf("  largest bucket touched          2^%d\n", bmax - 20);
        printf("  first half top-0.1%% bucket      2^%d\n", be - 20);
        printf("  second half top-0.1%% bucket     2^%d\n", bl - 20);
        printf("\n%s\n",
               be == bl ? "range is STABLE across the sequence"
                        : "range MOVES between halves - a constant es "
                          "would fit one and clip the other");
        /* A constant es maps the largest expected |x| onto the bound.
           Report what that costs the small end: the ratio between the
           largest and the 0.1%% point is how much of int16 is spent on
           outliers. */
        printf("outlier headroom: 2^%d between the 0.1%% point and the max\n",
               bmax - b999);
    }
    lz_state_free(&st);
    lz_free(&m);
    return 0;
}
