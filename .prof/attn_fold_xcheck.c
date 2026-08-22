/* Cross-check for lz_attn_score_q8's row epilogue, src/ops_gdn.c.
 *
 * TWO CHANGES, ONE OF WHICH NO AVAILABLE MODEL EXERCISES.
 *   - `(1.0f/256.0f) * scale` is folded once per call instead of once
 *     per row. Every head width reaches this.
 *   - `((a0+a2)+(a1+a3))` becomes `a0` when ng < 4, because the unrolled
 *     group loop needs ng >= 4 to run at all and a1/a2/a3 are still
 *     +0.0f below that. .prof/x87test has head_dim 128, so ng is 4 and
 *     an end-to-end logits comparison on it says NOTHING about this
 *     arm - it never takes it.
 *
 * SO THE COMPARISON IS AGAINST THE OLD SOURCE, NOT AGAINST A COPY OF
 * IT. attn_fold_xcheck.sh builds this same file against a tree with the
 * condition widened to `ng >= 0` (the four-accumulator form, always)
 * and against one with `(1.0f/256.0f) * scale` restored to the row, and
 * requires the bytes to match. A reimplementation of the epilogue here
 * would be checking my transcription, not the engine.
 *
 * Head widths swept: 32, 64, 96 (ng 1, 2, 3 - the arm the models do not
 * reach) and 128, 160 (ng 4, 5 - the arm they do). A sweep that only
 * ran the second set is the failure this file exists to avoid.
 *
 * Build: see attn_fold_xcheck.sh. Output is a hash of every att[] the
 * sweep produces; the script compares hashes across trees. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ops.h"

#define MAX_HD  160
#define TT      64
#define KVD     MAX_HD

static float  q[MAX_HD];
static int8_t kc[TT * KVD];
static float  ks[TT * (KVD / 32)];
static float  att[TT];

int main(void) {
    static const int HDS[] = { 32, 64, 96, 128, 160 };
    uint32_t st = 0x9E3779B9u, h = 2166136261u;
    int hi, i, t;
    lz_kernel_select(LZ_KERNEL_REF);
    for (i = 0; i < MAX_HD; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        q[i] = ((float)(int32_t)st / 2147483648.0f) * 2.0f;
    }
    for (i = 0; i < TT * KVD; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        kc[i] = (int8_t)((int)(st % 255u) - 127);
    }
    for (i = 0; i < TT * (KVD / 32); i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        ks[i] = ((float)(st >> 8) / 16777216.0f) * 0.05f + 1e-4f;
    }
    for (hi = 0; hi < (int)(sizeof(HDS) / sizeof(HDS[0])); hi++) {
        int hd = HDS[hi];
        lz_attn_score_q8(att, q, hd, kc, ks, KVD, TT - 1,
                         0.088388348f, 0, 0);
        for (t = 0; t < TT; t++) {
            union { float f; uint32_t u; } b;
            b.f = att[t];
            h = (h ^ b.u) * 16777619u;
        }
        printf("hd=%3d ng=%d  att[0]=%.9g att[%d]=%.9g\n",
               hd, hd / 32, (double)att[0], TT - 1, (double)att[TT - 1]);
    }
    printf("ATTFOLD %08lx\n", (unsigned long)h);
    return 0;
}
