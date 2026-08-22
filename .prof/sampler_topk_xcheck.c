/* Does skipping the whole-vocabulary softmax change the candidate
 * distribution? src/sampler.c, select_topk_exp.
 *
 * WHY NOT AN END-TO-END GENERATION DIFF. That is what I tried first and
 * it was worthless twice over: the control binary was built from HEAD
 * while the new one came from the working tree, so it also carried
 * another session's uncommitted forward.c (every seed "differed",
 * including --temp 0, which the sampler change cannot reach); and the
 * output line carries a wall-clock number, so even a correct pair
 * differs. Drive the sampler alone, from one snapshot, and neither
 * failure is possible.
 *
 * lz_target_dist is the entry that RETURNS the distribution instead of
 * drawing from it, so the comparison is over numbers rather than over
 * one sampled token - a token that matches proves much less than a
 * distribution that does.
 *
 * The vectors are shaped like real logits: one peak a few units above a
 * shoulder of plausible continuations, then a long tail, which is where
 * top_k's boundary and top_p's cumulative threshold actually sit. A
 * uniform vector would put the whole vocabulary inside the nucleus and
 * test nothing.
 *
 * Prints one line per case: the surviving indices' checksum and the
 * maximum relative difference the caller can compare between builds.
 *
 * Build: see sampler_topk_xcheck.sh. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sampler.h"

#define VOCAB 32768
#define NCASE 24

static float logits[VOCAB];
static float outp[VOCAB];
static LZProbIndex buf[VOCAB];

static uint32_t rs = 0x9E3779B9u;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}

int main(int argc, char **argv) {
    LZSampleParams pr;
    FILE *dump = (argc > 1) ? fopen(argv[1], "wb") : NULL;
    int c, i;
    for (c = 0; c < NCASE; c++) {
        int peak = (int)(rnd() % VOCAB);
        int shoulder = 40 + (int)(rnd() % 60);
        double h = 0.0;
        uint32_t sum = 2166136261u;
        /* FLAT ENOUGH THAT THE NUCLEUS HAS SOMETHING IN IT. The first
           version of this fixture put the peak 5 logits above the
           shoulder, which after a 0.6 temperature is a factor of 4000 -
           top_p 0.95 then kept ONE candidate per case and the whole
           comparison ran over 30 numbers. A real next-token
           distribution at temperature has a shoulder within a logit or
           so of the peak, which is exactly where top_k's boundary and
           top_p's threshold do their work. */
        for (i = 0; i < VOCAB; i++)
            logits[i] = -6.0f + ((float)(rnd() >> 8) / 16777216.0f) * 4.0f;
        for (i = 0; i < shoulder; i++)
            logits[(peak + i * 977) % VOCAB] =
                1.6f - (float)i * 0.02f + ((float)(rnd() >> 8) / 16777216.0f) * 0.3f;
        logits[peak] = 2.0f;

        memset(&pr, 0, sizeof(pr));
        pr.temperature = 0.6f + (float)(c % 4) * 0.2f;
        pr.topk = (c % 3 == 0) ? 20 : ((c % 3 == 1) ? 8 : 64);
        pr.topp = (c % 2) ? 0.8f : 0.95f;
        pr.minp = (c % 5 == 0) ? 0.05f : 0.0f;
        pr.minp_llamacpp = 1;

        lz_target_dist(&pr, logits, VOCAB, buf, outp);

        for (i = 0; i < VOCAB; i++) {
            if (outp[i] != 0.0f) {
                union { float f; uint32_t u; } b;
                b.f = outp[i];
                sum = (sum ^ (uint32_t)i) * 16777619u;
                sum = (sum ^ b.u) * 16777619u;
                h += (double)outp[i];
            }
        }
        printf("case %2d topk=%2d topp=%.2f minp=%.2f  idx+p=%08lx mass=%.9f\n",
               c, pr.topk, (double)pr.topp, (double)pr.minp,
               (unsigned long)sum, h);
        if (dump) fwrite(outp, sizeof(float), VOCAB, dump);
    }
    if (dump) fclose(dump);
    return 0;
}
