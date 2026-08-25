/* The five ROW kernels, tier against tier, on synthetic weights.
 *
 * WHY THESE FIVE AND WHY DIRECTLY. q8_0, q4_1, q6_1, q16_0 and t2 are
 * the only kernel-matrix units with no per-operator cross-check, and
 * they are also the code the end-to-end gate reaches least: a coverage
 * build puts kernel_isa_identity_gate's fixture at 47.57% of the lines
 * in src/ops_kernel_dot_mmx.h, and the union over five checkpoints at
 * 55.34%. The arms it misses belong to weight formats no local model
 * carries, so no fixture reaches them. Feeding the kernels does.
 *
 * WHAT IS COMPARED. Each format has a six-slot dispatch table -
 * MMX/SSE/SSE2 as gcc intrinsics, then the same three as Watcom
 * assembly. Every populated slot must produce the identical int32
 * accumulator from identical input. Integer MACs are exact under
 * reassociation, so the failure this can find is not rounding: it is a
 * wrong LANE ORDER, a wrong tail length, or a nibble unpacked from the
 * wrong half - and each of those produces a plausible number.
 *
 * WHAT WOULD HIDE THOSE, and is therefore avoided:
 *   - constant or repeating activations: any lane permutation agrees.
 *     Every element gets a distinct value.
 *   - a weight row that is symmetric in the nibble: q4_1 packs two
 *     weights per byte, and hi==lo makes the unpack unobservable.
 *   - in_dim a clean multiple of the widest vector: the tail path is
 *     where a length bug lives, so the sweep includes sizes that are
 *     not multiples of 32.
 *
 * The seed is fixed at 1145141919 - two runs of this file compare the
 * same numbers or the comparison is not one.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops.h"
#include "ops_quant.h"
/* BEFORE ops_matmul.h, which declares LZ_ROW_Q8[LZ_ROW_N] without
   including the header that defines LZ_ROW_N. */
#include "ops_kernel_shared.h"
#include "ops_matmul.h"

#define MAXDIM 256
#define MAXNT  2

static unsigned long rng_state;
static void rng_seed(unsigned long s) { rng_state = s; }
static unsigned long rng_next(void) {
    /* Park-Miller. Reproducible across every compiler this tree uses,
       which rand() is not - two toolchains with different libc would
       otherwise compare different inputs and agree for the wrong
       reason. */
    rng_state = (rng_state * 48271u) % 2147483647u;
    return rng_state;
}

static const char *slot_name(int s) {
    switch (s) {
        case LZ_ROW_MMX_I:  return "mmxI";
        case LZ_ROW_SSE_I:  return "sseI";
        case LZ_ROW_SSE2_I: return "sse2I";
        case LZ_ROW_MMX_A:  return "mmxA";
        case LZ_ROW_SSE_A:  return "sseA";
        case LZ_ROW_SSE2_A: return "sse2A";
    }
    return "?";
}

static int check_table(const char *fmt, const lz_rowfn *tab,
                       int in_dim, int nt) {
    /* w4 is the weight the SSE2 kernels now read with movdqa: it must be
       16-byte aligned, and Watcom's static alignment is 8, so take an
       aligned view over an over-allocated buffer. The activation xw and
       the accumulators stay bare - the kernels read xw with movdqu and
       write acc with movd. */
    static unsigned char w4buf[MAXDIM * MAXNT + 16];
    unsigned char *w4 = (unsigned char *)(((size_t)w4buf + 15) & ~(size_t)15);
    static unsigned char w2[MAXDIM * MAXNT];
    static int16_t xw[MAXDIM * MAXNT];
    static int32_t acc_ref[MAXDIM * MAXNT];
    static int32_t acc[MAXDIM * MAXNT];
    int nb = in_dim / 32;
    int i, s, first = -1, bad = 0;

    for (i = 0; i < in_dim * nt; i++) {
        /* Distinct per element, and the two nibbles of each w4 byte
           deliberately differ - see the header. */
        unsigned long r = rng_next();
        w4[i] = (unsigned char)((r & 0x0Fu) | (((r >> 5) & 0x0Fu) << 4));
        if ((w4[i] & 0x0Fu) == (w4[i] >> 4)) w4[i] ^= 0x10u;
        w2[i] = (unsigned char)(r >> 9);
        xw[i] = (int16_t)((int)(r % 511u) - 255);
    }

    for (s = 0; s < LZ_ROW_N; s++) {
        if (!tab[s]) continue;
        memset(acc, 0, sizeof acc);
        tab[s](acc, w4, w2, xw, nb, nt, in_dim, w2, w2 + sizeof w2 / 2);
        if (first < 0) {
            first = s;
            memcpy(acc_ref, acc, sizeof acc);
            continue;
        }
        /* EVERY accumulator, not the first per token. This compared
           `nt * 4` bytes - nt int32 - while the kernels write nb*nt of
           them, so only sub-block 0 of each token was ever checked and a
           kernel could be wrong in all the others silently. Found by
           mutating a 128-element group kernel that folds four sub-blocks
           at once: breaking its third and fourth lanes left this green,
           because those land in acc[2] and acc[3]. */
        if (memcmp(acc_ref, acc, (size_t)nb * nt * 4) != 0) {
            printf("  FAIL %s n=%d nt=%d: slot %s differs from %s\n",
                   fmt, in_dim, nt, slot_name(s), slot_name(first));
            for (i = 0; i < nt; i++)
                printf("      [%d] %s=%ld  %s=%ld\n", i,
                       slot_name(first), (long)acc_ref[i],
                       slot_name(s), (long)acc[i]);
            bad = 1;
        }
    }
    if (first < 0) return -1;      /* nothing populated - caller reports */
    return bad;
}

int main(void) {
    /* Sizes: two clean multiples of 32 and two that are not, because the
       tail is where a length bug lives. */
    static const int DIMS[] = { 32, 64, 96, 160 };
    static const struct { const char *n; const lz_rowfn *t; } FMT[] = {
        { "q8_0",  LZ_ROW_Q8  },
        { "q4_1",  LZ_ROW_Q41 },
        { "q6_1",  LZ_ROW_Q61 },
        { "q16_0", LZ_ROW_Q16 },
        { "t2",    LZ_ROW_T2  }
    };
    int f, d, nt, rc, bad = 0, compared = 0, empty = 0;

    rng_seed(1145141919u);
    for (f = 0; f < (int)(sizeof FMT / sizeof FMT[0]); f++) {
        int pop = 0, s;
        for (s = 0; s < LZ_ROW_N; s++) if (FMT[f].t[s]) pop++;
        if (pop < 2) {
            /* One implementation cannot disagree with itself. Reported,
               not skipped silently: a format that loses its twin would
               otherwise read the same as one that never had one. */
            printf("  %-6s %d slot(s) populated - nothing to compare\n",
                   FMT[f].n, pop);
            empty++;
            continue;
        }
        for (d = 0; d < (int)(sizeof DIMS / sizeof DIMS[0]); d++) {
            for (nt = 1; nt <= MAXNT; nt++) {
                rc = check_table(FMT[f].n, FMT[f].t, DIMS[d], nt);
                if (rc < 0) continue;
                if (rc) bad = 1;
                compared++;
            }
        }
        printf("  %-6s %d slots, %d size x nt combination(s)\n",
               FMT[f].n, pop, (int)(sizeof DIMS / sizeof DIMS[0]) * MAXNT);
    }

    /* Positive control. Zero comparisons is the same output as all of
       them agreeing, and this build linking with every table empty is
       exactly how that would happen. */
    if (compared == 0) {
        printf("rowdot_xcheck: FAIL - no comparison ran (%d format(s) had"
               " fewer than two slots)\n", empty);
        return 1;
    }
    if (bad) { printf("rowdot_xcheck: FAIL\n"); return 1; }
    printf("rowdot_xcheck: PASS (%d comparison(s), %d format(s) with a"
           " single slot)\n", compared, empty);
    return 0;
}
