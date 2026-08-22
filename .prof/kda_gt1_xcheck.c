/* Cross-check for gdn_step_impl's gt == 1.0f epilogue arm, ops_gdn.c.
 *
 * WHY IT NEEDS ITS OWN PROBE. lz_kda_step passes the literal 1.0f, so
 * `gt * u` and `gt * w` are exact no-ops and the two multiplies can go.
 * .prof/x87test cannot show that: its linear layers reach the SAME
 * function through lz_gdn_step with a real decay - counted, 0 of 600+
 * calls in a five-token forward took gt == 1.0f, with values like
 * 0x3F7D0798 instead. An end-to-end logits comparison on that model is
 * therefore GREEN NO MATTER WHAT this arm does, including if it were
 * deleted outright.
 *
 * So the check calls lz_kda_step DIRECTLY - the entry that carries the
 * literal - and kda_gt1_xcheck.sh compares the hash against a tree
 * whose condition is disabled (`gb.u == 0u`, never true, the whole
 * epilogue back on the general loop), which must be identical, plus a
 * tree whose fast arm is wrong, which must differ. The second one is
 * what proves the arm is reached; without it the first would pass by
 * comparing the general loop to itself.
 *
 * lz_gdn_step is driven too, with a decay gt, so a change that broke
 * the OTHER arm could not hide behind the KDA one.
 *
 * Build: see kda_gt1_xcheck.sh. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ops.h"

#define KD  32
#define VD  64

static int8_t  sq[KD * VD];
#if LZ_GDN_STATE_2PLANE
/* The low plane is part of the signature when the 2-plane state is
   compiled in, which is the default (build/switch_matrix_gate.sh lists
   LZ_GDN_STATE_2PLANE's #ifndef default as 1). */
static int8_t  sq2[KD * VD];
#define LZ_P2 sq2, sq2,
#else
#define LZ_P2
#endif
static float   ss[KD * (VD / 32)];
static float   qv[KD], kv[KD], vv[VD], gv[KD], out[VD];

static uint32_t h32 = 2166136261u;
static void feed(const float *p, int n) {
    union { float f; uint32_t u; } b;
    int i;
    for (i = 0; i < n; i++) { b.f = p[i]; h32 = (h32 ^ b.u) * 16777619u; }
}

/* THE STATE HAS TO BE REBUILT BETWEEN THE TWO SWEEPS. The step writes
   it back, so running the GDN sweep after the KDA one made the GDN hash
   a function of the KDA arm - and the mutation below then changed both,
   which the script reads (correctly) as the two arms being crossed. The
   check found a bug in this file before it found one in the engine. */
static void state_init(uint32_t *stp) {
    uint32_t st = *stp;
    int i;
    for (i = 0; i < KD * VD; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        sq[i] = (int8_t)((int)(st % 255u) - 127);
    }
#if LZ_GDN_STATE_2PLANE
    for (i = 0; i < KD * VD; i++) sq2[i] = 0;
#endif
    for (i = 0; i < KD * (VD / 32); i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        ss[i] = ((float)(st >> 8) / 16777216.0f) * 0.02f + 1e-4f;
    }
    *stp = st;
}

int main(void) {
    uint32_t st = 0x9E3779B9u, st0;
    int i, t;
    lz_kernel_select(LZ_KERNEL_REF);
    state_init(&st);
    st0 = st;
    /* Eight steps, state carried forward: the epilogue writes u, pass 2
       folds it into the state, and a one-shot call would not show a
       change that only reaches the output through the next step. */
    for (t = 0; t < 8; t++) {
        for (i = 0; i < KD; i++) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            qv[i] = ((float)(int32_t)st / 2147483648.0f) * 0.5f;
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            kv[i] = ((float)(int32_t)st / 2147483648.0f) * 0.5f;
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            gv[i] = 0.9f + ((float)(st >> 8) / 16777216.0f) * 0.1f;
        }
        for (i = 0; i < VD; i++) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            vv[i] = ((float)(int32_t)st / 2147483648.0f) * 1.5f;
        }
        lz_kda_step(out, sq, sq, LZ_P2 ss, ss, qv, kv, vv, gv, 0.7f, KD, VD);
        feed(out, VD);
    }
    printf("KDA  %08lx\n", (unsigned long)h32);
    h32 = 2166136261u;
    st = st0;
    { uint32_t s2 = 0x9E3779B9u; state_init(&s2); }
    for (t = 0; t < 8; t++) {
        for (i = 0; i < KD; i++) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            qv[i] = ((float)(int32_t)st / 2147483648.0f) * 0.5f;
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            kv[i] = ((float)(int32_t)st / 2147483648.0f) * 0.5f;
        }
        for (i = 0; i < VD; i++) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            vv[i] = ((float)(int32_t)st / 2147483648.0f) * 1.5f;
        }
        lz_gdn_step(out, sq, sq, LZ_P2 ss, ss, qv, kv, vv, 0.98f, 0.7f, KD, VD);
        feed(out, VD);
    }
    printf("GDN  %08lx\n", (unsigned long)h32);
    return 0;
}
