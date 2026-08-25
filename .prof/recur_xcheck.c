/* Cross-check for the GDN recurrence's pass-1 and pass-2 arms beyond
 * the gt==1.0f epilogue kda_gt1_xcheck already covers.
 *
 * WHY THIS FILE. kda_gt1_xcheck drives lz_kda_step/lz_gdn_step and
 * mutates the gt==1.0f fast arm. The recurrence's OTHER arms have no
 * per-operator check: the pass-2 fixed entry (gdn_p2_fixed_rows, the
 * census's 38.4% site), the p2 group scale (23.1%), and the pass-1
 * tap (15.4%) can all be silently wrong while kda_gt1 stays green,
 * because none of its mutations touch them.
 *
 * This probe drives the same two entries with the same synthetic state
 * and prints the same two hashes, so recur_xcheck.sh can mutate each
 * arm in turn and assert the hash moves. The mutations are surgical:
 * each flips ONE computation, so a hash change proves that arm runs.
 *
 * Build: see recur_xcheck.sh.
 */
#include <stdio.h>
#include <stdint.h>

#include "ops.h"

#define KD  32
#define VD  64

static int8_t  sq[KD * VD];
#if LZ_GDN_STATE_2PLANE
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

static void fill_io(uint32_t *stp) {
    uint32_t st = *stp;
    int i;
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
    *stp = st;
}

/* One sweep: 8 steps of kda (gt=1.0, exercises pass-2 fixed entry and
   pass-1 tap) then 8 of gdn (decay gt, exercises the general arm). The
   state carries forward within each half and is reinitialised between
   the halves, exactly like kda_gt1_xcheck. */
static void sweep(int mode) {
    uint32_t st = 0x9E3779B9u, st0;
    int i, t;
    state_init(&st);
    st0 = st;
    h32 = 2166136261u;
    for (t = 0; t < 8; t++) {
        fill_io(&st);
        lz_kda_step(out, sq, sq, LZ_P2 ss, ss, qv, kv, vv, gv, 0.7f, KD, VD);
        feed(out, VD);
    }
    printf("KDA  %08lx\n", (unsigned long)h32);
    h32 = 2166136261u;
    st = st0;
    { uint32_t s2 = 0x9E3779B9u; state_init(&s2); }
    for (t = 0; t < 8; t++) {
        fill_io(&st);
        lz_gdn_step(out, sq, sq, LZ_P2 ss, ss, qv, kv, vv, 0.98f, 0.7f, KD, VD);
        feed(out, VD);
    }
    printf("GDN  %08lx\n", (unsigned long)h32);
}

int main(void) {
    lz_kernel_select(LZ_KERNEL_REF);
    sweep(0);
    return 0;
}
