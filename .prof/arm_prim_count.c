/* Executed-instruction counts for the ARM operator primitives, both
 * tiers, on the REAL argument distributions .prof/arm_bias_dist.py
 * captured.
 *
 * Same harness shape as .prof/arm_bias_drv.c (which measured the C tier
 * for docs/arm-asm-audit.md's table): one arm per leaf, one control arm
 * with the call removed and nothing else changed, and the marginal
 * between n=256 and n=512 taken so no first-call table build lands in
 * the number. What is new here is PRIM_TIER, which pins g_kernel so the
 * same binary shape can be measured with the C body and with the
 * assembly one.
 *
 * lz_i32f is `static` and inlines, so mode 2 reaches it through a
 * noinline wrapper and mode 3 is that same wrapper with the body
 * replaced by a bit copy - mode 2 minus mode 3 is the body with the
 * wrapper's own frame subtracted out. */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "ops.h"
#include "ops_quant.h"
#include "ops_kernel_shared.h"

#ifndef PRIM_MODE
#define PRIM_MODE 0
#endif
#ifndef PRIM_TIER          /* 0 = --kernel arm-c, 1 = --kernel arm-asm */
#define PRIM_TIER 0
#endif
#ifndef PRIM_INTARG
#define PRIM_INTARG 0
#endif

#if PRIM_MODE == 1
#include "sp_args_q8r.h"        /* real norm_ss_fixed arguments */
#elif PRIM_MODE == 2 || PRIM_MODE == 3
#include "sp_args_i32f.h"       /* int32 accumulators */
#elif PRIM_MODE == 4
#include "sp_args_exp.h"        /* real KDA decay-gate g */
#elif PRIM_MODE == 5 || PRIM_MODE == 6
#include "sp_args_q8r.h"        /* only for SP_N; the arms use ac_* */
#elif PRIM_MODE == 7
#include "sp_args.h"            /* real KDA decay-gate g, lz_sigmoid's own */
#elif PRIM_MODE == 8
#include "sp_args_sigt.h"       /* real conv1d t = (x+12)*16 */
#elif PRIM_MODE == 9 || PRIM_MODE == 10
#include "sp_args_q8r.h"        /* real norm_ss_fixed row, pre-scale */
#elif PRIM_MODE == 11 || PRIM_MODE == 12
#include "sp_args_q8r.h"        /* only for SP_N; the arms use cv_* */
#else
# if PRIM_INTARG
#  include "sp_args_i32f.h"
# else
#  include "sp_args_q8r.h"
# endif
#endif

__attribute__((noinline)) static float i32f_wrap(int32_t v) {
    return lz_i32f(v);
}
__attribute__((noinline)) static float i32f_null(int32_t v) {
    union { float f; int32_t i; } u;
    u.i = v;
    return u.f;
}

volatile int32_t g_sink;

/* norm_ss_fixed's Q15 image and its exponent (mode 9). Live outside the
 * loop so the call, not the buffer, is what the marginal measures. */
static short ns_q[512];
static int   ns_e;

/* Attention arms (modes 5 and 6). One fixed head, set up before the
 * loop so the measured marginal is the call itself; T is 256 so the
 * per-call number is dominated by the 512-MAC-per-row inner kernel the
 * assembly replaces, which is what the ratio is about. */
#define AC_HD  64
#define AC_KVD 64
#define AC_T   256
static float   ac_q[AC_KVD], ac_out[AC_KVD], ac_att[AC_T], ac_cb[AC_T];
static int8_t  ac_c[AC_T * AC_KVD];
static float   ac_s[AC_T * (AC_KVD / 32)];
static int16_t ac_cq[AC_T];

static void ac_setup(void) {
    uint32_t st = 0x9E3779B9u;
    int i;
    for (i = 0; i < AC_KVD; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        ac_q[i] = ((float)(int32_t)st / 2147483648.0f) * 3.0f;
    }
    for (i = 0; i < AC_T * AC_KVD; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        ac_c[i] = (int8_t)((int)(st % 255u) - 127);
    }
    for (i = 0; i < AC_T * (AC_KVD / 32); i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        ac_s[i] = ((float)(st >> 8) / 16777216.0f) * 0.05f + 1e-4f;
    }
    lz_attn_score_q8(ac_att, ac_q, AC_HD, ac_c, ac_s, AC_KVD, AC_T - 1,
                     0.125f, 0, 0);
}

/* Conv arms (modes 11 and 12). TWO WIDTHS OF THE SAME CALL, not a call
 * and a control: the per-channel cost is (per-call at CV_NCH) minus
 * (per-call at CV_NCH/2), divided by CV_NCH/2, which subtracts the
 * function's own frame and its two loop set-ups exactly rather than
 * estimating them away.
 *
 * THE SHIPPED SETUP PATH BUILDS EVERY SCALE. lz_conv_norm_pow2 per
 * channel, then ldexp and lz_sig_q15_fold - the same three lines
 * conv_fixed_build runs in forward.c, in the same order. A hand-made
 * mw/sig_k1 pair would measure a distribution the engine never produces,
 * and lz_conv_build_scales' own comment records a bug that survived
 * precisely because a probe drove a copy instead of the shipped code.
 *
 * xi IS NON-NULL, which is the shipping path: int-pipeline milestone 3
 * has the q/k/v projections exiting as int16 at LZ_CONV_ES already
 * (kda_conv_proj, forward.c), so the float input and its q8_round are
 * not what the target machine runs. The int inputs are log-uniform over
 * [1, bound] with random sign - the same shape sp_args_i32f.h uses for
 * the accumulator band it documents, and what makes acc span its own
 * band rather than sitting in one binade.
 *
 * CV_ES is LZ_CONV_ES (9), which forward.c defines privately; nothing
 * here depends on the value beyond its being the exponent the taps were
 * normalized against, and the arm would measure the same shape at any
 * other. */
#define CV_K   4
#define CV_NCH 64
#define CV_ES  9
static short cv_mw[CV_NCH * CV_K];
static short cv_st[CV_NCH * (CV_K - 1)];
static short cv_xi[CV_NCH];
static float cv_o[CV_NCH];
static float cv_k1[CV_NCH], cv_o2[CV_NCH];
static float cv_k2;
static float cv_in_scale;
static int   cv_bound;

static void cv_setup(void) {
    static float w[CV_NCH * CV_K];
    uint32_t st = 0x9E3779B9u;
    int c, j;
    cv_bound = lz_conv_accum_bound(CV_K);
    cv_in_scale = (float)ldexp(1.0, CV_ES);
    cv_k2 = lz_sig_q15_t_offset();
    for (c = 0; c < CV_NCH; c++) {
        int e;
        for (j = 0; j < CV_K; j++) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            w[c * CV_K + j] = ((float)(int32_t)st / 2147483648.0f) * 0.5f;
        }
        e = lz_conv_norm_pow2(w + c * CV_K, CV_K, cv_mw + c * CV_K, cv_bound);
        lz_sig_q15_fold((float)ldexp(1.0, -(e + CV_ES)), &cv_k1[c], &cv_o2[c]);
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        {   /* log-uniform magnitude in [1, bound], sign from bit 31 */
            int b = (int)(st % 14u);            /* bound is 2^13 for k=4 */
            int v = (int)((st >> 8) & (uint32_t)((1 << b) - 1)) + (1 << b) / 2;
            if (v > cv_bound) v = cv_bound;
            cv_xi[c] = (short)((st & 0x80000000u) ? -v : v);
        }
    }
    /* Fill the history through the shipped kernel, so sh_in holds what
     * the kernel itself would have left there. */
    for (j = 0; j < CV_K; j++)
        lz_causal_conv1d_step_fixed(cv_o, (const float *)0, cv_xi,
                                    cv_st, cv_st, cv_mw, cv_k1, cv_o2,
                                    cv_k2, cv_in_scale, cv_bound,
                                    CV_NCH, CV_K);
}

int main(int argc, char **argv) {
    long n = (argc > 1) ? atol(argv[1]) : 0;
    long i;
    union { float f; int32_t i; } u;
    lz_kernel_select(PRIM_TIER ? LZ_KERNEL_ARM_ASM : LZ_KERNEL_ARM);
    /* Modes 5 and 6 ONLY. It used to read `>= 5`, which pulled a whole
     * 256-row attention head into the n=0 baseline of the sigmoid and
     * norm arms - harmless in the marginal, which is what is reported,
     * but it made the zero control unreadable across arms. */
#if PRIM_MODE == 5 || PRIM_MODE == 6
    ac_setup();
#endif
#if PRIM_MODE == 11 || PRIM_MODE == 12
    cv_setup();
#endif
    for (i = 0; i < n; i++) {
        int k = (int)(i & (SP_N - 1));
#if PRIM_MODE == 1
        g_sink = q8_round(SP_ARG[k]);
#elif PRIM_MODE == 2
        u.f = i32f_wrap(SP_ARG[k]);        g_sink = u.i;
#elif PRIM_MODE == 3
        u.f = i32f_null(SP_ARG[k]);        g_sink = u.i;
#elif PRIM_MODE == 4
        u.f = lz_exp(SP_ARG[k]);           g_sink = u.i;
#elif PRIM_MODE == 5
        lz_attn_score_q8(ac_att, ac_q, AC_HD, ac_c, ac_s, AC_KVD,
                         AC_T - 1, 0.125f, 0, 0);
        u.f = ac_att[k];                   g_sink = u.i;
#elif PRIM_MODE == 6
        lz_attn_wsum_q8(ac_out, ac_att, AC_HD, ac_c, ac_s, AC_KVD,
                        AC_T - 1, ac_cb, ac_cq, 0, 0);
        u.f = ac_out[k & 63];              g_sink = u.i;
#elif PRIM_MODE == 7
        g_sink = sigmoid_q15(SP_ARG[k]);
#elif PRIM_MODE == 8
        g_sink = sigmoid_q15_t(SP_ARG[k]);
#elif PRIM_MODE == 9
        /* One whole row per call, like the attention arms: the leaf is
         * a loop and its per-element cost is what the ratio is about.
         * SP_ARG is the row already rescaled, so undo the 2^14 the
         * capture applied and let norm_ss_fixed pick its own scale. */
        u.f = norm_ss_fixed(SP_ARG, SP_N, ns_q, &ns_e);
        g_sink = u.i + ns_q[k];
#elif PRIM_MODE == 10
        /* norm_ss_fixed's own amax scan, alone. Mode 9 minus this is
         * the ELEMENT LOOP, which is the part the assembly replaces -
         * subtracting a measured number rather than an estimated one. */
        g_sink = (int32_t)q8_amax(SP_ARG, SP_N);
#elif PRIM_MODE == 11 || PRIM_MODE == 12
        lz_causal_conv1d_step_fixed(cv_o, (const float *)0, cv_xi,
                                    cv_st, cv_st, cv_mw, cv_k1, cv_o2,
                                    cv_k2, cv_in_scale, cv_bound,
                                    (PRIM_MODE == 11) ? CV_NCH : CV_NCH / 2,
                                    CV_K);
        u.f = cv_o[k & (CV_NCH - 1)];      g_sink = u.i;
#else
# if PRIM_INTARG
        g_sink = SP_ARG[k];
# else
        u.f = SP_ARG[k];                   g_sink = u.i;
# endif
#endif
    }
    return 0;
}
