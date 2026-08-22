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
#elif PRIM_MODE == 13 || PRIM_MODE == 14 || PRIM_MODE == 15
#include "sp_args_q8r.h"        /* real norm rows; see sm_setup/nm_setup */
#elif PRIM_MODE == 16 || PRIM_MODE == 17 || PRIM_MODE == 19 || PRIM_MODE == 20
#include "sp_args_q8r.h"        /* real floats; only the shape matters */
#elif PRIM_MODE == 18
#include "sp_args_q8r.h"        /* only for SP_N; the arm uses gd_* */
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
volatile float g_mulc = 1.4142135f;      /* modes 16/17/19/20, see there */
volatile float g_facc = 0.7071068f;      /* mode 20's accumulator */

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
/* Guarded, unlike the arms above it, because this one calls a function
   whose SIGNATURE changes: lz_causal_conv1d_step_fixed has gained a
   parameter twice now, and an unguarded setup makes every OTHER mode
   fail to build the moment it does. A measurement harness that cannot
   be built is not a slow harness, it is no harness. */
#if PRIM_MODE == 11 || PRIM_MODE == 12
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
#endif /* PRIM_MODE == 11 || PRIM_MODE == 12 */

/* Softmax and rmsnorm arms (modes 13/15 and 14). These are the two rows
 * docs/arm-asm-audit.md still carries as `~100` ESTIMATES, and the conv
 * row is what an estimate of that shape is worth on this machine.
 *
 * NEITHER IS TIER-GATED, unlike every arm above: the softmax max scan
 * runs in the integer domain on every platform and the norm's (1+w) Q15
 * weight table is cached on every platform, so t0 and t1 differ here
 * only through the leaves they call (q8_round, lz_i32f, norm_ss_fixed).
 * A before/after for those two changes has to compile this same harness
 * against an older src tree, which is why the arms exist rather than a
 * tier flag.
 *
 * MODE 15 IS THE CONTROL FOR MODE 13, not for 14: lz_softmax works in
 * place, so the arm has to refill its input every iteration and that
 * refill is inside the measured window. 14 needs no control beyond c0 -
 * lz_rmsnorm does not touch its input. The output-loop figure is
 * 14 minus 9 (norm_ss_fixed on the SAME row), which is why both read
 * SP_ARG at SP_N rather than a length of their own. */
static float sm_x[SP_N];
static float sm_src[SP_N];
static float nm_o[SP_N];
static float nm_w[SP_N];

static void sm_setup(void) {
    int i;
    /* Attention logits, not norm inputs: q . k * scale lands within a
     * few units of zero either side, which is the range softmax's
     * exp and its max scan actually see. */
    for (i = 0; i < SP_N; i++) sm_src[i] = SP_ARG[i] * (8.0f / 16384.0f);
}

static void nm_setup(void) {
    int i;
    /* An RMSNorm weight row sits near 1. The Q15 (1+w) table this
     * exercises is keyed on the weight POINTER, so a single static
     * array is the shipping case (one table per layer weight), not a
     * best case invented for the measurement. */
    for (i = 0; i < SP_N; i++)
        nm_w[i] = 1.0f + SP_ARG[i] * (0.25f / 16384.0f);
}

/* GDN recurrence arm (mode 18). One whole lz_gdn_step per iteration -
 * the entry gdn_pass1_fixed_head and therefore gdn_sum_gg sit under,
 * which is where docs/arm-asm-audit.md's D2 row lives. Driven through
 * the SHIPPED entry rather than by calling the static directly, so the
 * pass-1 tier selection and the state write-back are in the measured
 * window exactly as the engine runs them. */
#define GD_KD 32
#define GD_VD 64
static int8_t  gd_sq[GD_KD * GD_VD];
#if LZ_GDN_STATE_2PLANE
static int8_t  gd_sq2[GD_KD * GD_VD];
#define GD_P2 gd_sq2, gd_sq2,
#else
#define GD_P2
#endif
static float   gd_ss[GD_KD * (GD_VD / 32)];
static float   gd_q[GD_KD], gd_k[GD_KD], gd_v[GD_VD], gd_out[GD_VD];

static void gd_setup(void) {
    uint32_t st = 0x9E3779B9u;
    int i;
    for (i = 0; i < GD_KD * GD_VD; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        gd_sq[i] = (int8_t)((int)(st % 255u) - 127);
    }
    for (i = 0; i < GD_KD * (GD_VD / 32); i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        gd_ss[i] = ((float)(st >> 8) / 16777216.0f) * 0.02f + 1e-4f;
    }
    for (i = 0; i < GD_KD; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        gd_q[i] = ((float)(int32_t)st / 2147483648.0f) * 0.5f;
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        gd_k[i] = ((float)(int32_t)st / 2147483648.0f) * 0.5f;
    }
    for (i = 0; i < GD_VD; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        gd_v[i] = ((float)(int32_t)st / 2147483648.0f) * 1.5f;
    }
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
#if PRIM_MODE == 13 || PRIM_MODE == 15
    sm_setup();
#endif
#if PRIM_MODE == 14
    nm_setup();
#endif
#if PRIM_MODE == 18
    gd_setup();
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
#elif PRIM_MODE == 13 || PRIM_MODE == 15
        {
            int j;
            for (j = 0; j < SP_N; j++) sm_x[j] = sm_src[j];
#if PRIM_MODE == 13
            lz_softmax(sm_x, SP_N);
#endif
        }
        u.f = sm_x[k];                     g_sink = u.i;
#elif PRIM_MODE == 14
        lz_rmsnorm(nm_o, SP_ARG, nm_w, SP_N, 1e-6f);
        u.f = nm_o[k];                     g_sink = u.i;
#elif PRIM_MODE == 16 || PRIM_MODE == 17
        /* ONE BARE SOFT-FLOAT OPERATION, against control arm c0 (which
         * is this loop with the operation removed). Every "is it worth
         * writing this in assembly" question in docs/arm-asm-audit.md
         * reduces to this number, and until it was measured it was
         * being INFERRED - the conv epilogue's row is what an inferred
         * one is worth. `g_mulc` is volatile so neither operand folds.
         *
         * The multiplier is a constant in the SOURCE because that is
         * the shape M5 is about (one operand loop-invariant); volatile
         * keeps the compiler from acting on it. */
#if PRIM_MODE == 16
        u.f = SP_ARG[k] * g_mulc;          g_sink = u.i;
#else
        u.f = SP_ARG[k] + g_mulc;          g_sink = u.i;
#endif
#elif PRIM_MODE == 19 || PRIM_MODE == 20
        /* THE TWO CHAINS docs/arm-asm-audit.md G2 rates worth writing,
         * measured instead of argued. Modes 16/17 answer what ONE bare
         * float operation costs; these answer what the PAIR costs when
         * both halves come from the same operands - which is the whole
         * G2 claim, that one unpack+align can serve both.
         *
         * 19 is lz_fwht's butterfly: a+b and a-b, two roundings of two
         * different values from one pair of operands. Nothing about
         * sharing the unpack changes either rounding, so a hand-written
         * cell here is rounding-NEUTRAL.
         *
         * 20 is lz_matmul's inner MAC: acc + w*x. The shareable part is
         * smaller than it looks and the reason is a correctness one -
         * skipping the product's rounding turns this into an FMA, which
         * is a DIFFERENT function from fl(fl(w*x)+acc) and would break
         * the x86<->ARM parity gate. So only the pack/unpack round trip
         * between the two operations is available, not the rounding.
         *
         * Both sinks are consumed so neither half can be dropped. */
#if PRIM_MODE == 19
        {
            float a = SP_ARG[k], b = g_mulc;
            u.f = a + b;  g_sink = u.i;
            u.f = a - b;  g_sink ^= u.i;
        }
#else
        {
            float w = SP_ARG[k];
            u.f = g_facc + w * g_mulc;      g_sink = u.i;
        }
#endif
#elif PRIM_MODE == 18
        lz_gdn_step(gd_out, gd_sq, gd_sq, GD_P2 gd_ss, gd_ss,
                    gd_q, gd_k, gd_v, 0.98f, 0.7f, GD_KD, GD_VD);
        u.f = gd_out[k & (GD_VD - 1)];     g_sink = u.i;
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
