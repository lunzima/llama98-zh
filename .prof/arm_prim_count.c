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
#include "ops_arm.h"

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
#elif PRIM_MODE == 27 || PRIM_MODE == 28
#include "sp_args_q8r.h"        /* the row arms build mm_w/mm_x from it */
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

#if PRIM_MODE == 27 || PRIM_MODE == 28
/* f32mm's row arms. 64 elements is a whole harness body's worth, so the
   per-iteration figure stays per element; the weights are the same real
   float stream the other arms use, which keeps the decline rate honest
   (a row of invented values could be all-normal by construction). */
#define MM_N 64
static float mm_w[MM_N], mm_x[MM_N], mm_acc[8];
static void mm_setup(void) {
    int i;
    for (i = 0; i < MM_N; i++) {
        mm_w[i] = SP_ARG[i];
        mm_x[i] = SP_ARG[(i + 17) & (SP_N - 1)];
    }
}
#endif /* PRIM_MODE == 27 || 28 */

#if PRIM_MODE == 24 && defined(LZ_ARM_PRIM_ASM)
/* THE QUESTION MODE 24 ANSWERS, and the only reason this body exists:
 * can a hand-written soft-float core in this tree beat libgcc's?
 *
 * It lives HERE and not in src/ops_arm_prim.h because nothing ships it.
 * f32mm armA is `acc + w*x`, and what a hand cell can take there is the
 * product's pack and the addend's unpack of that same word - about 14
 * of the chain's 74 instructions. That only pays if the multiply and
 * add cores are themselves no worse than __aeabi_fmul and __addsf3,
 * which are hand-tuned assembly.
 *
 * Half the answer was already in: lz_faddsub_arm_asm does two adds off
 * one unpack for 89, so its adder core runs about 44.5 an add against
 * __addsf3's 43.03. This is the other half, deliberately a MULTIPLY
 * ALONE rather than the whole MAC - a core that cannot match
 * __aeabi_fmul's 30 settles the cell without anyone writing the loop,
 * the group snapshot and the group-wide decline the full one needs.
 *
 * ANSWER: 49.0 against 30.0, with zero declines on this argument
 * stream (the fallback's `bl` executes 0 times in 1024 calls). It
 * overspends by 19 to save 14, so the cell is DECLINED in the matrix
 * rather than left TODO.
 *
 * Same decline set as the adder plus the product's own: an
 * out-of-range product exponent either way. UMULL gives the 48-bit
 * significand product in one instruction. */
static __inline__ int spw_fmul_arm(uint32_t ua, uint32_t ub,
                                   uint32_t *out) {
    uint32_t ok, R, e, pl, ph, t, a = ua, b = ub;
    __asm__(
        /* Four special-case tests in five instructions, by chaining
           conditions rather than branching between them: ANDS leaves Z
           when an exponent is 0, and each following test only runs if
           the previous one passed. */
        "mov    %[t],  #255\n\t"
        "ands   %[e],  %[t], %[a], lsr #23\n\t"
        "andnes %[R],  %[t], %[b], lsr #23\n\t"
        "teqne  %[e],  %[t]\n\t"
        "teqne  %[R],  %[t]\n\t"
        "beq    9f\n\t"
        "add    %[e],  %[e], %[R]\n\t"
        "eor    %[t],  %[a], %[b]\n\t"
        /* SIGNIFICANDS PRE-SCALED BY 2^4, which is what makes UMULL's
           HIGH WORD the answer: 28-bit factors give a 56-bit product,
           so its top 24 bits land in the high word already positioned
           as a significand. Extracting them from a 48-bit product
           straddling both words costs three more instructions and a
           second shift of the low word. */
        "mov    %[ph], #0x8000000\n\t"
        "mov    %[a],  %[a], lsl #9\n\t"
        "orr    %[a],  %[ph], %[a], lsr #5\n\t"
        "mov    %[b],  %[b], lsl #9\n\t"
        "orr    %[b],  %[ph], %[b], lsr #5\n\t"
        "umull  %[pl], %[ph], %[a], %[b]\n\t"
        "and    %[R],  %[t], #0x80000000\n\t"
        "cmp    %[ph], #0x800000\n\t"
        "movcc  %[ph], %[ph], lsl #1\n\t"
        "orrcc  %[ph], %[ph], %[pl], lsr #31\n\t"
        "movcc  %[pl], %[pl], lsl #1\n\t"
        "orr    %[R],  %[R], %[ph]\n\t"
        /* SBC, not SUB: the carry the compare above left is exactly
           "the product reached 2", so the normalisation's exponent step
           rides the bias subtraction. None of the three conditional
           moves between them writes flags. e ends one BELOW the biased
           result exponent, because the significand's implicit bit
           supplies the missing one in the ADC below. */
        "sbc    %[e],  %[e], #127\n\t"
        "cmp    %[e],  #253\n\t"
        "bhi    9f\n\t"                    /* over- or underflow */
        /* ADC, not ADD: the guard bit is the carry out of this compare,
           so rounding up and adding the exponent are one instruction -
           and a mantissa that carries to 2^24 walks into the exponent
           field on its own, needing no case. BICEQ is round-half-to-
           even entire: EQ is the exact tie, and clearing bit 0 is what
           that decides. */
        "cmp    %[pl], #0x80000000\n\t"
        "adc    %[R],  %[R], %[e], lsl #23\n\t"
        "biceq  %[R],  %[R], #1\n\t"
        "mov    %[ok], #1\n\t"
        "b      8f\n\t"
        "9:\n\t"
        "mov    %[ok], #0\n\t"
        "8:\n\t"
        : [ok] "=&r"(ok), [R] "=&r"(R), [e] "=&r"(e), [pl] "=&r"(pl),
          [ph] "=&r"(ph), [t] "=&r"(t), [a] "+r"(a), [b] "+r"(b)
        :
        : "cc");
    *out = R;
    return (int)ok;
}
#endif /* PRIM_MODE == 24 && LZ_ARM_PRIM_ASM */

#if PRIM_MODE == 26 && defined(LZ_ARM_PRIM_ASM)
#include "spw_fmac_core.h"
#endif /* PRIM_MODE == 26 && LZ_ARM_PRIM_ASM */

#if PRIM_MODE == 25 && defined(LZ_ARM_PRIM_ASM)
/* f32mm armA's cell, fused: fl(acc + fl(w*x)) with the product never
 * packed to memory and never handed to a second special-case scan.
 * Measured by mode 25 against mode 20, which is the same expression
 * through __aeabi_fmul and __addsf3.
 *
 * WHAT THE FUSION ACTUALLY BUYS, now that both halves are written the
 * way libgcc writes them: two call/return pairs, __addsf3's scan of an
 * operand this code just produced and therefore knows is normal, and
 * one shared decline exit instead of two. NOT the product's rounding -
 * skipping that would turn this into an FMA, a different function that
 * build/arm/parity_gate.sh would catch.
 *
 * The product half is spw_fmul_arm's, minus its pack; the sum half is
 * lz_faddsub_arm_asm's, minus the second result. Both use the ADC/BICEQ
 * round-and-pack and the SBC-folded normalisation step. */
static __inline__ int spw_fmac_arm(uint32_t uacc, uint32_t uw, uint32_t ux,
                                   uint32_t *out) {
    uint32_t ok, R, e, t, pl, ph, a = uw, b = ux, c = uacc;
    __asm__(
        /* ---- product ---- */
        "mov    %[t],  #255\n\t"
        "ands   %[e],  %[t], %[a], lsr #23\n\t"
        "andnes %[ph], %[t], %[b], lsr #23\n\t"
        "teqne  %[e],  %[t]\n\t"
        "teqne  %[ph], %[t]\n\t"
        "beq    9f\n\t"
        "add    %[e],  %[e], %[ph]\n\t"
        "eor    %[R],  %[a], %[b]\n\t"
        "mov    %[ph], #0x8000000\n\t"
        "mov    %[a],  %[a], lsl #9\n\t"
        "orr    %[a],  %[ph], %[a], lsr #5\n\t"
        "mov    %[b],  %[b], lsl #9\n\t"
        "orr    %[b],  %[ph], %[b], lsr #5\n\t"
        "umull  %[pl], %[ph], %[a], %[b]\n\t"
        "and    %[R],  %[R], #0x80000000\n\t"
        "cmp    %[ph], #0x800000\n\t"
        "movcc  %[ph], %[ph], lsl #1\n\t"
        "orrcc  %[ph], %[ph], %[pl], lsr #31\n\t"
        "movcc  %[pl], %[pl], lsl #1\n\t"
        "orr    %[R],  %[R], %[ph]\n\t"
        "sbc    %[e],  %[e], #127\n\t"
        "cmp    %[e],  #253\n\t"
        "bhi    9f\n\t"
        "cmp    %[pl], #0x80000000\n\t"
        "adc    %[R],  %[R], %[e], lsl #23\n\t"
        "biceq  %[R],  %[R], #1\n\t"
        /* ---- acc + product ---- */
        "eor    %[ok], %[c], %[R]\n\t"
        "mov    %[ok], %[ok], lsr #31\n\t"     /* 1 when the signs differ */
        "mov    %[a],  %[c], lsl #1\n\t"
        "mov    %[b],  %[R], lsl #1\n\t"
        "sub    %[e],  %[a], #0x01000000\n\t"
        "cmp    %[e],  #0xFE000000\n\t"
        "bhs    9f\n\t"                        /* acc 0/subnormal/inf/nan */
        /* The result's sign is the LARGER operand's, under both
           effective operations: when the signs agree they are the same
           sign, and when they differ the larger one wins. */
        "cmp    %[a],  %[b]\n\t"
        "and    %[pl], %[c], #0x80000000\n\t"
        "andlo  %[pl], %[R], #0x80000000\n\t"
        "movlo  %[e],  %[a]\n\t"
        "movlo  %[a],  %[b]\n\t"
        "movlo  %[b],  %[e]\n\t"
        "mov    %[e],  %[a], lsr #24\n\t"
        "mov    %[ph], %[b], lsr #24\n\t"
        "sub    %[ph], %[e], %[ph]\n\t"        /* d */
        "cmp    %[ph], #27\n\t"
        "bhi    9f\n\t"
        "mov    %[t],  %[b], lsl #8\n\t"
        "mov    %[t],  %[t], lsr #6\n\t"
        "orr    %[t],  %[t], #0x04000000\n\t"
        "rsb    %[b],  %[ph], #32\n\t"
        "mov    %[b],  %[t], lsl %[b]\n\t"
        "mov    %[t],  %[t], lsr %[ph]\n\t"
        "cmp    %[b],  #0\n\t"
        "orrne  %[t],  %[t], #1\n\t"           /* sticky */
        "mov    %[ph], %[a], lsl #8\n\t"
        "mov    %[ph], %[ph], lsr #6\n\t"
        "orr    %[ph], %[ph], #0x04000000\n\t"
        "tst    %[ok], #1\n\t"
        "bne    2f\n\t"
        "add    %[t],  %[ph], %[t]\n\t"        /* effective add */
        "sub    %[a],  %[e], #1\n\t"
        "cmp    %[t],  #0x08000000\n\t"
        "movhs  %[b],  %[t], lsr #1\n\t"
        "andhs  %[t],  %[t], #1\n\t"
        "orrhs  %[t],  %[b], %[t]\n\t"
        "addhs  %[a],  %[a], #1\n\t"
        "b      3f\n\t"
        "2:\n\t"
        "subs   %[t],  %[ph], %[t]\n\t"        /* effective subtract */
        "moveq  %[R],  #0\n\t"                 /* exact cancellation: +0 */
        "beq    4f\n\t"
        "clz    %[b],  %[t]\n\t"
        "sub    %[b],  %[b], #5\n\t"
        "mov    %[t],  %[t], lsl %[b]\n\t"
        "subs   %[a],  %[e], %[b]\n\t"
        "ble    9f\n\t"                        /* subnormal */
        "sub    %[a],  %[a], #1\n\t"
        "3:\n\t"
        "mov    %[b],  %[t], lsl #29\n\t"
        "mov    %[t],  %[t], lsr #3\n\t"
        "cmp    %[a],  #254\n\t"
        "bhs    9f\n\t"
        "orr    %[R],  %[pl], %[t]\n\t"
        "cmp    %[b],  #0x80000000\n\t"
        "adc    %[R],  %[R], %[a], lsl #23\n\t"
        "biceq  %[R],  %[R], #1\n\t"
        "4:\n\t"
        "mov    %[ok], #1\n\t"
        "b      8f\n\t"
        "9:\n\t"
        "mov    %[ok], #0\n\t"
        "8:\n\t"
        : [ok] "=&r"(ok), [R] "=&r"(R), [e] "=&r"(e), [t] "=&r"(t),
          [pl] "=&r"(pl), [ph] "=&r"(ph),
          [a] "+r"(a), [b] "+r"(b), [c] "+r"(c)
        :
        : "cc");
    *out = R;
    return (int)ok;
}
#endif /* PRIM_MODE == 25 && LZ_ARM_PRIM_ASM */
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
#if PRIM_MODE == 27 || PRIM_MODE == 28
    mm_setup();
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
#elif PRIM_MODE == 27 || PRIM_MODE == 28
        /* THE SHIPPED CELL, a whole row rather than one element: the
         * leaf's scan and decline are per group, so a per-element probe
         * would price machinery the real caller does not pay per
         * element. One 64-element row on the first of the harness's 64
         * bodies, so the per-iteration figure is again per ELEMENT.
         * 28 is the same row through the C loop lz_matmul actually
         * has - its own control, because mode 20's loop shape is not
         * this one. */
        {
            int j2, k2;
            /* A volatile read into the row, so the whole computation is
             * not loop-invariant and hoisted out - which it was on the
             * first attempt, and the giveaway was a per-element figure
             * of 25 where the C control alone costs 74. */
            mm_x[0] = g_mulc;
#if PRIM_MODE == 27
            j2 = lz_matmul_row_arm_asm(mm_w, mm_x, MM_N, mm_acc);
#else
            j2 = 0;
            for (k2 = 0; k2 < 8; k2++) mm_acc[k2] = 0.0f;
#endif
            for (; j2 + 7 < MM_N; j2 += 8)
                for (k2 = 0; k2 < 8; k2++) {
                    volatile float p = mm_w[j2 + k2] * mm_x[j2 + k2];
                    mm_acc[k2] = mm_acc[k2] + p;
                }
            g_sink = (int32_t)mm_acc[0];
        }
#elif PRIM_MODE == 26
        /* The fused core with no decline wrapper - an upper bound on
         * what a leaf kernel can reach, against mode 20. */
        {
            union { float f; uint32_t u; } bw, bx, bc;
            bw.f = SP_ARG[k];
            bx.f = g_mulc;
            bc.f = g_facc;
            g_sink = (int32_t)spw_fmac_core(bc.u, bw.u, bx.u);
        }
#elif PRIM_MODE == 25
        /* Mode 20's MAC through the fused cell, fallback included.
         * 25 against 20 is f32mm armA measured rather than projected. */
        {
            union { float f; uint32_t u; } bw, bx, bc;
            uint32_t r;
            bw.f = SP_ARG[k];
            bx.f = g_mulc;
            bc.f = g_facc;
            if (spw_fmac_arm(bc.u, bw.u, bx.u, &r)) {
                g_sink = (int32_t)r;
            } else {
                u.f = bc.f + bw.f * bx.f;   g_sink = u.i;
            }
        }
#elif PRIM_MODE == 24
        /* Mode 16's bare multiply through spw_fmul_arm, fallback
         * included. 24 against 16 is the whole f32mm armA question:
         * a hand core that cannot match __aeabi_fmul settles the cell,
         * because the MAC's shareable part is only the pack/unpack
         * round trip between the two operations. */
        {
            union { float f; uint32_t u; } ba, bb;
            uint32_t r;
            ba.f = SP_ARG[k];
            bb.f = g_mulc;
            if (spw_fmul_arm(ba.u, bb.u, &r)) {
                g_sink = (int32_t)r;
            } else {
                u.f = ba.f * bb.f;   g_sink = u.i;
            }
        }
#elif PRIM_MODE == 21 || PRIM_MODE == 22 || PRIM_MODE == 23
        /* 21 is mode 19's operand pair through the hand-written cell,
         * WITH the fallback the real caller runs - a declined pair pays
         * the check and then the two library calls anyway, so this is
         * the shipped cost and not the fast path's.
         *
         * 22 and 23 are the same two arms over a HADAMARD-LIKE pair.
         * Mode 19 puts a row element against a fixed constant, which
         * spreads the exponent difference across the whole float range;
         * a butterfly pairs two partial sums of one transform, whose
         * magnitudes track each other. Which distribution is measured
         * decides how often the cell declines, so both are here and
         * each has its own C control - 21 against 19, 23 against 22. */
        {
            union { float f; uint32_t u; } ba, bb;
            ba.f = SP_ARG[k];
#if PRIM_MODE == 21
            bb.f = g_mulc;
#else
            bb.f = SP_ARG[k ^ 1];
#endif
#if PRIM_MODE == 22
            u.f = ba.f + bb.f;  g_sink = u.i;
            u.f = ba.f - bb.f;  g_sink ^= u.i;
#else
            {
                uint32_t s, d;
                if (lz_faddsub_arm_asm(ba.u, bb.u, &s, &d)) {
                    g_sink = (int32_t)s;
                    g_sink ^= (int32_t)d;
                } else {
                    u.f = ba.f + bb.f;  g_sink = u.i;
                    u.f = ba.f - bb.f;  g_sink ^= u.i;
                }
            }
#endif
        }
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
