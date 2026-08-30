/* ops_conv1d.c - causal 1-D convolution: the SSM conv block's float and
   fixed tiers. A separate TU from ops.c so the ~500-line block (three
   step functions, the int16 exit, the per-channel scale builder, the
   tier's debug counters) stops sharing a translation unit with the
   attention/recurrence section it sat inside. Same reason ops_gdn.c and
   ops_norm.c exist: operator-to-file ownership that the kernel matrix
   can name.

   These call q8_round/sigmoid_q15_i/sigmoid_q15_t/lz_silu/pow2f from
   ops_quant.c and lz_i32f/epi_shr_half from ops_kernel_shared.h; the
   ARM tier uses the inline primitives from ops_arm_prim.h. The census
   bills this file's work at LZ_FC_CONV. */
#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include <stddef.h>
#include <math.h>
#include "ops.h"
#include "err.h"
#include "mmx_compat.h"
#include "ops_kernel_shared.h"
#include "ops_quant.h"
#include "ops_sched.h"
#if defined(LZ_ARM_PRIM_ASM)
#include "ops_arm_prim.h"
#endif /* LZ_ARM_PRIM_ASM */

void lz_causal_conv1d_step(float *o, const float *x,
                               const float *state_in, float *state_out,
                               const float *w, int n_ch, int k) {
    int hist = k - 1;
    int c, j;
    /* k multiplies and k adds per channel - the first add is against the
       0.0f initializer, which -ffp-contract=off without -ffast-math
       cannot fold away. The old census counted the taps and not the
       adds. silu bills itself. */
    LZ_FCX(LZ_FC_CONV, (long)n_ch * k, (long)n_ch * k, 0, 0, 0);
    for (c = 0; c < n_ch; c++) {
        const float *st_in = state_in + (size_t)c * hist;
        float *st_out = state_out + (size_t)c * hist;
        const float *wr = w + (size_t)c * k;
        float sum = 0.0f;
        for (j = 0; j < hist; j++) sum += wr[j] * st_in[j];
        sum += wr[hist] * x[c];
        o[c] = lz_silu(sum);
        /* shift left by one (reading the OLD history from st_in),
           append the current input at the end (writing to st_out) -
           safe when st_in==st_out too, same forward-order argument as
           the in-place case: iteration j writes st_out[j] and reads
           st_in[j+1], and no earlier iteration ever wrote index j+1
           (only indices < j). */
        for (j = 0; j + 1 < hist; j++) st_out[j] = st_in[j + 1];
        if (hist > 0) st_out[hist - 1] = x[c];
    }
}

/* The integer-coordinate activation's positive control: channel-steps
   the fixed conv epilogue took through sigmoid_q15_i. Zero under
   -DLZ_CONV_SIG_I=0, and zero is also what a float-tier run reports -
   the two arms differ only inside this tier's own tolerance, so no
   output comparison can tell "it ran" from "it did not". Defined
   outside LZ_CONV_FIXED so the front end's extern still links in a
   build that compiles the tier out. */
lz_i64 lz_debug_conv_sig_i = 0;

/* Same reason, and they were inside the #if until a -DLZ_CONV_FIXED=0
   row exists in build/recur_write_float_arm_gate.sh and would not
   LINK: cli_main.c's --debug-counters block reads both unconditionally.
   Nothing caught it earlier because the switch-matrix gate compiles the
   probe with -fsyntax-only, which never reaches the linker.

   lz_debug_convo_i16 is the int16 exit's positive control; the clamp
   counter is not one, because it reads 0 both when the exit is compiled
   out and when it ran without clipping, which is the normal case. */
long lz_conv_o_clamped;
lz_i64 lz_debug_convo_i16;

#if LZ_CONV_FIXED
#if defined(LZ_ARM_PRIM_ASM)
/* Is `v` an exact positive power of two, and if so at what exponent?
 *
 * Asked, not assumed. sig_k1/sig_oscale2 are lz_sig_q15_fold's outputs,
 * which are a pow2f scaled by 2^4 and by 2^-15, so both ARE powers of
 * two today - but that is a fact about a caller in another file, and the
 * exponent-field arithmetic it licenses is silently wrong for anything
 * else. Six integer instructions ask the question at the site that
 * depends on the answer, and a channel that answers no runs the C.
 *
 * `e >= 255` rejects a negative v as well as inf/nan: the sign bit lands
 * in bit 8 of e, so every negative float reads as e >= 256. */
static int conv_pow2_exp(float v, int32_t *k) {
    union { float f; uint32_t u; } b;
    uint32_t e;
    b.f = v;
    if ((b.u & 0x007FFFFFu) != 0u) return 0;
    e = b.u >> 23;
    if (e == 0u || e >= 255u) return 0;
    *k = (int32_t)e - 127;
    return 1;
}
#endif /* LZ_ARM_PRIM_ASM */

/* Fixed tier. The float path above is untouched and stays the control
   arm; this is a separate body, selected by lz_conv_mode().
 *
 * Where the win comes from, and it is not the multiply-adds: keeping the
 * HISTORY in fixed point means one conversion per new sample instead of
 * one per tap. Probe (.prof/wconv.c): 8 f32 operations per channel-step
 * down to 3, drift 4.8e-5 against a 3.9e-3 noise floor. The same shape
 * pass-2's fixed tier uses - its own comment says it does not accelerate
 * the write-back quantize, it deletes it.
 *
 * `sh_in`/`sh_out` are int16 history at exponent `es`, `mw` the int16
 * taps at per-channel exponent `ew`. Both were normalized by
 * lz_conv_norm_pow2 against lz_conv_accum_bound(k), which is what keeps
 * `acc` inside int32 - see that function for the derivation.
 *
 * The x87/SSE agreement argument is the same as the float path's: every
 * float multiply here (input, sig_k1, sig_oscale2) is by an exact power
 * of two, so none of them round.
 *
 * The activation is entered in the integer coordinate (LZ_CONV_SIG_I).
 * `acc` already IS the number the table wants indexing by - x == acc *
 * 2^-(ew[c]+es) - so sigmoid_q15_i takes its bits and the entire float
 * coordinate derivation disappears: three of the epilogue's six
 * int<->float converts per channel, 55,296 a token on kmr20's 18,432
 * channels. What is left is the exit, which has to produce a float:
 * one lz_i32f of the accumulator and one cast of the Q15 result.
 *
 * NOT bit-identical to the float entry below it, and the integer one is
 * the MORE accurate: sigmoid_q15's `x + 12` is exact only while
 * e <= 19, and the measured exponents here run 22..39 (8.24% of kmr20's
 * channels, 11.26% of kunmoe-v2-t2-probe's, sit past 27 and needed
 * sigmoid_q15_i's wide band to exist at all).
 *
 * The =0 arm keeps the folded float entry, which is what the census
 * control build and the pre-change bit-identity comparison run:
 * sigmoid_q15_t takes t = acc*k1 + k2 directly, sig_k1/sig_oscale2
 * (lz_sig_q15_fold) being oscale pre-multiplied by LZ_SIG_STEP and by
 * 1/32768 so acc_f is scaled by each once - bit-identical to the older
 * lz_silu(lz_i32f(acc)*oscale[c]) chain, .prof/verify_conv_fold3.c.
 *
 * `xi` is int-pipeline milestone 3: the producer (an int16-exit matmul,
 * lz_matmul_xq_nt_i16) has already delivered this vector at exponent
 * `es` and clamped to +-bound, so `x`/`in_scale` are not read and the
 * float multiply plus q8_round at the top of the channel loop are not
 * run. NULL keeps the float input this tier was written with. Not a
 * second body: the tap loop, the history shift and the activation are
 * the same code either way - only the source of `q` differs, and it is
 * the caller's job (forward.c) to make sure the two ways of obtaining
 * it are not mixed within one model's conv block. */

/* The int16 exit is a SEPARATE ENTRY POINT rather than two more
   parameters on lz_causal_conv1d_step_fixed, and the reason is the
   call-site count: six of that function's eight callers are probes
   (.prof/fwd_mine.c, .prof/arm_prim_count.c), where a signature change
   is churn with no reader. It stays as a wrapper passing NULL - one
   call per (layer, tensor), not per element, so it costs nothing
   measurable. */
void lz_causal_conv1d_step_fixed_o16(float *o, short *oi, int out_es,
                                 const float *x, const short *xi,
                                 const short *sh_in, short *sh_out,
                                 const LZConvParams *cp, int n_ch) {
    int hist = cp->k - 1;
    int c, j;
#if defined(LZ_ARM_PRIM_ASM)
    /* Hoisted, not per channel: the tier is fixed for the whole call and
       a compare inside the loop would bill 6,144 times a token for an
       answer that cannot change. */
    int prim = (g_kernel == LZ_KERNEL_ARM_ASM);
#endif /* LZ_ARM_PRIM_ASM */
#if LZ_CONV_SIG_I
    /* The folded float coordinate's two factors, unread here and kept in
       the signature for the =0 arm below - the caller builds both tables
       and one build must not need a different call shape from the other. */
    (void)cp->sig_k1; (void)cp->sig_k2;
    /* Per channel: input multiply (1), lz_i32f (1 cvt), the exit cast
       (1 cvt) and sig_oscale2 multiply (1), and the final product (1) =
       3 mul, 0 add, 2 cvt. The float entry's own three - sig_k1 multiply,
       sig_k2 add, sigmoid_q15_t's frac interpolation - are integers now
       and bill nothing, exactly as sigmoid_q15_i bills nothing at
       lz_swiglu_q15_i16's site. The int16 input drops the first multiply.
       q8_round bills nothing of its own (its magic add is not counted at
       any of its call sites), so 2 is the whole difference. */
    /* THE INT16 EXIT BILLS ITS OWN SHAPE, in the same commit as the
       code: with `oi` the three float operations after the accumulator
       - lz_i32f, the exit cast and the two multiplies - are a 64-bit
       integer product and a shift, so all that is left is the input
       multiply the float entry needs, and nothing at all with xi. A
       formula that kept charging the previous shape would report this
       change as worth zero, which is what the first run of it did. */
    if (oi) {
        LZ_FCX(LZ_FC_CONV, (long)n_ch * (xi ? 0 : 1), 0, 0, 0, 0);
        lz_debug_convo_i16 += n_ch;
    } else
        LZ_FCX(LZ_FC_CONV, (long)n_ch * (xi ? 2 : 3), 0, 0,
               (long)n_ch * 2, 0);
    lz_debug_conv_sig_i += n_ch;
#else
    (void)cp->sig_e;   /* the integer coordinate's table, unread in this arm */
    /* Per channel: input multiply (1), lz_i32f (1 cvt), sig_k1 multiply
       + sig_k2 add (the folded entry, 1 mul + 1 add), sigmoid_q15_t's
       own frac-interpolation multiply (1, not itemized here, same
       convention lz_sigmoid's flat bill already used for it), the exit
       cast (1 cvt) and sig_oscale2 multiply (1), and the final product
       (1) = 5 mul, 1 add, 2 cvt. The float entry costs 2 mul + 1 cvt
       here plus 4 mul + 1 add + 1 cvt at the sigmoid site through
       lz_silu, 9 in all, and reaches the sigmoid site; this one does
       not. The int16 input
       drops the first of those five multiplies; q8_round bills nothing
       of its own (its magic add is not counted at any of its call
       sites), so 4 is the whole difference. */
    LZ_FCX(LZ_FC_CONV, (long)n_ch * (xi ? 4 : 5), (long)n_ch, 0,
           (long)n_ch * 2, 0);
#endif /* LZ_CONV_SIG_I */
    for (c = 0; c < n_ch; c++) {
        const short *si = sh_in + (size_t)c * hist;
        short *so = sh_out + (size_t)c * hist;
        const short *wr = cp->mw + (size_t)c * cp->k;
        int acc = 0, q;
        if (xi) {
            q = xi[c];
        } else {
            float t = x[c] * cp->in_scale;
            q = q8_round(t);
            if (q > cp->bound) q = cp->bound;
            if (q < -cp->bound) q = -cp->bound;
        }
        for (j = 0; j < hist; j++) acc += (int)wr[j] * (int)si[j];
        acc += (int)wr[hist] * q;
        {
#if LZ_CONV_SIG_I
            /* x == acc * 2^-sig_e[c] by construction (sig_e is ew[c]+es,
               the exponent lz_sig_q15_fold turned into 2^(4-e) for the
               float entry), so the table index comes out of acc's own
               bits and the coordinate costs nothing. */
            int32_t s15 = sigmoid_q15_i(acc, cp->sig_e[c]);
            int done = 0;
            /* INT16 EXIT, and there is no float anywhere in it. The
               output is acc * s15 * sig_oscale2[c], and sig_oscale2 is
               an EXACT power of two by construction - forward.c builds
               oscale = ldexp(1, -(ew+LZ_CONV_ES)) and lz_sig_q15_fold
               divides it by 32768 - so its exponent is -(sig_e[c]+15)
               and conv_pow2_exp does not need to be asked. What is left
               is a 32x32 product and a shift.

               The product needs 64 bits: acc is int32 and s15 reaches
               32768, so it can hold 2^46. epi_shr_half is the same
               half-away-from-zero the fixed epilogue rounds with.

               The clamp is counted, not silent. LZ_CONVO_ES was chosen
               from .prof/convqk_range.c's clamp column at zero on both
               chains, and a checkpoint whose conv outputs are scaled
               differently would clip here with no other symptom - the
               same trap LZ_CONV_ES's own comment names. */
            if (oi) {
                lz_i64 p = (lz_i64)acc * (lz_i64)s15;
                lz_i64 r = epi_shr_half(p, (int)cp->sig_e[c] + 15 - out_es);
                if (r > 32767)       { r = 32767;  lz_conv_o_clamped++; }
                else if (r < -32767) { r = -32767; lz_conv_o_clamped++; }
                oi[c] = (short)r;
                done = 1;
            }
#if defined(LZ_ARM_PRIM_ASM)
            /* The remaining multiply is an exponent add. It scales acc_f
               by an exact power of two, which is exact in IEEE, so
               fl(acc_f * 2^oe) IS acc_f with oe added to its exponent
               field - the same number the C's __aeabi_fmul arrives at,
               not an approximation of it. lz_scale2_arm_asm declines
               every case where the add is not the whole story (a zero or
               subnormal acc_f, a result that leaves the normal range),
               and a decline here drops the channel to the C below.
               (float)s15 is a correctly-rounded int32->float32 either
               way, so lz_i32f_arm_asm and the cast agree over the whole
               int32 domain - s15's own [0, 32768] range is not what
               makes that true and is not relied on.

               No x-domain float entry in this branch - it would be a
               second lz_scale2_arm_asm to build x, then sigmoid_q15,
               and the C has none either: BOTH TIERS CALL sigmoid_q15_i,
               so the two agree by construction instead of by the
               table-edge coincidence .prof/conv_epi_xentry.sh had to
               sweep. One soft-float call left where the C tier makes
               three, against docs/arm-asm-audit.md E2's five. */
            if (prim) {
                union { float f; uint32_t u; } ab, pb;
                int32_t oe;
                if (conv_pow2_exp(cp->sig_oscale2[c], &oe)) {
                    ab.f = lz_i32f_arm_asm(acc);
                    if (lz_scale2_arm_asm(ab.u, oe, &pb.u)) {
                        o[c] = pb.f * lz_i32f_arm_asm(s15);
                        done = 1;
                    }
                }
            }
#endif /* LZ_ARM_PRIM_ASM */
            if (!done) o[c] = (lz_i32f(acc) * cp->sig_oscale2[c]) * (float)s15;
#else /* !LZ_CONV_SIG_I */
            int done = 0;
#if defined(LZ_ARM_PRIM_ASM)
            /* The two multiplies are exponent adds. Both scale the same
               acc_f by an exact power of two, which is exact in IEEE, so
               fl(acc_f * 2^k) IS acc_f with k added to its exponent
               field - the same number the C's __aeabi_fmul arrives at,
               not an approximation of it. lz_scale2_arm_asm declines
               every case where the add is not the whole story (a zero or
               subnormal acc_f, a result that leaves the normal range),
               and a decline here drops the channel to the C below.
               (float)s15 is a correctly-rounded int32->float32 either
               way, so lz_i32f_arm_asm and the cast agree over the whole
               int32 domain - s15's own [0, 32768] range is not what
               makes that true and is not relied on.

               The sigmoid is entered in the x domain, not the folded
               t one, which is what deletes the `+ sig_k2` __aeabi_fadd
               rather than moving it: sigmoid_q15's own assembly tier
               does that add in the integer domain (lz_sig_q15_arm_asm's
               "+12" case analysis) while sigmoid_q15_t's cannot, because
               by the time t arrives the add has already happened. The
               two entries are the same function here - t is exactly
               16*fl(x+12) and sigmoid_q15's coordinate IS 16*fl(x+12),
               scaling by a power of two commuting with rounding - with
               one fact left over that is about the TABLE and not the
               algebra: the upper clamps return a literal 32768 and
               g_sigtab[384] respectively, and those are equal only
               because sig_build makes them so. .prof/conv_epi_xentry.sh
               sweeps both halves and requires a mutation of each to go
               red. `ke - 4` is oscale's exponent, sig_k1 being oscale
               scaled by LZ_SIG_STEP == 2^4.

               What is left: sigmoid_q15 (which has its own assembly
               tier) and the final multiply, whose operands are both
               variable. Two calls where the C tier makes seven, and one
               soft-float call where it makes five. Measured 253.8 ->
               229.7 instructions per channel, docs/arm-asm-audit.md E2. */
            if (prim) {
                union { float f; uint32_t u; } ab, xb, pb;
                int32_t ke, oe;
                if (conv_pow2_exp(cp->sig_k1[c], &ke) &&
                    conv_pow2_exp(cp->sig_oscale2[c], &oe)) {
                    ab.f = lz_i32f_arm_asm(acc);
                    if (lz_scale2_arm_asm(ab.u, ke - 4, &xb.u) &&
                        lz_scale2_arm_asm(ab.u, oe, &pb.u)) {
                        o[c] = pb.f * lz_i32f_arm_asm(sigmoid_q15(xb.f));
                        done = 1;
                    }
                }
            }
#endif /* LZ_ARM_PRIM_ASM */
            if (!done) {
                float acc_f = lz_i32f(acc);
                int32_t s15 = sigmoid_q15_t(acc_f * cp->sig_k1[c] + cp->sig_k2);
                o[c] = (acc_f * cp->sig_oscale2[c]) * (float)s15;
            }
#endif /* LZ_CONV_SIG_I */
        }
        for (j = 0; j + 1 < hist; j++) so[j] = si[j + 1];
        if (hist > 0) so[hist - 1] = (short)q;
    }
}

void lz_causal_conv1d_step_fixed(float *o, const float *x, const short *xi,
                                 const short *sh_in, short *sh_out,
                                 const short *mw, const float *sig_k1,
                                 const float *sig_oscale2, float sig_k2,
                                 const signed char *sig_e,
                                 float in_scale, int bound,
                                 int n_ch, int k) {
    LZConvParams cp;
    cp.mw = mw; cp.sig_k1 = sig_k1; cp.sig_oscale2 = sig_oscale2;
    cp.sig_k2 = sig_k2; cp.sig_e = sig_e;
    cp.in_scale = in_scale; cp.bound = bound; cp.k = k;
    lz_causal_conv1d_step_fixed_o16(o, (short *)0, 0, x, xi, sh_in, sh_out,
                                    &cp, n_ch);
}

/* Build the per-channel output scale 2^-(ew[c]+es) and the input scale
   2^es, once, so the hot loop has no shift in it.
 *
 * NOT `1 << (ew + es)`: lz_conv_norm_pow2 caps its exponent at 30, so
 * the sum reaches 43 and that shift is undefined on a 32-bit int. The
 * first version of the kernel did exactly this and produced max drift
 * 4.2e8 on a uniform state while looking correct on a decayed one -
 * caught only because .prof/wconv2.c drives the shipped function rather
 * than a copy of it. ldexp walks the exponent field instead, and runs
 * once per channel at setup rather than per token. */
void lz_conv_build_scales(float *oscale, float *in_scale,
                          const signed char *ew, int es, int n_ch) {
    int c;
    *in_scale = pow2f(es);
    for (c = 0; c < n_ch; c++)
        oscale[c] = pow2f(-((int)ew[c] + es));
}
#endif /* LZ_CONV_FIXED */

/* Accumulator bound for the fixed conv tier, derived not chosen.
   k products of two values bounded by B reach k*B^2, which must stay
   inside int32: B <= sqrt((2^31-1)/k). For k=4 that is 23170, so the
   largest usable power of two is 2^14. Probe (.prof/wconv.c) measures a
   peak of 9.09e8 against the 1.07e9 bound - tight, never crossed, 2.36x
   inside int32.

   int64 accumulation is not the escape: ARMv5TE has SMLAL, but MMX's
   pmaddwd yields int32 pairs that wrap the same way, so the x86 side
   would need a carry chain per tap. One bound for both is cheaper. */
int lz_conv_accum_bound(int k) {
    int b = 1;
    while ((int64_t)k * (b * 2) * (b * 2) <= (int64_t)2147483647)
        b *= 2;
    return b;
}

/* Normalize n floats into int16 by a POWER OF TWO, clamped to +-bound.
   Returns e, so v[i] ~ out[i] * 2^-e. Power of two only: the scale is
   exact in binary, so this adds no rounding of its own and cannot
   differ between x87 and SSE. */
int lz_conv_norm_pow2(const float *v, int n, short *out, int bound) {
    /* q8_amax, not a hand-written scan: two __aeabi_fcmpXX per ELEMENT
       otherwise, and this was the fifth copy of the same loop. See
       epi_pack_act for why the integer-domain form is exact. */
    union { float f; uint32_t u; } ab;
    float amax;
    int i, e = 0;
    ab.u = q8_amax(v, n);
    amax = ab.f;
    if (amax == 0.0f) {
        for (i = 0; i < n; i++) out[i] = 0;
        return 0;
    }
    while (e < 30 && amax * (float)(1 << (e + 1)) < (float)bound) e++;
    for (i = 0; i < n; i++) {
        float t = v[i] * (float)(1 << e);
        int q = q8_round(t);
        if (q > bound) q = bound;
        if (q < -bound) q = -bound;
        out[i] = (short)q;
    }
    return e;
}
