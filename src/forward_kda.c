#include <string.h>

#include "forward.h"
#include "ops.h"
#include "ops_quant.h"  /* lz_quantize_q8(_int/_i16), lz_sigmoid */
#include "fwht.h"        /* lz_fwht, the SubLN Hadamard kernel */

/* KDA (linear-attention) + dense-FFN paths, moved out of forward.c (the
   four-path split). Pure code motion: the subn_* and kda_* helpers stay
   static; forward_kda and dense_ffn_step lost `static` because lz_forward
   / lz_mtp_draft_step still call them. */

/* ------------------------------------------------------ KDA (LZ_LT_KDA) block */

/* SubLN block-diagonal Hadamard + per-token absmax INT8 quantize for the
   o_proj / down_proj input (BitNet v2 H-BitLinear, minus the INT4 stage).
   x is the post-RMSNorm float activation row, n its width, blk the
   Hadamard block (a power of two dividing n; the training side caps it at
   512 for down_proj and uses value_dim for o_proj), fscr 2*n int32
   scratch.

   The transform is the fixed-point FWHT (lz_fwht_i32): the row is moved
   to int32 at a per-row power-of-two scale (row max ~2^14, so the
   unnormalized output's worst case blk*2^14 stays inside 2^24 and every
   int32->float is exact), then each blk-wide block butterflies in pure
   add/sub. The unnormalized transform grows RMS by sqrt(blk); 1/sqrt(blk)
   is folded into the STORED scale (the H.264 pattern), so the kernels
   stay integer and the dequant matches the training side's orthonormal
   _block_hadamard within quantization tolerance.

   gw is the width of one stored scale: n for BitNet v2's per-token
   absmax, or the engine's own activation group (32) when --subn-scale
   group is on. It is NOT the Hadamard block - the transform runs
   identically either way, and the grouping applies to the ROTATED int32
   values, where it is exact: within a group the dequant q[i]*s[g] still
   reproduces fscr[i]*inv_S/sqrt(blk), because inv_S and sqrt(blk) are
   row-wide constants and only the 127/tam fold is per group. Whichever
   gw is used, one xqs slot is still written per 32 elements, which is
   what the matmul reads.

   q/s semantics match lz_quantize_q8 (round-half-even, clamp
   [-127,127], all-zero group scale 1.0). */
static void subn_fwht_quant(const float *x, int n, int blk, int gw,
                            int8_t *q, float *s, int32_t *fscr) {
    union { float f; uint32_t u; } b;
    float amax, S, inv_S, inv_q, scale, rs;
    int e, i, b_i, base;
    b.u = q8_amax(x, n);
    amax = b.f;
    if (amax <= 0.0f) {          /* all-zero row: q all 0, scale 1.0 */
        for (i = 0; i < n; i++) q[i] = 0;
        for (i = 0; i < n / 32; i++) s[i] = 1.0f;
        return;
    }
    /* Power-of-two fixed-point scale: row max lands in [2^14, 2^15), so
       the worst-case unnormalized FWHT output is blk*2^15 <= 2^24. */
    e = (int)((b.u >> 23) & 0xFFu) - 127;
    S = pow2f(14 - e);
    inv_S = 1.0f / S;            /* exact: S is a power of two */
    for (i = 0; i < n; i++) fscr[i] = q8_round(x[i] * S);
    for (b_i = 0; b_i < n; b_i += blk)          /* block-diagonal Sylvester */
        lz_fwht_i32(fscr + n + b_i, fscr + b_i, blk);
    if (gw <= 0 || (gw & 31) != 0 || (n % gw) != 0) gw = n;
    /* Hoisted, not folded into a single constant: the multiply chain
       below must stay (tam*inv_S)*rs*INV127 in that order, or the
       per-token arm stops being bit-identical to the engine that had no
       gw. lz_rsqrt_float_body is pure and blk is row-wide, so calling it
       once here is the same call the row made before. */
    rs = lz_rsqrt_float_body((float)blk);
    for (base = 0; base < n; base += gw) {
        int32_t tam = 0;
        for (i = base; i < base + gw; i++) {
            int32_t a = fscr[n + i] < 0 ? (int32_t)(0u - (uint32_t)fscr[n + i])
                                        : fscr[n + i];
            if (a > tam) tam = a;
        }
        /* Dequant scale = post-Hadamard amax / sqrt(blk) / 127. The
           element loop rounds t*127/tam, identical to
           lz_quantize_q8_int's fold. tam == 0 is the group's fixed-point
           collapse and takes the same scale-1.0 exit q8_group_scale
           gives a zero group. */
        scale = (float)tam * inv_S * rs * LZ_Q8_INV127;
        if (tam == 0 || scale < LZ_Q8_MIN_SCALE_F) {
            scale = 1.0f;
            inv_q = 0.0f;
        } else {
            inv_q = 127.0f / (float)tam;
        }
        for (i = base; i < base + gw; i++) {
            int qv = q8_round((float)fscr[n + i] * inv_q);
            qv = (qv > 127) ? 127 : qv;
            q[i] = (int8_t)((qv < -127) ? -127 : qv);
        }
        for (i = base / 32; i < (base + gw) / 32; i++) s[i] = scale;
    }
}

/* Width of one stored activation scale on the SubLN o_proj/down_proj
   input, resolved from --subn-scale. 0 means "the whole row" and is
   what subn_fwht_quant / lz_quantize_q8_tok already do; gs is the
   site's lz_act_gs, which is 0 when the width is not a multiple of 32.
   Returning 0 for a gs this path cannot honour keeps the fallback in
   ONE place instead of at each of the four call sites. */
static int subn_scale_gs(int gs, int n) {
    if (!lz_subn_scale_group()) return 0;
    if (gs <= 0 || (gs & 31) != 0 || n < gs || (n % gs) != 0) return 0;
    return gs;
}

/* The SAME block-diagonal Hadamard, applied to the float row.
   subn_fwht_quant writes q/s and never touches x, so on the one path
   where lz_matmul_xq_nt reads x - an F32 weight - the rotation would
   otherwise not happen at all, and the model would compute W.x while
   the quantized build of the same config computes W.quant(Hx). Those
   are different functions, which is what made the unquantized run
   useless as a reference for the quantized one.

   Float, not the fixed-point twin, because this row IS the reference:
   lz_fwht takes no constants, so it is bit-identical across
   wcc386 and gcc, and the orthonormal form matches the training side's
   _block_hadamard exactly rather than within quantization tolerance.
   The 1/sqrt(blk) uses lz_rsqrt_float_body, the same call
   subn_fwht_quant folds into its stored scale, so the two paths differ
   by the transform's arithmetic and NOT by the normalisation constant.

   In place: the caller has already quantized from this row, and the
   matmul is the only reader left. */
static void subn_rot_f32(float *x, int n, int blk) {
    float inv = lz_rsqrt_float_body((float)blk);
    int b_i, i;
    for (b_i = 0; b_i < n; b_i += blk) lz_fwht(x + b_i, blk);
    for (i = 0; i < n; i++) x[i] *= inv;
}

/* Kimi Delta Attention: same role as forward_ssm's GatedDeltaNet block,
   generalized to a per-channel decay gate and three independent q/k/v
   projections (KdaAttention). See ops.h's
   lz_kda_step for the recurrence derivation - this function does not
   re-derive it, only wires the projections/conv/gate around it.

   Batches across nt for every projection that reads hidden-space input
   directly (q/k/v/f_a/b/g projections all share one xb quantization,
   the same trick forward_ssm/forward_attn use for their own hidden-space
   projections); f_b_proj and the conv are NOT nt-batched - f_b_proj
   because its input (kda_gate_lat) is a different width than hidden and
   needs its own quantization per token, the conv because it is
   inherently serial in t (same reasoning as forward_ssm's conv). */
/* One q/k/v projection feeding the fixed conv, int-pipeline milestone 3.
   With `*use_int` the matmul exits as int16 already at LZ_CONV_ES and
   clamped to s->conv_bound, and the conv takes it as-is: the producer's
   `(float)comb * pow2f(target)` and the consumer's `x[c] * in_scale` +
   q8_round both stop existing rather than moving somewhere else.

   All three or none, enforced by the flag. A refusal here
   cannot be recovered from per-projection: under SubLN each projection
   re-quantizes s->xq/s->xqs from its own norm, so an earlier one's float
   buffer can no longer be recomputed by the time a later one refuses.
   It also cannot happen - lz_matmul_xq_i16_ok is asked for q, k and v
   together before the first norm runs, and it IS the predicate the entry
   point asks, whose answer depends only on (dtype, in_dim, nt), the same
   three for all of q/k/v. Clearing the flag on the way to the float
   matmul is therefore correct for the only reachable case (the FIRST
   projection refusing, which sends the whole triple to float) and is
   here so that a future divergence degrades instead of reading a stale
   buffer. Same argument the fixed conv tier's own GDN comment makes
   about not splitting one SSM block across two tiers. */
static void kda_conv_proj(LZRunState *s, int *use_int, float *of, short *oi,
                          const float *x, const LZTensor *w,
                          int in_dim, int out_dim, int nt) {
#if LZ_CONV_FIXED
    if (*use_int) {
        if (lz_matmul_xq_nt_i16(oi, s->xq, s->xqs, w, in_dim, out_dim, nt,
                                LZ_CONV_ES, s->conv_bound))
            return;
        *use_int = 0;
    }
#else
    (void)use_int; (void)oi;
#endif /* LZ_CONV_FIXED */
    lz_matmul_xq_nt(of, x, s->xq, s->xqs, w, in_dim, out_dim, nt);
}

/* One SubLN input norm plus the activation quantization its projection
   reads. q, k, v and g each want this with a different norm weight and
   nothing else different, and kda_conv_proj above is the matmul half of
   the same four-way repetition.

   TWO ARMS, and which one ran decides where the result is. The norms
   tier writes a Q15 int product straight into s->subn_norm_int (single
   token, not nt-wide - see forward.h), the immediate tok-int quantize
   reads that, and s->xb2 IS NEVER WRITTEN. The float arm puts the norm
   in s->xb2 and quantizes from there. Callers pass s->xb2 to the matmul
   either way, which is correct only because a matmul handed xq/xqs
   never reads the float buffer - that is the invariant to check first
   if a caller here ever starts using s->xb2 for something else.

   `use_int` is the caller's, evaluated once for all four: it depends on
   hdim and gsa, not on which weight is being applied. */
static void subn_norm_q8(LZRunState *s, const LZTensor *nrm,
                         int hdim, int nt, int nsa, float eps, int use_int) {
    int tk;
    if (use_int) {
        for (tk = 0; tk < nt; tk++) {
            float deq;
            lz_rmsnorm_int(s->subn_norm_int, s->xb + (size_t)tk * hdim,
                           lz_t_f32(nrm, s->wscr), hdim, eps, &deq);
            lz_quantize_q8_tok_int(s->subn_norm_int, hdim, deq,
                                   s->xq + (size_t)tk * hdim,
                                   s->xqs + (size_t)tk * nsa);
        }
        return;
    }
    for (tk = 0; tk < nt; tk++)
        lz_rmsnorm(s->xb2 + (size_t)tk * hdim, s->xb + (size_t)tk * hdim,
                   lz_t_f32(nrm, s->wscr), hdim, eps);
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8_tok(s->xb2 + (size_t)tk * hdim, hdim,
                           s->xq + (size_t)tk * hdim,
                           s->xqs + (size_t)tk * nsa);
}

/* The same shape for the projections whose consumer is not the conv
   (int-pipeline milestone 5): one projection, one int16 exit at the
   consumer's own exponent, and the SAME all-or-nothing flag discipline
   kda_conv_proj documents at length.

   The capability comes IN as a value (`may_int`, from
   lz_matmul_xq_i16_ok); `*wrote_int` goes OUT and means one thing only:
   this call put its result in `oi`. Two separate objects on purpose. A
   single in/out flag reads as the capability everywhere it is set and
   as "the int buffer holds the output" everywhere it is consumed, so a
   branch that writes `of` WITHOUT coming through here leaves a
   capability behind for the consumer to read as a promise, and the
   consumer takes a stale int buffer. Splitting them makes that
   unrepresentable: `*wrote_int` starts at 0, nothing but this function
   ever raises it, and adding such a branch is now a compile error at
   the call rather than a silent wrong read at the consumer. */
void proj_i16(LZRunState *s, int may_int, int *wrote_int,
                         float *of, short *oi,
                         const float *x, const LZTensor *w,
                         int in_dim, int out_dim, int nt,
                         int target_e, int bound) {
    if (may_int &&
        lz_matmul_xq_nt_i16(oi, s->xq, s->xqs, w, in_dim, out_dim, nt,
                            target_e, bound)) {
        *wrote_int = 1;
        return;
    }
    *wrote_int = 0;
    lz_matmul_xq_nt(of, x, s->xq, s->xqs, w, in_dim, out_dim, nt);
}

/* Build the per-channel decay gate gt = f(A_log, dt_bias, pre) and
   leave it in s->kda_gate, plus the head-only decay_base in s->wscr.
   Extracted from forward_kda as pure code motion. */
/* The float path of kda_decay_gate_build, extracted so the gate-int
   ladder does not also carry a full second triple loop. See the parent
   for why the int path exists at all. */
static void kda_gate_build_float(LZRunState *s, const LZModelConfig *c,
                                 const float *decay_base, const float *dt_bias_arr,
                                 int nv, int kd, int nt) {
    int nvk = nv * kd;
    int h, i, tk;
    for (tk = 0; tk < nt; tk++) {
        float *gt = s->kda_gate + (size_t)tk * nvk;
        for (h = 0; h < nv; h++) {
            for (i = 0; i < kd; i++) {
                int idx = h * kd + i;
                float pre = gt[idx] + dt_bias_arr[idx];
                float g;
                if (c->kda_has_gate_lower_bound)
                    g = c->kda_gate_lower_bound * lz_sigmoid(decay_base[h] * pre);
                else
                    g = -decay_base[h] * lz_softplus(pre);
                gt[idx] = lz_exp(g);
            }
        }
    }
}

/* The gate-int arm of kda_decay_gate_build, extracted for the same
   reason as kda_gate_build_float: the parent is a two-way dispatch, not
   a ladder carrying both triple loops. The folding this body implements
   is the parent's own comment block (int-pipeline milestone 5). */
static void kda_gate_build_int(LZRunState *s, const LZModelConfig *c,
                               const float *decay_base, int li,
                               int nv, int kd, int nt) {
    int nvk = nv * kd;
    int h, i, tk;
    float lbq = c->kda_gate_lower_bound * (1.0f / 32768.0f);
    float gk = pow2f(4 - LZ_KGATE_ES);
    float t_off = lz_sig_q15_t_offset();
    const short *dtb = s->kda_dtb_i16 + (size_t)li * nvk;
#if LZ_KGATE_EXP_I
    /* The exp's fold: x == s15 * lower_bound * 2^-15, one constant for
       the whole model, so this is asked here only because here is where
       decay_base is built. lz_scalar_mode because lz_exp_t is
       lz_exp_FIXED's twin - with the scalar tier off, lz_exp is the
       float body and taking the integer entry would answer from the
       other tier. */
    int32_t em = 0;
    int es = 0;
    int exp_int = lz_scalar_mode() &&
                  lz_exp_t_fold(c->kda_gate_lower_bound, -15, &em, &es);
    if (lz_scalar_mode() && !exp_int) lz_debug_kgate_fold_no++;
    if (exp_int) lz_debug_kgate_exp_s = es;
#endif /* LZ_KGATE_EXP_I */
    for (tk = 0; tk < nt; tk++) {
        const short *gi = s->kda_gate_i16 + (size_t)tk * nvk;
        float *gt = s->kda_gate + (size_t)tk * nvk;
        for (h = 0; h < nv; h++) {
            float k1;
#if LZ_KGATE_EXP_I
            int32_t sm = 0;
            int ss = 0;
            /* The sigmoid's fold: x == pre * decay_base[h] *
               2^-LZ_KGATE_ES, per head. */
            int sig_int = exp_int &&
                          lz_fold_f32(decay_base[h], -LZ_KGATE_ES, &sm, &ss);
            if (exp_int && !sig_int) lz_debug_kgate_fold_no++;
            if (sig_int) {
                /* Integer to the last line of lz_exp_t. The site bills
                   nothing because there is nothing to bill - lz_exp_t
                   bills its own exit, exactly as sigmoid_q15_i bills
                   nothing at the conv epilogue's integer coordinate. */
                if (ss < lz_debug_kgate_sig_slo) lz_debug_kgate_sig_slo = ss;
                if (ss > lz_debug_kgate_sig_shi) lz_debug_kgate_sig_shi = ss;
                lz_debug_kgate_exp_i += kd;
                for (i = 0; i < kd; i++) {
                    int idx = h * kd + i;
                    int32_t pre = (int32_t)gi[idx] + (int32_t)dtb[idx];
                    gt[idx] = lz_exp_t(sigmoid_q15_ti(pre, sm, ss), em, es);
                }
                continue;
            }
#endif /* LZ_KGATE_EXP_I */
            k1 = decay_base[h] * gk;
            LZ_FCX(LZ_FC_SIGMOID, 3 * (long)kd, (long)kd, 0,
                   2 * (long)kd, 0);
            for (i = 0; i < kd; i++) {
                int idx = h * kd + i;
                /* A BARE cast, and the bound is why: both addends are
                   int16, so |pre| <= 65534, far inside the 2^24 where an
                   int32->float conversion is exact on x87 and SSE alike.
                   lz_i32f exists for the conversions that are NOT exact -
                   it costs two converts, a multiply and an add of its
                   own, so using it where the hazard cannot occur would
                   spend four float ops to avoid nothing. Past the
                   conversion both compilers round the multiply once and
                   the add once (PC=24). */
                int32_t pre = (int32_t)gi[idx] + (int32_t)dtb[idx];
                int32_t s15 = sigmoid_q15_t((float)pre * k1 + t_off);
                gt[idx] = lz_exp((float)s15 * lbq);
            }
        }
    }
}

static void kda_decay_gate_build(LZRunState *s, const LZModelConfig *c,
                                 const float *a_log_arr, const float *dt_bias_arr,
                                 int li, int nv, int kd, int nt, int gate_int) {
    int h;
    {
        float *decay_base = s->wscr;
        for (h = 0; h < nv; h++) decay_base[h] = lz_exp(a_log_arr[h]);
        /* The int-pipeline form, now kda_gate_build_int (int-pipeline
           milestone 5). With kda_gate arriving as int16 at
           2^-LZ_KGATE_ES and dt_bias pre-quantized into that same domain
           (kda_gate_build), `pre` is an INTEGER add, and the sigmoid is
           reached through sigmoid_q15_t's folded coordinate the way the
           fixed conv reaches it:
             t = pre_i * (decay_base[h] * 2^(4-ES)) + 192
           reproduces (decay_base[h]*pre - LZ_SIG_LO) * LZ_SIG_STEP with
           one multiply and one add, because both LZ_SIG_STEP and the
           offset are exact powers of two applied to an already-computed
           value (lz_sig_q15_t_offset's own argument). The exit folds the
           lower bound and the Q15 descale into one constant.
           Per channel: 3 mul, 1 add, 2 cvt, against the float form's
           add + multiply here and lz_sigmoid's 3 mul, 1 add, 1 cvt plus
           the epilogue's convert and multiply. */
        if (gate_int) {
            kda_gate_build_int(s, c, decay_base, li, nv, kd, nt);
        } else {
            kda_gate_build_float(s, c, decay_base, dt_bias_arr, nv, kd, nt);
        }
    }
}

/* advance_ring/ring_base: same contract as forward_ssm's own (see its
   header comment above) - reused verbatim, not re-derived, since both
   functions share the SAME ring buffers (ssm_state_q8/_lo/_s,
   conv_state) and the SAME per-chunk base forward_chunk computes once. */
void forward_kda(const LZModel *m, LZRunState *s,
                        const LZLayer *L, int layer, int nt,
                        int advance_ring, int ring_base) {
    const LZModelConfig *c = &m->config;
    int nk = c->lin_n_k_heads, nv = c->lin_n_v_heads;
    int kd = c->lin_k_head_dim, vd = c->lin_v_head_dim;
    int gate_rank = c->kda_gate_rank;
    /* Hoisted config scalar, passed to every norm call below: Watcom
       pushes a loop-invariant field as `push [mem]` (X86_TUNE_PUSH_MEMORY)
       and ARM reloads `ldr [c,#offset]` at each call site unless the
       value is named once here and carried in a register. */
    float rms_eps = c->rms_norm_eps;
    int li = s->cache_idx[layer];
    int ring_depth = s->ssm_ring_depth;
    size_t q8_slot_stride = (size_t)c->n_linear_layers * nv * kd * vd;
    size_t s_slot_stride  = q8_slot_stride / 32;
    size_t conv_slot_stride = (size_t)c->n_linear_layers *
                              c->lin_conv_dim * (c->conv_kernel - 1);
    size_t li_off_q8 = (size_t)li * nv * kd * vd;
    /* Same per-layer conv_state allocation GDN uses (sized by
       lin_conv_dim regardless of how that width is split), sliced into
       three independent regions [q | k | v] - matching the concat order
       kunmoe_modeling.py's own decode path uses for the same reason:
       three depthwise convs over disjoint channels == one depthwise
       conv over their concatenation, so the state layout is identical
       either way. hist/li_off_conv/q_off/k_off/v_off are the pieces a
       ring-slot base gets added to below - kept as offsets rather than
       resolved pointers because which slot applies varies per token
       (advance_ring). */
    int hist = c->conv_kernel - 1;
    size_t li_off_conv = (size_t)li * c->lin_conv_dim * hist;
    size_t q_off = 0;
    size_t k_off = (size_t)c->lin_key_dim * hist;
    size_t v_off = (size_t)2 * c->lin_key_dim * hist;
#if LZ_CONV_FIXED
    /* Packed fixed-conv geometry (one per q/k/v channel range), filled
       before the tk loop so each per-token call passes one struct pointer
       instead of seven scalar pushes - see LZConvParams in ops.h. */
    LZConvParams cp_q, cp_k, cp_v;
#endif /* LZ_CONV_FIXED */
    float scale = s->ssm_scale;
    int hdim = c->hidden_size, kdim = c->lin_key_dim, vdim = c->lin_value_dim;
    int nvk = nv * kd;
    int gsa = lz_act_gs(&L->kda_q_proj, hdim);
    int nsa = scale_groups(hdim, gsa);
    int h, i, tk;
    /* int-pipeline milestone 3: does the whole q/k/v triple qualify for
       the int16 matmul exit the fixed conv can consume directly? Decided
       ONCE, here, before any projection runs - see kda_conv_proj. The
       fixed conv tier is the consumer, so its absence answers this on its
       own; the three tensor questions are asked in full (no short-circuit
       reasoning from "they are the same dtype") because a checkpoint may
       quantize them differently. */
    int conv_int = 0;
    /* The conv's OUTPUT int16 exit for q and k (int-pipeline 9.4). It is
       independent of conv_int - that one is about the conv's INPUT, and
       either side can take its exit without the other. What it does need
       is the fixed tier and LZ_CONV_SIG_I, because the exit is integer
       only by way of the sig_e table those build. */
    int convo_i16 = 0;   /* set below, past the last declaration */
    /* int-pipeline milestone 5, decided here for the same reason
       conv_int is: the producing matmul has to know before it runs, and
       the consumer reads the answer far below.

       lz_sig_mode() gates bvec and the gate because the FLOAT sigmoid
       tier is not this table at all (lz_exp-based, a different value),
       so there is no integer entry to hand an integer to - the same
       shape as "the fixed conv tier is the consumer, so its absence
       answers this on its own". --fixed off therefore turns both
       off at runtime, which is what makes it a control. */
    /* *_may is the capability, *_int is "the int buffer holds it" and is
       raised only by proj_i16 - see its header for why they are two
       objects. */
    int bvec_int = 0, lat_int = 0, gate_int = 0;
    int bvec_may = 0, lat_may = 0;
    /* Assigned far below, where s->wscr is known free. Declared here
       because C89 wants the declaration ahead of the statements between,
       and dt_bias_arr outlives the block that would otherwise have held
       both. */
    const float *a_log_arr;
    const float *dt_bias_arr;
#if LZ_BVEC_I16
    bvec_may = lz_sig_mode() && s->bvec_i16 &&
               lz_matmul_xq_i16_ok(&L->kda_b_proj, hdim, nt);
#endif /* LZ_BVEC_I16 */
#if LZ_KLAT_I16
    lat_may = s->kda_lat_i16 && s->kda_lat_i32 && gate_rank > 0 &&
              lz_matmul_xq_i16_ok(&L->kda_f_a_proj, hdim, nt);
#endif /* LZ_KLAT_I16 */
#if LZ_KGATE_I16
    /* nt == 1 is not a restriction of the exit, it is what f_b_proj's
       call site is: this projection streams one token at a time (its
       in_dim is not hidden, so it cannot share the batched
       quantization), so the exit is asked for one token too.

       kda_has_gate_lower_bound is a CONSUMER question, not a taste one.
       With the lower bound the gate's transcendental is the sigmoid,
       which has an integer entry; without it the gate is
       lz_softplus(pre), and this tree has no fixed-point softplus at
       all, so that branch would have to rebuild the float the epilogue
       just declined to build - the work moved, not removed. */
    gate_int = lz_sig_mode() && s->kda_gate_i16 && s->kda_dtb_i16 &&
               c->kda_has_gate_lower_bound &&
               lz_matmul_xq_i16_ok(&L->kda_f_b_proj, gate_rank, 1);
#endif /* LZ_KGATE_I16 */

    /* Moved down past the declarations above, for the reason
       forward_moe's own note gives: C89 wants every declaration in a
       block ahead of every statement, and the declarations here each
       carry an explanation that should not be separated from them. */
#if LZ_CONV_FIXED
    if (s->conv_fixed && s->kda_q_i16 && s->kda_k_i16 && s->kda_v_i16) {
        int okq = lz_matmul_xq_i16_ok(&L->kda_q_proj, hdim, nt);
        int okk = lz_matmul_xq_i16_ok(&L->kda_k_proj, hdim, nt);
        int okv = lz_matmul_xq_i16_ok(&L->kda_v_proj, hdim, nt);
        conv_int = okq && okk && okv;
    }
#if LZ_CONVO_I16 && LZ_CONV_SIG_I
    convo_i16 = s->conv_fixed && s->kda_qc_i16 && s->kda_kc_i16
                && s->kda_vc_i16;
#endif /* LZ_CONVO_I16 && LZ_CONV_SIG_I */
#endif /* LZ_CONV_FIXED */

    if (c->use_subn) {
        /* SubLN per-projection input RMSNorms for the four hidden-space
           ternary projections (q/k/v/g): each normalizes the
           UN-normalized residual with its own norm weight before its
           matmul (the two pre-layer norms are deleted - see the layer
           loop in forward_chunk). They share s->xb2 as scratch, free
           until the o_proj matmul below overwrites it: each norm result
           is consumed by its own matmul before the next norm runs, and
           for Q8 weights the matmul reads the quantized xq/xqs, not the
           float scratch. f_a/b_proj stay on raw s->xb - no SubLN norm
           field exists for the low-rank decay-gate projections. */
        /* Once, not once per projection: the answer depends on hdim and
           gsa, and all four share both. */
        int nint = lz_norm_int() && lz_norm_can_fixed(hdim) &&
                   gsa > 0 && nsa * gsa == hdim;

        subn_norm_q8(s, &L->kda_q_norm, hdim, nt, nsa, rms_eps, nint);
        kda_conv_proj(s, &conv_int, s->kda_q, s->kda_q_i16, s->xb2,
                      &L->kda_q_proj, hdim, kdim, nt);

        subn_norm_q8(s, &L->kda_k_norm, hdim, nt, nsa, rms_eps, nint);
        kda_conv_proj(s, &conv_int, s->kda_k, s->kda_k_i16, s->xb2,
                      &L->kda_k_proj, hdim, kdim, nt);

        subn_norm_q8(s, &L->kda_v_norm, hdim, nt, nsa, rms_eps, nint);
        kda_conv_proj(s, &conv_int, s->kda_v, s->kda_v_i16, s->xb2,
                      &L->kda_v_proj, hdim, vdim, nt);

        subn_norm_q8(s, &L->kda_g_norm, hdim, nt, nsa, rms_eps, nint);
        lz_matmul_xq_nt(s->zbuf, s->xb2, s->xq, s->xqs, &L->kda_g_proj,
                        hdim, vdim, nt);

        /* f_a/b_proj: no SubLN norm field - quantize raw s->xb (the
           shared-activation-quantization trick still applies to these
           two). */
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8_tok(s->xb + (size_t)tk * hdim, hdim,
                               s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
        proj_i16(s, lat_may, &lat_int, s->kda_gate_lat, s->kda_lat_i16,
                     s->xb, &L->kda_f_a_proj, hdim, gate_rank, nt,
                     LZ_KLAT_ES, 32767);
        proj_i16(s, bvec_may, &bvec_int, s->bvec, s->bvec_i16,
                     s->xb, &L->kda_b_proj, hdim, nv, nt,
                     LZ_BVEC_ES, lz_sig_q15_domain(LZ_BVEC_ES));
    } else {
    /* Six hidden-space projections share one activation quantization,
       the same trick forward_ssm uses for its four (all have
       in_dim == hidden_size, so lz_act_gs agrees regardless of format). */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gsa,
                       s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
    kda_conv_proj(s, &conv_int, s->kda_q, s->kda_q_i16, s->xb,
                  &L->kda_q_proj, hdim, kdim, nt);
    kda_conv_proj(s, &conv_int, s->kda_k, s->kda_k_i16, s->xb,
                  &L->kda_k_proj, hdim, kdim, nt);
    kda_conv_proj(s, &conv_int, s->kda_v, s->kda_v_i16, s->xb,
                  &L->kda_v_proj, hdim, vdim, nt);
    proj_i16(s, lat_may, &lat_int, s->kda_gate_lat, s->kda_lat_i16,
                 s->xb, &L->kda_f_a_proj, hdim, gate_rank, nt,
                 LZ_KLAT_ES, 32767);
    proj_i16(s, bvec_may, &bvec_int, s->bvec, s->bvec_i16,
                 s->xb, &L->kda_b_proj, hdim, nv, nt,
                 LZ_BVEC_ES, lz_sig_q15_domain(LZ_BVEC_ES));
    lz_matmul_xq_nt(s->zbuf, s->xb, s->xq, s->xqs, &L->kda_g_proj, hdim, vdim, nt);
    }
    /* These read the FLOAT projection buffers, which hold nothing when
       the matching flag took the int16 exit - a tap build wanting
       q/k/v/f_a/b there has to dump kda_q_i16/kda_k_i16/kda_v_i16
       (conv_int), kda_lat_i16 (lat_int) or bvec_i16 (bvec_int). */
    LZ_TAP("kq", layer, s->kda_q, kdim);
    LZ_TAP("kk", layer, s->kda_k, kdim);
    LZ_TAP("kv", layer, s->kda_v, vdim);
    LZ_TAP("kfa", layer, s->kda_gate_lat, gate_rank);
    LZ_TAP("kb", layer, s->bvec, nv);
    LZ_TAP("kg", layer, s->zbuf, vdim);

    /* Depthwise causal convolution, SiLU activation - three independent
       calls over disjoint channel ranges of the same per-layer state
       buffer (q_off/k_off/v_off above), each serial in t like
       forward_ssm's single conv. Ring-aware per this function's own
       header comment - slot_in==slot_out when !advance_ring. */
#if LZ_CONV_FIXED
    /* Fill the packed fixed-conv params once, before the tk loop: the
       per-token call then passes one struct pointer (lea+push) where the
       flat signature made Watcom push six memory operands a call. The
       channel base is the history offset divided by hist - the two index
       different things (elements vs channels) and mixing them silently
       reads a neighbouring channel's taps. */
    if (s->conv_fixed) {
        size_t cb = (size_t)li * c->lin_conv_dim;
        size_t q_ch = 0, k_ch = (size_t)c->lin_key_dim;
        size_t v_ch = (size_t)2 * c->lin_key_dim;
        cp_q.mw = s->conv_mw + (cb + q_ch) * c->conv_kernel;
        cp_k.mw = s->conv_mw + (cb + k_ch) * c->conv_kernel;
        cp_v.mw = s->conv_mw + (cb + v_ch) * c->conv_kernel;
        cp_q.sig_k1 = s->conv_sig_k1 + cb + q_ch;
        cp_k.sig_k1 = s->conv_sig_k1 + cb + k_ch;
        cp_v.sig_k1 = s->conv_sig_k1 + cb + v_ch;
        cp_q.sig_oscale2 = s->conv_sig_oscale2 + cb + q_ch;
        cp_k.sig_oscale2 = s->conv_sig_oscale2 + cb + k_ch;
        cp_v.sig_oscale2 = s->conv_sig_oscale2 + cb + v_ch;
        cp_q.sig_k2 = cp_k.sig_k2 = cp_v.sig_k2 = s->conv_sig_k2;
        cp_q.sig_e = LZ_CONV_SIG_E(s, cb + q_ch);
        cp_k.sig_e = LZ_CONV_SIG_E(s, cb + k_ch);
        cp_v.sig_e = LZ_CONV_SIG_E(s, cb + v_ch);
        cp_q.in_scale = cp_k.in_scale = cp_v.in_scale = s->conv_in_scale;
        cp_q.bound = cp_k.bound = cp_v.bound = s->conv_bound;
        cp_q.k = cp_k.k = cp_v.k = c->conv_kernel;
    }
#endif /* LZ_CONV_FIXED */
    for (tk = 0; tk < nt; tk++) {
        int slot_in  = advance_ring ? (ring_base + tk) % ring_depth : ring_base;
        int slot_out = advance_ring ? ring_slot_next(slot_in, ring_depth)
                                    : ring_base;
#if LZ_CONV_FIXED
        if (s->conv_fixed) {
            /* Same three disjoint ranges, int16 history. */
            short *qi = s->conv_state_q + (size_t)slot_in  * conv_slot_stride + li_off_conv;
            short *qo = s->conv_state_q + (size_t)slot_out * conv_slot_stride + li_off_conv;
            /* `conv_int` says the projections above exited as int16 at
               LZ_CONV_ES already, so these read that and the float
               buffers hold nothing for this token. NULL is the float
               input the tier was written with. */
            const short *qxi = conv_int ? s->kda_q_i16 + (size_t)tk * kdim : NULL;
            const short *kxi = conv_int ? s->kda_k_i16 + (size_t)tk * kdim : NULL;
            const short *vxi = conv_int ? s->kda_v_i16 + (size_t)tk * vdim : NULL;
            /* q and k take the conv's int16 exit when it is compiled in;
               convo_i16 is what forward_kda's l2norm below reads to know
               which of the two buffers holds this token. LZ_CONV_SIG_I
               is a precondition, not a preference: the exit is integer
               only because sig_oscale2 is an exact power of two, and the
               =0 arm has no sig_e table to derive its exponent from. */
            lz_causal_conv1d_step_fixed_o16(
                s->kda_qc + (size_t)tk * kdim,
                convo_i16 ? s->kda_qc_i16 + (size_t)tk * kdim : (short *)0,
                LZ_CONVO_ES, s->kda_q + (size_t)tk * kdim,
                qxi, qi + q_off, qo + q_off,
                &cp_q, kdim);
            lz_causal_conv1d_step_fixed_o16(
                s->kda_kc + (size_t)tk * kdim,
                convo_i16 ? s->kda_kc_i16 + (size_t)tk * kdim : (short *)0,
                LZ_CONVO_ES, s->kda_k + (size_t)tk * kdim,
                kxi, qi + k_off, qo + k_off,
                &cp_k, kdim);
            lz_causal_conv1d_step_fixed_o16(
                s->kda_vc + (size_t)tk * vdim,
                convo_i16 ? s->kda_vc_i16 + (size_t)tk * vdim : (short *)0,
                LZ_CONVO_V_ES, s->kda_v + (size_t)tk * vdim,
                vxi, qi + v_off, qo + v_off,
                &cp_v, vdim);
            continue;
        }
#endif /* LZ_CONV_FIXED */
        {
        float *base_in  = s->conv_state + (size_t)slot_in  * conv_slot_stride + li_off_conv;
        float *base_out = s->conv_state + (size_t)slot_out * conv_slot_stride + li_off_conv;
        lz_causal_conv1d_step(s->kda_qc + (size_t)tk * kdim,
                              s->kda_q + (size_t)tk * kdim,
                              base_in + q_off, base_out + q_off,
                              lz_t_f32(&L->kda_q_conv1d, s->wscr), kdim,
                              c->conv_kernel);
        lz_causal_conv1d_step(s->kda_kc + (size_t)tk * kdim,
                              s->kda_k + (size_t)tk * kdim,
                              base_in + k_off, base_out + k_off,
                              lz_t_f32(&L->kda_k_conv1d, s->wscr), kdim,
                              c->conv_kernel);
        lz_causal_conv1d_step(s->kda_vc + (size_t)tk * vdim,
                              s->kda_v + (size_t)tk * vdim,
                              base_in + v_off, base_out + v_off,
                              lz_t_f32(&L->kda_v_conv1d, s->wscr), vdim,
                              c->conv_kernel);
        }
    }
    /* These three read the float buffers, which LZ_CONVO_I16 leaves
       unwritten - the conv puts the token in kda_*c_i16 instead. A tap
       build (the differential dump tool, .prof/convqk_range.c) must
       therefore pass -DLZ_CONVO_I16=0, and the symptom otherwise is
       silent: every sample reads zero. Found by convqk_range reporting
       max|kvc| = 0 over 3.7M samples, which is not a number a live
       model produces. */
    LZ_TAP("kqc", layer, s->kda_qc, kdim);
    LZ_TAP("kkc", layer, s->kda_kc, kdim);
    LZ_TAP("kvc", layer, s->kda_vc, vdim);

    /* f_b_proj: kda_gate_rank -> nvk, a DIFFERENT in_dim than hidden, so
       it needs its own quantization - lz_matmul_w does that internally,
       one token at a time (same reasoning forward_moe's per-expert
       matmuls use).

       Both of this matmul's sides are int-pipeline milestone 5 exits and
       they are INDEPENDENT: `lat_int` is its input (kda_gate_lat, from
       f_a_proj) and `gate_int` is its output (kda_gate, read by the decay
       gate below). The float body is lz_matmul_w spelled out - that
       function IS lz_quantize_q8 followed by lz_matmul_xq, and
       lz_matmul_xq is lz_matmul_xq_nt at nt == 1 - so that either side
       can be replaced without the other. */
    {
        int gsb = lz_act_gs(&L->kda_f_b_proj, gate_rank);
        int retry = 1;
        while (retry) {
            retry = 0;
            for (tk = 0; tk < nt; tk++) {
                if (lat_int) {
                    /* Widening, not requantizing: lz_quantize_q8_int
                       already takes exactly this - an integer vector plus
                       the power-of-two scale that turns it back into the
                       float the group scale must match - and it is the
                       entry the norm-int tier already uses at eight other
                       call sites. Only its int32 input width differs from
                       the exit's int16, and an int-to-int copy costs no
                       float op and introduces no rounding. */
                    int k;
                    const short *li = s->kda_lat_i16 + (size_t)tk * gate_rank;
                    lz_debug_klat_i16 += gate_rank;
                    for (k = 0; k < gate_rank; k++) s->kda_lat_i32[k] = li[k];
                    lz_quantize_q8_int(s->kda_lat_i32, gate_rank, gsb,
                                       pow2f(-LZ_KLAT_ES), s->xq, s->xqs);
                } else {
                    lz_quantize_q8(s->kda_gate_lat + (size_t)tk * gate_rank,
                                   gate_rank, gsb, s->xq, s->xqs);
                }
                if (gate_int) {
                    if (lz_matmul_xq_nt_i16(
                            s->kda_gate_i16 + (size_t)tk * nvk,
                            s->xq, s->xqs, &L->kda_f_b_proj,
                            gate_rank, nvk, 1, LZ_KGATE_ES, 32767))
                        continue;
                    /* Not reachable - the predicate above IS this entry
                       point's own, on the same (w, in_dim, nt). Handled
                       anyway, and handled by starting the chunk OVER on
                       the float exit: falling back for this token alone
                       would strand the earlier tokens' int buffers, the
                       mid-triple hazard kda_conv_proj names. */
                    gate_int = 0;
                    retry = 1;
                    break;
                }
                lz_matmul_xq_nt(s->kda_gate + (size_t)tk * nvk,
                                s->kda_gate_lat + (size_t)tk * gate_rank,
                                s->xq, s->xqs, &L->kda_f_b_proj,
                                gate_rank, nvk, 1);
            }
        }
    }

    /* Turn the pre-activation into the actual per-channel decay factor
       gt = exp(g). Without a lower bound, g = -exp(A_log[h]) *
       softplus(pre[h,k] + dt_bias[h,k]). With one (K3's gate_lower_bound),
       the kernel REPLACES that formula rather than clamping it:
       g = lower_bound * sigmoid(exp(A_log[h]) * (pre[h,k] + dt_bias[h,k]))
       (fla.ops.kda.chunk_kda's own docstring / naive_kda_lowerbound_gate;
       a plain floor on the unbounded formula is a DIFFERENT, wrong
       function - see KdaAttention.decay).
       This is that function, not a
       re-derivation of it - A_log stays PER HEAD (h only), dt_bias is
       PER CHANNEL (h, k), matching that file's own comment about why
       widening A_log was tried and reverted. */
    /* decay_base[h] = exp(A_log[h]) depends on the head ONLY, not on the
       token. Computed once per head here, then reused across nt tokens.
       A_log/dt_bias are per-head/per-channel weights, so their
       dequantization is hoisted out of the tk loop too. decay_base goes
       into s->wscr: A_log/dt_bias are F32 (lz_t_f32 returns t->f, no
       write to wscr), so wscr is free here, and it is sized >= nvk >= nv. */
    a_log_arr   = lz_t_f32(&L->kda_A_log, s->wscr);
    dt_bias_arr = lz_t_f32(&L->kda_dt_bias, s->wscr);
    kda_decay_gate_build(s, c, a_log_arr, dt_bias_arr, li, nv, kd, nt, gate_int);
    LZ_TAP("kgate", layer, s->kda_gate, nvk);

    /* Recurrence, head-major like forward_ssm's - same L1-residency
       argument applies (one head's state is kd*vd, resident for T
       steps; heads are independent, each strictly serial in t). */
    for (h = 0; h < nv; h++) {
        int kh = (nk == nv) ? h : (h * nk / nv);
        size_t h_off_q8 = li_off_q8 + (size_t)h * kd * vd;
        size_t h_off_s  = h_off_q8 / 32;

        for (tk = 0; tk < nt; tk++) {
            LZ_PROF_DECL(_tr);
            int slot_in  = advance_ring ? (ring_base + tk) % ring_depth : ring_base;
            int slot_out = advance_ring ? ring_slot_next(slot_in, ring_depth)
                                        : ring_base;
            const int8_t *sq_in  = s->ssm_state_q8 + (size_t)slot_in  * q8_slot_stride + h_off_q8;
            int8_t       *sq_out = s->ssm_state_q8 + (size_t)slot_out * q8_slot_stride + h_off_q8;
#if LZ_GDN_STATE_2PLANE
            const int8_t *sq2_in  = s->ssm_state_q8_lo + (size_t)slot_in  * q8_slot_stride + h_off_q8;
            int8_t       *sq2_out = s->ssm_state_q8_lo + (size_t)slot_out * q8_slot_stride + h_off_q8;
#endif /* LZ_GDN_STATE_2PLANE */
            const float *ss_in  = s->ssm_state_s + (size_t)slot_in  * s_slot_stride + h_off_s;
            float       *ss_out = s->ssm_state_s + (size_t)slot_out * s_slot_stride + h_off_s;
            const float *qp = s->kda_qc + (size_t)tk * kdim + (size_t)kh * kd;
            const float *kp = s->kda_kc + (size_t)tk * kdim + (size_t)kh * kd;
            const float *vp = s->kda_vc + (size_t)tk * vdim + (size_t)h * vd;
            const float *gv = s->kda_gate + (size_t)tk * nvk + (size_t)h * kd;
            float *out = s->ssm_out + (size_t)tk * vdim + (size_t)h * vd;
            float beta;

            /* The conv's int16 exit ends HERE, and the output stays
               float: this normalisation's consumer is gdn_build_table's
               ss[kk*ng+gg] * q[kk], whose scale varies with both indices
               and so has no common factor an integer q could ride. */
            if (convo_i16) {
                lz_l2norm_i16(s->qn,
                              s->kda_qc_i16 + (size_t)tk * kdim
                                            + (size_t)kh * kd,
                              kd, LZ_CONVO_ES, LZ_L2NORM_EPS);
                lz_l2norm_i16(s->kn,
                              s->kda_kc_i16 + (size_t)tk * kdim
                                            + (size_t)kh * kd,
                              kd, LZ_CONVO_ES, LZ_L2NORM_EPS);
            } else {
                lz_l2norm(s->qn, qp, kd, LZ_L2NORM_EPS);
                lz_l2norm(s->kn, kp, kd, LZ_L2NORM_EPS);
            }
            for (i = 0; i < kd; i++) s->qn[i] *= scale;

            /* bvec's only consumer, and with `bvec_int` it never became
               a float: the projection exited int16 at LZ_BVEC_ES and
               lz_sigmoid_i reaches the same table from there. */
            if (bvec_int) lz_debug_bvec_i16++;
            beta = bvec_int
                 ? lz_sigmoid_i(s->bvec_i16[(size_t)tk * nv + h], LZ_BVEC_ES)
                 : lz_sigmoid(s->bvec[(size_t)tk * nv + h]);
            LZ_PROF_BEG(_tr);
            if (convo_i16)
                lz_recur_step(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                        sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                        ss_in, ss_out, s->qn, s->kn,
                        (const float *)0,
                        s->kda_vc_i16 + (size_t)tk * vdim + (size_t)h * vd,
                        pow2f(-LZ_CONVO_V_ES), gv, 1.0f, beta, kd, vd);
            else
                lz_recur_step(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                        sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                        ss_in, ss_out, s->qn, s->kn, vp,
                        (const short *)0, 0.0f, gv, 1.0f, beta, kd, vd);
            LZ_PROF_END(_tr, LZ_PROF_REC);
        }
    }

    /* Raw recurrence output, pre-o_norm - separates "the recurrence is
       wrong" from "the gating is wrong" the same way GDN's sgdn tap
       alone cannot; this is what caught a test-fixture bug (v_head_dim
       not a multiple of 32) that "sgdn" alone would have reported as
       "everything downstream of KDA is wrong" instead. */
    LZ_TAP("kraw", layer, s->ssm_out, vdim);
    /* Same gated RMSNorm as GDN (plain-weight, silu gate) - see
       kda_o_norm's field comment in model.h for why the activation must
       stay silu on this engine. Fixed tier computes the full silu value
       in Q15 int-multiply (value-changing); float tier keeps the plain
       norm + quantize path. --norm-int takes the same int-output route
       as forward_ssm's sgdn block above (bit-identical, same guard). */
    {
        int gso = lz_act_gs(&L->kda_o_proj, vdim);
        int nso = scale_groups(vdim, gso);
        const float *kda_o_norm_w = lz_t_f32(&L->kda_o_norm, s->wscr);
        if (c->use_subn) {
            /* SubLN: o_proj's OWN input RMSNorm (kda_o_subn_norm) over
               the whole lin_value_dim span, applied AFTER the per-head
               gated silu norm (kda_o_norm). Computed in float regardless
               of the fixed/int tiers: the Q15 int-output shortcut
               materializes the gated result only as an int scale that
               cannot feed a second per-element weight multiply. */
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++)
                    lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                     kda_o_norm_w, vd,
                                     rms_eps, NULL);
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->ssm_out + (size_t)tk * vdim,
                           s->ssm_out + (size_t)tk * vdim,
                           lz_t_f32(&L->kda_o_subn_norm, s->wscr), vdim,
                           rms_eps);
            /* SubLN Hadamard: o_proj's block, settled at load time
               (model.c resolve_hadamard). 0 = no Hadamard. */
            {
                int oblk = lz_hadamard_on() ? c->hadamard_o : 0;
                int qgw = subn_scale_gs(gso, vdim);
                if (oblk > 1) {
                    for (tk = 0; tk < nt; tk++)
                        subn_fwht_quant(s->ssm_out + (size_t)tk * vdim, vdim,
                                        oblk, qgw, s->xq + (size_t)tk * vdim,
                                        s->xqs + (size_t)tk * nso, s->fwht_scratch);
                    /* Only when the matmul below will read the float row -
                       otherwise this is a second transform nobody reads,
                       on every token, on the target machine. */
                    if (lz_matmul_xq_reads_float_row(&L->kda_o_proj))
                        for (tk = 0; tk < nt; tk++)
                            subn_rot_f32(s->ssm_out + (size_t)tk * vdim,
                                         vdim, oblk);
                } else if (qgw > 0) {
                    for (tk = 0; tk < nt; tk++)
                        lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, qgw,
                                       s->xq + (size_t)tk * vdim,
                                       s->xqs + (size_t)tk * nso);
                } else {
                    for (tk = 0; tk < nt; tk++)
                        lz_quantize_q8_tok(s->ssm_out + (size_t)tk * vdim, vdim,
                                           s->xq + (size_t)tk * vdim,
                                           s->xqs + (size_t)tk * nso);
                }
            }
        } else if (lz_norm_int() && lz_norm_can_fixed(vd) &&
            gso > 0 && (vd % gso) == 0) {
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++) {
                    float deq;
                    lz_rmsnorm_gated_int(s->ssm_sig + (size_t)tk * vdim + (size_t)h * vd,
                                         s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                         s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                         kda_o_norm_w, vd,
                                         rms_eps, &deq);
                    lz_quantize_q8_int(s->ssm_sig + (size_t)tk * vdim + (size_t)h * vd,
                                       vd, gso, deq,
                                       s->xq + (size_t)tk * vdim + (size_t)h * vd,
                                       s->xqs + (size_t)tk * nso + (size_t)h * (vd / gso));
                }
        } else if (lz_norm_can_fixed(vd)) {
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++)
                    lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                     kda_o_norm_w, vd,
                                     rms_eps,
                                     NULL);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, gso,
                               s->xq + (size_t)tk * vdim, s->xqs + (size_t)tk * nso);
        } else {
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++)
                    lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                     kda_o_norm_w, vd,
                                     rms_eps, NULL);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, gso,
                               s->xq + (size_t)tk * vdim, s->xqs + (size_t)tk * nso);
        }
        LZ_TAP("sgdn", layer, s->ssm_out, vdim);
        lz_matmul_xq_nt(s->xb2, s->ssm_out, s->xq, s->xqs, &L->kda_o_proj, vdim,
                        hdim, nt);
    }
}

/* ------------------------------------------------------ dense FFN block */

/* Classic dense SwiGLU FFN (gate/up/down_proj): s->xb (nt x dim) ->
   s->xb2 (nt x dim), via s->hb/s->hb2 scratch. Extracted out of
   forward_chunk's per-layer loop so the MTP block's draft step
   (forward_mtp_draft_step) can share it exactly rather than carry a
   second, driftable copy - the MTP head's FFN is ALWAYS this dense
   form, never latent MoE, regardless of what the body does (model.h's
   LZMtp comment; model.c's model_walk filters MoE specs out of the MTP
   block for the same reason). `layer` is only for LZ_TAP (a no-op in
   production); the MTP call site passes LZ_MTP_CACHE_LAYER.

   `idim` is an explicit parameter, NOT read off c->intermediate_size
   internally: the MTP block's FFN width (c->mtp_intermediate_size) is
   an independent field (model.h's own comment on why
   reusing intermediate_size for both would silently couple two
   unrelated things on a checkpoint like kunmoe-v2). The body call site
   passes c->intermediate_size, the MTP call site passes
   c->mtp_intermediate_size - s->hb/s->hb2/s->xq/s->xqs are sized for
   the larger of the two by lz_state_alloc (see its own comment). */
void dense_ffn_step(const LZModel *m, LZRunState *s,
                           const LZLayer *L, int layer, int nt, int idim) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size;
    float rms_eps = c->rms_norm_eps;
    int gsg = lz_act_gs(&L->gate_proj, dim);
    int nsg = scale_groups(dim, gsg);
    int gsd = lz_act_gs(&L->down_proj, idim);
    int nsd = scale_groups(idim, gsd);
    int i, tk;
    LZ_PROF_DECL(_ta);
    (void)layer;   /* only used inside LZ_TAP, which is a no-op in a normal build */

    if (c->use_subn) {
        /* SubLN per-projection input RMSNorms for the dense FFN: gate
           and up each normalize the UN-normalized residual with their own
           norm weight before their matmul (s->xb2 as scratch, free until
           the down_proj matmul below), down normalizes the intermediate
           activation with down_norm (s->hb2 as scratch, dead after the
           SwiGLU loop). */
        /* Norms tier, int output: see forward_kda's matching comment -
           same subn_norm_int scratch, same fallback conditions. */
        if (lz_norm_int() && lz_norm_can_fixed(dim) &&
            gsg > 0 && (dim % gsg) == 0) {
            for (tk = 0; tk < nt; tk++) {
                float deq;
                lz_rmsnorm_int(s->subn_norm_int, s->xb + (size_t)tk * dim,
                               lz_t_f32(&L->gate_norm, s->wscr), dim,
                               rms_eps, &deq);
                lz_quantize_q8_tok_int(s->subn_norm_int, dim, deq,
                                       s->xq + (size_t)tk * dim,
                                       s->xqs + (size_t)tk * nsg);
            }
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb2 + (size_t)tk * dim, s->xb + (size_t)tk * dim,
                           lz_t_f32(&L->gate_norm, s->wscr), dim, rms_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8_tok(s->xb2 + (size_t)tk * dim, dim,
                                   s->xq + (size_t)tk * dim, s->xqs + (size_t)tk * nsg);
        }
        lz_matmul_xq_nt(s->hb, s->xb2, s->xq, s->xqs, &L->gate_proj, dim,
                        idim, nt);

        if (lz_norm_int() && lz_norm_can_fixed(dim) &&
            gsg > 0 && (dim % gsg) == 0) {
            for (tk = 0; tk < nt; tk++) {
                float deq;
                lz_rmsnorm_int(s->subn_norm_int, s->xb + (size_t)tk * dim,
                               lz_t_f32(&L->up_norm, s->wscr), dim,
                               rms_eps, &deq);
                lz_quantize_q8_tok_int(s->subn_norm_int, dim, deq,
                                       s->xq + (size_t)tk * dim,
                                       s->xqs + (size_t)tk * nsg);
            }
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb2 + (size_t)tk * dim, s->xb + (size_t)tk * dim,
                           lz_t_f32(&L->up_norm, s->wscr), dim, rms_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8_tok(s->xb2 + (size_t)tk * dim, dim,
                                   s->xq + (size_t)tk * dim, s->xqs + (size_t)tk * nsg);
        }
        lz_matmul_xq_nt(s->hb2, s->xb2, s->xq, s->xqs, &L->up_proj, dim,
                        idim, nt);
    } else {
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * dim, dim, gsg,
                       s->xq + (size_t)tk * dim, s->xqs + (size_t)tk * nsg);
    lz_matmul_xq_nt(s->hb, s->xb, s->xq, s->xqs, &L->gate_proj, dim,
                    idim, nt);
    lz_matmul_xq_nt(s->hb2, s->xb, s->xq, s->xqs, &L->up_proj, dim,
                    idim, nt);
    }
    {
        /* SwiGLU: one lz_exp per element, scalar. NOT vectorized, and
           not worth vectorizing - measured at 0.7%% of decode
           (--profile), because the three matmuls around it dominate the
           ffn phase and those are already SIMD.

           lz_swiglu_q15_i16 is deliberately NOT wired here, and the
           reason is the checkpoints rather than the code: gate/up/down
           are f32 on every dense model in /e/LLM/models/ that loads
           (.prof/dtype_by_role.c on all 30), so there is no int16 exit
           for the producers to take. The one exception is the throwaway
           ARM fixture _armgate2 (q4_1/q4_1/q6_1), where
           lz_matmul_xq_i16_ok does answer 1 - so this is unreached, not
           unreachable, and wiring it means calibrating three exponents
           on a fixture no quality gate runs.

           Do not read "scalar" as "impossible here". This repo already
           replaces float with integer MMX where it pays and controls the
           error with a second plane (LZ_GDN_STATE_2PLANE, LZ_KV_2PLANE),
           and lz_exp is itself a table-plus-polynomial hack rather than
           libm. The reason this one stays scalar is its SIZE, not its
           shape. */
        LZ_PROF_BEG(_ta);
        for (i = 0; i < nt * idim; i++)
            s->hb[i] = lz_silu(s->hb[i]) * s->hb2[i];
        LZ_PROF_END(_ta, LZ_PROF_ACT);
    }
    LZ_TAP("fh", layer, s->hb, idim);
    if (c->use_subn) {
        /* down_proj input = RMSNorm_{down_norm}(SwiGLU output), then the
           SubLN block Hadamard, whose block is settled at load time
           (model.c resolve_hadamard). 0 = no Hadamard. */
        {
            int dblk = lz_hadamard_on() ? c->hadamard_down : 0;
            /* The int norm output only applies when dblk<=1: the dblk>1
               branch feeds subn_fwht_quant, which needs the FLOAT row to
               run the Hadamard transform before it quantizes - there is
               no int-input twin of it, so that branch stays exactly as
               before regardless of the tier. */
            if (dblk <= 1 && lz_norm_int() && lz_norm_can_fixed(idim) &&
                gsd > 0 && (idim % gsd) == 0) {
                for (tk = 0; tk < nt; tk++) {
                    float deq;
                    lz_rmsnorm_int(s->subn_norm_int, s->hb + (size_t)tk * idim,
                                   lz_t_f32(&L->down_norm, s->wscr), idim,
                                   rms_eps, &deq);
                    lz_quantize_q8_tok_int(s->subn_norm_int, idim, deq,
                                           s->xq + (size_t)tk * idim,
                                           s->xqs + (size_t)tk * nsd);
                }
            } else {
                int qgw = subn_scale_gs(gsd, idim);
                for (tk = 0; tk < nt; tk++)
                    lz_rmsnorm(s->hb2 + (size_t)tk * idim, s->hb + (size_t)tk * idim,
                               lz_t_f32(&L->down_norm, s->wscr), idim, rms_eps);
                if (dblk > 1) {
                    for (tk = 0; tk < nt; tk++)
                        subn_fwht_quant(s->hb2 + (size_t)tk * idim, idim, dblk,
                                        qgw, s->xq + (size_t)tk * idim,
                                        s->xqs + (size_t)tk * nsd, s->fwht_scratch);
                    /* See forward_kda's twin: the float row is what an
                       F32 down_proj reads, and only then is this worth
                       running. */
                    if (lz_matmul_xq_reads_float_row(&L->down_proj))
                        for (tk = 0; tk < nt; tk++)
                            subn_rot_f32(s->hb2 + (size_t)tk * idim,
                                         idim, dblk);
                } else if (qgw > 0) {
                    for (tk = 0; tk < nt; tk++)
                        lz_quantize_q8(s->hb2 + (size_t)tk * idim, idim, qgw,
                                       s->xq + (size_t)tk * idim,
                                       s->xqs + (size_t)tk * nsd);
                } else {
                    for (tk = 0; tk < nt; tk++)
                        lz_quantize_q8_tok(s->hb2 + (size_t)tk * idim, idim,
                                           s->xq + (size_t)tk * idim,
                                           s->xqs + (size_t)tk * nsd);
                }
            }
        }
        lz_matmul_xq_nt(s->xb2, s->hb2, s->xq, s->xqs, &L->down_proj, idim,
                        dim, nt);
    } else {
    /* Quantization groups must match the target weight gs: when the
       intermediate dim is not a multiple of 32 (e.g. 1021) the weight's
       in-row gs falls back, and xqs tail entries must be filled with
       the same gs. */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->hb + (size_t)tk * idim, idim, gsd,
                       s->xq + (size_t)tk * idim, s->xqs + (size_t)tk * nsd);
    lz_matmul_xq_nt(s->xb2, s->hb, s->xq, s->xqs, &L->down_proj, idim,
                    dim, nt);
    }
}
