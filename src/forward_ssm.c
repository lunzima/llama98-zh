#include <string.h>

#include "forward.h"
#include "ops.h"
#include "ops_quant.h"  /* lz_quantize_q8(_int), lz_sigmoid */

/* Linear-attention (SSM) path, moved out of forward.c (the four-path split).
   Pure code motion: forward_ssm lost `static` because lz_forward still
   calls it. */

/* ------------------------------------------------ linear_attention block */

/* advance_ring/ring_base (the SSM/conv rollback ring - see forward.h's
   s->ssm_slot and forward_chunk's own ring_base comment):
   advance_ring is forward_chunk's all_logits, ring_base is s->ssm_slot
   as it stood BEFORE this chunk's layer loop started (read once by the
   caller, not here, so it does not drift as forward_ssm/forward_kda
   both get called once per LINEAR/KDA layer within the same chunk).
   advance_ring==0 (ordinary decode/prefill): every token reads and
   writes ring_base, bit-identical to the pre-ring single-slot version.
   advance_ring!=0 (a verify batch): token tk reads slot
   (ring_base+tk)%ring_depth and writes (ring_base+tk+1)%ring_depth -
   see lz_gdn_step's own header comment (ops.h) for why this costs zero
   extra bandwidth over the non-ring case, only memory. */
void forward_ssm(const LZModel *m, LZRunState *s,
                        const LZLayer *L, int layer, int nt,
                        int advance_ring, int ring_base) {
    const LZModelConfig *c = &m->config;
    int nk = c->lin_n_k_heads, nv = c->lin_n_v_heads;
    int kd = c->lin_k_head_dim, vd = c->lin_v_head_dim;
    int li = s->cache_idx[layer];
    int ring_depth = s->ssm_ring_depth;
    size_t q8_slot_stride = (size_t)c->n_linear_layers * nv * kd * vd;
    size_t s_slot_stride  = q8_slot_stride / 32;
    size_t conv_slot_stride = (size_t)c->n_linear_layers *
                              c->lin_conv_dim * (c->conv_kernel - 1);
    size_t li_off_q8 = (size_t)li * nv * kd * vd;
    size_t li_off_conv = (size_t)li * c->lin_conv_dim * (c->conv_kernel - 1);
    float scale = s->ssm_scale;
    int hdim = c->hidden_size, cdim = c->lin_conv_dim, vdim = c->lin_value_dim;
    int gsa = lz_act_gs(&L->in_proj_qkv, hdim);
    int nsa = scale_groups(hdim, gsa);
    int h, i, tk;

    /* quantize xb once; the four input projections share it (all in_dim = hidden) */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gsa,
                       s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
    lz_matmul_xq_nt(s->qkv, s->xb, s->xq, s->xqs, &L->in_proj_qkv, hdim,
                    cdim, nt);
    lz_matmul_xq_nt(s->zbuf, s->xb, s->xq, s->xqs, &L->in_proj_z, hdim,
                    vdim, nt);
    lz_matmul_xq_nt(s->avec, s->xb, s->xq, s->xqs, &L->in_proj_a, hdim, nv, nt);
    lz_matmul_xq_nt(s->bvec, s->xb, s->xq, s->xqs, &L->in_proj_b, hdim, nv, nt);

    /* depthwise causal convolution, SiLU activation */
    LZ_TAP("sqkv", layer, s->qkv, cdim);
    LZ_TAP("sz", layer, s->zbuf, vdim);
    LZ_TAP("sa", layer, s->avec, nv);
    LZ_TAP("sb", layer, s->bvec, nv);
    /* The conv advances serially over t: it reads/writes the rolling
       conv_state history, which is inherently ordered. **Deliberately
       not batched** - conv weights are only lin_conv_dim*k floats, 0.2%
       of per-token bytes; the bandwidth saved is not worth the risk of
       changing reduction order. Ring-aware per this function's
       own header comment - slot_in==slot_out when !advance_ring. */
    for (tk = 0; tk < nt; tk++) {
        int slot_in  = advance_ring ? (ring_base + tk) % ring_depth : ring_base;
        int slot_out = advance_ring ? ring_slot_next(slot_in, ring_depth)
                                    : ring_base;
#if LZ_CONV_FIXED
        if (s->conv_fixed) {
            /* One block here, unlike KDA's three: this path's conv
               covers lin_conv_dim in a single tensor. Wiring only KDA
               would leave a model with SSM layers half on each tier,
               which no banner line would show. */
            size_t cb = (size_t)li * c->lin_conv_dim;
            /* No int16 input here (int-pipeline milestone 3 wired only
               KDA's three): the producer is in_proj_qkv, and on every
               checkpoint in this tree that tensor is EITHER absent
               (kmr20/kunmoe-v2: n=0, so this whole function never runs)
               OR f32 (s1v3, kunkun98-pilot, kunmoe-v2-dense: n=1572864,
               dtype f32). f32 leaves lz_matmul_xq_nt at its first branch
               and never reaches the Q4_1/Q6_1/T2 epilogue that has the
               int16 exit, so there is nothing to wire and no gate could
               exercise it. Not a tier split of the kind the comment
               below warns about: this conv stays wholly on the float
               input, all lin_conv_dim channels of it. */
            lz_causal_conv1d_step_fixed(
                s->qkv_c + (size_t)tk * cdim, s->qkv + (size_t)tk * cdim, NULL,
                s->conv_state_q + (size_t)slot_in * conv_slot_stride + li_off_conv,
                s->conv_state_q + (size_t)slot_out * conv_slot_stride + li_off_conv,
                s->conv_mw + cb * c->conv_kernel,
                s->conv_sig_k1 + cb, s->conv_sig_oscale2 + cb, s->conv_sig_k2,
                LZ_CONV_SIG_E(s, cb),
                s->conv_in_scale, s->conv_bound, cdim, c->conv_kernel);
            continue;
        }
#endif /* LZ_CONV_FIXED */
        lz_causal_conv1d_step(s->qkv_c + (size_t)tk * cdim,
                              s->qkv + (size_t)tk * cdim,
                              s->conv_state + (size_t)slot_in * conv_slot_stride + li_off_conv,
                              s->conv_state + (size_t)slot_out * conv_slot_stride + li_off_conv,
                              lz_t_f32(&L->conv1d, s->wscr), cdim,
                              c->conv_kernel);
    }
    LZ_TAP("sconv", layer, s->qkv_c, cdim);

    /* Recurrence is organized head-major (fix a head, run T steps), not
       t-major. One head's state is kd*vd = 4 KB, resident in L1 for T
       steps - saving (T-1)/T of the per-token 1.33 MB state traffic.
       Heads are independent and each head is strictly serial in t, so
       this reordering changes no numbers. */
    for (h = 0; h < nv; h++) {
        /* In this model nv == nk, one-to-one; when nv > nk the reference
           uses repeat_interleave, equivalent to taking key head
           h * nk / nv. */
        int kh = (nk == nv) ? h : (h * nk / nv);
        size_t h_off_q8 = li_off_q8 + (size_t)h * kd * vd;
        size_t h_off_s  = h_off_q8 / 32;
        /* exp(A_log[h]) depends on the head only, not the token - hoist
           it out of the tk loop so it is computed once per head, not once
           per token per head (the same fix as forward_kda's decay_base). */
        float a_exp = lz_exp(lz_t_f32(&L->A_log, s->wscr)[h]);
        float dt_bias_h = lz_t_f32(&L->dt_bias, s->wscr)[h];

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
            const float *qp = s->qkv_c + (size_t)tk * cdim;
            const float *kp = qp + c->lin_key_dim;
            const float *vp = kp + c->lin_key_dim;
            const float *v_t = vp + (size_t)h * vd;
            float *out = s->ssm_out + (size_t)tk * vdim + (size_t)h * vd;
            float beta, g, gt;

            lz_l2norm(s->qn, qp + (size_t)kh * kd, kd, LZ_L2NORM_EPS);
            lz_l2norm(s->kn, kp + (size_t)kh * kd, kd, LZ_L2NORM_EPS);
            for (i = 0; i < kd; i++) s->qn[i] *= scale;

            beta = lz_sigmoid(s->bvec[(size_t)tk * nv + h]);
            /* A = exp(A_log) > 0 and softplus > 0, so g < 0 and
               gt = exp(g) lands in (0,1) - a decay factor. If gt >= 1
               ever comes out, a sign is flipped. */
            g = -a_exp *
                lz_softplus(s->avec[(size_t)tk * nv + h] + dt_bias_h);
            gt = lz_exp(g);

            /* State is read/written in two passes; quantization error
               decays per step with gt<1 (unlike KV cache, it does not
               accumulate). Self-consistency is guaranteed by
               lz_gdn_step: the next token reads the quantized value.
               sq_in/sq_out (and sq2/ss) differ only during a
               speculative verify batch - see this function's own
               signature comment. */
            LZ_PROF_BEG(_tr);
            lz_recur_step(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                        sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                        ss_in, ss_out, s->qn, s->kn, v_t,
                        (const short *)0, 0.0f, NULL, gt, beta, kd, vd);
            LZ_PROF_END(_tr, LZ_PROF_REC);
        }
    }

    /* This is the plain-weight gated RMSNorm, not the (1+w) kind used
       by the outer layers. The fixed tier computes the full silu value
       in Q15 int-multiply (value-changing) and writes it into o;
       lz_quantize_q8 quantizes it. With --norm-int the fixed tier
       writes the Q15 value as int into ssm_sig instead and hands the
       power-of-two scale out-of-band, so lz_quantize_q8_int folds it
       into the stored group scale and the dequant round-trip is gone.
       The fold itself is bit-identical; the element rounding is not any
       more, since that entry's loop became integer and exact - see
       lz_quantize_q8_int. The float tier keeps the plain norm +
       lz_quantize_q8 path. */
    {
        int gso = lz_act_gs(&L->out_proj, vdim);
        int nso = scale_groups(vdim, gso);
        const float *ssm_norm_w = lz_t_f32(&L->ssm_norm, s->wscr);
        if (lz_norm_int() && lz_norm_can_fixed(vd) &&
            gso > 0 && (vd % gso) == 0) {
            /* Per-head quantize, not one whole-vdim call: the norm runs
               per head and each head carries its own deq, so a Q8 group
               must not straddle two heads' different scales. gso divides
               vd, so the group boundaries and the s->xqs layout are
               identical to the whole-vdim call below. ssm_sig is
               otherwise unused in this path (the sig argument is NULL)
               and is allocated nt*vdim int32, the same span as ssm_out. */
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++) {
                    float deq;
                    lz_rmsnorm_gated_int(s->ssm_sig + (size_t)tk * vdim + (size_t)h * vd,
                                         s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                         s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                         ssm_norm_w, vd,
                                         c->rms_norm_eps, &deq);
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
                                     ssm_norm_w, vd,
                                     c->rms_norm_eps,
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
                                     ssm_norm_w, vd,
                                     c->rms_norm_eps, NULL);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, gso,
                               s->xq + (size_t)tk * vdim, s->xqs + (size_t)tk * nso);
        }
        LZ_TAP("sgdn", layer, s->ssm_out, vdim);
        lz_matmul_xq_nt(s->xb2, s->ssm_out, s->xq, s->xqs, &L->out_proj, vdim,
                        hdim, nt);
    }
}

