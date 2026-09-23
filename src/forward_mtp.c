/* MTP: the draft head, the DSpark slot loop with its Markov correction, and
   the two speculative-prefill entry points, moved out of forward.c. Pure
   code motion - every body here is letter-for-letter what forward.c held,
   and none of it is `static`: forward.h already declared all six functions
   for their other callers.

   The MTP block is not one of m->layers[], so it owns the reserved KV cache
   slot LZ_MTP_CACHE_LAYER names below; the scratch it reads (mtp_x,
   mtp_concat, mtp_emb_raw, mtp_chain, mtp_kdraft_*, mtp_pos) is allocated
   once in forward_state.c's lz_state_alloc and documented field by field in
   forward.h. lz_debug_mtp_attn_t1_slow's definition deliberately stayed in
   forward.c, beside the comment forward.h points at for it. */

#include <string.h>

#include "forward.h"     /* includes compat.h for lz_time_ms (LZ_PROF* macros) */
#include "lz_int.h"    /* lz_i64: the 64-bit type, portably */
#include "ops.h"
#include "ops_quant.h"   /* q8_round/q8_amax/pow2f/LZ_Q8_*, lz_rsqrt_float_body */

/* layer sentinel for the MTP block's own attention: it is not one of
   m->layers[], so it has no entry in s->cache_idx or s->dense_idx. Its
   reserved DENSE KV slot is the one past every layer that owns a plane
   (index n_dense_kv - see lz_state_alloc's "+1 layer slot" comment); the
   sentinel just tells forward_attn to use that slot directly instead of
   indexing dense_idx[layer]. */
#define LZ_MTP_CACHE_LAYER (-1)

/* --------------------------------------------------------- MTP draft step */

float *lz_mtp_draft_step(const LZModel *m, LZRunState *s, int next_token, int pos) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size, i;
    unsigned fpu;
    float t_us0;                              /* debug probe, see part 4 */

    if (!m || !m->mtp) return NULL;
    if (next_token < 0 || next_token >= c->vocab_size) return NULL;
    if (pos < 0 || pos >= s->seq_len) return NULL;

    t_us0 = lz_time_ms() * LZ_US_SCALE;
    fpu = lz_fpu_float_begin();

    /* fc input: concat(pre_fc_norm_embedding(next_token's embedding),
       pre_fc_norm_hidden(s->mtp_chain)) - embedding occupies the LOW
       half, hidden the HIGH half (model.h's LZMtp comment;
       torch.cat([inputs_embeds, hidden_states], dim=-1)). s->mtp_chain
       holds either the body's raw hidden (seeded by lz_forward_capture,
       first step of a round) or the PREVIOUS step's own post-norm
       output (chained steps) - read here before being overwritten
       below by this step's own result. */
    lz_t_row_f32(&m->embed_tokens, next_token, dim, s->mtp_emb_raw);
    lz_rmsnorm(s->mtp_concat, s->mtp_emb_raw,
              lz_t_f32(&m->mtp->pre_fc_norm_embedding, s->wscr), dim,
              c->rms_norm_eps);
    lz_rmsnorm(s->mtp_concat + dim, s->mtp_chain,
              lz_t_f32(&m->mtp->pre_fc_norm_hidden, s->wscr), dim,
              c->rms_norm_eps);
    lz_matmul_w(s->mtp_x, s->mtp_concat, &m->mtp->fc, 2 * dim, dim,
               s->xq, s->xqs);

    /* One ordinary decoder layer (model.h: "the block is an ordinary
       full_attention layer"), same pre-norm/residual shape forward_chunk
       uses for every body layer, applied to s->mtp_x instead of s->x and
       to the MTP's own reserved KV cache slot (LZ_MTP_CACHE_LAYER).

       SubLN (c->use_subn): same pre-LN deletion forward_chunk's body
       loop applies to every ordinary layer (that loop's own comment).
       forward_attn's q/k/v/o SubLN norms and dense_ffn_step's
       gate/up/down SubLN norms are gated by the SAME c->use_subn flag
       and read s->xb expecting the RAW residual in that case - applying
       input_layernorm/post_attention_layernorm here unconditionally
       would double-normalize (once here, once inside forward_attn /
       dense_ffn_step) instead of the pre-LN being replaced by them. */
    if (c->use_subn) {
        for (i = 0; i < dim; i++) s->xb[i] = s->mtp_x[i];
    } else {
        lz_rmsnorm(s->xb, s->mtp_x,
                  lz_t_f32(&m->mtp->blk.input_layernorm, s->wscr), dim,
                  c->rms_norm_eps);
    }
    /* kv_in is NULL: the MTP block is not in m->layers, so it is never an
       attn_kv_share target (model.h's comment on the rule). */
    forward_attn(m, s, &m->mtp->blk, LZ_MTP_CACHE_LAYER, pos, 1, NULL);
    if (lz_debug_mtp_attn_scale != 1.0f) {
        /* Investigative probe only - see this file's own comment on
           lz_debug_mtp_attn_scale above. */
        for (i = 0; i < dim; i++) s->xb2[i] *= lz_debug_mtp_attn_scale;
    }
    for (i = 0; i < dim; i++) s->mtp_x[i] += s->xb2[i];

    if (c->use_subn) {
        for (i = 0; i < dim; i++) s->xb[i] = s->mtp_x[i];
    } else {
        lz_rmsnorm(s->xb, s->mtp_x,
                  lz_t_f32(&m->mtp->blk.post_attention_layernorm, s->wscr), dim,
                  c->rms_norm_eps);
    }
    dense_ffn_step(m, s, &m->mtp->blk, LZ_MTP_CACHE_LAYER, 1, c->mtp_intermediate_size);
    for (i = 0; i < dim; i++) s->mtp_x[i] += s->xb2[i];

    /* Exit norm - this step's OWN post-norm output. Doubles as (a) the
       source for this step's draft logits below and (b) VERBATIM the
       next chained step's "hidden" input (s->mtp_chain's comment,
       forward.h): upstream returns exactly this value and feeds it
       straight back into the next call, no extra processing. */
    lz_rmsnorm(s->mtp_chain, s->mtp_x,
              lz_t_f32(&m->mtp->norm, s->wscr), dim, c->rms_norm_eps);

    /* lm_head is the shared embedding - same tie_word_embeddings path
       forward_chunk's own lm_head step uses. */
    lz_quantize_q8(s->mtp_chain, dim, lz_act_gs(&m->embed_tokens, dim),
                  s->xq, s->xqs);
    lz_matmul_xq(s->mtp_draft_logits, s->mtp_chain, s->xq, s->xqs,
                &m->embed_tokens, dim, c->vocab_size);

    lz_fpu_float_end(fpu);
    lz_debug_us_draft += (lz_i64)(lz_time_ms() * LZ_US_SCALE - t_us0);
    return s->mtp_draft_logits;
}

/* T=1, no-history attention for one DSpark draft slot (LZ_MTP_CACHE_LAYER,
   position 0 - lz_mtp_draft_batch's own contract, below). Same xb-in/
   xb2-out convention forward_attn itself uses, so the call site swaps
   one function for the other and nothing else changes.

   WHY THIS IS SAFE, not an approximation: with no prior cache entry,
   softmax over the single score at position 0 is 1.0 for ANY finite
   score (softmax of a length-1 vector is the identity), regardless of
   what K/Q/RoPE/Hadamard would have produced it - verified numerically
   against the PyTorch reference (Qwen3_5DecoderLayer at T=1 reduces to
   o_proj(v_proj(x) * sigmoid(gate)) with Q/K/RoPE contributing nothing
   but the gate value, max abs diff 0.0 over a random tiny fixture) and
   checked structurally here too: LZ_KV_WALK(s, 0, t, rslot, 0) always
   iterates t=0 exactly once (forward.h), and the eviction mask's own
   range collapses to empty at pos=0 for every attn_sink/attn_win
   setting (dead0==dead1==attn_sink there), so position 0 can never be
   evicted either. K itself is therefore provably irrelevant to the
   result - not merely unread by this shortcut, structurally unable to
   change it - which is why k_proj, q_norm, k_norm, RoPE, the K-side
   Hadamard, the K cache write and the whole score/softmax pass are all
   simply absent below rather than computed and discarded.

   FULL TIER COVERAGE, not the default configuration only: every V
   storage format attn_impl (forward_attn.c) supports - plain Q8
   (attn_wsum_plane), the fixed-tier int64 accumulator
   (lz_attn_wsum_q8_int, which is this engine's DEFAULT per --fixed
   all), Q8 2-plane (LZ_KV_2PLANE), 4-bit (v4) and the float reference
   arm (vf) - plus the optional Hadamard V-rotation (kv_rot_v). Reached
   by calling the SAME primitives attn_impl/attn_score_loop call for
   each tier (attn_wsum_plane, lz_attn_wsum_q8_int, lz_gdn_quantize_2p,
   lz_kv4_quantize, rot_head, lz_quantize_q8_int64), not by re-deriving
   any tier's arithmetic a second time - the only genuinely new code
   here is which one to reach and computing Q (for the gate only) and V
   in the first place, never K.

   NOT reproduced: the vtmp_i16 v-projection fast path (LZ_VPROJ_I16's
   v_may/v_int) - documented as an exact, cheaper route to the same
   quantize result (forward_attn.c's own comment on v_int), so always
   taking the plain float v_proj -> lz_quantize_q8 route here reaches
   the identical output by construction, just not via that shortcut's
   own faster input. kv_only is not a parameter: this function always
   computes the full o_proj output, which is all lz_mtp_draft_batch's
   caller ever wants.

   deliberately NOT static: .prof/mtp_kdraft_check.c calls it directly
   for the tier-by-tier bit-identity comparison against forward_attn. */
void mtp_attn_t1(const LZModel *m, LZRunState *s, const LZLayer *L) {
    const LZModelConfig *c = &m->config;
    int hd = c->head_dim, nh = c->n_heads, nkv = c->n_kv_heads;
    int kv_mul = nh / nkv;
    int hdim = c->hidden_size, kvd = c->attn_kv_dim, qd = c->attn_q_dim;
    int qgd = c->attn_qgate_dim;
    float rms_eps = c->rms_norm_eps;
    /* The MTP's own reserved slot, which follows the last DENSE plane -
       see attn_impl's own comment on why this is n_dense_kv and not
       n_full_layers. */
    int ci = s->n_dense_kv;
    size_t loff = (size_t)ci * s->kv_slots * kvd;
    size_t soff = loff / 32;
    int8_t *vc = s->vq8 ? s->vq8 + loff : NULL;
    float  *vf = s->vf32 ? s->vf32 + loff : NULL;
    unsigned char *v4 = s->v4 ? s->v4 + loff / 2 : NULL;
    float *vs4 = s->vs4 ? s->vs4 + (size_t)ci * s->kv_slots * nkv : NULL;
    float *vs  = s->vsq ? s->vsq + soff : NULL;
#if LZ_KV_2PLANE
    int8_t *vc_lo = s->vq8_lo ? s->vq8_lo + loff : NULL;
#endif /* LZ_KV_2PLANE */
    /* No nsa/nso here (attn_impl's own scale_groups() results): both are
       only needed to stride xqs across nt>1 tokens, and every quantize/
       matmul call below is nt=1, so lz_quantize_q8/lz_matmul_xq_nt's own
       internal (in_dim/gs) grouping is all either one needs. */
    int gsa = lz_act_gs(&L->q_proj, hdim);
    int gso = lz_act_gs(&L->o_proj, qd);
    int wslot = kv_slot(s, 0);
    int use_int;
    int h, d;

    /* Q (gate only) and V, sharing one activation quantization the same
       way attn_impl's non-SubLN branch shares it across q/k/v: all three
       read the same in_dim, so lz_act_gs and the quantized xb are
       identical for each. SubLN instead gives q and v their own
       norm/quantize/matmul, same shape as attn_impl's p[0]/p[2]
       (p[1] - k_proj - has no counterpart here at all). */
    if (c->use_subn) {
        lz_rmsnorm(s->xb2, s->xb, lz_t_f32(&L->attn_q_subn_norm, s->wscr),
                  hdim, rms_eps);
        lz_quantize_q8(s->xb2, hdim, gsa, s->xq, s->xqs);
        lz_matmul_xq_nt(s->qg, s->xb2, s->xq, s->xqs, &L->q_proj, hdim, qgd, 1);
        lz_rmsnorm(s->xb2, s->xb, lz_t_f32(&L->attn_v_subn_norm, s->wscr),
                  hdim, rms_eps);
        lz_quantize_q8(s->xb2, hdim, gsa, s->xq, s->xqs);
        lz_matmul_xq_nt(s->vtmp, s->xb2, s->xq, s->xqs, &L->v_proj, hdim, kvd, 1);
    } else {
        lz_quantize_q8(s->xb, hdim, gsa, s->xq, s->xqs);
        lz_matmul_xq_nt(s->qg, s->xb, s->xq, s->xqs, &L->q_proj, hdim, qgd, 1);
        lz_matmul_xq_nt(s->vtmp, s->xb, s->xq, s->xqs, &L->v_proj, hdim, kvd, 1);
    }

    if (s->kv_rot_v)
        for (h = 0; h < nkv; h++) rot_head(s->vtmp + (size_t)h * hd, hd, s->kv_rot_v);

    /* Write V into whichever plane is allocated - same calls, same
       group width, same wslot attn_impl's own K/V write loop uses. */
#if LZ_KV_2PLANE
    if (vc) lz_gdn_quantize_2p(s->vtmp, kvd, 32, vc + (size_t)wslot * kvd,
                               vc_lo + (size_t)wslot * kvd,
                               vs + (size_t)wslot * kvd / 32, -1);
#else
    if (vc) lz_quantize_q8(s->vtmp, kvd, 32, vc + (size_t)wslot * kvd,
                           vs + (size_t)wslot * kvd / 32);
#endif /* LZ_KV_2PLANE */
    if (vf) memcpy(vf + (size_t)wslot * kvd, s->vtmp, (size_t)kvd * sizeof(float));
    if (v4)
        for (h = 0; h < nkv; h++)
            lz_kv4_quantize(s->vtmp + (size_t)h * hd, hd,
                            v4 + (size_t)wslot * kvd / 2 + (size_t)h * (hd >> 1),
                            vs4 + (size_t)wslot * nkv + h);

    /* softmax of the single score at position 0 is exactly 1.0 - see
       this function's own header comment. */
    s->att[0] = 1.0f;

    /* Same predicate attn_score_loop uses to pick the int64 accumulator,
       minus the LZ_ATTN_NORING/hd-range/window terms that are always
       true here (nt=1, no window override reaches the MTP's own T=1
       slots) and minus s->kv_rot_v's own check (already excluded by
       construction: kv_rot_v!=0 falls through to the v4/vf-shaped float
       arm below the same way attn_impl's OWN predicate excludes it). */
#if (LZ_ATTN_FIXED & 2)
    use_int = lz_attn_int() && (lz_attn_mode() & 2) && s->wsum_cbuf &&
              !c->use_subn && hd > 0 && (hd % 32) == 0 &&
              hd <= LZ_ATTN_MAX_HD && s->kv_rot_v == 0 && gso == 32 &&
              !v4 && !vf && !LZ_KV_2PLANE;
#else
    use_int = 0;
#endif /* LZ_ATTN_FIXED & 2 */

    for (h = 0; h < nh; h++) {
        int kvh = h / kv_mul;
        const float *gate = s->qg + (size_t)h * 2 * hd + hd;
        float *out = s->attn_out + (size_t)h * hd;

        if (v4) {
            const unsigned char *vt = v4 + (size_t)wslot * kvd / 2 +
                                      (size_t)kvh * (hd >> 1);
            float a = vs4[(size_t)wslot * nkv + kvh];   /* att[0]==1.0 */
            for (d = 0; d < hd; d += 2) {
                unsigned char pk = vt[d >> 1];
                out[d]     = a * lz_kv4_cents[pk & 0x0F];
                out[d + 1] = a * lz_kv4_cents[pk >> 4];
            }
        } else if (vf) {
            const float *vt = vf + (size_t)wslot * kvd + (size_t)kvh * hd;
            for (d = 0; d < hd; d++) out[d] = vt[d];
        } else if (use_int) {
            /* Compiled only where the int16 weighted-sum kernel exists.
               Its declaration (ops.h) and the two LZRunState buffers it
               reads (forward.h) sit inside this same #if, as does use_int's
               own assignment above - so without the macro this branch is
               unreachable but still COMPILED, and its body would name a
               function and two members that do not exist at that setting.
               That is what the all-switches build caught. */
#if (LZ_ATTN_FIXED & 2)
            lz_attn_wsum_q8_int(s->attn_acc + (size_t)h * hd,
                                s->attn_ss + (size_t)(h * hd) / 32,
                                s->att, hd, vc + (size_t)kvh * hd,
                                vs + (size_t)(kvh * hd) / 32, kvd, 0,
                                s->wsum_cbuf, s->wsum_cq, s->attn_sink,
                                s->kv_ring);
#endif /* LZ_ATTN_FIXED & 2 */
        } else {
            attn_wsum_plane(s, out, vc, vs, hd, kvd, kvh, 0, 0,
                            1, 0, 0);
#if LZ_KV_2PLANE
            attn_wsum_plane(s, s->wsum_lo, vc_lo, vs, hd, kvd, kvh, 0, 0,
                            1, 0, 0);
            for (d = 0; d < hd; d++)
                out[d] += s->wsum_lo[d] * (1.0f / LZ_GDN_LO_SCALE);
#endif /* LZ_KV_2PLANE */
        }

        if (s->kv_rot_v && !use_int) {
            rot_head(out, hd, s->kv_rot_v);
            for (d = 0; d < hd; d++) out[d] *= 1.0f / (float)s->kv_rot_v;
        }

        /* Output gate - see attn_score_loop's own comment on the int/
           float split. use_int already excludes kv_rot_v/v4/vf above,
           so this else-branch's "out" is always what those three arms
           (which never set use_int) or the plain float arm produced. */
        if (use_int) {
            int sig_fixed = lz_sig_mode();
            int64_t *ac = s->attn_acc + (size_t)h * hd;
            if (sig_fixed) {
                int32_t s15[64];
                int base;
                for (base = 0; base < hd; base += 64) {
                    int mrun = hd - base;
                    if (mrun > 64) mrun = 64;
                    lz_sigmoid_q15_run(gate + base, s15, mrun);
                    for (d = 0; d < mrun; d++)
                        ac[base + d] = (ac[base + d] * (int64_t)s15[d]) >> 15;
                }
            } else {
                for (d = 0; d < hd; d++) {
                    int32_t s15 = (int32_t)(lz_sigmoid(gate[d]) * 32768.0f);
                    ac[d] = (ac[d] * (int64_t)s15) >> 15;
                }
            }
        } else {
            for (d = 0; d < hd; d++) out[d] *= lz_sigmoid(gate[d]);
        }
    }

    if (c->use_subn) {
        lz_rmsnorm(s->attn_out, s->attn_out,
                  lz_t_f32(&L->attn_o_subn_norm, s->wscr), qd, rms_eps);
    }
    if (use_int) {
        int g;
        for (g = 0; g < (qd >> 5); g++)
            lz_quantize_q8_int64(s->attn_acc + g * 32, 32, 32,
                                 s->attn_ss[g], s->xq + g * 32,
                                 s->xqs + g);
    } else {
        lz_quantize_q8(s->attn_out, qd, gso, s->xq, s->xqs);
    }
    lz_matmul_xq_nt(s->xb2, s->attn_out, s->xq, s->xqs, &L->o_proj, qd, hdim, 1);
}

/* DSpark (arXiv:2607.05147). See forward.h's own comment on the
   two-function split (base stage here, Markov correction separate) and
   on why a for-loop over k slots is a faithful scalar-reference
   implementation of "no self-chain", not a compromise of it.

   Wired into lz_generate/--spec via generate.c's lz_spec_round_dspark /
   lz_spec_round_dspark_temp. Each slot's attention runs through
   mtp_attn_t1 above (the verified T=1 shortcut) unless
   lz_debug_mtp_attn_t1_slow forces the ordinary forward_attn path for
   comparison. */
float *lz_mtp_draft_batch(const LZModel *m, LZRunState *s, int next_token,
                          int k) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size, i, slot;
    unsigned fpu;

    if (!m || !m->mtp) return NULL;
    if (c->mtp_n_draft <= 1) return NULL;
    if (next_token < 0 || next_token >= c->vocab_size) return NULL;
    if (k <= 0 || k > c->mtp_n_draft) return NULL;

    fpu = lz_fpu_float_begin();

    /* pre_fc_norm_hidden(h_body) is the SAME for every slot (the anchor
       hidden state is shared, not re-derived per slot - this sharing,
       not a batched matmul, is what "one forward" means here) -
       computed once into the concat buffer's low third and left alone
       across the loop below. */
    lz_rmsnorm(s->mtp_kdraft_concat, s->mtp_chain,
              lz_t_f32(&m->mtp->pre_fc_norm_hidden, s->wscr), dim,
              c->rms_norm_eps);

    for (slot = 0; slot < k; slot++) {
        float *logits_row = s->mtp_kdraft_logits + (size_t)slot * c->vocab_size;

        /* Middle third: slot 0 gets the anchor's real, known next
           token; every slot >=1 gets the SAME shared mask_embed row -
           never a real token, at train or inference alike (see
           engine/mtp_head.py's MtpHead.forward docstring - this is the
           fix, not a shortcut). */
        if (slot == 0) {
            lz_t_row_f32(&m->embed_tokens, next_token, dim, s->mtp_emb_raw);
        } else {
            for (i = 0; i < dim; i++)
                s->mtp_emb_raw[i] = lz_t_f32(&m->mtp->mask_embed, s->wscr)[i];
        }
        lz_rmsnorm(s->mtp_kdraft_concat + dim, s->mtp_emb_raw,
                  lz_t_f32(&m->mtp->pre_fc_norm_embedding, s->wscr), dim,
                  c->rms_norm_eps);

        /* High third: this slot's own depth row - the only per-slot
           signal once slot 0's real-token privilege is spent. */
        lz_t_row_f32(&m->mtp->depth_embed, slot, dim,
                    s->mtp_kdraft_concat + 2 * dim);

        lz_matmul_w(s->mtp_x, s->mtp_kdraft_concat, &m->mtp->main_proj,
                   3 * dim, dim, s->xq, s->xqs);

        /* One ordinary decoder layer, position 0 of the MTP's OWN
           reserved cache slot - EVERY slot, not advancing. This is not
           an approximation: at T=1 with no prior cache entry, causal
           self-attention's softmax collapses to a single weight of
           1.0 regardless of Q/K/RoPE (softmax of one score is always
           1), so writing each slot's fresh K/V over the same position-0
           entry and reading it straight back is exactly the T=1
           computation, not a reuse-by-coincidence - verified numerically
           against the PyTorch reference (Qwen3_5DecoderLayer at T=1
           reduces to o_proj(v_proj(x) * sigmoid(gate)) with Q/K/RoPE
           contributing nothing but the gate value, max abs diff 0.0
           over a random tiny fixture). mtp_attn_t1 (above) exploits
           this directly; lz_debug_mtp_attn_t1_slow forces the ordinary
           forward_attn call instead, for the bit-identity comparison
           .prof/mtp_kdraft_check.c runs between the two. */
        if (c->use_subn) {
            for (i = 0; i < dim; i++) s->xb[i] = s->mtp_x[i];
        } else {
            lz_rmsnorm(s->xb, s->mtp_x,
                      lz_t_f32(&m->mtp->blk.input_layernorm, s->wscr), dim,
                      c->rms_norm_eps);
        }
        if (lz_debug_mtp_attn_t1_slow)
            forward_attn(m, s, &m->mtp->blk, LZ_MTP_CACHE_LAYER, 0, 1, NULL);
        else
            mtp_attn_t1(m, s, &m->mtp->blk);
        for (i = 0; i < dim; i++) s->mtp_x[i] += s->xb2[i];

        if (c->use_subn) {
            for (i = 0; i < dim; i++) s->xb[i] = s->mtp_x[i];
        } else {
            lz_rmsnorm(s->xb, s->mtp_x,
                      lz_t_f32(&m->mtp->blk.post_attention_layernorm, s->wscr),
                      dim, c->rms_norm_eps);
        }
        dense_ffn_step(m, s, &m->mtp->blk, LZ_MTP_CACHE_LAYER, 1,
                       c->mtp_intermediate_size);
        for (i = 0; i < dim; i++) s->mtp_x[i] += s->xb2[i];

        /* Exit norm + lm_head, into this slot's own logits row - unlike
           lz_mtp_draft_step, nothing here feeds a NEXT call: there is
           no next call, every slot is independent. Reuses
           s->mtp_emb_raw as exit-norm scratch, NOT s->mtp_chain - that
           register belongs to the n_draft==1 chain (lz_mtp_draft_step's
           own contract, forward.h), and this function must leave it
           exactly as the caller seeded it (the anchor's own hidden
           state) rather than clobbering it with a mid-loop value that
           does not survive past this call anyway. mtp_emb_raw's own
           content (this slot's embed row) has already been fully
           consumed into mtp_kdraft_concat above, so reusing it here writes
           over nothing still needed. */
        lz_rmsnorm(s->mtp_emb_raw, s->mtp_x,
                  lz_t_f32(&m->mtp->norm, s->wscr), dim, c->rms_norm_eps);
        lz_quantize_q8(s->mtp_emb_raw, dim, lz_act_gs(&m->embed_tokens, dim),
                      s->xq, s->xqs);
        lz_matmul_xq(logits_row, s->mtp_emb_raw, s->xq, s->xqs,
                    &m->embed_tokens, dim, c->vocab_size);
    }

    lz_fpu_float_end(fpu);
    return s->mtp_kdraft_logits;
}

/* See forward.h's own comment. */
void lz_mtp_markov_correct(const LZModel *m, LZRunState *s, int x_prev,
                           float *logits) {
    const LZModelConfig *c = &m->config;
    int rank = c->mtp_markov_rank, v;
    unsigned fpu;
    /* mtp_kdraft_concat is (3*hidden) floats - always >= (rank), since rank
       is a small fraction of vocab_size (DSpark's own r=256 scaled) and
       hidden_size dominates markov_rank in every shape this project has
       built. Reused rather than a fourth buffer purely for a (rank)-
       sized row: mtp_kdraft_concat's OWN content is dead by the time this
       function runs (lz_mtp_draft_batch's loop has already finished). */
    float *embed_row = s->mtp_kdraft_concat;

    if (!m || !m->mtp || c->mtp_n_draft <= 1) return;
    if (x_prev < 0 || x_prev >= c->vocab_size) return;

    fpu = lz_fpu_float_begin();
    lz_t_row_f32(&m->mtp->markov_embed, x_prev, rank, embed_row);
    lz_matmul_w(s->mtp_kdraft_correction, embed_row, &m->mtp->markov_proj,
               rank, c->vocab_size, s->xq, s->xqs);
    for (v = 0; v < c->vocab_size; v++) logits[v] += s->mtp_kdraft_correction[v];
    lz_fpu_float_end(fpu);
}


/* Run the MTP block over n prompt positions purely to populate its own
   KV cache before the first speculative round (forward.h's s->mtp_pos
   comment). h_body_all[i]/next_tokens[i] are position pos0+i's body
   hidden and actual-next-token, exactly as lz_mtp_draft_step's own
   "hidden"/"next_token" inputs, just for every prefill position instead
   of one chained step.

   Only the ATTENTION half of the MTP block runs (fc -> input_layernorm
   -> forward_attn); the FFN half (dense_ffn_step) is skipped entirely.
   This is deliberate, not a shortcut: forward_attn's KV-cache write
   (this call's whole purpose) happens as a side effect of computing
   k/v for THIS position, independent of anything the FFN would add to
   the residual afterward - and the MTP block is exactly one layer, so
   there is no LATER position that ever reads this position's post-FFN
   mtp_x (unlike the body's multi-layer residual stream, where layer
   l+1 needs layer l's FFN output). The exit-norm/lm_head step is
   skipped for the same reason lz_spec_round never chains off a prefill
   position: s->mtp_chain is freshly reseeded by lz_forward_capture the
   moment a real round starts (generate.c), so nothing here needs to
   leave a valid chain register behind.

   fc's per-position concat + matmul stays a plain nt-loop (lz_matmul_w,
   one row at a time) - same shape forward_kda's f_b_proj loop uses for
   its own differently-shaped-than-hidden projection; the part that
   actually gets nt-batched is forward_attn, via its own nt parameter,
   exactly like every body layer's attention block. */
int lz_mtp_prefill(const LZModel *m, LZRunState *s, const float *h_body_all,
                   const int *next_tokens, int n, int pos0) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size;
    int cap, done = 0;
    unsigned fpu;

    if (!m || !m->mtp || !s || !h_body_all || !next_tokens || n < 1) return 1;

    cap = s->nt_cap;
    if (cap < 1) cap = 1;
    if (cap > LZ_BATCH_MAX) cap = LZ_BATCH_MAX;

    fpu = lz_fpu_float_begin();

    while (done < n) {
        int k = n - done, tk, i;
        if (k > cap) k = cap;

        for (tk = 0; tk < k; tk++) {
            const float *h_i = h_body_all + (size_t)(done + tk) * dim;
            float *concat_t = s->mtp_concat + (size_t)tk * 2 * dim;
            float *emb_t = s->mtp_emb_raw + (size_t)tk * dim;
            float *x_t = s->mtp_x + (size_t)tk * dim;

            lz_t_row_f32(&m->embed_tokens, next_tokens[done + tk], dim, emb_t);
            lz_rmsnorm(concat_t, emb_t,
                      lz_t_f32(&m->mtp->pre_fc_norm_embedding, s->wscr), dim,
                      c->rms_norm_eps);
            lz_rmsnorm(concat_t + dim, h_i,
                      lz_t_f32(&m->mtp->pre_fc_norm_hidden, s->wscr), dim,
                      c->rms_norm_eps);
            lz_matmul_w(x_t, concat_t, &m->mtp->fc, 2 * dim, dim,
                       s->xq, s->xqs);
        }

        /* SubLN (c->use_subn): same pre-LN deletion as lz_mtp_draft_step
           above - forward_attn's own q/k/v/o subn norms (gated by the
           same flag) expect the raw residual, not an already-normalized
           copy. See lz_mtp_draft_step's comment for the full argument. */
        if (c->use_subn) {
            for (i = 0; i < k * dim; i++) s->xb[i] = s->mtp_x[i];
        } else {
            for (tk = 0; tk < k; tk++)
                lz_rmsnorm(s->xb + (size_t)tk * dim, s->mtp_x + (size_t)tk * dim,
                          lz_t_f32(&m->mtp->blk.input_layernorm, s->wscr), dim,
                          c->rms_norm_eps);
        }
        /* The only thing this call needs: forward_attn writes this
           chunk's k/v rows into the MTP's reserved KV cache slot as a
           side effect (LZ_MTP_CACHE_LAYER) - s->xb2, its attention
           OUTPUT, is never read below. See this function's own comment
           above for why the FFN half is skipped. */
        forward_attn(m, s, &m->mtp->blk, LZ_MTP_CACHE_LAYER, pos0 + done, k, NULL);

        done += k;
    }

    lz_fpu_float_end(fpu);
    return 0;
}

/* Catch-up decode - see forward.h's own comment on why this is a
   separate entry point rather than callers just calling lz_mtp_prefill
   directly. Identical body to lz_mtp_prefill plus timing/counting;
   deliberately NOT sharing a common static helper with it - the two
   call sites' contracts read differently (one is "populate empty KV
   before the first round", the other is "replace speculative rows with
   verified ones after every round") even though the mechanics
   coincide, and forcing them through one name would blur exactly the
   distinction lz_debug_n_catchup exists to preserve. */
int lz_mtp_catchup(const LZModel *m, LZRunState *s, const float *h_body_all,
                   const int *next_tokens, int n, int pos0) {
    float t_us0 = lz_time_ms() * LZ_US_SCALE;
    int r = lz_mtp_prefill(m, s, h_body_all, next_tokens, n, pos0);
    lz_debug_us_catchup += (lz_i64)(lz_time_ms() * LZ_US_SCALE - t_us0);
    lz_debug_n_catchup++;
    return r;
}
