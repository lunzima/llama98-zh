#include <string.h>
#include <math.h>
#include "model.h"
#include "forward.h"
#include "ops.h"

/* S2, DeepSeek-V4.1 report SS2.3 (CSA2 - Compressed Sparse Attention).
   model.h's own LZModelConfig.csa2_on[]/csa2_compress_m comment has the
   full citation trail (arXiv:2512.02556 indexer, arXiv:2606.19348
   compression) and the list of deliberate simplifications this engine
   side does not reproduce; engine/csa2.py (training tree) is the
   reference this mirrors formula for formula.

   UNLIKE forward_attn/forward_kda, this function CANNOT batch across
   the chunk's nt tokens: whether position t's own compressed block is
   already finalized - and therefore visible to a LATER position within
   the SAME chunk - depends on whether an EARLIER token in this same
   chunk just completed it. Every token is processed one at a time,
   reading and then updating the ring state between iterations; decode
   (nt=1) is simply the nt==1 case of this same loop, not a separate
   path.

   SCALAR REFERENCE: a new mechanism in this engine lands a scalar
   reference first and gets SIMD kernels in a later, separate phase -
   not started here. Every projection goes through lz_matmul_w, which
   already dispatches F32 vs quantized weight tensors on its own
   (model.c's exporter has no writer for this format yet, so every
   checkpoint that can reach this function today is F32 - see
   docs/dsv41-optimization-menu.md's CSA2 section), and the indexer
   top-k selection is a plain O(k * n_visible) partial selection sort:
   correct, not fast, matching this project's own precedent for a new
   mechanism's first landing (DSpark's synthetic-fixture validation,
   E3/S4's own scalar-only first pass).

   NOT SUPPORTED (see model.h's own comment for why each is out of
   scope for this pass, not simply forgotten):
     - Composition with S4's shared low-rank query latent - this
       function always reads L->csa2_w_dq_idx, unlike the training-side
       reference, which reuses q_proj.down when S4 already converted it.
     - Incremental cache checkpoint/restore (LZStateCkpt) - CSA2's own
       ring state (csa2_pending_c and friends, csa2_comp_ring,
       csa2_n_blocks) is not included in a checkpoint's saved fields;
       speculative decode
       rollback (--spec) is therefore not yet correct on a csa2_on
       layer and is out of scope for this pass, same as SIMD. */
void forward_csa2(const LZModel *m, LZRunState *s, const LZLayer *L,
                  int layer, int pos0, int nt) {
    const LZModelConfig *c = &m->config;
    int hd = c->head_dim, nh = c->n_heads;
    int hdim = c->hidden_size;
    int qgd = c->attn_qgate_dim, qd = c->attn_q_dim;
    /* cm is the divisor of max_blocks and of pos % cm below, and the
       multiplier of every pend_* offset into buffers that
       forward_state.c sizes with `cm > 0 ? cm : 1`. Read raw, a cm of 0
       therefore divides by zero here while the allocation next door
       assumed 1 - two different values for one quantity. Same guard as
       forward.c's reset and forward_state.c's allocator; lz_open_bin
       also refuses cm < 1, so this is the third line rather than the
       only one. */
    int cm = c->csa2_compress_m > 0 ? c->csa2_compress_m : 1;
    int n_idx_heads = c->csa2_indexer_heads;
    int idx_hd = c->csa2_indexer_head_dim;
    int idx_r = c->csa2_indexer_q_lora_rank;
    int ci = s->cache_idx[layer];
    int max_blocks = (s->seq_len + cm - 1) / cm;
    int topk = c->csa2_topk;
    float rms_eps = c->rms_norm_eps;
    int rotary_dim = c->rotary_dim;
    float scale = 1.0f / (float)sqrt((double)hd);

    float *pend_c = s->csa2_pending_c + (size_t)ci * cm * hd;
    float *pend_z = s->csa2_pending_z + (size_t)ci * cm * hd;
    float *pend_c_idx = s->csa2_pending_c_idx + (size_t)ci * cm * idx_hd;
    float *pend_z_idx = s->csa2_pending_z_idx + (size_t)ci * cm * idx_hd;
    float *ring = s->csa2_comp_ring + (size_t)ci * max_blocks * hd;
    float *ring_idx = s->csa2_idx_comp_ring + (size_t)ci * max_blocks * idx_hd;
    int tk;

    for (tk = 0; tk < nt; tk++) {
        int pos = pos0 + tk;
        int off = pos % cm;
        int blk = pos / cm;
        const float *h = s->xb + (size_t)tk * hdim;
        int n_vis = s->csa2_n_blocks[ci];
        int k_eff;
        int hh;

        /* ---- 1. main query: q_proj/q_norm/RoPE, same as ordinary full
           attention (forward_attn.c's own attn_impl). Gated layout
           hardcoded ([q(hd), gate(hd)] per head in q_proj's output),
           same assumption forward_attn.c's own code already makes -
           see this file's own header comment on why generalizing to
           attn_output_gate=0 is not attempted here either. */
        lz_matmul_w(s->qg, h, &L->q_proj, hdim, qgd, s->xq, s->xqs);
        {
            const float *qn = lz_t_f32(&L->q_norm, s->wscr);
            for (hh = 0; hh < nh; hh++)
                lz_rmsnorm(s->qh + (size_t)hh * hd,
                          s->qg + (size_t)hh * 2 * hd, qn, hd, rms_eps);
        }
        lz_rope(s->qh, nh, hd, rotary_dim, pos, s->rope_cs);

        /* ---- 2. this position's own raw contribution to the block
           that is still open (csa2_compress's per-token half - the
           softmax/weighted-sum half only runs once the block closes,
           step 6 below). */
        lz_matmul_w(s->csa2_c_j, h, &L->csa2_w_kv, hdim, hd, s->xq, s->xqs);
        lz_matmul_w(s->csa2_z_j, h, &L->csa2_w_gate, hdim, hd, s->xq, s->xqs);
        memcpy(pend_c + (size_t)off * hd, s->csa2_c_j, (size_t)hd * sizeof(float));
        memcpy(pend_z + (size_t)off * hd, s->csa2_z_j, (size_t)hd * sizeof(float));
        lz_matmul_w(s->csa2_c_j_idx, h, &L->csa2_w_kv_idx, hdim, idx_hd, s->xq, s->xqs);
        lz_matmul_w(s->csa2_z_j_idx, h, &L->csa2_w_gate_idx, hdim, idx_hd, s->xq, s->xqs);
        memcpy(pend_c_idx + (size_t)off * idx_hd, s->csa2_c_j_idx, (size_t)idx_hd * sizeof(float));
        memcpy(pend_z_idx + (size_t)off * idx_hd, s->csa2_z_j_idx, (size_t)idx_hd * sizeof(float));

        /* ---- 3. indexer query + Eq 1's score, over exactly the
           n_vis blocks finalized by STRICTLY EARLIER positions - n_vis
           was read above, before step 6 below can finalize THIS
           position's own block, so a query can never see the block it
           is itself inside (this file's own header comment's causality
           argument, engine/csa2.py's csa2_causal_block_mask mirrored
           without needing an explicit mask tensor: there is simply
           nothing in the ring past index n_vis-1 yet). */
        if (n_vis > 0) {
            int b;
            lz_matmul_w(s->csa2_cq_idx, h, &L->csa2_w_dq_idx, hdim, idx_r, s->xq, s->xqs);
            lz_matmul_w(s->csa2_qidx, s->csa2_cq_idx, &L->csa2_w_uq_idx,
                       idx_r, n_idx_heads * idx_hd, s->xq, s->xqs);
            lz_matmul_w(s->csa2_widx, h, &L->csa2_w_weights, hdim, n_idx_heads, s->xq, s->xqs);
            for (b = 0; b < n_vis; b++) {
                float score = 0.0f;
                const float *kv = ring_idx + (size_t)b * idx_hd;
                for (hh = 0; hh < n_idx_heads; hh++) {
                    const float *qv = s->csa2_qidx + (size_t)hh * idx_hd;
                    float dot = 0.0f;
                    int d;
                    for (d = 0; d < idx_hd; d++) dot += qv[d] * kv[d];
                    if (dot < 0.0f) dot = 0.0f;   /* Eq 1's ReLU */
                    score += s->csa2_widx[hh] * dot;
                }
                s->csa2_scores[b] = score;
            }
            /* Top-k, plain partial selection sort: correct, not fast -
               see this file's own header comment. csa2_topk_idx[i]
               ends up strictly decreasing in score as i increases. */
            k_eff = topk < n_vis ? topk : n_vis;
            {
                int kk;
                for (kk = 0; kk < k_eff; kk++) {
                    int best = -1;
                    float best_val = 0.0f;
                    for (b = 0; b < n_vis; b++) {
                        int used = 0, u;
                        for (u = 0; u < kk; u++)
                            if (s->csa2_topk_idx[u] == b) { used = 1; break; }
                        if (used) continue;
                        if (best == -1 || s->csa2_scores[b] > best_val) {
                            best = b;
                            best_val = s->csa2_scores[b];
                        }
                    }
                    s->csa2_topk_idx[kk] = best;
                }
            }
        } else {
            k_eff = 0;
        }

        /* ---- 4. shared-K=V MQA core attention (arXiv:2606.19348
           Eq 19: each selected compressed entry serves as BOTH key and
           value) over the k_eff selected entries. Zero output when
           k_eff is 0 (this token's own block is still the very first
           one open - nothing causal to attend to via CSA2 yet, same
           "no legitimate contribution" case engine/csa2.py's own
           nan_to_num sites exist for; here there is simply no NaN to
           begin with, since the loop below never runs). */
        if (k_eff > 0) {
            for (hh = 0; hh < nh; hh++) {
                const float *qv = s->qh + (size_t)hh * hd;
                float *outh = s->attn_out + (size_t)hh * hd;
                float maxv, sum;
                int kk, d;
                for (kk = 0; kk < k_eff; kk++) {
                    const float *ent = ring + (size_t)s->csa2_topk_idx[kk] * hd;
                    float dot = 0.0f;
                    for (d = 0; d < hd; d++) dot += qv[d] * ent[d];
                    s->csa2_attn_probs[kk] = dot * scale;
                }
                maxv = s->csa2_attn_probs[0];
                for (kk = 1; kk < k_eff; kk++)
                    if (s->csa2_attn_probs[kk] > maxv) maxv = s->csa2_attn_probs[kk];
                sum = 0.0f;
                for (kk = 0; kk < k_eff; kk++) {
                    s->csa2_attn_probs[kk] = lz_exp(s->csa2_attn_probs[kk] - maxv);
                    sum += s->csa2_attn_probs[kk];
                }
                for (d = 0; d < hd; d++) outh[d] = 0.0f;
                for (kk = 0; kk < k_eff; kk++) {
                    const float *ent = ring + (size_t)s->csa2_topk_idx[kk] * hd;
                    float p = s->csa2_attn_probs[kk] / sum;
                    for (d = 0; d < hd; d++) outh[d] += p * ent[d];
                }
            }
        } else {
            memset(s->attn_out, 0, (size_t)qd * sizeof(float));
        }

        /* ---- 5. output gate (q_proj's own second half, same
           convention as forward_attn.c) + o_proj -> this layer's
           residual contribution. */
        for (hh = 0; hh < nh; hh++) {
            const float *gate = s->qg + (size_t)hh * 2 * hd + hd;
            float *outh = s->attn_out + (size_t)hh * hd;
            int d;
            for (d = 0; d < hd; d++) outh[d] *= lz_sigmoid(gate[d]);
        }
        lz_matmul_w(s->xb2 + (size_t)tk * hdim, s->attn_out, &L->o_proj,
                   qd, hdim, s->xq, s->xqs);

        /* ---- 6. finalize this position's block if it just completed
           one (csa2_compress's softmax/weighted-sum half). Main KV
           gets RMSNorm (k_norm, reused - both are exactly head_dim
           wide, arXiv:2606.19348 SS2.3.3) + RoPE at the block's nominal
           position (its own last raw position - engine/csa2.py's
           csa2_block_positions has the same choice and the same
           reasoning). Indexer keys get NEITHER, matching
           engine/csa2.py's own K_idx (Eq 1 does not call for either on
           the indexer side). */
        if (off == cm - 1) {
            int j, d;
            /* csa2_c_j/csa2_c_j_idx are dead by now - their only use
               this iteration was step 2's copy into pend_c/pend_c_idx
               above - so they double as this block's finalized-sum
               scratch instead of adding two more allocations sized
               identically to ones that already exist. */
            float *sum_c = s->csa2_c_j;
            float *sum_c_idx = s->csa2_c_j_idx;
            float maxv, denom;
            /* main KV: softmax over the cm pending gate logits, per
               output dim independently (arXiv:2606.19348 Eq 11's
               Softmax_row is across the m raw positions, for EACH of
               the c output dims separately - not a single scalar
               weight per position). */
            for (d = 0; d < hd; d++) {
                maxv = pend_z[d];
                for (j = 1; j < cm; j++)
                    if (pend_z[(size_t)j * hd + d] > maxv) maxv = pend_z[(size_t)j * hd + d];
                denom = 0.0f;
                for (j = 0; j < cm; j++) {
                    float e = lz_exp(pend_z[(size_t)j * hd + d] - maxv);
                    pend_z[(size_t)j * hd + d] = e;   /* reuse as the un-normalized weight */
                    denom += e;
                }
                sum_c[d] = 0.0f;
                for (j = 0; j < cm; j++)
                    sum_c[d] += (pend_z[(size_t)j * hd + d] / denom) * pend_c[(size_t)j * hd + d];
            }
            lz_rmsnorm(ring + (size_t)blk * hd, sum_c,
                      lz_t_f32(&L->k_norm, s->wscr), hd, rms_eps);
            lz_rope(ring + (size_t)blk * hd, 1, hd, rotary_dim, pos, s->rope_cs);

            for (d = 0; d < idx_hd; d++) {
                maxv = pend_z_idx[d];
                for (j = 1; j < cm; j++)
                    if (pend_z_idx[(size_t)j * idx_hd + d] > maxv) maxv = pend_z_idx[(size_t)j * idx_hd + d];
                denom = 0.0f;
                for (j = 0; j < cm; j++) {
                    float e = lz_exp(pend_z_idx[(size_t)j * idx_hd + d] - maxv);
                    pend_z_idx[(size_t)j * idx_hd + d] = e;
                    denom += e;
                }
                sum_c_idx[d] = 0.0f;
                for (j = 0; j < cm; j++)
                    sum_c_idx[d] += (pend_z_idx[(size_t)j * idx_hd + d] / denom) * pend_c_idx[(size_t)j * idx_hd + d];
            }
            memcpy(ring_idx + (size_t)blk * idx_hd, sum_c_idx, (size_t)idx_hd * sizeof(float));

            s->csa2_n_blocks[ci] = blk + 1;
        }
    }
}
