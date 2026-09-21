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

   FULL/REINDEX/REUSE (report Sec2.3.1's cross-layer KV reuse axis;
   model.h's own csa2_mode[]/csa2_source[] comment has the schema).
   L->csa2_mode picks which of the six numbered steps below actually
   run - forward_state.c already aliased a REINDEX/REUSE layer's own
   s->cache_idx[layer] onto its csa2_source's SAME value, so `ci`,
   `ring`, `ring_idx`, `pend_c` and friends below are, for such a layer,
   the exact same physical memory its source already reads and writes;
   no new per-layer state exists for that half. What DOES need a new
   per-layer distinction is which steps a call is allowed to touch that
   shared state with - see each step's own comment for which mode(s)
   run it and why running it under the wrong mode would corrupt a
   SEPARATE layer's own call for the same token:
     1. main query (q_proj/q_norm/RoPE)       - every mode
     2. own pending-block contribution         - FULL only
     3. finalize this position's block         - FULL only
     4. indexer query + Top-K selection        - FULL, REINDEX
     5. attention over the k_eff selected      - every mode
     6. output gate + o_proj                   - every mode
   Finalization sits at 3, ahead of the indexer, because a block is
   visible to the query standing on its own last token - the ordering
   that lets n_vis be one mode-independent formula. Step 3's own
   comment has that argument in full; the n_vis declaration and step
   4's comment point back to it rather than restating it, so there is
   exactly one copy of it to keep true.

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
     - RoPE on the indexer QUERY. The reference rotates it at the query
       token's own position - .prof/dsv41_ref/model.py:551,
       `apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])`
       - exactly as it rotates the indexer keys (:545), and .prof/
       dsv41_ref/model.py is the whole of the available specification
       (the report gives no formula for the indexer's own rotation, only
       the one-line mention of a fused FlashMLA kernel at
       .prof/dsv41_report.txt:969-970). Step 4 below scores the
       un-rotated s->csa2_qidx against ring_idx; the key half IS
       rotated (step 3), so the score is position-dependent through the
       keys alone - a different function from the reference's
       relative-position one, not an equivalent arrangement of it.
       engine/csa2.py carries the same entry on the training side, which
       rotates neither side of the indexer query either.

   Rollback, both kinds, is covered. LZStateCkpt's own comment has the
   cross-turn half (csa2_pending_c and friends were not among its saved
   fields, a live bug: a restore landing on an odd position at cm 2 left
   step 3 finalizing a block from the PREVIOUS turn's leftover slab for
   position p-1). A speculative round's own rollback rewinds nothing at
   all and needs to: lz_spec_round re-points s->ssm_slot and lets the
   next forward overwrite what the rejected rows wrote, which works for
   CSA2 only because of two properties of the code below - the pending
   slabs are banked by BLOCK (forward.h's csa2_pending_c comment), so a
   rejected row cannot reach a slab another block's closure will read,
   and step 3 leaves the gate logits it reads intact, so the re-forward
   of a block's last position finalizes it from the same data the
   rejected round would have used. */

/* The indexer key's own rotation, and the one place this file does not
   call lz_rope. The V4.1 reference rotates the TRAILING rotary_dim dims
   of that row and pairs ADJACENT elements as one complex number - its
   apply_rotary_emb is `view_as_complex(x.float().unflatten(-1, (-1, 2)))`
   times a complex frequency (.prof/dsv41_ref/model.py:545). lz_rope does
   neither: it rotates the LEADING rotary_dim dims, and pairs base[i] with
   base[i + rotary_dim/2] (src/ops_rope.c), which is the host family's
   rotate_half and stays the convention of the main query and the main KV
   entry this same step rotates. The indexer key and the main KV entry
   next to it are deliberately not in one convention.

   Scalar, like the rest of this file's first landing (header comment's
   SCALAR REFERENCE paragraph): rd/2 pairs per compressed block, against
   the projections this same step runs. `cs` is the same (pos, i*2 + 0/1)
   table lz_rope reads, and entry i is the angle pair i turns by. */
static void indexer_rope_adjacent(float *row, int width, int rotary_dim,
                                  int pos, const float *cs) {
    int half = rotary_dim / 2, base = width - rotary_dim, i;
    const float *row_cs = cs + (size_t)pos * half * 2;
    for (i = 0; i < half; i++) {
        float c = row_cs[i * 2];
        float sn = row_cs[i * 2 + 1];
        float x0 = row[base + 2 * i];
        float x1 = row[base + 2 * i + 1];
        row[base + 2 * i]     = x0 * c - x1 * sn;
        row[base + 2 * i + 1] = x1 * c + x0 * sn;
    }
}

void forward_csa2(const LZModel *m, LZRunState *s, const LZLayer *L,
                  int layer, int pos0, int nt, const float *kv_in) {
    const LZModelConfig *c = &m->config;
    int hd = c->head_dim, nh = c->n_heads;
    int hdim = c->hidden_size;
    int qgd = c->attn_qgate_dim, qd = c->attn_q_dim;
    /* cm is this layer's OWN compression ratio (model.h's own
       csa2_compress_m_layer comment on LZModelConfig): the divisor of
       pos below, the size of one pending block, and - through the two
       prefix sums this file gets its buffer bases from - the depth of
       this layer's ring. The resolver floors at 1 itself: lz_open_bin
       and lz_load_config both refuse a value below 1, and lz_csa2_cm
       exists so this file, the allocator and the reset cannot each
       answer that question differently. */
    int cm = lz_csa2_cm(c, layer);
    int n_idx_heads = c->csa2_indexer_heads;
    int idx_hd = c->csa2_indexer_head_dim;
    int idx_r = c->csa2_indexer_q_lora_rank;
    int ci = s->cache_idx[layer];
    int topk = c->csa2_topk;
    int mode = L->csa2_mode;
    float rms_eps = c->rms_norm_eps;
    int rotary_dim = c->rotary_dim;
    float scale = 1.0f / (float)sqrt((double)hd);

    /* Slabs are indexed by position modulo cm * ring_depth, not modulo
       cm: ring_depth slots are one bank, each bank holds cm raw
       positions, and bank (block % ring_depth) is where that block's
       positions land. forward.h's own csa2_pending_c comment derives
       why one bank is not enough under --spec; ring_depth is 1 for
       every state that never runs speculative decoding, where this is
       position % cm and the stride below is the single bank it always
       was. */
    size_t pdepth = (size_t)cm * (size_t)s->ssm_ring_depth;
    /* Both bases are PREFIX SUMS over the slots before this one, not
       ci * per-slot-size: a slot's own size is its owner layer's cm
       (lz_csa2_pend_base/lz_csa2_ring_base, model.c - the same two
       functions forward_state.c's allocator, forward.c's reset and
       forward_ckpt.c's save/restore size and index through). With no
       per-layer entry every slot's cm is the scalar and these are the
       single-cm layout again: ci * cm * ring_depth raw positions and
       ci * ceil(seq_len/cm) blocks. */
    float *pend_c = s->csa2_pending_c + lz_csa2_pend_base(c, ci, s->ssm_ring_depth) * hd;
    float *pend_z = s->csa2_pending_z + lz_csa2_pend_base(c, ci, s->ssm_ring_depth) * hd;
    /* s->csa2_pending_c_idx/csa2_pending_z_idx have no pointer here:
       nothing in this function reads or writes them, because the indexer
       K is projected from the finalized latent and has no per-position
       half. The buffers and their LZStateCkpt entries still exist
       (forward.h/forward_state.c/forward_ckpt.c own them). */
    float *ring = s->csa2_comp_ring + lz_csa2_ring_base(c, ci, s->seq_len) * hd;
    float *ring_idx = s->csa2_idx_comp_ring +
                      lz_csa2_ring_base(c, ci, s->seq_len) * idx_hd;
    /* This ci's published Top-K base (forward.h's own csa2_topk_idx_pub
       comment): one (topk)-wide slot per token-of-the-current-chunk, so
       a REUSE-mode layer processing token tk later in the dispatch
       order can still read what its (aliased) source published for
       THAT SAME tk, no matter how many other tokens/layers the source
       finished processing in between. */
    int *pub_idx_base = s->csa2_topk_idx_pub + (size_t)ci * LZ_BATCH_MAX * topk;
    int *pub_n_base = s->csa2_topk_n_pub + (size_t)ci * LZ_BATCH_MAX;
    int tk;

    for (tk = 0; tk < nt; tk++) {
        int pos = pos0 + tk;
        int off = pos % cm;
        int pslot = (int)((size_t)pos % pdepth);
        int blk = pos / cm;
        const float *h = s->xb + (size_t)tk * hdim;
        int *tok_idx = pub_idx_base + (size_t)tk * topk;
        int *tok_n = pub_n_base + tk;
        /* How many compressed blocks this position may attend to. A
           block becomes visible once the query has PASSED ITS LAST
           TOKEN, which for a 0-indexed pos is (pos + 1) / cm: the
           block containing pos is visible exactly on that block's own
           last raw position (off == cm - 1) and not one step earlier.
           That is the reference implementation's own compress_lens -
           (arange(1, seqlen+1) // ratio) while prefilling, end_pos //
           ratio while decoding, .prof/dsv41_ref/model.py 564 and 567 in
           the training tree - and engine/csa2.py's own
           csa2_causal_block_mask mirrors it there.

           ONE FORMULA, EVERY MODE, and one that has to be derived from
           the POSITION in every one of them, never counted in state
           shared with an alias: the ring a REINDEX layer reads also
           holds blocks belonging to LATER positions (step 3's own
           comment has the ordering argument, and why the alias root has
           already run by the time this layer reads - the reason a
           REINDEX-to-REINDEX chain works too), so for REINDEX the BOUND
           is the only thing keeping this query out of them, not the
           ring's contents. Sec2.3.1 defines the three modes by what
           each one obtains - they "differ in how they obtain main KV,
           indexer K, and Top-K indices" - and says nothing about
           masking, so this bound has to fall out of causality itself
           rather than out of a cache the layer only borrows.

           cm == 1 is the report's own named special case (SS2.3's
           "uncompressed main KV ... as a special case with a compression
           ratio of 1", which is what the reference's whole decoder half
           runs): n_vis is then pos + 1, every block holds exactly one
           raw position, and this degenerates into ordinary causal
           attention where a query sees its own position. */
        int n_vis = (pos + 1) / cm;
        int k_eff;
        int hh;

        /* ---- 1. main query: q_proj/q_norm/RoPE, same as ordinary full
           attention (forward_attn.c's own attn_impl). Gated layout
           hardcoded ([q(hd), gate(hd)] per head in q_proj's output),
           same assumption forward_attn.c's own code already makes -
           see this file's own header comment on why generalizing to
           attn_output_gate=0 is not attempted here either.

           q_proj keeps whatever the wiring put on it, SubLN included:
           a CSA2 layer is an LZ_LT_FULL layer, so _wire_full_attn_subn
           runs on it and attn_q_subn_norm is loaded here (model.c's
           present_subn spec). Under use_subn the pre-LN is gone and
           s->xb is the RAW residual, so q_proj reads it through that
           norm - the same norm-then-project forward_attn.c performs for
           q. s->xb2 is its scratch there too, and it is free here:
           this token's o_proj output has not been written yet.

           Every mode runs this step unchanged - Reindex/Reuse still
           need their own query to attend with in step 5 below, even
           when they compute none of their own K/V. */
        if (c->use_subn) {
            float *qin = s->xb2 + (size_t)tk * hdim;
            lz_rmsnorm(qin, h, lz_t_f32(&L->attn_q_subn_norm, s->wscr),
                       hdim, rms_eps);
            lz_matmul_w(s->qg, qin, &L->q_proj, hdim, qgd, s->xq, s->xqs);
        } else {
            lz_matmul_w(s->qg, h, &L->q_proj, hdim, qgd, s->xq, s->xqs);
        }
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
           step 3 directly below).

           FULL ONLY. A Reindex/Reuse layer's ci is ALIASED to its
           source's (forward_state.c's cache_idx fill) - if it also ran
           this step, it would write a SECOND, conflicting contribution
           into the exact same pend_c[pslot]/pend_z[pslot]/etc. slot its
           source layer's OWN, separate call to this function (each
           csa2_on layer gets its own top-level call from forward.c's
           per-layer dispatch loop) also writes for the SAME token: a
           genuine data race between two calls aliased to the same ci,
           not merely a wasted computation. */
        if (mode == LZ_CSA2_FULL) {
            /* CED "proj" (model.h's ced_kv comment): this layer's own
               main-KV compression input reads the encoder's frozen exit
               hidden instead of this position's own s->xb, when
               forward.c's kv_in resolution put this layer in CED's
               decoder region. The indexer K follows that substitution
               without being named here, because it is a projection of
               the latent this matmul feeds - the training side's own
               K_idx does the same (engine/csa2.py builds C_comp from
               hs_kv, then projects K_idx out of C_comp). Computed HERE,
               inside the FULL-only branch, rather than once per token
               above: kv_in is read ONLY by this step, and the training
               side's own _wire_ced never hooks CED's h_S substitution
               for a Reindex/Reuse-mode CSA2 layer either (report Sec2.3.1:
               "the Reindex and Reuse Modes are unchanged" under CED
               composition) - for those modes this line is not merely
               unused, it is never even evaluated. Step 1's query above
               stays on h regardless of mode - only the compression
               content moves. */
            const float *kvh = kv_in ? kv_in + (size_t)tk * hdim : h;
            lz_matmul_w(s->csa2_c_j, kvh, &L->csa2_w_kv, hdim, hd, s->xq, s->xqs);
            lz_matmul_w(s->csa2_z_j, h, &L->csa2_w_gate, hdim, hd, s->xq, s->xqs);
            memcpy(pend_c + (size_t)pslot * hd, s->csa2_c_j, (size_t)hd * sizeof(float));
            memcpy(pend_z + (size_t)pslot * hd, s->csa2_z_j, (size_t)hd * sizeof(float));
            /* No indexer half here. The indexer K is a projection OF the
               block's finalized latent (step 3), not a second per-token
               compression of hidden, so there is no per-position indexer
               intermediate for step 3 to weight - the whole
               csa2_pending_c_idx/csa2_pending_z_idx pair that used to
               carry one is unreferenced below. csa2_z_j_idx is likewise
               now written nowhere in this file. The allocations and the
               LZStateCkpt fields stay (forward_state.c/forward.h/forward_ckpt.c
               own them); only their readers here are gone. */
        }

        /* ---- 3. finalize this position's block if it just completed
           one (csa2_compress's softmax/weighted-sum half). Main KV
           gets RMSNorm (k_norm, reused - both are exactly head_dim
           wide, arXiv:2606.19348 SS2.3.3) + RoPE at the block's nominal
           position (its own last raw position - engine/csa2.py's
           csa2_block_positions has the same choice and the same
           reasoning). Indexer keys come out of THAT SAME latent, before
           the RoPE below rotates it in place: k_norm then wk then their
           own RMSNorm then RoPE - report Sec2.3.1's "CSA2 obtains
           indexer K by projecting main KV entries, replacing CSA's
           separate compression path from hidden states", and
           .prof/dsv41_ref/model.py:544-545's
           `k = self.k_norm(self.wk(latent))` followed by
           `apply_rotary_emb(k[..., -rd:], freqs)`. :749's own comment
           pins the ordering this depends on ("the indexer needs the
           latent before RoPE, so it runs before the cache is written") -
           .prof/dsv41_ref/model.py:758 rotates the latent into the
           cache AFTER the indexer has read it, exactly as the two
           lz_rope calls below are ordered.

           THE ORDERING ARGUMENT - AHEAD OF THE INDEXER, not after
           o_proj. This is where it lives in full; the file header, the
           n_vis declaration and step 4's comment each point back here
           rather than restating it, so a future move of this block has
           exactly one comment to keep true.

           Why it has to be ahead: a block becomes visible to the query
           standing on its own LAST raw token, so on exactly that token
           step 4 must find ring[blk]/ring_idx[blk] already written by
           the time it scores them. With finalization here, EVERY entry
           n_vis counts is written by the time step 4 reads it - which
           is what lets one mode-independent formula cover all three
           modes:
             FULL     wrote the newest of those entries itself, right
                      here, this same iteration;
             REINDEX  reads a ring filled by its alias ROOT's own call,
                      for every token of this chunk: forward_state.c
                      aliases this layer's ci onto the root's, and
                      forward.c dispatches layers in increasing order,
                      so the root ran strictly earlier. The root is not
                      necessarily this layer's DIRECT source - model.c's
                      own lz_csa2_last_writer_valid chains REINDEX
                      through REINDEX (`last_writer[r] = i` for a
                      REINDEX source too), and an intermediate REINDEX
                      source finalizes nothing at all, never running
                      this step. By the time this layer reads, the ring
                      therefore also holds blocks belonging to LATER
                      positions, which is why n_vis has to be a
                      per-position bound and never a shared counter (the
                      n_vis declaration's own comment has that half);
             REUSE    never reads the ring at all - step 4 skips REUSE
                      entirely - and takes its causal correctness from
                      whatever its source published into the shared
                      (ci, tk) slot.

           Moving it here is safe because finalization reads only
           pend_c/pend_z/pend_c_idx/pend_z_idx - this position's own
           contribution, written by step 2 directly above, plus the
           preceding cm-1 positions' from earlier iterations of this
           same loop or an earlier chunk - and produces nothing but
           ring[blk]/ring_idx[blk], a slot that was unwritten future
           capacity until now. Nothing it touches is produced by steps
           4-6, and nothing it writes was read by them beforehand.

           FULL ONLY, same reasoning as step 2: a Reindex/Reuse layer's
           aliased ci would double-finalize a block its source is also
           finalizing - and, since Reindex/Reuse never ran step 2 above,
           it would finalize from pend_c/pend_z it never wrote for THIS
           token, corrupting the source's own still-open block with a
           stale or zero contribution. */
        if (off == cm - 1 && mode == LZ_CSA2_FULL) {
            int j, d;
            /* This block's own cm slabs start at the base of its bank,
               (pslot - off), not at offset 0 of this ci's region: with
               more than one bank the block that just completed is not
               the first one. pbase is 0 whenever there is one bank. */
            int pbase = pslot - off;
            float *bz = pend_z + (size_t)pbase * hd;
            float *bc = pend_c + (size_t)pbase * hd;
            /* csa2_c_j is dead by now - its only use this iteration was
               step 2's copy into pend_c immediately above - so it doubles
               as this block's finalized-sum scratch instead of adding
               another allocation sized identically to one that already
               exists. csa2_c_j_idx is a different case: step 2 no longer
               writes it at all (the indexer has no per-token half left
               to store), so it arrives here free and the indexer
               projection below lands in it. */
            float *sum_c = s->csa2_c_j;
            float *sum_c_idx = s->csa2_c_j_idx;
            float maxv, denom;
            /* pend_z is READ here, never consumed: a block
               can be finalized more than once - a speculative round
               finalizes it from rows that are then rejected, and the
               re-forward of its last position finalizes it again - and
               the positions at or before the round's rollback point are
               never forwarded again. Storing the softmax weights back
               into pend_z would therefore leave that second finalization
               reading weights where it expects gate logits. lz_exp is
               deterministic, so recomputing it in the accumulation pass
               is bit-identical to the single evaluation it replaces. */
            /* main KV: softmax over the cm pending gate logits, per
               output dim independently (arXiv:2606.19348 Eq 11's
               Softmax_row is across the m raw positions, for EACH of
               the c output dims separately - not a single scalar
               weight per position). */
            for (d = 0; d < hd; d++) {
                maxv = bz[d];
                for (j = 1; j < cm; j++)
                    if (bz[(size_t)j * hd + d] > maxv) maxv = bz[(size_t)j * hd + d];
                denom = 0.0f;
                for (j = 0; j < cm; j++)
                    denom += lz_exp(bz[(size_t)j * hd + d] - maxv);
                sum_c[d] = 0.0f;
                for (j = 0; j < cm; j++)
                    sum_c[d] += (lz_exp(bz[(size_t)j * hd + d] - maxv) / denom)
                                * bc[(size_t)j * hd + d];
            }
            lz_rmsnorm(ring + (size_t)blk * hd, sum_c,
                      lz_t_f32(&L->k_norm, s->wscr), hd, rms_eps);

            /* Indexer K, projected out of the latent the line above just
               wrote - which is why this sits BETWEEN that lz_rmsnorm and
               the lz_rope below rather than after it. lz_rope rotates in
               place, so reading `ring` after it would read the rotated
               latent, and the reference explicitly hands the indexer the
               un-rotated one (model.py:749 "the indexer needs the latent
               before RoPE, so it runs before the cache is written"; its
               own lz_rope-equivalent, :758, runs after the indexer).
               wk is head_dim -> index_head_dim and k_norm_idx is
               index_head_dim wide, a different norm from the k_norm
               above (head_dim wide, main KV entries only).

               RoPE: the call below goes to indexer_rope_adjacent, not to
               lz_rope, because the reference differs from lz_rope on two
               axes at once - it rotates the row's TRAILING rotary_dim
               dims (`k[..., -rd:]`, model.py:545) and it pairs ADJACENT
               elements as one complex number, where lz_rope rotates the
               LEADING rotary_dim dims and pairs base[i] with
               base[i + rotary_dim/2]. Those slices are the same dims
               exactly when rotary_dim == indexer_head_dim, and the
               pairing is never lz_rope's. The main KV entry rotated on
               the next line keeps the leading slice and that same
               pairing, like every other rotated tensor in this engine, so
               these two rows are deliberately not in one convention.
               model.c refuses rotary_dim > csa2_indexer_head_dim
               (LZ_ERR_CFG_CSA2_IDX_ROTARY), so the trailing slice is
               inside the row and the offset is never negative. */
            lz_matmul_w(sum_c_idx, ring + (size_t)blk * hd, &L->csa2_w_kv_idx,
                        hd, idx_hd, s->xq, s->xqs);
            lz_rmsnorm(ring_idx + (size_t)blk * idx_hd, sum_c_idx,
                      lz_t_f32(&L->csa2_k_norm_idx, s->wscr), idx_hd, rms_eps);
            indexer_rope_adjacent(ring_idx + (size_t)blk * idx_hd, idx_hd,
                                  rotary_dim, pos, s->rope_cs);

            lz_rope(ring + (size_t)blk * hd, 1, hd, rotary_dim, pos, s->rope_cs);
        }

        /* ---- 4. indexer query + Eq 1's score, over exactly the
           n_vis blocks this position may see - n_vis was computed
           above as (pos + 1) / cm, one formula for every mode (see that
           declaration's own comment for the boundary and the reference
           it comes from), so a query CAN see the block it is itself
           inside, but only from that block's own last raw position
           onward, and never a block that closes later. Every one of
           those n_vis entries is present in the ring by now for every
           mode - step 3's own comment has that ordering argument in
           full, which is what makes the single n_vis formula valid -
           and for REINDEX it is still n_vis, not the ring's contents,
           that is the only thing enforcing causality here.
           No mask tensor is needed either way; the bound IS the mask.

           FULL and REINDEX: both compute their OWN fresh query/score/
           selection here, against the shared ring_idx - Reindex reads
           the same n_vis/ring_idx its aliased ci already gives it for
           free (no new state), but the selection itself uses THIS
           layer's own csa2_w_dq_idx/csa2_w_uq_idx/csa2_w_weights, which
           can and does differ from its source's. The result is
           published into this (ci, tk)'s slot for step 5 below to read
           - and, for a downstream REUSE layer sourced from THIS one,
           for that layer's own eventual call to read too.

           REUSE skips this entirely - "performs attention using this
           selection without computing indexer Q or evaluating index
           scores" (report Sec2.3.1) - and instead reads whatever is
           already published in its aliased ci's (ci, tk) slot: its
           source's FULL or REINDEX call already ran (forward.c
           dispatches layers in increasing order and a REUSE source is
           always strictly earlier), covering every token of this same
           chunk, so the slot for every tk in [0, nt) this loop will
           reach is already there by the time this call started. REUSE
           owns none of the seven csa2_w_* tensors (model.h's own
           csa2_mode[] comment; a REUSE layer's checkpoint carries none
           of them), so this branch reads no weight at all - the only
           way it could reach csa2_w_dq_idx/csa2_w_uq_idx/csa2_w_weights
           is by mistakenly falling into the branch below. */
        if (mode != LZ_CSA2_REUSE) {
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
                   see this file's own header comment. tok_idx[i]
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
                                if (tok_idx[u] == b) { used = 1; break; }
                            if (used) continue;
                            if (best == -1 || s->csa2_scores[b] > best_val) {
                                best = b;
                                best_val = s->csa2_scores[b];
                            }
                        }
                        tok_idx[kk] = best;
                    }
                }
            } else {
                k_eff = 0;
            }
            *tok_n = k_eff;
        } else {
            k_eff = *tok_n;
        }

        /* ---- 5. shared-K=V MQA core attention (arXiv:2606.19348
           Eq 19: each selected compressed entry serves as BOTH key and
           value) over the k_eff selected entries. Zero output when
           k_eff is 0, which at cm > 1 is every position before the
           first block closes - nothing causal to attend to via CSA2
           yet, the same "no legitimate contribution" case
           engine/csa2.py's own nan_to_num sites exist for; here there
           is simply no NaN to begin with, since the loop below never
           runs. At cm == 1 no position is ever in that case: every
           position closes its own block in step 3 and then attends to
           at least that one.

           Every mode runs this unchanged, reading tok_idx - for FULL/
           REINDEX this is what step 4 wrote in this same iteration;
           for REUSE it is whatever was already published there, which
           step 4's else-branch just read k_eff back out of (same
           pointer, same (ci, tk) slot - no mode branch needed here). */
        if (k_eff > 0) {
            for (hh = 0; hh < nh; hh++) {
                const float *qv = s->qh + (size_t)hh * hd;
                float *outh = s->attn_out + (size_t)hh * hd;
                float maxv, sum;
                int kk, d;
                for (kk = 0; kk < k_eff; kk++) {
                    const float *ent = ring + (size_t)tok_idx[kk] * hd;
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
                    const float *ent = ring + (size_t)tok_idx[kk] * hd;
                    float p = s->csa2_attn_probs[kk] / sum;
                    for (d = 0; d < hd; d++) outh[d] += p * ent[d];
                }
            }
        } else {
            memset(s->attn_out, 0, (size_t)qd * sizeof(float));
        }

        /* ---- 6. output gate (q_proj's own second half, same
           convention as forward_attn.c) + o_proj -> this layer's
           residual contribution.

           Every mode runs this unchanged - never shared, never moves:
           each layer's own o_proj always produces that layer's own
           residual contribution, whatever its csa2_mode.

           o_proj carries its own SubLN wrapper for the same reason
           q_proj does, over qd elements - the width forward_attn.c
           normalizes for the same projection. In place: s->attn_out is
           this token's only copy and is dead once the matmul has read
           it. */
        for (hh = 0; hh < nh; hh++) {
            const float *gate = s->qg + (size_t)hh * 2 * hd + hd;
            float *outh = s->attn_out + (size_t)hh * hd;
            int d;
            for (d = 0; d < hd; d++) outh[d] *= lz_sigmoid(gate[d]);
        }
        if (c->use_subn)
            lz_rmsnorm(s->attn_out, s->attn_out,
                       lz_t_f32(&L->attn_o_subn_norm, s->wscr), qd, rms_eps);
        lz_matmul_w(s->xb2 + (size_t)tk * hdim, s->attn_out, &L->o_proj,
                   qd, hdim, s->xq, s->xqs);
    }
}
