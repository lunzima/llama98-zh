/* Recurrent-state checkpoints: the four entry points and the one size
   derivation they share, split out of forward.c along the function seam.
   Code motion only - every body here is letter-for-letter what forward.c
   held, and ckpt_sizes stays `static` because nothing outside this file
   calls it.

   A checkpoint carries the recurrent half of LZRunState - SSM state, conv
   history, engram histories - and deliberately NOT the KV cache: restore
   refuses unless the epoch still matches, see its own comment on that. */

#include <stdlib.h>
#include <string.h>

#include "forward.h"
#include "err.h"

/* ------------------------------------------------ recurrent checkpoint */

/* Element counts of the buffers a checkpoint copies. Derived here once
   so save, restore and alloc cannot disagree about the shapes - they
   did not have to, and that is exactly how one of them ends up copying
   a stale size after a config field changes. n_engram_conv mirrors the
   row*dim sizing forward.c's alloc_engram block uses for
   s->engram_conv_hist, and is 0 when config.engram is 0 (same
   condition that block itself gates on). The two csa2_* counts mirror
   forward_state.c's alloc_csa2_ring the same way. */
static void ckpt_sizes(const LZModelConfig *c, size_t *n_q8, size_t *n_s,
                       size_t *n_conv, size_t *n_engram_conv,
                       size_t *n_csa2_pend, size_t *n_csa2_pend_idx) {
    *n_q8   = (size_t)c->n_linear_layers * c->lin_n_v_heads *
              c->lin_k_head_dim * c->lin_v_head_dim;
    *n_s    = *n_q8 / 32;
    *n_conv = (size_t)c->n_linear_layers * c->lin_conv_dim *
              (c->conv_kernel - 1);
    if (c->engram) {
        int rows = (c->engram_conv_kernel - 1) * c->engram_conv_dilation;
        if (rows < 0) rows = 0;
        *n_engram_conv = (size_t)rows * (size_t)c->hidden_size;
    } else {
        *n_engram_conv = 0;
    }
    /* CSA2's open block - forward.h's LZStateCkpt comment has the
       whole argument for why it is in the checkpoint at all, and for
       why it is ONE block (nf * cm * width, cm floored at 1) rather
       than the whole banked ring alloc_csa2_ring sizes: save and
       restore address the bank the checkpointed position selects, so
       this count does not move with ssm_ring_depth. Predicate is
       forward_state.c's alloc_csa2_ring copied exactly (any csa2_on
       layer) - L->csa2_on is c->csa2_on[li] verbatim (model.c's
       setup_layers), so reading the config array here is the same
       predicate the allocator ran on - and a checkpoint that sized
       these differently from the state they are copied in and out of is
       a memcpy past the end, not an error message. */
    {
        int any_csa2 = 0, li;
        for (li = 0; li < c->n_layers; li++)
            if (c->csa2_on[li]) { any_csa2 = 1; break; }
        if (any_csa2) {
            /* One block per slot - depth 1, since the pending buffers'
               own banked depth is the RUNNING state's business and this
               count is the checkpoint's copy of each slot's open block.
               lz_csa2_pend_base(c, n_full_layers, 1) is the end of the
               same prefix sum save/restore index through (model.c), so
               the two agree per slot by construction rather than by both
               sides repeating one cm. */
            size_t blocks = lz_csa2_pend_base(c, c->n_full_layers, 1);
            *n_csa2_pend = blocks * (size_t)c->head_dim;
            *n_csa2_pend_idx = blocks * (size_t)c->csa2_indexer_head_dim;
        } else {
            *n_csa2_pend = 0;
            *n_csa2_pend_idx = 0;
        }
    }
}


int lz_ckpt_alloc(LZStateCkpt *ck, const LZModel *m, char *errbuf, int errlen) {
    size_t n_q8, n_s, n_conv, n_engram_conv, n_csa2_pend, n_csa2_pend_idx;
    if (!ck || !m) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    memset(ck, 0, sizeof(*ck));
    ckpt_sizes(&m->config, &n_q8, &n_s, &n_conv, &n_engram_conv,
               &n_csa2_pend, &n_csa2_pend_idx);
    ck->ssm_q8 = (int8_t *)malloc(n_q8);
#if LZ_GDN_STATE_2PLANE
    ck->ssm_q8_lo = (int8_t *)malloc(n_q8);
#endif /* LZ_GDN_STATE_2PLANE */
    ck->ssm_s = (float *)malloc(n_s * sizeof(float));
    ck->conv  = (float *)malloc(n_conv * sizeof(float));
#if LZ_CONV_FIXED
    /* Allocated unconditionally, not on lz_conv_mode(): a checkpoint
       outlives the flag that was set when it was made, and half the
       size of the float one is not worth a conditional that can be
       wrong. */
    ck->conv_q = (short *)malloc(n_conv * sizeof(short));
    if (!ck->conv_q) { lz_ckpt_free(ck); if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_OOM); return LZ_ERR_OOM; }
#endif /* LZ_CONV_FIXED */
    /* NULL (not malloc(0)) when config.engram is 0, same convention as
       s->engram_conv_hist itself. */
    ck->engram_conv_hist = n_engram_conv > 0
        ? (float *)malloc(n_engram_conv * sizeof(float)) : NULL;
    /* Same convention for CSA2's open block: NULL on a model with no
       csa2_on layer, matching the run state's own pointers. */
    if (n_csa2_pend > 0) {
        ck->csa2_pend_c = (float *)malloc(n_csa2_pend * sizeof(float));
        ck->csa2_pend_z = (float *)malloc(n_csa2_pend * sizeof(float));
        ck->csa2_pend_c_idx = (float *)malloc(n_csa2_pend_idx * sizeof(float));
        ck->csa2_pend_z_idx = (float *)malloc(n_csa2_pend_idx * sizeof(float));
    }
    if (!ck->ssm_q8 || !ck->ssm_s || !ck->conv
#if LZ_GDN_STATE_2PLANE
        || !ck->ssm_q8_lo
#endif /* LZ_GDN_STATE_2PLANE */
        || (n_engram_conv > 0 && !ck->engram_conv_hist)
        || (n_csa2_pend > 0 && (!ck->csa2_pend_c || !ck->csa2_pend_z ||
                                !ck->csa2_pend_c_idx || !ck->csa2_pend_z_idx))
        ) {
        lz_ckpt_free(ck);
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_STATE_ALLOC);
        return LZ_ERR_STATE_ALLOC;
    }
    ck->n_ssm_q8 = n_q8;
    ck->n_ssm_s  = n_s;
    ck->n_conv   = n_conv;
    ck->n_engram_conv = n_engram_conv;
    ck->n_csa2_pend = n_csa2_pend;
    ck->n_csa2_pend_idx = n_csa2_pend_idx;
    ck->pos      = -1;          /* empty until a save */
    ck->epoch    = 0;
    return LZ_ERR_OK;
}

void lz_ckpt_free(LZStateCkpt *ck) {
    if (!ck) return;
    free(ck->ssm_q8);
#if LZ_GDN_STATE_2PLANE
    free(ck->ssm_q8_lo);
#endif /* LZ_GDN_STATE_2PLANE */
    free(ck->ssm_s);
    free(ck->conv);
    free(ck->conv_q);
    free(ck->engram_conv_hist);
    free(ck->csa2_pend_c);
    free(ck->csa2_pend_z);
    free(ck->csa2_pend_c_idx);
    free(ck->csa2_pend_z_idx);
    memset(ck, 0, sizeof(*ck));
}

int lz_ckpt_save(LZStateCkpt *ck, const LZRunState *s, const LZModel *m,
                 int pos, char *errbuf, int errlen) {
    size_t n_q8, n_s, n_conv, n_engram_conv, n_csa2_pend, n_csa2_pend_idx;
    size_t o_q8, o_s, o_conv;   /* byte offset of s's OWN active ring slot */
    if (!ck || !s || !m || pos < 0 || !ck->ssm_q8) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    ckpt_sizes(&m->config, &n_q8, &n_s, &n_conv, &n_engram_conv,
               &n_csa2_pend, &n_csa2_pend_idx);
    if (n_q8 != ck->n_ssm_q8 || n_s != ck->n_ssm_s || n_conv != ck->n_conv ||
        n_engram_conv != ck->n_engram_conv ||
        n_csa2_pend != ck->n_csa2_pend ||
        n_csa2_pend_idx != ck->n_csa2_pend_idx) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CKPT_SHAPE);
        return LZ_ERR_CKPT_SHAPE;
    }
    /* Read from s->ssm_slot's OWN slot, not always offset 0 (the ring -
       forward.h's s->ssm_slot comment). This checkpoint is a SEPARATE
       mechanism from the ring (LZPrefixCache's cross-turn reuse,
       unrelated to lz_spec_round's own per-round rollback, which does
       not use LZStateCkpt at all - see lz_spec_round's own module
       comment), but it shares the SAME LZRunState, and a prior turn's
       own speculative decoding can leave s->ssm_slot pointing anywhere
       in the ring by the time this runs - offset 0 would then be stale,
       unrelated data. ring_depth is 1 for a model with no MTP head
       (lz_state_alloc's own comment), so ssm_slot is always 0 there and
       this reduces to the single-slot behavior. */
    o_q8  = (size_t)s->ssm_slot * n_q8;
    o_s   = (size_t)s->ssm_slot * n_s;
    o_conv = (size_t)s->ssm_slot * n_conv;
    memcpy(ck->ssm_q8, s->ssm_state_q8 + o_q8, n_q8);
#if LZ_GDN_STATE_2PLANE
    memcpy(ck->ssm_q8_lo, s->ssm_state_q8_lo + o_q8, n_q8);
#endif /* LZ_GDN_STATE_2PLANE */
    memcpy(ck->ssm_s, s->ssm_state_s + o_s, n_s * sizeof(float));
#if LZ_CONV_FIXED
    /* Whichever history is live. Saving the float one while the fixed
       tier runs would restore a zero history and read as a subtle
       quality drift rather than as a broken checkpoint. */
    if (s->conv_fixed)
        memcpy(ck->conv_q, s->conv_state_q + o_conv, n_conv * sizeof(short));
    else
#endif /* LZ_CONV_FIXED */
    memcpy(ck->conv, s->conv_state + o_conv, n_conv * sizeof(float));
    /* Engram cross-chunk history - forward.h's engram_hist/
       engram_conv_hist field comments. Not ring-indexed (there is only
       one copy on s, unlike ssm/conv above): a speculative round's own
       rollback ring never touches these fields. */
    memcpy(ck->engram_hist, s->engram_hist, sizeof(ck->engram_hist));
    ck->engram_hist_n = s->engram_hist_n;
    if (n_engram_conv > 0)
        memcpy(ck->engram_conv_hist, s->engram_conv_hist,
              n_engram_conv * sizeof(float));
    ck->engram_conv_hist_n = s->engram_conv_hist_n;
    /* CSA2's open compressed block. The ring's other half
       (csa2_comp_ring) deliberately stays out - forward.h's LZStateCkpt
       comment derives why a restore can leave it alone. Guarded on the
       source pointers as well as the counts because the run state
       allocates these only when a layer is csa2_on, on a predicate that
       has to agree with ckpt_sizes' own; a disagreement is a memcpy of
       a length neither side sized, and it is cheaper to copy nothing
       than to find out at the next reset. */
    /* CSA2's open compressed block - one block per ci, read out of the
       BANK the block open at `pos` lives in (forward.h's csa2_pending_c
       comment has the layout, LZStateCkpt's own the reason a checkpoint
       carries one block and not the whole banked ring). */
    if (n_csa2_pend > 0 && s->csa2_pending_c) {
        size_t depth = (size_t)s->ssm_ring_depth;
        size_t hd = (size_t)m->config.head_dim;
        size_t ihd = (size_t)m->config.csa2_indexer_head_dim;
        int i;
        for (i = 0; i < m->config.n_full_layers; i++) {
            /* Per slot: the ci's own ratio decides both how big its
               block is and which bank position pos puts it in
               (lz_csa2_slot_cm/lz_csa2_pend_base, model.c - the layout
               the allocator and forward_csa2.c use). The checkpoint's
               copy is one block per slot, so its dst base is the same
               prefix sum evaluated at depth 1. */
            int cmi = lz_csa2_slot_cm(&m->config, i);
            size_t src, dst;
            if (cmi < 1) cmi = 1;
            src = lz_csa2_pend_base(&m->config, i, (int)depth) +
                  (((size_t)pos / (size_t)cmi) % depth) * (size_t)cmi;
            dst = lz_csa2_pend_base(&m->config, i, 1);
            memcpy(ck->csa2_pend_c + dst * hd, s->csa2_pending_c + src * hd,
                   (size_t)cmi * hd * sizeof(float));
            memcpy(ck->csa2_pend_z + dst * hd, s->csa2_pending_z + src * hd,
                   (size_t)cmi * hd * sizeof(float));
            memcpy(ck->csa2_pend_c_idx + dst * ihd,
                   s->csa2_pending_c_idx + src * ihd,
                   (size_t)cmi * ihd * sizeof(float));
            memcpy(ck->csa2_pend_z_idx + dst * ihd,
                   s->csa2_pending_z_idx + src * ihd,
                   (size_t)cmi * ihd * sizeof(float));
        }
    }
    ck->pos   = pos;
    ck->epoch = s->epoch;
    return LZ_ERR_OK;
}

int lz_ckpt_restore(const LZStateCkpt *ck, LZRunState *s, const LZModel *m,
                    int *out_pos, char *errbuf, int errlen) {
    size_t n_q8, n_s, n_conv, n_engram_conv, n_csa2_pend, n_csa2_pend_idx;
    if (!ck || !s || !m || !ck->ssm_q8) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    if (ck->pos < 0) {          /* never saved into */
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CKPT_EMPTY);
        return LZ_ERR_CKPT_EMPTY;
    }
    /* The KV cache is not in the checkpoint, so it has to still hold the
       prefix this state was saved with. A reset (or a fresh alloc) means
       it does not. Refusing here is the whole point of the epoch field:
       the alternative is a correct recurrent state sitting on top of a
       zeroed KV, which produces confident nonsense and no diagnostic. */
    if (ck->epoch != s->epoch) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CKPT_STALE);
        return LZ_ERR_CKPT_STALE;
    }
    ckpt_sizes(&m->config, &n_q8, &n_s, &n_conv, &n_engram_conv,
               &n_csa2_pend, &n_csa2_pend_idx);
    if (n_q8 != ck->n_ssm_q8 || n_s != ck->n_ssm_s || n_conv != ck->n_conv ||
        n_engram_conv != ck->n_engram_conv ||
        n_csa2_pend != ck->n_csa2_pend ||
        n_csa2_pend_idx != ck->n_csa2_pend_idx || ck->pos >= s->seq_len) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CKPT_SHAPE);
        return LZ_ERR_CKPT_SHAPE;
    }
    /* Write to slot 0 and REPOINT s->ssm_slot there (the ring): a
       restore establishes a fresh baseline, and slot 0 is as good a
       place as any to put it - lz_state_reset uses the same convention
       (ssm_slot=0), and this is exactly what "restore" ought to mean:
       as if the state had been freshly built up to ck->pos and nothing
       else. Every OTHER ring slot becomes irrelevant garbage until some
       future speculative round overwrites it, same as after any reset.
       Without this, s->ssm_slot would keep pointing at whatever slot a
       PRIOR turn's own speculative decoding last left it on - stale
       data unrelated to what was just restored - and every forward call
       after this one would silently read the wrong slot. Covered by
       test_cli_ckpt_parity.py / test_resume_parity.py's cross-turn
       checks against a session that also uses --spec. */
    memcpy(s->ssm_state_q8, ck->ssm_q8, n_q8);
#if LZ_GDN_STATE_2PLANE
    memcpy(s->ssm_state_q8_lo, ck->ssm_q8_lo, n_q8);
#endif /* LZ_GDN_STATE_2PLANE */
    memcpy(s->ssm_state_s, ck->ssm_s, n_s * sizeof(float));
#if LZ_CONV_FIXED
    if (s->conv_fixed)
        memcpy(s->conv_state_q, ck->conv_q, n_conv * sizeof(short));
    else
#endif /* LZ_CONV_FIXED */
    memcpy(s->conv_state, ck->conv, n_conv * sizeof(float));
    memcpy(s->engram_hist, ck->engram_hist, sizeof(s->engram_hist));
    s->engram_hist_n = ck->engram_hist_n;
    if (n_engram_conv > 0)
        memcpy(s->engram_conv_hist, ck->engram_conv_hist,
              n_engram_conv * sizeof(float));
    s->engram_conv_hist_n = ck->engram_conv_hist_n;
    /* CSA2's open block, back into the bank the resumed position's block
       uses - the same bank it was saved from, since ck->pos is the
       position the pass resumes at and it is that block's slabs that
       forward needs. The forward that resumes at ck->pos continues from
       the FIRST raw position of the open block only if that block does
       not straddle ck->pos; when it does, the slab for position
       ck->pos - 1 is the one this copy supplies and nothing else can -
       forward.h's LZStateCkpt comment has the measurement. */
    if (n_csa2_pend > 0 && s->csa2_pending_c) {
        size_t depth = (size_t)s->ssm_ring_depth;
        size_t hd = (size_t)m->config.head_dim;
        size_t ihd = (size_t)m->config.csa2_indexer_head_dim;
        int i;
        for (i = 0; i < m->config.n_full_layers; i++) {
            /* The save side's own layout, read back into the bank the
               resumed position's block uses - per slot, since each ci
               carries its own ratio. */
            int cmi = lz_csa2_slot_cm(&m->config, i);
            size_t dst, src;
            if (cmi < 1) cmi = 1;
            dst = lz_csa2_pend_base(&m->config, i, (int)depth) +
                  (((size_t)ck->pos / (size_t)cmi) % depth) * (size_t)cmi;
            src = lz_csa2_pend_base(&m->config, i, 1);
            memcpy(s->csa2_pending_c + dst * hd, ck->csa2_pend_c + src * hd,
                   (size_t)cmi * hd * sizeof(float));
            memcpy(s->csa2_pending_z + dst * hd, ck->csa2_pend_z + src * hd,
                   (size_t)cmi * hd * sizeof(float));
            memcpy(s->csa2_pending_c_idx + dst * ihd,
                   ck->csa2_pend_c_idx + src * ihd,
                   (size_t)cmi * ihd * sizeof(float));
            memcpy(s->csa2_pending_z_idx + dst * ihd,
                   ck->csa2_pend_z_idx + src * ihd,
                   (size_t)cmi * ihd * sizeof(float));
        }
    }
    s->ssm_slot = 0;
    if (out_pos) *out_pos = ck->pos;
    return LZ_ERR_OK;
}
