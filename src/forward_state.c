/* The state-allocation half of the engine: the 16-byte-aligned calloc
   pair every engine buffer goes through, the three per-tier build
   helpers (fixed conv taps, float conv taps, KDA decay-gate int16), and
   lz_state_alloc itself, split out of forward.c along the section seam.
   Code motion only - every body here is letter-for-letter what forward.c
   held.

   xcalloc and the three builders stay `static`: nothing outside this file
   calls them (the forward_kda/forward_ssm mentions of conv_f32_build are
   prose). xfree is shared - lz_state_free, still in forward.c, frees the
   same buffers - so it drops `static` and is declared in forward.h.

   lz_state_alloc is the entry point the front ends call; its contract,
   including what spec_k_max must be, lives at its declaration in
   forward.h. */

#include <stdlib.h>
#include <string.h>

#include "forward.h"
#include "lz_int.h"
#include "err.h"
#include "ops.h"
#include "ops_quant.h"   /* lz_sig_q15_fold/lz_sig_q15_t_offset */

/* ------------------------------------------------------------ state allocation */

/* 16-byte aligned calloc, so the SSE kernels that read these buffers can
   use movaps/movdqa (aligned) instead of movups/movdqu. Over-allocates
   by one pointer plus the pad and stores the ORIGINAL pointer just before
   the aligned one; xfree hands that back to free(). The SSE1/SSE2 loads
   in the operators that read these buffers then become aligned loads -
   the "unaligned loads throughout" note in ops_mmx_sse.c is what this
   removes. Not every engine buffer goes through here (the tensor slices
   from model.c do not), so not every _mm_loadu_* is convertible; only the
   ones whose pointer descends from xcalloc and whose offset is a multiple
   of 4 floats are. */
static void *xcalloc(size_t n, size_t sz, int *ok, lz_i64*acc) {
    char *base, *aligned;
    if (n == 0) return NULL;
    base = (char *)calloc(n * sz + 16 + sizeof(void *), 1);
    if (!base) { *ok = 0; return NULL; }
    aligned = base + sizeof(void *);
    aligned += (size_t)(16 - ((size_t)aligned & 15)) & 15;
    ((void **)(void *)aligned)[-1] = base;
    *acc += (lz_i64)n * (lz_i64)sz;
    return aligned;
}

void xfree(void *p) {
    if (p) free(((void **)p)[-1]);
}





#if LZ_CONV_FIXED
/* Quantize the conv taps once, laying them out per channel in the same
   order forward_kda/forward_ssm index the history.
 *
 * The offsets are duplicated from those two functions, which is a drift
 * surface, so the loop asserts that the channels it covered add up to
 * lin_conv_dim. A layout change then fails here instead of quantizing
 * the wrong tensor into the right slot. */
static int conv_fixed_build(LZRunState *s, const LZModel *m) {
    const LZModelConfig *c = &m->config;
    int layer, t, covered_ok = 1, built = 0;
    for (layer = 0; layer < c->n_layers; layer++) {
        const LZLayer *L;
        int li;
        size_t base;
        if (c->layer_types[layer] == LZ_LT_FULL) continue;
        L = &m->layers[layer];
        li = s->cache_idx[layer];
        base = (size_t)li * c->lin_conv_dim;
        built++;
        {
        size_t ch = 0;
        const LZTensor *tv[3];
        int cnt[3], ntv;
        if (L->kda_q_conv1d.q || L->kda_q_conv1d.f) {
            tv[0] = &L->kda_q_conv1d; cnt[0] = c->lin_key_dim;
            tv[1] = &L->kda_k_conv1d; cnt[1] = c->lin_key_dim;
            tv[2] = &L->kda_v_conv1d; cnt[2] = c->lin_value_dim;
            ntv = 3;
        } else {
            tv[0] = &L->conv1d; cnt[0] = c->lin_conv_dim; ntv = 1;
        }
        for (t = 0; t < ntv; t++) {
            int i;
            int row_dim = c->conv_kernel;  /* [C,1,K] -> row = K elements */
            for (i = 0; i < cnt[t]; i++) {
                /* Read one row (one channel's K taps) at a time instead of
                   the whole tensor through lz_t_f32. This lets BF16 conv
                   tensors stay narrow in RAM: lz_t_f32 would need the
                   entire [C,1,K] in s->wscr at once (4096+ elements for
                   a 1024-channel conv), overflowing it; lz_t_row_f32
                   needs only conv_kernel (=4) floats of scratch. */
                lz_t_row_f32(tv[t], i, row_dim, s->wscr);
                {
                signed char e = (signed char)lz_conv_norm_pow2(
                    s->wscr, row_dim,
                    s->conv_mw + (base + ch + i) * c->conv_kernel,
                    s->conv_bound);
                float oscale = pow2f(-((int)e + LZ_CONV_ES));
                lz_sig_q15_fold(oscale, &s->conv_sig_k1[base + ch + i],
                                &s->conv_sig_oscale2[base + ch + i]);
#if LZ_CONV_SIG_I
                s->conv_sig_e[base + ch + i] = (signed char)((int)e + LZ_CONV_ES);
#endif /* LZ_CONV_SIG_I */
                }
            }
            ch += (size_t)cnt[t];
        }
        if (ch != (size_t)c->lin_conv_dim) covered_ok = 0;
        }
    }
    /* Two coverage checks, not one: `covered_ok` catches a layer whose
       tensors do not add up to lin_conv_dim, `built` catches the case
       where the loop matched no layer at all - which would leave every
       tap zero and report success. */
    if (!covered_ok || built != c->n_linear_layers) return -1;
    s->conv_in_scale = pow2f(LZ_CONV_ES);
    return 0;
}
#endif /* LZ_CONV_FIXED */

/* Build a persistent, whole-tensor F32 cache of the causal conv taps for
 * the FLOAT tier - see conv_w32's own comment in forward.h for why this
 * exists and what per-token pattern it replaces. Deliberately NOT inside
 * an `#if LZ_CONV_FIXED`: in a build where that switch is 0 the float
 * tier is the only conv path there is, and this still has to run.
 *
 * Same per-layer walk, same layer-skip, same tv[]/cnt[] KDA-triple-vs-
 * single-conv1d branch and the same two coverage checks as
 * conv_fixed_build above - this is that function's un-quantized twin,
 * not a new design (the drift argument in conv_fixed_build's own comment
 * applies here too, which is why the offsets are duplicated rather than
 * shared: forward_ssm.c/forward_kda.c's own copies of this layout have
 * to keep agreeing with whichever of the two builders actually ran).
 *
 * One real difference: lz_t_row_f32 writes each row straight into its
 * `out` argument, so there is no separate scratch step here. `out` below
 * IS the persistent slot - unlike conv_fixed_build, which stages a row
 * through s->wscr before lz_conv_norm_pow2 quantizes it into conv_mw,
 * this is a straight copy with nothing to quantize, so the row can land
 * in its final home directly. */
static int conv_f32_build(LZRunState *s, const LZModel *m) {
    const LZModelConfig *c = &m->config;
    int layer, t, covered_ok = 1, built = 0;
    for (layer = 0; layer < c->n_layers; layer++) {
        const LZLayer *L;
        int li;
        size_t base;
        if (c->layer_types[layer] == LZ_LT_FULL) continue;
        L = &m->layers[layer];
        li = s->cache_idx[layer];
        base = (size_t)li * c->lin_conv_dim;
        built++;
        {
        size_t ch = 0;
        const LZTensor *tv[3];
        int cnt[3], ntv;
        if (L->kda_q_conv1d.q || L->kda_q_conv1d.f) {
            tv[0] = &L->kda_q_conv1d; cnt[0] = c->lin_key_dim;
            tv[1] = &L->kda_k_conv1d; cnt[1] = c->lin_key_dim;
            tv[2] = &L->kda_v_conv1d; cnt[2] = c->lin_value_dim;
            ntv = 3;
        } else {
            tv[0] = &L->conv1d; cnt[0] = c->lin_conv_dim; ntv = 1;
        }
        for (t = 0; t < ntv; t++) {
            int i;
            int row_dim = c->conv_kernel;  /* [C,1,K] -> row = K elements */
            for (i = 0; i < cnt[t]; i++)
                lz_t_row_f32(tv[t], i, row_dim,
                    s->conv_w32 + (base + ch + i) * (size_t)c->conv_kernel);
            ch += (size_t)cnt[t];
        }
        if (ch != (size_t)c->lin_conv_dim) covered_ok = 0;
        }
    }
    if (!covered_ok || built != c->n_linear_layers) return -1;
    return 0;
}

#if LZ_KGATE_I16
/* dt_bias in kda_gate's own integer domain, laid out per linear layer.
 *
 * This is what makes the decay gate's `gt[i] + dt_bias[i]` an INTEGER
 * add once kda_gate arrives as int16: with both terms at 2^-LZ_KGATE_ES
 * the sum is one too, and the only float work left per channel is the
 * one multiply and one add that reach sigmoid_q15_t's coordinate. Built
 * once here rather than per token for the same reason conv_mw is - it
 * is a function of the weights alone.
 *
 * REFUSES rather than clamps. A channel whose |dt_bias| * 2^ES does not
 * fit int16 gets the whole table rejected and every KDA layer keeps its
 * float exit; clamping instead would move the decay gate on a checkpoint
 * nobody measured, silently, which is exactly what LZ_CONV_ES's comment
 * warns about one tier over. MEASURED on kmr20: max |dt_bias| 13.062,
 * i.e. 6,688 of 32,767 at ES = 9, 4.9x of headroom.
 *
 * Returns 1 on success. A GDN linear layer has no kda_dt_bias at all, so
 * a model mixing the two refuses here as well - it has no kda_gate to
 * feed either. */
static int kda_gate_build(LZRunState *s, const LZModel *m) {
    const LZModelConfig *c = &m->config;
    int layer, built = 0;
    int nvk = c->lin_n_v_heads * c->lin_k_head_dim;
    float sc = pow2f(LZ_KGATE_ES);
    if (!s->kda_dtb_i16 || nvk <= 0) return 0;
    for (layer = 0; layer < c->n_layers; layer++) {
        const LZLayer *L = &m->layers[layer];
        const float *db;
        int li, i;
        if (c->layer_types[layer] == LZ_LT_FULL) continue;
        if (L->kda_dt_bias.n == 0) return 0;
        li = s->cache_idx[layer];
        db = lz_t_f32(&L->kda_dt_bias, s->wscr);
        if (!db) return 0;
        for (i = 0; i < nvk; i++) {
            /* Half-away-from-zero, the convention epi_align_i16 rounds
               the other addend with - one domain, one rounding rule. */
            float v = db[i] * sc;
            long q = (long)(v >= 0.0f ? v + 0.5f : v - 0.5f);
            if (q > 32767 || q < -32767) return 0;
            s->kda_dtb_i16[(size_t)li * nvk + i] = (short)q;
        }
        built++;
    }
    return built == c->n_linear_layers;
}
#endif /* LZ_KGATE_I16 */

/* Core activation and residual-stream buffers. */
static void alloc_core_buffers(LZRunState *s, const LZModel *m,
                               const LZModelConfig *c, int nt, int *okp) {
    int ok = 1;
    /* s->x carries GR's nr branches when the switch is on - see forward.h's
       own comment on it for why this is the one buffer whose stride moves.
       Sized for the widest the stream ever gets (nr*hidden, between
       gr_first and gr_last); xb/xb2 stay hidden wide throughout. */
    {
        int xw = c->gated_residual >= 2 ? c->gated_residual : 1;
        s->x = (float *)xcalloc((size_t)nt * xw * c->hidden_size,
                                sizeof(float), &ok, &s->bytes_alloc);
    }
    s->xb      = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
    s->xb2     = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
    /* S1 attn_kv_share: the source layer's attention input for this
       chunk (forward.h's own kvshare_x comment). One buffer for the whole
       stack, not one per target: every target layer of a model reads the
       SAME source. */
    s->kvshare_x = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
    /* Sized for the LARGER of the body's dense-FFN width and the MTP
       block's own (dense_ffn_step's shared scratch, forward.c) - the
       two are independent fields (model.h's mtp_intermediate_size
       comment) and neither is guaranteed to
       bound the other (kunmoe-v2: body intermediate_size 512 is an
       unrelated placeholder, a carved MTP head can be wider). */
    {
        int hb_dim = c->intermediate_size;
        if (m->mtp && c->mtp_intermediate_size > hb_dim) hb_dim = c->mtp_intermediate_size;
        s->hb  = (float *)xcalloc((size_t)nt * hb_dim, sizeof(float), &ok, &s->bytes_alloc);
        s->hb2 = (float *)xcalloc((size_t)nt * hb_dim, sizeof(float), &ok, &s->bytes_alloc);
    }
    s->logits  = (float *)xcalloc((size_t)c->vocab_size, sizeof(float), &ok, &s->bytes_alloc);
    if (!ok) *okp = 0;
}

/* Attention-path activation buffers and the fixed-tier accumulators. */
static void alloc_attn_buffers(LZRunState *s, const LZModelConfig *c,
                               int seq_len, int nt, int *okp) {
    int ok = 1;
    s->qg       = (float *)xcalloc((size_t)nt * c->attn_qgate_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->qh       = (float *)xcalloc((size_t)nt * c->attn_q_dim, sizeof(float), &ok, &s->bytes_alloc);
    /* seq_len + 1, not seq_len: the learned attention sink writes one
       extra logit column at index pos+1 (forward_attn.c), and pos reaches
       seq_len-1 (forward.c rejects pos0 + tk >= s->seq_len), so a
       seq_len-sized buffer would be written one element past its end on
       the last position. The softmax that consumes the column is sized
       off the same +1. wsum_cbuf/wsum_cq below stay seq_len-sized: they
       hold one weight per real position and never see the sink column. */
    s->att      = (float *)xcalloc((size_t)seq_len + 1, sizeof(float), &ok, &s->bytes_alloc);
#if (LZ_ATTN_FIXED & 2)
    /* Allocated whenever the kernel is compiled, not when the tier is
       selected: lz_float_ref_select can run after this point, and 6 bytes per
       context slot (196 KB at ctx 32768, counted in bytes_alloc) is a
       cheaper bargain than a buffer that appears only on some paths. */
    s->wsum_cbuf = (float *)xcalloc((size_t)seq_len, sizeof(float), &ok, &s->bytes_alloc);
    s->wsum_cq   = (int16_t *)xcalloc((size_t)seq_len, sizeof(int16_t), &ok, &s->bytes_alloc);
#endif /* LZ_ATTN_FIXED & 2 */
    s->attn_out = (float *)xcalloc((size_t)nt * c->attn_q_dim, sizeof(float), &ok, &s->bytes_alloc);
    /* Fixed-tier attention int path (--attn-int): attn_acc holds the exact
       int64 weighted sum and post-gate values, attn_ss the per-32-group
       dequant scale. Sized even when the knob is off, same reasoning as
       wsum_cbuf/wsum_cq: the selector can run after this point. */
    s->attn_acc = (int64_t *)xcalloc((size_t)nt * c->attn_q_dim, sizeof(int64_t), &ok, &s->bytes_alloc);
    s->attn_ss  = (float *)xcalloc((size_t)nt * (c->attn_q_dim >> 5), sizeof(float), &ok, &s->bytes_alloc);
    s->ktmp     = (float *)xcalloc((size_t)nt * c->attn_kv_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->vtmp     = (float *)xcalloc((size_t)nt * c->attn_kv_dim, sizeof(float), &ok, &s->bytes_alloc);
    if (!ok) *okp = 0;
}

/* KV-cache ring geometry, then the K/V planes of whichever format each
   side selected. Writes the ring fields (kfmt/vfmt/attn_win/attn_sink/
   kv_ring/kv_slots) as well as the planes. */
static void alloc_kv_cache(LZRunState *s, const LZModel *m,
                           const LZModelConfig *c, int seq_len, int *okp) {
    int ok = 1;
    /* KV cache Q8 (group 32, contiguous within a row). +1 layer slot when
       an MTP head is bound: its own full_attention block reuses this
       cache (index n_full_layers, see forward_attn's sentinel handling)
       rather than a separate buffer - same reasoning as everywhere else
       in this engine that a second copy of a KV cache was avoided. */
    {
        int kv_layers = c->n_full_layers + (m->mtp ? 1 : 0);
        s->kfmt = lz_kv_kfmt;
        s->vfmt = lz_kv_vfmt;
        /* Ring sizing, before anything is allocated. kv_slots is what the
           cache planes are actually indexed by; kv_slot() maps absolute
           positions into it. */
        /* lz_attn_window < 0 means AUTO, which is the default. The value
           is derived here rather than baked into the global because it
           depends on seq_len, which the global cannot see.

           AUTO = max(LZ_ATTN_WIN_FLOOR, seq_len/2), and it is chosen to
           be >= the measured-free point in BOTH readings of the data
           above. Read the numbers again: window 240 costs +5.351% at
           ctx 1024 and +5.428% at ctx 2048 - practically the same. The
           harm tracks the window's ABSOLUTE size, not its ratio to the
           context, which is why the floor is absolute. window 1008 at
           ctx 2048 is -0.142%, i.e. free, so the floor sits just above
           it at 1024. The seq_len/2 term only ever RAISES the window
           beyond the floor, and a wider window is strictly closer to
           full attention - so no ctx can land on a configuration more
           aggressive than one that was measured free.

           Nothing is claimed about ctx far past 2048: the measurements
           stop there. AUTO stays conservative there by construction
           (window grows with ctx), not by evidence. */
        if (lz_attn_window < 0) {
            int w = seq_len >> 1;
            if (w < LZ_ATTN_WIN_FLOOR) w = LZ_ATTN_WIN_FLOOR;
            /* A window that cannot be reached is not a window. Leaving
               it at 0 keeps the whole eviction path - and the ring, and
               the row skip - out of the picture for short contexts,
               which is what makes short-prompt behaviour bit-identical
               to the pre-default engine. */
            s->attn_win = (w + LZ_ATTN_SINK_DEFAULT < seq_len) ? w : 0;
        } else {
            s->attn_win = lz_attn_window > 0 ? lz_attn_window : 0;
        }
        /* A window with NO sink is the catastrophic configuration, not a
           cheaper one: measured +43.764%% PPL against +5.351%% with a
           sink of 16, on the same window budget. Nobody should be able to
           reach it by forgetting a flag, so a window on its own gets a
           sink. Setting --attn-sink 0 explicitly still gets zero, because
           that arm has to stay reachable for the comparison. */
        s->attn_sink = lz_attn_sink > 0 ? lz_attn_sink
                     : (lz_attn_sink < 0 ? 0
                        : (s->attn_win > 0 ? LZ_ATTN_SINK_DEFAULT : 0));
        s->kv_ring   = 0;
        s->kv_slots  = seq_len;
        if (s->attn_win > 0) {
            /* +LZ_SPEC_K_MAX+1 guard slots, see kv_slot()'s comment: a
               rejected speculative draft must only be able to clobber
               positions the read window has already dropped. */
            int ring = s->attn_win + LZ_SPEC_K_MAX + 1;
            if (s->attn_sink + ring < seq_len) {
                s->kv_ring  = ring;
                s->kv_slots = s->attn_sink + ring;
            }
            /* else the "ring" would be bigger than the context: leave it
               off rather than pretend to evict. */
        }
        /* Allocate ONLY the plane each side actually uses. Allocating a
           format that is not selected is not free bookkeeping: keeping
           the Q8 planes alongside the 4-bit ones makes the 4-bit cache
           cost MORE than the 8-bit one (29.1 MB vs 28.8 at ctx 2048) -
           the whole point inverted. */
        {
            size_t nrow = (size_t)kv_layers * s->kv_slots * c->attn_kv_dim;
            size_t nsc  = (size_t)kv_layers * s->kv_slots * c->n_kv_heads;
            if (s->kfmt == LZ_KVF_Q8) {
                s->kq8 = (int8_t *)xcalloc(nrow, 1, &ok, &s->bytes_alloc);
                s->ksq = (float *)xcalloc(nrow / 32, sizeof(float),
                                          &ok, &s->bytes_alloc);
#if LZ_KV_2PLANE
                s->kq8_lo = (int8_t *)xcalloc(nrow, 1, &ok, &s->bytes_alloc);
#endif /* LZ_KV_2PLANE */
            } else if (s->kfmt == LZ_KVF_F32) {
                s->kf32 = (float *)xcalloc(nrow, sizeof(float),
                                           &ok, &s->bytes_alloc);
            } else {
                s->k4 = (unsigned char *)xcalloc(nrow / 2, 1, &ok,
                                                 &s->bytes_alloc);
                s->ks4 = (float *)xcalloc(nsc, sizeof(float),
                                          &ok, &s->bytes_alloc);
            }
            if (s->vfmt == LZ_KVF_Q8) {
                s->vq8 = (int8_t *)xcalloc(nrow, 1, &ok, &s->bytes_alloc);
                s->vsq = (float *)xcalloc(nrow / 32, sizeof(float),
                                          &ok, &s->bytes_alloc);
#if LZ_KV_2PLANE
                s->vq8_lo = (int8_t *)xcalloc(nrow, 1, &ok, &s->bytes_alloc);
#endif /* LZ_KV_2PLANE */
            } else if (s->vfmt == LZ_KVF_F32) {
                s->vf32 = (float *)xcalloc(nrow, sizeof(float),
                                           &ok, &s->bytes_alloc);
            } else {
                s->v4 = (unsigned char *)xcalloc(nrow / 2, 1, &ok,
                                                 &s->bytes_alloc);
                s->vs4 = (float *)xcalloc(nsc, sizeof(float),
                                          &ok, &s->bytes_alloc);
            }
        }
    }
#if LZ_KV_2PLANE
    s->att_lo  = (float *)xcalloc((size_t)seq_len, sizeof(float), &ok, &s->bytes_alloc);
    s->wsum_lo = (float *)xcalloc((size_t)c->head_dim, sizeof(float), &ok, &s->bytes_alloc);
#endif /* LZ_KV_2PLANE */
    if (!ok) *okp = 0;
}

/* Per-token scratch for the linear-attention (KDA/GDN) path. */
static void alloc_linear_scratch(LZRunState *s, const LZModelConfig *c,
                                 int nt, int *okp) {
    int ok = 1;
    s->qkv    = (float *)xcalloc((size_t)nt * c->lin_conv_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->qkv_c  = (float *)xcalloc((size_t)nt * c->lin_conv_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->zbuf   = (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->avec   = (float *)xcalloc((size_t)nt * c->lin_n_v_heads, sizeof(float), &ok, &s->bytes_alloc);
    s->bvec   = (float *)xcalloc((size_t)nt * c->lin_n_v_heads, sizeof(float), &ok, &s->bytes_alloc);
    s->ssm_out= (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->ssm_sig = (int32_t *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(int32_t), &ok, &s->bytes_alloc);
    /* nt-wide, not one head-dim wide - allocated to match the other
       batched buffers; the serial recurrence only ever touches row 0.
       `nt` here is LZ_BATCH_MAX (line 407), the cap, not the current
       width - same rule the other batched buffers follow. */
    s->qn     = (float *)xcalloc((size_t)nt * c->lin_k_head_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kn     = (float *)xcalloc((size_t)nt * c->lin_k_head_dim, sizeof(float), &ok, &s->bytes_alloc);

    /* LZ_LT_KDA scratch. Zero-sized (NULL) when the model has no KDA
       layers (kda_gate_rank == 0, lin_key_dim/lin_n_v_heads still fine
       to multiply by since they'd only be 0 too on a config with no
       linear-attention layers at all). */
    s->kda_q  = (float *)xcalloc((size_t)nt * c->lin_key_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_k  = (float *)xcalloc((size_t)nt * c->lin_key_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_v  = (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_qc = (float *)xcalloc((size_t)nt * c->lin_key_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_kc = (float *)xcalloc((size_t)nt * c->lin_key_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_vc = (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_gate_lat = (float *)xcalloc((size_t)nt * c->kda_gate_rank, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_gate     = (float *)xcalloc((size_t)nt * c->lin_n_v_heads * c->lin_k_head_dim,
                                       sizeof(float), &ok, &s->bytes_alloc);
    if (!ok) *okp = 0;
}

/* Latent-MoE scratch. Zero-sized when the model has no MoE layers. */
static void alloc_moe_scratch(LZRunState *s, const LZModelConfig *c,
                              int nt, int *okp) {
    int ok = 1;
    /* latent MoE scratch - see forward.h. Zero-sized when the model has
       no MoE layers (num_experts == 0).

       TWO WIDTHS, split on the routing boundary. The router, the latent
       down/up projections and the shared expert read one weight stream
       for every token in the chunk, so they run batched and their
       buffers are nt-wide like every other block here. The ROUTED
       experts cannot: which expert a token wants is a property of that
       token, so moe_h2 and the per-expert selection stay one token's
       worth. See forward_moe. */
    {
        int moe_scratch_w = c->moe_intermediate_size;
        if (c->moe_shared_width > moe_scratch_w) moe_scratch_w = c->moe_shared_width;
        s->moe_router_logits = (float *)xcalloc((size_t)nt * c->num_experts, sizeof(float), &ok, &s->bytes_alloc);
        /* Sized to the EXPERT COUNT, not to num_experts_per_token, so
           --moe-topk can go up as well as down. At 16 experts that is
           128 bytes total - cheaper than the clamp it removes, and a
           clamp that silently only works downwards is exactly the kind
           of knob that produces a one-sided sweep. */
        s->moe_sel_idx = (int *)xcalloc((size_t)c->num_experts, sizeof(int), &ok, &s->bytes_alloc);
        s->moe_sel_w   = (float *)xcalloc((size_t)c->num_experts, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_lat_x = (float *)xcalloc((size_t)nt * c->moe_latent_dim, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_lat_y = (float *)xcalloc((size_t)nt * c->moe_latent_dim, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_h1 = (float *)xcalloc((size_t)nt * moe_scratch_w, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_h3 = (float *)xcalloc((size_t)nt * moe_scratch_w, sizeof(float), &ok, &s->bytes_alloc);
        /* One token's worth: this is the routed loop's own temporary,
           written and consumed inside one (token, expert) step. */
        s->moe_h2 = (float *)xcalloc((size_t)c->moe_latent_dim, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_shared_out = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
    }
    if (!ok) *okp = 0;
}

/* MTP draft-head scratch. Allocates nothing at all without a bound head. */
static void alloc_mtp_scratch(LZRunState *s, const LZModel *m,
                              const LZModelConfig *c, int nt, int *okp) {
    int ok = 1;
    /* MTP draft head scratch - only when a head is bound (see forward.h).
       mtp_x/mtp_concat/mtp_emb_raw are nt_cap-wide: the chained
       per-step draft path (lz_mtp_draft_step) only ever touches
       slot 0, but lz_mtp_prefill batches the SAME buffers across nt
       prompt positions at once (forward.h's own comment on why this is
       one set of scratch, not two). mtp_chain/mtp_draft_logits stay
       single-token - the draft chain register and one step's logits are
       inherently one-token concepts, prefill never touches either. */
    if (m->mtp) {
        s->mtp_x     = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_concat = (float *)xcalloc((size_t)nt * 2 * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_emb_raw = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_chain = (float *)xcalloc((size_t)c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_draft_logits = (float *)xcalloc((size_t)c->vocab_size, sizeof(float), &ok, &s->bytes_alloc);
        if (c->mtp_n_draft > 1) {
            s->mtp_kdraft_concat = (float *)xcalloc((size_t)3 * c->hidden_size,
                                                sizeof(float), &ok, &s->bytes_alloc);
            s->mtp_kdraft_logits = (float *)xcalloc((size_t)LZ_SPEC_K_MAX * c->vocab_size,
                                                sizeof(float), &ok, &s->bytes_alloc);
            s->mtp_kdraft_correction = (float *)xcalloc((size_t)c->vocab_size,
                                                    sizeof(float), &ok, &s->bytes_alloc);
        }
        s->mtp_logits = (float *)xcalloc((size_t)(LZ_SPEC_K_MAX + 1) * c->vocab_size,
                                         sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_verify_hidden = (float *)xcalloc((size_t)(LZ_SPEC_K_MAX + 1) * c->hidden_size,
                                                sizeof(float), &ok, &s->bytes_alloc);
        /* temp>0 speculative decoding phase 2: one row per DRAFTED
           token (LZ_SPEC_K_MAX, not +1 - the verify batch's own
           bonus/final row has no draft counterpart to accept/reject
           against, see forward.h's own comment). */
        s->mtp_draft_q = (float *)xcalloc((size_t)LZ_SPEC_K_MAX * c->vocab_size,
                                          sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_target_p = (float *)xcalloc((size_t)c->vocab_size,
                                           sizeof(float), &ok, &s->bytes_alloc);
    }
    if (!ok) *okp = 0;
}

/* SSM/conv recurrent state, and the ring depth its first dimension
   takes. */
static void alloc_recurrent_state(LZRunState *s, const LZModel *m,
                                  const LZModelConfig *c, int spec_k_max,
                                  int *okp) {
    int ok = 1;
    /* SSM/conv state: one (kd*vd) block per (layer, head), grouped
       within rows, times ssm_ring_depth slots - see forward.h's own
       comment on s->ssm_slot/s->ssm_ring_depth for the ring's shape
       and the "zero extra bandwidth, only memory" argument it depends
       on. Depth is 1 (no ring, byte-identical to the single-slot
       version) when there is no MTP head to draft with at all - a
       model that can never run --spec never allocates slots it cannot
       use - OR when the CALLER declares it will never request --spec
       (spec_k_max <= 0, this function's own parameter comment,
       forward.h). Otherwise spec_k_max+1, clamped to LZ_SPEC_K_MAX+1 -
       a caller that only ever wants k<=2 should not pay for slots
       3..6. Stored on s rather than recomputed at every site that
       needs it - see s->ssm_ring_depth's own comment for why. */
    if (!m->mtp || spec_k_max <= 0) {
        s->ssm_ring_depth = 1;
    } else {
        int d = spec_k_max + 1;
        if (d > LZ_SPEC_K_MAX + 1) d = LZ_SPEC_K_MAX + 1;
        s->ssm_ring_depth = d;
    }
    s->ssm_state_q8 = (int8_t *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_n_v_heads *
        c->lin_k_head_dim * c->lin_v_head_dim, 1, &ok, &s->bytes_alloc);
#if LZ_GDN_STATE_2PLANE
    s->ssm_state_q8_lo = (int8_t *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_n_v_heads *
        c->lin_k_head_dim * c->lin_v_head_dim, 1, &ok, &s->bytes_alloc);
#endif /* LZ_GDN_STATE_2PLANE */
    s->ssm_state_s = (float *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_n_v_heads *
        c->lin_k_head_dim * c->lin_v_head_dim / 32,
        sizeof(float), &ok, &s->bytes_alloc);
    s->conv_state = (float *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_conv_dim *
        (c->conv_kernel - 1), sizeof(float), &ok, &s->bytes_alloc);
    if (!ok) *okp = 0;
}

/* CSA2 compressed-block ring. Allocates nothing when no layer is
   csa2_on. */
static void alloc_csa2_ring(LZRunState *s, const LZModel *m,
                            const LZModelConfig *c, int seq_len, int *okp) {
    int ok = 1;
    /* S2 CSA2 (forward.h's own csa2_pending_c/csa2_comp_ring comment).
       Allocated only when at least one layer is csa2_on - unlike
       conv_state/ssm_state_q8 just above, which every KDA/GDN model
       needs, CSA2 is rare enough (S4's own mla_q_lora_rank precedent)
       that a model without it should not pay for a ring nobody reads.
       Sized against s->seq_len, not s->kv_slots - see forward.h's own
       comment on why.

       This scan reads m->layers[li2].csa2_on for EVERY layer, so the
       caller must pass a loaded model: m->layers has to be valid for
       every layer by the time this runs, whatever run-time switches are
       set. It is not conditional on any flag. */
    {
        int any_csa2 = 0, li2;
        for (li2 = 0; li2 < c->n_layers; li2++)
            if (m->layers[li2].csa2_on) { any_csa2 = 1; break; }
        if (any_csa2) {
            int cm = c->csa2_compress_m > 0 ? c->csa2_compress_m : 1;
            int max_blocks = (seq_len + cm - 1) / cm;
            size_t nf = (size_t)c->n_full_layers;
            s->csa2_pending_c = (float *)xcalloc(
                nf * cm * c->head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_pending_z = (float *)xcalloc(
                nf * cm * c->head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_pending_c_idx = (float *)xcalloc(
                nf * cm * c->csa2_indexer_head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_pending_z_idx = (float *)xcalloc(
                nf * cm * c->csa2_indexer_head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_comp_ring = (float *)xcalloc(
                nf * (size_t)max_blocks * c->head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_idx_comp_ring = (float *)xcalloc(
                nf * (size_t)max_blocks * c->csa2_indexer_head_dim, sizeof(float),
                &ok, &s->bytes_alloc);
            s->csa2_n_blocks = (int *)xcalloc(nf, sizeof(int), &ok, &s->bytes_alloc);
            s->csa2_scores = (float *)xcalloc((size_t)max_blocks, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_topk_idx = (int *)xcalloc((size_t)c->csa2_topk, sizeof(int), &ok, &s->bytes_alloc);
            s->csa2_topk_val = (float *)xcalloc((size_t)c->csa2_topk, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_c_j = (float *)xcalloc((size_t)c->head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_z_j = (float *)xcalloc((size_t)c->head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_c_j_idx = (float *)xcalloc((size_t)c->csa2_indexer_head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_z_j_idx = (float *)xcalloc((size_t)c->csa2_indexer_head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_qidx = (float *)xcalloc(
                (size_t)c->csa2_indexer_heads * c->csa2_indexer_head_dim, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_widx = (float *)xcalloc((size_t)c->csa2_indexer_heads, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_cq_idx = (float *)xcalloc((size_t)c->csa2_indexer_q_lora_rank, sizeof(float), &ok, &s->bytes_alloc);
            s->csa2_attn_probs = (float *)xcalloc((size_t)c->csa2_topk, sizeof(float), &ok, &s->bytes_alloc);
        }
    }
    if (!ok) *okp = 0;
}

/* The conv tap caches: the persistent FLOAT-tier one, and - only when
   the fixed tier is selected - that tier's own buffers. */
static void alloc_conv_tier(LZRunState *s, const LZModelConfig *c,
                            int nt, int *okp) {
    int ok = 1;
    /* Persistent F32 conv-tap cache for the FLOAT tier - see forward.h's
       conv_w32 field comment. Allocated unconditionally, NOT gated on
       s->conv_fixed (decided just below in this helper): that decision
       can still be reversed later, once conv_fixed_build's own coverage
       check runs, and the float path needs somewhere safe to widen into
       either way. Never ring-sized (unlike conv_state above): this holds
       weights, not rolling per-token history, exactly like conv_mw
       below. */
    s->conv_w32 = (float *)xcalloc(
        (size_t)c->n_linear_layers * c->lin_conv_dim * c->conv_kernel,
        sizeof(float), &ok, &s->bytes_alloc);

    /* Fixed conv tier. Decided ONCE, here, and recorded in s->conv_fixed
       rather than re-read per token: lz_conv_mode() can be moved by a
       flag, and a state whose buffers were not allocated must not start
       taking the fixed branch halfway through a run.
     *
     * The history exponent was the one parameter that kept this tier
     * unwired for a while, and the answer it eventually got is
     * LZ_CONV_ES: a CONSTANT power of two, measured once from the real
     * conv input distribution rather than derived per token, because the
     * history starts at zero and has nothing to derive one from. The
     * recurrence's answer (a scale that TRAVELS WITH THE STATE,
     * ssm_state_s / p2_group_scale) would cost a renormalization pass
     * over the history every time the range moved, which is the cost
     * this tier exists to avoid. See LZ_CONV_ES for the distribution it
     * was measured against and for the fact that a value past 32 clips
     * SILENTLY - re-run that probe on any model whose conv inputs might
     * be scaled differently. */
    s->conv_fixed = lz_conv_mode();
    if (s->conv_fixed) {
        size_t nch = (size_t)c->n_linear_layers * c->lin_conv_dim;
        s->conv_bound = lz_conv_accum_bound(c->conv_kernel);
        s->conv_state_q = (short *)xcalloc(
            (size_t)s->ssm_ring_depth * nch * (c->conv_kernel - 1),
            sizeof(short), &ok, &s->bytes_alloc);
        s->conv_mw = (short *)xcalloc(nch * c->conv_kernel,
                                      sizeof(short), &ok, &s->bytes_alloc);
        /* Sized like kda_q/kda_k/kda_v above, in int16. Zero-sized (and
           so NULL) on a model with no KDA layers, which forward_kda's
           own predicate then reads as "no int path". */
        s->kda_q_i16 = (short *)xcalloc((size_t)nt * c->lin_key_dim,
                                        sizeof(short), &ok, &s->bytes_alloc);
        s->kda_k_i16 = (short *)xcalloc((size_t)nt * c->lin_key_dim,
                                        sizeof(short), &ok, &s->bytes_alloc);
        s->kda_v_i16 = (short *)xcalloc((size_t)nt * c->lin_value_dim,
                                        sizeof(short), &ok, &s->bytes_alloc);
#if LZ_CONVO_I16
        /* The conv's own int16 exit - see forward.h. */
        s->kda_qc_i16 = (short *)xcalloc((size_t)nt * c->lin_key_dim,
                                         sizeof(short), &ok, &s->bytes_alloc);
        s->kda_kc_i16 = (short *)xcalloc((size_t)nt * c->lin_key_dim,
                                         sizeof(short), &ok, &s->bytes_alloc);
        s->kda_vc_i16 = (short *)xcalloc((size_t)nt * c->lin_value_dim,
                                         sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_CONVO_I16 */
        s->conv_sig_k1 = (float *)xcalloc(nch, sizeof(float), &ok,
                                          &s->bytes_alloc);
        s->conv_sig_oscale2 = (float *)xcalloc(nch, sizeof(float), &ok,
                                               &s->bytes_alloc);
#if LZ_CONV_SIG_I
        s->conv_sig_e = (signed char *)xcalloc(nch, sizeof(signed char), &ok,
                                               &s->bytes_alloc);
#endif /* LZ_CONV_SIG_I */
        s->conv_sig_k2 = lz_sig_q15_t_offset();
        if (!ok || !*okp) s->conv_fixed = 0;
        /* conv_fixed_build runs LATER, after cache_idx is filled - it
           needs the layer -> linear-index map and that is populated
           further down. Building here would read an all-zero map and
           quantize every layer's taps into slot 0. */
    }
    if (!ok) *okp = 0;
}

/* Q8 activation-quantization scratch: the capacity every matmul input
   must fit, and the buffers quantized activations land in. */
static void alloc_quant_scratch(LZRunState *s, const LZModel *m,
                                const LZModelConfig *c, int nt, int *okp) {
    int ok = 1;
    /* Q8 activation quantization buffers: capacity = max matmul input dim */
    {
        int qcap = c->hidden_size;
        int nvk = c->lin_n_v_heads * c->lin_k_head_dim;
        if (c->intermediate_size > qcap) qcap = c->intermediate_size;
        if (c->attn_q_dim > qcap) qcap = c->attn_q_dim;
        if (c->lin_conv_dim > qcap) qcap = c->lin_conv_dim;
        if (c->attn_qgate_dim > qcap) qcap = c->attn_qgate_dim;
        /* KDA's f_b_proj (kda_gate_rank -> nvk) and latent MoE's
           per-expert matmuls (moe_latent_dim <-> moe_intermediate_size)
           and shared expert (hidden <-> moe_shared_width) all quantize
           through lz_matmul_w with s->xq/s->xqs as scratch - every one
           of their in_dims must fit, not just the dense-path ones above. */
        if (c->kda_gate_rank > qcap) qcap = c->kda_gate_rank;
        if (nvk > qcap) qcap = nvk;
        if (c->moe_latent_dim > qcap) qcap = c->moe_latent_dim;
        if (c->moe_intermediate_size > qcap) qcap = c->moe_intermediate_size;
        if (c->moe_shared_width > qcap) qcap = c->moe_shared_width;
        /* MTP's fc reads 2*hidden_size (lz_mtp_draft_step quantizes
           s->mtp_concat through lz_matmul_w with this same scratch) -
           not covered by any of the dims above on a model whose
           attention/FFN widths are all below 2x hidden. Its dense FFN
           (dense_ffn_step's gate/up_proj quantization) reads
           mtp_intermediate_size, independent of intermediate_size -
           same reasoning as s->hb/s->hb2's sizing
           above, not assumed to be bounded by anything else here. */
        if (m->mtp && 2 * c->hidden_size > qcap) qcap = 2 * c->hidden_size;
        if (m->mtp && c->mtp_intermediate_size > qcap) qcap = c->mtp_intermediate_size;
        /* DSpark (mtp_n_draft > 1): main_proj reads 3*hidden_size
           (lz_mtp_draft_batch quantizes s->mtp_kdraft_concat through the
           same scratch); markov_proj reads mtp_markov_rank, always far
           smaller than hidden_size in practice but checked anyway
           rather than assumed. */
        if (m->mtp && c->mtp_n_draft > 1 && 3 * c->hidden_size > qcap)
            qcap = 3 * c->hidden_size;
        if (m->mtp && c->mtp_n_draft > 1 && c->mtp_markov_rank > qcap)
            qcap = c->mtp_markov_rank;
        /* GR's W_d and W_w read the whole flattened branch stream
           (nr*hidden), which is nr times wider than any single width
           above - on a GR model that is the widest thing this scratch
           ever quantizes, by construction. lz_gr_read passes s->xq/s->xqs
           to lz_matmul_w exactly like every other site here, and xq is
           indexed by in_dim, so the value has to fit an int. A gr_flat
           past INT_MAX is already an s->x this process allocated
           (alloc_core_buffers ran first and asked for nt*gr_flat*4 bytes),
           so the honest answer is to fail the allocation rather than size
           this buffer short and overrun it. No shipped checkpoint is
           within a factor of 10^5 of this. */
        if (c->gated_residual >= 2) {
            lz_i64 gr_flat = (lz_i64)c->gated_residual * c->hidden_size;
            if (gr_flat > 0x7FFFFFFF) { *okp = 0; return; }
            if ((int)gr_flat > qcap) qcap = (int)gr_flat;
        }
        s->qcap = qcap;
        s->xq = (int8_t *)xcalloc((size_t)nt * qcap, 1, &ok, &s->bytes_alloc);
        /* xqs sized for the worst case (in-row gs falling back to 1) */
        s->xqs = (float *)xcalloc((size_t)nt * qcap, sizeof(float), &ok, &s->bytes_alloc);
        /* AT LEAST LZ_MM_WIDEN_MAX, so the loader's bf16 rule is sound
           by construction rather than by inspection. model.c accepts a
           1-D tensor as bf16 when it is no longer than that constant,
           and lz_t_f32 widens such a tensor WHOLE into this buffer -
           two bounds that have to agree, and did only because every
           1-D tensor in the checkpoints to hand happens to be a norm
           or a per-head bias and so is already counted in qcap above.
           A model with a wider one would have overrun this. The cost
           is at most 16 KB. */
        {   int wcap = qcap;
            if (wcap < LZ_MM_WIDEN_MAX) wcap = LZ_MM_WIDEN_MAX;
            s->wscr = (float *)xcalloc((size_t)wcap, sizeof(float), &ok,
                                       &s->bytes_alloc); }
    }
    if (!ok) *okp = 0;
}

/* Engram scratch and its cross-chunk conv history. Allocates nothing at
   all when the model has no engram. */
static void alloc_engram_scratch(LZRunState *s, const LZModel *m,
                                 const LZModelConfig *c, int nt, int *okp) {
    int ok = 1;
    /* Engram's own scratch (see forward.h's field comment for why
       it cannot borrow s->wscr). Sized for the widest layout
       forward_engram.c uses: hidx (nt*ns ints) + value/qnorm/key
       (3*nt*dim floats) + a shared tail sized for the wider of
       eo+val (nt*po + nt*dim) and conv_out (nt*dim) - the tail is
       reused sequentially, never concurrently with the block ahead
       of it, so only the larger of the two needs counting.
       int and float are both 4 bytes here, so the two element
       counts add directly; +64 covers BOTH the alignment pad
       forward_engram.c inserts between hidx and the float region
       (up to 15 bytes) and xcalloc's own base-pointer/16-byte-align
       header (up to sizeof(void*)+15 more) - a tight +16 margin
       left exactly 1 byte of slack computed by hand, which is the
       kind of number that survives one config change and not the
       next. Rounding up to a clean 64 costs nothing measurable at
       this size. */
    if (m->engram) {
        int ns = c->engram_n_spaces, dim = c->hidden_size;
        int po = c->engram_per_order;
        size_t tail = (size_t)nt * po + (size_t)nt * dim;
        size_t n_elem = (size_t)nt * ns + (size_t)nt * dim * 3 + tail;
        s->engram_scratch_bytes = n_elem * sizeof(float) + 64;
        s->engram_scratch = xcalloc(s->engram_scratch_bytes, 1,
                                    &ok, &s->bytes_alloc);
        /* Cross-chunk VALUE history for the depthwise conv - see
           forward.h's engram_conv_hist comment. Rows needed:
           (kernel-1)*dilation, each dim wide. */
        {
            int rows = (c->engram_conv_kernel - 1) *
                      c->engram_conv_dilation;
            if (rows < 0) rows = 0;
            s->engram_conv_hist_cap = rows;
            s->engram_conv_hist = rows > 0
                ? (float *)xcalloc((size_t)rows * dim, sizeof(float),
                                   &ok, &s->bytes_alloc)
                : NULL;
        }
    }
    if (!ok) *okp = 0;
}

/* GR's own scratch: one block carved into the four views forward.h
   declares. Dedicated rather than pooled into xq/wscr/qcap for the reason
   engram_scratch's own comment gives - the widest block here is
   nt*nr*hidden, nr times the activation width any other site quantizes,
   and s->wscr is a single weight ROW.

   The two rhat/g blocks are sized nt*flat each rather than nt*flat and
   nt*r: the read gate's input IS the flattened branch stream, so its
   matmul input and the gate it produces are the same width. Everything
   below is counted off flat, so a change to nr or hidden moves all four
   views together - the failure mode a hand-summed capacity has is the
   silent overrun forward.h's engram note records. */
static void alloc_gr_scratch(LZRunState *s, const LZModelConfig *c,
                             int nt, int *okp) {
    int ok = 1;
    if (c->gated_residual >= 2) {
        int nr = c->gated_residual;
        int flat = nr * c->hidden_size;
        int r = c->hidden_size / 8 > 0 ? c->hidden_size / 8 : 1;
        size_t n_elem = (size_t)nt * 2 * flat + (size_t)nt * r + (size_t)nt * nr;
        s->gr_scratch_bytes = n_elem * sizeof(float);
        s->gr_scratch = xcalloc(s->gr_scratch_bytes, 1, &ok, &s->bytes_alloc);
        if (ok) {
            float *p = (float *)s->gr_scratch;
            s->gr_rhat = p;                      p += (size_t)nt * flat;
            s->gr_g    = p;                      p += (size_t)nt * flat;
            s->gr_h    = p;                      p += (size_t)nt * r;
            s->gr_s    = p;
        }
    }
    if (!ok) *okp = 0;
}

/* SubLN scratch: the Hadamard pair and the plain-norm int row, both
   sized by the Q8 capacity the caller passes. */static void alloc_subn_scratch(LZRunState *s, int qcap, int *okp) {
    int ok = 1;
    /* SubLN Hadamard scratch: two n-wide int32 halves, n <= qcap. */
    s->fwht_scratch = (int32_t *)xcalloc((size_t)2 * qcap, sizeof(int32_t),
                                         &ok, &s->bytes_alloc);
    /* SubLN plain-norm int output scratch: one n-wide int32 row,
       n <= qcap - see forward.h's field comment. */
    s->subn_norm_int = (int32_t *)xcalloc((size_t)qcap, sizeof(int32_t),
                                          &ok, &s->bytes_alloc);
    if (!ok) *okp = 0;
}

/* The q16_0 int16 exits' buffers, one per chain, each behind its own
   compile-time switch. */
static void alloc_i16_exits(LZRunState *s, const LZModelConfig *c,
                            int nt, int *okp) {
    int ok = 1;
    int nvk = c->lin_n_v_heads * c->lin_k_head_dim;
    /* ---- q16_0's int16 exits (int-pipeline milestone 5) ----------
       One buffer per chain, each the int16 twin of the float one it
       replaces for a token - exactly one of the two is written per
       call, the shape milestone 3's kda_q_i16 established. All are
       zero-sized (so NULL) on a model without the producing tensor,
       and every predicate below reads NULL as "no int path". */
#if LZ_BVEC_I16
    s->bvec_i16 = (short *)xcalloc((size_t)nt * c->lin_n_v_heads,
                                   sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_BVEC_I16 */
#if LZ_MOEGATE_I16
    s->moe_logits_i16 = (short *)xcalloc((size_t)nt * c->num_experts,
                                         sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_MOEGATE_I16 */
#if LZ_KLAT_I16
    s->kda_lat_i16 = (short *)xcalloc((size_t)nt * c->kda_gate_rank,
                                      sizeof(short), &ok, &s->bytes_alloc);
    /* The widening scratch lz_quantize_q8_int's int32 input needs -
       one token's worth, since f_b_proj is a per-token matmul. An
       int-to-int copy, so it bills nothing and adds no rounding. */
    s->kda_lat_i32 = (int32_t *)xcalloc((size_t)c->kda_gate_rank,
                                        sizeof(int32_t), &ok, &s->bytes_alloc);
#endif /* LZ_KLAT_I16 */
#if LZ_MLAT_I16
    s->moe_lat_i16 = (short *)xcalloc((size_t)nt * c->moe_latent_dim,
                                      sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_MLAT_I16 */
#if LZ_VPROJ_I16
    /* Sized like s->vtmp, in int16. Zero-sized (and so NULL) on a
       model with no full-attention layers, which forward_attn's
       predicate reads as "no int path". */
    s->vtmp_i16 = (short *)xcalloc((size_t)nt * c->attn_kv_dim,
                                   sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_VPROJ_I16 */
#if LZ_SWIGLU_I16
    /* Sized like moe_h1/moe_h3 - the wider of the routed experts'
       intermediate and the shared expert's width, since both sites
       write these. Zero-sized (and so NULL) on a dense model, which
       forward_moe's predicate reads as "no int path". */
    {
        int sw = c->moe_intermediate_size;
        if (c->moe_shared_width > sw) sw = c->moe_shared_width;
        s->moe_h1_i16 = (short *)xcalloc((size_t)nt * sw, sizeof(short),
                                         &ok, &s->bytes_alloc);
        s->moe_h3_i16 = (short *)xcalloc((size_t)nt * sw, sizeof(short),
                                         &ok, &s->bytes_alloc);
    }
#endif /* LZ_SWIGLU_I16 */
    /* Under neither LZ_MLAT_I16 nor LZ_MLAT_REUSE: the expert loop's
       hoisted activation quantize writes here on the float path, the
       int path, and the reuse control arm alike - what the switches
       select is whether the quantize is skipped, not where it lands.
       One token wide (the loop is per token) and the scale array is
       sized for the worst case, an in-row gs of 1, the same way
       s->xqs is. */
    s->moe_lat_q = (int8_t *)xcalloc((size_t)c->moe_latent_dim, 1,
                                     &ok, &s->bytes_alloc);
    s->moe_lat_qs = (float *)xcalloc((size_t)c->moe_latent_dim,
                                     sizeof(float), &ok, &s->bytes_alloc);
#if LZ_KGATE_I16
    s->kda_gate_i16 = (short *)xcalloc((size_t)nt * nvk, sizeof(short),
                                       &ok, &s->bytes_alloc);
    s->kda_dtb_i16 = (short *)xcalloc(
        (size_t)c->n_linear_layers * nvk, sizeof(short), &ok,
        &s->bytes_alloc);
#endif /* LZ_KGATE_I16 */
    if (!ok) *okp = 0;
}

int lz_state_alloc(LZRunState *s, const LZModel *m, int seq_len,
                       int spec_k_max, char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    int ok = 1, l, nf = 0, nl = 0;
    /* Activation buffers are allocated for LZ_BATCH_MAX slices. The
       generation phase only uses slice 0; the rest is wasted - but for
       s1v3 that is ~0.25 MB, well under 1% of the weights under any
       recipe still on the table (see LZ_BATCH_MAX in ops.h - the byte
       figure is not settled), and the alternative is a batch-width
       parameter that would infect every state_alloc caller.

       nt_cap starts at LZ_BATCH_DEFAULT, which is <= LZ_BATCH_MAX: the
       buffers are sized for the CAP so a caller may raise nt_cap up to
       it at any time (CLI --batch), but the width that actually runs is
       the DEFAULT. The two are separate because the optimal width is a
       property of the target's L1, and the only machine we can measure
       is not a target - see LZ_BATCH_MAX's comment for the sweep and
       for the mechanism by which the answer can reverse on a PII. */
    int nt = LZ_BATCH_MAX;

    memset(s, 0, sizeof(*s));
    /* --ced, REFUSED rather than ignored when it cannot apply. The guard
       that skips the layer sits inside the full-attention branch in
       forward_chunk, so on a stack whose last layer is linear attention
       it could never fire, the run would come out unchanged, and the
       flag would read as a speedup that nothing performed. m->layers is
       valid here - lz_state_alloc takes a loaded model - and the
       alloc_csa2_ring call below needs the same of every layer. */
    if (lz_ced) {
        if (m->layers[c->n_layers - 1].type != LZ_LT_FULL) {
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CED_LAST_LAYER);
            return 1;
        }
        s->ced = 1;
    }
    if (seq_len <= 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_SEQ_LEN);
        return 1;
    }
    s->seq_len = seq_len;
    s->nt_cap = LZ_BATCH_DEFAULT;

    alloc_core_buffers(s, m, c, nt, &ok);

    alloc_attn_buffers(s, c, seq_len, nt, &ok);

    alloc_kv_cache(s, m, c, seq_len, &ok);

    alloc_linear_scratch(s, c, nt, &ok);

    alloc_moe_scratch(s, c, nt, &ok);

    alloc_mtp_scratch(s, m, c, nt, &ok);

    alloc_recurrent_state(s, m, c, spec_k_max, &ok);

    alloc_csa2_ring(s, m, c, seq_len, &ok);

    alloc_conv_tier(s, c, nt, &ok);
    /* s->ssm_slot starts at 0 from this function's own memset(s,0,...)
       above - explicit here only as documentation, matching mtp_pos's
       own "reset to 0" comment; lz_state_reset (below) is the one that
       actually has to set it back to 0 on every subsequent reuse. */

    s->cache_idx = (int *)xcalloc((size_t)c->n_layers, sizeof(int), &ok, &s->bytes_alloc);

    alloc_quant_scratch(s, m, c, nt, &ok);
    alloc_engram_scratch(s, m, c, nt, &ok);
    alloc_gr_scratch(s, c, nt, &ok);
    alloc_subn_scratch(s, s->qcap, &ok);
    alloc_i16_exits(s, c, nt, &ok);
    if (!ok) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_STATE_ALLOC);
        lz_state_free(s);
        return 1;
    }
    /* RoPE precomputed table: pos * (rotary_dim/2) {cos, sin} pairs.
       Shared by all layers/heads under the same theta; the half-precision
       frequency formula matches the reference (denominator is
       rotary_dim, not head_dim). Fill itself is lz_rope_build_table's
       own doc comment. */
    /* RoPE/scale tables must be computed at PC=24 on the x87 targets too
       (Watcom): lz_mathf's float transcendentals are bit-identical across
       gcc and ARM only when every float operation rounds to 24 bits, and
       on a 387 that means the precision control word, not the default
       64-bit mantissa. gcc's SSE and the ARM soft-float ignore the
       control word (always 24-bit float), so begin/end is a no-op there. */
    {
        unsigned fpu = lz_fpu_float_begin();
        s->rope_cs = (float *)xcalloc((size_t)seq_len * c->rotary_dim,
                                      sizeof(float), &ok, &s->bytes_alloc);
        if (ok) lz_rope_build_table(s->rope_cs, seq_len, c);
        if (!ok) {
            lz_fpu_float_end(fpu);
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_STATE_ALLOC);
            lz_state_free(s);
            return 1;
        }

        s->attn_scale = lz_rsqrt((float)c->head_dim);
        s->ssm_scale  = lz_rsqrt((float)c->lin_k_head_dim);
        lz_fpu_float_end(fpu);
    }
    s->epoch      = 1;          /* 0 is reserved for "never allocated" */

    /* Hadamard widths (see forward.h's kv_rot_k comment). K takes the
       largest power of two that divides head_dim, V takes 64 - the sizes
       llama.cpp PR 21038 settled on. A head_dim that is not a multiple of
       64 gets NO rotation rather than a partial one. */
    s->kv_rot_k = s->kv_rot_v = 0;
    if (lz_kv_rot_enable && c->head_dim > 0 && (c->head_dim % 64) == 0) {
        int n = 64;
        while ((c->head_dim % (n * 2)) == 0) n *= 2;
        s->kv_rot_k = n;
        s->kv_rot_v = lz_kv_rot_v_dim > 0 ? lz_kv_rot_v_dim : 64;
        if (s->kv_rot_v > n) s->kv_rot_v = n;
        if ((c->head_dim % s->kv_rot_v) != 0) s->kv_rot_v = 64;
    }

    /* full and linear layers are numbered independently */
    for (l = 0; l < c->n_layers; l++)
        s->cache_idx[l] = (c->layer_types[l] == LZ_LT_FULL) ? nf++ : nl++;

#if LZ_CONV_FIXED
    /* After cache_idx, which conv_fixed_build reads. */
    if (s->conv_fixed && conv_fixed_build(s, m) != 0) s->conv_fixed = 0;
#endif /* LZ_CONV_FIXED */
    /* s->conv_fixed is final now - conv_fixed_build's own refusal, just
       above, is the last thing that can still flip it. Build the float
       tier's persistent conv cache exactly when the float tier is the
       one that will actually run; a build with LZ_CONV_FIXED compiled to
       0 always takes this branch, since s->conv_fixed is then always 0.
       Unlike conv_fixed_build's own refusal, there is no cheaper tier
       left to fall back to if THIS fails - the float path is the last
       resort - so a failure here is fatal, the same way the buffer
       allocations above it are. LZ_ERR_CONV_LAYOUT, not
       LZ_ERR_STATE_ALLOC: conv_f32_build's only failure is its coverage
       check (a layer's channel counts do not sum to lin_conv_dim), a
       checkpoint layout problem, not a failed xcalloc - see that code's
       own comment in err.h. */
    if (!s->conv_fixed && conv_f32_build(s, m) != 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CONV_LAYOUT);
        lz_state_free(s);
        return 1;
    }
#if LZ_KGATE_I16
    /* Same ordering reason as conv_fixed_build's: the table is indexed
       by linear-layer number. A refusal drops the buffer, and
       forward_kda's predicate reads that NULL as "no int path" - one
       place decides, one place asks. */
    if (!kda_gate_build(s, m)) {
        xfree(s->kda_dtb_i16);  s->kda_dtb_i16 = NULL;
        xfree(s->kda_gate_i16); s->kda_gate_i16 = NULL;
    }
#endif /* LZ_KGATE_I16 */

    return 0;
}
