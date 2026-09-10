/* Engram conditional n-gram memory forward pass.
 *
 * Implements kunmoe_modeling.py's EngramMemory.forward(), injected as a
 * hook on embed_tokens (before the layer loop) rather than as a decoder
 * layer.
 *
 * The forward:
 *   1. Rolling polynomial hash: [nt, n_spaces] int indices into flat table
 *   2. Per-order: table lookup -> key/value proj -> SPDA gate -> sum
 *   3. Depthwise dilated conv1d + SiLU
 *   4. residual += value + silu(conv(value))
 *
 * Parity target: tests/test_rectangular_state.py's f32 arm.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Watcom's C90 math.h has no sqrtf/fabsf: route to the double forms
   there. gcc keeps the float forms so the ARM build pulls no libgcc
   double soft-float. Same shape as ops_quant.c's LZ_SQRTF. */
#if defined(__WATCOMC__)
#define LZ_SQRTF(x) ((float)sqrt((double)(x)))
#define LZ_FABSF(x) ((float)fabs((double)(x)))
#else
#define LZ_SQRTF(x) sqrtf(x)
#define LZ_FABSF(x) fabsf(x)
#endif /* __WATCOMC__ */

#include "forward.h"

/* Rolling polynomial hash: h = (h * base + token) & MASK64.
   Per-order bases match kunmoe_modeling.py:
     bases[o] = 2*((o+1)*7+1)+1  -> {29, 43, 57, 71, ...}
   Each head reduces h mod its own prime to get the table row index.
   Window width = order = o+2, matching PyTorch's F.pad(ngram-1, 0)
   followed by a sliding window of width `order` starting at each
   position. */
/* Token at relative index k (k may be negative, reaching before this
   chunk's tokens[0]): k>=0 reads tokens[k]; k<0 reaches into the
   cross-chunk history hist[0..hist_n), where k=-1 is the token
   immediately before tokens[0] (hist[hist_n-1]), k=-2 is
   hist[hist_n-2], and so on. Past the start of the real sequence
   (hist_n exhausted, or no history at all - a fresh lz_state_reset)
   the token is padding, id 0 - matches F.pad's default value=0 in
   kunmoe_modeling.py's _multi_head_hash, the same rule a single chunk
   already used for k<0 before cross-chunk history existed. */
static int engram_tok_at(const int *tokens, const int *hist, int hist_n,
                         int k) {
    if (k >= 0) return tokens[k];
    {
        int back = -k;                 /* 1, 2, 3, ... tokens before tokens[0] */
        if (back <= hist_n) return hist[hist_n - back];
        return 0;                      /* before the real sequence start */
    }
}

static void engram_hash(const int *tokens, int nt,
                        const int *hist, int hist_n,
                        const int *primes, const int *offsets,
                        const lz_i64 *bases,
                        int n_orders, int n_head, int n_spaces,
                        int *out /* [nt, n_spaces] */) {
    int t, o, h_idx;
    for (t = 0; t < nt; t++) {
        for (o = 0; o < n_orders; o++) {
            lz_i64 base = bases[o];
            int w = o + 2;  /* order = o+2, window width = order */
            for (h_idx = 0; h_idx < n_head; h_idx++) {
                int sp = o * n_head + h_idx;
                lz_i64 hv = 0;
                int k;
                int pad = n_orders;   /* ngram - 1 */
                /* Window: padded[t .. t+order-1] over the pad-zero-
                   prefixed array = tokens[t+k-pad] for k=0..order-1.
                   For ngram=3: order 2 reads [t-2, t-1], order 3 reads
                   [t-2, t-1, t]. Negative indices reach into hist -
                   see engram_tok_at. */
                for (k = t - pad; k < t - pad + w; k++) {
                    int tk = engram_tok_at(tokens, hist, hist_n, k);
                    hv = (hv * base + (lz_i64)tk) & ((lz_i64)-1);
                }
                out[(size_t)t * n_spaces + sp] =
                    (int)((lz_i64)hv % primes[sp]) + offsets[sp];
            }
        }
    }
}

/* DEBUG: dump engine-side engram intermediates for parity check.
   Prints only for the first token (pos0=0, nt=1) to keep output small.
   Set ENGRAM_DUMP=1 in environment to enable. */
static int engram_dump_enabled(void) {
    const char *v = getenv("ENGRAM_DUMP");
    return v && v[0] == '1';
}

static void engram_dump(const char *stage, const float *data, int n) {
    float rms = 0.0f;
    int i;
    for (i = 0; i < n; i++) rms += data[i] * data[i];
    rms = LZ_SQRTF(rms / (float)n);
    fprintf(stderr, "[engram] %s rms=%.6f  first4=[%.6f, %.6f, %.6f, %.6f]\n",
            stage, rms, data[0], data[1], data[2], data[3]);
}

static void engram_dump_idx(const char *stage, const int *data, int n) {
    (void)n;
    fprintf(stderr, "[engram] %s first4=[%d, %d, %d, %d]\n",
            stage, data[0], data[1], data[2], data[3]);
}

static int engram_is_format(const unsigned char *bitmap, int tokid) {
    return (bitmap[tokid >> 3] >> (tokid & 7)) & 1;
}

/* kunmoe_modeling.py's EngramMemory._content_mask: value[t] is zeroed
   (no engram contribution) when EVERY token in the widest window
   (padded[t..t+ngram-1], ngram tokens wide) is a format token. Padding
   positions count as format - torch.nn.functional.pad's `value=True`
   in the Python source - which is a DIFFERENT pad rule from
   engram_hash's (that one pads with token id 0, matching the hash's
   own F.pad default of value=0). The two paddings must not be
   conflated: a mask that treated padding as "token 0" would ask "is
   token 0 a format id" instead of "assume format", changing the
   answer whenever id 0 is not itself a format id.

   "Padding" here means before the real sequence start, not before
   this CHUNK - an index reaching into hist[] is a real prior token
   (from an earlier chunk of the same prefill) and must get the real
   is_format(that token) answer, same as engram_hash reads the real
   token id there rather than 0. Only running past hist_n (nothing
   recorded that far back - the actual start of the sequence) is
   padding in the Python sense. */
static int engram_is_content(const int *tokens, const int *hist, int hist_n,
                             int t, int ngram, const unsigned char *bitmap) {
    int k;
    for (k = 0; k < ngram; k++) {
        int idx = t - (ngram - 1) + k;
        int is_fmt;
        if (idx >= 0) {
            is_fmt = engram_is_format(bitmap, tokens[idx]);
        } else {
            int back = -idx;
            is_fmt = (back <= hist_n)
                    ? engram_is_format(bitmap, hist[hist_n - back])
                    : 1;   /* before the sequence start: assume format */
        }
        if (!is_fmt) return 1;   /* one content token is enough */
    }
    return 0;   /* every token in the window is format (or padding) */
}
void forward_engram(const LZModel *m, LZRunState *s,
                    const int *tokens, int nt, int pos0,
                    const int *hist, int hist_n) {
    const LZModelConfig *c = &m->config;
    const LZEngram *eg = m->engram;
    int dim = c->hidden_size;
    int edim = c->engram_dim;
    int ns = c->engram_n_spaces;
    int no = c->engram_n_orders;
    int nh = c->engram_n_head;
    int po = c->engram_per_order;
    int tk, o, i;
    int dbg = engram_dump_enabled() && pos0 == 0 && nt >= 1;

    /* Scratch layout lives in s->engram_scratch, NOT s->wscr - that
       buffer is one weight row wide (forward.h's field comment) and a
       BF16 norm's f32 expansion, or nt=8 prefill, both overrun it
       silently (see forward.h's engram_scratch comment for the two
       ways this broke before the dedicated buffer existed).
       lz_state_alloc sizes engram_scratch for nt=LZ_BATCH_MAX, so any
       nt <= that fits. */
    int *hidx = (int *)s->engram_scratch;
    size_t hidx_sz = (size_t)nt * ns;
    size_t hidx_bytes = hidx_sz * sizeof(int);
    /* float scratch starts after hidx, aligned to 16 bytes */
    float *fscr = (float *)(((size_t)hidx + hidx_bytes + 15) & ~(size_t)15);
    /* 2-3 scratch layout, declared here rather than at first use so
       every declaration in this function precedes every statement -
       VC++4.0/MIPS's C89 floor (build/c89_floor_gate.sh) rejects mixed
       declarations and code. */
    float *value = fscr;                       /* [nt * dim] */
    float *qnorm = value + (size_t)nt * dim;   /* [nt * dim] */
    float *key   = qnorm + (size_t)nt * dim;   /* [nt * dim] */
    float *scratch = key + (size_t)nt * dim;    /* remaining scratch */

    /* 1. Hash */
    engram_hash(tokens, nt, hist, hist_n, eg->primes, eg->offsets, eg->bases,
                no, nh, ns, hidx);
    if (dbg) {
        engram_dump_idx("hash", hidx, ns);
    }

    /* 2-3. Per-order: table lookup -> proj -> gate -> sum */
    memset(value, 0, (size_t)nt * dim * sizeof(float));

    /* Query = rmsnorm(x). eg->norm's f32 view must NOT reuse `hidx`'s
       memory (the start of engram_scratch): a BF16 norm tensor makes
       lz_t_f32 WRITE dim(=512) floats into its scratch buffer, which
       would overrun hidx (at most LZ_BATCH_MAX*ns ints) and corrupt
       the very indices the table lookup below still needs to read.
       `scratch` is unused until the per-order loop begins, so it is a
       safe destination for this call - it needs only `dim` floats,
       well within scratch's later size (nt*po or nt*dim, both >= dim
       for every checkpoint here). */
    for (tk = 0; tk < nt; tk++)
        lz_rmsnorm(qnorm + (size_t)tk * dim, s->x + (size_t)tk * dim,
                   lz_t_f32(&eg->norm, scratch), dim, c->rms_norm_eps);
    if (dbg) engram_dump("qnorm", qnorm, dim);

    for (o = 0; o < no; o++) {
        float *eo = scratch;  /* [nt * po] scratch for table rows */
        float *val = scratch + (size_t)nt * po;
        float inv_sqrt_d = 1.0f / LZ_SQRTF((float)dim);

        /* Gather this order's table rows into eo */
        for (tk = 0; tk < nt; tk++) {
            for (i = 0; i < nh; i++) {
                int sp = o * nh + i;
                int idx = hidx[(size_t)tk * ns + sp];
                lz_t_row_f32(&eg->table, idx, edim,
                             eo + (size_t)tk * po + (size_t)i * edim);
            }
        }
        if (dbg) { char tag[32]; snprintf(tag, sizeof(tag), "emb_o%d", o); engram_dump(tag, eo, po); }

        /* Quantize eo into s->xq/s->xqs before either matmul below.
           lz_matmul_xq_nt's float-row arm (F32/BF16 weights) ignores
           xq/xqs and reads eo directly, but key_proj/value_proj may be
           any packed format (the shipping recipe puts them at Q6_1) -
           those arms return an all-zero row when xq/xqs are NULL
           (ops_matmul.c's defensive guard: "no silent miscalc" zeros
           the output rather than reading a null pointer). Every other
           matmul_xq_nt call site in this codebase quantizes first;
           this one was the missing case, invisible while key_proj/
           value_proj stayed F32 (the float-row arm never looks at
           xq/xqs) and only surfacing once the shipping recipe put them
           at Q6_1.

           Called with nt=1 per token (matching the loops below, which
           call lz_matmul_xq_nt once per tk rather than batched), so
           xq/xqs each hold exactly one token's worth: po int8s and
           po/gs floats. s->xq/s->xqs are sized nt_cap*qcap and
           nt_cap*(qcap/32) respectively (forward.c's lz_state_alloc),
           both >= po for every checkpoint here, so reusing token slot 0
           on each iteration is safe - forward_engram runs before the
           layer loop populates them for anything else. */
        for (tk = 0; tk < nt; tk++) {
            int gs_key = lz_act_gs(&eg->key_proj[o], po);
            lz_quantize_q8(eo + (size_t)tk * po, po, gs_key, s->xq, s->xqs);
            lz_matmul_xq_nt(key + (size_t)tk * dim, eo + (size_t)tk * po,
                            s->xq, s->xqs, &eg->key_proj[o], po, dim, 1);
        }
        if (dbg) { char tag[32]; snprintf(tag, sizeof(tag), "key_o%d", o); engram_dump(tag, key, dim); }

        /* val = value_proj[o] @ eo. Separate quantization: value_proj's
           gs need not match key_proj's, and the two matmuls are not
           fused, so there is no reuse to preserve by sharing xq/xqs
           between them. */
        for (tk = 0; tk < nt; tk++) {
            int gs_val = lz_act_gs(&eg->value_proj[o], po);
            lz_quantize_q8(eo + (size_t)tk * po, po, gs_val, s->xq, s->xqs);
            lz_matmul_xq_nt(val + (size_t)tk * dim, eo + (size_t)tk * po,
                            s->xq, s->xqs, &eg->value_proj[o], po, dim, 1);
        }
        if (dbg) { char tag[32]; snprintf(tag, sizeof(tag), "val_o%d", o); engram_dump(tag, val, dim); }

        /* SPDA gate per token: s = dot(q, key) / sqrt(d),
           gate = sigmoid(sqrt(|s|+eps) * sign(s)) */
        for (tk = 0; tk < nt; tk++) {
            float dot = 0.0f;
            float s, gate;
            for (i = 0; i < dim; i++)
                dot += qnorm[(size_t)tk * dim + i] * key[(size_t)tk * dim + i];
            s = dot * inv_sqrt_d;
            gate = lz_sigmoid(LZ_SQRTF(LZ_FABSF(s) + 1e-6f)
                              * (s >= 0 ? 1.0f : -1.0f));
            if (dbg && tk == 0)
                fprintf(stderr, "[engram] gate_o%d s=%.6f gate=%.6f\n",
                        o, s, gate);
            for (i = 0; i < dim; i++)
                value[(size_t)tk * dim + i] += gate * val[(size_t)tk * dim + i];
        }
        if (dbg) { char tag[32]; snprintf(tag, sizeof(tag), "value_o%d", o); engram_dump(tag, value, dim); }
    }

    /* Format-content mask, BEFORE the conv - kunmoe_modeling.py zeroes
       `value` here so no gradient reaches a format-only n-gram row
       through any path (the conv is a temporal mixer and would
       otherwise carry a format position's contribution into its
       neighbours' outputs even after zeroing the final sum instead).
       Skipping this at inference computes a different function than
       the one that was trained: a non-zero engram contribution at
       positions training made exactly zero. c->engram_format_bitmap
       is NULL when engram_format_ids is empty/absent in config.json -
       same as kunmoe_modeling.py's `if not self._format_ids`. */
    if (c->engram_format_bitmap) {
        for (tk = 0; tk < nt; tk++) {
            if (!engram_is_content(tokens, hist, hist_n, tk, c->engram_ngram,
                                   c->engram_format_bitmap)) {
                float *v_row = value + (size_t)tk * dim;
                for (i = 0; i < dim; i++) v_row[i] = 0.0f;
            }
        }
        if (dbg) engram_dump("value_masked", value, dim);
    }

    /* 4. Depthwise dilated conv1d + SiLU.
       PyTorch: Conv1d(padding=(kernel-1)*dilation, dilation=dil), then
       trim to T. Equivalent causal form: out[t] = sum_k
       value[t - (kernel-1-k)*dil] * w[c][k]. The stored weight is
       [dim, 1, kernel] (channel-major: w[c*kernel + k]). */
    {
        int kern = c->engram_conv_kernel;
        int dil = c->engram_conv_dilation;
        float *conv_out = scratch;

        for (tk = 0; tk < nt; tk++) {
            float *out_row = conv_out + (size_t)tk * dim;
            for (i = 0; i < dim; i++) out_row[i] = 0.0f;
            /* Dilated causal conv: tap k of the kernel reads
               value[t - (kern-1-k)*dil]; PyTorch's zero padding makes
               out-of-range taps contribute zero, which the skip below
               reproduces. Weight layout [channel, 1, kernel]:
               conv_w[c*kern + k].

               Read one channel's K taps at a time through lz_t_row_f32
               instead of the whole tensor through lz_t_f32. This lets
               BF16 conv tensors stay narrow in RAM: lz_t_f32 would need
               the entire [C,1,K] in s->wscr at once (2048 elements for
               engram's 512-channel conv), overflowing it; lz_t_row_f32
               needs only kern (=4) floats of scratch. */
            for (i = 0; i < dim; i++) {
                int k;
                /* One channel's K taps into wscr[0..kern-1] */
                lz_t_row_f32(&eg->conv, i, kern, s->wscr);
                for (k = 0; k < kern; k++) {
                    int src_t = tk - (kern - 1 - k) * dil;
                    float v;
                    if (src_t >= 0) {
                        v = value[(size_t)src_t * dim + i];
                    } else {
                        /* Before this chunk's first token: reach into
                           the cross-chunk VALUE history (forward.h's
                           engram_conv_hist comment) instead of treating
                           every chunk boundary as sequence start. `back`
                           counts rows before tk=0 the same way
                           engram_tok_at's does for the token history;
                           past what was recorded (real sequence start,
                           or a fresh lz_state_reset) the contribution is
                           zero - PyTorch's Conv1d zero-padding, which a
                           single chunk already relied on for src_t<0. */
                        int back = -src_t;
                        if (back <= s->engram_conv_hist_n) {
                            const float *hrow = s->engram_conv_hist +
                                (size_t)(s->engram_conv_hist_n - back) * dim;
                            v = hrow[i];
                        } else {
                            v = 0.0f;
                        }
                    }
                    out_row[i] += v * s->wscr[k];
                }
            }
            /* SiLU */
            for (i = 0; i < dim; i++)
                out_row[i] = lz_silu(out_row[i]);
        }

        /* 5. Residual add: x += value + conv_out */
        if (dbg) engram_dump("conv_out", conv_out, dim);
        for (tk = 0; tk < nt; tk++) {
            float *x_row = s->x + (size_t)tk * dim;
            float *v_row = value + (size_t)tk * dim;
            float *c_row = conv_out + (size_t)tk * dim;
            for (i = 0; i < dim; i++)
                x_row[i] += v_row[i] + c_row[i];
        }
    }
    (void)pos0;  /* position not needed; engram is position-agnostic */
    if (dbg) engram_dump("residual_add", s->x, dim);

    /* Roll this chunk's trailing `value` rows into engram_conv_hist for
       the NEXT chunk's conv taps - same cross-chunk carry engram_hist
       does for the hash/mask, one stage later (post-gate, pre-conv).
       cap==0 means the conv needs no history (kernel<=1 or
       dilation==0); nothing to do then. */
    if (s->engram_conv_hist && c->engram_conv_kernel > 1) {
        int cap = s->engram_conv_hist_cap;
        if (nt >= cap) {
            memcpy(s->engram_conv_hist, value + (size_t)(nt - cap) * dim,
                  (size_t)cap * dim * sizeof(float));
            s->engram_conv_hist_n = cap;
        } else {
            int old_keep = cap - nt;
            if (old_keep > s->engram_conv_hist_n)
                old_keep = s->engram_conv_hist_n;
            if (old_keep > 0)
                memmove(s->engram_conv_hist,
                       s->engram_conv_hist +
                           (size_t)(s->engram_conv_hist_n - old_keep) * dim,
                       (size_t)old_keep * dim * sizeof(float));
            memcpy(s->engram_conv_hist + (size_t)old_keep * dim, value,
                  (size_t)nt * dim * sizeof(float));
            s->engram_conv_hist_n = old_keep + nt;
        }
    }
}
