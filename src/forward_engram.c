/* Engram conditional n-gram memory forward pass.
 *
 * Implements kunmoe_modeling.py's EngramMemory.forward(), injected as a
 * hook on embed_tokens (before the layer loop) rather than as a decoder
 * layer.
 *
 * The forward:
 *   1. Rolling polynomial hash: [nt, n_spaces] int indices into flat table
 *   2. Per-order: table lookup → key/value proj → SPDA gate → sum
 *   3. Depthwise dilated conv1d + SiLU
 *   4. residual += value + silu(conv(value))
 *
 * Parity target: tests/test_rectangular_state.py's f32 arm.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"

/* Rolling polynomial hash: h = (h * base + token) & MASK64.
   Per-order bases match kunmoe_modeling.py:
     bases[o] = 2*((o+1)*7+1)+1  -> {29, 43, 57, 71, ...}
   Each head reduces h mod its own prime to get the table row index.
   Window width = order = o+2, matching PyTorch's F.pad(ngram-1, 0)
   followed by a sliding window of width `order` starting at each
   position. */
static void engram_hash(const int *tokens, int nt,
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
                   prefixed array = tokens[t+k-pad] for k=0..order-1,
                   clamped to 0. For ngram=3: order 2 reads [t-2, t-1],
                   order 3 reads [t-2, t-1, t]. */
                for (k = t - pad; k < t - pad + w; k++) {
                    int tk = (k < 0) ? 0 : tokens[k];
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
    rms = sqrtf(rms / (float)n);
    fprintf(stderr, "[engram] %s rms=%.6f  first4=[%.6f, %.6f, %.6f, %.6f]\n",
            stage, rms, data[0], data[1], data[2], data[3]);
}

static void engram_dump_idx(const char *stage, const int *data, int n) {
    (void)n;
    fprintf(stderr, "[engram] %s first4=[%d, %d, %d, %d]\n",
            stage, data[0], data[1], data[2], data[3]);
}
void forward_engram(const LZModel *m, LZRunState *s,
                    const int *tokens, int nt, int pos0) {
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

    /* Scratch layout (wscr is 4 * hidden_size floats, allocated by
       lz_state_alloc). Three nt*dim blocks fit when nt <= batch_max
       and hidden_size <= 512; for larger models, fall back to malloc.
       For now the scratch is enough for KunKun98/CE. */
    int *hidx = (int *)s->wscr;
    size_t hidx_sz = (size_t)nt * ns;
    size_t hidx_bytes = hidx_sz * sizeof(int);
    /* float scratch starts after hidx, aligned to 16 bytes */
    float *fscr = (float *)(((size_t)hidx + hidx_bytes + 15) & ~(size_t)15);

    /* 1. Hash */
    engram_hash(tokens, nt, eg->primes, eg->offsets, eg->bases,
                no, nh, ns, hidx);
    if (dbg) {
        engram_dump_idx("hash", hidx, ns);
    }

    /* 2-3. Per-order: table lookup → proj → gate → sum */
    float *value = fscr;                       /* [nt * dim] */
    float *qnorm = value + (size_t)nt * dim;   /* [nt * dim] */
    float *key   = qnorm + (size_t)nt * dim;   /* [nt * dim] */
    float *scratch = key + (size_t)nt * dim;    /* remaining scratch */

    memset(value, 0, (size_t)nt * dim * sizeof(float));

    /* Query = rmsnorm(x) */
    for (tk = 0; tk < nt; tk++)
        lz_rmsnorm(qnorm + (size_t)tk * dim, s->x + (size_t)tk * dim,
                   lz_t_f32(&eg->norm, s->wscr), dim, c->rms_norm_eps);
    if (dbg) engram_dump("qnorm", qnorm, dim);

    for (o = 0; o < no; o++) {
        float *eo = scratch;  /* [nt * po] scratch for table rows */

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

        /* key = key_proj[o] @ eo */
        for (tk = 0; tk < nt; tk++)
            lz_matmul_xq_nt(key + (size_t)tk * dim,
                            eo + (size_t)tk * po, NULL, NULL,
                            &eg->key_proj[o],
                            po, dim, 1);
        if (dbg) { char tag[32]; snprintf(tag, sizeof(tag), "key_o%d", o); engram_dump(tag, key, dim); }

        /* val = value_proj[o] @ eo */
        float *val = scratch + (size_t)nt * po;
        for (tk = 0; tk < nt; tk++)
            lz_matmul_xq_nt(val + (size_t)tk * dim,
                            eo + (size_t)tk * po, NULL, NULL,
                            &eg->value_proj[o],
                            po, dim, 1);
        if (dbg) { char tag[32]; snprintf(tag, sizeof(tag), "val_o%d", o); engram_dump(tag, val, dim); }

        /* SPDA gate per token: s = dot(q, key) / sqrt(d),
           gate = sigmoid(sqrt(|s|+eps) * sign(s)) */
        float inv_sqrt_d = 1.0f / sqrtf((float)dim);
        for (tk = 0; tk < nt; tk++) {
            float dot = 0.0f;
            for (i = 0; i < dim; i++)
                dot += qnorm[(size_t)tk * dim + i] * key[(size_t)tk * dim + i];
            float s = dot * inv_sqrt_d;
            float gate = lz_sigmoid(sqrtf(fabsf(s) + 1e-6f)
                                    * (s >= 0 ? 1.0f : -1.0f));
            if (dbg && tk == 0)
                fprintf(stderr, "[engram] gate_o%d s=%.6f gate=%.6f\n",
                        o, s, gate);
            for (i = 0; i < dim; i++)
                value[(size_t)tk * dim + i] += gate * val[(size_t)tk * dim + i];
        }
        if (dbg) { char tag[32]; snprintf(tag, sizeof(tag), "value_o%d", o); engram_dump(tag, value, dim); }
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
        const float *conv_w = lz_t_f32(&eg->conv, s->wscr);

        for (tk = 0; tk < nt; tk++) {
            float *out_row = conv_out + (size_t)tk * dim;
            for (i = 0; i < dim; i++) out_row[i] = 0.0f;
            /* Dilated causal conv: tap k of the kernel reads
               value[t - (kern-1-k)*dil]; PyTorch's zero padding makes
               out-of-range taps contribute zero, which the skip below
               reproduces. Weight layout [channel, 1, kernel]:
               conv_w[c*kern + k]. */
            for (int k = 0; k < kern; k++) {
                int src_t = tk - (kern - 1 - k) * dil;
                if (src_t < 0) continue;
                const float *src = value + (size_t)src_t * dim;
                for (i = 0; i < dim; i++)
                    out_row[i] += src[i] * conv_w[(size_t)i * kern + k];
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
}
