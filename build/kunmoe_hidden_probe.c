/* Per-layer intermediates of the real forward, to disk.
 *
 * ENGINE-SIDE INSTRUMENT: it links the engine's own sources and imports
 * nothing from the training tree. Its other half - the training side's
 * dump of the same quantities, and the subtractor - lives there
 * (tests/kunmoe_hidden_dump.py, tests/kunmoe_hidden_compare.py).
 *
 * BUILD/ IS GITIGNORED, so this file and kunmoe_hidden_tap.h need
 * `git add -f` to be committed, the same way build/loop_kv_plane_gate.sh
 * and the rest of the gates in this directory are.
 *
 * It opens a model directory with lz_open (the safetensors path - f32
 * weights, no quantisation), runs lz_forward_batch over ONE chunk, and
 * writes what the engine's own LZ_TAP sites hand it. No model math lives
 * here; the taps are the forward's own, and this file only names them.
 *
 * Built by hand (this is an instrument, not a gate): compile every engine
 * TU, but compile the three that carry taps (forward.c, forward_attn.c,
 * forward_kda.c) with
 *   -include build/kunmoe_hidden_tap.h
 *   -D'LZ_TAP(tag,li,ptr,n)=lz_dump_tap(tag,li,ptr,n)'
 * and link them with this file, so nothing in src/ is edited. The header
 * sits beside this file and declares this file's own dump function.
 *
 * The other half of the comparison - the training side's dump of the same
 * quantities, and the subtractor - lives in the TRAINING tree
 * (tests/kunmoe_hidden_dump.py, tests/kunmoe_hidden_compare.py): a tool
 * that imports kunmoe_modeling is not this tree's to hold.
 *
 * One chunk only: the taps fire once per layer per CHUNK, so a call that
 * the batcher splits would leave the last chunk's rows in the file. The
 * caller therefore passes at most LZ_BATCH_DEFAULT (4) tokens - the
 * activation buffers lz_state_alloc sizes at nt_cap - not LZ_BATCH_MAX.
 *
 * Usage: kunmoe_hidden_probe <model-dir> <out-prefix> <tok,tok,...>
 * Exit: 0 ok, 1 load/forward failure (message on stderr).
 *
 * The training half must be dumped at TRITON_F32_DEFAULT=ieee (its
 * default is TF32 inside fla's triton kernels, a ~5e-4 floor of the
 * reference's own - see the training tree's kunmoe_hidden_dump.py).
 * This side needs no such flag: the engine's fp32 path is fp32.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "forward.h"
#include "ops.h"

static const char *g_pfx;
static int g_nt = 1;

static void dump(const char *name, const void *p, size_t nbytes) {
    char path[512];
    FILE *f;
    if (!p || !nbytes) return;
    snprintf(path, sizeof path, "%s_%s.bin", g_pfx, name);
    f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        return;
    }
    fwrite(p, 1, nbytes, f);
    fclose(f);
}

void lz_dump_tap(const char *tag, int li, const float *v, int n) {
    char name[64];
    /* The layer-boundary buffers hold one row per token of the chunk;
       "fnorm" is handed the last token's row alone. */
    int rows = (strcmp(tag, "fnorm") == 0) ? 1 : g_nt;
    snprintf(name, sizeof name, "%s%d_%s",
             li < 0 ? "X" : "L", li < 0 ? 0 : li, tag);
    dump(name, v, (size_t)rows * (size_t)n * sizeof(float));
}

int main(int argc, char **argv) {
    LZModel m;
    LZRunState st;
    char err[512];
    int tokens[LZ_BATCH_DEFAULT];
    int n = 0;
    const char *p;
    float *lg;
    unsigned fpu;

    if (argc < 4) {
        fprintf(stderr, "usage: %s <model-dir> <out-prefix> <tok,...>\n",
                argv[0]);
        return 2;
    }
    g_pfx = argv[2];
    p = argv[3];
    while (*p && n < LZ_BATCH_DEFAULT) {
        char *end;
        long v = strtol(p, &end, 10);
        tokens[n++] = (int)v;
        p = (*end == ',') ? end + 1 : end;
        if (*p == '\0') break;
    }
    if (n < 1 || n > LZ_BATCH_DEFAULT) {
        fprintf(stderr, "token count %d outside [1, %d]\n", n, LZ_BATCH_DEFAULT);
        return 2;
    }
    g_nt = n;

    /* THE CALIBER. Without these the probe runs the SHIPPING defaults:
       --fixed all (fixed-point arithmetic at every site) and --kv auto
       (dense KV cache K q8 / V q4). Either one alone puts a ~1e-1 gap
       between this probe and an fp reference, which reads exactly like a
       structural disagreement. They are the same switches the CLI sets
       for `--kernel ref --fixed off --pair off --prefetch none
       --fastfp off --kv f32`, applied here because this probe calls the
       forward directly.

       The same caliber is spelled a second time in the training tree's
       tests/test_kunmoe_cross_side.py, which compares logits rather than
       intermediates. One caliber, two spellings; if a knob here gains a
       setting that matters, both move. */
    lz_float_ref_select("off");
    lz_pair_select("off");
    lz_prefetch_select("none");
    lz_fastfp_select("off");
    lz_kv_kfmt = LZ_KVF_F32;
    lz_kv_vfmt = LZ_KVF_F32;

    if (lz_open(&m, argv[1], err, sizeof err) != 0) {
        fprintf(stderr, "open: %s\n", err);
        return 1;
    }
    if (lz_read_weights(&m, err, sizeof err) != 0) {
        fprintf(stderr, "read: %s\n", err);
        return 1;
    }
    /* 32 positions is enough for the 8-token comparison and keeps the
       ring tables small; this probe never generates. */
    if (lz_state_alloc(&st, &m, 32, 0, err, sizeof err) != 0) {
        fprintf(stderr, "state: %s\n", err);
        return 1;
    }

    /* The weight half of "are the two sides reading the same bytes":
       three tensors straight out of lz_t_f32 - the same accessor every
       projection uses, not a re-read of the file. Chosen for the ladder
       fixture ([KDA, dense-FFN]): the embedding, one KDA conv tap, one
       dense-FFN projection. A tensor this fixture does not carry dumps
       nothing, which the compare step reports as a missing row. */
    {
        int d = m.config.hidden_size;
        dump("W_embed", lz_t_f32(&m.embed_tokens, st.wscr),
             (size_t)m.config.vocab_size * (size_t)d * sizeof(float));
        dump("W_l0_qconv", lz_t_f32(&m.layers[0].kda_q_conv1d, st.wscr),
             (size_t)m.layers[0].kda_q_conv1d.n * sizeof(float));
        dump("W_l0_down", lz_t_f32(&m.layers[0].down_proj, st.wscr),
             (size_t)m.layers[0].down_proj.n * sizeof(float));
    }

    fpu = lz_fpu_float_begin();
    lg = lz_forward_batch(&m, &st, tokens, n, 0);
    if (!lg) {
        lz_fpu_float_end(fpu);
        fprintf(stderr, "forward failed\n");
        return 1;
    }
    dump("logits", lg, (size_t)m.config.vocab_size * sizeof(float));
    lz_fpu_float_end(fpu);

    printf("probe: %d tokens, vocab %d\n", n, m.config.vocab_size);
    lz_state_free(&st);
    lz_free(&m);
    return 0;
}
