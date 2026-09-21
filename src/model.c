#include <stdarg.h>
#include <stddef.h>
#include "lz_int.h"   /* not <stdint.h>: VC++ 4.0 has no such header */
#include <ctype.h>    /* isdigit: csa2_key_idx's leading-character check */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "err.h"
#include "lfn.h"
#include "lz_int.h"   /* lz_i64: the 64-bit type, portably */
#include "lz4d.h"
#include "model.h"
#include "ops.h"   /* lz_bf16_store_g, LZ_MM_WIDEN_MAX, LZ_SPEC_K_MAX (alloc_mtp's mtp_n_draft bound) */

#if LZ_DBG_LAYER_HASH
/* Per-layer residual hash for the fast-mode (--fastfp) cross-target
   divergence hunt. PURE INTEGER over the buffer's bit patterns: no float
   op, so it cannot change the rounding of the arithmetic it measures.
   Lives here (a large, cold TU) rather than in forward.c so the compiler
   cannot inline it into the layer loop and reorder float work around it.
   -DLZ_DBG_LAYER_HASH=1 is a diagnostic build only; nothing in a normal
   build references it.

   The lines go to the file named by LHASH_FILE, one per layer visit -
   the engine has no console, and a divergence hunt compares two target
   builds by diffing two files. Nothing is written when the variable is
   unset or empty, so a build that defines the macro still dumps
   nothing until it is asked to. */
void lz_dbg_layer_hash(const char *tag, int li, const void *p, size_t nf) {
    static FILE *fp;
    static int tried;
    const unsigned char *b = (const unsigned char *)p;
    unsigned long long h = 1469598103934665603ull;   /* FNV-1a */
    size_t i, m;
    char line[96];
    int n;
    for (i = 0; i < nf * 4u; i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    if (!tried) {
        const char *path = getenv("LHASH_FILE");
        tried = 1;
        if (path && path[0]) fp = fopen(path, "w");
    }
    if (!fp) return;
    n = snprintf(line, sizeof line, "LHASH %s %d %llu\n", tag, li, h);
    if (n <= 0) return;
    m = (size_t)n;
    if (m > sizeof line - 1) m = sizeof line - 1;
    fwrite(line, 1, m, fp);
    fflush(fp);
}
#endif /* LZ_DBG_LAYER_HASH */

/* 32-byte aligned malloc/free for the weight buffers (field->f and the
   quantized planes field->q/scale/zero). The SSE1/SSE2 row kernels read
   these with movaps/movdqa and the AVX2 tier with 256-bit loads. Q8 and
   Q16 - 32- and 64-byte groups - become aligned 256-bit loads and Q4_1's
   16-byte group an aligned 128-bit one; Q6_1's two planes and T2 keep
   8-byte groups and stay unaligned. Sixteen would leave every odd Q8
   group at 16 mod 32, so it is strictly weaker and nothing depended on
   exactly sixteen.

   Over-allocates by one pointer plus the pad and stores the ORIGINAL
   pointer just before the aligned one, the same shape as forward.c's
   xcalloc/xfree. */
static void *aligned_malloc(size_t n, size_t sz) {
    char *base = (char *)malloc(n * sz + 32 + sizeof(void *));
    char *aligned;
    if (!base) return NULL;
    aligned = base + sizeof(void *);
    aligned += (size_t)(32 - ((size_t)aligned & 31)) & 31;
    ((void **)(void *)aligned)[-1] = base;
    return aligned;
}

static void aligned_free(void *p) {
    if (p) free(((void **)p)[-1]);
}

static void qerr(char *errbuf, int errlen, LZErr code, ...) {
    va_list ap;
    if (!errbuf || errlen <= 0) return;
    va_start(ap, code);
    lz_err_fmt_v(errbuf, errlen, code, ap);
    va_end(ap);
}

const char *lz_layer_type_name(int t) {
    if (t == LZ_LT_FULL) return "full_attention";
    if (t == LZ_LT_KDA)  return "linear_attention (kda)";
    return "linear_attention";
}

/* ---------------------------------------------------------------- config */

/* Settle hadamard_o / hadamard_down into concrete values and refuse the
   illegal ones. Called by BOTH loaders (config.json and model.bin) once
   lin_value_dim / intermediate_size are known, so the forward pass reads
   one already-resolved number and never re-derives.

   -1 means the product predates the fields; the two derivations below
   reproduce the forward pass's own inline computation, so a legacy
   model keeps its exact behaviour.

   An explicit block size that is not a power of two, or does not divide
   the projection's input dim, is a refusal rather than a fallback to 0:
   quietly dropping the transform would let a mis-exported model produce
   numbers that look reasonable and are not. Block size 1 IS legal and
   means off - a 1x1 Hadamard is the identity, so the forward pass's
   `blk > 1` guard is exact, not an approximation. */
static int hadamard_ok(int blk, int dim) {
    if (blk == 0) return 1;                    /* off */
    if (blk < 0) return 0;                     /* -1 is handled by the caller */
    if (blk & (blk - 1)) return 0;             /* not a power of two */
    return dim > 0 && dim % blk == 0;
}

/* The forward pass reads the gate half of every attention head
   unconditionally: forward_attn.c's output gate and forward_csa2.c's
   both index qg at h * 2 * hd + hd, and forward_mtp.c's draft head does
   the same. That half only exists when attn_output_gate is set -
   attn_qgate_dim, and so the s->qg allocation it sizes, is attn_q_dim
   wide otherwise - so with the key absent or false every one of those
   reads runs a second time past the end of the buffer. Refusing at load
   is the fix; running the three forwards ungated is a feature, and
   forward_csa2.c's header already records that it was not attempted. */
static int check_attn_gate(const LZModelConfig *c, char *errbuf, int errlen) {
    if (c->attn_output_gate) return 0;
    qerr(errbuf, errlen, LZ_ERR_CFG_ATTN_GATE);
    return 1;
}

static int resolve_hadamard(LZModelConfig *c, char *errbuf, int errlen) {
    int vdim = c->lin_value_dim;
    int idim = c->intermediate_size;
    if (c->hadamard_o == -1) {
        /* Derived only on the dense path. The training side's _ternarize
           returns early for num_experts != 0 ("KunMoe keeps fp
           projections; ternary is CE-only"), so its own default is 0
           there, and a non-zero block here would rotate an activation
           those weights never saw - invisible in float, because the float
           matmul never reads the rotated buffer. hadamard_down has no
           such condition: _make_dense_ffn_bitlinear still builds a MoE
           model's dense prefix layers as ternary. */
        int oblk = c->num_experts == 0 ? vdim : 0;
        if (oblk < 2 || (oblk & (oblk - 1))) oblk = 0;
        c->hadamard_o = oblk;
    } else if (!hadamard_ok(c->hadamard_o, vdim)) {
        qerr(errbuf, errlen, LZ_ERR_CFG_HADAMARD,
             "hadamard_o", c->hadamard_o, "lin_value_dim", vdim);
        return 1;
    }
    if (c->hadamard_down == -1) {
        int dblk = 512;
        while (dblk > 1 && idim % dblk) dblk >>= 1;
        if (dblk == 1) dblk = 0;
        c->hadamard_down = dblk;
    } else if (!hadamard_ok(c->hadamard_down, idim)) {
        qerr(errbuf, errlen, LZ_ERR_CFG_HADAMARD,
             "hadamard_down", c->hadamard_down, "intermediate_size", idim);
        return 1;
    }
    /* One hadamard_down serves two consumers: the body's dense FFN and
       the MTP block's, whose width is its own field. The block MUST
       divide both - forward.c's block loop steps b_i += blk to n and
       would write past the row otherwise. Equal widths (every product so
       far, and the default when the key is absent) make this vacuous. */
    if (c->use_subn && c->mtp_n_layers > 0 && c->hadamard_down > 1 &&
        (c->mtp_intermediate_size <= 0 ||
         c->mtp_intermediate_size % c->hadamard_down)) {
        qerr(errbuf, errlen, LZ_ERR_CFG_HADAMARD,
             "hadamard_down", c->hadamard_down,
             "mtp_intermediate_size", c->mtp_intermediate_size);
        return 1;
    }
    return 0;
}

/* ffn_activation and its per-mechanism params. Shared between
   lz_load_config (config.json is the primary source) and lz_open_bin's
   config.json sidecar read (Engram-style: the bin header has no field
   for this, so a .bin-loaded model reads it straight from the sidecar
   the same way it reads engram_*) so the two loading paths cannot
   drift out of parsing the same key two different ways. */
static int parse_ffn_activation(LZJson *j, const LZJsonNode *tc,
                                LZModelConfig *c, char *errbuf, int errlen) {
    const char *fa = lz_json_get_str(j, tc, "ffn_activation", "");
    if (fa[0] == '\0' || strcmp(fa, "none") == 0) {
        c->ffn_activation = LZ_FFN_ACT_NONE;
    } else if (strcmp(fa, "situ") == 0) {
        c->ffn_activation = LZ_FFN_ACT_SITU;
    } else if (strcmp(fa, "gptoss_clamp") == 0) {
        c->ffn_activation = LZ_FFN_ACT_GPTOSS_CLAMP;
    } else {
        /* Mirrors KunMoeConfig.__init__'s own ValueError - found there
           first (a typo'd value fell through silently and ran
           hidden_act's activation with no error, see that constructor's
           comment) and refused here for the same reason: a checkpoint
           carrying a name the engine does not recognize must not
           silently fall back to lz_silu*up. */
        qerr(errbuf, errlen, LZ_ERR_CFG_FFN_ACT, fa);
        return 1;
    }
    c->activation_situ_beta =
        (float)lz_json_get_num(j, tc, "activation_situ_beta", 4.0);
    {
        /* linear_beta's None (unbounded up) is a real, exercised branch
           on the training side (situ_glu's own default) - read the node
           directly, same reasoning as kda_gate_lower_bound. */
        const LZJsonNode *n = lz_json_get(j, tc, "activation_situ_linear_beta");
        c->ffn_has_situ_linear_beta = (n && n->type == LZ_JSON_NUM);
        c->activation_situ_linear_beta =
            c->ffn_has_situ_linear_beta ? (float)n->num : 0.0f;
    }
    c->swiglu_clamp_limit =
        (float)lz_json_get_num(j, tc, "swiglu_clamp_limit", 10.0);
    c->swiglu_clamp_alpha =
        (float)lz_json_get_num(j, tc, "swiglu_clamp_alpha", 1.702);
    return 0;
}

/* csa2_kv_mode/csa2_kv_source's member names: a layer index written as a
   JSON object key, which JSON has no integer form for. -1 on anything
   that is not ENTIRELY an ASCII digit string in [0, LZ_MAX_LAYERS) -
   the isdigit check on the leading byte is what actually delivers that,
   since strtol's own leading-whitespace and leading-sign handling would
   otherwise accept " 2" or "+2" as if they were "2". The upper bound is
   checked in `long` before any cast to `int`, so a key many digits long
   (past LONG_MAX, or merely past int's range) cannot become a small,
   in-range int by truncation - it becomes -1 like any other malformed
   key, not a wrapped-around layer index. */
static int csa2_key_idx(const char *key) {
    char *end;
    long v;
    if (!key || !key[0] || !isdigit((unsigned char)key[0])) return -1;
    v = strtol(key, &end, 10);
    if (*end != '\0' || v < 0 || v >= LZ_MAX_LAYERS) return -1;
    return (int)v;
}

static int st_path_from_index(const char *dir, char *out, int outlen,
                              char *errbuf, int errlen);

/* The one place that decides which file holds the weights: model.safetensors
   when it is there, otherwise the single shard a
   model.safetensors.index.json names (st_path_from_index has that case).
   Returns 0 and fills out, or non-zero with errbuf set.

   ONE resolver for both callers, because lz_open loads from what this
   returns and the architecture probe below reads that same file's header.
   Two resolvers could pick two files, and then the verdict would be about
   a checkpoint nobody is loading. */
static int st_resolve(const char *dir, char *out, int outlen,
                      char *errbuf, int errlen) {
    int ir;
    if (lz_lfn_exists(dir, "model.safetensors"))
        return lz_lfn_path(dir, "model.safetensors", out, outlen,
                           errbuf, errlen) != 0;
    ir = st_path_from_index(dir, out, outlen, errbuf, errlen);
    if (ir < 0) return 1;
    if (ir == 1) return 0;
    return lz_lfn_path(dir, "model.safetensors", out, outlen,
                       errbuf, errlen) != 0;
}

/* The directory a config.json path lives in, WITH its trailing separator
   kept. Keeping it is what gets the root right: "C:" names the current
   directory of drive C while "C:/" names the root, the distinction
   lfn.c's join() draws for itself. Both separators occur here -
   lz_lfn_path builds '/', and the user's own directory argument is passed
   through untouched. */
static void cfg_dir_of(const char *cfg_path, char *out, int cap) {
    int n = (int)strlen(cfg_path);
    int i;
    for (i = n - 1; i >= 0; i--)
        if (cfg_path[i] == '/' || cfg_path[i] == '\\') break;
    i++;                                    /* keep the separator itself */
    if (i > cap - 1) i = cap - 1;
    memcpy(out, cfg_path, (size_t)i);
    out[i] = '\0';
}

/* Which generation this checkpoint is, decided from what its linear layers
   are NAMED rather than from what its model_type says.

   GDN (Qwen3.5) writes `linear_attn.in_proj_qkv.weight`; KDA (KunMoe)
   writes `linear_attn.q_proj.weight` / `linear_attn.f_a_proj.weight`. The
   two sets are disjoint in every checkpoint on disk, and the tensor set is
   also the thing the loader has to bind - a model_type disagreeing with
   its own tensors would only move the failure later, to a missing-tensor
   error that no longer says which generation was expected.

   Both names carry the `linear_attn.` anchor: a full-attention layer's own
   `self_attn.q_proj.weight` shares the suffix and would otherwise answer
   for an architecture it does not belong to.

   Reads the header's MEMBER NAMES and nothing else, rather than going
   through lz_st_open. lz_st_open builds the whole tensor table, so its
   success also depends on the data_offsets validating against the file
   size - and those are JSON numbers. The verdict would then be "no
   verdict" for a checkpoint whose names state their generation plainly,
   whenever an unrelated offset fails to round-trip. Names are strings;
   the question is about names.

   Returns 1 KDA, 0 GDN, -1 neither (a checkpoint carrying both name sets
   is not either generation), -2 the header could not be read. -2 is
   deliberately not a verdict: the loader that owns that file reports it
   by name, and "neither generation" would be a claim about a file nobody
   read. */
static int arch_probe(const char *cfg_path) {
    unsigned char lenbuf[8];
    lz_i64 fsz;
    lz_u64 hdr_len = 0;
    char *hdr;
    FILE *fp;
    LZJson j;
    const LZJsonNode *root, *m;
    char dir[512], stp[512];
    int i, seen_kda = 0, seen_gdn = 0;

    cfg_dir_of(cfg_path, dir, (int)sizeof dir);
    if (st_resolve(dir, stp, (int)sizeof stp, NULL, 0) != 0) return -2;

    /* The safetensors envelope, same shape src/safetensors.c reads: an
       eight-byte little-endian header length, then that many bytes of
       JSON. The length is assembled byte by byte, and the bound is
       written as a subtraction, so neither host byte order nor a
       wrapped addition can turn a malformed header into a large malloc. */
    fsz = lz_fsize(stp);
    if (fsz < 8) return -2;
    fp = fopen(stp, "rb");
    if (!fp) return -2;
    if (fread(lenbuf, 1, 8, fp) != 8) { fclose(fp); return -2; }
    for (i = 7; i >= 0; i--) hdr_len = (hdr_len << 8) | (lz_u64)lenbuf[i];
    if (hdr_len == 0 || hdr_len > (lz_u64)fsz - LZ_U64_C(8)) {
        fclose(fp);
        return -2;
    }
    hdr = (char *)malloc((size_t)hdr_len);
    if (!hdr) { fclose(fp); return -2; }
    if (fread(hdr, 1, (size_t)hdr_len, fp) != (size_t)hdr_len) {
        free(hdr);
        fclose(fp);
        return -2;
    }
    fclose(fp);
    if (lz_json_parse(&j, hdr, (size_t)hdr_len, NULL, 0) != 0) {
        free(hdr);
        return -2;
    }
    free(hdr);   /* lz_json_parse copies: the names below point into that */

    root = lz_json_root(&j);
    if (root && root->type == LZ_JSON_OBJ) {
        for (m = lz_json_first(&j, root); m; m = lz_json_next(&j, m)) {
            const char *n = m->key;
            if (!n || strcmp(n, "__metadata__") == 0) continue;
            if (strstr(n, "linear_attn.in_proj_qkv.weight")) seen_gdn = 1;
            else if (strstr(n, "linear_attn.q_proj.weight") ||
                     strstr(n, "linear_attn.f_a_proj.weight")) seen_kda = 1;
        }
    }
    lz_json_free(&j);
    if (seen_kda == seen_gdn) return -1;
    return seen_kda;
}

int lz_load_config(LZModelConfig *c, const char *path,
                       char *errbuf, int errlen) {
    char *buf;
    size_t sz = 0;
    LZJson j;
    const LZJsonNode *root, *tc, *lt, *rp, *e;
    int i;
    int is_kunmoe;

    memset(c, 0, sizeof(*c));
    buf = (char *)lz_read_file(path, &sz, errbuf, errlen);
    if (!buf) return 1;

    if (lz_json_parse(&j, buf, sz, errbuf, errlen) != 0) {
        free(buf);
        return 1;
    }
    free(buf);

    root = lz_json_root(&j);
    if (!root || root->type != LZ_JSON_OBJ) {
        qerr(errbuf, errlen, LZ_ERR_CFG_ROOT);
        lz_json_free(&j);
        return 1;
    }
    /* The multimodal shell puts text params under text_config; plain-text checkpoints have them at top level */
    tc = lz_json_get(&j, root, "text_config");
    if (!tc) tc = root;

    /* GDN vs KDA, decided by the checkpoint's own linear-layer tensor
       names - see arch_probe. The config's layer_types entries stay the
       plain "linear_attention"/"full_attention" strings in both
       generations (kunmoe_modeling.py keeps the Qwen3.5 skeleton
       deliberately), so the string cannot say which of the two an entry
       means. This engine loads those two generations and nothing else; a
       checkpoint that is neither is refused here rather than half-bound
       against one of their specs. */
    {
        int arch = arch_probe(path);
        if (arch == -2) {
            /* The weights could not be opened. is_kunmoe only selects
               which spec the layers are bound against, and the weight
               load that follows reports the unreadable file by name. */
            is_kunmoe = 0;
        } else if (arch < 0) {
            qerr(errbuf, errlen, LZ_ERR_CFG_UNSUPPORTED_ARCH);
            lz_json_free(&j);
            return 1;
        } else {
            is_kunmoe = (arch == 1);
        }
    }

    c->vocab_size        = lz_json_get_int(&j, tc, "vocab_size", 0);
    c->hidden_size       = lz_json_get_int(&j, tc, "hidden_size", 0);
    c->n_layers          = lz_json_get_int(&j, tc, "num_hidden_layers", 0);
    c->n_heads           = lz_json_get_int(&j, tc, "num_attention_heads", 0);
    c->n_kv_heads        = lz_json_get_int(&j, tc, "num_key_value_heads", 0);
    c->head_dim          = lz_json_get_int(&j, tc, "head_dim", 0);
    c->lin_n_k_heads     = lz_json_get_int(&j, tc, "linear_num_key_heads", 0);
    c->lin_n_v_heads     = lz_json_get_int(&j, tc, "linear_num_value_heads", 0);
    c->lin_k_head_dim    = lz_json_get_int(&j, tc, "linear_key_head_dim", 0);
    c->lin_v_head_dim    = lz_json_get_int(&j, tc, "linear_value_head_dim", 0);
    c->conv_kernel       = lz_json_get_int(&j, tc, "linear_conv_kernel_dim", 0);
    c->intermediate_size = lz_json_get_int(&j, tc, "intermediate_size", 0);
    c->mtp_n_layers      = lz_json_get_int(&j, tc, "mtp_num_hidden_layers", 0);
    /* Defaults to intermediate_size (stock upstream never writes this
       key, because there the two ARE equal) - see model.h's field
       comment for why a checkpoint whose body doesn't use
       intermediate_size (kunmoe-v2) needs this to be independent. */
    c->mtp_intermediate_size =
        lz_json_get_int(&j, tc, "mtp_intermediate_size", c->intermediate_size);
    /* DSpark (arXiv:2607.05147). 1 = every checkpoint before this
       field existed and every upstream checkpoint - LZMtp binds `fc`,
       not the five
       main_proj/depth_embed/mask_embed/markov_embed/markov_proj
       tensors. mtp_markov_rank is meaningless (left 0) at
       mtp_n_draft<=1 and unread in that case. */
    c->mtp_n_draft       = lz_json_get_int(&j, tc, "mtp_n_draft", 1);
    c->mtp_markov_rank   = lz_json_get_int(&j, tc, "mtp_markov_rank", 0);

    /* CED bounded replay (model.h's own field comment has the full
       argument) - 0/0 defaults reproduce a config.json that never
       carried these keys. */
    c->ced_split      = lz_json_get_int(&j, tc, "ced_split", 0);
    c->ced_replay_w   = lz_json_get_int(&j, tc, "ced_replay_w", 0);

    /* ced_kv (model.h's own field comment). A string, not a bool - it
       has three possible values on the training side (off/proj/reuse)
       and this engine implements only two of them. */
    {
        const char *ck = lz_json_get_str(&j, tc, "ced_kv", "off");
        if (strcmp(ck, "off") == 0) {
            c->ced_kv = 0;
        } else if (strcmp(ck, "proj") == 0) {
            c->ced_kv = 1;
        } else {
            /* Covers "reuse" and any typo'd value alike - same posture
               kda_gate_activation/ffn_activation take on a name the
               engine does not recognize, rather than silently falling
               back to off. */
            qerr(errbuf, errlen, LZ_ERR_CFG_CED_KV, ck);
            lz_json_free(&j);
            return 1;
        }
    }

    /* S1 attn_kv_share (model.h's own field comment). Read as a BOOL,
       not with lz_json_get_int: the key is a JSON true/false in every
       config.json the training side writes, and lz_json_get_int answers
       its default for a LZ_JSON_BOOL node - the switch would never fire
       and the target layers' absent k/v tensors would come back as an
       ordinary missing-tensor error. Same reader use_subn and
       attn_output_gate use for their own boolean keys. */
    c->attn_kv_share  = lz_json_get_bool(&j, tc, "attn_kv_share", 0);

    /* Learned attention sink (model.h's own attn_sink_learned comment).
       A BOOL for the same reason attn_kv_share just above is one, and
       the .bin path reads the same switch out of the v5 tail
       (attn_sink_learned) - the two loaders have to agree on it or the
       same checkpoint is two different models depending on which file
       format it was written in. attn_sink_init is a float and has no
       header field at all; model.h's comment on it says what that
       costs. */
    c->attn_sink_learned = lz_json_get_bool(&j, tc, "attn_sink", 0);
    c->attn_sink_init    = (float)lz_json_get_num(&j, tc, "attn_sink_init", 0.0);

    /* GR (model.h's own gated_residual comment). An int, not a BOOL: the
       key is the branch count. This reader is the config.json half of the
       same switch lz_open_bin takes out of the v5 tail - the two loaders
       have to agree on it or one checkpoint is two different models
       depending on which file format it was written in, which is exactly
       the divergence setup_layers' gr_on would turn from invisible into a
       different forward pass. No range check here: the bin path's
       LZ_ERR_BIN_FIELD_RANGE and this reader both feed the same
       `>= 2` test in setup_layers, so a config.json asking for nr=1 gets
       no branches and no specs - and a checkpoint that carries gr_*
       tensors is then refused by the unclaimed-tensor check rather than
       quietly run without them. */
    c->gated_residual = lz_json_get_int(&j, tc, "gated_residual", 0);

    /* S4 MLA low-rank Q/O (model.h's own field comment) - 0/0 defaults
       reproduce a config.json that never carried these keys. */
    c->mla_q_lora_rank = lz_json_get_int(&j, tc, "mla_q_lora_rank", 0);
    c->mla_o_lora_rank = lz_json_get_int(&j, tc, "mla_o_lora_rank", 0);

    /* Engram (conditional n-gram memory). engram=0 skips everything;
       derive the structural fields the forward pass reads. */
    c->engram          = lz_json_get_int(&j, tc, "engram", 0);
    if (c->engram) {
        c->engram_ngram    = lz_json_get_int(&j, tc, "engram_ngram", 3);
        c->engram_n_head   = lz_json_get_int(&j, tc, "engram_n_head", 4);
        c->engram_dim      = lz_json_get_int(&j, tc, "engram_dim", 64);
        c->engram_size     = lz_json_get_int(&j, tc, "engram_size", 1 << 22);
        c->engram_n_orders = c->engram_ngram - 1;
        c->engram_n_spaces = c->engram_n_orders * c->engram_n_head;
        c->engram_per_order = c->engram_n_head * c->engram_dim;
        c->engram_conv_dim  = c->hidden_size;
        c->engram_conv_kernel   = 4;
        c->engram_conv_dilation = c->engram_ngram;
        /* head_space: the Python model uses
           max(256, hash_budget // (n_spaces * engram_dim)) */
        c->engram_head_space = c->engram_size / (c->engram_n_spaces * c->engram_dim);
        if (c->engram_head_space < 256) c->engram_head_space = 256;
        /* engram_total_rows: sum of the first n_spaces primes downward
           from head_space.  Computed here so model_walk can use it
           before alloc_engram runs. */
        {   int total = 0, found = 0, x = c->engram_head_space;
            while (found < c->engram_n_spaces) {
                int is_prime = (x >= 2), d;
                for (d = 2; is_prime && d * d <= x; d++)
                    if (x % d == 0) is_prime = 0;
                if (is_prime) { total += x; found++; }
                x--;
            }
            c->engram_total_rows = total;
        }
        /* engram_format_ids: kunmoe_modeling.py's EngramMemory zeroes
           `value` before the conv wherever the whole n-gram window is
           format tokens (punctuation/template scaffolding), so no
           gradient reaches those rows during training. Skipping the
           same mask at inference computes a DIFFERENT function than
           the one that was trained - a non-empty engram contribution
           at positions training made zero. Built as a bitmap for O(1)
           membership (mirrors torch.isin) rather than kept as a list. */
        {
            const LZJsonNode *fmt = lz_json_get(&j, tc, "engram_format_ids");
            if (fmt && fmt->type == LZ_JSON_ARR && fmt->n_children > 0
                && c->vocab_size > 0) {
                size_t nbytes = (size_t)(c->vocab_size + 7) / 8;
                unsigned char *bm = (unsigned char *)calloc(nbytes, 1);
                if (bm) {
                    const LZJsonNode *e;
                    for (e = lz_json_first(&j, fmt); e; e = lz_json_next(&j, e)) {
                        int id = (int)e->num;
                        if (id >= 0 && id < c->vocab_size)
                            bm[id >> 3] |= (unsigned char)(1u << (id & 7));
                    }
                    c->engram_format_bitmap = bm;
                }
            }
        }
    }
    c->seq_len = lz_json_get_int(&j, tc, "max_position_embeddings", 0);
    c->rms_norm_eps      = (float)lz_json_get_num(&j, tc, "rms_norm_eps", 1e-6);
    c->full_attention_interval =
        lz_json_get_int(&j, tc, "full_attention_interval", 0);

    /* "" (key absent) and "none" both mean LZ_FFN_ACT_NONE inside
       parse_ffn_activation - stock upstream and every model this
       project has produced omit the key entirely, so "absent" cannot be
       the refusal case an unrecognized name gets. */
    if (parse_ffn_activation(&j, tc, c, errbuf, errlen) != 0) {
        lz_json_free(&j);
        return 1;
    }

    {
        const LZJsonNode *n = lz_json_get(&j, tc, "tie_word_embeddings");
        if (!n) n = lz_json_get(&j, root, "tie_word_embeddings");
        c->tie_word_embeddings = (n && n->type == LZ_JSON_BOOL && n->num != 0.0);
    }
    {
        const LZJsonNode *n = lz_json_get(&j, tc, "attn_output_gate");
        c->attn_output_gate = (n && n->type == LZ_JSON_BOOL && n->num != 0.0);
    }
    c->use_subn = lz_json_get_bool(&j, tc, "use_subn", 0);
    /* LowRankLinear does not compose with SubLN's per-projection input
       norm - see model.h's mla_q_lora_rank comment. Refused rather than
       silently picking one, same posture as the rope_theta top-level-
       vs-nested collision check above. */
    if (c->use_subn && (c->mla_q_lora_rank > 0 || c->mla_o_lora_rank > 0)) {
        qerr(errbuf, errlen, LZ_ERR_CFG_MLA_SUBN);
        lz_json_free(&j);
        return 1;
    }
    /* Block-diagonal Hadamard block sizes. -1 = "derive" (resolved by
       resolve_hadamard below), which is what a config.json written
       before these keys existed gets. */
    c->hadamard_o    = lz_json_get_int(&j, tc, "hadamard_o", -1);
    c->hadamard_down = lz_json_get_int(&j, tc, "hadamard_down", -1);

    /* KDA gate (only meaningful when is_kunmoe; harmless zero otherwise).
       kda_gate_rank mirrors KunMoeConfig's "field or head_k_dim" default -
       0/absent means "use lin_k_head_dim", matching
       `self.gate_rank = config.kda_gate_rank or self.head_k_dim`. */
    c->kda_gate_rank = lz_json_get_int(&j, tc, "kda_gate_rank", 0);
    if (is_kunmoe && c->kda_gate_rank <= 0) c->kda_gate_rank = c->lin_k_head_dim;
    {
        /* Present-and-non-null vs absent-or-null must be distinguished:
           None (the default, no clamp) round-trips through JSON as an
           explicit `null`, not a missing key - KunMoeConfig always
           writes the key. lz_json_get_int/num both fold "missing" and
           "null" into the same default, so read the node directly. */
        const LZJsonNode *n = lz_json_get(&j, tc, "kda_gate_lower_bound");
        c->kda_has_gate_lower_bound = (n && n->type == LZ_JSON_NUM);
        c->kda_gate_lower_bound = c->kda_has_gate_lower_bound ? (float)n->num : 0.0f;
    }
    if (is_kunmoe) {
        const char *act = lz_json_get_str(&j, tc, "kda_gate_activation", "silu");
        if (strcmp(act, "silu") != 0) {
            /* The engine's gated RMSNorm (lz_rmsnorm_gated, shared with
               GDN) hardcodes silu, same as Qwen3_5RMSNormGated. A
               sigmoid-gated checkpoint (released Kimi-Linear's own
               default) would silently compute the wrong gate with no
               error - refuse instead. */
            qerr(errbuf, errlen, LZ_ERR_CFG_KDA_ACT, act);
            lz_json_free(&j);
            return 1;
        }
    }

    /* latent MoE (KunMoEGate / LatentMoE). num_experts == 0 (the default
       here, NOT KunMoeConfig's Python default of 8) means "no MoE in
       this checkpoint" - every layer keeps the classic dense FFN. A real
       kunmoe config.json always writes num_experts explicitly (HF
       configs serialize every attribute), so the 0 fallback only bites
       on a hand-written or non-kunmoe config, where "no MoE" is
       correct. */
    c->num_experts           = lz_json_get_int(&j, tc, "num_experts", 0);
    c->num_experts_per_token = lz_json_get_int(&j, tc, "num_experts_per_token", 0);
    c->num_shared_experts    = lz_json_get_int(&j, tc, "num_shared_experts", 0);
    c->moe_intermediate_size = lz_json_get_int(&j, tc, "moe_intermediate_size", 0);
    c->moe_latent_dim        = lz_json_get_int(&j, tc, "routed_expert_hidden_size", 0);
    c->first_k_dense_replace = lz_json_get_int(&j, tc, "first_k_dense_replace", 0);
    {
        const LZJsonNode *n = lz_json_get(&j, tc, "moe_renormalize");
        c->moe_renormalize = !n || (n->type == LZ_JSON_BOOL && n->num != 0.0);
    }
    c->moe_router_sigmoid =
        strcmp(lz_json_get_str(&j, tc, "moe_router_activation_func", "sigmoid"),
              "sigmoid") == 0;
    {
        const LZJsonNode *n = lz_json_get(&j, tc, "latent_moe_use_norm");
        c->moe_latent_use_norm = (n && n->type == LZ_JSON_BOOL && n->num != 0.0);
    }
    /* moe_shared_width: mirrors KunMoeConfig's derivation exactly (see
       LatentMoE.__init__) so config.json and the bin header agree on one
       resolved number instead of each re-deriving it. shared_explicit
       uses -1 as "absent/null", same reasoning as kda_gate_lower_bound
       above, because 0 is a legal explicit width. */
    {
        const LZJsonNode *n = lz_json_get(&j, tc, "shared_expert_intermediate_size");
        int shared_explicit = (n && n->type == LZ_JSON_NUM) ? (int)n->num : -1;
        if (shared_explicit >= 0) {
            c->moe_shared_width = c->num_shared_experts ? shared_explicit : 0;
        } else {
            c->moe_shared_width = c->moe_intermediate_size *
                                  (c->num_shared_experts > 0 ? c->num_shared_experts : 0);
        }
    }
    if (c->num_experts < 0 || c->num_experts > LZ_MAX_EXPERTS ||
        c->num_experts_per_token < 0 ||
        c->num_experts_per_token > c->num_experts ||
        c->first_k_dense_replace < 0) {
        qerr(errbuf, errlen, LZ_ERR_CFG_MOE_FIELDS,
             c->num_experts, c->num_experts_per_token, c->first_k_dense_replace);
        lz_json_free(&j);
        return 1;
    }

    /* rope_theta and partial_rotary_factor live under the nested
       rope_parameters; top-level same-name keys may be absent. Check
       both.

       A KEY PRESENT IN BOTH PLACES WITH TWO DIFFERENT VALUES IS
       REFUSED, not resolved. transformers resolves it the other way
       round - rope_parameters wins there - so any config carrying both
       trains under one value and infers under the other. Measured on
       kunkun-ce, whose top level said 0.25 while rope_parameters said
       1.0: training rotated 64 of 64 dims and this loader rotated 16.
       Nothing downstream reports that; the model simply has a different
       positional encoding at inference than the one it learned. The
       generator's intent (ce_carve.py's DST) was 1.0, so picking either
       side silently would also have been picking against the intent
       half the time. */
    c->rope_theta = 10000.0f;
    c->partial_rotary_factor = 1.0f;
    rp = lz_json_get(&j, tc, "rope_parameters");
    if (rp) {
        const LZJsonNode *n_rp_theta = lz_json_get(&j, rp, "rope_theta");
        const LZJsonNode *n_rp_prf =
            lz_json_get(&j, rp, "partial_rotary_factor");
        const LZJsonNode *n_top_theta = lz_json_get(&j, tc, "rope_theta");
        const LZJsonNode *n_top_prf =
            lz_json_get(&j, tc, "partial_rotary_factor");
        if (n_rp_theta && n_top_theta &&
            n_rp_theta->num != n_top_theta->num) {
            qerr(errbuf, errlen, LZ_ERR_CFG_ROPE_CONFLICT,
                 "rope_theta", (int)n_rp_theta->num, "top-level",
                 (int)n_top_theta->num);
            return 1;
        }
        if (n_rp_prf && n_top_prf &&
            n_rp_prf->num != n_top_prf->num) {
            /* Scaled by 1000 so a fraction survives the integer-only
               error format; 0.25 reads as 250. */
            qerr(errbuf, errlen, LZ_ERR_CFG_ROPE_CONFLICT,
                 "partial_rotary_factor x1000",
                 (int)(n_rp_prf->num * 1000.0), "top-level",
                 (int)(n_top_prf->num * 1000.0));
            return 1;
        }
        c->rope_theta = (float)lz_json_get_num(&j, rp, "rope_theta",
                                               c->rope_theta);
        c->partial_rotary_factor =
            (float)lz_json_get_num(&j, rp, "partial_rotary_factor",
                                   c->partial_rotary_factor);
    }
    c->rope_theta = (float)lz_json_get_num(&j, tc, "rope_theta", c->rope_theta);
    c->partial_rotary_factor =
        (float)lz_json_get_num(&j, tc, "partial_rotary_factor",
                               c->partial_rotary_factor);

    /* rope_parameters.rope_type: none/default/linear/yarn, plus
       factor, original_max_position_embeddings, beta_fast, beta_slow,
       attention_factor - all FLAT siblings of rope_theta inside
       rope_parameters, not nested under a "rope_scaling" sub-object.
       There is no such sub-object in what this project's own training
       side ever writes: checked against KunMoeConfig.to_dict() and
       every live config.json under E:/LLM/models, rope_parameters is
       {rope_theta, partial_rotary_factor, rope_type, and - only when
       non-default - factor/original_max_position_embeddings/
       beta_fast/beta_slow}. A prior version of this parser read
       rope_parameters.rope_scaling.type, matching an older nested
       transformers convention this project's config.json never
       actually produced - every field below was silently unreachable
       against every real config on disk, and rope_scaling_type stayed
       0 regardless of what rope_type said; the RoPE table this loader
       feeds never applied any of it either (see forward_rope.c's rope_cs
       fill). Defaults: no scaling. rope_scaling_type stays 0 unless
       the key exists AND names a supported type - an unsupported type
       name is refused, not silently read as "no scaling". */
    c->rope_scaling_type = 0;
    c->rope_scaling_factor = 1.0f;
    c->rope_scaling_orig_max = 0.0f;
    c->rope_scaling_beta_fast = 32.0f;
    c->rope_scaling_beta_slow = 1.0f;
    c->rope_scaling_attn_factor = 0.0f;   /* 0 = absent, forward.c derives it */
    if (rp) {
        const LZJsonNode *ty = lz_json_get(&j, rp, "rope_type");
        const char *s = ty && ty->type == LZ_JSON_STR ? ty->text : NULL;
        const LZJsonNode *n_factor = lz_json_get(&j, rp, "factor");
        const LZJsonNode *n_orig_max =
            lz_json_get(&j, rp, "original_max_position_embeddings");
        int t = 0;
        if (s && strcmp(s, "linear") == 0) t = 1;
        else if (s && strcmp(s, "yarn") == 0) t = 2;
        else if (s && strcmp(s, "default") != 0 && strcmp(s, "none") != 0 &&
                 strcmp(s, "null") != 0) {
            qerr(errbuf, errlen, LZ_ERR_CFG_ROPE_SCALING, s ? s : "?");
            lz_json_free(&j);
            return 1;
        }
        /* yarn's correction range divides by log(orig_max / ...) and by
           log(base) - an absent orig_max (sentinel 0) or a factor left
           at the "no scaling" sentinel (1.0) would compute a division
           by an effectively-undefined range rather than a refusal.
           Required explicit, same posture as LZ_ERR_CFG_MOE_FIELDS:
           silently deriving factor from max_position_embeddings /
           original_max_position_embeddings (as transformers does when
           factor is omitted) is a second code path this project does
           not carry, not a default worth guessing at. */
        if ((t == 2 && (!n_factor || !n_orig_max)) || (t == 1 && !n_factor)) {
            qerr(errbuf, errlen, LZ_ERR_CFG_ROPE_YARN_FIELDS,
                 (int)((n_factor ? n_factor->num : 0.0) * 1000.0),
                 (int)(n_orig_max ? n_orig_max->num : 0.0));
            lz_json_free(&j);
            return 1;
        }
        c->rope_scaling_type = t;
        c->rope_scaling_factor =
            (float)lz_json_get_num(&j, rp, "factor", c->rope_scaling_factor);
        c->rope_scaling_orig_max =
            (float)lz_json_get_num(&j, rp,
                                   "original_max_position_embeddings",
                                   c->rope_scaling_orig_max);
        c->rope_scaling_beta_fast =
            (float)lz_json_get_num(&j, rp, "beta_fast",
                                   c->rope_scaling_beta_fast);
        c->rope_scaling_beta_slow =
            (float)lz_json_get_num(&j, rp, "beta_slow",
                                   c->rope_scaling_beta_slow);
        c->rope_scaling_attn_factor =
            (float)lz_json_get_num(&j, rp, "attention_factor",
                                   c->rope_scaling_attn_factor);

        /* Keys transformers reads under rope_parameters that this engine
           does NOT implement. Ignoring one silently would run a rule the
           checkpoint was not trained under, and nothing downstream would
           say so - the posture the rope_theta/partial_rotary_factor
           conflict check above and the unknown rope_type check below
           already take. `truncate: false` drops yarn's floor/ceil on the
           correction range; mscale / mscale_all_dim replace the
           one-argument get_mscale(factor) with a two-argument ratio. No
           config.json on this machine carries any of the three. */
        {
            static const char *unsupported[] = {
                "truncate", "mscale", "mscale_all_dim"
            };
            int ui;
            for (ui = 0; ui < 3; ui++) {
                if (lz_json_get(&j, rp, unsupported[ui])) {
                    qerr(errbuf, errlen, LZ_ERR_CFG_ROPE_KEY,
                         unsupported[ui]);
                    lz_json_free(&j);
                    return 1;
                }
            }
        }
    }

    /* layer_types decides per-layer SSM vs attention; without it loading is impossible */
    lt = lz_json_get(&j, tc, "layer_types");
    if (!lt || lt->type != LZ_JSON_ARR) {
        qerr(errbuf, errlen, LZ_ERR_CFG_LAYER_TYPES);
        lz_json_free(&j);
        return 1;
    }
    if (lt->n_children != c->n_layers) {
        qerr(errbuf, errlen, LZ_ERR_CFG_LAYER_LEN,
             lt->n_children, c->n_layers);
        lz_json_free(&j);
        return 1;
    }
    if (c->n_layers <= 0 || c->n_layers > LZ_MAX_LAYERS) {
        qerr(errbuf, errlen, LZ_ERR_CFG_LAYER_RANGE,
             c->n_layers, LZ_MAX_LAYERS);
        lz_json_free(&j);
        return 1;
    }
    i = 0;
    for (e = lz_json_first(&j, lt); e; e = lz_json_next(&j, e), i++) {
        if (lz_json_str_eq(e, "full_attention")) {
            c->layer_types[i] = LZ_LT_FULL;
            c->n_full_layers++;
        } else if (lz_json_str_eq(e, "linear_attention")) {
            /* is_kunmoe decides GDN vs KDA - see the detection comment
               above. Either way this is a state-bearing "linear" layer
               for n_linear_layers/cache_idx purposes (see forward.c and
               lz_state_alloc), which is why both map into the same
               counter unchanged. */
            c->layer_types[i] = is_kunmoe ? LZ_LT_KDA : LZ_LT_LINEAR;
            c->n_linear_layers++;
        } else {
            qerr(errbuf, errlen, LZ_ERR_CFG_LAYER_TYPE, i,
                 e->text ? e->text : "(not-a-string)");
            lz_json_free(&j);
            return 1;
        }
    }

    /* S2 CSA2 (model.h's own csa2_on[]/csa2_compress_m comment). Parsed
       here, not earlier alongside ced_split/mla_q_lora_rank, because
       validating each named index against layer_types needs that array
       already filled - the loop just above is what fills it, and `j`
       (this function's own JSON handle) is freed a few lines down, so
       this is the only window where both are available at once. */
    lt = lz_json_get(&j, tc, "csa2_layers");
    if (lt && lt->type == LZ_JSON_ARR) {
        for (e = lz_json_first(&j, lt); e; e = lz_json_next(&j, e)) {
            int idx = (int)e->num;
            if (idx < 0 || idx >= c->n_layers ||
                c->layer_types[idx] != LZ_LT_FULL) {
                qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_LAYER, idx);
                lz_json_free(&j);
                return 1;
            }
            /* S4+CSA2 composition (model.h's own csa2_w_dq_idx comment;
               model_walk's own CSA2_SPECS comment has the mechanism):
               present_mla_q/present_not_mla_q gate q_proj.weight vs
               q_proj.down/up.weight PURELY on c->mla_q_lora_rank, for
               EVERY full_attention layer, csa2_on or not - so a csa2_on
               layer under S4 never gets q_proj.weight walked at all,
               while forward_csa2.c's step 1 always reads L->q_proj
               directly and never L->q_proj_down/up. Refused explicitly,
               here, rather than left to an accidental missing-tensor
               error on csa2_w_dq_idx (which only fires for a checkpoint
               that ALSO reproduces the training side's own composition
               choice of dropping that tensor - see model_walk's own
               comment for why that accident cannot be relied on at all
               on the bin path, which has no name to fail a lookup
               against). mla_q_lora_rank is parsed once, unconditionally,
               well before this loop runs (this function's own field
               order), so it is already final here. */
            if (c->mla_q_lora_rank > 0) {
                qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_MLA, idx);
                lz_json_free(&j);
                return 1;
            }
            c->csa2_on[idx] = 1;
        }
    }

    /* csa2_kv_mode / csa2_kv_source (model.h's own csa2_mode[]/
       csa2_source[] comment on LZModelConfig). Parsed here, right after
       csa2_on[] is filled, for the same reason csa2_layers itself is
       parsed here rather than alongside ced_split/mla_q_lora_rank above:
       validating an entry needs csa2_on[] already filled.

       c->csa2_mode[] is already LZ_CSA2_FULL (0) for every layer via
       lz_load_config's own memset(c, 0, ...) at entry - LZ_CSA2_FULL is
       0 for exactly this reason (model.h's own comment on the code).
       csa2_source[] needs an explicit fill: -1 is not memset's zero. */
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->csa2_source[i] = -1;

    /* Both are JSON OBJECTS keyed by the layer index AS A STRING (JSON
       has no integer keys) - walked the same way lz_json_first/
       lz_json_next walk an ARR's elements (st_path_from_index's
       weight_map loop and arch_probe's tensor-name loop, both above in
       this file, are the existing precedent for objects; this is the
       first object here keyed by something other than a fixed literal
       name). Each child node's own ->key is the member name, NUL-
       terminated (json.h's own LZJsonNode comment) - csa2_key_idx above
       is what rejects a key that is not ENTIRELY an in-range digit
       string, rather than silently reading a numeric prefix or letting
       strtol's own leading-whitespace/sign tolerance through. */
    lt = lz_json_get(&j, tc, "csa2_kv_mode");
    if (lt && lt->type == LZ_JSON_OBJ) {
        for (e = lz_json_first(&j, lt); e; e = lz_json_next(&j, e)) {
            int idx = csa2_key_idx(e->key);
            if (idx < 0 || idx >= c->n_layers || !c->csa2_on[idx]) {
                qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_MODE,
                     idx, "not a CSA2 layer");
                lz_json_free(&j);
                return 1;
            }
            if (lz_json_str_eq(e, "full")) {
                c->csa2_mode[idx] = LZ_CSA2_FULL;
            } else if (lz_json_str_eq(e, "reindex")) {
                c->csa2_mode[idx] = LZ_CSA2_REINDEX;
            } else if (lz_json_str_eq(e, "reuse")) {
                c->csa2_mode[idx] = LZ_CSA2_REUSE;
            } else {
                qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_MODE, idx,
                     e->type == LZ_JSON_STR && e->text ?
                     e->text : "(not-a-string)");
                lz_json_free(&j);
                return 1;
            }
        }
    }
    lt = lz_json_get(&j, tc, "csa2_kv_source");
    if (lt && lt->type == LZ_JSON_OBJ) {
        for (e = lz_json_first(&j, lt); e; e = lz_json_next(&j, e)) {
            int idx = csa2_key_idx(e->key);
            int src;
            /* idx must already have been resolved to a non-FULL mode by
               the csa2_kv_mode pass just above - a FULL-mode (explicit
               or the untouched default) idx getting a source entry is
               refused here, same as an out-of-range/non-CSA2 idx,
               because neither can have "a strictly earlier layer this
               entry aliases" mean anything. Bounds-checked before the
               csa2_mode[] read: an unresolved idx outside [0, n_layers)
               would otherwise index past the array. */
            if (idx < 0 || idx >= c->n_layers ||
                c->csa2_mode[idx] == LZ_CSA2_FULL) {
                qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_SOURCE, idx,
                     "mode is full or was never set to reindex/reuse");
                lz_json_free(&j);
                return 1;
            }
            /* The value's own type: jnode_new zeroes every new node, so
               a string/bool/null/array/object entry's ->num reads back
               0.0 (or 1.0 for `true`) exactly like a legitimate integer
               0/1 would - e->num alone cannot tell "the layer 0 source"
               from "this was never a number". Checked before reading
               ->num for that reason, not after. */
            if (e->type != LZ_JSON_NUM) {
                qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_SOURCE, idx,
                     "source is not a number");
                lz_json_free(&j);
                return 1;
            }
            src = (int)e->num;
            if (src < 0 || src >= idx || !c->csa2_on[src]) {
                qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_SOURCE, idx,
                     "source is not a strictly earlier CSA2 layer");
                lz_json_free(&j);
                return 1;
            }
            c->csa2_source[idx] = src;
        }
    }
    /* Cross-checks that need BOTH objects fully parsed: a non-FULL mode
       with no source (the csa2_kv_source pass above only ever fills a
       source for an idx it saw; an idx named only in csa2_kv_mode never
       reaches it), and - once every layer has a source at all -
       lz_csa2_last_writer_valid's own whole-config question: is each
       one's declared source actually the CURRENT last writer of the
       physical slot it aliases, not merely a strictly-earlier csa2_on
       layer. That single check subsumes the narrower per-layer one, so
       this loop does not check it ("a REUSE layer's source must not
       itself be REUSE mode" - model.c's own comment on
       lz_csa2_last_writer_valid has the proof: a REUSE layer never
       becomes anyone's last writer, so any layer naming one as its own
       source always fails the same way). */
    for (i = 0; i < c->n_layers; i++) {
        if (!c->csa2_on[i] || c->csa2_mode[i] == LZ_CSA2_FULL) continue;
        if (c->csa2_source[i] < 0) {
            qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_SOURCE, i,
                 "csa2_kv_mode names a non-full mode with no csa2_kv_source entry");
            lz_json_free(&j);
            return 1;
        }
    }
    {
        int bad = -1;
        if (!lz_csa2_last_writer_valid(c, &bad)) {
            qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_SOURCE, bad,
                 "source is not the current last writer of the shared "
                 "slot it aliases");
            lz_json_free(&j);
            return 1;
        }
    }

    c->csa2_compress_m          = lz_json_get_int(&j, tc, "csa2_compress_m", 2);
    c->csa2_topk                = lz_json_get_int(&j, tc, "csa2_topk", 64);
    c->csa2_indexer_heads       = lz_json_get_int(&j, tc, "csa2_indexer_heads", 4);
    c->csa2_indexer_head_dim    = lz_json_get_int(&j, tc, "csa2_indexer_head_dim", 32);
    c->csa2_indexer_q_lora_rank = lz_json_get_int(&j, tc, "csa2_indexer_q_lora_rank", 32);

    lz_json_free(&j);

    /* derived quantities computed once, here */
    c->attn_q_dim     = c->n_heads * c->head_dim;
    c->attn_qgate_dim = c->attn_output_gate ? c->attn_q_dim * 2 : c->attn_q_dim;
    c->attn_kv_dim    = c->n_kv_heads * c->head_dim;
    c->lin_key_dim    = c->lin_n_k_heads * c->lin_k_head_dim;
    c->lin_value_dim  = c->lin_n_v_heads * c->lin_v_head_dim;
    c->lin_conv_dim   = c->lin_key_dim * 2 + c->lin_value_dim;
    c->rotary_dim     = (int)((float)c->head_dim * c->partial_rotary_factor);

    if (c->vocab_size <= 0 || c->hidden_size <= 0 || c->head_dim <= 0 ||
        c->intermediate_size <= 0) {
        qerr(errbuf, errlen, LZ_ERR_CFG_FIELDS,
             c->vocab_size, c->hidden_size, c->head_dim, c->intermediate_size);
        return 1;
    }
    if (c->n_heads <= 0 || c->n_kv_heads <= 0 ||
        c->n_heads % c->n_kv_heads != 0) {
        qerr(errbuf, errlen, LZ_ERR_CFG_HEADS,
             c->n_heads, c->n_kv_heads);
        return 1;
    }
    if (c->rotary_dim <= 0 || c->rotary_dim > c->head_dim ||
        c->rotary_dim % 2 != 0) {
        qerr(errbuf, errlen, LZ_ERR_CFG_ROTARY,
             c->rotary_dim, c->head_dim,
             (int)(c->partial_rotary_factor * 100.0f + 0.5f));
        return 1;
    }

    /* lz_rope writes base[i] and base[i + rotary_dim/2] inside one row
       (src/ops_rope.c), so a row must be at least rotary_dim wide.
       head_dim is not the tightest row this model rotates: a FULL-mode
       CSA2 layer's indexer key is csa2_indexer_head_dim wide, and the
       rotation forward_csa2.c applies to it is this same lz_rope - so
       the bound above lets rotary_dim run off the end of that row, into
       the next block's entry and past the allocation. Refused here
       rather than silently narrowed, because narrowing it would rotate
       fewer dims than the caller asked for and report nothing.

       Gated on csa2_on[] and on FULL mode for two separate reasons:
       only a FULL layer owns an indexer key (REINDEX/REUSE read their
       source's), and csa2_indexer_head_dim carries a default (32) that
       every checkpoint gets whether or not it has any CSA2 layer at all.
       engine/csa2.py's _wire_csa2 refuses the same pairing on the
       training side. */
    for (i = 0; i < c->n_layers; i++) {
        if (!c->csa2_on[i] || c->csa2_mode[i] != LZ_CSA2_FULL) continue;
        if (c->rotary_dim > c->csa2_indexer_head_dim) {
            qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_IDX_ROTARY,
                 i, c->rotary_dim, c->csa2_indexer_head_dim);
            return 1;
        }
    }
    /* After lin_value_dim, which the derivations read. */
    if (check_attn_gate(c, errbuf, errlen) != 0) return 1;
    if (resolve_hadamard(c, errbuf, errlen) != 0) return 1;
    return 0;
}

/* ------------------------------------------------------ tensor spec table */

/* Expected shapes use symbolic dims so an architecture change makes
   the error point at exactly which tensor/dim */
enum {
    DK_END = 0,
    DK_HIDDEN, DK_INTER, DK_VOCAB,
    DK_ATTN_QGATE, DK_ATTN_KV, DK_ATTN_Q, DK_HEAD_DIM,
    /* n_heads: the width of a per-head vector. NOT DK_NVHEADS below,
       which is lin_n_v_heads - the linear-attention side's head count is
       an independent config field. */
    DK_ATTN_HEADS,
    DK_CONV, DK_VALUE, DK_NVHEADS, DK_VHEADDIM,
    DK_ONE, DK_KERNEL,
    DK_HIDDEN2,     /* 2 * hidden: mtp.fc reads [hidden, embedding] joined */
    DK_HIDDEN3,     /* 3 * hidden: mtp.main_proj (DSpark, mtp_n_draft > 1) */
    DK_MTP_NDRAFT,  /* mtp_n_draft: mtp.depth_embed row count */
    DK_MTP_MARKOV_RANK, /* mtp_markov_rank: markov_embed/markov_proj's rank dim */
    DK_MLA_Q_RANK,  /* mla_q_lora_rank: q_proj.down output / q_proj.up input */
    DK_MLA_O_RANK,  /* mla_o_lora_rank: o_proj.down output / o_proj.up input */
    /* S2 CSA2 (LZLayer's own csa2_w_* comment has the full tensor list;
       csa2_w_kv/csa2_w_gate's own output dim is DK_HEAD_DIM, reused
       rather than duplicated - CSA2's compressed entries are exactly
       head_dim wide, arXiv:2606.19348's own text). */
    DK_CSA2_IDX_HEAD_DIM,  /* csa2_indexer_head_dim: indexer K/Q per-head width */
    DK_CSA2_IDX_HEADS,     /* csa2_indexer_heads: csa2_w_weights output width */
    DK_CSA2_Q_LORA_RANK,   /* csa2_indexer_q_lora_rank: csa2_w_dq_idx output / csa2_w_uq_idx input */
    DK_CSA2_UQ_OUT,        /* csa2_indexer_heads * csa2_indexer_head_dim: csa2_w_uq_idx output */
    /* KDA (LZ_LT_KDA) */
    DK_KEY,         /* lin_key_dim: kda_q_proj/kda_k_proj output width */
    DK_GATE_RANK,   /* kda_gate_rank: f_a_proj output / f_b_proj input */
    DK_NVK,         /* lin_n_v_heads * lin_k_head_dim: f_b_proj output, dt_bias */
    DK_ENGRAM_ROWS, /* engram_total_rows: number of n-gram hash table rows */
    DK_ENGRAM_DIM,  /* engram_dim: embedding dimension per hash head */
    DK_ENGRAM_PER_ORDER, /* engram_per_order: n_head * engram_dim, key/value width */
    /* latent MoE */
    DK_NUM_EXPERTS, /* num_experts: router weight rows / bias width */
    DK_LATENT,      /* moe_latent_dim: routed_expert_down/up_proj's latent side */
    DK_SHARED_W,    /* moe_shared_width: shared expert inner width */
    DK_MOE_INTER,   /* moe_intermediate_size: per-expert w1/w2/w3 inner width */
    /* GR (LZLayer's own gr_attn / gr_mlp comment has the tensor list).
       DK_GR_FLAT is nr*hidden, the flattened branch stream every GR
       projection reads or writes - it is why the three are separate
       kinds rather than shares of DK_HIDDEN. */
    DK_GR_NR,       /* gated_residual: branch count, one norm per branch */
    DK_GR_RANK,     /* max(1, hidden/8): W_d's output / W_u's input */
    DK_GR_FLAT,     /* gated_residual * hidden: the flattened stream width */
    /* MTP head only (model_walk's separate MTP_FFN_SPECS, not
       LAYER_SPECS - see that table's own comment) */
    DK_MTP_INTER    /* mtp_intermediate_size: MTP block's own dense FFN width */
};

/* present() lets a spec depend on more than the two axes below (moe
   shared expert / latent norm are OPTIONAL even on a MoE layer, gated by
   a global config flag rather than by layer_type or ffn_kind). NULL
   means "always present given layer_type/ffn_kind already matched". */
typedef int (*LZPresentFn)(const LZModelConfig *c);

typedef struct {
    const char *suffix;
    int layer_type;          /* -1 = any attention kind; else LZ_LT_* */
    int ffn_kind;             /* -1 = any; 0 = dense FFN only; 1 = MoE FFN only */
    size_t field;            /* offsetof(LZLayer, field) */
    int dims[3];
    LZPresentFn present;
} LZLayerSpec;

#define LOFF(f) offsetof(LZLayer, f)

static int present_moe_shared(const LZModelConfig *c) { return c->moe_shared_width > 0; }
static int present_moe_latent_norm(const LZModelConfig *c) { return c->moe_latent_use_norm; }
static int present_subn(const LZModelConfig *c) { return c->use_subn; }
static int present_subn_shared(const LZModelConfig *c) {
    return c->use_subn && c->moe_shared_width > 0;
}
static int present_mtp_ndraft1(const LZModelConfig *c) { return c->mtp_n_draft <= 1; }
static int present_mtp_ndraft_gt1(const LZModelConfig *c) { return c->mtp_n_draft > 1; }
static int present_mla_q(const LZModelConfig *c) { return c->mla_q_lora_rank > 0; }
static int present_not_mla_q(const LZModelConfig *c) { return c->mla_q_lora_rank <= 0; }
static int present_mla_o(const LZModelConfig *c) { return c->mla_o_lora_rank > 0; }
static int present_not_mla_o(const LZModelConfig *c) { return c->mla_o_lora_rank <= 0; }

static const LZLayerSpec LAYER_SPECS[] = {
    /* shared by every layer kind and every FFN kind */
    { "input_layernorm.weight",          -1, -1, LOFF(input_layernorm),
      { DK_HIDDEN, DK_END, DK_END }, NULL },
    { "post_attention_layernorm.weight", -1, -1, LOFF(post_attention_layernorm),
      { DK_HIDDEN, DK_END, DK_END }, NULL },

    /* dense FFN only (ffn_kind 0 - i.e. NOT a latent-MoE layer) */
    { "mlp.gate_proj.weight",            -1, 0, LOFF(gate_proj),
      { DK_INTER, DK_HIDDEN, DK_END }, NULL },
    { "mlp.up_proj.weight",              -1, 0, LOFF(up_proj),
      { DK_INTER, DK_HIDDEN, DK_END }, NULL },
    { "mlp.down_proj.weight",            -1, 0, LOFF(down_proj),
      { DK_HIDDEN, DK_INTER, DK_END }, NULL },

    /* SubLN (use_subn=1 only): dense-FFN per-projection input RMSNorms. */
    { "mlp.gate_proj.norm.weight",       -1, 0, LOFF(gate_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "mlp.up_proj.norm.weight",         -1, 0, LOFF(up_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "mlp.down_proj.norm.weight",       -1, 0, LOFF(down_norm),
      { DK_INTER, DK_END, DK_END }, present_subn },

    /* full_attention only. q_proj is 2x wide; the second half is the gate */
    { "self_attn.q_proj.weight", LZ_LT_FULL, -1, LOFF(q_proj),
      { DK_ATTN_QGATE, DK_HIDDEN, DK_END }, present_not_mla_q },
    { "self_attn.k_proj.weight", LZ_LT_FULL, -1, LOFF(k_proj),
      { DK_ATTN_KV, DK_HIDDEN, DK_END }, NULL },
    { "self_attn.v_proj.weight", LZ_LT_FULL, -1, LOFF(v_proj),
      { DK_ATTN_KV, DK_HIDDEN, DK_END }, NULL },
    { "self_attn.o_proj.weight", LZ_LT_FULL, -1, LOFF(o_proj),
      { DK_HIDDEN, DK_ATTN_Q, DK_END }, present_not_mla_o },
    /* S4 MLA (mla_q_lora_rank/mla_o_lora_rank > 0 only): replaces the two
       entries above, matching engine/kunmoe_modeling.py's LowRankLinear
       replacing attn.q_proj/attn.o_proj in place - state-dict keys
       q_proj.down.weight/q_proj.up.weight (LowRankLinear.down/.up are
       nn.Linear submodules named that). See model.h's LZLayer comment
       on q_proj_down/q_proj_up/mla_q_on/mla_o_on for the per-layer
       flags attn_impl actually reads, and the mutual exclusion with
       use_subn enforced above (LZ_ERR_CFG_MLA_SUBN). */
    { "self_attn.q_proj.down.weight", LZ_LT_FULL, -1, LOFF(q_proj_down),
      { DK_MLA_Q_RANK, DK_HIDDEN, DK_END }, present_mla_q },
    { "self_attn.q_proj.up.weight", LZ_LT_FULL, -1, LOFF(q_proj_up),
      { DK_ATTN_QGATE, DK_MLA_Q_RANK, DK_END }, present_mla_q },
    { "self_attn.o_proj.down.weight", LZ_LT_FULL, -1, LOFF(o_proj_down),
      { DK_MLA_O_RANK, DK_ATTN_Q, DK_END }, present_mla_o },
    { "self_attn.o_proj.up.weight", LZ_LT_FULL, -1, LOFF(o_proj_up),
      { DK_HIDDEN, DK_MLA_O_RANK, DK_END }, present_mla_o },
    { "self_attn.q_norm.weight", LZ_LT_FULL, -1, LOFF(q_norm),
      { DK_HEAD_DIM, DK_END, DK_END }, NULL },
    { "self_attn.k_norm.weight", LZ_LT_FULL, -1, LOFF(k_norm),
      { DK_HEAD_DIM, DK_END, DK_END }, NULL },

    /* SubLN (use_subn=1 only): full attention's per-projection INPUT
       RMSNorms - not q_norm/k_norm above, which are the per-head QK
       norms over head_dim and are present regardless. */
    { "self_attn.q_proj.norm.weight", LZ_LT_FULL, -1, LOFF(attn_q_subn_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "self_attn.k_proj.norm.weight", LZ_LT_FULL, -1, LOFF(attn_k_subn_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "self_attn.v_proj.norm.weight", LZ_LT_FULL, -1, LOFF(attn_v_subn_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "self_attn.o_proj.norm.weight", LZ_LT_FULL, -1, LOFF(attn_o_subn_norm),
      { DK_ATTN_Q, DK_END, DK_END }, present_subn },

    /* linear_attention / GatedDeltaNet only */
    { "linear_attn.in_proj_qkv.weight", LZ_LT_LINEAR, -1, LOFF(in_proj_qkv),
      { DK_CONV, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.in_proj_z.weight",   LZ_LT_LINEAR, -1, LOFF(in_proj_z),
      { DK_VALUE, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.in_proj_a.weight",   LZ_LT_LINEAR, -1, LOFF(in_proj_a),
      { DK_NVHEADS, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.in_proj_b.weight",   LZ_LT_LINEAR, -1, LOFF(in_proj_b),
      { DK_NVHEADS, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.conv1d.weight",      LZ_LT_LINEAR, -1, LOFF(conv1d),
      { DK_CONV, DK_ONE, DK_KERNEL }, NULL },
    { "linear_attn.A_log",              LZ_LT_LINEAR, -1, LOFF(A_log),
      { DK_NVHEADS, DK_END, DK_END }, NULL },
    { "linear_attn.dt_bias",            LZ_LT_LINEAR, -1, LOFF(dt_bias),
      { DK_NVHEADS, DK_END, DK_END }, NULL },
    { "linear_attn.norm.weight",        LZ_LT_LINEAR, -1, LOFF(ssm_norm),
      { DK_VHEADDIM, DK_END, DK_END }, NULL },
    { "linear_attn.out_proj.weight",    LZ_LT_LINEAR, -1, LOFF(out_proj),
      { DK_HIDDEN, DK_VALUE, DK_END }, NULL },

    /* LZ_LT_KDA only - KdaAttention, tensor
       for tensor. NOT the same fields as GDN above (see model.h). */
    { "linear_attn.q_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_q_proj),
      { DK_KEY, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.k_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_k_proj),
      { DK_KEY, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.v_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_v_proj),
      { DK_VALUE, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.q_conv1d.weight", LZ_LT_KDA, -1, LOFF(kda_q_conv1d),
      { DK_KEY, DK_ONE, DK_KERNEL }, NULL },
    { "linear_attn.k_conv1d.weight", LZ_LT_KDA, -1, LOFF(kda_k_conv1d),
      { DK_KEY, DK_ONE, DK_KERNEL }, NULL },
    { "linear_attn.v_conv1d.weight", LZ_LT_KDA, -1, LOFF(kda_v_conv1d),
      { DK_VALUE, DK_ONE, DK_KERNEL }, NULL },
    { "linear_attn.A_log",           LZ_LT_KDA, -1, LOFF(kda_A_log),
      { DK_NVHEADS, DK_END, DK_END }, NULL },
    { "linear_attn.f_a_proj.weight", LZ_LT_KDA, -1, LOFF(kda_f_a_proj),
      { DK_GATE_RANK, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.f_b_proj.weight", LZ_LT_KDA, -1, LOFF(kda_f_b_proj),
      { DK_NVK, DK_GATE_RANK, DK_END }, NULL },
    { "linear_attn.dt_bias",         LZ_LT_KDA, -1, LOFF(kda_dt_bias),
      { DK_NVK, DK_END, DK_END }, NULL },
    { "linear_attn.b_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_b_proj),
      { DK_NVHEADS, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.g_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_g_proj),
      { DK_VALUE, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.o_norm.weight",   LZ_LT_KDA, -1, LOFF(kda_o_norm),
      { DK_VHEADDIM, DK_END, DK_END }, NULL },
    { "linear_attn.o_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_o_proj),
      { DK_HIDDEN, DK_VALUE, DK_END }, NULL },

    /* SubLN (use_subn=1 only): KDA per-projection input RMSNorms.
       kda_o_subn_norm is o_proj's OWN input norm over lin_value_dim - NOT
       kda_o_norm above (the per-head gated-silu norm). */
    { "linear_attn.q_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_q_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "linear_attn.k_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_k_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "linear_attn.v_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_v_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "linear_attn.g_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_g_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "linear_attn.o_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_o_subn_norm),
      { DK_VALUE, DK_END, DK_END }, present_subn },

    /* latent MoE FFN only (ffn_kind 1). Per-expert w1/w2/w3 are walked
       separately in model_walk - their count is a runtime config value,
       not something this fixed table can express. */
    { "mlp.gate.weight",                       -1, 1, LOFF(moe_gate_w),
      { DK_NUM_EXPERTS, DK_HIDDEN, DK_END }, NULL },
    { "mlp.gate.e_score_correction_bias",      -1, 1, LOFF(moe_gate_bias),
      { DK_NUM_EXPERTS, DK_END, DK_END }, NULL },
    { "mlp.routed_expert_down_proj.weight",    -1, 1, LOFF(moe_down_proj),
      { DK_LATENT, DK_HIDDEN, DK_END }, NULL },
    { "mlp.routed_expert_up_proj.weight",      -1, 1, LOFF(moe_up_proj),
      { DK_HIDDEN, DK_LATENT, DK_END }, NULL },
    { "mlp.routed_expert_norm.weight",         -1, 1, LOFF(moe_latent_norm),
      { DK_LATENT, DK_END, DK_END }, present_moe_latent_norm },
    { "mlp.shared_experts.gate_proj.weight",   -1, 1, LOFF(moe_shared_gate),
      { DK_SHARED_W, DK_HIDDEN, DK_END }, present_moe_shared },
    { "mlp.shared_experts.up_proj.weight",     -1, 1, LOFF(moe_shared_up),
      { DK_SHARED_W, DK_HIDDEN, DK_END }, present_moe_shared },
    { "mlp.shared_experts.down_proj.weight",   -1, 1, LOFF(moe_shared_down),
      { DK_HIDDEN, DK_SHARED_W, DK_END }, present_moe_shared },

    /* SubLN (use_subn=1 only): the latent-MoE block's per-projection
       input RMSNorms. The router carries one because its logits feed a
       softmax - an un-normalized input is a changed temperature, not a
       skipped normalization. */
    { "mlp.gate.norm.weight",                  -1, 1, LOFF(moe_gate_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "mlp.routed_expert_down_proj.norm.weight", -1, 1, LOFF(moe_down_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "mlp.routed_expert_up_proj.norm.weight", -1, 1, LOFF(moe_up_norm),
      { DK_LATENT, DK_END, DK_END }, present_subn },
    { "mlp.shared_experts.gate_proj.norm.weight", -1, 1,
      LOFF(moe_shared_gate_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn_shared },
    { "mlp.shared_experts.up_proj.norm.weight", -1, 1,
      LOFF(moe_shared_up_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn_shared },
    { "mlp.shared_experts.down_proj.norm.weight", -1, 1,
      LOFF(moe_shared_down_norm),
      { DK_SHARED_W, DK_END, DK_END }, present_subn_shared }
};
#define N_LAYER_SPECS ((int)(sizeof(LAYER_SPECS) / sizeof(LAYER_SPECS[0])))

/* ------------------------------------------------------ S1 attn_kv_share */

/* The five tensors a target layer takes from the source layer, as
   LZLayer offsets: the two k/v projection weights, the per-head k_norm,
   and the two SubLN input norms that live inside the projection modules.
   ONE table, read by all four callers below - the walk that must not look
   for them on a target layer, the alias that copies them, lz_free (they
   are the source's storage, freed once) and the parameter count (they are
   one tensor, counted once). Named to match the training side's own list,
   engine/kunmoe_modeling.py's _SHARED_KV_ATTRS, which is what decides
   which keys save_pretrained omits. */
static const size_t KV_SHARE_FIELDS[5] = {
    LOFF(k_proj), LOFF(v_proj), LOFF(k_norm),
    LOFF(attn_k_subn_norm), LOFF(attn_v_subn_norm)
};
#define N_KV_SHARE_FIELDS ((int)(sizeof(KV_SHARE_FIELDS) / sizeof(KV_SHARE_FIELDS[0])))

static int kv_share_field(size_t field) {
    int k;
    for (k = 0; k < N_KV_SHARE_FIELDS; k++)
        if (KV_SHARE_FIELDS[k] == field) return 1;
    return 0;
}

/* Layer index whose K/V the others reuse, or -1 when nothing is shared:
   attn_kv_share off, or fewer than two full_attention layers (the
   training side's own "there is nothing to share" case). Derived from
   the config rather than stored as a per-layer flag, so no second copy
   of this rule can go stale - m->mtp->blk is not in m->layers and is
   never a target, which is the same reason its own k_proj stays bound.
   Public (model.h) because forward.c needs the same answer to know whose
   input reaches a target layer's K/V projections. */
int lz_kv_share_src(const LZModelConfig *c) {
    int i, first = -1, nfull = 0;
    if (!c->attn_kv_share) return -1;
    for (i = 0; i < c->n_layers; i++) {
        if (c->layer_types[i] != LZ_LT_FULL) continue;
        if (first < 0) first = i;
        nfull++;
    }
    return nfull >= 2 ? first : -1;
}

/* 1 when layer li takes its k/v from the shared source. NOT when li
   is inside CED's decoder region (ced_split > 0 && ced_split < n and
   li >= ced_split - the same "CED on" test lz_gr_span elsewhere in
   this file uses): report Sec2.2 Eq(1) is unconditional there, and
   attn_kv_share (a project-invented generalization of the layer-reuse
   idea, not an Eq(1) rule) has no basis to override it. The training
   side's engine/kunmoe_modeling.py applies the same exclusion in
   _wire_attn_kv_share; that tree and this one share no git history, so
   its fix does not reach this file by any mechanism except being
   written again here. */
int lz_kv_share_target(const LZModelConfig *c, int li) {
    int src = lz_kv_share_src(c);
    if (src < 0 || li <= src || c->layer_types[li] != LZ_LT_FULL)
        return 0;
    if (c->ced_split > 0 && c->ced_split < c->n_layers && li >= c->ced_split)
        return 0;
    return 1;
}

/* model.h's own comment on this function has the contract. ced_split
   == 0 or == n_layers is not a decoder-region boundary (the same "CED
   on" test lz_kv_share_target and lz_gr_span use above), so ced_kv has
   nothing to substitute even when the config turns it on. */
int lz_ced_kv_target(const LZModelConfig *c, int li) {
    return c->ced_kv && c->ced_split > 0 && c->ced_split < c->n_layers && li >= c->ced_split;
}

/* model.h's own comment on this function has the contract. Mirrors
   lz_load_config's own csa2_kv_mode/csa2_kv_source cross-checks
   (csa2_key_idx's callers above), applied here to ONE layer's already-
   parsed fields rather than to a JSON object's entries, so lz_open_bin
   can use the identical rule on a raw header word.

   layer_types[src] == LZ_LT_FULL is checked in addition to csa2_on[src]
   rather than instead of it: csa2_on[] is only ever SET for a
   full_attention layer on the JSON path (lz_load_config's own
   csa2_layers loop refuses csa2_on[idx] for a non-full_attention idx
   before setting it), but the bin path reads csa2_on[] as a raw array
   with no such cross-check of its own - a source pointing at a
   csa2_on[]==1-but-actually-linear_attention layer would otherwise
   alias forward_state.c's cache_idx into the WRONG numbering space
   (nl, not nf), corrupting both. */
int lz_csa2_mode_source_valid(const LZModelConfig *c, int li) {
    int src;
    if (c->csa2_mode[li] != LZ_CSA2_FULL &&
        c->csa2_mode[li] != LZ_CSA2_REINDEX &&
        c->csa2_mode[li] != LZ_CSA2_REUSE) {
        return 0;
    }
    if (c->csa2_mode[li] == LZ_CSA2_FULL) return 1;
    if (!c->csa2_on[li]) return 0;
    src = c->csa2_source[li];
    return src >= 0 && src < li && c->csa2_on[src] &&
           c->layer_types[src] == LZ_LT_FULL;
}

/* model.h's own comment on this function has the contract. Cross-layer,
   unlike lz_csa2_mode_source_valid above: aliasing runs by shared
   PHYSICAL SLOT (forward_state.c's cache_idx fill, s->cache_idx[l] =
   s->cache_idx[c->csa2_source[l]] for a non-FULL mode), not by the
   declared csa2_source tree alone, and two layers can share the SAME
   slot without one being the other's own declared source - a FULL
   layer and every REINDEX layer chained off it (directly or
   transitively) all alias the one root's slot. forward_csa2.c's own
   step 4 has each FULL/REINDEX layer OVERWRITE that shared slot with
   its own fresh Top-K when its call runs (report Sec2.3.1's own
   header comment on csa2_topk_idx_pub), so by the time a REUSE (or a
   later REINDEX) layer's own call reads it, what is actually there is
   whichever csa2_on layer wrote to that root LAST - not necessarily
   the layer this one's own csa2_source names.

   Concretely: layer 0 FULL, layer 2 REINDEX(source 0), layer 3
   REUSE(source 0) - NOT source 2. lz_csa2_mode_source_valid alone
   passes layer 3 (0 is a strictly earlier, csa2_on, full_attention
   layer), but layer 2's own call already overwrote root 0's slot with
   layer 2's OWN Top-K before layer 3's call ever runs (forward.c
   dispatches layers in increasing order) - layer 3 reads layer 2's
   selection while declaring layer 0 as its source. The two are
   different questions and this function is what asks the second one.

   Assumes every layer already passed lz_csa2_mode_source_valid -
   REQUIRED, not merely convenient: this walk trusts csa2_source[i] to
   be a valid, strictly-earlier, in-bounds index for every non-FULL
   csa2_on layer it visits (both callers apply that check first, per
   layer, over the WHOLE range before ever calling this). Root-finding
   (the while loop below) therefore always terminates: csa2_source[r]
   is strictly less than r at every step, for as long as csa2_mode[r]
   is not yet FULL.

   Returns 1 if every layer's declared source is exactly its alias
   root's current last writer, 0 with *bad_layer set to the first
   violation otherwise (bad_layer may be NULL). */
int lz_csa2_last_writer_valid(const LZModelConfig *c, int *bad_layer) {
    /* static, not a stack array: this project's own convention for
       buffers of this shape - 128 ints is small on its own, but this
       function is exactly the one C1 (report Sec2.3.1 mode-dispatch
       task) found walking an ATTACKER-CONTROLLED n_layers against it
       before lz_open_bin's own n_layers bound had run; a static array
       does not remove the need for that bound (the loop below still
       has to stay inside LZ_MAX_LAYERS, which n_layers's own RD() site
       now guarantees for every caller), but it does mean an overrun
       here corrupts a fixed, adjacent-to-nothing-interesting data
       segment instead of a return address a few hundred bytes past a
       stack frame - defense in depth, not the fix itself.

       NON-REENTRANT as a consequence, which matters more than the size
       does: this engine has no threads and no signal handler that
       could call back into model loading, and the two loops right
       below always run back to back within one call - the first fully
       OVERWRITES every element to -1 before the second ever reads any
       of them, so there is no window where a second call (there is
       none, but if one were ever added) could observe a partially-
       written array from this one. Safe for THIS reason, not because
       the buffer happens to be small. */
    static int last_writer[LZ_MAX_LAYERS];
    int i;
    for (i = 0; i < LZ_MAX_LAYERS; i++) last_writer[i] = -1;
    for (i = 0; i < c->n_layers; i++) {
        int r;
        if (!c->csa2_on[i]) continue;
        if (c->csa2_mode[i] == LZ_CSA2_FULL) { last_writer[i] = i; continue; }
        r = c->csa2_source[i];
        while (c->csa2_mode[r] != LZ_CSA2_FULL) r = c->csa2_source[r];
        if (c->csa2_source[i] != last_writer[r]) {
            if (bad_layer) *bad_layer = i;
            return 0;
        }
        if (c->csa2_mode[i] == LZ_CSA2_REINDEX) last_writer[r] = i;
        /* REUSE writes nothing - forward_csa2.c's own step 4 skips it
           entirely - so last_writer[r] is untouched, and any LATER
           layer declaring THIS reuse layer as its own source fails the
           check above (last_writer[r] can only ever hold a FULL or
           REINDEX layer's index): a reuse chained off a reuse is a
           special case of the same violation this function already
           detects, which is why it replaces rather than joins the
           narrower "reuse source is itself reuse mode" check this
           project's own git history carried until this function
           existed. */
    }
    return 1;
}

/* Give every target layer the source layer's five K/V tensors.

   A STRUCT COPY, not a deep copy: the storage stays the source layer's,
   which is what the training side's module hand-over means and why
   lz_free skips these fields on a target layer.

   Runs AFTER the walk that fills the source's LZTensors, and never
   inside model_walk itself: visit_read/visit_read_bin overwrite the
   struct (not merely its buffers), so a copy taken while the files are
   being walked would keep the zeroed placeholder the field starts as.
   The two loaders therefore call it where their weights land - the bin
   path reads during lz_open_bin, the safetensors path during
   lz_read_weights, which then early-returns for a bin-loaded model. */
static void apply_kv_share(LZModel *m) {
    const LZModelConfig *c = &m->config;
    int src = lz_kv_share_src(c);
    int li, k;
    if (src < 0) return;
    for (li = src + 1; li < c->n_layers; li++) {
        if (!lz_kv_share_target(c, li)) continue;
        for (k = 0; k < N_KV_SHARE_FIELDS; k++) {
            LZTensor *dst = (LZTensor *)((char *)&m->layers[li] +
                                         KV_SHARE_FIELDS[k]);
            const LZTensor *s = (const LZTensor *)((const char *)&m->layers[src]
                                                   + KV_SHARE_FIELDS[k]);
            *dst = *s;
        }
    }
}

static lz_i64 resolve_dim(const LZModelConfig *c, int kind) {
    switch (kind) {
    case DK_HIDDEN:     return c->hidden_size;
    case DK_INTER:      return c->intermediate_size;
    case DK_VOCAB:      return c->vocab_size;
    case DK_ATTN_QGATE: return c->attn_qgate_dim;
    case DK_ATTN_KV:    return c->attn_kv_dim;
    case DK_ATTN_Q:     return c->attn_q_dim;
    case DK_HEAD_DIM:   return c->head_dim;
    case DK_ATTN_HEADS: return c->n_heads;
    case DK_CONV:       return c->lin_conv_dim;
    case DK_VALUE:      return c->lin_value_dim;
    case DK_NVHEADS:    return c->lin_n_v_heads;
    case DK_VHEADDIM:   return c->lin_v_head_dim;
    case DK_ONE:        return 1;
    case DK_KERNEL:     return c->conv_kernel;
    case DK_HIDDEN2:    return (lz_i64)c->hidden_size * 2;
    case DK_HIDDEN3:    return (lz_i64)c->hidden_size * 3;
    case DK_MTP_NDRAFT: return c->mtp_n_draft;
    case DK_MTP_MARKOV_RANK: return c->mtp_markov_rank;
    case DK_MLA_Q_RANK: return c->mla_q_lora_rank;
    case DK_MLA_O_RANK: return c->mla_o_lora_rank;
    case DK_CSA2_IDX_HEAD_DIM: return c->csa2_indexer_head_dim;
    case DK_CSA2_IDX_HEADS:    return c->csa2_indexer_heads;
    case DK_CSA2_Q_LORA_RANK:  return c->csa2_indexer_q_lora_rank;
    case DK_CSA2_UQ_OUT:       return (lz_i64)c->csa2_indexer_heads * c->csa2_indexer_head_dim;
    case DK_KEY:        return c->lin_key_dim;
    case DK_GATE_RANK:  return c->kda_gate_rank;
    case DK_NVK:        return (lz_i64)c->lin_n_v_heads * c->lin_k_head_dim;
    case DK_NUM_EXPERTS: return c->num_experts;
    case DK_LATENT:     return c->moe_latent_dim;
    case DK_SHARED_W:   return c->moe_shared_width;
    case DK_MOE_INTER:  return c->moe_intermediate_size;
    case DK_MTP_INTER:  return c->mtp_intermediate_size;
    case DK_GR_NR:      return c->gated_residual;
    /* max(1, hidden/8), not hidden/8: GatedResidual4.__init__ is
       `self.r = max(1, dim // 8)`, so a hidden smaller than 8 gets rank 1
       rather than rank 0. */
    case DK_GR_RANK:    return c->hidden_size / 8 > 0 ? c->hidden_size / 8 : 1;
    case DK_GR_FLAT:    return (lz_i64)c->gated_residual * c->hidden_size;
    case DK_ENGRAM_ROWS: return c->engram_total_rows;
    case DK_ENGRAM_DIM:  return c->engram_dim;
    case DK_ENGRAM_PER_ORDER: return c->engram_per_order;
    default:            return -1;
    }
}

static int check_shape(const LZModelConfig *c, const LZStTensor *t,
                       const int *dims, char *errbuf, int errlen) {
    int nd = 0, i;
    while (nd < 3 && dims[nd] != DK_END) nd++;

    if (t->n_dims != nd) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_DIMS,
             t->name, nd, t->n_dims);
        return 1;
    }
    for (i = 0; i < nd; i++) {
        lz_i64 want = resolve_dim(c, dims[i]);
        if (t->shape[i] != want) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_SHAPE,
                 t->name, i, want, t->shape[i]);
            return 1;
        }
    }
    return 0;
}

/* Is this tensor an indexer-K projection shaped for the HIDDEN-state
   input space?

   CSA2 obtains its indexer K by projecting the layer's own compressed
   latent - report Sec2.3.1: "CSA2 obtains indexer K by projecting main
   KV entries, replacing CSA's separate compression path from hidden
   states" - so csa2_w_kv_idx is (indexer_head_dim, head_dim). A
   checkpoint built against the earlier hidden-space projection carries
   it as (indexer_head_dim, hidden) instead: same name, same rank, a
   different function of a different tensor. Loading that would run the
   indexer on a projection of the wrong thing and produce confidently
   wrong Top-K indices with nothing to report it, which is the failure
   mode LZ_ERR_CFG_CED_KV refuses for "reuse" (model.h's own comment).
   The generic shape check below already rejects this tensor; what this
   arm adds is a message that names the cause instead of a dimension
   index.

   Safetensors only: on the bin path `t` is synthesized from `dims`
   immediately below, so it cannot disagree with them, and a stale bin is
   caught by the file's own element count instead.

   The hidden == head_dim guard is what keeps this from firing on a
   config where the two widths coincide - there the two spaces are
   indistinguishable by shape, and this arm must not claim one. */
static int csa2_idx_k_from_hidden(const LZModelConfig *c, const char *name,
                                  const LZStTensor *t) {
    lz_i64 hidden, head;
    if (!strstr(name, "self_attn.csa2_w_kv_idx.weight")) return 0;
    hidden = resolve_dim(c, DK_HIDDEN);
    head = resolve_dim(c, DK_HEAD_DIM);
    return hidden != head && t->n_dims == 2 &&
           t->shape[0] == resolve_dim(c, DK_CSA2_IDX_HEAD_DIM) &&
           t->shape[1] == hidden;
}

/* ------------------------------------------------------------ walk skeleton */

/* Call back once per expected tensor. Shared by bind and read so the
   name assembly and shape validation are not written twice - that kind
   of duplication is exactly how read/write layouts quietly diverge. */
typedef int (*LZVisit)(LZModel *m, const LZStTensor *t, LZTensor *field,
                          void *ctx, char *errbuf, int errlen);

static int m_uses_lz4(const LZModel *m);
static size_t lz4_reader(void *ctx, void *dst, size_t sz);

/* One tensor of the walk: resolve `name` (via the bin desc or the
   safetensors table), check its shape against `dims`, then hand it to
   `visit`. Extracted from model_walk so the obtain/check/visit sequence
   is written once - the per-site parts (name and `field`) are the
   callers'. */
static int model_walk_one(LZModel *m, const char *name, const int *dims,
                          LZTensor *field, LZVisit visit, void *ctx,
                          char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    const LZStTensor *t;
    LZStTensor desc;            /* bin-path desc; lives for this call so t stays valid */
    /* bin-path check is "reader set", not "bin_file non-NULL": the
       compressed sibling (model.bin.lz4) streams through m->rd with
       bin_file left NULL, and must take the same desc path. */
    if (m->bin_file || m_uses_lz4(m)) {
        lz_i64 ne = 1;
        int di;
        memset(&desc, 0, sizeof(desc));
        desc.name = name;
        for (di = 0; di < 3 && dims[di] != DK_END; di++) {
            lz_i64 d = resolve_dim(c, dims[di]);
            desc.shape[desc.n_dims++] = d;
            ne *= d;
        }
        desc.n_elem = ne;
        t = &desc;
    } else {
        t = lz_st_find(&m->st, name);
        if (!t) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_MISSING, name);
            return 1;
        }
    }
    if (!m->bin_file && !m_uses_lz4(m) &&
        csa2_idx_k_from_hidden(c, name, t)) {
        qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_IDX_K, name,
             (long long)t->shape[1]);
        return 1;
    }
    if (check_shape(c, t, dims, errbuf, errlen) != 0) return 1;
    if (visit(m, t, field, ctx, errbuf, errlen) != 0) return 1;
    return 0;
}

static int model_walk(LZModel *m, LZVisit visit, void *ctx,
                    char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    static const int EMBED_DIMS[3] = { DK_VOCAB, DK_HIDDEN, DK_END };
    static const int NORM_DIMS[3] = { DK_HIDDEN, DK_END, DK_END };
    char name[256];
    int li, si;

    /* top level: embedding and final norm */
    snprintf(name, sizeof(name), "%sembed_tokens.weight", m->prefix);
    if (model_walk_one(m, name, EMBED_DIMS, &m->embed_tokens, visit, ctx,
                       errbuf, errlen) != 0) return 1;

    snprintf(name, sizeof(name), "%snorm.weight", m->prefix);
    if (model_walk_one(m, name, NORM_DIMS, &m->final_norm, visit, ctx,
                       errbuf, errlen) != 0) return 1;

    /* per layer */
    for (li = 0; li < c->n_layers; li++) {
        LZLayer *L = &m->layers[li];
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            LZTensor *field;

            if (sp->layer_type >= 0 && sp->layer_type != L->type) continue;
            if (sp->ffn_kind >= 0 && sp->ffn_kind != L->ffn_moe) continue;
            if (sp->present && !sp->present(c)) continue;
            /* S1 attn_kv_share: a target layer owns none of the five
               shared tensors, so their specs name nothing in the
               checkpoint and the fields stay zeroed here -
               apply_kv_share fills them from the source afterwards. The
               bin stream is written in this same walk order, so the skip
               is also what keeps reader and writer in step. */
            if (kv_share_field(sp->field) && lz_kv_share_target(c, li)) continue;

            snprintf(name, sizeof(name), "%slayers.%d.%s",
                     m->prefix, li, sp->suffix);
            field = (LZTensor *)((char *)L + sp->field);
            if (model_walk_one(m, name, sp->dims, field, visit, ctx,
                               errbuf, errlen) != 0) return 1;
        }

        /* Per-expert triplet. Not in LAYER_SPECS above: the count
           (config.num_experts) is a runtime value, not something a
           fixed offsetof table can index. L->moe_expert_w1/w2/w3 are
           allocated to num_experts entries by setup_layers() before
           model_walk runs (see lz_open / lz_open_bin). */
        if (L->ffn_moe) {
            static const struct { const char *suf; size_t field; int dims[3]; }
            EXPERT_SPECS[] = {
                { "w1.weight", offsetof(LZLayer, moe_expert_w1), { DK_MOE_INTER, DK_LATENT, DK_END } },
                { "w2.weight", offsetof(LZLayer, moe_expert_w2), { DK_LATENT, DK_MOE_INTER, DK_END } },
                { "w3.weight", offsetof(LZLayer, moe_expert_w3), { DK_MOE_INTER, DK_LATENT, DK_END } }
            };
            int e, k;
            for (e = 0; e < c->num_experts; e++) {
                for (k = 0; k < 3; k++) {
                    LZTensor *arr, *field;
                    snprintf(name, sizeof(name), "%slayers.%d.mlp.experts.%d.%s",
                             m->prefix, li, e, EXPERT_SPECS[k].suf);
                    arr = *(LZTensor **)((char *)L + EXPERT_SPECS[k].field);
                    field = &arr[e];
                    if (model_walk_one(m, name, EXPERT_SPECS[k].dims, field,
                                       visit, ctx, errbuf, errlen) != 0) return 1;
                }
            }
        }

        /* S2 CSA2. Not in LAYER_SPECS above for the same class of reason
           the MoE per-expert triplet just above is not: LZPresentFn only
           sees the config, not which layer is being walked, and
           csa2_on[] is per-layer (model.h's own comment on why this
           differs from mla_q_on/mla_o_on, which a plain LZPresentFn CAN
           express because they are uniform across every full-attention
           layer). L->csa2_on is setup_layers-derived and already
           correct here (setup_layers runs before model_walk - see
           lz_open/lz_open_bin, same ordering the MoE block above
           depends on). */
        if (L->csa2_on) {
#define CSA2_OWN_FULL    1
#define CSA2_OWN_REINDEX 2
            /* CSA2_OWN_FULL/CSA2_OWN_REINDEX: which mode(s) actually own
               a given tensor, carried PER ENTRY rather than inferred
               from position in the array (an earlier version used a
               bare `k < 4` against the loop index, which encoded "the
               first four entries are FULL-only" as an unstated fact
               about array ORDER - silent and correct only for as long
               as nobody reorders or inserts a row; this project's own
               mutation coverage can only flip "does the walk gate on
               mode at all", never "is the cut at the right index", so a
               reordering bug would not even fail its own gate). Each
               row now states its own ownership, so grep or a diff on
               this table shows the fact directly, and a row inserted
               anywhere still carries the right bits with it. */
            static const struct { const char *suf; size_t field; int dims[3]; int own; }
            CSA2_SPECS[] = {
                { "self_attn.csa2_w_kv.weight",       offsetof(LZLayer, csa2_w_kv),
                  { DK_HEAD_DIM, DK_HIDDEN, DK_END }, CSA2_OWN_FULL },
                { "self_attn.csa2_w_gate.weight",     offsetof(LZLayer, csa2_w_gate),
                  { DK_HEAD_DIM, DK_HIDDEN, DK_END }, CSA2_OWN_FULL },
                { "self_attn.csa2_w_kv_idx.weight",   offsetof(LZLayer, csa2_w_kv_idx),
                  { DK_CSA2_IDX_HEAD_DIM, DK_HEAD_DIM, DK_END }, CSA2_OWN_FULL },
                { "self_attn.csa2_k_norm_idx.weight", offsetof(LZLayer, csa2_k_norm_idx),
                  { DK_CSA2_IDX_HEAD_DIM, DK_END, DK_END }, CSA2_OWN_FULL },
                { "self_attn.csa2_w_weights.weight",  offsetof(LZLayer, csa2_w_weights),
                  { DK_CSA2_IDX_HEADS, DK_HIDDEN, DK_END },
                  CSA2_OWN_FULL | CSA2_OWN_REINDEX },
                { "self_attn.csa2_w_dq_idx.weight",   offsetof(LZLayer, csa2_w_dq_idx),
                  { DK_CSA2_Q_LORA_RANK, DK_HIDDEN, DK_END },
                  CSA2_OWN_FULL | CSA2_OWN_REINDEX },
                { "self_attn.csa2_w_uq_idx.weight",   offsetof(LZLayer, csa2_w_uq_idx),
                  { DK_CSA2_UQ_OUT, DK_CSA2_Q_LORA_RANK, DK_END },
                  CSA2_OWN_FULL | CSA2_OWN_REINDEX }
            };
            /* Which of the seven a layer's csa2_mode actually owns -
             * forward_csa2.c's own dispatch already reads exactly this
             * subset (step 2, indices 0-3, is skipped whenever mode !=
             * FULL; step 4, indices 4-6, is skipped only for REUSE), so
             * this walk is catching up to code that was already
             * correct, not the other way around. Mirrors the training
             * side's engine/csa2.py _wire_csa2 field for field
             * (export_q8.py's csa2_suffixes_for is the same rule
             * re-derived on that side - a different repository, no
             * shared git history, so neither side's fix reaches the
             * other by any mechanism except being written again):
             *   FULL    all seven.
             *   REINDEX only indices 4-6 (weights, dq_idx, uq_idx - the
             *           indexer QUERY half). Main KV and indexer K
             *           (0-3) are both reused from csa2_source, never
             *           recomputed locally - a local copy would be dead
             *           weight nothing reads.
             *   REUSE   none - no compression, no indexer at all.
             * dq_idx (index 5) is NOT further conditioned on S4's
             * shared low-rank query latent (mla_q_lora_rank > 0) the
             * way the training side's own csa2_suffixes_for
             * additionally drops it there: this engine side does not
             * support that composition on ANY csa2 mode (model.h's own
             * csa2_w_dq_idx comment) and always expects its own
             * dedicated tensor regardless of mode. That composition is
             * refused by lz_load_config's own csa2_layers loop and by
             * lz_open_bin's own mla_q_lora_rank check (both run well
             * before this walk ever executes) - NOT by a missing-tensor
             * error here, whatever an earlier version of this comment
             * claimed. That claim held for a safetensors checkpoint
             * that genuinely dropped dq_idx the way the training side
             * does under the composition (name lookup fails), but never
             * for the .bin format: its reads are POSITIONAL, with no
             * name for a missing tensor to fail a lookup against, so a
             * hand-built or corrupted .bin header naming both fields
             * would have loaded and run - silently misaligned past this
             * walk's dq_idx read, not refused - with nothing here to
             * stop it before the explicit checks named above existed. */
            int k;
            if (L->csa2_mode != LZ_CSA2_REUSE) {
                int want = (L->csa2_mode == LZ_CSA2_FULL) ?
                           CSA2_OWN_FULL : CSA2_OWN_REINDEX;
                for (k = 0; k < (int)(sizeof(CSA2_SPECS) / sizeof(CSA2_SPECS[0])); k++) {
                    LZTensor *field;
                    if (!(CSA2_SPECS[k].own & want)) continue;
                    snprintf(name, sizeof(name), "%slayers.%d.%s",
                             m->prefix, li, CSA2_SPECS[k].suf);
                    field = (LZTensor *)((char *)L + CSA2_SPECS[k].field);
                    if (model_walk_one(m, name, CSA2_SPECS[k].dims, field,
                                       visit, ctx, errbuf, errlen) != 0) return 1;
                }
            }
#undef CSA2_OWN_FULL
#undef CSA2_OWN_REINDEX
        }

        /* Learned attention sink - one (n_heads) vector per layer the
           switch reaches. Not in LAYER_SPECS above, for exactly the
           reason the CSA2 block just above is not: LZPresentFn sees only
           the config, and attn_sink_on is per-layer (setup_layers skips
           the csa2_layers). Walked AFTER CSA2, which fixes this tensor's
           position in the .bin stream - the exporter's tensor_order has
           to place it here too, since a .bin carries no names.

           Key is "self_attn.sinks", not "...sinks.weight": the training
           side registers it as a bare nn.Parameter, so that is the name
           the checkpoint has. */
        if (L->attn_sink_on) {
            static const int SINK_DIMS[3] = { DK_ATTN_HEADS, DK_END, DK_END };
            snprintf(name, sizeof(name), "%slayers.%d.self_attn.sinks",
                     m->prefix, li);
            if (model_walk_one(m, name, SINK_DIMS, &L->attn_sinks,
                               visit, ctx, errbuf, errlen) != 0) return 1;
        }

        /* GR. Not in LAYER_SPECS above for both of the reasons the two
           blocks above are not: gr_on is per-layer (only the encoder half,
           LZLayer's own comment), and the branch norms are an array whose
           count nr is a runtime value while wd/wu/ww exist only on a
           gr_on layer. Walked AFTER the sink block, which fixes the GR
           family's position in the .bin stream - a .bin carries no names,
           so the exporter's tensor_order has to place it here too.

           The order WITHIN a layer is the checkpoint's own state-dict
           order, because the suffixes are literally its keys
           (GrDecoderLayer.gn_attn/gn_mlp are GatedResidual4 modules whose
           ModuleList `norms` was registered before wd/wu/ww): the four
           norms, then wd, wu, ww, then the same again for the MLP half. */
        if (L->gr_on) {
            static const char *GR_HALF[2] = { "gr_attn.", "gr_mlp." };
            static const size_t GR_NORMS[2] = {
                offsetof(LZLayer, gr_attn_norms),
                offsetof(LZLayer, gr_mlp_norms)
            };
            static const size_t GR_W[2][3] = {
                { offsetof(LZLayer, gr_attn_wd), offsetof(LZLayer, gr_attn_wu),
                  offsetof(LZLayer, gr_attn_ww) },
                { offsetof(LZLayer, gr_mlp_wd),  offsetof(LZLayer, gr_mlp_wu),
                  offsetof(LZLayer, gr_mlp_ww) }
            };
            static const char *GR_W_SUF[3] = { "wd.weight", "wu.weight", "ww.weight" };
            static const int GR_W_DIMS[3][3] = {
                { DK_GR_RANK, DK_GR_FLAT, DK_END },
                { DK_GR_FLAT, DK_GR_RANK, DK_END },
                { DK_GR_NR,   DK_GR_FLAT, DK_END }
            };
            int h, b, k;
            for (h = 0; h < 2; h++) {
                LZTensor *norms = *(LZTensor **)((char *)L + GR_NORMS[h]);
                for (b = 0; b < c->gated_residual; b++) {
                    snprintf(name, sizeof(name), "%slayers.%d.%snorms.%d.weight",
                             m->prefix, li, GR_HALF[h], b);
                    if (model_walk_one(m, name, NORM_DIMS, &norms[b],
                                       visit, ctx, errbuf, errlen) != 0) return 1;
                }
                for (k = 0; k < 3; k++) {
                    snprintf(name, sizeof(name), "%slayers.%d.%s%s",
                             m->prefix, li, GR_HALF[h], GR_W_SUF[k]);
                    if (model_walk_one(m, name, GR_W_DIMS[k],
                                       (LZTensor *)((char *)L + GR_W[h][k]),
                                       visit, ctx, errbuf, errlen) != 0) return 1;
                }
            }
        }
    }

    /* MTP head. Walked last so a checkpoint without one costs nothing.

       NOTE the names carry NO m->prefix: upstream puts them at the top
       level (`mtp.fc.weight`) while the body sits under
       `model.language_model.`. That asymmetry is exactly what let 15
       tensors slip past the leak check unnoticed - see lz_open. */
    if (m->mtp) {
        static const struct { const char *suffix; size_t field; int dims[3];
                              LZPresentFn present; }
        MTP_SPECS[] = {
            { "fc.weight",                     offsetof(LZMtp, fc),
              { DK_HIDDEN, DK_HIDDEN2, DK_END }, present_mtp_ndraft1 },
            { "norm.weight",                   offsetof(LZMtp, norm),
              { DK_HIDDEN, DK_END, DK_END }, NULL },
            { "pre_fc_norm_hidden.weight",     offsetof(LZMtp, pre_fc_norm_hidden),
              { DK_HIDDEN, DK_END, DK_END }, NULL },
            { "pre_fc_norm_embedding.weight",  offsetof(LZMtp, pre_fc_norm_embedding),
              { DK_HIDDEN, DK_END, DK_END }, NULL },
            /* mtp_n_draft > 1 only (DSpark, arXiv:2607.05147) -
               mutually exclusive with fc.weight above, never both bound
               on the same checkpoint. See LZMtp's own struct comment. */
            { "main_proj.weight",              offsetof(LZMtp, main_proj),
              { DK_HIDDEN, DK_HIDDEN3, DK_END }, present_mtp_ndraft_gt1 },
            { "depth_embed.weight",            offsetof(LZMtp, depth_embed),
              { DK_MTP_NDRAFT, DK_HIDDEN, DK_END }, present_mtp_ndraft_gt1 },
            { "mask_embed",                    offsetof(LZMtp, mask_embed),
              { DK_HIDDEN, DK_END, DK_END }, present_mtp_ndraft_gt1 },
            { "markov_embed.weight",           offsetof(LZMtp, markov_embed),
              { DK_VOCAB, DK_MTP_MARKOV_RANK, DK_END }, present_mtp_ndraft_gt1 },
            { "markov_proj.weight",            offsetof(LZMtp, markov_proj),
              { DK_VOCAB, DK_MTP_MARKOV_RANK, DK_END }, present_mtp_ndraft_gt1 }
        };
        /* The MTP block's own dense FFN, sized by mtp_intermediate_size -
           separate from LAYER_SPECS' body dense-FFN entries, even though
           the suffixes are identical. kunmoe-v2's body intermediate_size
           is an unrelated placeholder (first_k_dense_replace=0 means no
           body layer is dense), and reusing that field for the MTP
           block's width would silently couple two things that must be
           free to differ - see model.h's mtp_intermediate_size comment
           for the full argument. Everything else about the block still
           matches a body full_attention layer's shape one for one,
           which is why only this family of four specs needs to leave
           the generic reuse loop below.
           down_proj.norm.weight (present field, use_subn only) belongs
           to this family for the same reason the other three do: LOFF
           reuses LZLayer's down_norm field, but its width is
           mtp_intermediate_size, not the body's intermediate_size, so
           it cannot be picked up by the generic reuse loop either. A
           DK_INTER dim filter on that loop silently drops it
           (down_norm's LAYER_SPECS entry uses DK_INTER too, and such a
           filter matches on dim kind, not identity), which leaves
           L->down_norm a permanent zero LZTensor and crashes
           dense_ffn_step's SubLN norm on the first --spec MTP token. */
        static const struct { const char *suffix; size_t field; int dims[3];
                               LZPresentFn present; }
        MTP_FFN_SPECS[] = {
            { "mlp.gate_proj.weight",      LOFF(gate_proj), { DK_MTP_INTER, DK_HIDDEN, DK_END }, NULL },
            { "mlp.up_proj.weight",        LOFF(up_proj),   { DK_MTP_INTER, DK_HIDDEN, DK_END }, NULL },
            { "mlp.down_proj.weight",      LOFF(down_proj), { DK_HIDDEN, DK_MTP_INTER, DK_END }, NULL },
            { "mlp.down_proj.norm.weight", LOFF(down_norm), { DK_MTP_INTER, DK_END, DK_END },
              present_subn }
        };
        int k;

        /* The block reuses LAYER_SPECS for everything EXCEPT the four
           MTP_FFN_SPECS entries above, identified by field offset (NOT
           by dim kind - DK_INTER also appears on down_norm's LAYER_SPECS
           entry, which is exactly the false match that makes a
           dim-kind filter connascent with a dim taxonomy it has no
           business reading; see the MTP_FFN_SPECS comment above). Every
           other tensor in the block matches a body full_attention layer's
           shape one for one, which is what makes "reuse the same
           pruning path" possible on the training side too. */
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            LZTensor *field;
            if (sp->layer_type >= 0 && sp->layer_type != LZ_LT_FULL) continue;
            /* The MTP head's FFN is always the classic dense triplet -
               upstream ships it as an ordinary full_attention layer, and
               LZMtp has no ffn_moe flag to route MoE specs through even
               if it did. Without this filter the loop below also tries
               to load mtp.layers.0.mlp.gate.weight (the MoE router) on
               any kunmoe checkpoint, which does not exist. */
            if (sp->ffn_kind >= 0 && sp->ffn_kind != 0) continue;
            if (sp->present && !sp->present(c)) continue;
            if (sp->field == LOFF(gate_proj) || sp->field == LOFF(up_proj) ||
                sp->field == LOFF(down_proj) || sp->field == LOFF(down_norm))
                continue;   /* MTP_FFN_SPECS handles these, sized by mtp_intermediate_size */
            snprintf(name, sizeof(name), "mtp.layers.0.%s", sp->suffix);
            field = (LZTensor *)((char *)&m->mtp->blk + sp->field);
            if (model_walk_one(m, name, sp->dims, field, visit, ctx,
                               errbuf, errlen) != 0) return 1;
        }
        for (k = 0; k < (int)(sizeof(MTP_FFN_SPECS) / sizeof(MTP_FFN_SPECS[0])); k++) {
            LZTensor *field;
            if (MTP_FFN_SPECS[k].present && !MTP_FFN_SPECS[k].present(c)) continue;
            snprintf(name, sizeof(name), "mtp.layers.0.%s", MTP_FFN_SPECS[k].suffix);
            field = (LZTensor *)((char *)&m->mtp->blk + MTP_FFN_SPECS[k].field);
            if (model_walk_one(m, name, MTP_FFN_SPECS[k].dims, field, visit,
                               ctx, errbuf, errlen) != 0) return 1;
        }
        for (k = 0; k < (int)(sizeof(MTP_SPECS) / sizeof(MTP_SPECS[0])); k++) {
            LZTensor *field;
            if (MTP_SPECS[k].present && !MTP_SPECS[k].present(c)) continue;
            snprintf(name, sizeof(name), "mtp.%s", MTP_SPECS[k].suffix);
            field = (LZTensor *)((char *)m->mtp + MTP_SPECS[k].field);
            if (model_walk_one(m, name, MTP_SPECS[k].dims, field, visit, ctx,
                               errbuf, errlen) != 0) return 1;
        }
    }
    /* Engram: global module, same level as MTP in model_walk.
       Walked last so a checkpoint without engram costs nothing.
       alloc_engram ran before model_walk (lz_open), so m->engram
       is non-NULL when config.engram is set. */
    if (m->engram) {
        /* Named arrays, not anonymous C99 compound literals: this file's
           floor is C89/Watcom (see the lz_open_bin LZ4Stream comment
           above for the same rule applied to declaration placement),
           and `(int[]){...}` is a C99-only construct wcc386 -za99
           rejects. EMBED_DIMS/NORM_DIMS above use the same static
           const array idiom for the same reason. */
        static const int ENGRAM_TABLE_DIMS[3] =
            { DK_ENGRAM_ROWS, DK_ENGRAM_DIM, DK_END };
        static const int ENGRAM_PROJ_DIMS[3] =
            { DK_HIDDEN, DK_ENGRAM_PER_ORDER, DK_END };
        static const int ENGRAM_CONV_DIMS[3] =
            { DK_HIDDEN, DK_ONE, DK_KERNEL };
        static const int ENGRAM_NORM_DIMS[3] =
            { DK_HIDDEN, DK_END, DK_END };
        LZEngram *eg = m->engram;
        int no = c->engram_n_orders;
        int o;
        /* Engram tensors are NEW: a safetensors checkpoint may not
           carry them. alloc_engram already allocated zeroed f32
           buffers and filled norm with 1.0f. visit_bind overwrites
           any that the checkpoint ships; model_walk_one errors on
           missing tensors, so we guard each call with lz_st_find.

           BIN MODE (m->rd != NULL): the bin file is a sequential
           stream visited in tensor_order's order. Skipping a tensor
           here desyncs every subsequent read. In bin mode we must
           visit every tensor unconditionally - export_q8 already
           wrote them or didn't, and the stream has no names to
           search. */
        int bin_mode = (m->rd != NULL);
        if (bin_mode || lz_st_find(&m->st, "engram_memory.table.weight"))
            if (model_walk_one(m, "engram_memory.table.weight",
                               ENGRAM_TABLE_DIMS,
                               &eg->table, visit, ctx,
                               errbuf, errlen)) return 1;
        for (o = 0; o < no; o++) {
            snprintf(name, sizeof(name),
                     "engram_memory.key_proj.%d.weight", o);
            if (bin_mode || lz_st_find(&m->st, name))
                if (model_walk_one(m, name,
                                   ENGRAM_PROJ_DIMS,
                                   &eg->key_proj[o], visit, ctx,
                                   errbuf, errlen)) return 1;
            snprintf(name, sizeof(name),
                     "engram_memory.value_proj.%d.weight", o);
            if (bin_mode || lz_st_find(&m->st, name))
                if (model_walk_one(m, name,
                                   ENGRAM_PROJ_DIMS,
                                   &eg->value_proj[o], visit, ctx,
                                   errbuf, errlen)) return 1;
        }
        if (bin_mode || lz_st_find(&m->st, "engram_memory.conv.weight"))
            if (model_walk_one(m, "engram_memory.conv.weight",
                               ENGRAM_CONV_DIMS,
                               &eg->conv, visit, ctx,
                               errbuf, errlen)) return 1;
        if (bin_mode || lz_st_find(&m->st, "engram_memory.norm.weight"))
            if (model_walk_one(m, "engram_memory.norm.weight",
                               ENGRAM_NORM_DIMS,
                               &eg->norm, visit, ctx,
                               errbuf, errlen)) return 1;
    }
    return 0;
}

/* Engram norm fixup: the random-init checkpoints this engine is being
   brought up against carry engram_memory.norm.weight as ZEROS
   (Qwen3_5RMSNorm's own init is ones, but the arm generator loaded a
   checkpoint whose engram tensors did not exist, so save_pretrained
   wrote the freshly-initialised zeros back). RMSNorm with gamma=0
   zeroes its output, which zeroes the gate's query and with it the
   whole engram contribution - and the PyTorch side of the parity test
   applies the same fix, so both sides must agree on the repair.

   Runs AFTER model_walk for BOTH visit paths (visit_bind marks claimed
   without reading; visit_read overwrites the buffer), which is why
   this lives in lz_read_weights and not in model_walk. */
static void engram_norm_fixup(LZModel *m) {
    const LZModelConfig *c = &m->config;
    LZEngram *eg = m->engram;
    int k, all_zero = 1;
    if (!eg) return;
    /* Only reachable on LZ_FMT_F32: the checkpoints this workaround
       targets are freshly random-initialized by from_pretrained (see
       the comment above), which the export pipeline never narrows to
       BF16 - narrowing is export_q8's job, run against a TRAINED
       checkpoint, and a trained norm is not all-zero by construction
       (RMSNorm's own init is ones; zero only happens when
       from_pretrained materializes a tensor the checkpoint never
       carried). eg->norm.f is NULL for BF16/quantized dtypes, so the
       loop below is a silent no-op there rather than a crash - correct
       for the case that is actually reachable, but worth stating why
       it is not handled rather than leaving it implicit. */
    if (eg->norm.dtype != LZ_FMT_F32) return;
    for (k = 0; k < c->hidden_size; k++)
        if (eg->norm.f && eg->norm.f[k] != 0.0f) { all_zero = 0; break; }
    if (all_zero && eg->norm.f)
        for (k = 0; k < c->hidden_size; k++) eg->norm.f[k] = 1.0f;
}

/* --------------------------------------------------------------- binding */

typedef struct {
    unsigned char *claimed;     /* whether each tensor was claimed; for leak detection */
    lz_i64 n_params;
} LZBindCtx;

static int visit_bind(LZModel *m, const LZStTensor *t, LZTensor *field,
                      void *ctx, char *errbuf, int errlen) {
    LZBindCtx *bc = (LZBindCtx *)ctx;
    long idx = (long)(t - m->st.tensors);
    (void)field;
    (void)errbuf;
    (void)errlen;
    bc->claimed[idx] = 1;
    bc->n_params += t->n_elem;
    return 0;
}

/* The tensor-name prefix differs by shell: the multimodal wrapper uses
   model.language_model., the plain-text checkpoint model. Probe rather
   than hardcode. */
static int detect_prefix(LZModel *m, char *errbuf, int errlen) {    static const char *CANDIDATES[] = {
        "model.language_model.", "model.", "language_model.", ""
    };
    char name[256];
    size_t i;

    for (i = 0; i < sizeof(CANDIDATES) / sizeof(CANDIDATES[0]); i++) {
        snprintf(name, sizeof(name), "%sembed_tokens.weight", CANDIDATES[i]);
        if (lz_st_find(&m->st, name)) {
            snprintf(m->prefix, sizeof(m->prefix), "%s", CANDIDATES[i]);
            return 0;
        }
    }
    qerr(errbuf, errlen, LZ_ERR_PREFIX);
    return 1;
}

/* How many layers GR covers, from the top: the training side's own rule
   (engine/kunmoe_modeling.py's _wire_gr), `span = ced_split if
   0 < ced_split < n_layers else n_layers`. GR is scoped to the ENCODER
   half because CED's capture hook reads the split boundary's OUTPUT
   expecting a plain [B,S,D] stream - so the boundary layer is an early
   `gr_last`, collapsing exactly what the true last layer would. A
   ced_split with no decoder half (>= n_layers) is NOT a boundary, which
   is why the middle clause exists rather than a bare `ced_split ?`. */
static int lz_gr_span(const LZModelConfig *c) {
    int n = c->n_layers;
    if (c->ced_split > 0 && c->ced_split < n) return c->ced_split;
    return n;
}

/* Set each layer's attention type and FFN routing, and allocate the
   per-expert tensor arrays MoE layers need. Shared by lz_open (safetensors)
   and lz_open_bin so the two paths cannot disagree about which layers are
   MoE - that decision (config.num_experts > 0 && li >= first_k_dense_replace)
   is written once here. */
static int setup_layers(LZModel *m, char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    int i;
    for (i = 0; i < c->n_layers; i++) {
        LZLayer *L = &m->layers[i];
        L->type = c->layer_types[i];
        L->ffn_moe = (c->num_experts > 0 && i >= c->first_k_dense_replace) ? 1 : 0;
        /* S4 MLA (model.h's own comment on both LZModelConfig's fields
           and LZLayer's q_proj_down/mla_q_on): PER-LAYER flags derived
           here, in the loop that only ever touches m->layers, so
           m->mtp->blk - a separate LZLayer, calloc'd by alloc_mtp and
           never passed through this loop - keeps mla_q_on/mla_o_on at
           its calloc default of 0 regardless of these two global config
           fields. attn_impl (forward_attn.c) reads L->mla_q_on/mla_o_on,
           never c->mla_q_lora_rank/mla_o_lora_rank directly, for exactly
           that reason. */
        L->mla_q_on = c->mla_q_lora_rank > 0;
        L->mla_o_on = c->mla_o_lora_rank > 0;
        /* S2 CSA2: per-layer, unlike mla_q_on/mla_o_on above - csa2_on[]
           may name a SUBSET of the full-attention layers (model.h's own
           csa2_on[] comment). Same "only m->layers is touched, m->mtp->
           blk never is" safety this mirrors from mla_q_on/mla_o_on. */
        L->csa2_on = c->csa2_on[i];
        /* Sibling of csa2_on above, same reasoning: the forward reads
           L->csa2_mode (forward_csa2.c's own mode dispatch - FULL/
           REINDEX/REUSE, report Sec2.3.1), never c->csa2_mode[]
           directly, so the MTP block's own LZLayer - calloc'd
           separately, never touched by this loop - stays at its calloc
           default LZ_CSA2_FULL (0) regardless of what config.json's
           csa2_kv_mode says about the
           body layer index i happens to share with it. */
        L->csa2_mode = c->csa2_mode[i];
        /* Learned attention sink: three conditions, and all three are the
           training side's _wire_attn_sink rule - the global switch, a
           full-attention layer (a linear-attention layer has no
           self_attn to hang a sink on), and not one the csa2_layers
           turned into CSA2 (that writes its own .forward, so a sink
           bound there would be created and never read). Same
           "only m->layers is touched" safety as the two flags above. */
        L->attn_sink_on = c->attn_sink_learned &&
                          L->type == LZ_LT_FULL && !L->csa2_on;
        /* GR: three flags, all from the one span rule above. `>= 2` rather
           than `> 0` is the training side's own contract (GatedResidual4
           raises below nr=2, there is no single-stream GR); a config.json
           that asks for nr=1 therefore gets no branches AND no specs, so
           a checkpoint that does carry gr_* tensors is refused by the
           leak check rather than silently run without them. Same
           "per-layer flags derived here, m->mtp->blk never passes through
           this loop" safety as the three flags above. */
        L->gr_on = c->gated_residual >= 2 && i < lz_gr_span(c);
        L->gr_first = L->gr_on && i == 0;
        L->gr_last  = L->gr_on && i == lz_gr_span(c) - 1;
        if (L->gr_on) {
            /* One norm per branch, so an ARRAY whose count is a runtime
               config value - the same reason moe_expert_w1 above is one.
               Checked separately rather than with one guard: a failure on
               the second calloc leaves the first non-NULL, and lz_free
               indexes both. */
            L->gr_attn_norms = (LZTensor *)calloc((size_t)c->gated_residual, sizeof(LZTensor));
            L->gr_mlp_norms  = (LZTensor *)calloc((size_t)c->gated_residual, sizeof(LZTensor));
            if (!L->gr_attn_norms || !L->gr_mlp_norms) {
                qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
                return 1;
            }
        }
        if (L->ffn_moe) {
            L->moe_expert_w1 = (LZTensor *)calloc((size_t)c->num_experts, sizeof(LZTensor));
            L->moe_expert_w2 = (LZTensor *)calloc((size_t)c->num_experts, sizeof(LZTensor));
            L->moe_expert_w3 = (LZTensor *)calloc((size_t)c->num_experts, sizeof(LZTensor));
            if (!L->moe_expert_w1 || !L->moe_expert_w2 || !L->moe_expert_w3) {
                qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
                return 1;
            }
        }
    }
    return 0;
}

/* Bind the MTP head's storage when the config declares one. Shared by
   lz_open (safetensors) and lz_open_bin so the two loaders cannot
   disagree about when m->mtp exists - same reasoning as setup_layers
   above. Refused above 1: upstream ships exactly one block and nothing
   here is written for a stack of them, so a 2 would bind the first and
   silently ignore the rest - the failure this whole area exists to
   prevent. A no-op (m->mtp stays NULL) when mtp_n_layers is 0, which is
   every model this project has produced. */
static int alloc_mtp(LZModel *m, char *errbuf, int errlen) {
    if (m->config.mtp_n_layers <= 0) return 0;
    if (m->config.mtp_n_layers > 1) {
        qerr(errbuf, errlen, LZ_ERR_MTP_LAYERS, m->config.mtp_n_layers);
        return 1;
    }
    if (m->config.mtp_n_draft > LZ_SPEC_K_MAX) {
        qerr(errbuf, errlen, LZ_ERR_MTP_NDRAFT, m->config.mtp_n_draft,
             LZ_SPEC_K_MAX);
        return 1;
    }
    /* lz_mtp_markov_correct reads markov_embed's row into
       mtp_kdraft_concat, a fixed 3*hidden_size scratch buffer it shares
       with lz_mtp_draft_batch's base stage - see forward_mtp.c's own
       comment on embed_row. Only reachable with mtp_n_draft>1 (the
       markov head only exists in that tier). */
    if (m->config.mtp_n_draft > 1 &&
        m->config.mtp_markov_rank > 3 * m->config.hidden_size) {
        qerr(errbuf, errlen, LZ_ERR_MTP_MARKOV_RANK,
             m->config.mtp_markov_rank, 3 * m->config.hidden_size);
        return 1;
    }
    m->mtp = (LZMtp *)calloc(1, sizeof(LZMtp));
    if (!m->mtp) {
        qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
        return 1;
    }
    m->mtp->blk.type = LZ_LT_FULL;
    return 0;
}

static int alloc_engram(LZModel *m, char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    LZEngram *eg;
    int no, hdim, edim, o;
    if (!c->engram) return 0;

    eg = (LZEngram *)calloc(1, sizeof(LZEngram));
    if (!eg) { qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC); return 1; }
    m->engram = eg;

    no = c->engram_n_orders;
    hdim = c->hidden_size;
    edim = c->engram_dim;

    /* Primes, offsets, bases (derived from config, not in checkpoint).
       lz_load_config already computed engram_total_rows and
       engram_head_space from engram_size. */
    eg->primes = (int *)calloc(c->engram_n_spaces, sizeof(int));
    eg->offsets = (int *)calloc(c->engram_n_spaces, sizeof(int));
    eg->bases = (lz_i64 *)calloc(no, sizeof(lz_i64));
    if (!eg->primes || !eg->offsets || !eg->bases) {
        qerr(errbuf, errlen, LZ_ERR_OOM_ALLOC, "engram primes");
        return 1;
    }
    {   /* primes_down: largest primes <= head_space */
        int x = c->engram_head_space, found = 0;
        while (found < c->engram_n_spaces) {
            int is_prime = (x >= 2), d;
            for (d = 2; is_prime && d * d <= x; d++)
                if (x % d == 0) is_prime = 0;
            if (is_prime) eg->primes[found++] = x;
            x--;
        }
    }
    {   /* offsets: cumulative start row per space */
        int total = 0, k;
        for (k = 0; k < c->engram_n_spaces; k++) {
            eg->offsets[k] = total;
            total += eg->primes[k];
        }
    }
    { int o; for (o = 0; o < no; o++)
        eg->bases[o] = (lz_i64)(2 * ((o + 1) * 7 + 1) + 1);
    }

    /* Allocate tensors with aligned_malloc (lz_t_free's release path).
       norm starts at 1.0f, rest zero - training fills them. */
    eg->table.n = m->config.engram_total_rows * edim;
    eg->table.f = (float *)aligned_malloc((size_t)eg->table.n, sizeof(float));
    if (eg->table.f) memset(eg->table.f, 0, (size_t)eg->table.n * sizeof(float));
    eg->conv.n  = hdim * c->engram_conv_kernel;
    eg->conv.f  = (float *)aligned_malloc((size_t)eg->conv.n, sizeof(float));
    if (eg->conv.f) memset(eg->conv.f, 0, (size_t)eg->conv.n * sizeof(float));
    eg->norm.n  = hdim;
    eg->norm.f  = (float *)aligned_malloc(hdim, sizeof(float));
    if (eg->norm.f) { int i; for (i = 0; i < hdim; i++) eg->norm.f[i] = 1.0f; }
    eg->key_proj   = (LZTensor *)calloc(no, sizeof(LZTensor));
    eg->value_proj = (LZTensor *)calloc(no, sizeof(LZTensor));
    for (o = 0; o < no; o++) {
        int n = c->engram_per_order * hdim;
        eg->key_proj[o].f   = (float *)aligned_malloc(n, sizeof(float));
        if (eg->key_proj[o].f) memset(eg->key_proj[o].f, 0, n * sizeof(float));
        eg->key_proj[o].n   = n;
        eg->value_proj[o].f = (float *)aligned_malloc(n, sizeof(float));
        if (eg->value_proj[o].f) memset(eg->value_proj[o].f, 0, n * sizeof(float));
        eg->value_proj[o].n = n;
    }
    return 0;
}

/* Load config and all weights in one pass from model.bin */
static int lz_open_bin(LZModel *m, const char *dir, char *errbuf, int errlen);

/* Where the weights are when they are not in model.safetensors.
 *
 * A HUGGING FACE EXPORT NAMES ITS SHARDS AND LISTS THEM IN
 * model.safetensors.index.json, whose weight_map sends every tensor to a
 * file. Without this the loader looks for exactly one name, and a
 * perfectly ordinary export fails with "cannot find model.safetensors"
 * in a directory that plainly holds the weights - which is what
 * Qwen3.5-0.8B does here, its single shard being called
 * model.safetensors-00001-of-00001.safetensors.
 *
 * ONE SHARD ONLY, and the limit is deliberate rather than provisional.
 * LZSafetensors owns a single FILE * and one tensor table; spanning
 * several would mean a shard index per tensor and a handle array, which
 * is a real feature and not a redirect. Every index this has been shown
 * maps to one file, so the redirect is the whole of what is needed
 * today - and a multi-shard index gets a message SAYING it is
 * unsupported rather than the missing-file one, because the difference
 * matters to whoever hits it.
 *
 * Returns 1 and fills `out` when the index named a single shard, -1 with
 * errbuf set for a MULTI-shard one, and 0 for everything else - no
 * index, unreadable, malformed, empty weight_map. Those all fall back
 * to the caller's ordinary "cannot find model.safetensors", which is
 * the right message when the index could not be used: the file the
 * reader is missing is still that one. Only the multi-shard case earns
 * a message of its own, because there the weights ARE all present and
 * the loader is the thing that cannot take them.
 */
static int st_path_from_index(const char *dir, char *out, int outlen,
                              char *errbuf, int errlen) {
    char ipath[512];
    void *txt;
    size_t len;
    LZJson j;
    const LZJsonNode *map, *n;
    const char *first = NULL;
    int rc = 0;

    if (!lz_lfn_exists(dir, "model.safetensors.index.json")) return 0;
    if (lz_lfn_path(dir, "model.safetensors.index.json", ipath,
                    (int)sizeof ipath, errbuf, errlen) != 0) return 0;
    txt = lz_read_file(ipath, &len, errbuf, errlen);
    if (!txt) return 0;
    if (lz_json_parse(&j, (const char *)txt, len, errbuf, errlen) != 0) {
        free(txt);
        return 0;
    }
    map = lz_json_get(&j, lz_json_root(&j), "weight_map");
    if (!map) goto done;
    /* Distinct VALUES, not the member count: the map has one entry per
       tensor and they nearly all name the same file. */
    for (n = lz_json_first(&j, map); n; n = lz_json_next(&j, n)) {
        if (n->type != LZ_JSON_STR || !n->text) continue;
        if (!first) { first = n->text; continue; }
        if (strcmp(first, n->text) != 0) {
            qerr(errbuf, errlen, LZ_ERR_ST_INDEX_SHARDS);
            rc = -1;
            goto done;
        }
    }
    if (!first) goto done;
    if (lz_lfn_path(dir, first, out, outlen, errbuf, errlen) == 0) rc = 1;
done:
    lz_json_free(&j);
    free(txt);
    return rc;
}

int lz_open(LZModel *m, const char *dir, char *errbuf, int errlen) {
    char path[512];
    LZBindCtx bc;
    int i;

    memset(m, 0, sizeof(*m));
    memset(&bc, 0, sizeof(bc));
    if (errbuf && errlen > 0) errbuf[0] = '\0';

    /* Both loaders below convert from the file's little-endian order
       using LZ_BIG_ENDIAN, so a build that guessed its host's byte order
       wrong reads every weight reversed and reports nothing. Checked here,
       at the single door both paths go through, and before any file is
       touched - the answer does not depend on the model. */
    if (!lz_endian_ok()) {
        qerr(errbuf, errlen, LZ_ERR_ENDIAN_MISMATCH);
        return 1;
    }

    /* mixed-precision bin preferred: if model.bin (or the compressed
       sibling model.bin.lz4, the stunt path) exists, load everything at
       once. Resolved rather than fopen'd directly (src/lfn.h): this probe
       decides between the bin and safetensors loaders, so a wrong "no"
       sends the whole load down the other path. lz_open_bin prefers the
       .lz4 file when both are present. */
    if (lz_lfn_exists(dir, "model.bin") || lz_lfn_exists(dir, "model.bin.lz4"))
        return lz_open_bin(m, dir, errbuf, errlen);

    if (lz_lfn_path(dir, "config.json", path, (int)sizeof path,
                    errbuf, errlen) != 0) return 1;
    if (lz_load_config(&m->config, path, errbuf, errlen) != 0) return 1;

    /* model.safetensors first, then the shard index. Tried in this order
       so an export carrying both keeps the behaviour it already had. The
       resolver is st_resolve's, shared with the architecture probe inside
       lz_load_config, so the file probed and the file loaded are one. */
    if (st_resolve(dir, path, (int)sizeof path, errbuf, errlen) != 0) return 1;
    if (lz_st_open(&m->st, path, errbuf, errlen) != 0) return 1;

    if (detect_prefix(m, errbuf, errlen) != 0) goto fail;

    m->layers = (LZLayer *)calloc((size_t)m->config.n_layers,
                                     sizeof(LZLayer));
    if (!m->layers) {
        qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
        goto fail;
    }
    if (setup_layers(m, errbuf, errlen) != 0) goto fail;

    if (alloc_mtp(m, errbuf, errlen) != 0) goto fail;
    if (alloc_engram(m, errbuf, errlen) != 0) goto fail;

    bc.claimed = (unsigned char *)calloc((size_t)m->st.n_tensors, 1);
    if (!bc.claimed) {
        qerr(errbuf, errlen, LZ_ERR_CLAIM_ALLOC);
        goto fail;
    }
    if (model_walk(m, visit_bind, &bc, errbuf, errlen) != 0) {
        free(bc.claimed);
        goto fail;
    }

    /* Leak check: an unclaimed tensor means a gap in architecture
       understanding - must error rather than silently ignore, since
       silent ignoring makes "a whole module not computed" look
       perfectly fine.

       The scan covers names at EVERY level, not only names under
       m->prefix. Upstream Qwen3.5's MTP head is 15 tensors named
       `mtp.*` at TOP level, while the text prefix on that checkpoint
       is `model.language_model.`, so a prefix-scoped scan would skip
       every one of them without a word. That is the same silent drop
       transformers performs via _keys_to_ignore_on_load_unexpected,
       reached by a different route.

       So the rule is inverted: EVERY unclaimed tensor is an error unless
       its prefix is on the exemption list below. An exemption has to be
       written down; a name cannot be excused by falling outside a
       guard. This project's own exports have zero tensors outside
       m->prefix, so the strict rule costs nothing today. */
    {
        static const char *EXEMPT[] = {
            /* The multimodal wrapper's vision tower. This engine is text
               only and does not pretend otherwise; the tensors exist in
               upstream checkpoints and are deliberately not computed. */
            "model.visual.",
            NULL
        };
        for (i = 0; i < m->st.n_tensors; i++) {
            const LZStTensor *t = &m->st.tensors[i];
            int k, excused = 0;
            if (bc.claimed[i]) continue;
            for (k = 0; EXEMPT[k]; k++) {
                size_t el = strlen(EXEMPT[k]);
                if (strncmp(t->name, EXEMPT[k], el) == 0) { excused = 1; break; }
            }
            if (!excused) {
                qerr(errbuf, errlen, LZ_ERR_UNCLAIMED, t->name);
                free(bc.claimed);
                goto fail;
            }
            m->n_params_skipped += t->n_elem;   /* named exemptions only */
        }
    }
    free(bc.claimed);

    m->n_params = bc.n_params;
    m->weights_loaded = 0;
    return 0;

fail:
    lz_free(m);
    return 1;
}

/* --------------------------------------------------------------- reading */

static int visit_read(LZModel *m, const LZStTensor *t, LZTensor *field,
                      void *ctx, char *errbuf, int errlen) {
    (void)ctx;

    field->dtype = 0;
    field->n = (int)t->n_elem;
    field->gs = 0;
    field->q = NULL;
    field->scale = NULL;
    field->f = NULL;

    /* A BF16 TENSOR STAYS BF16, which halves what it occupies and what
       reading it costs in DDR traffic. Expanding it to f32 while loading
       would turn a two-bytes-per-parameter file into four bytes in
       RAM - on a 64MB target that is the difference between a model
       fitting and not.
     *
     * Nothing is lost and nothing rounds: a bf16 IS the high half of an
     * f32, so the widening lz_t_f32 does at the point of use is the
     * identical `<<16` the loader performs here, only deferred. Every
     * value a kernel sees is the same value it saw before, which is why
     * this needs no switch and no accuracy argument.
     *
     * F16 is deliberately NOT included: its widening is a real format
     * conversion (5-bit exponent to 8, subnormal renormalisation), so it
     * would have to be undone on every read rather than being a shift,
     * and it still expands here. */
    /* TWO-DIMENSIONAL, OR ONE-DIMENSIONAL AND SMALL, OR 3-D CONV.
       The reason is a consumer's contract rather than anything about bf16.

       THE 1-D CASE IS THE ONE THE EXPORTER CARES ABOUT. tools/
       export_q8.py narrows non-2D and small tensors to BF16 (plan()'s
       FMT_BF16 branch): norms (1D), conv1d (3D, row=4 < MIN_GS=32).
       bf16 is the middle option between Q8_0 (too coarse for these
       shapes) and F32 (more than they need). A first version of this
       guard was 2-D only and therefore served none of them.
     *
     * THE 3-D CASE (conv1d [C,1,K]) was added when forward.c's
     * conv_fixed_build and forward_engram.c's engram conv were changed
     * to read one row at a time through lz_t_row_f32, so the scratch
     * bound is conv_kernel (4), not the whole tensor.
     *
     * lz_t_f32() hands back a WHOLE-TENSOR f32 view. For an F32 tensor
     * that is free - it returns t->f and copies nothing - but a bf16
     * tensor has to be widened into the caller's scratch, and that
     * scratch is s->wscr, qcap floats. So a tensor read that way may be
     * narrow only if the WHOLE of it fits, which is what the 1-D clause
     * checks: qcap starts at hidden_size and only grows, so a norm
     * (hidden-wide) and A_log/dt_bias (per-head, smaller) all fit.
     *
     * 3-D IS STILL EXCLUDED, and conv1d is why. The checkpoint measured
     * here has it at [3072, 1, 4] - 12,288 elements against a qcap in
     * the hundreds - and conv_fixed_build reads it whole through
     * lz_t_f32. Storing it narrow overran s->wscr and crashed there;
     * found by running it, not by reading it. Serving it needs that
     * caller to move to lz_t_row_f32, which is a separate change.
     *
     * The 2-D tensors have no such problem: they are weight matrices,
     * reached through lz_matmul_w / lz_matmul_xq_nt, which widen a row
     * at a time into their own bounded buffer, or through
     * lz_t_row_f32(), whose output is a row by contract. They are also
     * where the memory is - 94 of this checkpoint's 161 tensors and
     * essentially all of its bytes, so excluding the other two shapes
     * costs almost nothing.
     *
     * The second clause is that bounded buffer's own limit, the same one
     * every quantized kernel in ops_matmul.c carries. Past it a tensor
     * stays f32, which costs memory and never correctness.
     *
     * 3-D (conv1d [C,1,K]) IS NOW INCLUDED: forward.c's
     * conv_fixed_build and forward_engram.c's engram conv were changed
     * to read one row at a time through lz_t_row_f32, so the scratch
     * bound is conv_kernel (4), not the whole tensor. The row-width
     * check (shape[last_dim] <= LZ_MM_WIDEN_MAX) is what matters.
     *
     * THAT LIST WAS INCOMPLETE. conv_fixed_build's row-wise read only
     * covers the FIXED conv tier; forward_ssm.c and forward_kda.c's own
     * FLOAT-tier per-token conv step still called lz_t_f32 directly on
     * L->conv1d / kda_q_conv1d / kda_k_conv1d / kda_v_conv1d - the exact
     * whole-tensor-into-s->wscr read this guard exists to rule out. A
     * checkpoint whose float tier actually runs (--fixed off, or any
     * model whose conv_fixed_build refuses) hit the heap-buffer-overflow
     * this comment already named above, on the caller it called "a
     * separate change" - found the same way, by running it. Closed by
     * forward_state.c's conv_f32_build, the float tier's un-quantized twin of
     * conv_fixed_build: it also reads one row at a time through
     * lz_t_row_f32, into its own persistent buffer (s->conv_w32), so
     * every reader of a narrow-stored conv1d tensor is row-wise now,
     * not just the fixed tier's. */
    if (t->dtype == LZ_DT_BF16 && lz_bf16_store_g &&
        ((t->n_dims == 2 && t->shape[1] <= LZ_MM_WIDEN_MAX) ||
         (t->n_dims == 1 && t->n_elem <= LZ_MM_WIDEN_MAX) ||
         (t->n_dims == 3 && t->shape[2] <= LZ_MM_WIDEN_MAX))) {
        lz_i64 nb = t->n_elem * (lz_i64)2;
        field->q = (int8_t *)aligned_malloc((size_t)t->n_elem, 2);
        if (!field->q) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC_BYTES, t->name, nb);
            return 1;
        }
        if (lz_st_read_raw(&m->st, t, field->q, errbuf, errlen) != 0) {
            aligned_free(field->q);
            field->q = NULL;
            return 1;
        }
        /* safetensors is little-endian by specification; the widening in
           lz_t_f32 reads the two bytes by name, so no swap is needed
           here and none would be correct on a big-endian host either. */
        field->dtype = LZ_FMT_BF16;
        m->bytes_alloc += nb;
        return 0;
    }

    /* Expanding to f32 needs 4*elem bytes; f16 raw data is read into
       the second half and expanded in place, so this buffer doubles as
       the read buffer - no extra scratch memory. */
    field->f = (float *)aligned_malloc((size_t)t->n_elem, sizeof(float));
    if (!field->f) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC_BYTES,
             t->name, t->n_elem * (lz_i64)sizeof(float));
        return 1;
    }
    if (lz_st_read_f32(&m->st, t, field->f, t->n_elem, errbuf, errlen) != 0) {
        aligned_free(field->f);
        field->f = NULL;
        return 1;
    }
    m->bytes_alloc += t->n_elem * (lz_i64)sizeof(float);
    return 0;
}

/* Read one quantized tensor's group size and vet it. Non-zero on either
   a short read or a value this format cannot use; the caller reports
   both as LZ_ERR_TENSOR_GS, which is what it did when the read and the
   checks shared one `if`.

   They cannot share one any more: the value has to be converted from
   little-endian BETWEEN being read and being divided by, and on a
   big-endian host an unconverted group size is not merely wrong but
   usually enormous, so `n % gs` would silently pass the "divides evenly"
   test by never dividing at all.

   mul32 says whether this format also requires a multiple of 32 - the
   sub-byte packings fix their sub-blocks at 32 elements (see model.h),
   Q8_0 does not. */
static size_t lz4_reader(void *ctx, void *dst, size_t sz);

static int m_uses_lz4(const LZModel *m) {
    return m->rd == lz4_reader;
}

static int read_gs(LZModel *m, uint32_t n, uint32_t *gs, int mul32) {
    if (m->rd(m->rd_ctx, gs, 4) != 4) return 1;
    lz_le32(gs, 1);
    if (*gs == 0 || (n % *gs) != 0) return 1;
    if (mul32 && (*gs % 32) != 0) return 1;
    return 0;
}

/* Read one tensor's header (dtype byte + LE element count). Shared by the
   bin path's tensor reads; factored out because every branch repeats it. */
static int read_tensor_hdr(LZModel *m, uint8_t *dtype, uint32_t *n) {
    if (m->rd(m->rd_ctx, dtype, 1) != 1 ||
        m->rd(m->rd_ctx, n, 4) != 4) return 1;
    return 0;
}

/* bin path: tensors are streamed from the reader in model_walk order
   (dtype 0=f32 / 1=Q8_0). All reads go through m->rd, so the same code
   streams from model.bin (file_reader) or model.bin.lz4 (lz4d_read). */
static int visit_read_bin(LZModel *m, const LZStTensor *t, LZTensor *field,
                          void *ctx, char *errbuf, int errlen) {
    uint8_t dtype;
    uint32_t n, gs;
    (void)ctx;

    /* Every multi-byte field below - the element count, the group size,
       the f32 planes and the Q16_0 int16 plane - is little-endian in the
       file and is converted after reading. The Q8_0, Q4_1, Q6_1 and T2
       code planes are NOT, and must not be: those are byte and sub-byte
       packings whose unpackers index bits within a byte, so their bytes
       are already in the only order they have. dtype is one byte.
       lz_le32/lz_le16 are empty on a little-endian build. */
    if (read_tensor_hdr(m, &dtype, &n)) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
        return 1;
    }
    lz_le32(&n, 1);
    if ((lz_i64)n != t->n_elem) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_ELEMS,
             t->name, t->n_elem, n);
        return 1;
    }
    field->n = (int)n;
    field->zero = NULL;
    if (dtype == LZ_FMT_F32) {
        field->dtype = LZ_FMT_F32;
        field->gs = 0;
        field->q = NULL;
        field->scale = NULL;
        field->f = (float *)aligned_malloc((size_t)n, sizeof(float));
        if (!field->f) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->f, (size_t)n * sizeof(float))
                != (size_t)n * sizeof(float)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            aligned_free(field->f);
            field->f = NULL;
            return 1;
        }
        lz_le32(field->f, (size_t)n);
        m->bytes_alloc += (lz_i64)n * sizeof(float);
    } else if (dtype == LZ_FMT_BF16) {
        /* BF16 from bin:2 bytes per element, no scale, no groups.
           Same storage as the safetensors bf16 path in visit_read.
           lz_t_f32/lz_t_row_f32 widen at the point of use. */
        lz_i64 nb = (lz_i64)n * 2;
        field->dtype = LZ_FMT_BF16;
        field->gs = 0;
        field->f = NULL;
        field->scale = NULL;
        field->q = (int8_t *)aligned_malloc((size_t)n, 2);
        if (!field->q) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC_BYTES, t->name, nb);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n * 2) != (size_t)n * 2) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            aligned_free(field->q);
            field->q = NULL;
            return 1;
        }
        /* bf16 is bytes - no endian swap needed (same as safetensors path) */
        m->bytes_alloc += nb;
    } else if (dtype == LZ_FMT_Q16_0) {
        /* int16: n*2 bytes of data + n/gs scales. Used for in_proj_a/b -
           they produce gt, the decay rate multiplied into the SSM state
           every token; a quantization error here is not a one-shot
           perturbation but a systematic decay-rate offset. Measured:
           Q4_1 gt relative error is 20.2x the
           activation quantization noise floor, Q8_0 1.65x, Q16_0 back
           to 1.01x. */
        field->dtype = LZ_FMT_Q16_0;
        field->f = NULL;
        if (read_gs(m, n, &gs, 1) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n * 2, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n * 2) != (size_t)n * 2 ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        /* The only quantized code plane that is not bytes: Q16_0 stores
           int16 and the kernels read it through an int16_t *, so its
           halves have to be put the right way round here. */
        lz_le16(field->q, (size_t)n);
        lz_le32(field->scale, (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n * 2 +
                          (lz_i64)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_Q6_1) {
        /* two planes: n/2 bytes 4bit + n/4 bytes 2bit + n/gs scales +
           n/gs mins. Both planes live contiguously in one malloc, 4bit
           first. */
        field->dtype = LZ_FMT_Q6_1;
        field->f = NULL;
        if (read_gs(m, n, &gs, 1) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n / 2 + (size_t)n / 4, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        field->zero = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale || !field->zero) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n / 2 + (size_t)n / 4)
                != (size_t)n / 2 + (size_t)n / 4 ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs) ||
            m->rd(m->rd_ctx, field->zero, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        lz_le32(field->scale, (size_t)(n / gs));
        lz_le32(field->zero,  (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n / 2 + (lz_i64)n / 4 +
                          2 * (lz_i64)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_T2) {
        /* ternary: n/4 bytes of 2-bit codes + n/gs scales. NO mins -
           ternary is symmetric, and `zero` staying NULL is what the
           kernels branch on to use -scale as the hoisted coefficient
           instead of a stored min (see model.h). */
        field->dtype = LZ_FMT_T2;
        field->f = NULL;
        if (read_gs(m, n, &gs, 1) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n / 4, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n / 4) != (size_t)n / 4 ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        lz_le32(field->scale, (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n / 4 +
                          (lz_i64)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_Q4_1) {
        /* nibbles: n/2 bytes of data + n/gs scales + n/gs mins.
           gs must be a multiple of 32 - nibble sub-blocks are fixed at
           32 elements (see model.h). */
        field->dtype = LZ_FMT_Q4_1;
        field->f = NULL;
        if (read_gs(m, n, &gs, 1) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n / 2, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        field->zero = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale || !field->zero) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n / 2) != (size_t)n / 2 ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs) ||
            m->rd(m->rd_ctx, field->zero, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        lz_le32(field->scale, (size_t)(n / gs));
        lz_le32(field->zero,  (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n / 2 +
                          2 * (lz_i64)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_Q8_0) {
        field->dtype = LZ_FMT_Q8_0;
        field->f = NULL;
        if (read_gs(m, n, &gs, 0) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            aligned_free(field->q);
            aligned_free(field->scale);
            field->q = NULL;
            field->scale = NULL;
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n) != (size_t)n ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            aligned_free(field->q);
            aligned_free(field->scale);
            field->q = NULL;
            field->scale = NULL;
            return 1;
        }
        lz_le32(field->scale, (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n + (lz_i64)(n / gs) * sizeof(float);
    } else {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_DTYPE, t->name, dtype);
        return 1;
    }
    return 0;
}

/* Load config and all weights in one pass from model.bin */
static int lz_open_bin(LZModel *m, const char *dir, char *errbuf, int errlen);

/* Two reader implementations behind LZModel.rd/rd_ctx:
   - file_reader over bin_file (plain model.bin),
   - lz4d_read over a LZ4Stream* (model.bin.lz4, the stunt path).
   Everything the bin path reads - header fields, group sizes, tensor
   planes - goes through rd, so whether the file is compressed is
   invisible below lz_open_bin. */
static size_t file_reader(void *ctx, void *dst, size_t sz) {
    return fread(dst, 1, sz, (FILE *)ctx);
}

/* Reader-shaped adapter for lz4d_read: the LZModel reader signature takes
   void *ctx; lz4d_read wants a LZ4Stream *.

   NAMED FOR THIS FILE, not for lz4d. It was lz4d_read_void, which put it
   in lz4d.c's family - and it is not lz4d's: it is model.c's half of the
   pair with file_reader above, it is static, and m_uses_lz4 identifies
   the reader by comparing against its ADDRESS, so it could not move even
   if the name were right. */
static size_t lz4_reader(void *ctx, void *dst, size_t sz) {
    return lz4d_read((LZ4Stream *)ctx, dst, sz);
}

static void close_reader(LZModel *m) {
    if (m_uses_lz4(m)) {
        lz4d_close((LZ4Stream *)m->rd_ctx);
    } else if (m->bin_file) {
        fclose(m->bin_file);
    }
    m->bin_file = NULL;
    m->rd = NULL;
    m->rd_ctx = NULL;
}

static int lz_open_bin(LZModel *m, const char *dir, char *errbuf, int errlen) {
    char path[512];
    char magic[4];
    uint32_t ver;
    LZModelConfig *c = &m->config;
    int i;
    lz_i64 n_params = 0;

    /* Decide BY MAGIC, not extension: read model.bin's first 4 bytes; an
       LZ4 frame's magic (04 22 4d 18 LE) means the weight file is itself
       LZ4-compressed - whatever its name - and is streamed through lz4d.
       Anything else is rewound and read plain with file_reader. The
       legacy model.bin.lz4 sibling is still accepted when model.bin is
       absent (lz_open's probe already proved one of the two opens, so a
       failure here is a race and keeps LZ_ERR_BIN_OPEN's wording). */
    {
        uint8_t probe[4];
        int is_lz4 = 0;
        int got = 0;
        if (lz_lfn_path(dir, "model.bin", path, (int)sizeof path,
                        errbuf, errlen) == 0) {
            m->bin_file = fopen(path, "rb");
            if (!m->bin_file) {
                qerr(errbuf, errlen, LZ_ERR_BIN_OPEN, path);
                return 1;
            }
            got = (int)fread(probe, 1, 4, m->bin_file) == 4;
            is_lz4 = got && (uint32_t)((uint32_t)probe[0]
                                       | ((uint32_t)probe[1] << 8)
                                       | ((uint32_t)probe[2] << 16)
                                       | ((uint32_t)probe[3] << 24))
                           == LZ4D_FRAME_MAGIC;
        } else if (lz_lfn_path(dir, "model.bin.lz4", path,
                               (int)sizeof path, errbuf, errlen) == 0) {
            is_lz4 = 1;          /* legacy sibling: must be a frame */
        } else {
            qerr(errbuf, errlen, LZ_ERR_BIN_OPEN, "model.bin(.lz4)");
            return 1;
        }
        if (is_lz4) {
            /* Declared before the statements, not beside its use: this
               tree's floor is a 1995 C89 compiler and
               build/c89_floor_gate.sh holds model.c at zero violations. */
            LZ4Stream *lz;
            if (m->bin_file) fclose(m->bin_file);
            m->bin_file = NULL;
            lz = lz4d_open(path, errbuf, errlen);
            if (!lz) return 1;
            m->rd = lz4_reader;
            m->rd_ctx = lz;
        } else {
            rewind(m->bin_file);     /* the 4-byte probe is the header's magic */
            m->rd = file_reader;
            m->rd_ctx = m->bin_file;
        }
    }

/* RDB reads BYTES and leaves them alone; RD reads 32-bit WORDS and
   converts them from the file's little-endian order.

   Every field in this header is 4 bytes wide - int32 or float32, and the
   conversion is the same for both - except the magic, which is four
   characters and is the one caller of RDB. Putting the conversion in the
   macro rather than at each of the sixty-odd call sites is not brevity:
   a field added later gets it automatically, whereas a list of explicit
   conversions is a list somebody has to remember to extend, and the
   symptom of forgetting is a plausible-looking wrong number rather than
   a failure. lz_le32 is empty on a little-endian build, so this is also
   byte-for-byte what it was on every shipping target. */
#define RDB(ptr, sz) do { \
    if (m->rd(m->rd_ctx, (ptr), (sz)) != (size_t)(sz)) { \
        qerr(errbuf, errlen, LZ_ERR_BIN_CFG_READ); \
        goto fail; \
    } } while (0)

#define RD(ptr, sz) do { \
    RDB((ptr), (sz)); \
    lz_le32((ptr), (size_t)(sz) / 4); \
    } while (0)

    RDB(magic, 4);
    if (memcmp(magic, "L98Z", 4) != 0) {
        qerr(errbuf, errlen, LZ_ERR_BIN_MAGIC);
        goto fail;
    }
    RD(&ver, 4);
    /* Two header shapes are read, and only two.

       v5 is the one layout this tree writes (tools/export_q8.py's
       BIN_VERSION): the base header, then every field any version ever
       added, written unconditionally, in the order it was added. The
       ORDER is the whole contract - neither side has a length prefix or
       a field tag, so a field moved across the boundary of a group is
       read back as a plausible wrong number rather than as a parse
       failure, and only a byte-level comparison of the two sequences
       finds it.

         base     magic "L98Z", ver, 28 fixed words, layer_types[128]
         tail (13) kda_gate_rank, kda_has_gate_lower_bound,
                   kda_gate_lower_bound, num_experts,
                   num_experts_per_token, num_shared_experts,
                   moe_intermediate_size, moe_latent_dim,
                   first_k_dense_replace, moe_renormalize,
                   moe_router_sigmoid, moe_latent_use_norm,
                   moe_shared_width
         tail  (2) mtp_n_layers, mtp_intermediate_size
         tail  (5) rope_scaling_type, rope_scaling_factor,
                   rope_scaling_orig_max, rope_scaling_beta_fast,
                   rope_scaling_beta_slow
         tail  (1) use_subn
         tail  (2) hadamard_o, hadamard_down
         tail  (2) mla_q_lora_rank, mla_o_lora_rank
         tail  (1) attn_kv_share
         appended (137) attn_sink_learned, gated_residual, ced_split,
                   csa2_on[128], ced_replay_w, csa2_compress_m,
                   csa2_topk, csa2_indexer_heads, csa2_indexer_head_dim,
                   csa2_indexer_q_lora_rank

       v4 is the published generation - github.com/lunzima/kunkun98-models
       ships a MODEL/model.bin of exactly this shape - so it is FROZEN
       here rather than retired with the rest of the ladder. It carries
       the 13-word tail and stops; the fields after it keep the defaults
       at the bottom of this function.

       Every other version is refused. They differ from v5 only in which
       tails they carry, and a shape nothing writes any more is a shape
       nothing exercises: reading it would be a compatibility branch no
       gate can reach. */
    if (ver != 4 && ver != 5) {
        qerr(errbuf, errlen, LZ_ERR_BIN_VERSION, ver);
        goto fail;
    }
    RD(&c->vocab_size, 4); RD(&c->hidden_size, 4);
    RD(&c->n_layers, 4);
    /* STRUCTURAL: this is n_layers's OWN bound, checked the instant its
       RD() lands and before any other statement in this function ever
       runs - not folded into the "derived quantities self-consistent"
       block further down (still there, unchanged, as an idempotent
       second gate - belt and suspenders, not a duplicate to remove).
       Every one of layer_types[]/csa2_on[]/csa2_mode[]/csa2_source[]/
       last_writer[] is LZ_MAX_LAYERS wide, and this loader (and
       lz_csa2_last_writer_valid, which it calls) is FULL of loops
       shaped `for (i = 0; i < c->n_layers; i++)` reading or writing one
       of those arrays - each one is safe only because n_layers cannot
       exceed their width by the time it runs. That holds by
       construction here, not by an accident of which loop sits
       textually after the single, late check - and
       twice already (report Sec2.3.1 mode-dispatch task's own C1 and
       then, one review round later, its own C-NEW-1/C-NEW-2 fixing
       Minor-item/I3 code) a NEW loop of exactly this shape landed
       BEFORE that late check, unbounded, because nothing at the point
       n_layers is first read stops it from being arbitrary. A header
       with n_layers=0x7FFFFFFF is refused HERE now, before the very
       first LZ_MAX_LAYERS-wide array is ever indexed by it anywhere
       in this function - so every loop below this point, present or
       future, inherits the bound for free instead of by remembering to
       place itself after line ~3450. The padding loop immediately below
       (`RD(c->layer_types, ...)` then `for (i = 0; i < LZ_MAX_LAYERS;
       i++) if (i >= c->n_layers ...)`) was already safe regardless -
       its own loop bound is LZ_MAX_LAYERS, and it only ever COMPARES
       against n_layers, never uses it as an index or an upper bound of
       its own - so it is unaffected by running before or after this
       check either way. */
    if (c->n_layers <= 0 || c->n_layers > LZ_MAX_LAYERS) {
        qerr(errbuf, errlen, LZ_ERR_BIN_CFG);
        goto fail;
    }
    RD(&c->seq_len, 4);
    RD(&c->n_heads, 4);    RD(&c->n_kv_heads, 4);
    RD(&c->head_dim, 4);   RD(&c->intermediate_size, 4);
    RD(&c->lin_n_k_heads, 4); RD(&c->lin_n_v_heads, 4);
    RD(&c->lin_k_head_dim, 4); RD(&c->lin_v_head_dim, 4);
    RD(&c->conv_kernel, 4);  RD(&c->n_linear_layers, 4);
    RD(&c->n_full_layers, 4); RD(&c->full_attention_interval, 4);
    RD(&c->tie_word_embeddings, 4); RD(&c->attn_output_gate, 4);
    RD(&c->rms_norm_eps, 4);
    RD(&c->rope_theta, 4); RD(&c->partial_rotary_factor, 4);
    RD(&c->attn_q_dim, 4); RD(&c->attn_qgate_dim, 4); RD(&c->attn_kv_dim, 4);
    RD(&c->lin_key_dim, 4); RD(&c->lin_value_dim, 4); RD(&c->lin_conv_dim, 4);
    RD(&c->rotary_dim, 4);
    RD(c->layer_types, LZ_MAX_LAYERS * 4);
    for (i = 0; i < LZ_MAX_LAYERS; i++) {
        if (i >= c->n_layers && c->layer_types[i] != 0) {
            qerr(errbuf, errlen, LZ_ERR_BIN_PAD);
            goto fail;
        }
    }
    /* n_full_layers/n_linear_layers are two more raw header words with
       no range check of their own - unlike n_layers, nothing above
       bounds them, and forward_state.c's alloc_csa2_ring trusts
       n_full_layers as its own `nf` with only lz_csa2_pub_overflows
       standing between an unvalidated word and xcalloc's arithmetic
       (that function's own comment has the overflow mechanics; this is
       the check that keeps its "nf <= LZ_MAX_LAYERS in production"
       premise actually true on the bin path, rather than merely
       documented). This loop is itself a `for (i = 0; i < c->n_layers;
       i++)` reading layer_types[i] - safe here for the structural
       reason n_layers's own RD() site now states: n_layers was already
       bounded to [1, LZ_MAX_LAYERS] there, well before this point, not
       because this loop happens to sit after some other check. LZ_LT_
       LINEAR and LZ_LT_KDA both count toward n_linear_layers, same as
       lz_load_config's own layer_types loop above (the JSON path's
       comment on that loop has the reasoning); an out-of-{0,1,2}
       layer_types[] entry - not yet refused at this point, the
       "unknown" check below runs later - falls into the linear side of
       this count too, which is harmless: this check only needs the two
       counts to disagree to catch a garbage n_full_layers/n_linear_
       layers pair, not to itself classify every legal value. */
    {
        int nf_count = 0, nl_count = 0;
        for (i = 0; i < c->n_layers; i++) {
            if (c->layer_types[i] == LZ_LT_FULL) nf_count++;
            else nl_count++;
        }
        if (c->n_full_layers != nf_count || c->n_linear_layers != nl_count) {
            qerr(errbuf, errlen, LZ_ERR_BIN_CFG);
            goto fail;
        }
    }
    /* The KDA gate + latent MoE tail. Both shapes carry it, and read it
       the same way, so it sits outside the branch below. */
    RD(&c->kda_gate_rank, 4);
    RD(&c->kda_has_gate_lower_bound, 4);
    RD(&c->kda_gate_lower_bound, 4);
    RD(&c->num_experts, 4);
    RD(&c->num_experts_per_token, 4);
    RD(&c->num_shared_experts, 4);
    RD(&c->moe_intermediate_size, 4);
    RD(&c->moe_latent_dim, 4);
    RD(&c->first_k_dense_replace, 4);
    RD(&c->moe_renormalize, 4);
    RD(&c->moe_router_sigmoid, 4);
    RD(&c->moe_latent_use_norm, 4);
    RD(&c->moe_shared_width, 4);
    if (c->num_experts < 0 || c->num_experts > LZ_MAX_EXPERTS ||
        c->num_experts_per_token < 0 ||
        c->num_experts_per_token > c->num_experts ||
        c->first_k_dense_replace < 0) {
        qerr(errbuf, errlen, LZ_ERR_BIN_CFG);
        goto fail;
    }
    if (ver == 5) {
        /* The rest of the v5 layout, in order and unconditionally. */
        RD(&c->mtp_n_layers, 4);
        RD(&c->mtp_intermediate_size, 4);
        RD(&c->rope_scaling_type, 4);
        RD(&c->rope_scaling_factor, 4);
        RD(&c->rope_scaling_orig_max, 4);
        RD(&c->rope_scaling_beta_fast, 4);
        RD(&c->rope_scaling_beta_slow, 4);
        RD(&c->use_subn, 4);
        RD(&c->hadamard_o, 4);
        RD(&c->hadamard_down, 4);
        RD(&c->mla_q_lora_rank, 4);
        RD(&c->mla_o_lora_rank, 4);
        RD(&c->attn_kv_share, 4);
        /* GR / learned attention sink / CSA2. Appended to the v5 tail
           rather than gated behind a version word of its own: v5 is "the
           base header plus every field any version added, written
           unconditionally, in the order it was added", so a group
           appended here stays self-describing. tools/export_q8.py's
           write_config writes this exact sequence and is the other half
           of the contract - nothing on either side carries a length
           prefix or a field tag. */
        RD(&c->attn_sink_learned, 4);
        RD(&c->gated_residual, 4);
        RD(&c->ced_split, 4);
        RD(&c->ced_kv, 4);
        /* Range-check the three scalars before anything downstream turns
           one into a shape: GR sizes the residual stream and its own
           weight rows at nr*d and takes its span from ced_split, and the
           sink flag decides which layers allocate a sinks tensor. Out of
           range they are allocation sizes, and nothing further down
           re-checks them. Reported by field name - the padding check
           below is about csa2_on and would send a reader to the wrong
           array.

           gated_residual: 0 is off, and 1 is refused rather than
           tolerated - training's GatedResidual4 raises below nr=2, there
           is no single-stream GR, and the control arm is the flag off
           rather than a degenerate nr=1. The upper bound is a guard on a
           file-supplied word, not an architecture limit (training bounds
           nr from below only); 64 is 16x the only branch count ever
           trained and keeps the widened stream nr*d at 32768 floats for
           this project's d=512.

           ced_split: the CED boundary is a layer index, so [0, n_layers]
           is its whole domain. _wire_gr reads 0 < ced_split < n_layers as
           "active" and anything else as "span the whole stack", so a
           garbage word would otherwise behave like a legal one. */
        if (c->attn_sink_learned != 0 && c->attn_sink_learned != 1) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "attn_sink_learned");
            goto fail;
        }
        if (c->gated_residual != 0 &&
            (c->gated_residual < 2 || c->gated_residual > 64)) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "gated_residual");
            goto fail;
        }
        if (c->ced_split < 0 || c->ced_split > c->n_layers) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "ced_split");
            goto fail;
        }
        /* ced_kv is an enum (0=off, 1=proj) written by the exporter,
           not a free-form scalar - "reuse" has no engine-side
           implementation (model.h's own field comment) and never
           reaches a header word, so any value past 1 is a garbage
           word rather than a legal-but-unimplemented one. */
        if (c->ced_kv != 0 && c->ced_kv != 1) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "ced_kv");
            goto fail;
        }
        RD(c->csa2_on, LZ_MAX_LAYERS * 4);
        for (i = 0; i < LZ_MAX_LAYERS; i++) {
            if (i >= c->n_layers && c->csa2_on[i] != 0) {
                qerr(errbuf, errlen, LZ_ERR_BIN_CSA2_PAD);
                goto fail;
            }
        }
        /* S4+CSA2 composition (lz_load_config's own csa2_layers loop has
           the full mechanism/reasoning - the same rule, checked here
           because this loader never calls that one). This IS the fix
           for the "bin-path never actually refuses" gap: unlike the
           safetensors path, this loader's tensor reads are POSITIONAL,
           so there is no missing-tensor lookup for a garbage or hand-
           built header to accidentally fail. mla_q_lora_rank was RD()'d
           unconditionally above (this function's own field order), so
           it is already final for every csa2_on[] entry just read.
           This loop, like the n_full_layers/n_linear_layers one above,
           is safe for the structural reason n_layers's own RD() site
           states: bounded to [1, LZ_MAX_LAYERS] there, before this
           point, not by virtue of this loop's own position. */
        if (c->mla_q_lora_rank > 0) {
            for (i = 0; i < c->n_layers; i++) {
                if (c->csa2_on[i]) {
                    qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "mla_q_lora_rank");
                    goto fail;
                }
            }
        }
        /* csa2_mode[]/csa2_source[] (model.h's own field comment on
           LZModelConfig) - appended to the v5 tail unconditionally, same
           posture as every other field in this block (ced_kv just
           above is the most recent precedent): no version word of its
           own, no length prefix, just the next fixed slot in a header
           that is "the base layout plus every field any version added,
           in the order it was added" (this function's own comment,
           above, at the ver==5 branch). tools/export_q8.py's
           write_config (E:\LLM\llama98-zh, a different repository, no
           shared git history) writes this pair now; build/csa2_last_
           writer_bin_gate.sh's own real exports depend on it having
           done so (the csa2_source[] signature search that locates
           its fixtures' bytes has nothing to find otherwise).

           THE COST, stated rather than left implicit: inserting a field
           in the MIDDLE of an unconditionally-ordered, untagged tail -
           rather than appending at the very end - desyncs every field
           read after it (ced_replay_w onward) for any model.bin whose
           export predates this commit but already carries every field
           through csa2_on. Not hypothetical: this machine has real
           (dev-fixture, non-shipping) v5 bin files in that exact state.
           Before this commit their header parsed cleanly all the way
           through and the load failed only much later, at an unrelated
           tensor-shape mismatch; after it, the load fails during header
           parsing itself, caught by the padding check just below -
           correctly refused either way, since RD() has no way to tell
           misaligned-but-in-range garbage from a legitimate value.
           (Which specific field a desync trips, absent that padding
           check, is a fact about one file's incidental byte content,
           not a property of this change - not worth naming, and not
           stable across edits to this very function, which is what
           happened to an earlier draft of this comment.) This is the
           established, accepted cost of this project's v5 format
           (every prior addition - csa2_on, attn_sink_learned,
           gated_residual, ced_replay_w, ced_kv - paid the same price
           against whatever predated IT), not a new defect this task
           introduces; the fix for a stale fixture is to re-export it,
           not to version the header. */
        RD(c->csa2_mode, LZ_MAX_LAYERS * 4);
        RD(c->csa2_source, LZ_MAX_LAYERS * 4);
        /* Two different checks for two different regions, not one loop
           applied uniformly: the padding past n_layers must be all-
           default (nothing named it, so nothing may have touched it),
           while an in-range layer's pair must be lz_csa2_mode_source_
           valid - unlike the padding check, THIS half is new: before
           it, a layer inside n_layers with a garbage csa2_mode/
           csa2_source pair was trusted outright, and forward_state.c's
           cache_idx fill (s->cache_idx[l] = s->cache_idx[c->
           csa2_source[l]] for a non-FULL mode) reads csa2_source[l] as
           an array index with no bounds check of its own - a negative
           sentinel left over from a FULL-mode default paired with a
           garbage non-zero csa2_mode reads cache_idx[-1] out of bounds,
           and a source naming a real but wrong-namespace or forward-
           referenced layer aliases into memory that index was never
           meant to reach. */
        for (i = 0; i < LZ_MAX_LAYERS; i++) {
            if (i >= c->n_layers) {
                if (c->csa2_mode[i] != 0 || c->csa2_source[i] != -1) {
                    qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE,
                         c->csa2_mode[i] != 0 ? "csa2_mode" : "csa2_source");
                    goto fail;
                }
            } else if (!lz_csa2_mode_source_valid(c, i)) {
                qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE,
                     (c->csa2_mode[i] != LZ_CSA2_FULL &&
                      c->csa2_mode[i] != LZ_CSA2_REINDEX &&
                      c->csa2_mode[i] != LZ_CSA2_REUSE) ?
                     "csa2_mode" : "csa2_source");
                goto fail;
            }
        }
        RD(&c->ced_replay_w, 4);
        RD(&c->csa2_compress_m, 4);
        RD(&c->csa2_topk, 4);
        RD(&c->csa2_indexer_heads, 4);
        RD(&c->csa2_indexer_head_dim, 4);
        RD(&c->csa2_indexer_q_lora_rank, 4);
        /* CSA2 geometry and CED replay. All six are config.json fields
           with no other route into this loader, so before these RD()
           calls the bin path ran the whole CSA2 mechanism on zeros: the
           three indexer words ARE tensor shapes (csa2_w_kv_idx and
           friends resolved to 0 elements, so the first CSA2 tensor with
           an indexer dim died with "want 0, file 16384"), csa2_compress_m
           sized buffers at the fallback 1 in forward_state.c while
           forward_csa2.c divided by the raw value, and csa2_topk is the
           top-k selection's own buffer length.

           Every bound is >= 1 because 0 is not a legal value for any of
           them and the failure mode of accepting it is a SYMBOLIC one,
           not a loud one: 0 elements of shape, 0 bytes of buffer, or a
           division by zero. Refusing them here is what keeps "a garbage
           word behaves like a legal one" from being true.

           csa2_compress_m is additionally capped at seq_len. Past that
           the arithmetic is unchanged - ceil(seq_len/cm) is 1 either way,
           so cm > seq_len compresses exactly as much as cm == seq_len,
           i.e. nothing - while the four csa2_pending_* buffers each grow
           as n_full_layers * cm * dim * ssm_ring_depth (forward.h's own
           csa2_pending_c comment has the banks). The cap costs no
           legitimate config and bounds those four allocations by
           n_full * seq_len * (LZ_SPEC_K_MAX + 1).

           csa2_topk is NOT bounded by csa2_compress_m: the two are in
           different units (blocks vs positions) and forward_csa2.c
           already clamps the selection with
           `k_eff = topk < n_vis ? topk : n_vis`, so a topk larger than
           the block count reads nothing out of range. A topk large
           enough to exhaust memory is loud through xcalloc, not silent
           - true again only because forward_state.c's alloc_csa2_ring
           checks lz_csa2_pub_overflows(nf, LZ_BATCH_MAX, csa2_topk,
           sizeof(int)) before it ever calls xcalloc: what a 32-bit
           size_t can wrap on is xcalloc's OWN arithmetic - element
           count times element size plus its alignment overhead - not
           csa2_topk alone and not the element count alone either, and
           an earlier version of this guard that checked only the
           element count still let a too-small allocation through
           silently, the report Sec2.3.1 mode-dispatch task
           (csa2_topk_idx_pub/csa2_topk_n_pub) introduced.

           The three indexer words size tensors, so a value that
           disagrees with the checkpoint is already caught by the tensor
           stream's own elem-count check; >= 1 is only here to keep 0
           from producing a shapeless tensor that the count check would
           then report as a mismatch far from its cause.

           ced_replay_w is >= 0 rather than >= 1 because 0 IS the legal
           setting - it is what every checkpoint in this tree carries and
           it means "no bounded replay". Only a negative word is refused.
           Its own consumer, lz_forward_batch's replay_boundary, reads
           `ced_replay_w > 0` and treats anything else as off, so a
           negative value would otherwise be a legal-looking "off". */
        if (c->ced_replay_w < 0) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "ced_replay_w");
            goto fail;
        }
        if (c->csa2_compress_m < 1 || c->csa2_compress_m > c->seq_len) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "csa2_compress_m");
            goto fail;
        }
        if (c->csa2_topk < 1) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "csa2_topk");
            goto fail;
        }
        if (c->csa2_indexer_heads < 1) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "csa2_indexer_heads");
            goto fail;
        }
        if (c->csa2_indexer_head_dim < 1) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "csa2_indexer_head_dim");
            goto fail;
        }
        if (c->csa2_indexer_q_lora_rank < 1) {
            qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE,
                 "csa2_indexer_q_lora_rank");
            goto fail;
        }
        if (c->rope_scaling_type < 0 || c->rope_scaling_type > 2 ||
            !(c->rope_scaling_factor > 0.0f) ||
            c->rope_scaling_beta_fast <= 0.0f ||
            c->rope_scaling_beta_slow <= 0.0f) {
            qerr(errbuf, errlen, LZ_ERR_BIN_CFG);
            goto fail;
        }
    } else {
        /* v4 stops after the tail above, so everything else keeps
           lz_open's memset(m, 0, ...) value - the same "off" defaults
           lz_load_config uses for a config.json without the key, so the
           two loading paths cannot disagree.

           hadamard_* is the exception, and the reason it needs one: 0 is
           a legal, meaningful "no Hadamard" block size, while an absent
           field means "this product predates the field, derive it from
           the dimensions". 0 cannot say that, so the sentinel -1 does -
           resolve_hadamard reads it as such. */
        c->hadamard_o = -1;
        c->hadamard_down = -1;
    }
#undef RD
#undef RDB

    /* Defense: derived quantities and key fields self-consistent (same checks as lz_load_config) */
    if (c->vocab_size <= 0 || c->hidden_size <= 0 || c->head_dim <= 0 ||
        c->intermediate_size <= 0 || c->n_layers <= 0 ||
        c->n_layers > LZ_MAX_LAYERS ||
        c->n_heads <= 0 || c->n_kv_heads <= 0 ||
        c->n_heads % c->n_kv_heads != 0 ||
        c->rotary_dim <= 0 || c->rotary_dim > c->head_dim ||
        c->rotary_dim % 2 != 0) {
        qerr(errbuf, errlen, LZ_ERR_BIN_CFG);
        goto fail;
    }

    /* The mirror of lz_load_config's own csa2_indexer_head_dim bound,
       reported with that path's message so the two loaders name the same
       defect identically: lz_rope writes base[i + rotary_dim/2] inside a
       row, and a FULL-mode CSA2 layer's indexer key row is only
       csa2_indexer_head_dim wide. csa2_on[] and csa2_mode[] are both
       read above (RD(c->csa2_on, ...) and RD(c->csa2_mode, ...)), and
       csa2_indexer_head_dim is read above this point too. */
    for (i = 0; i < c->n_layers; i++) {
        if (!c->csa2_on[i] || c->csa2_mode[i] != LZ_CSA2_FULL) continue;
        if (c->rotary_dim > c->csa2_indexer_head_dim) {
            qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_IDX_ROTARY,
                 i, c->rotary_dim, c->csa2_indexer_head_dim);
            goto fail;
        }
    }

    /* lz_csa2_last_writer_valid's own whole-config question. Its own
       loop is `for (i = 0; i < c->n_layers; i++)` against `last_writer[
       LZ_MAX_LAYERS]` and against csa2_on[]/csa2_mode[]/csa2_source[],
       each LZ_MAX_LAYERS wide - safe now unconditionally, because
       n_layers was already bounded to [1, LZ_MAX_LAYERS] the instant
       its own RD() landed (this function's own comment there has the
       full argument for WHY the check lives at that read site rather
       than here or at the "derived quantities" block just above: with
       the check anywhere else, this call, that block, and every
       c->n_layers-indexed loop in between each depend on remembering
       that ordering by hand, and relying on that is what went wrong
       twice (report Sec2.3.1 mode-dispatch task's own C1, then, one
       review round later, C-NEW-1/C-NEW-2).
       This call's own remaining ordering requirement is unrelated to
       n_layers: it must run only once every layer has already passed
       the per-layer csa2_mode/csa2_source loop above (this function's
       own documented precondition on lz_csa2_last_writer_valid) - a
       per-layer-valid csa2_source can still name a layer whose own
       contribution to their SHARED physical slot has already been
       overwritten by an intervening REINDEX layer at the same root;
       the per-layer check cannot see that, since it only ever looks at
       one layer's own fields (model.c's own comment on that function
       has the concrete example). */
    if (ver == 5 && !lz_csa2_last_writer_valid(c, NULL)) {
        qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, "csa2_source");
        goto fail;
    }

    /* Which generation this file is, decided from the layer_types[] words
       the header carries: 0 (LZ_LT_LINEAR) is GDN's linear attention, 2
       (LZ_LT_KDA) is KDA's, and 1 (gated full attention) belongs to both.
       A header with neither 0 nor 2 is a pure full-attention checkpoint,
       which is neither of the two generations this engine loads, and is
       refused here rather than half-bound against the GDN spec whose
       linear tensors it does not have.

       This is the bin path's half of the allowlist lz_load_config applies
       to safetensors (arch_probe there, from tensor names). The bin
       header is the only identity a .bin checkpoint carries - the
       config.json beside it is optional and read for engram and
       ffn_activation alone - so without this the allowlist does not reach
       a .bin model at all. */
    {
        int seen_gdn = 0, seen_kda = 0, unknown = 0;
        for (i = 0; i < c->n_layers; i++) {
            if (c->layer_types[i] == LZ_LT_KDA) seen_kda = 1;
            else if (c->layer_types[i] == LZ_LT_LINEAR) seen_gdn = 1;
            else if (c->layer_types[i] != LZ_LT_FULL) unknown = 1;
        }
        if (unknown || (!seen_gdn && !seen_kda)) {
            qerr(errbuf, errlen, LZ_ERR_CFG_UNSUPPORTED_ARCH);
            goto fail;
        }
    }

    if (check_attn_gate(c, errbuf, errlen) != 0) goto fail;

    if (resolve_hadamard(c, errbuf, errlen) != 0) goto fail;

    /* Engram fields live only in config.json, not in the bin header.
       Try to read them; absence means no engram. */
    {
        char cfg_path[1024];
        if (lz_lfn_path(dir, "config.json", cfg_path, (int)sizeof cfg_path,
                        NULL, 0) == 0) {
            size_t cfg_sz = 0;
            char *cfg_buf = (char *)lz_read_file(cfg_path, &cfg_sz, NULL, 0);
            if (cfg_buf) {
                LZJson jcfg;
                if (lz_json_parse(&jcfg, cfg_buf, cfg_sz, NULL, 0) == 0) {
                    const LZJsonNode *root = lz_json_root(&jcfg);
                    const LZJsonNode *tc = NULL;
                    if (root && root->type == LZ_JSON_OBJ)
                        tc = root;
                    if (tc) {
                        int eg_on = lz_json_get_int(&jcfg, tc, "engram", 0);
                        if (eg_on) {
                            c->engram = eg_on;
                            c->engram_ngram    = lz_json_get_int(&jcfg, tc, "engram_ngram", 3);
                            c->engram_n_head   = lz_json_get_int(&jcfg, tc, "engram_n_head", 4);
                            c->engram_dim      = lz_json_get_int(&jcfg, tc, "engram_dim", 64);
                            c->engram_size     = lz_json_get_int(&jcfg, tc, "engram_size", 1 << 22);
                            c->engram_n_orders = c->engram_ngram - 1;
                            c->engram_n_spaces = c->engram_n_orders * c->engram_n_head;
                            c->engram_per_order = c->engram_n_head * c->engram_dim;
                            c->engram_conv_dim  = c->hidden_size;
                            c->engram_conv_kernel   = 4;
                            c->engram_conv_dilation = c->engram_ngram;
                            c->engram_head_space =
                                c->engram_size / (c->engram_n_spaces * c->engram_dim);
                            if (c->engram_head_space < 256)
                                c->engram_head_space = 256;
                            {   int total = 0, found = 0, x = c->engram_head_space;
                                while (found < c->engram_n_spaces) {
                                    int is_prime = (x >= 2), d;
                                    for (d = 2; is_prime && d * d <= x; d++)
                                        if (x % d == 0) is_prime = 0;
                                    if (is_prime) { total += x; found++; }
                                    x--;
                                }
                                c->engram_total_rows = total;
                            }
                            /* Same format-gating bitmap as lz_load_config's
                               engram block; see that copy's comment. */
                            {
                                const LZJsonNode *fmt =
                                    lz_json_get(&jcfg, tc, "engram_format_ids");
                                if (fmt && fmt->type == LZ_JSON_ARR &&
                                    fmt->n_children > 0 && c->vocab_size > 0) {
                                    size_t nbytes =
                                        (size_t)(c->vocab_size + 7) / 8;
                                    unsigned char *bm =
                                        (unsigned char *)calloc(nbytes, 1);
                                    if (bm) {
                                        const LZJsonNode *e;
                                        for (e = lz_json_first(&jcfg, fmt); e;
                                             e = lz_json_next(&jcfg, e)) {
                                            int id = (int)e->num;
                                            if (id >= 0 && id < c->vocab_size)
                                                bm[id >> 3] |=
                                                    (unsigned char)(1u << (id & 7));
                                        }
                                        c->engram_format_bitmap = bm;
                                    }
                                }
                            }
                        }
                    }
                    /* ffn_activation: same sidecar-only field as engram
                       above, not a header field. Unlike engram's fields
                       this one has a real "invalid value" case
                       (parse_ffn_activation refuses an unrecognized
                       name), so a failure here must reach the caller
                       instead of the engram block's "absent means off" -
                       free what this scope allocated before goto fail,
                       since the normal exit below is skipped. */
                    if (parse_ffn_activation(&jcfg, tc, c, errbuf, errlen) != 0) {
                        lz_json_free(&jcfg);
                        free(cfg_buf);
                        goto fail;
                    }
                    lz_json_free(&jcfg);
                }
                free(cfg_buf);
            }
        }
    }

    m->layers = (LZLayer *)calloc((size_t)c->n_layers, sizeof(LZLayer));
    if (!m->layers) {
        qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
        goto fail;
    }
    if (setup_layers(m, errbuf, errlen) != 0) goto fail;

    if (alloc_mtp(m, errbuf, errlen) != 0) goto fail;
    if (alloc_engram(m, errbuf, errlen) != 0) goto fail;

    if (model_walk(m, visit_read_bin, NULL, errbuf, errlen) != 0) {
        goto fail;
    }
    /* The bin path reads weights here, during open, so this is where the
       shared k/v tensors exist to be handed out (lz_read_weights is an
       early return for a bin-loaded model). */
    apply_kv_share(m);
    for (i = 0; i < c->n_layers; i++) {
        int si;
        const LZLayer *L = &m->layers[i];
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            if (sp->layer_type >= 0 && sp->layer_type != L->type) continue;
            /* The five shared tensors are one tensor each, counted on the
               source layer: a target layer's fields are that same storage
               after apply_kv_share. Same predicate the safetensors path's
               visit_bind counts with (it sees one claimed tensor). */
            if (kv_share_field(sp->field) && lz_kv_share_target(c, i)) continue;
            n_params += ((LZTensor *)((char *)L + sp->field))->n;
        }
        if (L->ffn_moe) {
            int e;
            for (e = 0; e < c->num_experts; e++)
                n_params += L->moe_expert_w1[e].n + L->moe_expert_w2[e].n +
                            L->moe_expert_w3[e].n;
        }
        /* GR: the six weights are in the walk above, the two branch-norm
           arrays are not (model_walk's GR block). */
        if (L->gr_on) {
            int b;
            for (b = 0; b < c->gated_residual; b++)
                n_params += L->gr_attn_norms[b].n + L->gr_mlp_norms[b].n;
        }
    }
    n_params += m->embed_tokens.n + m->final_norm.n;
    if (m->mtp) {
        int si;
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            if (sp->layer_type >= 0 && sp->layer_type != LZ_LT_FULL) continue;
            if (sp->ffn_kind >= 0 && sp->ffn_kind != 0) continue;
            if (sp->present && !sp->present(c)) continue;
            n_params += ((LZTensor *)((char *)&m->mtp->blk + sp->field))->n;
        }
        n_params += m->mtp->fc.n + m->mtp->norm.n +
                    m->mtp->pre_fc_norm_hidden.n + m->mtp->pre_fc_norm_embedding.n;
    }
    if (m->engram) {
        LZEngram *eg = m->engram;
        int o;
        n_params += eg->table.n + eg->conv.n + eg->norm.n;
        for (o = 0; o < c->engram_n_orders; o++)
            n_params += eg->key_proj[o].n + eg->value_proj[o].n;
    }
    engram_norm_fixup(m);
    m->n_params = n_params;
    m->prefix[0] = '\0';
    m->weights_loaded = 1;
    close_reader(m);
    return 0;

fail:
    close_reader(m);
    lz_free(m);
    return 1;
}


void lz_t_free(LZTensor *t) {
    if (!t) return;
    aligned_free(t->f);
    aligned_free(t->q);
    aligned_free(t->scale);
    aligned_free(t->zero);
    /* Freed here, not through ops.c: this file owns the struct's memory,
       and model.c does not include ops.h. */
    free(t->sq);
    free(t->sexp);
    free(t->zq);
    free(t->zexp);
    t->f = NULL;
    t->q = NULL;
    t->scale = NULL;
    t->zero = NULL;
    t->sq = NULL;
    t->sexp = NULL;
    t->zq = NULL;
    t->zexp = NULL;
    t->sq_row = 0;
}

int lz_read_weights(LZModel *m, char *errbuf, int errlen) {
    if (!m || !m->layers) {
        qerr(errbuf, errlen, LZ_ERR_NOT_OPEN);
        return 1;
    }
    if (m->weights_loaded) return 0;
    if (model_walk(m, visit_read, NULL, errbuf, errlen) != 0) return 1;
    /* After the walk, not in lz_open's bind pass: visit_read lands the
       source layer's LZTensors here, and a target layer's five fields are
       copies of them (model.h's attn_kv_share comment). */
    apply_kv_share(m);
    engram_norm_fixup(m);
    m->weights_loaded = 1;
    return 0;
}

void lz_free(LZModel *m) {
    int li, si;
    if (!m) return;

    lz_t_free(&m->embed_tokens);
    lz_t_free(&m->final_norm);
    if (m->layers) {
        for (li = 0; li < m->config.n_layers; li++) {
            LZLayer *L = &m->layers[li];
            for (si = 0; si < N_LAYER_SPECS; si++) {
                const LZLayerSpec *sp = &LAYER_SPECS[si];
                LZTensor *field;
                if (sp->layer_type >= 0 && sp->layer_type != L->type) continue;
                /* S1 attn_kv_share: on a target layer these five fields are
                   the source layer's storage (apply_kv_share copied the
                   struct), so freeing them here would free it a second
                   time. Freed once, by the source layer's own pass. */
                if (kv_share_field(sp->field) &&
                    lz_kv_share_target(&m->config, li)) continue;
                field = (LZTensor *)((char *)L + sp->field);
                lz_t_free(field);
            }
            /* Per-expert arrays: not in LAYER_SPECS (see model_walk), so
               not covered by the loop above. free(NULL) is a no-op, so
               the three free() calls below are safe even for a non-MoE
               layer whose arrays were never allocated by setup_layers.

               INDEXING them is a different question: setup_layers does
               three separate calloc calls and checks all three, so a
               failure on the second leaves w1 set and w2 NULL. Indexing
               with a single `if (L->moe_expert_w1)` guard would then
               evaluate &((LZTensor *)0)[e] and crash on the cleanup
               path of an out-of-memory error, which is the one path a
               128MB Win98 box is most likely to take. Each array must
               therefore be checked on its own before indexing -
               free(NULL) says nothing about NULL[e]. */
            {
                int e;
                for (e = 0; e < m->config.num_experts; e++) {
                    if (L->moe_expert_w1) lz_t_free(&L->moe_expert_w1[e]);
                    if (L->moe_expert_w2) lz_t_free(&L->moe_expert_w2[e]);
                    if (L->moe_expert_w3) lz_t_free(&L->moe_expert_w3[e]);
                }
            }
            free(L->moe_expert_w1);
            free(L->moe_expert_w2);
            free(L->moe_expert_w3);
            /* GR's branch-norm arrays, same "not in LAYER_SPECS" story as
               the expert triplets above - and the six weight tensors
               themselves ARE in the walk, so they are NOT freed here. See
               model_walk's GR block. Both arrays are checked before
               indexing for the reason that comment gives. */
            if (L->gr_attn_norms) {
                int b;
                for (b = 0; b < m->config.gated_residual; b++)
                    lz_t_free(&L->gr_attn_norms[b]);
            }
            if (L->gr_mlp_norms) {
                int b;
                for (b = 0; b < m->config.gated_residual; b++)
                    lz_t_free(&L->gr_mlp_norms[b]);
            }
            free(L->gr_attn_norms);
            free(L->gr_mlp_norms);
        }
        free(m->layers);
    }
    if (m->mtp) {
        int si;
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            if (sp->layer_type >= 0 && sp->layer_type != LZ_LT_FULL) continue;
            /* Mirror the ffn_kind filter model_walk uses to bind this
               block (see the comment there): the MTP head's FFN is
               always the dense triplet, never latent MoE, so the MoE
               specs must be skipped here too. Currently a no-op either
               way - m->mtp is calloc'd and the load loop never binds
               those fields on a kunmoe checkpoint, so lz_t_free would
               just free(NULL) four times - but leaving the filter out
               here made this loop disagree with the one that fills the
               struct, which is exactly the kind of drift that let the
               load-side bug (looking for a nonexistent
               mtp.layers.0.mlp.gate.weight) go unnoticed. */
            if (sp->ffn_kind >= 0 && sp->ffn_kind != 0) continue;
            lz_t_free((LZTensor *)((char *)&m->mtp->blk + sp->field));
        }
        lz_t_free(&m->mtp->fc);
        lz_t_free(&m->mtp->norm);
        lz_t_free(&m->mtp->pre_fc_norm_hidden);
        lz_t_free(&m->mtp->pre_fc_norm_embedding);
        free(m->mtp);
    }
    if (m->engram) {
        LZEngram *eg = m->engram;
        int o;
        lz_t_free(&eg->table);
        if (eg->key_proj) {
            for (o = 0; o < m->config.engram_n_orders; o++)
                lz_t_free(&eg->key_proj[o]);
            free(eg->key_proj);
        }
        if (eg->value_proj) {
            for (o = 0; o < m->config.engram_n_orders; o++)
                lz_t_free(&eg->value_proj[o]);
            free(eg->value_proj);
        }
        lz_t_free(&eg->conv);
        lz_t_free(&eg->norm);
        free(eg->primes);
        free(eg->offsets);
        free(eg->bases);
        free(eg);
        m->engram = NULL;
    }
    free(m->config.engram_format_bitmap);
    close_reader(m);
    lz_st_close(&m->st);
    memset(m, 0, sizeof(*m));
}
