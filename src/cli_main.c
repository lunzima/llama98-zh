/* llama98 — KunMoe LLM inference engine CLI

   Usage:
     llama98 <model-dir> --check
     llama98 <model-dir> --tokens 1,2,3 --topk 5
     llama98 <model-dir> --prompt "Hello" -n 128
   Model dir must contain config.json + model.safetensors + tokenizer.json
   (or model.bin for quantized weights; loaded automatically if present).

   All output is English; column alignment uses fixed-width fields only
   (no CJK width tricks), so it stays aligned in any terminal.
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __DOS__
#include <io.h>       /* isatty, fileno - see console_is_tty */
#endif

#include "compat.h"
#include "forward.h"
#include "chat.h"
#include "cli_attr.h" /* console attributes for the styles */
#include "stream.h"   /* LZStream - see console_out */
#include "session.h"  /* LZSession conversation core (Task 3) */
#include "llama_zh.h"
#include "model.h"
#include "cpucheck.h" /* lz_cpu_check - the Socket 7 floor */
#include "gbk.h"      /* console code page, see console_out */
#include "unicode.h"  /* lz_utf8_valid - the console/argv UTF-8 sniff */
#include "lfn.h"      /* lz_lfn_path - see tok_path_for below */
#include "ops.h"      /* lz_prefetch_select for --prefetch */
#include "tokenizer.h"

/* Where this run's tokenizer.json actually lives (src/lfn.h). Shared by
 * the three call sites below so the rule cannot drift between them.
 *
 * Returns NULL with errbuf filled, so the caller reports the resolver's
 * message - which names the directory and says both forms were tried -
 * rather than a bare "cannot open file". --tokenizer bypasses this
 * entirely: second-guessing an explicit path would make the override
 * useless for pointing at a short name on purpose. */
static const char *tok_path_for(const char *dir, char *buf, int cap,
                                char *errbuf, int errlen) {
    if (lz_lfn_path(dir, "tokenizer.json", buf, cap, errbuf, errlen) != 0)
        return NULL;
    return buf;
}

/* Default context limit: the model supports up to max_position_embeddings
   (262144 on 0.8B), but KV cache is allocated for the actual need, capped
   at this practical default (~1.1KB per slot on 0.8B, ~0.3KB on s1v2).
   Override with --ctx. */
#define LZ_DEFAULT_CTX 2048

/* Investigative probes only - see forward.c's own comments
   on lz_debug_mtp_attn_scale / lz_debug_mtp_attn_window /
   lz_debug_mtp_attn_rows. Not part of the public API: no LZGenOpts
   fields for these three, deliberately not declared in any shipped
   header. (spec_debug_prefill_pos_value IS an LZGenOpts field - see
   llama_zh.h - since generate.c, not forward.c, needs to read it.) */
extern float lz_debug_mtp_attn_scale;
extern int lz_debug_mtp_attn_window;
extern long long lz_debug_mtp_attn_rows;
/* Part 4: microseconds inside each piece of a speculative
   round. Printed here rather than in the engine because iron law one
   forbids console output below cli_main.c - the engine only ever
   accumulates. */
extern long long lz_debug_us_verify;
extern long long lz_debug_us_draft;
extern long long lz_debug_us_capture;
/* Part 5: call-count companion to lz_debug_us_capture - see
   forward.c's own comment on why a removed call and a merely-fast call
   are indistinguishable without this. */
extern long long lz_debug_n_capture;
/* Part 6: timer + call-count pair for catch-up decode - see
   forward.c's own comment on lz_mtp_catchup for why it is a separate
   entry point from lz_mtp_prefill. */
extern long long lz_debug_us_catchup;
extern long long lz_debug_n_catchup;
/* Part 7: positive control for the ring-based rollback -
   defined in generate.c, not forward.c, unlike every counter above -
   see its own comment there for why. */
extern long long lz_debug_n_ring_rollback;

/* --kv-rot on|off, forward.c. On by default; off must reproduce the
   pre-rotation engine bit for bit. */
extern int lz_kv_rot_enable;
extern long long lz_debug_n_kv_rot;
/* --batch's positive control, forward.c. Prefilling n tokens at width T
   must take ceil(n/T) chunks; bit-identity alone cannot see the width. */
extern long long lz_debug_n_chunks;
/* skip_logits' positive control, forward.c. A prompt prefill must run
   lm_head once, not once per chunk. */
extern long long lz_debug_n_lmhead;
/* sink+window's row-skip positive control, forward.c. Zero when no
   window is set; non-zero only if evicted rows were really skipped. */
extern long long lz_debug_attn_skip;
extern int lz_kv_kfmt;
extern int lz_kv_vfmt;
extern int lz_attn_sink;
extern int lz_attn_window;
extern int lz_moe_topk;
extern float lz_moe_tau;
extern int lz_prof_enable;
extern double lz_prof_us[];

/* Short help: the options an END USER actually reaches for, nothing else.
   `-h` prints this; `--help` prints the full usage() below. Error paths
   print the full usage(), because a user who mistyped --kv needs to see
   its legal values, not a list that omits --kv entirely. */
static void usage_short(void) {
    printf(
    "llama98 - KunMoe LLM inference engine\n"
    "\n"
    "Usage: llama98 <model-dir> [options]\n"
    "\n"
    "Chat:\n"
    "  -i, --interactive  Multi-turn chat on stdin\n"
    "  --prompt TEXT      Single-turn prompt (bare continuation)\n"
    "  --chat             Render --prompt as a chat turn\n"
    "  -n N               Max new tokens (default 128)\n"
    "  --think            Thinking-mode defaults\n"
    "  --stop TEXT        Stop string (repeatable)\n"
    "  --ctx N            Context limit (default 2048)\n"
    "\n"
    "Sampling:\n"
    "  --temp T           Temperature (default 0.6; 0 = greedy)\n"
    "  --think-temp F     Inside <think> blocks use F instead of --temp\n"
    "                     (default off; 0 = greedy)\n"
    "  --topk K           Top-K (default 20)\n"
    "  --topp P           Nucleus (default 0.8)\n"
    "  --seed N           RNG seed (default 1; 0 = time)\n"
    "\n"
    "  --check            Validate structure only (fast)\n"
    "  -h                 This short help (--help for all options)\n"
    "\n"
    "llama98-zh - Copyright (c) 2026 Lunzima. Licensed under the\n"
    "Apache License 2.0.\n");
}

static void usage(void) {
    printf(
    "llama98 - KunMoe LLM inference engine\n"
    "\n"
    "Usage: llama98 <model-dir> [options]\n"
    "\n"
    "  -h, --help       Show the SHORT help\n"
    "\n"
    "Validation:\n"
    "  --check          Validate structure only (fast)\n"
    "  --tensors        List all tensors\n"
    "  --stats          Print tensor statistics\n"
    "  --layer N        Stats for layer N only\n"
    "  --tokens a,b,c   Forward on token ids\n"
    "  --dump-logits F  Write final logits (f32)\n"
    "  --dump-nll F     Write per-target NLL (f32, n_tokens-1 values)\n"
    "  --dump-all-logits F  Write every position's logits ((n_tokens-1) x\n"
    "                   vocab f32). Measurement only - the argmax/top-5\n"
    "                   agreement criterion (quantization-precision.md 6.2).\n"
    "  --dump-prompt-tokens F  Write the encoded prompt's token ids instead\n"
    "                   of generating (cross-build argv/encoding probe).\n"
    "  --tokens-file F  Read --tokens list from a file (long sequences)\n"
    "  --topk N         Show top-N candidates (default 5)\n"
    "\n"
    "Generation:\n"
    "  -i, --interactive  Multi-turn chat on stdin\n"
    "  --prompt TEXT    Prompt (UTF-8) - bare continuation (no chat\n"
    "                   template, no system prompt). Use --chat for a real\n"
    "                   chat turn.\n"
    "  --chat           Render --prompt as a single chat turn (system\n"
    "                   prompt + think tag).\n"
    "  -n N             Max new tokens (default 128)\n"
    "  --seed N         RNG seed (default 1; 0 = time)\n"
    "  --color M        Console attributes for Markdown styles:\n"
    "                   auto|on|off (default auto = on when a console).\n"
    "  --console C      Console code page: utf8 or gbk (default: gbk on\n"
    "                   DOS, utf8 elsewhere). Display only.\n"
    "  --temp T         Temperature (default 0.6; 0 = greedy)\n"
    "  --think-temp F   Inside <think> blocks use F instead of --temp\n"
    "                   (default off; 0 = greedy). Dynamic temperature.\n"
    "  --topk K         Top-K (default 20; <=0 off)\n"
    "  --topp P         Nucleus (default 0.8)\n"
    "  --minp P         Min-P, llama.cpp units (pre-temperature peak;\n"
    "                   default 0.05). This is the default unit.\n"
    "  --minp-vllm P    Min-P, vLLM units (post-temperature peak).\n"
    "  --presence F     Presence penalty (default 1.5)\n"
    "  --frequency F    Frequency penalty (default 0)\n"
    "  --repetition F   Repetition penalty (default 1.1)\n"
    "  --repeat-last-n N Penalty window (default 64; -1 all; 0 off)\n"
    "  --ctx N          Context limit (default 2048)\n"
    "  --batch N        Prefill batch width, 1..LZ_BATCH_MAX (default 4).\n"
    "                   Bit-identical at every width. 1 is batching off.\n"
    "  --lookahead W:D  Depth-limited lookahead, W<=4, D<=4 (default off).\n"
    "                   W<=1 is off; W:1 is the argmax control. TEST knob.\n"
    "  --lookahead-raw  Roll out without penalties.\n"
    "  --lookahead-lp F Length-penalty exponent (default 1.0).\n"
    "  --spec K         MTP speculative draft depth, 1..6 (default 0 = off).\n"
    "                   Needs an MTP head bound. Slower than 0 on amd64.\n"
    "  --p-min P        Stop drafting a round early below top-1 confidence P\n"
    "                   (default 0 = never; matches llama.cpp).\n"
    "  --n-min N        Discard the whole draft if fewer than N tokens got\n"
    "                   drafted (default 0 = never; matches llama.cpp).\n"
    "  --think          Thinking-mode defaults\n"
    "  --stop TEXT      Stop string (repeatable)\n"
    "  --no-eos         Do not stop on EOS\n"
    "  --no-ckpt        Disable cross-turn prefix reuse (-i only)\n"
    "  --kv MODE        q8|q4|f32 (default q8). q4 halves KV bytes and turns\n"
    "                   rotation on. f32 is the measuring arm, not a\n"
    "                   deployment mode.\n"
    "  --kv-k FMT       Override --kv's K format alone (same values).\n"
    "  --kv-v FMT       Override --kv's V format alone (same values).\n"
    "  --attn-sink N    StreamingLLM eviction: keep the first N positions.\n"
    "  --attn-window W  ... and the most recent W (default AUTO = max(1024,\n"
    "                   ctx/2), N=16; W=0 turns eviction off).\n"
    "  --kv-rot MODE    on|off (default OFF). Hadamard-rotate Q/K/V before\n"
    "                   KV quantization.\n"
    "  --moe-topk N     Route to N experts (default 0 = model's value).\n"
    "                   Alone it can only pick a losing arm; pair --moe-tau.\n"
    "  --moe-tau F      Router temperature, 0.1..10.0 (default 1.0 = off).\n"
    "                   Scales mixing weights; selection stays untempered.\n"
    "  --kernel TIER    auto|ref|mmx|sse2 (default auto, by CPUID).\n"
    "                   All tiers must be bit-identical; an uncompiled tier\n"
    "                   falls back loudly.\n"
    "  --recur-write TIER  auto|fixed (default auto). The recurrence state\n"
    "                   write-back tier: fixed keeps the integer result and\n"
    "                   skips the float re-quantize; auto resolves from\n"
    "                   hardware (fixed only where no SIMD write-back\n"
    "                   quantize exists).\n"
    "  --pair MODE      auto|on|off (default on). Share one weight unpack\n"
    "                   across two tokens (needs --batch >= 2).\n"
    "  --prefetch MODE  auto|none|load|nta|3dnow (default auto).\n"
    "  --profile        Print per-phase microsecond timing.\n"
    "  --tokenizer P    Override tokenizer.json\n"
    "\n"
    "Examples:\n"
    "  llama98 <model-dir> --check\n"
    "  llama98 <model-dir> --prompt \"Hello\" -n 256\n"
    "\n"
    "llama98-zh - Copyright (c) 2026 Lunzima. Licensed under the\n"
    "Apache License 2.0.\n"
#ifdef __DOS__
    "This build uses DOS/32 Advanced DOS Extender technology.\n"
    "DOS/32A - Copyright (C) 1996-2006 by Narech K.\n"
#endif
    );
}

/* KV cache format name -> LZ_KVF_*, -1 if unknown. Rotation comes on
   with any sub-8-bit format: without it 4-bit KV costs +6.7% PPL on
   English and +21.4% on Chinese, with it +0.6% and +1.7%, so shipping
   the pair unlinked would make the obvious invocation the broken one. A
   later --kv-rot off still wins, which is what keeps the comparison
   arms reachable. */
static int parse_kvfmt(const char *m) {
    if (strcmp(m, "q8") == 0)   return LZ_KVF_Q8;
    if (strcmp(m, "f32") == 0)  return LZ_KVF_F32;
    if (strcmp(m, "q4") == 0)   { lz_kv_rot_enable = 1; return LZ_KVF_Q4; }
    /* q4r2 (QJL residual sketch on keys) is not offered: it is dominated
       on this model rather than merely unproven - see forward.h's KV
       table. It spends MORE state than q8 (8.5 vs 8.6 MB is within
       noise of each other at 4017 slots, against q4's 6.1) for +0.055%
       bits/token WORSE, so it is not a knob, it is a trap. */
    return -1;
}

/* Parse "1,2,3" token list; returns count, -1 on error */
static int parse_tokens(const char *s, int **out) {
    int cap = 16, n = 0;
    int *v = (int *)malloc((size_t)cap * sizeof(int));
    const char *p = s;
    if (!v) return -1;
    while (*p) {
        char *end;
        long t;
        /* Newlines and tabs are separators too. Every normal way of
           writing a --tokens-file leaves a trailing newline:
           `echo "$ids" > f` produces "--tokens-file: parse error" while
           a Python write() of the same digits works. The error is
           reported, not silent - but a differential harness that
           redirects output and does not check the exit code reads it as
           "the two arms differ". */
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' ||
               *p == '\t') p++;
        if (!*p) break;
        t = strtol(p, &end, 10);
        if (end == p) { free(v); return -1; }
        if (n == cap) {
            int *nv = (int *)realloc(v, (size_t)(cap * 2) * sizeof(int));
            if (!nv) { free(v); return -1; }
            v = nv;
            cap *= 2;
        }
        v[n++] = (int)t;
        p = end;
    }
    *out = v;
    return n;
}

/* Single-pass top-k (k small; no full sort) */
static void print_topk(const float *logits, int n, int k) {
    int *idx;
    float *val;
    int cnt = 0, i, j;

    if (k <= 0) return;
    idx = (int *)malloc((size_t)k * sizeof(int));
    val = (float *)malloc((size_t)k * sizeof(float));
    if (!idx || !val) { free(idx); free(val); return; }
    for (i = 0; i < n; i++) {
        if (cnt < k || logits[i] > val[cnt - 1]) {
            j = (cnt < k) ? cnt : k - 1;
            while (j > 0 && val[j - 1] < logits[i]) {
                val[j] = val[j - 1];
                idx[j] = idx[j - 1];
                j--;
            }
            val[j] = logits[i];
            idx[j] = i;
            if (cnt < k) cnt++;
        }
    }
    for (i = 0; i < cnt; i++)
        printf("  #%-2d token %-8d logit %+.6f\n", i + 1, idx[i], (double)val[i]);
    free(idx);
    free(val);
}

static void print_config(const LZModelConfig *c) {
    int i;
    printf("Model:\n");
    printf("  vocab           %d\n", c->vocab_size);
    printf("  hidden          %d\n", c->hidden_size);
    printf("  layers          %d (linear %d / full %d, every %dth full)\n",
           c->n_layers, c->n_linear_layers, c->n_full_layers,
           c->full_attention_interval);
    printf("  context         %d\n", c->seq_len);
    /* WHICH ffn width is actually in use, not just the dense field.
       model.h's rule: layer li is MoE iff num_experts > 0 &&
       li >= first_k_dense_replace. So with first_k_dense_replace == 0 and
       experts present, intermediate_size applies to NO layer at all.
       This line printed it unconditionally, so `--check` on a KunMoe
       checkpoint whose every layer runs an 896-wide expert reported
       "ffn 512" - the one number that does not apply - and said nothing
       about the MoE at all. That is worse than printing nothing: it
       reads like a complete answer, and it is what sent one reader
       decoding the bin header by hand to find out whether the experts
       had been silently dropped (they had not). */
    if (c->num_experts > 0 && c->first_k_dense_replace < c->n_layers) {
        if (c->first_k_dense_replace > 0)
            printf("  ffn             dense %d (layers 0..%d) / MoE %d (layers %d..%d)\n",
                   c->intermediate_size, c->first_k_dense_replace - 1,
                   c->moe_intermediate_size, c->first_k_dense_replace,
                   c->n_layers - 1);
        else
            /* intermediate_size is NOT mentioned here, and that is the
               point of this branch existing separately. With
               first_k_dense_replace == 0 no layer has a dense FFN, and
               no dense FFN weights were exported either -
               kunmoe_modeling.py keeps the Qwen3.5 skeleton so
               config.json still carries the field, but every layer's
               mlp is experts / gate / routed_expert_{up,down}_proj /
               shared_experts and nothing else (checked on recover_r10:
               554 tensors in the checkpoint, 554 in the bin, not one a
               dense FFN).
               Printing it would mislead: every number in this block is
               supposed to describe something the model has, so printing
               one that corresponds to no tensor invites the reader to go
               looking for it. A field that does not apply belongs in
               config.json, not in a summary of what got loaded. */
            printf("  ffn             MoE %d (all layers)\n",
                   c->moe_intermediate_size);
        printf("  moe             %d experts, %d per token, %d shared"
               " (width %d) / latent %d\n",
               c->num_experts, c->num_experts_per_token,
               c->num_shared_experts, c->moe_shared_width, c->moe_latent_dim);
        printf("  moe router      %s%s%s\n",
               c->moe_router_sigmoid ? "sigmoid" : "softmax",
               c->moe_renormalize ? ", renormalized" : "",
               c->moe_latent_use_norm ? ", latent norm" : "");
    } else {
        printf("  ffn             %d\n", c->intermediate_size);
    }
    /* KDA's gate rank was equally invisible - a kunmoe checkpoint and a
       qwen3_5_text one printed identical linear-attn lines. */
    if (c->kda_gate_rank > 0)
        printf("  kda             gate rank %d%s\n", c->kda_gate_rank,
               c->kda_has_gate_lower_bound ? " [has lower bound]" : "");
    printf("  attention       %d heads / %d KV / head_dim %d%s\n",
           c->n_heads, c->n_kv_heads, c->head_dim,
           c->attn_output_gate ? " [gated]" : "");
    printf("  linear-attn     key %d x %d / value %d x %d / conv %d\n",
           c->lin_n_k_heads, c->lin_k_head_dim,
           c->lin_n_v_heads, c->lin_v_head_dim, c->conv_kernel);
    printf("  rope            theta %.0f / partial %.2f -> rotate first %d of head_dim\n",
           (double)c->rope_theta, (double)c->partial_rotary_factor,
           c->rotary_dim);
    printf("  rmsnorm eps     %g\n", (double)c->rms_norm_eps);
    printf("  tie embedding   %s\n",
           c->tie_word_embeddings ? "yes (no separate lm_head)" : "no");
    printf("  layer types     ");
    for (i = 0; i < c->n_layers; i++)
        printf("%c", c->layer_types[i] == LZ_LT_FULL ? 'A' : 'L');
    printf("  (L=linear, A=full)\n");
}

/* Streaming tensor stats: f32 direct scan; Q8 dequantize per group. */
static void tensor_stats_t(const char *name, const LZTensor *t) {
    double sum = 0.0, sumsq = 0.0, amax = 0.0;
    unsigned long long chk = 0;
    long long i, n_nan = 0;
    long long n = t ? t->n : 0;

    if (t->dtype == 0 && t->f) {
        for (i = 0; i < n; i++) {
            unsigned int u;
            double v = (double)t->f[i];
            memcpy(&u, &t->f[i], sizeof(u));
            chk += (unsigned long long)u * (unsigned long long)(i + 1);
            if (!(v == v)) { n_nan++; continue; }
            sum += v;
            sumsq += v * v;
            if (v < 0.0) v = -v;
            if (v > amax) amax = v;
        }
    } else if (t->dtype == 1 && t->q && t->scale) {
        long long g, k;
        for (g = 0; g < n / t->gs; g++) {
            double sv = (double)t->scale[g];
            for (k = 0; k < t->gs; k++) {
                double v = (double)t->q[g * t->gs + k] * sv;
                unsigned int u;
                float fv = (float)v;
                memcpy(&u, &fv, sizeof(u));
                chk += (unsigned long long)u * (unsigned long long)(g * t->gs + k + 1);
                if (!(v == v)) { n_nan++; continue; }
                sum += v;
                sumsq += v * v;
                if (v < 0.0) v = -v;
                if (v > amax) amax = v;
            }
        }
    } else {
        printf("  %-46s (not loaded)\n", name);
        return;
    }
    printf("  %-46s n=%-10lld chk=%llu sum=%+.17g rms=%.17g absmax=%.17g",
           name, n, chk, sum, sqrt(sumsq / (double)(n ? n : 1)), amax);
    if (n_nan) printf("  !! nan=%lld", n_nan);
    printf("\n");
}

/* ---------------- console code page (the DOS target) -----
 *
 * The engine speaks UTF-8 everywhere - history, tokenizer, files. A DOS
 * console with a Chinese character system resident (TechWay / twdos)
 * speaks GBK. Writing one to the other is the classic kun-jin-kao
 * mojibake: that string IS the UTF-8 encoding of U+FFFD read as GBK
 * (EF BF, BD EF, BF BD -> the three kun-jin-kao glyphs) - i.e. GBK
 * input mis-decoded as UTF-8 into replacement characters and shown on a
 * GBK screen. Both directions are wrong, so both are converted here.
 *
 * Not in the engine: iron law one keeps console I/O out of the other
 * src files and allows it only here. This is a DISPLAY layer - a code
 * point GBK cannot represent becomes '?', and that must never be
 * written back into history or the next render diverges from the KV
 * cache. Everything upstream of these two functions stays UTF-8.
 *
 * A knob, defaulted per platform rather than hardcoded (iron law nine).
 * DOS defaults on because that is the only target whose console is not
 * UTF-8-capable; anyone piping a DOS build's output into a file wants
 * --console utf8 and now has it. */
#ifdef __DOS__
#define LZ_CONSOLE_GBK_DEFAULT 1
#else
#define LZ_CONSOLE_GBK_DEFAULT 0
#endif
static int g_console_gbk = LZ_CONSOLE_GBK_DEFAULT;

/* Is stdout a console? Only DOS asks, and only DOS has the header for
   it in this translation unit; everywhere else the answer is unused
   because g_console_gbk is 0 there anyway. */
static int console_is_tty(void) {
#ifdef __DOS__
    static int known = -1;
    if (known < 0) known = isatty(fileno(stdout)) ? 1 : 0;
    return known;
#else
    return 1;
#endif
}

/* UTF-8 in, whatever the console wants out.
 *
 * STREAMING-SAFE, and it has to be: this is called once per generated
 * token, and a token boundary is not a character boundary - a three-
 * byte Chinese character routinely arrives split across two calls.
 * lz_gbk_from_utf8's `used` leaves an incomplete trailing sequence
 * unconsumed for exactly this, so the tail is held here and prepended
 * to the next chunk. Converting it immediately would emit '?' for a
 * character that is perfectly fine, one call later.
 *
 * static, not stack: iron law six rule 4, and the tail has to survive
 * between calls anyway. Single-threaded by construction - the CLI has
 * one generation at a time. */
/* The display sink LZStream drives. Styles arrive here and are ignored
   for now; the attribute mapper is what will consume them. */
/* ------------------------------------------------- table alignment
 *
 * LZStream turns a Markdown row's inner pipes into tabs and marks the
 * run LZ_STYLE_TABLE. A tab on a console advances to the next 8-column
 * stop, which aligns a table only while every cell is under eight
 * columns wide - and a Chinese cell is two columns per character, so
 * that runs out immediately.
 *
 * Real alignment needs every column's width, which is not knowable
 * until the table ends, and the stream is incremental by construction.
 * So the TABLE runs are BUFFERED here, in the front end, and emitted
 * when the style bit clears. stream.c is untouched: this is a display
 * decision and the GUI makes a different one (RichEdit tab stops), which
 * is exactly the kind of thing that belongs on this side of the sink.
 *
 * Width is counted in COLUMNS, not bytes: a GBK lead byte starts a
 * double-wide character. cli_main.c's own header says the CLI aligns
 * with fixed-width fields and no CJK width tricks - that holds for the
 * banner tables it was written about, which are ASCII. A Markdown table
 * carrying Chinese cannot be aligned without counting them, and getting
 * it wrong is visible on every row. */
#define LZ_TBL_BYTES 4096
#define LZ_TBL_COLS  16

static char g_tbl[LZ_TBL_BYTES];
static int  g_tbl_n;
static int  g_tbl_over;         /* the table outgrew the buffer */

/* Display columns of a GBK byte run. A lead byte (0x81..0xFE) followed
   by anything is one double-wide character; everything else is one
   column. Bytes are counted, not characters, so a truncated pair at the
   end still advances by one rather than looping. */
static int tbl_cols(const char *p, int n) {
    int i = 0, w = 0;
    while (i < n) {
        unsigned char c = (unsigned char)p[i];
        if (c >= 0x81 && c <= 0xFE && i + 1 < n) { w += 2; i += 2; }
        else { w += 1; i += 1; }
    }
    return w;
}

/* Emit the buffered table, padding every cell to its column's width.
   A row with more cells than the widths array holds simply stops being
   padded past that point - the text is still all there, which is the
   right failure for a display aid. */
static void tbl_flush(void) {
    int width[LZ_TBL_COLS];
    int i, c, start, col;

    if (g_tbl_n <= 0) { g_tbl_over = 0; return; }
    if (g_tbl_over) {          /* too big to align; print it as it came */
        lz_attr_write(g_tbl, g_tbl_n, LZ_STYLE_TABLE);
        g_tbl_n = 0; g_tbl_over = 0;
        return;
    }
    for (i = 0; i < LZ_TBL_COLS; i++) width[i] = 0;
    /* Pass one: the widest cell in each column. */
    start = 0; col = 0;
    for (i = 0; i <= g_tbl_n; i++) {
        int end = (i == g_tbl_n) || g_tbl[i] == '\t' || g_tbl[i] == '\n';
        if (!end) continue;
        if (col < LZ_TBL_COLS) {
            int w = tbl_cols(g_tbl + start, i - start);
            if (w > width[col]) width[col] = w;
        }
        col = (i < g_tbl_n && g_tbl[i] == '\n') ? 0 : col + 1;
        start = i + 1;
    }
    /* Pass two: write each cell, then pad it out to its column. The
       separator is two spaces so adjacent columns do not touch. */
    start = 0; col = 0;
    for (i = 0; i <= g_tbl_n; i++) {
        int end = (i == g_tbl_n) || g_tbl[i] == '\t' || g_tbl[i] == '\n';
        if (!end) continue;
        lz_attr_write(g_tbl + start, i - start, LZ_STYLE_TABLE);
        if (i < g_tbl_n && g_tbl[i] == '\t') {
            int pad = (col < LZ_TBL_COLS ? width[col] : 0)
                    - tbl_cols(g_tbl + start, i - start);
            for (c = 0; c < pad + 2; c++) lz_attr_write(" ", 1, LZ_STYLE_TABLE);
            col++;
        } else if (i < g_tbl_n) {          /* the row's newline */
            lz_attr_write("\n", 1, LZ_STYLE_TABLE);
            col = 0;
        }
        start = i + 1;
    }
    g_tbl_n = 0;
}

static void cli_stream_sink(void *ud, const char *gbk, int n, int style) {
    (void)ud;
    if (style & LZ_STYLE_TABLE) {
        if (g_tbl_n + n <= LZ_TBL_BYTES) {
            memcpy(g_tbl + g_tbl_n, gbk, (size_t)n);
            g_tbl_n += n;
        } else {
            g_tbl_over = 1;                /* flushed unaligned below */
        }
        return;
    }
    if (g_tbl_n) tbl_flush();
    lz_attr_write(gbk, n, style);
}

static void console_out(const char *bytes, int len) {
    /* One stream for the process. Single-threaded by construction - the
       CLI runs one generation at a time - and it has to persist between
       calls anyway, since a <think> tag or a UTF-8 sequence split across
       two chunks is precisely what it exists to carry. */
    static LZStream st;
    static int inited;

    if (!inited) {
        /* The display encoding is decided once, here, from the same two
           facts console_out used to branch on directly: the --console
           setting and whether stdout is still a console. A redirected
           stream stays UTF-8, the decision lz_init_stdout already makes
           when it chooses binary mode - the two have to agree or they
           disagree about what a redirected stream is.
           Measured while getting this wrong: with the tty test missing,
           `--prompt` redirected to a file came back GBK, which breaks
           every gate that reads the CLI's text output. */
        lz_stream_utf8_out(!g_console_gbk || !console_is_tty());
        lz_stream_init(&st);
        inited = 1;
    }
    /* BOTH paths go through the stream now. The hand-rolled UTF-8
       boundary buffer that used to sit here was a second implementation
       of what LZ_STREAM_PEND already does - and because the redirected
       path skipped it entirely, a </think> tag reached a piped file
       verbatim while the console never saw one. */
    lz_stream_push(&st, bytes, len, cli_stream_sink, NULL);
    fflush(stdout);
}

/* Is this already well-formed UTF-8? lz_utf8_valid in src/unicode.c is
 * the shared validator for both the console sniff and the engine's own
 * UTF-8 decoder, so the two cannot drift.
 *
 * THE DECIDING QUESTION FOR ALL INPUT, because "the console is GBK" is
 * a statement about a keyboard, not about every byte that reaches
 * argv. The conversion must not be applied unconditionally: input that
 * is ALREADY UTF-8 would be decoded as GBK and the model would receive
 * something else entirely. A DOS build is fed from more than one place
 * - a person typing at a TechWay console produces GBK, a generated DOSBox
 * [autoexec] or a batch file produces whatever encoding wrote it, and
 * a pipe produces whatever was upstream.
 *
 * Sniffing rather than trusting a flag is safe HERE and would not be
 * in general: valid multi-byte UTF-8 is a tightly constrained shape,
 * and GBK text that satisfies it by accident needs every one of its
 * lead bytes to fall in 0xC2..0xF4 with continuation bytes in
 * 0x80..0xBF - which the common Chinese range (lead 0xB0..0xF7,
 * trail 0xA1..0xFE) violates on the trail byte almost immediately.
 * Pure ASCII is valid UTF-8 and is also identical in GBK, so it takes
 * the no-conversion path and nothing is lost either way. */

/* GBK in, UTF-8 out, for one line the user typed. Returns `buf`
   unchanged when no conversion applies - already-UTF-8 input included -
   so the caller has nothing to free either way. */
static const char *console_in(char *buf, int len, char *dst, int dstcap) {
    if (!g_console_gbk) return buf;
    if (lz_utf8_valid(buf, len)) return buf;
    lz_gbk_to_utf8(buf, len, dst, dstcap, NULL);
    return dst;
}

/* Generation output callback: model bytes to the console */
static void sink_stdout(const char *bytes, int len, void *ctx) {
    (void)ctx;
    console_out(bytes, len);
}

/* The interactive loop does not accumulate a private copy of the
   reply: the LZSession's job sink accumulates the UTF-8 into the
   session's authority buffer and forwards the bytes to the caller's
   sink, which only has to print them. sink_stdout above is exactly
   that, so the interactive block passes sink_stdout to lz_session_job. */

static int parse_float(const char *s, float *out) {
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end) return -1;
    *out = (float)v;
    return 0;
}

/* The sampling block is printed by both the --prompt and --interactive
   paths and must stay identical - a drift here shows two different
   summaries for the same effective settings. One helper, two call sites. */
static void cli_print_sampling(const LZGenOpts *g) {
    printf("Sampling       temp %.2f topk %d topp %.2f minp %.2f "
           "presence %.2f freq %.2f rep %.2f win %d seed %llu\n",
           (double)g->sample.temperature, g->sample.topk,
           (double)g->sample.topp, (double)g->sample.minp,
           (double)g->sample.presence_penalty,
           (double)g->sample.frequency_penalty,
           (double)g->sample.repetition_penalty,
           g->sample.repeat_last_n, g->rng_seed);
    if (g->sample.think_temp_enabled)
        printf("DynTemp        think %.2f\n", (double)g->sample.temp_think);
}

/* Rebuild argv as UTF-8 from the OS's own copy of the command line.

   The engine's text interfaces are UTF-8 throughout, but a Windows
   process's narrow argv is converted by the CRT to the ANSI CODE PAGE.
   On this box, and on a Chinese Win98 install, that is 936 (GBK), so
   `--prompt <chinese>` from a UTF-8 shell arrives as GBK - invalid
   UTF-8, which the tokenizer turns into one U+FFFD per byte. lz_encode
   itself never overruns its caller's token buffer, but a GBK prompt is
   still garbage on arrival, which is the half that made the CLI unable
   to accept a Chinese prompt at all.

   TWO ROUTES, one per toolchain - a hardware split, not a preference.

   gcc: GetCommandLineW + CommandLineToArgvW + WideCharToMultiByte
   (CP_UTF8), below. Handles any ANSI code page, not only GBK - but
   every step of it is a W-suffixed / CP_UTF8 call, and Windows 9x's
   kernel32 is overwhelmingly stub W functions (that gap is the entire
   reason unicows.dll/MSLU exists at all). Current Microsoft Learn
   documentation lists GetCommandLineW's minimum supported client as
   Windows XP, not 95, and a contemporaneous (Win9x-era) bug report
   describes it returning nothing usable on Windows 95/98/2000.
   Correct and kept for gcc, which targets real Windows NT here; NOT
   used for Watcom, which is what ships to Win98.

   Watcom: lz_gbk_argv_convert (src/cli_argv.c/.h), gated on
   GetACP() == 936, below main(). Narrow argv IS the OS's canonical
   command line down-converted to the current ANSI code page - measured
   directly, not assumed: a one-line argv-dumping probe, built with this
   project's actual wcc386 and run with a Chinese argument from this
   dev box's shell, produced bytes matching the GBK encoding of that
   string exactly, byte for byte, not the UTF-8 the shell sent. The same
   holds even more directly on real Win98, which has no wide command
   line to down-convert from at all - the shell hands the process
   ANSI/GBK bytes from the moment they are typed. GetACP() carries none
   of the risk above: it has no A/W split (no GetACPA/GetACPW), returns
   a plain integer rather than a marshalled string, and every ANSI
   Win32 program - including Windows 9x's own components - has always
   depended on it. The CP_UTF8 route was rejected as unverifiable from
   this box.

   GetACP() != 936 (any non-Chinese locale) leaves argv untouched -
   today's long-standing behaviour, not a new failure mode. That is this
   route's one known failure mode, by design: it only knows GBK.

   Leaks the conversion by design: it lives as long as the process, and
   freeing it would mean tracking every const char * that points into it. */
#if defined(_WIN32) && !defined(__WATCOMC__)
#include <windows.h>
static char **utf8_argv(int *argc) {
    LPWSTR *wargv;
    char **out;
    int n = 0, i;

    wargv = CommandLineToArgvW(GetCommandLineW(), &n);
    if (!wargv || n <= 0) return NULL;
    out = (char **)calloc((size_t)n + 1, sizeof(char *));
    if (!out) { LocalFree(wargv); return NULL; }
    for (i = 0; i < n; i++) {
        int nb = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                     NULL, 0, NULL, NULL);
        if (nb <= 0) { LocalFree(wargv); free(out); return NULL; }
        out[i] = (char *)malloc((size_t)nb);
        if (!out[i]) { LocalFree(wargv); free(out); return NULL; }
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, out[i], nb, NULL, NULL);
    }
    LocalFree(wargv);
    *argc = n;
    return out;
}
#elif defined(_WIN32) && defined(__WATCOMC__)
#include <windows.h>
#include "cli_argv.h"
#elif defined(__DOS__)
/* Not a Windows fix-up despite the file's own header saying so: it is a
   pure GBK-to-UTF-8 pass over argv with no Win32 in it, and DOS wants
   exactly the same thing for exactly the same reason. */
#include "cli_argv.h"
#endif

int main(int argc, char **argv) {
    LZModel m;
    LZTokenizer tok;
    /* Zeroed up front so the single `cleanup:` tail below can free them
       from ANY error point, including the ones before lz_open / before
       lz_tokenizer_load ever ran. lz_free and lz_tokenizer_free are both
       safe on a zeroed struct (each frees NULL and re-zeroes), so one
       shared tail suffices for every error point. */
    memset(&m, 0, sizeof m);
    memset(&tok, 0, sizeof tok);
    /* static, not stack: iron law six, rule 4. The target is Win98 and
       this frame is the base of the CLI stack. Single-threaded, written
       by the callee before it is read, so one instance is enough. */
    static char err[1024];
    const char *dir = NULL;
    const char *prompt = NULL;
    const char *tok_path = NULL;
    const char *dump_path = NULL;
    const char *nll_path = NULL;
    const char *all_path = NULL;
    const char *prompt_tok_path = NULL;
    float *nll = NULL;
    FILE *af = NULL;    /* --dump-all-logits stream, fclosed after the loop */
    int only_check = 0, list_tensors = 0, want_stats = 0, only_layer = -1;
    int *tokens = NULL, n_tokens = 0, topk = 5;
    int n_gen = 0;                  /* -n new tokens for generation */
    int ctx_limit = LZ_DEFAULT_CTX;
    /* 0 = leave lz_state_alloc's default (LZ_BATCH_DEFAULT). A sweep
       knob, not a tuning suggestion: see LZ_BATCH_MAX in ops.h. */
    int batch_width = 0;
    int think = 0, no_eos = 0, interactive = 0, chat = 0;
    /* Which sampling fields the user set EXPLICITLY with a -- flag, as a
       bit mask. The think preset would otherwise overwrite them: it runs
       AFTER the parse loop and rewrites the whole sampling block, so a
       `--think --temp 0.9` today silently becomes 0.6 - the same defect
       the GUI fixed with manual_temp/manual_topp (common/settings.h).
       The GUI keeps one flag per preset-following field; the CLI has a
       WIDER sampling surface (--topk --minp --presence ... each
       overrides a think-preset value too), so it needs one bit per
       field rather than one flag for the two the GUI exposes. Same rule,
       carried to the larger surface: a value the user typed survives the
       preset, whatever order the flags arrived in. The LZ_MANUAL_* bits
       themselves live in sampler.h, next to lz_sample_apply_think_preset. */
    unsigned int manual = 0;
    int ckpt_on = 1;    /* --no-ckpt turns prefix reuse off (A/B knob) */
    LZGenOpts g;
    LZModelConfig *cfg;
    int i, rc = 0;
    double t0, t1;

    lz_init_stdout();
    lz_attr_mode("auto");

    /* Before anything else touches the CPU. The floor is Socket 7, and
       the two things that are actually required - CPUID and an FPU -
       are checked here so an unsupported machine gets a sentence it can
       act on instead of an invalid-opcode crash inside a kernel probe.
       Everything above the floor (MMX, SSE, SSE2, 3DNow) is optional
       and picked by lz_kernel_select. */
    if (lz_cpu_check(err, (int)sizeof(err)) != 0) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    lz_gen_opts_defaults(&g);

    /* --console IS PARSED FIRST, before the argv re-decode below and
       before the main loop. It has to be: the re-decode reads
       g_console_gbk, so a flag parsed in the ordinary place would be
       read one pass too late and silently ignored. Scanning the RAW
       argv is safe - the flag and its values are ASCII, which is the
       one thing both code pages agree on. */
    for (i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--console") != 0) continue;
        if (strcmp(argv[i + 1], "gbk") == 0)       g_console_gbk = 1;
        else if (strcmp(argv[i + 1], "utf8") == 0) g_console_gbk = 0;
    }

#if defined(_WIN32) && !defined(__WATCOMC__)
    {   /* Falling back to the CRT's narrow argv on failure is deliberate:
           a mangled non-ASCII prompt is worse than ASCII-only, but an
           exit is worse than both. */
        int wargc = 0;
        char **wargv = utf8_argv(&wargc);
        if (wargv && wargc > 0) { argc = wargc; argv = wargv; }
    }
#elif defined(_WIN32) && defined(__WATCOMC__)
    {   /* Same fallback stance as the gcc branch above: NULL (any code
           page other than 936) leaves argv exactly as the CRT built it,
           which is today's long-standing behaviour, not a regression. */
        char **conv = lz_gbk_argv_convert(GetACP(), argc, argv);
        if (conv) argv = conv;
    }
#elif defined(__DOS__)
    {   /* THE SAME CONVERSION, and it is the same problem: with a
           Chinese character system resident the bytes DOS puts in argv
           are GBK, and everything past this loop is UTF-8. Without it
           `--prompt 你好` reaches the tokenizer as four bytes that are
           not valid UTF-8, become U+FFFD, and come back out as 锟斤拷.

           936 as a constant rather than a query: DOS has no GetACP,
           and INT 21h/6601h reports the code page the OS was told
           about, which a TSR like 天汇 does not necessarily set. The
           honest statement is "this build assumes a GBK console on
           DOS", which is what LZ_CONSOLE_GBK_DEFAULT already says and
           what --console overrides. lz_gbk_argv_convert returns NULL
           for anything but 936, so this stays a single-code-page path
           by its own contract, not by accident. */
        /* Only when argv is NOT already UTF-8 - see lz_utf8_valid (and
           the console-sniff note above) for why sniffing beats trusting
           the flag here. Converting unconditionally would break input
           that arrived as UTF-8 (a generated DOSBox [autoexec], a batch
           file written by a UTF-8 shell), and the symptom is a model
           that has plainly been handed something other than what was
           typed. All-or-nothing across the whole argv, matching
           lz_gbk_argv_convert's own shape: a command line mixing the
           two encodings is not a thing any single source produces. */
        int all_utf8 = 1;
        for (i = 1; i < argc; i++)
            if (!lz_utf8_valid(argv[i], (int)strlen(argv[i]))) {
                all_utf8 = 0;
                break;
            }
        if (g_console_gbk && !all_utf8) {
            char **conv = lz_gbk_argv_convert(936, argc, argv);
            if (conv) argv = conv;
        }
    }
#endif

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        /* Consumed here too, so it is not reported as unknown - the
           value was already read by the pre-scan above. */
        if (strcmp(a, "--console") == 0 && i + 1 < argc) i++;
        else if (strcmp(a, "--color") == 0 && i + 1 < argc) {
            /* The mode actually in effect is what gets set, not the
               one asked for: "on" where stdout is not a console
               degrades to off. */
            if (!lz_attr_mode(argv[++i])) {
                printf("--color: unknown mode (auto|on|off)\n");
                return 2;
            }
        }
        else if (strcmp(a, "--check") == 0)   only_check = 1;
        else if (strcmp(a, "--tensors") == 0) list_tensors = 1;
        else if (strcmp(a, "--stats") == 0)   want_stats = 1;
        else if (strcmp(a, "--layer") == 0 && i + 1 < argc)
            only_layer = atoi(argv[++i]);
        else if (strcmp(a, "--topk") == 0 && i + 1 < argc) {
            topk = atoi(argv[++i]);             /* debug candidate count */
            g.sample.topk = topk;               /* sampling top_k too */
            manual |= LZ_MANUAL_TOPK;
        }
        else if (strcmp(a, "--dump-logits") == 0 && i + 1 < argc)
            dump_path = argv[++i];
        else if (strcmp(a, "--dump-nll") == 0 && i + 1 < argc)
            nll_path = argv[++i];
        else if (strcmp(a, "--dump-all-logits") == 0 && i + 1 < argc)
            all_path = argv[++i];
        else if (strcmp(a, "--dump-prompt-tokens") == 0 && i + 1 < argc)
            prompt_tok_path = argv[++i];
        else if (strcmp(a, "--tokens-file") == 0 && i + 1 < argc) {
            /* A 4k-token id list is ~24 KB of argv, which several shells
               and Win98's command line will not carry. Same parser, the
               text just arrives from a file. */
            size_t tsz = 0;
            char *tbuf = (char *)lz_read_file(argv[++i], &tsz, err, sizeof(err));
            char *tstr;
            if (!tbuf) { printf("--tokens-file: %s\n", err); return 2; }
            /* lz_read_file allocates EXACTLY size bytes and does not
               NUL-terminate - it exists for sized binary reads
               (safetensors, tokenizer.json, all parsed with an explicit
               length). parse_tokens walks until *p == 0, so handing it
               that buffer reads past the end.

               This bit: the first files that went through here happened
               to have a non-digit byte just past the buffer, so strtol
               stopped and the token count came out exactly right. It
               only surfaced on a 344-byte file whose trailing garbage
               made strtol fail outright. A bug that reports the correct
               answer until the allocator's mood changes. */
            tstr = (char *)malloc(tsz + 1);
            if (!tstr) { free(tbuf); printf("--tokens-file: out of memory\n"); return 2; }
            memcpy(tstr, tbuf, tsz);
            tstr[tsz] = '\0';
            free(tbuf);
            n_tokens = parse_tokens(tstr, &tokens);
            free(tstr);
            if (n_tokens <= 0) { printf("--tokens-file: parse error\n"); return 2; }
        }
        else if (strcmp(a, "--tokens") == 0 && i + 1 < argc) {
            n_tokens = parse_tokens(argv[++i], &tokens);
            if (n_tokens <= 0) {
                printf("--tokens: parse error\n");
                return 2;
            }
        }
        else if (strcmp(a, "-i") == 0 || strcmp(a, "--interactive") == 0)
            interactive = 1;
        else if (strcmp(a, "--chat") == 0)
            chat = 1;
        else if (strcmp(a, "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
            if (n_gen == 0) n_gen = 128;
        }
        else if (strcmp(a, "-n") == 0 && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        }
        else if (strcmp(a, "--seed") == 0 && i + 1 < argc) {
            g.rng_seed = strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(a, "--temp") == 0 && i + 1 < argc) {
            if (parse_float(argv[++i], &g.sample.temperature) != 0) { printf("--temp: parse error\n"); return 2; }
            manual |= LZ_MANUAL_TEMP;
        }
        else if (strcmp(a, "--think-temp") == 0 && i + 1 < argc) {
            /* Dynamic temperature, task #19: the VALUE and the ENABLE flag
               travel together under one manual bit, so the --think preset
               copy-back cannot overwrite half of a user's explicit choice
               (the value without the enable, or vice versa). */
            if (parse_float(argv[++i], &g.sample.temp_think) != 0) { printf("--think-temp: parse error\n"); return 2; }
            g.sample.think_temp_enabled = 1;
            manual |= LZ_MANUAL_THINK_TEMP;
        }
        else if (strcmp(a, "--topp") == 0 && i + 1 < argc) {
            if (parse_float(argv[++i], &g.sample.topp) != 0) { printf("--topp: parse error\n"); return 2; }
            manual |= LZ_MANUAL_TOPP;
        }
        else if (strcmp(a, "--minp") == 0 && i + 1 < argc) {
            /* llama.cpp units: fraction of the pre-temperature peak,
               converted at sample time by minp^(1/T). This is the
               DEFAULT unit (lz_sample_defaults sets minp_llamacpp). */
            if (parse_float(argv[++i], &g.sample.minp) != 0) { printf("--minp: parse error\n"); return 2; }
            g.sample.minp_llamacpp = 1;
            manual |= LZ_MANUAL_MINP;
        }
        else if (strcmp(a, "--minp-vllm") == 0 && i + 1 < argc) {
            /* Native (vLLM/OpenAI) units: fraction of the POST-temperature
               peak. Same meaning as the endpoint's min_p field. */
            if (parse_float(argv[++i], &g.sample.minp) != 0) { printf("--minp-vllm: parse error\n"); return 2; }
            g.sample.minp_llamacpp = 0;
            manual |= LZ_MANUAL_MINP;
        }
        else if (strcmp(a, "--presence") == 0 && i + 1 < argc) {
            if (parse_float(argv[++i], &g.sample.presence_penalty) != 0) { printf("--presence: parse error\n"); return 2; }
            manual |= LZ_MANUAL_PRES;
        }
        else if (strcmp(a, "--frequency") == 0 && i + 1 < argc) {
            if (parse_float(argv[++i], &g.sample.frequency_penalty) != 0) { printf("--frequency: parse error\n"); return 2; }
            manual |= LZ_MANUAL_FREQ;
        }
        else if (strcmp(a, "--repetition") == 0 && i + 1 < argc) {
            if (parse_float(argv[++i], &g.sample.repetition_penalty) != 0) { printf("--repetition: parse error\n"); return 2; }
            manual |= LZ_MANUAL_REP;
        }
        else if (strcmp(a, "--repeat-last-n") == 0 && i + 1 < argc) {
            g.sample.repeat_last_n = atoi(argv[++i]);
            manual |= LZ_MANUAL_RLAST;
        }
        else if (strcmp(a, "--spec") == 0 && i + 1 < argc) {
            g.spec_k = atoi(argv[++i]);
        }
        else if (strcmp(a, "--lookahead") == 0 && i + 1 < argc) {
            /* W:D in one argument, because the two are not independently
               meaningful - a width with no depth is a top-k pick and a
               depth with no width has nothing to compare. Bare "W" means
               W:1, which is the argmax control. */
            const char *v = argv[++i];
            const char *c = strchr(v, ':');
            g.look_width = atoi(v);
            g.look_depth = c ? atoi(c + 1) : 1;
        }
        else if (strcmp(a, "--lookahead-raw") == 0) {
            g.look_raw = 1;
        }
        else if (strcmp(a, "--lookahead-lp") == 0 && i + 1 < argc) {
            g.look_lp = (float)atof(argv[++i]);
        }
        else if (strcmp(a, "--spec-debug-break-rollback") == 0) {
            /* Test-only: see LZGenOpts.spec_debug_break_rollback
               (llama_zh.h) - deliberately corrupts conv_state on a
               rejected speculative round. Undocumented in --help on
               purpose (not a flag a normal run should ever pass). */
            g.spec_debug_break_rollback = 1;
        }
        else if (strcmp(a, "--spec-debug-skip-prefill") == 0) {
            /* Test-only: see LZGenOpts.spec_debug_skip_prefill
               (llama_zh.h) - same-binary A/B control for whether MTP
               prompt prefill changes anything measurable. Undocumented
               in --help on purpose, same reason as the flag above. */
            g.spec_debug_skip_prefill = 1;
        }
        else if (strcmp(a, "--spec-debug-prefill-pos-only") == 0) {
            /* Test-only: see LZGenOpts.spec_debug_prefill_pos_only
               (llama_zh.h) - condition "B" of the three-way A/B/C
               prefill control (advance s->mtp_pos, never write the
               MTP's own KV cache). Undocumented in --help on purpose,
               same reason as the two flags above. */
            g.spec_debug_prefill_pos_only = 1;
        }
        else if (strcmp(a, "--spec-debug-prefill-pos-value") == 0 && i + 1 < argc) {
            /* Test-only: see LZGenOpts.spec_debug_prefill_pos_value
               (llama_zh.h) - overrides the n_prompt-1 counter jump with
               an explicit value, letting the investigation test a
               starting position larger than any prompt would produce.
               Only takes effect together with --spec-debug-prefill-
               pos-only. Undocumented in --help on purpose. */
            g.spec_debug_prefill_pos_value = atoi(argv[++i]);
        }
        else if (strcmp(a, "--spec-debug-prefill-coverage") == 0 && i + 1 < argc) {
            /* Test-only: see LZGenOpts.spec_debug_prefill_coverage
               (llama_zh.h) - only prefill the last N prompt positions
               with real content, leaving the head zero-filled.
               Undocumented in --help on purpose. */
            g.spec_debug_prefill_coverage = atoi(argv[++i]);
        }
        else if (strcmp(a, "--spec-debug-skip-catchup") == 0) {
            /* Test-only: see LZGenOpts.spec_debug_skip_catchup
               (llama_zh.h) - same-binary A/B control for whether catch-
               up decode changes alpha measurably. Undocumented in
               --help on purpose, same reason as the flags above. */
            g.spec_debug_skip_catchup = 1;
        }
        else if (strcmp(a, "--spec-debug-attn-scale") == 0 && i + 1 < argc) {
            /* Investigative probe only: see forward.c's own comment on
               lz_debug_mtp_attn_scale. Undocumented in --help on
               purpose, same reason as the flags above. */
            if (parse_float(argv[++i], &lz_debug_mtp_attn_scale) != 0) {
                printf("--spec-debug-attn-scale: parse error\n"); return 2;
            }
        }
        else if (strcmp(a, "--spec-debug-attn-window") == 0 && i + 1 < argc) {
            /* Investigative probe only: see forward.c's own comment on
               lz_debug_mtp_attn_window. Undocumented in --help on
               purpose, same reason as the flags above. */
            lz_debug_mtp_attn_window = atoi(argv[++i]);
        }
        else if (strcmp(a, "--p-min") == 0 && i + 1 < argc) {
            if (parse_float(argv[++i], &g.p_min) != 0) { printf("--p-min: parse error\n"); return 2; }
        }
        else if (strcmp(a, "--n-min") == 0 && i + 1 < argc) {
            g.n_min = atoi(argv[++i]);
        }
        else if (strcmp(a, "--ctx") == 0 && i + 1 < argc) {
            ctx_limit = atoi(argv[++i]);
            if (ctx_limit < 16) ctx_limit = 16;
        }
        else if (strcmp(a, "--kernel") == 0 && i + 1 < argc) {
            /* Iron law 8 puts C / MMX / SSE / SSE2 paths on every
               operator, and the gate for that is a four-way bit-identity
               comparison. A path the CLI cannot select cannot be
               compared, so it might as well not exist - the same
               argument --prefetch's own tiers already carry. Selecting
               a tier the build does not compile in falls back to what
               it has (lz_kernel_select), so this never lies about what
               ran: --kernel-name reports the tier in effect. */
            const char *k = argv[++i];
            int want = strcmp(k, "ref") == 0  ? LZ_KERNEL_REF
                     : strcmp(k, "mmx") == 0  ? LZ_KERNEL_MMX
                     : strcmp(k, "sse2") == 0 ? LZ_KERNEL_SSE2
                     : strcmp(k, "auto") == 0 ? LZ_KERNEL_AUTO : -1;
            if (want < 0) {
                printf("--kernel: unknown tier (auto|ref|mmx|sse2)\n");
                return 2;
            }
            lz_kernel_select(want);
            /* SAY SO WHEN THE REQUEST WAS NOT HONOURED. lz_kernel_select
               silently clamps to a tier this build actually compiled,
               and that turns a four-way bit-identity check into a
               three-way one without anyone noticing - on the gcc MMX
               build `--kernel sse2` falls back to mmx (the dot32_*_sse2a
               kernels are dispatched only under __WATCOMC__, and
               kernel_detect hardcodes MMX for gcc+LZ_USE_MMX), so that
               arm compares mmx against mmx. */
            if (want != LZ_KERNEL_AUTO &&
                strcmp(argv[i], lz_kernel_name()) != 0)
                printf("[kernel: requested %s, this build has %s - "
                       "running %s]\n", argv[i], lz_build_paths(),
                       lz_kernel_tier());
        }
        else if (strcmp(a, "--recur-write") == 0 && i + 1 < argc) {
            /* State write-back tier. `fixed` keeps the recurrence's own
               integer result and skips the float re-quantize; `auto`
               resolves from hardware (fixed on a machine with no SIMD
               write-back quantize, float where the SIMD path exists). */
            const char *got = lz_gdn_p2_select(argv[++i]);
            if (!got) {
                printf("--recur-write: unknown tier (auto|fixed)\n");
                return 2;
            }
        }
        else if (strcmp(a, "--pair") == 0 && i + 1 < argc) {
            /* Which 32-element kernel the row loops call, not which ISA.
               Exists because pairing and the 128-element group kernel
               optimize different resources and the winner is per
               machine (iron law 3 / 9). Numerics-neutral by contract. */
            const char *got = lz_pair_select(argv[++i]);
            if (!got) {
                printf("--pair: unknown mode (auto|on|off)\n");
                return 2;
            }
        }
        else if (strcmp(a, "--batch") == 0 && i + 1 < argc) {
            /* Clamped rather than rejected, same as --ctx: the buffers
               are sized for LZ_BATCH_MAX and nothing above it is
               reachable, so asking for more is a sweep script running
               past the end of its range, not a user error. */
            batch_width = atoi(argv[++i]);
            if (batch_width < 1) batch_width = 1;
            if (batch_width > LZ_BATCH_MAX) batch_width = LZ_BATCH_MAX;
        }
        else if (strcmp(a, "--think") == 0) {
            think = 1;
        }
        else if (strcmp(a, "--stop") == 0 && i + 1 < argc) {
            if (g.n_stop < LZ_MAX_STOP) g.stop[g.n_stop++] = argv[++i];
        }
        else if (strcmp(a, "--no-eos") == 0) no_eos = 1;
        else if (strcmp(a, "--no-ckpt") == 0) ckpt_on = 0;
        else if (strcmp(a, "--kv") == 0 && i + 1 < argc) {
            int f = parse_kvfmt(argv[++i]);
            if (f < 0) { usage(); return 2; }
            lz_kv_kfmt = f;
            lz_kv_vfmt = f;
        }
        else if (strcmp(a, "--kv-k") == 0 && i + 1 < argc) {
            int f = parse_kvfmt(argv[++i]);
            if (f < 0) { usage(); return 2; }
            lz_kv_kfmt = f;
        }
        else if (strcmp(a, "--kv-v") == 0 && i + 1 < argc) {
            int f = parse_kvfmt(argv[++i]);
            if (f < 0) { usage(); return 2; }
            lz_kv_vfmt = f;
        }
        else if (strcmp(a, "--profile") == 0) { lz_prof_enable = 1; }
        else if (strcmp(a, "--moe-topk") == 0 && i + 1 < argc) {
            lz_moe_topk = atoi(argv[++i]);
        }
        else if (strcmp(a, "--moe-tau") == 0 && i + 1 < argc) {
            /* Refused rather than clamped, and the range is in the usage
               text: a silently clamped temperature would report a number
               in the banner that is not the one that ran, which is the
               shape iron law 4 keeps catching. */
            float t = (float)atof(argv[++i]);
            if (!(t >= LZ_MOE_TAU_MIN && t <= LZ_MOE_TAU_MAX)) {
                usage();
                return 2;
            }
            lz_moe_tau = t;
        }
        else if (strcmp(a, "--attn-sink") == 0 && i + 1 < argc) {
            /* -1 encodes "explicitly none", which is a comparison arm;
               0 means "unset" and lets the window supply a default. */
            int v = atoi(argv[++i]);
            lz_attn_sink = (v == 0) ? -1 : v;
        }
        else if (strcmp(a, "--attn-window") == 0 && i + 1 < argc) {
            lz_attn_window = atoi(argv[++i]);
        }
        /* --kv-rot-v and --qjl-dim are not offered. The V-side width
           stays at its swept value of 64 internally (forward.c's own
           comment carries the sweep and the asymmetry that explains it);
           the sketch width has nothing left to serve without q4r2. */
        else if (strcmp(a, "--kv-rot") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcmp(m, "on") == 0)       lz_kv_rot_enable = 1;
            else if (strcmp(m, "off") == 0) lz_kv_rot_enable = 0;
            else { usage(); return 2; }
        }
        else if (strcmp(a, "--prefetch") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            /* Print what was ACTUALLY selected, not what was asked for:
               nta/3dnow are downgraded on a CPU that lacks them, and a
               benchmark labelled with the wrong tier is worse than no
               benchmark. */
            const char *got = lz_prefetch_select(m);
            if (!got) { usage(); return 2; }
            printf("Prefetch       %s (requested %s)\n", got, m);
        }
        else if (strcmp(a, "--tokenizer") == 0 && i + 1 < argc)
            tok_path = argv[++i];
        else if (strcmp(a, "-h") == 0) {
            usage_short();
            return 0;
        } else if (strcmp(a, "--help") == 0) {
            usage();
            return 0;
        } else if (a[0] == '-') {
            printf("Unknown option: %s\n\n", a);
            usage();
            return 2;
        } else dir = a;
    }
    if (!dir) {
        usage();
        return 2;
    }
    if (interactive && prompt) {
        printf("--interactive and --prompt are mutually exclusive\n");
        return 2;
    }
    if (interactive && chat) {
        printf("--interactive and --chat are mutually exclusive\n");
        return 2;
    }
    if (chat && !prompt) {
        printf("--chat needs --prompt (single-turn chat render)\n");
        return 2;
    }
    if (think) {
        /* The think preset, but ONLY where the user did not set a value.
           The whole "snapshot the preset, copy back the fields the user
           claimed" rule lives in lz_sample_apply_think_preset (sampler.c),
           shared with the GUI - the `manual` mask above is the CLI's
           wider surface carried to that one function. */
        lz_sample_apply_think_preset(&g.sample, manual);
    }

    t0 = lz_time_ms();
    if (lz_open(&m, dir, err, sizeof(err)) != 0) {
        printf("Failed to load: %s\n", err);
        goto fail;
    }
    t1 = lz_time_ms();
    cfg = &m.config;

    printf("Model dir      %s\n", dir);
    printf("Tensor prefix  %s\n", m.prefix[0] ? m.prefix : "(none)");
    printf("safetensors    %lld bytes, %d tensors, data at %llu\n",
           m.st.file_size, m.st.n_tensors, m.st.data_start);
    printf("\n");
    print_config(cfg);
    printf("\n");
    printf("Params         %lld (%.1fM) text tower\n",
           m.n_params, (double)m.n_params / 1e6);
    if (m.n_params_skipped > 0)
        printf("Skipped        %lld (%.1fM) non-text (e.g. vision)\n",
               m.n_params_skipped, (double)m.n_params_skipped / 1e6);
    /* Two independent facts, both needed: what is COMPILED IN, and which
       of those the CPU picked. A build can be bit-perfect and still be
       running the scalar reference, and no output comparison can show
       it. */
    /* The third field is the JOIN of the other two, and it is the one a
       gate can accumulate across builds: neither "which impl was
       compiled" nor "which ISA the CPU picked" names an operator
       variant on its own. */
    /* "recurrence write", not "GDN pass2": the tier now covers the KDA
       write-back too (task #23), and "pass2" leaked the two-pass internal
       structure. The [impl] suffix names WHICH of the bit-identical
       bodies ran (task #27's MMX kernel, or the scalar one). Appended
       rather than substituted so the existing parsers - this file's
       gates match on "write <tier>" -
       keep matching. */
    printf("Pairing        %s, recurrence write %s[%s]\n",
           lz_pair_mode() ? "on" : "off",
           lz_gdn_p2_mode() ? (lz_gdn_p2_is_auto() ? "fixed(auto)" : "fixed")
                            : (lz_gdn_p2_is_auto() ? "float(auto)" : "float"),
           lz_gdn_p2_impl());
    printf("Kernels        %s built-in, %s selected -> tier %s\n",
           lz_build_paths(), lz_kernel_name(), lz_kernel_tier());
    printf("Structure OK   %.0f ms\n", t1 - t0);

    if (list_tensors) {
        printf("\nAll tensors:\n");
        for (i = 0; i < m.st.n_tensors; i++) {
            const LZStTensor *t = &m.st.tensors[i];
            int d;
            printf("  %-56s %-5s [", t->name, lz_st_dtype_name(t->dtype));
            for (d = 0; d < t->n_dims; d++)
                printf("%s%lld", d ? ", " : "", t->shape[d]);
            printf("]  @%llu +%llu\n", t->off_begin, t->nbytes);
        }
    }

    if (only_check) {
        goto cleanup;
    }

    printf("\nLoading weights (f32 expand ~%.2f GB)...\n",
           (double)m.n_params * 4.0 / 1e9);
    t0 = lz_time_ms();
    if (lz_read_weights(&m, err, sizeof(err)) != 0) {
        printf("Failed to load: %s\n", err);
        goto fail;
    }
    t1 = lz_time_ms();
    {
        /* A rate needs a denominator that exists. lz_time_ms moves in
           ~55 ms steps on both DOS (18.2 Hz) and a Win9x box without
           QueryPerformanceCounter - see compat.c - so a small or
           already-cached model can finish inside one tick and make
           t1 - t0 exactly zero. Printing "inf MB/s" for that is the
           trap CLAUDE.md names; saying the clock could not resolve it
           is the honest answer, and lz_common_tokcell guards the same
           way. */
        double secs = (t1 - t0) / 1000.0;
        printf("Done           %.2f GB / %.1f s", (double)m.bytes_alloc / 1e9,
               secs);
        if (secs > 0.0) printf(" = %.0f MB/s\n",
                               (double)m.bytes_alloc / 1e6 / secs);
        else            printf(" (below the clock's resolution)\n");
    }

    if (want_stats) {
        printf("\nTensor statistics (for parity with Python):\n");
        if (only_layer < 0) {
            tensor_stats_t("embed_tokens", &m.embed_tokens);
            tensor_stats_t("norm", &m.final_norm);
        }
        for (i = 0; i < cfg->n_layers; i++) {
            const LZLayer *L = &m.layers[i];
            char nm[128];
            if (only_layer >= 0 && i != only_layer) continue;
            printf("--- layer %d  %s ---\n", i, lz_layer_type_name(L->type));

#define ST(field, label) \
    do { if (L->field.n > 0) { \
             snprintf(nm, sizeof(nm), "%s", label); \
             tensor_stats_t(nm, &L->field); } } while (0)

            ST(input_layernorm, "input_layernorm.weight");
            ST(post_attention_layernorm, "post_attention_layernorm.weight");
            ST(gate_proj, "mlp.gate_proj.weight");
            ST(up_proj, "mlp.up_proj.weight");
            ST(down_proj, "mlp.down_proj.weight");
            ST(q_proj, "self_attn.q_proj.weight");
            ST(k_proj, "self_attn.k_proj.weight");
            ST(v_proj, "self_attn.v_proj.weight");
            ST(o_proj, "self_attn.o_proj.weight");
            ST(q_norm, "self_attn.q_norm.weight");
            ST(k_norm, "self_attn.k_norm.weight");
            ST(in_proj_qkv, "linear_attn.in_proj_qkv.weight");
            ST(in_proj_z, "linear_attn.in_proj_z.weight");
            ST(in_proj_a, "linear_attn.in_proj_a.weight");
            ST(in_proj_b, "linear_attn.in_proj_b.weight");
            ST(conv1d, "linear_attn.conv1d.weight");
            ST(A_log, "linear_attn.A_log");
            ST(dt_bias, "linear_attn.dt_bias");
            ST(ssm_norm, "linear_attn.norm.weight");
            ST(out_proj, "linear_attn.out_proj.weight");
#undef ST
        }
    }

    /* ---- debug: token-level forward ---- */
    if (n_tokens > 0) {
        LZRunState st;
        float *logits = NULL;
        int seq = n_tokens + (n_gen > 0 ? n_gen : 0) + 8;

        printf("\nForward: %d tokens\n", n_tokens);
        if (lz_state_alloc(&st, &m, seq, g.spec_k, err, sizeof(err)) != 0) {
            printf("State alloc failed: %s\n", err);
            goto fail;
        }
        /* Safe after alloc, never before: the buffers are sized for
           LZ_BATCH_MAX, so lowering or raising nt_cap within that bound
           cannot overrun anything. */
        if (batch_width) st.nt_cap = batch_width;
        printf("Runtime state  %.1f MB\n", (double)st.bytes_alloc / 1e6);

        t0 = lz_time_ms();
        for (i = 0; i < n_tokens; i++) {
            if (tokens[i] < 0 || tokens[i] >= cfg->vocab_size) {
                printf("Token %d out of range (vocab %d)\n", tokens[i], cfg->vocab_size);
                lz_state_free(&st);
                goto fail;
            }
            logits = lz_forward(&m, &st, tokens[i], i);
            if (all_path && i + 1 < n_tokens) {
                /* --dump-all-logits: one prediction-logits vector per
                   position that HAS a next token. lz_forward(tokens[i],
                   i) returns the logits predicting token i+1, so entry i
                   here is that prediction - index-for-index the same
                   target nll[i] scores just below. The condition
                   `i + 1 < n_tokens` drops only the final forward, which
                   predicts a token past the end and has no target; the
                   file is therefore (n_tokens-1) x vocab_size f32. The
                   help text spells this out because the first draft said
                   "every position's logits", which reads as "including
                   the last" and as "the logits OF position i" - both one
                   off from what is written, and exactly the off-by-one
                   that would silently mis-align a sec-6.2 harness. */
                if (!af) {
                    af = fopen(all_path, "wb");
                    if (!af) { printf("Cannot write %s\n", all_path); }
                }
                if (af)
                    fwrite(logits, sizeof(float), (size_t)cfg->vocab_size, af);
            }
            if (nll_path && i + 1 < n_tokens) {
                /* -log softmax(logits)[next]. Accumulated in double on
                   purpose: this is a MEASUREMENT path, not a shipped
                   one, and a 32k-term exp sum in float loses digits
                   exactly where the rare-token buckets live. It never
                   runs inside the PC=24 region (that closes with
                   lz_forward), so double is honest here - see iron law
                   six clause 3 for why that has to be checked. */
                float mx = logits[0];
                double sum = 0.0;
                int v;
                if (!nll) {
                    nll = (float *)calloc((size_t)n_tokens - 1, sizeof(float));
                    if (!nll) {
                        printf("NLL alloc failed\n");
                        lz_state_free(&st);
                        goto fail;
                    }
                }
                for (v = 1; v < cfg->vocab_size; v++)
                    if (logits[v] > mx) mx = logits[v];
                for (v = 0; v < cfg->vocab_size; v++)
                    sum += exp((double)(logits[v] - mx));
                nll[i] = (float)(log(sum) - (double)(logits[tokens[i + 1]] - mx));
            }
        }
        t1 = lz_time_ms();
        printf("Time           %.2f s, %.3f s/token\n",
               (t1 - t0) / 1000.0, (t1 - t0) / 1000.0 / n_tokens);
        printf("Top %d logits:\n", topk);
        print_topk(logits, cfg->vocab_size, topk);
        {
            LZTensor lt;
            memset(&lt, 0, sizeof(lt));
            lt.dtype = 0;
            lt.n = cfg->vocab_size;
            lt.f = logits;
            tensor_stats_t("logits", &lt);
        }

        if (dump_path) {
            FILE *f = fopen(dump_path, "wb");
            if (!f) {
                printf("Cannot write %s\n", dump_path);
            } else {
                fwrite(logits, sizeof(float), (size_t)cfg->vocab_size, f);
                fclose(f);
                printf("Wrote %s (%d f32)\n", dump_path, cfg->vocab_size);
            }
        }
        if (nll_path && nll) {
            FILE *f = fopen(nll_path, "wb");
            if (!f) {
                printf("Cannot write %s\n", nll_path);
            } else {
                double tot = 0.0;
                int v;
                fwrite(nll, sizeof(float), (size_t)n_tokens - 1, f);
                fclose(f);
                for (v = 0; v < n_tokens - 1; v++) tot += nll[v];
                printf("Wrote %s (%d f32), mean NLL %.4f\n",
                       nll_path, n_tokens - 1, tot / (n_tokens - 1));
            }
        }
        if (af) { fclose(af); af = NULL; }
        free(nll);
        nll = NULL;
        lz_state_free(&st);
        goto cleanup;
    }

    /* ---- generation: prompt -> sample ---- */
    if (prompt && !chat) {
        LZRunState st;
        /* static, same grounds as `line` in the interactive block below
           - and one more here: `tok_path = tpath` lets the pointer
           outlive nothing today, but it is a block-scoped array whose
           address escapes into a variable used further down. static
           removes both the frame cost and that hazard. */
        static char tpath[1024];
        int n_out = 0;
        double ms = 0.0;

        if (!tok_path) {
            tok_path = tok_path_for(dir, tpath, (int)sizeof tpath,
                                    err, sizeof(err));
            if (!tok_path) {
                printf("Tokenizer load failed: %s\n", err);
                goto fail;
            }
        }
        printf("\nTokenizer      %s\n", tok_path);
        t0 = lz_time_ms();
        if (lz_tokenizer_load(&tok, tok_path, err, sizeof(err)) != 0) {
            printf("Tokenizer load failed: %s\n", err);
            goto fail;
        }
        t1 = lz_time_ms();
        printf("Vocab          %d (%d special), merges %d, loaded %.0f ms\n",
               lz_tokenizer_vocab_size(&tok), lz_tokenizer_n_special(&tok),
               lz_tokenizer_n_merges(&tok), t1 - t0);

        if (!no_eos) lz_gen_opts_set_eos(&g, &tok);
        g.max_new_tokens = n_gen > 0 ? n_gen : 128;
        if (g.rng_seed == 0) g.rng_seed = lz_seed_mix((unsigned long long)t1);

        /* KV cache only covers prompt + generation; never preallocate the
           full max_position_embeddings. */
        {
            /* Count only - no buffer to size, and no buffer to get wrong.
               Sizing a buffer from the prompt's byte length would assume
               "tokens <= input bytes"; that is false (see lz_encode's
               header) and on Windows it fires every time, because MinGW
               hands main() an ANSI-converted argv, so a UTF-8 prompt from
               the shell arrives as invalid UTF-8 and expands 3x. */
            int n_prompt = lz_encode(&tok, prompt, (int)strlen(prompt),
                                     0, 0, NULL, 0);
            int seq;

            /* --dump-prompt-tokens: the cross-build argv/encoding gate
               needs the exact token id
               sequence the CHOSEN argv bytes encode to, not the
               generated text - two builds can produce visibly different
               text from the same correct ids (sampler noise) while a
               real argv-decoding bug produces the SAME ids on a
               different, wrong input. Exits here rather than falling
               through to generation: nothing past this point is part of
               what the gate checks, and skipping it keeps the gate fast
               enough to run on every build. n_prompt can be 0 (an empty
               or all-unmappable prompt) - out_cap 0 with a non-NULL out
               is a valid lz_encode call under its own count-only
               convention, so this is not special-cased. */
            if (prompt_tok_path) {
                int *pids = (int *)malloc((size_t)(n_prompt > 0
                                                    ? n_prompt : 1)
                                          * sizeof(int));
                int got = pids ? lz_encode(&tok, prompt, (int)strlen(prompt),
                                           0, 0, pids, n_prompt) : -1;
                FILE *pf;
                if (!pids || got != n_prompt) {
                    printf("--dump-prompt-tokens: re-encode mismatch "
                           "(%d vs %d)\n", got, n_prompt);
                    free(pids);
                    goto fail;
                }
                pf = fopen(prompt_tok_path, "w");
                if (!pf) {
                    printf("Cannot write %s\n", prompt_tok_path);
                    free(pids);
                    goto fail;
                }
                for (i = 0; i < n_prompt; i++) fprintf(pf, "%d\n", pids[i]);
                fclose(pf);
                free(pids);
                printf("Wrote %s (%d prompt tokens)\n", prompt_tok_path,
                       n_prompt);
                goto cleanup;
            }

            /* context = min(practical limit, model max); -n beyond ctx
               gets truncated with a note */
            if (ctx_limit > cfg->seq_len) ctx_limit = cfg->seq_len;
            if (n_prompt >= ctx_limit) {
                printf("Prompt encodes to %d tokens, exceeds context limit %d"
                       " (raise with --ctx)\n", n_prompt, ctx_limit);
                goto fail;
            }
            seq = n_prompt + g.max_new_tokens + 16;
            /* Investigative probe accommodation: this formula has no way
               to know about a debug-injected s->mtp_pos jump
               (LZGenOpts.spec_debug_prefill_pos_value), so a large
               override needs its own headroom or every later
               lz_mtp_draft_step call fails its own `pos >= s->seq_len`
               guard - loudly (good), but the guard itself is correct.
               Real user-facing runs never set this field (never touched
               outside the one investigation - see llama_zh.h's own
               comment on it), so this line changes nothing for anyone
               else. */
            if (g.spec_debug_prefill_pos_only && g.spec_debug_prefill_pos_value > 0) {
                int need = g.spec_debug_prefill_pos_value + g.max_new_tokens + 16;
                if (need > seq) seq = need;
            }
            if (seq > ctx_limit) seq = ctx_limit;
            if (seq < 64) seq = 64;
            if (lz_state_alloc(&st, &m, seq, g.spec_k, err, sizeof(err)) != 0) {
                printf("State alloc failed: %s\n", err);
                goto fail;
            }
            if (batch_width) st.nt_cap = batch_width;
            /* Through console_out, not printf. Any line that can carry
               model or user text needs the same treatment; pure ASCII
               labels and numbers do not, and are left alone rather
               than churned. */
            printf("Prompt         ");
            console_out(prompt, (int)strlen(prompt));
            console_out("\n", 1);
            printf("State          %.1f MB (%d slots, prefill batch %d)\n",
                   (double)st.bytes_alloc / 1e6, seq, st.nt_cap);
            cli_print_sampling(&g);
            printf("Context        %d slots (model max %d; --ctx to change)\n",
                   seq, cfg->seq_len);
            if (g.max_new_tokens >= seq - n_prompt)
                printf("Note: -n %d exceeds context headroom, "
                       "generation stops at %d tokens\n",
                       g.max_new_tokens, seq - n_prompt);
        }
        printf("\n---- generation start ----\n");
        rc = lz_generate(&m, &tok, &st, prompt, (int)strlen(prompt),
                         &g, sink_stdout, NULL, NULL,
                         &n_out, &ms, err, sizeof(err));
        printf("\n---- generation end ----\n");
        if (rc != 0) {
            printf("Generation failed: %s\n", err);
        } else {
            /* Head-rotations actually performed. Zero here with --kv-rot on
               means the rotation never ran, which an output comparison
               alone cannot distinguish from "ran and preserved the dot
               products" - see forward.c's lz_debug_n_kv_rot. */
            if (lz_prof_enable) {
                /* Coarse phases only. A profile's first job is to say
                   WHICH BLOCK, and on this model the answer is not the
                   one people assume. */
                static const char *pn[LZ_PROF_N] = {"attn", "linear", "ffn",
                                                    "lm_head", "norm",
                                                    "  \\_recur", "  \\_swiglu"};
                /* Only the top tier is summed - see LZ_PROF_TOP.
                   recur and swiglu are timed inside linear and ffn, so
                   a total that included them would double-count, and
                   did: it printed more microseconds than the run took.
                   They are reported against their PARENT, which is the
                   only denominator that makes a nested span mean
                   anything. */
                double tot = 0.0;
                int pi;
                for (pi = 0; pi < LZ_PROF_TOP; pi++) tot += lz_prof_us[pi];
                printf("[profile: us; top tier %% of instrumented time, "
                       "nested %% of its parent]\n");
                for (pi = 0; pi < LZ_PROF_N; pi++) {
                    double den = tot;
                    if (lz_prof_us[pi] <= 0.0) continue;
                    if (pi == LZ_PROF_REC) den = lz_prof_us[LZ_PROF_LIN];
                    if (pi == LZ_PROF_ACT) den = lz_prof_us[LZ_PROF_FFN];
                    printf("  %-8s %12.0f  %5.1f%%\n", pn[pi],
                           lz_prof_us[pi],
                           den > 0.0 ? 100.0 * lz_prof_us[pi] / den : 0.0);
                }
                printf("  %-8s %12.0f\n", "TOTAL", tot);
            }
            printf("[kv-rot: %s, k=%d v=%d, %lld head rotations]\n",
                   lz_kv_rot_enable ? "on" : "off",
                   st.kv_rot_k, st.kv_rot_v, lz_debug_n_kv_rot);
            printf("[batch: width %d, %lld forward chunks, %lld lm_head]\n",
                   st.nt_cap, lz_debug_n_chunks, lz_debug_n_lmhead);
            printf("[attn: sink %d window %d, %lld evicted rows skipped]\n",
                   st.attn_sink, st.attn_win, lz_debug_attn_skip);
            printf("\n[%d tokens, %.1f s, %.3f s/token]\n",
                   n_out, ms / 1000.0, ms / 1000.0 / (n_out > 0 ? n_out : 1));
            if (g.spec_k > 0) {
                /* alpha = accepted / draft_tokens: the on-policy
                   acceptance rate, measured through the SAME
                   lz_generate_resume path production traffic uses -
                   not a separate harness. See LZGenOpts's own comment
                   for what counts as a draft token. */
                double alpha = g.out_spec_draft_tokens > 0
                              ? (double)g.out_spec_accepted / g.out_spec_draft_tokens
                              : 0.0;
                printf("[spec: %d rounds, %d draft tokens, %d accepted, alpha=%.4f, "
                      "final mtp_pos=%d, mtp attn rows=%lld]\n",
                      g.out_spec_rounds, g.out_spec_draft_tokens,
                      g.out_spec_accepted, alpha, st.mtp_pos, lz_debug_mtp_attn_rows);
                printf("[spec us: capture=%lld draft=%lld verify=%lld"
                      " catchup=%lld (rounds=%d, n_capture=%lld,"
                      " n_catchup=%lld, n_ring_rollback=%lld)]\n",
                      lz_debug_us_capture, lz_debug_us_draft,
                      lz_debug_us_verify, lz_debug_us_catchup,
                      g.out_spec_rounds,
                      lz_debug_n_capture, lz_debug_n_catchup,
                      lz_debug_n_ring_rollback);
            }
        }
        lz_state_free(&st);
        goto cleanup;
    }

    /* ---- single-turn chat render (--chat) ----
       The GUI and --interactive feed the model through lz_chat_render,
       which injects the system prompt and the think-mode generation tail.
       --prompt does NOT: it feeds the raw bytes as a bare continuation.
       The two are different tasks - a bare "你好！" continuation and a
       chat turn produce different replies - and a CLI used to reproduce
       GUI behaviour must offer the chat path. --chat is that single turn,
       byte-identical to interactive's first turn. --prompt stays the
       continuation probe the numeric tests depend on. */
    if (chat) {
        LZRunState st;
        LZChatBuf cb;
        LZChatMsg msg;
        int n_out = 0;
        double ms = 0.0;

        static char tpath[1024];
        if (!tok_path) {
            tok_path = tok_path_for(dir, tpath, (int)sizeof tpath,
                                    err, sizeof(err));
            if (!tok_path) {
                printf("Tokenizer load failed: %s\n", err);
                goto fail;
            }
        }
        if (lz_tokenizer_load(&tok, tok_path, err, sizeof(err)) != 0) {
            printf("Tokenizer load failed: %s\n", err);
            goto fail;
        }
        if (!no_eos) lz_gen_opts_set_eos(&g, &tok);
        g.max_new_tokens = n_gen > 0 ? n_gen : 128;
        if (g.rng_seed == 0) g.rng_seed = lz_seed_mix((unsigned long long)lz_time_ms());

        {
            int seq = ctx_limit;
            if (seq > cfg->seq_len) seq = cfg->seq_len;
            if (seq < 256) seq = 256;
            if (lz_state_alloc(&st, &m, seq, g.spec_k, err, sizeof(err)) != 0) {
                printf("State alloc failed: %s\n", err);
                goto fail;
            }
            if (batch_width) st.nt_cap = batch_width;
        }

        lz_chat_buf_init(&cb);
        msg.role = LZ_ROLE_USER;
        msg.content = prompt;
        msg.len = -1;
        if (lz_chat_render(&msg, 1, 1, think, &cb, err, sizeof(err)) != 0) {
            printf("Render failed: %s\n", err);
            lz_chat_buf_free(&cb);
            lz_state_free(&st);
            goto fail;
        }

        printf("\n---- chat render (%s thinking) ----\n",
               think ? "on" : "off");
        rc = lz_generate(&m, &tok, &st, cb.s, cb.len,
                         &g, sink_stdout, NULL, NULL,
                         &n_out, &ms, err, sizeof(err));
        printf("\n---- chat end ----\n");
        if (rc != 0) printf("Generation failed: %s\n", err);

        lz_chat_buf_free(&cb);
        lz_state_free(&st);
        goto cleanup;
    }

    /* ---- interactive multi-turn chat ---- */
    if (interactive) {
        LZRunState st;
        LZSession sess;
        static char tpath[1024];       /* same grounds as `line` below */
        /* static, not stack: iron law six clause 4. main() is one frame
           at the base of a Win98 stack and already the largest in the
           tree; 8 KB of it was this. Safe as static here in a way it
           would not be in the engine - cli_main is the single-threaded
           frontend, and the DLL's thread-safety design (spec 4.5) is
           about engine state, which this is not. */
        static char line[8192];
        int seq;

        if (!tok_path) {
            tok_path = tok_path_for(dir, tpath, (int)sizeof tpath,
                                    err, sizeof(err));
            if (!tok_path) {
                printf("Tokenizer load failed: %s\n", err);
                goto fail;
            }
        }
        printf("\nTokenizer      %s\n", tok_path);
        if (lz_tokenizer_load(&tok, tok_path, err, sizeof(err)) != 0) {
            printf("Tokenizer load failed: %s\n", err);
            goto fail;
        }
        if (!no_eos) lz_gen_opts_set_eos(&g, &tok);
        g.max_new_tokens = n_gen > 0 ? n_gen : 128;
        if (g.rng_seed == 0) g.rng_seed = lz_seed_mix((unsigned long long)lz_time_ms());

        seq = ctx_limit;
        if (seq > cfg->seq_len) seq = cfg->seq_len;
        if (seq < 256) seq = 256;
        if (lz_state_alloc(&st, &m, seq, g.spec_k, err, sizeof(err)) != 0) {
            printf("State alloc failed: %s\n", err);
            goto fail;
        }
        if (batch_width) st.nt_cap = batch_width;
        printf("Context        %d slots (model max %d, prefill batch %d)\n",
               seq, cfg->seq_len, st.nt_cap);
        cli_print_sampling(&g);
        /* The session owns the conversation (history, prompt render,
           prefix cache, reply accumulation, turn trimming); it BORROWS
           the model, tokenizer and run state allocated above, and the
           CLI still frees all three at the end. */
        lz_session_init(&sess, &m, &tok, &st, think);
        /* The CLI's parsed sampling choices (think preset applied with
           the `manual` per-field overrides) live in `g`; copy the whole
           struct so the session generates with the user's numbers.
           EOS is already in `g` (set above, or left empty by --no-eos),
           so this copy carries it - there is no second set needed.
           --stop argv strings also come along in the copy. The core
           holds NO stop policy; the GUI applies its own EOS + STOPS in
           gui/session.c. The core also holds no seed policy:
           g.rng_seed is seeded once above and sess.opts.rng_seed is
           used verbatim per turn. */
        sess.opts = g;
        /* Cross-turn prefix reuse: a constant ~53 tokens per turn are
           genuinely new, the rest is re-derivation. --no-ckpt turns it
           off, which is also how the parity gate
           gets its reference output. The
           session's prefill field does NOT default to PREFIX -
           lz_session_init zeroes the struct - so the CLI must set it
           here; gui/main.c sets the same field for the GUI. */
        if (ckpt_on && lz_session_prefix_arm(&sess, &m, err, sizeof err) != 0) {
            printf("[prefix cache alloc failed: %s - reuse off]\n", err);
            ckpt_on = 0;
        }
        sess.prefill = ckpt_on ? LZ_PREFILL_PREFIX : LZ_PREFILL_FULL;
        printf("\n---- interactive chat: type a line, Ctrl-D to quit ----\n");
        for (;;) {
            int n_out, turn_rc, reused, n_prompt_tok;
            double ms = 0.0;
            /* Wall clock for the WHOLE turn, prefix preparation included.
               `ms` alone is lz_generate_resume's own span, and with reuse
               on the prefill has already happened inside
               lz_prefix_prepare by then - outside that span. A per-turn
               number that excludes it reads 0.2 s with reuse and 5.6 s
               without on the same conversation, suggesting ~30x, while
               the process wall clock says 12.16 s vs 17.57 s: 1.45x.
               A number that overstates its own feature by 20x on exactly
               the A/B it exists for is worse than no number.
               lz_session_begin (the render) is deliberately NOT timed -
               the span of the job is prefix_prepare + generate. */
            double turn_ms = 0.0;

            printf("You> "); fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) { printf("\n"); break; }
            {
                int l = (int)strlen(line);
                while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r'))
                    line[--l] = 0;
            }
            if (!line[0]) continue;
            /* GBK -> UTF-8 for what the user just typed. `line` is
               reused as the UTF-8 form so every use below is unchanged;
               UTF-8 is at most 3 bytes where GBK is 2, hence the
               separate, larger destination rather than converting in
               place. On a UTF-8 console this returns `line` untouched
               and costs a comparison. */
            {
                static char line_u8[2 * sizeof line];
                const char *conv = console_in(line, (int)strlen(line),
                                              line_u8, (int)sizeof line_u8);
                if (conv != line) {
                    strncpy(line, conv, sizeof line - 1);
                    line[sizeof line - 1] = 0;
                }
            }
            /* Two slots short of the cap, so this turn's user AND
               assistant messages both fit without a mid-turn reset. The
               session owns the history; the field is still reachable.
               lz_session_job's internal trim-and-retry is a safety net
               for a prompt that still overflows; this pre-emptive reset
               keeps the output from changing. */
            if (sess.hist.n >= LZ_CHAT_HIST_MAX - 2) {
                printf("[history full - resetting conversation]\n");
                lz_session_reset(&sess);
            }

            /* Push the user turn and render the whole conversation. */
            if (lz_session_begin(&sess, line, -1, NULL, err, sizeof err) != 0) {
                printf("[session begin failed: %s]\n", err);
                break;
            }

            printf("KunKun98> "); fflush(stdout);
            turn_ms = lz_time_ms();
            /* system=NULL: the engine's built-in identity, same as the
               single-turn --chat path. sink_stdout only prints - the
               job's own session_sink already accumulated the UTF-8 into
               sess.reply. ins=NULL: no inference inspector. */
            turn_rc = lz_session_job(&sess, NULL, sink_stdout, NULL, NULL,
                                err, sizeof err, NULL);
            printf("\n");
            if (turn_rc != 0) {
                printf("[generation failed: %s]\n", err);
                /* Propagate to the process exit status, not only to the
                   screen. `rc` at function scope is what `return rc ? 1
                   : 0` reads; without this, an interactive generation
                   failure would exit 0 and a wrapper script would see
                   success for a failed turn. */
                rc = turn_rc;
                break;
            }
            /* The job trims the oldest exchange silently when the window
               is full - say so, or the user just watches older turns
               disappear. */
            if (lz_session_trimmed(&sess) > 0)
                printf("[note: %d exchange(s) dropped - context window full]\n",
                       lz_session_trimmed(&sess));
            turn_ms = lz_time_ms() - turn_ms;
            n_out = lz_session_last_n_out(&sess);
            ms = lz_session_last_ms(&sess);
            reused = lz_session_last_reused(&sess);
            n_prompt_tok = lz_session_last_prompt_tok(&sess);
            printf("[%d tokens, %.1f s turn, %.1f s gen, %.3f s/token",
                   n_out, turn_ms / 1000.0, ms / 1000.0,
                   ms / 1000.0 / (n_out > 0 ? n_out : 1));
            if (reused > 0 && n_prompt_tok > 0)
                printf(", prompt %d reused %d", n_prompt_tok, reused);
            printf("]\n");
            /* Fold the reply back into history: normalize + push the
               assistant turn, reading the session's authority buffer. */
            if (lz_session_end(&sess, err, sizeof err) != 0) {
                printf("[history push failed: %s - turn dropped]\n", err);
            }
        }
        /* Frees what the SESSION allocated (hist, prompt, reply, pc) -
           NOT the borrowed model, tokenizer and run state below. */
        lz_session_free(&sess);
        lz_state_free(&st);
        goto cleanup;
    }

fail:
    rc = 1;
cleanup:
    lz_tokenizer_free(&tok);
    lz_free(&m);
    free(tokens);
    /* Collapse to 0/1. rc now carries an LZErr code (err.h) so that a
       library caller can map a failure onto an HTTP status without
       parsing the message text - but a PROCESS exit status is a
       different namespace, and letting engine codes leak into it would
       silently change what every wrapper script sees, for no gain: the
       CLI has already printed the message. */
    return rc ? 1 : 0;
}
