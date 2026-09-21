/* Full/Reindex/Reuse mode dispatch in forward_csa2.c (report Sec2.3.1,
   Part 5 Task 5.4 of the same plan build/csa2_modes_config_probe.c and
   build/csa2_cache_idx_alias_probe.c landed the schema/aliasing halves
   of). Those two tasks deliberately left forward_csa2.c reading nothing
   of L->csa2_mode; this is the task that makes it actually dispatch.

   Built and run by build/csa2_mode_dispatch_gate.sh against the real
   src/forward_csa2.c/forward_state.c/model.c/forward.h/model.h (not a
   transcription), and against SEVEN independent mutant copies - three
   from this task's own first landing, three more from a second,
   independent code-review round that found the first pass could not
   tell "reads the right physical ci" apart from "reads the only ci
   that exists" (every original fixture aliased every csa2_on layer to
   ONE ci) and that two of that round's own new validation predicates
   had no mutation of their own, and a seventh added by the causal-
   block-visibility fix that followed:
     - "dispatch reverted": every mode always runs the pre-this-task,
       unconditional six-step body (steps 2/3/4 lose their csa2_mode
       guards) - must redden the Reindex/Reuse-specific checks below
       while leaving check1's Full-only checks green.
     - "publish buffer collapsed to per-ci": the (ci, tk) indexing on
       csa2_topk_idx_pub/csa2_topk_n_pub loses its "+ tk" term (inside
       the per-token loop), back to a single (topk)-wide slot per ci -
       must redden specifically the per-token publish checks, since
       that is the one thing a per-ci-only buffer cannot get right for
       nt > 1.
     - "ci term dropped from the publish base pointers": a DIFFERENT
       pair of lines than the mutation above - pub_idx_base/pub_n_base
       THEMSELVES, declared once before the per-token loop, lose their
       own "+ ci * ..." term - must redden specifically check_ci_
       independence's cross-root read, since every OTHER fixture aliases
       every csa2_on layer to the same ci and cannot tell this apart
       from correct behavior.
     - "lz_csa2_pub_overflows neutered": forward_state.c's own overflow
       guard always answers "no overflow" - must redden exactly the
       must-overflow boundary assertion in check_overflow_guard.
     - "lz_csa2_mode_source_valid's source half neutered": model.c's own
       per-layer predicate's SOURCE-validity return always answers
       "valid" - must redden exactly the five bad-source assertions in
       check_bin_validation, none of the mode-code or control ones.
     - "lz_csa2_last_writer_valid neutered": model.c's own whole-config
       predicate's violation check is disabled - must redden exactly
       the one F9 whole-config assertion.
     - "finalization moved back to the end of the per-token loop": the
       whole finalize region - `if (off == cm - 1 && mode ==
       LZ_CSA2_FULL) { ... }` together with the comment introducing it -
       is cut from between step 2 and the indexer and re-inserted after
       step 6's o_proj. A move, not a substitution: no guard's text
       changes, so the other mutants' substitution greps cannot see it.
       What it covers is WHERE finalization sits, which is what makes
       n_vis = (pos + 1) / cm usable for every mode at once - ring[blk]
       has to already hold the block a query is standing on by the time
       step 5 reads it. Only an ABSOLUTE attention value can see this:
       the k_eff and selected-index assertions are direct per-position
       computations the move cannot reach (a one-candidate selection is
       index 0 whatever the scores are), and any check that compares an
       attn_out against another attn_out, or a published k_eff against
       another published k_eff, runs BOTH sides under the same
       reordering and agrees with itself whichever order is in force.
       Before the two absolute attn_out assertions check1a now carries,
       nothing in this probe looked at an absolute value.
   See that gate script's own header for the exact substitutions, the
   full expected-fallout list per mutant, and why each is a text
   mutation rather than a git-ref diff.

   No config.json, no safetensors: every LZModelConfig/LZLayer field is
   set by hand, the same shape build/csa2_cache_idx_alias_probe.c and
   build/ced_proj_arm_probe.c's PART 2 both use - unlike either of
   those, this probe calls the real forward_csa2() DIRECTLY (it is a
   non-static, externally-declared function, forward.h's own
   declaration) rather than routing through the whole per-layer
   lz_forward_batch stack: this task's change is entirely internal to
   forward_csa2.c's own per-layer, per-token bookkeeping, and calling it
   directly lets each fixture drive one layer's own multi-token chunk in
   full, explicit control of pos0/nt/kv_in, and inspect the persistent
   publish buffers afterward - exactly the granularity the design this
   task adds needs.

   Fixture weight tensors are grouped into three independent LCG
   streams per layer - "q" (q_proj/o_proj/q_norm, steps 1 and 6, every
   mode owns these), "kv" (k_norm + the four main/indexer COMPRESSION
   tensors, steps 2 and 3, FULL only) and "idx" (the three indexer
   QUERY tensors, step 4, FULL+REINDEX) - so a check can perturb exactly one
   group and attribute an observed change to exactly the step that
   reads it. A REINDEX layer is given a full "kv" group too even though
   the correct dispatch never reads it and a REAL checkpoint no longer
   carries it (model.c's CSA2_SPECS is mode-gated - build/csa2_mode_
   export_load_gate.sh is the gate that exercises a REINDEX layer's
   model.bin genuinely missing csa2_w_kv) - this fixture still gives it
   the tensor anyway, precisely so the "dispatch reverted" mutation
   above computes something (wrong, and caught) in that branch instead
   of dereferencing a genuinely absent tensor and crashing before this
   probe can print anything. build_layer_reuse_strict is the one
   deliberate exception - see its own comment and check3b's.

   csa2_compress_m=1 (every token finalizes its own block immediately)
   and csa2_topk=2 are chosen so a chunk exercises n_vis=0,1,2,... for
   whichever layer is doing the finalizing, and a real (not
   trivial-"select everything") Top-K choice as soon as n_vis exceeds
   topk. Most fixtures here run a 3-token chunk; check4b does not, and
   its own comment and F_CED_NT have the measurement for why - one token
   past the topk clamp is not yet enough for the published index SET to
   be a choice rather than a formality.

   Code-review addenda, added in two more rounds after the mode-
   dispatch checks above first landed and were reviewed:

   ROUND 2 (a Critical overflow hole and two Important gaps):
     - check_overflow_guard: lz_csa2_pub_overflows (forward_state.c)
       guards xcalloc's OWN arithmetic (element count times elem_sz
       plus its +16+sizeof(void*) overhead), not merely the element
       count nf*LZ_BATCH_MAX*csa2_topk itself - an EARLIER version of
       this function checked only the element count, which still let a
       real overflow through (nf=32, batch_max=8, topk=4194304 gives an
       element count that fits a 32-bit size_t, but that count times
       sizeof(int) is exactly 2^32, wrapping to 0 one multiplication
       later, inside xcalloc, where the element-count-only version
       could not see it). Tested with synthetic divisor values derived
       from size_t's own maximum, not from LZ_MAX_LAYERS/LZ_BATCH_MAX,
       because the real production dimensions cannot combine with an
       int-range csa2_topk to overflow a 64-bit size_t at all - see the
       function's own header comment. Also covers nf==0/batch_max==0
       (c->n_full_layers is a raw, unvalidated bin header word on the
       path that calls this).
     - check_bin_validation: lz_csa2_mode_source_valid (model.c) closes
       the gap left when build/csa2_cache_idx_alias_gate.sh's own task
       landed cache_idx aliasing without a bin-format consumer to
       validate against - lz_open_bin's own header parsing checked only
       that csa2_mode[]/csa2_source[] were untouched PAST n_layers, never
       that an IN-RANGE entry was internally consistent, so a corrupted
       or hand-crafted model.bin could alias forward_state.c's cache_idx
       fill into a negative index or the wrong numbering space. Tested
       directly against the predicate (a hand-built LZModelConfig, same
       shape as PART 1 of build/ced_proj_arm_probe.c for lz_ced_kv_target/
       lz_kv_share_target) rather than through a synthetic model.bin
       file - hand-encoding the ~1300-byte v5 header byte for byte would
       be a disproportionate amount of format plumbing for one
       predicate. "lz_open_bin calls the same predicate, so there is no
       separate real code path this could diverge from" was this
       comment's own claim once, and it is false: C1 (report Sec2.3.1
       mode-dispatch task, fifth review round) was a bug in lz_open_bin
       ITSELF - the ORDER it called lz_csa2_last_writer_valid in,
       relative to its own n_layers bound check - not in the predicate
       this file calls directly, and this file's own hand-built-
       LZModelConfig fixtures could not see it for exactly that reason.
       build/csa2_last_writer_bin_gate.sh is the gate that goes through
       the real byte stream and the real lz_open_bin call site.
     - check2's own third assertion (layer C, reuse sourced from B) and
       check_ci_independence (a whole new fixture, build_model_indep):
       the ORIGINAL round-1 checks never ran a REUSE layer whose source
       is a REINDEX layer whose OWN Top-K selection was perturbed away
       from the baseline, and never gave two csa2_on layers genuinely
       DIFFERENT physical ci values - every fixture aliased everything
       to ci=0, so a bug that stopped distinguishing ci entirely (this
       file's own mutation 3) was invisible.

   ROUND 3 (an architectural gap the PLAN itself missed, not a bug in
   this task's own round-2 fix): lz_csa2_mode_source_valid's per-layer
   question ("is csa2_source[i] a strictly earlier, csa2_on, full_
   attention layer") is necessary but not sufficient - cross-layer
   aliasing runs by shared PHYSICAL SLOT, and a layer's declared source
   can be per-layer-valid while still not being that slot's CURRENT
   last writer (model.c's own comment on lz_csa2_last_writer_valid has
   the concrete example: layer 0 FULL, layer 2 REINDEX(source 0), layer
   3 REUSE(source 0, not 2) - layer 2 already overwrote root 0's slot
   by the time layer 3 reads it). check_bin_validation's own F9-labeled
   assertions test this directly, including one that confirms the
   OLDER, per-layer-only predicate still (wrongly) accepts the same
   input the new whole-config one correctly refuses.

   Exit: 0 all checks pass, 1 a check failed (printed). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "forward.h"
#include "err.h"

static int g_fail = 0;

static void ok(int cond, const char *what) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    /* Flushed immediately, not left to exit-time buffering: check3b is
       deliberately last and deliberately crash-prone under this gate's
       own "dispatch reverted" mutation (its own comment) - a SIGSEGV
       there must not erase every earlier check's already-printed line
       along with it, and stdio is fully buffered (not line-buffered)
       once stdout is redirected to a file, which is exactly what the
       gate script does. */
    fflush(stdout);
    if (!cond) g_fail = 1;
}

/* ---------------------------------------------------------- fixture sizing */

/* head_dim a multiple of 32 - build/ced_proj_arm_probe.c's own comment
   on this exact trap (the KV cache's Q8 quantization group size), still
   allocated for every full-attention layer regardless of csa2_on. */
#define F_DIM     64
#define F_NH       2
#define F_HD      32
#define F_KVD     64
#define F_QD      64
#define F_QGD    128
#define F_INTER   64
#define F_VOCAB    4
#define F_ROT     32
#define F_SEQLEN  32
#define F_IDXH     2
#define F_IDXHD   16
#define F_IDXR    16
#define F_CM       1
/* check1a's own second compression ratio - see its comment. Every other
   fixture here runs at F_CM, where no position is ever left with an
   empty candidate set. */
#define F_CM2      2
#define F_TOPK     2
/* check4b's chunk length. It has to exceed F_TOPK, and by enough that
   the published Top-K set is a real choice rather than the whole visible
   range: at F_CM=1 a token's n_vis is tk+1, so every token before tk ==
   F_TOPK publishes every block it can see and its set cannot move
   whatever the indexer keys say. Eight is the first round number past
   that - check4b's own comment has the measurement that fixes the
   boundary. */
#define F_CED_NT   8

#define SEED0 1145141919u   /* this project's own fixed synthetic-input seed */

static unsigned lcg_next(unsigned *st) {
    *st = *st * 1103515245u + 12345u;
    return *st;
}
/* [-0.2, 0.2) - small, non-zero, decorrelated-enough; same shape as
   build/ced_proj_arm_probe.c's own p2_next. */
static float lcg_f(unsigned *st) {
    return (float)((int)((lcg_next(st) >> 8) % 2000u) - 1000) * 0.0002f;
}

static void fill_f(LZTensor *t, int n, unsigned *st) {
    int i;
    memset(t, 0, sizeof(*t));
    t->dtype = LZ_FMT_F32;
    t->n = n;
    t->gs = 0;
    t->f = (float *)malloc((size_t)n * sizeof(float));
    for (i = 0; i < n; i++) t->f[i] = lcg_f(st);
}

static void fill_xb(LZRunState *s, int nt, int dim, unsigned *st) {
    int tk, i;
    for (tk = 0; tk < nt; tk++)
        for (i = 0; i < dim; i++)
            s->xb[(size_t)tk * dim + i] = lcg_f(st);
}

/* ------------------------------------------------------------ layer fixtures */

/* q/o/q_norm (steps 1 and 6 - every mode) + k_norm and the four
   main-KV/indexer-K tensors (steps 2 and 3 - FULL only; the indexer
   half is a projection of the finalized latent plus its own norm, so
   it is two tensors of different widths - w_kv_idx is
   (indexer_head_dim, head_dim) and k_norm_idx is 1-D) + the three
   indexer QUERY tensors (step 4 - FULL and REINDEX). A REINDEX layer
   gets the SAME full set anyway - see this file's own header comment
   on why: the correct dispatch never reads the "kv" group for REINDEX,
   and a real checkpoint no longer carries it either (model.c's
   CSA2_SPECS is mode-gated now), but giving REINDEX the same tensors
   HERE, in this hand-built fixture, means a dispatch bug that wrongly
   ran FULL's steps for a REINDEX layer computes a wrong-but-finite
   answer instead of crashing before it can be caught by value
   comparison. */
static void fill_layer_common(LZLayer *L, unsigned seed_q, unsigned seed_kv, unsigned seed_idx) {
    unsigned stq = seed_q, stkv = seed_kv, stix = seed_idx;
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL;
    L->csa2_on = 1;
    fill_f(&L->q_proj, F_QGD * F_DIM, &stq);
    fill_f(&L->o_proj, F_DIM * F_QD, &stq);
    fill_f(&L->q_norm, F_HD, &stq);
    fill_f(&L->k_norm, F_HD, &stkv);
    fill_f(&L->csa2_w_kv, F_HD * F_DIM, &stkv);
    fill_f(&L->csa2_w_gate, F_HD * F_DIM, &stkv);
    fill_f(&L->csa2_w_kv_idx, F_IDXHD * F_HD, &stkv);
    fill_f(&L->csa2_k_norm_idx, F_IDXHD, &stkv);
    fill_f(&L->csa2_w_dq_idx, F_IDXR * F_DIM, &stix);
    fill_f(&L->csa2_w_uq_idx, (F_IDXH * F_IDXHD) * F_IDXR, &stix);
    fill_f(&L->csa2_w_weights, F_IDXH * F_DIM, &stix);
}

static void build_layer_full(LZLayer *L, unsigned sq, unsigned skv, unsigned six) {
    fill_layer_common(L, sq, skv, six);
    L->csa2_mode = LZ_CSA2_FULL;
}

static void build_layer_reindex(LZLayer *L, unsigned sq, unsigned skv, unsigned six) {
    fill_layer_common(L, sq, skv, six);
    L->csa2_mode = LZ_CSA2_REINDEX;
}

/* Same full tensor set as build_layer_full/build_layer_reindex, mode
   REUSE - used only to give the "dispatch reverted" mutation something
   finite to compute for this layer too (this file's own header
   comment). The CORRECT dispatch never reads any of these seven for a
   REUSE layer; build_layer_reuse_strict below is the fixture that
   actually proves that. */
static void build_layer_reuse_soft(LZLayer *L, unsigned sq, unsigned skv, unsigned six) {
    fill_layer_common(L, sq, skv, six);
    L->csa2_mode = LZ_CSA2_REUSE;
}

/* The genuine article for check3b: q_proj/o_proj/q_norm only. k_norm
   and all seven csa2_w_* tensors are left at memset's zero - dtype
   LZ_FMT_F32 (0), n 0, f NULL - because a REUSE layer owns none of
   them (model.h's csa2_mode[] comment; this file's own header comment
   on the seven-tensor grouping). If the dispatch ever mistakenly read
   one, lz_matmul_w would dereference a NULL LZTensor.f and this
   process would crash immediately rather than silently computing over
   garbage - see check3b_reuse_owns_nothing's own comment for why that
   crash, if it ever happens, is the assertion. */
static void build_layer_reuse_strict(LZLayer *L, unsigned sq) {
    unsigned stq = sq;
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL;
    L->csa2_on = 1;
    L->csa2_mode = LZ_CSA2_REUSE;
    fill_f(&L->q_proj, F_QGD * F_DIM, &stq);
    fill_f(&L->o_proj, F_DIM * F_QD, &stq);
    fill_f(&L->q_norm, F_HD, &stq);
}

/* ------------------------------------------------------------ model fixtures */

static void init_config_common(LZModelConfig *c, int n_layers) {
    int i;
    memset(c, 0, sizeof(*c));
    c->vocab_size = F_VOCAB;
    c->hidden_size = F_DIM;
    c->n_layers = n_layers;
    c->seq_len = F_SEQLEN;
    c->n_heads = F_NH;
    c->n_kv_heads = F_NH;
    c->head_dim = F_HD;
    c->intermediate_size = F_INTER;
    /* Rotation width, shared by the main query/KV and the indexer key.
       It must not exceed csa2_indexer_head_dim (set below): the indexer
       ring holds rows indexer_head_dim wide, lz_rope writes base[i] and
       base[i + rotary_dim/2] within each row, and a rotary_dim wider
       than the row runs off its end into the next row and past the
       allocation. F_IDXHD is the tighter of the two widths here. */
    c->rotary_dim = F_IDXHD;
    c->rope_theta = 10000.0f;
    c->rms_norm_eps = 1e-5f;
    c->attn_output_gate = 1;
    c->attn_q_dim = F_QD;
    c->attn_qgate_dim = F_QGD;
    c->attn_kv_dim = F_KVD;
    c->use_subn = 0;
    for (i = 0; i < n_layers; i++) c->layer_types[i] = LZ_LT_FULL;
    c->n_full_layers = n_layers;
    c->n_linear_layers = 0;
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->csa2_source[i] = -1;
    c->csa2_compress_m = F_CM;
    c->csa2_topk = F_TOPK;
    c->csa2_indexer_heads = F_IDXH;
    c->csa2_indexer_head_dim = F_IDXHD;
    c->csa2_indexer_q_lora_rank = F_IDXR;
}

/* 1 layer, FULL, csa2_on - check1's Full-only regression fixture. cm is
   a parameter here and nowhere else in this file: check1a needs both
   ratios to reach both sides of the "is this position's own block
   visible yet" boundary. */
static void build_model_single(LZModel *m, int cm, unsigned aq, unsigned akv, unsigned aix) {
    memset(m, 0, sizeof(*m));
    init_config_common(&m->config, 1);
    m->config.csa2_compress_m = cm;
    m->config.csa2_on[0] = 1;
    m->layers = (LZLayer *)calloc(1, sizeof(LZLayer));
    build_layer_full(&m->layers[0], aq, akv, aix);
}

/* 3-layer transitive chain A(FULL) -> B(REINDEX, source A) -> C(REUSE,
   source B) - the exact shape build/csa2_cache_idx_alias_probe.c's own
   case A already proved resolves cache_idx aliasing without an
   explicit walk. "Soft" C: see build_layer_reuse_soft's own comment. */
static void build_model_chain_soft(LZModel *m,
                                   unsigned aq, unsigned akv, unsigned aix,
                                   unsigned bq, unsigned bkv, unsigned bix,
                                   unsigned cq, unsigned ckv, unsigned cix) {
    memset(m, 0, sizeof(*m));
    init_config_common(&m->config, 3);
    m->config.csa2_on[0] = 1;
    m->config.csa2_on[1] = 1;
    m->config.csa2_on[2] = 1;
    m->config.csa2_source[1] = 0;
    m->config.csa2_source[2] = 1;
    m->layers = (LZLayer *)calloc(3, sizeof(LZLayer));
    build_layer_full(&m->layers[0], aq, akv, aix);
    build_layer_reindex(&m->layers[1], bq, bkv, bix);
    build_layer_reuse_soft(&m->layers[2], cq, ckv, cix);
}

/* Same chain shape, "strict" C - build_layer_reuse_strict's own comment. */
static void build_model_chain_strict(LZModel *m,
                                     unsigned aq, unsigned akv, unsigned aix,
                                     unsigned bq, unsigned bkv, unsigned bix,
                                     unsigned cq) {
    memset(m, 0, sizeof(*m));
    init_config_common(&m->config, 3);
    m->config.csa2_on[0] = 1;
    m->config.csa2_on[1] = 1;
    m->config.csa2_on[2] = 1;
    m->config.csa2_source[1] = 0;
    m->config.csa2_source[2] = 1;
    m->layers = (LZLayer *)calloc(3, sizeof(LZLayer));
    build_layer_full(&m->layers[0], aq, akv, aix);
    build_layer_reindex(&m->layers[1], bq, bkv, bix);
    build_layer_reuse_strict(&m->layers[2], cq);
}

/* 2-layer CED fixture: D1(FULL) in the decoder region, D2(REINDEX,
   source D1). No forward.c/lz_forward involved - kv_in is passed to
   forward_csa2 directly by this file, the same parameter forward.c's
   own kv_in resolution would have computed for a decoder-region CSA2
   layer under ced_kv=proj (this file's own header comment on why a
   direct call is the right granularity for this task). */
static void build_model_ced(LZModel *m,
                            unsigned aq, unsigned akv, unsigned aix,
                            unsigned bq, unsigned bkv, unsigned bix) {
    memset(m, 0, sizeof(*m));
    init_config_common(&m->config, 2);
    m->config.csa2_on[0] = 1;
    m->config.csa2_on[1] = 1;
    m->config.csa2_source[1] = 0;
    m->layers = (LZLayer *)calloc(2, sizeof(LZLayer));
    build_layer_full(&m->layers[0], aq, akv, aix);
    build_layer_reindex(&m->layers[1], bq, bkv, bix);
}

/* 3-layer independence fixture, code-review addendum 3: A(FULL) and
   D(FULL) are BOTH unaliased - each gets its own fresh ci from
   forward_state.c's cache_idx fill (neither has a csa2_source at all) -
   and E(REUSE) is sourced from A, NOT D. Every OTHER fixture in this
   file has every csa2_on layer ultimately aliasing the SAME physical
   ci (build/csa2_cache_idx_alias_probe.c's own case A shape, reused
   throughout this file for a reason - it is the shape the cache_idx
   aliasing task itself needed proving), which means none of them can
   tell "reads the right ci" apart from "reads whichever ci happens to
   be the only one that exists". This fixture is the one place two
   DIFFERENT physical ci values are ever live at once, so it is the
   only fixture that can catch csa2_topk_idx_pub/csa2_topk_n_pub's own
   ci-indexing being wrong (as opposed to their tk-indexing, which
   check5/check_chain_soft already cover) - see check_ci_independence's
   own comment and build/csa2_mode_dispatch_gate.sh's third mutation. */
static void build_model_indep(LZModel *m,
                              unsigned aq, unsigned akv, unsigned aix,
                              unsigned dq, unsigned dkv, unsigned dix,
                              unsigned eq, unsigned ekv, unsigned eix) {
    memset(m, 0, sizeof(*m));
    init_config_common(&m->config, 3);
    m->config.csa2_on[0] = 1;
    m->config.csa2_on[1] = 1;
    m->config.csa2_on[2] = 1;
    /* layer 1 (D) is FULL - csa2_source[1] stays -1, init_config_
       common's own default, and is never read for aliasing regardless.
       layer 2 (E) is REUSE, sourced from layer 0 (A) - NOT layer 1 (D),
       the whole point of this fixture. "Soft" E (build_layer_reuse_
       soft, a full seven-tensor set E never reads under the fixed
       dispatch), NOT "strict": this fixture is also exercised against
       mutation 1 (dispatch reverted) by virtue of running in the same
       probe binary, and a strict E would crash there for the same
       reason check3b's own strict fixture is deliberately ordered
       LAST - this one is not, so it must survive that mutation
       instead of being the thing that crashes under it. */
    m->config.csa2_source[2] = 0;
    m->layers = (LZLayer *)calloc(3, sizeof(LZLayer));
    build_layer_full(&m->layers[0], aq, akv, aix);
    build_layer_full(&m->layers[1], dq, dkv, dix);
    build_layer_reuse_soft(&m->layers[2], eq, ekv, eix);
}

/* ------------------------------------------------------------------ checks */

/* Item 1: Full-mode-only regression. Every currently-shipping checkpoint
   uses Full mode exclusively, so this task's dispatch must be a
   complete no-op for it. */
static void check1_full_only(void) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    unsigned stx;
    int ci;

    /* 1a: the empty-candidate-set case - k_eff exactly 0 and attn_out
       the exact zero vector (the already-shipped "no legitimate
       contribution" path). AT F_CM2, not this file's own F_CM: a
       position sees the block it is inside from that block's own last
       raw token onward, so at ratio 1 every position closes its block
       in step 3 and then attends to it, and there is no position left
       anywhere with nothing to attend to. Ratio 2 puts position 0
       (off 0, block still open) back on the empty side of that
       boundary, which is the only side this pair of assertions can
       test. The second arm below takes the other side. */
    build_model_single(&m, F_CM2, SEED0 + 1, SEED0 + 2, SEED0 + 3);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check1a: lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    ci = s.cache_idx[0];
    stx = SEED0 + 10;
    fill_xb(&s, 1, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 1, NULL);
    ok(s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + 0] == 0,
       "check1a: at cm=2, the first-ever token's published k_eff is 0");
    {
        int d, allz = 1;
        for (d = 0; d < F_QD; d++) if (s.attn_out[d] != 0.0f) allz = 0;
        ok(allz, "check1a: that k_eff==0 token's attn_out is the exact zero vector");
    }
    lz_state_free(&s);
    free(m.layers);

    /* The other side of that boundary, and the ratio the report names
       as its own special case (an uncompressed main KV, compression
       ratio 1 - which is also what the whole decoder half of the
       reference runs): position 0 closes block 0 in step 3 and must
       then attend to exactly that block. */
    build_model_single(&m, F_CM, SEED0 + 1, SEED0 + 2, SEED0 + 3);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check1a (cm=1): lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    ci = s.cache_idx[0];
    stx = SEED0 + 10;
    fill_xb(&s, 1, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 1, NULL);
    ok(s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + 0] == 1 &&
       s.csa2_topk_idx_pub[(size_t)ci * LZ_BATCH_MAX * F_TOPK + 0] == 0,
       "check1a: at cm=1, the first-ever token sees exactly the one block "
       "it just closed - k_eff 1, selecting block 0 (the block holding its "
       "own position)");
    /* The k_eff assertion above pins WHICH block was selected; this one
       pins that the selection could actually be attended over. A block
       writes ring[blk] when it is finalized, so the selection is only
       worth anything where that write has already happened - step 3 is
       ahead of the indexer for exactly this reason. With the
       finalization at the end of the per-token loop instead, ring[0] is
       still the allocation's own zeros when step 5 reads it, a
       one-candidate softmax over one all-zero entry weights it 1.0, and
       the output gate and o_proj carry that exact zero vector out -
       k_eff would still read 1, which is why the count alone is not
       evidence that anything was attended. */
    {
        int d, allz = 1;
        for (d = 0; d < F_QD; d++) if (s.attn_out[d] != 0.0f) allz = 0;
        ok(!allz,
           "check1a: at cm=1, that same token's attn_out is not the zero "
           "vector - the block it just closed was already in the ring when "
           "step 5 read it");
    }
    lz_state_free(&s);
    free(m.layers);

    /* The same boundary one ratio up, and the ratio a config.json that
       leaves "csa2_compress_m" out gets (model.c's lz_load_config
       default): at cm=2 a block closes on its own last raw position, so
       position 1 (off 1, block 0) must already see block 0 in the
       iteration that closes it. Position 0 is the control one line
       above - no block has closed yet there, so k_eff is 0 and attn_out
       IS the zero vector; the difference between the two positions is
       the whole rule. */
    build_model_single(&m, F_CM2, SEED0 + 1, SEED0 + 2, SEED0 + 3);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check1a (cm=2, last token): lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    ci = s.cache_idx[0];
    stx = SEED0 + 11;
    fill_xb(&s, 1, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 1, NULL);
    fill_xb(&s, 1, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 1, 1, NULL);
    ok(s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + 0] == 1 &&
       s.csa2_topk_idx_pub[(size_t)ci * LZ_BATCH_MAX * F_TOPK + 0] == 0,
       "check1a: at cm=2, the token that closes block 0 (pos 1) selects it "
       "- k_eff 1, selecting block 0");
    {
        int d, allz = 1;
        for (d = 0; d < F_QD; d++) if (s.attn_out[d] != 0.0f) allz = 0;
        ok(!allz,
           "check1a: ...and its attn_out is not the zero vector - block 0 "
           "was already final in the ring when step 5 read it, in the same "
           "iteration that closed it");
    }
    lz_state_free(&s);
    free(m.layers);

    /* 1b: batch-vs-unbatched bit-identity, this project's own
       established idiom for "the per-token ring bookkeeping does not
       depend on how many tokens one call covers" (build/csa2_gate.sh's
       own batch4-vs-batch1 arm), applied to the mode-dispatch code
       this task adds: three tokens through ONE forward_csa2 call must
       match three tokens through three separate single-token calls,
       token for token, including the published Top-K buffer this task
       reshapes. */
    {
        LZModel m3;
        LZRunState s3;
        int pub_n3[3], pub_idx3[3][F_TOPK];
        float attn_last3[F_QD];
        int tk, k;

        build_model_single(&m3, F_CM, SEED0 + 1, SEED0 + 2, SEED0 + 3);
        if (lz_state_alloc(&s3, &m3, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check1b: lz_state_alloc (batched): %s\n", errbuf);
            g_fail = 1;
            free(m3.layers);
            return;
        }
        ci = s3.cache_idx[0];
        stx = SEED0 + 20;
        fill_xb(&s3, 3, F_DIM, &stx);
        forward_csa2(&m3, &s3, &m3.layers[0], 0, 0, 3, NULL);
        for (tk = 0; tk < 3; tk++) {
            pub_n3[tk] = s3.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
            for (k = 0; k < F_TOPK; k++)
                pub_idx3[tk][k] = s3.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * F_TOPK + k];
        }
        memcpy(attn_last3, s3.attn_out, sizeof attn_last3);
        lz_state_free(&s3);
        free(m3.layers);

        {
            LZModel m1;
            LZRunState s1;
            unsigned stx1 = SEED0 + 20;
            int match_n = 1, match_idx = 1;

            build_model_single(&m1, F_CM, SEED0 + 1, SEED0 + 2, SEED0 + 3);
            if (lz_state_alloc(&s1, &m1, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
                printf("FAIL check1b: lz_state_alloc (unbatched): %s\n", errbuf);
                g_fail = 1;
                free(m1.layers);
                return;
            }
            ci = s1.cache_idx[0];
            for (tk = 0; tk < 3; tk++) {
                fill_xb(&s1, 1, F_DIM, &stx1);
                forward_csa2(&m1, &s1, &m1.layers[0], 0, tk, 1, NULL);
                if (s1.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + 0] != pub_n3[tk])
                    match_n = 0;
                for (k = 0; k < F_TOPK; k++)
                    if (s1.csa2_topk_idx_pub[(size_t)ci * LZ_BATCH_MAX * F_TOPK + k] != pub_idx3[tk][k])
                        match_idx = 0;
            }
            ok(match_n,
               "check1b: batch-vs-unbatched - published k_eff matches token for token");
            ok(match_idx,
               "check1b: batch-vs-unbatched - published Top-K indices match token for token");
            ok(memcmp(s1.attn_out, attn_last3, sizeof attn_last3) == 0,
               "check1b: batch-vs-unbatched - last token's attn_out is bit-identical");
            lz_state_free(&s1);
            free(m1.layers);
        }
    }
}

/* Items 2, 3a, 5: the 3-layer transitive chain, real (non-degenerate)
   weights throughout. */
static void check_chain_soft(void) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    int tk, k, ci;
    /* 4 tokens, not 3: at F_CM=1 a position's own visible-block count
       (forward_csa2.c's n_vis) is (pos + 1) / cm, so within an
       nt-token chunk the token at tk has a candidate pool of tk + 1.
       Only the LAST token's attn_out is ever externally inspectable -
       attn_out is a single-token scratch overwritten every iteration,
       so check2/check3a below can only ever look at the chunk's final
       position - and that pool must exceed topk or the selection is
       TRIVIAL: both candidates are always taken, whatever the scores,
       so no indexer-weight perturbation can change WHICH ones. nt=4
       gives the inspected token a 4-wide pool against topk=2, i.e. 6
       possible unordered pairs; a 3-token chunk would clear topk by a
       single candidate and leave only 3, which a perturbation is more
       likely to land back on (see check2's own comment). The same
       arithmetic is what sets check_ci_independence's own nt=4.
       (Layer C, REUSE, has no n_vis of its own - its own causal
       correctness comes entirely from reading whatever B published,
       per forward_csa2.c's own comment on the n_vis line.) */
    int a_n[4], b_n[4];
    int b_idx[4][F_TOPK];
    float attn_a[F_QD], attn_b[F_QD], attn_c[F_QD];
    unsigned stx;
    const unsigned AQ = SEED0 + 100, AKV = SEED0 + 101, AIX = SEED0 + 102;
    const unsigned BQ = SEED0 + 103, BKV = SEED0 + 104, BIX = SEED0 + 105;
    const unsigned CQ = SEED0 + 106, CKV = SEED0 + 107, CIX = SEED0 + 108;
    const unsigned XBA = SEED0 + 109, XBB = SEED0 + 110, XBC = SEED0 + 111;

    build_model_chain_soft(&m, AQ, AKV, AIX, BQ, BKV, BIX, CQ, CKV, CIX);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check_chain: lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    ci = s.cache_idx[0];
    ok(s.cache_idx[1] == ci && s.cache_idx[2] == ci,
       "check_chain: all three layers share the same physical ci "
       "(cache_idx aliasing from the prior task still holds)");

    stx = XBA;
    fill_xb(&s, 4, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 4, NULL);
    memcpy(attn_a, s.attn_out, sizeof attn_a);
    /* A's own published INDEX list is not captured: nothing below reads
       it. What this fixture asserts about A is a_n - its k_eff growth -
       and the publish buffer's survival across C's call, which is
       compared against b_idx below. */
    for (tk = 0; tk < 4; tk++)
        a_n[tk] = s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
    ok(a_n[0] == 1 && a_n[1] == 2 && a_n[2] == F_TOPK && a_n[3] == F_TOPK,
       "check_chain: layer A's own published k_eff grows 1,2,2,2(=topk) "
       "across the chunk");

    stx = XBB;
    fill_xb(&s, 4, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[1], 1, 0, 4, NULL);
    memcpy(attn_b, s.attn_out, sizeof attn_b);
    for (tk = 0; tk < 4; tk++) {
        b_n[tk] = s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
        for (k = 0; k < F_TOPK; k++)
            b_idx[tk][k] = s.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * F_TOPK + k];
    }
    /* B's own published k_eff is min(topk, n_vis), and n_vis is the
       per-POSITION bound (pos + 1) / cm - the reference implementation's
       own compress_lens (.prof/dsv41_ref/model.py:564, in the TRAINING
       tree at E:\LLM\llama98-zh; "a block becomes visible once the
       query has passed its last token") - computed per
       TOKEN of B's OWN loop, never however many blocks A's already-
       finished call left behind by the time B's call started. At F_CM=1
       every position closes its own block in that same iteration, so
       n_vis(tk) is exactly tk + 1 and k_eff must grow 1,2,2,2 across
       B's own chunk - the same shape check_chain's own a_n assertion
       above requires of A, and never the constant 4 that A's completed
       call leaves in their shared state. That constant is the causal
       leak this bound rules out: B's tk=0 would read n_vis=4, A's
       FINAL count for a 4-token chunk, and B's first token would see
       every block A ever finalizes for the chunk - including ones A
       had not yet written when B's tk=0 query happens, in the causal
       ordering the chunk models. */
    ok(b_n[0] == 1 && b_n[1] == F_TOPK && b_n[2] == F_TOPK && b_n[3] == F_TOPK,
       "check_chain: layer B (reindex) k_eff grows 1,2,2,2 across its "
       "own chunk - the per-position (pos + 1)/cm bound, exactly like "
       "layer A's, not a constant leaked in from A's already-finished "
       "call");
    ok(!(memcmp(b_idx[0], b_idx[1], sizeof(int) * F_TOPK) == 0 &&
         memcmp(b_idx[1], b_idx[2], sizeof(int) * F_TOPK) == 0 &&
         memcmp(b_idx[2], b_idx[3], sizeof(int) * F_TOPK) == 0),
       "check_chain: layer B's own Top-K selection is not identical "
       "across all four of its own tokens");

    stx = XBC;
    fill_xb(&s, 4, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, 4, NULL);
    memcpy(attn_c, s.attn_out, sizeof attn_c);

    /* Item 5: the shared ci's published buffer must be untouched by C's
       own call (Reuse writes nothing) - each of the four tokens' worth
       of B's own distinct selection, captured above before C ever ran,
       must survive intact. */
    {
        int all_match = 1;
        for (tk = 0; tk < 4; tk++) {
            if (s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk] != b_n[tk])
                all_match = 0;
            for (k = 0; k < F_TOPK; k++)
                if (s.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * F_TOPK + k] != b_idx[tk][k])
                    all_match = 0;
        }
        ok(all_match,
           "check_chain: after layer C's (reuse) call, all four tokens' "
           "published Top-K selections are still exactly layer B's own");
    }
    lz_state_free(&s);
    free(m.layers);

    /* Item 2: perturbing B's own indexer weights changes B's own
       attn_out (A's is untouched) AND C's - C (reuse, source B) reads
       whatever B's own call left in their shared, aliased ci slot, so
       a fresh selection from B's own perturbed indexer is exactly what
       C's own subsequent call sees too. Without running C in this same
       fixture, nothing in this probe ever exercised "reuse attends
       over a REINDEX source's Top-K" for a case where that Top-K
       actually differs from the unperturbed baseline - check3a below
       perturbs A instead, which changes C's INPUT VALUES (the ring
       content) but never changes WHICH indices get selected for C to
       read, a different (and weaker) claim. */
    {
        LZModel m2;
        LZRunState s2;
        float attn_a2[F_QD], attn_b2[F_QD], attn_c2[F_QD];

        /* +3, not an arbitrary round number: B's last token - the one
           attn_b/attn_out below actually holds - sees a 4-wide candidate
           pool at F_CM=1, and the perturbation has to move the SELECTED
           SET, not merely its order. Selecting the same pair in a
           different order leaves attn_out bit-identical, since summing
           two floats is commutative, so an order change is not evidence
           of anything. Verified empirically against this exact fixture
           (base picks the set {0,3}, +3 picks {1,2}) rather than
           assumed. */
        build_model_chain_soft(&m2, AQ, AKV, AIX, BQ, BKV, BIX + 3u, CQ, CKV, CIX);
        if (lz_state_alloc(&s2, &m2, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check2: lz_state_alloc: %s\n", errbuf);
            g_fail = 1;
        } else {
            stx = XBA;
            fill_xb(&s2, 4, F_DIM, &stx);
            forward_csa2(&m2, &s2, &m2.layers[0], 0, 0, 4, NULL);
            memcpy(attn_a2, s2.attn_out, sizeof attn_a2);
            stx = XBB;
            fill_xb(&s2, 4, F_DIM, &stx);
            forward_csa2(&m2, &s2, &m2.layers[1], 1, 0, 4, NULL);
            memcpy(attn_b2, s2.attn_out, sizeof attn_b2);
            stx = XBC;
            fill_xb(&s2, 4, F_DIM, &stx);
            forward_csa2(&m2, &s2, &m2.layers[2], 2, 0, 4, NULL);
            memcpy(attn_c2, s2.attn_out, sizeof attn_c2);

            ok(memcmp(attn_a2, attn_a, sizeof attn_a) == 0,
               "check2: layer A's own attn_out is untouched by a change to "
               "layer B's (unrelated) indexer weights");
            ok(memcmp(attn_b2, attn_b, sizeof attn_b) != 0,
               "check2: perturbing layer B's own indexer weights changes "
               "layer B's own attn_out");
            ok(memcmp(attn_c2, attn_c, sizeof attn_c) != 0,
               "check2: perturbing layer B's own indexer weights ALSO "
               "changes layer C's attn_out (C reads whatever B's own "
               "call left in their shared ci slot)");
            lz_state_free(&s2);
        }
        free(m2.layers);
    }

    /* Item 3a: perturbing A's compression weights changes BOTH B's and
       C's attn_out (both ultimately read A's physical ring). */
    {
        LZModel m3;
        LZRunState s3;
        float attn_b3[F_QD], attn_c3[F_QD];

        build_model_chain_soft(&m3, AQ, AKV + 7919u, AIX, BQ, BKV, BIX, CQ, CKV, CIX);
        if (lz_state_alloc(&s3, &m3, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check3a: lz_state_alloc: %s\n", errbuf);
            g_fail = 1;
        } else {
            stx = XBA;
            fill_xb(&s3, 4, F_DIM, &stx);
            forward_csa2(&m3, &s3, &m3.layers[0], 0, 0, 4, NULL);
            stx = XBB;
            fill_xb(&s3, 4, F_DIM, &stx);
            forward_csa2(&m3, &s3, &m3.layers[1], 1, 0, 4, NULL);
            memcpy(attn_b3, s3.attn_out, sizeof attn_b3);
            stx = XBC;
            fill_xb(&s3, 4, F_DIM, &stx);
            forward_csa2(&m3, &s3, &m3.layers[2], 2, 0, 4, NULL);
            memcpy(attn_c3, s3.attn_out, sizeof attn_c3);

            ok(memcmp(attn_b3, attn_b, sizeof attn_b) != 0,
               "check3a: perturbing layer A's compression weights changes "
               "layer B's attn_out");
            ok(memcmp(attn_c3, attn_c, sizeof attn_c) != 0,
               "check3a: perturbing layer A's compression weights also "
               "changes layer C's attn_out");
            lz_state_free(&s3);
        }
        free(m3.layers);
    }
}

/* Item 5, the direct proof: batch-vs-unbatched bit-identity (check1b's
   own idiom, generalized from one Full-mode layer to the full 3-layer
   chain). With nt==1, tk is always 0 - a per-ci-only publish buffer and
   a per-(ci, tk) one are indistinguishable, so the unbatched, one-
   position-at-a-time run is a correct oracle regardless of this task's
   own change.

   Compares EVERY token's s->xb2 (forward_csa2.c's own step 6 writes
   `s->xb2 + tk*hdim`, one slot per token, never overwritten again) for
   layer C, not s->attn_out - an earlier version of this check compared
   only the chunk's LAST token via attn_out (a single-token scratch
   overwritten every iteration, so it is the only one a multi-token call
   leaves inspectable) and was proven, empirically, structurally blind
   to the publish-buffer-collapse mutation this check exists to catch:
   collapsing csa2_topk_idx_pub/csa2_topk_n_pub from per-(ci, tk) back to
   per-ci changes what EARLIER tokens of a batched call read, but the
   LAST token's own slot is, by construction, always "the last write"
   under either addressing scheme - the same blindness class report
   Sec2.3.1 mode-dispatch task's own C2 fix hit in check_chain_soft
   (widened to 4 tokens for an unrelated but same-shaped reason) and
   build/csa2_reindex_causality_probe.c was written specifically to
   avoid from the start. check_chain_soft's own "buffer untouched by C"
   check is necessary but not sufficient for the same reason: it only
   proves C never WRITES the shared slot, not that each of its own
   tokens READS the right one. */
static void check5_batch_vs_unbatched_chain(void) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    float xb2_c_batched[3][F_DIM];
    const unsigned AQ = SEED0 + 400, AKV = SEED0 + 401, AIX = SEED0 + 402;
    const unsigned BQ = SEED0 + 403, BKV = SEED0 + 404, BIX = SEED0 + 405;
    const unsigned CQ = SEED0 + 406, CKV = SEED0 + 407, CIX = SEED0 + 408;
    const unsigned XBA = SEED0 + 409, XBB = SEED0 + 410, XBC = SEED0 + 411;
    unsigned stx;
    int tk;

    build_model_chain_soft(&m, AQ, AKV, AIX, BQ, BKV, BIX, CQ, CKV, CIX);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check5: lz_state_alloc (batched): %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    stx = XBA; fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 3, NULL);
    stx = XBB; fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[1], 1, 0, 3, NULL);
    stx = XBC; fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, 3, NULL);
    for (tk = 0; tk < 3; tk++)
        memcpy(xb2_c_batched[tk], s.xb2 + (size_t)tk * F_DIM,
               sizeof xb2_c_batched[tk]);
    lz_state_free(&s);
    free(m.layers);

    {
        LZModel m2;
        LZRunState s2;
        float xb2_c_unbatched[3][F_DIM];
        unsigned stxa = XBA, stxb = XBB, stxc = XBC;
        int pos, bad;

        build_model_chain_soft(&m2, AQ, AKV, AIX, BQ, BKV, BIX, CQ, CKV, CIX);
        if (lz_state_alloc(&s2, &m2, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check5: lz_state_alloc (unbatched): %s\n", errbuf);
            g_fail = 1;
            free(m2.layers);
            return;
        }
        /* Layer-major per position - exactly forward.c's own dispatch
           order, chunk width 1 (ordinary decode). */
        for (pos = 0; pos < 3; pos++) {
            fill_xb(&s2, 1, F_DIM, &stxa);
            forward_csa2(&m2, &s2, &m2.layers[0], 0, pos, 1, NULL);
            fill_xb(&s2, 1, F_DIM, &stxb);
            forward_csa2(&m2, &s2, &m2.layers[1], 1, pos, 1, NULL);
            fill_xb(&s2, 1, F_DIM, &stxc);
            forward_csa2(&m2, &s2, &m2.layers[2], 2, pos, 1, NULL);
            memcpy(xb2_c_unbatched[pos], s2.xb2, F_DIM * sizeof(float));
        }
        lz_state_free(&s2);
        free(m2.layers);

        bad = 0;
        for (tk = 0; tk < 3; tk++)
            if (memcmp(xb2_c_batched[tk], xb2_c_unbatched[tk],
                       sizeof xb2_c_batched[tk]) != 0)
                bad = 1;
        ok(!bad,
           "check5: layer C's (reuse) o_proj output for EVERY token of a "
           "3-token chunk processed as one nt=3 call per layer is bit-"
           "identical to the same 3 positions processed one at a time "
           "(nt=1, layer-major per position) - not merely the chunk's "
           "last token");
    }
}

/* Code-review addendum 3: two genuinely independent ci values, and a
   REUSE reader that must see only its OWN declared source's - not the
   OTHER, unrelated root's - published Top-K. build_model_indep's own
   comment has the full argument for why no earlier fixture in this
   file could catch this: every one of them aliases every csa2_on layer
   to the SAME physical ci, so "reads the right ci" and "reads the only
   ci that exists" are indistinguishable there. Empirically confirmed
   sensitive to build/csa2_mode_dispatch_gate.sh's third mutation
   (dropping the ci term from csa2_topk_idx_pub/csa2_topk_n_pub's own
   base pointers) - without it, every check in this file passes even
   with that mutation applied.

   nt=4, NOT the 3 every other fixture in this file uses: this has
   nothing to do with aliasing - D is a FRESH, unaliased FULL layer
   with no source to leak from either way. The trap is the plain
   arithmetic of this fixture's own F_CM=1/F_TOPK=2: a position's
   visible-block count is (pos + 1) / cm, so the LAST token of an
   nt-token chunk - the only one this check inspects, since attn_out is
   a single-token scratch - has an nt-wide candidate pool, and
   `k_eff = topk < n_vis ? topk : n_vis` takes a genuine, score-
   dependent 2 of the 4 nt=4 gives it. A pool of exactly topk would
   select EVERY visible block regardless of score, and this check's own
   perturbation could never show up in it - true of ANY csa2 layer at
   that width, aliased or not. */
static void check_ci_independence(void) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    float attn_e_base[F_QD];
    int ci_a, ci_d, ci_e;
    unsigned stx;
    const int nt = 4;
    const unsigned AQ = SEED0 + 500, AKV = SEED0 + 501, AIX = SEED0 + 502;
    const unsigned DQ = SEED0 + 503, DKV = SEED0 + 504, DIX = SEED0 + 505;
    const unsigned EQ = SEED0 + 506, EKV = SEED0 + 510, EIX = SEED0 + 511;
    const unsigned XBA = SEED0 + 507, XBD = SEED0 + 508, XBE = SEED0 + 509;

    build_model_indep(&m, AQ, AKV, AIX, DQ, DKV, DIX, EQ, EKV, EIX);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check_ci_independence: lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    ci_a = s.cache_idx[0];
    ci_d = s.cache_idx[1];
    ci_e = s.cache_idx[2];
    ok(ci_a != ci_d,
       "check_ci_independence: two independent FULL layers get MUTUALLY "
       "DISTINCT ci (no accidental aliasing between them)");
    ok(ci_e == ci_a,
       "check_ci_independence: the reuse layer's ci is aliased to its own "
       "declared source (A), not to the unrelated FULL layer (D)");

    stx = XBA; fill_xb(&s, nt, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, nt, NULL);
    stx = XBD; fill_xb(&s, nt, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[1], 1, 0, nt, NULL);
    stx = XBE; fill_xb(&s, nt, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, nt, NULL);
    memcpy(attn_e_base, s.attn_out, sizeof attn_e_base);
    lz_state_free(&s);
    free(m.layers);

    /* Perturb D's own INDEXER weights specifically (the group that
       controls what D itself publishes in its own step 4) - under
       correct per-ci addressing this can only ever change D's own
       physical slot, which E never reads (E's ci aliases A's, not D's).
       Under the mutation named above, D's step-4 write (still running
       - the mutation only changes WHERE it lands, not whether it
       fires) collides with A's own slot, and E ends up reading D's
       selection instead of A's. */
    {
        LZModel m2;
        LZRunState s2;
        float attn_e2[F_QD];

        /* +1, not an arbitrary round number: with n_vis=3 and topk=2 at
           D's own last token there are only 3 possible unordered pairs,
           and a perturbed indexer can legitimately land back on the
           SAME pair in a different order (attn_out is then bit-
           identical - summing two floats is commutative) without that
           meaning per-ci addressing is broken - check2's own comment
           has the same argument for layer B. Verified empirically
           against this exact fixture (base picks {0,1}, +1 picks
           {1,2} - a genuinely different set, not a reordering). */
        build_model_indep(&m2, AQ, AKV, AIX, DQ, DKV, DIX + 1u, EQ, EKV, EIX);
        if (lz_state_alloc(&s2, &m2, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check_ci_independence (perturbed): lz_state_alloc: %s\n", errbuf);
            g_fail = 1;
            free(m2.layers);
            return;
        }
        stx = XBA; fill_xb(&s2, nt, F_DIM, &stx);
        forward_csa2(&m2, &s2, &m2.layers[0], 0, 0, nt, NULL);
        stx = XBD; fill_xb(&s2, nt, F_DIM, &stx);
        forward_csa2(&m2, &s2, &m2.layers[1], 1, 0, nt, NULL);
        stx = XBE; fill_xb(&s2, nt, F_DIM, &stx);
        forward_csa2(&m2, &s2, &m2.layers[2], 2, 0, nt, NULL);
        memcpy(attn_e2, s2.attn_out, sizeof attn_e2);
        lz_state_free(&s2);
        free(m2.layers);

        ok(memcmp(attn_e2, attn_e_base, sizeof attn_e_base) == 0,
           "check_ci_independence: layer E's (reuse, source A) attn_out is "
           "untouched by a change to layer D's (unrelated, independently-"
           "aliased) indexer weights - per-ci addressing genuinely "
           "isolates the two roots, not merely the one this fixture "
           "happens to read");
    }
}

/* Item 4: the narrowed CED kv_in gate. */
static void check4_ced_gate(void) {
    /* F_CED_NT tokens' worth, not 3: kv_in is indexed as
       `kv_in + tk * hdim` (forward_csa2.c's own kvh), so a buffer sized
       for one arm's chunk length and handed to a longer one walks off
       the end of it. Both arms below read the same prefix, so sizing for
       the longer one costs the shorter arm nothing. */
    float poison[F_CED_NT * F_DIM];
    float attn_d1_null[F_QD], attn_d1_poison[F_QD];
    int i;
    char errbuf[256];
    unsigned stx;
    const unsigned AQ = SEED0 + 200, AKV = SEED0 + 201, AIX = SEED0 + 202;
    const unsigned BQ = SEED0 + 203, BKV = SEED0 + 204, BIX = SEED0 + 205;

    /* Obviously distinct from any lcg_f() draw (which stays within
       [-0.2, 0.2)) - if a bug read this where it should not, the
       divergence could not be missed. */
    for (i = 0; i < F_CED_NT * F_DIM; i++) poison[i] = 500.0f + (float)i;

    /* 4a: FULL mode still reads kv_in (Task 4's own behaviour, narrowed
       but not removed by this task). Two independent builds, kv_in
       NULL vs poisoned, same weights, same xb draw. */
    {
        LZModel ma;
        LZRunState sa;

        build_model_ced(&ma, AQ, AKV, AIX, BQ, BKV, BIX);
        if (lz_state_alloc(&sa, &ma, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check4a: lz_state_alloc (kv_in=NULL): %s\n", errbuf);
            g_fail = 1;
        } else {
            stx = SEED0 + 210;
            fill_xb(&sa, 3, F_DIM, &stx);
            forward_csa2(&ma, &sa, &ma.layers[0], 0, 0, 3, NULL);
            memcpy(attn_d1_null, sa.attn_out, sizeof attn_d1_null);
            lz_state_free(&sa);
        }
        free(ma.layers);
    }
    {
        LZModel mb;
        LZRunState sb;

        build_model_ced(&mb, AQ, AKV, AIX, BQ, BKV, BIX);
        if (lz_state_alloc(&sb, &mb, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check4a: lz_state_alloc (kv_in=poison): %s\n", errbuf);
            g_fail = 1;
        } else {
            stx = SEED0 + 210;
            fill_xb(&sb, 3, F_DIM, &stx);
            forward_csa2(&mb, &sb, &mb.layers[0], 0, 0, 3, poison);
            memcpy(attn_d1_poison, sb.attn_out, sizeof attn_d1_poison);
            lz_state_free(&sb);
        }
        free(mb.layers);
    }
    ok(memcmp(attn_d1_null, attn_d1_poison, sizeof attn_d1_null) != 0,
       "check4a: layer D1 (FULL, decoder-region CSA2) still reads kv_in in "
       "its own step 2");

    /* 4b: REINDEX never reads kv_in - D2's own call, given the SAME
       upstream ring state (D1 always run with kv_in=NULL to build it),
       must be byte-identical whether D2 itself is handed NULL or a
       poisoned non-NULL kv_in. One state, D2 called twice in a row:
       REINDEX writes nothing back into the ring/pending state (only
       FULL's steps 2 and 3 do), so replaying it is safe.

       F_CED_NT tokens, and every token's published row compared, not
       just one. The published Top-K set is a RANKING of the n_vis
       visible blocks, and at F_CM=1 n_vis is tk+1 - so while
       tk < F_TOPK the set is every block the token can see, and it
       stays put no matter what the indexer keys do. Only the tokens
       past that clamp can move at all. Measured against
       build/csa2_mode_dispatch_gate.sh's first arm ("dispatch reverted
       to unconditional Full-mode body", the one this check exists to
       catch): at 3 tokens both arms published the same set for every
       token, and from tk=3 on - the first token whose n_vis exceeds
       F_TOPK - every token published a different one. The chunk has to
       reach past the clamped prefix or this assertion cannot see the
       mutation it is named for. */
    {
        LZModel m;
        LZRunState s;
        float attn_d2_null[F_QD], attn_d2_poison[F_QD];
        int idx_null[F_CED_NT][F_TOPK], idx_poison[F_CED_NT][F_TOPK];
        int n_d2_null[F_CED_NT], n_d2_poison[F_CED_NT];
        int ci, k, tk;
        unsigned stx_d2;

        build_model_ced(&m, AQ, AKV, AIX, BQ, BKV, BIX);
        if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check4b: lz_state_alloc: %s\n", errbuf);
            g_fail = 1;
            free(m.layers);
            return;
        }
        ci = s.cache_idx[0];
        stx = SEED0 + 210;
        fill_xb(&s, F_CED_NT, F_DIM, &stx);
        forward_csa2(&m, &s, &m.layers[0], 0, 0, F_CED_NT, NULL);

        stx_d2 = SEED0 + 220;
        fill_xb(&s, F_CED_NT, F_DIM, &stx_d2);
        forward_csa2(&m, &s, &m.layers[1], 1, 0, F_CED_NT, NULL);
        memcpy(attn_d2_null, s.attn_out, sizeof attn_d2_null);
        for (tk = 0; tk < F_CED_NT; tk++) {
            n_d2_null[tk] = s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
            for (k = 0; k < F_TOPK; k++)
                idx_null[tk][k] =
                    s.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * F_TOPK + k];
        }

        stx_d2 = SEED0 + 220;
        fill_xb(&s, F_CED_NT, F_DIM, &stx_d2);   /* replay D2's own input exactly */
        forward_csa2(&m, &s, &m.layers[1], 1, 0, F_CED_NT, poison);
        memcpy(attn_d2_poison, s.attn_out, sizeof attn_d2_poison);
        for (tk = 0; tk < F_CED_NT; tk++) {
            n_d2_poison[tk] = s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
            for (k = 0; k < F_TOPK; k++)
                idx_poison[tk][k] =
                    s.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * F_TOPK + k];
        }

        ok(memcmp(attn_d2_null, attn_d2_poison, sizeof attn_d2_null) == 0,
           "check4b: layer D2 (reindex, decoder-region CSA2) attn_out is "
           "byte-identical whether kv_in is NULL or a poisoned non-NULL");
        ok(memcmp(n_d2_null, n_d2_poison, sizeof n_d2_null) == 0,
           "check4b: ...and so is the published k_eff");
        ok(memcmp(idx_null, idx_poison, sizeof idx_null) == 0,
           "check4b: ...and so is the published Top-K index set");

        lz_state_free(&s);
        free(m.layers);
    }
}

/* Item 3b: a REUSE layer owns none of the seven csa2_w_* tensors -
   ordered LAST in main(): the "dispatch reverted" mutation makes this
   fixture's own layer C dereference a NULL LZTensor.f and crash, and
   that crash must not swallow any earlier check's already-printed
   PASS/FAIL lines (build/csa2_mode_dispatch_gate.sh's own header
   comment has the exact reasoning). */
static void check3b_reuse_owns_nothing(void) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    unsigned stx;
    const unsigned AQ = SEED0 + 300, AKV = SEED0 + 301, AIX = SEED0 + 302;
    const unsigned BQ = SEED0 + 303, BKV = SEED0 + 304, BIX = SEED0 + 305;
    const unsigned CQ = SEED0 + 306;

    build_model_chain_strict(&m, AQ, AKV, AIX, BQ, BKV, BIX, CQ);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check3b: lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    stx = SEED0 + 310;
    fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 3, NULL);
    stx = SEED0 + 311;
    fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[1], 1, 0, 3, NULL);
    stx = SEED0 + 312;
    fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, 3, NULL);

    /* Reaching this line at all is most of the assertion: layer C
       (reuse) owns none of the seven csa2_w_* tensors in this fixture
       (all left at memset's zero - NULL data pointers). A dispatch bug
       that wrongly read even one of them would have dereferenced NULL
       a few lines up, and this process would already be dead. */
    {
        int d, finite = 1;
        for (d = 0; d < F_QD; d++)
            if (s.attn_out[d] != s.attn_out[d]) finite = 0;   /* NaN check */
        ok(finite,
           "check3b: layer C (reuse) completed its whole 3-token call "
           "without touching any of the seven csa2_w_* tensors it does "
           "not own, and its output is finite");
    }
    lz_state_free(&s);
    free(m.layers);
}

/* Code-review addendum 1: lz_csa2_pub_overflows (forward_state.c),
   guarding xcalloc's OWN arithmetic (element count times elem_sz plus
   its +16+sizeof(void*) alignment overhead - forward_state.c's own
   xcalloc comment) against wrapping size_t, not merely the element
   count nf*batch_max*topk itself. An EARLIER version of this function
   checked only the element count, which reviewer #2 found still let a
   real overflow through: nf=32, batch_max=8, topk=4194304 gives an
   element count (1073741824) that fits a 32-bit size_t, but that count
   times sizeof(int) is exactly 2^32, wrapping to 0 one multiplication
   later, inside xcalloc, where the element-count-only version could
   not see it - the fix folds elem_sz into the same check.

   This project's own csa2_topk=512/LZ_BATCH_MAX=8/nf-in-single-digits
   worst case (this file's own header comment) is nowhere near either a
   32-bit or a 64-bit wrap; the real production dimensions
   (c->n_full_layers <= LZ_MAX_LAYERS, LZ_BATCH_MAX, and c->csa2_topk as
   a 32-bit int) can never combine to overflow a 64-bit size_t either,
   which is exactly why this development host cannot reproduce the bug
   end to end through lz_state_alloc - only the function's own
   arithmetic can be driven directly, with synthetic nf values large
   enough to wrap on ANY width. (A prior version of this addendum
   asserted `!lz_csa2_pub_overflows(LZ_MAX_LAYERS, LZ_BATCH_MAX,
   INT_MAX, ...)` as a sanity check - true on this 64-bit host, but
   FALSE on the real 32-bit target this guard exists for, since
   LZ_MAX_LAYERS*LZ_BATCH_MAX=1024 is large enough there to trip it
   well before INT_MAX. Removed rather than fixed in place - the
   D-based checks below are the ones with a platform-independent proof
   of their own expected answer.) */
static void check_overflow_guard(void) {
    /* cap = ((size_t)-1) - 16 - sizeof(void*), elem_sz folded to 1 so
       cap IS the exact byte ceiling xcalloc's own n*sz+16+sizeof(void*)
       must not exceed (forward_state.c's own xcalloc, mirrored here
       with sizeof(void*) read the same way the real code does, not
       hardcoded, so this is exact on any width rather than assuming
       4 or 8). D = cap/4 + 1 is just over one quarter of cap, so with
       batch_max=1: n*topk = 4*D is PROVABLY > cap for any cap this
       large (4*floor(cap/4)+4 always exceeds cap, the remainder of a
       floor division being at most 3) while 3*D = 3*floor(cap/4)+3 is
       PROVABLY <= cap (the "+3" is negligible against a cap this size,
       unlike a genuinely tight boundary). Chosen via nf, not topk: an
       int cannot be made large enough on its own width to approach a
       64-bit size_t's own maximum, on this host or the real 32-bit
       target alike - a large nf (this project's own n_full_layers,
       read raw off an unvalidated bin header - forward_state.c's own
       comment on lz_csa2_pub_overflows has the full argument) is what
       actually carries the wrap in both cases. */
    size_t cap = ((size_t)-1) - 16 - sizeof(void *);
    size_t d = cap / 4 + 1;
    ok(lz_csa2_pub_overflows(d, 1, 4, 1),
       "overflow guard: nf = cap/4+1, topk=4, elem_sz=1 - 4x nf exceeds "
       "the exact byte ceiling - correctly flagged");
    ok(!lz_csa2_pub_overflows(d, 1, 3, 1),
       "overflow guard: the same nf at topk=3 stays under the ceiling - "
       "correctly NOT flagged");
    ok(!lz_csa2_pub_overflows(9, (size_t)LZ_BATCH_MAX, 512, sizeof(int)),
       "overflow guard: this project's own worst-case numbers (nf=9, "
       "batch_max=8, topk=512, elem_sz=sizeof(int)) do not trip it - "
       "true on every size_t width, not host-dependent");
    /* F7: nf==0 (or batch_max==0) must answer "no overflow" without
       dividing by zero - forward_state.c's own alloc_csa2_ring never
       calls this with a validated nf (c->n_full_layers is a raw bin
       header word), so nf==0 alongside a csa2_on layer claiming
       otherwise is a real, reachable input on that path, not a
       defensive case for something that cannot happen. */
    ok(!lz_csa2_pub_overflows(0, (size_t)LZ_BATCH_MAX, 512, sizeof(int)),
       "overflow guard: nf==0 does not divide by zero and answers "
       "'no overflow' (xcalloc(0, ...) itself short-circuits before "
       "ever reaching n*sz)");
    ok(!lz_csa2_pub_overflows((size_t)LZ_MAX_LAYERS, 0, 512, sizeof(int)),
       "overflow guard: batch_max==0 likewise does not divide by zero");
}

/* Code-review addendum 2: lz_csa2_mode_source_valid (model.c), the
   predicate lz_open_bin's own header validation now applies to every
   in-range layer's csa2_mode/csa2_source pair - previously only the
   padding past n_layers was checked at all. Tested directly with a
   hand-built LZModelConfig, the same shape ced_proj_arm_probe.c's PART
   1 already uses for lz_ced_kv_target/lz_kv_share_target: replicating
   the full ~1300-byte v5 model.bin header byte for byte to exercise
   this through lz_open would be a disproportionate amount of format
   plumbing for one predicate, and the predicate IS what lz_open_bin's
   own loop calls - there is no separate "real" behavior it could
   diverge from. */
static void check_bin_validation(void) {
    LZModelConfig c;
    int i;

    /* Fixture: 4 layers [FULL, LINEAR, FULL, FULL], layer 2 REINDEX
       sourced from 0, layer 3 REUSE sourced from 2 - the same shape
       build/csa2_cache_idx_alias_probe.c's own case A uses. */
#define VALIDATE_SETUP() do { \
        memset(&c, 0, sizeof(c)); \
        c.n_layers = 4; \
        c.layer_types[0] = LZ_LT_FULL; \
        c.layer_types[1] = LZ_LT_LINEAR; \
        c.layer_types[2] = LZ_LT_FULL; \
        c.layer_types[3] = LZ_LT_FULL; \
        for (i = 0; i < LZ_MAX_LAYERS; i++) c.csa2_source[i] = -1; \
        c.csa2_on[0] = 1; c.csa2_on[2] = 1; c.csa2_on[3] = 1; \
        c.csa2_mode[2] = LZ_CSA2_REINDEX; c.csa2_source[2] = 0; \
        c.csa2_mode[3] = LZ_CSA2_REUSE;   c.csa2_source[3] = 2; \
    } while (0)

    VALIDATE_SETUP();
    ok(lz_csa2_mode_source_valid(&c, 0) && lz_csa2_mode_source_valid(&c, 1) &&
       lz_csa2_mode_source_valid(&c, 2) && lz_csa2_mode_source_valid(&c, 3),
       "bin validation: the well-formed chain fixture passes on every layer");

    /* Corruption 1: csa2_mode[3] is a garbage word, not one of the
       three legal codes - the review's own named scenario (a raw
       header field read as garbage). */
    VALIDATE_SETUP();
    c.csa2_mode[3] = 47;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: csa2_mode outside {0,1,2} is refused");

    /* Corruption 2: csa2_source[3] is the untouched FULL-mode default
       (-1) despite csa2_mode[3] naming a non-FULL mode - the review's
       own "cache_idx[-1]" scenario. */
    VALIDATE_SETUP();
    c.csa2_source[3] = -1;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: a non-FULL mode with the FULL-mode sentinel "
       "source (-1) is refused, not read as a negative index");

    /* Corruption 3: csa2_source[3] names a layer at or after itself
       (not strictly earlier). */
    VALIDATE_SETUP();
    c.csa2_source[3] = 3;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: a source naming itself is refused");
    VALIDATE_SETUP();
    c.csa2_source[2] = 3;   /* forward reference: 3 has not been set up yet either, but 2 < 3 is what matters here */
    ok(!lz_csa2_mode_source_valid(&c, 2),
       "bin validation: a source naming a LATER layer is refused");

    /* Corruption 4: csa2_source[3] names layer 1, which is LINEAR, not
       full_attention - the review's own "wrong numbering space (nl,
       not nf)" scenario, reachable on the bin path because csa2_on[]
       there is a raw array with no cross-check against layer_types of
       its own. */
    VALIDATE_SETUP();
    c.csa2_source[3] = 1;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: a source naming a LINEAR (not full_attention) "
       "layer is refused");

    /* Corruption 5: csa2_source[3] names layer 1's index reinterpreted
       as csa2_on despite never being marked so - a source must itself
       be csa2_on, not merely full_attention. */
    VALIDATE_SETUP();
    c.csa2_on[1] = 0;             /* already 0 - restated for clarity */
    c.layer_types[1] = LZ_LT_FULL; /* layer 1 is FULL now, but never csa2_on */
    c.csa2_source[3] = 1;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: a source that is full_attention but never "
       "marked csa2_on is refused");

    /* Control: a layer that is not csa2_on at all, with csa2_mode left
       at its default (FULL, 0) - the ordinary, harmless case every
       non-CSA2 layer in a real checkpoint is in. Must NOT be refused -
       this predicate is not "must be csa2_on", it is "must be
       internally consistent". */
    VALIDATE_SETUP();
    ok(lz_csa2_mode_source_valid(&c, 1),
       "bin validation: an ordinary non-csa2_on layer (default FULL "
       "mode) is not refused");

    /* Code-review addendum F9 (a plan-level gap, not this task's own
       bug - model.c's own comment on lz_csa2_last_writer_valid has the
       full argument): the well-formed chain fixture itself (layer 2
       REINDEX source=0, layer 3 REUSE source=2) is what
       lz_csa2_last_writer_valid must ACCEPT - layer 2 becomes root 0's
       last writer, and layer 3 correctly names layer 2, not layer 0,
       as its own source. */
    VALIDATE_SETUP();
    ok(lz_csa2_last_writer_valid(&c, NULL),
       "bin validation (F9): the well-formed chain (layer 3's source is "
       "layer 2, the actual last writer of root 0's slot) passes");

    /* The named violation: layer 3 (REUSE) declares source=0 instead of
       2. lz_csa2_mode_source_valid alone (the per-layer question) still
       accepts this - 0 IS a strictly earlier, csa2_on, full_attention
       layer - which is exactly the gap this addendum closes:
       lz_csa2_last_writer_valid must refuse it, because by the time
       layer 3's own call runs, layer 2's call has already overwritten
       root 0's shared slot with layer 2's OWN Top-K, so layer 3 would
       actually read layer 2's selection while declaring layer 0. */
    VALIDATE_SETUP();
    c.csa2_source[3] = 0;
    ok(lz_csa2_mode_source_valid(&c, 3),
       "bin validation (F9): the per-layer predicate ALONE still "
       "accepts source=0 for layer 3 - demonstrating the gap this "
       "addendum closes, not just asserting the fix works");
    {
        int bad = -1;
        ok(!lz_csa2_last_writer_valid(&c, &bad) && bad == 3,
           "bin validation (F9): the whole-config predicate refuses it, "
           "naming layer 3 as the violation");
    }
#undef VALIDATE_SETUP
}

int main(void) {
    check1_full_only();
    check_chain_soft();
    check5_batch_vs_unbatched_chain();
    check_ci_independence();
    check4_ced_gate();
    check3b_reuse_owns_nothing();
    check_overflow_guard();
    check_bin_validation();

    if (g_fail) {
        printf("\ncsa2_mode_dispatch_probe: FAIL\n");
        return 1;
    }
    printf("\ncsa2_mode_dispatch_probe: PASS\n");
    return 0;
}
