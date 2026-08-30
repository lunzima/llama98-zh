#include <string.h>
#include <stdlib.h>   /* getenv, lz_moe_route_float_probe */

#include "forward.h"
#include "ops.h"
#include "ops_quant.h"  /* lz_quantize_q8(_i16), lz_sigmoid_q15_run */

/* Latent MoE FFN path, moved out of forward.c (the four-path split).
   Pure code motion: moe_expert_loop and lz_moe_route_float_probe stay
   static; forward_moe lost `static` because lz_forward still calls it. */

/* ------------------------------------------------ latent MoE FFN block */

/* Latent MoE (LatentMoE / KunMoEGate),
   replacing the dense gate/up/down_proj FFN for layers >=
   config.first_k_dense_replace (LZLayer.ffn_moe).

   FOUR PHASES, split on the one thing that is per-token: expert
   SELECTION. The router, the latent down/up projections and the shared
   expert all read one weight matrix for every token in the chunk, so
   they batch through lz_matmul_xq_nt exactly like dense_ffn_step. Only
   the routed experts cannot - two tokens in the same chunk may pick
   different experts entirely, so there is no shared weight stream.

   Why the tokens are not grouped by expert, which is what llama.cpp's
   ggml_compute_forward_mul_mat_id (a per-expert row bucket, no padding)
   and vLLM/SGLang's moe_align_block_size do: their scale is not ours.
   SGLang's CPU int8 MoE pads each expert's rows to BLOCK_M = 32 and
   drops the blocked path below M = 5; llama.cpp's tuning assumes
   batches in the thousands. LZ_BATCH_MAX is 8, and at 8 tokens over
   top-2 of 16 experts the expected DISTINCT expert count is
   16*(1-(15/16)^16) ~ 10.3 of 16 pairs - about a third of the weight
   streams, ~9% of prefill, in exchange for reordering the float
   accumulation in the kk loop. llama.cpp's own measurement that
   prompt-phase routing is flat (decode's is skewed) makes prefill the
   worst case for it.

   THE SPLIT, measured rather than assumed: ffn(topk) is
   linear in topk, so sweeping --moe-topk 1/2/4 separates the halves.
   On an 832-token prompt it fit ffn = 856,505 + 787,201*topk us
   exactly - at the trained topk=2 the routed experts are 64.8% of the
   ffn phase and everything batched here is the other 35.2%.

   nt=1 (generation) is unchanged: lz_matmul_xq_nt's t-loop degrades to
   one iteration, which that function's header states as a hard gate.

   Bit-identity between batched and per-token MoE prefill:
   verified across E:\LLM\models\kunmoe-v2 (widths
   1/2/3/4/8, 8 lengths each, plus 6 continuation steps) as well as
   s1v3 - the batch-parity claim reaches this function directly, not
   only a dense model that has no MoE and no KDA. It passes: the loop
   below has no state that carries across tk (xq/xqs are fully
   overwritten each iteration), so there was nothing FOR batch width to
   break here - the gap was real for throughput, not for correctness. */
static void moe_expert_loop(LZRunState *s, const LZModelConfig *c,
                            const LZLayer *L, int nt, int topk,
                            int mrt_int, int mlat_int, int rsw_may) {
    int latent = c->moe_latent_dim;
    int inter = c->moe_intermediate_size;
    int ne = c->num_experts;
    float rms_eps = c->rms_norm_eps;
    int tk, kk, vv;
    /* Loop-invariant router configuration, packed so the per-token call
       passes one struct pointer instead of nine arguments (see the
       LZMoeRouteParams note in ops.h). moe_gate_bias is F32, so
       lz_t_f32 returns t->f without touching s->wscr and the pointer is
       stable across the whole token loop. */
    LZMoeRouteParams rp;
    rp.li_e = LZ_MOEGATE_ES;
    rp.bias = lz_t_f32(&L->moe_gate_bias, s->wscr);
    rp.n_experts = ne;
    rp.top_k = topk;
    rp.sigmoid = c->moe_router_sigmoid;
    rp.renormalize = c->moe_renormalize;
    rp.tau = lz_moe_tau;
    rp.idx_out = s->moe_sel_idx;
    rp.w_out = s->moe_sel_w;
    for (tk = 0; tk < nt; tk++) {
        float *lat_x = s->moe_lat_x + (size_t)tk * latent;
        const short *lat_i = mlat_int ? s->moe_lat_i16 + (size_t)tk * latent
                                      : NULL;
        float *lat_y = s->moe_lat_y + (size_t)tk * latent;
        int gs_have = 0;        /* group size moe_lat_q currently holds */

        lz_moe_route(s->moe_router_logits + (size_t)tk * ne,
                     mrt_int ? s->moe_logits_i16 + (size_t)tk * ne : NULL,
                     &rp);

        /* One layer's contribution to the token's union. Guarded, so a
           caller that did not ask for an inspector pays one predictable
           branch per MoE layer and nothing else - no call, no write. */
        if (s->ins)
            lz_moe_hits_add(s->ins->expert_hits, LZ_INSPECT_EXPERT_MAX,
                            s->moe_sel_idx, topk);

        memset(lat_y, 0, (size_t)latent * sizeof(float));
        for (kk = 0; kk < topk; kk++) {
            int ei = s->moe_sel_idx[kk];
            float ww = s->moe_sel_w[kk];
            if (ei < 0) continue;               /* topk > n_experts padding */
            /* KdaExpert.forward: w2(act(w1(x)) * w3(x)) - w1 is the gate,
               w3 the up projection, w2 the down projection back to latent. */
            /* w1 and w3 both read the latent with the same in_dim, so
               quantize it once and share, instead of letting each
               lz_matmul_w re-quantize the identical vector. The
               quantization ALSO survives across experts, because the
               vector does not change and lz_act_gs is a function of
               (in_dim, whether the weight's gs is a sane multiple of
               32) - not of which expert. So it is redone only when the
               group size actually differs, which keeps the
               mixed-precision hazard forward_attn's q/k/v comment names
               (w1 and w3 must read the SAME quantization) while paying
               for it once per DISTINCT group size instead of once per
               expert. Measured invariant on both checkpoints: 128
               w1/w3 pairs each, lz_act_gs == 32 throughout, so the
               second expert's quantize is pure repetition.

               Its output is moe_lat_q/_qs and NOT s->xq/s->xqs: the w2
               matmul below quantizes moe_h1 into those, so a result
               left there would be gone before the next expert reads it.
               That, not the group size, is what forces the quantize
               back inside this loop. */
            {
                int gsl = lz_act_gs(&L->moe_expert_w1[ei], latent);
                /* LZ_MLAT_REUSE=0 is this saving's control arm: quantize
                   on every expert, as the loop did before the hoist.
                   Folds at compile time; the destination is moe_lat_q/_qs
                   either way, so the arm isolates the SKIP and not the
                   buffer move, and reproduces the pre-hoist cost exactly. */
                if (!LZ_MLAT_REUSE || gsl != gs_have) {
                    /* Positive controls for two switches at once, and
                       they say different things: mlat counts the int16
                       exit being consumed, reuse counts the
                       quantizations this guard let through. With reuse
                       on the second number is the smaller one, which is
                       the saving - and with it off the two are equal,
                       which is what "the guard is compiled out" looks
                       like from outside. */
                    lz_debug_mlat_quant++;
                    if (mlat_int) {
                        lz_debug_mlat_i16 += latent;
                        lz_quantize_q8_i16(lat_i, latent, gsl,
                                           pow2f(-LZ_MLAT_ES),
                                           s->moe_lat_q, s->moe_lat_qs);
                    } else
                        lz_quantize_q8(lat_x, latent, gsl,
                                       s->moe_lat_q, s->moe_lat_qs);
                    gs_have = gsl;
                }
            }
#if LZ_SWIGLU_I16
            /* The int path, when rsw_may said the whole triple can take
               it. Both exits are asked here rather than through a
               proj_i16-style helper because there is nothing to
               recover: rsw_may already ran lz_matmul_xq_i16_ok on both
               tensors, and that predicate IS the entry point's own, so a
               refusal below is unreachable. The `&&` is the
               all-or-nothing enforcement anyway - a w3 that somehow
               refused leaves moe_h1_i16 written and moe_h1 stale, and
               the float branch below would then read the stale float, so
               the second call's failure has to take the first one's
               result with it. It does: nothing between here and the
               requantize reads either buffer. */
            if (rsw_may &&
                lz_matmul_xq_nt_i16(s->moe_h1_i16, s->moe_lat_q, s->moe_lat_qs,
                                    &L->moe_expert_w1[ei], latent, inter, 1,
                                    LZ_SWIGLU_ES_P, 32767) &&
                lz_matmul_xq_nt_i16(s->moe_h3_i16, s->moe_lat_q, s->moe_lat_qs,
                                    &L->moe_expert_w3[ei], latent, inter, 1,
                                    LZ_SWIGLU_ES_G, 32767)) {
                lz_swiglu_q15_i16(s->moe_h1_i16, s->moe_h1_i16, s->moe_h3_i16,
                                  inter, LZ_SWIGLU_ES_P, LZ_SWIGLU_ES_G,
                                  LZ_SWIGLU_ES_O);
                /* lz_matmul_w's two halves, with the quantize taking the
                   int16 instead of a float. The float argument is NULL
                   because moe_h1 holds nothing on this path and the
                   weight is not F32 (rsw_may checked). */
                lz_quantize_q8_i16(s->moe_h1_i16, inter,
                                   lz_act_gs(&L->moe_expert_w2[ei], inter),
                                   pow2f(-LZ_SWIGLU_ES_O), s->xq, s->xqs);
                lz_matmul_xq(s->moe_h2, NULL, s->xq, s->xqs,
                             &L->moe_expert_w2[ei], inter, latent);
            } else
#endif /* LZ_SWIGLU_I16 */
            {
            lz_matmul_xq(s->moe_h1, lat_x, s->moe_lat_q, s->moe_lat_qs,
                        &L->moe_expert_w1[ei], latent, inter);
            lz_matmul_xq(s->moe_h3, lat_x, s->moe_lat_q, s->moe_lat_qs,
                        &L->moe_expert_w3[ei], latent, inter);
            for (vv = 0; vv < inter; vv++)
                s->moe_h1[vv] = lz_silu(s->moe_h1[vv]) * s->moe_h3[vv];
            lz_matmul_w(s->moe_h2, s->moe_h1, &L->moe_expert_w2[ei],
                       inter, latent, s->xq, s->xqs);
            }   /* the float SwiGLU */
            for (vv = 0; vv < latent; vv++)
                lat_y[vv] += ww * s->moe_h2[vv];
        }
        if (c->moe_latent_use_norm)
            lz_rmsnorm(lat_y, lat_y,
                      lz_t_f32(&L->moe_latent_norm, s->wscr), latent,
                      rms_eps);
    }
}

/* Sec 9.2 risk-path probe: force the router's float-logits path under an
   otherwise --fixed all engine, so the int16 quantization's effect on
   expert selection can be measured against a fresh float baseline. Not
   a shipping knob - LZ_MOE_ROUTE_FLOAT=1 in the environment, and only
   the router changes (lz_sig_mode() and every other site stay fixed).
   getenv is C89 and this file already links the CRT. */
static int lz_moe_route_float_probe(void) {
    static int cached = -1;
    if (cached < 0) cached = (getenv("LZ_MOE_ROUTE_FLOAT") != NULL);
    return cached;
}

void forward_moe(const LZModel *m, LZRunState *s,
                        const LZLayer *L, int layer, int nt) {
    const LZModelConfig *c = &m->config;
    int hdim = c->hidden_size, latent = c->moe_latent_dim;
    int shared_w = c->moe_shared_width;
    int ne = c->num_experts, topk = c->num_experts_per_token;
    float rms_eps = c->rms_norm_eps;
    /* --moe-topk: route to a different number of experts than the model
       was exported with. Nothing in this function or lz_moe_route is
       written for a particular k - the router pads unfillable slots with
       idx -1 / weight 0, the loop skips them, and moe_renormalize keeps
       the weights summing to one so the output scale does not move with
       k. So this is a knob, not a rewrite.
       What it is NOT is free: the model was TRAINED at its config value,
       and running off that value is off-distribution. It exists so the
       question can be measured - a parameter whose best value differs by
       machine gets swept, not baked - and the scratch buffers are sized from the
       config, so the override is clamped to them. */
    int gsh;
    int nsh;
    int tk, i;
    /* The topk override and the shared-expert group sizes are computed
       after the exit predicates below, not here. They are statements,
       and C89 wants every declaration in this block ahead of every
       statement in it - each of mrt, mlat and rsw carries a paragraph
       explaining a decision, so moving those up would separate three
       explanations from what they explain. These two have nothing to
       say and their first use is far below. */
    /* int-pipeline milestone 5: the router logits' only consumer is
       lz_moe_route, and the exit is the same one bvec takes - int16 at
       LZ_MOEGATE_ES, bounded by the sigmoid table's own domain rather
       than int16's edge. Decided before the matmul, like every other
       exit in this file, and lz_sig_mode() answers it for the same
       reason.
       moe_router_sigmoid is a consumer question too: the softmax scorer
       has no integer entry (lz_exp has no int-in form), so it would
       rebuild the float the epilogue declined to build - the same
       operation count, for a value that has moved. Refuse instead. */
    int mrt_int = 0, mrt_may = 0;   /* set below, past the declarations */
    /* int-pipeline milestone 6: the latent's only consumer is the
       routed experts' activation quantize, which lz_quantize_q8_i16
       serves directly from the int16 - so moe_down_proj's Q8_0
       epilogue can stop building a float that the next statement takes
       apart again. Decided before the matmul, like every other exit in
       this file. */
    int mlat_int = 0, mlat_may = 0;
    /* The predicate is set below with the others, past the last
       declaration in this block. Its refusal rule stays here, where it
       belongs to the reader: lz_matmul_xq reads its FLOAT argument when
       - and only when - the weight is F32, so an F32 expert would read
       a moe_lat_x the int exit never wrote. Refused for the whole layer
       rather than per expert: which experts a token routes to is not
       known until after the matmul that would have written the float.
       ne dtype loads per call and no float work. */
    /* int-pipeline milestone 8: the two SwiGLU sites in this function.
       Both are decided here, before any matmul, and both are
       all-or-nothing over their own triple (gate, up, down) for the
       reason kda_conv_proj's comment gives - one consumer fed half in
       int and half in float is worse than either.

       The routed predicate scans every expert rather than the two a
       token routes to, exactly as mlat_may above does and for the same
       reason: routing is not known until after the matmul that would
       have written the float. The down-projection is in the scan because
       lz_matmul_xq reads its FLOAT argument when the weight is F32, and
       on the int path that argument is a moe_h1 nothing wrote.

       use_subn refuses BOTH sites: SubLN puts an RMSNorm between the
       SwiGLU and its quantize (moe_shared_down_norm), and this tree has
       no int16-input norm for it to read - the float would have to be
       rebuilt, which is the swap this milestone exists to avoid. */
    int rsw_may = 0, ssw_may = 0;

    /* Moved down from the top of the block - see the note there. The
       router predicate joins them for the same reason: its declaration
       carries a paragraph, so the declaration stays where the paragraph
       is and the assignment comes here. */
#if LZ_MOEGATE_I16
    mrt_may = !lz_moe_route_float_probe() && lz_sig_mode() &&
              s->moe_logits_i16 && c->moe_router_sigmoid &&
              lz_matmul_xq_i16_ok(&L->moe_gate_w, hdim, nt);
#endif /* LZ_MOEGATE_I16 */
#if LZ_MLAT_I16
    mlat_may = s->moe_lat_i16 && L->moe_expert_w1 && L->moe_expert_w3 &&
               lz_matmul_xq_i16_ok(&L->moe_down_proj, hdim, nt);
    for (i = 0; mlat_may && i < ne; i++)
        /* bf16 excluded for the same reason F32 is: the i16 latent
           path needs a quantized weight with a group scale. */
        if (L->moe_expert_w1[i].dtype == LZ_FMT_F32 ||
            L->moe_expert_w3[i].dtype == LZ_FMT_F32 ||
            L->moe_expert_w1[i].dtype == LZ_FMT_BF16 ||
            L->moe_expert_w3[i].dtype == LZ_FMT_BF16)
            mlat_may = 0;
#endif /* LZ_MLAT_I16 */
    if (lz_moe_topk > 0 && lz_moe_topk <= ne)
        topk = lz_moe_topk;
    gsh = lz_act_gs(&L->moe_down_proj, hdim);
    nsh = scale_groups(hdim, gsh);

#if LZ_SWIGLU_I16
    rsw_may = s->moe_h1_i16 && s->moe_h3_i16 && !c->use_subn &&
              L->moe_expert_w1 && L->moe_expert_w3 && L->moe_expert_w2;
    for (i = 0; rsw_may && i < ne; i++)
        if (!lz_matmul_xq_i16_ok(&L->moe_expert_w1[i], latent, 1) ||
            !lz_matmul_xq_i16_ok(&L->moe_expert_w3[i], latent, 1) ||
            lz_matmul_xq_reads_float_row(&L->moe_expert_w2[i]))
            rsw_may = 0;
    ssw_may = s->moe_h1_i16 && s->moe_h3_i16 && !c->use_subn && shared_w > 0 &&
              lz_matmul_xq_i16_ok(&L->moe_shared_gate, hdim, nt) &&
              lz_matmul_xq_i16_ok(&L->moe_shared_up, hdim, nt) &&
              !lz_matmul_xq_reads_float_row(&L->moe_shared_down);
#endif /* LZ_SWIGLU_I16 */
    (void)layer;   /* only used inside LZ_TAP, which is a no-op in a normal build */

    /* PHASE 1 - everything that reads xt and does not depend on routing.
       The router and the latent down-projection are ordinary matmuls:
       one weight matrix serves every token in the chunk, so they run
       batched like every other block in this file, instead of streaming
       the same weights nt times.
       They also share one activation quantization, the same reasoning
       forward_attn's q/k/v_proj comment gives. */
    if (c->use_subn) {
        /* SubLN: this block is handed the RAW residual, so the router
           and the latent down-projection each normalize it with their
           own weight and the shared quantization above no longer
           applies. s->xb2 is this function's OUTPUT but nothing writes
           it until phase 3, so it is scratch here.

           The router is normalized like any other projection, and that
           is not cosmetic: its logits go through a softmax, so the input
           magnitude acts as a temperature and skipping it would move
           every routing decision. */
        int gsg = lz_act_gs(&L->moe_gate_w, hdim);
        int nsg = scale_groups(hdim, gsg);
        for (tk = 0; tk < nt; tk++)
            lz_rmsnorm(s->xb2 + (size_t)tk * hdim, s->xb + (size_t)tk * hdim,
                       lz_t_f32(&L->moe_gate_norm, s->wscr), hdim,
                       rms_eps);
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->xb2 + (size_t)tk * hdim, hdim, gsg,
                           s->xq + (size_t)tk * hdim,
                           s->xqs + (size_t)tk * nsg);
        proj_i16(s, mrt_may, &mrt_int, s->moe_router_logits,
                     s->moe_logits_i16,
                     s->xb2, &L->moe_gate_w, hdim, ne, nt,
                     LZ_MOEGATE_ES, lz_sig_q15_domain(LZ_MOEGATE_ES));
        for (tk = 0; tk < nt; tk++)
            lz_rmsnorm(s->xb2 + (size_t)tk * hdim, s->xb + (size_t)tk * hdim,
                       lz_t_f32(&L->moe_down_norm, s->wscr), hdim,
                       rms_eps);
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->xb2 + (size_t)tk * hdim, hdim, gsh,
                           s->xq + (size_t)tk * hdim,
                           s->xqs + (size_t)tk * nsh);
        lz_matmul_xq_nt(s->moe_lat_x, s->xb2, s->xq, s->xqs,
                        &L->moe_down_proj, hdim, latent, nt);
    } else {
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gsh,
                       s->xq + (size_t)tk * hdim,
                       s->xqs + (size_t)tk * nsh);
    proj_i16(s, mrt_may, &mrt_int, s->moe_router_logits,
                 s->moe_logits_i16,
                 s->xb, &L->moe_gate_w, hdim, ne, nt,
                 LZ_MOEGATE_ES, lz_sig_q15_domain(LZ_MOEGATE_ES));
    proj_i16(s, mlat_may, &mlat_int, s->moe_lat_x, s->moe_lat_i16,
                 s->xb, &L->moe_down_proj, hdim, latent, nt,
                 LZ_MLAT_ES, 32767);
    }
    /* Slot 0 is token 0's, so these tap what the per-token version
       taped under `if (tk == 0)`. Both read a FLOAT buffer that holds
       nothing when its chain took the int16 exit - dump
       moe_logits_i16 / moe_lat_i16 instead, same as the KDA taps
       above. */
    LZ_TAP("mrt", layer, s->moe_router_logits, ne);
    LZ_TAP("mlat", layer, s->moe_lat_x, latent);

    /* PHASE 2 - the routed experts, and the ONE part that stays per
       token: which expert a token wants is a property of that token, so
       there is no shared weight stream to batch over. Grouping the
       tokens by expert instead was measured and rejected - see the
       block comment above this function. */
    moe_expert_loop(s, c, L, nt, topk, mrt_int, mlat_int, rsw_may);
    LZ_TAP("mrou", layer, s->moe_lat_y, latent);

    /* PHASE 3 - back out of the latent space. Routing-independent
       again, so batched. */
    {
        int gsu = lz_act_gs(&L->moe_up_proj, latent);
        int nsu = scale_groups(latent, gsu);
        if (c->use_subn) {
            /* In place because the "mrou" tap above already recorded the
               pre-norm value and every later reader wants the post-norm
               one. Not because the buffer dies at the quantize below: it
               is still live at lz_matmul_xq_nt, which reads it as a float
               row whenever moe_up_proj is F32. */
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->moe_lat_y + (size_t)tk * latent,
                           s->moe_lat_y + (size_t)tk * latent,
                           lz_t_f32(&L->moe_up_norm, s->wscr), latent,
                           rms_eps);
        }
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->moe_lat_y + (size_t)tk * latent, latent, gsu,
                           s->xq + (size_t)tk * latent,
                           s->xqs + (size_t)tk * nsu);
        lz_matmul_xq_nt(s->xb2, s->moe_lat_y, s->xq, s->xqs,
                        &L->moe_up_proj, latent, hdim, nt);
    }

    /* PHASE 4 - the shared expert. _shared in kunmoe_modeling.py: same
       SwiGLU shape as the dense FFN this block replaces, just at
       shared_w width and reading hidden-space x directly (no latent).
       Every token goes through it, so all three of its matmuls batch -
       this is dense_ffn_step's shape, and it is written the same way. */
    if (shared_w > 0) {
        int gss = lz_act_gs(&L->moe_shared_gate, hdim);
        int nss = scale_groups(hdim, gss);
        int gsd = lz_act_gs(&L->moe_shared_down, shared_w);
        int nsd = scale_groups(shared_w, gsd);
        int ssw_int = 0;    /* the pair landed in moe_h1_i16/moe_h3_i16 */

        /* gate and up both read xt with in_dim == hidden_size, so
           quantize ONCE and share - the same fix, and the same hazard,
           as the routed experts' w1/w3 above. Re-quantized rather than
           reusing phase 1's: the shared gate may carry a different
           activation group size than the down-projection did. */
        if (c->use_subn) {
            /* SubLN: gate and up read the raw residual with their own
               weights, so they cannot share a quantization either.
               Scratch is moe_shared_out - hidden-wide and not written
               until this block's last matmul. xb2 is NOT free here:
               phase 3 has already put the routed output in it. */
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->moe_shared_out + (size_t)tk * hdim,
                           s->xb + (size_t)tk * hdim,
                           lz_t_f32(&L->moe_shared_gate_norm, s->wscr), hdim,
                           rms_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->moe_shared_out + (size_t)tk * hdim, hdim, gss,
                               s->xq + (size_t)tk * hdim,
                               s->xqs + (size_t)tk * nss);
            lz_matmul_xq_nt(s->moe_h1, s->moe_shared_out, s->xq, s->xqs,
                            &L->moe_shared_gate, hdim, shared_w, nt);
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->moe_shared_out + (size_t)tk * hdim,
                           s->xb + (size_t)tk * hdim,
                           lz_t_f32(&L->moe_shared_up_norm, s->wscr), hdim,
                           rms_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->moe_shared_out + (size_t)tk * hdim, hdim, gss,
                               s->xq + (size_t)tk * hdim,
                               s->xqs + (size_t)tk * nss);
            lz_matmul_xq_nt(s->moe_h3, s->moe_shared_out, s->xq, s->xqs,
                            &L->moe_shared_up, hdim, shared_w, nt);
        } else {
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gss,
                           s->xq + (size_t)tk * hdim,
                           s->xqs + (size_t)tk * nss);
#if LZ_SWIGLU_I16
        /* Same all-or-nothing pair as the routed experts above, and the
           same unreachability argument for the second call refusing -
           ssw_may already asked lz_matmul_xq_i16_ok for both. `ssw_int`
           goes OUT and means one thing only: moe_h1_i16/moe_h3_i16 hold
           the two projections, so nothing may read moe_h1/moe_h3. */
        if (ssw_may &&
            lz_matmul_xq_nt_i16(s->moe_h1_i16, s->xq, s->xqs,
                                &L->moe_shared_gate, hdim, shared_w, nt,
                                LZ_SWIGLU_ES_P, 32767) &&
            lz_matmul_xq_nt_i16(s->moe_h3_i16, s->xq, s->xqs,
                                &L->moe_shared_up, hdim, shared_w, nt,
                                LZ_SWIGLU_ES_G, 32767))
            ssw_int = 1;
#endif /* LZ_SWIGLU_I16 */
        if (!ssw_int) {
        lz_matmul_xq_nt(s->moe_h1, s->xb, s->xq, s->xqs,
                        &L->moe_shared_gate, hdim, shared_w, nt);
        lz_matmul_xq_nt(s->moe_h3, s->xb, s->xq, s->xqs,
                        &L->moe_shared_up, hdim, shared_w, nt);
        }
        }
        if (ssw_int) {
            /* Token-major with stride shared_w, so the whole batch is one
               contiguous run - the float loop below treats it the same
               way. */
            lz_swiglu_q15_i16(s->moe_h1_i16, s->moe_h1_i16, s->moe_h3_i16,
                              nt * shared_w, LZ_SWIGLU_ES_P, LZ_SWIGLU_ES_G,
                              LZ_SWIGLU_ES_O);
        } else {
        for (i = 0; i < nt * shared_w; i++)
            s->moe_h1[i] = lz_silu(s->moe_h1[i]) * s->moe_h3[i];
        if (c->use_subn) {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->moe_h1 + (size_t)tk * shared_w,
                           s->moe_h1 + (size_t)tk * shared_w,
                           lz_t_f32(&L->moe_shared_down_norm, s->wscr),
                           shared_w, rms_eps);
        }
        }
        for (tk = 0; tk < nt; tk++) {
            if (ssw_int)
                lz_quantize_q8_i16(s->moe_h1_i16 + (size_t)tk * shared_w,
                                   shared_w, gsd, pow2f(-LZ_SWIGLU_ES_O),
                                   s->xq + (size_t)tk * shared_w,
                                   s->xqs + (size_t)tk * nsd);
            else
                lz_quantize_q8(s->moe_h1 + (size_t)tk * shared_w, shared_w, gsd,
                               s->xq + (size_t)tk * shared_w,
                               s->xqs + (size_t)tk * nsd);
        }
        /* NULL, not moe_h1, on the int path: the down-projection is the
           one caller that could still read the float row, and ssw_may
           already established it will not (lz_matmul_xq_reads_float_row).
           Passing the stale buffer would leave that as a promise. */
        lz_matmul_xq_nt(s->moe_shared_out, ssw_int ? NULL : s->moe_h1,
                        s->xq, s->xqs,
                        &L->moe_shared_down, shared_w, hdim, nt);
        for (i = 0; i < nt * hdim; i++) s->xb2[i] += s->moe_shared_out[i];
    }
    /* No tap for the combined routed+shared output here: forward_chunk's
       unconditional LZ_TAP("ffn", l, s->xb2, dim) right after this
       function returns already captures it - same buffer, same point,
       for both this branch and the dense one, so a second tap here
       would only double the dump for zero extra signal. */
}

