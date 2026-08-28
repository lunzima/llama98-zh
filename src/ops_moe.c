/* MoE router: lz_moe_route and lz_moe_hits_add, extracted from ops.c
   as pure code motion (the bodies are verbatim copies). Both are
   declared in ops.h; the callers are in forward.c. The only
   non-ops.h dependencies are pow2f (ops_quant.h) and lz_i32f
   (ops_kernel_shared.h). */

#include "ops.h"               /* LZ_MAX_EXPERTS (via model.h), lz_sigmoid_i/
                                   lz_sigmoid/lz_exp, LZ_FCX/LZ_FC_EPI */
#include "ops_quant.h"         /* pow2f */
#include "ops_kernel_shared.h" /* lz_i32f */

void lz_moe_route(const float *logits, const short *li,
                  const LZMoeRouteParams *rp) {
    /* Unpack the loop-invariant fields into locals so the body below is
       unchanged from the flat-signature version. `rp` is const and no
       aliasing write reaches it, so each field loads once (base+offset)
       and the caller pays one struct pointer instead of pushing the
       nine arguments a flat signature needs. */
    int li_e = rp->li_e;
    const float *bias = rp->bias;
    int n_experts = rp->n_experts;
    int top_k = rp->top_k;
    int sigmoid = rp->sigmoid;
    int renormalize = rp->renormalize;
    float tau = rp->tau;
    int *idx_out = rp->idx_out;
    float *w_out = rp->w_out;
    /* static, not stack: LZ_MAX_EXPERTS is 128, and rule six clause 4
       ("large buffers stay off the stack") is written for a Win98
       target with a small default stack - four 512-byte arrays per
       call is exactly the kind of thing that rule exists for. */
    static float score[LZ_MAX_EXPERTS];
    static float tscore[LZ_MAX_EXPERTS];
    static float key[LZ_MAX_EXPERTS];
    /* Where an int16 input is rebuilt as floats for the readers that
       have no integer entry - see the `li` half of this function's
       header. Static for the same reason the three above are. */
    static float flog[LZ_MAX_EXPERTS];
    /* Sentinel for "already selected" in the top-k scan below - far
       enough from any real score/bias that excess precision could not
       change which elements it excludes, but not exactly representable
       in float32 either, so it still gets the same treatment as every
       other constant on this file's list (LZ_EXP_LOG2E32's comment in
       lz_exp has the mechanism). */
    static const float LZ_MOE_SENTINEL = -1e30f;
    int i, t;
    int tempered = (tau != 1.0f);
    float wsum = 0.0f;

    for (t = 0; t < top_k; t++) { idx_out[t] = -1; w_out[t] = 0.0f; }
    if (n_experts <= 0 || n_experts > LZ_MAX_EXPERTS || top_k <= 0) return;
    if (!(tau > 0.0f)) tempered = 0;   /* also catches NaN, which compares false */

    /* The int input only reaches the sigmoid scorer as integers. Every
       other reader below wants a float vector, so build it once here -
       that is the producer's declined convert and multiply arriving
       after all, which is why it bills at the epilogue's site rather
       than at this function's. */
    if (li && (!sigmoid || tempered)) {
        float deq = pow2f(-li_e);
        for (i = 0; i < n_experts; i++) flog[i] = lz_i32f(li[i]) * deq;
        LZ_FCX(LZ_FC_EPI, n_experts, 0, 0, n_experts, 0);
        logits = flog;
    }

    if (sigmoid) {
        if (li) for (i = 0; i < n_experts; i++) score[i] = lz_sigmoid_i(li[i], li_e);
        else    for (i = 0; i < n_experts; i++) score[i] = lz_sigmoid(logits[i]);
    } else {
        /* softmax: max-subtracted for stability, sum via lz_exp for the
           same cross-platform determinism the rest of the engine uses -
           see lz_exp's contract in ops.h. */
        float m = logits[0], s = 0.0f;
        for (i = 1; i < n_experts; i++) if (logits[i] > m) m = logits[i];
        for (i = 0; i < n_experts; i++) {
            score[i] = lz_exp(logits[i] - m);
            s += score[i];
        }
        for (i = 0; i < n_experts; i++) score[i] /= s;   /* s is a runtime value, not a constant - rule six item 1 */
    }

    /* Router temperature. SELECTION stays on the untempered scores - see
       the header for why that split is the whole point of the knob - so
       this is a second array rather than a rescale of the first.

       Divided by tau rather than multiplied by a precomputed 1/tau: tau
       is a runtime value, so the division is a real FDIV on both
       compilers and rule six item 1 is satisfied either way, but the
       reference this is calibrated against divides, and matching it is
       worth more here than saving a few
       cycles in a per-layer, n_experts-long loop.

       No flush-to-zero here; that is a measurement rather than an
       omission. Dividing by a small tau drives l/tau deep negative,
       which is exactly the operand class rule two clause (c) forbids.
       No floor is needed: lz_exp already returns a hard
       0.0f below -87.3f, one ULP-ish above FLT_MIN, so lz_sigmoid over
       the whole negative range produces 0 subnormals in 200001 samples
       (smallest nonzero 0084c390, exponent field 1). The softmax branch
       cannot reach one either - its post-division scan is also 0 -
       because s is only large when many experts sit near the max, and
       that is the same condition that keeps their exponentials away
       from the floor.
       So the guard against subnormals is lz_exp's own range clamp, one
       call down. That is the dependency, and t_moe_tau pins it. */
    if (tempered) {
        if (sigmoid) {
            for (i = 0; i < n_experts; i++)
                tscore[i] = lz_sigmoid(logits[i] / tau);
        } else {
            float m = logits[0], s = 0.0f;
            for (i = 1; i < n_experts; i++) if (logits[i] > m) m = logits[i];
            for (i = 0; i < n_experts; i++) {
                tscore[i] = lz_exp((logits[i] - m) / tau);
                s += tscore[i];
            }
            /* s >= 1 always: the max element contributes exp(0). */
            for (i = 0; i < n_experts; i++) tscore[i] /= s;
        }
    }

    for (i = 0; i < n_experts; i++)
        key[i] = score[i] + (bias ? bias[i] : 0.0f);

    /* top-k by (score + bias); the GATHERED weight is the plain score
       (see ops.h - the bias enters selection only). O(top_k * n_experts)
       selection, fine at this scale (n_experts is a handful to a few
       dozen in every reference config, and this is not a hot loop next
       to the matmuls around it). */
    for (t = 0; t < top_k && t < n_experts; t++) {
        int best = -1;
        float bk = 0.0f;
        for (i = 0; i < n_experts; i++) {
            if (key[i] <= LZ_MOE_SENTINEL) continue;      /* already selected */
            if (best < 0 || key[i] > bk) { best = i; bk = key[i]; }
        }
        if (best < 0) break;
        idx_out[t] = best;
        w_out[t] = tempered ? tscore[best] : score[best];
        wsum += w_out[t];
        key[best] = LZ_MOE_SENTINEL;              /* remove from further rounds */
    }
    /* Every selected weight underflowed to zero - lz_exp's clamp, not a
       gradual loss of precision. Only reachable in the tempered sigmoid
       path (the softmax one normalizes by a sum that is at least 1), and
       only at small tau with logits far below zero, which is exactly the
       configuration this model has: its router logits sit around -5 and
       --moe-tau goes down to 0.1, so l/tau reaches -50 inside the CLI's
       own bounds and a slightly colder router reaches the clamp. A real
       case, not a defensive branch.
       One-hot on the top-1 selection is the tau -> 0 limit, i.e. the
       answer the arithmetic heads for as it underflows.
       Leaving the zeros instead would silently drop the whole routed
       branch for that token, which no caller checks for. */
    if (tempered && wsum <= 0.0f && idx_out[0] >= 0) {
        w_out[0] = 1.0f;
        wsum = 1.0f;
    }
    if (renormalize && wsum > 0.0f) {
        for (t = 0; t < top_k; t++)
            if (idx_out[t] >= 0) w_out[t] /= wsum; /* wsum is a runtime value */
    }
}

void lz_moe_hits_add(unsigned char *hits, int cap, const int *idx, int k) {
    int t;
    if (!hits || !idx) return;
    for (t = 0; t < k; t++) {
        /* Both skips are real cases, not defensive padding: lz_moe_route
           writes -1 into a slot it could not fill, and cap is the width
           of the array. See the header for why the second one is the
           caller's problem to report rather than this function's to
           hide. */
        if (idx[t] < 0 || idx[t] >= cap) continue;
        if (hits[idx[t]] < 255) hits[idx[t]]++;
    }
}
