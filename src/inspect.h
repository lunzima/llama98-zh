#ifndef LZ_INSPECT_H
#define LZ_INSPECT_H

#define LZ_INSPECT_CAND_MAX 10
/* Experts the snapshot can describe. Sized to the panel's lamp row
   rather than to any model: a model with more experts than this is
   reported truncated, never silently shown as if it had exactly this
   many. */
#define LZ_INSPECT_EXPERT_MAX 32

/* One token's worth of "what just happened inside", filled by the engine
   only when the caller asks for it.

   Numbers only, no strings: turning a token id into text needs the
   tokenizer, and the tokenizer belongs to the caller. Keeping text out
   of here is also what keeps the engine layer free of user-facing
   strings (iron law one). */
typedef struct {
    /* How many of this token's MoE layers chose each expert. Zero means
       "not used by any layer"; the count, not just the fact, because the
       two are visibly different things and the display draws them
       differently.

       Measured on recover-r10-q8 (16 experts, top-2, 8 MoE layers, 200
       tokens): of every expert a token used at all, 60.0% were picked by
       exactly one layer, 31.5% by two, 7.7% by three, 0.8% by four. So
       the count has a real spread and a short one - which is why it is a
       count with a small ceiling rather than a weight.

       Routing WEIGHT is deliberately not here: on the same measurement
       79.6% of the per-expert maxima fell between 0.3 and 0.7, a band too
       narrow to encode as anything a reader could distinguish. A display
       driven by it would look like information and carry none. */
    unsigned char expert_hits[LZ_INSPECT_EXPERT_MAX];
    int n_experts;          /* 0 when the model has no MoE at all */
    int experts_truncated;  /* 1 when n_experts > LZ_INSPECT_EXPERT_MAX */

    /* The sampler's own shortlist, already sorted by probability
       descending. n_survived is the TRUE count after top_k/top_p/min_p,
       which can exceed LZ_INSPECT_CAND_MAX - the panel shows the first
       ten and reports the real number. */
    int   n_survived;
    int   n_cand;                        /* min(n_survived, CAND_MAX) */
    int   cand_id[LZ_INSPECT_CAND_MAX];
    float cand_p[LZ_INSPECT_CAND_MAX];   /* normalized over the survivors */
} LZInspect;

#endif
