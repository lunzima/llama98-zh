/* The RoPE table builder and the four YaRN helpers it reaches, split out
   of forward.c along the function seam. Code motion only - every body
   here is letter-for-letter what forward.c held, and the four helpers
   stay `static`: nothing outside this file calls them.

   lz_rope_build_table is the one entry point (forward.h; lz_state_alloc
   is its only caller besides tests). It reads only the config and the
   two rope runtime globals, never a tensor - which is what lets it be
   exercised without building a whole model. Callers must already be
   inside lz_fpu_float_begin()/end(): the transcendentals below are only
   cross-compiler bit-identical at PC=24. */

#include "lz_mathf.h"   /* lz_powf/lz_logf/lz_cosf/lz_sinf: float transcendentals, no libm, bit-identical x86/ARM */
#include "forward.h"     /* includes compat.h for lz_time_ms (LZ_PROF* macros) */
#include "lz_int.h"    /* lz_i64: the 64-bit type, portably */

/* yarn's correction-range floor/ceil (lz_state_alloc's rope_cs fill,
   rope_scaling_type==2). Manual truncation, not floorf/ceilf: nothing
   in this engine has called into libm's floor/ceil before, and floor/
   ceil are cheap enough at the small non-negative magnitudes (rotary
   dimension indices) this is ever called with that a library call buys
   nothing but an unaudited cross-compiler dependency - lz_mathf.h's own
   header comment is why every other transcendental here (sin/cos/pow/
   log/exp) is a project-owned implementation instead of libm's. */
static float lz_rope_floorf(float x) {
    float t = (float)(long)x;
    return (x < t) ? t - 1.0f : t;
}
static float lz_rope_ceilf(float x) {
    float t = (float)(long)x;
    return (x > t) ? t + 1.0f : t;
}

/* YaRN's "number of rotations at dimension boundary" inverse.
   Reference: transformers modeling_rope_utils.py, find_correction_dim -
   same name and shape, not renamed, so the two can be read side by
   side. num_rotations is beta_fast/beta_slow; dim
   is rotary_dim, not half of it. Both divisions are by runtime values
   (rule six #1: a literal divisor is not the same instruction on x87
   vs SSE - num_rotations/base/max_pos are all config-derived floats,
   never compile-time constants). */
static float lz_rope_find_correction_dim(float num_rotations, float dim,
                                         float base, float max_pos) {
    float two_pi = 6.283185307179586f;
    return (dim * lz_logf(max_pos / (num_rotations * two_pi))) /
           (2.0f * lz_logf(base));
}

/* find_correction_range, same reference. Always truncate=True (HF's
   own default; this project does not expose the flag - see model.c's
   rope_parameters comment). Clamped to [0, dim-1], same as the
   reference. */
static void lz_rope_find_correction_range(float beta_fast, float beta_slow,
                                          float dim, float base,
                                          float max_pos,
                                          float *out_low, float *out_high) {
    float low = lz_rope_floorf(
        lz_rope_find_correction_dim(beta_fast, dim, base, max_pos));
    float high = lz_rope_ceilf(
        lz_rope_find_correction_dim(beta_slow, dim, base, max_pos));
    if (low < 0.0f) low = 0.0f;
    if (high > dim - 1.0f) high = dim - 1.0f;
    *out_low = low;
    *out_high = high;
}


/* Fills rope_cs[0 .. seq_len*rotary_dim) with pos * (rotary_dim/2)
   {cos, sin} pairs. Factored out of lz_state_alloc so it can be
   exercised - and cross-checked against transformers' own RoPE
   reference - without building a whole model: it only reads `c` and
   the two --rope-ntk runtime globals, never any tensor. Caller must
   already be inside lz_fpu_float_begin()/end() (PC=24 - lz_mathf's
   transcendentals are only cross-compiler bit-identical there). */
void lz_rope_build_table(float *rope_cs, int seq_len, const LZModelConfig *c) {
    int half = c->rotary_dim >> 1;
    int l;
    /* Dynamic NTK: rescale theta once for the whole table, not once
       per position. Guarded on rotary_dim > 2 because the exponent's
       denominator is (rotary_dim - 2) - a model with rotary_dim <= 2
       has no valid ratio for this formula (undefined, not merely
       small), so it degrades to "off" rather than computing garbage.
       Both divisions below are by runtime values (lz_rope_orig_max,
       rotary_dim - 2), never by a float literal: a literal divisor is
       one a compiler may replace with a reciprocal multiply, and the
       x87 and SSE builds have to emit the same sequence here. */
    float theta_eff = c->rope_theta;
    if (lz_rope_dynamic_ntk && c->rotary_dim > 2 &&
        (float)seq_len > lz_rope_orig_max) {
        float ratio = (float)seq_len / lz_rope_orig_max;
        float expo  = (float)c->rotary_dim /
                      (float)(c->rotary_dim - 2);
        theta_eff = c->rope_theta * lz_powf(ratio, expo);
        lz_debug_n_rope_ntk += half;
    }
    /* rope_parameters.rope_type scaling (model.c's own comment on
       rope_scaling_type), composed with dynamic NTK above rather than
       replacing it - the two are independent knobs (one from
       config.json, one from the --rope-ntk runtime flag) and nothing
       says a caller cannot ask for both, though only the type==0
       combination (dynamic NTK alone) is what the reference below was
       ever checked against. Three separate loops, not one branchy
       one: type 0 is by far the common case, and this keeps its
       compiled code identical to what it was before rope_type scaling
       existed, rather than "the same numeric result through new
       branches that happen to be no-ops" - see rule two's own history
       of x87/SSE divergences that looked like no-ops and were not. */
    if (c->rope_scaling_type == 2) {
        /* yarn: NTK-by-parts blend between the extrapolated (unscaled)
           and interpolated (divided by factor) frequency, ramped per
           dimension by the beta_fast/beta_slow correction range, plus
           a post-scale on the final cos/sin (HF's "attention_factor"/
           mscale). Reference: transformers modeling_rope_utils.py,
           _compute_yarn_parameters + Qwen3_5TextRotaryEmbedding.
           forward's `cos = emb.cos() * attention_scaling`. Verified
           against it bit-for-bit at dynamic NTK off by a one-off probe
           that is not in the tree (wrope_yarn.c, under gitignored .prof/). */
        float low, high, attn_scale;
        lz_rope_find_correction_range(
            c->rope_scaling_beta_fast, c->rope_scaling_beta_slow,
            (float)c->rotary_dim, theta_eff,
            c->rope_scaling_orig_max, &low, &high);
        if (low == high) high += 0.001f;   /* HF's own singularity guard */
        attn_scale = c->rope_scaling_attn_factor;
        if (attn_scale <= 0.0f) {
            /* HF's get_mscale(factor) at the default mscale=1 (no
               DeepSeek-style mscale/mscale_all_dim pair - see
               model.h's own field comment on why not). */
            attn_scale = (c->rope_scaling_factor <= 1.0f) ? 1.0f :
                0.1f * lz_logf(c->rope_scaling_factor) + 1.0f;
        }
        for (l = 0; l < seq_len; l++) {
            int i;
            for (i = 0; i < half; i++) {
                float extrap = lz_powf(
                    theta_eff, -2.0f * (float)i / (float)c->rotary_dim);
                float interp = extrap / c->rope_scaling_factor;
                float ramp = ((float)i - low) / (high - low);
                float ang;
                if (ramp < 0.0f) ramp = 0.0f;
                if (ramp > 1.0f) ramp = 1.0f;
                /* ramp==1 -> pure interpolation (near dim 0, high-
                   frequency/short-wavelength rotations); ramp==0 ->
                   pure extrapolation (near dim half-1). Matches
                   interp*(1-extrap_factor) + extrap*extrap_factor with
                   extrap_factor = 1-ramp, algebraically the same
                   expression. */
                ang = (float)l * (interp * ramp + extrap * (1.0f - ramp));
                rope_cs[((size_t)l * half + i) * 2] =
                    lz_cosf(ang) * attn_scale;
                rope_cs[((size_t)l * half + i) * 2 + 1] =
                    lz_sinf(ang) * attn_scale;
            }
        }
    } else if (c->rope_scaling_type == 1) {
        /* linear: the frequency is divided by factor uniformly
           (equivalently, position ids scaled by 1/factor - see
           model.c's rope_parameters comment for why this is applied
           to the frequency instead, matching transformers' own
           _compute_linear_scaling_rope_parameters comment on the two
           being equivalent). No post-scale (attention_factor is 1.0
           for this type in the reference). */
        for (l = 0; l < seq_len; l++) {
            int i;
            for (i = 0; i < half; i++) {
                float freq = lz_powf(
                    theta_eff, -2.0f * (float)i / (float)c->rotary_dim);
                float ang = (float)l * (freq / c->rope_scaling_factor);
                rope_cs[((size_t)l * half + i) * 2]     = lz_cosf(ang);
                rope_cs[((size_t)l * half + i) * 2 + 1] = lz_sinf(ang);
            }
        }
    } else {
        for (l = 0; l < seq_len; l++) {
            int i;
            for (i = 0; i < half; i++) {
                float freq = lz_powf(theta_eff,
                                     -2.0f * (float)i / (float)c->rotary_dim);
                float ang = (float)l * freq;
                rope_cs[((size_t)l * half + i) * 2]     = lz_cosf(ang);
                rope_cs[((size_t)l * half + i) * 2 + 1] = lz_sinf(ang);
            }
        }
    }
}

/* The conjugate table: same {cos, sin} layout as lz_rope_build_table's
   output, every sin negated, cos untouched (cos is even, sin is odd, so
   this is R(-angle) element for element).

   lz_rope reads a pair as `x0*c - x1*s, x1*c + x0*s`; fed `(c, -s)` that
   same expression is `x0*c + x1*s, x1*c - x0*s`, which is R(angle)^T -
   the inverse rotation. Doing it here rather than in the kernels is what
   keeps the inverse available to every tier (scalar, SSE2, AVX2, and the
   hand-written Watcom ones) without a second implementation of each: they
   are the same code reading a different table, so they cannot disagree
   about what the inverse is.

   Negation is exact in IEEE 754 - the sign bit flips and nothing else -
   so this is the conjugate, not a rounding of it. A kernel that
   conjugated internally would compute the identical product.

   `n` is the element count, i.e. seq_len * rotary_dim; the odd-indexed
   elements are the sines, because the table is stored as {cos, sin}
   pairs. No transcendental runs here, so unlike lz_rope_build_table this
   needs no PC=24 region. */
void lz_rope_conjugate_table(float *dst, const float *src, size_t n) {
    size_t i;
    for (i = 0; i < n; i++)
        dst[i] = (i & 1u) ? -src[i] : src[i];
}
