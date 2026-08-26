/* RoPE (rotary position embedding). Extracted verbatim from src/ops.c as
   pure code motion - the only change is the translation unit. The SSE2
   tier it forwards to (lz_rope_sse2) stays in src/ops_sse2.c, for the
   same reason every %xmm-touching kernel lives there. */

#include <stddef.h>    /* size_t: pointer offsets in the rotation loop */

#include "ops.h"       /* lz_rope's prototype - definition checked against it */
#include "ops_sse2.h"  /* LZ_ROPE_SSE2_EXTERN, lz_rope_sse2 */

/* lz_rope_sse2's body lives in src/ops_sse2.c: it writes %xmm
   registers, so it belongs in the translation unit built with -msse2.
   Declared in src/ops_sse2.h, guarded there by LZ_ROPE_SSE2_EXTERN. */

void lz_rope(float *v, int n_heads, int head_dim, int rotary_dim,
             int pos, const float *cs) {
    /* Bit-exact contract: every kernel below reproduces the scalar path
       per-element (each rotation is an independent IEEE mul/add). MMX is
       deliberately NOT used here - MMX has no floating point, so any MMX
       RoPE would have to quantize (x Q8 / cos-sin Q14), which breaks
       bit-exactness; RoPE is not a PII bottleneck (~2-3us/token), so
       correctness wins and MMX builds fall through to scalar. */
    /* LZ_ROPE_SSE2_EXTERN is defined (in src/ops_sse2.h) exactly when
       src/ops_sse2.c's lz_rope_sse2 is part of this link - the single
       source of truth for both this call site and the extern
       declaration, so the two cannot drift the way two independent
       `#if defined(__SSE2__) && !defined(__WATCOMC__)` copies once
       could (one compiling the definition out while the call stayed,
       failing the build with an implicit-declaration error). */
#if defined(LZ_ROPE_SSE2_EXTERN)
    lz_rope_sse2(v, n_heads, head_dim, rotary_dim, pos, cs);
#else
    int half = rotary_dim / 2;
    int h, i;
    const float *row = cs + (size_t)pos * half * 2;
    for (i = 0; i < half; i++) {
        float c = row[i * 2];
        float s = row[i * 2 + 1];
        for (h = 0; h < n_heads; h++) {
            float *base = v + (size_t)h * head_dim;
            float x0 = base[i];
            float x1 = base[i + half];
            base[i]        = x0 * c - x1 * s;
            base[i + half] = x1 * c + x0 * s;
        }
    }
    /* components beyond rotary_dim stay as-is; they do not rotate */
#endif /* LZ_ROPE_SSE2_EXTERN */
}
