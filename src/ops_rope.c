/* RoPE (rotary position embedding). Extracted verbatim from src/ops.c as
   pure code motion - the only change is the translation unit. The SIMD
   tiers it forwards to (lz_rope_sse2, lz_rope_avx2) stay in
   src/ops_sse2.c and src/ops_avx2.c, for the same reason every
   %xmm/%ymm-touching kernel lives there. */

#include <stddef.h>    /* size_t: pointer offsets in the rotation loop */

#include "ops.h"       /* lz_rope's prototype - definition checked against it */
#include "ops_sse2.h"  /* LZ_ROPE_SSE2_EXTERN, lz_rope_sse2 */
#include "ops_avx2.h"  /* LZ_ROPE_AVX2_EXTERN, lz_rope_avx2 - gcc-only,
                          guarded on LZ_AVX2_TU && !__WATCOMC__ inside the
                          header itself. Without this include the AVX2 arm
                          of the dispatch below compiles out entirely and
                          --kernel avx2 falls through to SSE2 in silence. */

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
#if defined(LZ_ROPE_AVX2_EXTERN)
    /* Checked before the SSE2 arm below, as its own top-level
       g_kernel==AVX2 branch: an AVX2 request should take the AVX2
       kernel or nothing, not silently degrade to SSE2 through the
       unconditional call that follows. Both kernels owe the scalar
       path bit-identity, so a silent degradation would be invisible to
       every value-based comparison. */
    if (g_kernel == LZ_KERNEL_AVX2) {
        lz_rope_avx2(v, n_heads, head_dim, rotary_dim, pos, cs);
        return;
    }
#endif /* LZ_ROPE_AVX2_EXTERN */
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

/* Wiring-proof for lz_rope's direct g_kernel==AVX2 branch above, the
   same shape as ops_epi.c's lz_epi_avx2_compiled_in and fwht.c's
   lz_fwht_avx2_compiled_in: returns 1 iff LZ_ROPE_AVX2_EXTERN was
   defined when THIS file compiled, 0 otherwise. A value-only
   AVX2-vs-SSE2/scalar comparison cannot tell a real AVX2 rotation from
   a silently-correct fallback, because the tiers owe each other
   bit-identity by contract - so the test checks this before trusting
   any such comparison. It has to live in the dispatch translation
   unit, not in src/ops_avx2.c: there the macro comes from ops_avx2.c's
   own header under the same guard that wraps that whole TU, so it
   could only ever answer 1. Here it is defined exactly when this
   file's #include "ops_avx2.h" is, which is the wiring being proven. */
int lz_rope_avx2_compiled_in(void) {
#if defined(LZ_ROPE_AVX2_EXTERN)
    return 1;
#else
    return 0;
#endif /* LZ_ROPE_AVX2_EXTERN */
}
