/* Q8 activation-rounding kernels: scalar magic-add lives in ops.c, the
   SIMD tiers live here. NOT a standalone translation unit - ops.c
   #includes this, and it must stay that way.

   Why a header and not a .c: code LAYOUT alone moves decode 3-7% on the
   dev box (measured: -falign-functions=64 turned the same change from
   0.976x to 1.024x). Separate translation units would change
   inlining decisions and layout at once, turning a pure move into an
   unattributable performance change, and every `static` here would stop
   being static. Splitting the file is a readability change; it is not
   allowed to be a codegen change.

   The .h extension is a lie of convenience: there are no include guards
   because this is included exactly once, and including it twice would
   redefine every static. */
/* lz_q8round32_sse_asm / lz_q8round32_sse2_asm's #pragma aux bodies,
   and the group-level loops that used to call them (once per 32-element
   sub-chunk, in ops.c's lz_quantize_q8 and lz_gdn_quantize_2p), moved
   to src/ops_mmx_sse.c (SSE1+MMX tier, cvtps2pi writes %mm registers)
   and src/ops_sse2.c (pure SSE2 tier) in task #30 (4/N) - see those two
   call sites (search "one call per GROUP") for why the group, not the
   sub-chunk, is the granularity: gs can be much larger than 32, and
   this family's pragmas were called once per sub-chunk directly, unlike
   ops_kernel_amax.h's/ops_kernel_norm.h's, which were already behind a
   table-dispatched wrapper with no per-call-site change.

   No lz_q8round32_sse/lz_q8round32_simd bare-name aliases here anymore:
   they would reference names no longer declared in this TU, and
   nothing under __WATCOMC__ needs them - both call sites reach
   lz_q8round_group_sse/_sse2 and lz_q8round_2p_group_sse/_sse2
   directly now, the same cleanup task #30 (2/N) and (3/N) did for the
   p2 and wsum aliases. */
#if defined(__WATCOMC__)
#define LZ_HAVE_Q8R_SSE  1
#define LZ_HAVE_Q8R_SIMD 1
#else
/* lz_q8round32_sse's body and the LZ_HAVE_Q8R_SSE definition for this
   branch live in src/ops_mmx_sse.c / src/ops_mmx.h: cvtps2pi
   writes %mm registers, so it belongs in the one translation unit built
   with -mmmx. ops_mmx.h is included near the top of ops.c and defines
   LZ_HAVE_Q8R_SSE there when the MMX tier is in this build.

   lz_q8round32_simd's SSE2 body and the LZ_HAVE_Q8R_SIMD definition for
   this branch live in src/ops_sse2.c / src/ops_sse2.h: it writes
   %xmm registers, so it belongs in the translation unit built with
   -msse2. ops_sse2.h is included near the top of ops.c. */
#endif
