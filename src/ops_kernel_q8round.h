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
   and the group-level loops that call them, live in src/ops_mmx_sse.c
   (SSE1+MMX tier, cvtps2pi writes %mm registers) and src/ops_sse2.c
   (pure SSE2 tier) - see those two call sites (search "one call per
   GROUP") for why the group, not the sub-chunk, is the granularity: gs
   can be much larger than 32, and this family's pragmas are called once
   per sub-chunk directly, unlike ops_kernel_amax.h's/ops_kernel_norm.h's,
   which sit behind a table-dispatched wrapper with no per-call-site
   change.

   No lz_q8round32_sse/lz_q8round32_simd bare-name aliases in this file:
   they would reference names not declared in this TU, and nothing
   under __WATCOMC__ needs them - both call sites reach
   lz_q8round_group_sse/_sse2 and lz_q8round_2p_group_sse/_sse2
   directly, the same way the p2 and wsum aliases were retired. */
/* The q8round kernel bodies - lz_q8round32_sse_asm / lz_q8round32_sse2_asm's
   #pragma aux pairs and their group-level loops on Watcom,
   lz_q8round32_sse / lz_q8round32_simd intrinsics on gcc - live in
   src/ops_mmx_sse.c / src/ops_sse2.c. The LZ_HAVE_Q8R_SSE /
   LZ_HAVE_Q8R_SIMD flags that say "this tier is in the build" are
   defined in src/ops_mmx.h / src/ops_sse2.h under LZ_MMX_TU /
   LZ_SSE2_TU, both included near the top of ops.c before this file. */
