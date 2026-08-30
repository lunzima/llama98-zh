/* Every 32-element dot-product kernel, all ISAs and both impls. Included
   once by ops.c; see the header comment of ops_kernel_q8round.h for why
   this is a header rather than a translation unit, and why it has no
   include guards.

   Contents, in file order (grouped by FORMAT with the ISA branches
   nested inside, which is what makes the guards hard to follow):

     part32_* / fold4_sse2          sse2-intrin  (gcc, !LZ_USE_MMX)
     x16 / q16 / q41 / q61          each: #ifdef __WATCOMC__ asm
     nowiden                              #else  mmx-intrin
*/

/* THE GUARD MUST MATCH THE USE SITE. A symbol whose definition is
   compiled under a wider predicate than its caller is either an
   undefined reference in one configuration or dead code in another, and
   both only show up in a build nobody happens to run.

   The scalar reference for a 32-element dot product is
   matmul_scalar_ref in ops.c, not a private copy here: it is the path
   every SIMD tier is judged bit-identical against. Three symbols in
   this file and ops.c had drifted wider than their callers, and the
   ARMv5TE cross-build - where neither __SSE2__ nor __MMX__ is defined -
   is the configuration that reports them. */

/* fold4_sse2 and the five part32_* kernels (x16/q41/q61/t2/q16) are in
   src/ops_kernel_dot_sse2.h, included only by src/ops_sse2.c - they
   write %xmm registers. Declared (as the row_*_sse2_intrin functions
   that call them) in src/ops_sse2.h, visible in ops.c. */

/* MMX dot product with activations pre-expanded to int16, same
   algorithm as SSE2's part32_x16: weights via punpcklbw(0, w) (value
   x256, skips psraw), activations loaded directly as int16. SIMD
   instruction count per 32 elements drops from ~57 to ~35. Products
   scale 256x, cancelled once at the row end by the caller.

   dot32_x16_mmx / dot32_x16_mmx_2's gcc bodies live in
   src/ops_kernel_dot_mmx.h (included only by src/ops_mmx.c - both
   write %mm registers); this #ifdef's Watcom half is unaffected and
   unchanged. */
#ifdef __WATCOMC__
/* lz_dot32_x16_asm / lz_dot32_x16_asm_2 (MMX) and lz_dot32_x16_sse2_asm
   (SSE2), plus their dot32_x16_mmx/dot32_x16_sse2a wrappers, live in
   src/ops_mmx.c and src/ops_sse2.c respectively - together with
   row_q8_mmx_asm/row_q8_sse2_asm, which call them direct-by-name and
   therefore sit in the SAME TU to keep that call inline (see ops.c's
   own Q8 row-kernel section for the accounting). Algorithm
   documentation (weight-side widening via punpcklbw(zero,w), the
   paired kernel's register budget, why SSE2 needs no emms) is at the
   kernels' new site. */
#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_x16_mmx (single token) and dot32_x16_mmx_2 (two tokens, ONE
   weight unpack - 36/24 ceiling, NT=2 register budget) live in
   src/ops_kernel_dot_mmx.h, included only by src/ops_mmx.c. Declared in
   src/ops_mmx.h, already visible in this translation unit. */
#endif

/* ---- Q16_0 32/128-element kernels -------------------------------------
   Q16_0 stores int16 weights, so unlike Q8_0 there is no int8->int16
   unpack step and therefore no x256 product scaling: these return the
   raw int32 dot, and the float epilogue has no 1/256 to cancel.

   Accumulator bound is 32767*127*32 = 1.33e8, well past 2^24, so every
   int32->float conversion downstream MUST go through lz_i32f(). This is
   not a precaution: on 4096 real out_proj sub-blocks with |acc| > 2^24,
   a bare (float) cast disagrees between the two compilers on 1210 of
   them (29.5%, always exactly 1 ULP), while lz_i32f() disagrees on 0.
   The divergence is in the load, not the arithmetic,
   and why PC=24 does not cover it. */
#if defined(__WATCOMC__)
/* lz_dot32_q16_asm / lz_dot32_q16_sse2_asm live in src/ops_mmx.c /
   src/ops_sse2.c, alongside row_q16_mmx_asm / row_q16_sse2_asm which
   call them direct-by-name. */

/* Group-at-once (128 elements, 4 subblocks) version: same shape as
   lz_dot128_x16_asm. The 4 subblock sums are written out separately
   (each has its own xqs, so the float reduction cannot add them up
   inside the kernel). */
/* lz_dot128_q16_asm lives in src/ops_mmx.c, alongside row_q16_mmx_asm
   which calls it direct-by-name. Also declared (as lz_dot128_q16_mmx_w's
   twin) for -DLZ_SSE2_GROUP=0. */

/* lz_dot128_q16_sse2_asm lives in src/ops_sse2.c, alongside
   row_q16_sse2_asm which calls it direct-by-name. */

/* dot32_q16_mmx / dot32_q16_sse2a live in src/ops_mmx.c /
   src/ops_sse2.c, alongside row_q16_mmx_asm / row_q16_sse2_asm. */
#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_q16_mmx's body lives in src/ops_kernel_dot_mmx.h, included only
   by src/ops_mmx.c - it writes %mm registers. Declared in
   src/ops_mmx.h, already visible in this translation unit. */
#endif /* Q16_0 kernels */

/* Q4_1 32-element dot product (MMX). Same algorithm as SSE2's
   part32_q41: pand(b,0x0F) grabs elements 0..15,
   pand(psrlw(b,4),0x0F) elements 16..31, then punpcklbw(0,x) expands
   to x256 int16 for pmaddwd. Returns the 32-element dot scaled x256;
   the caller cancels once at the row end. */
#ifdef __WATCOMC__
/* lz_dot32_q41_asm / lz_dot32_q41_asm_2 (MMX) and lz_dot32_q41_sse2_asm
   (SSE2), plus their dot32_q41_mmx/dot32_q41_sse2a wrappers, live in
   src/ops_mmx.c and src/ops_sse2.c respectively - together with
   row_q41_mmx_asm/row_q41_sse2_asm, which call them direct-by-name and
   therefore sit in the SAME TU. Algorithm documentation travels with
   the code. */
#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_q41_mmx (single token) and dot32_q41_mmx_2 (two tokens, ONE
   nibble unpack - the kernel that matters most, see the Watcom twin's
   comment) live in src/ops_kernel_dot_mmx.h, included only by
   src/ops_mmx.c. Declared in src/ops_mmx.h, already visible here. */
#endif /* Q4_1 __WATCOMC__ branch end */

/* ---- T2 (ternary, one 2-bit plane) ------------------------------------ */

/* Originally emitted by tools/kernelgen/gen_t2.py IN THE TRAINING TREE -
   the engine repository carries no tools, so the generator lives there
   and only its output is pasted here. The generator is scaffolding, not
   the source of truth: hand-editing is fine, but if you regenerate,
   diff against what is here first - anything hand-tuned since will be
   silently reverted otherwise.

   MMX 70 instructions / 9 loads; SSE2 47 / 5. For scale, Q6_1 is 86 uop
   / 11 loads and 59 / 6 - T2 is the same kernel with the 4-bit plane
   deleted, which is exactly why it was given Q6_1'''s 2-bit layout.

   VERIFIED BIT-IDENTICAL:
   tools/kernelgen/wt2.c compares both against the scalar reference over
   a 32-position sweep (one nonzero weight at a time, both signs - a
   uniform block hides a wrong shift), six extreme patterns, and 200,000
   random trials. All three agree exactly; int32 accumulation is exact
   so there is no tolerance to argue about. */

#if defined(__WATCOMC__)
/* lz_dot32_t2_asm / lz_dot32_t2_sse2_asm live in src/ops_mmx.c /
   src/ops_sse2.c, alongside row_t2_mmx_asm / row_t2_sse2_asm which
   call them direct-by-name. No 128-wide group kernel exists for this
   format, so no -DLZ_SSE2_GROUP=0 wrapper is needed. */
#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_t2_mmx's body lives in src/ops_kernel_dot_mmx.h, included only
   by src/ops_mmx.c - it writes %mm registers. Declared in
   src/ops_mmx.h, already visible in this translation unit. */
#endif  /* __WATCOMC__ / __MMX__ */

/* ---- Q6_1 (6.5 bits: 4-bit + 2-bit dual planes) ------------------------ */
#if defined(__WATCOMC__)
/* lz_dot32_q61_asm / dot32_q61_mmx live in src/ops_mmx.c, alongside
   row_q61_mmx_asm which calls them direct-by-name. Both this format's
   SSE2 kernels are genuine %xmm-only implementations, not an MMX
   reuse. */

/* Originally generated, hand-edited since - if you regenerate, diff
   the generator's output against what is here first. 59 instructions / 6 loads.

   **Which is faster than the MMX version cannot be decided statically.**
   SSE2 is 59 instructions / ~116 uop; MMX is 78 / 86 uop - +35% uop,
   -24% instructions, the two directions oppose. Pentium M's
   (Banias/Dothan) SSE units are 64-bit wide, splitting 128-bit
   instructions into two uops - but split uops are not necessarily
   slower: Zen 1's AVX-256 also splits into two 128-bit uops with the
   same uop total as equivalent AVX-128 code, yet still wins, on front-
   end bandwidth and paired scheduling. The extra uops are not madds
   (those are even); they are fixed overhead paid twice (masks 6->12,
   epilogue 5->14) plus punpcklqdq, the inherent tax of the 128-bit
   layout. **That fixed overhead is fixable** - processing a whole
   gs=128 group at once amortizes it, and SSE2 benefits twice as much
   as MMX. So the SSE2 tier temporarily still uses the MMX kernel - not
   judged slower, just no target machine to measure on.

   Two SSE2-specific constraints (know before writing):
   - `pmaddwd xmm, m128` requires 16-byte alignment, which activation
     buffers do not guarantee, so `movdqu` to a register then
     `pmaddwd reg,reg` is mandatory - the instructions the MMX version
     saves via memory operands cannot be saved here.
   - 32-bit mode has only 8 xmms; the zero register becomes an in-place
     `pxor xmm2,xmm2` to fit. */
/* lz_dot32_q61_sse2_asm lives in src/ops_sse2.c, alongside
   row_q61_sse2_asm and dot32_q61_sse2a. */

/* lz_dot128_x16_sse2_asm (Q8_0's 128-wide SSE2 group kernel) lives in
   src/ops_sse2.c, alongside row_q8_sse2_asm which calls it
   direct-by-name. */

/* lz_dot128_q41_sse2_asm (Q4_1's 128-wide SSE2 group kernel) lives in
   src/ops_sse2.c, alongside row_q41_sse2_asm which calls it
   direct-by-name. Also declared (as lz_dot128_q41_mmx_w's twin, see
   that wrapper's own comment) for -DLZ_SSE2_GROUP=0. */

/* PAIRED twin: two tokens, ONE two-plane merge. Q6_1 gains the most of
   the three formats - the weight side is 61% of the kernel's ops, and
   on the recipe of record it carries the most bytes - so the ceiling is
   2.54x (the gcc twin measured 1.271x on prefill).

   Six parameters, so all six of Watcom's register slots are in use, and
   mm7 is NOT a persistent zero the way it is in the other two twins:
   with m0f, m03, the 2-bit plane, two accumulators and the composed
   6-bit values already resident, keeping a zero would leave nothing for
   the second token's copy of the widened half. `pxor mm4, mm4` before
   each punpck costs one instruction and buys that register back.

   The 2-bit plane's shifts (0/2/4/6) are 16-bit shifts, which is
   byte-wise equivalent here: after `& 3` only each byte's own low two
   bits survive and no shift used reaches across a byte boundary.

   No emms here: the caller emits one after the row loop.

   VERIFIED BIT-IDENTICAL: 200000 random plus 8 extreme trials against
   a scalar reference, 0 mismatches; sensitivity checked
   by mutating one 2-bit plane shift (psrlw 6 -> psrlw 5), which turned
   it into 199997 mismatches. */
/* lz_dot32_q61_asm_2 (PAIRED twin, MMX only - the SSE2 row function
   does not pair) lives in src/ops_mmx.c. dot32_q61_sse2a lives in
   src/ops_sse2.c alongside lz_dot32_q61_sse2_asm above. */

/* lz_dot128_x16_asm (Q8_0's 128-wide MMX group kernel - "one group at
   a time", amortizing the fixed mask/prologue overhead from once per
   32 elements to once per 128) lives in src/ops_mmx.c, alongside
   row_q8_mmx_asm which calls it direct-by-name. Also declared (as
   lz_dot128_x16_mmx_w's twin, see that wrapper's own comment) for
   src/ops_sse2.c's -DLZ_SSE2_GROUP=0 A/B control build. */

/* lz_dot128_q41_asm (Q4_1's 128-wide MMX group kernel) lives in
   src/ops_mmx.c, alongside row_q41_mmx_asm which calls it
   direct-by-name. Also declared (as lz_dot128_q41_mmx_w's twin, see
   that wrapper's own comment) for src/ops_sse2.c's -DLZ_SSE2_GROUP=0
   A/B control build. */

/* lz_dot128_q61_asm lives in src/ops_mmx.c, alongside row_q61_mmx_asm
   which calls it direct-by-name. Also declared (as lz_dot128_q61_mmx_w's
   twin) for -DLZ_SSE2_GROUP=0. */
/* lz_dot128_q61_sse2_asm lives in src/ops_sse2.c, alongside
   row_q61_sse2_asm which calls it direct-by-name. */
#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_q61_mmx (single token) and dot32_q61_mmx_2 (two tokens, ONE
   two-plane merge - the kernel that matters most on the recipe of
   record, see the Watcom twin's comment) live in
   src/ops_kernel_dot_mmx.h, included only by src/ops_mmx.c. Declared in
   src/ops_mmx.h, already visible here. */
#endif
