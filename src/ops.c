#include "lz_int.h"   /* lz_i64/lz_u64 and the width types:
                         Visual C++ 4.0 has neither <stdint.h> nor
                         `long long`, and 236 sites deciding that for
                         themselves is the shape one-authority forbids */
#include <stddef.h>   /* size_t: used a lot here for pointer offsets; math.h does not guarantee it */
#include <float.h>    /* FLT_MAX: overflow guard for quant reciprocal */
#include <math.h>
#include <stdlib.h>   /* malloc/free: the epilogue's derived scale plane
                         is the only allocation in this file. Watcom
                         reports W131 without it and -we makes that an
                         error; gcc picked the prototypes up from
                         elsewhere and said nothing. */
#include <string.h>   /* memcpy: unaligned loads in the MMX/SSE2 kernels */

#include "ops.h"
#include "err.h"
#include "cpucheck.h"   /* LZ_CPUID1_EDX_AUX - shared CPUID leaf 1 EDX asm */
#include "mmx_compat.h" /* _mm_empty(): called from the architecture-neutral
                           dispatch paths too, so it must be declared on
                           targets that have no MMX at all - see the header */
#include "ops_mmx.h"    /* declarations for src/ops_mmx.c's functions - the
                           %mm-touching kernels are defined there. Unconditional:
                           the macros and declarations inside are themselves
                           guarded on LZ_MMX_TU, a build flag (-D, from the
                           Makefile) rather than on __MMX__, which this file
                           must never be compiled with. */
#include "ops_sse.h"    /* the SSE1-on-%xmm half of src/ops_mmx_sse.c. A
                           separate include from ops_mmx.h on purpose: those
                           kernels need an SSE1 machine, and an include line
                           saying "mmx" would not say so. */
#include "ops_sse2.h"   /* declarations for src/ops_sse2.c's functions - the
                           %xmm-touching kernels are defined there. Same
                           LZ_SSE2_TU/__SSE2__ contract, one ISA tier up. */
#include "ops_kernel_shared.h" /* lz_i32f, LZ_WSUM_CHUNK, LZ_SLOT_NEXT,
                           gdn_tail_row, p2_shift_of - shared with the
                           suffixed TUs; included this early because lz_i32f's
                           first use is well before any of the ops_kernel_*.h
                           #includes below. */

/* LZ_GDN_FIXED and its rationale are in ops.h. */

#include "ops_quant.h"   /* pow2f, q8_round, q8_amax, sigmoid_q15,
                           sigmoid_q15_t, lz_sig_q15_fold,
                           lz_sig_q15_t_offset, norm_ss_fixed,
                           lz_exp_float_body, lz_rsqrt_float_body,
                           lz_exp_fixed, lz_rsqrt_fixed,
                           LZ_Q8_MIN_SCALE_F, LZ_Q8_INV127, LZ_Q15_INV */
#include "ops_sched.h"   /* lz_q8r_tier, q8r_have_simd */
#include "ops_matmul.h"  /* matmul impls, LZ_ROW_* tables, epilogue fns */
#include "ops_t2_arm.h"  /* LZ_T2_ARM_ASM_EXTERN */
#include "ops_arm.h"     /* LZ_ARM_ASM_EXTERN */

/* g_kernel defined in ops_sched.c; extern in ops_kernel_shared.h. */


/* g_sig and the sigmoid tier selectors are in ops_sched.c. */

/* ---- Q8 quantization per-element rounding: SSE2 version (32 at a time) ----

   The scalar version is ~12 instructions per element, of which the
   `fstp [tmp]; mov eax,[tmp]` pair is the price of magic-number
   rounding dodging control-word rewrites - the control word is
   avoided, the store/load is not.

   SSE2's `cvtps2dq` does this directly, and **the default MXCSR is
   round-half-even, the same rule as the magic add** - so this is not
   "numerically close" but BIT-IDENTICAL: exhaustively checked every
   quarter-integer within +/-127, 16 ULPs on each side of
   every half-integer, the zero neighborhood (incl. +/-0 and subnormals),
   and 40K random groups - 1,297,000 elements, 0 mismatches, verified
   on both compilers.

   Measured (same method, min of 7 rounds):

     Open Watcom 32-bit x87   24.50 ms -> 0.89 ms   27.4x
     gcc -O2 (SSE)            3.06 ms -> 0.80 ms    3.8x

   The only behavioral difference is NaN input: the scalar gives 0,
   SSE2's cvtps2dq gives 0x80000000 for unrepresentable inputs,
   saturating to -127. Both are garbage; NaN reaching here in
   production means the upstream is already destroyed - no slow path
   kept for it.

   The clamping stays. On the fast path |grp[k]| <= amax and
   inv = 127/amax, so qi is always in [-127,127] and pminsw/pmaxsw are
   dead code; but packsswb's saturation boundary is -128, not -127 -
   keeping these 4 instructions is what makes this match the scalar
   reference on all non-NaN inputs.

   **No emms emitted**: xmm and x87/MMX do not share register files
   (the no-emms convention only covers MMX), so this kernel can be called straight
   from lz_gdn_step's x87 section.

   No PIII tier - `cvtps2dq` is SSE2;
   SSE1 only has 2-lane `cvtps2pi` (writes MMX registers, emms
   management needed), that is a separate build. */
/* Included HERE, not at the top: the position is the whole point. These
   kernels see exactly the declarations in scope at this point, so
   including them here is provably a no-op - the code section is
   byte-identical across all three builds. Hoisting the #include would
   change that. */
#include "ops_kernel_q8round.h"

/* lz_q8r_tier and q8r_have_simd: declared in ops_sched.h,
   defined in ops_sched.c. */
/* lz_cpu_has_mmx is an inline read of g_lz_has_mmx, in ops_quant.h,
   which this file already includes. The one-time probe behind it is
   defined further down. */

/* lz_i32f (from ops_kernel_shared.h), q8_amax, q8_group_scale,
   LZ_Q8_MIN_SCALE_F, LZ_Q8_INV127 and LZ_Q15_INV are in
   ops_quant.c, as are the lz_quantize_q8* entry points (moved
   from this file). */

#include "ops_kernel_dot.h"

/* ---------------------------------------------- runtime kernel dispatch

   One Win98 binary must run on both a PII (MMX only) and a T42/Pentium M
   (SSE2), so both asm kernels are linked in and selected once at startup
   via CPUID.

   **The dispatch point is at the matmul level**: one branch per matmul,
   after which the 32-element kernel inside the row loop is a DIRECT call.
   Dispatching deeper would pay one indirect call per 32 MACs - exactly
   what this kernel design exists to avoid. */
/* Defined near the top of the file, above q8_amax, which also reads it. */

#if defined(__WATCOMC__)
/* CPUID leaf 1 EDX: bit 23 = MMX, bit 25 = SSE, bit 26 = SSE2. One
   symbol, one implementation, in cpucheck.c - this file calls it through
   the cpucheck.h declaration. A real __asm-block function, not a
   duplicated #pragma aux (see cpucheck.h for why those two cannot share
   a body across files). */
#endif /* __WATCOMC__ */

/* pf_detect, lz_prefetch_mode, g_pair, lz_pair_mode: in ops_sched.c. */

/* Derived bound for the fixed epilogue: per group the product is
   |acc| * |xw*ww| <= 2^27 * 2^30, so ng of them stay inside
   int64 while ng <= 64. Value in ops.h; the derivation stays here. */

/* The narrowing half of the above, in the MAGNITUDE domain, factored out
   because epi_q41_fixed's exponent join needs the same shift with an
   int64 rather than a bounded int16 exit. Both of the corners the
   verification probe covers live here - the +half bias added in unsigned
   64-bit (umag <= 2^63, half <= 2^62, so the sum cannot wrap) and the
   closed form for rs >= 64, where `>>rs` and `1ULL<<(rs-1)` are
   themselves undefined: umag * 2^-rs <= 0.5 there, with equality (and so
   a half-away-from-zero result of 1) only at rs == 64, umag == 2^63. */
lz_u64 epi_shr_half_u(lz_u64 umag, int rs) {
    if (rs <= 0) return umag;
    if (rs >= 64)
        return (rs == 64 && umag == ((lz_u64)1 << 63)) ? 1 : 0;
    return (umag + ((lz_u64)1 << (rs - 1))) >> rs;
}

/* Signed wrapper. The magnitude is taken by converting to unsigned FIRST
   for the LLONG_MIN reason epi_align_i16's header gives; the result is
   below 2^63 for every rs >= 1 (widest is rs == 1, where
   (2^63 + 2^62) >> 1 < 2^63), so negating it back is representable.

   The sign work is branchless for the TARGET's sake: the two `neg ? :`
   ternaries this replaced were 144,472,859 evaluations each with the
   minority arm at 34.5%, and a P5 has no predictor worth the name.
   `mask` comes from the sign bit of the UNSIGNED value - `um >> 63` is
   defined where `s >> 63` on a signed would be implementation-defined -
   and (x ^ mask) - mask is two's-complement abs in and negate out.

   The shift costs a helper call on wcc386, and no portable spelling
   avoids it. Four were measured there, function bodies including
   prologue:
     (lz_u64)0 - (um >> 63)          13 insns, 1 __U8RS call, no branch
     (lz_u64)0 - (lz_u64)(s < 0)     15 insns, no call, ONE BRANCH
     (lz_u64)(s >> 63)                8 insns, 1 call, implementation
                                      -defined for negative s
     (uint32_t)(um >> 32) >> 31      13 insns, 1 __U8RS call
   wcc386 routes every 64-bit shift through __U8RS whether or not the
   count is a compile-time constant, so narrowing first buys nothing;
   and the compare form trades the call for exactly the branch this
   code exists to avoid. The form here is the best of the four.
   What WOULD do it in three instructions is a #pragma aux (mov the high
   dword, sar 31), which is a new Watcom primitive and owes a gcc twin
   and a kernel-matrix cell - gcc already emits the optimal two
   instructions from the C, so that twin would exist only to satisfy the
   pairing. Not done; the cost is recorded so it need not be re-derived.

   `s == 0` needs no special case, and a fast path for it would never
   fire: the general path gives umag = 0, and epi_shr_half_u(0, rs) is 0
   for every rs.

   `rs <= 0` STAYS, and not for its share: at rs <= 0 with s == LLONG_MIN
   the magnitude is 2^63 and converting it back with (lz_i64) is
   implementation-defined. Every rs >= 1 keeps the result under 2^63 (see
   above), which is what makes the general path safe and this case not.
   Removing it is a correctness bug no data from this model would show. */
/* Non-static: the conv1d fixed tier (ops_conv1d.c) uses the same
   half-away-from-zero narrowing. The u variant is also non-static - the
   epilogue (ops_epi.c) shares it with this TU. */
lz_i64 epi_shr_half(lz_i64 s, int rs) {
    lz_u64 um = (lz_u64)s;
    lz_u64 mask = (lz_u64)0 - (um >> 63);
    lz_u64 umag;
    if (rs <= 0) return s;
    umag = epi_shr_half_u((um ^ mask) - mask, rs);
    return (lz_i64)((umag ^ mask) - mask);
}

int lz_act_gs(const LZTensor *w, int in_dim) {
    /* bf16 joins F32 here: it carries no group scale, so there is no
       activation group to follow. */
    if (!w || w->dtype == LZ_FMT_F32 || w->dtype == LZ_FMT_BF16) return 0;
    if (w->gs >= 32 && (w->gs % 32) == 0 && (in_dim % 32) == 0) return 32;
    return w->gs;                   /* degenerate exported tier: keep old "activation follows weight" */
}

/* Scalar reference path: every format must be CORRECT here.

   This is not a fallback; it is the CONTRACT. SIMD kernels are written
   per (format x instruction set); any missing cell falls here
   automatically. Absence is the norm, not the exception: Watcom's
   MMX intrinsics measure 6x slower than its own x87 scalar
   (21.7 ms vs 3.40 ms); only hand-written #pragma aux asm works
   (0.82 ms). I.e. **each new format means hand-writing another asm
   file**, so formats should be few, and a new format must first run on
   this path, measure its quality, then decide whether an asm is worth
   it.

   Semantics (covering both Q8_0 and Q4_1; weight groups gs are a
   multiple of 32, activation groups always 32):

     Q8_0: o[i] = sum_g  ws[g]*sum_{sub-block s in g} xqs[s]*dot32(w, x)
     Q4_1: o[i] = sum_g ( d[g]*sum_s xqs[s]*dot32(q, x)
                          + m[g]*sum_s xqs[s]*sum(x) )

   **The reference must reproduce the SIMD reduction ORDER bit for
   bit, not merely be "mathematically equivalent".** The contract is
   not the two summation formulas but HOW THE PARENTHESES GO - because
   SSM state is quantized, a 1e-6 association difference flips some
   +/-1 LSB quantization decision and then amplifies per token - eight
   tokens fork generation. So this code is deliberately
   dumb: int32 sub-block sums first go into arrays (on the SIMD side
   MMX/SSE2 does this), then the exact same float reduction structure
   finishes. Three details must not change:

     - `acc*(sx*sw)` not `(acc*sx)*sw` (gs=32 tier)
     - 4-way accumulators paired as `(a0+a2)+(a1+a3)`, tail ALL into a0 (gs=32 tier)
     - Q4_1's dot and zero each get their own accumulator, merged once at the end (not interleaved per group)

   The x256 fold exists on the SIMD side but not here: 256 is a power
   of two, scaling changes no rounding at any step, so the two sides
   remain bit-identical. */

/* Q4_1 matmul kernel.

   `o[i] = sum_g [ d[i][g]*sum_{sub-block s in g} xqs[s]*dot32(q_w, q_x)
                   + m[i][g]*xg[g] ]`

   Two savings:
   1. **Zero term hoisted out of the row loop**. `xg[g] = sum_{s in g}
      xqs[s]*sum_k xq[k]` depends only on activations - computed once per
      matmul, shared by all output rows. So Q4_1's inner loop differs
      from Q4_0's by not a single instruction, while the asymmetric
      tier's quality measures 2.7-4.4x better.
   2. **Nibble expansion needs no sign extension** (unsigned 0..15),
      keeping the x256 fold from `punpcklbw(0,x)`, cancelled once at
      the row end.

   Reduction order deliberately matches the Q8 wide-group path
   (per-sub-block sequential accumulation; SSE2's batched 4-vector
   reduction `((q0+q1)+q2)+q3` is equivalent), so the gcc/SSE2 and MMX
   builds are bit-identical. */

/* ---- Q4_1 row kernels, one per (ISA x impl) --------------------------- */

/* ---- T2 row kernels, one per (ISA x impl) -----------------------------
   The single 2-bit plane arrives in `w4`; `w2v` and both prefetch bounds
   are unused - 8 bytes per 32 elements is a quarter of Q4_1's weight
   traffic, too few cache lines for a prefetch distance to mean anything
   unmeasured.

   No pairing yet. Not a decision - unmeasured. `--pair` does nothing for
   this format today. */

/* row_t2_sse2_intrin's body lives in src/ops_sse2.c, same reason as
   row_q8_sse2_intrin above. Declared in src/ops_sse2.h. */

/* row_t2_mmx_asm / row_t2_sse2_asm live in src/ops_mmx.c /
   src/ops_sse2.c, alongside their leaf kernels (lz_dot32_t2_asm /
   lz_dot32_t2_sse2_asm) which they call direct-by-name, same-TU.
   Declared in src/ops_mmx.h / src/ops_sse2.h. */

/* SSE1 slots NULL: T2's unpack is `pand 0x03` + three `psrlw`, and SSE1
   adds no integer instruction that replaces any of them. */

/* ---- Q6_1 row kernels, one per (ISA x impl) ---------------------------
   The only format that uses BOTH plane pointers and BOTH prefetch
   bounds. The 2-bit plane needs its own: `pf_end` stops exactly at the
   plane boundary. At 22% of all weight bytes on the recipe of record,
   its miss frequency is half the 4-bit plane's, not 1/16 like a scale
   array: it IS a hot stream, just a narrower one. Both bounds are real,
   not belt-and-braces: LZ_PF_LOAD is a REAL dummy load on PII and
   faults past the end of the tensor. */

/* row_q61_sse2_intrin's body lives in src/ops_sse2.c, same reason as
   row_q8_sse2_intrin above. Declared in src/ops_sse2.h. */

/* T2 matmul. Same shape as matmul_q61_impl minus the second plane, with
   one addition: the zero coefficient is -scale, so the negated array is
   built per row into g_t2neg and handed to epi_q41 as its `wm`. */

/* lz_fwht, the float transform, is in src/fwht.c beside its int32 twin.
   It was here; one operator in two files meant changing the butterfly
   twice and made every per-file census see one operator as two. */

