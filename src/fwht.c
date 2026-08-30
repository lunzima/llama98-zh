#include "fwht.h"
#include "ops_arm.h"
/* Unconditional, for the same reason the three below are: both
   transforms in this file dispatch on g_kernel, so a guard that hides
   these on x86 hides the dispatch with them. */
#include "ops.h"                  /* LZ_KERNEL_*, lz_fwht's prototype */
#include "ops_kernel_shared.h"    /* g_kernel */
#include "ops_sse.h"              /* LZ_FWHT_F32_SSE_EXTERN, lz_fwht_stage_f32_sse */

/* OUTSIDE the ARM guard above, deliberately: these three are what the
   x86 MMX stage kernel needs, and putting them inside it is exactly the
   mistake this file made first - on x86 LZ_ARM_ASM_EXTERN is not
   defined, so the includes vanished, LZ_FWHT_MMX_EXTERN was never seen,
   and the MMX branch below compiled out silently. A mutation of the
   kernel then changed nothing, which is how it was found. */
#include "mmx_compat.h"           /* _mm_empty, on both toolchains */
#include "ops_mmx.h"              /* LZ_FWHT_MMX_EXTERN, lz_fwht_stage_mmx */
#include "ops_sse2.h"             /* LZ_FWHT_SSE2_EXTERN, lz_fwht_stage_sse2 */
#include "ops_quant.h"            /* lz_cpu_has_mmx */

/* ONE OPERATOR, TWO FILES, AND THIS ONE IS NAMED FOR IT. lz_fwht, the
   float transform, lives in ops.c with its own SSE1 tier; the int32
   twin below lives here with its MMX, SSE2 and ARM tiers. The split is
   by dtype, not by operator, and nothing about the transform asks for
   it - the two even cross-reference each other's guard shape. Anyone
   changing the butterfly has to change it in two files, and the census
   that ranks operator files sees one operator as two.
   Not moved here yet: lz_fwht is called from forward.c in two places
   and moving it across a translation unit needs a byte-identity run on
   both toolchains, which is a commit of its own rather than a hunk in
   someone else's.

   Sylvester recursion, in place, mirroring lz_fwht (float) in ops.c.
   Every stage is a butterfly of one add and one sub - no multiply,
   sqrt or divide in the transform body, so it is exact in int32 up to
   the additions themselves. The caller folds the 1/sqrt(n) energy
   factor into the downstream activation-quantization scale, keeping
   this a pure integer kernel for the ARMv5TE target (no FPU, no SIMD). */
void lz_fwht_i32(int32_t *restrict y, const int32_t *restrict x, int n) {
    int len, i, j;
    int32_t *tmp = y;          /* caller provides output; butterflies in place */
#if defined(LZ_ARM_ASM_EXTERN)
    /* Selected, not automatic: `--kernel arm-c` and `ref` must reach the
       C body below, or the tier this assembly is COMPARED AGAINST would
       be the assembly itself and build/arm/parity_gate.sh would be
       comparing a thing to itself. Same shape as lz_wsum_pair_arm and
       norm_ss_fixed's guard.

       n >= 8 is the assembly's precondition, not a preference: every
       loop there moves eight words and has no epilogue. Smaller n means
       blk 2 or 4, which no shipping config produces, so the C body stays
       the whole answer for those rather than growing a tail. */
    if (g_kernel == LZ_KERNEL_ARM_ASM && n >= 8) {
        lz_fwht_i32_arm_asm(y, x, n);
        return;
    }
#endif /* LZ_ARM_ASM_EXTERN */
    for (i = 0; i < n; i++) tmp[i] = x[i];
#if defined(LZ_FWHT_MMX_EXTERN)
    /* MMX from the second stage on. The butterfly at stride len pairs
       tmp[i+j] with tmp[i+j+len], so for len >= 2 the two halves are
       CONTIGUOUS pairs of int32 - one movq each, paddd, psubd, two movq
       back, no shuffle anywhere. At len == 1 the partners are adjacent
       words and an unpack per butterfly would cost more than it saves,
       so that one stage stays in the C below.

       n >= 4 because len reaches 2 only then, and (n & 3) == 0 because
       the stage kernel moves two int32 per iteration over a half-block
       of `len` words; every shipping block (hadamard_o/hadamard_down,
       powers of two from 64 up) satisfies both.

       EXACT BY CONSTRUCTION, not by tolerance: int32 add and subtract,
       wrapping, no rounding and no reassociation - each output word is
       the same two inputs under the same operator as the C. That is why
       this cell needs no bit-identity argument of the kind every float
       kernel in this engine owes.

       ONE emms FOR THE WHOLE TRANSFORM, here rather than per stage: the
       stage loop contains no x87, and the emms can be hoisted
       exactly that far. What follows lz_fwht_i32 does (subn_fwht_quant's
       amax and q8_round), which is why it cannot be hoisted further.

       SELECTED BY TIER, not by CPUID, and the first version of this got
       it wrong. `lz_cpu_has_mmx()` alone would give --kernel ref the MMX
       body too, and then the tier the kernel is COMPARED AGAINST would
       be the kernel - the same trap the ARM guard above spells out. The
       CPUID test stays as well: a tier can be asked for on a machine
       that cannot run it (lz_kernel_select clamps, but the emms below
       would still be reached on a 486 without it).

       LZ_KERNEL_SSE is listed because this is an MMX body and every SSE1
       machine has MMX. A tier that dropped to the scalar loop for
       `--kernel sse` would be reporting a narrower ISA than it has. */
    if ((g_kernel == LZ_KERNEL_MMX || g_kernel == LZ_KERNEL_SSE ||
         g_kernel == LZ_KERNEL_SSE2) &&
        lz_cpu_has_mmx() && n >= 4 && (n & 3) == 0) {
        int start = 2;
        for (i = 0; i < n; i += 2) {
            int32_t u = tmp[i], v = tmp[i + 1];
            tmp[i]     = u + v;
            tmp[i + 1] = u - v;
        }
#if defined(LZ_FWHT_SSE2_EXTERN)
        /* SSE2 takes the stages it can and hands the rest back, rather
           than replacing the MMX path: four int32 per xmm needs len >= 4
           for the halves to be whole vectors, so len == 2 stays on MMX
           and everything above it widens. Same reason LZ_DEFINE_PICK
           falls back from the SSE2 row to the MMX one instead of to
           nothing. */
        if (g_kernel == LZ_KERNEL_SSE2 && n >= 8 && (n & 7) == 0) {
            lz_fwht_stage_mmx(tmp, n, 2);
            for (len = 4; len < n; len <<= 1)
                lz_fwht_stage_sse2(tmp, n, len);
            _mm_empty();
            return;
        }
#endif /* LZ_FWHT_SSE2_EXTERN */
        for (len = start; len < n; len <<= 1) lz_fwht_stage_mmx(tmp, n, len);
        _mm_empty();
        return;
    }
#endif /* LZ_FWHT_MMX_EXTERN */
    for (len = 1; len < n; len <<= 1) {
        for (i = 0; i < n; i += (len << 1)) {
            for (j = 0; j < len; j++) {
                int32_t u = tmp[i + j];
                int32_t v = tmp[i + j + len];
                tmp[i + j]       = u + v;
                tmp[i + j + len] = u - v;
            }
        }
    }
}

/* The float transform, moved here from ops.c so one operator is one
   file. Same Sylvester recursion as the int32 twin above, same
   butterfly, and both dispatch on g_kernel - the split was by dtype and
   nothing about the transform asked for it. Its ISA floor is unchanged
   by the move: ops.c and this file are both compiled at the target's
   ordinary floor, neither takes HIGH_CPU_FLAGS, which was measured
   rather than read off the Makefile's comment. */
void lz_fwht(float *v, int n) {
    /* Sylvester recursion, in place. Every stage is a butterfly of one
       add and one sub, so the whole transform is exact in binary float
       up to the additions themselves - no constants enter, which is why
       this is bit-identical between wcc386 and gcc without any of the
       care the multiply-by-reciprocal paths demand. */
    int len, i, j;
#if defined(LZ_FWHT_F32_SSE_EXTERN)
    /* SSE1 from the third stage on. The butterfly at stride len pairs
       v[i+j] with v[i+j+len], so for len >= 4 both halves are whole
       16-byte vectors - one load each, addps, subps, two stores, no
       shuffle. len 1 and 2 stay in the loop below, where an unpack per
       butterfly would cost more than it saves.

       Selected by tier, not by CPUID: --kernel ref must reach the C or
       the arm this is compared against would be this. Same guard shape
       the int32 twin above carries, and the same reason. */
    if ((g_kernel == LZ_KERNEL_SSE2 || g_kernel == LZ_KERNEL_SSE) &&
        n >= 8 && (n & 7) == 0) {
        for (len = 1; len < 4 && len < n; len <<= 1) {
            for (i = 0; i < n; i += (len << 1)) {
                for (j = 0; j < len; j++) {
                    float a = v[i + j];
                    float b = v[i + j + len];
                    v[i + j]       = a + b;
                    v[i + j + len] = a - b;
                }
            }
        }
        for (len = 4; len < n; len <<= 1) lz_fwht_stage_f32_sse(v, n, len);
        return;
    }
#endif /* LZ_FWHT_F32_SSE_EXTERN */
#if defined(LZ_ARM_PRIM_ASM)
    /* ARMv5TE has no FPU, so each butterfly below is `bl __addsf3` plus
       `bl __subsf3` - two full soft-float adders over one pair of
       operands. lz_faddsub_arm_asm does the unpack and the alignment
       once and declines the cases it cannot prove identical, which the
       C tail then answers. 95.0 instructions a butterfly against 112.7
       (.prof/arm_chain_count.sh modes 23 and 22).

       Selected by tier, not by CPU: ref and arm-c must reach the loop
       below or the arm this is compared against would be this one. */
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        for (len = 1; len < n; len <<= 1) {
            for (i = 0; i < n; i += (len << 1)) {
                for (j = 0; j < len; j++) {
                    union { float f; uint32_t u; } ba, bb;
                    uint32_t s, d;
                    ba.f = v[i + j];
                    bb.f = v[i + j + len];
                    if (lz_faddsub_arm_asm(ba.u, bb.u, &s, &d)) {
                        ba.u = s;
                        bb.u = d;
                    } else {
                        float a = ba.f, b = bb.f;
                        ba.f = a + b;
                        bb.f = a - b;
                    }
                    v[i + j]       = ba.f;
                    v[i + j + len] = bb.f;
                }
            }
        }
        return;
    }
#endif /* LZ_ARM_PRIM_ASM */
    for (len = 1; len < n; len <<= 1) {
        for (i = 0; i < n; i += (len << 1)) {
            for (j = 0; j < len; j++) {
                float a = v[i + j];
                float b = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
}
