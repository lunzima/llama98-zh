/* Helpers used from BOTH src/ops.c and one or more of the suffixed
   MMX/SSE1/SSE2 translation units (src/ops_mmx.c, src/ops_mmx_sse.c,
   src/ops_sse2.c). Shared helpers live here once, not duplicated under
   per-TU names (lz_i32f_wsum, gdn_tail_row_mmx) or as bare
   textually-identical copies (LZ_WSUM_CHUNK, LZ_SLOT_NEXT,
   p2_shift_of): lz_i32f carries a project-wide invariant ("every
   int32->float conversion goes through this, never a bare cast") that
   a rename makes ungreppable, and LZ_WSUM_CHUNK/LZ_SLOT_NEXT drive a
   ring-buffer walk that must stay in lock-step across whichever TU
   boundary a caller and its row/group function sit on - this codebase
   has that exact failure mode on record already (LZ_GDN_LO_SCALE, 254
   vs 256, silently wrong because nothing compared the two copies).

   ONE definition each, here, included by every TU that needs it -
   `static` for the functions (each TU gets its own internal-linkage
   copy of the CODE, but there is only one COPY OF THE TEXT to keep in
   step, which is the property that matters) and #define for the
   macros. Every name keeps the name it had at its original (ops.c)
   site - no _mmx/_wsum/_sse2 suffix - so a grep for the name that
   matters finds every use.

   Needs its own include guard, unlike ops_kernel_amax.h/norm.h/
   q8round.h/p2.h/dot.h: those are included exactly once (by ops.c) and
   rely on that; this one is included by up to four different files. */
#ifndef LZ_OPS_KERNEL_SHARED_H
#define LZ_OPS_KERNEL_SHARED_H

#include "lz_int.h"   /* <stdint.h> is not on the language floor */

#include "ops_arm_prim.h"   /* LZ_ARM_PRIM_ASM, lz_i32f_arm_asm */

/* The runtime-selected ISA tier, defined in ops_sched.c. Declared here
   rather than beside LZ_DEFINE_PICK below because lz_i32f reads it too.
   Every TU that includes this header has already included ops.h, which
   is where LZ_KERNEL_* come from. */
extern int g_kernel;

/* int32 -> float. **Every int32-to-float conversion in the engine goes
   through this**, not a bare `(float)v` - unified by construction rather
   than policed case by case.

   `(float)v` is exact for |v| < 2^24 on both sides; BEYOND 2^24 there
   is rounding, and the rounding timing differs between compilers:
   x87's `fild` loads exactly into an 80-bit register and can `fmul`
   directly (rounding the product once), while SSE's `cvtsi2ss` must
   first round the integer to 24 bits, then multiply (rounding twice).
   PC=24 cannot govern this - precision control applies to arithmetic
   results, not to loads.

   Q8_0's sub-block accumulator bound is 127*127*32 = 5.2e5 and Q4_1's
   6.1e4, both within 2^24, so this road ran a long time untouched.
   Q16_0 is the first format that can exceed it (32767*127*32 =
   1.33e8); measured: some sub-block sum of 19834503 in out_proj line
   160 flipped 1 ULP, amplified by the reduction to 4 ULP, max
   end-to-end logits difference 1.4e-5.

   Split into two parts: |v>>12| < 2^19 and |v & 0xFFF| < 2^12, both
   exact as float; multiplying by 4096 (a power of two) is exact; only
   the final addition rounds - and addition is arithmetic, which PC=24
   governs. In two's complement (v>>12)*4096 + (v & 0xFFF) == v always
   holds (`>>` is arithmetic shift).

   ARM's hand-written tier (--kernel arm-asm) computes the same
   correctly-rounded conversion with CLZ and a shift instead of the four
   __aeabi_* calls the split costs on a machine with no FPU. Selected
   per call rather than per matmul because this has no dispatch point to
   hoist to - three instructions against the 112.8 the C tier measures
   (docs/arm-asm-audit.md). Compiled out entirely off ARM, so no x86
   build gains a branch.

   AND THE SPLIT IS ONLY NEEDED WHERE EXCESS PRECISION EXISTS. Everything
   above is about ONE hazard: a compiler that may keep the intermediate
   wider than float32, which is x87 and nothing else. Where
   __FLT_EVAL_METHOD__ is 0 the standard says float expressions evaluate
   to float's own range and precision, so `(float)v` IS the correctly
   rounded float32 the split constructs by hand - and it is the same
   number, not merely a close one, because the split's own derivation
   shows it computes exactly that.

   Verified rather than argued: 87,108,878 values compared on
   gcc/x86-64, the whole +-2^25 band (every rounding decision lives
   there, since |v| < 2^24 is exact either way) plus 20 M random plus
   INT32_MIN/MAX and the neighbours of 2^24 - 0 mismatches. On ARM the
   check already exists and runs: .prof/arm_leafgate.c compares this C
   tier against lz_i32f_arm_asm, so a `(float)v` that disagreed with a
   correctly rounded conversion would fail it.

   Costs: 9 SSE2 instructions become 2, and on the ARM C tier four
   __aeabi_* calls (116.8 instructions) become one __aeabi_i2f. Watcom
   and any 32-bit x87 gcc build do not define __FLT_EVAL_METHOD__ as 0
   and keep the split, which is the whole reason it was written. */
/* v * 2^e for a float v and an integer e, WITHOUT the multiply.
 *
 * Scaling by a power of two only moves the exponent field, so on the
 * normal range this is an integer add on the bit pattern - about six
 * instructions - where `v * pow2f(e)` is a __aeabi_fmul at 30 on the
 * soft-float target. pow2f itself is already bit construction and costs
 * nothing to speak of; the multiply is the whole expense.
 *
 * WHY IT IS THE SAME NUMBER AND NOT A CLOSE ONE: for a normal v and an
 * in-range result, v * 2^e is exactly representable with the same
 * significand, so the IEEE-754 product IS v with e added to its
 * exponent. Neither form rounds, which is why this owes its callers
 * bit-identity rather than a tolerance.
 *
 * THE CASES IT DECLINES, returning 0 so the caller keeps the multiply
 * it already has written:
 *   - v is zero or subnormal: the exponent field is 0 and adding to it
 *     would build a different number entirely.
 *   - the result would reach 255: overflow to inf. The multiply gets
 *     that right; an exponent add would walk into the sign bit.
 *   - the result would reach 0: gradual underflow to a subnormal, which
 *     an exponent add cannot express.
 * inf and NaN are excluded by the same first test - their exponent
 * field is 255 - rather than by a separate case.
 *
 * Callers: both epilogue exits (ops.c, once per output element) and
 * p2_group_norm (ops_gdn.c, twice per recurrence group, where the
 * multiply it replaces already carried the comment "exact: multiply by
 * a power of two"). */
LZ_MAYBE_UNUSED static int f32_scale_pow2(float v, int e, float *out) {
    union { float f; uint32_t u; } b;
    uint32_t bex;
    int ex;
    b.f = v;
#if defined(LZ_ARM_PRIM_ASM)
    /* lz_scale2_arm_asm is this function in ARM assembly, with the same
       four declines, and it was already in the tree serving the conv
       kernel before this one existed. Dispatched the way lz_i32f
       dispatches below, so the arm-asm tier does not run the C form of
       an operator it has a cell for - which is what the kernel matrix
       is for. Bit-identical by contract, not by tolerance: both forms
       decline exactly where the exponent add is not the whole story. */
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        uint32_t r;
        if (!lz_scale2_arm_asm(b.u, (int32_t)e, &r)) return 0;
        b.u = r;
        *out = b.f;
        return 1;
    }
#endif /* LZ_ARM_PRIM_ASM */
    bex = (b.u >> 23) & 0xFFu;
    ex  = (int)bex + e;
    /* TWO compares, not four. `bex == 0 || bex == 0xFF` is exactly
       `(uint32_t)(bex - 1) >= 0xFE`, and `ex <= 0 || ex >= 255` is
       exactly `(uint32_t)(ex - 1) >= 254` - both by unsigned wraparound
       putting the low out-of-range case above the high one. Written
       this way because the guard is the whole cost of this helper: it
       replaces a 30-instruction __aeabi_fmul, and the four-compare form
       measured 18 of those 30 back (ARM, .prof/arm_l2f_count.sh mode 5
       against mode 3, 134 -> 122). On a target with no branch
       predictor worth the name, halving the branch count in a
       four-branch guard is most of what is left to win here. */
    if ((uint32_t)(bex - 1u) >= 0xFEu) return 0;     /* zero/subnormal/inf/NaN */
    if ((uint32_t)(ex - 1) >= 254u) return 0;        /* under/overflow */
    b.u = (b.u & 0x807FFFFFu) | ((uint32_t)ex << 23);
    *out = b.f;
    return 1;
}

LZ_MAYBE_UNUSED static float lz_i32f(int32_t v) {
#if defined(LZ_ARM_PRIM_ASM)
    if (g_kernel == LZ_KERNEL_ARM_ASM) return lz_i32f_arm_asm(v);
#endif /* LZ_ARM_PRIM_ASM */
#if defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ == 0
    return (float)v;
#else
    return (float)(v >> 12) * 4096.0f + (float)(v & 0xFFF);
#endif /* __FLT_EVAL_METHOD__ == 0 */
}

/* The chunk fold: accf[d] += (float)acc32[d] for 32 elements, at
 * whichever tier the machine has.
 *
 * HERE, like LZ_WSUM_CHUNK and LZ_SLOT_NEXT below, and for the same
 * reason: the two arms of the LZ_WSUM_MMX_EXTERN split - src/ops_mmx.c's
 * group kernel and src/ops_gdn.c's inline walk - both run this fold, and
 * a tier decision that could differ between them is a divergence nothing
 * would report. One text, both sides.
 *
 * A MACRO rather than a static function because the SIMD cells are
 * declared in src/ops_sse.h and src/ops_sse2.h, which this header does
 * not include and must not: it is included by four TUs, two of which
 * have no business seeing the x86 declarations at all. Expanding at the
 * call site lets each consumer's own includes decide which cells exist.
 * `dv` is the caller's loop variable, passed in for the same reason -
 * C89 wants declarations at the top of a block and the fallback loop
 * needs one.
 *
 * The kernels are bit-identical to the fallback, not close: lz_i32f's
 * own comment above records the split form agreeing with the correctly
 * rounded conversion on 87,108,878 values, and cvtdq2ps/cvtsi2ss are
 * that conversion. */
#if defined(LZ_HAVE_I32FACC_SSE2) && defined(LZ_HAVE_I32FACC_SSE)
#define LZ_I32F_ACC32(accf, acc32, dv) do { \
        int lz_tr_ = lz_i32facc_tier(); \
        if (lz_tr_ >= 2)      lz_i32f_acc32_simd((accf), (acc32)); \
        else if (lz_tr_ >= 1) lz_i32f_acc32_sse((accf), (acc32)); \
        else for ((dv) = 0; (dv) < 32; (dv)++) \
                 (accf)[dv] += lz_i32f((acc32)[dv]); \
    } while (0)
#elif defined(LZ_HAVE_I32FACC_SSE2)
#define LZ_I32F_ACC32(accf, acc32, dv) do { \
        if (lz_i32facc_tier() >= 2) lz_i32f_acc32_simd((accf), (acc32)); \
        else for ((dv) = 0; (dv) < 32; (dv)++) \
                 (accf)[dv] += lz_i32f((acc32)[dv]); \
    } while (0)
#elif defined(LZ_HAVE_I32FACC_SSE)
#define LZ_I32F_ACC32(accf, acc32, dv) do { \
        if (lz_i32facc_tier() >= 1) lz_i32f_acc32_sse((accf), (acc32)); \
        else for ((dv) = 0; (dv) < 32; (dv)++) \
                 (accf)[dv] += lz_i32f((acc32)[dv]); \
    } while (0)
#else
#define LZ_I32F_ACC32(accf, acc32, dv) do { \
        for ((dv) = 0; (dv) < 32; (dv)++) \
            (accf)[dv] += lz_i32f((acc32)[dv]); \
    } while (0)
#endif /* LZ_HAVE_I32FACC_* */

/* Attention wsum's chunking constant. lz_attn_wsum_q8's int32
   accumulator only holds 516 rows before the worst-case product sum
   (4,161,409 per row) risks overflow, so it chunks at 512 (2.13e9,
   ~0.8% under INT32_MAX) and folds each chunk through lz_i32f() before
   joining the running float total - both ops.c's own loop and
   src/ops_mmx.c's lz_wsum_group_mmx walk chunks of exactly this size,
   and must, since T's chunk boundaries are otherwise implicit in the
   ring-buffer slot each side computes independently. */
#define LZ_WSUM_CHUNK 512

/* forward.c's kv_slot(), walked instead of computed: t only ever
   increases in these kernels, so the slot advances by one and wraps at
   sink+ring rather than costing a division per row. A division is 39
   cycles on the x87 machines this tier exists for, against one compare.
   The closed form in forward.c stays as the gate's reference. */
#define LZ_SLOT_NEXT(slot, sink, ring)                          \
    ((slot) = (ring) && (slot) + 1 >= (sink) + (ring)           \
              ? (sink) : (slot) + 1)

/* GDN/KDA pass 1's odd-kd tail row: the eight 4-lane sub-windows the
   MMX/asm kernels cover leave one row uncovered when kd is odd, and
   this closes it. Pure integer, touches neither x87 nor MMX, so it is
   safe to call from inside a row's SIMD phase without disturbing
   either register file. */
LZ_MAYBE_UNUSED static void gdn_tail_row(const int8_t *sq_gg, int kd, int vd,
                         int32_t cu, int32_t cw,
                         int32_t *au_gg, int32_t *aw_gg) {
    if (!(kd & 1)) return;
    {
        const int8_t *row = sq_gg + (size_t)(kd - 1) * vd;
        int j;
        for (j = 0; j < 32; j++) {
            au_gg[j] += (int32_t)row[j] * cu;
            aw_gg[j] += (int32_t)row[j] * cw;
        }
    }
}

/* SHIFT, NOT DIVIDE. A divide would land amax exactly on 32512
   (hn = A*32512/amax) at one integer divide per element; MMX has no
   divide, so that alone keeps an MMX twin out of reach.

   Shifting instead maps amax somewhere into [16256, 32512] rather than
   onto 32512, which gives up to one bit of code range. PREDICTED to cost
   accuracy; MEASURED to improve it (4.27e-05 -> 2.98e-05)
   because the divide's own rounding was worth more than the lost bit.

   32512 rather than 32767 is what makes |hn| <= 32513 and therefore
   |hi| <= 127 after the split - the two saturating packs in the MMX twin
   depend on that bound and must never fire. */
LZ_MAYBE_UNUSED static int p2_shift_of(int32_t amax) {
    int sh = 0;
    while ((amax >> sh) > 32512) sh++;
    return sh;
}

/* ---- matmul row-kernel prefetch ---------------------------------------
   Every format's Watcom row function (row_q8_mmx_asm etc., in
   src/ops_mmx.c/ops_sse2.c) issues one of these per 32 (or 128)
   elements, so - like LZ_WSUM_CHUNK/LZ_SLOT_NEXT above - this is
   infrastructure the row functions need wherever they end up, not
   something specific to any one format. Not one of the five names
   shared via this header, but the identical principle: one text,
   included by every TU that needs it, rather than a copy that can
   drift.

   P6 has only 4 fill buffers and a 40-entry ROB. The matmul inner loop
   executes ~50 instructions per 32 weight bytes; to keep 4 misses in
   flight you must look ~128 bytes ahead = ~200 instructions - far
   beyond the ROB. So a PII most likely serializes per 32 bytes at one
   full memory latency: 32 B / ~180 ns ~ 178 MB/s, not the 300-400 MB/s
   the bus could deliver.

   Three tiers:
     PII        no PREFETCH instruction (that is SSE); only a dummy
                load - occupies one register and one ROB slot, but
                feeds the fill buffers.
     PIII/PM    `prefetchnta`. The nta variant fits this workload:
                weights are used once per token and must never enter L2
                (Coppermine's L2 is only 256KB and would be flushed by
                weights, evicting SSM state and activations along the
                way).
     ref        none.

   **Prefetch changes no numbers** - the gate is "all three tiers'
   logits are byte-identical to ref". If they differ, the prefetch is a
   real access or out of bounds.

   The distance is machine-dependent, kept as a compile-time constant
   to sweep 0/2/4/8 on target hardware. */
#ifndef LZ_PF_DIST
#define LZ_PF_DIST 4                    /* look ahead this many 32-byte cache lines */
#endif

/* The four prefetch-tier values lz_prefetch_mode() (src/ops.c, real
   extern linkage - unaffected by this move) returns. Kernel tier picks
   instruction-set width; this picks the memory hint - orthogonal, per
   ops.c's own note beside g_pf/pf_detect(): K6-2/K7 have MMX only but
   DO have 3DNow! prefetch, which beats the PII's dummy load. */
#define LZ_PF_NONE 0
#define LZ_PF_LOAD 1     /* PII: no prefetch instruction, dummy load only */
#define LZ_PF_NTA  2     /* PIII onward: prefetchnta (weights used once per token) */
#define LZ_PF_AMD  3     /* K6-2 / K7: 3DNow! prefetch */

#if defined(__WATCOMC__)
/* Watcom side has two: dummy load (PII) and prefetchnta (PIII+).
   __modify declared truthfully. Each including TU gets its
   own inlined instance (no address of these is ever taken, so there is
   no linkable symbol to collide across TUs - the same #pragma aux
   semantics every other kernel in this project relies on). */
extern void lz_pf_load(const void *p);
#pragma aux lz_pf_load =            \
    "mov eax, [eax]"                \
    __parm [__eax] __modify [__eax]

extern void lz_pf_nta(const void *p);
#pragma aux lz_pf_nta =             \
    "db 0x0F, 0x18, 0x00"           \
    __parm [__eax] __modify []
extern void lz_pf_amd(const void *p);
#pragma aux lz_pf_amd =                 ".686"                              "db 0x0F, 0x0D, 0x00"               __parm [__eax] __modify []
#define LZ_PFI_LOAD(p) lz_pf_load(p)
#define LZ_PFI_NTA(p)  lz_pf_nta(p)
#define LZ_PFI_AMD(p)  lz_pf_amd(p)
#elif defined(__GNUC__)
#define LZ_PFI_LOAD(p) __builtin_prefetch((p), 0, 0)
#define LZ_PFI_NTA(p)  __builtin_prefetch((p), 0, 0)
#define LZ_PFI_AMD(p)  __builtin_prefetch((p), 0, 0)
#else
#define LZ_PFI_LOAD(p) ((void)0)
#define LZ_PFI_NTA(p)  ((void)0)
#define LZ_PFI_AMD(p)  ((void)0)
#endif

/* Compile-time A/B control for the 128-element SSE2 group kernels that
   sit alongside the 32-element MMX ones - every format uses the same
   knob.

   `-DLZ_SSE2_GROUP=0` makes an SSE2 CPU take the MMX group path
   instead, i.e. the MMX group kernels alone. Two binaries from one
   source tree, so the "did this change any number" question is
   answered by `cmp` on --dump-logits rather than by swapping source
   trees and hoping the rest matched. Same reason LZ_BATCH_MAX exists.

   IT STAYS BECAUSE THERE IS NO RUNTIME TIER FOR IT. The FORCE_* escapes
   once cited here are gone - each was deleted when `--kernel sse`/`mmx`
   came to select the same body at run time. Nothing selects between the
   32- and 128-element group kernels that way, so this one is not the
   same case.

   Only the Watcom build has these kernels at all - the gcc translation
   unit does not contain them (they live inside `#if defined(__WATCOMC__)`),
   so the gcc side is unaffected by this switch by construction. */
#ifndef LZ_SSE2_GROUP
#define LZ_SSE2_GROUP 1
#endif

/* gcc's 128-element MMX group kernels. Watcom ships its group kernels
   always-on (its LZ_G128_* are 1). gcc's were "not written, not
   nothing-to-offer": the mask construction the Watcom assembly hoists
   across its four sub-blocks (0x0F/0x03 and a zero register left live
   in %mm) is exactly what a compiler does with a loop-invariant, and
   whether a hand-written gcc intrinsics version would emit a different
   shape than four separate 32-element calls was never measured. The
   kernels exist (dot128_*_mmx, src/ops_kernel_dot_mmx.h), bit-identical
   by construction: each sub-block's int32 dot folds and stores
   separately, so the four accumulators are the same four ints the
   per-32 path writes.

   Whether the group path WINS on any given machine is a separate,
   unmeasured question (the dev machine is not the target),
   so it is a knob. Default 1 matches Watcom's shipped shape; build with
   -DLZ_G128_GCC=0 to run the per-32 path on the dev machine. */
#ifndef LZ_G128_GCC
#define LZ_G128_GCC 1
#endif

/* No prefetch at all. Not a platform fallback - a deliberately
   reachable mode, because the LATENCY TIER is a thing we need to
   measure and could not.

   The performance model has two tiers per machine: latency (no
   prefetch, assumed 178 MB/s on PIII) and bandwidth (prefetchnta,
   assumed 440). Every performance conclusion that matters - J's payoff,
   whether MTP has headroom to trade into - is a ratio between those two
   numbers. The dispatch otherwise has no way to reach "no prefetch":
   pf_detect() returns LOAD/NTA/AMD and the selection's else-branch
   falls to NTA, so a run on real hardware would only ever measure the
   bandwidth tier. This mode is the instrument for the latency half of
   every one of those questions. */
#define LZ_PFI_NONE(p) ((void)(p))

/* ---- dispatch infrastructure (per-TU) ----------------------------------

   Slot order for EVERY dispatch table in this file and others.
   g_kernel (declared at the top of this header) is the runtime-selected
   ISA tier. LZ_DEFINE_PICK generates a static picker for each
   function-pointer type; it reads g_kernel to decide MMX vs SSE2.

   Here so ops_kernel_amax.h, ops_kernel_norm.h and ops_kernel_q8round.h
   compile in every TU that includes this header. */
#define LZ_ROW_MMX_I   0
#define LZ_ROW_SSE_I   1
#define LZ_ROW_SSE2_I  2
#define LZ_ROW_MMX_A   3
#define LZ_ROW_SSE_A   4
#define LZ_ROW_SSE2_A  5
#define LZ_ROW_N       6

#define LZ_DEFINE_PICK(NAME, TYPE)                                   \
    static TYPE NAME(const TYPE *tab) {                                   \
        int want_sse2 = (g_kernel == LZ_KERNEL_SSE2);                     \
        TYPE f;                                                           \
        if (!g_kernel) return (TYPE)0;                                    \
        if (g_kernel == LZ_KERNEL_REF) return (TYPE)0;                    \
        f = tab[want_sse2 ? LZ_ROW_SSE2_A : LZ_ROW_MMX_A];              \
        if (f) return f;                                                  \
        f = tab[want_sse2 ? LZ_ROW_SSE2_I : LZ_ROW_MMX_I];              \
        if (f) return f;                                                  \
        return tab[LZ_ROW_MMX_I];                                         \
    }

#endif /* LZ_OPS_KERNEL_SHARED_H */
