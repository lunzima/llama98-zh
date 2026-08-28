/* ops_gdn.c - SSM recurrence (GDN/KDA), fixed-point attention, and
   pass-2 write-back. A separate TU from ops.c to keep each file's #if
   guard density low.

   Contains the full recurrence: pass 1 (coefficient table +
   gdn_sum_gg), pass 2 (p2_group_* + gdn_p2_row_simd), the attention
   scoring/weighted-sum kernels, and the LZGdnTable machinery.
   Depends on ops_quant (q8_round/q8_amax), ops_sched (tier
   selectors), ops_kernel_shared (lz_i32f, LZ_WSUM_CHUNK,
   LZ_SLOT_NEXT, gdn_tail_row, p2_shift_of). */
#include <stddef.h>
#include <float.h>
#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include <stdlib.h>
#include <string.h>

/* LZ_GDN_FIXED, LZ_ATTN_MAX_HD: ops.h, included below. */

#include "ops.h"
#include "err.h"
#include "mmx_compat.h"
#include "ops_mmx.h"
#include "ops_sse.h"     /* lz_i32f_acc32_sse - LZ_I32F_ACC32's SSE1 arm.
                            Between the two, like the ISA ladder itself. */
#include "ops_sse2.h"
#include "ops_quant.h"
#include "ops_sched.h"
#include "ops_kernel_shared.h"
#include "ops_matmul.h"
#include "ops_arm.h"     /* LZ_ARM_ASM_EXTERN, lz_dot32_q8_arm_asm,
                            lz_wsum_pair_arm_asm */
#include "ops_gdn.h"


/* ---- fixed-point attention -------------------------------------------

   Two independent pieces, both replacing f32 x int8 inner loops in
   forward_attn():

   1. Scoring. q is quantized to Q8 once per head and widened to int16
      once, so q.k becomes int8(k) x int16(q) per 32-group - exactly the
      shape the production matmul already uses. No new kernel: it calls
      the same dot32_x16 matmul_q8_impl does, already validated. The x256
      the int8->int16 unpack introduces is cancelled once at the end,
      same as there.

   2. Weighted sum. out[d] = sum_t c_t[g] * vq_t[d], with
      c_t[g] = att[t] * vs[t][g], is isomorphic to the SSM recurrence's
      pass 1 (t plays kk, d plays vv), so it DOES need a new kernel.

   The weighted sum is the only place in the engine whose accumulator
   bound depends on context length: one term is at most 32767*127 =
   4,161,409, so T rows fit int32 only while T <= 516. Accumulation is
   chunked at 512 rows (512 * 4,161,409 = 2.13e9, ~0.8% under INT32_MAX)
   and each chunk goes through lz_i32f() before joining the running float
   total. The chunking is load-bearing, not decorative - test_overflow()
   drives a worst-case input through an unchunked variant and asserts
   the result IS corrupted.

   The default lives in ops.h (LZ_ATTN_FIXED), not here - a second
   #ifndef with a different default would be a lie. The error-magnitude
   argument that would normally settle this is NOT usable here: the
   logit-difference metric saturates, so "this error is 200x that one"
   says nothing about which output is better (section 6.1). */
#if LZ_ATTN_FIXED
/* LZ_ATTN_MAX_HD comes from ops.h, which this file already includes.
   Do NOT redefine it here to the same 256 - legal C, because an
   identical redefinition is allowed, and therefore silent until someone
   changed one copy. The comment three lines below has said "a constant
   duplicated across that TU boundary is a failure mode" the whole time,
   about a different constant. */
/* LZ_WSUM_CHUNK, LZ_SLOT_NEXT: defined in src/ops_kernel_shared.h.
   src/ops_mmx.c's lz_wsum_group_mmx walks the exact same ring buffer
   in the exact same chunks, and a constant duplicated across that TU
   boundary is a failure mode. Defined outside both `& 1`/`& 2` bit
   guards because BOTH kernels need LZ_SLOT_NEXT - under `& 1` only,
   the -DLZ_ATTN_FIXED=2 build would not compile. */

#if (LZ_ATTN_FIXED & 2)
#if defined(__WATCOMC__)
/* lz_wsum_pair_asm's #pragma aux body, and the per-pair loop, live in
   src/ops_mmx.c's lz_wsum_group_mmx - see that function and ops.c's
   lz_attn_wsum_q8 (search "one call per GROUP") for why: T is
   context-length-dependent and unbounded, so a per-pair or even
   per-chunk call here would scale the wrong way. No LZ_WSUM_PAIR macro
   in this TU: it would alias a name not declared here, and nothing
   under __WATCOMC__ needs it - lz_attn_wsum_q8's Watcom branch calls
   lz_wsum_group_mmx directly. */

#elif defined(LZ_WSUM_MMX_EXTERN)
/* lz_wsum_pair_mmx's body lives in src/ops_mmx.c now - it writes %mm
   registers. Declared in src/ops_mmx.h, already visible here. */
#if defined(LZ_ATTN_SSE2_EXTERN)
/* The x86 tier split, and the shape is the ARM one twelve lines down,
   for the same reason: this runs once per ROW PAIR, so an indirect call
   would be paid T/2 times per group while this test folds into a branch
   the compiler already emits. Same-tier for a whole run, so the two
   kernels never interleave.

   Tested on g_kernel and on NOTHING ELSE, unlike the SSE1 float cells
   in src/ops.c which also ask lz_cpu_has_sse(): there is no
   lz_cpu_has_sse2() to ask, because lz_kernel_select already clamps
   LZ_KERNEL_SSE2 to whatever kernel_detect found, so this value cannot
   be set on a machine without SSE2. Adding a CPUID test here would be a
   second answer to a question already settled, and testing CPUID
   INSTEAD would hand --kernel mmx the SSE2 body and destroy the control
   arm kernel_isa_identity_gate compares against. */
static void lz_wsum_pair_x86(const int8_t *rowA, const int8_t *rowB,
                             const int16_t *coef, int32_t *acc32) {
    if (g_kernel == LZ_KERNEL_SSE2) {
        lz_wsum_pair_sse2(rowA, rowB, coef, acc32);
        return;
    }
    lz_wsum_pair_mmx(rowA, rowB, coef, acc32);
}
#define LZ_WSUM_PAIR(ra, rb, co, ac) lz_wsum_pair_x86(ra, rb, co, ac)
#else
#define LZ_WSUM_PAIR(ra, rb, co, ac) lz_wsum_pair_mmx(ra, rb, co, ac)
#endif /* LZ_ATTN_SSE2_EXTERN */

#else
/* Non-MMX builds. Integer sums are exact under reassociation, so this is
   bit-identical to both kernels above - the same reason the SSM pass 1
   has a scalar twin. Without it, enabling this tier would make the output
   depend on which compiler built the binary, and the cross-compiler byte-compare
   gate would stop meaning anything. */
static void lz_wsum_pair_ref(const int8_t *rowA, const int8_t *rowB,
                             const int16_t *coef, int32_t *acc32) {
    int32_t ckA = coef[0], ckB = coef[1];
    int d;
    for (d = 0; d < 32; d++)
        acc32[d] += (int32_t)rowA[d] * ckA + (int32_t)rowB[d] * ckB;
}
#if defined(LZ_ARM_ASM_EXTERN)
/* The ARM tier split. Picked here rather than by a function pointer for
   the reason matmul gives at row_q16_arm_asm: this is called T/2 times
   per group and an indirect call would be paid every time, while the
   test below folds into the branch gcc already has. --kernel arm-c runs
   the shared C above, which is also what ref runs - the two are the
   same code, and the assembly is what the comparison is against. */
static void lz_wsum_pair_arm(const int8_t *rowA, const int8_t *rowB,
                             const int16_t *coef, int32_t *acc32) {
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        lz_wsum_pair_arm_asm(rowA, rowB, coef, acc32);
        return;
    }
    lz_wsum_pair_ref(rowA, rowB, coef, acc32);
}
#define LZ_WSUM_PAIR(ra, rb, co, ac) lz_wsum_pair_arm(ra, rb, co, ac)
#else
#define LZ_WSUM_PAIR(ra, rb, co, ac) lz_wsum_pair_ref(ra, rb, co, ac)
#endif /* LZ_ARM_ASM_EXTERN */
#endif /* __WATCOMC__ */
#endif /* LZ_ATTN_FIXED & 2 */

#if (LZ_ATTN_FIXED & 1)
#if !defined(__WATCOMC__) && !defined(LZ_DOT_MMX_EXTERN)
/* Same story for the scoring dot. Mirrors dot32_x16_mmx INCLUDING its
   x256 product scaling, so the caller's single 1/256 is correct either
   way. */
static int32_t dot32_x16_attn_ref(const int8_t *w, const int16_t *x) {
    int32_t acc = 0;
    int k;
    for (k = 0; k < 32; k++) acc += (int32_t)w[k] * 256 * (int32_t)x[k];
    return acc;
}
#if defined(LZ_ARM_ASM_EXTERN)
/* NO NEW KERNEL ON ARM EITHER. This dot is int8 weights x int16
   activations with the x256 fold, which is exactly what q8_0's row leaf
   computes, so --kernel arm-asm calls that one rather than growing a
   second copy of the same LDM + halfword-selection body
   (docs/arm-asm-audit.md section 6 asked for this explicitly).
   256*sum(w*x) and sum(256*w*x) are the same value in int32 - two's
   complement multiplication distributes over addition even where the
   sum wraps - so the reuse owes identity, not a tolerance. */
static int32_t dot32_x16_attn_arm(const int8_t *w, const int16_t *x) {
    if (g_kernel == LZ_KERNEL_ARM_ASM) return lz_dot32_q8_arm_asm(w, x);
    return dot32_x16_attn_ref(w, x);
}
#define LZ_ATTN_DOT32(w, x) dot32_x16_attn_arm(w, x)
#else
#define LZ_ATTN_DOT32(w, x) dot32_x16_attn_ref(w, x)
#endif /* LZ_ARM_ASM_EXTERN */
#else
/* The MMX/SSE2 split, same tier test as LZ_WSUM_PAIR above. The SSE2
   arm is a REUSE of the matmul's own SSE2 Q8 leaf, exactly as the ARM
   arm above reuses q8_0's - and it owes identity for the same reason,
   the x256 fold distributing over the sum in int32. */
#if defined(LZ_ATTN_SSE2_EXTERN)
static int32_t dot32_x16_attn_x86(const int8_t *w, const int16_t *x) {
    if (g_kernel == LZ_KERNEL_SSE2) return lz_dot32_x16_sse2(w, x);
    return dot32_x16_mmx(w, x);
}
#define LZ_ATTN_DOT32(w, x) dot32_x16_attn_x86(w, x)
#else
#define LZ_ATTN_DOT32(w, x) dot32_x16_mmx(w, x)
#endif /* LZ_ATTN_SSE2_EXTERN */
#endif /* !__WATCOMC__ && !LZ_DOT_MMX_EXTERN */

/* Empty everywhere but the ARM cross-build: Watcom never sees __arm__
   and has no __attribute__, same shape as matmul's LZ_XW_ALIGN. */
#if defined(LZ_ARM_ASM_EXTERN)
#define LZ_ATTN_QW_ALIGN __attribute__((aligned(4)))
#else
#define LZ_ATTN_QW_ALIGN
#endif /* LZ_ARM_ASM_EXTERN */

void lz_attn_score_q8(float *att, const float *qhh, int hd,
                      const int8_t *kc, const float *ks, int kvd,
                      int pos, float scale, int sink, int ring) {
    /* static, not stack (Win98 has a small stack). qw carries the same aligned(4)
       as matmul's g_xw and for the same reason: the hand-written leaf
       reads it with LDM, and int16_t promises two bytes, so whatever
       alignment gcc happens to give the array is not the contract. */
    static int8_t  qq[LZ_ATTN_MAX_HD];
    static float   qs[LZ_ATTN_MAX_HD / 32];
    static int16_t qw[LZ_ATTN_MAX_HD] LZ_ATTN_QW_ALIGN;
    int ng = hd / 32, g, t, i, slot = 0;
    float sc;

    /* Once per head: the q quantization - billed by lz_quantize_q8
       itself under LZ_FC_QUANT, NOT here. This site carries no
       approximation for it (an mul=hd, div=ng estimate here would
       double-count, since the quantizer now has its own site). Then
       per t: a convert, the qs*kts product
       and a multiply-add per group, plus 3 adds and 2 multiplies to
       fold. This is the term that makes the fixed tier LOSE below ~200
       tokens - it does not shrink with context, and the per-t term it
       replaces is 2*hd.

       Per-t, per group: cvt=1, mul=2 (qs*kts, then the outer product),
       add=1 (accumulate) - times ng. Per-t fold: 1 mul for `* sc`, and
       the three fold adds ONLY when ng >= 4.

       NEITHER TAIL IS FLAT, and only one of the two flat forms was
       actually wrong here - measuring is what separated them:
         - mul is 2*ng+1, not 2*ng+2. The extra one would be
           `* (1.0f/256.0f)` applied after `* scale`, and the 2^-8 is
           folded into sc above, so no such multiply is performed.
           Worth 4.8% of the class: kmr20 at 603 tokens, 101,472
           against 96,640.
         - add is ng+3 only when ng >= 4. Below that the three fold adds
           are adds of +0.0f and the conditional below skips them. On
           this model ng is 4 (head_dim 128), so the three ARE performed
           and the condition changes the census by exactly zero here;
           it is written for the 32- and 64-wide heads where it is not.
       A guess put the first at 38%, on ng being 2. An optimisation that
       does not update its own site leaves the row claiming work nobody
       performs; a correction that does not measure replaces one wrong
       number with another. */
    LZ_FCX(LZ_FC_ATTN_SCORE,
           (long)(pos + 1) * (2 * ng + 1),
           (long)(pos + 1) * (ng + (ng >= 4 ? 3 : 0)),
           0,
           (long)(pos + 1) * ng,
           0);
    lz_quantize_q8(qhh, hd, 32, qq, qs);
    for (i = 0; i < hd; i++) qw[i] = (int16_t)qq[i];

    /* The 2^-8 that undoes LZ_ATTN_DOT32's `* 256`, folded into scale
       ONCE instead of once per row. Exact, and therefore the same
       number the two-step form produces: multiplying by a power of two
       only moves the exponent, so fl(fold * 2^-8) is fold * 2^-8
       exactly and fl(2^-8 * scale) is 2^-8 * scale exactly - both forms
       round the identical product fold * scale * 2^-8 exactly once.
       The one place that stops being true is an operand the scaling
       pushes out of the normal range, and the caller's scale is
       1/sqrt(head_dim); a fold small enough to go subnormal at 2^-8 is
       an attention score under 2^-118. */
    sc = (1.0f / 256.0f) * scale;

    for (t = 0; t <= pos; t++, LZ_SLOT_NEXT(slot, sink, ring)) {
        const int8_t *kt = kc + (size_t)slot * kvd;
        const float *kts = ks + (size_t)slot * (kvd / 32);
        int32_t rr[LZ_ATTN_MAX_HD / 32];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

        /* Phase 1 is MMX only: collect every group's raw dot before any
           x87 runs. Interleaving the two leaves MMX registers dirty
           across an x87 fld/fmul. It
           showed 0 mismatches in a short bit-identity test and -nan in a
           longer one, on the Watcom build only. */
        for (g = 0; g < ng; g++)
            rr[g] = LZ_ATTN_DOT32(kt + (size_t)g * 32, qw + g * 32);
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)
        if (lz_cpu_has_mmx()) _mm_empty();
#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */
        /* Phase 2, x87 only. Raw magnitude reaches 256*127*127*32 =
           1.33e8 > 2^24, so lz_i32f() is mandatory here,
           not a stylistic choice. */
        for (g = 0; g + 3 < ng; g += 4) {
            a0 += lz_i32f(rr[g + 0]) * (qs[g + 0] * kts[g + 0]);
            a1 += lz_i32f(rr[g + 1]) * (qs[g + 1] * kts[g + 1]);
            a2 += lz_i32f(rr[g + 2]) * (qs[g + 2] * kts[g + 2]);
            a3 += lz_i32f(rr[g + 3]) * (qs[g + 3] * kts[g + 3]);
        }
        for (; g < ng; g++)
            a0 += lz_i32f(rr[g]) * (qs[g] * kts[g]);
        /* THREE OF THESE ADDS ARE ADDS OF ZERO whenever ng < 4, and a
           float add of zero costs 43 instructions on a soft-float
           target - the unrolled loop above needs ng >= 4 to run at all,
           so for a 32- or 64-wide head every group lands in a0 and a1,
           a2, a3 are still their +0.0f initializers.

           Dropping them is exact, not close: x + (+0.0f) is x for every
           x except -0.0f, which would come back as +0.0f - and a0
           cannot be -0.0f, because it starts at +0.0f and 0.0f + y is y
           (or +0.0f when y is -0.0f). NaN and infinities pass through
           either form unchanged. */
        att[t] = (ng >= 4 ? ((a0 + a2) + (a1 + a3)) : a0) * sc;
    }
}
#endif /* LZ_ATTN_FIXED & 1 */

#if (LZ_ATTN_FIXED & 2)
/* cbuf/cq are caller-provided with seq_len entries each: the coefficient
   run is as long as the context, so unlike the per-head buffers above it
   cannot be a fixed static array. */
void lz_attn_wsum_q8(float *out, const float *att, int hd,
                     const int8_t *vc, const float *vs, int kvd,
                     int pos, float *cbuf, int16_t *cq, int sink, int ring) {
    int ng = hd / 32, g, T = pos + 1;
    /* Two independent walkers rather than one slot array: both passes
       below visit t in ascending order from 0, so each can carry its own
       running slot. An array would need a fixed cap, and a cap that a
       long context silently exceeds is the same class of defect this
       whole change exists to fix. */

    /* Per group: T to build the coefficients (mul, the cmax tracking
       compare is not billed), 1 divide plus 2T to quantize them (mul
       + q8_round add per t, common path), 64 per int32 chunk to drain
       the accumulator (32 convert + 32 add), and 33 to scale the 32
       outputs (all mul). The chunk term is why the count is not a
       clean multiple of T.

       THAT ONE DIVIDE PER GROUP STAYS, here and in the int16 twin
       below, and the decision was measured. `inv = 32767/cmax` and the
       `sscale = cmax/32767` at the end are reciprocals of each other
       and both live inside the function, so normalising cmax to a power
       of two - the move that took gdn_build_table's recurrence divides
       to zero - would make both exact exponent writes and remove it. It
       does not pay for itself:

         gain  64 divides per token of the engine's 688, at 117
               soft-float instructions each: 7,488 against roughly
               2.0e7 per token. 0.037%.
         cost  a numeric change, hence the paired-NLL run over 1262
               positions before it could ship. 200,000 synthetic groups
               put the worst reconstruction error at 3.29e-04 of cmax
               and the mean at 6.13e-05 - about ten times the 15-bit
               grid, in line with what the GDN table's normalisation
               costs there.

       Measured against cmax, NOT against the summed output: that sum is
       a difference of signed terms and goes near zero by cancellation,
       where the same absolute error reads as 2.05% and looks like a
       reason to refuse. cmax is what sets the quantiser's grid. */
    LZ_FCX(LZ_FC_ATTN_WSUM,
           (long)ng * (2L * T + 33L),
           (long)ng * ((long)T + 32L * ((T + LZ_WSUM_CHUNK - 1) / LZ_WSUM_CHUNK)),
           (long)ng,
           (long)ng * 32L * ((T + LZ_WSUM_CHUNK - 1) / LZ_WSUM_CHUNK),
           0);
    for (g = 0; g < ng; g++) {
        int32_t acc32[32];
        float accf[32];
        float cmax = 0.0f, inv;
        int t, d, t0;

        int slot = 0;
        /* THE MAGNITUDE SCAN RUNS IN THE INTEGER DOMAIN, the same move
           lz_softmax's max scan makes and for the same reason: `ac >
           cmax` is a bl __aeabi_fcmpgt, about 30 instructions, once per
           coefficient, and the sign-strip that feeds it is a compare
           and a negate on top.

           Simpler than softmax's key, because both operands are already
           non-negative: clearing the sign bit IS fabs, and non-negative
           floats compare exactly as their bit patterns do when read as
           unsigned - no order-preserving transform is needed. The
           winning pattern is then the float itself, so cmax comes back
           by a plain reinterpret with nothing to undo.

           NaN is the one input where this differs from the float form,
           and it takes the same disposition lz_softmax's scan does:
           float `>` is false for NaN so a float scan never selects one,
           while NaN's magnitude bits exceed every finite value and this
           one would. A NaN reaching attention weighting means the
           upstream is already destroyed. */
        {
            union { float f; uint32_t u; } b;
            uint32_t best = 0;
            for (t = 0; t < T; t++, LZ_SLOT_NEXT(slot, sink, ring)) {
                uint32_t a;
                b.f = att[t] * vs[(size_t)slot * (kvd / 32) + g];
                cbuf[t] = b.f;
                a = b.u & 0x7FFFFFFFu;
                if (a > best) best = a;
            }
            b.u = best;
            cmax = b.f;
        }
        if (cmax > 0.0f) {
            inv = 32767.0f / cmax;
            if (!(inv <= FLT_MAX)) {
                /* cmax subnormal: 32767/cmax overflows to inf and (int)inf
                   is undefined behavior. Same guard lz_quantize_q8 carries. */
                for (t = 0; t < T; t++) {
                    int qi = q8_round(cbuf[t] / (cmax * LZ_Q15_INV));
                    if (qi >  32767) qi =  32767;
                    if (qi < -32767) qi = -32767;
                    cq[t] = (int16_t)qi;
                }
            } else {
                for (t = 0; t < T; t++) {
                    int qi = q8_round(cbuf[t] * inv);
                    if (qi >  32767) qi =  32767;
                    if (qi < -32767) qi = -32767;
                    cq[t] = (int16_t)qi;
                }
            }
        } else {
            for (t = 0; t < T; t++) cq[t] = 0;
        }

#if defined(__WATCOMC__)
        /* Watcom: one call per GROUP, not per pair. The #pragma aux
           body expands inline at the call site, once per pair, with
           zero real call overhead; because the body lives in
           src/ops_mmx.c, the pair loop is there too - a per-pair
           cross-TU call would cost up to T/2 of them, T being
           context-length-dependent and unbounded (262144 on this
           model). lz_wsum_group_mmx does the ENTIRE multi-chunk
           second walk (all of LZ_WSUM_CHUNK's chunks, all pairs, the
           ring-buffer slot walk, the per-chunk emms and lz_i32f fold)
           for this one group, so ops.c's call count per group is
           exactly one, not T/2. */
        lz_wsum_group_mmx(vc, kvd, g, sink, ring, cq, T, accf);
#else
        for (d = 0; d < 32; d++) accf[d] = 0.0f;
        slot = 0;                       /* second walk, same t order */
        for (t0 = 0; t0 < T; t0 += LZ_WSUM_CHUNK) {
            int tn = T - t0;
            int tp;
            if (tn > LZ_WSUM_CHUNK) tn = LZ_WSUM_CHUNK;
            memset(acc32, 0, sizeof(acc32));
            for (tp = t0; tp + 1 < t0 + tn; tp += 2) {
                int16_t coef[4];
                const int8_t *rowA, *rowB;
                int slotA = slot;
                LZ_SLOT_NEXT(slot, sink, ring);
                rowA = vc + (size_t)slotA * kvd + (size_t)g * 32;
                rowB = vc + (size_t)slot  * kvd + (size_t)g * 32;
                LZ_SLOT_NEXT(slot, sink, ring);
                coef[0] = cq[tp];     coef[1] = cq[tp + 1];
                coef[2] = cq[tp];     coef[3] = cq[tp + 1];
                LZ_WSUM_PAIR(rowA, rowB, coef, acc32);
            }
            if (tp < t0 + tn) {            /* odd leftover row in this chunk */
                int16_t coef[4];
                const int8_t *rowA = vc + (size_t)slot * kvd + (size_t)g * 32;
                coef[0] = cq[tp]; coef[1] = 0;
                coef[2] = cq[tp]; coef[3] = 0;
                LZ_WSUM_PAIR(rowA, rowA, coef, acc32);
                LZ_SLOT_NEXT(slot, sink, ring);
            }
#if defined(LZ_WSUM_MMX_EXTERN)
            if (lz_cpu_has_mmx()) _mm_empty();      /* no x87 inside the row-pair loop above */
#endif /* LZ_WSUM_MMX_EXTERN */
            LZ_I32F_ACC32(accf, acc32, d);
        }
#endif /* __WATCOMC__ */
        {
            float sscale = cmax * LZ_Q15_INV;
            for (d = 0; d < 32; d++) out[g * 32 + d] = accf[d] * sscale;
        }
    }
}

/* int-output twin of lz_attn_wsum_q8: same coefficient pass, but the
   second walk accumulates the full-context sum in int64 (each 512-row
   int32 chunk fits by the bound above; their sum does not - T*4,161,409
   passes INT32_MAX at T=516) and hands out that exact sum plus the
   per-32-group dequant scale, so the caller can carry the weighted sum
   through the output gate and lz_quantize_q8_int64 without the
   int32->float->int8 double conversion. Integer sums are exact under
   reassociation, so every chunk order is bit-identical. */
void lz_attn_wsum_q8_int(int64_t *acc, float *sscale, const float *att,
                         int hd, const int8_t *vc, const float *vs,
                         int kvd, int pos, float *cbuf, int16_t *cq,
                         int sink, int ring) {
    int ng = hd / 32, g, T = pos + 1;
    LZ_FCX(LZ_FC_ATTN_WSUM,
           (long)ng * (2L * T + 33L),
           (long)ng * ((long)T + 32L * ((T + LZ_WSUM_CHUNK - 1) / LZ_WSUM_CHUNK)),
           (long)ng,
           (long)ng * 32L * ((T + LZ_WSUM_CHUNK - 1) / LZ_WSUM_CHUNK),
           0);
    for (g = 0; g < ng; g++) {
        int64_t acc64[32];
        float cmax = 0.0f, inv;
        int t, d, t0;

        int slot = 0;
        /* THE MAGNITUDE SCAN RUNS IN THE INTEGER DOMAIN, the same move
           lz_softmax's max scan makes and for the same reason: `ac >
           cmax` is a bl __aeabi_fcmpgt, about 30 instructions, once per
           coefficient, and the sign-strip that feeds it is a compare
           and a negate on top.

           Simpler than softmax's key, because both operands are already
           non-negative: clearing the sign bit IS fabs, and non-negative
           floats compare exactly as their bit patterns do when read as
           unsigned - no order-preserving transform is needed. The
           winning pattern is then the float itself, so cmax comes back
           by a plain reinterpret with nothing to undo.

           NaN is the one input where this differs from the float form,
           and it takes the same disposition lz_softmax's scan does:
           float `>` is false for NaN so a float scan never selects one,
           while NaN's magnitude bits exceed every finite value and this
           one would. A NaN reaching attention weighting means the
           upstream is already destroyed. */
        {
            union { float f; uint32_t u; } b;
            uint32_t best = 0;
            for (t = 0; t < T; t++, LZ_SLOT_NEXT(slot, sink, ring)) {
                uint32_t a;
                b.f = att[t] * vs[(size_t)slot * (kvd / 32) + g];
                cbuf[t] = b.f;
                a = b.u & 0x7FFFFFFFu;
                if (a > best) best = a;
            }
            b.u = best;
            cmax = b.f;
        }
        if (cmax > 0.0f) {
            inv = 32767.0f / cmax;
            if (!(inv <= FLT_MAX)) {
                /* cmax subnormal: 32767/cmax overflows to inf and (int)inf
                   is undefined behavior. Same guard lz_quantize_q8 carries. */
                for (t = 0; t < T; t++) {
                    int qi = q8_round(cbuf[t] / (cmax * LZ_Q15_INV));
                    if (qi >  32767) qi =  32767;
                    if (qi < -32767) qi = -32767;
                    cq[t] = (int16_t)qi;
                }
            } else {
                for (t = 0; t < T; t++) {
                    int qi = q8_round(cbuf[t] * inv);
                    if (qi >  32767) qi =  32767;
                    if (qi < -32767) qi = -32767;
                    cq[t] = (int16_t)qi;
                }
            }
        } else {
            for (t = 0; t < T; t++) cq[t] = 0;
        }

#if defined(__WATCOMC__)
        /* Same one-call-per-GROUP shape as lz_attn_wsum_q8's Watcom
           branch: lz_wsum_group_mmx_int does the entire multi-chunk
           second walk and the int64 fold behind one cross-TU call. */
        lz_wsum_group_mmx_int(vc, kvd, g, sink, ring, cq, T, acc64);
#else
        {
        int32_t acc32[32];
        for (d = 0; d < 32; d++) acc64[d] = 0;
        slot = 0;                       /* second walk, same t order */
        for (t0 = 0; t0 < T; t0 += LZ_WSUM_CHUNK) {
            int tn = T - t0;
            int tp;
            if (tn > LZ_WSUM_CHUNK) tn = LZ_WSUM_CHUNK;
            memset(acc32, 0, sizeof(acc32));
            for (tp = t0; tp + 1 < t0 + tn; tp += 2) {
                int16_t coef[4];
                const int8_t *rowA, *rowB;
                int slotA = slot;
                LZ_SLOT_NEXT(slot, sink, ring);
                rowA = vc + (size_t)slotA * kvd + (size_t)g * 32;
                rowB = vc + (size_t)slot  * kvd + (size_t)g * 32;
                LZ_SLOT_NEXT(slot, sink, ring);
                coef[0] = cq[tp];     coef[1] = cq[tp + 1];
                coef[2] = cq[tp];     coef[3] = cq[tp + 1];
                LZ_WSUM_PAIR(rowA, rowB, coef, acc32);
            }
            if (tp < t0 + tn) {            /* odd leftover row in this chunk */
                int16_t coef[4];
                const int8_t *rowA = vc + (size_t)slot * kvd + (size_t)g * 32;
                coef[0] = cq[tp]; coef[1] = 0;
                coef[2] = cq[tp]; coef[3] = 0;
                LZ_WSUM_PAIR(rowA, rowA, coef, acc32);
                LZ_SLOT_NEXT(slot, sink, ring);
            }
#if defined(LZ_WSUM_MMX_EXTERN)
            if (lz_cpu_has_mmx()) _mm_empty();      /* no x87 inside the row-pair loop above */
#endif /* LZ_WSUM_MMX_EXTERN */
            for (d = 0; d < 32; d++) acc64[d] += acc32[d];
        }
        }
#endif /* __WATCOMC__ */
        sscale[g] = cmax * LZ_Q15_INV;
        for (d = 0; d < 32; d++) acc[g * 32 + d] = acc64[d];
    }
}
#endif /* LZ_ATTN_FIXED & 2 */
#endif /* LZ_ATTN_FIXED */

/* ---- SSM recurrence pass 1, fixed point --------------------------------

   Pass 1 is the two contractions u[vv] = sum_kk S[kk][vv]*ss[kk][gg]*k[kk]
   and the same with q. In float it is kd*vd multiply-adds over an int8
   state converted to float one element at a time - the state is ALREADY
   quantized, and the float math only carries the coefficients at full
   width. Quantizing the coefficients to int16 as well lets the whole
   contraction run as pmaddwd directly on the int8 state, with one dequant
   per 32 lanes instead of one float convert per element.

   The summation is exact - int32 accumulation of int8 x int16 products
   loses nothing. The ONLY new error source is quantizing the coefficients,
   which is why the precision is a dial rather than a fixed cost:

     LZ_GDN_FIXED=0  float pass 1, the original code.
     LZ_GDN_FIXED=1  one int16 coefficient plane. Coefficient error <= su/2,
                     measured single-step relative error 1.8e-05 on both
                     compilers (closed form: 169/(32767*337) ~= 1.5e-5).
     LZ_GDN_FIXED=2  two planes - the residual the first plane rounds away,
                     re-quantized 32767x finer and summed with the SAME
                     kernel against a second table. Coefficient error drops
                     to su/(2*32767) ~= 4.7e-10 relative, i.e. below float32
                     epsilon: the error this optimization introduces stops
                     being measurable. Costs 2x the integer work in pass 1
                     and needs no new assembly.

   Tier 1 is the default, and the reason is a measurement worth keeping:

   Tier 2 does exactly what it claims arithmetically - on s1v3 the logit
   difference against tier 0 is 2.4e-07 at ntok=2 and 2.1e-07 at ntok=3,
   i.e. 1.9e-05 of tier 1's, matching the predicted 1/32767. Then at ntok=4
   it jumps to 1.5e-02, the same band tier 1 sits in, and the ratio between
   the tiers becomes O(1) with random sign.

   The discontinuity is pass 2. It re-quantizes the state to int8 with
   lz_quantize_q8, and that is a DISCRETE decision: the moment any element
   lands on the other side of a rounding boundary the state differs by one
   LSB, which is ~8.5e-4 - orders of magnitude above either tier's
   coefficient error. There are millions of such decisions per token, so a
   boundary crossing is a matter of when, not if. Past that point the logit
   difference is set by LSB flips, not by coefficient precision.

   Which means the end-to-end ~1e-2 is NOT a precision defect of the fixed
   point pass. It is this engine's standing sensitivity to any perturbation
   of pass 1 - a legal reassociation would produce the same magnitude. All
   tier 2 buys is delaying the first divergence from token 2 to token 4, at
   2x the pass-1 cost. Not worth it for a chat workload; kept compiled-out
   because it becomes the right tier the day the state stops being int8.

   The place to spend on SSM fidelity is therefore pass 2's state
   quantization, not pass 1's coefficients.

   It applies to EVERY build, not only the MMX ones. A tier that is
   deliberately non-bit-identical must not be selected by which compiler
   you used, or the cross-compiler byte-compare gate stops meaning
   anything. So the non-MMX path gets a scalar fixed-point summation that
   shares the coefficient table; integer sums are exact under
   reassociation, so the two are bit-identical to each other. */
#if LZ_GDN_FIXED
/* Table is int16_t[LZ_GDN_MAX_KD/2][8]; bump both together. A kd past this
   takes the float pass 1 rather than truncating silently. */
#define LZ_GDN_MAX_KD 256

#if defined(__WATCOMC__) || defined(LZ_GDN1_MMX_EXTERN)
#define LZ_GDN_HAVE_MMX 1
#endif /* __WATCOMC__ || LZ_GDN1_MMX_EXTERN */

#if defined(LZ_GDN_HAVE_MMX)
#if defined(__WATCOMC__)
/* lz_gdn1_x4_asm's Watcom body, and the whole of gdn_sum_gg's MMX
   branch that calls it 8 times per group, live in src/ops_mmx.c as
   lz_gdn_sum_gg_mmx - it writes %mm registers, and wcc386's
   -Nr/-Ns ceiling cannot be raised per-function the way a #pragma aux
   ".586" directive implies, so it needs its own translation unit now
   that ops.c targets the lower floor. Declared in src/ops_mmx.h,
   already visible here. gdn_sum_gg's Watcom branch below calls it once
   per group - the same call count this function has from its own
   caller, just crossing a TU boundary now. */
#else
/* lz_gdn1_x4_asm's gcc body lives in src/ops_mmx.c - it writes %mm
   registers. Declared in src/ops_mmx.h, already visible here. */
#endif /* __WATCOMC__ */
#endif /* LZ_GDN_HAVE_MMX */

/* Coefficients for one gg (32 vv lanes): kd/2 kk-pair entries of 8 int16,
   [ckA,ckB,ckA,ckB, cqA,cqB,cqA,cqB] - each broadcast to 4 lanes so
   pmaddwd can take it straight as a memory operand. An odd kd leaves one
   unpaired row, whose coefficients go to tail_cu/tail_cw and are summed in
   scalar code. Every current model has even kd, so that path is
   defensive only. */
typedef struct {
    int16_t t[LZ_GDN_MAX_KD / 2][8];
    int16_t tail_cu, tail_cw;
    float su, sw;         /* dequant scale: amax/32767, or 1/32767 for an all-zero group */
#if LZ_GDN_FIXED >= 2
    /* Second plane: the residual left by the first plane's rounding,
       re-quantized at 32767x finer scale. |residual| <= 0.5 by
       construction, so |t2| <= 16384 - inside int16 with a bit to spare,
       and the accumulator bound is 16384*127*256 = 5.3e8, still inside
       int32. Same kernel, second table: no new assembly, so the
       bit-identity already established for lz_gdn1_x4_asm carries over
       unchanged. */
    int16_t t2[LZ_GDN_MAX_KD / 2][8];
    int16_t tail2_cu, tail2_cw;
    float su2, sw2;       /* = su/32767, sw/32767 */
#endif /* LZ_GDN_FIXED >= 2 */
} LZGdnTable;

static void gdn_build_table(const float *ss, const float *q, const float *k,
                            int kd, int vd, int gg, LZGdnTable *tb) {
    /* static, not stack (Win98 has a small stack) - and the reason they exist
       at all is to keep the second loop from RECOMPUTING ss*k and ss*q.
       Reusing the first loop's products is exact, not close: that loop
       evaluates left to right, so `ss*k*ik` is fl(fl(ss*k)*ik) either
       way, the same two roundings of the same two products. Worth
       2*kd multiplies per call - 24,576 a token on kmr20. */
    static float ka[LZ_GDN_MAX_KD], qa[LZ_GDN_MAX_KD];
    int ng = vd / 32, kk;
    float amk = 0.0f, amq = 0.0f, ik, iq;
    /* The block exponents: ik is 2^-ek and iq is 2^-eq exactly, so
       loop 2 scales by an exponent add. ik_p2/iq_p2 say whether that
       holds - they go 0 on the amax==0 arm, where ik is 0 and no power
       of two describes it. */
    int ek = 0, eq = 0, ik_p2 = 1, iq_p2 = 1;

    /* THIS FUNCTION HAD NO CENSUS SITE. It runs once per group inside
       gdn_pass1_fixed, which itself was unbilled until one commit ago,
       so the whole coefficient table build read as zero. Found by
       .prof/unbilled_scan.awk. Per call, at LZ_GDN_FIXED == 1:
         loop 1   0 mul per kk. ss is the state's group scale and
                  p2_group_scale writes it as pow2f(ee), so both
                  products are exponent adds - free, no value traded,
                  the power of two was already there.
         scales   0 mul, 0 div. su/sw and ik/iq are pow2f of the block
                  exponent - bit construction, nothing to bill. This is
                  what the block normalization buys: without it the pair
                  costs 2 mul (amk*LZ_Q15_INV) and 2 div (32767/amk),
                  and the divides are why the normalization is there.
         loop 2   0 mul (the two ik/iq products are exponent adds after
                  the block normalization) and 2 add per kk (q8_round's
                  magic add, billed the way every other quantize site in
                  this file bills it)
       The absmax compares stay unbilled - the census's standing
       convention, not an omission here.

       The LZ_GDN_FIXED >= 2 residual plane below adds 2 cvt, 2 add and
       2 mul per kk and is NOT in this formula: it compiles out at the
       default level, and a term for a path this build does not take is
       a term nobody can check. */
    LZ_FCX(LZ_FC_RECUR, 0, (long)kd * 2, 0, 0, 0);
    /* THE TWO ABSMAX SCANS RUN IN THE INTEGER DOMAIN. They were four
       float comparisons per kk - `a < 0.0f` and `b < 0.0f` for the sign,
       `t > amk` and `t > amq` for the max - which on a soft-float target
       is four `bl __aeabi_fcmpXX` at ~25 instructions each. Masking off
       the sign bit gives the magnitude, and for magnitudes the unsigned
       bit-pattern order IS the numeric order, so this is a permutation
       of the same total order and owes the float loop bit-identity.
       Same transformation as gdn_p2_fixed_rows' delta scan, q8_amax and
       lz_softmax's max scan. NaN cannot reach here - a
       and b are finite products of finite state and inputs - and a
       negative zero masks to +0, which the float form also read as 0. */
    /* THE TWO PRODUCTS ARE EXPONENT ADDS TOO, and this one is free -
       no value change, nothing traded. ss is the STATE's group scale,
       written by p2_group_scale, which returns pow2f(ee) on its main
       path: it is already an exact power of two and nobody had used
       that. One bit test per kk serves both products, since a and b
       scale by the same ss. The guard is live - p2_group_scale's
       out-of-range arm returns sacc*pow2f(...) and its zero arm returns
       1.0f, and on the FLOAT recurrence ss comes from q8_group_scale's
       amax*(1/127) instead, which is not a power of two at all. */
    {
        union { float f; uint32_t u; } mb, sb;
        uint32_t amku = 0u, amqu = 0u;
        for (kk = 0; kk < kd; kk++) {
            float ssv = ss[(size_t)kk * ng + gg];
            float a, b;
            uint32_t t;
            int se, sp2;
            sb.f = ssv;
            sp2 = ((sb.u & 0x007FFFFFu) == 0u)
                  && (uint32_t)(((sb.u >> 23) & 0xFFu) - 1u) < 0xFEu;
            se = (int)((sb.u >> 23) & 0xFFu) - 127;
            if (!sp2 || !f32_scale_pow2(k[kk], se, &a)) a = ssv * k[kk];
            if (!sp2 || !f32_scale_pow2(q[kk], se, &b)) b = ssv * q[kk];
            ka[kk] = a;
            qa[kk] = b;
            mb.f = a; t = mb.u & 0x7FFFFFFFu; if (t > amku) amku = t;
            mb.f = b; t = mb.u & 0x7FFFFFFFu; if (t > amqu) amqu = t;
        }
        mb.u = amku; amk = mb.f;
        mb.u = amqu; amq = mb.f;
    }
    /* BLOCK-NORMALIZED to 32767 * 2^k, which makes su = 2^k and
       ik = 2^-k EXACTLY - so loop 2's two multiplies per kk become
       exponent adds, and the two divides go away entirely rather than
       going through LZ_FDIV.
     *
     * su and ik are built from pow2f, NOT from amk * LZ_Q15_INV and
     * 32767/amk. Those would be a ULP off: LZ_Q15_INV is fl(1/32767)
     * and 32767 * fl(1/32767) is not exactly 1, so the product is not
     * exactly 2^k and f32_scale_pow2 would be folding the wrong number.
     * The division IS exact at the normalized amk (both operands exact,
     * quotient exactly 2^-k), but there is no reason to spend it.
     *
     * VALUE-CHANGING, and measured before landing rather than after.
     * Rounding amk UP to the next 32767*2^k costs up to one bit of the
     * coefficient table's range. Paired NLL, three models x two
     * fixtures, n=3786: +6.45e-04 at t=+0.48 - indistinguishable, and
     * in the same band as the attention int wsum's default change,
     * which shipped at t=0.65. The GLOBAL version of this idea, snapping
     * every quantizer's group scale, was rejected at +4.72e-03 and
     * t=+2.70 (docs/int-pipeline-project.md 0.6); localising it to this
     * one table is what brings the cost down sevenfold. That is the
     * whole reason this lands and that one did not.
     *
     * The same caution applies to any future code that wants the
     * un-normalized `amk / 32767.0f` ratio: Watcom strength-
     * reduces it and gcc does not, 1 ULP apart, so write the multiply. */
    if (amk > 0.0f) {
        float r = amk * LZ_Q15_INV;
        ek = expof(r);
        if (pow2f(ek) < r) ek++;          /* round the block UP */
        tb->su = pow2f(ek);
        ik = pow2f(-ek);
    } else {
        tb->su = LZ_Q15_INV;
        ik = 0.0f;
        ik_p2 = 0;
    }
    if (amq > 0.0f) {
        float r = amq * LZ_Q15_INV;
        eq = expof(r);
        if (pow2f(eq) < r) eq++;
        tb->sw = pow2f(eq);
        iq = pow2f(-eq);
    } else {
        tb->sw = LZ_Q15_INV;
        iq = 0.0f;
        iq_p2 = 0;
    }
    tb->tail_cu = 0; tb->tail_cw = 0;
#if LZ_GDN_FIXED >= 2
    tb->su2 = tb->su * LZ_Q15_INV;
    tb->sw2 = tb->sw * LZ_Q15_INV;
    tb->tail2_cu = 0; tb->tail2_cw = 0;
#endif /* LZ_GDN_FIXED >= 2 */

    for (kk = 0; kk < kd; kk++) {
        /* ik and iq are exact powers of two after the block
           normalization above, so these are exponent adds. The float
           multiply stays as the fallback for the amax==0 arm and for
           the operands f32_scale_pow2 declines (a zero ka[kk] is
           ordinary), which is also what keeps this bit-identical to the
           multiply wherever the multiply was already exact. */
        float fa, fb;
        int ia, ib;
        int kp = kk >> 1, half = kk & 1;
        int tail = (kk == kd - 1 && (kd & 1));
        if (!ik_p2 || !f32_scale_pow2(ka[kk], -ek, &fa)) fa = ka[kk] * ik;
        if (!iq_p2 || !f32_scale_pow2(qa[kk], -eq, &fb)) fb = qa[kk] * iq;
        ia = q8_round(fa);
        ib = q8_round(fb);
        if (ia >  32767) ia =  32767;
        if (ia < -32767) ia = -32767;
        if (ib >  32767) ib =  32767;
        if (ib < -32767) ib = -32767;
        if (tail) {
            tb->tail_cu = (int16_t)ia;
            tb->tail_cw = (int16_t)ib;
        } else {
            int16_t *slot = tb->t[kp];
            if (half == 0) {
                slot[0] = (int16_t)ia; slot[2] = (int16_t)ia;
                slot[4] = (int16_t)ib; slot[6] = (int16_t)ib;
            } else {
                slot[1] = (int16_t)ia; slot[3] = (int16_t)ia;
                slot[5] = (int16_t)ib; slot[7] = (int16_t)ib;
            }
        }
#if LZ_GDN_FIXED >= 2
        {
            /* Residual of the first plane, in units of su. Clamped
               defensively: |fa - ia| <= 0.5 holds whenever the clamp above
               did not fire, and when it did the residual is bounded by the
               clamp distance instead - which is why the range check is not
               an assert. */
            int i2a = q8_round((fa - (float)ia) * 32767.0f);
            int i2b = q8_round((fb - (float)ib) * 32767.0f);
            if (i2a >  32767) i2a =  32767;
            if (i2a < -32767) i2a = -32767;
            if (i2b >  32767) i2b =  32767;
            if (i2b < -32767) i2b = -32767;
            if (tail) {
                tb->tail2_cu = (int16_t)i2a;
                tb->tail2_cw = (int16_t)i2b;
            } else {
                int16_t *s2 = tb->t2[kp];
                if (half == 0) {
                    s2[0] = (int16_t)i2a; s2[2] = (int16_t)i2a;
                    s2[4] = (int16_t)i2b; s2[6] = (int16_t)i2b;
                } else {
                    s2[1] = (int16_t)i2a; s2[3] = (int16_t)i2a;
                    s2[5] = (int16_t)i2b; s2[7] = (int16_t)i2b;
                }
            }
        }
#endif /* LZ_GDN_FIXED >= 2 */
    }
}

/* gdn_tail_row: defined in src/ops_kernel_shared.h - src/ops_mmx.c's
   MMX branch needs the exact same odd-tail-row fixup. */

#if !defined(LZ_GDN_HAVE_MMX)
/* Non-MMX builds. Same table, same integer arithmetic, different summation
   order - and integer sums are exact under reassociation, so this is
   bit-identical to the MMX version. Verified over 200000+ random groups
   plus extremes. */
/* DO NOT "FIX" THE DOUBLE LOAD IN THIS LOOP FROM C. It looks like free
   money and it is not; three shapes were tried and disassembled.
 *
 * gcc emits 16 ARM instructions per lane here, and rowA[j]/rowB[j] are
 * each read TWICE - int8_t is a character type and may alias int32_t,
 * so after the store to au_gg[j] the compiler must assume the state row
 * changed under it, and C89 has no `restrict` to say otherwise.
 * Hoisting them into locals does remove the reload: 16 -> 14.
 *
 * IT ALSO TURNS FOUR SMULBB/SMLABB INTO FOUR MUL/MLA. gcc fuses the
 * narrow load into the halfword multiply only while each byte feeds one
 * multiply; CSE the load and both operands become plain ints. On
 * ARM926EJ-S SMULxy/SMLAxy are one cycle and MUL/MLA are two to five
 * (early termination on Rs, a 16-bit coefficient costing three). The
 * loop goes from about 16 cycles to about 22 - two instructions FEWER
 * and six cycles MORE.
 *
 * The other two shapes fail differently: putting kp inside keeps each
 * lane's accumulator in a register but re-loads the four coefficients
 * per lane instead of once per 32, and at two lanes per block gcc
 * spilled the accumulators anyway (16.5 per lane); unrolling kp by two
 * needs eight coefficients live and spills for the same reason.
 *
 * So the 16 stands until somebody writes the assembly the audit's D2
 * row describes - LDM for four bytes at once, SMLABB/SMLATT halfword
 * selection, accumulators in registers - which can have BOTH. From C
 * the compiler will give one or the other.
 *
 * This is the one place in this file where the project's usual metric
 * lies: static instruction count transfers across machines and cycle
 * counts do not, which is why everything else is counted rather than
 * timed - but the multiply family's cycle difference is an ISA fact,
 * documented, and large enough to invert the ranking. */
static void gdn_sum_gg_ref(const int8_t *sq_gg, int kd, int vd,
                           const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                           int32_t *au_gg, int32_t *aw_gg) {
    int kpairs = kd / 2, kp, j;
    for (j = 0; j < 32; j++) { au_gg[j] = 0; aw_gg[j] = 0; }
    for (kp = 0; kp < kpairs; kp++) {
        const int8_t *rowA = sq_gg + (size_t)(2 * kp) * vd;
        const int8_t *rowB = sq_gg + (size_t)(2 * kp + 1) * vd;
        int32_t ckA = tab[kp][0], ckB = tab[kp][1];
        int32_t cqA = tab[kp][4], cqB = tab[kp][5];
        for (j = 0; j < 32; j++) {
            au_gg[j] += (int32_t)rowA[j] * ckA + (int32_t)rowB[j] * ckB;
            aw_gg[j] += (int32_t)rowA[j] * cqA + (int32_t)rowB[j] * cqB;
        }
    }
    gdn_tail_row(sq_gg, kd, vd, tcu, tcw, au_gg, aw_gg);
}

#if defined(LZ_ARM_ASM_EXTERN)
/* ARM dispatch. --kernel arm-asm reaches the hand-written 4-lane kernel
   (lz_gdn1_x4_arm_asm), arm-c/ref the scalar above; the two are the
   control arm and the arm this column's cell is for. Same eight
   sub-window + out8 round-trip structure as the gcc MMX branch below,
   minus the emms (ARM has no MMX state to clear). */
static void gdn_sum_gg(const int8_t *sq_gg, int kd, int vd,
                       const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                       int32_t *au_gg, int32_t *aw_gg) {
    int kpairs = kd / 2, vb;
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        for (vb = 0; vb < 8; vb++) {
            int v = vb * 4;
            int32_t out8[8];
            lz_gdn1_x4_arm_asm(sq_gg + v, vd, &tab[0][0], kpairs, out8);
            au_gg[v + 0] = out8[0]; au_gg[v + 1] = out8[1];
            au_gg[v + 2] = out8[2]; au_gg[v + 3] = out8[3];
            aw_gg[v + 0] = out8[4]; aw_gg[v + 1] = out8[5];
            aw_gg[v + 2] = out8[6]; aw_gg[v + 3] = out8[7];
        }
        gdn_tail_row(sq_gg, kd, vd, tcu, tcw, au_gg, aw_gg);
        return;
    }
    gdn_sum_gg_ref(sq_gg, kd, vd, tab, tcu, tcw, au_gg, aw_gg);
}
#else
static void gdn_sum_gg(const int8_t *sq_gg, int kd, int vd,
                       const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                       int32_t *au_gg, int32_t *aw_gg) {
    gdn_sum_gg_ref(sq_gg, kd, vd, tab, tcu, tcw, au_gg, aw_gg);
}
#endif /* LZ_ARM_ASM_EXTERN */
#elif defined(__WATCOMC__)
/* Watcom: one call to src/ops_mmx.c's lz_gdn_sum_gg_mmx, which does the
   eight sub-window lz_gdn1_x4_asm calls internally with the pragma
   genuinely inline - the same call count this function has from its
   own caller, just crossing a TU boundary. */
static void gdn_sum_gg(const int8_t *sq_gg, int kd, int vd,
                       const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                       int32_t *au_gg, int32_t *aw_gg) {
    lz_gdn_sum_gg_mmx(sq_gg, kd, vd, tab, tcu, tcw, au_gg, aw_gg);
}
#else
/* Eight 4-lane sub-windows cover this gg's 32 vv lanes.

   The emms belongs HERE, once per gg - not per sub-window, and not after
   the whole loop. Hoisting the emms out is allowed only when the loop
   body has no x87, and that precondition fails one level up: the
   next gg's gdn_build_table runs q8_round and 32767/amax on x87 right
   after this returns. Hoisting it would reopen the trap that broke the
   mmx tier. */
static void gdn_sum_gg(const int8_t *sq_gg, int kd, int vd,
                       const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                       int32_t *au_gg, int32_t *aw_gg) {
    int kpairs = kd / 2, vb;
    for (vb = 0; vb < 8; vb++) {
        int v = vb * 4;
        int32_t out8[8];
        lz_gdn1_x4_asm(sq_gg + v, vd, &tab[0][0], kpairs, out8);
        au_gg[v + 0] = out8[0]; au_gg[v + 1] = out8[1];
        au_gg[v + 2] = out8[2]; au_gg[v + 3] = out8[3];
        aw_gg[v + 0] = out8[4]; aw_gg[v + 1] = out8[5];
        aw_gg[v + 2] = out8[6]; aw_gg[v + 3] = out8[7];
    }
    gdn_tail_row(sq_gg, kd, vd, tcu, tcw, au_gg, aw_gg);
    if (lz_cpu_has_mmx()) _mm_empty();
}
#endif /* LZ_GDN_HAVE_MMX */

/* au/aw come back as scaled-up int32 with su/sw the matching dequant
   scale. The accumulator bound is 127*32767*256 ~= 1.06e9, far past 2^24,
   so the caller must convert with lz_i32f(). */
static void gdn_pass1_fixed(const int8_t *sq,
#if LZ_GDN_STATE_2PLANE
                            const int8_t *sq_lo, int32_t *aul, int32_t *awl,
#endif /* LZ_GDN_STATE_2PLANE */
                            const float *ss,
                            const float *q, const float *k, int kd, int vd,
                            int32_t *au, int32_t *aw, float *su, float *sw,
                            int32_t *au2, int32_t *aw2, float *su2, float *sw2) {
    int gg, ng = vd / 32;
    /* static, not stack: 2-4 KB of table on a Win98 stack. */
    static LZGdnTable tb;
#if LZ_GDN_FIXED < 2
    (void)au2; (void)aw2; (void)su2; (void)sw2;
#endif /* LZ_GDN_FIXED < 2 */
    for (gg = 0; gg < ng; gg++) {
        const int8_t *sg = sq + (size_t)gg * 32;
        gdn_build_table(ss, q, k, kd, vd, gg, &tb);
        gdn_sum_gg(sg, kd, vd, tb.t, tb.tail_cu, tb.tail_cw,
                   au + gg * 32, aw + gg * 32);
        su[gg] = tb.su; sw[gg] = tb.sw;
#if LZ_GDN_STATE_2PLANE
        /* Low state plane: same kernel, same coefficient table, different
           state plane. Its dequant weight is a constant 1/254 relative to
           the high plane, so no second scale array is needed. */
        gdn_sum_gg(sq_lo + (size_t)gg * 32, kd, vd, tb.t, tb.tail_cu, tb.tail_cw,
                   aul + gg * 32, awl + gg * 32);
#endif /* LZ_GDN_STATE_2PLANE */
#if LZ_GDN_FIXED >= 2
        /* Second plane, same kernel, second table. The MMX gdn_sum_gg emits
           its own emms, so this runs a redundant one - kept rather than
           hoisted, because the emms-hoisting precondition ("no x87 in the
           loop body") fails here: gdn_build_table for the NEXT gg is x87.
           One extra tag-word write per gg is not worth reopening that trap. */
        gdn_sum_gg(sg, kd, vd, tb.t2, tb.tail2_cu, tb.tail2_cw,
                   au2 + gg * 32, aw2 + gg * 32);
        su2[gg] = tb.su2; sw2[gg] = tb.sw2;
#endif /* LZ_GDN_FIXED >= 2 */
    }
}
#endif /* LZ_GDN_FIXED */

/* ---- measurement scaffolding: exact float SSM state --------------------

   Never in a shipping build. `-DLZ_GDN_STATE_F32=1` makes the recurrence
   keep an exact float state instead of the int8 one, so the question
   "how far above an exact recurrence does this engine already sit, before
   pass 1's coefficients are quantized at all" is measurable rather than
   argued from amax/127.

   Keyed by the int8 state pointer the caller passes, which is 1:1 with
   (layer, head), so no signature change and no forward.c change - the
   scaffolding cannot perturb the shipping path by construction. The
   linear scan is O(layers*heads) per call and irrelevant at these
   sizes. */
#ifndef LZ_GDN_STATE_F32
#define LZ_GDN_STATE_F32 0
#endif /* LZ_GDN_STATE_F32 */

#if LZ_GDN_STATE_2PLANE
#if LZ_GDN_FIXED >= 2
#error "LZ_GDN_STATE_2PLANE with LZ_GDN_FIXED=2 is not implemented: it would be four kernel passes, and the coefficient plane was measured not to be the bottleneck. Use LZ_GDN_FIXED=0 or 1."
#endif /* LZ_GDN_FIXED >= 2 */

/* LZ_GDN_LO_SCALE now lives in ops.h - forward.c's KV path needs the same
   constant (LZ_KV_2PLANE), and two copies of a dequant weight is exactly
   the kind of duplicate that drifts. */

/* Two-plane quantize of one state row. Deliberately mirrors
   lz_quantize_q8: same integer-domain absmax, same amax*(1/127) scale
   written as a multiply, same q8_round, and the SAME
   three rounding tiers. `hi` alone therefore matches what
   lz_quantize_q8 would have written, which makes
   -DLZ_GDN_STATE_2PLANE=0 vs 1 a clean A/B rather than two unrelated
   quantizers.

   Both planes go through the vectorized rounding kernel, the low one by
   feeding it the residual with a multiplier of 254 - the kernel computes
   round(x*inv) and does not care what x means. Rounding both planes in
   scalar code instead would cost +51.8% per token on s1v3: pass 2's
   quantize is the dominant term in lz_gdn_step, and the SSE2 rounding
   tier is ~25x the scalar magic-number path, so dropping to scalar
   there would swamp everything the two planes buy. What is left scalar
   is only the residual itself (one multiply, one int8->float, one
   subtract per element).

   `fc_site`: which census site to bill this call's f32 work to, or -1
   to bill nothing. Billed here, not at the call site: the per-element
   cost depends on q8_group_scale's `fast` flag and on lz_q8r_tier(),
   both internal to this loop, and duplicating that dispatch at a caller
   is a second copy of it that will drift. */
void lz_gdn_quantize_2p(const float *x, int n, int gs,
                        int8_t *hi, int8_t *lo, float *s, int fc_site) {
    int g, k, ng = n / gs, fast;
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
    int tier = ((gs & 31) == 0) ? lz_q8r_tier() : 0;
#ifdef LZ_Q8R_COUNT
    /* Which tier this call actually took, counted rather than inferred.
       Off unless a probe defines the macro.
       Exists because "the dispatch condition looks right" is not an
       answer to "which one ran" - the same distinction that let a whole
       tier of hand-written assembly stay out of the Watcom build for
       months. */
    extern long lz_q8r_hits[3];
    lz_q8r_hits[tier < 0 ? 0 : (tier > 2 ? 2 : tier)]++;
#endif /* LZ_Q8R_COUNT */
    static const float lo_mul = LZ_GDN_LO_SCALE;
    /* static, not stack; one row's residual at most. */
    static float res[LZ_GDN_MAX_VD];
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */
    for (g = 0; g < ng; g++) {
        const float *grp = x + (size_t)g * gs;
        int8_t *ho = hi + (size_t)g * gs;
        int8_t *lw = lo + (size_t)g * gs;
        float inv;
        fast = q8_group_scale(grp, gs, &s[g], &inv);
        /* Paid once per group regardless of which branch below runs:
           amax*(1/127) mul, 127/amax div, and q8_group_scale's three
           float comparisons (see lz_quantize_q8's identical bill in
           ops.c for which three, and for why the amax scan is not one
           of them). */
        if ((unsigned)fc_site < (unsigned)LZ_FC_N) LZ_FCX(fc_site, 1, 0, 1, 0, 3);
        if (!fast) {
            /* Same shape as the fast&&tier==0 branch below (that one
               multiplies instead of dividing), same count. */
            if ((unsigned)fc_site < (unsigned)LZ_FC_N) LZ_FCX(fc_site, gs, 3 * gs, gs, gs, 0);  /* div, q8_round(add), convert, sub(add), mul, q8_round(add) */
            for (k = 0; k < gs; k++) {
                float f = grp[k] / s[g];
                int h = q8_round(f);
                int l;
                if (h >  127) h =  127;
                if (h < -127) h = -127;
                l = q8_round((f - (float)h) * LZ_GDN_LO_SCALE);
                /* +-127, NOT +-128, even though int8 holds -128 and L =
                   256 can produce it. The SIMD rounding kernel clamps at
                   +-127 (pminsw/pmaxsw against 127/-127) and cannot be
                   told otherwise, so allowing -128 here would make the
                   scalar and SIMD tiers disagree - the quality probe
                   catches that as the FLOAT path getting 2.2x worse
                   when it should improve. */
                if (l >  127) l =  127;
                if (l < -127) l = -127;
                ho[k] = (int8_t)h;
                lw[k] = (int8_t)l;
            }
            continue;
        }
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
        if (tier) {
            /* The lz_q8round32_simd/_sse calls bracketing the visible
               residual line below are not free - reading only the C
               between them would miss 4 of the 7. */
            if ((unsigned)fc_site < (unsigned)LZ_FC_N) LZ_FCX(fc_site, 3 * gs, gs, 0, 3 * gs, 0);  /* residual mul+convert+sub=3, 2x kernel mul+convert=4 */
#if defined(__WATCOMC__)
            /* One call per GROUP for the WHOLE two-plane round (both
               kernel loops and the residual arithmetic between them),
               not one per 32-element sub-chunk - same reasoning as
               lz_quantize_q8 above. lz_q8round_2p_group_sse2/_sse own
               the residual computation too (plain float arithmetic,
               safe in a Watcom #pragma-aux TU the way it is not in a
               gcc -mmmx one - see src/ops_mmx.c's header for why that
               split is toolchain-specific), so the hand-off is still
               one call, not "kernel, kernel, residual" three. */
#ifdef LZ_HAVE_Q8R_SIMD
            if (tier == 2)
                lz_q8round_2p_group_sse2(grp, ho, lw, gs, &inv, &lo_mul, res);
            else
#endif /* LZ_HAVE_Q8R_SIMD */
                lz_q8round_2p_group_sse(grp, ho, lw, gs, &inv, &lo_mul, res);
#else
#ifdef LZ_HAVE_Q8R_SIMD
            if (tier == 2) {
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_simd(grp + k, ho + k, &inv);
                for (k = 0; k < gs; k++)
                    res[k] = grp[k] * inv - (float)ho[k];
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_simd(res + k, lw + k, &lo_mul);
            } else
#endif /* LZ_HAVE_Q8R_SIMD */
            {
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_sse(grp + k, ho + k, &inv);
                /* emms before the x87 residual arithmetic below, and
                   again before the next group's amax/inv - same reason as
                   the identical placement in lz_quantize_q8: the loop
                   body contains x87, so the hoist that works elsewhere
                   does not apply here. lz_q8round32_sse uses pmaxsw,
                   which is SSE1 on an MMX REGISTER, so this tier leaves
                   MMX state behind exactly as the MMX one does. */
                if (lz_cpu_has_mmx()) _mm_empty();
                for (k = 0; k < gs; k++)
                    res[k] = grp[k] * inv - (float)ho[k];
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_sse(res + k, lw + k, &lo_mul);
                if (lz_cpu_has_mmx()) _mm_empty();
            }
#endif /* __WATCOMC__ */
            continue;
        }
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */
        /* fast && tier==0: no SIMD rounding tier, e.g. --kernel ref. Same
           shape as the !fast branch above, multiply instead of divide. */
        if ((unsigned)fc_site < (unsigned)LZ_FC_N) LZ_FCX(fc_site, 2 * gs, 3 * gs, 0, gs, 0);  /* mul, q8_round(add), convert, sub(add), mul, q8_round(add) */
        for (k = 0; k < gs; k++) {
            float f = grp[k] * inv;
            int h = q8_round(f);
            int l;
            if (h >  127) h =  127;
            if (h < -127) h = -127;
            l = q8_round((f - (float)h) * LZ_GDN_LO_SCALE);
            if (l >  127) l =  127;
            if (l < -127) l = -127;
            ho[k] = (int8_t)h;
            lw[k] = (int8_t)l;
        }
    }
}
#endif /* LZ_GDN_STATE_2PLANE */

#if LZ_GDN_STATE_F32
#define LZ_GDN_SHADOW_MAX 1024
static const int8_t *g_shk[LZ_GDN_SHADOW_MAX];
static float *g_shv[LZ_GDN_SHADOW_MAX];
static int g_shn;
static float *gdn_shadow(const int8_t *sq, int n) {
    int i;
    for (i = 0; i < g_shn; i++) if (g_shk[i] == sq) return g_shv[i];
    if (g_shn >= LZ_GDN_SHADOW_MAX) return NULL;
    g_shv[g_shn] = (float *)calloc((size_t)n, sizeof(float));
    if (!g_shv[g_shn]) return NULL;
    g_shk[g_shn] = sq;
    return g_shv[g_shn++];
}
#endif /* LZ_GDN_STATE_F32 */

/* ---- SSM recurrence pass 2, fixed point --------------------------------

   The counterpart to gdn_pass1_fixed, and the reason it exists is a
   measurement rather than symmetry: on the all-scalar tier - which is
   the ONLY tier a Pentium II has, since lz_q8round32 structurally
   cannot have an MMX one - the write-back quantize is 83% of
   lz_gdn_step and pass 2's arithmetic a further 11% (Watcom). Both
   are stuck on x87 for one reason: pass 2 hands the
   quantizer a FLOAT. Producing int32 instead turns the quantize into
   int32 -> int8, which is packssdw/packsswb - pure MMX. The blocker is
   not the instruction set, it is the type at the boundary.

   TWO FACTS MAKE THIS SMALL, and both are worth stating because the
   first design draft treated the opposite as make-or-break:

   1. THE TWO-PLANE STATE IS ALREADY AN int16 STATE. With
      H = hi*256 + lo the value is exactly H*(s/256), and
      |H| <= 127*256+127 = 32639 < 32767. So there
      are no "two planes" in the integer path: pack on the way in, split
      on the way out, and the split is EXACT and division-free
      (hi = (H+128)>>8, lo = H - hi*256) where the float
      path rounds twice.

   2. THE SCALE ARITHMETIC STAYS IN FLOAT. The 83% is per ELEMENT
      (kd*vd); s_acc, the multipliers and the reciprocal are per GROUP
      (kd*ng, one thirty-second as many). Leaving those on x87 costs
      nothing measurable and removes the whole problem of aligning two
      fixed-point exponents. Same ratio argument that makes "integer
      contraction, float epilogue" right for the matmul kernels.

   Per element what remains is A = m1*H + m2*Dq, which is exactly what
   pmaddwd computes - one instruction for the whole accumulate.

   THE 32-BIT TRAP: A reaches 2.1e9, so the natural A*32258/amax rescale
   is 3.5e13 and
   overflows. `long` is 32-bit on BOTH Watcom and MinGW (LLP64), so this
   is not a host-only hazard. amax is shifted into 16 bits first, which
   keeps exactly the 15 bits H needs and is also the shape the MMX
   kernel wants (a per-group int16 reciprocal for pmulhw).

   AND BOTH SHIFTS ROUND. An arithmetic shift right truncates toward
   -inf, which is a BIAS; bias adds linearly where quantization noise
   adds in quadrature. Measured on the way here: 5.88e-05 with the two
   shifts truncating, 4.16e-05 with them rounded, against a float pass 2
   at 1.78e-05.

   ACCURACY, stated plainly: "no worse than the float path" is
   structurally unachievable - the float path's only error source is
   the final
   quantization, and this path has that PLUS coefficient quantization
   PLUS delta quantization PLUS the rescale. Measured added error is
   ~3.8e-05, about 2x the float path's own. For comparison the pass-1
   fixed tier that ships by default adds ~1.5e-05 to the same baseline.
   That is why this tier is OFF by default and gated on an end-to-end
   paired-NLL measurement, not on this operator-level number.

   Bit-identity across builds is free HERE and not by luck: everything
   per element is integer, and integer sums are exact under
   reassociation, so a future MMX twin agrees with this scalar path by
   construction - the same argument gdn_pass1_fixed's own comment makes.
   The float lines write every constant divisor as
   a multiply by its reciprocal; the one real division has a variable
   divisor, which both compilers emit as a true divide). */

/* expof and norm_ss_fixed are in ops_quant.c. */

#if LZ_GDN_FIXED
/* THE SPLIT IS A SHIFT, AND THE SHIFT IS 8 BECAUSE L IS 256. Every
   `>> 8`, `<< 8` and `* 256` in the fixed pass 2 is that one constant
   written three ways; ops.h's LZ_GDN_LO_SCALE is the same number a
   fourth time, in float. Building with -DLZ_GDN_LO_SCALE=254.0f - the
   A/B control ops.h advertises - would leave all of them behind and
   compute a wrong answer silently, so it is now a build error. */
typedef char lz_p2_needs_lo_scale_256[
    ((int)(LZ_GDN_LO_SCALE) == 256) ? 1 : -1];
/* The same constant a fifth time, as the exponent. p2_group_scale folds
   it into a pow2f argument, which is only legal because the assert above
   pins the base at 2. */
#define LZ_GDN_LO_SHIFT 8

#include "ops_kernel_p2.h"        /* the MMX pair; defines lz_p2_blk and,
                                     on Watcom, LZ_HAVE_P2_MMX_ASM */

/* p2_blk() is in src/ops_mmx.c, declared unconditionally in
   src/ops_mmx.h - both Watcom's lz_p2_rows_mmx/lz_p2_rows_sse and
   gcc's LZ_P2_MMX_EXTERN case need the SAME 16-byte-aligned scratch
   buffer and reach across a TU boundary to the code that fills it, so
   it has to be one real exported function rather than a static one the
   #pragma aux bodies could reach for free. */

#if defined(LZ_HAVE_P2_MMX_ASM) || defined(LZ_P2_MMX_EXTERN)
#define LZ_HAVE_P2_MMX 1
/* No lz_p2_mul32_mmx/lz_p2_split32_mmx bare-name aliases in this file:
   the #pragma aux bodies live in src/ops_mmx.c (declared in
   src/ops_mmx.h for both toolchains), and Watcom's Phase 2 below calls
   lz_p2_rows_mmx/_sse/_sse2 directly. The bare per-group names are
   gcc's alone, used only by the gcc branch of gdn_p2_row_simd's
   Phase 2. */
#endif /* LZ_HAVE_P2_MMX_ASM || LZ_P2_MMX_EXTERN */

/* ---- the SSE1 tier: one instruction of content, and it is the split ---
   The A term and the absmax have no SSE1 form at all (ops_kernel_p2.h
   enumerates SSE1's whole addition to the MMX register file against what
   this kernel needs), so this tier is the MMX mul32 paired with a split
   whose low-plane clamp is a single pmaxsw. Named for the ISA a machine
   must HAVE to run it, which is what dispatch turns on. */
#if defined(LZ_HAVE_P2_SSE_ASM) || defined(LZ_P2_MMX_EXTERN)
#define LZ_HAVE_P2_SSE 1
/* No lz_p2_split32_sse bare-name alias in this file - its body lives
   in src/ops_mmx_sse.c for BOTH toolchains, named "sse" for the ISA a
   machine must HAVE to run it even though it still operates on %mm
   (SSE1's whole addition to the integer side of the MMX register file
   is used here exactly once: pmaxsw does the low plane's clamp to -127
   directly). Watcom's Phase 2 reaches it only through lz_p2_rows_sse,
   declared in src/ops_mmx.h; gcc's per-group loop calls it by its real
   name, also declared there. */
#endif /* LZ_HAVE_P2_SSE_ASM || LZ_P2_MMX_EXTERN */

/* ---- the 128-bit tier -------------------------------------------------
   Every instruction of the MMX pair exists on xmm in SSE2 - pmaddwd,
   psrad, packssdw, packsswb, the pcmpgtd/pand/pandn/por max tree, the
   saturating pair that clamps the low plane - so this is the same kernel
   twice as wide, not a different algorithm. It is a real tier rather
   than a rewrite of an existing one, and the machines it is for are the
   P4 Northwood and the Pentium M.

   NOT pmaxsd: that is SSE4.1. On SSE2 an int32 maximum still costs the
   four-instruction mask dance, so the absmax stays the expensive half
   here exactly as it is in MMX.

   Alignment is now load-bearing: movdqa and the m128 shift-count operand
   FAULT on a misaligned address, which is why p2_blk() aligns by hand.
   The caller's state planes get movdqu/movq, since their alignment is
   not ours to assume. */
#if defined(LZ_HAVE_P2_SSE2_ASM) || defined(LZ_P2_SSE2_EXTERN)
#define LZ_HAVE_P2_SSE2 1
/* One flag, two triggers: LZ_HAVE_P2_SSE2_ASM means Watcom's #pragma
   aux SSE2 pair is in the build, LZ_P2_SSE2_EXTERN means gcc's
   extern-linked intrinsics are - different code for the same operator
   on different compilers, both bodies in src/ops_sse2.c. No bare-name
   aliases here (same reasoning as the MMX and SSE1 tiers above):
   Watcom's Phase 2 reaches this tier only through lz_p2_rows_sse2,
   declared in src/ops_sse2.h alongside the gcc-only per-group names,
   which the intrinsics' declarations share. */
#endif /* LZ_HAVE_P2_SSE2_ASM || LZ_P2_SSE2_EXTERN */

#if defined(LZ_HAVE_P2_SSE2) && !defined(LZ_HAVE_P2_MMX)
#error "SSE2 pass 2 without the MMX one: gdn_p2_row_simd is gated on MMX"
#endif /* LZ_HAVE_P2_SSE2 && !LZ_HAVE_P2_MMX */

#ifdef LZ_HAVE_P2_MMX
/* 0 scalar, 1 MMX, 2 SSE2. Simpler than lz_q8r_tier for
   one reason: this operator's tiers are pure integer SIMD, so they line
   up exactly with the kernel tier g_kernel already carries - unlike the
   q8 rounding, where a PIII has 64-bit integer SIMD but 128-bit float
   SSE and the two axes come apart.

   MMX is the floor of the whole target family (the machine table has a
   check in the MMX column on every row), so a machine that lands on
   scalar here is one where kernel_detect already said LZ_KERNEL_REF.
   No FORCE knob: unlike the SSE1 tiers, both of these are selected by
   default on some machine that runs the suite, so both are validated by
   every run rather than only by a probe. */
static int p2_tier(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return 0;
#ifdef LZ_HAVE_P2_SSE2
    if (g_kernel == LZ_KERNEL_SSE2) return 3;
#endif /* LZ_HAVE_P2_SSE2 */
    /* SSE1 comes from g_kernel now that LZ_KERNEL_SSE exists. It used to
       come from CPUID, which meant --kernel mmx ran this arm on every
       machine in the suite and the pure MMX split below was unreachable
       - the state LZ_P2_FORCE_MMX was a separate build to escape. The
       tiers are bit-identical, so nothing reported the substitution.

       kernel_detect resolves a PIII to LZ_KERNEL_SSE, so the machines
       this tier exists for still get it from auto. */
#if defined(LZ_HAVE_P2_SSE)
    if (g_kernel == LZ_KERNEL_SSE) return 2;
#endif /* LZ_HAVE_P2_SSE */
    return 1;
}
#endif /* LZ_HAVE_P2_MMX */

/* --- the per-group float and integer work, shared by both tiers -------

   These four exist as functions rather than as lines inside the scalar
   loop for one reason: the SIMD twin below needs every one of them, and
   a threshold computed in two places is a threshold that will end up
   with two values. What differs between the tiers is the 32-element
   integer body and nothing else. */

/* c1 weights the decayed old state, c2 the new delta term. ONE copy of
   these two expressions in the engine - the scalar and SIMD rows both
   call this - because they are the only float arithmetic left that is
   not exact, so two textually-equal copies could still differ by which
   intermediate the compiler chose to keep in an x87 register. */

/* TAKES decs = dec * 2^-8, NOT dec. The 1/LZ_GDN_LO_SCALE factor is a
   property of the ROW (dec is per row, the constant is per build), so
   paying it per GROUP was ng multiplies where one would do. Both callers
   now scale once above their group loop.

   EXACT, not close, and for the same reason lz_attn_score_q8's 2^-8 fold
   is: LZ_GDN_LO_SCALE is 256, so c = 2^-8 only moves an exponent.
   fl(x*c) is fl(x)*c for any x, so fl(fl(dec*rs)*c) - scaling per
   element - and fl(fl(dec*c)*rs) - scaling once above the loop, which
   is what this does - both equal fl(dec*rs*c), the single
   rounding of the same real product. A static assert two hundred lines
   up already pins LZ_GDN_LO_SCALE at 256; if it ever stops being a power
   of two this argument goes with it.

   The one operand that breaks it is one the scaling pushes out of the
   normal range: dec*2^-8 goes subnormal below dec = 2^-118. dec is a
   decay gate, so that is a gate whose state contribution is already
   zero to 24 bits - but it is a bound, not an impossibility, which is
   why the end-to-end byte comparison is what this change was accepted
   on and not the algebra alone. */
static void p2_group_coef(float decs, float rs, float kv, float sd,
                          float *c1, float *c2) {
    /* rs IS THE STATE'S GROUP SCALE, which p2_group_scale writes as
       pow2f(ee) - so c1 is an exponent add, free and bit-identical.
       Its three other arms (out-of-range, all-zero, and the float
       recurrence's q8_group_scale scale) are not powers of two and keep
       the multiply, which is why the guard is here rather than assumed
       away.
       c2 stays a multiply here: sd is the DELTA group scale and this
       function does not know whether it was block-normalized. Its
       caller's own site folds it when it was. */
    union { float f; uint32_t u; } rb;
    rb.f = rs;
    if (((rb.u & 0x007FFFFFu) != 0u)
        || (uint32_t)(((rb.u >> 23) & 0xFFu) - 1u) >= 0xFEu
        || !f32_scale_pow2(decs, (int)((rb.u >> 23) & 0xFFu) - 127, c1))
        *c1 = decs * rs;
    *c2 = kv * sd;
}

/* The float prologue, reduced to what the integer body can consume: two
   int16 multipliers and the power of two they were normalized against.
   Returns 0 for a group that flushes to zero.

   Why the normalization is by powers of two only, and why that is the
   whole reason this tier survives cross-compiler comparison, is written
   out at the top of gdn_p2_group_fixed's shift section below.

   THE +-32767 CLAMP IS NEW AND IT MOVES NUMBERS, so it is stated rather
   than buried. Normalization puts |t| in [2^14, 2^15), and q8_round of a
   t at or above 32767.5 returns 32768 - one past int16. The scalar path
   held that in an int32 and multiplied happily; pmaddwd cannot, and
   32768 truncated to int16 is -32768, a sign flip on the dominant
   coefficient. So BOTH paths clamp, and the fixed tier moves by one ULP
   of its own multiplier (3e-5 relative) on the ~1.5e-5 of groups that
   land in that last half-step. -32768 would not need clamping, but the
   bound is kept symmetric: a one-sided bound reads like an oversight and
   gets "fixed" in the wrong direction later. */
/* esac_out is the EXPONENT sacc was built from, handed out alongside the
   float so p2_group_scale can fold two power-of-two factors into one
   pow2f instead of multiplying them. Kept as a second output rather than
   replacing sacc: the flush case signals itself through sacc == 0, and
   an exponent has no spare value to mean that. */
static int p2_group_norm(float c1, float c2, int16_t *mul4, float *sacc,
                         int *esac_out) {
    float cm1 = (c1 < 0.0f) ? -c1 : c1;
    float cm2 = (c2 < 0.0f) ? -c2 : c2;
    float cmax = (cm1 > cm2) ? cm1 : cm2;
    float p, t1, t2;
    int e, m1, m2;

    /* FLUSH BEFORE NORMALIZING, and this is a correctness guard rather
       than a shortcut. If cmax is small enough that the accumulator
       scale lands in the subnormal range, x87 and SSE stop agreeing
       about it - PC=24 fixes the mantissa at 24 bits but leaves the
       EXPONENT range extended, so a value held in an x87 register is
       normal where its float32 store is subnormal, and whether it gets
       stored is a register-allocation decision. Not theoretical: without
       this guard the two builds diverge at recurrence step 26, and
       compiling the tracing probe - whose only effect is to STORE these
       intermediates - makes the divergence disappear. A bug that hides
       when you look at it is this mechanism, every time.

       Same constant as lz_quantize_q8 (LZ_Q8_MIN_SCALE, ops.h): two
       operators disagreeing about where zero starts would disagree about
       results. A group with cmax below 1e-30 contributes |term| <=
       3e-26 next to activations of order 1. */
    if (!(cmax >= LZ_Q8_MIN_SCALE_F)) {
        mul4[0] = mul4[1] = mul4[2] = mul4[3] = 0;
        *sacc = 0.0f; *esac_out = 0;
        return 0;
    }
    e  = expof(cmax);               /* cmax >= LZ_Q8_MIN_SCALE, so normal */
    /* Exact: multiplying by a power of two only moves the exponent, so
       f32_scale_pow2 (ops_kernel_shared.h) does it as an integer add
       and the __aeabi_fmul goes away - twice per group, in pass 2's
       per-row block, which is the second-largest piece of the census's
       largest row. It declines a zero/subnormal c and an out-of-range
       result; pow2f keeps those, so the pair below is the same number
       either way and the fallback is not dead code (c1 or c2 being
       exactly zero is ordinary). */
    if (!f32_scale_pow2(c1, 14 - e, &t1) ||
        !f32_scale_pow2(c2, 14 - e, &t2)) {
        p  = pow2f(14 - e);         /* exact */
        t1 = c1 * p;
        t2 = c2 * p;
    }
    *esac_out = e - 14;
    *sacc = pow2f(e - 14);          /* exact reciprocal of p */
    m1 = (t1 > -0.5f && t1 < 0.5f) ? 0 : q8_round(t1);
    m2 = (t2 > -0.5f && t2 < 0.5f) ? 0 : q8_round(t2);
    if (m1 >  32767) m1 =  32767;
    if (m1 < -32767) m1 = -32767;
    if (m2 >  32767) m2 =  32767;
    if (m2 < -32767) m2 = -32767;
    mul4[0] = (int16_t)m1;
    mul4[1] = (int16_t)m2;
    mul4[2] = (int16_t)m1;         /* the pair, twice: pmaddwd reads two */
    mul4[3] = (int16_t)m2;         /* dwords per register */
    return 1;
}

/* p2_shift_of: defined in src/ops_kernel_shared.h - the "shift not
   divide" rationale lives there, alongside the SIMD tiers
   (src/ops_mmx.c, src/ops_mmx_sse.c, src/ops_sse2.c) that need the
   exact same rescale shift. */

/* The float epilogue. s_new stays exact: sacc is already a power of two
   from the normalization, so multiplying by 2^sh and by L introduces no
   rounding of its own. Returns 0 when the group must be written as an
   exact zero - either the prologue flushed it (sacc == 0) or the new
   scale itself falls under the floor.

   THE TWO POWER-OF-TWO MULTIPLIES ARE FUSED, and the thing that makes
   that legal is a BOUND, not the algebra. pow2f clamps at both ends, so
   pow2f(sh) * 256 and pow2f(sh + 8) disagree exactly where the clamp
   bites: at sh in [120, 127] the first overflows to inf and the second
   returns 2^127, and at sh in [-134, -127] the first is 0 and the second
   is a normal number.

   Neither region is reachable. Both call sites pass sh straight from
   p2_shift_of, whose whole body is `while ((amax >> sh) > 32512) sh++`
   on an int32 - so sh is in [0, 31] and sh + 8 in [8, 39]. Away from
   the clamps, multiplying by a power of two only moves an exponent, so
   the fused form is the same float and not merely a close one.

   (An earlier version of this comment said the bound did not exist. It
   does; I had not read p2_shift_of.)

   THE REMAINING MULTIPLY GOES THE SAME WAY, but it needs a GUARD rather
   than a bound, because the exponent it folds comes from live data.
   sacc is pow2f(esac) by construction, so sacc * pow2f(sh + 8) is
   2^(esac + sh + 8) - one pow2f, no multiply - EXCEPT where pow2f
   clamps. The guard is epi_fixed's, one file over: take the fused form
   only inside pow2f's normal range and fall back to the ORIGINAL
   expression outside it, so the two are bit-identical everywhere by
   construction rather than by an argument about reachability.

   The two edges are already closed on the other side: esac < -126 makes
   pow2f return 0, and sacc == 0 is the flush case that returns above;
   esac > 127 makes pow2f clamp, and then ee > 127 takes the fallback.
   So inside the fused branch both esac and ee are in range and the
   identity is exact. */
static int p2_group_scale(float sacc, int esac, int sh, float *os) {
    float s_new;
    int ee;
    if (!(sacc > 0.0f)) { *os = 1.0f; return 0; }
    ee = esac + sh + LZ_GDN_LO_SHIFT;
    if (ee < -126 || ee > 127) s_new = sacc * pow2f(sh + LZ_GDN_LO_SHIFT);
    else                       s_new = pow2f(ee);
    if (!(s_new >= LZ_Q8_MIN_SCALE_F)) { *os = 1.0f; return 0; }
    *os = s_new;
    return 1;
}

/* The whole fixed pass 2 for one head, shared by lz_gdn_step and
   lz_kda_step. THIS WRAPPER IS THE UNIT THAT KEEPS GETTING FORGOTTEN,
   which is why it is a function rather than two copies.

   Omission is silent - the tier simply does nothing on the family that
   lacks it, both settings produce the same output, and it reads as
   "this tier is not worth much". KunMoe is 0 gdn / 6 kda / 2 full, so
   forgetting to wire pass 1 or pass 2 fixed into a new family is the
   only way it goes missing.

   `gvec` NULL means the scalar-decay family (GDN, one gt for the head);
   non-NULL means one decay per row (KDA). That is the ONLY difference
   between the two families here - everything else, including the
   per-token delta quantization, is character-for-character identical. */
static void gdn_p2_fixed_rows(const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                              const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                              const float *ss_in, float *ss_out,
                              const float *delta, const float *k,
                              const float *gvec, float gt,
                              int kd, int vd);

static void gdn_p2_group_fixed(const int8_t *ph,
#if LZ_GDN_STATE_2PLANE
                               const int8_t *pl,
#endif /* LZ_GDN_STATE_2PLANE */
                               float c1, const int16_t *dq, float c2,
                               int8_t *oh,
#if LZ_GDN_STATE_2PLANE
                               int8_t *ol,
#endif /* LZ_GDN_STATE_2PLANE */
                               float *os) {
    int32_t A[32];
    int16_t mul4[4];
    int32_t amax = 0;
    float sacc;
    int esac;
    int j, sh = 0;

    if (p2_group_norm(c1, c2, mul4, &sacc, &esac)) {
        int32_t m1 = mul4[0], m2 = mul4[1];
        for (j = 0; j < 32; j++) {
#if LZ_GDN_STATE_2PLANE
            int32_t H = ((int32_t)ph[j] << 8) + (int32_t)pl[j];
#else
            /* hi*L, and L is 256 - NOT 254. The caller's c1 carries 1/L
               in both representations, so a single-plane H of hi*254
               would shrink every decay term by 0.78% per step -
               invisible in the default build, which is two-plane; the
               A/B control build is the only one that would run it.
               Caught by writing the MMX twin, which has to state what H
               is. */
            int32_t H = (int32_t)ph[j] << 8;
#endif /* LZ_GDN_STATE_2PLANE */
            A[j] = m1 * H + m2 * (int32_t)dq[j];
            if (A[j] > amax) amax = A[j];
            if (-A[j] > amax) amax = -A[j];
        }
        sh = p2_shift_of(amax);

        for (j = 0; j < 32; j++) {
            /* Rounded shift, not truncating: an arithmetic shift right
               rounds toward -inf, which is a BIAS, and bias adds
               linearly where quantization noise adds in quadrature.
               Measured on this routine: two truncating shifts cost
               5.88e-05 against 4.16e-05 for two rounded ones. */
            int32_t hn = (sh > 0) ? ((A[j] + (1 << (sh - 1))) >> sh) : A[j];
            /* Split by SHIFT, which is what LZ_GDN_LO_SCALE = 256
               bought. (hn + 128) >> 8 is round-to-nearest for a
               power-of-two divisor, exact, and native on MMX (paddw +
               psraw); the /254 form needs a divide, which has no MMX
               equivalent (ops.h has the exhaustive search that closed
               that door). */
            int h = (int)((hn + 128) >> 8);
            if (h >  127) h =  127;
            if (h < -127) h = -127;
            oh[j] = (int8_t)h;
#if LZ_GDN_STATE_2PLANE
            {
                int32_t l = hn - ((int32_t)h << 8);
                /* Same +-127 as the float quantizer, for the same
                   reason: the two must agree about what a state code
                   word can be. Only -128 is reachable here, which is
                   what lets the MMX twin clamp with a saturating
                   subtract instead of a compare. */
                if (l >  127) l =  127;
                if (l < -127) l = -127;
                ol[j] = (int8_t)l;
            }
#endif /* LZ_GDN_STATE_2PLANE */
        }
    }
    if (!p2_group_scale(sacc, esac, sh, os)) {
        for (j = 0; j < 32; j++) {
            oh[j] = 0;
#if LZ_GDN_STATE_2PLANE
            ol[j] = 0;
#endif /* LZ_GDN_STATE_2PLANE */
        }
    }
}

/* --- the SIMD twin, one whole ROW rather than one group ---------------

   The MMX cell for this operator. Bit-identical to the scalar
   above by construction, not by tolerance: every step is the same
   integer operation on the same integers, verified at 100k scale.

   WHY A ROW AND NOT A GROUP - it is entirely about emms hoisting. The
   emms may only be hoisted out of a loop whose body has no x87, and the
   per-group body has plenty: the coefficients, the normalization, the
   new scale. So the row is PHASE-SPLIT instead - all the float work of
   the row first, then all the MMX work, then one emms, then the float
   epilogue. One emms per row instead of one per 32 elements, which is
   the ratio that costs 16% of the matmul compute budget on
   a P6 and worse on a P4.

   The price is three loops over ng and three small static arrays, and
   the risk is that this sequencing drifts from the scalar one. That is
   why every float decision inside it - coefficients, multipliers,
   flush thresholds, new scale - is a call into the SAME helper the
   scalar path calls. What is duplicated here is the order of the
   phases, nothing else. */
#ifdef LZ_HAVE_P2_MMX
static void gdn_p2_row_simd(const int8_t *ph_row,
#if LZ_GDN_STATE_2PLANE
                            const int8_t *pl_row,
#endif /* LZ_GDN_STATE_2PLANE */
                            const float *rs_in, float decs,
                            const int16_t *dq, const float *s_dg, float kv,
                            int8_t *oh_row,
#if LZ_GDN_STATE_2PLANE
                            int8_t *ol_row,
#endif /* LZ_GDN_STATE_2PLANE */
                            float *rs_out, int ng, int tier) {
#if !defined(__WATCOMC__)
    lz_p2_blk *blk = p2_blk();     /* Watcom's row functions get their
                                       own via their own p2_blk() call -
                                       see the Watcom Phase 2 branch
                                       below. */
#endif /* !__WATCOMC__ */
    static float sacc[LZ_GDN_MAX_VD / 32];
    static int   esac[LZ_GDN_MAX_VD / 32];
    static int   shv[LZ_GDN_MAX_VD / 32];
    static int16_t mul[LZ_GDN_MAX_VD / 32][4];
#if !LZ_GDN_STATE_2PLANE
    /* The single-plane build feeds the kernel a zero low plane and
       throws its low output away, rather than carrying a second pair of
       kernels that would be exercised by nothing. H = hi*256 + 0 is
       exactly the single-plane H. */
    static const int8_t zero_lo[32];
    static int8_t sink_lo[32];
#endif /* !LZ_GDN_STATE_2PLANE */
    int gg, j;

    /* Not billed here - gdn_p2_fixed_rows bills this row's cost once,
       above its call to this function. */

    /* PHASE 1 - float only. */
    for (gg = 0; gg < ng; gg++) {
        float c1, c2;
        p2_group_coef(decs, rs_in[gg], kv, s_dg[gg], &c1, &c2);
        p2_group_norm(c1, c2, mul[gg], &sacc[gg], &esac[gg]);
    }

    /* PHASE 2 - integer and MMX only, NO x87 anywhere in this loop.
       Changing that is what would make the single emms below wrong. */
#if defined(__WATCOMC__)
    /* Watcom: one call per row, not per group. The #pragma aux bodies
       are inline inside whichever row function `tier` selects
       (src/ops_mmx.c's lz_p2_rows_mmx, src/ops_mmx_sse.c's
       lz_p2_rows_sse, src/ops_sse2.c's lz_p2_rows_sse2); this loop
       calls it once per row, in place of the one call per group that
       wrapping the group loop unchanged would cost.
       pl_stride/ol_stride carry the single-plane fallback (the same
       fixed zero/sink buffer at every group) across the TU boundary
       without teaching those files about LZ_GDN_STATE_2PLANE. */
    {
        const int8_t *plg0;
        int8_t *olg0;
        int pstride;
#if LZ_GDN_STATE_2PLANE
        plg0 = pl_row; olg0 = ol_row; pstride = 32;
#else
        plg0 = zero_lo; olg0 = sink_lo; pstride = 0;
#endif /* LZ_GDN_STATE_2PLANE */
#ifdef LZ_HAVE_P2_SSE2
        if (tier >= 3)
            lz_p2_rows_sse2(ph_row, plg0, pstride, dq, mul,
                            oh_row, olg0, pstride, shv, ng);
        else
#endif /* LZ_HAVE_P2_SSE2 */
#ifdef LZ_HAVE_P2_SSE
        if (tier >= 2)
            lz_p2_rows_sse(ph_row, plg0, pstride, dq, mul,
                           oh_row, olg0, pstride, shv, ng);
        else
#endif /* LZ_HAVE_P2_SSE */
        lz_p2_rows_mmx(ph_row, plg0, pstride, dq, mul,
                       oh_row, olg0, pstride, shv, ng);
    }
#else /* gcc: per-group loop with extern calls into ops_mmx.c /
         ops_mmx_sse.c / ops_sse2.c. The row-at-a-time form above is
         Watcom-only because its bodies are #pragma aux, which gcc has
         no equivalent for; the two arms must stay bit-identical. */
    for (gg = 0; gg < ng; gg++) {
        int32_t amax;
        int sh;
        const int8_t *plg;
        /* Every constant replicated across the full 16 bytes. MMX reads
           the low 8 and SSE2 reads all 16, so one fill serves both -
           see the block's comment in ops_kernel_p2.h. */
        for (j = 0; j < 8; j += 2) {
            blk->mul[j]     = mul[gg][0];
            blk->mul[j + 1] = mul[gg][1];
        }
        /* A whole statement in each arm rather than one statement split
           across the #if: half a statement per branch reads as a
           conditional expression and is not one. plg is declared at the
           top of the block because C89 has no declaration after a
           statement. */
#if LZ_GDN_STATE_2PLANE
        plg = pl_row + gg * 32;
#else
        plg = zero_lo;
#endif /* LZ_GDN_STATE_2PLANE */
#ifdef LZ_HAVE_P2_SSE2
        if (tier >= 3) lz_p2_mul32_sse2(ph_row + gg * 32, plg, dq + gg * 32, blk);
        else
#endif /* LZ_HAVE_P2_SSE2 */
        /* No SSE1 arm here on purpose: this half has no SSE1 content,
           so tier 2 runs the MMX mul32 unchanged. */
        lz_p2_mul32_mmx(ph_row + gg * 32, plg, dq + gg * 32, blk);
        amax = blk->amax[0];
        for (j = 1; j < 4; j++) if (blk->amax[j] > amax) amax = blk->amax[j];
        sh = p2_shift_of(amax);
        shv[gg] = sh;
        for (j = 0; j < 4; j++) {
            blk->rnd[j] = (sh > 0) ? (int32_t)(1 << (sh - 1)) : 0;
            blk->cnt[j] = (j == 0) ? (int32_t)sh : 0;
        }
        for (j = 0; j < 8; j++) {
            blk->k128[j] = 128;
            blk->kclp[j] = 32641;   /* MMX/SSE2: psubsw then paddsw */
            blk->kmin[j] = -127;    /* SSE1: one pmaxsw does the same */
        }
        {
            int8_t *olg;
#if LZ_GDN_STATE_2PLANE
            olg = ol_row + gg * 32;
#else
            olg = sink_lo;
#endif /* LZ_GDN_STATE_2PLANE */
#ifdef LZ_HAVE_P2_SSE2
            if (tier >= 3) lz_p2_split32_sse2(blk, oh_row + gg * 32, olg);
            else
#endif /* LZ_HAVE_P2_SSE2 */
#ifdef LZ_HAVE_P2_SSE
            if (tier >= 2) lz_p2_split32_sse(blk, oh_row + gg * 32, olg);
            else
#endif /* LZ_HAVE_P2_SSE */
            lz_p2_split32_mmx(blk, oh_row + gg * 32, olg);
        }
    }
#endif /* __WATCOMC__ */
    /* Only the 64-bit tiers leave MMX state behind, and the SSE1 one is
       a 64-bit tier: pmaxsw is an SSE1 instruction operating on an MMX
       REGISTER, so tier 2 needs the emms exactly as tier 1 does. Tier 3
       touches xmm and nothing else, so there is no emms to
       clear there, and skipping it is worth a few cycles a row on the
       P4 - the machine the SSE2 tier is for. */
    if (tier < 3) _mm_empty();

    /* PHASE 3 - float again. A group the prologue or the epilogue
       flushed is written as an exact zero here, on top of whatever the
       kernel put there; the kernel ran on a zero multiplier pair in that
       case and produced zeros anyway, but relying on that would be
       relying on an accident. */
    for (gg = 0; gg < ng; gg++)
        if (!p2_group_scale(sacc[gg], esac[gg], shv[gg], rs_out + gg))
            for (j = 0; j < 32; j++) {
                oh_row[gg * 32 + j] = 0;
#if LZ_GDN_STATE_2PLANE
                ol_row[gg * 32 + j] = 0;
#endif /* LZ_GDN_STATE_2PLANE */
            }
}
#endif /* LZ_HAVE_P2_MMX */
#endif /* LZ_GDN_FIXED */

/* WHICH pass-2 body ran - the same join lz_kernel_tier() reports for the
   matmul kernels, for the same reason. "The MMX pass 2 is selected" is
   otherwise unfalsifiable from outside: it is required to be bit
   identical to the scalar one, so no output changes when it silently
   is not compiled in, and that is not hypothetical - the Watcom build
   can carry none of the hand assembly and every bit-identity gate
   stays green.

   "-" means the question does not apply: the float pass 2 is selected,
   or this build has no fixed pass at all. */
const char *lz_gdn_p2_impl(void) {
#if LZ_GDN_FIXED
    if (!lz_gdn_p2_mode()) return "-";
#if defined(LZ_HAVE_P2_SSE2_ASM)
    if (p2_tier() >= 3) return "sse2-asm";
#elif defined(LZ_HAVE_P2_SSE2)
    if (p2_tier() >= 3) return "sse2-intrin";
#endif /* LZ_HAVE_P2_SSE2_ASM */
#if defined(LZ_HAVE_P2_SSE_ASM)
    if (p2_tier() >= 2) return "sse-asm";
#elif defined(LZ_HAVE_P2_SSE)
    if (p2_tier() >= 2) return "sse-intrin";
#endif /* LZ_HAVE_P2_SSE_ASM */
#if defined(LZ_HAVE_P2_MMX_ASM)
    if (p2_tier()) return "mmx-asm";
#elif defined(LZ_HAVE_P2_MMX)
    if (p2_tier()) return "mmx-intrin";
#endif /* LZ_HAVE_P2_MMX_ASM */
    return "ref";
#else
    return "-";
#endif /* LZ_GDN_FIXED */
}

#if LZ_GDN_FIXED
/* The fixed pass 1 for one head, shared by both families - the twin of
   gdn_p2_fixed_rows below, and just as easy to forget to wire into a
   new family.

   `gvec` NULL is the scalar-decay family: q and k go to the coefficient
   table as they are. Non-NULL is the per-channel one, where they are
   premultiplied first - which is the whole trick, since
   ss*gvec*k == ss*(gvec*k) means the SAME table builder and the SAME
   hand-written assembly serve both. Returns k.q, RAW and unweighted in
   both cases: it is not part of the contraction, the epilogue uses it
   against delta, and weighting it would be invisible at kd == 1 and
   wrong everywhere else. */
static float gdn_pass1_fixed_head(const int8_t *sq_in,
#if LZ_GDN_STATE_2PLANE
                                  const int8_t *sq2_in,
#endif /* LZ_GDN_STATE_2PLANE */
                                  const float *ss_in,
                                  const float *q, const float *k,
                                  const float *gvec, int kd, int vd,
                                  float *u, float *w) {
    static int32_t au[LZ_GDN_MAX_VD], aw[LZ_GDN_MAX_VD];
    static float   su[LZ_GDN_MAX_VD / 32], sw[LZ_GDN_MAX_VD / 32];
    static int32_t au2[LZ_GDN_MAX_VD], aw2[LZ_GDN_MAX_VD];
    static float   su2[LZ_GDN_MAX_VD / 32], sw2[LZ_GDN_MAX_VD / 32];
    static float   kg[LZ_GDN_MAX_KD], qg[LZ_GDN_MAX_KD];
#if LZ_GDN_STATE_2PLANE
    static int32_t aul[LZ_GDN_MAX_VD], awl[LZ_GDN_MAX_VD];
#endif /* LZ_GDN_STATE_2PLANE */
    const float *qq = q, *kk_ = k;
    float kq = 0.0f;
    int ng = vd / 32, kk, gg, j;

    if (gvec) {
        for (kk = 0; kk < kd; kk++) {
            float gc = gvec[kk];
            kg[kk] = k[kk] * gc;
            qg[kk] = q[kk] * gc;
            kq += k[kk] * q[kk];
        }
        qq = qg; kk_ = kg;
    } else {
        for (kk = 0; kk < kd; kk++) kq += k[kk] * q[kk];
    }

    /* PASS 1's FLOAT WORK HAD NO SITE. gdn_p2_fixed_rows bills pass 2
       and nothing billed this, so the recurrence class read as pass 2
       alone. Found by .prof/unbilled_scan.awk, which keys on the
       OPERATION (lz_i32f / q8_round) rather than on the signature the
       way build/f32count_billing_gate.sh does - that gate's own header
       names this rule and says why it was not implemented.

       Billed here rather than at the call site so the two arms below
       cannot drift apart: the epilogue's shape is a compile-time choice
       and the count has to move with it.

       Prologue: 3 mul + 1 add per kk with a gvec (KDA), 1 mul + 1 add
       without (GDN). Epilogue, per ELEMENT and for each of u and w:
       2 lz_i32f (2 cvt), 0 mul, 1 add under the two-plane shape. BOTH
       multiplies are exponent adds now - the low plane's
       1/LZ_GDN_LO_SCALE always was one, and `* su[gg]` became one when
       gdn_build_table started block-normalizing to 32767*2^k, which
       makes su and sw exact powers of two. The two-TABLE shape
       (LZ_GDN_FIXED>=2) still spends 2 and shares this bill; it is a
       non-default arm, so the
       count below follows the default and understates that one - said
       here rather than left for a reader to find by subtraction.
       Single-plane: 1 cvt and 1 mul.

       NOT COVERED, and it is the larger half: gdn_build_table runs once
       per group inside gdn_pass1_fixed and does its own float work
       (.prof/unbilled_scan.awk lists it). Deriving it needs the table
       shape audited, and a half-derived formula is exactly what this
       file has twice been wrong by. */
#if LZ_GDN_STATE_2PLANE || LZ_GDN_FIXED >= 2
    LZ_FCX(LZ_FC_RECUR, (long)kd * (gvec ? 3 : 1),
           (long)vd * 2 + (long)kd, 0, (long)vd * 4, 0);
#else
    LZ_FCX(LZ_FC_RECUR, (long)kd * (gvec ? 3 : 1),
           (long)kd, 0, (long)vd * 2, 0);
#endif /* LZ_GDN_STATE_2PLANE || LZ_GDN_FIXED >= 2 */
#if LZ_GDN_STATE_2PLANE
    gdn_pass1_fixed(sq_in, sq2_in, aul, awl, ss_in, qq, kk_, kd, vd,
                    au, aw, su, sw, au2, aw2, su2, sw2);
#else
    gdn_pass1_fixed(sq_in, ss_in, qq, kk_, kd, vd,
                    au, aw, su, sw, au2, aw2, su2, sw2);
#endif /* LZ_GDN_STATE_2PLANE */
    for (gg = 0; gg < ng; gg++) {
      /* `* su[gg]` IS AN EXPONENT ADD NOW, because gdn_build_table
         block-normalizes to 32767*2^k and so writes su and sw as exact
         powers of two. The exponent is read back out of the float
         rather than threaded through LZGdnTable - su is pow2f(ek), so
         its stored exponent IS ek, and a parallel int array would be a
         second copy of a number already in the first one.
         WITH A BIT TEST, NOT A FLOAT COMPARE: mantissa zero and the
         exponent field neither 0 nor 255 is exactly "this is a power of
         two", costs two integer ops per GROUP, and never asks
         __aeabi_fcmp. The guard is live, not decoration - the all-zero
         group takes su = LZ_Q15_INV, which is 1/32767 and not a power
         of two, and those groups fall back to the multiply.
         Hoisted per gg because that is where su changes. */
      union { float f; uint32_t u; } sb;
      int su_e, sw_e, su_p2, sw_p2;
      sb.f = su[gg];
      su_p2 = ((sb.u & 0x007FFFFFu) == 0u)
              && (uint32_t)(((sb.u >> 23) & 0xFFu) - 1u) < 0xFEu;
      su_e = (int)((sb.u >> 23) & 0xFFu) - 127;
      sb.f = sw[gg];
      sw_p2 = ((sb.u & 0x007FFFFFu) == 0u)
              && (uint32_t)(((sb.u >> 23) & 0xFFu) - 1u) < 0xFEu;
      sw_e = (int)((sb.u >> 23) & 0xFFu) - 127;
        for (j = 0; j < 32; j++) {
            int vi = gg * 32 + j;
#if LZ_GDN_STATE_2PLANE
            /* 1/LZ_GDN_LO_SCALE is 2^-8 - the static assert further
               down pins LO_SCALE at 256 - so the low plane's weighting
               is an exponent SUBTRACT, not a __aeabi_fmul. Two per
               element, in pass 1's tail, which is 61,440 operations a
               token on kmr20 and the third-largest block of the
               census's largest row. */
            {
                float lo, acc;
                if (!f32_scale_pow2(lz_i32f(aul[vi]), -LZ_GDN_LO_SHIFT, &lo))
                    lo = lz_i32f(aul[vi]) * (1.0f / LZ_GDN_LO_SCALE);
                acc = lz_i32f(au[vi]) + lo;
                if (!su_p2 || !f32_scale_pow2(acc, su_e, &u[vi]))
                    u[vi] = acc * su[gg];
                if (!f32_scale_pow2(lz_i32f(awl[vi]), -LZ_GDN_LO_SHIFT, &lo))
                    lo = lz_i32f(awl[vi]) * (1.0f / LZ_GDN_LO_SCALE);
                acc = lz_i32f(aw[vi]) + lo;
                if (!sw_p2 || !f32_scale_pow2(acc, sw_e, &w[vi]))
                    w[vi] = acc * sw[gg];
            }
#elif LZ_GDN_FIXED >= 2
            u[vi] = lz_i32f(au[vi]) * su[gg] + lz_i32f(au2[vi]) * su2[gg];
            w[vi] = lz_i32f(aw[vi]) * sw[gg] + lz_i32f(aw2[vi]) * sw2[gg];
#else
            if (!su_p2 || !f32_scale_pow2(lz_i32f(au[vi]), su_e, &u[vi]))
                u[vi] = lz_i32f(au[vi]) * su[gg];
            if (!sw_p2 || !f32_scale_pow2(lz_i32f(aw[vi]), sw_e, &w[vi]))
                w[vi] = lz_i32f(aw[vi]) * sw[gg];
#endif /* LZ_GDN_STATE_2PLANE */
        }
    }
    return kq;
}

static void gdn_p2_fixed_rows(const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                              const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                              const float *ss_in, float *ss_out,
                              const float *delta, const float *k,
                              const float *gvec, float gt,
                              int kd, int vd) {
    static int16_t dq[LZ_GDN_MAX_VD];
    static float s_dg[LZ_GDN_MAX_VD / 32];
    int ng = vd / 32, kk, gg, vv;
#ifdef LZ_HAVE_P2_MMX
    /* Hoisted out of the kk loop below - the whole loop, not just the
       one line that calls it twice. The tier cannot change inside this
       function - nothing here calls lz_kernel_select except p2_tier's
       own first line, and that is a no-op once the tier is resolved -
       so asking once per CALL rather than once per ROW is the same
       answer. build/dispatch_hotness.sh: 73,729 executions per line
       before this, ~500 after. */
    int p2t;
#endif /* LZ_HAVE_P2_MMX */

    /* delta quantized PER 32-GROUP, not once for the whole vd vector.
       A single scale for the whole delta - the shape gdn_build_table
       uses for its coefficients - would be the thing most likely to
       break the error budget, given delta's dynamic range. Measured:
       one scale gives 4.34e-05 state error against an exact float
       recurrence; per group is measurably better. The cost is one extra
       absmax pass over
       32 elements per group and one more multiplier -
       both per group, nothing per element, so the 83% this whole tier
       exists to move is untouched.

       Group boundaries match the state's own quantization groups, which
       is what lets c2 stay a single scalar per (row, group) call. */
    /* THIS LOOP HAD NO CENSUS SITE AT ALL. gdn_p2_fixed_rows billed only
       its per-ROW work, so the whole delta quantization - a divide and 32
       multiply-and-round per group, every group, every call - read as
       zero. That is the defect class build/f32count_billing_gate.sh
       exists for, and it does not catch this one: the gate looks for a
       function returning float from an int accumulator, and this is a
       void function quantizing a float vector.

       THE AMAX SCAN RUNS IN THE INTEGER DOMAIN, and used not to. It was
       two float comparisons per element - `delta[vv] < 0.0f` for the
       sign and `a > amd` for the max - which on a soft-float target is
       two `bl __aeabi_fcmpXX` at ~25 instructions each, 12,480 of them
       per token on kmr20. It was also UNBILLED, under the census's
       "compares are not counted" convention, so the largest row in the
       table was hiding its second-largest per-element cost.

       Same IEEE-754 trick q8_amax (ops_quant.c) and lz_softmax's max
       scan (ops_norm.c) already use, and the reason it is exact here is
       the same: masking off the sign bit gives the magnitude, and for
       magnitudes the unsigned bit-pattern order IS the numeric order,
       so this is a permutation of the same total order rather than an
       approximation. It owes the float loop bit-identity and delivers
       it. NaN cannot reach here - delta is a finite recurrence delta -
       and a negative zero masks to +0, which the float form also
       treated as 0.

       BILLED AS ZERO, and that is a rule rather than an omission.
       This is the f32 census; an integer compare is not an f32
       operation, and billing one would make the cmp column
       inhomogeneous - some entries costing 25 instructions and some
       costing 1, summed into one number. The loop is not hidden by
       being unbilled any more, it is empty of float work. What it cost
       before is measured, not estimated: ng*32 is 6,144 elements per
       token on kmr20/zh, so 12,288 __aeabi_fcmp at ~25 instructions,
       307,200 per token, against 25.6M for the whole token - 1.2%,
       removed for 6,144 ands and 6,144 integer compares. */
    for (gg = 0; gg < ng; gg++) {
        float amd, inv_d, sd;
        int ed;
        union { float f; uint32_t u; } dbit;
        uint32_t amu = 0;
        for (vv = gg * 32; vv < gg * 32 + 32; vv++) {
            uint32_t a;
            dbit.f = delta[vv];
            a = dbit.u & 0x7FFFFFFFu;
            if (a > amu) amu = a;
        }
        dbit.u = amu;
        amd = dbit.f;
        /* BLOCK-NORMALIZED to 32767 * 2^ed, the same trade
           gdn_build_table's coefficient table takes: sd becomes 2^ed
           and inv_d becomes 2^-ed EXACTLY, so the 32 per-element
           multiplies below are exponent adds, p2_group_coef's c2 folds,
           and the divide disappears.
         *
         * Built from pow2f rather than from amd*LZ_Q15_INV and
         * 32767/amd: LZ_Q15_INV is fl(1/32767), so the product would be
         * a ULP off 2^ed and the fold would be of the wrong number.
         *
         * VALUE-CHANGING, measured before landing. Rounding amd UP
         * costs at most one bit of the delta group's range. Paired NLL,
         * three models x two fixtures, n=3786: +7.88e-04 at t=+0.62 -
         * indistinguishable, and the same band as the coefficient
         * table's +6.45e-04/t=0.48 and the attention int wsum's default
         * at t=0.65. */
        {
            float r = amd * LZ_Q15_INV;
            ed = expof(r);
            if (pow2f(ed) < r) ed++;      /* round the block UP */
            sd = pow2f(ed);
        }
        LZ_FCX(LZ_FC_RECUR, 1, 0, 0, 0, 0);   /* the amd*LZ_Q15_INV above */
        /* Same floor as everywhere else: a delta group below this is
           numerically zero, and letting its scale reach the subnormal
           range is how x87 and SSE stop agreeing (ops.h's
           LZ_Q8_MIN_SCALE has the mechanism). */
        if (!(sd >= LZ_Q8_MIN_SCALE_F)) {
            s_dg[gg] = 1.0f;
            for (vv = gg * 32; vv < gg * 32 + 32; vv++) dq[vv] = 0;
            continue;
        }
        s_dg[gg] = sd;
        inv_d = pow2f(-ed);               /* exact; was 32767/amd */
        /* 0 div and 0 mul per element now: inv_d is 2^-ed, so the scale
           is an exponent add. Only q8_round's magic add is left, the
           same one the other quantize sites in this file bill. */
        LZ_FCX(LZ_FC_RECUR, 0, 32, 0, 0, 0);
        for (vv = gg * 32; vv < gg * 32 + 32; vv++) {
            float dv;
            int t;
            if (!f32_scale_pow2(delta[vv], -ed, &dv)) dv = delta[vv] * inv_d;
            t = q8_round(dv);
            if (t >  32767) t =  32767;
            if (t < -32767) t = -32767;
            dq[vv] = (int16_t)t;
        }
    }

#ifdef LZ_HAVE_P2_MMX
    p2t = p2_tier();
#endif /* LZ_HAVE_P2_MMX */
    for (kk = 0; kk < kd; kk++) {
        const int8_t *row_in = sq_in + (size_t)kk * vd;
        int8_t *row_out = sq_out + (size_t)kk * vd;
        const float *rs_in = ss_in + (size_t)kk * ng;
        float *rs_out = ss_out + (size_t)kk * ng;
        float dec = gvec ? gvec[kk] : gt;    /* the one family difference */
        /* Scaled ONCE per row, not once per group: see p2_group_coef,
           which now takes this rather than dec. Above the tier fork so
           both row paths get the same value from the same multiply -
           two textually-equal copies is the hazard that comment names. */
        float decs;
#if LZ_GDN_STATE_2PLANE
        const int8_t *row_lo_in = sq2_in + (size_t)kk * vd;
        int8_t *row_lo_out = sq2_out + (size_t)kk * vd;
#endif /* LZ_GDN_STATE_2PLANE */
        /* Declarations first, for the C89 floor. This assignment
           sat above the two above it until a C89 build said so. */
        if (!f32_scale_pow2(dec, -LZ_GDN_LO_SHIFT, &decs))
            decs = dec * (1.0f / LZ_GDN_LO_SCALE);
        /* Both SIMD (gdn_p2_row_simd) and scalar (gg loop below) row
           paths cost the same per group. Billed once, above the fork -
           billing only inside gdn_p2_row_simd would read zero on
           --kernel ref, where p2_tier()==0 takes the scalar path only.
           coef 1 mul (was 3, then 2: the 2^-8 moved to the row and is
           an exponent add there too, and c1's `decs * rs` folds because
           rs is the state's group scale, pow2f. Only c2's `kv * sd`
           is left - sd is the DELTA scale and nothing normalized it);
           norm
           0 mul (was 2: p2_group_norm's `c * pow2f(14-e)` pair is an
           exponent add now, f32_scale_pow2 - same number, no multiply)
           + 2 q8_round(add); scale 0 mul (was 2: both power-of-two
           factors are folded into one pow2f, which bills nothing - see
           p2_group_scale). The fallback arms outside pow2f's range
           still spend theirs and are not billed: billing a branch by
           its worst case would overstate every row that has one. */
        LZ_FCX(LZ_FC_RECUR, ng, 2 * ng, 0, 0, 0);
#ifdef LZ_HAVE_P2_MMX
        /* ONE CALL, NOT TWO. This line asked p2_tier() and then passed
           p2_tier() again as the last argument, and it sits in the
           per-row loop: build/dispatch_hotness.sh measured 147,457
           executions of EACH over eight tokens, the second and third
           hottest tier dispatches in the engine. The function reads a
           global and cannot be folded by the compiler because it calls
           lz_kernel_select on its first line.

           Not cached in a static inside p2_tier: --kernel switches tiers
           at runtime and the suite relies on it, so a cache there would
           make the second tier of a run silently be the first. */
        if (p2t) {
            gdn_p2_row_simd(row_in,
#if LZ_GDN_STATE_2PLANE
                            row_lo_in,
#endif /* LZ_GDN_STATE_2PLANE */
                            rs_in, decs, dq, s_dg, k[kk], row_out,
#if LZ_GDN_STATE_2PLANE
                            row_lo_out,
#endif /* LZ_GDN_STATE_2PLANE */
                            rs_out, ng, p2t);
            continue;
        }
#endif /* LZ_HAVE_P2_MMX */
        for (gg = 0; gg < ng; gg++) {
            float c1, c2;
            p2_group_coef(decs, rs_in[gg], k[kk], s_dg[gg], &c1, &c2);
            gdn_p2_group_fixed(row_in + gg * 32,
#if LZ_GDN_STATE_2PLANE
                               row_lo_in + gg * 32,
#endif /* LZ_GDN_STATE_2PLANE */
                               c1, dq + gg * 32, c2,
                               row_out + gg * 32,
#if LZ_GDN_STATE_2PLANE
                               row_lo_out + gg * 32,
#endif /* LZ_GDN_STATE_2PLANE */
                               rs_out + gg);
        }
    }
}
/* Both of the above read the coefficient table, the pass-1 kernel and
   the per-group helpers, all of which live inside LZ_GDN_FIXED. Their
   only call sites are already guarded the same way, so guarding the
   definitions costs nothing and keeps the capacity-fallback build
   compilable. */
#endif /* LZ_GDN_FIXED */

/* ONE recurrence, two families. lz_gdn_step and lz_kda_step are now
   both thin wrappers over this.

   The identity that makes it exact - not approximate, exact:

       lz_kda_step(...) == lz_recur_step(..., gvec, gt = 1.0f)

   because KDA's per-channel decay is already folded into pass 1 (its
   coefficients are gvec[kk]*k[kk]), so the only place GDN's scalar gt
   still appears is the epilogue's `gt*u` and `gt*w` and pass 2's
   `rs*gt`. Multiplying by 1.0f is exact for every finite value, so
   passing gt = 1.0f reproduces KDA's arithmetic bit for bit rather than
   merely closely.

   Why bother: a tier added to one family and forgotten in the other is
   SILENT - it just does nothing on that architecture, both settings
   agree, and it reads as "this tier is not worth much". KunMoe is
   0 gdn / 6 kda / 2 full, so the forgotten half is every time the only
   half that matters.

   `gvec` NULL selects the scalar-decay family. The gate is bit-exactness
   against a 10-hash baseline (2 models x --gdn x --gdn-p2); nothing may
   move. */
void lz_recur_step(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 const short *vi, float vscale,
                 const float *gvec, float gt, float beta, int kd, int vd) {
    float u[LZ_GDN_MAX_VD];      /* holds S^T k first, then reused in place as delta */
    float w[LZ_GDN_MAX_VD];      /* S^T q */
    float kq = 0.0f;
    int kk, gg, j, vv, ng;

    if (vd <= 0 || (vd % 32) != 0 || vd > LZ_GDN_MAX_VD || kd <= 0) {
        for (vv = 0; vv < vd; vv++) out[vv] = 0.0f;   /* defense: no silent miscalc */
        return;
    }
    ng = vd / 32;

#if LZ_GDN_STATE_F32
    /* NOTE: this shadow is keyed by the READ pointer's own address,
       1:1 with (layer, head) ONLY when every call for a given head
       always passes the SAME sq_in - true for ordinary decode (in==out,
       unchanged) but NOT during a speculative verify batch, where sq_in
       advances through the ring slot by slot. Enabling LZ_GDN_STATE_F32
       together with --spec K would silently look up the wrong shadow
       entry per ring step. Flagged rather than fixed, since fixing it
       means re-keying the shadow map by (layer,head) instead of by
       pointer, a change to a scaffold this feature does not otherwise
       touch. */
    {
        float *sf = gdn_shadow(sq_in, kd * vd);
        if (sf) {
            /* Same recurrence, exact float state: no quantization anywhere
               in the loop, so what is left is only float rounding.

               gvec IS THE WHOLE DECAY ON A KDA MODEL, where gt is the
               literal 1.0f - so this block must read gvec and not only
               gt, which is the GDN family's scalar decay. Reading gt
               alone makes the "exact float state" arm an UNDECAYED
               recurrence on every KunMoe checkpoint, worth +1.29 NLL
               against the shipping default at t=27.9: a scaffold that
               answers, loudly, about a model this engine does not run.
               Mirrors the reference float path below exactly - kcw/qcw
               for the contraction, raw kc for the update, dec for the
               state scale. */
            for (vv = 0; vv < vd; vv++) { u[vv] = 0.0f; w[vv] = 0.0f; }
            for (kk = 0; kk < kd; kk++) {
                const float *row = sf + (size_t)kk * vd;
                float kc = k[kk], qc = q[kk];
                float kcw = gvec ? kc * gvec[kk] : kc;
                float qcw = gvec ? qc * gvec[kk] : qc;
                kq += kc * qc;              /* RAW dot, both families */
                for (vv = 0; vv < vd; vv++) {
                    u[vv] += row[vv] * kcw;
                    w[vv] += row[vv] * qcw;
                }
            }
            /* `v` is NULL whenever the caller took the int16 v entry
               (lz_kda_step_vi16), so this reads it the way the shipping
               loops below do. It did not, and this scaffold segfaulted
               on every KunMoe model - a compile switch the switch-matrix
               gate builds and nothing runs. */
            for (vv = 0; vv < vd; vv++) {
                float vc = vi ? lz_i32f((int32_t)vi[vv]) * vscale : v[vv];
                u[vv] = (vc - gt * u[vv]) * beta;
                out[vv] = gt * w[vv] + u[vv] * kq;
            }
            for (kk = 0; kk < kd; kk++) {
                float *row = sf + (size_t)kk * vd;
                float kc = k[kk];
                float dec = gvec ? gvec[kk] : gt;
                for (vv = 0; vv < vd; vv++)
                    row[vv] = row[vv] * dec + kc * u[vv];
            }
            return;
        }
    }
#endif /* LZ_GDN_STATE_F32 */

    /* Pass 1: state read once, gathering both contractions. Always reads
       sq_in/sq2_in/ss_in - the "_in" side, whether or not it aliases
       "_out" (see this function's own header comment on the ring). */
#if LZ_GDN_FIXED
    /* lz_gdn_mode() resolves from --fixed, the same knob pass 2 reads;
       the kd bound is a hard capacity limit (the coefficient table is
       int16_t[LZ_GDN_MAX_KD/2][8]). Either one falling through lands on
       the same float path below, which is exactly why that path is the
       baseline worth being able to ask for. */
    if (lz_gdn_mode() && kd <= LZ_GDN_MAX_KD) {
        kq = gdn_pass1_fixed_head(sq_in,
#if LZ_GDN_STATE_2PLANE
                                  sq2_in,
#endif /* LZ_GDN_STATE_2PLANE */
                                  ss_in, q, k, gvec, kd, vd, u, w);
    } else
#endif /* LZ_GDN_FIXED */
    {
    /* Float fallback: only 2 mul + 2 add per element - scale and k/q
       components are pre-multiplied. */
    for (vv = 0; vv < vd; vv++) { u[vv] = 0.0f; w[vv] = 0.0f; }
    for (kk = 0; kk < kd; kk++) {
        const int8_t *row = sq_in + (size_t)kk * vd;
        const float *rs = ss_in + (size_t)kk * ng;
        float kc = k[kk], qc = q[kk];
        /* The one family difference in this loop: KDA's contraction is
           against gvec-weighted k and q, GDN's against them raw. With
           gvec NULL the expressions below are character-for-character
           what the GDN-only version had. */
        float kcw = gvec ? kc * gvec[kk] : kc;
        float qcw = gvec ? qc * gvec[kk] : qc;
        kq += kc * qc;                     /* RAW dot, both families */
        for (gg = 0; gg < ng; gg++) {
            const int8_t *p = row + gg * 32;
            float *uu = u + gg * 32, *ww = w + gg * 32;
            float sck = rs[gg] * kcw, scq = rs[gg] * qcw;
#if LZ_GDN_STATE_2PLANE
            const int8_t *pl = sq2_in + (size_t)kk * vd + gg * 32;
#endif /* LZ_GDN_STATE_2PLANE */
            for (j = 0; j < 32; j++) {
#if LZ_GDN_STATE_2PLANE
                float e = lz_i32f((int32_t)p[j]) + lz_i32f((int32_t)pl[j]) * (1.0f / LZ_GDN_LO_SCALE);
#else
                float e = lz_i32f((int32_t)p[j]);
#endif /* LZ_GDN_STATE_2PLANE */
                uu[j] += e * sck;
                ww[j] += e * scq;
            }
        }
    }
    }

    /* Scalar epilogue: output needs no second state read.
       S_new = gt*S + outer(k, delta)
       =>  S_new^T q = gt*(S^T q) + delta*dot(k, q)
     *
     * gt IS EXACTLY 1.0f ON EVERY KDA CALL - lz_kda_step passes the
     * literal, because KDA's decay already lives in pass 1's
     * coefficients (see that function's header). Multiplying by 1.0f is
     * exact for every value, zeros and infinities included, so dropping
     * the multiply is the SAME number rather than a close one. That
     * deletes two __aeabi_fmuls per element, 30 instructions each on
     * this target, for one integer compare hoisted out of the loop.
     *
     * TESTED ON THE BIT PATTERN, not with `gt == 1.0f`: a float compare
     * is itself a library call here and would put back a third of what
     * the branch saves. The two tests ask the same question - 0x3F800000
     * is the only encoding of 1.0f, and a NaN gt fails both. */
    {
        union { float f; uint32_t u; } gb;
        int vs_e = 0;
        gb.f = gt;
        if (vi) {
            /* vscale's exponent, or an exponent no operand can reach if
               vscale is not a power of two - f32_scale_pow2 then
               declines on every element and the multiply stays. Read
               with a bit test rather than a float compare, the same way
               su/sw are read in pass 1's tail. */
            union { float f; uint32_t u; } vb;
            vb.f = vscale;
            vs_e = (((vb.u & 0x007FFFFFu) == 0u)
                    && (uint32_t)(((vb.u >> 23) & 0xFFu) - 1u) < 0xFEu)
                   ? (int)((vb.u >> 23) & 0xFFu) - 127
                   : 0x40000000;
        }
        /* THIS LOOP HAD NO SITE. Pass 1 bills at gdn_pass1_fixed_head
           and pass 2 at gdn_p2_fixed_rows, so the recurrence class read
           as those two and this scalar epilogue was invisible - 5 or 6
           operations an element, 6,144 elements a token on kmr20, about
           31K the census could not see. Found while pricing v's int16
           exit, whose whole cost lands here: without a site the exit
           would have shown the producer's saving and none of the
           consumer's cost, which is the shape of a fake win.

           Billed per ARM because they differ by two multiplies, and gt
           is exactly 1.0f on every KDA call - averaging them would
           misreport whichever family is running. */
        /* v's INT16 ENTRY (int-pipeline 9.4's second half). It is a
           convert and a multiply here where the float entry has
           neither, and the conv it comes from drops two of each - so
           the trade is two operations an element, in this loop's
           favour. NOT re-associated into vi*(vscale*beta): that would
           be a second value change on top of the quantization and it
           has not been measured apart from it. */
        if (gb.u == 0x3F800000u) {
            /* vi costs the SAME two multiplies as the float v now, not
               three: the convert's `* vscale` is an exponent add. */
            LZ_FCX(LZ_FC_RECUR, (long)vd * 2, (long)vd * 2, 0,
                   vi ? (long)vd : 0, 0);
            if (vi)
                for (vv = 0; vv < vd; vv++) {
                    /* vscale is pow2f(-LZ_CONVO_V_ES) at the one caller
                       that passes it (forward.c's lz_kda_step_vi16
                       site) - a compile-time power of two, so the
                       convert's scaling is an exponent add. The guard
                       keeps the multiply for a zero vi[vv], which is
                       ordinary, and for any caller that ever passes a
                       vscale that is not a power of two. */
                    float vf;
                    if (!f32_scale_pow2(lz_i32f((int32_t)vi[vv]), vs_e, &vf))
                        vf = lz_i32f((int32_t)vi[vv]) * vscale;
                    u[vv] = (vf - u[vv]) * beta;
                    out[vv] = w[vv] + u[vv] * kq;
                }
            else
                for (vv = 0; vv < vd; vv++) {
                    u[vv] = (v[vv] - u[vv]) * beta;   /* u becomes delta */
                    out[vv] = w[vv] + u[vv] * kq;
                }
        } else {
            LZ_FCX(LZ_FC_RECUR, (long)vd * 4, (long)vd * 2, 0,
                   vi ? (long)vd : 0, 0);
            if (vi)
                for (vv = 0; vv < vd; vv++) {
                    float vf;                 /* see the gt==1 twin above */
                    if (!f32_scale_pow2(lz_i32f((int32_t)vi[vv]), vs_e, &vf))
                        vf = lz_i32f((int32_t)vi[vv]) * vscale;
                    u[vv] = (vf - gt * u[vv]) * beta;
                    out[vv] = gt * w[vv] + u[vv] * kq;
                }
            else
                for (vv = 0; vv < vd; vv++) {
                    u[vv] = (v[vv] - gt * u[vv]) * beta;
                    out[vv] = gt * w[vv] + u[vv] * kq;
                }
        }
    }

    /* Pass 2: read the OLD state from sq_in/ss_in (same as pass 1 - the
       decay/update formula below needs the pre-step value), write the
       NEW state to sq_out/ss_out. When in==out (ordinary decode) this
       is bit-identical to the single-pointer form: every read for row
       kk completes before that row's one write, so self-aliasing is
       safe and the split is just an option, not a requirement (see this
       function's own header comment). New values first land
       in one stack row buffer (vd entries, stays in L1), then the whole
       row is quantized back at once - reusing lz_quantize_q8, the same
       rounding semantics as everywhere else. Called per ROW, not per
       32-group: the 0.8B has 18 layers x 16 heads x 128 rows per
       token; per-group calls would be 147K function calls whose
       overhead alone eats the pass savings. */
#if LZ_GDN_FIXED
    if (lz_gdn_p2_mode()) {
        gdn_p2_fixed_rows(sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                          sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                          ss_in, ss_out, u, k, gvec, gt, kd, vd);
        /* docs/x87-gcc-reg-stack-leak.md: the row/group loop inside
           gdn_p2_fixed_rows is where the leak was bisected to. This is
           the finer of the two checkpoints; forward.c's per-layer one
           is the coarser one that first surfaced it. */
        LZ_X87_STACK_CHECK("gdn-p2-fixed-rows-return");
        return;
    }
#endif /* LZ_GDN_FIXED */

    for (kk = 0; kk < kd; kk++) {
        float buf[LZ_GDN_MAX_VD];
        const int8_t *row_in = sq_in + (size_t)kk * vd;
        int8_t *row_out = sq_out + (size_t)kk * vd;
        const float *rs_in = ss_in + (size_t)kk * ng;
        float *rs_out = ss_out + (size_t)kk * ng;
        float kc = k[kk];
        float dec = gvec ? gvec[kk] : gt;   /* the one family difference */
#if LZ_GDN_STATE_2PLANE
        const int8_t *row_lo_in = sq2_in + (size_t)kk * vd;
        int8_t *row_lo_out = sq2_out + (size_t)kk * vd;
#endif /* LZ_GDN_STATE_2PLANE */
        for (gg = 0; gg < ng; gg++) {
            const int8_t *p = row_in + gg * 32;
            const float *dd = u + gg * 32;
            float *bb = buf + gg * 32;
            float scg = rs_in[gg] * dec;
#if LZ_GDN_STATE_2PLANE
            const int8_t *pl = row_lo_in + gg * 32;
            for (j = 0; j < 32; j++)
                bb[j] = (lz_i32f((int32_t)p[j]) +
                         lz_i32f((int32_t)pl[j]) * (1.0f / LZ_GDN_LO_SCALE)) * scg +
                        kc * dd[j];
#else
            for (j = 0; j < 32; j++)
                bb[j] = lz_i32f((int32_t)p[j]) * scg + kc * dd[j];
#endif /* LZ_GDN_STATE_2PLANE */
        }
#if LZ_GDN_STATE_2PLANE
        /* Reads zero when the recurrence is fixed: the work is billed
           through gdn_p2_fixed_rows, not here - it did not vanish. */
        lz_gdn_quantize_2p(buf, vd, 32, row_out, row_lo_out, rs_out,
                           LZ_FC_RECUR);
#else
        /* No LZ_FCX here: lz_quantize_q8 bills its own three branches -
           SIMD, scalar-multiply, scalar-divide - under LZ_FC_QUANT (see
           its own comment in ops.c). An approximation at this site
           would double-count it. */
        lz_quantize_q8(buf, vd, 32, row_out, rs_out);
#endif /* LZ_GDN_STATE_2PLANE */
    }
}

void lz_gdn_step(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 float gt, float beta, int kd, int vd) {
    lz_recur_step(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                  sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                  ss_in, ss_out, q, k, v, (const short *)0, 0.0f,
                  NULL, gt, beta, kd, vd);
}

/* gt = 1.0f is not a placeholder. KDA's decay is already inside pass 1's
   coefficients, so the only surviving uses of the scalar gt are `gt*u`,
   `gt*w` and `rs*gt` - and multiplying by 1.0f is exact. This call
   therefore reproduces the hand-written KDA recurrence bit for bit, not
   approximately; the 10-hash baseline confirms it. */
void lz_kda_step(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 const float *gvec, float beta, int kd, int vd) {
    lz_recur_step(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                  sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                  ss_in, ss_out, q, k, v, (const short *)0, 0.0f,
                  gvec, 1.0f, beta, kd, vd);
}

/* Same call with v arriving as int16 at `vscale` (int-pipeline 9.4).
   A separate entry rather than two more parameters for the reason
   lz_causal_conv1d_step_fixed_o16 is one: five of the six callers of
   lz_kda_step are probes. `vi` NULL is exactly lz_kda_step. */
void lz_kda_step_vi16(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const short *vi,
                 float vscale, const float *gvec, float beta,
                 int kd, int vd) {
    lz_recur_step(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                  sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                  ss_in, ss_out, q, k, (const float *)0, vi, vscale,
                  gvec, 1.0f, beta, kd, vd);
}
