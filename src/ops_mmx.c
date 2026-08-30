/* Every function that writes a %mm register - the MMX register file,
   which aliases the x87 stack. The ONLY translation unit built with
   -mmmx (and -msse, for the SSE1-on-%mm cell) on EITHER toolchain: on
   a 32-bit x87 target, gcc and clang both use MMX for an 8-byte block
   store when -mmmx is on even with no MMX intrinsic anywhere nearby in
   the source, and neither issues emms before the x87 code that follows
   - which leaves the tag word full and makes the next fld overflow to
   real indefinite (0xFFC00000). Keeping every %mm-touching function
   out of ops.c is what makes "ops.c is never built with -mmmx" safe;
   see src/ops_mmx.h for the macro that tells ops.c's dispatch code
   these kernels still exist.

   Watcom compiles this file too - $(ENG_MMX_GCC) in the Makefile is
   misnamed; see its own comment. Watcom's #pragma aux bodies live here,
   not in ops.c or the ops_kernel_*.h headers, because -Nr/-Ns is a hard
   ceiling wcc386's own per-function ".586"/".686" directives cannot
   exceed - ops.c staying at the i486/i386 floor and this file's
   hand-written kernels needing a higher one are incompatible unless the
   kernels live in a separate TU with its own ceiling. The ARM
   cross-build must still never see this file: it has no MMX register
   file at all.

   Kernels do NOT emit emms themselves - the caller emits one after its
   loop, so a row loop pays one instead of one per row. That contract is the same in every TU that
   compiles %mm instructions; only the translation unit differs.

   A #pragma aux body has no address, so a kernel named only inside its
   pragma cannot be called from another TU - a probe, a cross-TU
   caller, or a linker that needs a symbol. The thin wrappers below
   (lz_dot128_x16_mmx_w, dot32_q16_mmx, etc.) exist for that reason.
   Any new kernel must arrive with one, or kernel_reach_gate reddens. */
#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include <string.h>

#include "ops_mmx.h"

#if defined(LZ_MMX_TU)

#if defined(__WATCOMC__)
#include "mmx_compat.h"
#include "ops.h"               /* LZ_KERNEL_* - the tier constants
                           wsum_pair_tier below compares g_kernel
                           against. ops_kernel_shared.h declares
                           g_kernel but names the constants only inside
                           a macro, which expands at ITS call sites and
                           so never required them here before. */
#include "ops_sse.h"           /* lz_i32f_acc32_sse - the SSE1 tier of the
                           chunk fold. Between ops_mmx.h and ops_sse2.h,
                           like the ISA ladder: a reader scanning the
                           include block sees the tiers in order. */
#include "ops_sse2.h"          /* lz_wsum_pair_sse2 - the SSE2 tier of
                           the pair MAC the group loops below call. This
                           TU owns the loop for BOTH tiers because the
                           loop is the call boundary on Watcom (see
                           lz_wsum_group_mmx), so the tier test lives
                           here rather than at a caller that does not
                           exist. */
#include "ops_sched.h"         /* lz_i32facc_tier - LZ_I32F_ACC32 calls it.
                           Both ISA headers come first: that macro picks
                           its arm from the LZ_HAVE_I32FACC_* they
                           define. */
#include "ops_kernel_shared.h" /* lz_i32f, LZ_WSUM_CHUNK, LZ_SLOT_NEXT,
                           gdn_tail_row, p2_shift_of - shared with
                           ops.c and the other suffixed TUs, one
                           definition each rather than a per-TU copy
                           under a per-TU name. gcc's branch below needs
                           none of these five - see ops.c's unmoved
                           gdn_sum_gg/lz_attn_wsum_q8 gcc branches,
                           which own the equivalent work there instead. */

/* ---- GDN/KDA pass 1, 4-lane table contraction - whole in this TU,
   called from ops.c's gdn_sum_gg. It is the real call boundary: a
   function called once per 32-lane group from its own caller, doing 8
   sub-window lz_gdn1_x4_asm calls internally. The whole body here adds
   no new call site - ops.c's call count to it is unchanged, only the
   destination TU differs. gdn_tail_row comes from
   src/ops_kernel_shared.h. */

/* Eight 4-lane sub-windows cover this gg's 32 vv lanes.

   The emms belongs HERE, once per gg - not per sub-window, and not after
   the whole loop. Hoisting the emms out is allowed only when the loop
   body has no x87, and that precondition fails one level up: the
   next gg's gdn_build_table runs q8_round and 32767/amax on x87 right
   after this returns. Hoisting it would reopen the trap that broke the
   mmx tier. */
extern void lz_gdn1_x4_asm(const int8_t *rowA, int32_t vd,
                           const int16_t *ctab, int32_t kpairs,
                           int32_t *out8);
#pragma aux lz_gdn1_x4_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "pxor      mm2, mm2" \
    "pxor      mm7, mm7" \
    "test      ecx, ecx" \
    "jz        LZ_GDN1_DONE" \
    "LZ_GDN1_LOOP:" \
    "movd      mm3, [eax]" \
    "punpcklbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movd      mm4, [eax+edx]" \
    "punpcklbw mm4, mm4" \
    "psraw     mm4, 8" \
    "movq      mm5, mm3" \
    "punpcklwd mm5, mm4" \
    "punpckhwd mm3, mm4" \
    "movq      mm6, mm5" \
    "pmaddwd   mm6, [ebx]" \
    "pmaddwd   mm5, [ebx+8]" \
    "paddd     mm0, mm6" \
    "paddd     mm2, mm5" \
    "movq      mm6, mm3" \
    "pmaddwd   mm6, [ebx]" \
    "pmaddwd   mm3, [ebx+8]" \
    "paddd     mm1, mm6" \
    "paddd     mm7, mm3" \
    "lea       eax, [eax+edx*2]" \
    "add       ebx, 16" \
    "dec       ecx" \
    "jnz       LZ_GDN1_LOOP" \
    "LZ_GDN1_DONE:" \
    "movq      [esi], mm0" \
    "movq      [esi+8], mm1" \
    "movq      [esi+16], mm2" \
    "movq      [esi+24], mm7" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [__eax __ebx __ecx 8087]

/* gdn_sum_gg's MMX branch, named lz_gdn_sum_gg_mmx here - ops.c's
   non-MMX branch keeps the bare name; this is the branch behind
   #ifdef LZ_GDN_HAVE_MMX in gdn_sum_gg. */
void lz_gdn_sum_gg_mmx(const int8_t *sq_gg, int kd, int vd,
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
    _mm_empty();
}

/* ---- attention scoring's weighted-sum dot, GROUP granularity - in
   this TU, called from ops.c's lz_attn_wsum_q8. LZ_ATTN_FIXED defaults
   to 3 (ops.h) so this is a live, always-compiled path, not a
   reservation - AUTO just does not select it on an SSE-capable host,
   the same way the GDN pass-2 fixed tier is compiled everywhere but
   AUTO-selected only where float is slow.

   LZ_WSUM_CHUNK, LZ_SLOT_NEXT and lz_i32f all come from
   src/ops_kernel_shared.h - LZ_WSUM_CHUNK/LZ_SLOT_NEXT in particular
   drive a ring-buffer walk that MUST stay in lock-step with ops.c's
   own copy of the same walk, which a duplicated literal cannot
   guarantee the way a single #define can. */

/* Originally generated, hand-edited since - if you regenerate, diff
   the generator's output against what is here first.
   acc32[d] += rowA[d]*coef[0] + rowB[d]*coef[1] for d in [0,32).
   coef is [ckA,ckB,ckA,ckB] so one pmaddwd covers a 2-int32 lane pair.
   Reads [eax]/[edx]/[ecx], writes only [ebx] and MMX - every offset is a
   literal, no GP register is modified, so `__modify [8087]` is the truth
   and not an `__exact []`-style lie: modify has to tell the truth or
   the compiler keeps a live value in a register this body overwrites.
   Emits no emms; the driver does one per chunk. */
extern void lz_wsum_pair_asm(const int8_t *rowA, const int8_t *rowB,
                const int16_t *coef, int32_t *acc32);
#pragma aux lz_wsum_pair_asm = \
    ".586" \
    "movq      mm0, [ecx]" \
    "movq      mm1, [eax]" \
    "movq      mm2, [edx]" \
    "movq      mm3, mm1" \
    "movq      mm4, mm2" \
    "punpcklbw mm3, mm3" \
    "punpcklbw mm4, mm4" \
    "psraw     mm3, 8" \
    "psraw     mm4, 8" \
    "movq      mm5, mm3" \
    "punpcklwd mm5, mm4" \
    "punpckhwd mm3, mm4" \
    "pmaddwd   mm5, mm0" \
    "pmaddwd   mm3, mm0" \
    "movq      mm6, [ebx]" \
    "movq      mm7, [ebx+8]" \
    "paddd     mm6, mm5" \
    "paddd     mm7, mm3" \
    "movq      [ebx], mm6" \
    "movq      [ebx+8], mm7" \
    "movq      mm3, mm1" \
    "movq      mm4, mm2" \
    "punpckhbw mm3, mm3" \
    "punpckhbw mm4, mm4" \
    "psraw     mm3, 8" \
    "psraw     mm4, 8" \
    "movq      mm5, mm3" \
    "punpcklwd mm5, mm4" \
    "punpckhwd mm3, mm4" \
    "pmaddwd   mm5, mm0" \
    "pmaddwd   mm3, mm0" \
    "movq      mm6, [ebx+16]" \
    "movq      mm7, [ebx+24]" \
    "paddd     mm6, mm5" \
    "paddd     mm7, mm3" \
    "movq      [ebx+16], mm6" \
    "movq      [ebx+24], mm7" \
    "movq      mm1, [eax+8]" \
    "movq      mm2, [edx+8]" \
    "movq      mm3, mm1" \
    "movq      mm4, mm2" \
    "punpcklbw mm3, mm3" \
    "punpcklbw mm4, mm4" \
    "psraw     mm3, 8" \
    "psraw     mm4, 8" \
    "movq      mm5, mm3" \
    "punpcklwd mm5, mm4" \
    "punpckhwd mm3, mm4" \
    "pmaddwd   mm5, mm0" \
    "pmaddwd   mm3, mm0" \
    "movq      mm6, [ebx+32]" \
    "movq      mm7, [ebx+40]" \
    "paddd     mm6, mm5" \
    "paddd     mm7, mm3" \
    "movq      [ebx+32], mm6" \
    "movq      [ebx+40], mm7" \
    "movq      mm3, mm1" \
    "movq      mm4, mm2" \
    "punpckhbw mm3, mm3" \
    "punpckhbw mm4, mm4" \
    "psraw     mm3, 8" \
    "psraw     mm4, 8" \
    "movq      mm5, mm3" \
    "punpcklwd mm5, mm4" \
    "punpckhwd mm3, mm4" \
    "pmaddwd   mm5, mm0" \
    "pmaddwd   mm3, mm0" \
    "movq      mm6, [ebx+48]" \
    "movq      mm7, [ebx+56]" \
    "paddd     mm6, mm5" \
    "paddd     mm7, mm3" \
    "movq      [ebx+48], mm6" \
    "movq      [ebx+56], mm7" \
    "movq      mm1, [eax+16]" \
    "movq      mm2, [edx+16]" \
    "movq      mm3, mm1" \
    "movq      mm4, mm2" \
    "punpcklbw mm3, mm3" \
    "punpcklbw mm4, mm4" \
    "psraw     mm3, 8" \
    "psraw     mm4, 8" \
    "movq      mm5, mm3" \
    "punpcklwd mm5, mm4" \
    "punpckhwd mm3, mm4" \
    "pmaddwd   mm5, mm0" \
    "pmaddwd   mm3, mm0" \
    "movq      mm6, [ebx+64]" \
    "movq      mm7, [ebx+72]" \
    "paddd     mm6, mm5" \
    "paddd     mm7, mm3" \
    "movq      [ebx+64], mm6" \
    "movq      [ebx+72], mm7" \
    "movq      mm3, mm1" \
    "movq      mm4, mm2" \
    "punpckhbw mm3, mm3" \
    "punpckhbw mm4, mm4" \
    "psraw     mm3, 8" \
    "psraw     mm4, 8" \
    "movq      mm5, mm3" \
    "punpcklwd mm5, mm4" \
    "punpckhwd mm3, mm4" \
    "pmaddwd   mm5, mm0" \
    "pmaddwd   mm3, mm0" \
    "movq      mm6, [ebx+80]" \
    "movq      mm7, [ebx+88]" \
    "paddd     mm6, mm5" \
    "paddd     mm7, mm3" \
    "movq      [ebx+80], mm6" \
    "movq      [ebx+88], mm7" \
    "movq      mm1, [eax+24]" \
    "movq      mm2, [edx+24]" \
    "movq      mm3, mm1" \
    "movq      mm4, mm2" \
    "punpcklbw mm3, mm3" \
    "punpcklbw mm4, mm4" \
    "psraw     mm3, 8" \
    "psraw     mm4, 8" \
    "movq      mm5, mm3" \
    "punpcklwd mm5, mm4" \
    "punpckhwd mm3, mm4" \
    "pmaddwd   mm5, mm0" \
    "pmaddwd   mm3, mm0" \
    "movq      mm6, [ebx+96]" \
    "movq      mm7, [ebx+104]" \
    "paddd     mm6, mm5" \
    "paddd     mm7, mm3" \
    "movq      [ebx+96], mm6" \
    "movq      [ebx+104], mm7" \
    "movq      mm3, mm1" \
    "movq      mm4, mm2" \
    "punpckhbw mm3, mm3" \
    "punpckhbw mm4, mm4" \
    "psraw     mm3, 8" \
    "psraw     mm4, 8" \
    "movq      mm5, mm3" \
    "punpcklwd mm5, mm4" \
    "punpckhwd mm3, mm4" \
    "pmaddwd   mm5, mm0" \
    "pmaddwd   mm3, mm0" \
    "movq      mm6, [ebx+112]" \
    "movq      mm7, [ebx+120]" \
    "paddd     mm6, mm5" \
    "paddd     mm7, mm3" \
    "movq      [ebx+112], mm6" \
    "movq      [ebx+120], mm7" \
    __parm [__eax] [__edx] [__ecx] [__ebx] \
    __modify [8087]

/* The MMX pair MAC's address. Its SSE2 twin has had one
   (lz_wsum_pair_sse2) since the tier split; this side did not, so the
   two could never be compared even though they are required to agree
   bit for bit. build/wsum_pair_probe.c does that now. */
void lz_wsum_pair_mmx_w(const int8_t *rowA, const int8_t *rowB,
                        const int16_t *coef, int32_t *acc32) {
    lz_wsum_pair_asm(rowA, rowB, coef, acc32);
}

/* The x86 tier split for the pair MAC, and it has to be HERE rather
   than in src/ops_gdn.c the way the gcc build's is: on Watcom the group
   loop below IS the call boundary (ops.c's lz_attn_wsum_q8 calls
   lz_wsum_group_mmx and never sees a pair), so there is no caller
   outside this file to put the test in.

   One branch per row PAIR, against ~56 instructions of work - the same
   trade the ARM tier split makes at lz_wsum_pair_arm, and for the same
   reason an indirect call would be worse. g_kernel alone, no CPUID:
   lz_kernel_select cannot leave LZ_KERNEL_SSE2 set on a machine that
   lacks it. */
#if defined(LZ_ATTN_SSE2_EXTERN)
static void wsum_pair_tier(const int8_t *rowA, const int8_t *rowB,
                           const int16_t *coef, int32_t *acc32) {
    if (g_kernel == LZ_KERNEL_SSE2) {
        lz_wsum_pair_sse2(rowA, rowB, coef, acc32);
        return;
    }
    lz_wsum_pair_asm(rowA, rowB, coef, acc32);
}
#else
#define wsum_pair_tier(a, b, c, d) lz_wsum_pair_asm(a, b, c, d)
#endif /* LZ_ATTN_SSE2_EXTERN */

/* lz_wsum_group_mmx(): the ENTIRE multi-chunk second walk for one
   group - all of LZ_WSUM_CHUNK's chunks, all pairs, the ring-buffer
   slot walk, the per-chunk emms and lz_i32f fold - behind ONE call
   per group from ops.c's lz_attn_wsum_q8. T is context-length-
   dependent and unbounded (up to 262144 on this model), so amortizing
   at group granularity rather than per-pair or per-chunk is what
   keeps the call count O(ng) instead of O(T). */
void lz_wsum_group_mmx(const int8_t *vc, int kvd, int g, int sink, int ring,
                       const int16_t *cq, int T, float *accf) {
    int32_t acc32[32];
    int d, t0, slot = 0;

    for (d = 0; d < 32; d++) accf[d] = 0.0f;
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
            wsum_pair_tier(rowA, rowB, coef, acc32);
        }
        if (tp < t0 + tn) {            /* odd leftover row in this chunk */
            int16_t coef[4];
            const int8_t *rowA = vc + (size_t)slot * kvd + (size_t)g * 32;
            coef[0] = cq[tp]; coef[1] = 0;
            coef[2] = cq[tp]; coef[3] = 0;
            wsum_pair_tier(rowA, rowA, coef, acc32);
            LZ_SLOT_NEXT(slot, sink, ring);
        }
        _mm_empty();      /* no x87 inside the row-pair loop above */
        /* The fold, at the SSE2 tier when the machine has one. Asked
           here rather than at the caller because THIS loop is what the
           tier changes: the row-pair walk above stays MMX either way -
           a P4 running this kernel has cvtdq2ps for the fold and no
           reason to spend lz_i32f's x87 split form on it. The emms
           above has already run, so the cell owes nothing. */
        LZ_I32F_ACC32(accf, acc32, d);
    }
}

/* int-output twin of lz_wsum_group_mmx: the same whole multi-chunk
   second walk, but the per-chunk int32 sums fold into an int64
   accumulator instead of a float one, so the caller gets the exact
   full-context integer weighted sum (the float fold rounds each chunk
   at 24 bits - the precision loss the attention int path exists to
   remove). Integer addition is exact under reassociation, so this is
   bit-identical to the scalar twin lz_attn_wsum_q8_int uses on non-Watcom
   builds. */
void lz_wsum_group_mmx_int(const int8_t *vc, int kvd, int g, int sink, int ring,
                           const int16_t *cq, int T, int64_t *acc64) {
    int32_t acc32[32];
    int d, t0, slot = 0;

    for (d = 0; d < 32; d++) acc64[d] = 0;
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
            wsum_pair_tier(rowA, rowB, coef, acc32);
        }
        if (tp < t0 + tn) {            /* odd leftover row in this chunk */
            int16_t coef[4];
            const int8_t *rowA = vc + (size_t)slot * kvd + (size_t)g * 32;
            coef[0] = cq[tp]; coef[1] = 0;
            coef[2] = cq[tp]; coef[3] = 0;
            wsum_pair_tier(rowA, rowA, coef, acc32);
            LZ_SLOT_NEXT(slot, sink, ring);
        }
        _mm_empty();      /* no x87 inside the row-pair loop above */
        for (d = 0; d < 32; d++) acc64[d] += acc32[d];
    }
}

/* ---- GDN/KDA fixed-point pass 2, MMX tier - in this TU, not
   src/ops_kernel_p2.h. Algorithm documentation (WHAT THE OPERATOR IS,
   ISA INVENTORY, why the lo clamp is a saturating subtract) is in
   that header, unaffected by the code's location. */

#include "ops_p2_blk.h"

/* 16-byte-aligned scratch, by hand: neither compiler can be asked for
   it here (Watcom is built with -zp4, capping member alignment at 4;
   C89/C99 have no alignment specifier), and SSE2's movdqa faults on a
   misaligned address. ONE buffer for the whole engine, shared with
   src/ops_mmx_sse.c's lz_p2_rows_sse across the TU boundary - the
   split step there reads what the mul32 step here wrote, so both need
   the SAME pointer, not merely the same layout. Nothing re-enters
   between the two calls a row makes. */
lz_p2_blk *p2_blk(void) {
    static char raw[sizeof(lz_p2_blk) + 15];
    size_t off = (size_t)((uintptr_t)(void *)raw & 15u);
    return (lz_p2_blk *)(void *)(raw + ((16u - off) & 15u));
}

/* p2_shift_of: defined in src/ops_kernel_shared.h. */

/* PASS A: blk->a[0..31] = m1*H + m2*dq, and blk->amax[0..3] = four
   partial maxima of |A| (the caller folds the four; doing it in MMX
   would cost more than the three scalar compares it saves).

   Per quad of elements:
     movd/punpcklbw(zero,hi)   hi<<8 without a psraw - putting a byte in
                               the high half of a word IS a multiply by
                               256, sign and all (byte 0xFF becomes
                               0xFF00 = -256). Same trick the Q8 matmul
                               kernels use on the weight side.
     punpcklbw(lo,lo)+psraw 8  lo sign-extended, the ordinary way
     paddw                     H, still int16 because |hi*256+lo| <=
                               32639 - which is why LZ_GDN_LO_SCALE had
                               to stay at 256 and the planes clamp at
                               +-127
     punpcklwd/punpckhwd       interleave H with dq so each dword holds
                               the pair (H, dq) pmaddwd will contract
     pmaddwd x2                A, four elements, no overflow: the bound
                               is 32767*32639 + 32767*32767 = 2.143e9,
                               inside int32 by 4e6. That margin is the
                               reason p2_group_norm clamps m to +-32767
                               rather than letting q8_round return 32768.
     psrad 31 / pxor / psubd   |A|, the branchless two's-complement form
     pcmpgtd/pand/pandn/por    the running max, five instructions per
                               two lanes because MMX has no pmaxsd

   eax=hi edx=lo ebx=dq ecx=blk. Single-plane callers pass a zeroed lo
   buffer rather than getting a second kernel: H = hi*256 + 0 is exactly
   the single-plane H, so one kernel serves both and cannot drift. */
extern void lz_p2_mul32_mmx_asm(const signed char *hi, const signed char *lo,
                                const short *dq, void *blk);
#pragma aux lz_p2_mul32_mmx_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "pxor      mm2, mm2" \
    "movq      mm3, [ecx]" \
    \
    "movd      mm4, [eax]" \
    "movd      mm5, [edx]" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "punpcklbw mm5, mm5" \
    "psraw     mm5, 8" \
    "paddw     mm6, mm5" \
    "movq      mm7, [ebx]" \
    "movq      mm4, mm6" \
    "punpcklwd mm4, mm7" \
    "punpckhwd mm6, mm7" \
    "pmaddwd   mm4, mm3" \
    "pmaddwd   mm6, mm3" \
    "movq      [ecx+96], mm4" \
    "movq      [ecx+104], mm6" \
    "movq      mm5, mm4" \
    "psrad     mm5, 31" \
    "pxor      mm4, mm5" \
    "psubd     mm4, mm5" \
    "movq      mm7, mm6" \
    "psrad     mm7, 31" \
    "pxor      mm6, mm7" \
    "psubd     mm6, mm7" \
    "movq      mm5, mm1" \
    "pcmpgtd   mm5, mm4" \
    "pand      mm1, mm5" \
    "pandn     mm5, mm4" \
    "por       mm1, mm5" \
    "movq      mm7, mm2" \
    "pcmpgtd   mm7, mm6" \
    "pand      mm2, mm7" \
    "pandn     mm7, mm6" \
    "por       mm2, mm7" \
    \
    "movd      mm4, [eax+4]" \
    "movd      mm5, [edx+4]" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "punpcklbw mm5, mm5" \
    "psraw     mm5, 8" \
    "paddw     mm6, mm5" \
    "movq      mm7, [ebx+8]" \
    "movq      mm4, mm6" \
    "punpcklwd mm4, mm7" \
    "punpckhwd mm6, mm7" \
    "pmaddwd   mm4, mm3" \
    "pmaddwd   mm6, mm3" \
    "movq      [ecx+112], mm4" \
    "movq      [ecx+120], mm6" \
    "movq      mm5, mm4" \
    "psrad     mm5, 31" \
    "pxor      mm4, mm5" \
    "psubd     mm4, mm5" \
    "movq      mm7, mm6" \
    "psrad     mm7, 31" \
    "pxor      mm6, mm7" \
    "psubd     mm6, mm7" \
    "movq      mm5, mm1" \
    "pcmpgtd   mm5, mm4" \
    "pand      mm1, mm5" \
    "pandn     mm5, mm4" \
    "por       mm1, mm5" \
    "movq      mm7, mm2" \
    "pcmpgtd   mm7, mm6" \
    "pand      mm2, mm7" \
    "pandn     mm7, mm6" \
    "por       mm2, mm7" \
    \
    "movd      mm4, [eax+8]" \
    "movd      mm5, [edx+8]" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "punpcklbw mm5, mm5" \
    "psraw     mm5, 8" \
    "paddw     mm6, mm5" \
    "movq      mm7, [ebx+16]" \
    "movq      mm4, mm6" \
    "punpcklwd mm4, mm7" \
    "punpckhwd mm6, mm7" \
    "pmaddwd   mm4, mm3" \
    "pmaddwd   mm6, mm3" \
    "movq      [ecx+128], mm4" \
    "movq      [ecx+136], mm6" \
    "movq      mm5, mm4" \
    "psrad     mm5, 31" \
    "pxor      mm4, mm5" \
    "psubd     mm4, mm5" \
    "movq      mm7, mm6" \
    "psrad     mm7, 31" \
    "pxor      mm6, mm7" \
    "psubd     mm6, mm7" \
    "movq      mm5, mm1" \
    "pcmpgtd   mm5, mm4" \
    "pand      mm1, mm5" \
    "pandn     mm5, mm4" \
    "por       mm1, mm5" \
    "movq      mm7, mm2" \
    "pcmpgtd   mm7, mm6" \
    "pand      mm2, mm7" \
    "pandn     mm7, mm6" \
    "por       mm2, mm7" \
    \
    "movd      mm4, [eax+12]" \
    "movd      mm5, [edx+12]" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "punpcklbw mm5, mm5" \
    "psraw     mm5, 8" \
    "paddw     mm6, mm5" \
    "movq      mm7, [ebx+24]" \
    "movq      mm4, mm6" \
    "punpcklwd mm4, mm7" \
    "punpckhwd mm6, mm7" \
    "pmaddwd   mm4, mm3" \
    "pmaddwd   mm6, mm3" \
    "movq      [ecx+144], mm4" \
    "movq      [ecx+152], mm6" \
    "movq      mm5, mm4" \
    "psrad     mm5, 31" \
    "pxor      mm4, mm5" \
    "psubd     mm4, mm5" \
    "movq      mm7, mm6" \
    "psrad     mm7, 31" \
    "pxor      mm6, mm7" \
    "psubd     mm6, mm7" \
    "movq      mm5, mm1" \
    "pcmpgtd   mm5, mm4" \
    "pand      mm1, mm5" \
    "pandn     mm5, mm4" \
    "por       mm1, mm5" \
    "movq      mm7, mm2" \
    "pcmpgtd   mm7, mm6" \
    "pand      mm2, mm7" \
    "pandn     mm7, mm6" \
    "por       mm2, mm7" \
    \
    "movd      mm4, [eax+16]" \
    "movd      mm5, [edx+16]" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "punpcklbw mm5, mm5" \
    "psraw     mm5, 8" \
    "paddw     mm6, mm5" \
    "movq      mm7, [ebx+32]" \
    "movq      mm4, mm6" \
    "punpcklwd mm4, mm7" \
    "punpckhwd mm6, mm7" \
    "pmaddwd   mm4, mm3" \
    "pmaddwd   mm6, mm3" \
    "movq      [ecx+160], mm4" \
    "movq      [ecx+168], mm6" \
    "movq      mm5, mm4" \
    "psrad     mm5, 31" \
    "pxor      mm4, mm5" \
    "psubd     mm4, mm5" \
    "movq      mm7, mm6" \
    "psrad     mm7, 31" \
    "pxor      mm6, mm7" \
    "psubd     mm6, mm7" \
    "movq      mm5, mm1" \
    "pcmpgtd   mm5, mm4" \
    "pand      mm1, mm5" \
    "pandn     mm5, mm4" \
    "por       mm1, mm5" \
    "movq      mm7, mm2" \
    "pcmpgtd   mm7, mm6" \
    "pand      mm2, mm7" \
    "pandn     mm7, mm6" \
    "por       mm2, mm7" \
    \
    "movd      mm4, [eax+20]" \
    "movd      mm5, [edx+20]" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "punpcklbw mm5, mm5" \
    "psraw     mm5, 8" \
    "paddw     mm6, mm5" \
    "movq      mm7, [ebx+40]" \
    "movq      mm4, mm6" \
    "punpcklwd mm4, mm7" \
    "punpckhwd mm6, mm7" \
    "pmaddwd   mm4, mm3" \
    "pmaddwd   mm6, mm3" \
    "movq      [ecx+176], mm4" \
    "movq      [ecx+184], mm6" \
    "movq      mm5, mm4" \
    "psrad     mm5, 31" \
    "pxor      mm4, mm5" \
    "psubd     mm4, mm5" \
    "movq      mm7, mm6" \
    "psrad     mm7, 31" \
    "pxor      mm6, mm7" \
    "psubd     mm6, mm7" \
    "movq      mm5, mm1" \
    "pcmpgtd   mm5, mm4" \
    "pand      mm1, mm5" \
    "pandn     mm5, mm4" \
    "por       mm1, mm5" \
    "movq      mm7, mm2" \
    "pcmpgtd   mm7, mm6" \
    "pand      mm2, mm7" \
    "pandn     mm7, mm6" \
    "por       mm2, mm7" \
    \
    "movd      mm4, [eax+24]" \
    "movd      mm5, [edx+24]" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "punpcklbw mm5, mm5" \
    "psraw     mm5, 8" \
    "paddw     mm6, mm5" \
    "movq      mm7, [ebx+48]" \
    "movq      mm4, mm6" \
    "punpcklwd mm4, mm7" \
    "punpckhwd mm6, mm7" \
    "pmaddwd   mm4, mm3" \
    "pmaddwd   mm6, mm3" \
    "movq      [ecx+192], mm4" \
    "movq      [ecx+200], mm6" \
    "movq      mm5, mm4" \
    "psrad     mm5, 31" \
    "pxor      mm4, mm5" \
    "psubd     mm4, mm5" \
    "movq      mm7, mm6" \
    "psrad     mm7, 31" \
    "pxor      mm6, mm7" \
    "psubd     mm6, mm7" \
    "movq      mm5, mm1" \
    "pcmpgtd   mm5, mm4" \
    "pand      mm1, mm5" \
    "pandn     mm5, mm4" \
    "por       mm1, mm5" \
    "movq      mm7, mm2" \
    "pcmpgtd   mm7, mm6" \
    "pand      mm2, mm7" \
    "pandn     mm7, mm6" \
    "por       mm2, mm7" \
    \
    "movd      mm4, [eax+28]" \
    "movd      mm5, [edx+28]" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "punpcklbw mm5, mm5" \
    "psraw     mm5, 8" \
    "paddw     mm6, mm5" \
    "movq      mm7, [ebx+56]" \
    "movq      mm4, mm6" \
    "punpcklwd mm4, mm7" \
    "punpckhwd mm6, mm7" \
    "pmaddwd   mm4, mm3" \
    "pmaddwd   mm6, mm3" \
    "movq      [ecx+208], mm4" \
    "movq      [ecx+216], mm6" \
    "movq      mm5, mm4" \
    "psrad     mm5, 31" \
    "pxor      mm4, mm5" \
    "psubd     mm4, mm5" \
    "movq      mm7, mm6" \
    "psrad     mm7, 31" \
    "pxor      mm6, mm7" \
    "psubd     mm6, mm7" \
    "movq      mm5, mm1" \
    "pcmpgtd   mm5, mm4" \
    "pand      mm1, mm5" \
    "pandn     mm5, mm4" \
    "por       mm1, mm5" \
    "movq      mm7, mm2" \
    "pcmpgtd   mm7, mm6" \
    "pand      mm2, mm7" \
    "pandn     mm7, mm6" \
    "por       mm2, mm7" \
    \
    "movq      [ecx+16], mm1" \
    "movq      [ecx+24], mm2" \
    __parm [__eax] [__edx] [__ebx] [__ecx] \
    __modify [8087]

/* Real, addressable wrapper - needed cross-TU by src/ops_mmx_sse.c's
   lz_p2_rows_sse, which has no mul32 kernel of its own (SSE1 adds
   nothing to this half - see ops_kernel_p2.h's ISA inventory).
   lz_p2_rows_mmx below never calls through this: it inlines the
   pragma directly above, so this wrapper adds no overhead to the tier
   that does not already need it. */
void lz_p2_mul32_mmx(const int8_t *hi, const int8_t *lo,
                     const int16_t *dq, lz_p2_blk *blk) {
    lz_p2_mul32_mmx_asm((const signed char *)hi, (const signed char *)lo,
                        (const short *)dq, blk);
}

/* PASS B: the rescale, the split and the narrowing, blk->a[] -> two
   int8 planes. eax=blk edx=hi_out ebx=lo_out.

   The rescale is a shift with a rounding addend rather than a truncating
   one because truncation is a BIAS and bias adds linearly where
   quantization noise adds in quadrature - measured on this routine,
   5.88e-05 truncating against 4.16e-05 rounded. rnd and cnt are both
   read from memory, which costs nothing here and keeps sh out of a
   register the unrolled body has no room for.

   packssdw and packsswb both saturate, and neither one is ALLOWED to:
   |hn| <= 32513 and |hi| <= 127 are what the shift and the +-127 clamps
   guarantee. Saturation firing would mean the scalar path and this one
   disagree, which is exactly what the differential probe looks for.

   THE lo CLAMP IS THE ONE TRICK IN HERE. lo = hn - hi*256 lands in
   [-128, 127] by construction, and the scalar path clamps it to
   [-127, 127] to match what the float quantizer can emit. Only the
   single value -128 is affected, so instead of a four-instruction
   compare-and-select it is psubsw then paddsw by 32641: x - 32641
   saturates to -32768 only for x = -128 (x = -127 gives exactly -32768
   without saturating), and adding 32641 back returns x for every x >=
   -127 and -127 for x = -128. Two instructions, no mask register, and
   the high end is untouched because 127 - 32641 does not saturate. */
extern void lz_p2_split32_mmx_asm(const void *blk, signed char *hi_out,
                                  signed char *lo_out);
#pragma aux lz_p2_split32_mmx_asm = \
    ".586" \
    "movq      mm0, [eax+96]" \
    "movq      mm1, [eax+104]" \
    "movq      mm2, [eax+112]" \
    "movq      mm3, [eax+120]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "psubsw    mm0, [eax+80]" \
    "paddsw    mm0, [eax+80]" \
    "psubsw    mm2, [eax+80]" \
    "paddsw    mm2, [eax+80]" \
    "packsswb  mm1, mm3" \
    "movq      [edx], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx], mm0" \
    \
    "movq      mm0, [eax+128]" \
    "movq      mm1, [eax+136]" \
    "movq      mm2, [eax+144]" \
    "movq      mm3, [eax+152]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "psubsw    mm0, [eax+80]" \
    "paddsw    mm0, [eax+80]" \
    "psubsw    mm2, [eax+80]" \
    "paddsw    mm2, [eax+80]" \
    "packsswb  mm1, mm3" \
    "movq      [edx+8], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+8], mm0" \
    \
    "movq      mm0, [eax+160]" \
    "movq      mm1, [eax+168]" \
    "movq      mm2, [eax+176]" \
    "movq      mm3, [eax+184]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "psubsw    mm0, [eax+80]" \
    "paddsw    mm0, [eax+80]" \
    "psubsw    mm2, [eax+80]" \
    "paddsw    mm2, [eax+80]" \
    "packsswb  mm1, mm3" \
    "movq      [edx+16], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+16], mm0" \
    \
    "movq      mm0, [eax+192]" \
    "movq      mm1, [eax+200]" \
    "movq      mm2, [eax+208]" \
    "movq      mm3, [eax+216]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "psubsw    mm0, [eax+80]" \
    "paddsw    mm0, [eax+80]" \
    "psubsw    mm2, [eax+80]" \
    "paddsw    mm2, [eax+80]" \
    "packsswb  mm1, mm3" \
    "movq      [edx+24], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+24], mm0" \
    __parm [__eax] [__edx] [__ebx] \
    __modify [8087]

/* The split kernel's address. */
void lz_p2_split32_mmx_w(const void *blk, signed char *hi_out,
                         signed char *lo_out) {
    lz_p2_split32_mmx_asm(blk, hi_out, lo_out);
}

/* lz_p2_rows_mmx(): all of a row's groups, tier 1. One call per row
   from ops.c's gdn_p2_row_simd - see that function's Watcom branch
   for the full accounting. Both pragma-aux kernels above are
   textually visible in THIS translation unit, so this loop is fully
   inline; the row boundary is the only call. */
void lz_p2_rows_mmx(const int8_t *ph_row, const int8_t *pl_row,
                    int pl_stride, const int16_t *dq,
                    const int16_t (*mul)[4], int8_t *oh_row,
                    int8_t *ol_row, int ol_stride, int *shv, int ng) {
    lz_p2_blk *blk = p2_blk();
    int gg, j;
    const int8_t *plg = pl_row;
    int8_t *olg = ol_row;
    for (gg = 0; gg < ng; gg++, plg += pl_stride, olg += ol_stride) {
        int32_t amax;
        int sh;
        for (j = 0; j < 8; j += 2) {
            blk->mul[j]     = mul[gg][0];
            blk->mul[j + 1] = mul[gg][1];
        }
        lz_p2_mul32_mmx_asm((const signed char *)(ph_row + gg * 32),
                            (const signed char *)plg,
                            (const short *)(dq + gg * 32), blk);
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
        lz_p2_split32_mmx_asm(blk, oh_row + gg * 32, olg);
    }
}

/* ---- Q8 group-scale amax, MMX tier - in this TU; the wrapper
   lz_amax32_mmx_w sits between the pragma and LZ_AMAX_TAB
   (src/ops_kernel_amax.h). A #pragma aux body cannot have its address
   taken, so the table always needs a real function to point at, and
   that indirect call happens once per operator invocation regardless
   of which TU the wrapper's body lives in. */

/* Two lanes of uint32 in one MMX register. The mask is built in-register
   (pcmpeqd then psrld 1) rather than loaded, so the routine touches no
   memory but its input. Leaves MMX state dirty - the caller emits one
   emms after the group loop, never the kernel - one per loop rather
   than one per group. */
extern unsigned lz_amax32_mmx_asm(const float *x, int n);
#pragma aux lz_amax32_mmx_asm = \
    "pxor      mm0, mm0"                 \
    "pcmpeqd   mm1, mm1"                 \
    "psrld     mm1, 1"                   \
    "shr       edx, 1"                   \
    "jz        L2"                       \
    "L1:"                               \
    "movq      mm2, [eax]"               \
    "pand      mm2, mm1"                 \
    "movq      mm3, mm2"                 \
    "pcmpgtd   mm3, mm0"                 \
    "pand      mm2, mm3"                 \
    "pandn     mm3, mm0"                 \
    "por       mm2, mm3"                 \
    "movq      mm0, mm2"                 \
    "add       eax, 8"                   \
    "dec       edx"                      \
    "jnz       L1"                       \
    "L2:"                               \
    "movq      mm2, mm0"                 \
    "psrlq     mm2, 32"                  \
    "movq      mm3, mm0"                 \
    "pcmpgtd   mm3, mm2"                 \
    "pand      mm0, mm3"                 \
    "pandn     mm3, mm2"                 \
    "por       mm0, mm3"                 \
    "movd      eax, mm0"                 \
    parm [eax] [edx]                    \
    value [eax]                         \
    modify [eax edx 8087];

/* Thin wrapper so LZ_AMAX_TAB can hold this. The wrapper is the only
   caller, so the asm is still inlined once and the indirect call is
   one per operator invocation, not one per element. */
unsigned lz_amax32_mmx_w(const float *x, int n) {
    return lz_amax32_mmx_asm(x, n);
}

/* ---- Q8 activation rounding, the PACK HALF, MMX tier, Watcom twin.
   gcc twin: lz_q8round32_pack_mmx, the intrinsics body in the #else
   arm below - read that one for why this is a half, why packsswb's
   saturation IS the +127 clamp, and why pminsw/pmaxsw are unavailable
   here (they are SSE1, not MMX).

   mm7 holds -127 in every word for the whole run; the -127 floor is
   the pcmpgtw/pand/pandn/por idiom, four instructions per four lanes.
   Eight elements a block, four blocks, one emms at the exit - the
   caller's x87 pass has already run and its next x87 instruction comes
   after this returns. */
extern void lz_q8round32_pack_mmx_asm(const int32_t *q, int8_t *out);
#pragma aux lz_q8round32_pack_mmx_asm = \
    ".586"                              \
    "pcmpeqw   mm0, mm0"                  \
    "psrlw     mm0, 9"                    /* 127 per word */ \
    "pxor      mm1, mm1"                  \
    "psubw     mm1, mm0"                  /* -127 per word */ \
    "movq      mm0, mm1"                  \
    "movq      mm2, [eax]"                \
    "movq      mm3, [eax+8]"              \
    "packssdw  mm2, mm3"                 \
    "movq      mm4, [eax+16]"             \
    "movq      mm5, [eax+24]"             \
    "packssdw  mm4, mm5"                 \
    "movq      mm6, mm0"                  \
    "pcmpgtw   mm6, mm2"                  \
    "movq      mm7, mm6"                  \
    "pand      mm6, mm0"                  \
    "pandn     mm7, mm2"                  \
    "por       mm6, mm7"                  \
    "movq      mm3, mm0"                  \
    "pcmpgtw   mm3, mm4"                  \
    "movq      mm5, mm3"                  \
    "pand      mm3, mm0"                  \
    "pandn     mm5, mm4"                  \
    "por       mm3, mm5"                  \
    "packsswb  mm6, mm3"                 \
    "movq      [edx], mm6"                \
    "movq      mm2, [eax+32]"             \
    "movq      mm3, [eax+40]"             \
    "packssdw  mm2, mm3"                 \
    "movq      mm4, [eax+48]"             \
    "movq      mm5, [eax+56]"             \
    "packssdw  mm4, mm5"                 \
    "movq      mm6, mm0"                  \
    "pcmpgtw   mm6, mm2"                  \
    "movq      mm7, mm6"                  \
    "pand      mm6, mm0"                  \
    "pandn     mm7, mm2"                  \
    "por       mm6, mm7"                  \
    "movq      mm3, mm0"                  \
    "pcmpgtw   mm3, mm4"                  \
    "movq      mm5, mm3"                  \
    "pand      mm3, mm0"                  \
    "pandn     mm5, mm4"                  \
    "por       mm3, mm5"                  \
    "packsswb  mm6, mm3"                 \
    "movq      [edx+8], mm6"              \
    "movq      mm2, [eax+64]"             \
    "movq      mm3, [eax+72]"             \
    "packssdw  mm2, mm3"                 \
    "movq      mm4, [eax+80]"             \
    "movq      mm5, [eax+88]"             \
    "packssdw  mm4, mm5"                 \
    "movq      mm6, mm0"                  \
    "pcmpgtw   mm6, mm2"                  \
    "movq      mm7, mm6"                  \
    "pand      mm6, mm0"                  \
    "pandn     mm7, mm2"                  \
    "por       mm6, mm7"                  \
    "movq      mm3, mm0"                  \
    "pcmpgtw   mm3, mm4"                  \
    "movq      mm5, mm3"                  \
    "pand      mm3, mm0"                  \
    "pandn     mm5, mm4"                  \
    "por       mm3, mm5"                  \
    "packsswb  mm6, mm3"                 \
    "movq      [edx+16], mm6"             \
    "movq      mm2, [eax+96]"             \
    "movq      mm3, [eax+104]"            \
    "packssdw  mm2, mm3"                 \
    "movq      mm4, [eax+112]"            \
    "movq      mm5, [eax+120]"            \
    "packssdw  mm4, mm5"                 \
    "movq      mm6, mm0"                  \
    "pcmpgtw   mm6, mm2"                  \
    "movq      mm7, mm6"                  \
    "pand      mm6, mm0"                  \
    "pandn     mm7, mm2"                  \
    "por       mm6, mm7"                  \
    "movq      mm3, mm0"                  \
    "pcmpgtw   mm3, mm4"                  \
    "movq      mm5, mm3"                  \
    "pand      mm3, mm0"                  \
    "pandn     mm5, mm4"                  \
    "por       mm3, mm5"                  \
    "packsswb  mm6, mm3"                 \
    "movq      [edx+24], mm6"             \
    "emms"                              \
    __parm [__eax] [__edx]              \
    __modify [8087];

/* The real function the other translation units call: a #pragma aux
   body expands at its call site and has no address. */
void lz_q8round32_pack_mmx(const int32_t *q, int8_t *out) {
    lz_q8round32_pack_mmx_asm(q, out);
}

/* ---- norm_ss_fixed's pack-and-square half, MMX tier, Watcom twin.
   gcc twin: lz_norm_ss_pack_mmx in the #else arm - read that one for
   the clamp derivation and the overflow bound.

   FOUR ELEMENTS a pass, which is one %mm of int16 and one pmaddwd. The
   two int32 lanes it produces are drained to the caller's int64
   immediately: each is at most 2147352578 and a second add would
   overflow, and paddq is SSE2 even on %mm.

   ecx counts the eight passes, edi holds the running sum in memory
   because there is no spare register pair for it and one add/adc per
   four elements is cheaper than the register juggling would be.
   qout NULL is a branch per pass rather than two loop bodies: this
   kernel is called once per 32 elements, not per element, so the
   predictable test costs nothing beside the work it guards. */
extern void lz_norm_ss_pack_mmx_asm(const int32_t *q, short *qout,
                                    lz_i64 *acc);
#pragma aux lz_norm_ss_pack_mmx_asm = \
    ".586"                              \
    "pcmpeqw   mm0, mm0"                  \
    "psrlw     mm0, 1"                    /* 32767 per word */ \
    "pxor      mm1, mm1"                  \
    "psubw     mm1, mm0"                  /* -32767 per word */ \
    "movq      mm0, mm1"                  \
    "xor       ecx, ecx"                  \
    "mov       [ebx], ecx"                \
    "mov       [ebx+4], ecx"              \
    "mov       ecx, 8"                    \
    "nsploop:"                          \
    "movq      mm2, [eax]"                \
    "movq      mm3, [eax+8]"              \
    "packssdw  mm2, mm3"                 \
    "movq      mm4, mm0"                  \
    "pcmpgtw   mm4, mm2"                  \
    "movq      mm5, mm4"                  \
    "pand      mm4, mm0"                  \
    "pandn     mm5, mm2"                  \
    "por       mm4, mm5"                  \
    "test      edx, edx"                  \
    "jz        nspnostore"                \
    "movq      [edx], mm4"                \
    "add       edx, 8"                    \
    "nspnostore:"                       \
    "movq      mm6, mm4"                  \
    "pmaddwd   mm6, mm4"                  \
    "movd      esi, mm6"                  \
    "psrlq     mm6, 32"                   \
    "movd      edi, mm6"                  \
    "add       [ebx], esi"                \
    "adc       dword ptr [ebx+4], 0"      \
    "add       [ebx], edi"                \
    "adc       dword ptr [ebx+4], 0"      \
    "add       eax, 16"                   \
    "dec       ecx"                       \
    "jnz       nsploop"                   \
    "emms"                              \
    __parm [__eax] [__edx] [__ebx]      \
    __modify [__eax __ecx __edx __esi __edi 8087];

lz_i64 lz_norm_ss_pack_mmx(const int32_t *q, short *qout) {
    lz_i64 acc;
    lz_norm_ss_pack_mmx_asm(q, qout, &acc);
    return acc;
}

/* ---- the Q15 table interpolation, MMX tier, Watcom twin.
   gcc twin: lz_lerp_q15_mmx in the #else arm - read that one for the
   precondition (|b - a| <= 32767, which the callers meet because a and
   b are adjacent table entries) and for why `a` stays int32.

   32 ELEMENTS a call, eight passes of four, one emms at the exit. A
   four-element pragma would put an emms every four values, and the
   caller returns to x87 immediately, so the granularity is the whole
   difference between this being a kernel and being a regression - the
   same reason lz_norm_ss_pack_mmx_asm loops internally. */
extern void lz_lerp_q15_32_mmx_asm(const int32_t *a, const int32_t *b,
                                   const int32_t *frac, int32_t *out);
#pragma aux lz_lerp_q15_32_mmx_asm = \
    ".586"                              \
    "push    esi"                       \
    "mov     esi, 8"                    \
    "lerploop:"                         \
    "movq    mm0, [eax]"                \
    "movq    mm1, [eax+8]"              \
    "movq    mm2, [edx]"                \
    "movq    mm3, [edx+8]"              \
    "psubd   mm2, mm0"                  \
    "psubd   mm3, mm1"                  \
    "packssdw mm2, mm3"                 \
    "movq    mm4, [ebx]"                \
    "movq    mm5, [ebx+8]"              \
    "packssdw mm4, mm5"                 \
    "movq    mm5, mm2"                  \
    "pmullw  mm5, mm4"                  \
    "pmulhw  mm2, mm4"                  \
    "movq    mm6, mm5"                  \
    "punpcklwd mm6, mm2"                \
    "punpckhwd mm5, mm2"                \
    "psrad   mm6, 15"                   \
    "psrad   mm5, 15"                   \
    "paddd   mm6, mm0"                  \
    "paddd   mm5, mm1"                  \
    "movq    [ecx], mm6"                \
    "movq    [ecx+8], mm5"              \
    "add     eax, 16"                   \
    "add     edx, 16"                   \
    "add     ebx, 16"                   \
    "add     ecx, 16"                   \
    "dec     esi"                       \
    "jnz     lerploop"                  \
    "pop     esi"                       \
    "emms"                              \
    __parm [__eax] [__edx] [__ebx] [__ecx] \
    __modify [__eax __ebx __ecx __edx 8087];

void lz_lerp_q15_mmx(const int32_t *a, const int32_t *b,
                     const int32_t *frac, int32_t *out, int n) {
    int k = 0;
    for (; k + 31 < n; k += 32)
        lz_lerp_q15_32_mmx_asm(a + k, b + k, frac + k, out + k);
    for (; k < n; k++)
        out[k] = a[k] + (((b[k] - a[k]) * frac[k]) >> 15);
}

/* ---- lz_exp_fixed's Q20 Taylor, MMX tier, Watcom arm.
   gcc twin: lz_exp_q20_mmx in the #else arm, which carries the whole
   derivation - why every product is rebuilt from pmaddwd-sized limbs,
   why s is carried as u = s >> 1 plus a masked odd bit, and why the
   output multiply's big half cancels against the shift.

   THE CONSTANTS STAY IN MEMORY rather than in registers, which is the
   one structural difference from the SSE2 twin. MMX has eight registers
   and this body needs eight live values at its widest; MMX instructions
   also take a 64-bit memory operand directly, so a constant costs an
   addressing mode instead of a register. The SSE2 arm went the other
   way for the opposite reason: it had four to spare.

   int64 ELEMENTS, NOT int32 PAIRS, so the block is 8-byte aligned and
   every movq/pmaddwd against it is an aligned access. An int32[18]
   would hold the same bytes with 4-byte alignment and every other
   operand would straddle. */
#define LZ_D2(v) ((((lz_i64)(v)) << 32) | (lz_i64)(unsigned)(v))
static const lz_i64 exp_q20_mk[9] = {
    LZ_D2(1), LZ_D2(1023), LZ_D2(2047), LZ_D2(1 << 19),
    LZ_D2(726817), LZ_D2(22), LZ_D2(5921), LZ_D2(7), LZ_D2(22530)
};
#undef LZ_D2

/* np is the number of TWO-element passes. One emms at the exit, not one
   per pass: the caller returns to x87 immediately and a per-pass emms
   would be most of the kernel - same reason lz_lerp_q15_32_mmx_asm
   loops internally. */
extern void lz_exp_q20_2_mmx_asm(const int32_t *tab, const int32_t *s,
                                 int32_t *prod, const void *k, int np);
#pragma aux lz_exp_q20_2_mmx_asm =  \
    ".586"                          \
    "expq20mloop:"                  \
    "movq    mm0, [edx]"            \
    "movq    mm1, mm0"              \
    "psrld   mm1, 1"                \
    "movq    mm2, mm0"              \
    "pand    mm2, [ecx]"            \
    "pxor    mm3, mm3"              \
    "psubd   mm3, mm2"              \
    "movq    mm4, mm1"              \
    "pmaddwd mm4, [ecx+40]"         \
    "movq    mm5, mm1"              \
    "pmaddwd mm5, [ecx+48]"         \
    "movq    mm6, [ecx+32]"         \
    "pand    mm6, mm3"              \
    "pslld   mm5, 1"                \
    "paddd   mm5, mm6"              \
    "paddd   mm5, [ecx+24]"         \
    "psrld   mm5, 16"               \
    "paddd   mm4, mm5"              \
    "psrld   mm4, 4"                \
    "movq    mm5, mm1"              \
    "pmaddwd mm5, mm1"              \
    "movq    mm6, mm1"              \
    "pslld   mm6, 2"                \
    "paddd   mm6, [ecx]"            \
    "pand    mm6, mm3"              \
    "pslld   mm5, 2"                \
    "paddd   mm5, mm6"              \
    "movq    mm6, mm5"              \
    "pand    mm6, [ecx+8]"          \
    "movq    mm7, mm5"              \
    "psrld   mm7, 10"               \
    "pand    mm7, [ecx+8]"          \
    "psrld   mm5, 20"               \
    "movq    mm3, mm6"              \
    "movq    mm0, mm7"              \
    "movq    mm1, mm5"              \
    "pmaddwd mm3, [ecx+64]"         \
    "pmaddwd mm0, [ecx+64]"         \
    "pmaddwd mm1, [ecx+64]"         \
    "pmaddwd mm6, [ecx+56]"         \
    "pmaddwd mm7, [ecx+56]"         \
    "pmaddwd mm5, [ecx+56]"         \
    "pslld   mm6, 15"               \
    "pslld   mm7, 15"               \
    "pslld   mm5, 15"               \
    "paddd   mm3, mm6"              \
    "paddd   mm0, mm7"              \
    "paddd   mm1, mm5"              \
    "psrld   mm3, 10"               \
    "paddd   mm0, mm3"              \
    "psrld   mm0, 10"               \
    "paddd   mm1, [ecx+24]"         \
    "paddd   mm1, mm0"              \
    "psrld   mm1, 20"               \
    "paddd   mm4, mm1"              \
    "movq    mm0, [eax]"            \
    "movq    mm1, mm0"              \
    "psrld   mm1, 11"               \
    "movq    mm2, mm0"              \
    "pand    mm2, [ecx+16]"         \
    "pmaddwd mm1, mm4"              \
    "pmaddwd mm2, mm4"              \
    "paddd   mm2, [ecx+24]"         \
    "psrld   mm2, 11"               \
    "paddd   mm1, mm2"              \
    "psrld   mm1, 9"                \
    "paddd   mm0, mm1"              \
    "movq    [ebx], mm0"            \
    "add     eax, 8"                \
    "add     edx, 8"                \
    "add     ebx, 8"                \
    "dec     esi"                   \
    "jnz     expq20mloop"           \
    "emms"                          \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [__eax __ebx __ecx __edx __esi 8087];

void lz_exp_q20_mmx(const int32_t *tab, const int32_t *s,
                    int32_t *prod, int n) {
    int k = n & ~1;
    if (k) lz_exp_q20_2_mmx_asm(tab, s, prod, exp_q20_mk, k >> 1);
    for (; k < n; k++) {
        int32_t sk = s[k], cq;
        cq = (1 << 20)
           + (int32_t)((((lz_i64)726817 * sk) + (1 << 19)) >> 20)
           + (int32_t)((((lz_i64)251906 * ((lz_i64)sk * sk))
                        + (LZ_I64_C(1) << 39)) >> 40);
        prod[k] = (int32_t)((((lz_i64)tab[k] * cq) + (1 << 19)) >> 20);
    }
}

/* ---- Q8_0 32/128-element dot products and row kernel, MMX tier -
   in this TU. Algorithm documentation (why the weight side is widened
   via punpcklbw(zero,w), the register-budget accounting for the paired
   kernel, the emms placement rule) travels WITH these kernels and is
   immediately below, which is what src/ops_kernel_dot.h's Q8_0 section
   points at. This comment does NOT point back at that file: the two
   references would then form a circle around a text that lives here. */
#include "ops.h"           /* lz_prefetch_mode/lz_pair_mode - row_q8_mmx_asm
                              needs both, and neither is static, so a
                              real declaration is all that is needed. */
#include "ops_kernel_dot_shared.h" /* LZ_Q8_ACC, LZ_Q8_PFSEL, LZ_Q8_GROUP4 */

/* __modify must be truthful; never write __exact [].
   `__exact []` means "this code touches no register" - a lie: the
   asm writes eax (the return value), MMX/XMM, and MMX also moves the
   x87 stack. -oe (inline expansion) or -ol (loop optimization) alone
   are fine; together (-otel, -ox, -otexan) it blows up: after inlining,
   the loop optimizer keeps loop variables in eax across the asm
   because it believes nothing changed.
   The symptom is deeply misleading - end-to-end logits become
   nondeterministic garbage (three runs of the same exe give rms
   5418/5415/5412, drifting each time), while kernel-level differential
   tests of 200K groups all pass, because that call shape is simple and
   the optimizer has nothing to move.
   Writing [8087] makes Watcom treat it conservatively as "modified FP
   stack + default volatile registers". */
/* Watcom hand-written MMX dot product: 32 elements (int8 weights x
   int16 activations), pure asm pipeline. Same algorithm as the gcc
   version below (w expanded via punpcklbw(zero,w) = w<<8, cancelled by
   the caller's 1/256 at the row end). Written as a single pragma aux
   function to avoid Watcom's __m64 struct-return stack round trip
   (measured: the intrinsics version was 6x slower than x87; this one
   40x faster than it). Correctness: bit-matches the scalar over 100K
   random/extreme groups. eax=w, edx=x, returns eax;
   contains an emms (the caller's row-level EMMS is idempotent, no
   harm). */
/* Originally generated, hand-edited since - if you regenerate, diff
   the generator's output against what is here first.
   Kernels do NOT emit emms: the loop body is all integer and MMX; x87
   is only used in the whole-row reduction, and the caller already does
   one _mm_empty() after the loop. One emms per 32 elements would be
   pure redundancy - ~1.8M sub-block calls per token, at 6 cycles on a
   P6 that is 36 ms / 16% of the compute budget, worse on P4.
   Changing this REQUIRES proving the caller's loop body has no x87.
   Differential-checked over 200K groups (random + extremes,
   bit-identical to the scalar reference).
   Counts (uop, P6 accounting): Q8_0 52, Q4_1 63, Q6_1 86. */
extern int32_t lz_dot32_x16_asm(const int8_t *w, const int16_t *x);
#pragma aux lz_dot32_x16_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "movq      mm2, [eax]" \
    "movq      mm3, mm0" \
    "movq      mm4, mm0" \
    "punpcklbw mm3, mm2" \
    "pmaddwd   mm3, [ecx]" \
    "punpckhbw mm4, mm2" \
    "pmaddwd   mm4, [ecx+8]" \
    "paddd     mm1, mm3" \
    "movq      mm2, [eax+8]" \
    "movq      mm3, mm0" \
    "paddd     mm1, mm4" \
    "movq      mm4, mm0" \
    "punpcklbw mm3, mm2" \
    "pmaddwd   mm3, [ecx+16]" \
    "punpckhbw mm4, mm2" \
    "pmaddwd   mm4, [ecx+24]" \
    "paddd     mm1, mm3" \
    "movq      mm2, [eax+16]" \
    "movq      mm3, mm0" \
    "paddd     mm1, mm4" \
    "movq      mm4, mm0" \
    "punpcklbw mm3, mm2" \
    "pmaddwd   mm3, [ecx+32]" \
    "punpckhbw mm4, mm2" \
    "pmaddwd   mm4, [ecx+40]" \
    "paddd     mm1, mm3" \
    "movq      mm2, [eax+24]" \
    "movq      mm3, mm0" \
    "paddd     mm1, mm4" \
    "movq      mm4, mm0" \
    "punpcklbw mm3, mm2" \
    "pmaddwd   mm3, [ecx+48]" \
    "punpckhbw mm4, mm2" \
    "pmaddwd   mm4, [ecx+56]" \
    "paddd     mm1, mm3" \
    "paddd     mm1, mm4" \
    "movq      mm2, mm1" \
    "psrlq     mm2, 32" \
    "paddd     mm1, mm2" \
    "movd      eax, mm1" \
    __parm [__eax] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* PAIRED twin of the above: two tokens, ONE weight unpack.
   The gcc side is dot32_x16_mmx_2; this is the Watcom half, i.e. the
   half that runs on the target. Q8's weight side is 12 of 36 ops per
   32 MACs, so sharing it across NT tokens costs 12 + 24*NT instead of
   36*NT - ceiling 1.50x, 1.20x at NT=2.

   Register budget is why NT is 2 and not more: mm7 zero, mm6/mm5 the
   two accumulators, mm1 the raw weight octet, mm2 the widened half,
   mm3 its copy for the second token - six of eight, with none to spare
   for a third token's accumulator plus its activation.

   No emms here: the caller emits one after the row loop.

   VERIFIED BIT-IDENTICAL: 200000 random plus 8 extreme trials against
   the scalar reference, 0 mismatches - and the
   differential's own sensitivity checked by mutating one operand offset
   ([ebx+8] -> [ebx+16]), which turned it into 200000 mismatches. */
extern void lz_dot32_x16_asm_2(const int8_t *w, const int16_t *xa,
                               const int16_t *xb, int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_x16_asm_2 = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "pxor      mm2, mm2" \
    "movq      mm3, [eax]" \
    "movq      mm4, mm0" \
    "movq      mm5, mm0" \
    "punpcklbw mm4, mm3" \
    "punpckhbw mm5, mm3" \
    "movq      mm6, mm4" \
    "movq      mm7, mm5" \
    "pmaddwd   mm4, [edx]" \
    "pmaddwd   mm5, [edx+8]" \
    "pmaddwd   mm6, [ebx]" \
    "pmaddwd   mm7, [ebx+8]" \
    "paddd     mm1, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm3, [eax+8]" \
    "movq      mm4, mm0" \
    "movq      mm5, mm0" \
    "punpcklbw mm4, mm3" \
    "punpckhbw mm5, mm3" \
    "movq      mm6, mm4" \
    "movq      mm7, mm5" \
    "pmaddwd   mm4, [edx+16]" \
    "pmaddwd   mm5, [edx+24]" \
    "pmaddwd   mm6, [ebx+16]" \
    "pmaddwd   mm7, [ebx+24]" \
    "paddd     mm1, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm3, [eax+16]" \
    "movq      mm4, mm0" \
    "movq      mm5, mm0" \
    "punpcklbw mm4, mm3" \
    "punpckhbw mm5, mm3" \
    "movq      mm6, mm4" \
    "movq      mm7, mm5" \
    "pmaddwd   mm4, [edx+32]" \
    "pmaddwd   mm5, [edx+40]" \
    "pmaddwd   mm6, [ebx+32]" \
    "pmaddwd   mm7, [ebx+40]" \
    "paddd     mm1, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm3, [eax+24]" \
    "movq      mm4, mm0" \
    "movq      mm5, mm0" \
    "punpcklbw mm4, mm3" \
    "punpckhbw mm5, mm3" \
    "movq      mm6, mm4" \
    "movq      mm7, mm5" \
    "pmaddwd   mm4, [edx+48]" \
    "pmaddwd   mm5, [edx+56]" \
    "pmaddwd   mm6, [ebx+48]" \
    "pmaddwd   mm7, [ebx+56]" \
    "paddd     mm1, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm3, mm1" \
    "psrlq     mm3, 32" \
    "paddd     mm1, mm3" \
    "movd      [ecx], mm1" \
    "movq      mm3, mm2" \
    "psrlq     mm3, 32" \
    "paddd     mm2, mm3" \
    "movd      [esi], mm2" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [8087]

/* Both are compiled in; the runtime picks by CPUID (see
   lz_kernel_select). One binary must run on both PII (MMX only) and
   T42 (SSE2). Non-static now: src/ops.c's LZ_ATTN_DOT32 macro
   (attention scoring's dot, lz_attn_score_q8) calls this by name
   across the TU boundary - it always did, even before this move, just
   within one TU; declared in src/ops_mmx.h. */
int32_t dot32_x16_mmx(const int8_t *w, const int16_t *x) {
    return lz_dot32_x16_asm(w, x);
}

/* The paired form's wrapper, which the Watcom build did not have.
   src/ops_mmx.h declares dot32_x16_mmx_2 and the gcc side defines it in
   src/ops_kernel_dot_mmx.h; on this side the pragma was only ever
   called by name from row_q8_mmx_asm below, so the symbol the header
   promises did not exist here. Nothing linked against it, so nothing
   complained - one build answering for a declaration the other does
   not is the shape a pair gate exists to catch. It also gives the
   kernel an ADDRESS: a #pragma aux body has none, so without this a
   probe cannot reach it. */
void dot32_x16_mmx_2(const int8_t *w, const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob) {
    lz_dot32_x16_asm_2(w, xa, xb, oa, ob);
}

/* Q8_0, 128 elements: four sub-blocks, four accumulators, one fold.
   Held to lz_dot32_x16_asm called four times AND to lz_dot32_x16_sse2
   by build/dot128_x16_probe.c.

   TWO CHUNKS IN FLIGHT. The lo and hi halves of a weight octet are
   independent, so they take mm2 and mm3 and run side by side rather
   than one chain of movq/punpck/pmaddwd/paddd after another. Costs
   nothing: mm0 mm3 mm4 mm5 were all free.

   FOUR ACCUMULATORS, NOT ONE, so the four sub-blocks do not each pay a
   horizontal reduction. mm6 mm5 mm4 mm0 carry them; the transposing
   fold at the end pairs them with punpckldq/punpckhdq so one paddd
   finishes two, and each pair leaves whole - two 8-byte stores instead
   of four movd. Same trick as lz_dot128_t2_asm, same reason.

   The accumulation order into each sub-block's own accumulator is
   unchanged, so this is bit-identical rather than merely equal. */
extern void lz_dot128_x16_asm(const int8_t *w,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_x16_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "pxor      mm2, mm2" \
    "pxor      mm3, mm3" \
    "pxor      mm4, mm4" \
    "movq      mm5, [eax]" \
    "movq      mm6, mm0" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+8]" \
    "paddd     mm1, mm6" \
    "movq      mm5, [eax+8]" \
    "movq      mm6, mm0" \
    "paddd     mm1, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+16]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+24]" \
    "paddd     mm1, mm6" \
    "movq      mm5, [eax+16]" \
    "movq      mm6, mm0" \
    "paddd     mm1, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+32]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+40]" \
    "paddd     mm1, mm6" \
    "movq      mm5, [eax+24]" \
    "movq      mm6, mm0" \
    "paddd     mm1, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+48]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+56]" \
    "paddd     mm1, mm6" \
    "movq      mm5, [eax+32]" \
    "movq      mm6, mm0" \
    "paddd     mm1, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+64]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+72]" \
    "paddd     mm2, mm6" \
    "movq      mm5, [eax+40]" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+80]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+88]" \
    "paddd     mm2, mm6" \
    "movq      mm5, [eax+48]" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+96]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+104]" \
    "paddd     mm2, mm6" \
    "movq      mm5, [eax+56]" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+112]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+120]" \
    "paddd     mm2, mm6" \
    "movq      mm5, [eax+64]" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+128]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+136]" \
    "paddd     mm3, mm6" \
    "movq      mm5, [eax+72]" \
    "movq      mm6, mm0" \
    "paddd     mm3, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+144]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+152]" \
    "paddd     mm3, mm6" \
    "movq      mm5, [eax+80]" \
    "movq      mm6, mm0" \
    "paddd     mm3, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+160]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+168]" \
    "paddd     mm3, mm6" \
    "movq      mm5, [eax+88]" \
    "movq      mm6, mm0" \
    "paddd     mm3, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+176]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+184]" \
    "paddd     mm3, mm6" \
    "movq      mm5, [eax+96]" \
    "movq      mm6, mm0" \
    "paddd     mm3, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+192]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+200]" \
    "paddd     mm4, mm6" \
    "movq      mm5, [eax+104]" \
    "movq      mm6, mm0" \
    "paddd     mm4, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+208]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+216]" \
    "paddd     mm4, mm6" \
    "movq      mm5, [eax+112]" \
    "movq      mm6, mm0" \
    "paddd     mm4, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+224]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+232]" \
    "paddd     mm4, mm6" \
    "movq      mm5, [eax+120]" \
    "movq      mm6, mm0" \
    "paddd     mm4, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+240]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+248]" \
    "paddd     mm4, mm6" \
    "movq      mm5, mm1" \
    "paddd     mm4, mm7" \
    "punpckldq mm5, mm2" \
    "punpckhdq mm1, mm2" \
    "paddd     mm5, mm1" \
    "movq      [ebx], mm5" \
    "movq      mm5, mm3" \
    "punpckldq mm5, mm4" \
    "punpckhdq mm3, mm4" \
    "paddd     mm5, mm3" \
    "movq      [ebx+8], mm5" \
    "emms" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

/* Real, addressable twin of lz_dot128_x16_asm - needed cross-TU by
   src/ops_sse2.c's row_q8_sse2_asm, ONLY in the -DLZ_SSE2_GROUP=0 A/B
   control build (an explicit measurement knob, not the default: see
   ops.c's own comment on it): that build point asks an SSE2-tier row
   to fall back to the MMX group kernel. LZ_Q8_GROUP4 calls its DOT128
   argument by name, and a #pragma aux body cannot have its address
   taken, so this wrapper is what makes that call legal across the TU
   boundary. row_q8_mmx_asm above never calls through this: it inlines
   the pragma directly. */
void lz_dot128_x16_mmx_w(const int8_t *w, const int16_t *x, int32_t *out4) {
    lz_dot128_x16_asm(w, x, out4);
}

/* row_q8_mmx_asm: the real call boundary from its own caller
   (matmul_q8_impl, src/ops.c, dispatched once per matmul via
   lz_row_pick's LZ_ROW_Q8 table). The whole function, together with
   the leaf kernels it calls direct-by-name inside its own loop, adds
   no new call sites. g_pair is accessed via lz_pair_mode() - the row
   function cannot see ops.c's static global directly, so it goes
   through the same accessor lz_prefetch_mode() already used one line
   below. */
void row_q8_mmx_asm(const lz_row_ctx *c) {
    const int8_t *wr = (const int8_t *)c->w4;
    const int8_t *wend = (const int8_t *)c->pf_end;
    int tk, g;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pairing beats the 128-element group kernel whenever there is
               a second token: the group kernel amortizes FIXED overhead
               (mask setup, prologue) from once per 32 elements to once per
               128, while pairing removes the weight unpack itself for the
               second token - and on the target the weight side is the
               memory side. Same choice the gcc twin makes, which also keeps
               the two impls' shapes comparable when they are diffed.
               At nt == 1 there is nothing to pair, so the group kernel
               keeps that path unchanged.

               NOT MEASURED ON TARGET. The op-count argument is 12 + 24*NT
               against 36*NT; whether it beats the group kernel's amortized
               prologue on a real PII is a wall-clock question, and the
               machine for it is not here. This is a knob-shaped
               decision recorded as such, not a measured result. */
            if (c->nt - tk >= 2 && lz_pair_mode()) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                int g2;
                for (g2 = 0; g2 < c->nb; g2++) {
                    const int8_t *pf_ = wr + (size_t)(g2 + LZ_PF_DIST) * 32;
                    if (pf_ < wend) {
                        int pf_m_ = lz_prefetch_mode();
                        if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);
                        else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);
                        else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);
                        else                          LZ_PFI_NTA(pf_);
                    }
                    lz_dot32_x16_asm_2(wr + (size_t)g2 * 32,
                                       xwt + (size_t)g2 * 32,
                                       xw2 + (size_t)g2 * 32,
                                       acc + g2, accb + g2);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            if ((c->nb & 3) == 0) LZ_Q8_GROUP4(lz_dot128_x16_asm);
            else               LZ_Q8_PFSEL(dot32_x16_mmx);
        }
    }
}

/* ---- Q4_1 32/128-element dot products and row kernel, MMX tier -
   in this TU. Algorithm documentation is in src/ops_kernel_dot.h,
   unchanged by the code's location. */

/* Q4_1 32-element dot product (MMX). Same algorithm as SSE2's
   part32_q41: pand(b,0x0F) grabs elements 0..15,
   pand(psrlw(b,4),0x0F) elements 16..31, then punpcklbw(0,x) expands
   to x256 int16 for pmaddwd. Returns the 32-element dot scaled x256;
   the caller cancels once at the row end. Watcom hand-written version,
   same rationale as lz_dot32_x16_asm (_m_* intrinsics are 6x slower
   than its own x87 scalar). Mask built on the fly instead of a memory
   load: all-ones -> psrlw 12 gives 0x000F x4 -> packuswb saturating
   pack to 0x0F x8 - three instructions replacing one memory operand.
   Registers: mm0=low nibbles mm4=high nibbles mm1=activations
   mm2=expand temp mm3=mask mm6=accumulator mm7=zero mm5=reduce temp -
   exactly 8.
   Verified: bit-matches the scalar over 200K random + all-15 x
   all-+/-127 extreme groups; measured 3.75 ns/32MAC vs x87
   scalar 23.50 ns (6.3x). */
extern int32_t lz_dot32_q41_asm(const unsigned char *w, const int16_t *x);
/* w from [eax], x from [ecx] - two registers, and no third. A trailing
   [__ebx] sat here and was inert, Watcom ignoring registers past the
   last argument; it is gone because it made "three registers, two
   parameters" read as this file's house style, and the edit that added
   it put the same spare at the FRONT of lz_dot32_t2_asm's list, where
   it is not inert at all. */
#pragma aux lz_dot32_q41_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pcmpeqb   mm1, mm1" \
    "psrlw     mm1, 12" \
    "packuswb  mm1, mm1" \
    "pxor      mm2, mm2" \
    "movq      mm3, [eax]" \
    "movq      mm4, mm3" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+8]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+32]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+40]" \
    "paddd     mm2, mm6" \
    "movq      mm3, [eax+8]" \
    "movq      mm4, mm3" \
    "paddd     mm2, mm7" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx+16]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+24]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+48]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+56]" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm4, mm2" \
    "psrlq     mm4, 32" \
    "paddd     mm2, mm4" \
    "movd      eax, mm2" \
    __parm [__eax] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* PAIRED twin: two tokens, ONE nibble unpack. Q4_1's weight side is 16
   of the 40 ops per 32 MACs (2 loads + 6 mask/shift + 8 punpck) - a
   nibble has to be split before it can be widened - so sharing it costs
   16 + 24*NT instead of 40*NT: ceiling 1.67x, 1.25x at NT=2. The gcc
   twin is dot32_q41_mmx_2; this is the half that runs on the target.

   mm0 holds 0x0F x8, built without touching a general register:
   pcmpeqb gives all ones, psrlw 12 leaves 0x000F per word, packuswb
   saturates those to 0x0F per byte. Register budget: mm0 mask, mm7
   zero, mm6/mm5 accumulators, mm1 lo nibbles, mm4 hi nibbles, mm2
   widened half, mm3 its copy for the second token - all eight.

   No emms here; the caller emits one after the row loop.

   VERIFIED BIT-IDENTICAL: 200000 random plus 8 extreme trials against
   a scalar reference, 0 mismatches; the differential's
   own sensitivity checked by mutating one operand offset ([edx+32] ->
   [edx+48]), which turned it into 199998 mismatches. */
extern void lz_dot32_q41_asm_2(const unsigned char *w, const int16_t *xa,
                               const int16_t *xb, int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_q41_asm_2 = \
    ".586" \
    "pcmpeqb   mm0, mm0" \
    "psrlw     mm0, 12" \
    "packuswb  mm0, mm0" \
    "pxor      mm1, mm1" \
    "pxor      mm2, mm2" \
    "pxor      mm3, mm3" \
    "movq      mm4, [eax]" \
    "movq      mm5, mm4" \
    "pand      mm4, mm0" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm0" \
    "movq      mm6, mm1" \
    "punpcklbw mm6, mm4" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [edx]" \
    "pmaddwd   mm7, [ebx]" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm6, mm1" \
    "punpckhbw mm6, mm4" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [edx+8]" \
    "pmaddwd   mm7, [ebx+8]" \
    "movq      mm4, [eax+8]" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm6, mm1" \
    "punpcklbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [edx+32]" \
    "pmaddwd   mm7, [ebx+32]" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm6, mm1" \
    "punpckhbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [edx+40]" \
    "pmaddwd   mm7, [ebx+40]" \
    "movq      mm5, mm4" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "pand      mm4, mm0" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm0" \
    "movq      mm6, mm1" \
    "punpcklbw mm6, mm4" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [edx+16]" \
    "pmaddwd   mm7, [ebx+16]" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm6, mm1" \
    "punpckhbw mm6, mm4" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [edx+24]" \
    "pmaddwd   mm7, [ebx+24]" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm6, mm1" \
    "punpcklbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [edx+48]" \
    "pmaddwd   mm7, [ebx+48]" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm6, mm1" \
    "punpckhbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [edx+56]" \
    "pmaddwd   mm7, [ebx+56]" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, mm2" \
    "psrlq     mm4, 32" \
    "paddd     mm2, mm4" \
    "movd      [ecx], mm2" \
    "movq      mm4, mm3" \
    "psrlq     mm4, 32" \
    "paddd     mm3, mm4" \
    "movd      [esi], mm3" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [8087]

/* Not static: ops_mmx.h declares it with external linkage. */
int32_t dot32_q41_mmx(const unsigned char *w, const int16_t *x) {
    return lz_dot32_q41_asm(w, x);
}

/* The paired kernel's address, on the same argument. */
void dot32_q41_mmx_2(const unsigned char *w, const int16_t *xa,
                     const int16_t *xb, int32_t *oa, int32_t *ob) {
    lz_dot32_q41_asm_2(w, xa, xb, oa, ob);
}

extern void lz_dot128_q41_asm(const unsigned char *w,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q41_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pcmpeqb   mm1, mm1" \
    "psrlw     mm1, 12" \
    "packuswb  mm1, mm1" \
    "pxor      mm2, mm2" \
    "movq      mm3, [eax]" \
    "movq      mm4, mm3" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+8]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+32]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+40]" \
    "paddd     mm2, mm6" \
    "movq      mm3, [eax+8]" \
    "movq      mm4, mm3" \
    "paddd     mm2, mm7" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx+16]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+24]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+48]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+56]" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm4, mm2" \
    "psrlq     mm4, 32" \
    "paddd     mm2, mm4" \
    "movd      [ebx], mm2" \
    "pxor      mm2, mm2" \
    "movq      mm3, [eax+16]" \
    "movq      mm4, mm3" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx+64]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+72]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+96]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+104]" \
    "paddd     mm2, mm6" \
    "movq      mm3, [eax+24]" \
    "movq      mm4, mm3" \
    "paddd     mm2, mm7" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx+80]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+88]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+112]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+120]" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm4, mm2" \
    "psrlq     mm4, 32" \
    "paddd     mm2, mm4" \
    "movd      [ebx+4], mm2" \
    "pxor      mm2, mm2" \
    "movq      mm3, [eax+32]" \
    "movq      mm4, mm3" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx+128]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+136]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+160]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+168]" \
    "paddd     mm2, mm6" \
    "movq      mm3, [eax+40]" \
    "movq      mm4, mm3" \
    "paddd     mm2, mm7" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx+144]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+152]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+176]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+184]" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm4, mm2" \
    "psrlq     mm4, 32" \
    "paddd     mm2, mm4" \
    "movd      [ebx+8], mm2" \
    "pxor      mm2, mm2" \
    "movq      mm3, [eax+48]" \
    "movq      mm4, mm3" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx+192]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+200]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+224]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+232]" \
    "paddd     mm2, mm6" \
    "movq      mm3, [eax+56]" \
    "movq      mm4, mm3" \
    "paddd     mm2, mm7" \
    "movq      mm5, mm3" \
    "pand      mm4, mm1" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm1" \
    "movq      mm6, mm0" \
    "punpcklbw mm6, mm4" \
    "pmaddwd   mm6, [ecx+208]" \
    "movq      mm7, mm0" \
    "punpckhbw mm7, mm4" \
    "pmaddwd   mm7, [ecx+216]" \
    "paddd     mm2, mm6" \
    "movq      mm6, mm0" \
    "paddd     mm2, mm7" \
    "movq      mm7, mm0" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+240]" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+248]" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm4, mm2" \
    "psrlq     mm4, 32" \
    "paddd     mm2, mm4" \
    "movd      [ebx+12], mm2" \
    "emms" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

/* Real, addressable twin of lz_dot128_q41_asm - needed cross-TU by
   src/ops_sse2.c's row_q41_sse2_asm, ONLY in the -DLZ_SSE2_GROUP=0 A/B
   control build. Same reasoning as lz_dot128_x16_mmx_w (Q8_0). */
void lz_dot128_q41_mmx_w(const unsigned char *w, const int16_t *x, int32_t *out4) {
    lz_dot128_q41_asm(w, x, out4);
}

/* row_q41_mmx_asm: the real call boundary from its own caller
   (matmul_q41_impl, src/ops.c, dispatched once per matmul via
   LZ_ROW_Q41). The whole function adds no new call sites - same
   argument as row_q8_mmx_asm. g_pair is accessed via lz_pair_mode(),
   same as Q8_0's row function. */
void row_q41_mmx_asm(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *wend = (const unsigned char *)c->pf_end;
    int tk, s;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair whenever a second token exists - same trade as Q8 and
               worth more here (weight side 16 of 40 ops against Q8's 12 of
               36). Not measured on target; see row_q8_mmx_asm. */
            if (c->nt - tk >= 2 && lz_pair_mode()) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                int s2;
                for (s2 = 0; s2 < c->nb; s2++) {
                    const unsigned char *pf_ = wn + (size_t)(s2 + LZ_PF_DIST) * 16;
                    if (pf_ < wend) {
                        int pf_m_ = lz_prefetch_mode();
                        if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);
                        else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);
                        else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);
                        else                          LZ_PFI_NTA(pf_);
                    }
                    lz_dot32_q41_asm_2(wn + (size_t)s2 * 16,
                                       xwt + (size_t)s2 * 32,
                                       xw2 + (size_t)s2 * 32,
                                       acc + s2, accb + s2);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            if ((c->nb & 3) == 0) LZ_Q41_GROUP4(lz_dot128_q41_asm);
            else               LZ_Q41_PFSEL(dot32_q41_mmx);
        }
    }
}

/* ---- T2 (ternary) dot product and row kernel, MMX tier - in this TU.
   No 128-wide group kernel exists for this format (the
   -DLZ_SSE2_GROUP=0 A/B control concern does not apply here), so
   row_t2_mmx_asm calls lz_dot32_t2_asm direct-by-name, same-TU, no
   wrapper needed. */
extern int32_t lz_dot32_t2_asm(const unsigned char *w2,
                               const int16_t *x);
/* TWO CHUNKS IN FLIGHT, not one, because the target core is in-order
   and dual-issue. Written as one chain this body was seven instructions
   each reading what the last wrote - pxor, punpck, psllw, psrlw, psllw,
   pmaddwd, paddd - and build/pair_census.sh scored it 75% dependent
   adjacent pairs, the worst in the file. A dependent pair cannot issue
   together on a Pentium, so most of the second pipe was unreachable.

   The low and high nibbles of the same weight byte are independent, so
   the two halves of each shift level run side by side on separate
   scratch (mm0/mm2 against mm1/mm3) into separate accumulators.

   BIT-IDENTICAL, not close: the same eight products in a different
   summation grouping, and int32 addition is associative and exact. The
   two accumulators join before the horizontal fold. */
/* TWO registers for TWO parameters, in the ORDER THE BODY LOADS THEM:
   w2 from [edx], x from [ecx]. An extra [__eax] once sat at the FRONT of
   the __parm list, so Watcom passed w2 in eax and x in edx and never
   wrote ecx - which the body's second instruction dereferences. It read
   whatever the caller left there: .prof/rowdot_xcheck.c died on ECX=0,
   and the engine would have summed from a stale address instead.
   A spare register is harmless at the END of the list (q41 above carried
   one) and shifts every argument when it is not, which is why the check
   is on arity and not on "looks like the others" -
   build/aux_parm_arity_gate.sh. */
#pragma aux lz_dot32_t2_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "movq      mm2, [edx]" \
    "movq      mm3, [ecx]" \
    "movq      mm4, [ecx+8]" \
    "pxor      mm5, mm5" \
    "pxor      mm6, mm6" \
    "punpcklbw mm5, mm2" \
    "punpckhbw mm6, mm2" \
    "psllw     mm5, 6" \
    "psllw     mm6, 6" \
    "psrlw     mm5, 14" \
    "psrlw     mm6, 14" \
    "psllw     mm5, 8" \
    "psllw     mm6, 8" \
    "pmaddwd   mm5, mm3" \
    "pmaddwd   mm6, mm4" \
    "paddd     mm0, mm5" \
    "movq      mm3, [ecx+16]" \
    "movq      mm4, [ecx+24]" \
    "paddd     mm1, mm6" \
    "pxor      mm5, mm5" \
    "pxor      mm6, mm6" \
    "punpcklbw mm5, mm2" \
    "punpckhbw mm6, mm2" \
    "psllw     mm5, 4" \
    "psllw     mm6, 4" \
    "psrlw     mm5, 14" \
    "psrlw     mm6, 14" \
    "psllw     mm5, 8" \
    "psllw     mm6, 8" \
    "pmaddwd   mm5, mm3" \
    "pmaddwd   mm6, mm4" \
    "paddd     mm0, mm5" \
    "movq      mm3, [ecx+32]" \
    "movq      mm4, [ecx+40]" \
    "paddd     mm1, mm6" \
    "pxor      mm5, mm5" \
    "pxor      mm6, mm6" \
    "punpcklbw mm5, mm2" \
    "punpckhbw mm6, mm2" \
    "psllw     mm5, 2" \
    "psllw     mm6, 2" \
    "psrlw     mm5, 14" \
    "psrlw     mm6, 14" \
    "psllw     mm5, 8" \
    "psllw     mm6, 8" \
    "pmaddwd   mm5, mm3" \
    "pmaddwd   mm6, mm4" \
    "paddd     mm0, mm5" \
    "movq      mm3, [ecx+48]" \
    "movq      mm4, [ecx+56]" \
    "paddd     mm1, mm6" \
    "pxor      mm5, mm5" \
    "pxor      mm6, mm6" \
    "punpcklbw mm5, mm2" \
    "punpckhbw mm6, mm2" \
    "psrlw     mm5, 14" \
    "psrlw     mm6, 14" \
    "psllw     mm5, 8" \
    "psllw     mm6, 8" \
    "pmaddwd   mm5, mm3" \
    "pmaddwd   mm6, mm4" \
    "paddd     mm0, mm5" \
    "paddd     mm1, mm6" \
    "paddd     mm0, mm1" \
    "movq      mm3, mm0" \
    "psrlq     mm3, 32" \
    "paddd     mm0, mm3" \
    "movd      eax, mm0" \
    "emms" \
    __parm [__edx] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* Addressable twin of lz_dot32_t2_asm. row_t2_mmx_asm inlines the
   pragma directly and needs no such thing; build/dot128_t2_probe.c does,
   because the group kernel's reference IS this kernel called four times
   and a #pragma aux body cannot be called from another TU. */
int32_t lz_dot32_t2_mmx_w(const unsigned char *w2, const int16_t *x) {
    return lz_dot32_t2_asm(w2, x);
}

/* ---- T2, 128 elements: four sub-blocks folded together.
   gcc twins: dot128_t2_mmx (src/ops_kernel_dot_mmx.h) and
   dot128_t2_sse2 (src/ops_kernel_dot_sse2.h).

   WHAT THE GROUP FORM BUYS, since an earlier reason string said
   "nothing" and was wrong about where to look. The unpack is unchanged
   - T2 hoists only a zero register across sub-blocks, so grouping
   trades pxor for movq one for one and gains nothing there. It pays at
   the two ENDS: the per-32 kernel finishes every sub-block with its own
   `movq/psrlq/paddd/movd` horizontal reduction and its own emms, and
   four of those become one fold of ten instructions and one emms.

   THE FOUR BLOCKS ARE TEXTUALLY IDENTICAL except for which accumulator
   they add into, because the pointers ADVANCE between them rather than
   the offsets growing. Written the other way this would carry
   thirty-two hand-typed displacements up to [ecx+248], and a single
   digit wrong in one of them is a wrong answer in one lane of one
   sub-block - the kind of defect that survives a spot check. Two `add`s
   per block is what that costs.

   THE FOLD transposes rather than reducing each accumulator alone.
   punpckldq/punpckhdq bring the two lanes of two accumulators together
   so one paddd finishes a pair, and each pair leaves in one register to
   be stored whole - two 8-byte stores instead of four movd.

   THE CHUNKS RUN TWO AT A TIME, for the in-order dual-issue target:
   written as one chain each was pxor, punpck, psllw, psrlw, psllw,
   pmaddwd, paddd with every instruction reading what the last wrote,
   and a dependent pair cannot issue together. The natural pair is the
   lo and hi unpack of the SAME mm4 at the same shift level, so they
   share every constant and need only a second scratch register - mm7,
   the one this kernel had spare. The activation register is shared
   rather than doubled: mm0 is reloaded between the two pmaddwd, which
   the four accumulators leave no room to avoid.

   Bit-identical, and not by an appeal to associativity: the products
   reach the accumulator in the order they did before. */
/* THE TWO `add`s COME BEFORE THE TWO LOADS AT EACH CHUNK BOUNDARY, and
   that is not a scheduling preference. `movq mm4, [edx]` depends on
   `add edx, 8` through EDX, which no MMX register records: a pass that
   tracks mm0-mm7 sees two independent instructions and will happily
   hoist the load above the increment. It did, once. Chunks 1, 2 and 3
   then multiplied the PREVIOUS chunk's weights and first activation
   octet by their own remaining seven, which is a well formed kernel over
   valid codes returning a plausible number - chunk 0 stayed correct, so
   a spot check of the first sub-block agreed.
   build/dot128_t2_probe.c counted 11523 of 49156 values wrong. It is
   hand-run and nothing in CI calls it, which is how the defect lived in
   a shipping kernel: no local checkpoint carries a T2 tensor, so the
   parity gate, the logits comparison and every bit-identity claim passed
   over it untouched. */
extern void lz_dot128_t2_asm(const unsigned char *w2,
                             const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_t2_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "pxor      mm2, mm2" \
    "pxor      mm3, mm3" \
    "movq      mm4, [edx]" \
    "movq      mm5, [ecx]" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 6" \
    "psllw     mm7, 6" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+8]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm0, mm6" \
    "movq      mm5, [ecx+16]" \
    "paddd     mm0, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 4" \
    "psllw     mm7, 4" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+24]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm0, mm6" \
    "movq      mm5, [ecx+32]" \
    "paddd     mm0, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 2" \
    "psllw     mm7, 2" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+40]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm0, mm6" \
    "movq      mm5, [ecx+48]" \
    "paddd     mm0, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+56]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm0, mm6" \
    "add       ecx, 64" \
    "add       edx, 8" \
    "movq      mm4, [edx]" \
    "movq      mm5, [ecx]" \
    "paddd     mm0, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 6" \
    "psllw     mm7, 6" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+8]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm1, mm6" \
    "movq      mm5, [ecx+16]" \
    "paddd     mm1, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 4" \
    "psllw     mm7, 4" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+24]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm1, mm6" \
    "movq      mm5, [ecx+32]" \
    "paddd     mm1, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 2" \
    "psllw     mm7, 2" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+40]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm1, mm6" \
    "movq      mm5, [ecx+48]" \
    "paddd     mm1, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+56]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm1, mm6" \
    "add       ecx, 64" \
    "add       edx, 8" \
    "movq      mm4, [edx]" \
    "movq      mm5, [ecx]" \
    "paddd     mm1, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 6" \
    "psllw     mm7, 6" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+8]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm2, mm6" \
    "movq      mm5, [ecx+16]" \
    "paddd     mm2, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 4" \
    "psllw     mm7, 4" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+24]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm2, mm6" \
    "movq      mm5, [ecx+32]" \
    "paddd     mm2, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 2" \
    "psllw     mm7, 2" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+40]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm2, mm6" \
    "movq      mm5, [ecx+48]" \
    "paddd     mm2, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+56]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm2, mm6" \
    "add       ecx, 64" \
    "add       edx, 8" \
    "movq      mm4, [edx]" \
    "movq      mm5, [ecx]" \
    "paddd     mm2, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 6" \
    "psllw     mm7, 6" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+8]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm3, mm6" \
    "movq      mm5, [ecx+16]" \
    "paddd     mm3, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 4" \
    "psllw     mm7, 4" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+24]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm3, mm6" \
    "movq      mm5, [ecx+32]" \
    "paddd     mm3, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psllw     mm6, 2" \
    "psllw     mm7, 2" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+40]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm3, mm6" \
    "movq      mm5, [ecx+48]" \
    "paddd     mm3, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "punpcklbw mm6, mm4" \
    "punpckhbw mm7, mm4" \
    "psrlw     mm6, 14" \
    "psrlw     mm7, 14" \
    "psllw     mm6, 8" \
    "psllw     mm7, 8" \
    "pmaddwd   mm6, mm5" \
    "movq      mm5, [ecx+56]" \
    "pmaddwd   mm7, mm5" \
    "paddd     mm3, mm6" \
    "movq      mm5, mm0" \
    "paddd     mm3, mm7" \
    "punpckldq mm5, mm1" \
    "punpckhdq mm0, mm1" \
    "paddd     mm5, mm0" \
    "movq      [ebx], mm5" \
    "movq      mm5, mm2" \
    "punpckldq mm5, mm3" \
    "punpckhdq mm2, mm3" \
    "paddd     mm5, mm2" \
    "movq      [ebx+8], mm5" \
    "emms" \
    __parm [__edx] [__ecx] [__ebx] \
    __modify [__ecx __edx 8087]

/* Real, addressable twin of lz_dot128_t2_asm - a #pragma aux body has
   no address, and the probe and any cross-TU caller need one. Same
   reason lz_dot128_x16_mmx_w exists for Q8_0. */
void lz_dot128_t2_mmx_w(const unsigned char *w2, const int16_t *x,
                        int32_t *out4) {
    lz_dot128_t2_asm(w2, x, out4);
}

/* row_t2_mmx_asm: ALREADY the real call boundary (matmul_t2_impl,
   src/ops.c, dispatched once per matmul via LZ_ROW_T2). Moving the
   whole function here adds NO new call sites - same argument as
   row_q8_mmx_asm/row_q41_mmx_asm. No pairing (unmeasured, per the
   original comment at this function's old home in ops.c). */
void row_t2_mmx_asm(const lz_row_ctx *c) {
    const unsigned char *p2 = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Whole groups of four first, the per-32 kernel for the nb % 4
               tail - the same split the gcc twin makes in ops_matmul.c. The
               two write the same values (build/dot128_t2_probe.c holds the
               group form to four calls of the per-32 one), so a row divided
               between them has two instruction counts and one answer. */
            s = 0;
            for (; s + 4 <= c->nb; s += 4)
                lz_dot128_t2_asm(p2 + (size_t)s * 8, xwt + (size_t)s * 32,
                                 acc + s);
            for (; s < c->nb; s++)
                acc[s] = lz_dot32_t2_asm(p2 + (size_t)s * 8, xwt + (size_t)s * 32);
        }
    }
}

/* ---- Q6_1 32/128-element dot products and row kernel, MMX tier -
   in this TU. Algorithm documentation is in src/ops_kernel_dot.h,
   unchanged by the code's location. */
extern int32_t lz_dot32_q61_asm(const unsigned char *w4, const unsigned char *w2, const int16_t *x);
#pragma aux lz_dot32_q61_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "pcmpeqb   mm2, mm2" \
    "psrlw     mm2, 12" \
    "packuswb  mm2, mm2" \
    "pcmpeqb   mm3, mm3" \
    "psrlw     mm3, 14" \
    "packuswb  mm3, mm3" \
    "movq      mm4, [edx]" \
    "movq      mm5, [eax]" \
    "movq      mm6, mm5" \
    "pand      mm6, mm2" \
    "movq      mm7, mm4" \
    "pand      mm7, mm3" \
    "psllw     mm7, 4" \
    "por       mm6, mm7" \
    "pxor      mm7, mm7" \
    "punpcklbw mm7, mm6" \
    "pmaddwd   mm7, [ecx]" \
    "pxor      mm0, mm0" \
    "punpckhbw mm0, mm6" \
    "pmaddwd   mm0, [ecx+8]" \
    "movq      mm6, mm5" \
    "psrlw     mm6, 4" \
    "paddd     mm1, mm7" \
    "paddd     mm1, mm0" \
    "pand      mm6, mm2" \
    "movq      mm7, mm4" \
    "psrlw     mm7, 4" \
    "pand      mm7, mm3" \
    "psllw     mm7, 4" \
    "por       mm6, mm7" \
    "pxor      mm7, mm7" \
    "punpcklbw mm7, mm6" \
    "pmaddwd   mm7, [ecx+32]" \
    "pxor      mm0, mm0" \
    "punpckhbw mm0, mm6" \
    "pmaddwd   mm0, [ecx+40]" \
    "movq      mm5, [eax+8]" \
    "movq      mm6, mm5" \
    "paddd     mm1, mm7" \
    "paddd     mm1, mm0" \
    "pand      mm6, mm2" \
    "movq      mm7, mm4" \
    "psrlw     mm7, 2" \
    "pand      mm7, mm3" \
    "psllw     mm7, 4" \
    "por       mm6, mm7" \
    "pxor      mm7, mm7" \
    "punpcklbw mm7, mm6" \
    "pmaddwd   mm7, [ecx+16]" \
    "pxor      mm0, mm0" \
    "punpckhbw mm0, mm6" \
    "pmaddwd   mm0, [ecx+24]" \
    "movq      mm6, mm5" \
    "psrlw     mm6, 4" \
    "paddd     mm1, mm7" \
    "paddd     mm1, mm0" \
    "pand      mm6, mm2" \
    "movq      mm7, mm4" \
    "psrlw     mm7, 6" \
    "pand      mm7, mm3" \
    "psllw     mm7, 4" \
    "por       mm6, mm7" \
    "pxor      mm7, mm7" \
    "punpcklbw mm7, mm6" \
    "pmaddwd   mm7, [ecx+48]" \
    "pxor      mm0, mm0" \
    "punpckhbw mm0, mm6" \
    "pmaddwd   mm0, [ecx+56]" \
    "paddd     mm1, mm7" \
    "paddd     mm1, mm0" \
    "movq      mm6, mm1" \
    "psrlq     mm6, 32" \
    "paddd     mm1, mm6" \
    "movd      eax, mm1" \
    __parm [__eax] [__edx] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* Not static: ops_mmx.h declares it with external linkage. */
int32_t dot32_q61_mmx(const unsigned char *w4, const unsigned char *w2,
                             const int16_t *x) {
    return lz_dot32_q61_asm(w4, w2, x);
}

/* PAIRED twin: two tokens, ONE two-plane merge. Q6_1 gains the most of
   the three formats from pairing - see row_q61_mmx_asm's own comment. */
extern void lz_dot32_q61_asm_2(const unsigned char *w4, const unsigned char *w2,
                               const int16_t *xa, const int16_t *xb,
                               int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_q61_asm_2 = \
    ".586" \
    "pcmpeqb   mm0, mm0" \
    "psrlw     mm0, 12" \
    "packuswb  mm0, mm0" \
    "pcmpeqb   mm1, mm1" \
    "psrlw     mm1, 14" \
    "packuswb  mm1, mm1" \
    "movq      mm2, [edx]" \
    "pxor      mm3, mm3" \
    "pxor      mm4, mm4" \
    "movq      mm5, [eax]" \
    "pand      mm5, mm0" \
    "movq      mm6, mm2" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [ebx]" \
    "pmaddwd   mm7, [ecx]" \
    "paddd     mm3, mm6" \
    "paddd     mm4, mm7" \
    "pxor      mm6, mm6" \
    "punpckhbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [ebx+8]" \
    "pmaddwd   mm7, [ecx+8]" \
    "movq      mm5, [eax]" \
    "paddd     mm3, mm6" \
    "paddd     mm4, mm7" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm0" \
    "movq      mm6, mm2" \
    "psrlw     mm6, 4" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [ebx+32]" \
    "pmaddwd   mm7, [ecx+32]" \
    "paddd     mm3, mm6" \
    "paddd     mm4, mm7" \
    "pxor      mm6, mm6" \
    "punpckhbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [ebx+40]" \
    "pmaddwd   mm7, [ecx+40]" \
    "movq      mm5, [eax+8]" \
    "paddd     mm3, mm6" \
    "paddd     mm4, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm2" \
    "psrlw     mm6, 2" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [ebx+16]" \
    "pmaddwd   mm7, [ecx+16]" \
    "paddd     mm3, mm6" \
    "paddd     mm4, mm7" \
    "pxor      mm6, mm6" \
    "punpckhbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [ebx+24]" \
    "pmaddwd   mm7, [ecx+24]" \
    "movq      mm5, [eax+8]" \
    "paddd     mm3, mm6" \
    "paddd     mm4, mm7" \
    "psrlw     mm5, 4" \
    "pand      mm5, mm0" \
    "movq      mm6, mm2" \
    "psrlw     mm6, 6" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [ebx+48]" \
    "pmaddwd   mm7, [ecx+48]" \
    "paddd     mm3, mm6" \
    "paddd     mm4, mm7" \
    "pxor      mm6, mm6" \
    "punpckhbw mm6, mm5" \
    "movq      mm7, mm6" \
    "pmaddwd   mm6, [ebx+56]" \
    "pmaddwd   mm7, [ecx+56]" \
    "paddd     mm3, mm6" \
    "paddd     mm4, mm7" \
    "movq      mm6, mm3" \
    "psrlq     mm6, 32" \
    "paddd     mm3, mm6" \
    "movd      [esi], mm3" \
    "movq      mm6, mm4" \
    "psrlq     mm6, 32" \
    "paddd     mm4, mm6" \
    "movd      [edi], mm4" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] [__edi] \
    __modify [8087]

/* The paired kernel's address. */
void dot32_q61_mmx_2(const unsigned char *w4, const unsigned char *w2,
                     const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob) {
    lz_dot32_q61_asm_2(w4, w2, xa, xb, oa, ob);
}

extern void lz_dot128_q61_asm(const unsigned char *w4, const unsigned char *w2,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q61_asm = \
    ".586" \
    "pcmpeqb   mm0, mm0" \
    "psrlw     mm0, 12" \
    "packuswb  mm0, mm0" \
    "pcmpeqb   mm1, mm1" \
    "psrlw     mm1, 14" \
    "packuswb  mm1, mm1" \
    "pxor      mm2, mm2" \
    "movq      mm3, [edx]" \
    "movq      mm4, [eax]" \
    "movq      mm5, mm4" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+8]" \
    "movq      mm5, mm4" \
    "psrlw     mm5, 4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 4" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+32]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+40]" \
    "movq      mm4, [eax+8]" \
    "movq      mm5, mm4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 2" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+16]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+24]" \
    "movq      mm5, mm4" \
    "psrlw     mm5, 4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 6" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+48]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+56]" \
    "movq      mm3, [edx+8]" \
    "movq      mm4, [eax+16]" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm5, mm2" \
    "psrlq     mm5, 32" \
    "paddd     mm2, mm5" \
    "movd      [ebx], mm2" \
    "pxor      mm2, mm2" \
    "movq      mm5, mm4" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+64]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+72]" \
    "movq      mm5, mm4" \
    "psrlw     mm5, 4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 4" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+96]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+104]" \
    "movq      mm4, [eax+24]" \
    "movq      mm5, mm4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 2" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+80]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+88]" \
    "movq      mm5, mm4" \
    "psrlw     mm5, 4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 6" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+112]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+120]" \
    "movq      mm3, [edx+16]" \
    "movq      mm4, [eax+32]" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm5, mm2" \
    "psrlq     mm5, 32" \
    "paddd     mm2, mm5" \
    "movd      [ebx+4], mm2" \
    "pxor      mm2, mm2" \
    "movq      mm5, mm4" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+128]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+136]" \
    "movq      mm5, mm4" \
    "psrlw     mm5, 4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 4" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+160]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+168]" \
    "movq      mm4, [eax+40]" \
    "movq      mm5, mm4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 2" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+144]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+152]" \
    "movq      mm5, mm4" \
    "psrlw     mm5, 4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 6" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+176]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+184]" \
    "movq      mm3, [edx+24]" \
    "movq      mm4, [eax+48]" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm5, mm2" \
    "psrlq     mm5, 32" \
    "paddd     mm2, mm5" \
    "movd      [ebx+8], mm2" \
    "pxor      mm2, mm2" \
    "movq      mm5, mm4" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+192]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+200]" \
    "movq      mm5, mm4" \
    "psrlw     mm5, 4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 4" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+224]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+232]" \
    "movq      mm4, [eax+56]" \
    "movq      mm5, mm4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 2" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+208]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+216]" \
    "movq      mm5, mm4" \
    "psrlw     mm5, 4" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "pand      mm5, mm0" \
    "movq      mm6, mm3" \
    "psrlw     mm6, 6" \
    "pand      mm6, mm1" \
    "psllw     mm6, 4" \
    "por       mm5, mm6" \
    "pxor      mm6, mm6" \
    "punpcklbw mm6, mm5" \
    "pmaddwd   mm6, [ecx+240]" \
    "pxor      mm7, mm7" \
    "punpckhbw mm7, mm5" \
    "pmaddwd   mm7, [ecx+248]" \
    "paddd     mm2, mm6" \
    "paddd     mm2, mm7" \
    "movq      mm5, mm2" \
    "psrlq     mm5, 32" \
    "paddd     mm2, mm5" \
    "movd      [ebx+12], mm2" \
    "emms" \
    __parm [__eax] [__edx] [__ecx] [__ebx] \
    __modify [8087]

/* Real, addressable twin of lz_dot128_q61_asm - needed cross-TU by
   src/ops_sse2.c's row_q61_sse2_asm, ONLY in the -DLZ_SSE2_GROUP=0 A/B
   control build. Same reasoning as lz_dot128_x16_mmx_w (Q8_0) and
   lz_dot128_q41_mmx_w (Q4_1). */
void lz_dot128_q61_mmx_w(const unsigned char *w4, const unsigned char *w2,
                         const int16_t *x, int32_t *out4) {
    lz_dot128_q61_asm(w4, w2, x, out4);
}

/* row_q61_mmx_asm: the real call boundary from its own caller
   (matmul_q61_impl, src/ops.c, dispatched once per matmul via
   LZ_ROW_Q61). The whole function adds no new call sites - same
   argument as row_q8_mmx_asm/row_q41_mmx_asm. g_pair is accessed via
   lz_pair_mode(), same as the other two formats' row functions. */
void row_q61_mmx_asm(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *w2 = (const unsigned char *)c->w2;
    const unsigned char *wend = (const unsigned char *)c->pf_end;
    const unsigned char *wend2 = (const unsigned char *)c->pf_end2;
    int tk, s;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair whenever a second token exists. Q6_1 gains the most of
               the three formats (weight side 61% of the kernel's ops);
               the gcc twin measured 1.271x on prefill. Not measured on
               target - see row_q8_mmx_asm. */
            if (c->nt - tk >= 2 && lz_pair_mode()) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                int s2;
                for (s2 = 0; s2 < c->nb; s2++) {
                    LZ_Q61_PF(s2);
                    lz_dot32_q61_asm_2(wn + (size_t)s2 * 16, w2 + (size_t)s2 * 8,
                                       xwt + (size_t)s2 * 32,
                                       xw2 + (size_t)s2 * 32,
                                       acc + s2, accb + s2);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            if ((c->nb & 3) == 0) LZ_Q61_GROUP4(lz_dot128_q61_asm);
            else               LZ_Q61_ACC(dot32_q61_mmx);
        }
    }
}

/* ---- Q16_0 32/128-element dot products and row kernel, MMX tier -
   in this TU. No pairing (Q16_0 never had it - matches T2's shape,
   not Q8/Q41/Q61's). */
extern int32_t lz_dot32_q16_asm(const int16_t *w, const int16_t *x);
#pragma aux lz_dot32_q16_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "pxor      mm2, mm2" \
    "pxor      mm3, mm3" \
    "movq      mm4, [eax]" \
    "movq      mm5, [eax+16]" \
    "movq      mm6, [eax+32]" \
    "movq      mm7, [eax+48]" \
    "pmaddwd   mm4, [ecx]" \
    "pmaddwd   mm5, [ecx+16]" \
    "pmaddwd   mm6, [ecx+32]" \
    "pmaddwd   mm7, [ecx+48]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, [eax+8]" \
    "movq      mm5, [eax+24]" \
    "movq      mm6, [eax+40]" \
    "movq      mm7, [eax+56]" \
    "pmaddwd   mm4, [ecx+8]" \
    "pmaddwd   mm5, [ecx+24]" \
    "pmaddwd   mm6, [ecx+40]" \
    "pmaddwd   mm7, [ecx+56]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "paddd     mm0, mm1" \
    "paddd     mm2, mm3" \
    "paddd     mm0, mm2" \
    "movq      mm4, mm0" \
    "psrlq     mm4, 32" \
    "paddd     mm0, mm4" \
    "movd      eax, mm0" \
    __parm [__eax] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* Not static: ops_mmx.h declares it with external linkage. */
int32_t dot32_q16_mmx(const int16_t *w, const int16_t *x) {
    return lz_dot32_q16_asm(w, x);
}

/* Two tokens, one weight load - the Watcom twin of dot32_q16_mmx_2
   (src/ops_kernel_dot_mmx.h), and the q16_0 member of the g_pair
   family q8_0/q4_1/q6_1 already had.
   THE WEIGHT GOES TO mm0 AND IS COPIED, not re-loaded: pmaddwd
   destroys its first operand, so the single-token form's
   `movq mm1,[eax+N]` + `pmaddwd mm1,[ecx+N]` cannot simply run twice
   against two activation pointers - that would reload the weight and
   save nothing, which is the whole point of the pair. mm6 and mm7 are
   the two accumulators, mm1/mm2 the per-step temporaries.
   Same parameter registers as lz_dot32_x16_asm_2 above. */
extern void lz_dot32_q16_asm_2(const int16_t *w, const int16_t *xa,
                               const int16_t *xb, int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_q16_asm_2 = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "movq      mm2, [eax]" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [edx]" \
    "movq      mm4, mm2" \
    "pmaddwd   mm4, [ebx]" \
    "movq      mm2, [eax+8]" \
    "paddd     mm0, mm3" \
    "paddd     mm1, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [edx+8]" \
    "movq      mm4, mm2" \
    "pmaddwd   mm4, [ebx+8]" \
    "movq      mm2, [eax+16]" \
    "paddd     mm0, mm3" \
    "paddd     mm1, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [edx+16]" \
    "movq      mm4, mm2" \
    "pmaddwd   mm4, [ebx+16]" \
    "movq      mm2, [eax+24]" \
    "paddd     mm0, mm3" \
    "paddd     mm1, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [edx+24]" \
    "movq      mm4, mm2" \
    "pmaddwd   mm4, [ebx+24]" \
    "movq      mm2, [eax+32]" \
    "paddd     mm0, mm3" \
    "paddd     mm1, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [edx+32]" \
    "movq      mm4, mm2" \
    "pmaddwd   mm4, [ebx+32]" \
    "movq      mm2, [eax+40]" \
    "paddd     mm0, mm3" \
    "paddd     mm1, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [edx+40]" \
    "movq      mm4, mm2" \
    "pmaddwd   mm4, [ebx+40]" \
    "movq      mm2, [eax+48]" \
    "paddd     mm0, mm3" \
    "paddd     mm1, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [edx+48]" \
    "movq      mm4, mm2" \
    "pmaddwd   mm4, [ebx+48]" \
    "movq      mm2, [eax+56]" \
    "paddd     mm0, mm3" \
    "paddd     mm1, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [edx+56]" \
    "movq      mm4, mm2" \
    "pmaddwd   mm4, [ebx+56]" \
    "paddd     mm0, mm3" \
    "movq      mm3, mm0" \
    "paddd     mm1, mm4" \
    "psrlq     mm3, 32" \
    "paddd     mm0, mm3" \
    "movd      [ecx], mm0" \
    "movq      mm4, mm1" \
    "psrlq     mm4, 32" \
    "paddd     mm1, mm4" \
    "movd      [esi], mm1" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [8087]

/* The paired kernel's address. */
void dot32_q16_mmx_2(const int16_t *w, const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob) {
    lz_dot32_q16_asm_2(w, xa, xb, oa, ob);
}

extern void lz_dot128_q16_asm(const int16_t *w, const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q16_asm = \
    ".586" \
    "pxor      mm0, mm0" \
    "pxor      mm1, mm1" \
    "pxor      mm2, mm2" \
    "pxor      mm3, mm3" \
    "movq      mm4, [eax]" \
    "movq      mm5, [eax+64]" \
    "movq      mm6, [eax+128]" \
    "movq      mm7, [eax+192]" \
    "pmaddwd   mm4, [ecx]" \
    "pmaddwd   mm5, [ecx+64]" \
    "pmaddwd   mm6, [ecx+128]" \
    "pmaddwd   mm7, [ecx+192]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, [eax+8]" \
    "movq      mm5, [eax+72]" \
    "movq      mm6, [eax+136]" \
    "movq      mm7, [eax+200]" \
    "pmaddwd   mm4, [ecx+8]" \
    "pmaddwd   mm5, [ecx+72]" \
    "pmaddwd   mm6, [ecx+136]" \
    "pmaddwd   mm7, [ecx+200]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, [eax+16]" \
    "movq      mm5, [eax+80]" \
    "movq      mm6, [eax+144]" \
    "movq      mm7, [eax+208]" \
    "pmaddwd   mm4, [ecx+16]" \
    "pmaddwd   mm5, [ecx+80]" \
    "pmaddwd   mm6, [ecx+144]" \
    "pmaddwd   mm7, [ecx+208]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, [eax+24]" \
    "movq      mm5, [eax+88]" \
    "movq      mm6, [eax+152]" \
    "movq      mm7, [eax+216]" \
    "pmaddwd   mm4, [ecx+24]" \
    "pmaddwd   mm5, [ecx+88]" \
    "pmaddwd   mm6, [ecx+152]" \
    "pmaddwd   mm7, [ecx+216]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, [eax+32]" \
    "movq      mm5, [eax+96]" \
    "movq      mm6, [eax+160]" \
    "movq      mm7, [eax+224]" \
    "pmaddwd   mm4, [ecx+32]" \
    "pmaddwd   mm5, [ecx+96]" \
    "pmaddwd   mm6, [ecx+160]" \
    "pmaddwd   mm7, [ecx+224]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, [eax+40]" \
    "movq      mm5, [eax+104]" \
    "movq      mm6, [eax+168]" \
    "movq      mm7, [eax+232]" \
    "pmaddwd   mm4, [ecx+40]" \
    "pmaddwd   mm5, [ecx+104]" \
    "pmaddwd   mm6, [ecx+168]" \
    "pmaddwd   mm7, [ecx+232]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, [eax+48]" \
    "movq      mm5, [eax+112]" \
    "movq      mm6, [eax+176]" \
    "movq      mm7, [eax+240]" \
    "pmaddwd   mm4, [ecx+48]" \
    "pmaddwd   mm5, [ecx+112]" \
    "pmaddwd   mm6, [ecx+176]" \
    "pmaddwd   mm7, [ecx+240]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, [eax+56]" \
    "movq      mm5, [eax+120]" \
    "movq      mm6, [eax+184]" \
    "movq      mm7, [eax+248]" \
    "pmaddwd   mm4, [ecx+56]" \
    "pmaddwd   mm5, [ecx+120]" \
    "pmaddwd   mm6, [ecx+184]" \
    "pmaddwd   mm7, [ecx+248]" \
    "paddd     mm0, mm4" \
    "paddd     mm1, mm5" \
    "paddd     mm2, mm6" \
    "paddd     mm3, mm7" \
    "movq      mm4, mm0" \
    "punpckldq mm4, mm1" \
    "punpckhdq mm0, mm1" \
    "paddd     mm4, mm0" \
    "movq      [ebx], mm4" \
    "movq      mm4, mm2" \
    "punpckldq mm4, mm3" \
    "punpckhdq mm2, mm3" \
    "paddd     mm4, mm2" \
    "movq      [ebx+8], mm4" \
    "emms" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

/* Real, addressable twin of lz_dot128_q16_asm - needed cross-TU by
   src/ops_sse2.c's row_q16_sse2_asm, ONLY in the -DLZ_SSE2_GROUP=0 A/B
   control build. Same reasoning as the other three formats' *_mmx_w
   wrappers. */
void lz_dot128_q16_mmx_w(const int16_t *w, const int16_t *x, int32_t *out4) {
    lz_dot128_q16_asm(w, x, out4);
}

/* row_q16_mmx_asm: the real call boundary (matmul_q16-adjacent
   dispatch, src/ops.c, via LZ_ROW_Q16). The whole function adds no
   new call sites. */
void row_q16_mmx_asm(const lz_row_ctx *c) {
    const int16_t *wr = (const int16_t *)c->w4;
    const int16_t *wend = (const int16_t *)c->pf_end;
    int tk, g;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair before grouping, the same order and the same reason as
               the gcc twin: the pair halves the weight traffic while the
               group only amortizes the per-32 call, and a token pair cannot
               be expressed once the group loop owns the sub-block index.
               An odd nt leaves its last token on the paths below. */
            if (c->nt - tk >= 2 && lz_pair_mode()) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (g = 0; g < c->nb; g++) {
                    LZ_Q16_PF(g);
                    lz_dot32_q16_asm_2(wr + (size_t)g * 32,
                                       xwt + (size_t)g * 32,
                                       xw2 + (size_t)g * 32,
                                       acc + g, accb + g);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            if ((c->nb & 3) == 0) LZ_Q16_GROUP4(lz_dot128_q16_asm);
            else               LZ_Q16_ACC(dot32_q16_mmx);
        }
    }
}

/* ---- Hadamard: one butterfly stage of lz_fwht_i32, Watcom twin.
   gcc twin: lz_fwht_stage_mmx, the intrinsics body below.

   Two int32 per movq, paddd and psubd for the two halves of the
   butterfly, no shuffle - see the gcc twin's comment for why len >= 2
   makes the partners contiguous and why len == 1 stays in the C.

   The inner loop is written over BYTE offsets so `len` enters as a byte
   count once (shl 2) instead of being scaled at every index. edi walks
   the low half, esi the high; both advance by 8 per iteration, which is
   the two int32 a movq moves. */
extern void lz_fwht_stage_mmx_asm(int32_t *y, int n, int len);
#pragma aux lz_fwht_stage_mmx_asm = \
    "shl    ecx, 2"                 /* len -> bytes */          \
    "shl    edx, 2"                 /* n   -> bytes */          \
    "lea    ebx, [ecx+ecx]"         /* stride = 2*len bytes */  \
    "add    edx, eax"               /* end of y */              \
    "outer:"                                                    \
    "mov    edi, eax"                                           \
    "lea    esi, [eax+ecx]"                                     \
    "mov    ebp, ecx"                                           \
    "inner:"                                                    \
    "movq   mm0, [edi]"                                         \
    "movq   mm1, [esi]"                                         \
    "movq   mm2, mm0"                                           \
    "paddd  mm0, mm1"                                           \
    "psubd  mm2, mm1"                                           \
    "movq   [edi], mm0"                                         \
    "movq   [esi], mm2"                                         \
    "add    edi, 8"                                             \
    "add    esi, 8"                                             \
    "sub    ebp, 8"                                             \
    "jnz    inner"                                              \
    "add    eax, ebx"                                           \
    "cmp    eax, edx"                                           \
    "jb     outer"                                              \
    parm [eax] [edx] [ecx]                                      \
    modify [eax ebx ecx edx esi edi ebp];

/* Thin wrapper, and it is not optional: a #pragma aux body expands at
   the CALL SITE, and the call site is src/fwht.c, which is not this TU
   and therefore never sees the pragma - it emitted a call to a symbol
   nothing defines and the link failed. Same shape as
   lz_dot128_q16_mmx_w and lz_amax32_sse2_w, and for the same reason:
   the wrapper is the real symbol, the pragma expands inside it. */
void lz_fwht_stage_mmx(int32_t *y, int n, int len) {
    lz_fwht_stage_mmx_asm(y, n, len);
}

#else /* gcc: intrinsics */

#include "mmx_compat.h"
#include <mmintrin.h>

/* ---- Q8 activation rounding, the PACK HALF, MMX tier.
   Watcom twin: lz_q8round32_pack_mmx_asm, the #pragma aux above.

   WHY A HALF. Rounding a Q8 group is x[k] * inv, a round, and a clamp
   to [-127, 127]. The multiply and the round are float and MMX has
   neither, which is why this cell read "impossible" for as long as the
   question was whether the WHOLE loop runs here. It is not: the caller
   runs the float half into an int32 scratch with x87, takes one emms,
   and this runs the clamp-and-pack four wide. One emms per block, not
   one per element - that difference is what separates partial
   acceleration from a regression.

   NO pminsw OR pmaxsw, which are SSE1 additions to the MMX register
   file and not MMX; lz_q8round32_sse uses both and this cannot.
   It does not need them:

     packssdw   int32 -> int16, saturating at +-32768. Harmless: the
                clamp below is tighter.
     the floor  max(w, -127) as pcmpgtw/pand/pandn/por. Four
                instructions where SSE1 spends one, and still four per
                FOUR lanes against the C's two compares per element.
     packsswb   int16 -> int8, saturating at [-128, 127]. The +127 half
                of the C clamp IS this saturation, exactly; only the
                -128 lane needed the floor above, and it got it before
                this pack rather than after, where it would be
                unrecoverable.

   Bit-identical to the C by construction: every value the C would
   clamp, these two saturations and one floor clamp to the same
   number, and the arithmetic that produced the int32 is the C's own. */
void lz_q8round32_pack_mmx(const int32_t *q, int8_t *out) {
    const __m64 flr = _mm_set1_pi16((short)-127);
    int b;

    for (b = 0; b < 4; b++) {
        const int32_t *p = q + b * 8;
        __m64 a, c, w0, w1, g0, g1, r;

        memcpy(&a, p,     8);
        memcpy(&c, p + 2, 8);
        w0 = _mm_packs_pi32(a, c);
        memcpy(&a, p + 4, 8);
        memcpy(&c, p + 6, 8);
        w1 = _mm_packs_pi32(a, c);

        g0 = _mm_cmpgt_pi16(flr, w0);
        g1 = _mm_cmpgt_pi16(flr, w1);
        w0 = _mm_or_si64(_mm_and_si64(g0, flr), _mm_andnot_si64(g0, w0));
        w1 = _mm_or_si64(_mm_and_si64(g1, flr), _mm_andnot_si64(g1, w1));

        r = _mm_packs_pi16(w0, w1);
        memcpy(out + b * 8, &r, 8);
    }
    _mm_empty();
}

/* ---- norm_ss_fixed's element loop, the PACK-AND-SQUARE HALF, MMX tier.
   Watcom twin: lz_norm_ss_pack_mmx_asm, the #pragma aux in the other
   arm. SSE1 twin: lz_norm_ss_sse, which does the whole loop because
   cvtps2pi lets it convert in %mm; this cannot, so the caller runs the
   scale and the round in x87 and hands over the int32.

   THE SAME SHAPE AS q8round32_pack_mmx above and for the same reason:
   packssdw saturates int32 to [-32768, 32767], so the C's +32767 clamp
   IS that saturation and only the -32768 lane needs lifting to -32767.
   pmaxsw would do it in one; it is SSE1, so this pays the
   pcmpgtw/pand/pandn/por idiom instead.

   THEN pmaddwd, which is exactly the pairwise sum of products this
   accumulate wants, and the SSE2 twin's bound carries over unchanged:
   |v| <= 32767 gives v*v <= 1073676289 and a pair sums to 2147352578,
   131069 under INT32_MAX. The pair is safe and the NEXT add is not,
   so each pmaddwd result is drained to the int64 immediately rather
   than accumulated in %mm - paddq is SSE2 even there.

   Returns the sum for the 32 elements; qout is written only when the
   caller wants the quantized row, matching the C. */
lz_i64 lz_norm_ss_pack_mmx(const int32_t *q, short *qout) {
    const __m64 flr = _mm_set1_pi16((short)-32767);
    lz_i64 acc = 0;
    int b;

    for (b = 0; b < 8; b++) {
        const int32_t *p = q + b * 4;
        __m64 a, c, w, g, s;
        union { __m64 m; int32_t d[2]; } o;

        memcpy(&a, p,     8);
        memcpy(&c, p + 2, 8);
        w = _mm_packs_pi32(a, c);
        g = _mm_cmpgt_pi16(flr, w);
        w = _mm_or_si64(_mm_and_si64(g, flr), _mm_andnot_si64(g, w));
        if (qout) memcpy(qout + b * 4, &w, 8);
        s = _mm_madd_pi16(w, w);
        o.m = s;
        acc += (lz_i64)o.d[0] + (lz_i64)o.d[1];
    }
    _mm_empty();
    return acc;
}

/* ---- the Q15 table interpolation, MMX tier.
   Watcom twin: lz_lerp_q15_mmx_asm, the #pragma aux in the other arm.

   out[k] = a[k] + (((b[k] - a[k]) * frac[k]) >> 15), which is the tail
   of sigmoid_q15 and lz_exp_fixed. The two TABLE LOOKUPS stay scalar -
   a per-element index needs a gather and that is AVX2, off this target
   family - and the arithmetic around them does not have to.

   PRECONDITION: |b - a| <= 32767, and it is a precondition rather than
   a property because this saturates rather than wrapping if it is
   broken. The callers satisfy it - a and b are ADJACENT entries of a
   384-step table whose whole range is 32768, so their difference is in
   the hundreds - but "the caller happens to" is not a contract, and
   the first version of this comment said "b - a is in the hundreds" as
   though that were one. build/lerp_q15_probe.c drives the boundary
   deliberately: a = 0 with b = 32768 is off by one here, which is what
   found the omission.

   ONLY TWO OF THE THREE VALUES GO THROUGH THE 16-BIT MULTIPLY, and
   getting that wrong is how this silently loses the table's endpoint.
   b - a and frac do; `a` itself reaches 32768 exactly - s = 1.0 rounds
   there - which is one past INT16_MAX, so it stays int32 and joins
   with paddd.

   pmullw and pmulhw give the low and high halves of the signed
   product; punpck[lh]wd rebuilds them into two full int32 each. psrad
   is an ARITHMETIC shift, which is what >> 15 on a signed value has to
   be - b - a is negative wherever the table falls. */
void lz_lerp_q15_mmx(const int32_t *a, const int32_t *b,
                     const int32_t *frac, int32_t *out, int n) {
    int k;

    for (k = 0; k + 3 < n; k += 4) {
        __m64 a0, a1, b0, b1, f0, f1, d, f, lo, hi, p0, p1;

        memcpy(&a0, a + k,     8);
        memcpy(&a1, a + k + 2, 8);
        memcpy(&b0, b + k,     8);
        memcpy(&b1, b + k + 2, 8);
        d = _mm_packs_pi32(_mm_sub_pi32(b0, a0), _mm_sub_pi32(b1, a1));

        memcpy(&f0, frac + k,     8);
        memcpy(&f1, frac + k + 2, 8);
        f = _mm_packs_pi32(f0, f1);

        lo = _mm_mullo_pi16(d, f);
        hi = _mm_mulhi_pi16(d, f);
        p0 = _mm_srai_pi32(_mm_unpacklo_pi16(lo, hi), 15);
        p1 = _mm_srai_pi32(_mm_unpackhi_pi16(lo, hi), 15);

        p0 = _mm_add_pi32(p0, a0);
        p1 = _mm_add_pi32(p1, a1);
        memcpy(out + k,     &p0, 8);
        memcpy(out + k + 2, &p1, 8);
    }
    _mm_empty();
    for (; k < n; k++)
        out[k] = a[k] + (((b[k] - a[k]) * frac[k]) >> 15);
}

/* ---- lz_exp_fixed's Q20 Taylor, MMX tier.
   Watcom twin: lz_exp_q20_2_mmx_asm, the #pragma aux in the other arm.
   SSE2 twin: lz_exp_q20_simd (src/ops_sse2.c), which has pmuludq and so
   spends one instruction where this spends a limb decomposition.

   THE WHOLE CELL IS ABOUT ONE MISSING INSTRUCTION. Every multiply in
   this operator is 32 x 32 -> 64; MMX's widest is pmaddwd, 16 x 16 ->
   32. So each product is rebuilt from pieces small enough that pmaddwd
   holds them and the running value never leaves an int32 lane. Two
   elements a pass, against SSE2's four.

   pmaddwd IS USED AS A PLAIN 16x16->32 MULTIPLY here, not as a dot
   product: the operands sit in int32 lanes with zero high halves, so
   as int16 they read [x, 0, x', 0] and the pairwise add contributes
   0*0. That costs nothing and is the only 16x16->32 MMX has - pmullo
   and pmulhi give two halves that then need reassembling.

   EVERY OPERAND MUST FIT SIGNED int16, which is what drives the splits
   below, and s is exactly the value that does not: it reaches 2^15,
   one past int16. So s is carried as u = s >> 1 and b = s & 1 - u is
   at most 2^14 - and b is folded back with a mask instead of a branch.
   Splitting the CONSTANTS at 2^15 rather than 2^16 is the same
   constraint seen from the other side.

     t1  LN2 = 22*2^15 + 5921, so LN2*s = 22*u*2^16 + 2*(5921*u) +
         LN2*b. The 2^16 term needs no multiply at all: shifting the
         sum right by 16 turns it into a plain add.
     ss  s*s = 4*u^2 + (4u + 1 masked by b), exact, one pmaddwd.
     t2  LN2SQ*(s*s) >> 40 with s*s up to 2^30. Three 10-bit limbs of
         s*s, each times LN2SQ (itself split 7*2^15 + 22530), then two
         >>10 stages instead of one >>20 so no partial sum exceeds
         int32. 10 bits is the largest limb width that keeps
         LN2SQ*limb inside int32.
     out cq = 2^20 + d, so tab*cq = tab*2^20 + tab*d and
         prod = tab + ((tab*d + 2^19) >> 20) - the big half of the
         product cancels against the shift and never has to be formed.
         d < 2^15 and tab < 2^21, so one 2^11 split of tab finishes it.

   NO SATURATION ANYWHERE. Every value above is proved to fit its lane,
   so packssdw never appears and there is nothing to clamp; the shifts
   are all psrld (logical) because every operand is non-negative. */
void lz_exp_q20_mmx(const int32_t *tab, const int32_t *s,
                    int32_t *prod, int n) {
    const __m64 k1     = _mm_set1_pi32(1);
    const __m64 k1023  = _mm_set1_pi32(1023);
    const __m64 k2047  = _mm_set1_pi32(2047);
    const __m64 kr19   = _mm_set1_pi32(1 << 19);
    const __m64 kln2   = _mm_set1_pi32(726817);
    const __m64 k22    = _mm_set1_pi32(22);
    const __m64 k5921  = _mm_set1_pi32(5921);
    const __m64 k7     = _mm_set1_pi32(7);
    const __m64 k22530 = _mm_set1_pi32(22530);
    const __m64 zero   = _mm_setzero_si64();
    int k;

    for (k = 0; k + 1 < n; k += 2) {
        __m64 sv, tv, u, b, mb, av, bv, xv, t1, u2, ss;
        __m64 l0, l1, l2, m0, m1, m2, inner, t2, d, a1, a0;

        memcpy(&sv, s + k, 8);
        memcpy(&tv, tab + k, 8);
        u  = _mm_srli_pi32(sv, 1);
        b  = _mm_and_si64(sv, k1);
        mb = _mm_sub_pi32(zero, b);          /* 0 or all-ones */

        av = _mm_madd_pi16(u, k22);          /* 22*u   <= 2^19 */
        bv = _mm_madd_pi16(u, k5921);        /* 5921*u <= 2^27 */
        xv = _mm_add_pi32(_mm_slli_pi32(bv, 1), _mm_and_si64(kln2, mb));
        xv = _mm_add_pi32(xv, kr19);
        t1 = _mm_srli_pi32(_mm_add_pi32(av, _mm_srli_pi32(xv, 16)), 4);

        u2 = _mm_madd_pi16(u, u);            /* u*u <= 2^28 */
        ss = _mm_add_pi32(_mm_slli_pi32(u2, 2),
                          _mm_and_si64(_mm_add_pi32(_mm_slli_pi32(u, 2),
                                                    k1), mb));

        l0 = _mm_and_si64(ss, k1023);
        l1 = _mm_and_si64(_mm_srli_pi32(ss, 10), k1023);
        l2 = _mm_srli_pi32(ss, 20);
        m0 = _mm_add_pi32(_mm_madd_pi16(l0, k22530),
                          _mm_slli_pi32(_mm_madd_pi16(l0, k7), 15));
        m1 = _mm_add_pi32(_mm_madd_pi16(l1, k22530),
                          _mm_slli_pi32(_mm_madd_pi16(l1, k7), 15));
        m2 = _mm_add_pi32(_mm_madd_pi16(l2, k22530),
                          _mm_slli_pi32(_mm_madd_pi16(l2, k7), 15));
        inner = _mm_srli_pi32(_mm_add_pi32(m1, _mm_srli_pi32(m0, 10)), 10);
        t2 = _mm_srli_pi32(_mm_add_pi32(_mm_add_pi32(m2, kr19), inner), 20);

        d  = _mm_add_pi32(t1, t2);           /* cq - 2^20, < 2^15 */
        a1 = _mm_srli_pi32(tv, 11);
        a0 = _mm_and_si64(tv, k2047);
        {
            __m64 p1 = _mm_madd_pi16(a1, d);
            __m64 p0 = _mm_add_pi32(_mm_madd_pi16(a0, d), kr19);
            __m64 r  = _mm_srli_pi32(
                           _mm_add_pi32(p1, _mm_srli_pi32(p0, 11)), 9);
            r = _mm_add_pi32(tv, r);
            memcpy(prod + k, &r, 8);
        }
    }
    _mm_empty();
    for (; k < n; k++) {
        int32_t sk = s[k], cq;
        cq = (1 << 20)
           + (int32_t)((((lz_i64)726817 * sk) + (1 << 19)) >> 20)
           + (int32_t)((((lz_i64)251906 * ((lz_i64)sk * sk))
                        + (LZ_I64_C(1) << 39)) >> 40);
        prod[k] = (int32_t)((((lz_i64)tab[k] * cq) + (1 << 19)) >> 20);
    }
}

/* ---- wsum: attention scoring's weighted-sum dot, LZ_WSUM_PAIR's gcc
   twin. Watcom twin: lz_wsum_pair_asm, src/ops.c ~3960-4041.
   Sign-extend two int8 rows to int16, interleave so one pmaddwd covers a
   (rowA,rowB) pair against (ckA,ckB), accumulate into acc32 in place.
   Emits no emms; the driver does one per chunk, and that chunk's
   row-pair loop contains no x87. */
void lz_wsum_pair_mmx(const int8_t *rowA, const int8_t *rowB,
                      const int16_t *coef, int32_t *acc32) {
    __m64 c, ra, rb, a0, b0, p, q, acc;
    int off;
    memcpy(&c, coef, 8);
    for (off = 0; off < 32; off += 8) {
        memcpy(&ra, rowA + off, 8);
        memcpy(&rb, rowB + off, 8);

        a0 = _mm_srai_pi16(_mm_unpacklo_pi8(ra, ra), 8);
        b0 = _mm_srai_pi16(_mm_unpacklo_pi8(rb, rb), 8);
        p = _mm_unpacklo_pi16(a0, b0);        /* [a0,b0,a1,b1] */
        q = _mm_unpackhi_pi16(a0, b0);        /* [a2,b2,a3,b3] */
        p = _mm_madd_pi16(p, c);
        q = _mm_madd_pi16(q, c);
        memcpy(&acc, acc32 + off, 8);
        acc = _mm_add_pi32(acc, p);
        memcpy(acc32 + off, &acc, 8);
        memcpy(&acc, acc32 + off + 2, 8);
        acc = _mm_add_pi32(acc, q);
        memcpy(acc32 + off + 2, &acc, 8);

        a0 = _mm_srai_pi16(_mm_unpackhi_pi8(ra, ra), 8);
        b0 = _mm_srai_pi16(_mm_unpackhi_pi8(rb, rb), 8);
        p = _mm_unpacklo_pi16(a0, b0);
        q = _mm_unpackhi_pi16(a0, b0);
        p = _mm_madd_pi16(p, c);
        q = _mm_madd_pi16(q, c);
        memcpy(&acc, acc32 + off + 4, 8);
        acc = _mm_add_pi32(acc, p);
        memcpy(acc32 + off + 4, &acc, 8);
        memcpy(&acc, acc32 + off + 6, 8);
        acc = _mm_add_pi32(acc, q);
        memcpy(acc32 + off + 6, &acc, 8);
    }
}

/* ---- GDN/KDA fixed-point pass 2. Watcom twins:
   lz_p2_mul32_mmx_asm / lz_p2_split32_mmx_asm / lz_p2_split32_sse_asm,
   src/ops_kernel_p2.h. Full algorithm description (what each field of
   lz_p2_blk means, the ISA inventory, why SSE1 has exactly one
   instruction of content here) lives with that file - it is the
   authority on the "what" and "why". */

/* 16-byte-aligned scratch, by hand: neither compiler can be asked for
   it here (Watcom is built with -zp4, capping member alignment at 4;
   C89/C99 have no alignment specifier), and SSE2's movdqa faults on a
   misaligned address. One buffer for the whole engine - the kernels
   below are called back to back inside one group and nothing
   re-enters between them. */
lz_p2_blk *p2_blk(void) {
    static char raw[sizeof(lz_p2_blk) + 15];
    size_t off = (size_t)((uintptr_t)(void *)raw & 15u);
    return (lz_p2_blk *)(void *)(raw + ((16u - off) & 15u));
}

/* memcpy for every load and store, not a cast to __m64*: the state
   planes belong to the caller and nothing here may assume their
   alignment. */
void lz_p2_mul32_mmx(const int8_t *hi, const int8_t *lo,
                     const int16_t *dq, lz_p2_blk *blk) {
    __m64 zero = _mm_setzero_si64();
    __m64 acc1 = _mm_setzero_si64();
    __m64 acc2 = _mm_setzero_si64();
    __m64 mul;
    int q;

    memcpy(&mul, blk->mul, 8);
    for (q = 0; q < 8; q++) {
        __m64 hw, lw, H, d, p0, p1, s0, s1, a0, a1, m;
        int32_t hv, lv;

        memcpy(&hv, hi + q * 4, 4);
        memcpy(&lv, lo + q * 4, 4);
        hw = _mm_unpacklo_pi8(zero, _mm_cvtsi32_si64(hv));   /* hi * 256 */
        lw = _mm_cvtsi32_si64(lv);
        lw = _mm_unpacklo_pi8(lw, lw);
        lw = _mm_srai_pi16(lw, 8);                           /* lo, signed */
        H  = _mm_add_pi16(hw, lw);
        memcpy(&d, dq + q * 4, 8);
        p0 = _mm_madd_pi16(_mm_unpacklo_pi16(H, d), mul);
        p1 = _mm_madd_pi16(_mm_unpackhi_pi16(H, d), mul);
        memcpy(blk->a + q * 4,     &p0, 8);
        memcpy(blk->a + q * 4 + 2, &p1, 8);
        s0 = _mm_srai_pi32(p0, 31);
        a0 = _mm_sub_pi32(_mm_xor_si64(p0, s0), s0);         /* |A| */
        s1 = _mm_srai_pi32(p1, 31);
        a1 = _mm_sub_pi32(_mm_xor_si64(p1, s1), s1);
        m    = _mm_cmpgt_pi32(acc1, a0);
        acc1 = _mm_or_si64(_mm_and_si64(acc1, m), _mm_andnot_si64(m, a0));
        m    = _mm_cmpgt_pi32(acc2, a1);
        acc2 = _mm_or_si64(_mm_and_si64(acc2, m), _mm_andnot_si64(m, a1));
    }
    memcpy(blk->amax,     &acc1, 8);
    memcpy(blk->amax + 2, &acc2, 8);
}

void lz_p2_split32_mmx(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m64 rnd, cnt, k128, kclp;
    int o;

    memcpy(&rnd,  blk->rnd,  8);
    memcpy(&cnt,  blk->cnt,  8);
    memcpy(&k128, blk->k128, 8);
    memcpy(&kclp, blk->kclp, 8);
    for (o = 0; o < 4; o++) {
        __m64 a0, a1, a2, a3, hn0, hn1, h0, h1, l0, l1, t;

        memcpy(&a0, blk->a + o * 8,     8);
        memcpy(&a1, blk->a + o * 8 + 2, 8);
        memcpy(&a2, blk->a + o * 8 + 4, 8);
        memcpy(&a3, blk->a + o * 8 + 6, 8);
        a0 = _mm_sra_pi32(_mm_add_pi32(a0, rnd), cnt);
        a1 = _mm_sra_pi32(_mm_add_pi32(a1, rnd), cnt);
        a2 = _mm_sra_pi32(_mm_add_pi32(a2, rnd), cnt);
        a3 = _mm_sra_pi32(_mm_add_pi32(a3, rnd), cnt);
        hn0 = _mm_packs_pi32(a0, a1);
        hn1 = _mm_packs_pi32(a2, a3);
        h0 = _mm_srai_pi16(_mm_add_pi16(hn0, k128), 8);
        h1 = _mm_srai_pi16(_mm_add_pi16(hn1, k128), 8);
        l0 = _mm_sub_pi16(hn0, _mm_slli_pi16(h0, 8));
        l1 = _mm_sub_pi16(hn1, _mm_slli_pi16(h1, 8));
        l0 = _mm_adds_pi16(_mm_subs_pi16(l0, kclp), kclp);   /* max(l, -127) */
        l1 = _mm_adds_pi16(_mm_subs_pi16(l1, kclp), kclp);
        t = _mm_packs_pi16(h0, h1);
        memcpy(oh + o * 8, &t, 8);
        t = _mm_packs_pi16(l0, l1);
        memcpy(ol + o * 8, &t, 8);
    }
}

/* lz_p2_split32_sse is in src/ops_mmx_sse.c: it needs genuine SSE1,
   and gcc treats -msse as license to assume i686 (CMOV included) for
   the WHOLE translation unit, not just the SSE-touching functions in
   it - measured, see that file's header comment. This file stays
   MMX-only (-march=i586 -mmmx, no -msse) so its plain-MMX functions,
   reachable on Socket-7 parts with no CMOV, cannot pick one up. */

/* ---- GDN/KDA pass 1, 4-lane table contraction. Watcom twin:
   the #pragma aux declaration of this same name in src/ops.c (search
   LZ_GDN_HAVE_MMX). Sign-extend two int8 rows to int16, interleave so
   one pmaddwd consumes a (rowA,rowB) pair against a (cA,cB) coefficient
   pair, four accumulators for u/w x the low/high halves of the 4-lane
   window. */
void lz_gdn1_x4_asm(const int8_t *rowA, int32_t vd, const int16_t *ctab,
                    int32_t kpairs, int32_t *out8) {
    __m64 accU0 = _mm_setzero_si64(), accU1 = _mm_setzero_si64();
    __m64 accW0 = _mm_setzero_si64(), accW1 = _mm_setzero_si64();
    const int8_t *rA = rowA;
    int32_t kp;
    for (kp = 0; kp < kpairs; kp++) {
        __m64 a, b, pl, ph, t, cu, cw;
        int32_t a32, b32;
        memcpy(&a32, rA, 4);
        memcpy(&b32, rA + vd, 4);
        a = _mm_cvtsi32_si64(a32);
        b = _mm_cvtsi32_si64(b32);
        a = _mm_unpacklo_pi8(a, a);
        a = _mm_srai_pi16(a, 8);           /* 4 int16, sign-extended */
        b = _mm_unpacklo_pi8(b, b);
        b = _mm_srai_pi16(b, 8);
        pl = _mm_unpacklo_pi16(a, b);      /* [a0,b0,a1,b1] */
        ph = _mm_unpackhi_pi16(a, b);      /* [a2,b2,a3,b3] */
        memcpy(&cu, ctab + (size_t)kp * 8,     8);
        memcpy(&cw, ctab + (size_t)kp * 8 + 4, 8);
        t = _mm_madd_pi16(pl, cu); accU0 = _mm_add_pi32(accU0, t);
        t = _mm_madd_pi16(pl, cw); accW0 = _mm_add_pi32(accW0, t);
        t = _mm_madd_pi16(ph, cu); accU1 = _mm_add_pi32(accU1, t);
        t = _mm_madd_pi16(ph, cw); accW1 = _mm_add_pi32(accW1, t);
        rA += (size_t)vd * 2;
    }
    memcpy(out8 + 0, &accU0, 8);
    memcpy(out8 + 2, &accU1, 8);
    memcpy(out8 + 4, &accW0, 8);
    memcpy(out8 + 6, &accW1, 8);
    _mm_empty();
}

/* ---- Q8 group-scale amax. Watcom twin: lz_amax32_mmx_asm,
   src/ops_kernel_amax.h. Two lanes of uint32 in one MMX register; the
   mask is built in-register (pcmpeqd then psrld 1) rather than loaded,
   so the routine touches no memory but its input. Leaves MMX state
   dirty - the caller emits one emms after the group loop. */
unsigned lz_amax32_mmx(const float *x, int n) {
    __m64 acc = _mm_setzero_si64();
    __m64 msk = _mm_set1_pi32(0x7FFFFFFF);
    int k;
    for (k = 0; k + 1 < n; k += 2) {
        __m64 v = _mm_and_si64(*(const __m64 *)(const void *)(x + k), msk);
        __m64 gt = _mm_cmpgt_pi32(v, acc);
        acc = _mm_or_si64(_mm_and_si64(v, gt), _mm_andnot_si64(gt, acc));
    }
    {
        union { __m64 m; unsigned u[2]; } o;
        o.m = acc;
        return o.u[0] > o.u[1] ? o.u[0] : o.u[1];
    }
}

/* ---- Hadamard: one butterfly stage of lz_fwht_i32, gcc twin.
   Watcom twin: lz_fwht_stage_mmx_asm, the #pragma aux above.

   ONE STAGE, NOT THE WHOLE TRANSFORM, and the split is where the
   vectorization stops being free. The butterfly at stride `len` pairs
   y[i+j] with y[i+j+len], so for len >= 2 those are two CONTIGUOUS
   pairs of int32 - one movq each, one paddd, one psubd, two movq back,
   and not a single shuffle. At len == 1 the partners are adjacent
   words, which needs an unpack per butterfly and buys nothing, so that
   stage stays in the C.

   BIT-IDENTITY IS BY CONSTRUCTION HERE, unlike every other kernel in
   this file: the transform is int32 add and subtract, wrapping, with no
   rounding, no saturation and no reassociation - each output word is
   the same two inputs combined by the same operator. There is no
   tolerance to argue about and no float path to agree with.

   Emits no emms. The caller does one after the last stage, which is
   where the x87 that follows it begins. */
void lz_fwht_stage_mmx(int32_t *y, int n, int len) {
    __m64 u, v;
    int i, j;
    for (i = 0; i < n; i += (len << 1)) {
        int32_t *a = y + i;
        int32_t *b = y + i + len;
        for (j = 0; j < len; j += 2) {
            memcpy(&u, a + j, 8);
            memcpy(&v, b + j, 8);
            {
                __m64 s = _mm_add_pi32(u, v);
                __m64 d = _mm_sub_pi32(u, v);
                memcpy(a + j, &s, 8);
                memcpy(b + j, &d, 8);
            }
        }
    }
}

/* lz_q8round32_sse is in src/ops_mmx_sse.c, same reason as
   lz_p2_split32_sse above. */

#include "ops_kernel_dot_mmx.h"

#endif /* __WATCOMC__ */

#endif /* LZ_MMX_TU */
