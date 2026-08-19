/* Every function that writes a %mm register - the MMX register file,
   which aliases the x87 stack. The ONLY translation unit built with
   -mmmx (and -msse, for the SSE1-on-%mm cell) on EITHER toolchain (task
   #30): on a 32-bit x87 target, gcc and clang both use MMX for an
   8-byte block store when -mmmx is on even with no MMX intrinsic
   anywhere nearby in the source, and neither issues emms before the x87
   code that follows - which leaves the tag word full and makes the next
   fld overflow to real indefinite (0xFFC00000). Moving every
   %mm-touching function out of ops.c is what makes "ops.c is never
   built with -mmmx" a safe thing to do; see src/ops_mmx.h for the macro
   that tells ops.c's dispatch code these kernels still exist.

   Watcom NOW compiles this file too (it did not before task #30) -
   $(ENG_MMX_GCC) in the Makefile is misnamed as of this split; see its
   own comment. Watcom's #pragma aux bodies moved here from ops.c and
   the ops_kernel_*.h headers because -Nr/-Ns is a hard ceiling wcc386's
   own per-function ".586"/".686" directives cannot exceed (task #28's
   finding, on this project's own cpucheck.c) - ops.c staying at the new
   i486/i386 floor and this file's hand-written kernels needing a higher
   one are incompatible unless the kernels live in a separate TU with
   its own ceiling. The ARM cross-build must still never see this file:
   it has no MMX register file at all.

   Kernels do NOT emit emms themselves (rule 6 item 6) - the caller
   emits one after its loop. That contract does not change here; only
   which translation unit the code compiling to %mm instructions lives
   in changes. */
#include <stdint.h>
#include <string.h>

#include "ops_mmx.h"

#if defined(LZ_MMX_TU)

#if defined(__WATCOMC__)
#include "mmx_compat.h"

/* ---- GDN/KDA pass 1, 4-lane table contraction - moved here whole from
   ops.c's gdn_sum_gg (task #30). This function was ALREADY the real
   call boundary: a static function called once per 32-lane group from
   its own caller, doing 8 sub-window lz_gdn1_x4_asm calls internally.
   Moving the whole body here adds NO new call site - ops.c's call
   count to it is unchanged, only the destination TU is. gdn_tail_row
   is duplicated rather than shared across a TU boundary: 13 lines,
   pure integer, touches neither x87 nor MMX (ops.c's own comment on
   it), so sharing it via extern linkage would cost a real call for
   something this small and stateless, and duplication cannot drift
   since it has no state to drift out of. */
static void gdn_tail_row_mmx(const int8_t *sq_gg, int kd, int vd,
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

/* Eight 4-lane sub-windows cover this gg's 32 vv lanes.

   The emms belongs HERE, once per gg - not per sub-window, and not after
   the whole loop. Rule 6 item 6 permits hoisting the emms out only when
   the loop body has no x87, and that precondition fails one level up: the
   next gg's gdn_build_table runs q8_round and 32767/amax on x87 right
   after this returns. Hoisting it would reopen the trap that broke the
   mmx tier. */
extern void lz_gdn1_x4_asm(const int8_t *rowA, int32_t vd,
                           const int16_t *ctab, int32_t kpairs,
                           int32_t *out8);
#pragma aux lz_gdn1_x4_asm = \
    ".586" \
    "pxor      mm4, mm4" \
    "pxor      mm5, mm5" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "test      ecx, ecx" \
    "jz        LZ_GDN1_DONE" \
    "LZ_GDN1_LOOP:" \
    "movd      mm0, [eax]" \
    "punpcklbw mm0, mm0" \
    "psraw     mm0, 8" \
    "movd      mm1, [eax+edx]" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "movq      mm2, mm0" \
    "punpcklwd mm2, mm1" \
    "punpckhwd mm0, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [ebx]" \
    "paddd     mm4, mm3" \
    "pmaddwd   mm2, [ebx+8]" \
    "paddd     mm6, mm2" \
    "movq      mm3, mm0" \
    "pmaddwd   mm3, [ebx]" \
    "paddd     mm5, mm3" \
    "pmaddwd   mm0, [ebx+8]" \
    "paddd     mm7, mm0" \
    "lea       eax, [eax+edx*2]" \
    "add       ebx, 16" \
    "dec       ecx" \
    "jnz       LZ_GDN1_LOOP" \
    "LZ_GDN1_DONE:" \
    "movq      [esi], mm4" \
    "movq      [esi+8], mm5" \
    "movq      [esi+16], mm6" \
    "movq      [esi+24], mm7" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [__eax __ebx __ecx 8087]

/* gdn_sum_gg's MMX branch, unchanged from ops.c except for the name
   (ops.c's non-MMX branch keeps the bare name; this is the one that
   used to sit behind #ifdef LZ_GDN_HAVE_MMX in the same function). */
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
    gdn_tail_row_mmx(sq_gg, kd, vd, tcu, tcw, au_gg, aw_gg);
    _mm_empty();
}

/* ---- attention scoring's weighted-sum dot, GROUP granularity - moved
   here from src/ops.c (task #30, 3/N). LZ_ATTN_FIXED defaults to 3
   (ops.h) so this is a live, always-compiled path, not a reservation -
   AUTO just does not select it on an SSE-capable host, the same way
   the GDN pass-2 fixed tier is compiled everywhere but AUTO-selected
   only where float is slow.

   Duplicated rather than shared, same precedent as gdn_tail_row_mmx/
   p2_shift_of above: LZ_WSUM_CHUNK is one integer literal, LZ_SLOT_NEXT
   is a five-line ring-buffer walk with no state of its own, and
   lz_i32f is the one-line int32->float split ops.c's own comment
   requires EVERY conversion in the engine to go through (not a bare
   `(float)v`) - copying its body here is "keep in step by comment",
   the same word-for-word text as its ops.c twin, not a policy
   exception. */
#define LZ_WSUM_CHUNK 512

#define LZ_SLOT_NEXT(slot, sink, ring)                          \
    ((slot) = (ring) && (slot) + 1 >= (sink) + (ring)           \
              ? (sink) : (slot) + 1)

static float lz_i32f_wsum(int32_t v) {
    return (float)(v >> 12) * 4096.0f + (float)(v & 0xFFF);
}

/* Originally generated, hand-edited since - if you regenerate, diff
   the generator's output against what is here first.
   acc32[d] += rowA[d]*coef[0] + rowB[d]*coef[1] for d in [0,32).
   coef is [ckA,ckB,ckA,ckB] so one pmaddwd covers a 2-int32 lane pair.
   Reads [eax]/[edx]/[ecx], writes only [ebx] and MMX - every offset is a
   literal, no GP register is modified, so `__modify [8087]` is the truth
   and not an `__exact []`-style lie (rule 2 item 3).
   Emits no emms; the driver does one per chunk. */
extern void lz_wsum_pair_asm(const int8_t *rowA, const int8_t *rowB,
                const int16_t *coef, int32_t *acc32);
#pragma aux lz_wsum_pair_asm = \
    ".586" \
    "movq      mm7, [ecx]" \
    "movq      mm0, [eax]" \
    "movq      mm1, [edx]" \
    "movq      mm2, mm0" \
    "punpcklbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpcklbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx]" \
    "paddd     mm5, mm4" \
    "movq      [ebx], mm5" \
    "movq      mm5, [ebx+8]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+8], mm5" \
    "movq      mm2, mm0" \
    "punpckhbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpckhbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+16]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+16], mm5" \
    "movq      mm5, [ebx+24]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+24], mm5" \
    "movq      mm0, [eax+8]" \
    "movq      mm1, [edx+8]" \
    "movq      mm2, mm0" \
    "punpcklbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpcklbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+32]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+32], mm5" \
    "movq      mm5, [ebx+40]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+40], mm5" \
    "movq      mm2, mm0" \
    "punpckhbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpckhbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+48]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+48], mm5" \
    "movq      mm5, [ebx+56]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+56], mm5" \
    "movq      mm0, [eax+16]" \
    "movq      mm1, [edx+16]" \
    "movq      mm2, mm0" \
    "punpcklbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpcklbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+64]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+64], mm5" \
    "movq      mm5, [ebx+72]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+72], mm5" \
    "movq      mm2, mm0" \
    "punpckhbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpckhbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+80]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+80], mm5" \
    "movq      mm5, [ebx+88]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+88], mm5" \
    "movq      mm0, [eax+24]" \
    "movq      mm1, [edx+24]" \
    "movq      mm2, mm0" \
    "punpcklbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpcklbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+96]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+96], mm5" \
    "movq      mm5, [ebx+104]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+104], mm5" \
    "movq      mm2, mm0" \
    "punpckhbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpckhbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+112]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+112], mm5" \
    "movq      mm5, [ebx+120]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+120], mm5" \
    __parm [__eax] [__edx] [__ecx] [__ebx] \
    __modify [8087]

/* lz_wsum_group_mmx(): the ENTIRE multi-chunk second walk for one
   group - all of LZ_WSUM_CHUNK's chunks, all pairs, the ring-buffer
   slot walk, the per-chunk emms and lz_i32f fold - byte-for-byte the
   loop body that used to live in src/ops.c's lz_attn_wsum_q8, now
   behind ONE call per group instead of the zero real calls the
   pre-split #pragma-aux-inline loop had. T is context-length-dependent
   and unbounded (up to 262144 on this model), so amortizing at group
   granularity rather than per-pair or per-chunk is what keeps the
   added call count O(ng) instead of O(T). */
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
            lz_wsum_pair_asm(rowA, rowB, coef, acc32);
        }
        if (tp < t0 + tn) {            /* odd leftover row in this chunk */
            int16_t coef[4];
            const int8_t *rowA = vc + (size_t)slot * kvd + (size_t)g * 32;
            coef[0] = cq[tp]; coef[1] = 0;
            coef[2] = cq[tp]; coef[3] = 0;
            lz_wsum_pair_asm(rowA, rowA, coef, acc32);
            LZ_SLOT_NEXT(slot, sink, ring);
        }
        _mm_empty();      /* no x87 inside the row-pair loop above */
        for (d = 0; d < 32; d++) accf[d] += lz_i32f_wsum(acc32[d]);
    }
}

/* ---- GDN/KDA fixed-point pass 2, MMX tier - moved here from
   src/ops_kernel_p2.h (task #30, 2/N). Algorithm documentation (WHAT
   THE OPERATOR IS, ISA INVENTORY, why the lo clamp is a saturating
   subtract) stays there, unaffected by the code's move. */

#include "ops_p2_blk.h"

/* 16-byte-aligned scratch, by hand: neither compiler can be asked for
   it here (Watcom is built with -zp4, capping member alignment at 4;
   C89/C99 have no alignment specifier), and SSE2's movdqa faults on a
   misaligned address. ONE buffer for the whole engine, shared with
   src/ops_mmx_sse.c's lz_p2_rows_sse across the TU boundary (task #30,
   2/N: this used to be ops.c's own static function) - the split step
   there reads what the mul32 step here wrote, so both need the SAME
   pointer, not merely the same layout. Nothing re-enters between the
   two calls a row makes. */
lz_p2_blk *p2_blk(void) {
    static char raw[sizeof(lz_p2_blk) + 15];
    size_t off = (size_t)((uintptr_t)(void *)raw & 15u);
    return (lz_p2_blk *)(void *)(raw + ((16u - off) & 15u));
}

/* Duplicated from ops.c's own p2_shift_of, same precedent as
   gdn_tail_row_mmx above: five lines, pure integer, cheaper to keep in
   step by comment than to cost a real call for. */
static int p2_shift_of(int32_t amax) {
    int sh = 0;
    while ((amax >> sh) > 32512) sh++;
    return sh;
}

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
    "pxor      mm7, mm7" \
    "pxor      mm4, mm4" \
    "pxor      mm3, mm3" \
    "movq      mm5, [ecx]" \
    \
    "movd      mm0, [eax]" \
    "movd      mm1, [edx]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+96], mm0" \
    "movq      [ecx+104], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+4]" \
    "movd      mm1, [edx+4]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+8]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+112], mm0" \
    "movq      [ecx+120], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+8]" \
    "movd      mm1, [edx+8]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+16]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+128], mm0" \
    "movq      [ecx+136], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+12]" \
    "movd      mm1, [edx+12]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+24]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+144], mm0" \
    "movq      [ecx+152], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+16]" \
    "movd      mm1, [edx+16]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+32]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+160], mm0" \
    "movq      [ecx+168], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+20]" \
    "movd      mm1, [edx+20]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+40]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+176], mm0" \
    "movq      [ecx+184], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+24]" \
    "movd      mm1, [edx+24]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+48]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+192], mm0" \
    "movq      [ecx+200], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+28]" \
    "movd      mm1, [edx+28]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+56]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+208], mm0" \
    "movq      [ecx+216], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movq      [ecx+16], mm4" \
    "movq      [ecx+24], mm3" \
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

/* lz_p2_rows_mmx(): all of a row's groups, tier 1. One call per row
   from ops.c's gdn_p2_row_simd (task #30, 2/N), replacing the zero real
   calls the pre-split #pragma-aux-inline loop had - see that function's
   Watcom branch for the full accounting. Both pragma-aux kernels above
   are textually visible in THIS translation unit, so this loop is
   exactly as inline as the one it replaces; the row boundary is the
   only new call. */
void lz_p2_rows_mmx(const int8_t *ph_row, const int8_t *pl_row,
                    int pl_stride, const int16_t *dq,
                    const int16_t (*mul)[4], int8_t *oh_row,
                    int8_t *ol_row, int ol_stride, int *shv, int ng) {
    lz_p2_blk *blk = p2_blk();
    int gg, j;
    for (gg = 0; gg < ng; gg++) {
        int32_t amax;
        int sh;
        const int8_t *plg = pl_row + (size_t)gg * pl_stride;
        int8_t *olg = ol_row + (size_t)gg * ol_stride;
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

/* ---- Q8 group-scale amax, MMX tier - moved here from
   src/ops_kernel_amax.h (task #30, 4/N: the mechanical low-risk sweep).
   Zero call-shape change: lz_amax32_mmx_w already sat between the
   pragma and LZ_AMAX_TAB (src/ops_kernel_amax.h) before the move - a
   #pragma aux body cannot have its address taken, so the table always
   needed a real function to point at - and that indirect call happens
   once per operator invocation regardless of which TU the wrapper's
   body lives in. */

/* Two lanes of uint32 in one MMX register. The mask is built in-register
   (pcmpeqd then psrld 1) rather than loaded, so the routine touches no
   memory but its input. Leaves MMX state dirty - the caller emits one
   emms after the group loop, never the kernel (rule 6 item 6). */
extern unsigned lz_amax32_mmx_asm(const float *x, int n);
#pragma aux lz_amax32_mmx_asm =         \
    "pxor     mm0, mm0"                 \
    "pcmpeqd  mm3, mm3"                 \
    "psrld    mm3, 1"                   \
    "shr      edx, 1"                   \
    "jz       L2"                       \
    "L1:"                               \
    "movq     mm1, [eax]"               \
    "pand     mm1, mm3"                 \
    "movq     mm2, mm1"                 \
    "pcmpgtd  mm2, mm0"                 \
    "pand     mm1, mm2"                 \
    "pandn    mm2, mm0"                 \
    "por      mm1, mm2"                 \
    "movq     mm0, mm1"                 \
    "add      eax, 8"                   \
    "dec      edx"                      \
    "jnz      L1"                       \
    "L2:"                               \
    "movq     mm1, mm0"                 \
    "psrlq    mm1, 32"                  \
    "movq     mm2, mm0"                 \
    "pcmpgtd  mm2, mm1"                 \
    "pand     mm0, mm2"                 \
    "pandn    mm2, mm1"                 \
    "por      mm0, mm2"                 \
    "movd     eax, mm0"                 \
    parm [eax] [edx]                    \
    value [eax]                         \
    modify [eax edx 8087];

/* Thin wrapper so LZ_AMAX_TAB can hold this. The wrapper is the only
   caller, so the asm is still inlined once and the indirect call is
   one per operator invocation, not one per element. */
unsigned lz_amax32_mmx_w(const float *x, int n) {
    return lz_amax32_mmx_asm(x, n);
}

#else /* gcc: intrinsics */

#include "mmx_compat.h"
#include <mmintrin.h>

/* ---- wsum: attention scoring's weighted-sum dot, LZ_WSUM_PAIR's gcc
   twin. Watcom twin: lz_wsum_pair_asm, src/ops.c ~3960-4041.
   Sign-extend two int8 rows to int16, interleave so one pmaddwd covers a
   (rowA,rowB) pair against (ckA,ckB), accumulate into acc32 in place.
   Emits no emms; the driver does one per chunk, and that chunk's
   row-pair loop contains no x87 (rule 6 item 6). */
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
   instruction of content here) lives with that file - it is unaffected
   by this split and stays the authority on the "what" and "why". */

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

/* lz_p2_split32_sse moved to src/ops_mmx_sse.c: it needs genuine SSE1,
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

/* lz_q8round32_sse moved to src/ops_mmx_sse.c, same reason as
   lz_p2_split32_sse above. */

#include "ops_kernel_dot_mmx.h"

#endif /* __WATCOMC__ */

#endif /* LZ_MMX_TU */
