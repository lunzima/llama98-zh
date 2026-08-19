/* Helpers used from BOTH src/ops.c and one or more of the suffixed
   MMX/SSE1/SSE2 translation units (src/ops_mmx.c, src/ops_mmx_sse.c,
   src/ops_sse2.c). Task #30's kernel extraction kept re-duplicating
   these under per-TU names (lz_i32f_wsum, gdn_tail_row_mmx) or as bare
   textually-identical copies (LZ_WSUM_CHUNK, LZ_SLOT_NEXT,
   p2_shift_of) - flagged as a real defect (task #31), not tidiness:
   lz_i32f carries a project-wide invariant ("every int32->float
   conversion goes through this, never a bare cast") that a rename
   makes ungreppable, and LZ_WSUM_CHUNK/LZ_SLOT_NEXT drive a ring-buffer
   walk that must stay in lock-step across whichever TU boundary a
   caller and its row/group function sit on - this codebase has that
   exact failure mode on record already (LZ_GDN_LO_SCALE, 254 vs 256,
   silently wrong because nothing compared the two copies).

   ONE definition each, here, included by every TU that needs it -
   `static` for the functions (each TU gets its own internal-linkage
   copy of the CODE, but there is only one COPY OF THE TEXT to keep in
   step, which is the actual property task #31 asked for) and #define
   for the macros. Every name keeps the name it had at its original
   (ops.c) site - no _mmx/_wsum/_sse2 suffix - so a grep for the name
   that matters finds every use.

   Needs its own include guard, unlike ops_kernel_amax.h/norm.h/
   q8round.h/p2.h/dot.h: those are included exactly once (by ops.c) and
   rely on that; this one is included by up to four different files. */
#ifndef LZ_OPS_KERNEL_SHARED_H
#define LZ_OPS_KERNEL_SHARED_H

#include <stdint.h>

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
   holds (`>>` is arithmetic shift). */
static float lz_i32f(int32_t v) {
    return (float)(v >> 12) * 4096.0f + (float)(v & 0xFFF);
}

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
static void gdn_tail_row(const int8_t *sq_gg, int kd, int vd,
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
static int p2_shift_of(int32_t amax) {
    int sh = 0;
    while ((amax >> sh) > 32512) sh++;
    return sh;
}

#endif /* LZ_OPS_KERNEL_SHARED_H */
