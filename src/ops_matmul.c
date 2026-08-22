/* ops_matmul.c - Matmul kernels: scalar reference, per-format row
   kernels, dispatch tables, and the tiered matmul impls.

   A separate TU from ops.c to keep #if guard density low. Linkage is
   extern only where a function is called from ops.c (matmul_q*_impl,
   LZ_ROW_* tables, matmul_scalar_ref).
   lz_row_pick reads the tier through lz_kernel_sel() instead of the
   raw g_kernel, breaking the sched<->matmul cycle. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "ops.h"
#include "err.h"
#include "mmx_compat.h"
#include "ops_mmx.h"
#include "ops_sse2.h"
#include "ops_kernel_shared.h"
#include "ops_quant.h"
#include "ops_sched.h"
#include "ops_matmul.h"
#include "ops_t2_arm.h"
#include "ops_arm.h"

#ifndef LZ_GDN_FIXED
#define LZ_GDN_FIXED 1
#endif /* LZ_GDN_FIXED */

#ifndef LZ_EPI_MAX_NG
#define LZ_EPI_MAX_NG 64
#endif /* LZ_EPI_MAX_NG */

/* ---- LZ_MM_WIDEN_MAX + scratch ---- */
#define LZ_MM_WIDEN_MAX 4096

/* Scratch shared by the three SIMD matmuls (batched version widens by
   LZ_BATCH_MAX). The three are never alive simultaneously
   (single-threaded, no nested calls), so one block serves all of them.
   matmul_scalar_ref's accref/xgref are deliberately NOT merged: on
   non-SIMD builds q41/q61 call scalar_ref after filling xw, and sharing
   would alias.

   token-major: g_xw[t*in_dim + k], g_acc32[t*nsb + s], g_xg[t*ng + g]. */

/* The hand-written ARM leaf kernels read g_xw with LDM, which faults on
   ARM below a word boundary. int16_t promises 2 bytes, so the alignment
   gcc happens to give a 64KB array is not the contract - this is. The
   offsets stay aligned on their own: every block pointer is g_xw plus a
   multiple of 64 bytes, in_dim being a multiple of 32 by the dispatch
   guards. Empty off ARM; Watcom never sees __arm__ and has no
   __attribute__. */
#if defined(LZ_T2_ARM_ASM_EXTERN)
#define LZ_XW_ALIGN __attribute__((aligned(4)))
#else
#define LZ_XW_ALIGN
#endif /* LZ_T2_ARM_ASM_EXTERN */
static int16_t g_xw[LZ_BATCH_MAX * LZ_MM_WIDEN_MAX] LZ_XW_ALIGN;
static float   g_xg[LZ_BATCH_MAX * (LZ_MM_WIDEN_MAX / 32)];
/* Unguarded: the SSE2-intrinsics path writes here too now that it fills
   acc32 instead of fusing the float work into the integer loop. Costs
   4KB of BSS in every build. */
static int32_t g_acc32[LZ_BATCH_MAX * (LZ_MM_WIDEN_MAX / 32)];

/* ---- lz_rowfn typedef + fwd decl ---- */
typedef void (*lz_rowfn)(int32_t *acc32, const void *w4, const void *w2,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end,
                         const void *pf_end2);

/* Defined below; the tiered matmuls fall back to it when they have no
   row kernel for the selected tier, so it must be visible up here. */
void matmul_scalar_ref(float *o, const int8_t *xq, const float *xqs,
                              const LZTensor *w, int in_dim, int out_dim,
                              int nt);

/* ---- lz_row_pick ---- */
static lz_rowfn lz_row_pick(const lz_rowfn *tab) {
    int want_sse2 = (lz_kernel_sel() == LZ_KERNEL_SSE2);
#ifdef __WATCOMC__
    lz_rowfn f = tab[want_sse2 ? LZ_ROW_SSE2_A : LZ_ROW_MMX_A];
    return f ? f : tab[LZ_ROW_MMX_A];
#else
    lz_rowfn f = tab[want_sse2 ? LZ_ROW_SSE2_I : LZ_ROW_MMX_I];
    if (f) return f;
    return tab[LZ_ROW_SSE2_I] ? tab[LZ_ROW_SSE2_I] : tab[LZ_ROW_MMX_I];
#endif /* __WATCOMC__ */
}

/* ---- q41-family epilogue exit ----------------------------------------
   The per-row epilogue loop of matmul_q41_impl / matmul_t2_impl /
   matmul_q61_impl, in one place because the choice it makes is the same
   in all three: epi_q41 (no fixed epilogue), epi_q41_fixed (integer
   join, float exit) or epi_q41_align_i16 (same join, int16 exit, no
   float materialized at all - int-pipeline milestone 3). The row kernel
   above it and the int32 accumulators below it do not know which exit
   ran; that is the whole point of putting the exit here.

   `wd`/`wm` are read by the epi_q41 arm only. The int16 result is
   already clamped to +-`out->bound` inside epi_align_i16, so the
   narrowing cast cannot lose a bit. */
static void epi_q41_row(const LZMatOut *out, int i, int out_dim, int nt,
                        const LZTensor *w, int ng, int r, int nb, int zneg,
                        int fixed_epi, const float *xqs,
                        const float *wd, const float *wm) {
    int tk;
    if (out->oi) {
        for (tk = 0; tk < nt; tk++)
            out->oi[(size_t)tk * out_dim + i] = (short)epi_q41_align_i16(
                g_acc32 + (size_t)tk * nb, w, i, tk, ng, r, zneg,
                out->target_e, out->bound);
        return;
    }
    for (tk = 0; tk < nt; tk++)
        out->o[(size_t)tk * out_dim + i] = fixed_epi
            ? epi_q41_fixed(g_acc32 + (size_t)tk * nb, w, i, tk, ng, r, zneg)
            : epi_q41(g_acc32 + (size_t)tk * nb, xqs + (size_t)tk * nb,
                      g_xg + (size_t)tk * ng, wd, wm, ng, r);
}

/* ---- Q8 guard block ---- */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)
#include "ops_kernel_dot_shared.h" /* LZ_Q8_ACC, LZ_Q8_PFSEL, LZ_Q8_GROUP4 -
                           needed here (gcc's row_q8_mmx_intrin below)
                           AND by src/ops_mmx.c's/ops_sse2.c's
                           row_q8_mmx_asm/row_q8_sse2_asm - shared
                           rather than duplicated across that TU
                           boundary. */

#ifdef __WATCOMC__
/* row_q8_mmx_asm / row_q8_sse2_asm live in src/ops_mmx.c / src/ops_sse2.c.
   Each is ALREADY the real call boundary from its own caller (dispatched
   once per matmul via lz_row_pick's function-pointer table in LZ_ROW_Q8
   below - never once per 32-element group), and calls its leaf kernels
   direct-by-name inside its own loop (lz_dot32_x16_asm, the pragma
   inlines into THIS function; lz_dot32_x16_asm_2, lz_dot128_x16_asm,
   dot32_x16_mmx/_sse2a) - so the split adds no new call sites. Splitting
   the leaf kernels out alone and leaving the row function here would
   have cost up to nb (32-groups per row, up to 112 on this project's
   largest in_dim) real cross-TU calls per matmul row instead. Declared
   in src/ops_mmx.h/ src/ops_sse2.h, already visible in this translation
   unit. */

#else  /* gcc: intrinsics */

static void row_q8_mmx_intrin(int32_t *acc32, const void *w4, const void *w2,
                              const int16_t *xw, int nb, int nt, int in_dim,
                              const void *pf_end,
                         const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    const int8_t *wend = (const int8_t *)pf_end;
    int tk, g;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        /* Pair this token with the next so ONE weight unpack serves
           both. The prefetch is issued once for the pair, not once per
           token: it is a hint about the WEIGHT row, and the second
           token reads the bytes the first just pulled in. An odd nt
           leaves the last token on the single-token kernel. */
        if (tk + 1 < nt && g_pair) {
            const int16_t *xw2 = xwt + in_dim;
            int32_t *accb = acc + nb;
            for (g = 0; g < nb; g++) {
                const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                if (pf_ < wend) LZ_PFI_NTA(pf_);
                dot32_x16_mmx_2(wr + (size_t)g * 32,
                                xwt + (size_t)g * 32,
                                xw2 + (size_t)g * 32,
                                acc + g, accb + g);
            }
            tk++;
        } else {
#if defined(LZ_G128_GCC) && LZ_G128_GCC
            /* Group-of-four path behind the LZ_G128_GCC knob (default on,
               ops_kernel_shared.h): one call per 128 elements with the
               mask construction hoisted, mirroring the Watcom row. The
               per-32 fallback stays LZ_Q8_ACC/NTA exactly as before, so
               toggling the knob changes only the group path. */
            if ((nb & 3) == 0) LZ_Q8_GROUP4(dot128_x16_mmx);
            else               LZ_Q8_ACC(dot32_x16_mmx, LZ_PFI_NTA);
#else
            LZ_Q8_ACC(dot32_x16_mmx, LZ_PFI_NTA);
#endif /* LZ_G128_GCC */
        }
    }
}

#endif /* __WATCOMC__ */
#endif /* __MMX__ */

/* ---- LZ_ROW_Q8 + matmul_q8_impl ---- */
/* row_q8_sse2_intrin's body lives in src/ops_sse2.c: it writes %xmm
   registers, so it belongs in the translation unit built with -msse2.
   Declared in src/ops_sse2.h, already visible in this translation unit. */

#if defined(__arm__)
/* The two ARM tiers, picked in matmul_q8_impl rather than from the
   table below - those six slots are x86 slots. Leaves in
   src/ops_arm.c; the w4 pointer is the int8 weight row, the same slot
   row_q8_mmx_intrin reads. No pairing: g_pair's two-token kernels are
   an x86 register-budget trick, and there is no ARM twin to select. */
static void row_q8_arm(int32_t *acc32, const void *w4, const void *w2,
                       const int16_t *xw, int nb, int nt, int in_dim,
                       const void *pf_end, const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    int tk, s;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_q8_arm(wr + (size_t)s * 32,
                                     xwt + (size_t)s * 32);
    }
}

#if defined(LZ_ARM_ASM_EXTERN)
static void row_q8_arm_asm(int32_t *acc32, const void *w4, const void *w2,
                           const int16_t *xw, int nb, int nt, int in_dim,
                           const void *pf_end, const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    int tk, s;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_q8_arm_asm(wr + (size_t)s * 32,
                                         xwt + (size_t)s * 32);
    }
}
#endif /* LZ_ARM_ASM_EXTERN */
#endif /* __arm__ */

/* The table. A NULL slot is a tier this build does not carry; the two
   SSE slots stay NULL because no SSE row kernel exists for this format. */
const lz_rowfn LZ_ROW_Q8[LZ_ROW_N] = {
#if defined(LZ_DOT_MMX_EXTERN)
    row_q8_mmx_intrin,
#else
    NULL,
#endif /* LZ_DOT_MMX_EXTERN */
    NULL,        /* sse-intrin: no SSE1 op applies, see lz_row_pick */
#if defined(LZ_ROW_SSE2_EXTERN)
    row_q8_sse2_intrin,
#else
    NULL,
#endif /* LZ_ROW_SSE2_EXTERN */
#if defined(__WATCOMC__)
    row_q8_mmx_asm,
    NULL,        /* sse-asm: ditto */
    row_q8_sse2_asm
#else
    NULL, NULL, NULL
#endif /* __WATCOMC__ */
};

/* int8 matmul kernel for gs==32, dispatched by target platform
   (LZ_USE_MMX forces MMX for local verification; SSE2 is the x86-64
   default; ref is the fallback). */

/* Fixed-epi predicate shared by the float and int-output Q8 impls, so
   the two cannot disagree on when the int output applies. The tier is
   decided once per call, ANDed with whether this tensor actually has
   its int16 plane - lz_epi_prep declines shapes outside the derived
   bound, so a tensor it refused keeps the float epilogue without the
   selector knowing about in_dim. The plane is built here rather than
   in the loader, which does not know each tensor's in_dim - every call
   site does. Idempotent and one-time, so the cost lands on the first
   token and nowhere after. The cast is the price of that: the plane is
   derived state, not the weights, and making every matmul signature
   non-const to say so would be a larger lie than this one. */
static int q8_fixed_epi(const LZTensor *w, int in_dim, int nb, int nt) {
    if (lz_epi_mode() && !lz_epi_ready(w)) lz_epi_prep((LZTensor *)w, in_dim);
    return lz_epi_mode() && lz_epi_ready(w) && nb <= LZ_EPI_MAX_NG
           && nt <= LZ_BATCH_MAX;
}

void matmul_q8_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                           const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, tk, t;
    int r  = w->gs / 32;    /* 32-sub-blocks per weight group (gs=128 -> 4) */
    int nb = in_dim / 32;   /* 32-sub-block count */
    const int8_t *wend = w->q + (size_t)w->n;   /* prefetch upper bound */
    lz_rowfn row;
    /* Same predicate as the float epilogue - see q8_fixed_epi. */
    int fixed_epi = q8_fixed_epi(w, in_dim, nb, nt);
    int want_i16 = (out->oi != NULL) || out->probe;
    /* 2^-8, the kernels' x256 undone - taken from the same place the
       fixed epilogue takes its exponent fold, not spelled out again. */
    int post_e = lz_epi_post_e(w);
    float post = pow2f(post_e);

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_Q8);
#if defined(__arm__)
#if defined(LZ_ARM_ASM_EXTERN)
    row = (g_kernel == LZ_KERNEL_ARM_ASM) ? row_q8_arm_asm : row_q8_arm;
#else
    row = row_q8_arm;
#endif /* LZ_ARM_ASM_EXTERN */
#endif /* __arm__ */

    /* Three ways this function cannot serve the call, and all of them
       must land on the scalar reference rather than on a second inline
       implementation:
       - this build carries no row kernel for the selected ISA (all-NULL
         table);
       - in_dim exceeds the pre-expansion buffer;
       - the format has no entry in lz_epi_post_e, so neither epilogue
         below has a scaling anyone derived for it.
       Inline fallbacks (an MMX dot32_nowiden loop, an SSE2 part32_sse2
       loop, the ref branch) would be three DIFFERENT code paths, none of
       them bit-identical to the others, all of them unreachable - the
       largest in_dim in any current model is 3584 against a 4096 cap.
       Dead code in three association orders is worse than dead code in
       one, and matmul_scalar_ref is the contract. */
    /* Same all-or-nothing refusal as matmul_q16_impl, and for the same
       reason: the int16 exit IS the fixed epilogue's exit
       (epi_fixed_align_i16 is epi_fixed_raw plus epi_align_i16), so
       !fixed_epi refuses here even though the float exit below would
       have been happy to run without it. Refusing leaves out->ok at 0
       and writes nothing; the caller falls back. */
    if (want_i16 && (!fixed_epi || !row || in_dim > LZ_MM_WIDEN_MAX
                     || post_e == LZ_EPI_POST_NA)) return;
    if (!row || in_dim > LZ_MM_WIDEN_MAX || post_e == LZ_EPI_POST_NA) {
        matmul_scalar_ref(out->o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    out->ok = 1;
    if (out->probe) return;

    /* Expand the whole batch's activations to int16 once (in_dim <=
       4096, resident in L1); amortized over out_dim rows. */
    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    if (fixed_epi) epi_pack_act(xqs, nb, nt);

    for (i = 0; i < out_dim; i++) {
        const int8_t *wr = w->q + (size_t)i * in_dim;
        const float *ws = w->scale + (size_t)(i * in_dim) / w->gs;
        /* The weight row loads once here and serves all nt tokens; one
           EMMS per row, not per token - MMX and x87 share a register
           file and the epilogue below is x87 on the Watcom build. */
        row(g_acc32, wr, NULL, g_xw, nb, nt, in_dim, wend, NULL);
        if (lz_cpu_has_mmx()) _mm_empty();
        /* The clamp inside epi_align_i16 has already brought the result
           within +-out->bound, so the narrowing cast cannot lose a bit. */
        if (out->oi) {
            for (tk = 0; tk < nt; tk++)
                out->oi[(size_t)tk * out_dim + i] = (short)epi_fixed_align_i16(
                    g_acc32 + (size_t)tk * nb, w, i, tk, nb, r,
                    out->target_e, out->bound);
            continue;
        }
        for (tk = 0; tk < nt; tk++)
            out->o[(size_t)tk * out_dim + i] = fixed_epi
                ? epi_fixed(g_acc32 + (size_t)tk * nb, w, i, tk, nb, r)
                : epi_q8(g_acc32 + (size_t)tk * nb, xqs + (size_t)tk * nb,
                         ws, nb, r, post);
    }
}

/* ---- matmul_scalar_ref_one + matmul_scalar_ref ---- */
void matmul_scalar_ref_one(float *o, const int8_t *xq, const float *xqs,
                                  const LZTensor *w, int in_dim, int out_dim) {
    int i, g, s, k;
    int gs = w->gs, r, ng, nsb;
    int t2 = (w->dtype == LZ_FMT_T2);
    /* T2 rides the q4 flag for the ACTIVATION-SUM hoist, which is the
       same computation for both (xgref[g] = Σ xqs·Σ xq). What differs
       is the coefficient it is multiplied by at the end: Q4_1/Q6_1 use
       the per-group min, T2 uses -scale. See model.h's T2 note. */
    int q4 = (w->dtype == LZ_FMT_Q4_1 || w->dtype == LZ_FMT_Q6_1 || t2);
    int q6 = (w->dtype == LZ_FMT_Q6_1);
    int q16 = (w->dtype == LZ_FMT_Q16_0);
    /* static not stack: Win98's stack is tight; see the xw/acc32 note in this file. */
    static int32_t accref[LZ_MM_WIDEN_MAX / 32];
    static float   xgref[LZ_MM_WIDEN_MAX / 32];

    if (gs < 32 || (gs % 32) != 0 || (in_dim % gs) != 0) {
        /* degenerate tier (gs < 32, activations share weight groups): Q8_0 only */
        for (i = 0; i < out_dim; i++) {
            const int8_t *wr = w->q + (size_t)i * in_dim;
            const float *ws = w->scale + (size_t)(i * in_dim) / gs;
            float sum = 0.0f;
            for (g = 0; g < in_dim / gs; g++) {
                int32_t acc = 0;
                const int8_t *wq = wr + (size_t)g * gs;
                const int8_t *xr = xq + (size_t)g * gs;
                for (k = 0; k < gs; k++)
                    acc += (int32_t)wq[k] * (int32_t)xr[k];
                sum += lz_i32f(acc) * xqs[g] * ws[g];
            }
            o[i] = sum;
        }
        return;
    }
    r  = gs / 32;                   /* 32 sub-blocks per weight group */
    ng = in_dim / gs;
    nsb = in_dim / 32;              /* total number of 32-element sub-blocks */
    if (nsb > LZ_MM_WIDEN_MAX / 32) return;      /* defense: no silent miscalc */

    /* Q4_1's zero term xg[g] = Σ_{sub-block s∈g} xqs[s]·Σ_k xq[k] depends
       only on activations and is shared by all output rows - it MUST be
       summed in the same order as matmul_q41_impl, or the
       bit-identical contract breaks on this branch. */
    if (q4) {
        for (g = 0; g < ng; g++) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int sb = g * r + s;
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xq[sb * 32 + k];
                a += lz_i32f(sx) * xqs[sb];
            }
            xgref[g] = a;
        }
    }

    for (i = 0; i < out_dim; i++) {
        const unsigned char *wn = (const unsigned char *)w->q +
                                  (size_t)i * in_dim / 2;
        const unsigned char *w2 = (const unsigned char *)w->q +
                                  (size_t)w->n / 2 + (size_t)i * in_dim / 4;
        const int8_t *wr = w->q + (size_t)i * in_dim;
        const int16_t *w16 = (const int16_t *)(const void *)w->q +
                             (size_t)i * in_dim;
        const float *ws = w->scale + (size_t)i * ng;
        const float *wz = (q4 && !t2) ? w->zero + (size_t)i * ng : NULL;

        /* Step 1: the row's int32 sub-block sums. On the SIMD side
           this segment is the MMX/SSE2 kernel. */
        for (s = 0; s < nsb; s++) {
            int base = s * 32;
            int32_t acc = 0;
            if (q6) {
                /* dot(q,x) = dot(lo,x) + 16·dot(hi,x), exact in
                   integers. The SIMD side splits exactly this way: the
                   lo plane reuses the Q4_1 kernel, the hi plane goes
                   through the 2-bit sub-kernel, ending as
                   acc_lo + (acc_hi<<4). */
                const unsigned char *b4 = wn + (size_t)base / 2;
                const unsigned char *b2 = w2 + (size_t)base / 4;
                int32_t alo = 0, ahi = 0;
                for (k = 0; k < 16; k++) {
                    alo += (int32_t)(b4[k] & 15) * (int32_t)xq[base + k];
                    alo += (int32_t)(b4[k] >> 4) * (int32_t)xq[base + k + 16];
                }
                for (k = 0; k < 32; k++)
                    ahi += (int32_t)((b2[k & 7] >> (2 * (k >> 3))) & 3) *
                           (int32_t)xq[base + k];
                acc = alo + (ahi << 4);
            } else if (t2) {
                /* Same addressing as Q6_1's 2-bit plane, but it is the
                   ONLY plane: 8 bytes per 32 elements. Codes are 0..2
                   read unsigned; the -1 lives in the hoisted term. */
                const unsigned char *b2 = (const unsigned char *)w->q +
                                          (size_t)i * in_dim / 4 +
                                          (size_t)base / 4;
                for (k = 0; k < 32; k++)
                    acc += (int32_t)((b2[k & 7] >> (2 * (k >> 3))) & 3) *
                           (int32_t)xq[base + k];
            } else if (q4) {
                const unsigned char *p = wn + (size_t)base / 2;
                for (k = 0; k < 16; k++) {
                    acc += (int32_t)(p[k] & 15) * (int32_t)xq[base + k];
                    acc += (int32_t)(p[k] >> 4) * (int32_t)xq[base + k + 16];
                }
            } else if (q16) {
                for (k = 0; k < 32; k++)
                    acc += (int32_t)w16[base + k] * (int32_t)xq[base + k];
            } else {
                for (k = 0; k < 32; k++)
                    acc += (int32_t)wr[base + k] * (int32_t)xq[base + k];
            }
            accref[s] = acc;
        }

        /* Step 2: float reduction, byte-for-byte matching the SIMD side's epilogue. */
        if (q4) {
            float dotsum = 0.0f, zsum = 0.0f;
            for (g = 0; g < ng; g++) {
                float dot = 0.0f;
                for (s = 0; s < r; s++)
                    dot += lz_i32f(accref[g * r + s]) * xqs[g * r + s];
                dotsum += dot * ws[g];
                /* T2's zero coefficient is -scale, not a stored min:
                   Σ w·x = d·[Σ code·x - Σ x]. Written as a negated
                   multiply rather than folded into the dot term so the
                   two accumulators stay separate, matching the SIMD
                   epilogue's merge order. */
                zsum += xgref[g] * (t2 ? -ws[g] : wz[g]);
            }
            o[i] = dotsum + zsum;
        } else if (r > 1) {
            float sum = 0.0f;
            for (g = 0; g < ng; g++) {
                float dot = 0.0f;
                for (s = 0; s < r; s++)
                    dot += lz_i32f(accref[g * r + s]) * xqs[g * r + s];
                sum += dot * ws[g];
            }
            o[i] = sum;
        } else {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (g = 0; g + 3 < ng; g += 4) {
                a0 += lz_i32f(accref[g])     * (xqs[g]     * ws[g]);
                a1 += lz_i32f(accref[g + 1]) * (xqs[g + 1] * ws[g + 1]);
                a2 += lz_i32f(accref[g + 2]) * (xqs[g + 2] * ws[g + 2]);
                a3 += lz_i32f(accref[g + 3]) * (xqs[g + 3] * ws[g + 3]);
            }
            for (; g < ng; g++)                  /* tail all into a0, same as SIMD */
                a0 += lz_i32f(accref[g]) * (xqs[g] * ws[g]);
            o[i] = (a0 + a2) + (a1 + a3);
        }
    }
}

/* Scalar-reference batched wrapper. Here we DELIBERATELY do no weight
   reuse - the reference path's job is to be the oracle, not fast.
   Reruns per token as-is; batched and unbatched results must agree
   down to the rounding. */
void matmul_scalar_ref(float *o, const int8_t *xq, const float *xqs,
                              const LZTensor *w, int in_dim, int out_dim, int nt) {
    int gsa = lz_act_gs(w, in_dim);
    int ss = (gsa > 0) ? in_dim / gsa : in_dim / 32;
    int t;
    for (t = 0; t < nt; t++)
        matmul_scalar_ref_one(o + (size_t)t * out_dim, xq + (size_t)t * in_dim,
                              xqs + (size_t)t * ss, w, in_dim, out_dim);
}

/* ---- Q41 guard block ---- */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)
#include "ops_kernel_dot_shared.h" /* LZ_Q41_ACC, LZ_Q41_PFSEL, LZ_Q41_GROUP4 */

#ifdef __WATCOMC__
/* row_q41_mmx_asm / row_q41_sse2_asm live in src/ops_mmx.c /
   src/ops_sse2.c - same reasoning as Q8_0 above: each is already the
   real call boundary from LZ_ROW_Q41 below, so the whole function plus
   the leaf kernels it calls direct-by-name adds no new call sites.
   Declared in src/ops_mmx.h / src/ops_sse2.h. row_q41_sse2_asm's
   -DLZ_SSE2_GROUP=0 branch calls lz_dot128_q41_mmx_w, src/ops_mmx.c's
   real wrapper around the MMX #pragma aux group kernel - same cross-TU
   need as Q8_0's lz_dot128_x16_mmx_w. */

#else  /* gcc: intrinsics */

static void row_q41_mmx_intrin(int32_t *acc32, const void *w4, const void *w2,
                               const int16_t *xw, int nb, int nt, int in_dim,
                               const void *pf_end,
                         const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *wend = (const unsigned char *)pf_end;
    int tk, s;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        /* Pair with the next token so ONE nibble unpack serves both. */
        if (tk + 1 < nt && g_pair) {
            const int16_t *xw2 = xwt + in_dim;
            int32_t *accb = acc + nb;
            for (s = 0; s < nb; s++) {
                const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16;
                if (pf_ < wend) LZ_PFI_NTA(pf_);
                dot32_q41_mmx_2(wn + (size_t)s * 16,
                                xwt + (size_t)s * 32,
                                xw2 + (size_t)s * 32,
                                acc + s, accb + s);
            }
            tk++;
        } else {
#if defined(LZ_G128_GCC) && LZ_G128_GCC
            if ((nb & 3) == 0) LZ_Q41_GROUP4(dot128_q41_mmx);
            else               LZ_Q41_ACC(dot32_q41_mmx, LZ_PFI_NTA);
#else
            LZ_Q41_ACC(dot32_q41_mmx, LZ_PFI_NTA);
#endif /* LZ_G128_GCC */
        }
    }
}

#endif /* __WATCOMC__ */
#endif /* __MMX__ */

/* ---- LZ_ROW_Q41 + matmul_q41_impl ---- */
/* row_q41_sse2_intrin's body lives in src/ops_sse2.c, same reason as
   row_q8_sse2_intrin above. Declared in src/ops_sse2.h. */

#if defined(__arm__)
/* The two ARM tiers, picked in matmul_q41_impl rather than from the
   table below. Leaves in src/ops_arm.c; the w4 pointer is the 4-bit
   plane (16 bytes per 32-wide block), the same slot
   row_q41_mmx_intrin reads. */
static void row_q41_arm(int32_t *acc32, const void *w4, const void *w2,
                        const int16_t *xw, int nb, int nt, int in_dim,
                        const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    int tk, s;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_q41_arm(wn + (size_t)s * 16,
                                      xwt + (size_t)s * 32);
    }
}

#if defined(LZ_ARM_ASM_EXTERN)
static void row_q41_arm_asm(int32_t *acc32, const void *w4, const void *w2,
                            const int16_t *xw, int nb, int nt, int in_dim,
                            const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    int tk, s;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_q41_arm_asm(wn + (size_t)s * 16,
                                          xwt + (size_t)s * 32);
    }
}
#endif /* LZ_ARM_ASM_EXTERN */
#endif /* __arm__ */

const lz_rowfn LZ_ROW_Q41[LZ_ROW_N] = {
#if defined(LZ_DOT_MMX_EXTERN)
    row_q41_mmx_intrin,
#else
    NULL,
#endif /* LZ_DOT_MMX_EXTERN */
    NULL,        /* sse-intrin: no SSE1 op applies, see lz_row_pick */
#if defined(LZ_ROW_SSE2_EXTERN)
    row_q41_sse2_intrin,
#else
    NULL,
#endif /* LZ_ROW_SSE2_EXTERN */
#if defined(__WATCOMC__)
    row_q41_mmx_asm,
    NULL,        /* sse-asm: ditto */
    row_q41_sse2_asm
#else
    NULL, NULL, NULL
#endif /* __WATCOMC__ */
};

/* Fixed-epi predicate for the Q4_1/Q6_1/T2 dot/scale term (int-pipeline
   milestone 1, docs/int-pipeline-project.md #2.1) - shared by
   matmul_q41_impl, matmul_t2_impl and matmul_q61_impl, all three of
   which call epi_q41/epi_q41_fixed.

   Deliberately its own function rather than a literal call to
   q8_fixed_epi above, even though the two bodies are identical: the
   project doc calls this out explicitly (docs/int-pipeline-project.md
   §2.1's numerical-risk note) - q8_fixed_epi's bound (LZ_EPI_MAX_NG) was
   derived for Q8_0's SIGNED int8 x int8 accumulator range, and while
   Q4_1/Q6_1/T2's UNSIGNED nibble/2-bit codes give a PROVABLY SMALLER
   |acc| (Q6_1's worst case is 63*127*32*256 = 65,544,192 < 2^26, against
   Q8_0's 127*127*32*256 = 132,111,616 ~ 2^27 - measured on the real
   model too: .prof/ probe against kmr20, max |acc32| observed across
   Q4_1/Q6_1/T2 was 16,977,664, ~2^24, comfortably under both), the two
   predicates must not be coupled by sharing one function - a future
   change to either tier's gating must not silently move the other.

   `zneg` is T2, which needs no zero plane at all: its zero coefficient
   is -scale, already quantized into sq/sexp, so demanding zq/zexp there
   would refuse a tier that is in fact ready. `r` carries LZ_EPI_MAX_R,
   the zero term's own int32 bound (src/ops.h) - a second shape limit
   beside LZ_EPI_MAX_NG, checked here for the same reason that one is:
   past it the integer fold would wrap silently, so the tensor keeps the
   float epilogue instead. No tensor on kmr20 is excluded by it
   (measured over both gate corpora: r is 4, 8 or 16 on 100% of calls). */
static int q41_fixed_epi(const LZTensor *w, int in_dim, int nb, int nt,
                         int r, int zneg) {
    if (!lz_epi_mode()) return 0;
    if (!lz_epi_ready(w)) lz_epi_prep((LZTensor *)w, in_dim);
    if (!zneg && !lz_epi_zero_ready(w)) lz_epi_prep_zero((LZTensor *)w, in_dim);
    return lz_epi_ready(w) && (zneg || lz_epi_zero_ready(w))
           && nb <= LZ_EPI_MAX_NG && r <= LZ_EPI_MAX_R
           && nt <= LZ_BATCH_MAX;
}

/* The zero term's ACTIVATION half, shared by Q4_1/Q6_1/T2:
   g_xg[t*ng + g] = Sum_{s in g} (Sum_k xq[k]) * xqs[s], computed once
   per token and reused by every output row. The weight half (the
   per-group min, or -scale for T2) meets it later inside epi_q41 /
   epi_q41_fixed.

   One function rather than the three identical copies matmul_q41_impl,
   matmul_t2_impl and matmul_q61_impl each carried: the accumulation
   ORDER is part of the same bit-identity contract epi_q41's own comment
   describes, and three copies of it drift the moment one is touched.

   The FLOAT path's copy only: the fixed path folds the same quantity in
   integers (epi_zero_act_int, src/ops.c) and the two are exclusive, one
   or the other per call. The inner 32-wide sum is integer and bills
   nothing, per the census convention; the r-long fold is nt*ng*r
   converts, multiplies and adds of real x87 work that this site failed
   to declare until it was given a name. */
static void epi_zero_act(const int8_t *xq, const float *xqs, int in_dim,
                         int ng, int r, int nb, int nt) {
    int t, g, s, k;
    LZ_FCX(LZ_FC_EPI, (long)nt * ng * r, (long)nt * ng * r, 0,
           (long)nt * ng * r, 0);
    for (t = 0; t < nt; t++) {
        const int8_t *xqt = xq + (size_t)t * in_dim;
        const float *xst = xqs + (size_t)t * nb;
        float *xgt = g_xg + (size_t)t * ng;
        for (g = 0; g < ng; g++) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int sb = g * r + s;
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xqt[sb * 32 + k];
                a += lz_i32f(sx) * xst[sb];
            }
            xgt[g] = a;
        }
    }
}

void matmul_q41_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    int i, t;
    int r  = w->gs / 32;              /* 32 sub-blocks per weight group */
    int ng = in_dim / w->gs;          /* weight-group count */
    int nb = in_dim / 32;             /* 32-sub-block count */
    /* Same predicate as the Q8_0 float epilogue, own copy - see
       q41_fixed_epi's comment. */
    int fixed_epi = q41_fixed_epi(w, in_dim, nb, nt, r, 0);
    int want_i16 = (out->oi != NULL) || out->probe;
    lz_rowfn row;

    row = lz_row_pick(LZ_ROW_Q41);
#if defined(__arm__)
#if defined(LZ_ARM_ASM_EXTERN)
    row = (g_kernel == LZ_KERNEL_ARM_ASM) ? row_q41_arm_asm : row_q41_arm;
#else
    row = row_q41_arm;
#endif /* LZ_ARM_ASM_EXTERN */
#endif /* __arm__ */
    /* The int16 exit is the fixed epilogue's exit, and matmul_scalar_ref
       has no counterpart for it. Both refusals leave `ok` at 0 and write
       nothing rather than silently handing back the other domain. */
    if (want_i16 && (!fixed_epi || !row || in_dim > LZ_MM_WIDEN_MAX)) return;
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(out->o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    out->ok = 1;
    if (out->probe) return;

    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    if (fixed_epi) {
        epi_pack_act(xqs, nb, nt);
        epi_zero_act_int(xq, in_dim, ng, r, nt);
    } else {
        epi_zero_act(xq, xqs, in_dim, ng, r, nb, nt);
    }

    for (i = 0; i < out_dim; i++) {
        const unsigned char *wn =
            (const unsigned char *)w->q + (size_t)i * in_dim / 2;
        const unsigned char *wend =            /* prefetch upper bound */
            (const unsigned char *)w->q + (size_t)w->n / 2;
        const float *wd = w->scale + (size_t)i * ng;
        const float *wm = w->zero + (size_t)i * ng;
        /* Weight row loads once and serves all nt tokens; one EMMS per
           row, not per token - the epilogue below is x87 on Watcom. */
        row(g_acc32, wn, NULL, g_xw, nb, nt, in_dim, wend, NULL);
        if (lz_cpu_has_mmx()) _mm_empty();
        epi_q41_row(out, i, out_dim, nt, w, ng, r, nb, 0, fixed_epi,
                    xqs, wd, wm);
    }
}

/* ---- T2 guard block ---- */
#if defined(LZ_DOT_MMX_EXTERN)
static void row_t2_mmx_intrin(int32_t *acc32, const void *w4, const void *w2v,
                              const int16_t *xw, int nb, int nt, int in_dim,
                              const void *pf_end, const void *pf_end2) {
    const unsigned char *p2 = (const unsigned char *)w4;
    int tk, s;
    (void)w2v; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = dot32_t2_mmx(p2 + (size_t)s * 8, xwt + (size_t)s * 32);
    }
}
#endif /* LZ_DOT_MMX_EXTERN */

#if defined(__arm__)
/* ARMv5TE row kernel for the ternary tier. All six LZ_ROW_T2 slots are
   x86 SIMD (MMX/SSE2 or Watcom #pragma aux) and compile to NULL on the
   ARM cross-build, so without this override the tier would fall back to
   matmul_scalar_ref on every call. leaf = lz_dot32_t2_arm (src/ops_t2_arm.c),
   same 256*sum(code*x) contract as lz_dot32_t2_scalar; the w4 pointer is
   the 2-bit plane (8 bytes per 32-wide block), matching the w4 slot that
   row_t2_mmx_intrin reads on x86. */
static void row_t2_arm(int32_t *acc32, const void *w4, const void *w2v,
                       const int16_t *xw, int nb, int nt, int in_dim,
                       const void *pf_end, const void *pf_end2) {
    const unsigned char *p2 = (const unsigned char *)w4;
    int tk, s;
    (void)w2v; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_t2_arm(p2 + (size_t)s * 8, xwt + (size_t)s * 32);
    }
}

#if defined(LZ_T2_ARM_ASM_EXTERN)
/* Same row, hand-written leaf. A separate row function rather than a
   function pointer inside one: the leaf is called nb times per row, and
   an indirect call there would cost more than the kernel it dispatches
   to (dispatch is once per matmul, the same rule lz_row_pick follows). */
static void row_t2_arm_asm(int32_t *acc32, const void *w4, const void *w2v,
                           const int16_t *xw, int nb, int nt, int in_dim,
                           const void *pf_end, const void *pf_end2) {
    const unsigned char *p2 = (const unsigned char *)w4;
    int tk, s;
    (void)w2v; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_t2_arm_asm(p2 + (size_t)s * 8,
                                         xwt + (size_t)s * 32);
    }
}
#endif /* LZ_T2_ARM_ASM_EXTERN */
#endif /* __arm__ */

/* ---- LZ_ROW_T2 ---- */
const lz_rowfn LZ_ROW_T2[LZ_ROW_N] = {
#if defined(LZ_DOT_MMX_EXTERN)
    row_t2_mmx_intrin,
#else
    NULL,
#endif /* LZ_DOT_MMX_EXTERN */
    NULL,        /* sse-intrin: no SSE1 op applies */
#if defined(LZ_ROW_SSE2_EXTERN)
    row_t2_sse2_intrin,
#else
    NULL,
#endif /* LZ_ROW_SSE2_EXTERN */
#if defined(__WATCOMC__)
    row_t2_mmx_asm,
    NULL,        /* sse-asm: ditto */
    row_t2_sse2_asm
#else
    NULL, NULL, NULL
#endif /* __WATCOMC__ */
};

/* ---- Q61 guard block ---- */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)

/* LZ_Q61_PF / LZ_Q61_ACC / LZ_Q61_GROUP4 are in
   src/ops_kernel_dot_shared.h - shared with gcc's row_q61_mmx_intrin
   below via the same guard. */

#ifdef __WATCOMC__
/* row_q61_mmx_asm / row_q61_sse2_asm live in src/ops_mmx.c /
   src/ops_sse2.c, alongside their leaf kernels which they call
   direct-by-name. Declared in src/ops_mmx.h / src/ops_sse2.h.
   row_q61_sse2_asm's -DLZ_SSE2_GROUP=0 branch calls
   lz_dot128_q61_mmx_w, src/ops_mmx.c's real wrapper around the MMX
   #pragma aux group kernel. */

#else  /* gcc: intrinsics */

static void row_q61_mmx_intrin(int32_t *acc32, const void *w4, const void *w2v,
                               const int16_t *xw, int nb, int nt, int in_dim,
                               const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *w2 = (const unsigned char *)w2v;
    const unsigned char *wend = (const unsigned char *)pf_end;
    const unsigned char *wend2 = (const unsigned char *)pf_end2;
    int tk, s;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        /* Q6_1 gains the most of the three formats from pairing: the
           weight side is 61% of the kernel's ops against Q4_1's 40% and
           Q8's 33%, and on the recipe of record it also carries the
           most bytes. Both planes' prefetches are issued once for the
           pair - the second token reads bytes the first just pulled. */
        /* No LZ_Q61_FORCE_SSE2 macro here: a compile-time switch whose
           only job is to make the SSE2 kernel reachable on a machine
           where the dispatch would not pick it has nothing left to do,
           since one gcc build carries both tiers and `--kernel sse2`
           selects row_q61_sse2_intrin for real. A knob that duplicates
           a knob is worse than no knob - the rule asks for a switch
           plus a gate proving it changes something, and this one would
           change nothing. */
        if (tk + 1 < nt && g_pair) {
            const int16_t *xw2 = xwt + in_dim;
            int32_t *accb = acc + nb;
            for (s = 0; s < nb; s++) {
                LZ_Q61_PF(s);
                dot32_q61_mmx_2(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                xwt + (size_t)s * 32, xw2 + (size_t)s * 32,
                                acc + s, accb + s);
            }
            tk++;
            continue;
        }
#if defined(LZ_G128_GCC) && LZ_G128_GCC
        if ((nb & 3) == 0) LZ_Q61_GROUP4(dot128_q61_mmx);
        else               LZ_Q61_ACC(dot32_q61_mmx);
#else
        LZ_Q61_ACC(dot32_q61_mmx);
#endif /* LZ_G128_GCC */
    }
}

#endif /* __WATCOMC__ */
#endif /* __MMX__ */

/* ---- LZ_ROW_Q61 ---- */
#if defined(__arm__)
/* The two ARM tiers, picked in matmul_q61_impl rather than from the
   table below. Leaves in src/ops_arm.c. The only format that uses
   BOTH plane pointers: w4 is the 4-bit plane (16 bytes per 32-wide
   block), w2v the 2-bit one (8 bytes), exactly as row_q61_mmx_intrin
   reads them. */
static void row_q61_arm(int32_t *acc32, const void *w4, const void *w2v,
                        const int16_t *xw, int nb, int nt, int in_dim,
                        const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *w2 = (const unsigned char *)w2v;
    int tk, s;
    (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_q61_arm(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                      xwt + (size_t)s * 32);
    }
}

#if defined(LZ_ARM_ASM_EXTERN)
static void row_q61_arm_asm(int32_t *acc32, const void *w4, const void *w2v,
                            const int16_t *xw, int nb, int nt, int in_dim,
                            const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *w2 = (const unsigned char *)w2v;
    int tk, s;
    (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_q61_arm_asm(wn + (size_t)s * 16,
                                          w2 + (size_t)s * 8,
                                          xwt + (size_t)s * 32);
    }
}
#endif /* LZ_ARM_ASM_EXTERN */
#endif /* __arm__ */

const lz_rowfn LZ_ROW_Q61[LZ_ROW_N] = {
#if defined(LZ_DOT_MMX_EXTERN)
    row_q61_mmx_intrin,
#else
    NULL,
#endif /* LZ_DOT_MMX_EXTERN */
    NULL,        /* sse-intrin: no SSE1 op applies, see lz_row_pick */
#if defined(LZ_ROW_SSE2_EXTERN)
    row_q61_sse2_intrin,
#else
    NULL,
#endif /* LZ_ROW_SSE2_EXTERN */
#if defined(__WATCOMC__)
    row_q61_mmx_asm,
    NULL,        /* sse-asm: ditto */
    row_q61_sse2_asm
#else
    NULL, NULL, NULL
#endif /* __WATCOMC__ */
};

/* ---- matmul_t2_impl + matmul_q61_impl ---- */
void matmul_t2_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                           const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, g, t;
    int gs = w->gs, r = gs / 32, ng = in_dim / gs;
    int nb = in_dim / 32;
    const unsigned char *p2b = (const unsigned char *)w->q;
    static float g_t2neg[LZ_MM_WIDEN_MAX / 32];
    /* Same predicate as matmul_q41_impl - T2 shares epi_q41/epi_q41_fixed
       and reads the same w->scale field the dot/scale term quantizes
       (T2's zero coefficient is a negated COPY of it, g_t2neg below, not
       a different tensor field). Hence the zneg argument: the fixed path
       negates the sq/sexp plane in place of a zero plane it does not
       have, so the predicate must not require one. */
    int fixed_epi = q41_fixed_epi(w, in_dim, nb, nt, r, 1);
    int want_i16 = (out->oi != NULL) || out->probe;
    lz_rowfn row;

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_T2);
#if defined(__arm__)
    /* All LZ_ROW_T2 slots are x86 SIMD and NULL on the ARM build, so the
       ARM column is picked here instead of from the table. Two cells,
       one enum value each (rule 8): the unrolled C and the hand-written
       asm, both in src/ops_t2_arm.c. LZ_KERNEL_REF never reaches this
       function - lz_matmul_xq sends it to matmul_scalar_ref first. */
#if defined(LZ_T2_ARM_ASM_EXTERN)
    row = (g_kernel == LZ_KERNEL_ARM_ASM) ? row_t2_arm_asm : row_t2_arm;
#else
    row = row_t2_arm;
#endif /* LZ_T2_ARM_ASM_EXTERN */
#endif /* __arm__ */
    /* Same all-or-nothing refusal as matmul_q41_impl - see LZMatOut. */
    if (want_i16 && (!fixed_epi || !row || in_dim > LZ_MM_WIDEN_MAX
                     || ng > LZ_MM_WIDEN_MAX / 32)) return;
    if (!row || in_dim > LZ_MM_WIDEN_MAX || ng > LZ_MM_WIDEN_MAX / 32) {
        matmul_scalar_ref(out->o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    out->ok = 1;
    if (out->probe) return;

    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    if (fixed_epi) {
        epi_pack_act(xqs, nb, nt);
        epi_zero_act_int(xq, in_dim, ng, r, nt);
    } else {
        epi_zero_act(xq, xqs, in_dim, ng, r, nb, nt);
    }

    for (i = 0; i < out_dim; i++) {
        const unsigned char *w2 = p2b + (size_t)i * in_dim / 4;
        const float *wd = w->scale + (size_t)i * ng;
        /* Only the float epilogue reads it - the fixed one negates the
           int16 plane instead. */
        if (!fixed_epi) for (g = 0; g < ng; g++) g_t2neg[g] = -wd[g];
        row(g_acc32, w2, NULL, g_xw, nb, nt, in_dim, NULL, NULL);
        if (lz_cpu_has_mmx()) _mm_empty();
        epi_q41_row(out, i, out_dim, nt, w, ng, r, nb, 1, fixed_epi,
                    xqs, wd, g_t2neg);
    }
}

void matmul_q61_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, t;
    int gs = w->gs, r = gs / 32, ng = in_dim / gs;
    int nb = in_dim / 32;
    const unsigned char *p4b = (const unsigned char *)w->q;
    const unsigned char *p2b = p4b + (size_t)w->n / 2;
    /* Same predicate as matmul_q41_impl - see q41_fixed_epi's comment. */
    int fixed_epi = q41_fixed_epi(w, in_dim, nb, nt, r, 0);
    int want_i16 = (out->oi != NULL) || out->probe;
    lz_rowfn row;

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_Q61);
#if defined(__arm__)
#if defined(LZ_ARM_ASM_EXTERN)
    row = (g_kernel == LZ_KERNEL_ARM_ASM) ? row_q61_arm_asm : row_q61_arm;
#else
    row = row_q61_arm;
#endif /* LZ_ARM_ASM_EXTERN */
#endif /* __arm__ */
    /* Same all-or-nothing refusal as matmul_q41_impl - see LZMatOut. */
    if (want_i16 && (!fixed_epi || !row || in_dim > LZ_MM_WIDEN_MAX)) return;
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(out->o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    out->ok = 1;
    if (out->probe) return;

    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    if (fixed_epi) {
        epi_pack_act(xqs, nb, nt);
        epi_zero_act_int(xq, in_dim, ng, r, nt);
    } else {
        epi_zero_act(xq, xqs, in_dim, ng, r, nb, nt);
    }

    for (i = 0; i < out_dim; i++) {
        const unsigned char *wn = p4b + (size_t)i * in_dim / 2;
        const unsigned char *w2 = p2b + (size_t)i * in_dim / 4;
        const float *wd = w->scale + (size_t)i * ng;
        const float *wm = w->zero + (size_t)i * ng;
        row(g_acc32, wn, w2, g_xw, nb, nt, in_dim,
            p2b,                        /* 4-bit plane ends where 2-bit starts */
            p2b + (size_t)w->n / 4);    /* 2-bit plane's own bound */
        if (lz_cpu_has_mmx()) _mm_empty();
        epi_q41_row(out, i, out_dim, nt, w, ng, r, nb, 0, fixed_epi,
                    xqs, wd, wm);
    }
}

/* ---- int16 exit: entry point and its predicate -----------------------
   One dtype switch, shared by both, so "which impl serves this tensor"
   is not written twice. The shape guards here are only the ones the
   impls do NOT make themselves; everything else (the fixed-epilogue
   predicate, the row-kernel pick, LZ_MM_WIDEN_MAX) is decided inside
   the impl and reported back through LZMatOut.ok.

   LZ_KERNEL_REF is refused rather than served: it means matmul_scalar_ref,
   whose float epilogue is its own code and has no int16 exit. That is
   not a new asymmetry - --kernel ref already computes different logits
   from the SIMD tiers whenever the fixed epilogue is live, for the same
   reason.

   Q16_0 sits behind the same build guard its impl and its float
   dispatch in lz_matmul_xq_nt already sit behind, not a new one: where
   matmul_q16_impl is not compiled, q16_0 reaches matmul_scalar_ref for
   the float exit too, and there is no int16 exit for that path to
   refuse into. A build without it answers 0 here and every caller
   falls back, which is the same answer it gives for f32 weights. */
static int mm_i16_dispatch(LZMatOut *out, const int8_t *xq, const float *xqs,
                           const LZTensor *w, int in_dim, int out_dim, int nt) {
    if (!w || w->gs < 32 || (w->gs % 32) != 0 || (in_dim % w->gs) != 0)
        return 0;
    if (nt < 1 || nt > LZ_BATCH_MAX) return 0;
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return 0;
    switch (w->dtype) {
        case LZ_FMT_Q8_0:
            /* Symmetric like Q16_0 - no zero plane to require. No
               in_dim cap restated either: matmul_q8_impl's own
               want_i16 refusal covers LZ_MM_WIDEN_MAX, and it is the
               one place that also knows whether a row kernel exists.
               Unlike Q16_0 this needs no build guard - the float
               dispatch reaches matmul_q8_impl unconditionally. */
            matmul_q8_impl(out, xq, xqs, w, in_dim, out_dim, nt);
            return 1;
        case LZ_FMT_Q4_1:
            if (!w->zero) return 0;
            matmul_q41_impl(out, xq, xqs, w, in_dim, out_dim, nt);
            return 1;
        case LZ_FMT_Q6_1:
            if (!w->zero) return 0;
            matmul_q61_impl(out, xq, xqs, w, in_dim, out_dim, nt);
            return 1;
        case LZ_FMT_T2:
            /* Ternary carries no min; a T2 tensor that HAS one was
               mis-loaded (lz_matmul_xq_nt's own guard). */
            if (w->zero) return 0;
            matmul_t2_impl(out, xq, xqs, w, in_dim, out_dim, nt);
            return 1;
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN) || defined(__arm__)
        case LZ_FMT_Q16_0:
            /* Symmetric like Q8_0 - no zero plane to require, and the
               widen cap is the impl's own bail. */
            if (in_dim > LZ_MM_WIDEN_MAX) return 0;
            matmul_q16_impl(out, xq, xqs, w, in_dim, out_dim, nt);
            return 1;
#endif /* Q16_0 SIMD matmul */
        default:
            return 0;
    }
}

int lz_matmul_xq_i16_ok(const LZTensor *w, int in_dim, int nt) {
    LZMatOut ex;
    memset(&ex, 0, sizeof(ex));
    ex.probe = 1;
    if (!mm_i16_dispatch(&ex, NULL, NULL, w, in_dim, 0, nt)) return 0;
    return ex.ok;
}

int lz_matmul_xq_nt_i16(short *o, const int8_t *xq, const float *xqs,
                        const LZTensor *w, int in_dim, int out_dim, int nt,
                        int target_e, int bound) {
    LZMatOut ex;
    if (!o || !xq || !xqs) return 0;
    memset(&ex, 0, sizeof(ex));
    ex.oi = o;
    ex.target_e = target_e;
    ex.bound = bound;
    if (!mm_i16_dispatch(&ex, xq, xqs, w, in_dim, out_dim, nt)) return 0;
    return ex.ok;
}

/* ---- Q16 block ---- */
/* __arm__ joins the guard for the row TABLE and the impl, not for the
   MMX kernels below: the ARM build reaches matmul_q16_impl through the
   armC/armA tiers picked inside it, and its LZ_ROW_Q16 is all-NULL just
   like the other four formats' tables are there. */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN) || defined(__arm__)
/* Q16_0 SIMD matmul. Same shape as matmul_q8_impl - weight row loads
   once, nt tokens' int32 sub-block sums computed under MMX, ONE emms,
   then the x87 float epilogue - with two differences that both come
   from the weights already being int16:

     - no x256 product scaling, so no 1/256 to cancel at the end;
     - the weight row is int16, so a 32-element sub-block is 64 bytes,
       i.e. two cache lines rather than one. Prefetch distance is
       therefore expressed in the same units as the Q8 path (32-element
       sub-blocks) and lands twice as far ahead in bytes, which is what
       we want: the byte stream is twice as fast.

   The epilogue is character-for-character the non-q4 branch of
   matmul_scalar_ref_one, including the (a0+a2)+(a1+a3) reduction order
   and lz_i32f() on every accumulator. That is the contract (rule 2):
   this function is only allowed to be faster, never different. */
/* ---- Q16_0 row kernels -------------------------------------------------
   int16 weights, so no int8->int16 unpack and no x256 product scaling -
   the kernels return the raw int32 dot and the epilogue's `post` is 1.0.
   No sse2-intrin slot yet: part32_q16 does not exist. */

/* LZ_Q16_PF / LZ_Q16_ACC / LZ_Q16_GROUP4 are in
   src/ops_kernel_dot_shared.h - shared with gcc's row_q16_mmx_intrin
   below via the same guard. */

#ifdef __WATCOMC__
/* row_q16_mmx_asm / row_q16_sse2_asm live in src/ops_mmx.c /
   src/ops_sse2.c, alongside their leaf kernels which they call
   direct-by-name. Declared in src/ops_mmx.h / src/ops_sse2.h.
   row_q16_sse2_asm's -DLZ_SSE2_GROUP=0 branch calls
   lz_dot128_q16_mmx_w, src/ops_mmx.c's real wrapper around the MMX
   #pragma aux group kernel. */

#elif defined(LZ_DOT_MMX_EXTERN)  /* gcc: intrinsics */

static void row_q16_mmx_intrin(int32_t *acc32, const void *w4, const void *w2,
                               const int16_t *xw, int nb, int nt, int in_dim,
                               const void *pf_end, const void *pf_end2) {
    const int16_t *wr = (const int16_t *)w4;
    const int16_t *wend = (const int16_t *)pf_end;
    int tk, g;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
#if defined(LZ_G128_GCC) && LZ_G128_GCC
        if ((nb & 3) == 0) LZ_Q16_GROUP4(dot128_q16_mmx);
        else               LZ_Q16_ACC(dot32_q16_mmx);
#else
        LZ_Q16_ACC(dot32_q16_mmx);
#endif /* LZ_G128_GCC */
    }
}

#endif /* __WATCOMC__ */

/* row_q16_sse2_intrin's body lives in src/ops_sse2.c, same reason as
   row_q8_sse2_intrin above. Declared in src/ops_sse2.h. */

#if defined(__arm__)
/* The two ARM tiers. All six LZ_ROW_Q16 slots are x86 SIMD and NULL on
   this build, so matmul_q16_impl picks between these directly, the way
   matmul_t2_impl does - one enum value per code body (rule 8), so
   lz_kernel_name() can answer which one ran. Leaves in src/ops_arm.c;
   the w4 pointer is the int16 weight row, matching the slot
   row_q16_mmx_intrin reads on x86. */
static void row_q16_arm(int32_t *acc32, const void *w4, const void *w2,
                        const int16_t *xw, int nb, int nt, int in_dim,
                        const void *pf_end, const void *pf_end2) {
    const int16_t *wr = (const int16_t *)w4;
    int tk, s;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_q16_arm(wr + (size_t)s * 32,
                                      xwt + (size_t)s * 32);
    }
}

#if defined(LZ_ARM_ASM_EXTERN)
/* Same row, hand-written leaf. A separate row function rather than a
   function pointer inside one: the leaf is called nb times per row, and
   an indirect call there would cost more than the kernel it dispatches
   to (dispatch is once per matmul, the rule lz_row_pick follows). */
static void row_q16_arm_asm(int32_t *acc32, const void *w4, const void *w2,
                            const int16_t *xw, int nb, int nt, int in_dim,
                            const void *pf_end, const void *pf_end2) {
    const int16_t *wr = (const int16_t *)w4;
    int tk, s;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++)
            acc[s] = lz_dot32_q16_arm_asm(wr + (size_t)s * 32,
                                          xwt + (size_t)s * 32);
    }
}
#endif /* LZ_ARM_ASM_EXTERN */
#endif /* __arm__ */

const lz_rowfn LZ_ROW_Q16[LZ_ROW_N] = {
#if defined(LZ_DOT_MMX_EXTERN)
    row_q16_mmx_intrin,
#else
    NULL,
#endif /* LZ_DOT_MMX_EXTERN */
    NULL,        /* sse-intrin: no SSE1 op applies, see lz_row_pick */
#if defined(LZ_ROW_SSE2_EXTERN)
    row_q16_sse2_intrin,
#else
    NULL,
#endif /* LZ_ROW_SSE2_EXTERN */
#ifdef __WATCOMC__
    row_q16_mmx_asm,
    NULL,        /* sse-asm: ditto */
    row_q16_sse2_asm
#else
    NULL, NULL, NULL
#endif /* __WATCOMC__ */
};

/* Fixed-epi predicate for Q16_0. Its own function rather than a call to
   q8_fixed_epi, for the reason q41_fixed_epi's comment gives at length:
   the bound behind LZ_EPI_MAX_NG is derived per format, and coupling two
   tiers through one predicate lets a change to either move the other in
   silence. Q16_0 needs no zero plane - it is symmetric, like Q8_0 - so
   lz_epi_ready alone is the readiness question.

   BOUND, derived against epi_fixed_raw's own MAC rather than restated
   from Q8_0's: lz_epi_mac sums ng products of an int32 accumulator by an
   int16 x int16 mantissa product, and LZ_EPI_MAX_NG = 64 is exactly
   where 64 * 2^27 * 2^30 fills int64. The Q16_0 kernels return an
   UNSCALED dot of int16 weights against int8 activations, so
   |acc| <= 32767 * 127 * 32 = 133,151,488 < 2^27 - the same place
   Q8_0's 127*127*32*256 = 132,111,616 lands, because the int16 weights
   are 256x larger in magnitude exactly where Q8_0 has its 256x fold.
   The other factor is unchanged: epi_pack_act and epi_prep_plane both
   clamp to +-32767, so |xq_i16 * sq| <= 2^30 whatever the format. So
   nothing in lz_epi_mac's derivation moves and LZ_EPI_MAX_NG carries
   over as-is.

   MEASURED on kmr20, both gate corpora, instrumented build: max |acc32|
   27,083,396 (zh) / 23,290,399 (en) against the 133,151,488 ceiling -
   4.9x of headroom - over three shapes (512x64 gs=512, 512x16 gs=512,
   64x1024 gs=64). The weight plane does reach the full +-32767, so the
   derivation above is not vacuous on this checkpoint. */
static int q16_fixed_epi(const LZTensor *w, int in_dim, int nb, int nt) {
    if (lz_epi_mode() && !lz_epi_ready(w)) lz_epi_prep((LZTensor *)w, in_dim);
    return lz_epi_mode() && lz_epi_ready(w) && nb <= LZ_EPI_MAX_NG
           && nt <= LZ_BATCH_MAX;
}

void matmul_q16_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, tk;
    int r = w->gs / 32;
    int ng = in_dim / 32;
    const int16_t *wbase = (const int16_t *)(const void *)w->q;
    const int16_t *wend = wbase + (size_t)w->n;   /* prefetch upper bound */
    int t;
    lz_rowfn row;
    int fixed_epi = q16_fixed_epi(w, in_dim, ng, nt);
    int want_i16 = (out->oi != NULL) || out->probe;
    /* 1.0: the Q16_0 kernels do not scale by 256 the way the Q8_0 ones
       do. Read from lz_epi_post_e, not written here, so the float exit
       and the fixed one cannot end up on different scalings. */
    int post_e = lz_epi_post_e(w);
    float post = pow2f(post_e);

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_Q16);
#if defined(__arm__)
    /* Picked here rather than from the table, same as matmul_t2_impl:
       those six slots are x86 slots. LZ_KERNEL_REF never reaches this
       function - lz_matmul_xq_nt sends it to matmul_scalar_ref first. */
#if defined(LZ_ARM_ASM_EXTERN)
    row = (g_kernel == LZ_KERNEL_ARM_ASM) ? row_q16_arm_asm : row_q16_arm;
#else
    row = row_q16_arm;
#endif /* LZ_ARM_ASM_EXTERN */
#endif /* __arm__ */
    /* Same all-or-nothing refusal as matmul_q41_impl - see LZMatOut. The
       int16 exit is the FIXED epilogue's exit (epi_fixed_align_i16 is
       epi_fixed_raw plus epi_align_i16), so !fixed_epi refuses here even
       though the float exit would have been happy to run without it. */
    if (want_i16 && (!fixed_epi || !row || in_dim > LZ_MM_WIDEN_MAX
                     || post_e == LZ_EPI_POST_NA)) return;
    /* Third bail alongside the two kernel ones, same reasoning as
       matmul_q8_impl's: a format with no lz_epi_post_e entry has no
       scaling for either epilogue below, and matmul_scalar_ref is the
       one path here that needs none. */
    if (!row || in_dim > LZ_MM_WIDEN_MAX || post_e == LZ_EPI_POST_NA) {
        matmul_scalar_ref(out->o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    out->ok = 1;
    if (out->probe) return;

    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    /* Once per token, before the row loop: a per-row rebuild would cost
       what it saves. Same call the Q8_0 path makes. */
    if (fixed_epi) epi_pack_act(xqs, ng, nt);

    for (i = 0; i < out_dim; i++) {
        const int16_t *wr = wbase + (size_t)i * in_dim;
        const float *ws = w->scale + (size_t)(i * in_dim) / w->gs;
        row(g_acc32, wr, NULL, g_xw, ng, nt, in_dim, wend, NULL);
        if (lz_cpu_has_mmx()) _mm_empty();                    /* exit MMX state before x87 */
        /* Term for term the same epilogue as Q8_0's, fixed branch
           included: Q16_0 is symmetric, so it belongs to the
           lz_epi_prep / epi_fixed_raw family and not to epi_q41's - and
           so its int16 exit is epi_fixed_align_i16, not the q41 one.
           The clamp inside epi_align_i16 has already brought the result
           within +-out->bound, so the narrowing cast cannot lose a bit. */
        if (out->oi) {
            for (tk = 0; tk < nt; tk++)
                out->oi[(size_t)tk * out_dim + i] = (short)epi_fixed_align_i16(
                    g_acc32 + (size_t)tk * ng, w, i, tk, ng, r,
                    out->target_e, out->bound);
            continue;
        }
        for (tk = 0; tk < nt; tk++)
            out->o[(size_t)tk * out_dim + i] = fixed_epi
                ? epi_fixed(g_acc32 + (size_t)tk * ng, w, i, tk, ng, r)
                : epi_q8(g_acc32 + (size_t)tk * ng, xqs + (size_t)tk * ng,
                         ws, ng, r, post);
    }
}

#endif /* Q16_0 SIMD matmul */

