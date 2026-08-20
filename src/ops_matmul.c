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
static int16_t g_xw[LZ_BATCH_MAX * LZ_MM_WIDEN_MAX];
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

void matmul_q8_impl(float *o, const int8_t *xq, const float *xqs,
                           const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, tk, t;
    int r  = w->gs / 32;    /* 32-sub-blocks per weight group (gs=128 -> 4) */
    int nb = in_dim / 32;   /* 32-sub-block count */
    const int8_t *wend = w->q + (size_t)w->n;   /* prefetch upper bound */
    lz_rowfn row;
    /* Same predicate as the float epilogue - see q8_fixed_epi. */
    int fixed_epi = q8_fixed_epi(w, in_dim, nb, nt);

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (fixed_epi) epi_pack_act(xqs, nb, nt);
    row = lz_row_pick(LZ_ROW_Q8);

    /* Two ways to have no row kernel, and both must land on the scalar
       reference rather than on a second inline implementation:
       - this build carries none for the selected ISA (all-NULL table);
       - in_dim exceeds the pre-expansion buffer.
       Both land on the scalar reference rather than on a second inline
       implementation. Inline fallbacks (an MMX dot32_nowiden loop, an
       SSE2 part32_sse2 loop, the ref branch) would be three DIFFERENT
       code paths, none of them bit-identical to the others, all of them
       unreachable - the largest in_dim in any current model is 3584
       against a 4096 cap. Dead code in three association orders is
       worse than dead code in one, and matmul_scalar_ref is the
       contract. */
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }

    /* Expand the whole batch's activations to int16 once (in_dim <=
       4096, resident in L1); amortized over out_dim rows. */
    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];

    for (i = 0; i < out_dim; i++) {
        const int8_t *wr = w->q + (size_t)i * in_dim;
        const float *ws = w->scale + (size_t)(i * in_dim) / w->gs;
        /* The weight row loads once here and serves all nt tokens; one
           EMMS per row, not per token - MMX and x87 share a register
           file and the epilogue below is x87 on the Watcom build. */
        row(g_acc32, wr, NULL, g_xw, nb, nt, in_dim, wend, NULL);
        if (lz_cpu_has_mmx()) _mm_empty();
        for (tk = 0; tk < nt; tk++)
            o[(size_t)tk * out_dim + i] = fixed_epi
                ? epi_fixed(g_acc32 + (size_t)tk * nb, w, i, tk, nb, r,
                            1.0f / 256.0f)
                : epi_q8(g_acc32 + (size_t)tk * nb, xqs + (size_t)tk * nb,
                         ws, nb, r, 1.0f / 256.0f);
    }
}

/* ---- int-accumulator output path (Q8_0) -------------------------------
   Hands out what epi_q8 consumes - the per-32-input-group int32
   accumulator and the two per-group scale factors (activation xsb,
   weight ws) - so a caller can quantize in the int domain without the
   float dequant round-trip. The accumulator is exact: int8 x int8
   products summed in int32 have no rounding; the only float rounding in
   the whole chain is epi_q8's sum order. o_acc carries the kernels'
   256-scaled sub-block sum (the same value epi_q8's post=1/256
   cancels), so a caller reproducing the float output from it must apply
   that 1/256 post exactly like epi_q8 does - the 256 is a kernel
   convention, not part of the scale factors. Both functions return 1
   when the int output was written, 0 when the tensor is on the fixed-epi
   tier (the float path would dequantize via epi_fixed, not epi_q8, so
   the reconstruct-to-epi_q8 contract would not hold - keep the float
   path instead). matmul_q8_int_scalar_ref is the contract; the impl
   routes to it under --kernel ref and on shapes its row kernels
   decline. Layouts (token-major, the same strides lz_matmul_xq_nt
   documents for o):
     o_acc[tk * (out_dim*nb) + i*nb + g]   nb = in_dim/32
     o_xsb[tk * nb + g]
     o_ws[i * (in_dim/w->gs) + g128] */

int matmul_q8_int_scalar_ref(int32_t *o_acc, float *o_xsb, float *o_ws,
                             const int8_t *xq, const float *xqs,
                             const LZTensor *w, int in_dim, int out_dim,
                             int nt) {
    int i, g, k, tk;
    int ng = in_dim / w->gs;   /* weight-group count */
    int nb = in_dim / 32;      /* 32-sub-block count */

    if (nt < 1) return 0;
    if (!w || w->dtype != LZ_FMT_Q8_0 || !xq || !xqs || !o_acc || !o_xsb ||
        !o_ws || w->gs < 32 || (w->gs % 32) != 0 || (in_dim % w->gs) != 0)
        return 0;
    if (q8_fixed_epi(w, in_dim, nb, nt)) return 0;

    for (tk = 0; tk < nt; tk++) {
        const int8_t *xqt = xq + (size_t)tk * in_dim;
        memcpy(o_xsb + (size_t)tk * nb, xqs + (size_t)tk * nb,
               (size_t)nb * sizeof(float));
        for (i = 0; i < out_dim; i++) {
            const int8_t *wr = w->q + (size_t)i * in_dim;
            const float *ws = w->scale + (size_t)i * ng;
            int32_t *acc = o_acc + ((size_t)tk * out_dim + i) * nb;
            for (g = 0; g < nb; g++) {
                const int8_t *wq = wr + (size_t)g * 32;
                const int8_t *xr = xqt + (size_t)g * 32;
                int32_t s = 0;
                for (k = 0; k < 32; k++)
                    s += (int32_t)wq[k] * (int32_t)xr[k];
                acc[g] = s * 256; /* kernels hand out the dot scaled
                                     x256; keep that contract exact */
            }
            if (tk == 0) /* ws is per row, shared by all tokens */
                memcpy(o_ws + (size_t)i * ng, ws, (size_t)ng * sizeof(float));
        }
    }
    return 1;
}

int matmul_q8_int_impl(int32_t *o_acc, float *o_xsb, float *o_ws,
                       const int8_t *xq, const float *xqs,
                       const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, tk, t;
    int ng, nb;
    const int8_t *wend;
    lz_rowfn row;

    if (nt < 1) return 0;
    if (nt > LZ_BATCH_MAX) nt = LZ_BATCH_MAX;
    if (!w || w->dtype != LZ_FMT_Q8_0 || !xq || !xqs || !o_acc || !o_xsb ||
        !o_ws || w->gs < 32 || (w->gs % 32) != 0 || (in_dim % w->gs) != 0)
        return 0;
    ng = in_dim / w->gs; /* weight-group count */
    nb = in_dim / 32;    /* 32-sub-block count */
    if (q8_fixed_epi(w, in_dim, nb, nt)) return 0;

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF)
        return matmul_q8_int_scalar_ref(o_acc, o_xsb, o_ws, xq, xqs, w,
                                        in_dim, out_dim, nt);

    row = lz_row_pick(LZ_ROW_Q8);
    if (!row || in_dim > LZ_MM_WIDEN_MAX)
        return matmul_q8_int_scalar_ref(o_acc, o_xsb, o_ws, xq, xqs, w,
                                        in_dim, out_dim, nt);

    /* Same widening as the float impl; row kernels and the acc32
       scratch are shared. */
    wend = w->q + (size_t)w->n;
    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    /* xsb is per token, shared by all output rows - copied once, before
       the row loop. */
    for (tk = 0; tk < nt; tk++)
        memcpy(o_xsb + (size_t)tk * nb, xqs + (size_t)tk * nb,
               (size_t)nb * sizeof(float));

    for (i = 0; i < out_dim; i++) {
        const int8_t *wr = w->q + (size_t)i * in_dim;
        const float *ws = w->scale + (size_t)i * ng;
        row(g_acc32, wr, NULL, g_xw, nb, nt, in_dim, wend, NULL);
        if (lz_cpu_has_mmx()) _mm_empty();
        memcpy(o_ws + (size_t)i * ng, ws, (size_t)ng * sizeof(float));
        for (tk = 0; tk < nt; tk++)
            memcpy(o_acc + ((size_t)tk * out_dim + i) * nb,
                   g_acc32 + (size_t)tk * nb, (size_t)nb * sizeof(int32_t));
    }
    return 1;
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

void matmul_q41_impl(float *o, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    int i, g, s, t, tk;
    int r  = w->gs / 32;              /* 32 sub-blocks per weight group */
    int ng = in_dim / w->gs;          /* weight-group count */
    int nb = in_dim / 32;             /* 32-sub-block count */
    lz_rowfn row;

    row = lz_row_pick(LZ_ROW_Q41);
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }

    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    /* The zero term depends only on activations; computed per token,
       shared by all of that token's output rows */
    for (t = 0; t < nt; t++) {
        const int8_t *xqt = xq + (size_t)t * in_dim;
        const float *xst = xqs + (size_t)t * nb;
        float *xgt = g_xg + (size_t)t * ng;
        for (g = 0; g < ng; g++) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int sb = g * r + s, k;
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xqt[sb * 32 + k];
                a += lz_i32f(sx) * xst[sb];
            }
            xgt[g] = a;
        }
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
        for (tk = 0; tk < nt; tk++)
            o[(size_t)tk * out_dim + i] =
                epi_q41(g_acc32 + (size_t)tk * nb, xqs + (size_t)tk * nb,
                        g_xg + (size_t)tk * ng, wd, wm, ng, r);
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
void matmul_t2_impl(float *o, const int8_t *xq, const float *xqs,
                           const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, g, s, t, tk;
    int gs = w->gs, r = gs / 32, ng = in_dim / gs;
    int nb = in_dim / 32;
    const unsigned char *p2b = (const unsigned char *)w->q;
    static float g_t2neg[LZ_MM_WIDEN_MAX / 32];
    lz_rowfn row;

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_T2);
    if (!row || in_dim > LZ_MM_WIDEN_MAX || ng > LZ_MM_WIDEN_MAX / 32) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }

    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    for (t = 0; t < nt; t++) {
        const int8_t *xqt = xq + (size_t)t * in_dim;
        const float *xst = xqs + (size_t)t * nb;
        float *xgt = g_xg + (size_t)t * ng;
        for (g = 0; g < ng; g++) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int sb = g * r + s, k;
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xqt[sb * 32 + k];
                a += lz_i32f(sx) * xst[sb];
            }
            xgt[g] = a;
        }
    }

    for (i = 0; i < out_dim; i++) {
        const unsigned char *w2 = p2b + (size_t)i * in_dim / 4;
        const float *wd = w->scale + (size_t)i * ng;
        for (g = 0; g < ng; g++) g_t2neg[g] = -wd[g];
        row(g_acc32, w2, NULL, g_xw, nb, nt, in_dim, NULL, NULL);
        if (lz_cpu_has_mmx()) _mm_empty();
        for (tk = 0; tk < nt; tk++)
            o[(size_t)tk * out_dim + i] =
                epi_q41(g_acc32 + (size_t)tk * nb, xqs + (size_t)tk * nb,
                        g_xg + (size_t)tk * ng, wd, g_t2neg, ng, r);
    }
}

void matmul_q61_impl(float *o, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, g, s, t, tk;
    int gs = w->gs, r = gs / 32, ng = in_dim / gs;
    int nb = in_dim / 32;
    const unsigned char *p4b = (const unsigned char *)w->q;
    const unsigned char *p2b = p4b + (size_t)w->n / 2;
    lz_rowfn row;

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_Q61);
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }

    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    for (t = 0; t < nt; t++) {
        const int8_t *xqt = xq + (size_t)t * in_dim;
        const float *xst = xqs + (size_t)t * nb;
        float *xgt = g_xg + (size_t)t * ng;
        for (g = 0; g < ng; g++) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int sb = g * r + s, k;
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xqt[sb * 32 + k];
                a += lz_i32f(sx) * xst[sb];
            }
            xgt[g] = a;
        }
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
        for (tk = 0; tk < nt; tk++)
            o[(size_t)tk * out_dim + i] =
                epi_q41(g_acc32 + (size_t)tk * nb, xqs + (size_t)tk * nb,
                        g_xg + (size_t)tk * ng, wd, wm, ng, r);
    }
}

/* ---- Q16 block ---- */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)
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

#else  /* gcc: intrinsics */

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

const lz_rowfn LZ_ROW_Q16[LZ_ROW_N] = {
#ifndef __WATCOMC__
    row_q16_mmx_intrin,
#else
    NULL,
#endif /* __WATCOMC__ */
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

void matmul_q16_impl(float *o, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, tk;
    int r = w->gs / 32;
    int ng = in_dim / 32;
    const int16_t *wbase = (const int16_t *)(const void *)w->q;
    const int16_t *wend = wbase + (size_t)w->n;   /* prefetch upper bound */
    int t;
    lz_rowfn row;

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_Q16);
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];

    for (i = 0; i < out_dim; i++) {
        const int16_t *wr = wbase + (size_t)i * in_dim;
        const float *ws = w->scale + (size_t)(i * in_dim) / w->gs;
        row(g_acc32, wr, NULL, g_xw, ng, nt, in_dim, wend, NULL);
        if (lz_cpu_has_mmx()) _mm_empty();                    /* exit MMX state before x87 */
        /* post = 1.0: the Q16_0 kernels do not scale by 256 the way the
           Q8_0 ones do. Otherwise term for term the same epilogue. */
        for (tk = 0; tk < nt; tk++)
            o[(size_t)tk * out_dim + i] =
                epi_q8(g_acc32 + (size_t)tk * ng, xqs + (size_t)tk * ng,
                       ws, ng, r, 1.0f);
    }
}

#endif /* Q16_0 SIMD matmul */

