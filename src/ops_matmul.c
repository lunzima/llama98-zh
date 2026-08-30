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
#include "ops_sse.h"
#include "ops_sse2.h"
#include "ops_kernel_shared.h"
#include "ops_quant.h"
#include "ops_sched.h"
#include "ops_matmul.h"
#include "ops_t2_arm.h"
#include "ops_arm.h"

/* LZ_GDN_FIXED, LZ_EPI_MAX_NG, LZ_MM_WIDEN_MAX: ops.h. */

/* ---- scratch, sized by LZ_MM_WIDEN_MAX (ops.h) ---- */

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
   guards.

   SIXTEEN, NOT FOUR. The SSE2 kernels read
   this array through _mm_loadu_si128, and an alignment probe over one
   full fixture says they never needed to: 189,486,720 calls to
   part32_x16, and the OR of every pointer's low four bits is ZERO on
   both the weight and the activation side. That measurement is about
   one model on one host, though - what makes it a property is
   declaring it. gcc happens to give a 64KB array 32-byte alignment;
   int16_t promises 2. Between "happens to" and "promises" there was
   nothing, which is exactly the gap that made the ARM LDM note
   necessary in the first place.

   Watcom has no __attribute__ and its 32-bit malloc/static alignment is
   8, so the aligned SSE2 form cannot be switched on there off the back
   of this - the declaration is what the gcc twins can rely on, and the
   Watcom twins keep their unaligned loads until their own probe says
   otherwise. */
#if defined(__GNUC__)
#define LZ_XW_ALIGN __attribute__((aligned(16)))
#else
#define LZ_XW_ALIGN
#endif /* __GNUC__ */
static int16_t g_xw[LZ_BATCH_MAX * LZ_MM_WIDEN_MAX] LZ_XW_ALIGN;
static float   g_xg[LZ_BATCH_MAX * (LZ_MM_WIDEN_MAX / 32)];
/* Unguarded: the SSE2-intrinsics path writes here too now that it fills
   acc32 instead of fusing the float work into the integer loop. Costs
   4KB of BSS in every build. */
static int32_t g_acc32[LZ_BATCH_MAX * (LZ_MM_WIDEN_MAX / 32)];

/* ---- lz_rowfn typedef + fwd decl ---- */
typedef void (*lz_rowfn)(const lz_row_ctx *c);

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
        short *op = out->oi + i;
        const int32_t *accp = g_acc32;
        for (tk = 0; tk < nt; tk++, op += out_dim, accp += nb)
            *op = (short)epi_q41_align_i16(accp, w, i, tk, ng, r, zneg,
                                           out->target_e, out->bound);
        return;
    }
    {
        float *op = out->o + i;
        const int32_t *accp = g_acc32;
        const float *xsp = xqs;
        const float *xgp = g_xg;
        for (tk = 0; tk < nt; tk++, op += out_dim, accp += nb,
             xsp += nb, xgp += ng)
            *op = fixed_epi
                ? epi_q41_fixed(accp, w, i, tk, ng, r, zneg)
                : epi_q41(accp, xsp, xgp, wd, wm, ng, r);
    }
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

static void row_q8_mmx_intrin(const lz_row_ctx *c) {
    const int8_t *wr = (const int8_t *)c->w4;
    const int8_t *wend = (const int8_t *)c->pf_end;
    int tk, g;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        /* Pair this token with the next so ONE weight unpack serves
           both. The prefetch is issued once for the pair, not once per
           token: it is a hint about the WEIGHT row, and the second
           token reads the bytes the first just pulled in. An odd nt
           leaves the last token on the single-token kernel. */
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (g = 0; g < c->nb; g++) {
                    const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                    if (pf_ < wend) LZ_PFI_NTA(pf_);
                    dot32_x16_mmx_2(wr + (size_t)g * 32,
                                    xwt + (size_t)g * 32,
                                    xw2 + (size_t)g * 32,
                                    acc + g, accb + g);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
#if defined(LZ_G128_GCC) && LZ_G128_GCC
            /* Group-of-four path behind the LZ_G128_GCC knob (default on,
               ops_kernel_shared.h): one call per 128 elements with the
               mask construction hoisted, mirroring the Watcom row. The
               per-32 fallback stays LZ_Q8_ACC/NTA exactly as before, so
               toggling the knob changes only the group path. */
            if ((c->nb & 3) == 0) LZ_Q8_GROUP4(dot128_x16_mmx);
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
static void row_q8_arm(const lz_row_ctx *c) {
    const int8_t *wr = (const int8_t *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_q8_arm(wr + (size_t)s * 32,
                                         xwt + (size_t)s * 32);
        }
    }
}

#if defined(LZ_ARM_ASM_EXTERN)
static void row_q8_arm_asm(const lz_row_ctx *c) {
    const int8_t *wr = (const int8_t *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_q8_arm_asm(wr + (size_t)s * 32,
                                             xwt + (size_t)s * 32);
        }
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
    /* Both reads hoisted: lz_epi_mode is a cross-TU call and lz_epi_ready
       a pointer test, and neither can change between the two statements -
       lz_epi_prep only ever makes ready go false->true, which is why the
       second read is of the value AFTER the prep, not a re-ask. */
    int on = lz_epi_mode();
    if (on && !lz_epi_ready(w)) lz_epi_prep((LZTensor *)w, in_dim);
    return on && lz_epi_ready(w) && nb <= LZ_EPI_MAX_NG
           && nt <= LZ_BATCH_MAX;
}

void matmul_q8_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                           const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, tk, t;
    int r  = (unsigned)w->gs >> 5;    /* 32-sub-blocks per weight group (gs=128 -> 4) */
    int nb = (unsigned)in_dim >> 5;   /* 32-sub-block count */
    const int8_t *wend = w->q + (size_t)w->n;   /* prefetch upper bound */
    lz_rowfn row;
    const int8_t *wr;   /* C89: declarations before statements */
    /* THE EMMS GUARD IS INVARIANT AND WAS ASKED PER ROW. `_mm_empty()`
       is the caller's duty after a row kernel touches MMX state, and
       the guard exists because the instruction faults on a machine
       without MMX - but which machine this is cannot change inside the
       loop. build/dispatch_hotness.sh, kmr20, eight tokens: 1,000,256
       executions of lz_cpu_has_mmx's body across the four matmul row
       loops, the largest single category in the engine once the tier
       dispatches were hoisted.

       It is already a static inline reading a cached global (ops_quant.h
       carries the measurement behind that: 75,080,736 calls when it was
       an extern) - so this is not "make the check cheaper" a second
       time. It is
       asking it once instead of once per row. */
    int emms = lz_cpu_has_mmx();
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

    wr = w->q;
    for (i = 0; i < out_dim; i++, wr += in_dim) {
        /* The weight row loads once here and serves all nt tokens; one
           EMMS per row, not per token - MMX and x87 share a register
           file and the epilogue below is x87 on the Watcom build. */
        lz_row_ctx rctx;
        rctx.acc32 = g_acc32;
        rctx.w4 = wr;
        rctx.w2 = NULL;
        rctx.xw = g_xw;
        rctx.nb = nb;
        rctx.nt = nt;
        rctx.in_dim = in_dim;
        rctx.pf_end = wend;
        rctx.pf_end2 = NULL;
        row(&rctx);
        if (emms) _mm_empty();
        /* The clamp inside epi_align_i16 has already brought the result
           within +-out->bound, so the narrowing cast cannot lose a bit. */
        if (out->oi) {
            short *op = out->oi + i;
            const int32_t *accp = g_acc32;
            for (tk = 0; tk < nt; tk++, op += out_dim, accp += nb)
                *op = (short)epi_fixed_align_i16(accp, w, i, tk, nb, r,
                                                 out->target_e, out->bound);
            continue;
        }
        /* fixed_epi does not change inside the row loop, so it picks the
           loop rather than each element: one predictable branch per row
           instead of nt unpredictable ones, which is the difference the
           target's predictor notices.
           It also puts the float scale pointer where it is used. `ws` is
           read by nothing but epi_q8, and computing it up here cost an
           integer divide on every row of every matmul - 131,072 of them
           in eight tokens on kmr20, where the fixed epilogue runs and
           epi_q8 never does. The plane is indexed over the flat element
           array, so the expression stays (i*in_dim)/gs rather than
           becoming i*(in_dim/gs): those differ when gs does not divide
           in_dim, and matmul_scalar_ref_one treats that as a tier of its
           own rather than an impossibility. */
        if (fixed_epi) {
            float *op = out->o + i;
            const int32_t *accp = g_acc32;
            for (tk = 0; tk < nt; tk++, op += out_dim, accp += nb)
                *op = epi_fixed(accp, w, i, tk, nb, r, post_e);
        } else {
            const float *ws = w->scale + (size_t)(i * in_dim) / w->gs;
            float *op = out->o + i;
            const int32_t *accp = g_acc32;
            const float *xsp = xqs;
            for (tk = 0; tk < nt; tk++, op += out_dim, accp += nb, xsp += nb)
                *op = epi_q8(accp, xsp, ws, nb, r, post);
        }
    }
}

/* ---- matmul_scalar_ref_one + matmul_scalar_ref ---- */
void matmul_scalar_ref_one(float *o, const int8_t *xq, const float *xqs,
                                  const LZTensor *w, int in_dim, int out_dim) {
    int i, g, s, k;
    int gs = w->gs, r, ng, nsb;
    int t2 = (w->dtype == LZ_FMT_T2);
    /* T2 rides the q4 flag for the ACTIVATION-SUM hoist, which is the
       same computation for both (xgref[g] = sum(xqs)*sum(xq)). What differs
       is the coefficient it is multiplied by at the end: Q4_1/Q6_1 use
       the per-group min, T2 uses -scale. See model.h's T2 note. */
    int q4 = (w->dtype == LZ_FMT_Q4_1 || w->dtype == LZ_FMT_Q6_1 || t2);
    int q6 = (w->dtype == LZ_FMT_Q6_1);
    int q16 = (w->dtype == LZ_FMT_Q16_0);
    /* static not stack: Win98's stack is tight; see the xw/acc32 note in this file. */
    static int32_t accref[LZ_MM_WIDEN_MAX / 32];
    static float   xgref[LZ_MM_WIDEN_MAX / 32];
    const int8_t *wr;
    const int16_t *w16;
    const char *ws;
    const char *wz;   /* C89: declarations before statements */

    if (gs < 32 || (gs % 32) != 0 || (in_dim % gs) != 0) {
        /* degenerate tier (gs < 32, activations share weight groups): Q8_0 only */
        const int8_t *wr = w->q;
        for (i = 0; i < out_dim; i++, wr += in_dim) {
            const float *ws = w->scale + (size_t)(i * in_dim) / gs;
            float sum = 0.0f;
            const int8_t *wq = wr;
            const int8_t *xr = xq;
            for (g = 0; g < (int)((unsigned)in_dim / (unsigned)gs); g++, wq += gs, xr += gs) {
                int32_t acc = 0;
                for (k = 0; k < gs; k++)
                    acc += (int32_t)wq[k] * (int32_t)xr[k];
                sum += lz_i32f(acc) * xqs[g] * ((const float *)ws)[g];
            }
            o[i] = sum;
        }
        return;
    }
    r  = (unsigned)gs >> 5;                   /* 32 sub-blocks per weight group */
    ng = (unsigned)in_dim / (unsigned)gs;
    nsb = (unsigned)in_dim >> 5;              /* total number of 32-element sub-blocks */
    if (nsb > LZ_MM_WIDEN_MAX / 32) return;      /* defense: no silent miscalc */

    /* Q4_1's zero term xg[g] = sum_{sub-block s in g} xqs[s]*sum_k xq[k] depends
       only on activations and is shared by all output rows - it MUST be
       summed in the same order as matmul_q41_impl, or the
       bit-identical contract breaks on this branch. */
    if (q4) {
        const float *xsp = xqs;
        const int8_t *xqp = xq;
        for (g = 0; g < ng; g++, xsp += r, xqp += r * 32) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xqp[s * 32 + k];
                a += lz_i32f(sx) * xsp[s];
            }
            xgref[g] = a;
        }
    }

    wr = w->q;
    w16 = (const int16_t *)(const void *)w->q;
    ws = (const char *)w->scale;
    wz = (q4 && !t2) ? (const char *)w->zero
                     : (const char *)w->scale;
    for (i = 0; i < out_dim; i++, wr += in_dim, w16 += in_dim,
         ws += (unsigned)ng << 2, wz += (unsigned)ng << 2) {
        const unsigned char *wn = (const unsigned char *)w->q +
                                  (size_t)i * in_dim / 2;
        const unsigned char *w2 = (const unsigned char *)w->q +
                                  (size_t)w->n / 2 + (size_t)i * in_dim / 4;

        /* Step 1: the row's int32 sub-block sums. On the SIMD side
           this segment is the MMX/SSE2 kernel. */
        for (s = 0; s < nsb; s++) {
            int base = s * 32;
            int32_t acc = 0;
            if (q6) {
                /* dot(q,x) = dot(lo,x) + 16*dot(hi,x), exact in
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
            const int32_t *ap = accref;
            const float *xp = xqs;
            for (g = 0; g < ng; g++, ap += r, xp += r) {
                float dot = 0.0f;
                for (s = 0; s < r; s++)
                    dot += lz_i32f(ap[s]) * xp[s];
                dotsum += dot * ((const float *)ws)[g];
                /* T2's zero coefficient is -scale, not a stored min:
                   sum(w*x) = d*[sum(code*x) - sum(x)]. Written as a negated
                   multiply rather than folded into the dot term so the
                   two accumulators stay separate, matching the SIMD
                   epilogue's merge order. */
                zsum += xgref[g] * (t2 ? -((const float *)ws)[g] : ((const float *)wz)[g]);
            }
            o[i] = dotsum + zsum;
        } else if (r > 1) {
            float sum = 0.0f;
            const int32_t *ap = accref;
            const float *xp = xqs;
            for (g = 0; g < ng; g++, ap += r, xp += r) {
                float dot = 0.0f;
                for (s = 0; s < r; s++)
                    dot += lz_i32f(ap[s]) * xp[s];
                sum += dot * ((const float *)ws)[g];
            }
            o[i] = sum;
        } else {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (g = 0; g + 3 < ng; g += 4) {
                a0 += lz_i32f(accref[g])     * (xqs[g]     * ((const float *)ws)[g]);
                a1 += lz_i32f(accref[g + 1]) * (xqs[g + 1] * ((const float *)ws)[g + 1]);
                a2 += lz_i32f(accref[g + 2]) * (xqs[g + 2] * ((const float *)ws)[g + 2]);
                a3 += lz_i32f(accref[g + 3]) * (xqs[g + 3] * ((const float *)ws)[g + 3]);
            }
            for (; g < ng; g++)                  /* tail all into a0, same as SIMD */
                a0 += lz_i32f(accref[g]) * (xqs[g] * ((const float *)ws)[g]);
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
    int ss = (gsa > 0) ? (unsigned)in_dim / (unsigned)gsa : (unsigned)in_dim >> 5;
    int t;
    float *op = o;
    const int8_t *xqp = xq;
    const float *xsp = xqs;
    for (t = 0; t < nt; t++, op += out_dim, xqp += in_dim, xsp += ss)
        matmul_scalar_ref_one(op, xqp, xsp, w, in_dim, out_dim);
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

static void row_q41_mmx_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *wend = (const unsigned char *)c->pf_end;
    int tk, s;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        /* Pair with the next token so ONE nibble unpack serves both. */
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (s = 0; s < c->nb; s++) {
                    const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16;
                    if (pf_ < wend) LZ_PFI_NTA(pf_);
                    dot32_q41_mmx_2(wn + (size_t)s * 16,
                                    xwt + (size_t)s * 32,
                                    xw2 + (size_t)s * 32,
                                    acc + s, accb + s);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
#if defined(LZ_G128_GCC) && LZ_G128_GCC
            if ((c->nb & 3) == 0) LZ_Q41_GROUP4(dot128_q41_mmx);
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
static void row_q41_arm(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_q41_arm(wn + (size_t)s * 16,
                                          xwt + (size_t)s * 32);
        }
    }
}

#if defined(LZ_ARM_ASM_EXTERN)
static void row_q41_arm_asm(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_q41_arm_asm(wn + (size_t)s * 16,
                                              xwt + (size_t)s * 32);
        }
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
   section 2.1's numerical-risk note) - q8_fixed_epi's bound
   (LZ_EPI_MAX_NG) was
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
    const int8_t *xqt;
    const float *xst;
    float *xgt;   /* C89: declarations before statements */
    LZ_FCX(LZ_FC_EPI, (long)nt * ng * r, (long)nt * ng * r, 0,
           (long)nt * ng * r, 0);
    xqt = xq;
    xst = xqs;
    xgt = g_xg;
    for (t = 0; t < nt; t++, xqt += in_dim, xst += nb, xgt += ng) {
        const float *xsp = xst;
        const int8_t *xqp = xqt;
        for (g = 0; g < ng; g++, xsp += r, xqp += r * 32) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xqp[s * 32 + k];
                a += lz_i32f(sx) * xsp[s];
            }
            xgt[g] = a;
        }
    }
}

void matmul_q41_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, t;
    int r  = (unsigned)w->gs >> 5;              /* 32 sub-blocks per weight group */
    int ng = in_dim / w->gs;          /* weight-group count */
    int nb = (unsigned)in_dim >> 5;             /* 32-sub-block count */
    int emms = lz_cpu_has_mmx();   /* invariant; see matmul_q8_impl */
    /* Same predicate as the Q8_0 float epilogue, own copy - see
       q41_fixed_epi's comment. */
    int fixed_epi = q41_fixed_epi(w, in_dim, nb, nt, r, 0);
    int want_i16 = (out->oi != NULL) || out->probe;
    lz_rowfn row;
    const char *wd;
    const char *wm;   /* C89: declarations before statements */

    /* After the declarations, not before them: the tier selector has no
       bearing on any initialiser above and the language floor puts
       declarations first. */
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);

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

    wd = (const char *)w->scale;
    wm = (const char *)w->zero;
    for (i = 0; i < out_dim; i++, wd += (unsigned)ng << 2,
         wm += (unsigned)ng << 2) {
        const unsigned char *wn =
            (const unsigned char *)w->q + (size_t)i * in_dim / 2;
        const unsigned char *wend =            /* prefetch upper bound */
            (const unsigned char *)w->q + (size_t)w->n / 2;
        /* Weight row loads once and serves all nt tokens; one EMMS per
           row, not per token - the epilogue below is x87 on Watcom. */
        lz_row_ctx rctx;
        rctx.acc32 = g_acc32;
        rctx.w4 = wn;
        rctx.w2 = NULL;
        rctx.xw = g_xw;
        rctx.nb = nb;
        rctx.nt = nt;
        rctx.in_dim = in_dim;
        rctx.pf_end = wend;
        rctx.pf_end2 = NULL;
        row(&rctx);
        if (emms) _mm_empty();
        epi_q41_row(out, i, out_dim, nt, w, ng, r, nb, 0, fixed_epi,
                    xqs, (const float *)wd, (const float *)wm);
    }
}

/* ---- T2 guard block ---- */
#if defined(LZ_DOT_MMX_EXTERN)
static void row_t2_mmx_intrin(const lz_row_ctx *c) {
    const unsigned char *p2 = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        /* gcc-side pairing only: dot32_t2_mmx_2 shares the ternary unpack
           across two tokens because the C intrinsic keeps its operand
           live. The Watcom twin is deliberately absent - its pmaddwd
           destroys the unpack register, so sharing it there costs a movq
           per group plus two more accumulators than eight %mm hold. */
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (s = 0; s < c->nb; s++)
                    dot32_t2_mmx_2(p2 + (size_t)s * 8,
                                   xwt + (size_t)s * 32,
                                   xw2 + (size_t)s * 32,
                                   acc + s, accb + s);
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            s = 0;
            /* Whole groups of four first, the per-32 fold for the nb % 4
               tail. Both write the same values - dot128_t2_mmx folds the
               same partials pairwise instead of singly - so the row is split
               between two instruction counts, never two answers.
               The knob is spelled this way because LZ_G128_GCC lives in
               ops_kernel_shared.h, which is not on every TU's include path;
               the A/B is run as -DLZ_G128_GCC=0. */
#if !defined(LZ_G128_GCC) || LZ_G128_GCC
            for (; s + 4 <= c->nb; s += 4)
                dot128_t2_mmx(p2 + (size_t)s * 8, xwt + (size_t)s * 32, acc + s);
#endif /* !defined(LZ_G128_GCC) || LZ_G128_GCC */
            for (; s < c->nb; s++)
                acc[s] = dot32_t2_mmx(p2 + (size_t)s * 8, xwt + (size_t)s * 32);
        }
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
static void row_t2_arm(const lz_row_ctx *c) {
    const unsigned char *p2 = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_t2_arm(p2 + (size_t)s * 8, xwt + (size_t)s * 32);
        }
    }
}

#if defined(LZ_T2_ARM_ASM_EXTERN)
/* Same row, hand-written leaf. A separate row function rather than a
   function pointer inside one: the leaf is called nb times per row, and
   an indirect call there would cost more than the kernel it dispatches
   to (dispatch is once per matmul, the same rule lz_row_pick follows). */
static void row_t2_arm_asm(const lz_row_ctx *c) {
    const unsigned char *p2 = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_t2_arm_asm(p2 + (size_t)s * 8,
                                             xwt + (size_t)s * 32);
        }
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

static void row_q61_mmx_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *w2 = (const unsigned char *)c->w2;
    const unsigned char *wend = (const unsigned char *)c->pf_end;
    const unsigned char *wend2 = (const unsigned char *)c->pf_end2;
    int tk, s;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
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
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (s = 0; s < c->nb; s++) {
                    LZ_Q61_PF(s);
                    dot32_q61_mmx_2(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                    xwt + (size_t)s * 32, xw2 + (size_t)s * 32,
                                    acc + s, accb + s);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
#if defined(LZ_G128_GCC) && LZ_G128_GCC
            if ((c->nb & 3) == 0) LZ_Q61_GROUP4(dot128_q61_mmx);
            else               LZ_Q61_ACC(dot32_q61_mmx);
#else
            LZ_Q61_ACC(dot32_q61_mmx);
#endif /* LZ_G128_GCC */
        }
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
static void row_q61_arm(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *w2 = (const unsigned char *)c->w2;
    int tk, s;
    (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_q61_arm(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                          xwt + (size_t)s * 32);
        }
    }
}

#if defined(LZ_ARM_ASM_EXTERN)
static void row_q61_arm_asm(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *w2 = (const unsigned char *)c->w2;
    int tk, s;
    (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_q61_arm_asm(wn + (size_t)s * 16,
                                              w2 + (size_t)s * 8,
                                              xwt + (size_t)s * 32);
        }
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
    int gs = w->gs, r = (unsigned)gs >> 5, ng = (unsigned)in_dim / (unsigned)gs;
    int nb = (unsigned)in_dim >> 5;
    const unsigned char *p2b = (const unsigned char *)w->q;
    static float g_t2neg[LZ_MM_WIDEN_MAX / 32];
    int emms = lz_cpu_has_mmx();   /* invariant; see matmul_q8_impl */
    /* Same predicate as matmul_q41_impl - T2 shares epi_q41/epi_q41_fixed
       and reads the same w->scale field the dot/scale term quantizes
       (T2's zero coefficient is a negated COPY of it, g_t2neg below, not
       a different tensor field). Hence the zneg argument: the fixed path
       negates the sq/sexp plane in place of a zero plane it does not
       have, so the predicate must not require one. */
    int fixed_epi = q41_fixed_epi(w, in_dim, nb, nt, r, 1);
    int want_i16 = (out->oi != NULL) || out->probe;
    lz_rowfn row;
    const float *wd;   /* C89: declarations before statements */

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_T2);
#if defined(__arm__)
    /* All LZ_ROW_T2 slots are x86 SIMD and NULL on the ARM build, so the
       ARM column is picked here instead of from the table. Two cells,
       one enum value each: the unrolled C and the hand-written
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

    wd = w->scale;
    for (i = 0; i < out_dim; i++, wd += ng) {
        const unsigned char *w2 = p2b + (size_t)i * in_dim / 4;
        lz_row_ctx rctx;
        /* Only the float epilogue reads it - the fixed one negates the
           int16 plane instead. */
        if (!fixed_epi) for (g = 0; g < ng; g++) g_t2neg[g] = -wd[g];
        rctx.acc32 = g_acc32;
        rctx.w4 = w2;
        rctx.w2 = NULL;
        rctx.xw = g_xw;
        rctx.nb = nb;
        rctx.nt = nt;
        rctx.in_dim = in_dim;
        rctx.pf_end = NULL;
        rctx.pf_end2 = NULL;
        row(&rctx);
        if (emms) _mm_empty();
        epi_q41_row(out, i, out_dim, nt, w, ng, r, nb, 1, fixed_epi,
                    xqs, (const float *)wd, g_t2neg);
    }
}

void matmul_q61_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, t;
    int gs = w->gs, r = (unsigned)gs >> 5, ng = (unsigned)in_dim / (unsigned)gs;
    int nb = (unsigned)in_dim >> 5;
    const unsigned char *p4b = (const unsigned char *)w->q;
    const unsigned char *p2b = p4b + (size_t)w->n / 2;
    int emms = lz_cpu_has_mmx();   /* invariant; see matmul_q8_impl */
    /* Same predicate as matmul_q41_impl - see q41_fixed_epi's comment. */
    int fixed_epi = q41_fixed_epi(w, in_dim, nb, nt, r, 0);
    int want_i16 = (out->oi != NULL) || out->probe;
    lz_rowfn row;
    const char *wd;
    const char *wm;
    const unsigned char *wn;
    const unsigned char *w2;   /* C89: declarations before statements */

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

    wd = (const char *)w->scale;
    wm = (const char *)w->zero;
    wn = p4b;
    w2 = p2b;
    for (i = 0; i < out_dim; i++, wd += (unsigned)ng << 2,
         wm += (unsigned)ng << 2, wn += in_dim >> 1, w2 += in_dim >> 2) {
        lz_row_ctx rctx;
        rctx.acc32 = g_acc32;
        rctx.w4 = wn;
        rctx.w2 = w2;
        rctx.xw = g_xw;
        rctx.nb = nb;
        rctx.nt = nt;
        rctx.in_dim = in_dim;
        rctx.pf_end = p2b;                      /* 4-bit plane ends where 2-bit starts */
        rctx.pf_end2 = p2b + (size_t)w->n / 4;  /* 2-bit plane's own bound */
        row(&rctx);
        if (emms) _mm_empty();
        epi_q41_row(out, i, out_dim, nt, w, ng, r, nb, 0, fixed_epi,
                    xqs, (const float *)wd, (const float *)wm);
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
   and lz_i32f() on every accumulator. That is the contract:
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

static void row_q16_mmx_intrin(const lz_row_ctx *c) {
    const int16_t *wr = (const int16_t *)c->w4;
    const int16_t *wend = (const int16_t *)c->pf_end;
    int tk, g;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        /* Pair this token with the next so ONE weight load serves both,
           the same shape q8_0/q4_1/q6_1 use above. Less is shared here
           than there - int16 weights need no unpack, so it is the load
           alone - and whether that clears the wider prologue is the
           wall-clock question g_pair exists to let someone answer. An
           odd nt leaves the last token on the single-token path.
           Ahead of the group form deliberately: pairing halves the
           weight traffic, grouping only amortizes the per-32 call, and
           a token pair cannot be expressed once the group loop owns
           the sub-block index. */
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (g = 0; g < c->nb; g++) {
                    const int16_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                    if (pf_ < wend) LZ_PFI_NTA(pf_);
                    dot32_q16_mmx_2(wr + (size_t)g * 32,
                                    xwt + (size_t)g * 32,
                                    xw2 + (size_t)g * 32,
                                    acc + g, accb + g);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
#if defined(LZ_G128_GCC) && LZ_G128_GCC
            if ((c->nb & 3) == 0) LZ_Q16_GROUP4(dot128_q16_mmx);
            else               LZ_Q16_ACC(dot32_q16_mmx);
#else
            LZ_Q16_ACC(dot32_q16_mmx);
#endif /* LZ_G128_GCC */
        }
    }
}

#endif /* __WATCOMC__ */

/* row_q16_sse2_intrin's body lives in src/ops_sse2.c, same reason as
   row_q8_sse2_intrin above. Declared in src/ops_sse2.h. */

#if defined(__arm__)
/* The two ARM tiers. All six LZ_ROW_Q16 slots are x86 SIMD and NULL on
   this build, so matmul_q16_impl picks between these directly, the way
   matmul_t2_impl does - one enum value per code body, so
   lz_kernel_name() can answer which one ran. Leaves in src/ops_arm.c;
   the w4 pointer is the int16 weight row, matching the slot
   row_q16_mmx_intrin reads on x86. */
static void row_q16_arm(const lz_row_ctx *c) {
    const int16_t *wr = (const int16_t *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_q16_arm(wr + (size_t)s * 32,
                                          xwt + (size_t)s * 32);
        }
    }
}

#if defined(LZ_ARM_ASM_EXTERN)
/* Same row, hand-written leaf. A separate row function rather than a
   function pointer inside one: the leaf is called nb times per row, and
   an indirect call there would cost more than the kernel it dispatches
   to (dispatch is once per matmul, the rule lz_row_pick follows). */
static void row_q16_arm_asm(const lz_row_ctx *c) {
    const int16_t *wr = (const int16_t *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_q16_arm_asm(wr + (size_t)s * 32,
                                              xwt + (size_t)s * 32);
        }
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
   from Q8_0's: that MAC sums ng products of an int32 accumulator by an
   int16 x int16 mantissa product, and LZ_EPI_MAX_NG = 64 is exactly
   where 64 * 2^27 * 2^30 fills int64. The Q16_0 kernels return an
   UNSCALED dot of int16 weights against int8 activations, so
   |acc| <= 32767 * 127 * 32 = 133,151,488 < 2^27 - the same place
   Q8_0's 127*127*32*256 = 132,111,616 lands, because the int16 weights
   are 256x larger in magnitude exactly where Q8_0 has its 256x fold.
   The other factor is unchanged: epi_pack_act and epi_prep_plane both
   clamp to +-32767, so |xq_i16 * sq| <= 2^30 whatever the format. So
   nothing in that derivation moves and LZ_EPI_MAX_NG carries
   over as-is.

   MEASURED on kmr20, both gate corpora, instrumented build: max |acc32|
   27,083,396 (zh) / 23,290,399 (en) against the 133,151,488 ceiling -
   4.9x of headroom - over three shapes (512x64 gs=512, 512x16 gs=512,
   64x1024 gs=64). The weight plane does reach the full +-32767, so the
   derivation above is not vacuous on this checkpoint. */
static int q16_fixed_epi(const LZTensor *w, int in_dim, int nb, int nt) {
    /* Hoisted for the reason q8_fixed_epi gives. Hoisted SEPARATELY, and
       the body stays its own: this predicate must not become a call to
       that one - see the note above epi_q41_fixed on why the two bounds
       are derived from different accumulator ranges and must be free to
       diverge. */
    int on = lz_epi_mode();
    if (on && !lz_epi_ready(w)) lz_epi_prep((LZTensor *)w, in_dim);
    return on && lz_epi_ready(w) && nb <= LZ_EPI_MAX_NG
           && nt <= LZ_BATCH_MAX;
}

void matmul_q16_impl(LZMatOut *out, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, tk;
    int r = (unsigned)w->gs >> 5;
    int ng = (unsigned)in_dim >> 5;
    const int16_t *wbase = (const int16_t *)(const void *)w->q;
    const int16_t *wend = wbase + (size_t)w->n;   /* prefetch upper bound */
    int t;
    lz_rowfn row;
    int emms = lz_cpu_has_mmx();   /* invariant; see matmul_q8_impl */
    int fixed_epi = q16_fixed_epi(w, in_dim, ng, nt);
    int want_i16 = (out->oi != NULL) || out->probe;
    /* 1.0: the Q16_0 kernels do not scale by 256 the way the Q8_0 ones
       do. Read from lz_epi_post_e, not written here, so the float exit
       and the fixed one cannot end up on different scalings. */
    int post_e = lz_epi_post_e(w);
    float post = pow2f(post_e);
    const int16_t *wr;   /* C89: declarations before statements */

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

    wr = wbase;
    for (i = 0; i < out_dim; i++, wr += in_dim) {
        lz_row_ctx rctx;
        rctx.acc32 = g_acc32;
        rctx.w4 = wr;
        rctx.w2 = NULL;
        rctx.xw = g_xw;
        rctx.nb = ng;
        rctx.nt = nt;
        rctx.in_dim = in_dim;
        rctx.pf_end = wend;
        rctx.pf_end2 = NULL;
        row(&rctx);
        if (emms) _mm_empty();                    /* exit MMX state before x87 */
        /* Term for term the same epilogue as Q8_0's, fixed branch
           included: Q16_0 is symmetric, so it belongs to the
           lz_epi_prep / epi_fixed_raw family and not to epi_q41's - and
           so its int16 exit is epi_fixed_align_i16, not the q41 one.
           The clamp inside epi_align_i16 has already brought the result
           within +-out->bound, so the narrowing cast cannot lose a bit. */
        if (out->oi) {
            short *op = out->oi + i;
            const int32_t *accp = g_acc32;
            for (tk = 0; tk < nt; tk++, op += out_dim, accp += ng)
                *op = (short)epi_fixed_align_i16(accp, w, i, tk, ng, r,
                                                 out->target_e, out->bound);
            continue;
        }
        /* Loop picked by fixed_epi, and `ws` computed only in the arm
           that reads it - the same two reasons matmul_q8_impl's row loop
           gives, and the same divide: 54,016 executions in eight tokens
           on kmr20 for a pointer epi_q8 was never handed. */
        if (fixed_epi) {
            float *op = out->o + i;
            const int32_t *accp = g_acc32;
            for (tk = 0; tk < nt; tk++, op += out_dim, accp += ng)
                *op = epi_fixed(accp, w, i, tk, ng, r, post_e);
        } else {
            const float *ws = w->scale + (size_t)(i * in_dim) / w->gs;
            float *op = out->o + i;
            const int32_t *accp = g_acc32;
            const float *xsp = xqs;
            for (tk = 0; tk < nt; tk++, op += out_dim, accp += ng, xsp += ng)
                *op = epi_q8(accp, xsp, ws, ng, r, post);
        }
    }
}

#endif /* Q16_0 SIMD matmul */

/* The float exit of the impls that take an LZMatOut, which is what every
   lz_matmul_xq_nt caller wants; the int16 one is reached through
   lz_matmul_xq_nt_i16 instead. */
static void mat_out_f32(LZMatOut *x, float *o) {
    memset(x, 0, sizeof(*x));
    x->o = o;
}

/* Matmul consuming already-quantized activations: the forward hot path.
   xq/xqs quantized by the caller at lz_act_gs()'s group (one activation
   may be reused by several matmuls). f32 weights fall back to f32 -
   one call site serves all weight representations. */
void lz_matmul_xq(float *o, const float *x, const int8_t *xq,
                  const float *xqs, const LZTensor *w,
                  int in_dim, int out_dim) {
    lz_matmul_xq_nt(o, x, xq, xqs, w, in_dim, out_dim, 1);
}

/* Which of lz_matmul_xq_nt's two inputs it will actually read.
   Exported because a caller that PREPARES those inputs differently -
   the SubLN Hadamard prepares the quantized row only - has to ask the
   same question, and two spellings of it drift. That drift already
   happened once: the rotation reached xq and never the float row, so
   an unquantized model computed W.x while a quantized one computed
   W.quant(Hx), silently, from one config. */
int lz_matmul_xq_reads_float_row(const LZTensor *w) {
    /* bf16 reads the float row too - it is the f32 arm with a wider
       load, not a quantized format. */
    return !w || w->dtype == LZ_FMT_F32 || w->dtype == LZ_FMT_BF16;
}

void lz_matmul_xq_nt(float *o, const float *x, const int8_t *xq,
                     const float *xqs, const LZTensor *w,
                     int in_dim, int out_dim, int nt) {
    int i;

    /* An out-of-range nt would read past g_xw/g_acc32. The caller
       (lz_forward_batch) already chunks by LZ_BATCH_MAX; reaching here
       is a bug - clamp, never silently overrun. */
    if (nt < 1) return;
    if (nt > LZ_BATCH_MAX) nt = LZ_BATCH_MAX;
    if (lz_matmul_xq_reads_float_row(w)) {
        float *op = o;
        const float *xp = x;
        /* TWO FORMATS READ A FLOAT ROW, not one. bf16 has no f32 array
           to point at - the whole point of it is that the tensor stays
           two bytes per element - so taking w->f here handed lz_matmul a
           null pointer and the run died in the first matmul. The
           predicate above says "reads a float row"; it does not say
           "has ->f", and that distinction is the bug. */
        for (i = 0; i < nt; i++, op += out_dim, xp += in_dim) {
            if (w && w->dtype == LZ_FMT_BF16)
                lz_matmul_bf16(op, xp, (const unsigned char *)w->q,
                               in_dim, out_dim);
            else
                lz_matmul(op, xp, w ? w->f : NULL, in_dim, out_dim);
        }
        return;
    }
    if (!xq || !xqs || w->gs <= 0 || (in_dim % w->gs) != 0) {
        for (i = 0; i < nt * out_dim; i++) o[i] = 0.0f;  /* defense: no silent miscalc */
        return;
    }
    /* The widen buffers are LZ_BATCH_MAX * LZ_MM_WIDEN_MAX; nt tokens'
       activations always fit: the g_xw-filling path carries its own
       `in_dim <= LZ_MM_WIDEN_MAX` check (Q8_0's !widened branch does
       not widen and goes per-token scalar), and Q4_1/Q6_1 dispatch
       already requires in_dim <= LZ_MM_WIDEN_MAX. */
    /* LZ_KERNEL_REF must genuinely reach the scalar reference, not just
       affect the name reported. Dispatch is by dtype, so without this
       check `--kernel ref` would actually run SIMD, unnoticed because
       all kernels are bit-identical. The scalar reference is the
       CONTRACT for new formats; an oracle that cannot be
       selected is no oracle. */
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    /* No ARM interception here, and the reason it is absent rather than
       forgotten: all five formats have their armC/armA pair
       (src/ops_arm.c, plus ternary in src/ops_t2_arm.c), so the dispatch
       below is the whole story on ARM too. A format WITHOUT one needs
       the interception - route that dtype to matmul_scalar_ref, so the
       tiers differ by the kernel under test and not by which fallback
       they fell through. Such a format lands as a `t` in km_reg. */
    /* The SIMD fast path covers all five quantised formats - Q8_0,
       Q4_1, Q6_1, T2 and Q16_0 - at any group size that is a multiple
       of 32 dividing in_dim, with the four non-Q8 ones additionally
       needing in_dim <= LZ_MM_WIDEN_MAX. `gs` is a family, not a set of
       named sub-types: kmr20 alone carries 64, 128, 256, 512 and 1024.
       Measured on it (build/tier_reach_gate.sh, eight tokens):
       matmul_scalar_ref executes ZERO times - every quantised matmul
       reaches its tiered kernel, gs=1024 and gs=64 included. T2 is 0
       there because the checkpoint has no T2 tensor, not because it
       falls back.

       This is not a question of "whether asm is worth it" decided by
       measurement. The rule requires every operator to have C /
       MMX / SSE / SSE2 paths, because the bottleneck is a seesaw -
       relieve the bytes and it moves to the ALU, relieve the ALU and it
       moves back, so "measure which side binds today, then optimise
       only that side" chases a moving target and leaves both sides
       undone. Measurement says how fast and whether it is bit-identical,
       not whether to write the kernel. Every remaining
       scalar-reference fallthrough below is therefore a TODO with a
       name, not a decision. */
    /* Both SIMD paths pre-expand activations to int16; beyond the
       buffer cap they fall back to the scalar reference. Q8 has no
       non-expanding MMX intrinsics fallback - it would be 6x slower
       than x87 scalar on Watcom; keeping it would only make a real
       trigger worse. */
    if (w->dtype == LZ_FMT_Q8_0 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0) {
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_q8_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    /* Q4_1's SIMD path pre-expands activations to int16; beyond the
       buffer cap it falls back to the scalar reference. s1v3's max
       in_dim is 1024, the 0.8B's 3584 - both within 4096. */
    if (w->dtype == LZ_FMT_Q4_1 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX && w->zero) {
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_q41_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    if (w->dtype == LZ_FMT_Q6_1 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX && w->zero) {
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_q61_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    /* No `w->zero` in the guard: ternary carries no min, and a T2 tensor
       that HAS one was mis-loaded. */
    if (w->dtype == LZ_FMT_T2 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX && !w->zero) {
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_t2_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN) || defined(__arm__)
    /* Q16_0 has no non-widened fallback: unlike Q8_0 it always needs the
       activation expanded to int16, so in_dim past the buffer cap goes to
       the scalar reference. The gcc/SSE2-only dev build has no Q16_0 SIMD
       kernel either and lands there too - that is deliberate, not an
       oversight. The differential oracle the bit-identity contract requires is a gcc build
       with LZ_MMX_TU set (src/ops_mmx.c linked in), which compiles the
       intrinsics twin and exercises this same path. */
    if (w->dtype == LZ_FMT_Q16_0 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX) {
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_q16_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */
    matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
}

/* o = W x: unified entry for weight tensors.
   dtype=0 goes f32; dtype=1 quantizes x (Q8_0, group = w->gs) into the
   caller-provided xq/xqs buffers, int8*int8 -> int32 accumulation,
   dequantized per group by (sx * sw). xq needs in_dim bytes, xqs
   in_dim/w->gs f32s. */
void lz_matmul_w(float *o, const float *x, const LZTensor *w,
                 int in_dim, int out_dim, int8_t *xq, float *xqs) {
    int i;

    if (!w || w->dtype == LZ_FMT_F32) {
        lz_matmul(o, x, w ? w->f : NULL, in_dim, out_dim);
        return;
    }
    /* bf16 is stored, not quantized: no activation quantization, no
       group scale, just a wider-on-read version of the f32 arm above.
       The in_dim bound is the widening scratch's, the same one every
       quantized kernel here carries; model.c keeps a tensor past that
       bound expanded to f32 at load, so this arm is always reachable
       when the dtype says bf16. */
    if (w->dtype == LZ_FMT_BF16) {
        lz_matmul_bf16(o, x, (const unsigned char *)w->q, in_dim, out_dim);
        return;
    }
    if (w->gs <= 0 || (in_dim % w->gs) != 0 || !xq || !xqs) {
        /* Defense: normal export guarantees in_dim % gs == 0 (in-row
           grouping). Triggering means a bug - zero the output and
           return; no silent miscalc, no crash. */
        for (i = 0; i < out_dim; i++) o[i] = 0.0f;
        return;
    }
    /* Activation groups follow lz_act_gs, not w->gs - weights may be
       coarse to gs=128 while activations stay 32. */
    lz_quantize_q8(x, in_dim, lz_act_gs(w, in_dim), xq, xqs);
    lz_matmul_xq(o, x, xq, xqs, w, in_dim, out_dim);
}

/* NEITHER OF THESE RUNS ON A QUANTISED MODEL, and the number is worth
   having before anyone optimises them. Instrumenting lz_matmul_xq_nt -
   the real dispatch point, which lz_matmul_w is not; that one takes 3
   call sites against its 29 - and running the zh fixture for 8 tokens
   on the ARM recipe counted:

       f32 0    bf16 0    quantised 728

   So the f32 row leaf below, and the bf16 twin beside it, are for
   UNQUANTISED models only. That is not a defect: it is exactly the case
   where bf16 storage matters, since a quantised tensor is already
   narrower than bf16 and has its own kernels. It does mean that
   shaving instructions here buys nothing on the shipping recipe, and
   that an earlier "150 million multiplies per token" framing for this
   cell was wrong twice over - the census it came from is a 603-token
   total, and it counts every __aeabi_fmul in the engine rather than
   this path.

   ONE ROW BODY, TWO ROW SOURCES. `w` is a plain f32 matrix; `wb` is the
   same matrix stored as bf16, two bytes per element. Exactly one is
   non-NULL. A bf16 row is widened into g_bfrow first and the body below
   then runs on it unchanged - which is what makes the two paths produce
   identical bits rather than merely close ones. The alternative, a
   second loop that widens inside the dot product, would have to
   reproduce this one's eight-accumulator order exactly and would be a
   second copy of the trickiest arithmetic in the file.
 *
 * The widening is `bits << 16`, the same operation model.c used to
 * perform at load time, so the numbers reaching the accumulators are
 * the ones a f32-expanded tensor would have supplied. Assembled from
 * two bytes by name: correct on a big-endian host, and needs no
 * alignment past 2. */
static float g_bfrow[LZ_MM_WIDEN_MAX];

static void matmul_f32_rows(float *o, const float *x, const float *w,
                            const unsigned char *wb,
                            int in_dim, int out_dim) {    int i;
    const float *row = w;
    /* THE TIER TEST IS LOOP-INVARIANT AND WAS BEING REDONE PER ROW.
       Measured with build/dispatch_hotness.sh: 835,840 executions over
       eight tokens, 73% of every kernel-tier dispatch in the engine -
       far more than any other site, and all of it re-deciding something
       that cannot change inside this loop.

       gcc cannot hoist it. `g_kernel` is a global and the loop body
       calls lz_matmul_row_sse / lz_matmul_row_arm_asm, so the compiler
       must assume the call may have written it and reloads it every
       iteration. Nothing here does write it - no lz_kernel_select on
       any path below - so the hoist is safe in fact even though it is
       not provable from the types.

       Note what this is NOT: a change of dispatch mechanism. A
       function-pointer table would not help here, because the cost is
       the RELOAD, not the branch; an indirect call per row would be
       worse than a perfectly-predicted test on a local. */
#if defined(LZ_MATMUL_F32_SSE_EXTERN)
    const int use_sse = ((g_kernel == LZ_KERNEL_SSE2
                          || g_kernel == LZ_KERNEL_SSE) && in_dim >= 8);
#endif /* LZ_MATMUL_F32_SSE_EXTERN */
#if defined(LZ_ARM_ASM_EXTERN)
    const int use_arm = (g_kernel == LZ_KERNEL_ARM_ASM && in_dim >= 8);
#endif /* LZ_ARM_ASM_EXTERN */
    /* The hottest loop in the whole forward, ~750M MACs per token.

       8 independent accumulators here are not for loop unrolling but
       to break the dependency chain: a single accumulator makes every
       iteration wait for the previous float add to retire (~4 cycles
       latency on Zen 3), pinning throughput at 2 flops/cycle.
       Measured: single accumulator 9.1 GFLOP/s, eight accumulators
       19.6 GFLOP/s, while L1-vs-memory for weights makes almost no
       difference - the bottleneck is the latency chain, not bandwidth.

       Note the summation order therefore changes, giving ~1e-7-level
       float differences. Differential tests use tolerances at the
       same order as the reference implementation; this change does not
       break them. */
    for (i = 0; i < out_dim; i++, row += (wb ? 0 : in_dim)) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        float sum;
        int j = 0;
        if (wb) {
            const unsigned char *p = wb + (size_t)i * in_dim * 2;
            int k;
            for (k = 0; k < in_dim; k++) {
                uint32_t u = ((uint32_t)p[k * 2 + 1] << 24) |
                             ((uint32_t)p[k * 2] << 16);
                memcpy(&g_bfrow[k], &u, sizeof(float));
            }
            row = g_bfrow;
        }
#if defined(LZ_MATMUL_F32_SSE_EXTERN)
        /* The eight accumulators above ARE two SSE vectors - lane k of
           them accumulates exactly the sequence scalar a_k does, in the
           same order, so this is bit-identical rather than a
           re-association. Selected by tier, like every other kernel: a
           --kernel ref run must reach the loop below or the arm this is
           compared against would be this.

           SSE1, not SSE2: mulps and addps are SSE1's, and every SSE2
           machine has them - so one kernel serves both tiers and the
           SSE2 column holds the same body rather than a copy of it.
           No CPUID in the test: lz_kernel_select clamps both of these
           values to what CPUID found, so reaching either one is already
           the answer, and reading CPUID here would only ask it again. */
        if (use_sse) {
            float acc8[8];
            j = lz_matmul_row_sse(row, x, in_dim, acc8);
            a0 = acc8[0]; a1 = acc8[1]; a2 = acc8[2]; a3 = acc8[3];
            a4 = acc8[4]; a5 = acc8[5]; a6 = acc8[6]; a7 = acc8[7];
        }
#endif /* LZ_MATMUL_F32_SSE_EXTERN */
#if defined(LZ_ARM_ASM_EXTERN)
        /* Same shape, same contract - the kernel INITIALISES acc8, so
           the eight scalars above are overwritten rather than added to,
           and lane k still accumulates the sequence scalar a_k does.
           On a target with no FPU each of these MACs is two library
           calls; the kernel fuses them and returns where it stopped.
           Selected by tier for the reason the SSE arm gives. */
        if (use_arm) {
            float acc8[8];
            j = lz_matmul_row_arm_asm(row, x, in_dim, acc8);
            a0 = acc8[0]; a1 = acc8[1]; a2 = acc8[2]; a3 = acc8[3];
            a4 = acc8[4]; a5 = acc8[5]; a6 = acc8[6]; a7 = acc8[7];
        }
#endif /* LZ_ARM_ASM_EXTERN */
        for (; j + 7 < in_dim; j += 8) {
            a0 += row[j]     * x[j];
            a1 += row[j + 1] * x[j + 1];
            a2 += row[j + 2] * x[j + 2];
            a3 += row[j + 3] * x[j + 3];
            a4 += row[j + 4] * x[j + 4];
            a5 += row[j + 5] * x[j + 5];
            a6 += row[j + 6] * x[j + 6];
            a7 += row[j + 7] * x[j + 7];
        }
        /* pair-wise merge rather than sequential adds, reducing rounding error in these last steps */
        sum = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
        for (; j < in_dim; j++) sum += row[j] * x[j];
        o[i] = sum;
    }
}

void lz_matmul(float *o, const float *x, const float *w,
               int in_dim, int out_dim) {
    matmul_f32_rows(o, x, w, NULL, in_dim, out_dim);
}

/* The bf16 twin. in_dim is bounded by the widening scratch, the same
   bound every quantized kernel in this file carries; past it the caller
   falls back, so this never truncates silently. */
void lz_matmul_bf16(float *o, const float *x, const unsigned char *wb,
                    int in_dim, int out_dim) {
    matmul_f32_rows(o, x, NULL, wb, in_dim, out_dim);
}

