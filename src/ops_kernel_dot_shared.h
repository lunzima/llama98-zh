/* Per-format row-loop macros for the matmul kernels, shared between
   src/ops.c (gcc's row_*_intrin functions, which never move) and
   whichever of src/ops_mmx.c/src/ops_sse2.c hosts that format's Watcom
   row_*_asm function.
   Each format has its OWN macro family under its own LZ_<FORMAT>_*
   name (LZ_Q8_ACC vs LZ_Q41_ACC etc. - a pre-existing pattern, not
   something new) because the row loop's shape - which pointer
   arithmetic, which stride - differs per format; nothing here is
   generic across formats the way LZ_WSUM_CHUNK/LZ_SLOT_NEXT are.

   Needs its own include guard, unlike ops_kernel_amax.h/norm.h/
   q8round.h/p2.h/dot.h: those are included exactly once (by ops.c) and
   rely on that; this one is included by up to three different files -
   same reason src/ops_kernel_shared.h has one.

   Each macro assumes local variables already in scope at its call
   site (g, wr, wend, xwt, acc, and the row-context pointer c) - the
   same convention every
   other LZ_*_PAIR/LZ_*_ACC macro in this project already uses; not
   re-derived here. */
#ifndef LZ_OPS_KERNEL_DOT_SHARED_H
#define LZ_OPS_KERNEL_DOT_SHARED_H

/* ---- Q8_0 --------------------------------------------------------------
   Both macros used by gcc's row_q8_mmx_intrin (src/ops.c, unmoved) AND
   by Watcom's row_q8_mmx_asm/row_q8_sse2_asm (src/ops_mmx.c/
   src/ops_sse2.c) - guarded the same way ops.c's own Q8 section always
   was, so each including TU gets exactly the macros it is allowed to
   use. */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)

/* Prefetch distance clamped by a pointer upper bound once: PIII's
   prefetchnta is harmless out of bounds (pure hint, no fault), but
   PII's dummy load is a REAL access - reading an unmapped page
   segfaults. One compare buys safety for the whole path; the branch is
   highly predictable. */
#define LZ_Q8_ACC(DOT32, PF)                                        \
    for (g = 0; g < c->nb; g++) {                                      \
        const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;     \
        if (pf_ < wend) PF(pf_);                                    \
        acc[g] = DOT32(wr + (size_t)g * 32, xwt + (size_t)g * 32);  \
    }

/* Kernel choice sits at the ROW level; the inner 32-element kernel
   stays a direct call. Prefetch tier is orthogonal and resolved here
   too, which is why this expands the loop body four times rather than
   testing per sub-block. */
#define LZ_Q8_PFSEL(DOT32)                                              \
    do {                                                                \
        int pf_m_ = lz_prefetch_mode();                                 \
        if (pf_m_ == LZ_PF_AMD)       { LZ_Q8_ACC(DOT32, LZ_PFI_AMD); }  \
        else if (pf_m_ == LZ_PF_LOAD) { LZ_Q8_ACC(DOT32, LZ_PFI_LOAD); } \
        else if (pf_m_ == LZ_PF_NONE) { LZ_Q8_ACC(DOT32, LZ_PFI_NONE); } \
        else                          { LZ_Q8_ACC(DOT32, LZ_PFI_NTA); }  \
    } while (0)

#if defined(__WATCOMC__) || (defined(LZ_G128_GCC) && LZ_G128_GCC)
/* One group (128 elements) at a time when nb is a multiple of 4: the
   fixed overhead - mask construction, prologue/epilogue - is amortized
   from once per 32 elements to once per 128. Integer results are
   bit-identical to the per-32 version. All current tensors have in_dim
   a multiple of 128 (512/768/1024/3072). Watcom always carries the
   kernels; gcc carries them behind the LZ_G128_GCC knob
   (ops_kernel_shared.h), which defaults on to match Watcom's shape. */
#define LZ_Q8_GROUP4(DOT128)                                            \
    do {                                                                \
        for (g = 0; g < c->nb; g += 4) {                                   \
            const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;     \
            if (pf_ < wend) {                                           \
                int pf_m_ = lz_prefetch_mode();                         \
                if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);          \
                else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);         \
                else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);         \
                else                          LZ_PFI_NTA(pf_);          \
            }                                                           \
            DOT128(wr + (size_t)g * 32, xwt + (size_t)g * 32, acc + g); \
        }                                                               \
    } while (0)
#endif /* __WATCOMC__ || LZ_G128_GCC */

#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */

/* ---- Q4_1 -------------------------------------------------------------- */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)

#define LZ_Q41_ACC(DOT32, PF)                                          \
    for (s = 0; s < c->nb; s++) {                                         \
        const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16; \
        if (pf_ < wend) PF(pf_);                                       \
        acc[s] = DOT32(wn + (size_t)s * 16, xwt + (size_t)s * 32);     \
    }

#define LZ_Q41_PFSEL(DOT32)                                              \
    do {                                                                 \
        int pf_m_ = lz_prefetch_mode();                                  \
        if (pf_m_ == LZ_PF_AMD)       { LZ_Q41_ACC(DOT32, LZ_PFI_AMD); }  \
        else if (pf_m_ == LZ_PF_LOAD) { LZ_Q41_ACC(DOT32, LZ_PFI_LOAD); } \
        else if (pf_m_ == LZ_PF_NONE) { LZ_Q41_ACC(DOT32, LZ_PFI_NONE); } \
        else                          { LZ_Q41_ACC(DOT32, LZ_PFI_NTA); }  \
    } while (0)

#if defined(__WATCOMC__) || (defined(LZ_G128_GCC) && LZ_G128_GCC)
/* Same 128-element step as Q8_0: fixed overhead amortized from once per
   32 elements to once per 128, integer results bit-identical. */
#define LZ_Q41_GROUP4(DOT128)                                              \
    do {                                                                   \
        for (s = 0; s < c->nb; s += 4) {                                      \
            const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16; \
            if (pf_ < wend) {                                              \
                int pf_m_ = lz_prefetch_mode();                            \
                if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);             \
                else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);            \
                else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);            \
                else                          LZ_PFI_NTA(pf_);             \
            }                                                              \
            DOT128(wn + (size_t)s * 16, xwt + (size_t)s * 32, acc + s);    \
        }                                                                  \
    } while (0)
#endif /* __WATCOMC__ || LZ_G128_GCC */

#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */

/* ---- Q6_1 --------------------------------------------------------------
   Two weight streams (4-bit plane wn, 2-bit plane w2), so its own
   prefetch macro covers both bounds - see the original comment at this
   macro's old home in src/ops.c: the 2-bit plane is a real, narrower
   hot stream, not belt-and-braces. */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)

#define LZ_Q61_PF(S)                                                    \
    do {                                                                \
        const unsigned char *pf_ = wn + (size_t)((S) + LZ_PF_DIST) * 16; \
        const unsigned char *pf2_ = w2 + (size_t)((S) + LZ_PF_DIST) * 8; \
        int pf_m_ = lz_prefetch_mode();                                 \
        if (pf_ < wend) {                                               \
            if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);              \
            else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);             \
            else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);             \
            else                          LZ_PFI_NTA(pf_);              \
        }                                                               \
        if (pf2_ < wend2) {                                             \
            if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf2_);             \
            else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf2_);            \
            else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf2_);            \
            else                          LZ_PFI_NTA(pf2_);             \
        }                                                               \
    } while (0)

#define LZ_Q61_ACC(DOT32)                                               \
    for (s = 0; s < c->nb; s++) {                                          \
        LZ_Q61_PF(s);                                                   \
        acc[s] = DOT32(wn + (size_t)s * 16, w2 + (size_t)s * 8,         \
                       xwt + (size_t)s * 32);                           \
    }

#if defined(__WATCOMC__) || (defined(LZ_G128_GCC) && LZ_G128_GCC)
#define LZ_Q61_GROUP4(DOT128)                                           \
    do {                                                                \
        for (s = 0; s < c->nb; s += 4) {                                   \
            LZ_Q61_PF(s);                                               \
            DOT128(wn + (size_t)s * 16, w2 + (size_t)s * 8,             \
                   xwt + (size_t)s * 32, acc + s);                      \
        }                                                               \
    } while (0)
#endif /* __WATCOMC__ || LZ_G128_GCC */

#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */

/* ---- Q16_0 --------------------------------------------------------------
   int16 weights: a 32-element sub-block is 64 bytes (two cache lines),
   twice Q8_0's, so the prefetch distance lands twice as far ahead in
   bytes for the same sub-block count - see the original comment at
   this macro's old home in src/ops.c. */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)

#define LZ_Q16_PF(G)                                                      \
    do {                                                                  \
        const int16_t *pf_ = wr + (size_t)((G) + LZ_PF_DIST) * 32;        \
        if (pf_ < wend) {                                                 \
            int pf_m_ = lz_prefetch_mode();                               \
            if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);                \
            else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);               \
            else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);               \
            else                          LZ_PFI_NTA(pf_);                \
        }                                                                 \
    } while (0)

#define LZ_Q16_ACC(DOT32)                                                 \
    for (g = 0; g < c->nb; g++) {                                            \
        LZ_Q16_PF(g);                                                     \
        acc[g] = DOT32(wr + (size_t)g * 32, xwt + (size_t)g * 32);        \
    }

#if defined(__WATCOMC__) || (defined(LZ_G128_GCC) && LZ_G128_GCC)
#define LZ_Q16_GROUP4(DOT128)                                             \
    do {                                                                  \
        for (g = 0; g < c->nb; g += 4) {                                     \
            LZ_Q16_PF(g);                                                 \
            DOT128(wr + (size_t)g * 32, xwt + (size_t)g * 32, acc + g);   \
        }                                                                 \
    } while (0)
#endif /* __WATCOMC__ || LZ_G128_GCC */

#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */

#endif /* LZ_OPS_KERNEL_DOT_SHARED_H */
