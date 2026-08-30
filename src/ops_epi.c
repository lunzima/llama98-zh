#include "lz_int.h"   /* lz_i64/lz_u64 and the width types:
                         Visual C++ 4.0 has neither <stdint.h> nor
                         `long long`, and 236 sites deciding that for
                         themselves is the shape one-authority forbids */
#include <stddef.h>   /* size_t: used a lot here for pointer offsets; math.h does not guarantee it */
#include <float.h>    /* FLT_MAX: overflow guard for quant reciprocal */
#include <math.h>
#include <stdlib.h>   /* malloc/free: the epilogue's derived scale plane
                         is the only allocation in this file. Watcom
                         reports W131 without it and -we makes that an
                         error; gcc picked the prototypes up from
                         elsewhere and said nothing. */
#include <string.h>   /* memcpy: unaligned loads in the MMX/SSE2 kernels */

#include "ops.h"
#include "err.h"
#include "cpucheck.h"   /* LZ_CPUID1_EDX_AUX - shared CPUID leaf 1 EDX asm */
#include "mmx_compat.h" /* _mm_empty(): called from the architecture-neutral
                           dispatch paths too, so it must be declared on
                           targets that have no MMX at all - see the header */
#include "ops_mmx.h"    /* declarations for src/ops_mmx.c's functions - the
                           %mm-touching kernels are defined there. Unconditional:
                           the macros and declarations inside are themselves
                           guarded on LZ_MMX_TU, a build flag (-D, from the
                           Makefile) rather than on __MMX__, which this file
                           must never be compiled with. */
#include "ops_sse.h"    /* the SSE1-on-%xmm half of src/ops_mmx_sse.c. A
                           separate include from ops_mmx.h on purpose: those
                           kernels need an SSE1 machine, and an include line
                           saying "mmx" would not say so. */
#include "ops_sse2.h"   /* declarations for src/ops_sse2.c's functions - the
                           %xmm-touching kernels are defined there. Same
                           LZ_SSE2_TU/__SSE2__ contract, one ISA tier up. */
#include "ops_kernel_shared.h" /* lz_i32f, LZ_WSUM_CHUNK, LZ_SLOT_NEXT,
                           gdn_tail_row, p2_shift_of - shared with the
                           suffixed TUs; included this early because lz_i32f's
                           first use is well before any of the ops_kernel_*.h
                           #includes below. */

/* LZ_GDN_FIXED and its rationale are in ops.h. */

#include "ops_quant.h"   /* pow2f, q8_round, q8_amax, sigmoid_q15,
                           sigmoid_q15_t, lz_sig_q15_fold,
                           lz_sig_q15_t_offset, norm_ss_fixed,
                           lz_exp_float_body, lz_rsqrt_float_body,
                           lz_exp_fixed, lz_rsqrt_fixed,
                           LZ_Q8_MIN_SCALE_F, LZ_Q8_INV127, LZ_Q15_INV */
#include "ops_sched.h"   /* lz_q8r_tier, q8r_have_simd */
#include "ops_matmul.h"  /* matmul impls, LZ_ROW_* tables, epilogue fns */
#include "ops_t2_arm.h"  /* LZ_T2_ARM_ASM_EXTERN */
#include "ops_arm.h"     /* LZ_ARM_ASM_EXTERN */

int lz_epi_ready(const LZTensor *w) { return w && w->sq && w->sexp; }
int lz_epi_zero_ready(const LZTensor *w) { return w && w->zq && w->zexp; }

/* Only for the half-built cases below; lz_t_free owns the normal path. */
static void epi_drop(LZTensor *w) {
    free(w->sq);   w->sq = NULL;
    free(w->sexp); w->sexp = NULL;
}

static void epi_drop_zero(LZTensor *w) {
    free(w->zq);   w->zq = NULL;
    free(w->zexp); w->zexp = NULL;
}

/* The exponent search in epi_prep_plane only ever raises e from 0, so a
   plane already reaching 32767 cannot be represented and would clamp
   every group in that row to +-32767 in silence. Asked over the WHOLE
   plane before anything is allocated, so a tensor that fails simply
   never gets a plane, keeps the float epilogue, and does not re-attempt
   the build on every later call. Not reachable on either plane of a
   real checkpoint (measured on kmr20: max |scale| 0.0198, max |zero|
   0.512), which is exactly why it has to be a check and not a comment. */
static int epi_plane_fits(const float *src, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        float a = src[i] < 0.0f ? -src[i] : src[i];
        if (a >= 32767.0f) return 0;
    }
    return 1;
}

/* One row-plane of `ngw` floats -> int16 mantissas plus one shared
   integer exponent for the row. The scale plane (lz_epi_prep) and the
   zero plane (lz_epi_prep_zero) are the same decomposition applied to
   two different tensor fields, so they share this loop rather than
   carrying two conventions that can drift apart. */
static void epi_prep_plane(const float *src, int16_t *out, signed char *ex,
                           int ngw, int rows) {
    int row, g;
    const float *ws = src;
    int16_t *o = out;
    for (row = 0; row < rows; row++, ws += ngw, o += ngw) {
        float amax = 0.0f, se;
        int e = 0;
        for (g = 0; g < ngw; g++) {
            float a = ws[g] < 0.0f ? -ws[g] : ws[g];
            if (a > amax) amax = a;
        }
        if (amax <= 0.0f) {
            for (g = 0; g < ngw; g++) o[g] = 0;
            ex[row] = 0;
            continue;
        }
        /* The exponent is a read of amax, not a search for it.
           amax = M x 2^E with M in [1,2), so the condition
           `amax * 2^e < 32767` turns on E + e: below 14 it always
           holds, at 15 it never does, and at exactly 14 it reduces to
           M < 32767/16384, which is a comparison of the mantissa field
           against 0x7FFE00. Hence a bias subtract, one correction, and
           the clamp from starting at 0 and stopping at
           100. Subnormals fall out: their biased exponent is 0, giving
           141, which clamps to 100 - the same place the loop capped.
           Verified against the loop over all 2^32 bit patterns, not a
           sample; the two can only part at a boundary a sample would
           have to already know about to hit. */
        {
            union { float f; uint32_t u; } ab;
            ab.f = amax;
            e = 141 - (int)((ab.u >> 23) & 0xFFu);
            if ((ab.u & 0x7FFFFFu) >= 0x7FFE00u) e--;
            if (e < 0) e = 0;
            if (e > 100) e = 100;
        }
        se = pow2f(e);
        for (g = 0; g < ngw; g++) {
            float t = ws[g] * se;
            int q = q8_round(t);
            if (q >  32767) q =  32767;
            if (q < -32767) q = -32767;
            o[g] = (int16_t)q;
        }
        ex[row] = (signed char)e;
    }
}

/* One shared exponent per ROW, not per tensor: .prof/wepirange.c
   measured the per-row spread of the weight scales at 99.6% within 8x,
   while across rows it is far wider. A per-tensor exponent would spend
   that whole range on every row. */
void lz_epi_prep(LZTensor *w, int in_dim) {
    int ngw, rows;
    if (!w || !w->scale || w->gs <= 0 || in_dim <= 0) return;
    if (w->sq) return;                       /* idempotent */
    if (in_dim % w->gs) return;
    ngw = (unsigned)in_dim / (unsigned)w->gs;
    if (in_dim / 32 > LZ_EPI_MAX_NG) return; /* outside the derived bound */
    rows = w->n / in_dim;
    if (rows <= 0) return;
    if (!epi_plane_fits(w->scale, (size_t)ngw * (size_t)rows)) return;
    w->sq   = (int16_t *)malloc((size_t)ngw * (size_t)rows * sizeof(int16_t));
    w->sexp = (signed char *)malloc((size_t)rows);
    if (!w->sq || !w->sexp) { epi_drop(w); return; }
    w->sq_row = ngw;
    epi_prep_plane(w->scale, w->sq, w->sexp, ngw, rows);
}

/* The zero plane's counterpart, same shape and the same shared loop.
   Only Q4_1/Q6_1 reach it: T2 carries no `zero` field (its coefficient
   is -scale, already in sq/sexp) and Q8_0/Q16_0 have no zero term, so
   both leave here immediately and lz_epi_zero_ready stays false for
   them. */
void lz_epi_prep_zero(LZTensor *w, int in_dim) {
    int ngw, rows;
    if (!w || !w->zero || w->gs <= 0 || in_dim <= 0) return;
    if (w->zq) return;                       /* idempotent */
    if (in_dim % w->gs) return;
    ngw = (unsigned)in_dim / (unsigned)w->gs;
    if (in_dim / 32 > LZ_EPI_MAX_NG) return; /* outside the derived bound */
    rows = w->n / in_dim;
    if (rows <= 0) return;
    if (!epi_plane_fits(w->zero, (size_t)ngw * (size_t)rows)) return;
    w->zq   = (int16_t *)malloc((size_t)ngw * (size_t)rows * sizeof(int16_t));
    w->zexp = (signed char *)malloc((size_t)rows);
    if (!w->zq || !w->zexp) { epi_drop_zero(w); return; }
    w->sq_row = ngw;                  /* same stride; either prep may be first */
    epi_prep_plane(w->zero, w->zq, w->zexp, ngw, rows);
}

/* int32 x int32 -> int64, which C cannot ask for: `(lz_i64)a *
   (lz_i64)b` is 64x64 by the standard and Watcom emits `call __I8M`,
   2.75x the cost of the one `imul r/m32` x86 has had since the 386.

   In both of the kernels below, ebp is pushed and popped rather
   than declared modified - it is the frame pointer, and the rule is
   that modify tells the truth. imul owns edx:eax, so the two
   accumulator halves are ebx and ebp.

   Both entries read the tensor's int16 plane directly: the accumulator
   bound is n products of an int32 accumulator by an int16, summed at 64
   bits, which is the bound ops_matmul.c's LZ_EPI_MAX_NG comments cite. */

/* Counts calls that left the scalar body for a vector one, whichever
   tier this build has. Read by cli_main's --debug line; the dispatch
   below says why a counter is needed when a byte-identity gate already
   passes.

   OUTSIDE the toolchain split, and defined even where nothing can
   increment it. cli_main.c reads it unconditionally, so a definition
   under one arm is a link error under the other; and zero is the true
   answer for a build with no epi kernel, which the km_reg row records
   as a todo rather than as silence. One counter and not one per tier:
   no build has two of these bodies, and --debug prints the tier on the
   line above. */
lz_i64 lz_debug_epi_kern = 0;

#if defined(__WATCOMC__)
/* int16-multiplier twin, for a caller whose m plane is int16 and who
   would otherwise widen it into a scratch buffer first. epi_q41_join
   was that caller: its widening loop was the third-hottest branch in the
   engine (gcov -b -c, kmr20/zh, 603 tokens: 145,723,392 evaluations,
   minority arm 41.9%, because ng is 2 and a two-iteration loop's
   back-edge is a coinflip to a not-taken default), and it also cost
   241k copies a token and a global buffer that existed only to hold
   them.

   Identical values, not merely close: the widen was exact and the
   accumulator is int64 either way, so this computes the same sum the
   copy-then-mac did. Verified byte for byte on both fixtures.

   The asm differs from its int32 sibling in two instructions - movsx
   instead of mov, and add edi,2 instead of 4. The label is L1i rather
   than L1 because two pragma-aux bodies in one translation unit must
   not both define L1. */
/* One GROUP of the fused epilogue MAC: sum a[i] * (xq[i] * w) over n
   elements, w being the group's shared int16 weight. Fusing the two
   means epi_fixed_raw needs no g_epi_m plane - the products are
   consumed as they are formed, so nothing has to be stored.

   w RIDES THE STACK because the body has no register left for it: esi
   (a), edi (xq), ecx (the counter), ebx/ebp (the int64 accumulator) and
   edx:eax (imul's fixed pair) is already seven. The 386's two-operand
   `imul r32, r/m32` takes a memory operand and leaves edx alone, so w
   stays where the caller put it: after `push ebp` the frame is
   [esp]=old ebp, [esp+4]=w. NO RETURN ADDRESS between them - wcc386
   INLINES a pragma-aux body, so [esp+8] would read a saved register.

   `parm caller [esi] [edi] [ecx]` - three registers for four arguments,
   with the cleanup named. Both halves matter and neither errors when
   wrong: an explicit empty slot (`[esi] [edi] [ecx] []`) makes wcc386
   hand every register the LAST argument, and omitting `caller` leaves
   the callee's plain `ret` with the stack argument still on it.

   Same per-element product and the same ascending order as the flat
   loop it replaces. NOT the same as hoisting w out of the sum - that
   reassociates the reduction and changes what can overflow; this does
   not. */
lz_i64 lz_epi_mac_grp(const int32_t *a, const int16_t *xq, int n, int32_t w);
#pragma aux lz_epi_mac_grp =            \
    "push ebp"                          \
    "xor ebx,ebx"                       \
    "xor ebp,ebp"                       \
    "Lgrp:"                             \
    "movsx eax,word ptr [edi]"          \
    "imul eax,dword ptr [esp+4]"        \
    "mov edx,[esi]"                     \
    "imul edx"                          \
    "add ebx,eax"                       \
    "adc ebp,edx"                       \
    "add esi,4"                         \
    "add edi,2"                         \
    "dec ecx"                           \
    "jnz Lgrp"                          \
    "mov eax,ebx"                       \
    "mov edx,ebp"                       \
    "pop ebp"                           \
    parm caller [esi] [edi] [ecx]       \
    value [edx eax]                     \
    modify [eax ebx ecx edx esi edi 8087];
/* One-operand imul IS a signed 32x32 -> 64 with the result in edx:eax,
   so this needs none of the 16-bit splitting the SSE2 twin does. Eight
   instructions an element, which is why that twin's win is in the
   branch count and not the instruction count. */
lz_i64 lz_epi_mac_i16_asm(const int32_t *a, const int16_t *m, int n);
#pragma aux lz_epi_mac_i16_asm =        \
    "push ebp"                          \
    "xor ebx,ebx"                       \
    "xor ebp,ebp"                       \
    "L1i:"                              \
    "mov eax,[esi]"                     \
    "movsx edx,word ptr [edi]"          \
    "imul edx"                          \
    "add ebx,eax"                       \
    "adc ebp,edx"                       \
    "add esi,4"                         \
    "add edi,2"                         \
    "dec ecx"                           \
    "jnz L1i"                           \
    "mov eax,ebx"                       \
    "mov edx,ebp"                       \
    "pop ebp"                           \
    parm [esi] [edi] [ecx]              \
    value [edx eax]                     \
    modify [eax ebx ecx edx esi edi 8087];

/* The tier dispatch, which the pragma above cannot carry: a #pragma aux
   body expands at the call site and has no C in it. Without this the
   Watcom SSE2 kernel is a registered cell nobody reaches - the matrix
   would report it present, because the symbol is, and no call would
   ever arrive.

   n <= 0 is answered here rather than in either body. The C loop in the
   gcc branch below returns 0 for it by not iterating; `dec ecx / jnz`
   with ecx zero goes round 2^32 times. */
static lz_i64 lz_epi_mac_i16(const int32_t *a, const int16_t *m, int n) {
    if (n <= 0) return 0;
#if defined(LZ_EPI_SSE2_EXTERN)
    /* Same lazy guard the other dispatch sites carry: the GUI never
       calls lz_kernel_select, so g_kernel reaches its value here. */
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_SSE2) {
        lz_debug_epi_kern++;
        return lz_epi_mac_i16_sse2(a, m, n);
    }
#endif /* LZ_EPI_SSE2_EXTERN */
    return lz_epi_mac_i16_asm(a, m, n);
}
#else
/* See the Watcom twin above for the register argument this shape
   satisfies; here it is just the loop. */
static lz_i64 lz_epi_mac_grp(const int32_t *a, const int16_t *xq,
                                int n, int32_t w) {
    lz_i64 s = 0;
    int i;
    for (i = 0; i < n; i++)
        s += (lz_i64)a[i] * (lz_i64)((int32_t)xq[i] * w);
    return s;
}
/* See the Watcom twin above for why this exists and what it replaced.

   The kernel is selected at the TIER, not by an ifdef alone: the C loop
   stays reachable so --kernel ref and --kernel mmx still exercise it,
   which is what the identity gate compares against. */
static lz_i64 lz_epi_mac_i16(const int32_t *a, const int16_t *m, int n) {
    lz_i64 s = 0;
    int g;
    /* BELOW FOUR ELEMENTS THERE IS NO KERNEL TO DISPATCH TO, so do not
       ask. Both SIMD wrappers compute their block count as `n >> 2` and
       skip the vector body entirely when it is zero - falling through to
       a scalar tail that is the same loop as the one at the bottom of
       this function. For n < 4 the tier tests, the counter and the call
       are pure overhead in front of code that was going to run anyway.

       This is the engine's single hottest dispatch:
       build/dispatch_hotness.sh measures 811,008 executions on kmr20
       over eight tokens, where ng is 2 - i.e. EVERY call took this
       shape. It is one call site (epi_q41_join) with n = ng, so which
       side of the threshold it lands on is a property of the model's
       group count, not of the input.

       Not hoisted the way the tier tests in ops_matmul.c were: this
       dispatch is the callee's own prologue and its only caller is
       itself per-row, so hoisting would mean threading a resolved
       kernel down four levels. Declining to dispatch at all is smaller
       and strictly better. */
    if (n < 4) {
        for (g = 0; g < n; g++) s += (lz_i64)a[g] * (lz_i64)m[g];
        return s;
    }
#if defined(LZ_EPI_SSE2_EXTERN) || defined(LZ_ARM_ASM_EXTERN)
    /* Same lazy guard the other dispatch sites carry: the GUI never
       calls lz_kernel_select, so g_kernel reaches its value here.

       lz_debug_epi_kern counts calls that REACH a kernel body. NO GATE
       READS IT, whatever a reader might assume from a counter that
       distinguishes "agrees" from "agrees and was reached": cli_main.c
       prints it in the run banner and that is the whole consumer. The
       count is a diagnostic for a person, and the early-out above keeps
       it honest - a call that enters a SIMD wrapper only to take its
       scalar tail does not reach a kernel body and is not counted. */
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
#endif /* LZ_EPI_SSE2_EXTERN || LZ_ARM_ASM_EXTERN */
#if defined(LZ_EPI_SSE2_EXTERN)
    if (g_kernel == LZ_KERNEL_SSE2) {
        lz_debug_epi_kern++;
        return lz_epi_mac_i16_sse2(a, m, n);
    }
#endif /* LZ_EPI_SSE2_EXTERN */
#if defined(LZ_ARM_ASM_EXTERN)
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        lz_debug_epi_kern++;
        return lz_epi_mac_i16_arm_asm(a, m, n);
    }
#endif /* LZ_ARM_ASM_EXTERN */
    for (g = 0; g < n; g++) s += (lz_i64)a[g] * (lz_i64)m[g];
    return s;
}
#endif /* __WATCOMC__ */


/* Cap for the activation pre-expansion buffers (element count). Above
   it, fall back to the non-expanded path. 4096 covers the 0.8B's max
   in_dim (intermediate 3584) and s1v2's 2048; 8KB on the stack, only
   present in SSE2 builds (the MMX tier has its own cap). */
/* These pre-expansion buffers MUST be static, not on the stack.
   8KB xw[] plus 512B acc32[] pushes the stack pointer two pages down
   on function entry, and Open Watcom's -ox includes -s (no stack
   probes); skipping the guard page reads uncommitted garbage - measured
   end-to-end logits becoming nondeterministic garbage (rms
   5418/5415/5412 drifting run to run), while -ot or -od works. Win98's
   default stack is even tighter than NT's; relying on compiler flags
   is not safe. The engine is single-threaded and non-recursive, so
   static is safe. */

/* Cap for the activation pre-expansion buffers (element count). One
   value, in ops.h, because lz_matmul_xq_nt checks it too and two copies
   of a cap are two chances for a buffer to be sized against the one
   nobody edited. */

/* ---- prefetch (lever J) ------------------------------------------------
   LZ_PF_DIST, LZ_PFI_LOAD/NTA/AMD/NONE and the lz_pf_load/lz_pf_nta/
   lz_pf_amd pragmas live in src/ops_kernel_shared.h: every format's row
   function needs them, wherever it lives, the same reasoning
   LZ_WSUM_CHUNK/LZ_SLOT_NEXT established. Full rationale (why three
   tiers, why the distance is a knob, why "no prefetch" is a real mode
   and not a fallback) lives there. */

/* LZ_SSE2_GROUP (the -DLZ_SSE2_GROUP=0 A/B control for the 128-element
   SSE2 group kernels) is in src/ops_kernel_shared.h: the row functions
   that test it live in src/ops_mmx.c/src/ops_sse2.c, and it must be
   visible wherever one of them references it, same reasoning as
   LZ_PF_DIST, the LZ_PFI_ family and LZ_PF_NONE above. */

/* ---- row-kernel tier tables (stage 3) ---------------------------------

   ONE signature for every (format x tier). A row function fills
   acc32[tk*nb + s] for ONE weight row across nt tokens and touches no
   float at all - the float work is epi_q8 / epi_q41, which stage 4
   already reduced to one copy each.

   w4/w2 are the weight row's planes (w2 is the Q6_1 2-bit plane, NULL
   elsewhere); pf_end is the prefetch upper bound - PIII's prefetchnta
   is a harmless hint out of bounds but PII's dummy load is a REAL
   access and segfaults on an unmapped page.

   The dispatch is at row level, never inside. At in_dim=1024 that is
   one indirect call per 32 kernel calls; one per 32 MACs would be the
   disaster the file's opening comment warns about. */

/* The seven variants, as table slots. A build fills at
   most three of them (gcc cannot compile #pragma aux; Watcom's _m_*
   intrinsics are 6x slower than its own x87 scalar), and REF is not
   here at all - lz_matmul_xq_nt intercepts that tier before reaching
   any impl, so that the scalar reference stays a real oracle.

   An unfilled slot is NULL and the pick falls back, loudly rather than
   silently: a tier that cannot be reached must not report itself as
   the tier that ran. That confusion is the whole reason this file is
   being restructured. */
/* The slot constants are defined near the top of the file, beside
   LZ_DEFINE_PICK, because the operator tables there are read first. */

/* ISA from the runtime tier, impl from the build - the same join
   lz_kernel_tier() reports, in one place instead of four #if mazes.

   THE SSE SLOTS ARE EMPTY BY INSTRUCTION SET, NOT BY BACKLOG, and that
   distinction is the whole reason gaps need a reason - checked against
   the actual kernels rather than reasoning about the ISA.
   A row kernel's operation sequence is pand / psrlw / punpcklbw(z,.) /
   punpckhbw(z,.) / pmaddwd / paddd. SSE1's additions to the MMX integer
   set are pshufw, pinsrw, pextrw, pmovmskb, pmulhuw, pavgb, pavgw,
   pmaxsw, pminsw, pmaxub, pminub, psadbw, maskmovq, movntq. None of
   them replaces a step or removes one:
     pshufw  only permutes existing 16-bit lanes; it cannot zero-extend
             bytes to int16, so it cannot stand in for punpck
     pextrw  is 16-bit; the horizontal reduction here is 32-bit
     pmaxsw  belongs to lz_quantize_q8, not to matmul
     movntq  would write acc32, which the float epilogue reads on the
             very next line - a non-temporal store is a pessimisation
             exactly where the data is reused immediately
   So a PIII runs the MMX kernel, and asking for SSE lands there - one
   readable line with the reasoning attached, rather than an accident of
   #if nesting. Prefetch - the one thing SSE really does add here - is
   orthogonal and lives in --prefetch. */

/* THE Q8 FLOAT EPILOGUE - one copy, shared by every kernel tier.
   accb[g] holds the g-th 32-element sub-block's integer sum SCALED BY
   256 (what part32_x16 and the MMX/asm kernels all produce); this turns
   it into one output element.

   The SSE2-intrinsics branch fills acc32 rather than fusing integer and
   float work, so it can be expressed as a row function that "only fills
   acc32" - which is what lets this epilogue stay a single shared copy.

   EVERY LINE OF ASSOCIATION ORDER HERE IS LOAD-BEARING; the order is
   what makes two builds agree bit for bit:
     - `acc * (sx * sw)`, never `(acc * sx) * sw` - one last-bit apart;
     - the r>1 path accumulates within a weight group in s order and
       multiplies by ws ONCE, with no cross-group 4-way accumulator;
     - the r==1 path reduces ((a0+a2)+(a1+a3)), matching the SSE2
       horizontal fold, not the natural ((a0+a1)+(a2+a3)).
   A 1e-6 difference here is not cosmetic: SSM state requantization
   amplifies it token by token, and eight tokens are enough to fork the
   generated text - and note that |logit diff|
   saturates and cannot rank such changes.

   `post` cancels the kernel's own scaling: 1/256 for Q8_0 (part32_x16
   and the MMX/asm kernels all scale their products by 256) and 1.0 for
   Q16_0, whose kernels do not. A power of two either way, so the cancel
   is exact, and `x * 1.0f` is bit-identical to `x` - which is what lets
   Q16_0 share this instead of carrying a second copy differing in one
   multiply. Both call sites pass a literal, so it folds away. */
/* The fixed counterpart. Same signature plus the tensor's int16 plane
   and the row, because the shared exponent is per row.

   xq16/mprod are static rather than stack (Win98 has a small stack) and sized by
   the same bound lz_epi_prep enforces, so a shape it refused cannot
   reach here. */
static int16_t g_epi_xq[LZ_BATCH_MAX][LZ_EPI_MAX_NG];
static int     g_epi_xe[LZ_BATCH_MAX];
/* The zero term's activation half in integer form (epi_zero_act_int) and
   the widened weight half epi_q41_fixed pairs it with. Same reason as
   the three above for being static rather than stack. */
static int32_t g_epi_zg[LZ_BATCH_MAX][LZ_EPI_MAX_NG];
/* g_epi_zm is gone with the widening copy it existed to hold - see
   lz_epi_mac_i16. It was duplicate memory in the strict sense: a buffer
   whose every write was read exactly once, by the next statement. */

/* All tokens packed up front, because the row loop is OUTER and the
   token loop inner - packing inside would run once per (row, token) and
   cost exactly what the epilogue it replaces costs. That mistake is
   what made the first measured candidate 1.88x instead of 32x. */
void epi_pack_act(const float *xqs, int nb, int nt) {
    int tk, g;
    const float *xsb;   /* C89: declarations before statements */
    /* One multiply per group plus the amax scan, per token. Counted so
       the census cannot report the fixed tier as free - the work moved,
       it did not vanish, and a site that bills nothing is how the old
       hand-made table lost the two heaviest ones. The amax scan is a
       comparison, not billed (same convention as the four scans B1
       leaves alone); the round itself is q8_round's magic add, so there
       is no compare left in the packing loop to bill.

       Zero multiplies now, not one per group: the pack's scale and the
       exponent search's are both powers of two, so both go through
       f32_scale_pow2 as exponent adds. The bill drops from nt*nb to
       nothing, which is the honest number and also the reason this
       comment stays - a site that bills zero because its work moved is
       one edit away from being mistaken for a site that bills zero
       because nobody wrote the bill. */
    LZ_FCX(LZ_FC_EPI, 0, 0, 0, 0, 0);
    xsb = xqs;
    for (tk = 0; tk < nt; tk++, xsb += nb) {
        /* The amax scan is q8_amax, not a hand-written loop. It was two
           __aeabi_fcmpXX per group - `xsb[g] < 0.0f` for the sign and
           `a > amax` for the max - and q8_amax already does exactly this
           in the integer domain, with a SIMD tier on top. Calling it
           removes the compares AND a fourth open-coded copy of the same
           scan. Exact either way: the maximum of magnitudes cannot
           depend on lane order or on comparing bit patterns instead of
           values (the unsigned order of magnitudes IS their numeric
           order), which is the argument lz_softmax's max scan already
           makes for its own version. */
        union { float f; uint32_t u; } ab;
        float amax;
        int e = 0;
        ab.u = q8_amax(xsb, nb);
        amax = ab.f;
        if (amax <= 0.0f) {
            for (g = 0; g < nb; g++) g_epi_xq[tk][g] = 0;
            g_epi_xe[tk] = 0;
            continue;
        }
        /* Both scalings here are by a power of two, so both are
           exponent adds rather than __aeabi_fmul - the search below
           runs up to 100 times per token and the pack once per group.
           f32_scale_pow2 declines zero/subnormal and out-of-range, and
           pow2f keeps those: amax is already known positive at this
           point, but xsb[g] being exactly zero is ordinary. */
        {
            float pr;
            while (e < 100) {
                if (!f32_scale_pow2(amax, e + 1, &pr)) pr = amax * pow2f(e + 1);
                if (!(pr < 32767.0f)) break;
                e++;
            }
        }
        for (g = 0; g < nb; g++) {
            float t, xs = xsb[g];
            int q;
            if (!f32_scale_pow2(xs, e, &t)) t = xs * pow2f(e);
            q = q8_round(t);
            if (q >  32767) q =  32767;
            if (q < -32767) q = -32767;
            g_epi_xq[tk][g] = (int16_t)q;
        }
        g_epi_xe[tk] = e;
    }
}

/* The zero term's ACTIVATION half in the integer domain - the fixed
   path's replacement for ops_matmul.c's epi_zero_act, not an addition
   to it: where that one folds `lz_i32f(Sum_k xq) * xqs[sb]` in float,
   this pairs the same 32-wide integer sums with the int16 sub-block
   scales epi_pack_act has already produced, so the whole function is
   integer and bills nothing. Runs once per token like its float twin,
   and like epi_pack_act must run BEFORE the row loop for the same
   reason (a per-row rebuild would cost what it saves).

   Its output means xgb[g] == g_epi_zg[tk][g] * 2^-g_epi_xe[tk], the
   integer counterpart of epi_zero_act's float xgb.

   BOUND, derived: lz_quantize_q8 clamps to +-127 on every tier (the
   scalar loop and the SSE1/SSE2 kernels alike, src/ops_sse2.c's
   _mm_max_epi16/_mm_min_epi16 pair), so |Sum_{k<32} xq| <= 32*127 =
   4064; epi_pack_act clamps to +-32767; one product is therefore at
   most 133,165,088 < 2^27, and r of them stay inside int32 while
   r * 133,165,088 <= 2^31-1, i.e. r <= 16 - which is LZ_EPI_MAX_R,
   enforced by the caller's predicate. MEASURED on kmr20 over both gate
   corpora: max |Sum_{k<32} xq| = 1502 of the 4064 allowed, max |zg| =
   76,280,004, 3.6% of the int32 ceiling. */
void epi_zero_act_int(const int8_t *xq, int in_dim, int ng, int r, int nt) {
    int t, g, s, k;
    const int8_t *xqt = xq;
    for (t = 0; t < nt; t++, xqt += in_dim) {
        const int16_t *xqi = g_epi_xq[t];
        for (g = 0; g < ng; g++) {
            int32_t a = 0;
            for (s = 0; s < r; s++) {
                int sb = g * r + s;
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xqt[(size_t)sb * 32 + k];
                a += sx * (int32_t)xqi[sb];
            }
            g_epi_zg[t][g] = a;
        }
    }
}

/* Shared by epi_fixed (float exit) and epi_fixed_align_i16 (aligned-
   int16 exit, below): the row's raw int64 dot product against the
   packed activation and its integer exponent, without committing to
   either exit's conversion. Splitting this out is what makes the
   int16 exit "build on top of epi_fixed" rather than a second,
   independent accumulator - both exits share the exact MAC epi_fixed
   has always used, not a reimplementation of it. */
static lz_i64 epi_fixed_raw(const int32_t *accb, const LZTensor *w,
                               int row, int tk, int ng, int r, int *e_out) {
    /* w->sq_row, not ng/r. The two are the same number - ng is the
       sub-block count and r the sub-blocks per group, so ng/r is the
       group count lz_epi_prep allocated the plane with - but this
       function runs once per output element, and build/div_hotness.sh
       counts that divide at 996,096 executions in eight tokens on
       kmr20, an order of magnitude above every other divide in the
       engine. The stride is a property of the plane, so it is read
       from the plane. */
    const int16_t *ww = w->sq + (size_t)row * (size_t)w->sq_row;
    lz_i64 s = 0;
    int g = 0, j = 0;
    /* ONE PASS over ng, not two: lz_epi_mac_grp reads the int16 plane
       and the group's scalar directly, so no widened copy is
       materialised for a MAC to read back.

       Bit-identical to the flat form by construction: the same
       per-element int32 product xq[g]*ww[g/r] and the same ascending g,
       which is NOT what hoisting ww[j] out of the inner sum would be -
       that reassociates the reduction and changes what can overflow.

       Cost, on the branch census (kmr20/zh, 603 tokens): the two flat
       loops were 1,135,588,896 evaluations each; the fused inner loop
       and this group loop are 1,159,052,832 and 173,625,408, so
       2,271,177,792 became 1,332,678,240.

       The bound is derived the way the flat loop derived it - walk g to
       ng and clip the last group - rather than assuming r divides ng.
       It does on every checkpoint here; the clip costs one compare per
       group, and assuming it would be an invariant nothing states. */
    while (g < ng) {
        int n = ng - g; if (n > r) n = r;
        s += lz_epi_mac_grp(accb + g, g_epi_xq[tk] + g, n, (int32_t)ww[j]);
        g += n; j++;
    }
    /* The exponent is per ROW, so a libm call here would run once per
       output element and cost more than the epilogue it replaces. */
    *e_out = -g_epi_xe[tk] - (int)w->sexp[row];  /* int arithmetic, no f32 work */
    return s;
}

/* How much the row kernels feeding this epilogue family scale their
   products, as a power of two. Q8_0's kernels widen int8 to int16 and
   return 256x the true product; Q16_0's weights are already int16 and
   theirs return it unscaled.

   One source for both epilogues - the float one's `post` factor and the
   fixed one's exponent fold - and an EXPONENT rather than a float
   factor. That is what makes a wrong scaling inexpressible instead of
   merely unused: epi_fixed has no `post` parameter to be handed a value
   it then ignores on one of its two exits, only this integer, which
   both exits fold into the same pow2f argument. LZ_EPI_POST_NA for a
   format with no entry here: both matmul call sites then hand the whole
   call to matmul_scalar_ref, which computes from w->scale directly and
   has no `post` to get wrong. Refusing at the FUNCTION and not at the
   fixed-epi predicate is the point - a predicate that only steers away
   from the fixed branch would leave the float one running on a scaling
   nobody derived, which is the same defect one door down. */
int lz_epi_post_e(const LZTensor *w) {
    if (!w) return LZ_EPI_POST_NA;
    switch (w->dtype) {
        case LZ_FMT_Q8_0:  return -8;
        case LZ_FMT_Q16_0: return 0;
        default:           return LZ_EPI_POST_NA;
    }
}

/* f32_scale_pow2: src/ops_kernel_shared.h. Two TUs need it - this one
   for both epilogue exits, ops_gdn.c for p2_group_norm - and a second
   copy of a bit-layout helper is what build/dedupe_gate.sh exists to
   refuse. */

/* `pe` IS A PARAMETER, not a lz_epi_post_e(w) call in here, because this
   function runs once per output element and that value depends only on
   w->dtype. It was a call plus a switch on every one of 59,904 elements
   per token on kmr20 - the dequant epilogue is the census's second
   largest row - for an answer the caller already had before it entered
   its row loop. Not a float op, so the census cannot see it; on a P5,
   which is the target, a call and a switch per element is not nothing.
   `e` cannot be hoisted the same way and is not: it is
   -g_epi_xe[tk] - w->sexp[row], genuinely per element.

   This is the minority exit. Counted on kmr20/zh, 603 tokens:
   epi_fixed_raw runs 75,080,736 times and hands its (s, e) pair to
   epi_q41_fixed 27,477,504 times, epi_fixed_align_i16 5,306,400, and
   this float conversion 8,644,608 - 11.5%. The integer exits are the
   main road already.

   What keeps those 8.6M is not a missing operator but the two call
   sites in ops_matmul.c, which write out->o, a float buffer their own
   callers read. Same shape as lz_rmsnorm's five consumers: the int
   entry exists, and the work left is in whoever is asking for a
   float. */
float epi_fixed(const int32_t *accb, const LZTensor *w, int row,
                       int tk, int ng, int r, int pe) {
    int e;
    lz_i64 s = epi_fixed_raw(accb, w, row, tk, ng, r, &e);
    /* The kernels' scaling is a power of two, so pow2f(e) * 2^pe is
       pow2f(e + pe) and folding it into the exponent drops a multiply.
       That fold is exact wherever pow2f itself is exact; the exception
       is where pow2f(e) is still a normal float but pow2f(e + pe)
       undershoots pow2f's -126 floor and returns zero outright instead
       of the gradual-underflow denormal the two-step form produces.
       g_epi_xe (epi_pack_act) and w->sexp (lz_epi_prep) are each capped
       at 100 by their own packing loops, so e can run all the way to
       -200 and cross that band - fall back to the two-step form there.
       Both exits read the same pe, so neither can carry a scaling the
       other does not. */
    if (e >= -126 && e + pe < -126) {
        LZ_FCX(LZ_FC_EPI, 2, 0, 0, 1, 0);      /* convert, two multiplies */
        return (float)s * pow2f(e) * pow2f(pe);
    }
    {
        float sf = (float)s, sc;
        if (f32_scale_pow2(sf, e + pe, &sc)) {
            LZ_FCX(LZ_FC_EPI, 0, 0, 0, 1, 0);  /* convert; the scale is integer */
            return sc;
        }
        LZ_FCX(LZ_FC_EPI, 1, 0, 0, 1, 0);      /* convert, one multiply */
        return sf * pow2f(e + pe);
    }
}

/* ---- exponent alignment primitive (int pipeline project, batch 1 #1) --
 *
 * epi_fixed's (s, e) pair means "value == s * 2^(e-8)" (see epi_fixed_raw
 * and epi_fixed above), and e is per OUTPUT ROW (w->sexp is malloc'd per
 * row - lz_epi_prep's own comment: weight scale spread is 99.6% within
 * 8x WITHIN a row but far wider ACROSS rows, which is why there is no
 * single tensor-wide exponent to begin with). A consumer that wants a
 * whole output vector as one int16 array at ONE shared exponent - namely
 * lz_causal_conv1d_step_fixed, whose `x[c] * in_scale` step already
 * assumes a single scalar in_scale for the whole call - needs those
 * per-row exponents folded down to that one target before it can take
 * the int64 output directly, instead of the float the epilogue would
 * otherwise have to materialize.
 *
 * That consumer is live: epi_q41_align_i16 (below) reaches it from the
 * Q4_1/Q6_1/T2 epilogue, which is what KDA's q/k/v projections actually
 * use. epi_fixed_align_i16, the Q8_0-shaped wrapper this was written
 * for first, is still unwired - see its own comment.
 *
 * `target_e`/`bound` are NOT invented here: for the conv family they are
 * the consumer's own already-established constants (LZ_CONV_ES,
 * lz_conv_accum_bound(k), forward.c) - a fixed power-of-two exponent
 * measured once from the real conv input distribution, not a fresh
 * per-token statistic computed from this vector's own row spread. That
 * sidesteps option (a) in docs/int-pipeline-project.md #2.1 (take
 * min(e[row]), a new probe) in favour of reusing a budget the consumer
 * already pays for - this function's job is purely the shift+round+
 * clamp arithmetic, not choosing target_e.
 *
 * shift = e - 8 + target_e is the number of bits the stored mantissa
 * moves to land at target_e: value * 2^target_e == s * 2^shift. Pure
 * integer end to end - unlike every other int-to-float conversion this
 * project has had to guard (lz_i32f, the volatile intermediate in
 * lz_quantize_q8_int64), there is no float step here for gcc/SSE and
 * Watcom/x87 to round differently around, so this primitive is
 * cross-compiler-safe by construction rather than by verification.
 *
 * Two overflow hazards, both avoided rather than checked-after-the-fact
 * because s can be as large as +-2^63 (the epilogue MAC's derivation:
 * ng * 2^27 * 2^30 stays inside int64 only up to ng==64==LZ_EPI_MAX_NG,
 * i.e. right at the boundary, and the magnitude of LLONG_MIN itself is
 * exactly 2^63):
 *   - widening (shift >= 0): `umag << shift` is only ever computed once
 *     the caller-independent bound check below has already established
 *     it cannot exceed roughly bound + 2^32, nowhere near overflow -
 *     everything past that check saturates through the ordinary clamp.
 *   - narrowing (shift < 0, down to -63): the +half rounding bias is
 *     added in unsigned 64-bit (umag <= 2^63, half <= 2^62, sum < 2^64:
 *     no wraparound, and unsigned wraparound is well-defined even where
 *     it did occur, unlike signed overflow).
 *   - narrowing past -64: `>>rs`/`1ULL<<(rs-1)` are themselves undefined
 *     once rs reaches 64, so shift<=-64 is handled by a closed form
 *     instead of widening the shift: umag*2^shift <= 2^63*2^-64 == 0.5,
 *     with equality (rounds to +-1 under half-away-from-zero) only at
 *     shift==-64 and umag==2^63 (s==LLONG_MIN exactly) - every other
 *     input in that range rounds to 0.
 * The magnitude itself is taken by converting to unsigned FIRST (well-
 * defined: the two's-complement bit pattern reinterpreted) rather than
 * negating the signed value: `-s` when s==LLONG_MIN is not representable
 * in a 64-bit signed type and is undefined behaviour - caught by this
 * primitive's own verification probe before this comment was written.
 *
 * Rounding is half-away-from-zero (add half the LSB in magnitude space,
 * then shift), matching C99 round()/roundl() - NOT q8_round's half-to-
 * even, because q8_round's magic-add trick is a float-domain optimization
 * for x87's slow (int) cast (ops_quant.c's own comment) and has nothing
 * to reuse in a pure-integer shift. The two conventions only disagree on
 * exact tie inputs, which quantization noise already dominates (q8_round's
 * own comment: 254x).
 *
 * Verified against an independent long-double reference (long double's
 * 64-bit mantissa on x87 holds any int64 exactly, so `(long double)s *
 * powl(2, shift)` is exact with no rounding of its own to disagree
 * with) over a sweep spanning the full domain this function accepts,
 * including the LLONG_MIN/shift==-64 corner above -
 * .prof/verify_epi_align.c, 1,647,376 cases, 0 mismatches. */
static int32_t epi_align_i16(lz_i64 s, int e, int target_e, int bound) {
    lz_u64 umag;
    lz_i64 r;
    int neg, shift;
    if (s == 0) return 0;
    neg = (s < 0);
    umag = neg ? (lz_u64)0 - (lz_u64)s
               : (lz_u64)s;
    shift = e - 8 + target_e;
    if (shift > 32) shift = 32;    /* umag<<32 already dwarfs any real bound */
    if (shift >= 0) {
        lz_u64 capped = (lz_u64)bound >> shift; /* shift<=32: safe */
        if (umag > capped + 1)
            r = (lz_i64)bound + 1;          /* clamps below regardless */
        else
            r = (lz_i64)(umag << shift);    /* <= (capped+1)<<shift <=
                                                   bound+2^32: far inside
                                                   unsigned 64 bits */
    } else {
        r = (lz_i64)epi_shr_half_u(umag, -shift);
    }
    if (r > bound) r = bound;
    return (int32_t)(neg ? -r : r);
}

/* Epilogue-shaped wrapper: same call signature as epi_fixed (drop-in at
   a matmul row loop's call site) but exits through epi_align_i16 instead
   of a float. Serves both symmetric formats: matmul_q16_impl (q16_0's
   four exits) and matmul_q8_impl (the MoE latent). The candidate
   producers it was written for first, kda_q/k/v_proj, are q6_1 on the
   checkpoints this project measures, so epi_fixed never runs for them
   and they reach epi_align_i16 through epi_q41_align_i16 instead.

   epi_align_i16 reads its (s, e) pair as `s * 2^(e-8)`, the convention
   from when Q8_0 was the only format that could reach it and its x256
   was still uncancelled. The value here is `s * 2^(e + pe)`, so the
   pair handed over is (s, e + pe + 8) - not a fudge factor, the same
   quantity written in the primitive's units, exactly as
   epi_q41_align_i16's `target + 8` is. Taking pe from lz_epi_post_e
   rather than assuming Q8_0's is what keeps this exit clear of the
   256x error a hardcoded `post` makes possible. */
int32_t epi_fixed_align_i16(const int32_t *accb, const LZTensor *w, int row,
                             int tk, int ng, int r, int target_e, int bound) {
    int e;
    lz_i64 s = epi_fixed_raw(accb, w, row, tk, ng, r, &e);
    return epi_align_i16(s, e + lz_epi_post_e(w) + 8, target_e, bound);
}

/* ---- SwiGLU in the integer domain (int-pipeline milestone 8) ---------
 *
 * silu(p)*g == p * sigmoid(p) * g, and with p and g arriving as int16 at
 * fixed exponents (lz_matmul_xq_nt_i16) every factor is already an
 * integer: sigmoid_q15_i takes p's bits directly and returns Q15, so the
 * whole product can be done with two int32 multiplies and two shifts and
 * no float appears anywhere between the two matmuls and the requantize.
 *
 * Lives here, next to epi_align_i16, because it is that primitive's
 * first consumer outside a matmul epilogue - the final shift+round+clamp
 * IS epi_align_i16, not a second copy of it, and the intermediate
 * narrowing is epi_shr_half for the same reason.
 *
 * TWO int32 MULTIPLIES, NOT ONE int64 ONE. p*sig*g reaches 2^45 and
 * would need a 64-bit product (Watcom emits `call __I8M` for
 * int32*int32->int64, which is a soft-multiply per element); narrowing
 * in between keeps both products inside int32, and the second one sits
 * at the edge:
 *   |p| <= 32767 and sig <= 32768 (sigmoid_q15's range is [0, 32768],
 *   the upper end INCLUSIVE), so |p*sig| <= 1,073,709,056 < 2^31.
 *   sv = epi_shr_half(p*sig, 14) then has |sv| <= 65,534, attained -
 *   twice int16's range, so stage 2 cannot be an int16 multiply.
 *   |sv*g| <= 65,534 * 32,767 = 2,147,352,578 <= INT32_MAX, 131,069 to
 *   spare. At a narrowing of 13 the same product is 4,294,705,156 and
 *   does not fit, so 14 is the largest usable shift, not a taste; 15
 *   would cost a bit of the silu value for margin already there.
 * Verified exhaustively stage by stage in .prof/verify_swiglu_i16.c -
 * every (p, sig) pair for stage 1, every (sv, g) pair for stage 2, every
 * reachable int32 accumulator for the exit. Its A1 asserts the 65,534 is
 * ATTAINED, which is what rejected the 65,536 this comment first
 * carried.
 *
 * The (s, e) pair handed to epi_align_i16 means `s * 2^(e-8)`, and this
 * accumulator's value is `sv*g * 2^-(ep+1+eg)` (sv carries ep+15-14 ==
 * ep+1 fractional bits), so e == 8 - (ep + 1 + eg). Same bookkeeping as
 * epi_q41_align_i16's `target + 8`, written in the same units.
 *
 * `o` may alias `p`: element i reads p[i] and g[i] before writing o[i],
 * and nothing later reads p again. The routed expert relies on this.
 *
 * Bills nothing because there is nothing to bill - sigmoid_q15_i is a
 * table lookup on integers, and the rest is multiplies and shifts. */
lz_i64 lz_debug_swiglu_i16 = 0;   /* elements the int path produced */
/* Post-clamp maxima, the silent-clamp probe the LZ_SWIGLU_ES_* comment
   in forward.c reports from: a maximum strictly below the bound proves
   no element clamped, which a clamp inside epi_align_i16 cannot say for
   itself. */
int lz_debug_swiglu_pmax = 0;
int lz_debug_swiglu_gmax = 0;
int lz_debug_swiglu_omax = 0;

void lz_swiglu_q15_i16(short *o, const short *p, const short *g, int n,
                       int ep, int eg, int eo) {
    int i;
    int e = 8 - (ep + 1 + eg);
    for (i = 0; i < n; i++) {
        int32_t pv = p[i];
        int32_t gv = g[i];
        int32_t sv = (int32_t)epi_shr_half((lz_i64)(pv * sigmoid_q15_i(pv, ep)), 14);
        int32_t q  = epi_align_i16((lz_i64)(sv * gv), e, eo, 32767);
        int32_t a;
        o[i] = (short)q;
        a = pv < 0 ? -pv : pv; if (a > lz_debug_swiglu_pmax) lz_debug_swiglu_pmax = a;
        a = gv < 0 ? -gv : gv; if (a > lz_debug_swiglu_gmax) lz_debug_swiglu_gmax = a;
        a = q  < 0 ? -q  : q;  if (a > lz_debug_swiglu_omax) lz_debug_swiglu_omax = a;
    }
    lz_debug_swiglu_i16 += n;
}

float epi_q8(const int32_t *accb, const float *xsb, const float *ws,
                    int ng, int r, float post) {
    int g;
    /* r>1: per group a convert, a multiply and an add, then per r-run one
       multiply and one add, then the post scale. r==1: convert plus two
       multiplies plus an add per group, and 4 to fold the accumulators.
       This is the row the census put at 1,995,552 by a rule nobody
       could reconstruct. The convert (lz_i32f, once per group either
       way) is the same ng in both branches. */
    LZ_FCX(LZ_FC_EPI,
           r > 1 ? ng + (ng / r) + 1 : 2 * ng + 1,     /* mul */
           r > 1 ? ng + (ng / r)     : ng + 3,          /* add */
           0,                                            /* div */
           ng,                                           /* cvt */
           0);
    if (r > 1) {
        float sum = 0.0f;
        int g128;
        const int32_t *ap = accb;
        const float *xp = xsb;
        for (g128 = 0; g128 < ng / r; g128++, ap += r, xp += r) {
            float dot = 0.0f;
            for (g = 0; g < r; g++) {
                dot += lz_i32f(ap[g]) * xp[g];
            }
            sum += dot * ws[g128];
        }
        return sum * post;
    } else {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (g = 0; g + 3 < ng; g += 4) {
            a0 += lz_i32f(accb[g])     * (xsb[g]     * ws[g]);
            a1 += lz_i32f(accb[g + 1]) * (xsb[g + 1] * ws[g + 1]);
            a2 += lz_i32f(accb[g + 2]) * (xsb[g + 2] * ws[g + 2]);
            a3 += lz_i32f(accb[g + 3]) * (xsb[g + 3] * ws[g + 3]);
        }
        for (; g < ng; g++)
            a0 += lz_i32f(accb[g]) * (xsb[g] * ws[g]);
        return ((a0 + a2) + (a1 + a3)) * post;
    }
}

/* ---- Q8_0 row kernels, one per (ISA x impl) ---------------------------
   Each lives next to its table entry under ONE guard, so a definition
   and its use cannot drift apart - the drift that caused link breaks. */

/* Get the f32 view. In ops.c rather than model.c: this is a FORMAT
   matter, sharing the same layout knowledge as the matmul dispatch
   below - adding a format only touches this one file. */

/* Q6_1 single-element read: 4-bit plane b4 (16 bytes) + 2-bit plane
   b2 (8 bytes), both pointing at the start of the SAME 32-element
   sub-block. k in [0,32), returns [0,63].

   The two planes' groupings are deliberately aligned: on the 4-bit
   side k<16 takes the low nibble, k>=16 the high nibble; on the 2-bit
   side byte k&7 holds the (k>>3)-th bit pair. On the SIMD side, the
   four groups from `pand 0x03` + `psrlw 2/4/6` are exactly
   {0-7}{8-15}{16-23}{24-31}, corresponding group-for-group with the
   4-bit plane's two 8-byte loads; pre-expanded activation registers
   are reused as-is. */
static int lz_q61_get(const unsigned char *b4, const unsigned char *b2, int k) {
    int lo = (k < 16) ? (b4[k] & 15) : (b4[k - 16] >> 4);
    int hi = (b2[k & 7] >> (2 * (k >> 3))) & 3;
    return lo + (hi << 4);
}

float *lz_t_f32(const LZTensor *t, float *scratch) {
    int g, k;
    if (!t) return NULL;
    if (t->dtype == LZ_FMT_F32) return t->f;
    /* BF16 AHEAD OF THE SCALE GUARD, because it has no scale: it is
       storage, not quantization. The widening is `bits << 16` and the
       result is exactly what the file holds - the same operation the
       loader performs when bf16 storage is off, done here instead so
       the tensor can stay two bytes per element in RAM. Assembled from the two bytes by name
       rather than by casting to uint16_t*, so it is correct on a
       big-endian host and needs no alignment beyond 2. */
    if (t->dtype == LZ_FMT_BF16) {
        const unsigned char *p = (const unsigned char *)t->q;
        int i;
        if (!scratch || !t->q) return NULL;
        for (i = 0; i < t->n; i++) {
            uint32_t u = ((uint32_t)p[i * 2 + 1] << 24) |
                         ((uint32_t)p[i * 2] << 16);
            memcpy(&scratch[i], &u, sizeof(float));
        }
        return scratch;
    }
    if (!scratch || !t->q || !t->scale) return NULL;
    if (t->dtype == LZ_FMT_Q4_1) {
        /* nibbles: each 32-element sub-block is 16 bytes; byte j's low
           nibble = element j, high nibble = j+16 */
        const unsigned char *p = (const unsigned char *)t->q;
        if (!t->zero) return NULL;
        for (g = 0; g < t->n / t->gs; g++) {
            float d = t->scale[g], m = t->zero[g];
            int base;
            for (base = g * t->gs; base < (g + 1) * t->gs; base += 32) {
                const unsigned char *b = p + base / 2;
                for (k = 0; k < 16; k++) {
                    scratch[base + k]      = (float)(b[k] & 15) * d + m;
                    scratch[base + k + 16] = (float)(b[k] >> 4) * d + m;
                }
            }
        }
        return scratch;
    }
    if (t->dtype == LZ_FMT_Q6_1) {
        const unsigned char *p4 = (const unsigned char *)t->q;
        const unsigned char *p2 = p4 + (size_t)t->n / 2;
        if (!t->zero) return NULL;
        for (g = 0; g < t->n / t->gs; g++) {
            float d = t->scale[g], m = t->zero[g];
            int base;
            for (base = g * t->gs; base < (g + 1) * t->gs; base += 32) {
                const unsigned char *b4 = p4 + base / 2;
                const unsigned char *b2 = p2 + (unsigned)base / 4;
                for (k = 0; k < 32; k++)
                    scratch[base + k] =
                        (float)(lz_q61_get(b4, b2, k)) * d + m;
            }
        }
        return scratch;
    }
    if (t->dtype == LZ_FMT_T2) {
        /* One 2-bit plane, Q6_1's addressing. code-1 in {-1,0,+1}; no
           min term, so a code of 1 dequantizes to exactly 0.0f for
           every group regardless of scale - which is the property the
           format exists for. */
        const unsigned char *p2 = (const unsigned char *)t->q;
        for (g = 0; g < t->n / t->gs; g++) {
            float d = t->scale[g];
            int base;
            for (base = g * t->gs; base < (g + 1) * t->gs; base += 32) {
                const unsigned char *b2 = p2 + (unsigned)base / 4;
                for (k = 0; k < 32; k++)
                    scratch[base + k] =
                        (float)(((b2[k & 7] >> (2 * (k >> 3))) & 3) - 1) * d;
            }
        }
        return scratch;
    }
    if (t->dtype == LZ_FMT_Q16_0) {
        const int16_t *p = (const int16_t *)(const void *)t->q;
        for (g = 0; g < t->n / t->gs; g++) {
            float s = t->scale[g];
            for (k = 0; k < t->gs; k++)
                scratch[g * t->gs + k] = (float)p[g * t->gs + k] * s;
        }
        return scratch;
    }
    for (g = 0; g < t->n / t->gs; g++) {
        float s = t->scale[g];
        const int8_t *q = t->q + g * t->gs;
        float *d = scratch + g * t->gs;
        for (k = 0; k < t->gs; k++) d[k] = q[k] * s;
    }
    return scratch;
}

void lz_t_row_f32(const LZTensor *t, int row, int dim, float *out) {
    size_t base;
    int g0, i;
    if (!t || !out || dim <= 0) return;
    if (t->dtype == LZ_FMT_F32) {
        if (t->f) memcpy(out, t->f + (size_t)row * dim, (size_t)dim * sizeof(float));
        return;
    }
    /* BF16: same `<<16` widening as lz_t_f32, one row of it. Ahead of
       the scale/group guard below for the same reason - no scale. */
    if (t->dtype == LZ_FMT_BF16) {
        const unsigned char *p;
        int i;
        if (!t->q) return;
        p = (const unsigned char *)t->q + (size_t)row * dim * 2;
        for (i = 0; i < dim; i++) {
            uint32_t u = ((uint32_t)p[i * 2 + 1] << 24) |
                         ((uint32_t)p[i * 2] << 16);
            memcpy(&out[i], &u, sizeof(float));
        }
        return;
    }
    if (!t->q || !t->scale || t->gs <= 0) return;
    base = (size_t)row * dim;
    g0 = (int)(base / (size_t)t->gs);
    if (t->dtype == LZ_FMT_Q4_1) {
        const unsigned char *p = (const unsigned char *)t->q + base / 2;
        if (!t->zero) return;
        for (i = 0; i < dim; i += 32) {
            int g = g0 + i / t->gs;
            float d = t->scale[g], m = t->zero[g];
            const unsigned char *b = p + i / 2;
            int k;
            for (k = 0; k < 16; k++) {
                out[i + k]      = (float)(b[k] & 15) * d + m;
                out[i + k + 16] = (float)(b[k] >> 4) * d + m;
            }
        }
        return;
    }
    if (t->dtype == LZ_FMT_Q6_1) {
        const unsigned char *p4 = (const unsigned char *)t->q + base / 2;
        const unsigned char *p2 = (const unsigned char *)t->q +
                                  (size_t)t->n / 2 + (unsigned)base / 4;
        if (!t->zero) return;
        for (i = 0; i < dim; i += 32) {
            int g = g0 + i / t->gs;
            float d = t->scale[g], m = t->zero[g];
            const unsigned char *b4 = p4 + i / 2;
            const unsigned char *b2 = p2 + i / 4;
            int k;
            for (k = 0; k < 32; k++)
                out[i + k] = (float)lz_q61_get(b4, b2, k) * d + m;
        }
        return;
    }
    if (t->dtype == LZ_FMT_T2) {
        const unsigned char *p2 = (const unsigned char *)t->q + (unsigned)base / 4;
        for (i = 0; i < dim; i += 32) {
            int g = g0 + i / t->gs, k;
            float d = t->scale[g];
            const unsigned char *b2 = p2 + i / 4;
            for (k = 0; k < 32; k++)
                out[i + k] =
                    (float)(((b2[k & 7] >> (2 * (k >> 3))) & 3) - 1) * d;
        }
        return;
    }
    if (t->dtype == LZ_FMT_Q16_0) {
        const int16_t *p = (const int16_t *)(const void *)t->q + base;
        for (i = 0; i < dim; i++)
            out[i] = (float)p[i] * t->scale[g0 + i / t->gs];
        return;
    }
    {
        const int8_t *q = t->q + base;
        for (i = 0; i < dim; i++)
            out[i] = (float)q[i] * t->scale[g0 + i / t->gs];
    }
}

/* THE Q4_1 / Q6_1 FLOAT EPILOGUE - one copy, shared by every tier.
   Same shape as epi_q8 plus the zero-point term: accb[sb] is the
   sub-block's integer sum scaled by 256, xgb[g] the activation-only
   zero term for weight group g (computed once per token, reused across
   output rows), wd/wm the group's scale and min.

   The association order is again load-bearing, again what makes the MMX
   and SSE2 paths agree bit for bit: dot accumulates
   in s order within a weight group, is multiplied by wd ONCE, and the
   256 cancel happens on dotsum, AFTER which zsum is added. The SSE2
   fold has the same shape - a 4-lane multiply and a
   ((q0+q1)+q2)+q3 fold; term for term that is this loop. */
/* `wm` is the per-group ZERO coefficient, and T2 shares this epilogue by
   passing a negated scale rather than a stored min - ternary carries no
   min, and Sum w*x = d*[Sum code*x - Sum x] makes the coefficient -d.
   One function rather than a near-copy on purpose: the reduction ORDER
   is what the bit-identity contract is about, and two copies of it drift
   the moment one is touched. Callers hand in whichever array is theirs;
   matmul_t2_impl builds the negated one once per row. */
float epi_q41(const int32_t *accb, const float *xsb, const float *xgb,
                     const float *wd, const float *wm, int ng, int r) {
    float dotsum = 0.0f, zsum = 0.0f;
    int g, s;
    /* Per group: r sub-blocks each a convert (lz_i32f) plus a multiply
       plus an add (the dot accumulation), then two multiplies and two
       adds closing the group (dotsum += dot*wd[g], zsum += xgb[g]*wm[g]).
       Plus one closing multiply and add for the 1/256 fold and the zero
       term. Verified against an independent raw-counter re-derivation
       (sum of ng, ng*r and call count taken separately from this
       formula): both agree to the op, 5,089,280 f32 ops/token on kmr20
       / .prof/tz.bin / --fixed all - this site's entire prior invisible
       share (LZ_FC_EPI moved 119,616 -> 5,208,896/token, i.e. from
       13.2% to 86.9% of the whole census). */
    LZ_FCX(LZ_FC_EPI, (long)ng * (r + 2) + 1, (long)ng * (r + 2) + 1,
           0, (long)ng * r, 0);
    {
        const int32_t *ap = accb;
        const float *xp = xsb;
        for (g = 0; g < ng; g++, ap += r, xp += r) {
            float dot = 0.0f;
            for (s = 0; s < r; s++) {
                dot += lz_i32f(ap[s]) * xp[s];
            }
            dotsum += dot * wd[g];
            zsum += xgb[g] * wm[g];
        }
    }
    return dotsum * (1.0f / 256.0f) + zsum;
}

/* int-pipeline milestone 2 (docs/int-pipeline-project.md #2.1): BOTH of
   epi_q41's terms in one integer domain, joined by an integer shift, so
   exactly one float is materialized per output element and the float
   cost is O(1) instead of O(ng).

   Milestone 1 left the zero term in float and gave the dot term its own
   float exit, which costs 2 cvt + 2 mul + 1 add against epi_q41's
   2*ng+3 - nothing at all where ng == 1, and ng == 1 is 81.8% of the
   calls on this checkpoint (measured, kmr20, both gate corpora:
   ng=1/r=16 67.68%, ng=1/r=8 14.14%, ng=2/r=4 14.14%, ng=4/r=4 4.04%).
   The join below is what makes the saving independent of ng.

   The two halves, each `mantissa * 2^exponent` with an integer exponent
   the row and the token fix between them:

     dot  == s_dot  * 2^(e_dot)    e_dot  = -g_epi_xe[tk] - w->sexp[row]
     zero == s_zero * 2^(e_zero)   e_zero = -g_epi_xe[tk] - w->zexp[row]

   Both carry the same -g_epi_xe[tk], so their difference is a per-ROW
   integer decided entirely on the weight side; the token only moves
   both together.

   THE x256 IS TAKEN OFF THE DOT SUM FIRST, and that is load-bearing
   twice over. The row kernels return acc = 256 * (integer dot), which
   epi_fixed cancels with a 1/256 post-multiply; here it is an exact
   integer shift instead (measured: 953,024,512 acc32 entries over the
   English corpus, 0 of them not a multiple of 256), and doing it before
   the join leaves 8 bits of headroom under the epilogue MAC's worst case
   so the addition that follows provably cannot overflow int64:
   |s_dot| <= 2^63 by LZ_EPI_MAX_NG's derivation, hence |s_dot>>8| <=
   2^55, while |s_zero| <= 64 * 2^31 * 2^15 = 2^52.

   Which side gets shifted is not a free choice: shifting up would have
   to fit inside the 3 bits of int64 headroom |s_dot| already leaves, so
   the join always aligns DOWN, to the coarser of the two exponents.
   Measured on kmr20, e_zero is coarser on 100% of calls, by 0 to 6 bits
   (the raw sexp+8-zexp spread is 8..14 and the leading 8 are the exact
   x256 shift above) - the zero plane's mantissas are the group MINs,
   which run about 16x the group SCALES, so its exponent sits about four
   below. Six bits off a sum whose measured magnitude is 2^54.8 leaves
   ~49 significant bits against float32's 24, and the rounding it costs
   is 2^20 times smaller than the int16 quantization of w->zero that the
   zero term already carries. The `target` below is still the general
   max() rather than a hardcoded direction: the opposite ordering is one
   comparison to handle correctly and a silent wrong answer to assume
   away.

   `ng`/`r` are epi_q41's own convention (ng weight groups, r 32-wide
   sub-blocks per group); epi_fixed_raw's own `ng` parameter means
   "sub-block count" - ng*r converts between the two.

   `zneg` is T2, which stores no min: its zero coefficient is -scale, so
   it reuses the sq/sexp plane lz_epi_prep already built and negates the
   accumulated int64 (|s_zero| <= 2^52, nowhere near the LLONG_MIN
   corner that makes negation UB). Q4_1/Q6_1 pass 0 and read the zq/zexp
   plane. */
/* The join itself, with no exit committed: `comb * 2^*e_out`, both
   halves already in one integer domain. Split out for the same reason
   epi_q41's own comment gives for being one function - the reduction
   ORDER is the bit-identity contract, and the float exit
   (epi_q41_fixed) and the aligned-int16 exit (epi_q41_align_i16) must
   share this arithmetic rather than each carry a copy of it. */
static lz_i64 epi_q41_join(const int32_t *accb, const LZTensor *w, int row,
                              int tk, int ng, int r, int zneg, int *e_out) {
    int e_dot, e_zero, target;
    lz_i64 s_dot, s_zero;
    const int16_t *zp = zneg ? w->sq : w->zq;
    s_dot = epi_fixed_raw(accb, w, row, tk, ng * r, r, &e_dot);
    s_dot = epi_shr_half(s_dot, 8);        /* exact: acc32 is the x256 */
    e_zero = -g_epi_xe[tk] - (int)(zneg ? w->sexp[row] : w->zexp[row]);
    /* Was a widening copy into a global scratch, then a mac. Three
       things at once and all three measured: the engine's third-hottest
       branch (145,723,392 evaluations at 41.9%, ng being 2), 241k
       duplicate copies a token, and a buffer whose only purpose was to
       carry the copy to the next statement. lz_epi_mac_i16 takes the
       int16 plane directly; see its header. */
    s_zero = lz_epi_mac_i16(g_epi_zg[tk], zp + (size_t)row * ng, ng);
    if (zneg) s_zero = -s_zero;
    target = e_dot > e_zero ? e_dot : e_zero;
    *e_out = target;
    /* ONE OF THESE TWO SHIFTS IS ALWAYS ZERO, because target is the max
       of the two exponents - so one term is already aligned and the
       call can only reach epi_shr_half's `if (rs <= 0) return s;` and
       come back.

       It was reaching it 811,008 times per eight tokens on kmr20 (half
       of these two lines' 1,622,016 calls), each one a cross-TU call
       into a 67-instruction function to return its argument. The branch
       census reads the same fact from the other end: ops.c:204 scores
       a 29.7% minority arm over 2,727,936 executions, and 811,008 /
       2,727,936 is 29.7%.

       Exactly equivalent, not an approximation: `target == e_dot` and
       `target - e_dot == 0` are the same condition on ints, and that is
       the condition the guard tests. */
    return (target == e_dot  ? s_dot  : epi_shr_half(s_dot,  target - e_dot))
         + (target == e_zero ? s_zero : epi_shr_half(s_zero, target - e_zero));
}

float epi_q41_fixed(const int32_t *accb, const LZTensor *w, int row,
                    int tk, int ng, int r, int zneg) {
    int target;
    lz_i64 comb = epi_q41_join(accb, w, row, tk, ng, r, zneg, &target);
    /* pow2f returns 0 below 2^-126 rather than a denormal, and |comb|
       reaches 2^55, so at a deep enough target a single multiply would
       zero a result that is still an ordinary normal float. e_dot and
       e_zero are each two exponents capped at 100 by their packing
       loops, so target can reach -200 - the same floor epi_fixed's own
       fold guards, reachable here by a wider margin. */
    if (target < -126) {
        LZ_FCX(LZ_FC_EPI, 2, 0, 0, 1, 0);  /* convert, two multiplies */
        return (float)comb * pow2f(-126) * pow2f(target + 126);
    }
    {
        float cf = (float)comb, sc;
        /* Same exponent-add as epi_fixed's exit. This is the busier of
           the two on kmr20 - 45,568 of the row's 59,904 calls a token,
           of which the tied LM head is 32,768. */
        if (f32_scale_pow2(cf, target, &sc)) {
            LZ_FCX(LZ_FC_EPI, 0, 0, 0, 1, 0);  /* convert; the scale is integer */
            return sc;
        }
        LZ_FCX(LZ_FC_EPI, 1, 0, 0, 1, 0);      /* convert, one multiply */
        return cf * pow2f(target);
    }
}

/* int-pipeline milestone 3: the same join, exited as an int16 already
   aligned to the CONSUMER's exponent, so no float is materialized at
   this boundary at all. Bills nothing because there is nothing to bill -
   epi_q41_join is integer end to end and epi_align_i16 is a shift.
 *
 * `epi_align_i16` reads its `(s, e)` pair as `s * 2^(e-8)` (epi_fixed's
 * convention, where the -8 is the row kernel's x256 still uncancelled).
 * The join has already taken that shift off exactly, so this exit's
 * value is `comb * 2^target` and the pair handed over is (comb,
 * target + 8) - not a fudge factor, the same quantity written in the
 * primitive's units.
 *
 * `target_e`/`bound` belong to the consumer (LZ_CONV_ES,
 * lz_conv_accum_bound(k) for the conv family) - see epi_align_i16's
 * header for why they are taken rather than derived here. The clamp to
 * +-bound happens inside it, so the consumer needs no clamp of its
 * own. */
int32_t epi_q41_align_i16(const int32_t *accb, const LZTensor *w, int row,
                          int tk, int ng, int r, int zneg,
                          int target_e, int bound) {
    int target;
    lz_i64 comb = epi_q41_join(accb, w, row, tk, ng, r, zneg, &target);
    return epi_align_i16(comb, target + 8, target_e, bound);
}
