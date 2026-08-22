/* ops_sched.c - Tier globals, CPU detection, kernel dispatch, and all
   mode selectors (--kernel, --pair, --gdn, --conv, --attn, etc.).

   A separate TU from ops.c to keep #if guard density low and make the
   tier dispatch structure legible. Only the public API is extern; the
   rest is static to this TU. */
#include <stddef.h>
#include <string.h>    /* strcmp: tier selectors */

#include "ops.h"
#include "err.h"
#include "cpucheck.h"
#include "mmx_compat.h"
#include "ops_mmx.h"
#include "ops_sse2.h"
#include "ops_kernel_shared.h"
#include "ops_quant.h"  /* lz_cpu_has_mmx (non-static, from module 1) */
#include "ops_sched.h"
#include "ops_t2_arm.h" /* LZ_T2_ARM_ASM_EXTERN: one home for "this build
                           carries the ARM asm tier" */

#ifndef LZ_GDN_FIXED
#define LZ_GDN_FIXED 1
#endif /* LZ_GDN_FIXED */

/* ---- g_kernel (the selected ISA tier) ---------------------------------
   Defined here (was in ops.c). Declared extern in ops_kernel_shared.h. */
int g_kernel = 0;

/* Accessor for g_kernel. lz_row_pick (ops_matmul.c) reads the tier
   through this rather than the raw global, so the matmul module never
   needs direct access to the sched global. */
int lz_kernel_sel(void) { return g_kernel; }

/* ---- CPUID caching (SSE bit, for hot-path dispatch) ------------------- */

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86)
static int g_has_sse = -1;

int lz_cpu_has_sse(void) {
#ifdef LZ_CPUID_NOCACHE
    return (lz_cpuid1_edx() & (1u << 25)) != 0;
#else
    if (g_has_sse < 0) g_has_sse = (lz_cpuid1_edx() & (1u << 25)) != 0;
    return g_has_sse;
#endif /* LZ_CPUID_NOCACHE */
}

#endif /* __i386__ || __x86_64__ || _M_IX86 */

/* lz_cpu_has_mmx: non-static, defined in ops.c, declared in ops_quant.h.
   g_has_mmx stays in ops.c (its cache variable). */

/* ---- prefetch detection ----------------------------------------------- */

static int g_pf = -1;

#if defined(__WATCOMC__)
extern unsigned lz_cpuid_ext_edx(void);
#pragma aux lz_cpuid_ext_edx =      ".586"                          "push ebx"                      "mov  eax, 80000000h"           "cpuid"                         "cmp  eax, 80000001h"           "jb   short no_ext"             "mov  eax, 80000001h"           "cpuid"                         "mov  eax, edx"                 "jmp  short done_ext"           "no_ext:"                       "xor  eax, eax"                 "done_ext:"                     "pop  ebx"                      __value [__eax]                 __modify [__eax __ecx __edx]
#endif /* __WATCOMC__ */

static int pf_detect(void) {
#if defined(__WATCOMC__)
    unsigned edx = lz_cpuid1_edx();
    if (edx & (1u << 25)) return LZ_PF_NTA;
    if (lz_cpuid_ext_edx() & (1u << 31)) return LZ_PF_AMD;
    if (edx & (1u << 23)) return LZ_PF_LOAD;
    return LZ_PF_NONE;
#else
    return LZ_PF_NTA;
#endif /* __WATCOMC__ */
}

int lz_prefetch_mode(void) {
    if (g_pf < 0) g_pf = pf_detect();
    return g_pf;
}

/* ---- pairing --------------------------------------------------------- */

int g_pair = 1;
static int g_pair_auto = 1;

int lz_pair_mode(void) { return g_pair; }

const char *lz_pair_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "auto") == 0)      { g_pair = 1; g_pair_auto = 1; }
    else if (strcmp(name, "on") == 0)   { g_pair = 1; g_pair_auto = 0; }
    else if (strcmp(name, "off") == 0)  { g_pair = 0; g_pair_auto = 0; }
    else return NULL;
    return g_pair ? "on" : "off";
}

int lz_pair_is_auto(void) { return g_pair_auto; }

/* ---- q8r_have_simd / lz_q8r_tier ------------------------------------ */

/* Which q8-round implementation to use, and nothing else. No tier's
   AUTO asks it any more - the shipping build is fixed-point whatever
   the machine, so auto answers "fixed" without a CPUID.

   CPUID SSE, never MMX, for the choice it does still make: every
   Socket 7 part has MMX and no SSE, so an MMX-worded test would claim a
   SIMD tier on exactly the machines that have none - Cyrix 6x86MX/MII
   and IDT WinChip C6/2 alongside the K6, the weakest x87 in the family.
   Enumerating machines in a comment is how that edit gets rationalised;
   the predicate is the contract. */
int q8r_have_simd(void) {
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
    return lz_cpu_has_sse();
#else
    return 0;
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */
}

/* lz_soft_float_target ("is this machine's float slow relative to its
   integer" - a one-bit CPUID proxy) USED to be what all seven write
   tiers' `auto` resolved from, six of them directly and lz_scalar_mode
   deliberately not (commit 508e015 already made scalar fixed
   unconditionally, ignoring this question, before this file did).

   REMOVED (this commit): the user settled the question these tiers
   existed to keep open. The shipping build is fixed-point
   unconditionally, on every host, regardless of what this predicate
   would have answered there - so nothing asks it anymore, and every one
   of the seven tiers now has the same shape lz_scalar_mode already did.
   Float is not gone; it is reachable exactly like every other explicit
   tier value (--recur-write float et al.) and in one command for all
   seven at once (--fixed off, cli_main.c) - the correctness oracle every
   tolerance in CLAUDE.md and every cross-compiler gate is measured
   against, not an auto outcome. See
   docs/fixed-tier-pipeline-breakpoints.md for which of the resulting
   float-to-fixed conversions are removable now that fixed is
   unconditional (most - they existed only because a stage's neighbor
   might have been float) and which are forced by data regardless (a
   short list: the residual stream's cross-layer dynamic range, and
   the SubLN Hadamard transform's float-only kernel). q8r_have_simd
   (this predicate's own building block) still has its
   LZ_HAVE_Q8R_SIMD/LZ_HAVE_Q8R_SSE-guarded caller below, untouched -
   that is a different axis (the Q8 row-kernel tier), not a write tier,
   and out of scope here. */

#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
int lz_q8r_tier(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return 0;
#if defined(LZ_HAVE_Q8R_SSE)
#ifdef LZ_Q8R_FORCE_SSE
    return 1;
#endif /* LZ_Q8R_FORCE_SSE */
#endif /* LZ_HAVE_Q8R_SSE */
#if defined(LZ_HAVE_Q8R_SIMD)
    if (g_kernel == LZ_KERNEL_SSE2) return 2;
#endif /* LZ_HAVE_Q8R_SIMD */
#if defined(LZ_HAVE_Q8R_SSE)
    if (lz_cpu_has_sse()) return 1;
#endif /* LZ_HAVE_Q8R_SSE */
    return 0;
}
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */

/* ---- GDN recurrence tier (--recur-write): BOTH passes ---------------- */

#define LZ_P2_FLOAT 0
#define LZ_P2_FIXED 1
#define LZ_P2_AUTO  2
static int g_gdn_p2 = LZ_P2_AUTO;

int lz_gdn_p2_mode(void) {
#if LZ_GDN_FIXED
    if (g_gdn_p2 == LZ_P2_AUTO) return LZ_P2_FIXED;
    return g_gdn_p2;
#else
    return 0;
#endif /* LZ_GDN_FIXED */
}

/* Pass 1 reads the SAME selector as pass 2. One axis for the whole
   recurrence, so --fixed off is a complete float arm for it: a pass-1
   tier with no runtime value would leave the float oracle running fixed
   arithmetic for the larger of the two passes. */
int lz_gdn_mode(void) { return lz_gdn_p2_mode(); }

int lz_gdn_p2_is_auto(void) { return g_gdn_p2 == LZ_P2_AUTO; }

const char *lz_gdn_p2_select(const char *name) {
    if (!name) return NULL;
    /* float used to have no explicit spelling here - auto asked
       hardware and an SSE machine landed on float that way, so a caller
       who wanted float explicitly had nothing to type. Auto no longer
       asks hardware (see lz_soft_float_target's retirement note above),
       so this value is what makes float reachable at all now - the
       measurement/correctness-oracle path, not a deployment choice. */
    if (strcmp(name, "float") == 0) g_gdn_p2 = LZ_P2_FLOAT;
    else if (strcmp(name, "fixed") == 0) g_gdn_p2 = LZ_P2_FIXED;
    else return NULL;
    return lz_gdn_p2_mode() ? "fixed" : "float";
}

/* ---- conv1d tier ----------------------------------------------------- */

#define LZ_CONV_FLOAT 0
#define LZ_CONV_FIXEDT 1
#define LZ_CONV_AUTO  2
static int g_conv = LZ_CONV_AUTO;

int lz_conv_mode(void) {
#if LZ_CONV_FIXED
    if (g_conv == LZ_CONV_AUTO) return LZ_CONV_FIXEDT;
    return g_conv;
#else
    return 0;
#endif /* LZ_CONV_FIXED */
}

int lz_conv_is_auto(void) { return g_conv == LZ_CONV_AUTO; }

const char *lz_conv_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_conv = LZ_CONV_FLOAT;
    else if (strcmp(name, "fixed") == 0) g_conv = LZ_CONV_FIXEDT;
    else return NULL;
    return lz_conv_mode() ? "fixed" : "float";
}

/* ---- attention tier -------------------------------------------------- */

/* The three low bits are the scoring/wsum bitmask; LZ_ATTN_TIER_INT sits
   above them as a fourth VALUE on the same axis rather than a separate
   boolean. The int output rides on the fixed wsum, so selecting it
   selects that too - and "float + int", which used to be spellable and
   silently inert, no longer is. */
#define LZ_ATTN_TIER_INT 4

static int g_attn = -1;

int lz_attn_mode(void) {
    int m = (g_attn < 0) ? 3 : ((g_attn == LZ_ATTN_TIER_INT) ? 3 : g_attn);
    return m & LZ_ATTN_FIXED;
}

int lz_attn_is_auto(void) { return g_attn < 0; }

int lz_attn_int(void) { return g_attn == LZ_ATTN_TIER_INT; }

const char *lz_attn_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_attn = 0;
    else if (strcmp(name, "scoring") == 0) g_attn = 1;
    else if (strcmp(name, "wsum") == 0) g_attn = 2;
    else if (strcmp(name, "fixed") == 0) g_attn = 3;
    else if (strcmp(name, "int") == 0) g_attn = LZ_ATTN_TIER_INT;
    else return NULL;
    if (lz_attn_int()) return "int";
    switch (lz_attn_mode()) {
        case 1:  return "scoring";
        case 2:  return "wsum";
        case 3:  return "fixed";
        default: return "float";
    }
}

/* ---- sigmoid/silu tier ----------------------------------------------- */

static int g_sig = -1;

int lz_sig_mode(void) {
    return (g_sig < 0) ? 1 : g_sig;
}

int lz_sig_is_auto(void) { return g_sig < 0; }

const char *lz_sig_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_sig = 0;
    else if (strcmp(name, "fixed") == 0) g_sig = 1;
    else return NULL;
    return lz_sig_mode() ? "fixed" : "float";
}

/* ---- scalar tier ----------------------------------------------------- */

/* Used to be the one axis whose auto ignored hardware (commit 508e015)
   while the other six asked lz_soft_float_target - an odd one out, and
   worth a warning at the CLI (cli_main.c's own note that "pass no flags"
   was not a float baseline). That asymmetry is gone: lz_soft_float_target
   is retired and every one of the seven tiers now has exactly this
   shape, `(g_X < 0) ? 1 : g_X`. Kept as its own section rather than
   folded into a shared helper - it is one line, and a helper would only
   hide that this was already the answer six other places needed to
   copy, not the other way round. */
/* NOT file-static, unlike its eleven siblings: lz_scalar_mode is a
   macro in ops.h so that softmax's per-element lz_exp does not pay a
   cross-TU call for it. This file is still its only writer. */
int lz_scalar_g = -1;
#define g_scalar lz_scalar_g

int lz_scalar_is_auto(void) { return g_scalar < 0; }

const char *lz_scalar_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_scalar = 0;
    else if (strcmp(name, "fixed") == 0) g_scalar = 1;
    else return NULL;
    return lz_scalar_mode() ? "fixed" : "float";
}

/* ---- norm tier ------------------------------------------------------- */

/* TWO values, not three. `int` used to be the third, meaning "the fixed
   tier writes its Q15 result as int instead of dequantizing it and
   letting the consumer requantize". It is gone as a SELECTABLE value
   because there is nothing left to select between: on the gated norms
   the two spellings are bit-identical (the dequantize/requantize pair
   cancels - proof at lz_quantize_q8_int), and on the plain norm the int
   form is ~9.5x cheaper per element on the target and is the same
   float-to-fixed call the user already settled for every other write
   tier. A knob whose one setting is never worth choosing is a knob that
   only records history.

   Float is not gone - `--norm-write float` still reaches the float
   output loop, and that arm is the oracle, not an auto outcome. Same
   shape as this file's lz_soft_float_target retirement note above. */
/* FIXEDT, not FIXED: LZ_NORM_FIXED is the compile switch in ops.h that
   decides whether the fixed bodies exist at all. Same collision
   LZ_CONV_FIXEDT dodges above. */
#define LZ_NORM_FLOAT  0
#define LZ_NORM_FIXEDT 1

static int g_norm = -1;

int lz_norm_mode(void) {
    return (g_norm < 0) ? 1 : (g_norm != LZ_NORM_FLOAT);
}

int lz_norm_is_auto(void) { return g_norm < 0; }

/* Kept as the one place that says "the fixed tier writes int", now
   answered by the tier itself. Not folded into its call sites: they
   read `lz_norm_int() && lz_norm_can_fixed(n) && ...`, and the name is
   what tells a reader WHICH of the two fixed output loops the branch
   selects - lz_norm_can_fixed alone would not. */
int lz_norm_int(void) { return lz_norm_mode(); }

const char *lz_norm_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_norm = LZ_NORM_FLOAT;
    else if (strcmp(name, "fixed") == 0) g_norm = LZ_NORM_FIXEDT;
    else return NULL;
    return lz_norm_mode() ? "fixed" : "float";
}

#ifndef LZ_NORM_MAX_N
#define LZ_NORM_MAX_N 4096
#endif /* LZ_NORM_MAX_N */

int lz_norm_can_fixed(int n) {
#if LZ_NORM_FIXED
    return lz_norm_mode() && n > 0 && n <= LZ_NORM_MAX_N;
#else
    /* The fixed bodies are not in this build, so no answer from
       --norm-write can make them runnable. Saying no here is what keeps
       the call sites off a compiled-out path; norm_ss_fixed declines
       for the same reason. */
    (void)n;
    return 0;
#endif /* LZ_NORM_FIXED */
}

/* ---- SubLN activation scale granularity ------------------------------ */

/* How wide one INT8 activation scale is on the SubLN o_proj/down_proj
   input. NOT a fixed-point tier: it changes no float/int decision and is
   deliberately absent from cli_main.c's --fixed axes[] table, which
   floats every fixed-point site. Putting it there would make `--fixed
   off` silently change the scale layout too.

   `tok` is BitNet v2's per-token absmax: one scale for the whole row,
   replicated into every n/32 slot. That is where the SubLN path started,
   because the reference quantizes per token EVERYWHERE - weights per
   tensor, activations per token - so it had no finer grouping to give
   up. This engine's native activation granularity is per 32 elements
   (lz_act_gs), which the non-SubLN path uses, so `tok` hands back a
   grouping the rest of the engine keeps.

   `group` keeps that per-32 grouping across the Hadamard. The transform
   does not require a row-wide scale: the fixed-point FWHT's own scale S
   is per row because a butterfly mixes a whole blk-wide block, but that
   is the INPUT representation; once the rotated int32 values exist,
   quantizing them in 32-wide groups is exact - each group's dequant
   q[i]*s[g] reproduces fscr[i]*inv_S/sqrt(blk) as before.

   A knob rather than a straight replacement: which one wins is measured,
   not derived, and the two disagree on the wire. */
#define LZ_SUBN_SCALE_TOK   0
#define LZ_SUBN_SCALE_GROUP 1

static int g_subn_scale = LZ_SUBN_SCALE_TOK;

int lz_subn_scale_group(void) { return g_subn_scale == LZ_SUBN_SCALE_GROUP; }

const char *lz_subn_scale_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "tok") == 0) g_subn_scale = LZ_SUBN_SCALE_TOK;
    else if (strcmp(name, "group") == 0) g_subn_scale = LZ_SUBN_SCALE_GROUP;
    else return NULL;
    return lz_subn_scale_group() ? "group" : "tok";
}

/* ---- SubLN Hadamard on/off ------------------------------------------- */

/* The control arm for the block Hadamard. `off` forces the resolved
   hadamard_o / hadamard_down to 0 at their use sites, which is the same
   thing a model that never had the fields does.

   It exists because the feature had NO way to be turned off at runtime,
   and a transform with no control arm cannot be measured or gated: the
   only way to ask "did the rotation happen" was to export a second
   model with the field zeroed. It also makes the f32-vs-quantized
   agreement checkable without a threshold - see build/subn_scale_gate.sh.

   Turning it off on weights TRAINED in the rotated basis produces a
   worse model, not a broken one; that is what a control arm is. */
static int g_hadamard_off = 0;

int lz_hadamard_on(void) { return !g_hadamard_off; }

const char *lz_hadamard_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "on") == 0) g_hadamard_off = 0;
    else if (strcmp(name, "off") == 0) g_hadamard_off = 1;
    else return NULL;
    return lz_hadamard_on() ? "on" : "off";
}

/* ---- epilogue tier --------------------------------------------------- */

static int g_epi = -1;

int lz_epi_mode(void) {
    return (g_epi < 0) ? 1 : g_epi;
}

int lz_epi_is_auto(void) { return g_epi < 0; }

const char *lz_epi_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_epi = 0;
    else if (strcmp(name, "fixed") == 0) g_epi = 1;
    else return NULL;
    return lz_epi_mode() ? "fixed" : "float";
}

/* ---- prefetch selector ----------------------------------------------- */

const char *lz_prefetch_select(const char *name) {
    int have = pf_detect();
    int mode;
    if (!name)                        return NULL;
    else if (strcmp(name, "auto")  == 0) mode = -1;
    else if (strcmp(name, "none")  == 0) mode = LZ_PF_NONE;
    else if (strcmp(name, "load")  == 0) mode = LZ_PF_LOAD;
    else if (strcmp(name, "nta")   == 0) mode = LZ_PF_NTA;
    else if (strcmp(name, "3dnow") == 0) mode = LZ_PF_AMD;
    else                              return NULL;

    if (mode == -1) {
        g_pf = have;
    } else if (mode == LZ_PF_NONE || mode == LZ_PF_LOAD) {
        g_pf = mode;
    } else if (mode == have) {
        g_pf = mode;
    } else {
        g_pf = have;
    }
    return lz_prefetch_name();
}

const char *lz_prefetch_name(void) {
    switch (lz_prefetch_mode()) {
    case LZ_PF_NTA:  return "nta";
    case LZ_PF_AMD:  return "3dnow";
    case LZ_PF_LOAD: return "load";
    default:         return "none";
    }
}

/* ---- kernel detection and dispatch ----------------------------------- */

static int kernel_detect(void) {
#if defined(LZ_T2_ARM_ASM_EXTERN)
    /* No CPUID question on this target: -march=armv5te is a build-time
       floor (build/arm/build_engine.sh), and the asm tier needs exactly
       that and nothing later. */
    return LZ_KERNEL_ARM_ASM;
#elif defined(__WATCOMC__)
    unsigned edx = lz_cpuid1_edx();
    if (edx & (1u << 26)) return LZ_KERNEL_SSE2;
    if (edx & (1u << 23)) return LZ_KERNEL_MMX;
    return LZ_KERNEL_REF;
#elif defined(LZ_ROW_SSE2_EXTERN)
    if (lz_cpuid1_edx() & (1u << 26)) return LZ_KERNEL_SSE2;
#if defined(LZ_DOT_MMX_EXTERN)
    return LZ_KERNEL_MMX;
#else
    return LZ_KERNEL_REF;
#endif /* LZ_DOT_MMX_EXTERN */
#elif defined(LZ_DOT_MMX_EXTERN)
    return LZ_KERNEL_MMX;
#else
    return LZ_KERNEL_REF;
#endif /* __WATCOMC__ */
}

int lz_kernel_select(int which) {
    int have = kernel_detect();
    /* The ARM values are not "a tier this x86 machine happens to lack",
       they are a different instruction set: accepting one here would set
       g_kernel to a value no x86 dispatch tests for, and every
       `g_kernel == LZ_KERNEL_REF` site would quietly answer no. Clamped
       first, before the per-build clamps below, so exactly one of the
       two mistakes is possible - falling back - and never the other. */
#if !defined(LZ_T2_ARM_ASM_EXTERN)
    if (which == LZ_KERNEL_ARM || which == LZ_KERNEL_ARM_ASM)
        which = LZ_KERNEL_AUTO;
#endif /* !LZ_T2_ARM_ASM_EXTERN */
    if (which == LZ_KERNEL_AUTO) {
        g_kernel = have;
    } else {
#if defined(LZ_T2_ARM_ASM_EXTERN)
        /* Three tiers really run here and all three must be selectable,
           or the bit-identity comparison between them is unavailable:
           ref (the shared scalar), arm-c, arm-asm. Anything else is an
           x86 tier and clamps. */
        g_kernel = (which == LZ_KERNEL_REF || which == LZ_KERNEL_ARM ||
                    which == LZ_KERNEL_ARM_ASM) ? which : have;
#elif defined(__WATCOMC__)
        g_kernel = (which == LZ_KERNEL_SSE2 && have != LZ_KERNEL_SSE2)
                   ? have : which;
#elif defined(LZ_DOT_MMX_EXTERN) && defined(LZ_ROW_SSE2_EXTERN)
        g_kernel = (which == LZ_KERNEL_SSE2 && have != LZ_KERNEL_SSE2)
                   ? have : which;
#else
        g_kernel = (which == LZ_KERNEL_REF) ? LZ_KERNEL_REF : have;
#endif /* __WATCOMC__ */
    }
    return g_kernel;
}

const char *lz_kernel_name(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_SSE2) return "sse2";
    if (g_kernel == LZ_KERNEL_MMX) return "mmx";
    if (g_kernel == LZ_KERNEL_ARM_ASM) return "arm-asm";
    if (g_kernel == LZ_KERNEL_ARM) return "arm-c";
    return "ref";
}

const char *lz_build_paths(void) {
#if defined(LZ_T2_ARM_ASM_EXTERN)
    return "arm-c+arm-asm";
#elif defined(__WATCOMC__)
    return "mmx-asm+sse2-asm";
#elif defined(LZ_DOT_MMX_EXTERN) && defined(LZ_ROW_SSE2_EXTERN)
    return "mmx-intrin+sse2-intrin";
#elif defined(LZ_DOT_MMX_EXTERN)
    return "mmx-intrin";
#elif defined(LZ_ROW_SSE2_EXTERN)
    return "sse2-intrin";
#else
    return "scalar";
#endif /* __WATCOMC__ */
}

const char *lz_kernel_tier(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return "ref";
#if defined(LZ_T2_ARM_ASM_EXTERN)
    return (g_kernel == LZ_KERNEL_ARM_ASM) ? "arm-asm" : "arm-c";
#elif defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)
#  ifdef __WATCOMC__
    return (g_kernel == LZ_KERNEL_SSE2) ? "sse2-asm" : "mmx-asm";
#  else
    return (g_kernel == LZ_KERNEL_SSE2) ? "sse2-intrin" : "mmx-intrin";
#  endif
#elif defined(LZ_ROW_SSE2_EXTERN)
    return "sse2-intrin";
#else
    return "ref";
#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */
}

/* ---- tier report and pin string -------------------------------------- */

/* CROSS-PLATFORM pins: a fixed, deliberately mostly-float string for
   build/arm/parity_gate.sh to compare one code path across x86 and ARM
   with, not a report of what is running and not the auto resolution.

   NOT USABLE AS A SINGLE-PLATFORM REGRESSION PIN, on x86 or on ARM: auto
   is now unconditionally fixed on every platform (lz_soft_float_target,
   the per-platform hardware predicate this comment used to name, is
   retired - see its own former site's note, still readable in this
   file's history, and docs/fixed-tier-pipeline-breakpoints.md), so this
   string's mostly-float values are guaranteed to differ from what EVERY
   platform's auto now gives, everywhere, not just on the one platform a
   stale version of this comment singled out. Pinning from here for a
   regression run forces most tiers back to float and switches OFF the
   very kernels a change may have touched. A regression run pinned from
   this string once compared two builds with both new fixed kernels
   disabled and looked clean. Regress on plain `auto` (now the same
   fixed-everywhere answer on every platform), or pin the knob under test
   explicitly; use this string only for the x86<->ARM comparison it
   exists for. lz_tier_report() below is the one that reads live state. */
const char *lz_tier_pins(void) {
    return "--kernel ref --recur-write fixed --conv-write float "
           "--attn-write float --epi-write float --sig-write float "
           "--norm-write float --scalar-fixed float --pair off "
           "--prefetch none";
}

const char *lz_tier_report(void) {
    static char buf[512];
    int p = 0;
    p += sprintf(buf + p, "  recurrence   %s%s\n",
                 lz_gdn_p2_mode() ? "fixed" : "float",
                 lz_gdn_p2_is_auto() ? " (default)" : "");
    p += sprintf(buf + p, "  conv1d       %s%s\n",
                 lz_conv_mode() ? "fixed" : "float",
                 lz_conv_is_auto() ? " (default)" : "");
    p += sprintf(buf + p, "  attention    %s%s\n",
                 lz_attn_mode() == 3 ? "fixed" :
                 lz_attn_mode() == 1 ? "scoring" :
                 lz_attn_mode() == 2 ? "wsum" : "float",
                 lz_attn_is_auto() ? " (default)" : "");
    p += sprintf(buf + p, "  epilogue     %s%s\n",
                 lz_epi_mode() ? "fixed" : "float",
                 lz_epi_is_auto() ? " (default)" : "");
    p += sprintf(buf + p, "  sigmoid/silu %s%s\n",
                 lz_sig_mode() ? "fixed" : "float",
                 lz_sig_is_auto() ? " (default)" : "");
    p += sprintf(buf + p, "  norms        %s%s\n",
                 lz_norm_mode() ? "fixed" : "float",
                 lz_norm_is_auto() ? " (default)" : "");
    p += sprintf(buf + p, "  scalar       %s%s\n",
                 lz_scalar_mode() ? "fixed" : "float",
                 lz_scalar_is_auto() ? " (default)" : "");
    p += sprintf(buf + p, "  pairing      %s%s\n",
                 lz_pair_mode() ? "on" : "off",
                 lz_pair_is_auto() ? " (auto = on; no hardware question)"
                                   : "");
    buf[p] = '\0';
    return buf;
}
