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

/* ---- GDN pass-1 tier ------------------------------------------------- */

#if LZ_GDN_FIXED
static int g_gdn_fixed = 1;
#endif /* LZ_GDN_FIXED */

int lz_gdn_mode(void) {
#if LZ_GDN_FIXED
    return g_gdn_fixed;
#else
    return 0;
#endif /* LZ_GDN_FIXED */
}

/* ---- q8r_have_simd / lz_q8r_tier ------------------------------------ */

/* THE question every fixed-point tier's AUTO asks.
   CPUID SSE, never MMX. Every Socket 7 part has MMX and no SSE, and
   those are exactly the machines whose x87 is weakest. */
int q8r_have_simd(void) {
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
    return lz_cpu_has_sse();
#else
    return 0;
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */
}

/* lz_soft_float_target: "is this machine's float slow relative to its
   integer". One-bit CPUID proxy. */
static int lz_soft_float_target(void) {
#ifdef LZ_NO_FPU
    return 1;
#else
    return q8r_have_simd() == 0;
#endif /* LZ_NO_FPU */
}

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

/* ---- GDN pass-2 tier ------------------------------------------------- */

#define LZ_P2_FLOAT 0
#define LZ_P2_FIXED 1
#define LZ_P2_AUTO  2
static int g_gdn_p2 = LZ_P2_AUTO;

int lz_gdn_p2_mode(void) {
#if LZ_GDN_FIXED
    if (g_gdn_p2 == LZ_P2_AUTO) return lz_soft_float_target();
    return g_gdn_p2;
#else
    return 0;
#endif /* LZ_GDN_FIXED */
}

int lz_gdn_p2_is_auto(void) { return g_gdn_p2 == LZ_P2_AUTO; }

const char *lz_gdn_p2_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "fixed") == 0) g_gdn_p2 = LZ_P2_FIXED;
    else if (strcmp(name, "auto") == 0) g_gdn_p2 = LZ_P2_AUTO;
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
    if (g_conv == LZ_CONV_AUTO) return lz_soft_float_target();
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
    else if (strcmp(name, "auto") == 0) g_conv = LZ_CONV_AUTO;
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
    int m = (g_attn < 0) ? (lz_soft_float_target() ? 3 : 0)
                         : ((g_attn == LZ_ATTN_TIER_INT) ? 3 : g_attn);
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
    else if (strcmp(name, "auto") == 0) g_attn = -1;
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
    return (g_sig < 0) ? lz_soft_float_target() : g_sig;
}

int lz_sig_is_auto(void) { return g_sig < 0; }

const char *lz_sig_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_sig = 0;
    else if (strcmp(name, "fixed") == 0) g_sig = 1;
    else if (strcmp(name, "auto") == 0) g_sig = -1;
    else return NULL;
    return lz_sig_mode() ? "fixed" : "float";
}

/* ---- scalar tier ----------------------------------------------------- */

static int g_scalar = -1;

int lz_scalar_mode(void) {
    return (g_scalar < 0) ? 1 : g_scalar;
}

int lz_scalar_is_auto(void) { return g_scalar < 0; }

const char *lz_scalar_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_scalar = 0;
    else if (strcmp(name, "fixed") == 0) g_scalar = 1;
    else if (strcmp(name, "auto") == 0) g_scalar = -1;
    else return NULL;
    return lz_scalar_mode() ? "fixed" : "float";
}

/* ---- norm tier ------------------------------------------------------- */

/* ONE ordered axis, not a tier plus an independent boolean: the int
   output only exists on top of the fixed tier, so "int" is its third
   value. As two flags the pair (float, int) was representable and
   silently did nothing - a combination that cannot be spelled cannot be
   mis-set. */
#define LZ_NORM_FLOAT 0
#define LZ_NORM_FIXED 1
#define LZ_NORM_INT   2

static int g_norm = -1;

int lz_norm_mode(void) {
    /* int implies fixed - every caller asking "is this tier fixed" must
       see yes, so the int value folds down here rather than at each site. */
    return (g_norm < 0) ? lz_soft_float_target() : (g_norm != LZ_NORM_FLOAT);
}

int lz_norm_is_auto(void) { return g_norm < 0; }

int lz_norm_int(void) { return g_norm == LZ_NORM_INT; }

const char *lz_norm_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_norm = LZ_NORM_FLOAT;
    else if (strcmp(name, "fixed") == 0) g_norm = LZ_NORM_FIXED;
    else if (strcmp(name, "int") == 0) g_norm = LZ_NORM_INT;
    else if (strcmp(name, "auto") == 0) g_norm = -1;
    else return NULL;
    if (lz_norm_int()) return "int";
    return lz_norm_mode() ? "fixed" : "float";
}

#ifndef LZ_NORM_MAX_N
#define LZ_NORM_MAX_N 4096
#endif /* LZ_NORM_MAX_N */

int lz_norm_can_fixed(int n) {
#if LZ_GDN_FIXED
    return lz_norm_mode() && n > 0 && n <= LZ_NORM_MAX_N;
#else
    (void)n;
    return 0;
#endif /* LZ_GDN_FIXED */
}

/* ---- matmul int output ------------------------------------------------
   A build macro, not a CLI flag: it is measurably inert on the shipping
   configuration. matmul_q8_int_impl/_consume both decline once the fixed
   epilogue is on, and lz_epi_mode's auto turns that on for exactly the
   machines this was meant to help (see docs/matmul-int-output-design.md).
   What remains is a developer A/B, and those belong at build time -
   a runtime knob that cannot change a shipped run is noise in --help. */
#ifndef LZ_MATMUL_INT
#define LZ_MATMUL_INT 0
#endif /* LZ_MATMUL_INT */

int lz_matmul_int(void) { return LZ_MATMUL_INT != 0; }

/* ---- epilogue tier --------------------------------------------------- */

static int g_epi = -1;

int lz_epi_mode(void) {
    return (g_epi < 0) ? lz_soft_float_target() : g_epi;
}

int lz_epi_is_auto(void) { return g_epi < 0; }

const char *lz_epi_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "float") == 0) g_epi = 0;
    else if (strcmp(name, "fixed") == 0) g_epi = 1;
    else if (strcmp(name, "auto") == 0) g_epi = -1;
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
                 lz_gdn_p2_is_auto() ? " (auto)" : "");
    p += sprintf(buf + p, "  conv1d       %s%s\n",
                 lz_conv_mode() ? "fixed" : "float",
                 lz_conv_is_auto() ? " (auto)" : "");
    p += sprintf(buf + p, "  attention    %s%s\n",
                 lz_attn_mode() == 3 ? "fixed" :
                 lz_attn_mode() == 1 ? "scoring" :
                 lz_attn_mode() == 2 ? "wsum" : "float",
                 lz_attn_is_auto() ? " (auto)" : "");
    p += sprintf(buf + p, "  epilogue     %s%s\n",
                 lz_epi_mode() ? "fixed" : "float",
                 lz_epi_is_auto() ? " (auto)" : "");
    p += sprintf(buf + p, "  sigmoid/silu %s%s\n",
                 lz_sig_mode() ? "fixed" : "float",
                 lz_sig_is_auto() ? " (auto)" : "");
    p += sprintf(buf + p, "  norms        %s%s\n",
                 lz_norm_mode() ? "fixed" : "float",
                 lz_norm_is_auto() ? " (auto)" : "");
    p += sprintf(buf + p, "  scalar       %s%s\n",
                 lz_scalar_mode() ? "fixed" : "float",
                 lz_scalar_is_auto() ? " (auto)" : "");
    p += sprintf(buf + p, "  pairing      %s%s\n",
                 lz_pair_mode() ? "on" : "off",
                 lz_pair_is_auto() ? " (auto = on; no hardware question)"
                                   : "");
    buf[p] = '\0';
    return buf;
}
