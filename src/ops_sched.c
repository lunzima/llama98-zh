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
#include "ops_avx2.h"   /* LZ_HAVE_Q8R_AVX2 - without this, lz_q8r_tier's
                           tier 3 arm is compiled out in THIS TU and
                           --kernel avx2 resolves to tier 0 (the scalar
                           path) for both call sites: LZ_KERNEL_AVX2
                           matches neither the SSE2 nor the SSE1 check
                           below. Empty body on Watcom/non-AVX2 builds
                           (this header's own guard), same shape as the
                           SSE2 include above. */
#include "ops_kernel_shared.h"
#include "ops_quant.h"  /* lz_cpu_has_mmx (non-static, from module 1) */
#include "ops_sched.h"
#include "ops_t2_arm.h" /* LZ_T2_ARM_ASM_EXTERN: one home for "this build
                           carries the ARM asm tier" */

/* LZ_GDN_FIXED, LZ_NORM_MAX_N: ops.h. */

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

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86)
static int g_has_avx2 = -1;

/* OSXSAVE + XGETBV + leaf-7 AVX2, in that order - CPUID reporting AVX2
   is not sufficient by itself; the OS must also have enabled YMM state
   saving (XCR0 bits 1:2), or executing an AVX2 instruction traps. Leaf
   1 ECX bit 28 (AVX) is not checked separately: the architecture never
   sets leaf 7 EBX bit 5 (AVX2) without also setting AVX, since AVX2 is
   defined as an extension of the AVX instruction set, so testing AVX2
   alone is already the tighter of the two conditions. On Watcom this
   always returns 0: no Watcom build ever links src/ops_avx2.c, so
   there is nothing here to detect, and the request path
   (LZ_KERNEL_AVX2 accepted by the CLI on every build) must still clamp
   safely rather than claim support. */
int lz_cpu_has_avx2(void) {
#ifdef __WATCOMC__
    return 0;
#elif defined(LZ_FORCE_NO_AVX2)
    /* Test-only override: lets the "requested avx2, don't have it,
       clamp to `have`" branch in lz_kernel_select be exercised on a
       host that genuinely has AVX2 - otherwise that branch could ship
       broken and nothing on this dev machine would ever run it.

       NOT a member of the LZ_<THING>_FORCE_<TIER> family documented
       above LZ_KERNEL_SSE in ops.h (LZ_Q8R_FORCE_SSE etc., mostly
       retired) - those forced a specific tier's BODY to run in place
       of runtime dispatch. This instead forces the CAPABILITY PROBE
       itself to lie, leaving dispatch and every tier body untouched;
       it exists to make an unreachable branch reachable, not to bypass
       one. */
    return 0;
#else
    unsigned ecx1, xcr0, ebx7;
#ifdef LZ_CPUID_NOCACHE
    ecx1 = lz_cpuid1_ecx();
    if (!(ecx1 & (1u << 27))) return 0;
    xcr0 = lz_xgetbv0();
    if ((xcr0 & 0x6u) != 0x6u) return 0;
    ebx7 = lz_cpuid7_ebx();
    return (ebx7 & (1u << 5)) != 0;
#else
    if (g_has_avx2 < 0) {
        ecx1 = lz_cpuid1_ecx();
        if (!(ecx1 & (1u << 27))) {
            g_has_avx2 = 0;
        } else {
            xcr0 = lz_xgetbv0();
            if ((xcr0 & 0x6u) != 0x6u) {
                g_has_avx2 = 0;
            } else {
                ebx7 = lz_cpuid7_ebx();
                g_has_avx2 = (ebx7 & (1u << 5)) != 0;
            }
        }
    }
    return g_has_avx2;
#endif /* LZ_CPUID_NOCACHE */
#endif /* __WATCOMC__ */
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

/* Nothing asks whether this machine's float is slow. There is no
   hardware predicate behind the arithmetic tier: the shipping build is
   fixed-point on every host, and float is reached only by asking for it
   (--fixed off), which is the correctness oracle rather than an
   outcome. docs/fixed-tier-pipeline-breakpoints.md lists which
   float-to-fixed conversions that makes removable and which are forced
   by data regardless - the residual stream's cross-layer dynamic range
   and the SubLN Hadamard's float-only kernel.

   q8r_have_simd below is a different axis - the Q8 row-kernel tier, not
   the arithmetic one - and keeps its CPUID question. */

/* Q8 rounding: 0 scalar, 1 SSE1 (lz_q8round32_sse), 2 SSE2
   (lz_q8round32_simd), 3 AVX2 (lz_q8round32_avx2, gcc-only). The tier is
   read from g_kernel, and LZ_Q8R_FORCE_SSE is gone for the same reason
   LZ_I32FACC_FORCE_SSE below is: `--kernel sse` selects the SSE1 body
   at run time, so the compile-time escape was a second way to say it
   and the two could disagree. kernel_isa_identity_gate's `sse` flagset
   compares the two tiers out of one binary, which the control build
   never did - it was compiled by switch_matrix_gate and run by
   nothing. */

#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
int lz_q8r_tier(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return 0;
#if defined(LZ_HAVE_Q8R_AVX2)
    if (g_kernel == LZ_KERNEL_AVX2) return 3;
#endif /* LZ_HAVE_Q8R_AVX2 */
#if defined(LZ_HAVE_Q8R_SIMD)
    if (g_kernel == LZ_KERNEL_SSE2) return 2;
#endif /* LZ_HAVE_Q8R_SIMD */
#if defined(LZ_HAVE_Q8R_SSE)
    /* g_kernel, not CPUID: `lz_cpu_has_sse()` answers yes for --kernel mmx
       on every machine in the suite, so the mmx tier would run the SSE1
       body and nothing would say so. */
    if (g_kernel == LZ_KERNEL_SSE) return 1;
#endif /* LZ_HAVE_Q8R_SSE */
    return 0;
}
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */

/* Attention wsum's chunk fold: 0 scalar, 1 SSE1, 2 SSE2, 3 AVX2. Same
   shape as lz_q8r_tier above and for the same reason - the tiers here
   are a scalar convert (cvtsi2ss, SSE1), a packed four-wide one
   (cvtdq2ps, SSE2) and a packed eight-wide one (vcvtdq2ps, AVX2), all
   of them the same correctly rounded conversion (see
   ops_kernel_shared.h's LZ_I32F_ACC32 note).

   LZ_I32FACC_FORCE_SSE is gone. It existed because the SSE1 cell was
   unreachable on any machine with SSE2 - every machine that runs this
   suite - and a path that cannot be selected cannot be validated. That
   is what `--kernel sse` is, so the compile-time escape was a second
   way to say it and the two could disagree. */
#if defined(LZ_HAVE_I32FACC_SSE2) || defined(LZ_HAVE_I32FACC_SSE) || \
    defined(LZ_HAVE_I32FACC_AVX2)
int lz_i32facc_tier(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return 0;
#if defined(LZ_HAVE_I32FACC_AVX2)
    if (g_kernel == LZ_KERNEL_AVX2) return 3;
#endif /* LZ_HAVE_I32FACC_AVX2 */
#if defined(LZ_HAVE_I32FACC_SSE2)
    if (g_kernel == LZ_KERNEL_SSE2) return 2;
#endif /* LZ_HAVE_I32FACC_SSE2 */
#if defined(LZ_HAVE_I32FACC_SSE)
    if (g_kernel == LZ_KERNEL_SSE) return 1;
#endif /* LZ_HAVE_I32FACC_SSE */
    return 0;
}
#endif /* LZ_HAVE_I32FACC_SSE2 || LZ_HAVE_I32FACC_SSE */

/* ---- THE float knob -------------------------------------------------- */

/* ONE axis for every fixed-point site the engine has, and one is not an
   accident of how it grew. Seven are possible - a --recur-write,
   --conv-write, --attn-write, --epi-write, --sig-write, --norm-write
   and --scalar-fixed, each with its own global, selector and
   is_default, with --fixed off|all a table-driven group setter over
   the lot. build/float_knob_gate.sh asserts that none of those seven
   names is accepted.

   Seven axes could only ever buy MIXED settings, and mixing is the one
   thing that measured worse than either end. Paired NLL over 1262
   positions on kmr20/zh+en, one tier floated at a time against the
   all-fixed default: six of the seven arms came out WORSE than both
   all-fixed and all-float, and all-fixed against all-float is +3.49e-05
   at t=0.01, i.e. the two ends are indistinguishable while every point
   between them is measurably off. What the seven axes were selling was
   a menu of ways to be worse.

   So the knob that remains is the one the mixing was hiding: float or
   fixed, whole engine, no in-between. Float exists as the arm you pin
   when cross-checking operator precision against another
   implementation - llama.cpp, HF transformers, vLLM, SGLang - which is
   a measurement path, not a deployment choice. Nothing selects it
   automatically on any platform.

   THE PER-TOKEN NLL CROSS-CHECK AGAINST HF TRANSFORMERS HAS BEEN RUN:
   .prof/bench_quality_check.py compares llama98's Q8 export against
   transformers' f32 on the same token ids and reports the PAIRED
   per-token NLL difference (its header spells out why paired, not
   marginal). What it does NOT do is the finer per-layer-activation
   comparison against llama.cpp/vLLM/SGLang - .prof/bench_vs_llamacpp.py
   names llama.cpp but measures only SPEED, and no vLLM/SGLang reference
   runs on this machine. `--fixed off` is the arm to pin when someone
   has those. What the tree ALSO checks today is internal: gcc against
   Watcom, MMX against SSE2 against C, x87 against ARM soft-float, all
   bit-identical (build/xcheck_gate.sh). That answers "the tiers agree"
   and, with bench_quality_check, "the engine agrees with PyTorch at
   logit level".

   NOT file-static: lz_scalar_mode is a macro in ops.h so that softmax's
   per-element lz_exp does not pay a cross-TU call for its tier, and
   that macro reads this. This file is still the only writer. */
int lz_float_ref_g = 0;

const char *lz_float_ref_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "all") == 0) lz_float_ref_g = 0;
    else if (strcmp(name, "off") == 0) lz_float_ref_g = 1;
    else return NULL;
    return lz_float_ref_g ? "off" : "all";
}

int lz_float_ref(void) { return lz_float_ref_g; }

/* ---- fast float semantics ------------------------------------------- */

/* ON by default (RTZ+FTZ is the shipping float semantics; off restores
   RNE for a diagnostic run).

   It is not an x86 speed knob, measured: FTZ/DAZ move the zh fixture's
   logits by zero bytes and its runtime by -0.7%, inside a 110ms sigma,
   because x86 scalar float is already one hardware instruction. It is
   the x86 half of the fast-mode pairing: on ARM the same arithmetic is
   LZ_SOFTFP_FAST (round toward zero + flush denormals, compiled in),
   and the two describe ONE rounding, so a fast-mode ARM run byte-
   compares with a fast-mode x86 run the way the strict pair already
   does. See that switch's comment in lz_softfp.c for the coverage and
   the one surviving difference (NaN sign, unreachable in practice).

   THE X86 SIDE MUST NOT AUTO-VECTORISE for the comparison to hold:
   gcc -O2 turns a scalar float reduction into addps/mulps, which
   reassociates it, and under round-toward-zero reassociation changes
   the result where round-to-nearest happened to match. The fast-parity
   build therefore passes -fno-tree-vectorize; the pairing is about the
   arithmetic, and the ARM side has no SIMD to reassociate with.

   x87 (Watcom) can do the rounding half but has no FTZ/DAZ. That gap is
   harmless here and measured rather than assumed: setting FTZ/DAZ on
   the SSE build leaves every logit byte unchanged, so denormals never
   reach a result on this path. */
int lz_fastfp_g = 1;   /* default ON: round-toward-zero is the shipping semantics */

const char *lz_fastfp_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "on") == 0) lz_fastfp_g = 1;
    else if (strcmp(name, "off") == 0) lz_fastfp_g = 0;
    else return NULL;
    return lz_fastfp_g ? "on" : "off";
}

int lz_fastfp(void) {
    /* On ARM the compile flag ORs in so --fastfp off cannot fight the
       compile-time RTZ soft-float. Elsewhere LZ_SOFTFP_FAST is 0. */
    return lz_fastfp_g || LZ_SOFTFP_FAST;
}

/* DEFAULT ON WHERE THERE IS NO FPU, OFF WHERE THERE IS, and the split is
   measured rather than assumed. Widening a stored bf16 costs about five
   integer instructions per element. What that is worth depends entirely
   on what it is being compared against:

     ARM soft-float   a MAC is __aeabi_fmul (33) + __aeabi_fadd (64),
                      so the widening is ~5% on top - and the tensor
                      occupies half the RAM, on a 64MB part whose
                      bottleneck is DDR. Clearly worth it.
     x86 with SSE     a MAC is a vector instruction, and the widening
                      loop is scalar. Measured on cpt100m-frozen, 60
                      tokens: 12.6 ms/token expanded against 37.5
                      ms/token widened-on-read. THREE TIMES slower, to
                      save memory a desktop does not need.

   Loading is faster either way (189 ms against 213 ms for one token,
   half the bytes read), so this is a forward-pass trade, not a load one.

   PER-TARGET DEFAULTS ARE SAFE HERE, which they would not be for any
   other knob in this file: the widening is lossless - `bits << 16`, the
   same value the expand-at-load path produces - so an x86 run with it
   off and an ARM run with it on still produce identical logits.
   build/bf16_store_gate.sh asserts exactly that equality, and
   build/arm/parity_gate.sh compares the two targets. A knob that
   changed results could not do this.

   Either target can be told otherwise with --bf16-store; the switch is
   not target-locked, only its default is. */
#if defined(__arm__) || defined(LZ_NO_FPU)
int lz_bf16_store_g = 1;
#else
int lz_bf16_store_g = 0;
#endif /* __arm__ || LZ_NO_FPU */

const char *lz_bf16_store_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "on") == 0) lz_bf16_store_g = 1;
    else if (strcmp(name, "off") == 0) lz_bf16_store_g = 0;
    else return NULL;
    return lz_bf16_store_g ? "on" : "off";
}

int lz_bf16_store(void) { return lz_bf16_store_g; }

/* ---- GDN recurrence tier: BOTH passes -------------------------------- */

/* Pass 1 resolves from the same knob as pass 2, so the float arm is a
   complete one for the recurrence: a pass-1 tier with no runtime value
   would leave the float oracle running fixed arithmetic for the larger
   of the two passes.

   LZ_GDN_FIXED (ops.h) is the COMPILE switch deciding whether the fixed
   bodies exist at all; this is the runtime one. Both must say yes. */
int lz_gdn_p2_mode(void) {
#if LZ_GDN_FIXED
    return !lz_float_ref_g;
#else
    return 0;
#endif /* LZ_GDN_FIXED */
}

int lz_gdn_mode(void) { return lz_gdn_p2_mode(); }

/* ---- conv1d tier ----------------------------------------------------- */

int lz_conv_mode(void) {
#if LZ_CONV_FIXED
    return !lz_float_ref_g;
#else
    return 0;
#endif /* LZ_CONV_FIXED */
}

/* ---- attention tier -------------------------------------------------- */

/* Two questions, one answer. lz_attn_mode is the scoring/wsum bitmask
   that the kernels branch on; lz_attn_int is whether the wsum hands its
   int64 accumulator straight to lz_quantize_q8_int64 instead of
   draining to float and letting the quantizer re-round. The int exit
   rides on the fixed wsum, so they cannot disagree - one global answers
   both, and there is no second place for them to diverge.

   WHAT IT COSTS, measured over 3786 paired positions (kmr20,
   kunmoe-v2-t2-probe, _armgate2 x zh+en): +5.09e-04 NLL at t=0.65
   against the float exit - indistinguishable - for 8,064 fewer float
   ops per forward, of which the sigmoid row alone accounts for 10,688
   down to 448. */
int lz_attn_mode(void) { return lz_float_ref_g ? 0 : LZ_ATTN_FIXED; }

int lz_attn_int(void) { return !lz_float_ref_g; }

/* ---- sigmoid/silu tier ----------------------------------------------- */

int lz_sig_mode(void) { return !lz_float_ref_g; }

/* ---- norm tier ------------------------------------------------------- */

/* lz_norm_int kept as its own name rather than folded into its call
   sites: they read `lz_norm_int() && lz_norm_can_fixed(n) && ...`, and
   the name is what tells a reader WHICH of the two fixed output loops
   the branch selects - lz_norm_can_fixed alone would not. */
int lz_norm_mode(void) { return !lz_float_ref_g; }

int lz_norm_int(void) { return lz_norm_mode(); }

/* LZ_NORM_MAX_N: ops.h. */

int lz_norm_can_fixed(int n) {
#if LZ_NORM_FIXED
    return lz_norm_mode() && n > 0 && n <= LZ_NORM_MAX_N;
#else
    /* The fixed bodies are not in this build, so no value of --fixed
       can make them runnable. Saying no here is what keeps the call
       sites off a compiled-out path; norm_ss_fixed declines for the
       same reason. */
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

/* LZ_EPI_FIXED is the compile switch, the same shape LZ_GDN_FIXED and
   LZ_CONV_FIXED have. It exists because --kernel ref structurally has
   no fixed epilogue - lz_matmul_xq_nt routes LZ_KERNEL_REF to
   matmul_scalar_ref before any epilogue is selected - so ref and sse2
   differ on the shipping default for that one reason, and
   build/kernel_isa_identity_gate.sh needs a build where they do not in
   order to compare every OTHER fixed kernel against its C reference.
   A per-site runtime flag would serve that; a whole-engine float switch
   does not, because floating everything takes conv, norms, sigmoid,
   recurrence and attention out of the comparison, which is the coverage
   the row exists for. Hence the compile-time switch here. */
int lz_epi_mode(void) {
#if LZ_EPI_FIXED
    return !lz_float_ref_g;
#else
    return 0;
#endif /* LZ_EPI_FIXED */
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
/* SSE1 sits between MMX and SSE2 here: a PIII or Athlon auto-detects as
   MMX, so without this tier it would reach the SSE1 bodies through CPUID
   behind its back. The kernels are the same ones; the tier has the name
   of what it runs, so --kernel reports it and lz_kernel_select can be
   asked for it. */
#elif defined(__WATCOMC__)
    unsigned edx = lz_cpuid1_edx();
    if (edx & (1u << 26)) return LZ_KERNEL_SSE2;
    if (edx & (1u << 25)) return LZ_KERNEL_SSE;
    if (edx & (1u << 23)) return LZ_KERNEL_MMX;
    return LZ_KERNEL_REF;
#elif defined(LZ_ROW_SSE2_EXTERN)
#if LZ_HAVE_AVX2_TU
    /* The gcc ceiling, above SSE2. Two answers, because the kernels can
       be absent from a link the CPU would run them in: the machine's
       (lz_cpu_has_avx2) and the build's (LZ_HAVE_AVX2_TU) - the same
       pair lz_kernel_select clamps an explicit --kernel avx2 against. */
    if (lz_cpu_has_avx2()) return LZ_KERNEL_AVX2;
#endif /* LZ_HAVE_AVX2_TU */
    if (lz_cpuid1_edx() & (1u << 26)) return LZ_KERNEL_SSE2;
    if (lz_cpu_has_sse()) return LZ_KERNEL_SSE;
#if defined(LZ_DOT_MMX_EXTERN)
    return LZ_KERNEL_MMX;
#else
    return LZ_KERNEL_REF;
#endif /* LZ_DOT_MMX_EXTERN */
#elif defined(LZ_DOT_MMX_EXTERN)
    if (lz_cpu_has_sse()) return LZ_KERNEL_SSE;
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
#elif defined(__WATCOMC__) || \
      (defined(LZ_DOT_MMX_EXTERN) && defined(LZ_ROW_SSE2_EXTERN))
        /* Asking DOWN is always honoured - that is what the knob is for,
           and running the SSE1 tier on a machine that also has SSE2 is
           the only way to compare them. Asking UP clamps to what CPUID
           found: sse2 needs SSE2, and sse needs SSE, which an SSE2
           machine also has.
           Each test names the CAPABILITY its tier needs rather than
           comparing against `have`: have is the best of four now, so
           `have == LZ_KERNEL_SSE2` would promote an explicit
           --kernel sse2 to avx2 on exactly the machines that have it,
           and the four-way bit-identity comparison would spend its run
           comparing one tier against itself.
           AVX2 asks two questions, the machine's and the link's: a tier
           with no bodies falls back per site - table pickers to SSE2,
           the inline tests to MMX, the fixed-point tiers to C - so one
           run would mix three implementations under one tier name. The
           CLI reports the fallback. */
        if (which == LZ_KERNEL_AVX2 &&
            (!lz_cpu_has_avx2() || !LZ_HAVE_AVX2_TU))
            which = have;
        else if (which == LZ_KERNEL_SSE2 && !(lz_cpuid1_edx() & (1u << 26)))
            which = have;
        else if (which == LZ_KERNEL_SSE && !(lz_cpuid1_edx() & (1u << 25)))
            which = have;
        g_kernel = which;
#else
        g_kernel = (which == LZ_KERNEL_REF) ? LZ_KERNEL_REF : have;
#endif /* __WATCOMC__ */
    }
    return g_kernel;
}

const char *lz_kernel_name(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_SSE2) return "sse2";
    if (g_kernel == LZ_KERNEL_SSE) return "sse";
    if (g_kernel == LZ_KERNEL_MMX) return "mmx";
    if (g_kernel == LZ_KERNEL_ARM_ASM) return "arm-asm";
    if (g_kernel == LZ_KERNEL_ARM) return "arm-c";
    if (g_kernel == LZ_KERNEL_AVX2) return "avx2";
    return "ref";
}

const char *lz_build_paths(void) {
#if defined(LZ_T2_ARM_ASM_EXTERN)
    return "arm-c+arm-asm";
#elif defined(__WATCOMC__)
    return "mmx-asm+sse2-asm";
#elif defined(LZ_DOT_MMX_EXTERN) && defined(LZ_ROW_SSE2_EXTERN)
#if LZ_HAVE_AVX2_TU
    return "mmx-intrin+sse2-intrin+avx2-intrin";
#else
    return "mmx-intrin+sse2-intrin";
#endif /* LZ_HAVE_AVX2_TU */
#elif defined(LZ_DOT_MMX_EXTERN)
#if LZ_HAVE_AVX2_TU
    return "mmx-intrin+avx2-intrin";
#else
    return "mmx-intrin";
#endif /* LZ_HAVE_AVX2_TU */
#elif defined(LZ_ROW_SSE2_EXTERN)
#if LZ_HAVE_AVX2_TU
    return "sse2-intrin+avx2-intrin";
#else
    return "sse2-intrin";
#endif /* LZ_HAVE_AVX2_TU */
#else
    return "scalar";
#endif /* __WATCOMC__ */
}

const char *lz_kernel_tier(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return "ref";
    if (g_kernel == LZ_KERNEL_AVX2) return "avx2-intrin";
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

/* CROSS-PLATFORM pins: the FLOAT arm, for comparing one code path
   across SSE2 / SSE / MMX / x87 / ARMv5TE soft-float and across gcc and
   Watcom. Not a report of what is running, and not the shipping
   configuration.

   This string names ONE axis, not seven tiers at a mixed setting -
   which would pin a configuration nothing ships and no gate defaults
   to. `--fixed off` for the same reason the seven axes are one: the
   float
   arm is what a cross-implementation precision check needs, and any
   setting between the two ends is neither the shipping engine nor a
   clean oracle.

   NOT USABLE AS A SINGLE-PLATFORM REGRESSION PIN. The shipping default
   is fixed everywhere, so pinning from here switches OFF the very
   kernels a change may have touched - a regression run pinned from this
   string once compared two builds with both new fixed kernels disabled
   and looked clean. Regress on the default, or pin `--fixed off`
   explicitly when the float path is the thing under test. Cross-target
   gates want BOTH arms: the fixed arm is integer and must match
   bit-for-bit, the float arm is where rounding regimes are allowed to
   differ and the tolerance lives. lz_tier_report() reads live state. */
const char *lz_tier_pins(void) {
    /* --fastfp is here even though it resolves from a default rather
       than from AUTO: it changes float SEMANTICS, so a comparison that
       left it implicit would silently compare a strict run against a
       fast one if the default ever moved. Pinning it costs one token
       and removes that failure mode. */
    return "--kernel ref --fixed off --pair off --prefetch none "
           "--fastfp off";
}

/* Every arithmetic row resolves from one global, so they cannot
   disagree and printing seven of them would be printing the same bit
   seven times. They are still listed individually: the row names are
   what a reader matches against a census row or a kernel-matrix cell,
   and "the engine is in fixed mode" does not tell anyone that the
   attention wsum is the thing feeding lz_quantize_q8_int64. */
const char *lz_tier_report(void) {
    static char buf[512];
    const char *t = lz_float_ref_g ? "float" : "fixed";
    const char *d = lz_float_ref_g ? "" : " (default)";
    int p = 0;
    p += sprintf(buf + p, "  recurrence   %s%s\n", t, d);
    p += sprintf(buf + p, "  conv1d       %s%s\n", t, d);
    p += sprintf(buf + p, "  attention    %s%s\n",
                 lz_float_ref_g ? "float" : "fixed+int", d);
    p += sprintf(buf + p, "  epilogue     %s%s\n", t, d);
    p += sprintf(buf + p, "  sigmoid/silu %s%s\n", t, d);
    p += sprintf(buf + p, "  norms        %s%s\n", t, d);
    p += sprintf(buf + p, "  scalar       %s%s\n", t, d);
    p += sprintf(buf + p, "  pairing      %s%s\n",
                 lz_pair_mode() ? "on" : "off",
                 lz_pair_is_auto() ? " (auto = on; no hardware question)"
                                   : "");
    buf[p] = '\0';
    return buf;
}
