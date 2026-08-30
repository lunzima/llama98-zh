/* ops_kernel.c - engine infrastructure: the f32 census accounting, the
   MMX CPUID probe, the kernel coverage matrix, and the x87 stack-empty
   debug gate.

   Pure code motion out of src/ops.c (no logic change). The shared
   epilogue helpers (epi_shr_half_u / epi_shr_half / lz_act_gs) stayed
   behind in ops.c.

   Compile environment mirrors ops.c exactly: same includes, and like
   ops.c this TU is on the ordinary CPU floor and must never be
   compiled with __MMX__. */

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

#if LZ_F32COUNT
float lz_f32_class[LZ_FC_N][LZ_FCC_N];
int lz_f32_reached[LZ_FC_N];
#endif /* LZ_F32COUNT */

static const char *const LZ_FC_NAMES[LZ_FC_N] = {
    "dequant epilogue", "attention scoring", "attention wsum",
    "softmax", "exp", "sigmoid", "softplus", "conv1d", "norms", "rsqrt",
    "recurrence write", "quantize"
};

static const char *const LZ_FCC_NAMES[LZ_FCC_N] = {
    "mul", "add", "div", "cvt", "cmp", "other"
};

const char *lz_f32_count_name(int site) {
    return (site >= 0 && site < LZ_FC_N) ? LZ_FC_NAMES[site] : "?";
}

float lz_f32_count_get(int site) {
#if LZ_F32COUNT
    int c;
    float tot;
    if (site < 0 || site >= LZ_FC_N) return 0.0f;
    tot = 0.0f;
    for (c = 0; c < LZ_FCC_N; c++) tot += lz_f32_class[site][c];
    return tot;
#else
    (void)site;
    return -1.0f;   /* not "zero work": the build cannot answer */
#endif /* LZ_F32COUNT */
}

float lz_f32_class_get(int site, int cls) {
#if LZ_F32COUNT
    if (site < 0 || site >= LZ_FC_N || cls < 0 || cls >= LZ_FCC_N) return 0.0f;
    return lz_f32_class[site][cls];
#else
    (void)site; (void)cls;
    return -1.0f;   /* not "zero work": the build cannot answer */
#endif /* LZ_F32COUNT */
}

const char *lz_f32_class_name(int cls) {
    return (cls >= 0 && cls < LZ_FCC_N) ? LZ_FCC_NAMES[cls] : "?";
}

int lz_f32_count_reached(int site) {
#if LZ_F32COUNT
    return (site >= 0 && site < LZ_FC_N) ? lz_f32_reached[site] : 0;
#else
    (void)site;
    return 0;
#endif /* LZ_F32COUNT */
}

void lz_f32_count_reset(void) {
#if LZ_F32COUNT
    int i, c;
    for (i = 0; i < LZ_FC_N; i++) {
        for (c = 0; c < LZ_FCC_N; c++) lz_f32_class[i][c] = 0.0;
        lz_f32_reached[i] = 0;
    }
#endif /* LZ_F32COUNT */
}

/* CPUID leaf 1 EDX bit 23 = MMX. Cached for the same reason g_has_sse is:
   the _mm_empty() call sites sit in row/group loops and would turn into a
   serializing cpuid per row otherwise. Bit 23, NOT bit 25: Pentium MMX
   has MMX without SSE, and a machine that has MMX is exactly the machine
   where an unconditional emms is a legal (if redundant) instruction. On a
   machine WITHOUT MMX (486DX, early Pentium) this is false and every
   unconditional emms that followed a scalar row is skipped - the fix for
   "emms is an MMX instruction, so it is #UD on an MMX-less CPU". Defined
   on every target (not just x86): the call sites test it on ARM too, where
   _mm_empty() is a no-op and this returns 0 so the test compiles away. */
/* The cache is not enough on its own, and the reason is the call. This
   was one extern function reading a cached global, and the call sites
   are in row loops: 75,080,736 of them over 603 tokens on kmr20, which
   is 124,512 per token. gcc hoists most of them; Open Watcom hoists
   none, because an extern function may have side effects it cannot see
   - all five sites in ops_matmul.c survive as real calls there.

   So the FAST path is inline, in ops_quant.h, reading g_lz_has_mmx
   directly; this is only the one-time probe it falls back to. Same
   shape as q8_round, and for the same reason: the call and the
   dispatch cost more than the body does. */
int g_lz_has_mmx = -1;

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86)
int lz_cpu_has_mmx_probe(void) {
    g_lz_has_mmx = (lz_cpuid1_edx() & (1u << 23)) != 0;
    return g_lz_has_mmx;
}
#else
int lz_cpu_has_mmx_probe(void) { g_lz_has_mmx = 0; return 0; }
#endif /* __i386__ || __x86_64__ || _M_IX86 */

/* ---- kernel coverage matrix ------------------------------------------

   At the end of the file and outside every #if, because it must exist on
   EVERY target: cli_main.c calls it unconditionally, and the first
   version sat inside `#if defined(__MMX__)` and broke the ARM
   cross-build at link. That build is the one nobody runs, which is
   exactly why it found this.

   The 128-group column is DECLARED, not detected: whether a row kernel
   calls a group-of-four kernel is a compile-time fact inside a static
   function and cannot be read from outside. Both halves have to be
   edited together; the TRAINING tree's tests/test_kernel_matrix.py
   (E:\LLM\llama98-zh, pointed at this tree by LZ_ENGINE - there is no
   tests/ here by policy) compares this against the source, and
   build/kernel_matrix_gate.sh checks the registry below out of the
   shipped binaries. The previous version of this map lived in a
   document and drifted for months. */
#if defined(__WATCOMC__)
#define LZ_G128_Q8  1
#define LZ_G128_Q41 1
#define LZ_G128_Q61 1
#define LZ_G128_Q16 1
#elif defined(LZ_DOT_MMX_EXTERN)
/* The gcc group kernels (dot128_*_mmx) exist and are wired into the row
   functions behind the LZ_G128_GCC knob (default 1, ops_kernel_shared.h)
   - see that knob's comment for the A/B rationale. This column tracks
   the wiring: LZ_G128_GCC=0 reverts it to 'o' and the row functions to
   the per-32 path. The claim is "wired in", not "measured faster" -
   whether the group path wins on any machine is unmeasured. */
#define LZ_G128_Q8  LZ_G128_GCC
#define LZ_G128_Q41 LZ_G128_GCC
#define LZ_G128_Q61 LZ_G128_GCC
#define LZ_G128_Q16 LZ_G128_GCC
#else
/* Neither compiler's group kernels are in this build: dot128_*_mmx sit
   behind LZ_DOT_MMX_EXTERN in src/ops_mmx.c, the #pragma aux twins
   behind __WATCOMC__. The ARM cross-build is the case - it has no row
   kernel for these four formats either, so there is nothing that could
   call a group kernel. Without this branch the ARM binary reports 'G'
   for all four - a column lit on a target whose row is empty, which is
   the exact shape of signal this map exists to refuse. */
#define LZ_G128_Q8  0
#define LZ_G128_Q41 0
#define LZ_G128_Q61 0
#define LZ_G128_Q16 0
#endif /* __WATCOMC__ */
/* T2's group form now exists on all three: dot128_t2_sse2
   (ops_kernel_dot_sse2.h), dot128_t2_mmx (ops_kernel_dot_mmx.h) and
   lz_dot128_t2_asm (ops_mmx.c, the Watcom #pragma aux).

   The argument that kept this at 0 examined the unpack. It said q6_1
   hoists two mask constants across four sub-blocks while T2's unpack has
   only a zero register, which trades pxor for movq one instruction for
   one. Both halves are true, and the group form does not pay there. It
   pays on the HORIZONTAL FOLD: per-32 every sub-block reduces its own
   lanes and extracts, and folding four together costs about half that
   and leaves the results in one register to store whole. The Watcom
   kernel also drops four emms to one, which the per-format twins above
   get for the same reason.

   THE THIRD BRANCH IS NOT OPTIONAL, and the four formats above carry
   the same one for the same reason. On the ARM cross-build neither
   __WATCOMC__ nor the gcc group kernels exist, so a plain #else lights
   this column on a target whose row is empty - kernel_matrix_gate
   reports it as "t2/g128 present but not registered for this build
   class", which is what it did. */
#if defined(__WATCOMC__)
#define LZ_G128_T2  1
#elif defined(LZ_ROW_SSE2_EXTERN)
#define LZ_G128_T2  LZ_G128_GCC
#else
#define LZ_G128_T2  0
#endif /* __WATCOMC__ */

/* ---- the reason registry ---------------------------------------------

   A missing cell must carry a reason, and "still empty"
   must be distinguishable from "impossible". Three classes, one letter
   each in the matrix (ops.h's LZKmCell documents the full alphabet):

     have != 0   some build class carries it. Absent HERE is 'o' (another
                 build has it) and absent THERE is '!' - a gate failure,
                 because that is what a path silently vanishing looks
                 like from inside the binary it vanished from.
     IMPOSSIBLE  no build will ever have it; the audit is below.
     TODO        nobody has written it. Not a defect - a registration.
                 An unregistered cell is '?', which IS a defect.
     DECLINED    it WAS written and measured, and the measurement said
                 the hand cell is not better. Settled, like IMPOSSIBLE,
                 but for a reason that a faster idea could overturn.

   Filling in IMPOSSIBLE where TODO belongs is the expensive mistake:
   it retires the cell permanently and the next person reads it as
   settled. TODO only waits.

   DECLINED exists because that pair could not express f32mm armA.
   IMPOSSIBLE was false - the code compiles and is bit-identical - and
   TODO invited the next reader to spend a day arriving at a number
   already measured. A class that cannot say "tried, worse" makes
   "tried, worse" indistinguishable from "nobody got to it".

   ---- IMPOSSIBLE, cell by cell -----------------------------------------

   sseI, sseA (all five formats). Audited instruction by instruction,
   not by impression:

   What a row kernel executes, every format (src/ops_kernel_dot_mmx.h,
   Watcom twins in src/ops_mmx.c): movq loads; the unpack -
   punpcklbw/punpckhbw for q8_0 and q16_0, pand plus psrlw/psrld/psllw
   plus pxor/por for the 4-bit, 6-bit and 2-bit planes; pmaddwd; paddd;
   then psrlq + paddd + movd to fold the pair. Every one is MMX.

   What SSE1 adds on top of MMX, complete list: pavgb, pavgw, pextrw,
   pinsrw, pmaxsw, pmaxub, pminsw, pminub, pmovmskb, pmulhuw, psadbw,
   pshufw, maskmovq, movntq, prefetchnta/t0/t1/t2, sfence - plus the
   %xmm file, which under SSE1 holds packed float32 ONLY. Integer %xmm
   arrives with SSE2, and that is the sse2I/sse2A column.

   Against the sequence above: none of them is a signed 16x16->32
   multiply-add, so pmaddwd stands; pshufw permutes words and cannot
   zero-extend bytes, so the punpck pair stands; psadbw sums |a-b| over
   UNSIGNED bytes and produces no products; pmulhuw is unsigned and
   throws away the low half a dot product needs; pavg/pmax/pmin/pextrw/
   pinsrw/pmovmskb have no role in a dot product at all. The two that DO
   help are prefetchnta and movntq - and they are ALREADY in the MMX
   kernels (LZ_PFI_NTA, src/ops_kernel_shared.h, emitted as the raw
   bytes 0F 18 00 on Watcom). They are a memory hint, selected by
   --prefetch, which is orthogonal to --kernel by construction.

   Structural half, and it is independent of the ISA argument:
   lz_row_pick (src/ops_matmul.c) never indexes LZ_ROW_SSE_I or
   LZ_ROW_SSE_A, and ops.h has no LZ_KERNEL_SSE value, so `--kernel sse`
   cannot be spelled. A path the CLI cannot select cannot be
   compared, so a kernel written into those two slots would be dead on
   arrival.

   t2/g128. The argument is at LZ_G128_T2 above: q6_1 unpacks the same
   2-bit plane and does have a group kernel because it hoists two mask
   constants across four sub-blocks; t2's unpack is a zero register and
   three immediate shifts, so hoisting trades pxor for movq one for one
   and the group kernel would emit what four separate calls emit.

   ---- TODO -------------------------------------------------------------

   None. Every row unit carries both ARM cells: the C tier and the
   hand-written one, for all five formats (src/ops_arm.c, plus ternary
   in src/ops_t2_arm.c). The eleven remaining empty cells are the sseI
   and sseA columns and t2/g128, all IMPOSSIBLE and all audited above.

   A new format arrives with two empty ARM cells, and filling them is
   two edits, not one: the kernel, and the line in lz_matmul_xq_nt that
   lets its dtype past. That function carries no interception - the list
   of dtypes needing one is empty - so a new format reaches its impl
   with an all-NULL row table and falls back from inside it instead.
   Register the cells as TODO and add the
   interception for that one dtype until the pair exists - otherwise
   the two tiers differ by which fallback they fell through rather
   than by the kernel under test. */
#define LZ_KM_IMPOSSIBLE 1
#define LZ_KM_TODO       2
#define LZ_KM_DECLINED   3
/* NOT APPLICABLE, which is not the same empty as impossible and was
   spelled the same for as long as there were only three states.
   IMPOSSIBLE means the instruction set cannot express the work.
   NA means this COLUMN does not ask a question about this ROW - the
   g128 column is the matmul row kernel's 128-element group hoist, and
   an attention weighted sum whose group is the context length has no
   answer to give it, wrong or right.
   Keeping them apart matters because the headline count is read as a
   work list: eleven of the forty-nine "impossible" cells were category
   errors, so the ISA-blocked surface read 29% larger than it is, and
   nine of the eleven can never be implemented because there is nothing
   there to implement. */
#define LZ_KM_NA         4

/* g128's gcc expectation follows the knob rather than being hand-
   declared: -DLZ_G128_GCC=0 is the documented A/B control arm, the row
   functions really do revert to the per-32 path, and a control build
   that trips the gate is a control build nobody runs - an A/B arm
   with no gate rots. Watcom's group kernels
   have no knob and are always expected. */
#if defined(LZ_G128_GCC) && LZ_G128_GCC
#define LZ_KMB_G128 (LZ_KMB_GCC | LZ_KMB_WAT)
#else
#define LZ_KMB_G128 LZ_KMB_WAT
#endif /* LZ_G128_GCC */

/* A UNIT is one dispatchable kernel family - the row axis. Every unit
   today is a matmul row kernel and is named for its weight format, so
   the grid prints one line per unit and the two look like the same
   thing. They are not, and the ARM columns are why: `--kernel arm-c` /
   `arm-asm` has exactly ONE consumer (matmul_t2_impl's ternary row,
   src/ops_matmul.c). The moment a second operator gets an ARM tier -
   docs/arm-asm-audit.md ranks the candidates by calls per token - a
   single global enum value cannot say WHICH operator has the
   assembly -
   one LZ_KERNEL_MMX covering both Watcom assembly and gcc intrinsics
   would make lz_kernel_name unable to answer what ran. The registry therefore
   keys on the unit, not on the build.

   LZ_KM_ROW units print in the grid. LZ_KM_OP units are registry-only:
   the grid's six characters ARE the six LZ_ROW_* slots, and an operator
   whose dispatch table has a different shape does not belong in them.

   Adding a unit is one line in km_unit, one row in km_reg and one arm
   in km_present. Nothing in build/kernel_matrix_gate.sh changes. */
#define LZ_KM_ROW 0
#define LZ_KM_OP  1

#define LZ_KM_NUNIT 17
#define LZ_KM_NCOL 10
/* The one unit with ARM kernels, by index - km_tab and km_present both
   need it, and a named constant is what stops a reorder of km_unit from
   silently moving them onto another format. */
#define LZ_KM_UNIT_T2 4
/* The units with ARM row kernels, same reason: km_arm_row reads them,
   and a named constant is what stops a reorder of km_unit from moving
   an ARM cell onto a format that has none. */
#define LZ_KM_UNIT_Q16 3
#define LZ_KM_UNIT_Q8  0
#define LZ_KM_UNIT_Q41 1
#define LZ_KM_UNIT_Q61 2
/* The operator units. Registry-only (LZ_KM_OP), same reason as above:
   km_arm_asm reads them by name, so a reorder of km_unit cannot move a
   filled cell onto an operator that has no assembly. */
#define LZ_KM_UNIT_I32F  5
#define LZ_KM_UNIT_Q8RND 6
#define LZ_KM_UNIT_EXPFX 7
#define LZ_KM_UNIT_ATDOT 8
#define LZ_KM_UNIT_WSUMP 9
#define LZ_KM_UNIT_SIGQ  10
#define LZ_KM_UNIT_NRMSS 11
/* The two Hadamard transforms. TWO units, not one: they are separate
   bodies in separate files serving separate tiers (lz_fwht_i32 is the
   shipping path's, lz_fwht the float reference's), and one row would
   let a kernel appearing for either report the pair as covered. */
#define LZ_KM_UNIT_FWHTI 12
#define LZ_KM_UNIT_FWHTF 13
/* lz_matmul's F32-weight row. A unit of its own rather than a sixth
   format row: the five format rows are selected through LZ_ROW_*, and
   this one is reached by lz_matmul_xq_nt's f32 short-circuit before any
   dtype dispatch runs - a different mechanism, so a different row. */
#define LZ_KM_UNIT_F32MM 14
/* The two largest float census sites, which had no row until now.
   recur is the GDN recurrence (LZ_FC_RECUR, 26.7% of per-token float);
   epi is the matmul epilogue (LZ_FC_EPI, 10.0%). Their absence was not
   a statement that the operators lack kernels - recur has four x86
   bodies - it was that nothing in the table could say either way. */
#define LZ_KM_UNIT_RECUR 15
#define LZ_KM_UNIT_EPI   16
/* Column 0 is ref and columns 7..9 are armC/armA/g128; columns 1..6 are
   the six LZ_ROW_* slots in slot order, so column c reads tab[c - 1]. */
#define LZ_KM_COL_ROW0 1
#define LZ_KM_COL_ARMC 7
#define LZ_KM_COL_ARMA 8
#define LZ_KM_COL_G128 9

static const struct { const char *name; unsigned char kind; }
km_unit[LZ_KM_NUNIT] = {
    { "q8_0",  LZ_KM_ROW }, { "q4_1", LZ_KM_ROW }, { "q6_1", LZ_KM_ROW },
    { "q16_0", LZ_KM_ROW }, { "t2",   LZ_KM_ROW },
    { "i32f",  LZ_KM_OP  }, { "q8rnd", LZ_KM_OP },
    { "expfx", LZ_KM_OP  }, { "atdot", LZ_KM_OP }, { "wsump", LZ_KM_OP },
    { "sigq",  LZ_KM_OP  }, { "nrmss", LZ_KM_OP },
    { "fwhti", LZ_KM_OP  }, { "fwhtf", LZ_KM_OP },
    { "f32mm", LZ_KM_OP },
    { "recur", LZ_KM_OP  }, { "epi",   LZ_KM_OP }
};
static const char *const km_col[LZ_KM_NCOL] = {
    "ref", "mmxI", "sseI", "sse2I", "mmxA", "sseA", "sse2A",
    "armC", "armA", "g128"
};

/* Printed, so one line each; the audit above is the long form. */
/* The standard an empty ISA cell has to meet, because two weaker ones
   were used here before it and both let a writable kernel go unwritten:
   the tier must be unable to express ANY PART of the work. Partial
   relief obliges a kernel - a loop that goes half as fast in SIMD and
   finishes in scalar is still a cell, and "an SSE1 kernel would just be
   an MMX kernel with mulps in front" is a sentence about taxonomy that
   cost nrmss its two cells for as long as it stood.
   AND "THE COMPILER ALREADY DOES IT" IS NOT A REASON. gcc -O2 does
   vectorise the wsum chunk fold on x86-64, and that cell is written
   anyway: an auto-vectorisation has no name the matrix can register, no
   symbol a cross-toolchain comparison can reach, and it disappears when
   a compiler version or a flag moves. It is not a kernel, it is a
   compiler's current mood.
   So o3_probe's zero at -march=pentium3 is evidence in ONE direction
   only. It cannot license an empty cell; only the instruction list can.

   Confirmed by the compiler, not only by the instruction audit. gcc
   -O3 on the same C, build/o3_probe.sh: ALL FIFTEEN probed units emit
   zero packed ARITHMETIC at -m32 -march=pentium3, every one of them
   with a non-empty body, so the zero is the vectoriser declining and
   not the probe missing its symbol. The five weight formats these
   cells belong to are among them - q8_0 at 388 instructions, q4_1 at
   340 - as are lz_attn_score_q8 (393) and lz_attn_wsum_q8 (538), which
   emit 63 and 73 at -march=x86-64.

   THE FIGURE READ 151 AND 122 HERE for one commit and was wrong: the
   pattern counted movaps and movdqa, which is how gcc copies an xmm
   register in ordinary SCALAR float code. A packed MOVE is not
   evidence of vectorisation - separating the two dropped lz_exp from
   "22 packed" to zero arithmetic in 163 instructions.

   And the instruction list, which is the half that licenses the empty
   cell. What the MMX dot kernels execute is pmaddwd, punpcklbw, psllw,
   psrlw and pxor - measured out of src/ops_kernel_dot_mmx.h, not
   recalled. Against that, every integer instruction SSE1 adds:

     pmaxsw pminsw   a saturating clamp; these loops have none
     pavgb pavgw     rounded average; not a MAC
     pmulhuw         UNSIGNED high multiply; the operands are signed
     psadbw          unsigned byte SAD; not a signed MAC
     pshufw          a lane permute; the shifts here are bit-field
                     extraction inside a lane, not a permute
     pextrw pinsrw   scalar lane access; slower than the load it replaces
     pmovmskb        a mask extract with no consumer here
     maskmovq movntq stores; these kernels accumulate in a register

   None of them applies. The one SSE1 instruction that does is
   prefetchnta, and it is already emitted - 14 in the shipped binary,
   from the row loops in ops_matmul.c under --prefetch - so an SSE1
   cell would be the MMX kernel with nothing added.

   That is the test being met: not "gcc declined", which is only
   evidence, but "the tier has no instruction that touches this work". */
static const char km_why_sse1[] =
    "SSE1 adds no signed 16x16->32 MAC and no integer %xmm; its own MMX "
    "additions (pmaxsw/pshufw/psadbw) are real but are not a MAC";
/* The operator units' x86 and group columns. docs/arm-asm-audit.md is
   an ARM-only audit, so it cannot answer for an x86 tier here and this
   says so rather than guessing 'impossible'. */
/* AUDITED NOW, and the answer is the same for all four of the scalar
   leaves: a SIMD cell needs a caller that hands it more than one value,
   and these have none. lz_i32f, q8_round, lz_exp_fixed and sigmoid_q15
   are each called with one scalar from inside somebody else's loop, and
   the loops that ARE vectorizable (the quantize group, the norm element
   loop, the matmul rows) already carry their own kernels - a vector
   twin here would have to be called four values at a time by code that
   does not have four. That is a fact about the call sites, not about
   the ISA, so it does not change with the instruction set and the six
   columns get one reason.

   The two table leaves have a second, independent block: lz_exp_fixed
   and sigmoid_q15 index a table by a value computed per element, and
   neither MMX, SSE1 nor SSE2 has a gather - VPGATHERDD is AVX2. Four
   lanes would have to be scattered back to scalar loads, which is the
   scalar code with extra shuffling in front of it. */
/* This reason initially listed the quantize group, the norm
   element loop and the matmul rows as the only loops that could batch a
   scalar leaf, and concluded that none of them would because each
   carries its own kernel. It missed a fourth: lz_attn_wsum_q8's chunk
   fold, which hands over thirty-two at a time and sits inside a SIMD
   kernel's own epilogue, after the emms. i32f has four x86 cells now.

   What survives is the part that was never about batching. expfx and
   sigq index a table per element, and a per-element table index needs a
   gather - AVX2, which is off the target family entirely. That holds no
   matter who calls them. */
/* SPLIT IN TWO under the standard above. The gather argument closes the
   cells where the LOOKUP is the work, and does not close the ones where
   arithmetic happens around it. sigmoid_q15 ends in a linear
   interpolation - a + (((b - a) * frac) >> 15) - and its two table
   loads stay scalar while that does not have to. lz_exp_fixed has
   arithmetic around its lookup too, but not this arithmetic; see the
   next block, which is where saying otherwise here led.
   THE MULTIPLY IS int16 x int16, which is the fact that decides
   whether this is writable at all, and it was checked rather than
   assumed. g_sigtab holds (int32_t)(s * 32768 + 0.5) for s in [0, 1],
   so a and b are in [0, 32768] and their adjacent difference is in the
   hundreds - 384 steps across the table and a sigmoid slope that never
   exceeds 0.25. frac is (t - idx) * 32768 with t - idx in [0, 1), so
   [0, 32767]. Both operands fit int16, so the product is pmaddwd, which
   is MMX.
   HAD IT BEEN int32 x int32 these cells would be genuinely shut: the
   instruction for that is pmulld and pmulld is SSE4.1, off this target
   family entirely. Worth writing down because reaching for it is the
   obvious first move and it does not exist here.

   ONLY TWO OF THE THREE OPERANDS ARE int16, and mixing that up is the
   trap this note exists to spring. b - a and frac are; `a` ITSELF IS
   NOT - g_sigtab holds up to 32768 exactly (s = 1.0 rounds there, and
   both early exits return it), which is one past INT16_MAX. So the
   sequence keeps a in int32 and adds it with paddd, and only the
   difference and the fraction go through the 16-bit multiply:
     pmullw + pmulhw on (b-a, frac), punpck[lh]wd to rebuild two full
     int32 products, psrad 15, paddd a.
   Eight MMX instructions per four elements against the C's four
   scalar sub/mul/shift/add. Packing `a` into int16 to shorten that
   would silently clamp the one value the table's own endpoint is.

   The win is still small - two lanes on MMX, four on SSE2, against
   scalar loads that stay scalar. Small is not none, and none is the
   bar. */
/* expfx IS NOT THE SAME SHAPE as sigq. lz_exp_fixed does no linear
   interpolation: it is a Q20 Taylor series - s = rq >> 5, then
   cq = QSCALE + ((LN2*s + rnd) >> Q) + ((LN2SQ*s*s + rnd) >> 2Q), then
   prod = (g_exptab[idx] * cq + rnd) >> Q - whose multiplies are
   32 x 32 -> 64, not the 16 x 16 -> 32 that made sigq's tail a pmaddwd.
   That changes which instruction each tier needs:
     SSE2  pmuludq is exactly 32 x 32 -> 64, two lanes, and it needs NO
           sign handling here - every operand is non-negative. r is
           [0,1) by construction, so rq >= 0 and s = rq >> 5 >= 0;
           LN2 and LN2SQ are positive Q20 constants; g_exptab holds
           exp() times QSCALE; cq is QSCALE plus two non-negative
           terms. An earlier version of this note said the sign had to
           be folded in by hand, which would send the next reader down
           a path the arithmetic does not need.
     MMX   has nothing wider than pmaddwd. The product has to be built
           from 16-bit halves with shifts and adds - several
           instructions where SSE2 spends one. Owed, not impossible:
           expressible is the bar, not convenient.

   THE SECOND TAYLOR TERM IS NOT THE 64 x 32 IT LOOKS LIKE, and this
   note said twice that it was. C writes LN2SQ*s*s as
   ((int64)LN2SQ * s) * s, whose first product is 2^33 and whose second
   is therefore 64 x 32 - which pmuludq cannot do. But s = rq >> 5 with
   rq at most 2^20, so s <= 2^15 and s*s <= 2^30 fits int32. Grouped as
   LN2SQ * (s*s) both factors are under 2^32 and pmuludq takes it in
   one, and the two groupings are the SAME NUMBER: integer
   multiplication is associative and exact, and every intermediate here
   is at most 2^48, far inside int64.
   So all four multiplies in this operator are 32 x 32 -> 64:
   LN2*s, s*s, LN2SQ*(s*s), and g_exptab[idx]*cq. On SSE2 that is
   pmuludq four times (two lanes each); on MMX it is the 16-bit-halves
   construction, which is where the cost actually sits.

   The 32-entry table lookup stays scalar on both, same as sigq's. */
/* sigq's SSE1 cells, and this one CLOSES rather than opens. The
   interpolation is psubd, packssdw, pmullw, pmulhw, punpck[lh]wd,
   psrad and paddd - every one of them MMX, at the width MMX has. SSE1
   adds pmaxsw, pminsw, pshufw, pavgw, pmulhuw and psadbw to the MMX
   register file and not one of them appears in that list, so an SSE1
   cell would be the MMX kernel with nothing added. That is the test
   the standard sets, and it is the same instruction-list argument the
   fourteen dot-product cells pass. The wider form is SSE2's, and SSE2
   has its own cell. */
static const char km_why_sigq_sse1[] =
    "SSE1 adds nothing this sequence uses: psubd, packssdw, pmullw, "
    "pmulhw, punpck[lh]wd, psrad and paddd are all MMX, and SSE1's own "
    "additions (pmaxsw/pshufw/psadbw) appear nowhere in it. The wider "
    "form is SSE2's, which has its own cell";
/* km_why_expfx_x86 stood here and is gone with the last cell it
   explained. What it argued is now in the three bodies themselves
   (ops_sse2.c, ops_mmx.c, ops_mmx_sse.c), which is where an
   implemented kernel's reasoning belongs - a reason string is for a
   cell that does NOT exist. The 32-entry table lookup still stays
   scalar at every tier: a per-element index needs a gather and that is
   AVX2, which km_why_sse1's block above covers for both table units. */
/* expfx and sigq only, and the scope line is here because this string
   is wrong for any unit that HAS a group form. i32f's is the wsum
   chunk fold's 32 and q8rnd's is lz_q8round_group_sse's gs, so each of
   those carries a reason naming its own axis instead. For a per-element
   table lookup both clauses below are true: there is no group form, and the
   leaf really has nothing to hoist across. */
static const char km_why_op_g128[] =
    "no 128-element group form exists - a per-element table lookup has "
    "no group to hoist across";
static const char km_why_attn_g128[] =
    "this column is the 128-ELEMENT group; attention's group runs over "
    "context length, which is unbounded, and lz_wsum_group_mmx already "
    "covers it - a different axis, not this cell";
/* norm_ss_fixed walks a whole norm row (up to LZ_NORM_MAX_N), not a
   quantization group, so the 128-element column is not its axis. The
   one group-shaped thing it does - the amax scan - is q8_amax's, and
   q8_amax has its own kernels; billing them here would count them
   twice. */
/* nrmss is the one operator unit that is NOT a one-value leaf, so it
   does not get km_why_op_x86's reason: it is an element loop of the
   same shape lz_quantize_q8 already vectorizes, and whether an x86
   kernel belongs here is a real open question rather than a category
   error. The ARM-only audit cannot answer it either way. */
/* Unlike the scalar leaves this one IS an element loop - x[i]*sc, round
   to int16, accumulate v*v - so the question is which instruction set
   can do all three steps, and the answer splits the six columns rather
   than filling them with one reason.

   SSE2 can, and lz_norm_ss_sse2 is that cell: mulps for the scale,
   cvtps2dq for the round (round-to-nearest under the default MXCSR,
   which is q8_round's own rule), packssdw + pmaxsw for the clamp, and
   pmaddwd for the squares, which is exactly the accumulate this loop
   wants. Gate: .prof/norm_ss_sse2_xcheck.sh.

   SSE1 finishes it too, and lz_norm_ss_sse is that cell: the same five
   steps with cvtps2pi in place of cvtps2dq, so everything after the
   scale lands in %mm and the kernel owes an emms. Four elements a pass
   against eight. This row read "impossible" for as long as it did
   because the reason argued taxonomy - an SSE1/MMX hybrid is still an
   SSE1 cell, it needs SSE1 to run.

   MMX gets a cell too. "MMX alone cannot start it: the scale is a
   FLOAT multiply and MMX is integer-only" answers whether the loop runs
   in ONE pass, which is not what the cell asks. It does not have to.
   lz_norm_ss_pack_mmx takes the int32 an x87 pass
   produced and does the clamp and the pmaddwd sum of squares, which is
   the part MMX can reach; the emms is once per 32 elements rather than
   once per element, which is what separates it from a regression. */

static const char km_why_nrmss_g128[] =
    "this column is the 128-ELEMENT quantization group; norm_ss_fixed's "
    "loop runs over a whole norm row, and the group-shaped part of it "
    "is q8_amax, which carries its own kernels";
/* The Hadamards are the opposite of the scalar leaves above: a butterfly
   loop of pure add/sub over a power-of-two block, which is the shape
   SIMD is best at, and no multiply or constant enters.

   The two transforms take OPPOSITE answers per ISA because one is
   integer and the other is float - which is the whole reason they are
   two units. lz_fwht_i32 is paddd/psubd: MMX does two lanes, SSE2 does
   four, and SSE1 has no integer add at all. lz_fwht is addps/subps:
   SSE1 and SSE2 both do four lanes, and MMX is integer-only. So fwhti
   fills its four x86 cells with TWO bodies and fwhtf with one.

   Neither needs a shuffle. The butterfly at stride `len` pairs y[i]
   with y[i+len], and for every stage with len >= the vector width those
   are two CONTIGUOUS vectors - one load each, one add, one sub, two
   stores. The last stages, where len drops below the vector width, stay
   scalar: an unpack per butterfly would cost more than it saves, and
   they are the cheap ones anyway.

   Both are bit-identical to the C body rather than close, and that is a
   structural claim, not a measurement that happened to come out even:
   each output is one add or one sub of the same pair the scalar loop
   forms, so nothing is reassociated and no constant enters. The gates
   are .prof/fwht_mmx_xcheck.sh and .prof/fwht_f32_xcheck.sh, both with
   a mutation, and .prof/wbuild_xcheck.sh runs the same probes through
   the target compiler. */
static const char km_why_fwhti_sse1[] =
    "impossible: SSE1 has no integer add - paddd is MMX/SSE2. The int32 "
    "butterfly cannot be expressed in xmm under SSE1";
/* Audited with the kernel, which is how the row above it was found to
   be wrong. q8rnd had all six x86 cells marked impossible on the
   scalar-leaf reason, and four of them ship: lz_q8round32_sse
   (ops_mmx_sse.c) and lz_q8round32_simd (ops_sse2.c) on gcc,
   lz_q8round32_sse_asm and lz_q8round32_sse2_asm as #pragma aux on
   Watcom, selected at runtime by lz_q8r_tier. The function is named for
   the thirty-two values it takes at once. Only the MMX pair is
   genuinely out, and for a different reason than the one it carried. */
/* q8rnd DOES have a group form, which the shared operator reason denied
   twice over. lz_q8round_group_sse (src/ops_mmx.h) takes gs and runs
   the whole gs/32-chunk loop - hand it 128 and it runs four chunks. So
   neither "no group form exists" nor "a scalar leaf has no group" is
   true here; the second clause is the one this file already retired at
   km_why_i32f_g128.
   What is true is narrower, and it is what the column asks: this column
   is the matmul row kernel's relationship to a group-of-128 kernel
   (dot128_*), and q8round's group kernel is called by lz_quantize_q8
   per group, not hoisted across sub-blocks by a row. */
static const char km_why_q8rnd_g128[] =
    "this column is the ROW kernel's 128-element group hoist (dot128_*); "
    "q8round does have a group form - lz_q8round_group_sse takes gs and "
    "runs gs/32 chunks - but lz_quantize_q8 calls it per group rather "
    "than a row hoisting it across sub-blocks. A different axis";
static const char km_why_i32f_g128[] =
    "this column is the 128-ELEMENT quantization group; the fold's group "
    "is the wsum chunk's 32, which is where its cells are. Not \"a scalar "
    "leaf has no group\" - it has one, just not this one";
static const char km_why_i32f_mmx[] =
    "impossible: MMX has no integer-to-float conversion at all - the "
    "first one on this ladder is SSE1's cvtsi2ss. lz_i32f_acc32_sse is "
    "that cell";
static const char km_why_fwhtf_mmx[] =
    "impossible: MMX is integer-only and this transform is over floats. "
    "The int32 twin (fwhti) is the one MMX can take";
static const char km_why_fwht_g128[] =
    "this column is the 128-ELEMENT quantization group; the Hadamard's "
    "block is hadamard_o/hadamard_down (up to 512, capped by the "
    "fixed-point headroom) and is a different axis, not this cell";
/* Audited with the kernel. MMX is integer-only and this is a float
   multiply-add, so those two cells cannot exist. SSE1 is where mulps
   and addps arrive, and every SSE2 machine has them - so ONE body
   serves both tiers and the SSE2 cells hold it rather than a copy.
   The ARM cell is lz_matmul_row_arm_asm, a leaf rather than an inline
   block: the fusion's win is per GROUP (one scan, one decline, the
   first group needing no addend at all) and an inline per-element form
   gives it all back in wrapper. 69.8 an element against the C loop's
   78.1, .prof/arm_chain_count.sh mode 27 against 28. */
static const char km_why_f32mm_mmx[] =
    "impossible: MMX is integer-only and this is a float multiply-add. "
    "SSE1 is the first tier that can express it";
static const char km_why_f32mm_g128[] =
    "this column is the 128-ELEMENT quantization group; lz_matmul's "
    "unit of work is a whole weight ROW against a whole activation "
    "vector, a different axis";
/* The four 64-bit-lane cells. SSE1 shares this reason with MMX because
   for THIS operator it adds nothing: its integer additions are pmaxsw,
   pshufw, pextrw and movntq on %mm, none of which is a multiply and
   none of which is a 64-bit add.

   A BODY WAS WRITTEN AND MEASURED, which is why the cell is declined
   rather than impossible - and it is NOT IN THIS TREE. It was built in
   a scratch copy, run against the oracle, and discarded; grep for an
   MMX epi kernel here and there is none. "The body exists" is what this
   said, and a reader taking it at its word would go looking for a file
   to enable. The measurements below stand; the code has to be written
   again. The intrinsics version passes the oracle's
   20106 cases and costs 19 instructions an element on the 32-bit i486
   build against the scalar imul loop's 8: the int32 lane bound forces
   a spill every two elements and gcc stacks the accumulators on top of
   that. A Pentium MMX resolves auto to this tier, so shipping it would
   slow down the machine the tier exists for. The pack is {b,ah}
   against {m,m}, which makes one pmaddwd lane the whole high term;
   Open Watcom's mmintrin.h declares __m64 and none of the intrinsics,
   so the twin has to be a #pragma aux.

   That paragraph is a COMMENT and this is a RUNTIME STRING. The cell
   reason has to fit C89's 509-character literal, which VC++ 4.0 holds
   to - it does not get to carry the whole argument. */
static const char km_why_epi_x86[] =
    "declined: the products are pmaddwd's shape but the ACCUMULATOR is "
    "not. paddq is SSE2 even on %mm, so a 64-bit sum has to leave the "
    "vector unit: |ah*m| reaches 2^30 and two of them overflow an int32 "
    "lane, and a lane takes ONE madd result, not two: 2*2^30 lands exactly "
    "on 2^31. Written and measured, then not shipped: 19 instructions an "
    "element against the scalar loop's 8 - slower than pure C, which is "
    "the one exception (regression), not partial relief";
typedef struct { unsigned char have; unsigned char why; const char *text; } LZKmReg;
#define KM_H(mask) { (unsigned char)(mask), 0, (const char *)0 }
#define KM_I(text) { 0, LZ_KM_IMPOSSIBLE, (text) }
#define KM_T(text) { 0, LZ_KM_TODO, (text) }
#define KM_D(text) { 0, LZ_KM_DECLINED, (text) }
#define KM_N(text) { 0, LZ_KM_NA, (text) }
#define KM_ALL (LZ_KMB_GCC | LZ_KMB_WAT | LZ_KMB_ARM)

/* ref is the one DECLARED-present column with no per-format variation:
   lz_matmul_xq_nt routes LZ_KERNEL_REF to matmul_scalar_ref before any
   dtype dispatch, and matmul_scalar_ref_one branches on all five. It
   cannot go empty for one format only, so it is KM_ALL by construction
   and carries no signal of its own - it is here because the registry
   counts C as a tier and a column left out of the table is a column
   nobody can miss. */
/* FLAT, not [LZ_KM_NUNIT][LZ_KM_NCOL], and the reason is a compiler
   limit rather than taste: Open Watcom accepts fourteen rows of this
   shape and reports "E1175: Too many initializers" on the fifteenth,
   with NUNIT and NCOL both verified correct by #error probes at the
   declaration. One brace level fewer is what it takes. km_at() below is
   the only place the two-dimensional index lives. */
static const LZKmReg km_reg[LZ_KM_NUNIT * LZ_KM_NCOL] = {
/*          ref          mmxI              sseI                sse2I             mmxA              sseA                sse2A             armC                 armA                 g128            */
/* q8_0 */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* q4_1 */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* q6_1 */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* q16_0*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* t2   */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* operator units (LZ_KM_OP): registry only, no grid line. ref is the
   shared C body, which is also what --kernel arm-c runs - so armC is
   present on ARM for the same reason ref is present everywhere. */
/* i32f */ KM_H(KM_ALL), KM_I(km_why_i32f_mmx), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_I(km_why_i32f_mmx), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_i32f_g128),
/* q8rnd*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_q8rnd_g128),
/* expfx*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_op_g128),
/* The two attention MAC units DO have x86 kernels, and they are
   macro-selected rather than table-selected, so km_op_present answers
   for them. atdot's assembly is q8_0's leaf, reused rather than copied
   (docs/arm-asm-audit.md section 6); wsump's is its own. That holds at
   BOTH x86 tiers now - the SSE2 cells were empty not because SSE2 could
   not express these but because the two call sites named the MMX body
   with no tier test, so no SSE2 twin was selectable. Gate:
   .prof/attn_sse2_xcheck.sh (kernel against kernel against C) plus
   build/kernel_isa_identity_gate.sh (whether the dispatch reaches it). */
/* atdot*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_attn_g128),
/* wsump*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_attn_g128),
/* sigq */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sigq_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sigq_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_op_g128),
/* nrmss*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_nrmss_g128),
/* Registering these two rows is what makes a gate able to tell "no
   assembly" from "no such operator": without them src/fwht.c could be
   deleted and not a single cell would change. */
/* fwhti*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_fwhti_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_fwhti_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM),       KM_N(km_why_fwht_g128),
/* fwhtf*/ KM_H(KM_ALL), KM_I(km_why_fwhtf_mmx), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_I(km_why_fwhtf_mmx), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM),          KM_N(km_why_fwht_g128),
/* f32mm*/ KM_H(KM_ALL), KM_I(km_why_f32mm_mmx), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_I(km_why_f32mm_mmx), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM),        KM_N(km_why_f32mm_g128),
/* recur's SSE1 cell is NOT km_why_sse1. That string fills the SSE1
   column for every weight-format row above and copying it down is the
   reflex; lz_p2_rows_sse (ops_mmx_sse.c) exists, and a false absent is
   worse than an empty cell because the gate then enforces it. The
   Watcom SSE columns stay with the gcc class: LZ_P2_SSE_EXTERN and
   LZ_P2_SSE2_EXTERN are what km_op_present can key off, and the SSE2
   one is declared under !__WATCOMC__. */
/* recur*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_op_g128),
/* epi  */ KM_H(KM_ALL), KM_D(km_why_epi_x86), KM_D(km_why_epi_x86), KM_H(LZ_KMB_GCC), KM_D(km_why_epi_x86), KM_D(km_why_epi_x86), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_op_g128)
};

#if defined(__WATCOMC__)
#define LZ_KMB_THIS LZ_KMB_WAT
#elif defined(__arm__)
#define LZ_KMB_THIS LZ_KMB_ARM
#else
#define LZ_KMB_THIS LZ_KMB_GCC
#endif /* __WATCOMC__ */

/* Q16's table is the one row inside the MMX block; the other four are
   unguarded and come out all-NULL on a target with no SIMD, which is
   the honest answer there. */
static const lz_rowfn *km_tab(int f) {
    switch (f) {
    case 0: return LZ_ROW_Q8;
    case 1: return LZ_ROW_Q41;
    case 2: return LZ_ROW_Q61;
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN) || defined(__arm__)
    case 3: return LZ_ROW_Q16;
#else
    case 3: return (const lz_rowfn *)0;
#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN || __arm__ */
    case LZ_KM_UNIT_T2: return LZ_ROW_T2;
    /* An operator unit has its own table with its own shape; it gets an
       arm here when it gets a row in km_reg, not a share of this one. */
    default: return (const lz_rowfn *)0;
    }
}

#if defined(__arm__)
/* Does THIS build carry an ARM row kernel for unit f? The list grows as
   cells get filled, in step with lz_matmul_xq_nt's dtype list - that
   line is what makes a filled cell actually run, and the two going out
   of step is a cell that reports present and never executes. */
static int km_arm_row(int f) {
    (void)f;
    /* Every unit, and deliberately without enumerating them: the C tier
       IS the shared body every build carries, so `--kernel arm-c` runs
       the same code `--kernel ref` runs. A list here would have to be
       extended with each new unit and would say nothing a reader could
       check - it was already out of date once. */
    return 1;
}

/* The asm column is answered per unit, not per build: each format's
   hand-written leaf sits behind its own file's EXTERN macro, so a
   toolchain that compiles one and not the other is a state this can
   report rather than one it averages away. */
static int km_arm_asm(int f) {
#if defined(LZ_T2_ARM_ASM_EXTERN)
    if (f == LZ_KM_UNIT_T2) return 1;
#endif /* LZ_T2_ARM_ASM_EXTERN */
#if defined(LZ_ARM_ASM_EXTERN)
    if (f == LZ_KM_UNIT_Q16 || f == LZ_KM_UNIT_Q8 ||
        f == LZ_KM_UNIT_Q41 || f == LZ_KM_UNIT_Q61) return 1;
#endif /* LZ_ARM_ASM_EXTERN */
#if defined(LZ_ARM_PRIM_ASM)
    /* sigq is both sigmoid_q15 entries: one table core, two coordinate
       front ends, one macro. They share a cell because they share the
       thing that makes the cell - splitting them would register two
       rows for one body. */
    if (f == LZ_KM_UNIT_I32F || f == LZ_KM_UNIT_Q8RND ||
        f == LZ_KM_UNIT_EXPFX || f == LZ_KM_UNIT_SIGQ) return 1;
    /* fwhtf's cell is lz_faddsub_arm_asm, an inline block like the four
       above rather than a loop leaf: the butterfly's operands come from
       two strides of the caller's array, so a leaf would pass them
       through memory and lose what the shared unpack bought. */
    if (f == LZ_KM_UNIT_FWHTF) return 1;
#endif /* LZ_ARM_PRIM_ASM */
#if defined(LZ_ARM_ASM_EXTERN)
    /* Both attention MACs, and atdot's cell IS q8_0's leaf - the reuse
       section 6 of the audit asked for, not a second copy. nrmss is a
       loop, so it is a leaf function in src/ops_arm.c rather than an
       inline block: its per-call setup amortizes over a whole row. */
    if (f == LZ_KM_UNIT_ATDOT || f == LZ_KM_UNIT_WSUMP ||
        f == LZ_KM_UNIT_NRMSS || f == LZ_KM_UNIT_F32MM) return 1;
    /* epi's is another loop leaf in src/ops_arm.c. It needs no split of
       its own: SMLAL is the whole operator, which is why this column
       filled before any of the six x86 ones. */
    if (f == LZ_KM_UNIT_EPI) return 1;
    /* fwhti's is a loop leaf in src/ops_arm.c; the float twin's is the
       inline block registered above, under a different guard. */
    if (f == LZ_KM_UNIT_FWHTI) return 1;
    /* recur's is lz_gdn1_x4_arm_asm, the pass-1 4-lane contraction leaf
       (docs/arm-asm-audit.md D2). */
    if (f == LZ_KM_UNIT_RECUR) return 1;
#endif /* LZ_ARM_ASM_EXTERN */
    (void)f;
    return 0;
}
#endif /* __arm__ */

static int km_g128(int f) {
    switch (f) {
    case 0:  return LZ_G128_Q8;
    case 1:  return LZ_G128_Q41;
    case 2:  return LZ_G128_Q61;
    case 3:  return LZ_G128_Q16;
    case LZ_KM_UNIT_T2: return LZ_G128_T2;
    /* Operator units have no group kernel; the registry says so and
       this must agree, or the cell reports present-and-unregistered. */
    default: return 0;
    }
}

/* The x86 tiers of the two attention MAC units. They have no dispatch
   table - both call sites pick their kernel with a #if, so the presence
   answer has to come from the same macros the #if reads, and it does. */
static int km_op_present(int f, int c) {
    /* The GDN recurrence. Four bodies, and until LZ_P2_SSE_EXTERN landed
       the SSE1 one had no macro to key a cell off - which is why an
       operator with x86 assembly showed as having none.
       Columns by build class, like FWHTI below: LZ_P2_MMX_EXTERN and
       LZ_P2_SSE_EXTERN are defined for both toolchains, so naming the
       gcc column unconditionally would make a Watcom build report the
       gcc cell present. */
#if defined(LZ_P2_MMX_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_P2_MMX_EXTERN */
#if defined(LZ_P2_SSE_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_P2_SSE_EXTERN */
#if defined(LZ_P2_SSE2_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_P2_SSE2_EXTERN */
#if defined(LZ_EPI_SSE2_EXTERN)
    /* Both toolchains have an epi SSE2 body, so this one IS split by
       build class - the macro no longer is. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_EPI && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_EPI && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_EPI_SSE2_EXTERN */

#if defined(LZ_FWHT_MMX_EXTERN)
    /* The Hadamard's MMX stage kernel. LZ_FWHT_MMX_EXTERN is defined for
       BOTH toolchains, unlike the dot/wsum _EXTERNs below - so the COLUMN
       has to be chosen by build class here, or a gcc build reports the
       Watcom cell present and the matrix gate fails with "present but not
       registered for this build class", which is exactly what it did. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_FWHTI && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_FWHTI && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_FWHT_MMX_EXTERN */
#if defined(LZ_FWHT_SSE2_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_FWHTI && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_FWHTI && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_FWHT_SSE2_EXTERN */
#if defined(LZ_NORM_SS_SSE2_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_NORM_SS_SSE2_EXTERN */
#if defined(LZ_NORM_SS_SSE_EXTERN)
    /* The SSE1 cell, a body of its own rather than a second column on
       the SSE2 one: cvtps2pi puts the pack, the clamp and the
       multiply-add in %mm, four elements a pass against eight. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_NORM_SS_SSE_EXTERN */
#if defined(LZ_MATMUL_F32_SSE_EXTERN)
    /* FOUR cells from one body, unlike every other entry here: the
       kernel is SSE1 (mulps/addps), and an SSE2 machine runs the same
       instructions - so the SSE2 column holds this body rather than a
       second one, and lz_matmul selects it at either tier. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_F32MM &&
        (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A ||
         c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A)) return 1;
#else
    if (f == LZ_KM_UNIT_F32MM &&
        (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I ||
         c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I)) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_MATMUL_F32_SSE_EXTERN */
    /* The two MMX PACK HALVES, which are cells the "part of it counts"
       standard opened and which now exist. Both take the int32 the
       caller's x87 pass produced and do the part MMX can reach:
       q8round the clamp and the two packs, nrmss the clamp and the
       pmaddwd sum of squares. Registered from the same macros the call
       sites test, so a registration cannot outlive its kernel. */
#if defined(LZ_HAVE_Q8R_PACK_MMX)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_Q8R_PACK_MMX */
#if defined(LZ_HAVE_NORM_SS_PACK_MMX)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_NORM_SS_PACK_MMX */
    /* sigq's interpolation tail, and expfx's Q20 Taylor beneath it.
       SEPARATE MACROS even though the two units are neighbours in the
       registry: they are different kernels with different wiring, and
       registering both from one flag would let either cell outlive its
       own body. expfx's SSE2 pair is wired (lz_exp_fixed_run, reached
       from softmax); its MMX and SSE1 cells are still owed. */
#if defined(LZ_HAVE_LERP_Q15_MMX)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_SIGQ && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_SIGQ && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_LERP_Q15_MMX */
#if defined(LZ_HAVE_LERP_Q15_SIMD)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_SIGQ && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_SIGQ && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_LERP_Q15_SIMD */
#if defined(LZ_HAVE_EXP_Q20_SIMD)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_EXP_Q20_SIMD */
#if defined(LZ_HAVE_EXP_Q20_MMX)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_EXP_Q20_MMX */
#if defined(LZ_HAVE_EXP_Q20_SSE)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_EXP_Q20_SSE */

    /* i32f, the attention wsum chunk fold. Four cells from two bodies
       per toolchain: SSE1 converts one value at a time (cvtsi2ss,
       lz_i32f_acc32_sse) because the packed convert at that tier reads
       an %mm register the caller has just emms'd away from, and SSE2
       converts four (cvtdq2ps, lz_i32f_acc32_simd). */
#if defined(LZ_HAVE_I32FACC_SSE)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_I32F && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_I32F && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_I32FACC_SSE */
#if defined(LZ_HAVE_I32FACC_SSE2)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_I32F && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_I32F && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_I32FACC_SSE2 */

    /* q8rnd, whose four x86 cells were registered impossible on the
       scalar-leaf reason until they were audited against the kernels.
       TWO bodies here, not one shared across tiers like f32mm above:
       SSE1 rounds thirty-two values through two MMX registers
       (lz_q8round32_sse / lz_q8round32_sse_asm) and SSE2 does the same
       thirty-two in xmm (lz_q8round32_simd / lz_q8round32_sse2_asm), so
       each tier holds its own. lz_q8r_tier picks between them. */
#if defined(LZ_HAVE_Q8R_SSE)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_Q8R_SSE */
#if defined(LZ_HAVE_Q8R_SIMD)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_Q8R_SIMD */
#if defined(LZ_FWHT_F32_SSE_EXTERN)
    /* Four cells from one body, same as f32mm just above and for the
       same reason - addps/subps is SSE1, and lz_fwht offers it at both
       the mmx and sse2 tiers. Note the asymmetry with fwhti two entries
       up, which needs TWO bodies for its four cells because MMX's paddd
       is two lanes and SSE2's is four. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_FWHTF &&
        (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A ||
         c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A)) return 1;
#else
    if (f == LZ_KM_UNIT_FWHTF &&
        (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I ||
         c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I)) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_FWHT_F32_SSE_EXTERN */
    if (f != LZ_KM_UNIT_ATDOT && f != LZ_KM_UNIT_WSUMP) return 0;
#if defined(LZ_ATTN_SSE2_EXTERN)
    /* Both attention MACs at the SSE2 tier. Defined for BOTH toolchains
       like LZ_FWHT_MMX_EXTERN above, so the column is chosen by build
       class rather than by the macro. atdot's SSE2 cell is a REUSE of
       the matmul's own Q8 leaf - the same relationship its ARM cell has
       to q8_0's - while wsump's is a new body; the matrix records
       PRESENCE, and a reused body is present. */
#if defined(__WATCOMC__)
    if (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_ATTN_SSE2_EXTERN */
#if defined(__WATCOMC__)
    if (c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#endif /* __WATCOMC__ */
#if defined(LZ_DOT_MMX_EXTERN)
    if (f == LZ_KM_UNIT_ATDOT && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* LZ_DOT_MMX_EXTERN */
#if defined(LZ_WSUM_MMX_EXTERN)
    if (f == LZ_KM_UNIT_WSUMP && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* LZ_WSUM_MMX_EXTERN */
    (void)c;
    return 0;
}

/* Is the cell in THIS binary? Columns 1..6 are read out of the real
   dispatch tables, so deleting a kernel is visible here; ref, armC,
   armA and g128 are declared, for the reasons noted at their columns. */
static int km_present(int f, int c) {
    const lz_rowfn *tab;
    if (c == 0) return 1;
    if (c >= LZ_KM_COL_ROW0 && c < LZ_KM_COL_ROW0 + LZ_ROW_N) {
        tab = km_tab(f);
        if (!tab) return km_op_present(f, c);
        return tab[c - LZ_KM_COL_ROW0] != 0;
    }
    if (c == LZ_KM_COL_G128) return km_g128(f);
#if defined(__arm__)
    /* The ARM row functions are picked inside each matmul_*_impl, not
       from the LZ_ROW_* tables - those six slots are x86 slots. One
       list, read by both ARM columns: a format whose C tier exists but
       whose asm tier does not is a real state (the asm needs
       __GNUC__), a format with the asm and no C tier is not. */
    if (c == LZ_KM_COL_ARMC) return km_arm_row(f);
    if (c == LZ_KM_COL_ARMA) return km_arm_asm(f);
#else
    (void)f;
#endif /* __arm__ */
    return 0;
}

/* The one place the two-dimensional index lives - see km_reg's own
   comment for why that table is flat. */
static const LZKmReg *km_at(int f, int c) {
    return &km_reg[(size_t)f * LZ_KM_NCOL + (size_t)c];
}

static char km_status(int f, int c) {
    unsigned have = km_at(f, c)->have;
    int here = (have & (unsigned)LZ_KMB_THIS) != 0;
    if (km_present(f, c)) return here ? 'x' : '+';
    if (here) return '!';
    if (have) return 'o';
    if (km_at(f, c)->why == LZ_KM_IMPOSSIBLE) return 'i';
    if (km_at(f, c)->why == LZ_KM_TODO) return 't';
    if (km_at(f, c)->why == LZ_KM_DECLINED) return 'd';
    if (km_at(f, c)->why == LZ_KM_NA) return 'n';
    return '?';
}

const char *lz_kernel_matrix(void) {
    static char buf[512];
    int i, j, p = 0;
    for (i = 0; i < LZ_KM_NUNIT; i++) {
        /* Row-kernel units only: the six characters below ARE the six
           LZ_ROW_* slots. An operator unit prints in the registry. */
        if (km_unit[i].kind != LZ_KM_ROW) continue;
        p += sprintf(buf + p, "%-6s", km_unit[i].name);
        for (j = 0; j < LZ_ROW_N; j++)
            buf[p++] = km_status(i, LZ_KM_COL_ROW0 + j);
#if defined(__arm__)
        /* Printed only on the build that has them: a matrix of six dots
           is what "this build has no ternary kernel" looks like, and on
           the ARM cross-build that would be false. */
        buf[p++] = ' ';
        buf[p++] = km_status(i, LZ_KM_COL_ARMC);
        buf[p++] = km_status(i, LZ_KM_COL_ARMA);
#endif /* __arm__ */
        buf[p++] = ' ';
        /* 'G' rather than 'x' when present: the column is a declaration,
           not a tier, and the two should not read alike. */
        buf[p] = km_status(i, LZ_KM_COL_G128);
        if (buf[p] == 'x') buf[p] = 'G';
        p++;
        buf[p++] = '\n';
    }
    buf[p] = '\0';
    return buf;
}

unsigned lz_kernel_build_class(void) { return (unsigned)LZ_KMB_THIS; }

int lz_kernel_cell(int i, LZKmCell *out) {
    int f, c;
    if (!out || i < 0 || i >= LZ_KM_NUNIT * LZ_KM_NCOL) return 0;
    f = i / LZ_KM_NCOL;
    c = i % LZ_KM_NCOL;
    out->unit   = km_unit[f].name;
    out->tier   = km_col[c];
    out->why    = km_at(f, c)->text;
    out->have   = km_at(f, c)->have;
    out->status = km_status(f, c);
    out->kind   = km_unit[f].kind;
    return 1;
}

/* lz_tier_pins and lz_tier_report are in ops_sched.c. */

/* ---- x87 stack-empty gate (debug only) --------------------------------

   See the header comment on LZ_X87_STACK_CHECK (ops.h) for what this is
   and its own limits. Record-only: the first dirty call site wins and
   stays latched for the run, same convention as lz_debug_attn_skip and
   friends - a counter that resets per call could not distinguish "never
   dirty" from "dirty once, then this call site happened to be clean".

   Real implementation only where fnstenv means what this file needs it
   to mean: gcc-family (not Watcom, whose inline asm is a different
   syntax entirely and whose 32-bit x87 output is already known correct
   - this gate exists for gcc's -O2 -mfpmath=387 bug specifically) on an
   x86 target (not the ARM cross-build, which has no x87 unit at all).
   Everywhere else the gate compiles to a stub that never reports dirty -
   stated here rather than left to be discovered, because a silent no-op
   gate is indistinguishable from a passing one otherwise. */
#if defined(LZ_X87_STACK_GATE)
#if !defined(__WATCOMC__) && (defined(__i386__) || defined(__x86_64__))
static const char *g_x87_dirty_where = NULL;
static unsigned g_x87_dirty_tag = 0;

void lz_x87_stack_check(const char *where) {
    unsigned short env[14];
    __asm__ volatile ("fnstenv %0" : "=m" (env));
    if (env[4] != 0xFFFFu && !g_x87_dirty_where) {
        g_x87_dirty_where = where;
        g_x87_dirty_tag = (unsigned)env[4];
    }
}

int lz_x87_stack_dirty(const char **where, unsigned *tag) {
    if (!g_x87_dirty_where) return 0;
    if (where) *where = g_x87_dirty_where;
    if (tag) *tag = g_x87_dirty_tag;
    return 1;
}
#else
/* Stub: LZ_X87_STACK_GATE was requested on a target this check does not
   cover. Compiles and links, always reports clean - NOT because the
   target is known clean, but because there is nothing here to ask. */
void lz_x87_stack_check(const char *where) { (void)where; }
int  lz_x87_stack_dirty(const char **where, unsigned *tag) {
    (void)where; (void)tag; return 0;
}
#endif /* !__WATCOMC__ && (__i386__ || __x86_64__) */
#endif /* LZ_X87_STACK_GATE */
