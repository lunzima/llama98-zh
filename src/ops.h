#ifndef LZ_OPS_H
#define LZ_OPS_H

#include "lz_int.h"   /* lz_i64. Here rather than in
                         model.h so every ops_*.c inherits it from the
                         header it already includes. */
#include "model.h"

/* Operators needed by the Qwen3.5 forward.

   Points where this differs from llama2 and is easy to get wrong:

   1. Qwen3_5RMSNorm scales by (1 + weight), not weight.
      Measured: input_layernorm weights have mean +0.24 and contain
      negatives (min -0.11); multiplying by weight directly distorts
      everything.
   2. The RMSNormGated inside SSM uses the PLAIN weight.
      Measured: mean +0.95, all positive (min +0.52). Both conventions
      coexist in the same model.
   3. l2norm uses sum(x^2), not mean(x^2), and eps is added to the sum.
   4. RoPE acts only on each head's leading rotary_dim dims (64 of 256
      in this model), paired as (i, i + rotary_dim/2) half-rotation,
      not interleaved adjacent elements. */

/* Per-site f32 operation counts. Off unless -DLZ_F32COUNT=1, where it
   costs one add per site execution and is not meant to ship.

   It exists because the census it feeds was hand-computed and could not
   be reconstructed: rows in different units (operations for some sites,
   CALL counts for the transcendental ones, and one lz_exp is 13 f32
   operations), taps counted without their adds, and the two heaviest
   sites - attention scoring and the weighted sum - absent altogether,
   which put the roadmap's ordering 100x wrong.

   TWO RULES, both of which that table broke:

   - The op count per execution is written AT the loop it describes, not
     in a table somewhere else. It cannot drift from code it sits in.
   - Every site counts its OWN arithmetic only. lz_softmax does not
     count the lz_exp it calls; lz_silu does not count lz_sigmoid. So
     the rows sum to the total exactly once, and a site's row means the
     same thing as every other site's.

   Per-token census, kmr20 133M, fixed tiers, 8 tokens, seed 1145141919:

     recurrence write    159936  40.1%
     norms               106774  26.8%
     dequant epilogue     59904  15.0%
     quantize             47328  11.9%
     exp                  14090   3.5%
     attention wsum        7136   1.8%
     attention scoring     1440   0.4%
     rsqrt                 1103   0.3%
     sigmoid                448   0.1%
     softmax                232   0.1%
     softplus            (not run)
     conv1d                  0   0.0%
     total              398390

   A divide counts as one, which understates it on every target this
   engine cares about. Convert (int->float) counts as one, because on a
   soft-float target it is a call like any other. */
enum {
    LZ_FC_EPI, LZ_FC_ATTN_SCORE, LZ_FC_ATTN_WSUM, LZ_FC_SOFTMAX,
    LZ_FC_EXP, LZ_FC_SIGMOID,
    /* Its own row and not a share of LZ_FC_EXP: until this site existed
       lz_softplus was libm double, which the census counts as zero, and
       a row that reads zero for a reason is what keeps that from
       happening again. LZ_FC_NAMES in ops.c is positional - the two
       lists move together. */
    LZ_FC_SOFTPLUS,
    LZ_FC_CONV,
    /* SECOND LARGEST ROW, 26.8% per token on kmr20 with every tier
       fixed, and thirteen billing sites in ops_norm.c of which six run
       on that checkpoint. Their executions, from the coverage build
       with -DLZ_F32COUNT=1 over the same ten tokens the census row
       comes from:

         1920  lz_rmsnorm_gated, fixed+int branch
         1920  lz_l2norm_i16
          720  lz_rmsnorm, fixed-ss branch
          360  lz_rmsnorm's normout table kernel
          360  lz_rmsnorm's scalar tail
           20  lz_rmsnorm_gated, the 16-wide arm

       THE SHARES DO NOT FOLLOW FROM THOSE, and the difference from
       LZ_FC_RECUR is worth stating because the method looks the same.
       Every formula here is linear in n, and n VARIES between calls -
       head width, hidden width, latent width all reach these
       functions - so executions times formula needs a sum of n per
       site, not a count. The recurrence sites happen to be called at
       one geometry, which is why solving their three class totals
       there gives whole numbers and reconstructs the row exactly; the
       same solve here gives n+1 = 64.31 and the right conclusion is
       that the model is wrong, not that n is fractional.
       What closes it is the loop line beside each site: gcov counts
       `for (i = 0; i < n; i++)` sum(n) + count times, which is the
       missing term. DONE, 8-token coverage run on kmr20,
       -DLZ_F32COUNT=1 -O2 -march=x86-64. The LZ_SSN ternaries inside
       LZ_FCX arguments inflate gcov line counts 2x (two branches per
       macro invocation); all counts below are AFTER dividing out that
       inflation. For-loop body counts give sum(n); function call counts
       give reaches. Where no for-loop exists (the normout kernel path),
       sum(n) is inferred from model geometry (n=512 for layer norms).

       site                          reaches  sum(n)   avg n  share
       ------------------------------------------------------------------------
       lz_rmsnorm_gated_int:417        768    49152     64    17.2%
       lz_rmsnorm_int:548                8     4096    512     1.8%
       lz_rmsnorm fixed-ss:607         288   147456    512    50.9%
       lz_rmsnorm kernel:631           288   147456    512    12.7%
       lz_rmsnorm scalar tail:646      288   147456    512     0.0%
       lz_l2norm_i16:778              1536    98304     64    17.5%
                                                         -------
                                               total:   100.1%

       THE DOMINANT SITE IS lz_rmsnorm's fixed-ss branch (607) at 50.9%
       of the row. Together with its kernel add (631) the lz_rmsnorm
       function accounts for 63.6% of NORM float ops. The second and
       third sites (l2norm_i16 at 17.5% and rmsnorm_gated_int at 17.2%)
       are nearly equal. The scalar tail (646) is always zero: when the
       normout kernel runs (n >= 4, always true here), it processes all n
       elements and n-i = 0. lz_rmsnorm_int (548) is a minor site at
       1.8% that the census listing omitted.

       Float ops per site (mul+add+div+cvt, 8 tokens):
         199,680  site 417  (2n+2 mul, 2n+1 add, 1 cvt per call)
          20,528  site 548  (2n+2 mul, 3n+2 add, 1 cvt per call)
         590,976  site 607  (3n+2 mul, n+1 add, 1 cvt per call)
         147,456  site 631  (n4<<2 add per call, kernel's +k2[1])
               0  site 646  (tail empty)
         202,752  site 778  (n+2 mul, 1 add, n+1 cvt per call)
       ---------
       1,161,392  total NORM float ops */
    LZ_FC_NORM, LZ_FC_RSQRT,
    /* The recurrence state write-back - pass 2 only, which is LESS than
       what the recurrence tier covers. Pass 1 has no site in either arm,
       neither the int16 head nor the float body, so this row reads the
       same whichever pass-1 arm runs and the axis's census marginal
       under-reports its float arm by more than the row itself. Measured
       counts for both arms are in .prof/census_audit.

       THE LARGEST ROW IN THE CENSUS, and one row over eight billing
       sites in ops_gdn.c, so the row alone says where to look only if
       there is one site. There is not. Split by running the coverage
       build with -DLZ_F32COUNT=1 (build/cov_run.sh takes it as its
       fourth argument) and multiplying each site's gcov execution count
       by the formula written at it. On kmr20, ten tokens, every tier
       resolved fixed - kd 64, vd 64, ng 2, gvec on - the shares are:

         38.4%  the pass-2 fixed entry (mul kd*3, add vd*2+kd, cvt vd*4)
         23.1%  the p2 group scale (mul ng, add 2*ng), 49,152 executions
         19.2%  the gt==1 delta arm (mul vd*2, add vd*2, cvt vd)
         15.4%  the pass-1 tap (add kd*2)
          3.8%  the delta quantize's magic add (add 32)

       The five reconstruct the row exactly, which is the check that the
       attribution is of this row and not of a similar one. Read the
       shares, not the absolute counts: they are per model geometry. */
    LZ_FC_RECUR,
    /* lz_quantize_q8 and its int/int64/silu twins: the amax*(1/127) mul,
       127/amax div and the per-element round, at every one of their many
       call sites. Before this site existed only a couple of callers
       billed an approximation of this work to their OWN site (ops_gdn.c's
       attention score and recurrence write-back); everywhere else a
       quantize that moved or disappeared was invisible to the census. */
    LZ_FC_QUANT,
    LZ_FC_N
};

/* What kind of f32 operation a site's count is made of. A scalar count
   cannot be reweighted for a target: a divide costs 4.6x an add on
   soft-float ARM and 19.5x on a Pentium, and a compare costs 1.4x and
   19.2x while a plain count bills it nothing. LZ_FCC_OTHER is the
   unclassified bucket every site starts in, so a half-migrated tree
   still reports a correct total - that is what lets sites move here
   one at a time instead of needing a flag day.

   Outside the #if below, like LZ_FC_N above it: cli_main.c's census
   table loops over both enums at compile time regardless of
   LZ_F32COUNT, dispatching on the runtime -1.0 sentinel instead. Only
   the storage and the billing macros need the flag. */
enum { LZ_FCC_MUL, LZ_FCC_ADD, LZ_FCC_DIV, LZ_FCC_CVT, LZ_FCC_CMP,
       LZ_FCC_OTHER, LZ_FCC_N };

#if LZ_F32COUNT
extern float lz_f32_count[LZ_FC_N];
extern int lz_f32_reached[LZ_FC_N];
extern float lz_f32_class[LZ_FC_N][LZ_FCC_N];

/* The comma operator, so both macros stay single expressions and keep
   working in the `if (c) LZ_FC(...);` positions they already sit in. A
   braced block would need every one of those visited. `site` is now
   evaluated twice in LZ_FC and six times in LZ_FCX - every call site
   passes a constant or the plain variable `fc_site`, never an
   expression with a side effect, and must keep doing so.

   LZ_FC is the unclassified form: it bills the whole count to
   LZ_FCC_OTHER, and is what every existing site uses.

   LZ_FCX bills mul/add/div/cvt/cmp by name. All five arguments are
   written even when zero - an omitted class is how a divide goes
   missing - and must sum to what LZ_FC would have billed for the same
   code, since lz_f32_count_get() reports their sum. */
#define LZ_FC(site, ops) (lz_f32_reached[site] = 1, \
                          lz_f32_class[site][LZ_FCC_OTHER] += (float)(ops))
#define LZ_FCX(site, mul, add, div, cvt, cmp) (lz_f32_reached[site] = 1, \
    lz_f32_class[site][LZ_FCC_MUL] += (float)(mul), \
    lz_f32_class[site][LZ_FCC_ADD] += (float)(add), \
    lz_f32_class[site][LZ_FCC_DIV] += (float)(div), \
    lz_f32_class[site][LZ_FCC_CVT] += (float)(cvt), \
    lz_f32_class[site][LZ_FCC_CMP] += (float)(cmp))
#else
#define LZ_FC(site, ops) ((void)0)
#define LZ_FCX(site, mul, add, div, cvt, cmp) ((void)0)
#endif
const char *lz_f32_count_name(int site);
float lz_f32_count_get(int site);
/* Per-class read behind lz_f32_count_get's sum. Bounds-checked at both
   ends for site and cls, same as the accessor above; -1.0 in a
   non-counting build ("this build cannot answer"), 0.0 out of range in
   one that counts. */
float lz_f32_class_get(int site, int cls);
/* "mul"/"add"/"div"/"cvt"/"cmp"/"other", or "?" out of range. */
const char *lz_f32_class_name(int cls);
/* 1 once a site has been billed at least once in this process, else 0.
   Separate from the count because a site that ran and did no work and a
   site that never ran print identically otherwise, and the second one
   means the measurement is of a different code path than intended.
   Without -DLZ_F32COUNT=1 this always returns 0, which reads as "not
   reached" - a confidently wrong answer, unlike lz_f32_count_get's
   -1.0 for "this build cannot answer". Callers must gate on
   lz_f32_count_get(0) >= 0.0 first, same as for the count itself. */
int lz_f32_count_reached(int site);
void lz_f32_count_reset(void);

/* o = x * rsqrt(mean(x^2) + eps) * (1 + w)   - Qwen3_5RMSNorm */
void lz_rmsnorm(float *o, const float *x, const float *w, int n, float eps);

/* o = x * rsqrt(mean(x^2) + eps) * w * silu(g)  - Qwen3_5RMSNormGated
   Order matters: normalize and multiply by weight first, gate last.

   In the fixed tier, o holds the FINAL silu value (Q15 int-multiply,
   value-changing vs the float chain). sig is unused (API stability);
   callers pass NULL. Callers gate on lz_norm_can_fixed(). */
void lz_rmsnorm_gated(float *o, const float *x, const float *g,
                          const float *w, int n, float eps, int32_t *sig);

/* Same operator's fixed tier with INT output: t receives the Q15
   result (no per-element dequant) and *deq the power-of-two scale that
   the float path would have multiplied by, so lz_quantize_q8_int can
   fold it into the stored group scale. Bit-identical to
   lz_rmsnorm_gated(fixed) + lz_quantize_q8. Callers gate on
   lz_norm_can_fixed(n). */
void lz_rmsnorm_gated_int(int32_t *t, const float *x, const float *g,
                          const float *w, int n, float eps, float *deq);

/* Same idea for the PLAIN norm's fixed tier: t receives a Q15
   int-multiply result and *deq its power-of-two scale, for
   lz_quantize_q8_tok_int. UNLIKE lz_rmsnorm_gated_int, this is NOT
   bit-identical to lz_rmsnorm(fixed) + lz_quantize_q8_tok - the plain
   norm's fixed tier has no existing Q15-chain-then-dequant sibling to
   match (its float output loop multiplies x*inv*(1+w) directly, never
   quantizing w or inv), so this int chain is a genuinely different,
   value-changing computation, same footing as the scalar/conv/attn
   fixed tiers. Callers gate on lz_norm_can_fixed(n). */
void lz_rmsnorm_int(int32_t *t, const float *x, const float *w, int n,
                    float eps, float *deq);

/* o = x * rsqrt(sum(x^2) + eps)  - note sum, not mean */
void lz_l2norm(float *o, const float *x, int n, float eps);
/* Same normalisation from an int16 input at 2^-e. Output is float and
   stays float - see the body for why the chain ends there, and why eps
   is what stops the exponent from cancelling the way the mathematics
   says it should. */
void lz_l2norm_i16(float *o, const short *x, int n, int e, float eps);

/* exp = 2^(x*log2e) via bit construction. Default path; -DLZ_EXACT_MATH falls back to libm.

   ~69K exps per token, nearly all from conv1d's SiLU (lin_conv_dim *
   linear layer count) and the FFN SiLU.

   Speed-wise it is NOT faster than libm: measured 11.8 ns each on
   Open Watcom 32-bit, and Watcom's own exp is also 11.8 ns. It stays
   the default for consistency - once the engine carries all
   transcendental functions, the gcc and Watcom builds are bit-identical
   and the Win98 side can differential-check against the dev machine's
   output. libm exp implementations differ; mixing one in would kill
   that property.

   Approximation contract: relative error <= 1e-5 on [-87, 20],
   monotonic non-decreasing, no
   wrap-around at either end. Outside the range (x < -87.3) give 0
   directly - the true value is already subnormal there; in the engine
   only sigmoid(-very large) reaches it, and 0 vs the subnormal is
   unobservable. */
float lz_exp(float x);

/* 1/sqrt(x). Default: 0x5f3759df + 2 Newton iterations (relative error
   4.6e-6); -DLZ_EXACT_MATH falls back to 1/sqrt. */
float lz_rsqrt(float x);

void  lz_softmax(float *x, int n);
float lz_sigmoid(float x);
/* Same value with the argument already in integers (x == v * 2^-e) -
   the int-pipeline entry, for producers that exit int16 at a known
   exponent. Only the fixed sigmoid tier has an integer table to reach,
   so callers ask lz_sig_mode() before choosing the int exit at all. */
float lz_sigmoid_i(int32_t v, int e);
float lz_silu(float x);

/* log(1+e^x). A table+poly float body (lz_softplus_poly, ops_quant.c),
   under the same lz_scalar_mode() tier as lz_exp - it was the one
   scalar transcendental outside it, because it was a pair of libm
   DOUBLE calls. That pair is what LZ_SOFTPLUS_POLY=0 restores. */
float lz_softplus(float x);

/* The libm double pair, kept as the reference lz_softplus is checked
   against rather than as its body. Measured on soft-float
   ARM under QEMU (.prof/softplus_weight.c, ratio only - QEMU models no
   cycles) it is 2.59x the cost of the shipped body in guest
   instructions, and the f32 census scores it zero because it counts
   neither libm nor doubles. 0 restores it, as the control arm every
   number here was measured against; it is also what LZ_EXACT_MATH
   selects. */
#ifndef LZ_SOFTPLUS_POLY
#define LZ_SOFTPLUS_POLY 1
#endif

/* Whether softmax reaches lz_exp_fixed through its RUN entry (see
   ops_quant.h) instead of one call per element. Only the fixed tier
   changes - lz_scalar_mode() gates it - and the two forms are
   bit-identical, including the order the sum accumulates in, so 0 is a
   control arm rather than a second numerical behaviour. It exists
   because the run is what gives the Q20 Taylor cell a caller: without a
   consumer the kernel is unreachable, and an unreachable kernel is a
   kernel no gate can measure. */
#ifndef LZ_SOFTMAX_RUN
#define LZ_SOFTMAX_RUN 1
#endif

/* x87 precision control: set the control word to 24 bits (single
   precision) so x87 float semantics become isomorphic to SSE - the
   prerequisite for bit-identical Watcom and gcc builds. Returns the
   original control word; lz_fpu_float_end restores it. No-op wherever
   scalar float already goes through SSE (x86-64, or any 32-bit build
   that opts into SSE scalar float) - real on Watcom and on 32-bit x86
   gcc/clang built without SSE scalar float (build/x87/build_engine.sh's
   target), both of which actually run float arithmetic on the x87
   stack and need this to agree with SSE bit for bit.

   No double inside the region (incl. libm double routines): PC=24
   compresses double to 24 bits too. lz_exp is all-float, and so is
   lz_softplus now - only its LZ_SOFTPLUS_POLY=0 control arm still has
   to restore precision internally. */
unsigned lz_fpu_float_begin(void);
void     lz_fpu_float_end(unsigned save);

/* Runtime kernel dispatch. On Win98 one binary must run on both PII
   (MMX only) and Pentium M / P4 (SSE2); both asm kernels are compiled
   in and selected once via CPUID.

   The switch point is the matmul layer, not the 32-element inner loop -
   the inner loop stays direct calls, otherwise every 32 MACs would pay
   one indirect call.

   lz_kernel_select(LZ_KERNEL_AUTO) picks via CPUID; specifying a
   concrete tier is for debugging, but it will never select a tier the
   local CPU does not support. Returns the actually selected tier.

   This enum is the CLI/API selector; which of the seven variants
   actually ran is reported by lz_build_paths() and lz_kernel_tier()
   (below) - scalar C, plus MMX/SSE/SSE2 as intrinsics, plus
   MMX/SSE/SSE2 as hand-written asm. Two things are wrong with the
   plain enum, and both are recorded below rather than silently
   patched, because each one hid a real bug.

   WRONG 1 - one enum value, two different code bodies. LZ_KERNEL_MMX
   means #pragma aux asm under Watcom and intrinsics under gcc. So
   lz_kernel_name() cannot say which of the two ran, `--kernel mmx`
   "succeeds" even when the asm was never compiled in, and the
   cross-compiler bit-identity gate stays green throughout - all paths
   are required to agree bit for bit, so two builds running different
   paths look identical. That is exactly how a Watcom build can run
   pure scalar without any gate noticing (see lz_build_paths below).
   asm and intrinsics must be separate selectable values.

   WRONG 2 - "SSE is not a tier, only prefetch is". The premise is
   right (SSE's 128-bit XMM is float-only; PIII's integer SIMD is
   still 64-bit MMX) but the inventory behind the conclusion listed
   exactly one SSE capability, prefetchnta. SSE also adds a whole set
   of INTEGER extensions operating on MMX registers, none of which a
   PII has: pshufw (nibble reordering for Q4_1/Q6_1), pextrw/pinsrw
   (horizontal fold without going through memory), pmaxsw/pminsw
   (absmax in lz_quantize_q8), and movntq (streaming stores, skipping
   the read-for-ownership) - the last one aims straight at the
   measured PIII bottleneck, which is DRAM/bus throughput rather than
   latency. lz_q8round32_sse_asm and lz_q8round32_sse in this same
   file are already a counterexample to "SSE is not a tier".

   > The point of the "PIII tier" note, kept because half of it
   > still holds: an LZ_KERNEL_MMX_SSE value tied to "PIII has SSE, so
   > it can prefetchnta" stuffed two orthogonal dimensions into one
   > enum and missed K6-2/K7 - they have no SSE but do have 3DNow!'s
   > prefetch.
   >
   > Still true: PREFETCH IS ORTHOGONAL to the kernel tier and must
   > stay its own knob. Not true: that this leaves nothing else in an
   > SSE tier.

   SSE also has 128-bit mulps/addps, which could make the float
   epilogue 4-wide. Do NOT: that changes the association order of a
   float reduction, and the tiers owe each other bit-identity
   (see the "no 4-way
   float accumulator across groups" note in the r>1 path). A
   vectorized epilogue can only ever be a separate tier that is bit
   identical to itself.

   MEASURED, and worth stating because a plausible argument says the
   opposite: on a real Pentium M the SSE2 path beats the MMX path by
   more than 20%, even though Pentium M splits a 128-bit SSE2 op into
   two micro-ops and therefore has the SAME nominal throughput as
   64-bit MMX. Width is not the only variable - XMM has 16 registers
   against MMX's 8, SSE2 does not alias the x87 stack (so the emms
   accounting for who owes an emms goes to zero), and one 128-bit
   instruction occupies one decode slot where two 64-bit ones occupy
   two.

   So do NOT "simplify" Pentium M back onto the MMX kernel on the
   grounds that its 128-bit ops are 2 uops. That reasoning is exactly
   what the measurement refutes. */
#define LZ_KERNEL_AUTO 0
#define LZ_KERNEL_REF  1
#define LZ_KERNEL_MMX  2
#define LZ_KERNEL_SSE2 3

/* SSE1, and WRONG 2 above is what it closes. Five sites had an SSE1
   kernel reachable only from CPUID, each carrying the same sentence -
   "there is no LZ_KERNEL_SSE, and on a PIII g_kernel is LZ_KERNEL_MMX
   while the machine does have SSE" - and each carrying a compile-time
   escape so the cell could be tested at all: LZ_Q8R_FORCE_SSE,
   LZ_I32FACC_FORCE_SSE, LZ_P2_FORCE_MMX, LZ_EXP_Q20_FORCE_MMX,
   LZ_NORM_SS_FORCE_MMX. Five separate control BUILDS for one missing
   selector value. Four are now deleted - the P2, I32FACC, EXP_Q20 and
   Q8R ones - because a compile-time escape beside a runtime tier is a
   second way to say the same thing, and two ways can disagree. The
   NORM_SS pair is the last one left.

   6, not 3-with-everything-renumbered. The value only has to be
   distinct; renumbering would move SSE2 and MMX under every comparison
   in the tree at once, for tidiness, and the enum is not in tier order
   anyway (ARM is 4 and 5).

   WHAT THIS MEANS ABOUT --kernel mmx: it means MMX. The sites below
   read `g_kernel != LZ_KERNEL_REF && lz_cpu_has_sse()`, so on any
   machine with SSE1 - which is every machine that runs the suite -
   asking for mmx runs the SSE1 body. The tiers owe each other
   bit-identity, so nothing reports it. */
#define LZ_KERNEL_SSE  6

/* The ARM column. Two values because the ARM side is two cells, C and
   hand-written asm, and WRONG 1 above is precisely about not letting one
   enum value stand for two code bodies - `--kernel arm-c` and
   `--kernel arm-asm` name them apart and lz_kernel_name() reports which
   one ran. Only the ARM cross-build ever selects either: kernel_detect
   returns them under __arm__ and lz_kernel_select clamps them to the
   local tier everywhere else, the same way `sse2` clamps on a PII.

   These are NOT a full ISA tier the way MMX/SSE2 are, but they are no
   longer ternary-only: every quantized row format has an ARM cell now,
   plus the operator leaves (wsum pair, the int32 Hadamard, norm_ss) and
   the F32 matmul row. Whatever still has no ARM kernel goes to the
   scalar reference by the same route `--kernel ref` takes, so the two
   tiers differ by the kernel under test and not by which fallback they
   picked. */
#define LZ_KERNEL_ARM     4
#define LZ_KERNEL_ARM_ASM 5
int lz_kernel_select(int which);

/* Prefetch tier, ORTHOGONAL to the kernel tier - the kernel picks
   instruction-set width, prefetch picks memory hints. K6-2/K7 have
   only MMX but do have 3DNow!'s prefetch; tying prefetch to "has SSE"
   would miss them. Returns an LZ_PF_* internal constant; name via
   lz_prefetch_name(). */
int lz_prefetch_mode(void);
const char *lz_prefetch_name(void);

/* Force a prefetch tier; LZ_PF_AUTO restores CPUID detection. Returns
   the tier actually in effect - a request the CPU cannot honour is
   downgraded rather than crashing (prefetchnta on a PII is an invalid
   opcode, and a benchmark that faults is not a slower benchmark).

   This exists to make the LATENCY TIER measurable. The model has two
   tiers per machine - latency (no prefetch) and bandwidth (prefetchnta)
   - and every conclusion that matters is a RATIO between them: whether
   J paid off, whether speculative decoding has headroom to trade into.
   Without this accessor the dispatch had no reachable "no prefetch"
   path (pf_detect returns LOAD/NTA/AMD and the else-branch fell to
   NTA), so a run on real hardware could only ever produce the bandwidth
   number. This is the instrument for the latency half of every one of
   those ratios.

   Measured, the ratio collapsed the two-tier model: on a real PIII
   Coppermine 1000 the tier ratio is <=1.05, not the 2.47 the model
   predicted - the machine is bound by DRAM/bus THROUGHPUT, not by
   memory latency, so prefetching cannot buy back much (corroborated by
   PC133 capping the same chip and by Apollo Pro 266 helping). Do not
   reuse the two-tier structure without rereading the current
   measurement - the instrument stays useful, the model it was built to
   feed does not.

   Takes a NAME, not an LZ_PF_* constant, because those are internal to
   ops.c by design (see lz_prefetch_mode above) and this accessor has no
   business making them public. Accepts "auto", "none", "load", "nta",
   "3dnow"; returns the name of the tier actually in effect, or NULL if
   the string is unrecognised. Callers should print the RETURNED name -
   a benchmark labelled with what it asked for rather than what it got
   is worse than no benchmark.

   Prefetch is a HINT: changing this tier must not change a single
   output bit. */

/* ---- THE float knob: --fixed all | off ------------------------------- */

/* `all` is the shipping engine and the default. `off` floats every
   arithmetic site at once - recurrence (both passes), conv1d,
   attention scoring/wsum/int-exit, dequant epilogue, sigmoid/silu,
   norms, and the scalar transcendentals - and exists so operator
   precision can be cross-checked against another implementation
   (llama.cpp, HF transformers, vLLM, SGLang) or across the rounding
   regimes this engine targets: SSE2, SSE, MMX, x87, ARMv5TE
   soft-float, gcc and Watcom.

   ORTHOGONAL to --kernel (which instruction set) and to --pair and
   --prefetch. Those stay separate axes because they answer "which
   implementation of the same arithmetic", not "which arithmetic".

   There is no third setting and no per-tier flag. Mixing was measured:
   floating exactly one tier at a time, six of the seven arms came out
   worse than BOTH ends, while the two ends are indistinguishable from
   each other (+3.49e-05, t=0.01, n=1262). See ops_sched.c.

   NOT numerics-neutral, obviously - that is the point of it. A gate
   that switches this needs a tolerance, not cmp. lz_float_ref_g is
   read directly by the lz_scalar_mode macro below so that softmax's
   per-element lz_exp does not pay a cross-TU call; use lz_float_ref()
   everywhere else. Only lz_float_ref_select writes it. */
extern int lz_float_ref_g;
const char *lz_float_ref_select(const char *name);
int lz_float_ref(void);

/* Fast float semantics: "on"|"off" (default on). RTZ+FTZ, the shipping
   float semantics; x86/ARM byte-identical under it (parity_gate fastfp
   row). off restores RNE. Read through lz_fastfp(). */
extern int lz_fastfp_g;
const char *lz_fastfp_select(const char *name);
int lz_fastfp(void);

/* bf16 WEIGHT STORAGE: 1 = keep a BF16 tensor at two bytes per element
   and widen it on read, 0 = expand it to f32 while loading. DEFAULT ON
   ON A TARGET WITH NO FPU, off elsewhere - see ops_sched.c for the
   measurement behind that split. Per-target defaults are safe here and
   would not be for any other knob in this file, because of the one
   property the others do not have: it changes nothing. A bf16 is
   the high half of an f32, so the widening is `bits << 16` - the very
   operation model.c performs at load time when this is off, only
   deferred to the read - and every value the kernels see is identical
   either way. What changes is that the tensor
   occupies half as much RAM and costs half as much DDR traffic to read,
   on a part whose bottleneck is DDR.
 *
 * It exists as a switch at all so that a parity comparison has
 * somewhere to stand: if two builds ever disagree, turning this off
 * puts both back on the expand-at-load path and says whether the
 * storage change was involved. cli_main prints the state for the same
 * reason - a run whose logits differ should not leave the reader
 * guessing which arm produced them.
 *
 * NOT a licence to narrow an F32 file into bf16. That would be lossy
 * and is a different decision; this only declines to WIDEN what is
 * already bf16. */
extern int lz_bf16_store_g;
const char *lz_bf16_store_select(const char *name);
int lz_bf16_store(void);

/* Token pairing: "on" | "off" | "auto" (= on). Returns the setting in
   effect, or NULL if the name is unknown.

   Orthogonal to the kernel tier and to prefetch: it selects WHICH
   32-element kernel a row loop calls, not which instruction set. The
   two options optimize different resources - pairing removes the weight
   unpack for a second token (memory side), the 128-element group kernel
   amortizes fixed prologue overhead (instruction side) - and which wins
   is a per-machine question nobody has answered on the target yet.

   Needs nt >= 2 to do anything, so at decode (nt == 1) both settings
   run identical code. Must not change one output bit. */
/* GDN/KDA pass-1 tier. NOT a selector of its own: it resolves from
   lz_gdn_p2_mode(), so one tier names the whole recurrence and
   `--fixed off` floats all of it. Returns 0 for the
   runtime value `float`, and also when the fixed path was not compiled
   in (-DLZ_GDN_FIXED=0). The float body is additionally the capacity
   fallback for kd > LZ_GDN_MAX_KD. */
int lz_gdn_mode(void);

/* "fixed" | "float" | "auto". Selects BOTH recurrence passes (pass 1
   through lz_gdn_mode above): pass 1's fixed path adds ~1.5e-05, pass
   2's adds ~1.2e-05 on top of a 1.8e-05 baseline. See
   gdn_p2_group_fixed in ops.c.

   Auto asks nothing - it returns fixed on every machine. The shipping
   build is fixed-point unconditionally, so speed is not what selects a
   tier here; lz_soft_float_target() and q8r_have_simd() are no longer
   consulted by any tier's auto (ops_sched.c carries the retirement note
   for the first, and the second now only picks a q8-round
   implementation). The knob exists so a control arm can be built, not
   so a machine can choose.

   Two earlier assumptions behind that reasoning have been re-measured:

     - "off pending an end-to-end measurement": that measurement was
       made. Paired per-position NLL, fixed minus float, mean -9.94e-04,
       t = -1.26 - indistinguishable, with the point estimate favouring
       fixed.
     - "on a machine that HAS the SIMD tier, fixed trades a fast path
       for a slow one": false on this host now that the fixed body has
       MMX and SSE2 kernels. Watcom, whole lz_gdn_step:
       fixed is 2.36x/1.30x/1.44x at --kernel ref/mmx/sse2 for GDN and
       2.45x/1.27x/1.42x for KDA. The fixed tier does not accelerate the
       write-back quantize, it DELETES it - the state never becomes
       float - and that quantize was 83% of this operator.

   One host cannot decide a family of targets, and no target-machine
   wall clock has been taken yet - but that does not gate the default,
   which is fixed because the shipping build is. The knob is the
   decision point, not this comment. */
int lz_gdn_p2_mode(void);
/* Which of the fixed pass-2 bodies is selected: "ref", "mmx-intrin",
   "mmx-asm", or "-" when the float tier is running. The same join
   lz_kernel_tier() reports, and it exists for the same reason: the
   bodies are required to be bit identical, so nothing in the output can
   tell you one of them stopped being compiled in. */
const char *lz_gdn_p2_impl(void);

/* Causal conv1d fixed tier, in the HEADER rather than in ops.c because
   forward.c guards its own buffers on it. LZ_GDN_FIXED's comment records
   what happens otherwise: `#if` on an undefined macro is a silent 0, so
   the guarded code simply is not compiled and the knob looks inert
   instead of absent. That is the shape this exact define hit on its
   first attempt here. */
#ifndef LZ_CONV_FIXED
#define LZ_CONV_FIXED 1
#endif

/* Route the engine's divides through lz_fdiv_recip (ops_quant.c) instead
   of the compiler's. BIT-IDENTICAL either way - the two arms of this
   switch give the same logits, which is unusual here and is what makes
   it checkable by cmp rather than by tolerance.
   Default on only where floats are software. On a host with a hardware
   divider the reciprocal is not obviously a win and has not been
   measured there; -DLZ_NO_FPU=1 is what the ARM cross build passes and
   is the one target the 117-against-30 number was taken on. Socket 7's
   x87 FDIV is 39 cycles against FMUL's 5, so it is a candidate too -
   also unmeasured, which is why it is not the default. */
#ifndef LZ_FDIV_RECIP
#if defined(LZ_NO_FPU)
#define LZ_FDIV_RECIP 1
#else
#define LZ_FDIV_RECIP 0
#endif
#endif /* LZ_FDIV_RECIP */
/* LZ_SOFTFP_FAST: RTZ for the software-float targets, where it is the
   shipping semantics (x86 --fastfp on is byte-identical). Strict (RNE,
   =libgcc) builds pin -DLZ_SOFTFP_FAST=0. */
#ifndef LZ_SOFTFP_FAST
#if defined(LZ_NO_FPU) || defined(__arm__)
#define LZ_SOFTFP_FAST 1
#else
#define LZ_SOFTFP_FAST 0
#endif
#endif /* !LZ_SOFTFP_FAST */

#if LZ_FDIV_RECIP
#define LZ_FDIV(a, b) lz_fdiv_recip((a), (b))
#else
#define LZ_FDIV(a, b) ((a) / (b))
#endif

/* The fixed conv1d epilogue's activation enters sigmoid in the INTEGER
   coordinate (sigmoid_q15_i) instead of the folded float one
   (sigmoid_q15_t). Three of the epilogue's six int<->float converts per
   channel go away - the whole coordinate derivation was the float work,
   and the accumulator is already the number the table wants indexing by.
   NOT bit-identical, and the integer entry is the MORE accurate of the
   two: every measured channel sits at e >= 22, far past the e <= 19
   band where sigmoid_q15's `x + 12` is still exact.
   In the header for LZ_CONV_FIXED's reason - forward.c builds the
   exponent table this reads, so both files must see the same value. */
#ifndef LZ_CONV_SIG_I
#define LZ_CONV_SIG_I 1
#endif

/* Causal conv1d tier, resolved from the one float knob above.
   NOT numerics-neutral: the fixed tier drifts ~4.8e-5 from float
   (.prof/wconv.c), so switching tiers needs a tolerance gate, not cmp. */
int lz_conv_mode(void);

/* Fixed-tier helpers, exposed because forward.c builds the quantized
   weight and history buffers with them. See their definitions for why
   the bound is derived and why the scale must be a power of two. */
int lz_conv_accum_bound(int k);
int lz_conv_norm_pow2(const float *v, int n, short *out, int bound);

/* Sigmoid/SiLU tier: "float" | "fixed" | "auto".

   These two are 36% of the f32 operations per token once the epilogue
   and attention tiers are on, and they are the cheapest 36% to take:
   sigmoid tabulates directly, so no chain changes and no new formats.

   lz_exp costs 13 f32 operations at 2.38e-7 relative error against a
   3.9e-3 noise floor - four orders nobody uses. The fixed form is a Q15
   table over [-8, 8] with linear interpolation; beyond that sigmoid is
   within 3.4e-4 of 0 or 1, under the floor, so it clamps.

   NOT the KDA decay, which also goes through lz_exp and must not: its
   gt multiplies the recurrent state every step, so a relative offset
   compounds N-fold. Measured at 1.64e-01 over 512 steps against the
   floor's 3.9e-3 - forty-two times over. The tier is named for its two
   callers for that reason.

   BOTH TIERS SHIP - a contested choice is a knob: which is faster
   depends on the
   target's float-to-integer cost ratio, and this host answers for
   itself only. */
/* The kernel coverage matrix: one line per weight format, six characters
   for the six row-kernel slots (mmx-intrin, sse-intrin, sse2-intrin,
   mmx-asm, sse-asm, sse2-asm) and one for whether that format has a
   128-element group kernel in this build.

   It exists because a hand-maintained version of this table claimed
   gcc 128-group kernels for three formats that had none at the time,
   and no gate could see the difference - the tier-union assertion
   checks that every TIER NAME appears somewhere, which a format
   missing a kernel does not change. (Those kernels exist today; the
   drift is the point, not the value.)

   Two builds are needed to read it: gcc carries the intrinsics columns,
   Watcom the assembly ones, and only their union is the answer.
   build/kernel_matrix_gate.sh is what forms that union.

   Empty cells do not print '.': see lz_kernel_cell below for the
   letters and what each one asserts. */
const char *lz_kernel_matrix(void);

/* Build classes. The registry says which of them each cell belongs to,
   so "absent from this binary" and "absent from every binary" stop
   being the same character. */
#define LZ_KMB_GCC 1
#define LZ_KMB_WAT 2
#define LZ_KMB_ARM 4

/* Row-axis kinds. A UNIT is one dispatchable kernel family. Every unit
 * today is a matmul row kernel named for its weight format (LZ_KM_ROW,
 * printed in the grid), but the axis is deliberately not "weight
 * format": --kernel arm-c/arm-asm has exactly one consumer right now,
 * and when a second operator gets an ARM tier the global enum stops
 * being able to say which one has the assembly. Such an operator joins
 * as LZ_KM_OP - registry only, because the grid's six characters are
 * the six LZ_ROW_* slots. src/ops.c's km_unit is the table. */
#define LZ_KM_ROW 0
#define LZ_KM_OP  1

/* One cell of the coverage matrix: unit x tier.
 *
 * status, for THIS build:
 *   'x'  present, and the registry expects it here
 *   'o'  absent here; another build class carries it (`have` says which)
 *   'i'  absent everywhere, structurally impossible - `why` is the audit
 *   't'  absent everywhere, not written yet - `why` names what is missing
 *   '!'  the registry expects it HERE and it is gone     -> gate failure
 *   '?'  absent with no registry entry at all            -> gate failure
 *   '+'  present but the registry does not expect it here -> gate failure
 *
 * '!' is the shape this registry was built after: the Watcom assembly vanished from the
 * build and every numeric gate stayed green, because a kernel that
 * falls back to the scalar reference is bit-identical to one that does
 * not. '?' is the other direction - a cell nobody registered reads as
 * an intentional blank.
 *
 * Iterate to exhaustion; the count lives in one place, src/ops.c:
 *     for (i = 0; lz_kernel_cell(i, &c); i++) ...
 * Returns 0 when i is past the last cell. */
typedef struct {
    const char *unit;   /* unit name, e.g. "q8_0" */
    const char *tier;   /* column name, e.g. "mmxA" */
    const char *why;    /* why no build has it; NULL when some build does */
    unsigned    have;   /* LZ_KMB_* mask: build classes expected to carry it */
    char        status; /* this build - see above */
    unsigned char kind; /* LZ_KM_ROW or LZ_KM_OP */
} LZKmCell;

int lz_kernel_cell(int i, LZKmCell *out);

/* Which LZ_KMB_* class this binary is. Exactly one bit. The union check
   needs a binary per class and has to know which one it is holding;
   deriving it from the cells would be guessing, and re-deriving it from
   __WATCOMC__/__arm__ in the front end would be a second authority. */
unsigned lz_kernel_build_class(void);

/* Every AUTO-resolved tier, in one list.
 *
 * lz_tier_pins() is the CLI string that pins all of them. Any comparison
 * across targets or builds must use it: an unpinned AUTO resolves from
 * hardware and the two sides then run different arithmetic, which makes
 * the comparison compare nothing. That is not hypothetical - the ARM
 * parity gate pinned two knobs by hand, four more were added later with
 * the same predicate, and the first real run reported 94505 of 131072
 * bytes differing for that reason alone.
 *
 * lz_tier_report() is the same list resolved, for the banner, so
 * "chosen" and "defaulted" are distinguishable at a glance. */
const char *lz_tier_pins(void);
const char *lz_tier_report(void);
int lz_pair_is_auto(void);

int lz_sig_mode(void);

/* Fixed norms tier, compiled in or out. Its own axis: the recurrence's
   LZ_GDN_FIXED says nothing about the norms and this says nothing about
   the recurrence, so -DLZ_GDN_FIXED=0 isolates the recurrence and
   -DLZ_NORM_FIXED=0 isolates the norms. Guards norm_ss_fixed
   (ops_quant.c), the three fixed output loops (ops_norm.c) and
   lz_norm_can_fixed (ops_sched.c). In the HEADER, like LZ_CONV_FIXED, so
   build/switch_matrix_gate.sh's -dM proof reaches it through ops.c and
   forward.c. */
#ifndef LZ_NORM_FIXED
#define LZ_NORM_FIXED 1
#endif /* LZ_NORM_FIXED */

/* GDN pass-1 tier default. IN THE HEADER FOR THE SAME REASON
   LZ_NORM_FIXED and LZ_CONV_FIXED are, and it took four copies to get
   here: ops.c, ops_gdn.c, ops_matmul.c and ops_sched.c each carried
   their own `#ifndef LZ_GDN_FIXED / #define 1`. All four agreed, so
   nothing was wrong - and nothing would have SAID anything if one had
   been edited and the others not, because each TU only ever sees its
   own. build/dedupe_gate.sh now asserts the single definition.

   ops.c's original note, kept because it is the reason the default is
   spelled out at all rather than left to `#if` on an undefined macro:
   `#if LZ_GDN_FIXED` on an UNDEFINED macro is silently 0, which made
   lz_gdn_mode() return "float" always AND, through it, disabled the
   fixed path outright. The knob's two settings then produced identical
   logits - it looked numerics-neutral instead of broken. A knob whose
   effect is not observable is indistinguishable from a knob that does
   nothing. */
#ifndef LZ_GDN_FIXED
#define LZ_GDN_FIXED 1
#endif /* LZ_GDN_FIXED */

/* Sizing constants that were also once per-TU, and these are the ones
   where drift is not a style problem: each bounds a STATIC ARRAY, so
   two TUs disagreeing is a buffer one side thinks is longer than the
   other side allocated. All were 4096 / 64 / 4096 in every copy.
     LZ_NORM_MAX_N    ops_norm.c, ops_quant.c, ops_sched.c
     LZ_EPI_MAX_NG    ops.c, ops_matmul.c
     LZ_MM_WIDEN_MAX  ops.c, ops_matmul.c */
#ifndef LZ_NORM_MAX_N
#define LZ_NORM_MAX_N 4096
#endif /* LZ_NORM_MAX_N */
#ifndef LZ_EPI_MAX_NG
#define LZ_EPI_MAX_NG 64
#endif /* LZ_EPI_MAX_NG */
#ifndef LZ_MM_WIDEN_MAX
#define LZ_MM_WIDEN_MAX 4096
#endif /* LZ_MM_WIDEN_MAX */
/* Whether the fixed dequant epilogue exists at all. Same shape as
   LZ_GDN_FIXED / LZ_CONV_FIXED / LZ_NORM_FIXED, and set to 0 by
   build/kernel_isa_identity_gate.sh for the one build where --kernel
   ref and --kernel sse2 must be byte-identical: ref never reaches an
   epilogue, so that is the single axis on which the two legitimately
   differ. See lz_epi_mode. */
#ifndef LZ_EPI_FIXED
#define LZ_EPI_FIXED 1
#endif /* LZ_EPI_FIXED */

/* x87 precision-control availability, one detect instead of two. ops.c
   and ops_quant.c each ran this same three-condition test; they are the
   same test, and a target where they answered differently would have
   one file saving the control word and the other not. */
#if defined(__i386__) && !defined(__SSE_MATH__) && !defined(__SSE2_MATH__)
#define LZ_X87_FLOAT_CW 1
#endif /* x87 float control word */

/* The SSE counterpart, and the mirror image of the test above: where
   scalar float goes through SSE there IS no x87 control word to set,
   but there IS an MXCSR - which strict mode leaves alone and fast mode
   uses for FTZ/DAZ/round-toward-zero. gcc-only: the asm is stmxcsr /
   ldmxcsr, and Watcom's x86 targets are covered by the x87 arm. */
#if defined(__GNUC__) && (defined(__SSE_MATH__) || defined(__SSE2_MATH__))
#define LZ_SSE_FLOAT_CSR 1
#endif /* SSE float control/status register */

/* The gated norm's sigmoid reads the INTEGER coordinate.
   lz_rmsnorm_gated_out_fixed_int already quantizes the gate to Q15 (qg
   at exponent eg) for its product chain, and then reached sigmoid_q15
   through the float g a second time - three converts per element
   deriving a table coordinate that sigmoid_q15_i gets with a shift and
   a mask. With this on the sigmoid comes from sigmoid_q15_i(qg, eg).

   VALUE-CHANGING, and consistently so: silu(g) = g * sigmoid(g), the
   product's g factor was already qg, and now both factors read the same
   quantized gate instead of one reading g and the other its Q15 image.

   0 or 1 only - ops_norm.c's census formula is arithmetic on this value,
   not a preprocessor branch. 0 is the control arm the saving was
   measured against. In the HEADER, like LZ_NORM_FIXED, so
   build/switch_matrix_gate.sh's -dM proof reaches it through ops.c and
   forward.c. */
#ifndef LZ_GATE_SIG_I
#define LZ_GATE_SIG_I 1
#endif /* LZ_GATE_SIG_I */

/* The gated norm's weight Q15 table, cached per weight pointer.
   lz_rmsnorm_gated_out_fixed_int scanned w for its amax and rounded
   every element to Q15 on every token, although w is a norm weight -
   the same object for the life of the model. With this on it reads
   build_qw's table instead (ops_norm.c), which the plain norm's int
   output loop already used.

   BIT-IDENTICAL by construction, not by tolerance: the table is filled
   by the same arithmetic in the same order the inline loop ran.

   0 or 1 only, same reason as LZ_GATE_SIG_I. 0 is the control arm and
   computes the scan and the rounding per call exactly as before. */
#ifndef LZ_GATE_QW
#define LZ_GATE_QW 1
#endif /* LZ_GATE_QW */

/* The RMS/L2 norm tier. fixed accumulates the sum of squares in int64
   from the int16 image of x rather than in f32, so the per-element
   multiply-add leaves the float domain; the reciprocal square root and
   the output scale stay float. */
int lz_norm_mode(void);
/* Whether lz_rmsnorm_gated will take the fixed tier for an n-element
   norm: false in a LZ_NORM_FIXED=0 build, where the fixed body is
   compiled out and norm_ss_fixed declines. The fixed-tier call sites
   that hand the pre-silu product and the Q15 sigmoid to
   lz_quantize_q8_silu gate on this, so the float tier keeps the plain
   lz_rmsnorm_gated + lz_quantize_q8 path. */
int lz_norm_can_fixed(int n);

/* Whether a norm's fixed tier writes its Q15 result as int (consumed by
   lz_quantize_q8_int) instead of dequantizing to float and
   re-quantizing.

   NOT SELECTABLE, and there is nothing to select between: it answers
   yes whenever the fixed tier is on. On the gated norms the two
   spellings are bit-identical (the dequantize/requantize pair cancels
   - proof at lz_quantize_q8_int), and on the plain norm the int form
   is ~9.5x cheaper per element on the target. Read the rest of this
   comment for the half that was a defect rather than a preference.

   Bit-identical for the gated norm, not for the plain one. The identity
   is a CANCELLATION, not a property of int arithmetic: it needs the
   fixed tier to already be computing the very same Q15 product, so that
   `int` only elides a Q15 -> float -> int8 round trip. That premise
   holds for lz_rmsnorm_gated, whose fixed output loop
   (lz_rmsnorm_gated_out_fixed) ends in `o[i] = (float)t * pow2f(...)`
   over exactly the chain its _int twin writes out - the proof lives at
   lz_quantize_q8_int.

   It does NOT hold for lz_rmsnorm. That one's fixed tier replaces the
   sum of squares ONLY; its output loop is float end to end
   (`o[i] = x[i] * inv * (1.0f + w[i])`, ops_norm.c). So
   lz_rmsnorm_out_fixed_int does not elide a round trip - it introduces
   a Q15 chain that the fixed path never had. Different arithmetic, no
   cancellation, and nothing was wrong with the proof: the flag's REACH
   grew past it.

   Do NOT scope this line to "the SGDN norm's fixed tier". That scope
   would make the bit-identity clause true, and it is not: SubLN reuses
   the flag for six lz_rmsnorm sites - forward_kda's four projection
   input norms and dense_ffn_step's gate/up pair, all inside a use_subn
   branch - which the narrower qualifier does not cover.

   Measured on kunkun-ce q8, 500 tokens of real corpus, fixed
   against int: |diff| rms 0.144 on a signal rms of 2.01 (7%), top-1
   88.58%, NOT ONE element equal. Disabling exactly those six restores
   bit-identity, which is how the list was arrived at rather than read
   off a diff.

   Do not reach for s1v3 as the control: fixed and int agree there, and
   they go on agreeing with lz_quantize_q8_int mutated to a constant.
   s1v3 reaches no int site at all. Of what is on disk, kunkun-ce is the
   only model that exercises this value.

   The int form is what the fixed tier does, so the six plain-norm
   sites ship it, and the ~7% is a deliberate float-to-fixed step of the
   kind already taken for every other write tier. It buys 114 -> 12
   instructions per element on the target (measured, ops_norm.c) with
   the (1+w) table cached; `--fixed off` remains the oracle. */
int lz_norm_int(void);

/* SubLN activation scale granularity: "tok" | "group" (default tok).
   Selected as `--subn-scale`. Applies ONLY where use_subn is on, to the
   o_proj / down_proj input - the two projections H-BitLinear covers.

   tok:   one absmax over the whole row, replicated into every n/32 slot
          (BitNet v2's per-token scheme).
   group: the engine's own per-32 grouping (lz_act_gs), kept across the
          Hadamard.

   A value-changing knob, like the conv/attn/epi tiers: the two produce
   different int8 and different scales. `tok` is bit-identical to the
   engine before this knob existed. See ops_sched.c for why the Hadamard
   does not force a row-wide scale. */
int lz_subn_scale_group(void);
const char *lz_subn_scale_select(const char *name);

/* SubLN block Hadamard: "on" | "off" (default on), `--hadamard`.
   off forces the resolved hadamard_o/hadamard_down to 0 at their use
   sites. The feature's control arm - without it the only way to ask
   whether the rotation ran was to export a second model. */
int lz_hadamard_on(void);
const char *lz_hadamard_select(const char *name);

/* Scalar transcendental tier (lz_exp and lz_rsqrt together):
   "float" | "fixed" | "auto".

   The last two scalar x87 operators the fixed tiers leave on float:
   softmax's exp and the norms' rsqrt. Both fixed bodies are
   integer-domain - a Q20 table + Taylor correction for exp, a Q23
   seed table + 2 integer Newtons for rsqrt - and CHANGE the value vs
   the float path by design (measured, not promised: ~1e-6 relative
   for exp, ~1e-7 for rsqrt, both far under the Q8 floor). So this
   tier is a documented value-changing knob, like the conv/attn/epi
   tiers.

   AUTO asks the same CPUID SSE question every other tier asks, so a
   no-SIMD (weak-x87) target lands on fixed. The float path stays the
   default anywhere x87 is not the bottleneck. */
/* INLINE, unlike its eleven siblings, and the reason is where it is
   called from. Every other tier getter is asked once per operator call;
   this one is asked once per lz_exp / lz_rsqrt / lz_sigmoid, which
   softmax invokes PER ELEMENT. Left out of line it made lz_exp's
   dispatch shell fifteen instructions on ARM - a push/pop pair, an
   argument shuffle and a cross-TU `bl` for a one-line getter - in front
   of a body that is 169.

   The global stops being file-static to allow it, which is the same
   trade g_kernel already makes in ops_kernel_shared.h. lz_float_ref_g
   is now that global for every tier, not just this one, and
   lz_float_ref_select is still its only writer; nothing outside
   ops_sched.c may assign it. */
#define lz_scalar_mode() (!lz_float_ref_g)

/* Dequant epilogue tier: "float" | "fixed" | "auto".

   The epilogue is 66.4% of the f32 operations per token once the fixed
   attention tier is on. The fixed form quantizes both scale planes to
   int16 - weights at load time, activations once per token - and
   accumulates acc[g]*(xw[g]*ww[g]) in int64, leaving one convert and one
   multiply per row instead of four per group.

   The bound is DERIVED, not observed: ng * 2^27 * 2^30 = 2.27e18 against
   int64's 9.22e18, holding while ng <= 64, i.e. in_dim <= 2048. Wider
   rows keep the float path rather than silently overflowing. Measured
   max relative error 1.5e-4 against a Q8 noise floor of 3.9e-3.

   BOTH TIERS SHIP. On this host the fixed one is 1.65x SLOWER (Watcom,
   22.35 vs 13.52 ns per row) - and this host decides nothing: per group
   it is one imul against one fild, two fmul and one fadd, and a P6
   issues three uops a cycle against fmul's one-per-two. Which wins is a
   target-machine question, so it is a knob.

   AUTO asks the same CPUID SSE question the conv and pass-2 tiers ask. */
int lz_epi_mode(void);

/* Build a tensor's int16 scale plane. Idempotent, and a no-op when the
   shape is outside the derived bound - callers do not have to know the
   limit, they check lz_epi_ready() afterwards. */
void lz_epi_prep(LZTensor *w, int in_dim);
int  lz_epi_ready(const LZTensor *w);

/* The same for the `zero` plane (Q4_1/Q6_1 group minimums), which the
   Q4_1/Q6_1/T2 epilogue needs to fold its zero term into the integer
   domain. A no-op with lz_epi_zero_ready() staying false wherever
   w->zero is NULL, which is every other format including T2 - T2's zero
   coefficient is -scale and rides the plane above. */
void lz_epi_prep_zero(LZTensor *w, int in_dim);
int  lz_epi_zero_ready(const LZTensor *w);

/* Half-away-from-zero arithmetic shift right of a 64-bit signed value,
   the fixed epilogue's narrowing. Shared with the conv1d fixed tier
   (ops_conv1d.c), which is why it is exported from ops.c rather than
   static. See its definition for the branchless sign handling and the
   rs <= 0 / rs >= 64 corners. */
lz_i64 epi_shr_half(lz_i64 s, int rs);
lz_u64 epi_shr_half_u(lz_u64 umag, int rs);

/* Sub-blocks per weight group (gs/32) that the zero term's integer
   activation fold can take. DERIVED, like LZ_EPI_MAX_NG above it, and
   from the same kind of worst case: |Sum_{k<32} xq| <= 32*127 = 4064
   and |packed sub-block scale| <= 32767 put one product at 133,165,088,
   and r of them stay inside int32 only while r <= 16. Shapes past it
   keep the float epilogue rather than silently wrapping. See
   epi_zero_act_int (src/ops.c) for the full derivation and the measured
   margin. */
#define LZ_EPI_MAX_R 16

const char *lz_pair_select(const char *name);
int lz_pair_mode(void);

const char *lz_prefetch_select(const char *name);
const char *lz_kernel_name(void);

/* The PRIMARY matmul path this BINARY was compiled with: "mmx-asm",
   "mmx-intrin", "sse2-intrin" or "scalar". Compile-time truth, not a
   runtime tier - lz_kernel_name() answers "which kernel did CPUID
   pick among those compiled in", this answers "which #if branch the
   matmul bodies took", and the two are independent.

   NOT a complete inventory, and the name overstates it: the Watcom
   build returns "mmx-asm" while also carrying the sse2-asm kernels
   (dispatched at runtime by g_kernel inside that same branch). Use
   lz_kernel_tier() for what actually ran; the per-format row-kernel
   tables in ops.c are the enumerable dispatch.

   It exists because the documented Watcom build can compile NONE of
   the hand-written assembly: wcc386 predefines neither __MMX__ nor
   __SSE2__, and every #pragma aux kernel plus its dispatch sits inside
   `LZ_USE_MMX && __MMX__`, so the engine silently runs the scalar
   #else. No gate sees it - the cross-compiler bit-identity check
   passes precisely BECAUSE all paths agree bit for bit, so two builds
   running different paths look identical. A consistency gate cannot
   report which path ran; this can. */
const char *lz_build_paths(void);

/* Which ONE of the seven operator variants is in effect: "ref",
   "mmx-intrin", "sse-intrin", "sse2-intrin", "mmx-asm", "sse-asm" or
   "sse2-asm". The join of lz_kernel_name() (runtime ISA) and
   lz_build_paths() (compile-time impl) - neither alone identifies a
   variant, because one enum value covers both impls of an ISA.
   Full rationale at the definition in ops.c. */
const char *lz_kernel_tier(void);

/* Q8_0 activation quantization: q[i] = round_half_even(x[i]/s[i/gs]), s
   being the group's absmax/127 (all-zero group scale=1.0). q needs n
   bytes, s n/gs f32s. */
void lz_quantize_q8(const float *x, int n, int gs, int8_t *q, float *s);

/* Per-token absmax INT8 activation quantize (SubLN path). ONE scale for
   the whole n-wide row - the training side's activation_quant_absmax
   (gamma = max|x| over the token, dequant scale = gamma/127) - stored in
   every n/32 xqs slot so the matmul's per-32 xqs read stays consistent
   with lz_act_gs. q/s semantics match lz_quantize_q8: round-half-even,
   clamp [-127,127], all-zero row scale 1.0. n must be a multiple of 32. */
void lz_quantize_q8_tok(const float *x, int n, int8_t *q, float *s);

/* Int-domain twin of lz_quantize_q8_tok, for the fixed-tier plain
   norm's int output (lz_rmsnorm_int): t holds the Q15-ish result the
   norm wrote as int and deq its power-of-two scale. ONE scale over the
   whole n-wide row, replicated into every n/32 xqs slot, same layout as
   lz_quantize_q8_tok; same deq-folds-into-the-stored-scale fold as
   lz_quantize_q8_int (see its comment) - the element loop rounds
   t*127/t_amax and deq enters the scale only. Unlike the gated norm's
   t, this t is NOT bounded to Q15 magnitude (lz_rmsnorm_out_fixed_int's
   chain has one fewer shift), so (float)t can round values above 2^24
   to 24-bit mantissa - the same ~1e-7 relative error
   lz_quantize_q8_int64 already accepts for its wider accumulator,
   negligible next to Q8's own granularity. n must be a multiple of 32. */
void lz_quantize_q8_tok_int(const int32_t *t, int n, float deq,
                            int8_t *q, float *s);

/* Int-domain variant for the fixed-tier gated norm: the norm wrote its
   Q15 result t[] as int and handed the dequant scale deq (a power of
   two) out separately. The stored group scale is t_amax*deq/127,
   matching lz_quantize_q8 on the float input (float)t[i]*deq bit for
   bit, while the element loop rounds t*127/t_amax - deq enters the
   scale only, never the element loop (the same fold lz_quantize_q8_silu
   applies to 2^-15). deq is a non-negative power of two; a zero deq
   collapses every group to all-zero q with scale 1.0, same as the float
   path's zero output.

   The element loop is INTEGER (LZ_Q8INT_INT, src/ops.c) and returns the
   exactly-rounded quotient, so it is not a float32 multiply any more
   and is bit-identical across toolchains by construction. t is Q15 in
   ORIGIN only - the gated norm's four-factor chain leaves it up to
   32767^2, and 4.6e8 is measured - so the loop normalizes the group
   amax before applying the shared reciprocal. */
void lz_quantize_q8_int(const int32_t *t, int n, int gs, float deq,
                        int8_t *q, float *s);

/* int16-input twin of lz_quantize_q8_int, for a producer whose int16
   exit (epi_align_i16) already delivered the vector at a fixed
   exponent. Same deq fold, same MIN_SCALE floor, same stored scale.

   It exists because at int16 width the element loop needs no float at
   all: 127*t/tam is rounded exactly in int32 by a per-group reciprocal
   and a multiply-shift (LZ_Q8I16_INT, src/ops.c), and no normalizing
   pre-shift is needed to get there, the amax already being in the
   reciprocal's range. The int32 entry reaches the same answer with one
   (LZ_Q8INT_INT), so the two agree again on the values they share -
   measured over all 536,887,295 of them. Cross-compiler bit-identity
   comes for free in both, there being no float rounding left for
   gcc/SSE and Watcom/x87 to resolve differently. */
void lz_quantize_q8_i16(const short *t, int n, int gs, float deq,
                        int8_t *q, float *s);

/* SwiGLU with both operands already int16 (int-pipeline milestone 8):
   o[i] = silu(p[i]) * g[i], entirely in integers.

   p is at exponent ep and g at eg (value == q * 2^-ES, epi_align_i16's
   convention); o comes out at eo, clamped to +-32767 by epi_align_i16.
   sigmoid comes from sigmoid_q15_i, which takes p's bits directly, so
   the chain from the two producing matmuls to lz_quantize_q8_i16 crosses
   no int<->float boundary at all. `o` may alias `p`.

   ep must be within sigmoid_q15_i's domain (0 <= ep <= 50); outside it
   that function reconstructs a float and this function's whole point is
   gone. See src/ops.c for the exact accumulator bounds - both int32
   products sit at their edges by construction, not by margin. */
void lz_swiglu_q15_i16(short *o, const short *p, const short *g, int n,
                       int ep, int eg, int eo);

/* int64-input twin of lz_quantize_q8_int, for the fixed-tier attention
   weighted sum. The attention accumulator is exact only at int64 width:
   each 512-row int32 chunk fits, but their full-context SUM does not
   (T*32767*127 exceeds INT32_MAX past T=516), so the gated values come
   at int64. Same deq-folds-into-scale mechanism, same MIN_SCALE floor. */
void lz_quantize_q8_int64(const int64_t *t, int n, int gs, float deq,
                          int8_t *q, float *s);

/* Fused fixed-tier gated-RMSNorm quantize. p is the pre-silu product
   and sig the Q15 sigmoid that lz_rmsnorm_gated wrote in its fixed tier
   (g is the gate); the silu value
   o[i] = p[i]*(g[i]*(float)sig[i])*2^-15 is reconstructed and quantized
   to q/s. Bit-identical to lz_quantize_q8 on the float silu output: the
   2^-15 is a power of two, so amax over the pre-scale product scales
   exactly and the products round identically. o receives the
   reconstructed float (the f32-weight matmul fallback and the sgdn tap
   read it) and doubles as the amax-pass scratch. gs <= 0 (f32 weights)
   reconstructs into o and returns without quantizing. */
void lz_quantize_q8_silu(const float *p, const float *g, const int32_t *sig,
                         int n, int gs, int8_t *q, float *s, float *o);

/* Which group size this weight tensor requires for quantizing activations.

   Activations are ALWAYS Q8 group 32, independent of the weight format
   and gs - coarser weight groups save per-token bytes; activations
   have only in_dim elements, nothing to save, no reason to lose
   precision. Returns 0 when no quantization is needed (f32 weights).
   The only exception is the exported degenerate tier (in_dim not a
   multiple of 32, gs halved below 32), where activations follow the
   weight groups to keep old behavior. */
int lz_act_gs(const LZTensor *w, int in_dim);

/* o = W x (x = f32 activations).
   W is a weight tensor (LZTensor): dtype=0 goes f32; dtype=1 first
   quantizes x, then int8*int8->int32 accumulation, dequantized per
   group by (sx * sw).
   xq/xqs are caller-provided quantization buffers (capacity in_dim
   bytes / in_dim/gs f32s). */
void lz_matmul_w(float *o, const float *x, const LZTensor *w,
                 int in_dim, int out_dim, int8_t *xq, float *xqs);

/* Matmul consuming ALREADY-QUANTIZED activations: the forward hot path
   reuses one quantization for q/k/v/gate/up sharing xb's xq/xqs,
   avoiding repeated activation scans.

   The suffix names the activation side, not the weights. `xq` is the
   parameter it takes; the weight format is whatever w->dtype says, and
   this one entry point dispatches every one of them - F32, Q8_0, Q4_1,
   Q6_1, Q16_0, T2. A name that mentions one format gets read as "the
   path for that format", which happened here more than once, including in a
   review that concluded the tied lm_head was pinned to Q8 and therefore
   could not go ternary. It never was: forward.c's lm_head call sites
   have always dispatched on dtype like every other call site.

   The three members of this family differ ONLY in who quantizes the
   activations and how many tokens are done at once:
     lz_matmul_w      f32 activations, quantizes internally
     lz_matmul_xq     caller pre-quantized, 1 token
     lz_matmul_xq_nt  caller pre-quantized, nt tokens */
void lz_matmul_xq(float *o, const float *x, const int8_t *xq,
                  const float *xqs, const LZTensor *w,
                  int in_dim, int out_dim);

/* Max tokens per batch for batched prefill.

   Generation streams all weights once per token, arithmetic intensity
   1; prefill of T tokens shares one weight load, intensity T. T=4 is
   enough to flip prefill from memory-bound to compute-bound (the
   intensity identity); beyond that, gains diminish while activation
   scratch grows linearly.

   No byte figure is quoted here on purpose. The recipe direction is
   settled, but "any concrete recipe number appearing elsewhere,
   including the volume column, is a placeholder, not a conclusion".
   Nine of thirteen roles are still open. The recipe of record (p0)
   currently weighs ~45.8 MB, but with nine
   roles open that total is itself a placeholder, so quoting one number
   as if it were the weight budget would just be ranking placeholders.

   T=4 holds across that whole range, which is why it can be decided
   without the number. What cannot: any RATIO built on a byte figure.

   MEASURED, and it does NOT say T=4 flips prefill to
   compute-bound. Qwen3.5-0.8B, 512-token prompt, gcc -O2 -mmmx
   -DLZ_USE_MMX=1 (the batched branch), wall time, best of 3:

     T=1  56.94 s     T=2  45.13 s (1.26x)
     T=4  39.50 s (1.44x)          T=8  36.94 s (1.54x)

   Fitting 1/(f/T + (1-f)) to all three ratios gives f = 0.412 / 0.408 /
   0.401 - the same number three times. **Only 41% of prefill cost scales
   with batch width; the ceiling from batching alone is 1/0.59 = 1.69x**,
   and T=8 is already at 1.54x of it. The 59% is per-token work that a
   wider batch cannot touch: the 32-element kernel reloads and re-widens
   the weight row for every token in the batch (4 loads + 8 unpacks of
   its ~36 ops), plus the SSM recurrence, the float epilogue and the
   activation quantization. Batching amortizes DRAM traffic, not micro-ops.

   Two ways to share that weight-side work, and only one of them helps.
   Pre-widening the weights into an int16 scratch buffer does NOT: each
   token would then load 64 bytes per sub-block instead of 32 - 8 loads
   replacing 4 loads + 8 unpacks, plus a widening pass amortised over T,
   so at T=4 the op count goes 36 -> 37. Keeping the widened weight in
   REGISTERS across the batch does: 12 + 24*T against 36*T, ceiling
   36/24 = 1.5x. dot32_x16_mmx_2 (ops.c) is that, at T=2, which is what
   eight MMX registers allow once each token needs its own accumulator.
   Both numbers are op counts, not measurements.

   Note what this does NOT decide: whether to write the kernel. The
   rule is C / MMX / SSE / SSE2 paths on every operator
   unconditionally, because the bottleneck is a seesaw and "optimise
   whichever side binds today" leaves both undone.

   THE CAP IS 8, THE DEFAULT IS 4, AND THAT SPLIT IS DELIBERATE. The cap
   only sizes buffers (kunkun98-pilot state 2.2 -> 2.6 MB, the 0.8B
   28.8 -> 29.8, plus g_xw 32K -> 64K static). The default is what runs.

   **4 IS SETTLED (user decision after testing). Do not
   reopen it on the strength of the dev-box number.** T=8 is faster
   here - kunkun98-pilot 512-token prefill 3.591 s vs 4.051 s - and that
   is exactly the kind of number that must not be promoted into a
   target default. The mechanism for it to reverse is concrete: g_xw is
   T x in_dim x 2 bytes, so on kunkun98 (in_dim 1024) T=8 is 16 KB of
   activations, exactly a PII's whole L1, which would evict the weight
   lines the batch exists to reuse. T=4 is 8 KB.

   The knob stays runtime (`--batch`, LZRunState.nt_cap) anyway: a new
   target, a new model geometry or a new in_dim moves the L1 arithmetic
   above, and re-deciding must not need a Watcom rebuild. Note 4 is
   not a measured value either - it comes from the two-tier
   bandwidth model that the measurement records as collapsed. It is
   measured now.

   Overridable at compile time, same reason as LZ_PF_DIST.
   **`-DLZ_BATCH_MAX=1` equals batching off** and is the control for
   bit-identical differential checks. */
#ifndef LZ_BATCH_MAX
#define LZ_BATCH_MAX 8
#endif

/* Runtime default for LZRunState.nt_cap. Must be <= LZ_BATCH_MAX. */
#ifndef LZ_BATCH_DEFAULT
#define LZ_BATCH_DEFAULT 4
#endif
#if LZ_BATCH_DEFAULT > LZ_BATCH_MAX
#error "LZ_BATCH_DEFAULT exceeds LZ_BATCH_MAX"
#endif

/* Batched lz_matmul_xq: one weight stream serves nt tokens.

   All three buffers are token-major with hardcoded strides, no extra
   params:
     o  [t * out_dim + i]
     x  [t * in_dim  + k] (f32-weight fallback path only)
     xq [t * in_dim  + k]
     xqs[t * (in_dim / lz_act_gs(w, in_dim)) + g]

   **nt=1 is bit-identical to lz_matmul_xq** - not "numerically close",
   the t-loop in the same code degrades to one iteration and the
   reduction structure is untouched byte for byte. This is a hard
   gate: batching changes weight-load count, not what is computed. */
void lz_matmul_xq_nt(float *o, const float *x, const int8_t *xq,
                     const float *xqs, const LZTensor *w,
                     int in_dim, int out_dim, int nt);

/* 1 if the call above will read `x` (the f32-weight fallback), 0 if it
   will read `xq`/`xqs`. A caller that prepares the two inputs
   DIFFERENTLY must branch on this and not on its own copy of the dtype
   test - see the definition for the bug that produced. */
int lz_matmul_xq_reads_float_row(const LZTensor *w);

/* Same matmul, int16 exit (int-pipeline milestone 3). The row kernel
   and the int32 accumulators are the same ones lz_matmul_xq_nt uses -
   only the epilogue differs, exiting through epi_q41_align_i16 (the
   Q4_1/Q6_1/T2 family) or epi_fixed_align_i16 (the symmetric Q8_0 /
   Q16_0 one) instead of materializing a float. `o` is token-major
   [t * out_dim + i] like lz_matmul_xq_nt's, in int16 at exponent
   `target_e`, clamped to +-bound; there is no f32-weight fallback
   because there is no int16 exit for one.

   Serves every quantized format the fixed epilogue reaches, and only
   where it is live. Returns 1 if it ran and 0 if it refused, having
   written nothing - ask lz_matmul_xq_i16_ok FIRST if the caller needs
   to choose between tiers for a GROUP of tensors, which is the normal
   case: one consumer fed half in int and half in float is worse than
   either. */
int lz_matmul_xq_nt_i16(short *o, const int8_t *xq, const float *xqs,
                        const LZTensor *w, int in_dim, int out_dim, int nt,
                        int target_e, int bound);

/* Whether lz_matmul_xq_nt_i16 would run for this tensor. Same answer
   for the same arguments, and it is the predicate the entry point
   itself asks, so a caller that gets 1 here cannot then be refused.
   Idempotent side effect: it may build the tensor's fixed-epilogue
   planes (lz_epi_prep), exactly as the float dispatch does. */
int lz_matmul_xq_i16_ok(const LZTensor *w, int in_dim, int nt);

/* Pure f32 matmul (for differential/debug use; semantics same as
   lz_matmul_w's dtype=0 branch) */
void lz_matmul(float *o, const float *x, const float *w,
                   int in_dim, int out_dim);

/* The same matmul over a bf16-stored weight matrix: each row is widened
   into a scratch buffer and then run through the identical row body, so
   the result is bit-identical to expanding the tensor at load time and
   calling lz_matmul. wb is two bytes per element, row-major (out, in).
   in_dim must be <= LZ_MM_WIDEN_MAX; model.c keeps wider tensors as f32
   so this is guaranteed at the call site. */
void lz_matmul_bf16(float *o, const float *x, const unsigned char *wb,
                    int in_dim, int out_dim);

/* Apply partial RoPE in place. v is laid out (n_heads, head_dim);
   each head rotates only its leading rotary_dim dims, the rest stay
   as-is. cs is a precomputed (cos, sin) table:
   cs[(pos*half + i)*2 + 0/1], half = rotary_dim/2; built by the caller
   at state allocation (same theta). */
void lz_rope(float *v, int n_heads, int head_dim, int rotary_dim,
             int pos, const float *cs);

/* In-place fast Walsh-Hadamard transform over n contiguous floats, n a
   power of two. UNNORMALIZED: the matrix entries are +-1, so this is the
   orthonormal Hadamard scaled by sqrt(n), and applying it twice gives
   n * v.

   Why unnormalized. The orthonormal form needs a 1/sqrt(n) factor, which
   is not a power of two for odd log2(n) (n=128: 1/sqrt(128)), and
   dividing by a float constant is barred: the two compilers
   strength-reduce it differently. Leaving the sqrt(n) in and folding the
   resulting factor n into a scale the caller already owns keeps every
   step exact: n is a power of two, so 1/n is exact in binary float and
   both compilers emit the same thing.

   Callers must therefore account for the factor:
     scoring   (H q) . (H k) = n (q . k)   -> divide the attention scale by n
     weighted  H (H a) = n a               -> divide the result by n
   Both divisors are powers of two, so this costs no accuracy.

   Cost: n*log2(n) adds/subs and NO multiplies - which is the whole
   reason to prefer a Hadamard over a general rotation on a P6, where
   FMUL throughput is one per two cycles and FADD is one per cycle.

   Scalar only, deliberately (a new transform lands as a
   scalar reference first). It runs once per token per head, not once per
   cached row, so it is not on the hot path that would justify two
   hand-written kernels. */
void lz_fwht(float *v, int n);

/* ---- 4-bit KV cache rows (IsoQuant paper's Algorithm 1, stage 1) ----
   Format, per (layer, position, kv head):
     scale = ||x||_2 / sqrt(n)          one float, NOT a per-32 absmax
     code  = argmin_c |x[d]/scale - cents[c]|   4 bits, 2 per byte, low
                                                nibble first
     x_hat[d] = scale * cents[code[d]]

   Why one scale per row and not per 32. This is the difference that
   decides whether rotating the activations is worth anything. Q8_0's
   per-32 absmax already adapts to local variance, so spreading the
   energy with a Hadamard takes resolution away from the groups that had
   little of it - measured on this engine, that costs 0.08-0.30%% PPL
   (see forward.c's lz_kv_rot_enable). A single per-row scale cannot
   adapt, so the rotation's job - making every coordinate the same size -
   is exactly what it needs. The two choices only make sense together.

   Why a Gaussian codebook. After the norm is split off, x/||x|| is a
   direction in R^n, whose per-coordinate law is (1-z^2)^((n-3)/2)
   (IsoQuant eq. 36); at n = 256 that is Gaussian well past 4-bit
   resolution, and sqrt(n) = 16 makes the scaling exact. The table is
   the Lloyd-Max fixed point for N(0,1) (MSE 0.00950, SNR 20.22 dB -
   the textbook
   value for 4 bits, which is the check that it converged).

   The table is a constant of the FORMAT: a row decoded against a
   different table is wrong and looks plausible, so it is not
   configurable. */
extern const float lz_kv4_cents[16];

void lz_kv4_quantize(const float *x, int n, unsigned char *out, float *scale);

/* QJL (1-bit quantized JL sketch for KEY scores, arXiv:2406.03482)
   is NOT in the build. It is kept out rather than left
   compiled-but-unreachable because the rule is explicit that a
   path nothing can select does not exist.

   Both of its forms were built and measured, and both lost. As a
   standalone key cache (the paper's own usage) it was dominated on
   both axes - at the width where it stopped being terrible it already
   cost more memory than the 4-bit path (+7.7% PPL at s=1024 for
   37.3 MB, against q4's +0.6% for 36.3). As a stage-2 residual
   correction (--kv q4r2) it did worse on the actual target model at
   20,885 positions: +0.055% bits/token WORSE than plain q8 while
   spending 8.5 MB against q8's 8.6 - i.e. buying 0.1 MB with a
   significant quality loss. src/forward.c's KV comment carries that
   table.

   The negative result is the reason this comment stays. The reference
   construction lives in git history; that is not an invitation. */
/* GatedDeltaNet single-head single-step recurrence. State (kd, vd)
   row-major, stored as Q8_0 (group 32, contiguous within a row, hence
   vd % 32 == 0 required).

       delta = (v - gt*(S^T k))*beta
       S    <- gt*S + outer(k, delta)
       out   = S_new^T q

   The implementation touches state twice: pass 1 gathers both S^T k
   and S^T q, pass 2 updates and quantizes. Output uses the identity
   S_new^T q = gt*(S^T q) + delta*dot(k, q), so the state is not
   re-read. q
   must be pre-multiplied by 1/sqrt(kd) by the caller.

   sq_in/ss_in (read) and sq_out/ss_out (write) are DELIBERATELY
   separate pointers (the speculative-decode rollback ring - generate.c's
   lz_spec_round, forward.h's s->ssm_slot): passing the SAME pointer for
   both (sq_in==sq_out, ss_in==ss_out) reproduces the in-place behavior
   exactly bit for bit - pass 1 finishes reading before pass 2 writes
   anything, so self-aliasing is safe, this just makes it a caller
   CHOICE instead of the only option. Passing DIFFERENT pointers (read a
   live ring slot, write the NEXT one) is what lets a rollback become
   "point the ring index elsewhere" instead of "restore a snapshot and
   re-forward": pass 1 reads count-old-state bytes from sq_in exactly as
   before, pass 2 reads those SAME old bytes a second time (unchanged -
   it needs the pre-decay value) and writes the new ones to sq_out -
   same total bytes touched (2 read + 1 write) whichever way the
   pointers point, so a ring costs zero extra bandwidth over the
   single-slot version, only extra memory (see forward.h's s->ssm_slot
   comment for the account this depends on: every recurrence step here
   ALREADY reads and rewrites the entire state, so widening it to a ring
   adds no new traffic - if that stops being true, e.g. an
   incremental-update recurrence replaces this one, this argument must
   be re-derived). */
/* Fixed-point attention scoring and weighted sum. `-DLZ_ATTN_FIXED=1`
   replaces forward_attn's two f32 x int8 inner loops:

     scoring     q -> Q8 once per head, then int8(k) x int16(q) per
                 32-group through the SAME dot32_x16 the matmul uses
     weighted    out[d] = sum_t c_t[g]*vq_t[d], isomorphic to the SSM
     sum         recurrence's pass 1, so it has its own kernel

   It is a BITMASK, not a flag, because the two halves quantize at very
   different granularity and there was no measurement saying which one
   costs what:
     1  fixed scoring     (q -> int8; k was already int8 in the KV cache)
     2  fixed weighted sum (coefficients -> int16; v was already int8)
     3  both
   Scoring is the coarse one at 8 bits; the weighted sum's int16
   coefficients sit at the same granularity the SSM pass 1 uses, which was
   measured NOT to be a bottleneck. Splitting the knob makes that testable
   instead of assumed.

   This macro says which kernels are COMPILED; which one runs is
   lz_attn_mode(), a runtime selector. They were the same knob until the
   measurement below existed, and a compile-time-only tier cannot be
   compared against itself on one machine.

   Both kernels take kv_slot()'s sink and ring and walk the mapping
   themselves. Getting that wrong is silent - the read lands on a real
   row belonging to another position - and it cost a full diagnosis
   once: `scoring` wrote NaN into the last layer's attention output from
   the first evicted position on, and short measurements could not see it
   because the AUTO window does not open until about 1032 tokens.

   Quality: on 363 tokens all three arms sit within +-0.52 paired t of
   float with top-5 unchanged, against a bit-identical null arm; at 1440
   tokens, within 0.8 t, argmax agreement 99.8%.

   Speed on x86, where the float path is SSE2-vectorized, best of 5:
   1.021x at 180 tokens, 1.081x at 720, 1.136x at 1440 - measured
   ratios, not a fitted O(ctx) coefficient. A quadratic fitted through
   three timings turns three percent of noise into a factor of two, and
   did: the same procedure produced 3.7x and 1.54x on the same code.
   The x87 targets are where this tier is meant to pay; x86 only bounds
   it from below.

   lz_attn_wsum_q8's cbuf/cq are caller-owned scratch of seq_len entries
   each; its int32 accumulator only holds 516 rows, so it chunks at 512
   and converts each chunk through lz_i32f(). */
#ifndef LZ_ATTN_FIXED
#define LZ_ATTN_FIXED 3
#endif

/* Attention tier: "float" | "scoring" | "wsum" | "fixed" | "auto".
   Returns the resolved name, NULL on an unknown one. lz_attn_mode()
   returns the same bitmask LZ_ATTN_FIXED uses, masked by what was
   actually compiled, so a build with the kernels out cannot select them.

   AUTO asks the same hardware question as the conv and pass-2 tiers -
   CPUID SSE, never MMX - so a Socket 7 part lands on fixed and an SSE
   machine keeps the float path it has always had. It answers only "is
   this machine slow at float"; whether the kernels may run on this
   cache is a separate per-call test in forward.c. Context length is not
   part of either: the buffers are sized once, and a tier that changes
   underneath a running sequence is a worse bargain than one chosen once.

   NOT numerics-neutral, so switching tiers needs a tolerance gate. */
int lz_attn_mode(void);

/* Whether the fixed attention weighted sum keeps its int accumulator
   for lz_quantize_q8_int64 instead of dequantizing to float and
   re-quantizing (default off). Value-changing: the exact int64 sum
   replaces the float per-chunk drain, which rounds each 512-row chunk
   to 24-bit float, and the per-32-group dequant scale (cmax*LZ_Q15_INV)
   is not a power of two, so the stored group scales shift too. Rides on
   the wsum fixed tier rather than carrying a selector of its own. */
int lz_attn_int(void);

#if LZ_ATTN_FIXED
#define LZ_ATTN_MAX_HD 256
#if (LZ_ATTN_FIXED & 1)
/* sink/ring are forward.c's kv_slot() parameters, passed rather than
   looked up: these kernels get a cache pointer, not the state. ring 0
   means the cache is addressed by absolute position. Getting this wrong
   is silent - the read lands on a real row belonging to some other
   position - so the row walk is incremental and its wrap is asserted
   against the same formula in the gate. */
void lz_attn_score_q8(float *att, const float *qhh, int hd,
                      const int8_t *kc, const float *ks, int kvd,
                      int pos, float scale, int sink, int ring);
#endif
#if (LZ_ATTN_FIXED & 2)
void lz_attn_wsum_q8(float *out, const float *att, int hd,
                     const int8_t *vc, const float *vs, int kvd,
                     int pos, float *cbuf, int16_t *cq, int sink, int ring);
void lz_attn_wsum_q8_int(int64_t *acc, float *sscale, const float *att,
                         int hd, const int8_t *vc, const float *vs,
                         int kvd, int pos, float *cbuf, int16_t *cq,
                         int sink, int ring);
#endif
#endif

#define LZ_GDN_MAX_VD 256

/* Two-plane int8 SSM state (ON by default; -DLZ_GDN_STATE_2PLANE=0 to
   revert). The state is a high int8 plane plus a low int8 plane holding
   what the high plane rounds away, giving ~15 bits at 2x the state
   memory.

   Why two planes and not simply int16: pmaddwd sums two int16 products
   into one int32, and 2*32767^2 = 2147483648 is one past INT32_MAX, so an
   int16 state overflows inside a SINGLE instruction. Two int8 planes keep
   the existing int8 x int16 kernels and their proven bound
   (127*32767*256 = 1.07e9) unchanged - the second plane is just a second
   pass over the same kernel with the same coefficient table.

   Why it is worth it: the int8 state, not pass 1's coefficient precision,
   is the dominant error term in the recurrence. Measured on
   Qwen3.5-0.8B over 24 positions against an exact float-state reference:

     int8 state  + float pass 1                         argmax 24/24  top5 14/24
     2-plane st. + fixed pass 1 (this)                   argmax 24/24  top5 20/24

   Note the second row also settles a question the first could not: with an
   accurate state, the fixed-point pass 1 costs nothing. The
   logit-difference metric hid that, because it saturates.

   The 0.8B numbers above are the only valid evidence: the same sweep on
   s1v3 showed the same direction more strongly, but that model has
   misaligned embeddings (kept as models/s1v3_broken) - it computes
   deterministic garbage, whose logit distribution says nothing about
   how often a perturbation flips a REAL model's argmax.

   Cost, in the three currencies that matter, measured on
   s1v3-testfrozen (a FROZEN copy of a valid model):

     bytes    0.56 -> 1.12 MB of state, +4.12% of per-token streamed
              bytes. Transfers to the target.
     static   +15.6% instructions across the GDN path, counted from
              `wdis` on wcc386 -otexan objects built both ways
              (gdn_pass1_fixed +4, lz_gdn_step +1, and the 12-instruction
              lz_gdn_quantize_2p that replaces lz_quantize_q8).
              Transfers as a COUNT, not as a time.
     time     +7.1% end-to-end s/token, gcc/x86-64 dev box, 7 rounds min.
              **Does NOT transfer** - quoted only so a target
              run has something to disagree with.

   To get the target number, one command, no new instrument - the CLI
   already prints absolute s/token:

     llama98 <model> --prompt hello -n 48 --temp 0
     # -> [48 tokens, X s, Y s/token];  rebuild with
     #    -DLZ_GDN_STATE_2PLANE=0 for the other half of the ratio

   Note the static count is more than twice the dev-box time delta. That
   gap is the thing a target run decides: on an in-order-ish P6 with no
   spare issue width the count is closer to the truth than an
   out-of-order machine's wall clock. */
#ifndef LZ_GDN_STATE_2PLANE
#define LZ_GDN_STATE_2PLANE 1
#endif

/* LZ_KV_2PLANE's write path calls lz_gdn_quantize_2p, which this switch
   owns - so the two are coupled and nothing said so. Both at their
   non-default values (KV on, state off) built every TU cleanly and
   failed at the LINK, which is a place no gate was looking: the
   switch-matrix gate compiled ops.c and forward.c to /dev/null, and the
   call sits in forward.c under LZ_KV_2PLANE while the declaration sits
   here under LZ_GDN_STATE_2PLANE. Found by that gate's new all-at-once
   link. A contradiction that can be a compile error should be one. */
#if defined(LZ_KV_2PLANE) && LZ_KV_2PLANE && !LZ_GDN_STATE_2PLANE
#error "LZ_KV_2PLANE=1 needs LZ_GDN_STATE_2PLANE: it calls lz_gdn_quantize_2p"
#endif

#if LZ_GDN_STATE_2PLANE
/* State quantizer for the two-plane representation: `hi` and `s` come out
   exactly as lz_quantize_q8 would have written them, `lo` carries the
   residual at 1/254 weight. Exposed because the scalar reference has
   to write the same representation the implementation reads - the same
   reason it already calls
   lz_quantize_q8. `fc_site`: LZ_FC_* site to bill, or -1 for none. */
void lz_gdn_quantize_2p(const float *x, int n, int gs,
                        int8_t *hi, int8_t *lo, float *s, int fc_site);
#endif

/* Dequant weight of a low plane: value ~= (hi + lo/254) * s.

   256, not 254, and the choice is worth its paragraph because 254
   looked like the obviously right number.

   With a residual split - hi = round(v/s), lo = round((v/s - hi)*L) -
   the residual is bounded by half a step, so L = 254 makes lo fill int8
   exactly (0.5*254 = 127) and L = 256 would need 128, one past the end.
   That is the whole case for 254, and it is a real one.

   What it costs is that H = hi*L + lo can then only be split back apart
   by DIVIDING by L. MMX has no divide, and an exhaustive search found no
   magic multiply that keeps the residual inside int8 either (hi off by
   one throws lo by a full L, so the magic would have to be exactly
   round-to-nearest for all 64771 reachable H, which one arithmetic
   shift cannot be across both signs). So 254 blocks an MMX pass 2
   outright.

   L = 256 keeps the same two-step quantizer - hi through the SIMD
   rounding kernel exactly as before, lo through it again with a
   multiplier of 256 instead of 254 - and clamps the one boundary case
   (lo = +-128) to +-127, which the code already did. In exchange:

     - the split becomes hi = (H + 128) >> 8, lo = H - hi*256: shifts,
       no divide. Verified exhaustively that this round-trips every
       (hi, lo) pair in range, |H| <= 32639 < 32767.
     - the step is s/256 rather than s/254. PREDICTED to improve
       accuracy; MEASURED as a wash (float pass 2 1.78e-05 -> 1.82e-05,
       integer 4.35e-05 -> 4.27e-05). The finer step is
       cancelled by clamping: the residual reaches +-128 at L = 256 and
       the +-127 clamp discards the extremes, which L = 254 never had to
       do. So the case for 256 rests on the division-free split alone,
       not on precision - the prediction did not hold and is left here
       rather than quietly dropped.
     - 1/256 is exact in float where 1/254 is not, so every dequant
       loses a rounding.

   NOT claimed: fewer roundings in the quantizer. The split cannot be
   one rounding instead of two: that would need H computed directly,
   and the SIMD rounding kernel clamps to +-127 and cannot produce a
   +-32512 value. Keeping two roundings is what keeps the SSE path
   intact, and losing it would slow down exactly the machines that are
   already fast.

   Runtime representation only - the two-plane form is the in-memory
   SSM state and KV cache, never a file format - so no model needs
   re-exporting.

   Lives here rather than in ops.c because forward.c's KV path needs it
   too - see LZ_KV_2PLANE. */
/* #ifndef so the 254-vs-256 A/B stays buildable - a plain #define
   silently overrides the -D and the control reports "no difference",
   which reads exactly like "the change does not matter".

   That control is a build error, not a comparison, and the note
   belongs here rather than only at the assertion (ops.c,
   lz_p2_needs_lo_scale_256). The fixed pass 2 splits H with `>> 8` and
   `<< 8`, both hardcoded to L=256; a -DLZ_GDN_LO_SCALE=254.0f override
   runs a quantizer at 254 against a splitter at 256 and reports the
   result as if it were an A/B. A silent wrong answer is a refused
   build. The float pass 2 and the quantizer would still be comparable
   at 254; if that comparison is ever wanted again, the fixed pass 2 has
   to be compiled out with it, and that is more machinery than a settled
   question deserves. */
#ifndef LZ_GDN_LO_SCALE
#define LZ_GDN_LO_SCALE 256.0f
#endif

/* The subnormal floor, one number for the whole engine rather
   than a per-operator constant, because two operators that disagree
   about where zero starts would disagree about results.

   Three preconditions are on record for Watcom-vs-gcc bit-identity;
   the third is "no subnormal operands or results", and unlike the other
   two it cannot be met by writing the source differently. x87's PC=24
   fixes the MANTISSA at 24 bits but leaves the EXPONENT range extended,
   so a product stays normalized in the register, rounds once to 24 bits,
   and rounds AGAIN on the store down into the subnormal range; SSE
   rounds once. There is no exponent-range control on x87 to turn off.
   Measured: four 1-ULP disagreements at a subnormal group scale, with
   the SSE2 assembly and gcc's scalar agreeing against Watcom's x87.

   So the subnormal range is removed from the DATA. One place enforces
   it, and nothing downstream can reach it:

     - lz_quantize_q8 / lz_gdn_quantize_2p emit an exactly-zero group
       when the scale would fall below this, so no dequantized value is
       ever subnormal (smallest nonzero is s/254 >= 3.9e-33).

   Why 1e-30 and not something just above FLT_MIN: this makes the
   argument airtight rather than merely narrower. A value at or above
   1e-30 has an ULP of ~6e-38, while the largest possible x87-vs-SSE
   disagreement is one subnormal ULP (1.4e-45) - it cannot change a value
   that survives the floor, and a value that does not survive is zero on
   both sides. Against activations of order 1, discarding magnitudes
   below 1e-30 is not an approximation anyone can measure.

   It is #ifndef, not a plain #define, so a probe can build with the
   floor disabled (-DLZ_Q8_MIN_SCALE=0.0f) and check that the divergence
   comes back. A plain #define silently overrides the command-line value
   and the control then reports "identical without the floor" - a
   control proving the probe is blind reads exactly like a control
   proving the probe is unnecessary. The only warning is one line of
   "macro redefinition". */
#ifndef LZ_Q8_MIN_SCALE
#define LZ_Q8_MIN_SCALE 1.0e-30f
#endif

/* Two-plane int8 KV cache (OFF by default; -DLZ_KV_2PLANE=1 to enable).
   Same representation as LZ_GDN_STATE_2PLANE, applied to the persistent
   K/V cache instead of the recurrent state.

   MEASURED TWICE, and it stays off. The first measurement
   (s1v3-testfrozen, KV depths 4..73) gave argmax 23/24 and top-5 0.975
   - but it was taken in the wrong regime and the conclusion did not
   stand on it: deployment context is 2048, and KV quantization error is
   expected to bite at LONG context, where more cached rows enter each
   softmax and a small logit error has more chances to flip which
   position dominates. Measuring at depth <=73 and reporting "small" is
   the same defect as comparing a recurrence at sequence length 1
   (a criterion has to be shown sensitive before it is trusted).

   Re-measured on cpt100m-frozen (a 100M-token CPT product,
   frozen copy - never the live training dir), one plane vs two, ten
   independent windows per depth:

     KV depth    argmax agreement   top-5 overlap
        64            9/10              0.960
       512           10/10              0.960
      2048            9/10              1.000

   **Flat across the whole deployment range, including at full context.**
   Ten samples cannot separate 9/10 from 10/10 - what the sweep shows is
   the absence of a trend, which is exactly what the first measurement
   could not have shown.

   And the cost side was stated backwards the first time. Per element:

     one plane (shipping)  1 + f32 scale/32 = 1.125 B   ~8 bit
     two planes            2.125 B                      ~15 bit
     the F16 KV cache other engines default to   2.0 B  16 bit

   So the second plane is not "double for a marginal gain" - it is THIS
   ENGINE'S F16-KV EQUIVALENT, at F16's price, built out of int8 planes
   so the existing kernels still apply. The shipping single-plane path is
   the aggressive one, roughly llama.cpp's q8_0 KV.

   Why this model diverges from the common "F16 KV is noticeably more
   stable" experience: **the KV cache serves 3 of 12 layers here.** The
   other nine are GatedDeltaNet and carry their recurrent state instead -
   and that state DOES run two planes by default, because it was measured
   to matter (numbers above). The precision is already being spent where
   this architecture keeps its memory. In a pure Transformer every layer
   reads KV, and the same reasoning would come out the other way.

   Compare the SSM state's own numbers above: int8 state scored top-5
   14/24 against an exact-float reference, the two-plane state 20/24.
   **The KV cache's second plane is worth far less than the state's**,
   which is the direction the priors below predicted - the recurrence
   feeds its own error back, an append-only cache does not. Paying an
   extra ~1 B/element on the allocation that decides how many session
   slots fit in RAM, for no measurable change at full context, is not a
   default worth taking. The flag stays, so a deployment that wants
   F16-class KV can buy it with one define.

   Criterion note: agreement rates, NOT |logit difference| - that metric
   saturates.

   What is already known, and cuts both ways:
     - KV is only ~1.1% of per-token streamed bytes, so the DOUBLING here
       is cheap in bandwidth terms - unlike the SSM state's +4.12%.
     - But it doubles a PER-SEQUENCE allocation that scales with seq_len,
       which is the one that decides how many session slots fit in RAM.
     - And unlike the SSM state, KV error does not compound: the cache is
       append-only and each row is read back as written. The recurrence's
       error feeds itself; this one does not. So the prior should be that
       it matters LESS here, which is exactly why it needs measuring
       rather than assuming.

   Implementation note: no new kernel. Scoring and the weighted sum are
   both linear in the cached value, so the low plane is a SECOND PASS
   over the existing kernels, combined at 1/254 - the same trick the SSM
   state uses. That keeps bit-identity without writing two more
   assembly kernels. */
#ifndef LZ_KV_2PLANE
#define LZ_KV_2PLANE 0
#endif

/* Speculative decoding draft depth (OFF by default).
   0 disables it entirely; 1..6 is the legal range if it is ever enabled.

   A KNOB, not a constant, and for a reason this repo has already paid
   for once: LZ_BATCH_MAX above carries the same rule ("target machines
   must be sweepable"), because the optimum depends on how much
   memory-bound headroom a machine has, and that ratio FLIPS inside the
   target family. The plan's own sweep put the best k at 1, 2 or 4
   depending on tier, draft vocabulary and acceptance rate. Writing any
   one of them into the source is the "one machine's answer
   taken for the family's".

   THE COST, in the three currencies, because counting only one is how
   this feature's benefit case was wrong twice:

     bytes    a draft step streams the shared embedding as an output
              projection - upstream's config says
              mtp_use_dedicated_embeddings=false, so there is no smaller
              head to use. On this model that is embed_tokens at the p0
              recipe, which is a THIRD of the whole per-token weight
              stream. The draft block itself is small next to that.
     uop      verifying k+1 tokens costs (k+1) x the single-token ALU.
              Batching does NOT amortise it: the weight bytes come from
              L1 the second time, but the unpack/mask/shift work is
              redone per token (see LZ_BATCH_MAX's note - T=4 already
              flips prefill to compute-bound). So speculation trades ALU
              for bandwidth and only wins where bandwidth is the wall.
     time     NOT ESTABLISHED. Every multiple the plan published came
              out of a two-tier bandwidth model that the owner's PIII
              Coppermine 1000 refuted (prefetch measured <=5%, so the
              tiers collapse into one). Direction survives - PIII is
              still memory-bound - the numbers do not. Nothing here may
              carry a figure until a real s/token is measured on target.

   Transferable: the uop line (a static count, P6 ports). Not
   transferable: the byte line depends on the quantization recipe in
   force, and the time line does not exist yet.

   Not wired to anything: there is no draft head. This model has zero
   mtp.* weights, so a draft step cannot run at all. What IS implemented
   is lz_spec_accept (llama_zh.h), the acceptance rule,
   because that is the piece that decides whether speculation can change
   the output, and it can be judged right or wrong without a head. */
#ifndef LZ_SPEC_K
#define LZ_SPEC_K 0
#endif
/* Runtime ceiling for the CLI/opts override (--spec K in cli_main.c,
   LZGenOpts.spec_k in llama_zh.h): LZ_SPEC_K above is only the
   COMPILE-TIME default, same relationship LZ_BATCH_MAX has to nt_cap.
   State buffers sized off this constant (forward.h's s->mtp_logits,
   the MTP block's own KV cache depth is unaffected - see forward.h)
   must cover the full runtime range regardless of what LZ_SPEC_K
   defaults to. */
#define LZ_SPEC_K_MAX 6
#if LZ_SPEC_K < 0 || LZ_SPEC_K > LZ_SPEC_K_MAX
#error "LZ_SPEC_K must be 0 (off) or 1..6"
#endif

/* Depth-limited lookahead - generate.c's lz_look_pick.
   *** THIS IS NOT BEAM SEARCH, and the distinction is not pedantry. ***
   A beam search CARRIES W hypotheses forward: every step selects the
   next W survivors out of W x vocab (or, as SGLang narrows it, out of
   W x 2W), and beam state persists across the whole generation. This
   restarts from scratch at every emitted token: take the top-W
   candidates, roll each one forward D steps GREEDILY, score, keep the
   winner's FIRST token, discard the rest of the rollout. So it cannot
   return W ranked sequences - the thing beam search is actually used
   for in production (retrieval/recall, e.g. Kuaishou's OneRec) - and
   calling it beam would promise exactly that.

   BOTH limits are deliberately small, and the cost model is why: it
   runs W*D EXTRA forward passes per emitted token, SEQUENTIALLY. A GPU
   engine packs its W branches into one batch and pays little at the
   margin; a single-stream CPU engine has no such lever, so 4:4 is a
   16x slowdown on a machine whose whole reason for existing is that it
   has no compute to spare. There is no width/depth here that is
   "free"; these ceilings only stop a typo from turning a 3 s reply into
   a 20 minute one. */
#define LZ_LOOK_W_MAX 4
#define LZ_LOOK_D_MAX 4

/* The one recurrence body lz_gdn_step / lz_kda_step / lz_kda_step_vi16 all
   forward to. The three wrappers exist so the .prof probes have a stable
   entry point with the narrow signature they were written against; the
   engine's own callers (forward_ssm / forward_kda) call this DIRECTLY and
   fill in the wrapper's constants themselves, so the per-head call pays
   the 17-argument push once instead of twice (caller -> wrapper ->
   here). vi/vscale are the int16-v entry, gvec/gt the per-channel vs
   scalar decay - the three wrappers differ only in which are NULL/1.0f. */
void lz_recur_step(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                   const int8_t *sq2_in, int8_t *sq2_out,
#endif
                   const float *ss_in, float *ss_out,
                   const float *q, const float *k, const float *v,
                   const short *vi, float vscale,
                   const float *gvec, float gt, float beta, int kd, int vd);

void lz_gdn_step(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 float gt, float beta, int kd, int vd);

/* KDA (Kimi Delta Attention) single-head single-step recurrence: lz_gdn_step
   generalized from a single scalar decay `gt` to a PER-CHANNEL decay vector
   `gvec[kd]` (one already-exponentiated factor per key channel, gvec[i] =
   exp(g[i]) - same convention as lz_gdn_step's `gt`, just one value per row
   instead of one for the whole state).

   Derived from flash-linear-attention's naive_recurrent_kda /
   torch_recurrent_kda (per-timestep loop):

       S     <- diag(gvec) . S                      (decay, per k-row)
       delta = (v - S^T k) . beta                    (dot uses the DECAYED S)
       S     <- S + k (x) delta                      (outer product update)
       out   = S_new^T q

   Rewritten (see ops.c for the derivation) to the same two-pass shape
   lz_gdn_step uses, so it reads the Q8/2-plane state exactly once:

       u[v] = sum_k S_old[k,v] * (gvec[k]*k[k])       (= decayed S^T k)
       w[v] = sum_k S_old[k,v] * (gvec[k]*q[k])
       delta[v] = (v_t[v] - u[v]) * beta
       out[v]   = w[v] + delta[v] * (k . q)           (raw, unweighted dot)
       S_new[k,v] = gvec[k]*S_old[k,v] + k[k]*delta[v]

   q must already be L2-normalized and pre-multiplied by 1/sqrt(kd) by the
   caller, k must already be L2-normalized - same contract as lz_gdn_step.

   sq_in/sq_out/ss_in/ss_out split the same way lz_gdn_step's own does -
   see that function's comment for the full derivation (why in==out
   reproduces the in-place behavior exactly, and why in!=out costs
   no extra bandwidth). */
void lz_kda_step(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 const float *gvec, float beta, int kd, int vd);
/* Same, with v arriving as int16 at `vscale` - the conv's int16 exit
   (int-pipeline 9.4). The convert this loop then pays is smaller than
   the two the conv's float exit paid to avoid it. */
void lz_kda_step_vi16(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const short *vi,
                 float vscale, const float *gvec, float beta,
                 int kd, int vd);

/* Causal depthwise separable conv, single-step advance.
   state_in holds the last (k-1) inputs, layout (n_ch, k-1);
   state_out receives the rolled history. o[c] = silu( sum_{j<k}
   w[c*k + j] * hist[j] ), hist = [state_in columns..., x[c]].

   state_in/state_out split the same way lz_gdn_step's sq_in/sq_out
   does (the speculative-decode rollback ring) - passing the same
   pointer for both reproduces the in-place roll exactly: each output
   column j only ever reads state_in[j+1] (the OLD value, at an index
   no earlier iteration of this same call has written), then writes
   state_out[j], so self-aliasing is safe. See forward.h's s->ssm_slot
   comment for why a ring costs zero extra bandwidth here specifically:
   this function already reads the whole (k-1)-wide history and writes
   it back every call, incremental or not. */
void lz_causal_conv1d_step(float *o, const float *x,
                               const float *state_in, float *state_out,
                               const float *w, int n_ch, int k);

/* Fixed tier of the same operator, selected by lz_conv_mode(). History
   and taps are int16 at power-of-two exponents `es` and `ew[c]`, both
   produced by lz_conv_norm_pow2 against lz_conv_accum_bound(k). Aliasing
   sh_in == sh_out is safe on the same forward-order argument the float
   path documents. Drifts ~4.8e-5 from the float path - a tolerance
   gate, not cmp.

   `xi` non-NULL takes the input already quantized at `es` and clamped
   to +-bound (lz_matmul_xq_nt_i16's exit), and `x`/`in_scale` go
   unread: the int pipeline's producer-consumer join, one float
   multiply and one q8_round per channel deleted rather than moved.

   `sig_e` is the same per-channel exponent as `sig_k1`, ew[c] + es,
   carried as the integer it always was instead of as the float 2^(4-e)
   the folded coordinate needed. Under LZ_CONV_SIG_I it is what the
   activation reads and sig_k1/sig_k2 go unread; under =0 the reverse,
   and the caller passes NULL because it never allocated the table. */
void lz_causal_conv1d_step_fixed(float *o, const float *x, const short *xi,
                                 const short *sh_in, short *sh_out,
                                 const short *mw, const float *sig_k1,
                                 const float *sig_oscale2, float sig_k2,
                                 const signed char *sig_e,
                                 float in_scale, int bound,
                                 int n_ch, int k);
/* Same kernel with an INT16 EXIT: `oi` non-NULL writes the output there
   at 2^-out_es and leaves `o` alone. The exit is purely integer, which
   is a property of how the caller builds the scales rather than of this
   code - sig_oscale2 is an exact power of two whose exponent is
   -(sig_e[c]+15), so the epilogue is one 32x32 product and a shift.
   NULL is the float exit, which is what the wrapper above passes.
   Clamps are counted in lz_conv_o_clamped rather than being silent. */
/* Loop-invariant fixed-conv geometry, packed so the per-token call passes
   one struct pointer where Watcom would otherwise push seven memory
   operands (X86_TUNE_PUSH_MEMORY). Built once by the caller before its
   token loop; sig_e rides along because it is the same per-channel table
   as sig_k1, just the integer coordinate's exponent instead of the folded
   float one. */
typedef struct LZConvParams {
    const short *mw;
    const float *sig_k1;
    const float *sig_oscale2;
    float sig_k2;
    const signed char *sig_e;
    float in_scale;
    int bound;
    int k;
} LZConvParams;
void lz_causal_conv1d_step_fixed_o16(float *o, short *oi, int out_es,
                                 const float *x, const short *xi,
                                 const short *sh_in, short *sh_out,
                                 const LZConvParams *cp, int n_ch);
/* Non-zero means the int16 exit above hit its bound. LZ_CONVO_ES was
   picked at zero on both measured chains, so any count is a checkpoint
   this exponent was not measured on. */
extern long lz_conv_o_clamped;
/* Elements that took the int16 exit. The clamp counter above cannot
   serve as the switch's positive control - it is 0 in the healthy case
   AND 0 when the exit is compiled out. */
extern lz_i64 lz_debug_convo_i16;

/* Precompute the scales the fixed kernel multiplies by, once per model
   rather than per token. Built with ldexp: `1 << (ew+es)` overflows,
   the exponents reach 43 together. */
void lz_conv_build_scales(float *oscale, float *in_scale,
                          const signed char *ew, int es, int n_ch);

/* Latent MoE router (KunMoEGate). `logits` is the caller-computed
   x @ gate_w^T (n_experts elements, matmul done by lz_matmul_w - this
   function is only the small score/select/normalize math that has no
   matmul in it). `bias` (e_score_correction_bias) shifts SELECTION only,
   never the mixing weight - DeepSeek-V3's aux-loss-free balancing
   (noaux_tc): the top-k is chosen on (score + bias), but the weight
   gathered for each selected expert is its plain score. NULL bias is
   treated as all-zero. idx_out/w_out must hold top_k entries; an
   unfillable slot (top_k > n_experts) gets idx -1 / weight 0.

   `tau` is the router temperature: the WEIGHT for a selected expert
   becomes act(logit/tau) instead of act(logit), while SELECTION stays on
   the untempered scores. tau == 1.0f takes a branch that touches nothing,
   so the default is bit-identical to the engine that had no such
   parameter; tau <= 0 and NaN are treated as 1.0f.

   The two halves of that split are the point. tau < 1 sharpens the
   mixture without changing WHICH experts run, so it is the only way to
   trade off "how much does rank 3 get" separately from "does rank 3 run
   at all" - and on kunmoe-v2 those two questions have opposite answers:

     k=3, tau=1.0   -0.0005 nats vs the trained (k=2, tau=1) config
     k=4, tau=1.0   +0.0149          <- more experts alone is WORSE
     k=4, tau=0.5   -0.0210          <- more experts, weighted less
     k=2, tau=0.5   +0.0102          <- sharpening alone is WORSE

   (16384 tokens of cci3-hq, paired per-token NLL.) So --moe-topk without
   --moe-tau could only
   ever select a losing arm, which is why this parameter exists: a knob
   whose winning setting is unreachable is not a knob.

   NOT the default, and the reason is bandwidth, not quality: k=4 reads
   twice the expert weight bytes per token, and that is the dominant cost
   on every machine in the target family, for -2.07% ppl. The measured
   optimum for the DEPLOYED point stays (k=2, tau=1) - both directions on
   the tau axis are worse there, which is itself a measurement rather
   than an assumption (see moe_tau_flat.py's flattening half). */
/* `li`/`li_e` are the int-pipeline milestone 5 entry: non-NULL takes the
   router logits already int16 at exponent li_e (logits[i] == li[i] *
   2^-li_e, lz_matmul_xq_nt_i16's exit) and `logits` goes unread. Only
   the sigmoid scorer consumes them as integers - it has lz_sigmoid_i -
   and the softmax scorer and the tempered weight both rebuild the float
   vector first, since neither lz_exp nor a divide by a runtime tau has
   an integer entry to reach. NULL keeps the float input this function
   was written with. */
/* Loop-invariant router configuration, packed so the per-token call
   passes one struct pointer instead of nine arguments: Watcom's
   X86_TUNE_PUSH_MEMORY prices the seven that land past its register
   window as a memory push each, and ARM's AAPCS spills the same seven
   to the stack and reloads them every token. Built once by the caller
   before its token loop; only `logits` and `li` vary per token.
   idx_out/w_out ride along because they are the same fixed slots every
   token, not per-token destinations. */
typedef struct LZMoeRouteParams {
    int li_e;
    const float *bias;
    int n_experts;
    int top_k;
    int sigmoid;
    int renormalize;
    float tau;
    int *idx_out;
    float *w_out;
} LZMoeRouteParams;
void lz_moe_route(const float *logits, const short *li,
                  const LZMoeRouteParams *rp);

/* Accepted range for --moe-tau. Wide enough to contain everything that
   was measured (the sharpening half went to 0.25, the flattening half to
   6.0 and to the tau -> inf limit) with margin on both sides, and no
   wider, because outside it the answer is already known and the
   arithmetic gets worse at the same time:

     below 0.1  sigmoid(logit/tau) goes subnormal for this model's router
                logits, which the flush in lz_moe_route turns into the
                one-hot tau -> 0 limit - a defined answer, but one
                --moe-topk 1 gives more directly and more cheaply;
     above 10   the mixture is already within a few percent of the tau ->
                inf limit, and that limit measured +0.0637 nats, i.e.
                strictly worse than every point below it.

   A range is a claim about what was measured, so it moves only when
   somebody measures further out. */
#define LZ_MOE_TAU_MIN 0.1f
#define LZ_MOE_TAU_MAX 10.0f

/* Fold one layer's routing decision into a running per-expert tally for
   the current token. Called once per MoE layer; hits[e] ends up holding
   how many layers chose expert e.

   A count rather than a flag because the two are visibly different: an
   expert one layer glanced at and an expert three layers leaned on are
   not the same event, and on the real model they happen in a 60/8 ratio
   rather than being rare corners of each other.

   Takes the router's own idx_out array rather than an LZInspect, so that
   ops.h does not have to know what an inspector is - the caller owns
   that struct and this stays a plain function on plain numbers, which is
   also what makes it testable without a model.

   Two entries are skipped rather than counted: idx < 0, which is how
   lz_moe_route reports an unfillable slot, and idx >= cap, which does
   not fit the array. The second is NOT silently equivalent to "not
   selected" for the caller - it has to report the expert count
   separately, or a model with more experts than the array holds would
   look exactly like one that fits.

   Saturates at 255 rather than wrapping: a 256-layer model is not
   expected, and a count that silently returns to zero would draw an
   expert every layer used as one no layer used. */
void lz_moe_hits_add(unsigned char *hits, int cap, const int *idx, int k);

/* ---- x87 stack-empty gate (debug only) --------------------------------

   Root-caused this session (docs/x87-gcc-reg-stack-leak.md): gcc
   -m32 -mfpmath=387 -O2 can leave the x87 register stack non-empty on
   return from a function, and the residue survives to fault an
   unrelated LATER push elsewhere in the same forward pass. This checks
   the x87 tag word is all-empty (0xFFFF) at a call site and, if not,
   records where and what - it does NOT print or abort here. The
   console-output ban applies to this file same as any other in src/;
   cli_main.c is where a caught failure gets reported.

   Deliberately architected as "record, don't report" rather than one
   function that does both: the useful call sites (forward.c's per-layer
   loop) are deep inside the forward pass, with no access to stderr's
   caller-appropriateness the way cli_main.c has, and a debug facility
   that could only ever be called from one file would not have caught
   this bug at the layer boundary where it was actually found.

   CAVEAT, and it belongs here rather than only in the doc: this gate is
   itself a probe. Section "Why this is hard to reduce further" in
   docs/x87-gcc-reg-stack-leak.md found that even a pure-read fnstenv
   probe changes gcc's register allocation for nearby code enough to
   make the bug it was checking for stop reproducing. This gate can
   therefore be silent at one call site while its own presence in the
   binary is why a DIFFERENT, unchecked function nearby never leaked in
   the first place. A clean run under LZ_X87_STACK_GATE is evidence this
   gate did not observe a leak AT ITS OWN CALL SITES - it is not
   evidence the file has none.

   Mutation test: .prof/x87_gate_mutation.c (not built by anything, run
   by hand) proves the gate reddens on a deliberately unmatched `fld`
   and stays quiet on ordinary balanced code - see that file's own
   header for the exact command. */
#if defined(LZ_X87_STACK_GATE)
void lz_x87_stack_check(const char *where);
int  lz_x87_stack_dirty(const char **where, unsigned *tag);
#define LZ_X87_STACK_CHECK(where) lz_x87_stack_check(where)
#else
#define LZ_X87_STACK_CHECK(where) ((void)0)
#endif

#endif
