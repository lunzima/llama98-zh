#ifndef LZ_OPS_H
#define LZ_OPS_H

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

/* o = x * rsqrt(mean(x^2) + eps) * (1 + w)   - Qwen3_5RMSNorm */
void lz_rmsnorm(float *o, const float *x, const float *w, int n, float eps);

/* o = x * rsqrt(mean(x^2) + eps) * w * silu(g)  - Qwen3_5RMSNormGated
   Order matters: normalize and multiply by weight first, gate last. */
void lz_rmsnorm_gated(float *o, const float *x, const float *g,
                          const float *w, int n, float eps);

/* o = x * rsqrt(sum(x^2) + eps)  - note sum, not mean */
void lz_l2norm(float *o, const float *x, int n, float eps);

/* exp = 2^(x·log2e) via bit construction. Default path; -DLZ_EXACT_MATH falls back to libm.

   ~69K exps per token, nearly all from conv1d's SiLU (lin_conv_dim ×
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
float lz_silu(float x);
float lz_softplus(float x);

/* x87 precision control: set the control word to 24 bits (single
   precision) so x87 float semantics become isomorphic to SSE - the
   prerequisite for bit-identical Watcom and gcc builds. Returns the
   original control word; lz_fpu_float_end restores it. No-op on
   non-Watcom builds.

   No double inside the region (incl. libm double routines): PC=24
   compresses double to 24 bits too. lz_exp is all-float now;
   lz_softplus restores precision internally. */
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
   float reduction, which iron law two forbids (see the "no 4-way
   float accumulator across groups" note in the r>1 path). A
   vectorized epilogue can only ever be a separate tier that is bit
   identical to itself.

   MEASURED, and worth stating because a plausible argument says the
   opposite: on a real Pentium M the SSE2 path beats the MMX path by
   more than 20%, even though Pentium M splits a 128-bit SSE2 op into
   two micro-ops and therefore has the SAME nominal throughput as
   64-bit MMX. Width is not the only variable - XMM has 16 registers
   against MMX's 8, SSE2 does not alias the x87 stack (so the emms
   accounting in iron law six clause 6 goes to zero), and one 128-bit
   instruction occupies one decode slot where two 64-bit ones occupy
   two.

   So do NOT "simplify" Pentium M back onto the MMX kernel on the
   grounds that its 128-bit ops are 2 uops. That reasoning is exactly
   what the measurement refutes. */
#define LZ_KERNEL_AUTO 0
#define LZ_KERNEL_REF  1
#define LZ_KERNEL_MMX  2
#define LZ_KERNEL_SSE2 3
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
/* GDN pass-1 tier is always "fixed" now (there is no --gdn selector);
   lz_gdn_mode() returns 0 only when the fixed path was not compiled in
   (-DLZ_GDN_FIXED=0). The float body remains reachable as a capacity
   fallback for kd > LZ_GDN_MAX_KD. */
int lz_gdn_mode(void);

/* "fixed" | "float" | "auto". SEPARATE from the pass-1 tier above:
   pass 1's fixed path adds ~1.5e-05 and ships by default; pass 2's adds
   ~1.2e-05 on top of a 1.8e-05 baseline. See
   gdn_p2_group_fixed in ops.c.

   AUTO ASKS THE HARDWARE QUESTION - does this machine have a SIMD tier
   for the write-back quantize? - and answers "fixed" only where it does
   not, i.e. a Pentium II or K6-2. Two earlier assumptions behind that
   reasoning have been re-measured:

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

   One host row cannot decide a family (iron law 3), so the default is
   unchanged pending a PII/PIII measurement. The knob is the decision
   point, not this comment. */
const char *lz_gdn_p2_select(const char *name);
int lz_gdn_p2_mode(void);
/* Was the tier chosen by AUTO rather than asked for? The banner says so,
   because "fixed" that a machine picked and "fixed" that a flag forced
   are different facts and only one of them travels to another machine. */
int lz_gdn_p2_is_auto(void);
/* Which of the fixed pass-2 bodies is selected: "ref", "mmx-intrin",
   "mmx-asm", or "-" when the float tier is running. The same join
   lz_kernel_tier() reports, and it exists for the same reason: the
   bodies are required to be bit identical, so nothing in the output can
   tell you one of them stopped being compiled in. */
const char *lz_gdn_p2_impl(void);

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
   quantizes x, then int8×int8->int32 accumulation, dequantized per
   group by (sx * sw).
   xq/xqs are caller-provided quantization buffers (capacity in_dim
   bytes / in_dim/gs f32s). */
void lz_matmul_w(float *o, const float *x, const LZTensor *w,
                 int in_dim, int out_dim, int8_t *xq, float *xqs);

/* Q8 matmul consuming ALREADY-QUANTIZED activations: the forward hot
   path reuses one quantization for q/k/v/gate/up sharing xb's xq/xqs,
   avoiding repeated activation scans. dtype=0 (f32 weights) uses x in
   f32 - one call site serves both weight representations. */
void lz_matmul_q8(float *o, const float *x, const int8_t *xq,
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
   iron law puts C / MMX / SSE / SSE2 paths on every operator
   unconditionally, because the bottleneck is a seesaw and "optimise
   whichever side binds today" leaves both undone.

   THE CAP IS 8, THE DEFAULT IS 4, AND THAT SPLIT IS DELIBERATE. The cap
   only sizes buffers (kunkun98-pilot state 2.2 -> 2.6 MB, the 0.8B
   28.8 -> 29.8, plus g_xw 32K -> 64K static). The default is what runs.

   **4 IS SETTLED (user decision after testing). Do not
   reopen it on the strength of the dev-box number.** T=8 is faster
   here - kunkun98-pilot 512-token prefill 3.591 s vs 4.051 s - and that
   is exactly the kind of number iron law 3 says not to promote into a
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

/* Batched lz_matmul_q8: one weight stream serves nt tokens.

   All three buffers are token-major with hardcoded strides, no extra
   params:
     o  [t * out_dim + i]
     x  [t * in_dim  + k] (f32-weight fallback path only)
     xq [t * in_dim  + k]
     xqs[t * (in_dim / lz_act_gs(w, in_dim)) + g]

   **nt=1 is bit-identical to lz_matmul_q8** - not "numerically close",
   the t-loop in the same code degrades to one iteration and the
   reduction structure is untouched byte for byte. This is a hard
   gate: batching changes weight-load count, not what is computed. */
void lz_matmul_q8_nt(float *o, const float *x, const int8_t *xq,
                     const float *xqs, const LZTensor *w,
                     int in_dim, int out_dim, int nt);

/* Pure f32 matmul (for differential/debug use; semantics same as
   lz_matmul_w's dtype=0 branch) */
void lz_matmul(float *o, const float *x, const float *w,
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
   is not a power of two for odd log2(n) (n=128: 1/sqrt(128)), and iron
   law 6 forbids dividing by a float constant that the two compilers
   strength-reduce differently. Leaving the sqrt(n) in and folding the
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

   Scalar only, deliberately (iron law 2: a new transform lands as a
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

   WHY ONE SCALE PER ROW AND NOT PER 32. This is the difference that
   decides whether rotating the activations is worth anything. Q8_0's
   per-32 absmax already adapts to local variance, so spreading the
   energy with a Hadamard takes resolution away from the groups that had
   little of it - measured on this engine, that costs 0.08-0.30%% PPL
   (see forward.c's lz_kv_rot_enable). A single per-row scale cannot
   adapt, so the rotation's job - making every coordinate the same size -
   is exactly what it needs. The two choices only make sense together.

   WHY A GAUSSIAN CODEBOOK. After the norm is split off, x/||x|| is a
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
   compiled-but-unreachable because iron law eight is explicit that a
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

       delta = (v - gt·Sᵀk)·beta
       S    <- gt·S + k ⊗ delta
       out   = S_newᵀ q

   The implementation touches state twice: pass 1 gathers both Sᵀk and
   Sᵀq, pass 2 updates and quantizes. Output uses the identity
   S_newᵀq = gt·(Sᵀq) + delta·(k·q), so the state is not re-read. q
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

   Default off: whether it is worth enabling has to be decided by
   argmax/top-5 agreement against an exact reference, not by comparing
   error magnitudes - the logit-difference metric saturates.

   lz_attn_wsum_q8's cbuf/cq are caller-owned scratch of seq_len entries
   each; its int32 accumulator only holds 516 rows, so it chunks at 512
   and converts each chunk through lz_i32f(). */
#ifndef LZ_ATTN_FIXED
#define LZ_ATTN_FIXED 0
#endif

#if LZ_ATTN_FIXED
#define LZ_ATTN_MAX_HD 256
#if (LZ_ATTN_FIXED & 1)
void lz_attn_score_q8(float *att, const float *qhh, int hd,
                      const int8_t *kc, const float *ks, int kvd,
                      int pos, float scale);
#endif
#if (LZ_ATTN_FIXED & 2)
void lz_attn_wsum_q8(float *out, const float *att, int hd,
                     const int8_t *vc, const float *vs, int kvd,
                     int pos, float *cbuf, int16_t *cq);
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
              bytes. Transfers to the target (rule 3).
     static   +15.6% instructions across the GDN path, counted from
              `wdis` on wcc386 -otexan objects built both ways
              (gdn_pass1_fixed +4, lz_gdn_step +1, and the 12-instruction
              lz_gdn_quantize_2p that replaces lz_quantize_q8).
              Transfers as a COUNT, not as a time.
     time     +7.1% end-to-end s/token, gcc/x86-64 dev box, 7 rounds min.
              **Does NOT transfer** (rule 3) - quoted only so a target
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

#if LZ_GDN_STATE_2PLANE
/* State quantizer for the two-plane representation: `hi` and `s` come out
   exactly as lz_quantize_q8 would have written them, `lo` carries the
   residual at 1/254 weight. Exposed because the scalar reference has
   to write the same representation the implementation reads - the same
   reason it already calls
   lz_quantize_q8. */
void lz_gdn_quantize_2p(const float *x, int n, int gs,
                        int8_t *hi, int8_t *lo, float *s);
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

   THAT CONTROL IS A BUILD ERROR, not a comparison, and the note
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

/* THE SUBNORMAL FLOOR, and it is one number for the whole engine rather
   than a per-operator constant, because two operators that disagree
   about where zero starts would disagree about results.

   Iron law 2 lists three preconditions for Watcom-vs-gcc bit-identity;
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

   Both are #ifndef so a probe can build with the floor disabled
   (-DLZ_Q8_MIN_SCALE=0.0f) and check that the divergence comes back.
   The first attempt at that control used a plain #define here, so the
   command-line value was silently overridden and the control reported
   "identical without the floor" - a control that proves the probe is
   blind, reading exactly like a control that proves the probe is
   unnecessary. The compiler said so ("1 warnings", macro redefinition)
   and the line scrolled past. */
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
   (iron law 4).

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
   state uses. That keeps iron law 2 satisfied without writing two more
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
   one of them into the source is iron law three's "one machine's answer
   taken for the family's".

   THE COST, in the three currencies, because counting only one is how
   this feature's benefit case was wrong twice:

     bytes    a draft step streams the shared embedding as an output
              projection - upstream's config says
              mtp_use_dedicated_embeddings=false, so there is no smaller
              head to use. On this model that is embed_tokens at the p0
              recipe, which is a THIRD of the whole per-token weight
              stream. The draft block itself is small next to that.
     µop      verifying k+1 tokens costs (k+1) x the single-token ALU.
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

   Transferable: the µop line (a static count, P6 ports). Not
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
   whose winning setting is unreachable is not a knob. See iron law 9.

   NOT the default, and the reason is bandwidth, not quality: k=4 reads
   twice the expert weight bytes per token, and that is the dominant cost
   on every machine in iron law 3's table, for -2.07% ppl. The measured
   optimum for the DEPLOYED point stays (k=2, tau=1) - both directions on
   the tau axis are worse there, which is itself a measurement rather
   than an assumption (see moe_tau_flat.py's flattening half). */
void lz_moe_route(const float *logits, const float *bias, int n_experts,
                  int top_k, int sigmoid, int renormalize, float tau,
                  int *idx_out, float *w_out);

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

#endif
