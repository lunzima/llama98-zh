# x87 stack leak in gcc -O2 -m32 -mfpmath=387, reproduced in llama98

Status: **ROOT CAUSE: gcc holds seven x87 values live across a call into an eight-slot stack. See "The mechanism, named". The workaround remains a workaround - no source fix is possible, the C is correct.** Worked
around with a build flag (`-fno-rerun-cse-after-loop`, see
`build/x87/build_engine.sh`); not fixed in source, because the trigger
condition could not be reproduced in isolation (see "Why this is hard to
reduce further" below).

An earlier version of this document, and of the build script's own
comment, stated the mechanism below as an established fact - "leaves
the x87 stack non-empty on return from a function reachable from
`gdn_p2_fixed_rows`'s per-row/per-group loop." **That claim is wrong**,
disproven by the static verifier described in "Static verification: the
depth-imbalance theory is dead" below, and left here struck through
rather than deleted because the evidence chain that built it (the tag
word walk, the ruled-out candidates, the probe-sensitivity finding) is
still real and still the best evidence available - it just does not
prove what it was believed to prove. Read the whole document; do not
stop at the first mechanism paragraph.

## One-sentence mechanism (SUPERSEDED by "The mechanism, named")

The paragraph below was the first attempt and it is wrong in its
attribution, not in its symptom. The confirmed mechanism is that gcc
holds seven x87 values live across a call into an eight-slot stack; see
that section. The struck-through text stays because the evidence that
built it is still real.


~~`gcc -m32 -march=i686 -mmmx -mfpmath=387 -O2` leaves the x87
floating-point register stack non-empty on return from a function
reachable from `gdn_p2_fixed_rows`'s per-row/per-group loop
(`src/ops.c`), and this residue survives long enough that an unrelated,
later `fld` elsewhere in the same forward pass (`lz_rope`'s scalar
rotation, also in `src/ops.c`) overflows the 8-deep physical stack,
producing the hardware's masked "real indefinite" QNaN (`0xFFC00000`)
silently, which then propagates through ordinary IEEE-754 NaN contagion
into the whole output.~~

What is still true: the symptom is exactly this (a QNaN from an
overflowing push, propagating through NaN contagion). What is now known
false: that it is a **push/pop count imbalance**, and that
`gdn_p2_fixed_rows` is where it happens. A static verifier
(`build/x87/x87_stack_check.py`) that reads the compiled instructions
directly - no execution, no probe, cannot be perturbed by its own
presence - finds every join balanced and every return at ABI depth,
**everywhere in the binary**, including `gdn_p2_fixed_rows`'s fully-
inlined body. The three tail calls and the other unresolved control-flow
edges the verifier could not follow by itself have since been resolved
by hand (see below); none of them hide a discrepancy either. The
contradiction between that result and a real, reproducible `gdb`
observation (three x87 registers occupied at a specific program point)
is the actual open question - see "Static verification" below.

## Environment

```
$ gcc --version
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
$ uname -a
Linux ... 6.6.87.2-microsoft-standard-WSL2 ... x86_64 GNU/Linux
```

Reproduced under WSL2 on an x86-64 host, compiling and running a genuine
32-bit ELF binary (not emulated - the host CPU executes it natively).
Target flags: `-m32 -march=i686 -mmmx -mfpmath=387 -O2 -std=c99
-ffp-contract=off`. This is the exact configuration AOSC OS / Afterglow
package the engine in (Socket-7-class hardware: MMX present, no SSE, x87
the only float unit), which is why `-mfpmath=387` is forced rather than
letting gcc pick SSE2 scalar float at `-m32`.

## Reproduction

Clean checkout, `build/x87/build_engine.sh` **without** the
`-fno-rerun-cse-after-loop` workaround (i.e. drop that one flag from the
script, or build directly):

```
gcc -m32 -march=i686 -mmmx -mfpmath=387 -O2 -std=c99 -ffp-contract=off \
    -Isrc -Icommon -o llama98-x87 \
    src/err.c src/compat.c src/lfn.c src/json.c src/safetensors.c \
    src/model.c src/ops.c src/forward.c src/sampler.c src/generate.c \
    src/tokenizer.c src/unicode.c src/chat.c src/gbk.c src/cpucheck.c \
    src/cli_main.c src/cli_argv.c src/cli_attr.c \
    common/session.c common/stream.c -lm

./llama98-x87 <model-dir> --kernel ref --attn-write float --tokens 1 \
    --dump-logits /dev/null
```

Result: all 32768 output logits are `0xFFC00000` (bit-identical, single
token, first forward pass). The same source, same flags except dropping
`-O2` for `-O0`/`-O1`/`-Og`, produces the numerically correct (and, for
`-O0`, bit-identical-to-itself-across-multiple-runs) result. Same source
compiled by Open Watcom (32-bit x87 target) and by this same gcc at
`-m64` (SSE2 scalar float, not x87) both produce the correct result and
agree with each other bit-for-bit. Only this one combination -
`-m32 -mfpmath=387 -O2`, this exact runtime configuration - fails, and it
fails on the very first token.

## Why this is not the engine's arithmetic

- Bisected across optimization levels: clean at `-O0`/`-O1`/`-Og`, broken
  only at `-O2`, same C source throughout.
- `-fno-strict-aliasing` does not fix it (rules out an aliasing
  violation as the underlying UB).
- Forcing a guaranteed-empty x87 stack immediately before the group
  computation that eventually produces the NaN (`emms` inserted right
  before `p2_group_coef`'s call, in a scratch copy, never landed in
  `src/`) makes that computation produce the *correct* answer every
  time (`c1 = 0.0`, matching the true zero SSM state at `t=0`). The
  arithmetic is right; something about which physical register gets read
  is wrong.
- The corrupted register's x87 tag word entries are `Zero` (`01`) and,
  once corrupted, `Special` (`10`, hardware NaN/Inf/denormal
  classification) - never `Valid` (`00`). MMX writes to an MM register
  unconditionally set the corresponding tag to `00` regardless of the
  64-bit pattern being written; the FPU's own value classification
  (which produces `01`/`10`/`11`) is only performed by genuine x87
  instructions. This rules out "an MMX kernel forgot to call `emms`" as
  the mechanism, even though this build does have at least one MMX
  kernel (`lz_gdn1_x4_asm`, GDN pass 1) that runs unconditionally
  regardless of `--kernel ref` - traced separately, confirmed to leave
  the tag word byte-identical before and after it runs (`0x3fff` both
  times, on the failing binary), so it is not the source.

## Evidence chain

### 1. Isolation matrix (from the original bisection)

| configuration | result |
|---|---|
| `--kernel ref --attn-write float` | all 32768 logits `0xFFC00000` |
| `--kernel ref --attn-write scoring/wsum/fixed/auto` | correct |
| `--attn-write float` with `--kernel mmx` (default) | correct |
| Open Watcom, 32-bit x87 | correct |
| gcc `-m64` (SSE2) | correct, bit-identical to Watcom |

`--attn-write scoring/wsum/fixed` do not visibly fail because they
quantize the query vector to int8 before using it; a NaN input to that
quantizer's group-scale computation (`q8_group_scale`) hits
`if (amax > 0.0f)`, which is false for NaN under IEEE-754, and falls back
to a defensive `sg=1.0, inv=0.0` - the corrupted element stays wrong, its
31 group-mates become a defensible zero, and nothing propagates further.
Confirmed by direct gdb inspection: `qht[0]` is `0xFFC00000` under
`--attn-write fixed` and `--attn-write scoring` too, identically to
`float` - the corruption happens upstream of attention-mode dispatch and
is present on every run of this binary; only the float path's plain IEEE
arithmetic propagates it visibly.

### 2. Tag-word walk, forward through a forward pass

Using `gdb` breakpoints on the unmodified failing binary (no source
instrumentation - see the probe-sensitivity section for why that
distinction matters), reading the x87 tag word via `info float`
(equivalent to `fnstenv`'s tag-word field):

```
layer 0 entry                                  tag=0xffff  clean
head 0, before beta = lz_sigmoid(...)          tag=0xffff  clean
head 0, after beta, before g = ...*softplus()  tag=0xffff  unchanged by sigmoid
head 0, after softplus, before gt = lz_exp(g)  tag=0x3fff  +1 (the in-flight softplus return value)
head 0, after exp, before lz_gdn_step          tag=0x3fff  unchanged, exp adds nothing
head 0, after GDN pass 1 returns               tag=0x3fff  UNCHANGED from before pass 1
head 0, entering gdn_p2_fixed_rows             tag=0xffff  clean again - gt was flushed
head 0, entering the row loop (8 heads sampled) tag=0xffff  clean every time
head 1, entry (next head, top of per-head loop) tag=0xffd5  dirty: 3 registers occupied
```

Everything up to and including entry to `gdn_p2_fixed_rows`'s row loop is
provably balanced. The state is clean on entry to every sampled head's
row loop, yet dirty (`0xffd5`) by the time the *next* head's own
processing begins.

**Superseded conclusion, kept for the record:** this session originally
concluded from this table alone that "the leak is inside
`gdn_p2_fixed_rows`'s per-row/per-group loop body... not before it and
not in the code that consumes its results afterward." The static
verifier (below) finds that entire loop body, fully inlined into
`gdn_step_impl`, depth-balanced with no exceptions. The `gdb` table
above is not wrong as a set of measurements - the tag word really does
read `0xffff` entering the row loop and `0xffd5` at the next head's
entry - but the INFERENCE that the code between those two points must
be where a push/pop count goes wrong is what the static check disproves.
Something about the runtime state changes between those two
observations; a static, count-only analysis of the code that runs
between them says the count is not it.

Two plausible-looking candidates were disassembled and ruled out
directly rather than by inference:

- **GDN pass 1's MMX kernel** (`lz_gdn1_x4_asm`, confirmed via a direct
  `fprintf` probe to run unconditionally under `--kernel ref` - a
  separate, real finding, since `lz_gdn_mode()` has no runtime override
  at all): tag word before and after are bit-identical, so it does not
  touch the depth.
- **The `lz_softplus`/`lz_exp` caller sequence**, including a
  profiling-timer branch (`LZ_PROF_BEG`/`LZ_PROF_END`-shaped, calling
  `lz_time_ms`) that duplicates `gt` on the stack before branching:
  disassembled both control-flow arms, both reach the rejoin point at
  identical depth (`D_in+1`, i.e. `gt` correctly sitting there as an
  ordinary in-flight return value). Not a leak.

### 3. The stack was full, not partially occupied

An earlier decode of one captured tag word (`0x2015`) was miscounted as
"5 of 8 registers occupied." Corrected: `0x2015` has zero `11` (Empty)
pairs - **all eight registers were occupied**, i.e. the very next `fld`
in that state has nowhere to go and must fault. This matters for
interpretation: the row loop does not need to leak gradually across many
iterations to explain the failure; a single leaked push, on top of
whatever was already resident, is sufficient once the physical stack is
already near capacity. Row-by-row monotonic growth (predicted and tested
explicitly - `fnstenv` at the top of the `kk` loop, expecting
`0xFFFF -> 0x3FFF -> 0x0FFF -> ...`) was **not observed**; the tag word
goes dirty once and then holds at a fixed, non-growing signature
(`0xffd5`) for dozens of subsequent iterations. This is more consistent
with one (or a small, fixed number of) leaked push(es) whose effect then
stabilizes, not a per-iteration accumulation.

## Why this is hard to reduce further

`p2_group_norm` (`src/ops.c`) has a comment, written before this NaN bug
was ever found, describing the exact same class of fragility for a
*different*, already-fixed symptom (x87-extended-precision vs. SSE
divergence at a specific recurrence step):

> "Not theoretical: without this guard the two builds diverge at
> recurrence step 26, and compiling the tracing probe - whose only
> effect is to STORE these intermediates - makes the divergence
> disappear. A bug that hides when you look at it is this mechanism,
> every time."

This session reproduced that mechanism independently, for the NaN bug,
before finding the comment: inserting `fnstenv`-only (pure read, no
writes) diagnostic probes inside `gdn_p2_fixed_rows`'s loop - intended
purely as instrumentation - was itself enough to change gcc's register
allocation for the surrounding code and make the NaN stop reproducing
entirely (12 samples, all clean, where the uninstrumented binary reliably
fails). Static disassembly of `p2_group_norm` was read, at the time,
as showing a plausible mechanism: six-plus branches implementing a
min/max/clamp chain, at least one exit path that manually drains two
`fstp %st(0)` before returning where the "normal" exit needs none - the
kind of asymmetric, hand-balanced stack accounting across multiple
control-flow paths that a compiler's register-stack allocator can get
wrong on exactly one path. **The static verifier below checked this
directly and found `p2_group_norm` depth-balanced on every path, so
this specific candidate is now ruled out, not just unconfirmed** - kept
here because the probe-sensitivity lesson above is still the reason
nothing further was learned by looking harder at any one function by
hand, and because the fragility class it demonstrates
(exactly this kind of asymmetric multi-exit branch code) may still be
where the real, non-depth mechanism lives even though this particular
manifestation of it was not the one.

## The mechanism, named: seven values live across a call into eight slots

Open Watcom compiles this same source for the same 32-bit x87 target and
is correct. Comparing the two disassemblies at the same call answers it:

| | live x87 depth at the call to `p2_group_norm` |
|---|---|
| Open Watcom | **0** |
| gcc 13.3 `-m32 -march=i686 -mfpmath=387 -O2` | **7** (tag word `0xe015`) |
| gcc 14.2, same flags | **7** (tag word `0xe005`) |
| gcc 16.1 (mingw), same flags | peak **7** in the same function (static count; not run - no 32-bit libs) |

**Not a version regression.** Three gcc releases spanning 13.3 to 16.1
put the same peak in the same function, and 13.3 and 14.2 both produce
all-NaN output and are both repaired by the same flag. Upgrading the
toolchain is not a route out.

The x87 stack is one fixed eight-slot resource. Nothing reserves any of
it for a callee - see "What rule is broken, stated carefully" below for
what the ABI does and does not say about this.

**The hardware says which fault it is, and it is underflow.** At the
breakpoint the status word reads `0x0061`: `IE` (invalid operation),
`PE`, and `SF` (stack fault) set, with **`C1` clear**. `C1` is what
separates the two: set means overflow, a push onto a full stack; clear
means **underflow, a read from a register the tag word marks empty**.
So the code is not pushing a ninth value - it is reading a slot it never
wrote. An earlier version of this document said overflow; that was an
inference, this is the flag.

The register file at that point names the hoisted values outright:

```
st3  1.0000000000000000e-30          LZ_Q8_MIN_SCALE
st5  12582912                        q8_round's magic constant, 1.5 * 2^23
st6  0xffff0000000000000000          never written
st7  0xffffc000000000000000          real indefinite
ftag 0xe015                          seven occupied
```

Two loop-invariant float *constants* are sitting in x87 registers.
That is exactly what a hoisting pass does, and it is why
`-fno-rerun-cse-after-loop` is the flag that helps. `p2_group_norm` pushes on top of those
seven, the push past the eighth raises a masked stack overflow, and the
hardware delivers real indefinite (`0xFFC00000`) instead of a value.
Every NaN downstream is that one, propagated by ordinary IEEE contagion.

**`p2_group_norm` is not at fault.** It is balanced, its own peak is
small, and it computes correctly whenever it is entered on an empty
stack. Three rounds of investigation examined it because that is where
the NaN first becomes visible; the defect is at the call site, in what
the caller is still holding.

**What builds the pressure is loop-invariant hoisting, not inlining.**
An earlier version of this section blamed inlining, on the strength of
`-fno-inline` also hiding the symptom and of Watcom keeping the two
helpers as real functions. Direct test refutes it: marking either
`gdn_p2_group_fixed` or `gdn_p2_fixed_rows` `__attribute__((noinline))`
leaves the tag word at `0xe015` - seven occupied - at the call, and the
output still all-NaN. The seven live values are held inside
`gdn_p2_fixed_rows` itself across its own call, whether or not it is
inlined into its caller.

That is consistent with which flag works: `-fno-rerun-cse-after-loop`
disables the pass that hoists loop-invariant values, and `-fno-inline`
helps only because changing inlining changes what is loop-invariant.

**Five source-level fixes were tried and measured. All five fail.**

| attempted fix | result |
|---|---|
| `__attribute__((noinline))` on `gdn_p2_group_fixed` | still all-NaN, tag word unchanged at `0xe015` |
| `__attribute__((noinline))` on `gdn_p2_fixed_rows` | still all-NaN, tag word unchanged |
| `__attribute__((always_inline))` on `p2_group_norm` (remove the call boundary entirely) | still all-NaN |
| `volatile` on `c1`/`c2`, forcing the coefficients through memory | still all-NaN |
| the `LZ_Q8_MIN_SCALE` comparison rewritten in the integer domain, so that constant never needs a register | still all-NaN |

The pattern is the point: **displace one value and the allocator hoists
another into the slot.** Each of these removes a real live value and the
stack still fills. A source fix that works would have to reduce the
peak below what the allocator wants, not move one item out of it, and
nothing in C expresses that.

**There is therefore no source-level fix, and `noinline` is not one.**
No annotation in C says "do not keep values in x87 registers across this
call" - that is what the register allocator is for.

**What rule is broken, stated carefully.** An earlier version of this
document claimed the i386 ABI requires the x87 stack to be empty across
a call. **That claim is not supported by the specification.** The
Intel386 psABI (1.0, 2015) Table 2.3 says only:

| register | usage | preserved across calls |
|---|---|---|
| `%st0` | scratch; also returns float/double/long double | No |
| `%st1`-`%st7` | scratch | No |

"Scratch, not preserved" binds the *callee's* freedom to clobber. It
does not require an empty stack at entry, and no such sentence appears
in that document. `p2_group_norm` is also `static`, so gcc sees every
call site and is entitled to use a non-standard convention for it; the
psABI does not govern that call at all.

The defect does not need an ABI to be a defect. **There are eight
physical registers and gcc arranged for more than eight values to be
live at one point.** The callee pushes onto a stack that already holds
seven, the ninth push raises a masked overflow, and the hardware returns
real indefinite. That is a resource over-allocation - a miscompilation
on its own terms, whatever the calling convention says.

The one ABI rule verified as relevant here is the MMX one, and it is
not the mechanism: "The CPU shall be in x87 mode upon entry to a
function. Therefore, every function that uses the MMX registers is
required to issue an emms or femms instruction after using MMX
registers, before returning or calling another function." That was
ruled out separately by the tag-word classification.

This is a register-allocation defect, not a source defect: the same C is
correct under two other toolchains, and the allocator's obligation is to
spill before a call it cannot fit.

## Static verification: the depth-imbalance theory is dead

Every runtime probe attempted above changed gcc's register allocation
enough to stop the bug reproducing (the previous section). The fix is
an instrument that cannot be perturbed by its own presence: read the
already-compiled bytes instead of running or re-running anything.
`build/x87/x87_stack_check.py` does this - objdump text in, basic
blocks and a per-mnemonic x87 stack-depth delta, propagated over the
CFG, reporting any join point whose predecessors disagree and any
`ret` reached at the wrong ABI depth.

**Result on the actual failing binary: zero depth violations, anywhere.**
`gdn_step_impl` (which contains `gdn_p2_fixed_rows`, `gdn_p2_group_fixed`
and `p2_group_coef` fully inlined - none of the three exist as separate
symbols on this build), `p2_group_norm` itself (not inlined, called),
and `lz_rope` all report clean individually. Across the whole binary:
82 functions contain x87 instructions, all 82 checked, only 8 findings,
and every one of the 8 is `UNRESOLVED` (an indirect call or a tail-call
jump the checker cannot statically follow) rather than an actual
violation.

**Getting a trustworthy "clean" required finding and fixing five real
bugs in the checker itself**, all caught by the x86-64/Watcom control
runs the tool was required to also report nothing on:

1. GCC clone suffixes (`lz_exp.part.0`) not recognized as sharing the
   original function's signature, scoring a real float-returning call
   as void and cascading into phantom joins through `lz_sigmoid`.
2. Bare `nop` misclassified as an x87 instruction, making `has_x87()`
   fire on almost any function (nop is padding, everywhere) and running
   full CFG analysis - not hardened for large non-numeric control flow -
   on functions with no real x87 content. Produced 236 phantom findings
   on the x86-64 control alone.
3. `float *lz_mtp_draft_step(...)` (pointer return, EAX) included in the
   float-returning table from an unchecked grep hit instead of the real
   header declaration.
4. Watcom's PE disassembly prints `0x`-prefixed branch targets; the
   parser only handled bare hex, so every branch on the Watcom control
   looked indirect.
5. Watcom appends a trailing underscore to every exported symbol
   (`lz_sigmoid_`), and x86-64's ABI returns float/double through
   `%xmm0` rather than `%st(0)` at all - two independent naming/ABI
   gaps, both caught by the same control requirement.

Every one of these was an OVER-report (a phantom finding), never a
false "clean" - the tool has been wrong five times and every time in
the direction of crying wolf, not staying silent. That asymmetry is the
actual argument for trusting the final negative result, not the result
by itself.

### The 8 unresolved sites, resolved by hand

The checker cannot statically follow an indirect call or a tail-call
jump, so those 8 sites were the one place a real leak could still be
hiding despite the "clean" verdict - `lz_gdn_step`'s tail call into
`gdn_step_impl` was the likeliest, since that is exactly the call this
whole investigation is about. Resolved one at a time:

- **`lz_gdn_step` -> `gdn_step_impl`** (tail call, `jmp`): the function's
  entire x87 content is `flds 0x58(%esp); flds 0x5c(%esp); fstps
  0x54(%esp); fstps 0x50(%esp)` - two pushes marshaling `gt`/`beta`
  into `gdn_step_impl`'s different argument frame (it takes an extra
  `gvec=NULL` `lz_gdn_step` doesn't), two pops consuming them right back
  out. Net zero. Depth is exactly 0 at the `jmp`, matching
  `gdn_step_impl`'s own (separately verified) depth-0 entry.
- **`lz_kda_step` -> `gdn_step_impl`** (tail call): one `flds`/one
  `fstps` (only `beta` needs marshaling; `gt=1.0f` is a compile-time
  constant for this family). Net zero, same conclusion.
- **`lz_matmul_xq_nt` -> `matmul_scalar_ref`** (tail call): traced with
  the checker's own CFG machinery rather than by eye (the function is
  too large to read reliably) - depth is 0 at every path reaching the
  jump instruction, 20 predecessor paths sampled, all agreeing.
- **`q8_amax`, `lz_rmsnorm`, `lz_softmax` (x2)** (indirect calls through
  a dispatch-table function pointer, e.g. `lz_amax_pick(...)`): depth is
  0 immediately before each call, on every path reaching it. Separately,
  under `--kernel ref` (the failing configuration), `LZ_DEFINE_PICK`
  returns `NULL` for every one of these tables, so the indirect call is
  behind an `if (f) f(...)` that is never taken at runtime on this
  configuration at all - moot for this specific bug regardless of the
  static depth result.

**All 8 resolve cleanly. None hides a depth discrepancy.** The
tail-call hypothesis - the most promising remaining place for a
count-mismatch to be hiding from the checker - is disproven.

### The open contradiction

Two measurements, both individually trustworthy, disagree:

- `gdb` on the pristine, unmodified binary: tag word `0xffd5` (three
  registers occupied) at a reproducible program point (entry to a
  head's own per-head setup, right after the previous head's full GDN
  processing completes).
- The static checker, cross-validated three ways (self-test, x86-64
  control, Watcom control) and with all 8 of its own blind spots
  resolved by hand: no push/pop count is ever unbalanced anywhere in
  the compiled code that runs between one observation and the next.

One piece of evidence about which side of this to trust for the FIX,
not the diagnosis: `emms` inserted before `lz_rope`'s call produces the
EXACT correct answer (`rms=1.4744957526166522`, bit-identical to the
`-O0` build), not merely a different one. If the actual mechanism were
gcc reading the wrong `%st(i)` - a real value sitting in the wrong slot -
clearing the tag word would not fix that; it would still compute from
whatever garbage now occupies the slot `emms` didn't change the
identity of, just its emptiness. That the answer becomes exactly right
argues the register INDICES gcc's code uses are correct, and only the
incoming TAG STATE is wrong - which is a depth/occupancy question, not
an identity one, and pushes the open question back to where those three
occupied slots come from if not from a compiled-code count mismatch.

Not yet pursued, because if the mechanism above is right the answer
would not need it: whether the `Edge A->B: correcting stack to [...]`
lines in gcc's own `-fdump-rtl-all` `326r.stack` pass output are
reconciliation-pass overrides of an earlier per-block estimate (routine)
or a genuine internal disagreement (informative) was left an open
question rather than guessed at - see the RTL dump note below if this
document's next step is to resolve it after all.

## What fixes it, and what was tried and rejected

| flag | fixes it | notes |
|---|---|---|
| `-ffloat-store` | **no** | tried first, by analogy with the fix pattern in `p2_group_norm`'s own comment (force intermediates to memory). Does not reproduce that fix here - reported so nobody retries it expecting the same result. |
| `-fno-rerun-cse-after-loop` | **yes** | disables exactly one optimization pass. Chosen fix - see `build/x87/build_engine.sh`. Measured cost: no regression (500-token prefill, best of 3: 0.044-0.045 s/token without, 0.040-0.042 s/token with - the flagged build was measured faster, within noise, not a claimed speedup). |
| `-fno-inline` | yes | disables all inlining project-wide; strictly more invasive than the above for the same fix. Not used. |
| `-fno-strict-aliasing` | no | rules out aliasing UB as an alternative explanation. |
| `-fno-tree-vectorize`, `-fno-schedule-insns[2]`, `-fno-reorder-blocks`, `-fno-gcse`, `-fno-caller-saves`, `-fno-ipa-ra`, `-fno-lra-remat`, `-fno-crossjumping`, `-fno-code-hoisting`, `-fno-hoist-adjacent-loads`, `-fno-tree-pre`, `-fno-tree-vrp`, `-fno-unroll-loops`, `-fno-cse-follow-jumps` | no | full diff of `-O1`-vs-`-O2`-enabled passes, tested individually. |

## What is NOT yet known

- **The actual mechanism.** Depth/count imbalance is ruled out by static
  analysis of the real binary, cross-validated three ways, with all 8
  of the checker's own blind spots (indirect calls, tail calls) since
  resolved by hand and none of them hiding a discrepancy either. What
  is left is a real, reproducible runtime observation (`gdb`: 3 x87
  registers occupied at a specific point) that a push/pop-count reading
  of the compiled code cannot explain. See "The open contradiction"
  above for the one piece of evidence bearing on which side to trust
  for a fix (the `emms` result argues register INDICES are correct and
  only occupancy/depth is wrong at runtime - which is odd, since that is
  exactly what the static checker also says is fine. This tension is
  itself unresolved.)
- The leading open hypothesis - a "right count, wrong `%st(i)` index"
  bug invisible to a depth-only checker - is now in tension with the
  `emms`-produces-the-exact-right-answer evidence, which argues INDICES
  are fine and only depth/occupancy is wrong. Neither hypothesis is
  currently fully consistent with all the evidence; this is the actual
  state of the investigation, not a gap to be embarrassed about hiding.
- Whether this is a gcc bug at all, as opposed to something in how the
  compiled program's runtime FPU state evolves that a static reading of
  the instructions - correct as far as it goes - cannot see (interrupt/
  signal handling, CRT startup state, something outside the reg-stack
  pass's own accounting).
- Whether other gcc versions reproduce this (only 13.3.0 was available
  in this environment).

## Suggested next steps for whoever picks this up

1. **Build a symbolic slot-identity tracker** on top of
   `build/x87/x87_stack_check.py`'s existing CFG machinery - track which
   pseudo-value, not just how many, occupies each physical slot, and
   flag a join point where predecessors agree on depth but disagree on
   identity. This is the most direct way to test the "wrong index" idea,
   despite the `emms` tension noted above - it would either confirm an
   index bug the depth checker cannot see, or rule that out too and
   narrow the remaining space further.
2. Reconcile why `gdb`'s tag-word reading and a from-source static count
   disagree at all, given the static analysis is now trusted. One
   concrete angle not yet tried: read the tag word with `gdb` at the
   EXACT same program counter the static checker computes a depth for,
   rather than at a nearby source-line breakpoint - the two might not
   currently be looking at the literal same instruction.
3. Try other gcc major versions (12, 14, trunk) on the same reproduction
   command to bound whether this is version-specific.
4. A non-perturbing trace (Bochs with x87 state logging, or a QEMU TCG
   plugin that reads FPU state without recompiling the guest binary)
   would sidestep the probe-sensitivity that blocked `gdb stepi` and
   inline instrumentation this session, and could settle the
   contradiction directly rather than through more inference.
5. Resolve the `Edge A->B: correcting stack to [...]` semantics in the
   `-fdump-rtl-all` `326r.stack` dump (see "The open contradiction"
   above) - deliberately not spent effort on this while the UNRESOLVED
   sites were still open; now that they are closed and clean, this is a
   reasonable next thing to spend the effort on rather than defer again.
6. If reduced to a small standalone reproducer, file upstream at
   <https://gcc.gnu.org/bugzilla/> - this document is written to make
   that filing possible without re-deriving the evidence chain, but a
   minimal reproducer was not constructed this session (every attempt to
   isolate the failing function in a smaller test file changed its
   compiled shape enough to stop reproducing), and given the depth
   theory is now dead, a reproducer would need to target whatever
   mechanism step 1 or step 2 above actually finds, not the original
   leaked-push theory.
