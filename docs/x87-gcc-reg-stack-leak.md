# x87 stack leak in gcc -O2 -m32 -mfpmath=387, reproduced in llama98

Status: **root-caused to a containing function; exact leaking instruction
not identified**. Worked around with a build flag
(`-fno-rerun-cse-after-loop`, see `build/x87/build_engine.sh`); not fixed
in source, because the trigger condition could not be reproduced in
isolation (see "Why this is hard to reduce further" below).

## One-sentence mechanism

`gcc -m32 -march=i686 -mmmx -mfpmath=387 -O2` leaves the x87 floating-point
register stack non-empty on return from a function reachable from
`gdn_p2_fixed_rows`'s per-row/per-group loop (`src/ops.c`), and this
residue survives long enough that an unrelated, later `fld` elsewhere in
the same forward pass (`lz_rope`'s scalar rotation, also in `src/ops.c`)
overflows the 8-deep physical stack, producing the hardware's masked
"real indefinite" QNaN (`0xFFC00000`) silently, which then propagates
through ordinary IEEE-754 NaN contagion into the whole output.

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
processing begins. The leak is inside `gdn_p2_fixed_rows`'s per-row/
per-group loop body (`p2_group_coef`, `gdn_p2_group_fixed`,
`p2_group_norm`, `p2_shift_of`, `p2_group_scale`), not before it and not
in the code that consumes its results afterward.

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
fails). Static disassembly of `p2_group_norm` shows a plausible
mechanism: six-plus branches implementing a min/max/clamp chain, at
least one exit path that manually drains two `fstp %st(0)` before
returning where the "normal" exit needs none - the kind of asymmetric,
hand-balanced stack accounting across multiple control-flow paths that a
compiler's register-stack allocator can get wrong on exactly one path
without it showing up anywhere else. The exact path and the exact
instruction were not identified, because every tool tried to look closer
(inline probes, `gdb` `stepi` single-stepping through the pristine
binary) either changed the outcome or ran out of a reliable way to
correlate a specific loop iteration (`kk`/`gg`) with a specific tag-word
reading in this optimized build.

## What fixes it, and what was tried and rejected

| flag | fixes it | notes |
|---|---|---|
| `-ffloat-store` | **no** | tried first, by analogy with the fix pattern in `p2_group_norm`'s own comment (force intermediates to memory). Does not reproduce that fix here - reported so nobody retries it expecting the same result. |
| `-fno-rerun-cse-after-loop` | **yes** | disables exactly one optimization pass. Chosen fix - see `build/x87/build_engine.sh`. Measured cost: no regression (500-token prefill, best of 3: 0.044-0.045 s/token without, 0.040-0.042 s/token with - the flagged build was measured faster, within noise, not a claimed speedup). |
| `-fno-inline` | yes | disables all inlining project-wide; strictly more invasive than the above for the same fix. Not used. |
| `-fno-strict-aliasing` | no | rules out aliasing UB as an alternative explanation. |
| `-fno-tree-vectorize`, `-fno-schedule-insns[2]`, `-fno-reorder-blocks`, `-fno-gcse`, `-fno-caller-saves`, `-fno-ipa-ra`, `-fno-lra-remat`, `-fno-crossjumping`, `-fno-code-hoisting`, `-fno-hoist-adjacent-loads`, `-fno-tree-pre`, `-fno-tree-vrp`, `-fno-unroll-loops`, `-fno-cse-follow-jumps` | no | full diff of `-O1`-vs-`-O2`-enabled passes, tested individually. |

## What is NOT yet known

- The exact instruction or control-flow path inside
  `p2_group_coef`/`gdn_p2_group_fixed`/`p2_group_norm`/`p2_shift_of`/
  `p2_group_scale` that leaves the stack non-empty.
- Whether this is a gcc `reg-stack` pass bug specifically, or a more
  general x87-extended-precision register-allocation issue that happens
  to manifest as a stack-depth leak in this instance (the sibling issue
  `p2_group_norm`'s own comment documents was a *value* divergence, not
  a stack-depth leak - the two may or may not share a root cause inside
  gcc).
- Whether other gcc versions reproduce this (only 13.3.0 was available
  in this environment).

## Suggested next steps for whoever picks this up

1. Try other gcc major versions (12, 14, trunk) on the same reproduction
   command to bound whether this is version-specific.
2. `-fdump-rtl-all` (or specifically the `reg-stack`/`stack` pass dump)
   on `src/ops.c` at `-O2 -m32 -mfpmath=387`, to see the allocator's own
   accounting for `p2_group_norm` rather than inferring it from
   disassembly and runtime probes.
3. A non-perturbing trace (Bochs with x87 state logging, or a QEMU TCG
   plugin that reads FPU state without recompiling the guest binary)
   would sidestep the probe-sensitivity that blocked `gdb stepi` and
   inline instrumentation this session.
4. If reduced to a small standalone reproducer, file upstream at
   <https://gcc.gnu.org/bugzilla/> - this document is written to make
   that filing possible without re-deriving the evidence chain, but a
   minimal reproducer was not constructed this session (every attempt to
   isolate the failing function in a smaller test file changed its
   compiled shape enough to stop reproducing).
