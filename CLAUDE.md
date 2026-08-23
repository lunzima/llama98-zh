# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

llama98 is a C99 inference engine for the KunMoe LLM architecture (KDA linear
attention + latent MoE), plus the 昆昆98 (kunkun98) Win32 front end. The target
hardware is Pentium-class machines running Windows 9x, NT 3.51 and DOS. The dev
machine is amd64/MinGW. Both must produce **bit-identical** output.

## Iron law zero: what belongs here, and what never will

**This repository holds the inference engine and the front end. It never holds
a test suite and it never holds training tools.** Not "does not yet" - never.
`src/`, `common/`, `gui/`, `build/`, `docs/`, `assets/`, `contrib/` and nothing
else. Verification is what the `Verify` section below describes: modes of the
shipped binaries.

The training side - corpus packing, tokenizer and charset construction,
quantized export, the Python test suite - lives in a **separate repository**,
and `E:\LLM\llama98-zh` is where it lives today. That tree is ALSO where this
engine's source used to live; **its copy of the engine is dead and awaiting
deletion**. Nothing there is authoritative for `src/`.

Two consequences that have already caused work to land in the wrong place:

- **The two trees share no history.** 20 commits here, 536 there, different
  root commits, neither contains the other's. They are not clones of each
  other and `git` will not help you reconcile them. A change made in the old
  tree does not arrive here by any mechanism except being written again.
- **mtime lies about which is current.** The dead tree gets touched daily
  because the training work is there, so its `src/*.c` will usually look
  newer. The evidence that decides is this file plus the remote: this
  repository has `origin` at `github.com/lunzima/llama98-zh`; the other has
  no remote at all.

A format added to the engine therefore lands **half here and half there** by
construction - the kernel and the loader here, the exporter that produces the
weights over there - and neither half can be tested against the other inside
one repository. Say so when it happens; the failure mode is a format the
engine can read and nothing can write.

## Build

One Makefile drives two toolchains. Source lists live once, in `ENG` / `SRV` /
`GUI` / `COM` at the top of the Makefile — a new source file is added in exactly
one place.

```sh
make native gui serve        # gcc: llama98, kunkun98-gui, kunkun98-serve
make check                   # Open Watcom: watcom-cli + watcom-gui + watcom-dll
make watcom-dos              # DOS target (needs the extender stub)
make watcom-dos LZ_DOS_EXTENDER=dos32a   # DOS/32A instead of dos4gw
make clean
```

Requirements: gcc/MinGW for the `native`/`gui`/`serve` targets; Open Watcom 2.x
at `C:\WATCOM` for every `watcom-*` target (cross-built from `binnt64` under Git
Bash, `binl64` under WSL — the Makefile probes and picks); `python3` for
`watcom-dll` (`build/watcom/gen_exports.py`); DOSBox-X to run the DOS build.

**`make check` IS the correctness gate for the target build.** A clean DLL link
proves `llama98.def`'s export surface still matches the engine, and `-D__MMX__=1`
is baked in so the hand-written assembly cannot silently vanish (wcc386 does not
predefine it; without it every `#pragma aux` kernel disappears and matmul falls
back to scalar — *while staying bit-identical*, which is why no numeric gate
catches it).

## Verify

There is no unit-test framework. Verification is done with the shipped binaries:

```sh
./llama98 <model-dir> --check                 # structural validation, fast
./llama98 <model-dir> --tokens 1,2,3 --topk 5 # forward on fixed ids
./kunkun98-gui --selftest report.txt          # GUI: builds the window, checks
                                              # every control, writes
                                              # "checks N / fails N / skips N",
                                              # exit != 0 on any failure
kunkun98-serve <model-dir> --requests 5       # serve N requests then exit
```

`--selftest` is a mode of the *shipped* GUI binary, not a separate build. It is
the closest thing to a test suite in the repo — when touching anything in `gui/`,
run it and diff the report.

A **skip** is a check that could not be run, not one that failed: it does not
affect the exit code, and `checks + skips` is constant, so a check that silently
stopped running shows up in a diff. Two things skip today — the clipboard checks
when another process holds the clipboard through the retry window, and the
status-strip pixel comparison when the counterpart image is missing or predates
the current binary. The strip comparison needs **two runs** of the same build
(`classic_ui=0` and `classic_ui=1`) and splits its differing-byte count: only
`drawn by this program` is a claim about the port, since the comctl32 build puts
a real progress control in that one cell and comctl32 v6 draws it its own way on
a modern host.

Numeric changes are gated by **bit-identity comparisons**, using
`--dump-logits F` / `--dump-nll F` / `--dump-all-logits F` and swapping one knob:

```sh
--kernel ref|mmx|sse2     # every tier must agree bit for bit
--batch 1..LZ_BATCH_MAX   # every prefill width must agree
--spec 0 vs --spec K      # greedy speculation must not change output at all
--kv-rot off              # must reproduce the pre-rotation engine exactly
--prefetch none|nta|...   # numerics-neutral by contract
```

A KV-precision or PPL claim needs ~20k predicted positions; 1k cannot resolve
effects this size (recorded in `forward.h`, with the measurement that showed it).

### The tiers that are NOT bit-identical, and what gates them instead

`--conv-write`, `--attn-write`, `--epi-write` and `--sig-write` select
fixed-point arithmetic. Swapping them changes the numbers by construction, so
the comparison above does not apply and a tolerance takes its place — against
the Q8 activation noise floor of 3.9e-3 relative, which is the smallest
difference anything downstream can feel.

Both tiers of each ship. Which is faster depends on the target's
float-to-integer cost ratio, and this development host answers only for itself:
the fixed tiers do 2.7x to 15x less f32 work and run at 0.87x to 1.14x the
speed here, because a modern core's float is fast and vectorized. Iron law nine
- a contested choice is a knob.

**Cross-compiler bit-identity still holds inside every tier**, which is the
part that is NOT a tolerance. A tier changes the arithmetic, so gcc-fixed and
gcc-float differ; gcc-fixed and Watcom-fixed must not. Verified on kunmoe
d_ref, every tier alone and all four together, at ntok 8 and at ctx 1440 where
the KV ring engages: identical bytes in all cases. The tiers therefore use
`pow2f()` rather than `ldexp` - a libm double inside the PC=24 region is barred
by iron law six clause 3, and three ways to build a power of two would be three
chances to disagree.

`|logits diff|` is NOT the criterion for any of them: it saturates. Two
measurements at different scales, both reported because the number means
nothing without its convention:

- **Per perturbation, one position**: 1.89e-2 for the smallest possible
  perturbation and 1.30e-1 for a larger one, on the same model. A change a
  thousand times bigger moves this by less than a decade.
- **Worst over a whole sequence**: 1.55 to 1.66 across the tier configurations,
  taken as the maximum over all 152 positions of a zh fixture on kmr20. Two
  orders of magnitude above the per-perturbation figures, from the same
  arithmetic — the maximum over many positions is a different quantity, not a
  contradiction.

Neither ranks tiers, which is the point. Use paired NLL, argmax and top-5
agreement (`--dump-nll`, `--dump-top5`).

### Where the numeric probes live

`.prof/` holds 60-plus C probes and 20-plus Python ones - the instruments behind
every number in these laws. It is **gitignored by decision**, so it does not
survive a clone; `.prof/README.md` classifies what is there. Probes that drive
the shipped code (`wfwd.c`, `wconv2.c`, `wattnbreak.c`, `wepi.c`) compile
`src/*.c` directly rather than transcribing it, which is the difference between
measuring the engine and measuring a copy of it.

The numbered iron laws themselves live in the TRAINING tree's `CLAUDE.md`
(`E:\LLM\llama98-zh`), not here - a split this file's iron law zero explains and
does not endorse. When a numeric law changes, it changes there.

## Architecture

### Layering, and the one rule that defines it

```
src/          the engine       — no console I/O, no sockets, no Win32
common/       front-end core   — LZSession: history, render, prefix reuse
gui/          Win32 front end  — kunkun98
src/cli_main.c, src/server_main.c   the CLI and HTTP front ends
```

Console output exists **only** in `cli_main.c`, `server_main.c` and `gui/`. The
engine never prints and never exits: every failure returns an `LZErr` code and
fills a caller-supplied `errbuf` (`src/err.h`; text is bilingual EN/ZH, switched
at runtime by `lz_set_error_lang`). `SRV` (`net`/`http`/`openai`) is deliberately
outside `ENG` — there are no sockets on the target machines.

### The model, and the state asymmetry that drives everything

Hybrid layers alternate: linear attention (`LZ_LT_LINEAR` GatedDeltaNet, or
`LZ_LT_KDA` Kimi Delta Attention with per-channel decay) and gated full
attention (`LZ_LT_FULL`). FFN is either dense SiLU-gated or a latent MoE
(top-k routed experts through a compressed latent space + optional shared
expert), resolved per layer at load time into `LZLayer.ffn_moe`.

Two kinds of runtime state, and they do not behave alike:

- full_attention → a **KV cache** that grows with position. "Rolling back" is
  just not advancing the position pointer; stale rows get overwritten.
- linear_attention → a **fixed-size SSM recurrent state + conv history**. It has
  no rollback. That single fact explains why trimming a conversation re-renders
  the whole prompt, why cross-turn reuse goes through an `LZStateCkpt`
  checkpoint keyed on the *token sequence* (not a position), and why a session
  pool slot is a whole `LZRunState` rather than a checkpoint ring.

`src/llama_zh.h` is the public API and carries the contracts (ownership of
borrowed buffers, what `spec_k = 0` must be identical to, why `out_stop` exists).
Read the header comment before changing a signature — the reasoning for most
decisions is written down at the declaration, not in a separate doc.

### Front-end core

`common/session.h`'s `LZSession` owns the conversation loop for **all three**
front ends: history (in the engine's `LZChatHist`, never a second copy), prompt
render, prefix-cache reuse, reply accumulation, turn trimming.

`gui/session.h`'s `LZGuiSession` **flat-inherits** it: the first fields are
`LZSession`'s in exact order and type, so `(LZSession *)&gui_session` is valid,
enforced by `offsetof` typedef-array asserts at the bottom of that header (C99
has no `_Static_assert`). Reordering either struct is a compile error by design.

`gui/session.h` includes the core as `"../common/session.h"` — the plain quoted
spelling would resolve to itself.

### GUI specifics

- `gui/worker.c` runs generation on a background thread. The sequence before
  touching anything the worker can see is **always** `lz_worker_join_drain()` —
  join alone leaves posted `WM_APP_TOKENS`/`WM_APP_GEN_DONE` in the queue.
- `gui/compat40.c/h` is the NT 3.51 floor: the build compiles at
  `WINVER=0x0351` with `-we` (W131 "no prototype" becomes fatal), and every
  4.0-era nicety is resolved at run time and degrades when absent. That gate
  catches *functions*, not constants — so `compat40.h` never self-declares a
  hidden Win32 prototype (that would undo the floor for the one thing it is
  good at); it supplies a small annotated set of constants, and its own
  `lz_*` helpers wrap the run-time-resolved calls.
- Text is UTF-8 everywhere except at the display boundary. Win9x has no
  `CP_UTF8`, so the GUI converts to GBK when drawing. **Never convert back:** a
  character GBK cannot represent has already become `?`, and writing that into
  history diverges the next render from the KV cache permanently. The reply
  buffer (`LZSession.reply`, UTF-8, filled beside the sink on the worker thread)
  is the authority, never the RichEdit control.

## Conventions that will bite you

Comments cite **"iron law N"** — the numbered project constitution. The document
itself lives outside this repo; what the tree evidences:

1. **No console output below the front end.** The engine accumulates counters;
   `cli_main.c` prints them.
2. **Cross-compiler bit-identity.** The Watcom build is not optional. Hence
   `-ffp-contract=off` (gcc's FMA folds one rounding where x87 does two),
   x87 precision control set to 24 bits (`lz_fpu_float_begin`), the engine's own
   `lz_exp`/`lz_rsqrt` instead of libm, and no reassociation of float reductions
   — a 4-wide SSE epilogue changes association order and is forbidden unless it
   is a separate, self-identical tier.
3. **Measure; a ratio needs a measurable denominator.** `--kv f32` and
   `--prefetch none` exist purely so the other arms mean something. Timing
   traps are documented at `compat.c`'s `lz_time_ms` (18.2 Hz on DOS, 15.6 ms on
   Windows — "all readings are multiples of one number" is the tell).
4. **Self-consistency is not correctness**, and a clamped value that gets
   reported as the requested one is the recurring bug shape — refuse instead.
5. (no in-tree citation found)
6. **Nothing large on the stack** — `static` instead, with a note on why it is
   safe there (single-threaded front end, written before read). Rule 1 bans
   divides in the forward path; wall-clock conversions are the stated exemption.
7. **Source files stay ASCII.** Every user-visible string lives in a table
   (`src/err.c`, `gui/localized_strings.c`: enum + parallel bilingual arrays +
   designated initialisers), and Chinese inside `src/` is hex-escaped
   (`"\xCB\xCE\xCC\xE5"`). The string tables themselves are the exemption —
   there the Chinese is data.
8. **Every operator carries C / MMX / SSE / SSE2 paths**, and the gate is a
   four-way bit-identity comparison. A path the CLI cannot select cannot be
   compared, so every tier is reachable via `--kernel`.
9. **A contested choice is a knob, not a decision.** If the right value differs
   across the target family, it becomes an option with a default, not a
   constant. (`LZ_DOS_EXTENDER`, `--pair`, `--prefetch`, `stream_ms`, …)
10. **The target family includes big-endian machines and a 1995 C89 compiler.**
    Windows NT on MIPS, built with Visual C++ 4.0, is a target. So the language
    floor is C89 as that compiler implements it, and the data floor is "byte
    order is not an assumption you get to make".

    What that costs, measured rather than feared — `grep` over `src/` today:
    **213 uses of `long long`** and **23 files including `<stdint.h>`**. VC++ 4.0
    has neither: the 64-bit type is `__int64` and the fixed-width names do not
    exist. Both need to go behind one header that each toolchain answers in its
    own terms; scattering `#ifdef _MSC_VER` at 213 sites is the shape rule
    "one authority per fact" exists to forbid. Also absent there: `//`
    comments (the tree is already clean — 0 occurrences), declarations after
    statements, `inline`, `__attribute__`, and anything from C99's library.

    **UNALIGNED ACCESS IS A CORRECTNESS BUG, not a slow path.** On ARMv5TE an
    unaligned word load does not fault — it returns the word ROTATED, so the
    engine computes confidently wrong numbers. On MIPS it faults. On a Pentium
    it costs three cycles. One of those three is silent, which is why this is a
    law and not a tuning note. Buffers that SIMD or `LDM` will read declare
    their alignment (`LZ_XW_ALIGN`); they do not inherit it from whatever the
    compiler happened to give a large array.

    **`build/mips/endian_gate.sh` is the only gate that can see a byte-order
    bug.** Every other target — x86 under gcc/MinGW/Watcom, x86-64, ARMv5TE — is
    little-endian, so a load path that assumes little-endian passes the entire
    rest of the suite, ARM bit-identity included.

    **WHERE THE WEIGHT PATH'S ALIGNMENT ACTUALLY COMES FROM**, surveyed
    2026-08-24 rather than assumed. `src/` holds 204 casts to a wider type; 111
    are `(float *)xcalloc(...)` and alignment-free by construction. The other 93
    reduce to a handful of operands — `w->q`, `t->q`, and the `w4`/`w2`/`pf_end`
    the row kernels derive from them — and the whole surface then rests on ONE
    invariant: every `q` plane comes from `malloc` (`model.c` 1232–1349, so the
    base is fundamentally aligned), and every offset added before the cast is a
    multiple of the target's size *because `gs` and `in_dim` are multiples of
    32*. That is enforced, not hoped: `model.c:1167` rejects a non-multiple at
    load, and `ops.c` 2373 / 2745 / 2754 / 2760 re-check it per format before
    any kernel runs.

    `-Wcast-align=strict` CANNOT audit this and must not be quoted as if it
    had. It reports 9 sites, all `(LZTensor *)((char *)L + offsetof)` in
    `model.c` — aligned by construction, false positives — and stays silent on
    every real one, because the tree writes `(const int16_t *)(const void *)w->q`
    and an intermediate `void *` silences the warning. An instrument that finds
    only the benign cases is the always-green probe this project keeps
    re-learning about.

Two more, not numbered but enforced everywhere:

- **A new feature turned off must be bit-identical to a build that never had
  it.** `spec_k = 0`, `look_width <= 1`, `--kv-rot off`, `--batch 1` all carry
  this contract explicitly.
- **One authority per fact.** The Makefile's source lists, `llama98.def`, the
  string tables, `lz_chat_norm_history` (history normalisation — the GUI
  re-implementing it is a mistake this project already made once),
  `common/session.c`'s render. Two copies drift; the second one is always the
  wrong one.

## Model-level traps (all measured, all easy to get silently wrong)

- `q_proj`'s output is `2 * n_heads * head_dim`, split **per head** as
  interleaved `[q(head_dim), gate(head_dim)]` — *not* `[all q][all gate]`.
- `lz_rmsnorm` scales by `(1 + w)`; `lz_rmsnorm_gated` (inside SSM) uses plain
  `w`. Both conventions coexist in the same model.
- `lz_l2norm` uses `sum(x^2)`, not `mean`, and eps is added to the sum.
- RoPE rotates only each head's leading `rotary_dim` dims, paired as
  `(i, i + rotary_dim/2)` — a half-rotation, not adjacent-element interleaving.
- Never read `LZTensor.q` directly. Quantized-format layout knowledge lives only
  in `ops.c`; go through `lz_t_row_f32` / `lz_t_f32`. A quantized row misread as
  plain ints still yields finite logits (measured cosine 0.34).
- MTP `fc` input concat order: **embedding in the low half, hidden in the high
  half**.
