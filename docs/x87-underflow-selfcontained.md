# x87 stack underflow under `gcc -m32 -mfpmath=387 -O2`: a self-contained report

This document is written to be handed to someone with no access to the
project it came from. Everything needed to reason about the defect is
here: the environment, the symptom, the hardware state at the fault, what
has been ruled out and by what evidence, and what has been tried and
failed. Nothing below depends on reading the source tree.

English rather than the project's usual language, because the audience is
compiler people.

---

## 1. Summary

A 32-bit x87 build of a small C inference engine produces `0xFFC00000` —
x87 "real indefinite" — for every element of its output, from the very
first token, on the first forward pass. The same C source is correct
when built by Open Watcom for the same 32-bit x87 target, and correct
when built by the same gcc at `-m64` (SSE2 scalar float); those two agree
with each other bit-for-bit.

The x87 status word at the fault has **SF (stack fault) set and C1
clear** — the hardware's encoding for a **stack underflow**, i.e. a read
from a register the tag word marks empty. Seven of the eight registers
are occupied at that moment, two of them holding loop-invariant float
*constants* that the compiler has hoisted there.

Reproduced on gcc 13.3.0 and 14.2.0. gcc 16.1 produces the same peak
register pressure in the same function by static count (not run — no
32-bit libraries available on that host). It is not a version regression.

## 2. Environment

```
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0     reproduces
gcc (Ubuntu 14.2.0-4ubuntu2~24.04.1) 14.2.0     reproduces
gcc (MinGW-W64 x86_64-ucrt-posix-seh) 16.1.0    same static peak; not run
Linux 6.6.87.2-microsoft-standard-WSL2, x86-64 host running a genuine
32-bit ELF binary (not emulated)
```

Flags: `-m32 -march=i686 -mmmx -mfpmath=387 -O2 -std=c99 -ffp-contract=off`

`-mfpmath=387` is forced rather than letting gcc pick SSE2 scalar float,
because the deployment target is Socket-7-class hardware: MMX present, no
SSE, x87 the only float unit. This is a shipping configuration on at
least one distribution, not a curiosity.

## 3. Symptom and bisection

All 32768 output values are `0xFFC00000`, bit-identical, on a single
token. Not scattered garbage — the hardware's canonical masked
invalid-operation sentinel, propagated onward by ordinary IEEE NaN
contagion.

| variation | result |
|---|---|
| `-O0`, `-O1`, `-Og` (same source) | correct |
| `-O2` | all-NaN |
| `-O2 -fno-strict-aliasing` | all-NaN (so: not an aliasing violation) |
| `-O2 -fno-tree-vectorize` | all-NaN |
| `-O2 -fno-schedule-insns`, `-fno-schedule-insns2` | all-NaN |
| `-O2 -fno-reorder-blocks` | all-NaN |
| `-O2 -fno-gcse` | all-NaN |
| Open Watcom, 32-bit x87 | correct |
| gcc `-m64` (SSE2 scalar float) | correct, bit-identical to Watcom |

## 4. The hardware state at the fault

Breakpoint on the leaf function where the bad value first becomes
visible, on the unmodified failing binary (`gdb`, no source
instrumentation — see §7 for why that distinction is load-bearing):

```
st0   0
st1   0
st2   0
st3   1.0000000000000000e-30      <- a loop-invariant constant from the source
st4   12583514.057...             <- the magic-round constant plus a value
st5   12582912                    <- 1.5 * 2^23, the magic-round constant
st6   0xffff0000000000000000      <- never written
st7   0xffffc000000000000000      <- real indefinite
fctrl 0x037f
fstat 0x0061                      <- IE, PE, SF set; C1 clear
ftag  0xe015                      <- seven registers occupied, one empty
```

**`SF` set with `C1` clear is underflow**, not overflow: the code read a
register the tag word marks empty, rather than pushing a ninth value onto
a full stack. This distinction was measured, not inferred; an earlier
reading of this bug assumed overflow and was wrong.

`st3` and `st5` are the two things that make this look like a
register-allocation decision rather than a stray push: they are
**compile-time float constants**, loop-invariant, parked in x87 registers
across a call.

Tag word walked forward through one forward pass, same method:

```
layer 0 entry                      0xffff   empty
inner loop, head 0, top            0xffff   empty
inner loop, head 0, group loop     0xa015   all eight occupied
leaf function entry                0xe015   seven occupied, st7 already indefinite
leaf function exit                 0xffd5   three occupied
head 1 entry                       0xffd5   and it stays there
```

## 5. The shape of the code

Reduced to what matters. A per-row loop calls, per group, a small
coefficient helper and then a fixed-point group routine; the group
routine calls a leaf that normalizes two floats into int16 multipliers.
The leaf is `static`, the helpers are `static`, and gcc inlines some of
them and not others.

```c
#define MIN_SCALE 1.0e-30f

static int leaf_norm(float c1, float c2, int16_t *mul4, float *sacc)
{
    float cm1 = (c1 < 0.0f) ? -c1 : c1;
    float cm2 = (c2 < 0.0f) ? -c2 : c2;
    float cmax = (cm1 > cm2) ? cm1 : cm2;
    float p, t1, t2;
    int e, m1, m2;

    if (!(cmax >= MIN_SCALE)) {              /* MIN_SCALE ends up in st3 */
        mul4[0] = mul4[1] = mul4[2] = mul4[3] = 0;
        *sacc = 0.0f;
        return 0;
    }
    e  = exponent_of(cmax);                  /* pure bit manipulation */
    p  = pow2f(14 - e);                      /* pure bit manipulation */
    t1 = c1 * p;
    t2 = c2 * p;
    *sacc = pow2f(e - 14);
    m1 = (t1 > -0.5f && t1 < 0.5f) ? 0 : magic_round(t1);
    m2 = (t2 > -0.5f && t2 < 0.5f) ? 0 : magic_round(t2);
    /* clamp to +-32767, store into mul4[0..3] */
    return 1;
}

static int magic_round(float qv)             /* 1.5*2^23 ends up in st5 */
{
    union { float f; uint32_t u; } t;
    t.f = qv + 12582912.0f;
    return (int)(t.u & 0x007FFFFFu) - 0x00400000;
}

/* caller, simplified */
for (kk = 0; kk < kd; kk++) {
    float dec = gvec ? gvec[kk] : gt;
    for (gg = 0; gg < ng; gg++) {
        float c1, c2;
        group_coef(dec, rs_in[gg], k[kk], s_dg[gg], &c1, &c2);  /* 2 muls */
        group_fixed(row_in + gg*32, c1, dq + gg*32, c2,
                    row_out + gg*32, rs_out + gg);              /* calls leaf_norm */
    }
}
```

`leaf_norm` has six-plus conditional branches and several exit paths;
some drain the stack explicitly with `fstp %st(0)` and some do not.

## 6. What is ruled out, and by what

**Not the arithmetic.** Forcing a guaranteed-empty stack immediately
before the coefficient computation (an `emms` inserted in a scratch copy)
makes that computation produce the correct answer every time, and the
whole run comes out numerically right. Given a clean stack the code is
correct; something about which physical register gets read is wrong.

**Not MMX residue.** This build does contain MMX kernels, and one of them
runs unconditionally. But an MMX write sets the corresponding x87 tag to
`00` (Valid) regardless of the bit pattern — the FPU's own value
classification, which produces `01` (Zero) / `10` (Special) / `11`
(Empty), is performed only by genuine x87 instructions. The observed tag
words carry Zero and Special and **no Valid at all**. Separately, the tag
word is bit-identical before and after the MMX kernel runs.

**Not a push/pop count imbalance.** A static verifier was written that
reads the emitted instructions directly — `objdump` text in, basic
blocks, a per-mnemonic depth delta, propagated over the CFG — and reports
any join whose predecessors disagree on depth and any `ret` reached at
the wrong ABI depth. It cannot be perturbed by its own presence, which
matters here (§7). It finds **every join balanced and every return at ABI
depth, everywhere in the binary**. Its own indirect-branch and tail-call
blind spots were then resolved by hand; none hides a discrepancy. The
same verifier reports nothing on the Watcom and x86-64 builds, and its
self-test confirms it does flag a deliberately unmatched `fld`.

**Not a calling-convention violation, at least not by the letter.** The
Intel386 psABI (1.0, 2015) Table 2.3 lists `%st0` and `%st1`–`%st7` as
scratch, "preserved across function calls: No". That constrains the
callee's freedom to clobber; it does not require an empty stack at entry,
and no such sentence appears in that document. The leaf is `static`
besides, so the compiler sees every call site and may use whatever
convention it likes for it.

What remains, and needs no ABI to be a defect: **there are eight physical
registers, and the generated code arranges for a read from a register
that was never written.**

## 7. Why this has been hard to reduce

**Observing it changes it.** Adding inline instrumentation — even a pure
`fnstenv` read that writes nothing — is enough to change gcc's register
allocation and stop the bug reproducing. Twelve samples across six rows
came back clean, tag word `0xffff` everywhere, correct output, purely
from the presence of three reads.

`gdb` breakpoints on the unmodified binary do not have this problem, and
every measurement in §4 was taken that way. But it rules out the usual
reduction technique: every attempt to shrink the test case changed its
shape enough to stop it reproducing, and no standalone minimal reproducer
exists as a result.

This is not a novel observation about this code — a comment in the very
function involved, written for an unrelated earlier bug in the same
place, says: *"compiling the tracing probe, whose only effect is to STORE
these intermediates, makes the divergence disappear. A bug that hides
when you look at it is this mechanism, every time."*

## 8. Source-level fixes attempted, all measured, all failed

| attempt | result |
|---|---|
| `__attribute__((noinline))` on the group routine | still all-NaN, tag word unchanged |
| `__attribute__((noinline))` on the row-loop function | still all-NaN, tag word unchanged |
| `__attribute__((always_inline))` on `leaf_norm`, removing the call boundary entirely | still all-NaN |
| `volatile` on `c1`/`c2`, forcing the coefficients through memory | still all-NaN |
| the `MIN_SCALE` comparison rewritten as an unsigned compare of sign-cleared bit patterns, so that constant never needs a register | still all-NaN |

The pattern is itself evidence: **displace one live value and the
allocator hoists another into the slot.** Each attempt removes a real
live value and the stack still fills.

## 9. The question

Is a compiler entitled to keep seven x87 values — two of them
loop-invariant constants — live across a call, on a target where the
stack is eight deep and the callee needs some of it?

If yes, then something in this source is doing something the compiler is
permitted to interpret this way, and identifying it is the ask.

If no, then the reg-stack pass is over-allocating a fixed hardware
resource, and the diagnosis wanted is which decision does it and why the
same pass gets the `-O0`/`-O1` and x86-64 cases right.

Either way the concrete unanswered question is narrow: **which
instruction reads the empty register, and which pass decided that
register would be live there?**

## 10. What would help

- A way to see the reg-stack pass's decisions that does not perturb them.
  `-fdump-rtl-all` does not change `.text` (verified by comparing the
  extracted section with and without it), and its `*.stack` dump covers
  the functions involved, but the dump prints both a per-block
  `Input stack:`/`Output stack:` listing and a later
  `Edge A->B: correcting stack to [...]` summary, and for one edge those
  two disagree textually. Whether the edge correction overrides the
  earlier per-block estimate (working as intended) or records a genuine
  internal disagreement could not be settled from the dump alone.
- An emulator or tracing approach that records x87 tag-word transitions
  instruction by instruction without recompiling.
- Confirmation or refutation from a fourth toolchain on the same target
  shape: 32-bit, x87-only, no SSE.
