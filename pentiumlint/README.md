# pentiumlint (a.k.a. cyrixlint)

A pre-P6 / P4 x86 inefficiency linter. Finds instruction combinations
that are slow on specific x86 processors, in the spirit of
[gaul/armlint](https://github.com/gaul/armlint) and
[CSharperMantle/loonglint](https://github.com/CSharperMantle/loonglint).

The per-processor split is the point: `inc`/`dec` stall on P4's trace
cache (2 uops) but not on P5/P6/K6. A flat "inc is bad" rule would be
wrong on half the family. The rule table grows against gcc's
`config/i386/x86-tune.def` (`X86_TUNE_*`), the authority on which
instruction is slow where.

## Rule table

Rules come in three categories:

1. **quirk** — gcc's `x86-tune.def` says this instruction is slow here.
2. **cost-delta** — the instruction cost table shows it much pricier here
   than on the Pentium II baseline (x86) / ARMv7-A (ARM).
3. **behavioral** — correct in principle but costly here, e.g. avoid
   unpredictable branches on P4's long pipeline.

| cat | mnemonic(s) | processors | why |
|---|---|---|---|
| 1 | `inc`, `dec` | `p4` | trace cache stores 2 uops per inc/dec; add/sub wins. NOT p5/p6/k6 |
| 1 | `not` | `p5,p55c` | not pairable on P5 (X86_TUNE_NOT_UNPAIRABLE); `xor reg,-1` |
| 1 | `cdq`, `cltd` | `p5,p55c,k6,k6-2` | USE_CLTD excludes m_PENT and m_K6; `mov edx,eax; sar edx,31` |
| 1 | `stc`, `clc`, `cmc` | `p4` | slow on NetBurst (SLOW_STC); use sbb idiom |
| 1 | `sahf` | `p5,p55c` | USE_SAHF does not list m_PENT |
| 2 | `movsx` | `p5,p55c,k6,k6-2` | 3 insns P5 / 2 K6 vs 1 PPro (3x/2x) |
| 2 | `movzx` | `p5,p55c,k6,k6-2` | 2 insns P5 / 2 K6 vs 1 PPro (2x); P5 prefers AND |
| 2 | `imul`, `mul` | `p5,p55c,p4` | 11 insns P5 / 15 P4 vs 4 PPro (2.75x/3.75x) |
| 2 | `idiv` | `p4` | 56 insns vs 17 PPro (3.29x). P5 (1.47x) / K6 (1.06x) excluded |
| 1 | `movs`, `stos` | `p5,p55c,k6,k6-2` | gcc avoids single rep movs/stos here (SINGLE_STRINGOP); use explicit loop |
| 2 | `shl`, `sal`, `shr`, `sar`, `rol`, `ror`, `rcl`, `rcr` | `p4` | 4 insns vs 1 PPro (4x), all shifts/rotates, variable and constant |
| 2 | `lea` | `p4,k6` | 3 insns P4 / 2 K6 vs 1 PPro (3x/2x) |
| 3 | `jcc`, `jmp` | `p4` | NetBurst ~20-stage pipeline (not gcc's cost table): mispredict ~20 cycles |

The next seven are **operand rules**: they need the operand text, not just
the mnemonic (the `OpRule` kind, as opposed to the mnemonic-count `Rule`).

| 2 | `xor reg,reg` | `k6` | `mov $0,reg` beats `xor reg,reg` on K6 (USE_MOV0) |
| 1 | `add/adc/sub/sbb/... mem,reg` | `p5` | read-modify-write memory dest avoided (READ_MODIFY) |
| 1 | `push [mem]` | `p5` | separate load+push beats `push mem` (PUSH_MEMORY) |
| 1 | `not mem` | `k6` | K6 vector-decodes `not mem` without modRM (NOT_VECTORMODE) |
| 1 | `rol/ror r16,8` | `p4` | byte swap: `xchgb %rh,%rl` (USE_XCHGB) |
| 1 | `mov reg,-1` | `p5` | `or reg,-1` beats `mov reg,-1` (MOVE_M1_VIA_OR) |
| 2 | `shl/shr/sar reg,cl` | `p5` | variable shift 4 insns vs 1 constant (shift_var/shift_const) |

Processors in the engine's target family: `p5` `p55c` `cyrix` `winchip`
`k6` `k6-2` `p4`. A rule fires only on the processors it names. `cyrix`
and `winchip` have no gcc cost table (mapped to i486, removed since
gcc-9), so no rule keys on them directly.

**Soundness-first.** Each rewrite is only correct when the flags it
changes are dead; the reader proves that. This lints, it does not judge.

## Usage

```bash
# Watcom OMF (shipping build); wdis reads it:
wdis build/watcom/cli/ops_quant.obj > /tmp/ops_quant.wdis

# Or gcc Intel-syntax objdump (either is accepted):
gcc -m32 -O2 -c src/ops_quant.c -o /tmp/ops_quant.o
objdump -d -M intel /tmp/ops_quant.o > /tmp/ops_quant.dis

python3 pentiumlint.py /tmp/ops_quant.wdis /tmp/ops_quant.dis
```

## Tests

```bash
python3 test_pentiumlint.py   # or: make test
```

The unit tests plant one mnemonic and assert the rule fires (positive
control); `test_integration_on_real_gcc_intel_fixture` runs against a
real `objdump -M intel` fixture to exercise the gcc parser + per-function
splitting that the synthetic fixtures cannot.

## Adding a rule

Append a `Rule(category, (mnems...), "procs", "reason")` to `RULES` (from
gcc's `x86-tune.def` / `x86-tune-costs.h`, not memory), and a
`test_<name>` that asserts it fires on a planted fixture and is silent
otherwise. A rule that needs the operands (not just the mnemonic) is an
`OpRule(category, name, mnems, procs, reason, pred)` in `OP_RULES`, where
`pred(operands) -> bool`; its test asserts both the fire and the
same-mnemonic-different-operands silence (the negative control that
proves the predicate sees the operands).
