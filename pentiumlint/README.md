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

| mnemonic(s) | processors | why |
|---|---|---|
| `inc`, `dec` | `p4` | trace cache stores 2 uops per inc/dec; add/sub wins. NOT p5/p6/k6 |

Processors in the engine's target family: `p5` `p55c` `cyrix` `winchip`
`k6` `k6-2` `p4`. A rule fires only on the processors it names.

**Soundness-first.** Each rewrite is only correct when the flags it
changes are dead; the reader proves that. This lints, it does not judge.

## Usage

```bash
# The shipping build is 32-bit Watcom (OMF); wdis reads it:
wdis build/watcom/cli/ops_quant.obj > /tmp/ops_quant.wdis

python3 pentiumlint.py /tmp/ops_quant.wdis
```

## Tests

```bash
python3 test_pentiumlint.py
```

## Adding a rule

Append a `Rule((mnems...), "procs", "reason")` to `RULES` (from gcc's
`x86-tune.def`, not memory), and a `test_<name>` that asserts it fires
on a planted fixture and is silent otherwise.
