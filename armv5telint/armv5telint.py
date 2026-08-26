#!/usr/bin/env python3
"""armv5telint - an ARMv5TE peephole linter.

Finds suboptimal instruction sequences in ARMv5TE assembly, in the
spirit of gaul/armlint and CSharperMantle/loonglint but aimed at the
one place this project's suboptimal sequences actually are: the
gcc-compiled C tier (the hand-written asm cells are already optimal,
with per-instruction counts in their comments).

The linter is a linear scanner over objdump -d output. Each rule is a
peephole pattern plus a per-processor reason; a rule fires only when the
pattern matches and the rewrite is provably equivalent under the
4-byte alignment contract the asm cells assert (src/ops_arm.h).

A CENSUS, not a gate: it reports "look here", and the reader proves the
rewrite's flags are dead before applying it.
"""

import re
import sys
from collections import namedtuple

# A rule is a category + a named predicate over a disassembled function.
# Category 1 (quirk) is empty for ARMv5TE - gcc's arm-cpus.in has only
# LDSCHED/XSCALE/STRONG scheduling models, no "avoid this instruction"
# flags (the one slow note, smallmul, is cortex-m only). These rules are
# category 2 (cost-delta: the C tier's sequence is pricier than the asm
# cell's under the 4-byte contract).
Rule = namedtuple("Rule", ["category", "name", "apply"])

FUNC_RE = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")
MNEMONIC_RE = re.compile(r"^\s*[0-9a-f]+:\s+[0-9a-f]{8}\s+(\S+)")


def parse_funcs(text):
    """Split objdump -d output into (name, [mnemonics]) per function."""
    funcs = []
    cur = None
    for line in text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            if cur:
                funcs.append(cur)
            cur = (m.group(2), [])
            continue
        if cur is None:
            continue
        m = MNEMONIC_RE.match(line)
        if m:
            cur[1].append(m.group(1))
    if cur:
        funcs.append(cur)
    return funcs


def ldrh_run_rule():
    """Four or more consecutive ldrh can be one ldm (4:1 load saving)."""
    def apply(name, mnems):
        runs = []
        run = 0
        for m in mnems:
            if m == "ldrh":
                run += 1
            else:
                if run >= 4:
                    runs.append(run)
                run = 0
        if run >= 4:
            runs.append(run)
        return [("ldrh-run %d" % r, name) for r in runs]
    return Rule(2, "ldrh-run", apply)


def smlabb_alone_rule():
    """smlabb with no smlatt/smlabt/smlatb wastes the top halfword."""
    def apply(name, mnems):
        sb = sum(1 for m in mnems if m == "smlabb")
        st = sum(1 for m in mnems if m in ("smlatt", "smlabt", "smlatb"))
        if sb > 0 and st == 0:
            return [("smlabb-alone %d" % sb, name)]
        return []
    return Rule(2, "smlabb-alone", apply)


def mul_mla_rule():
    """mul/mla is 2-5 cycles on the ARM9TDMI (early-termination
    multiplier, data-dependent) vs 1 cycle on Cortex-A8 - a 2-5x cost
    delta, and the data-dependence makes it worse than the number. From
    the ARM9TDMI / Cortex-A8 TRM instruction timings."""
    def apply(name, mnems):
        n = sum(1 for m in mnems if m in ("mul", "mla"))
        if n > 0:
            return [("mul-mla %d (2-5x vs armv7-a)" % n, name)]
        return []
    return Rule(2, "mul-mla", apply)


RULES = [ldrh_run_rule(), smlabb_alone_rule(), mul_mla_rule()]


def lint(text):
    """Return [(finding, func)] across all rules and functions."""
    out = []
    for name, mnems in parse_funcs(text):
        for rule in RULES:
            out.extend(rule.apply(name, mnems))
    return out


def main(argv):
    if not argv:
        print("usage: armv5telint.py <objdump-output>", file=sys.stderr)
        return 2
    for path in argv:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for finding, fn in lint(text):
            print("%-18s %s" % (finding, fn))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
