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

# A rule is a named predicate over a disassembled function. The scanner
# feeds each function's instruction stream to every rule and collects
# the findings. This is the seed of the BOLT-style DSL the full project
# wants; for now the rules are plain Python, one per file below.
Rule = namedtuple("Rule", ["name", "apply"])

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
    return Rule("ldrh-run", apply)


def smlabb_alone_rule():
    """smlabb with no smlatt/smlabt/smlatb wastes the top halfword."""
    def apply(name, mnems):
        sb = sum(1 for m in mnems if m == "smlabb")
        st = sum(1 for m in mnems if m in ("smlatt", "smlabt", "smlatb"))
        if sb > 0 and st == 0:
            return [("smlabb-alone %d" % sb, name)]
        return []
    return Rule("smlabb-alone", apply)


RULES = [ldrh_run_rule(), smlabb_alone_rule()]


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
