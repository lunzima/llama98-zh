#!/usr/bin/env python3
"""pentiumlint / cyrixlint - a pre-P6 / P4 x86 inefficiency linter.

Finds instruction combinations that are slow on specific x86 processors
(Pentium P5/P55C, Cyrix 6x86MX/MII, IDT WinChip C6/2, AMD K6/K6-2,
Pentium 4), in the spirit of gaul/armlint and CSharperMantle/loonglint.

The per-processor split is the whole point: inc/dec stall on P4's trace
cache (2 uops) but not on P5/P6/K6 (the partial-flags stall there is
negligible next to the 1-byte encoding). One flat "inc is bad" rule
would be wrong on half the family. The rule table below is meant to
grow against gcc's config/i386/x86-tune.def (the X86_TUNE_* flags),
which is the authority on which instruction is slow where.

A CENSUS, not a gate. Each rewrite a rule implies is only correct when
the flags it changes are dead; the reader proves that.
"""

import re
import sys
from collections import namedtuple

# A rule has a CATEGORY and a per-processor scope:
#   1 quirk       - gcc's X86_TUNE_* says this instruction is slow here
#   2 cost-delta  - the instruction cost table shows it much pricier here
#                   than on the Pentium II baseline (x86) / ARMv7-A (ARM)
#   3 behavioral  - correct in principle but costly here, e.g. avoid
#                   unpredictable branches on P4's long pipeline
Rule = namedtuple("Rule", ["category", "mnems", "procs", "reason"])

# The seed table, keyed to the engine's target family. grow this from
# gcc's x86-tune.def (category 1), the instruction cost table (category
# 2), and branch-predictability analysis (category 3), not from memory.
RULES = [
    Rule(1, ("inc", "dec"), "p4",
         "trace cache stores 2 uops per inc/dec (read-modify-write); "
         "add/sub wins. NOT p5/p6/k6: gcc X86_TUNE_USE_INCDEC keeps "
         "inc/dec there (stall negligible vs 1-byte encoding)"),
]

# wdis line: ADDR  HEXBYTES  MNEMONIC  OPERANDS, tab-separated.
MNEMONIC_RE = re.compile(r"^[0-9A-Fa-f]{4}\s+[0-9A-Fa-f ]{1,40}\t+(\w+)")


def mnemonics(text):
    """Yield the mnemonic of each wdis instruction line."""
    for line in text.splitlines():
        m = MNEMONIC_RE.match(line)
        if m:
            yield m.group(1)


def lint(text):
    """Return [(mnem, procs, reason, count)] for every fired rule."""
    counts = {}
    for m in mnemonics(text):
        counts[m] = counts.get(m, 0) + 1
    out = []
    for rule in RULES:
        total = sum(counts.get(m, 0) for m in rule.mnems)
        if total > 0:
            out.append(("+".join(rule.mnems), rule.procs, rule.reason, total))
    return out


def main(argv):
    if not argv:
        print("usage: pentiumlint.py <wdis-output>", file=sys.stderr)
        return 2
    for path in argv:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for mn, procs, reason, n in lint(text):
            print("%-10s %-12s %d  %s" % (mn, procs, n, reason))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
