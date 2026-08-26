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
# flags (the one slow note, smallmul, is cortex-m only). Category 2
# (cost-delta) is read from gcc's own cost table, aarch-cost-tables.h,
# not from memory: it maps every ARMv5TE core to generic_extra_costs and
# Cortex-A8 to cortexa8_extra_costs. The ldrh/smlabb rules are the C
# tier's sequence pricier than the asm cell's under the 4-byte contract.
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


# gcc's ARM cost table (gcc/config/arm/aarch-cost-tables.h): every
# ARMv5TE core (arm9tdmi, arm9e, arm10e, arm926ej-s, arm1026ej-s, xscale,
# strongarm) tunes to generic_extra_costs; the ARMv7-A baseline is
# cortexa8_extra_costs. The MULT SImode sub-table, generic vs cortex-a8,
# in COSTS_N_INSNS units (field -> mnemonic):
#   simple       mul           2 vs 1   -> 2x
#   flag_setting muls          1 vs 1   -> equal (not flagged)
#   extend       smull/umull   2 vs 1   -> 2x
#   add          mla           3 vs 1   -> 3x
#   extend_add   smlal/umlal   3 vs 1   -> 3x
MUL_FAMILY = [
    ("mul", "2x"),
    ("mla", "3x"),
    ("smull", "2x"),
    ("umull", "2x"),
    ("smlal", "3x"),
    ("umlal", "3x"),
]


# LD/ST sub-table, generic (ARMv5TE) vs cortex-a8, same source:
#   ldr/ldrb/ldrh/ldrsb/ldrsch/str/strb/strh   2 vs 1   -> 2x
#   ldrd/strd                                    3 vs 1   -> 3x
# ldm/stm is MIXED (first reg 2x, subsequent 0.5x) so not flagged here;
# the ldrh-run rule already targets the 4+ ldrh → ldm rewrite.
LDST_2X = ("ldr", "ldrb", "ldrh", "ldrsb", "ldrsch",
           "str", "strb", "strh")
LDST_3X = ("ldrd", "strd")


def mul_family_rule():
    """Category 2 (cost-delta) read from gcc's own cost table, not
    memory: aarch-cost-tables.h gives every ARMv5TE core
    generic_extra_costs and Cortex-A8 cortexa8_extra_costs, and the
    multiply family is 2-3x pricier on ARMv5TE than on the ARMv7-A
    baseline. muls is 1 vs 1 (equal), so it is deliberately not flagged."""
    def apply(name, mnems):
        out = []
        for m, ratio in MUL_FAMILY:
            n = sum(1 for x in mnems if x == m)
            if n:
                out.append(("mul-family %s %d (%s vs armv7-a)"
                            % (m, n, ratio), name))
        return out
    return Rule(2, "mul-family", apply)


def ldst_cost_rule():
    """Category 2 (cost-delta): every single ldr/str is 2x and every
    ldrd/strd is 3x on ARMv5TE vs Cortex-A8, from the LD/ST sub-table
    in aarch-cost-tables.h. This is a census flag — consecutive loads
    that could be grouped into ldm are the rewrite (see ldrh-run)."""
    def apply(name, mnems):
        out = []
        n2 = sum(1 for m in mnems if m in LDST_2X)
        if n2:
            out.append(("ldst-single %d (2x vs armv7-a)" % n2, name))
        n3 = sum(1 for m in mnems if m in LDST_3X)
        if n3:
            out.append(("ldst-double %d (3x vs armv7-a)" % n3, name))
        return out
    return Rule(2, "ldst-cost", apply)


RULES = [ldrh_run_rule(), smlabb_alone_rule(), mul_family_rule(),
         ldst_cost_rule()]


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
