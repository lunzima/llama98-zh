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
INSTR_RE = re.compile(r"^\s*[0-9a-f]+:\s+[0-9a-f]{8}\s+(\S+)\s*(.*)$")


def parse_funcs(text):
    """Split objdump -d output into (name, [(mnemonic, operands)]) per
    function. `operands` is the raw text after the mnemonic (may be empty)."""
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
        m = INSTR_RE.match(line)
        if m:
            cur[1].append((m.group(1), m.group(2).strip()))
    if cur:
        funcs.append(cur)
    return funcs


def ldrh_run_rule():
    """Four or more consecutive ldrh can be one ldm (4:1 load saving)."""
    def apply(name, instrs):
        runs = []
        run = 0
        for m, _ in instrs:
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
    def apply(name, instrs):
        sb = sum(1 for m, _ in instrs if m == "smlabb")
        st = sum(1 for m, _ in instrs if m in ("smlatt", "smlabt", "smlatb"))
        if sb > 0 and st == 0:
            return [("smlabb-alone %d" % sb, name)]
        return []
    return Rule(2, "smlabb-alone", apply)


# gcc's ARM cost table: every ARMv5TE core (arm9tdmi, arm9e, arm10e,
# arm926ej-s, arm1026ej-s, xscale, strongarm) tunes to generic_extra_costs
# (gcc/config/arm/aarch-cost-tables.h). The ARMv7-A baseline, cortex-a8,
# moved to gcc/config/arm/arm.cc (L1214) with unchanged values. The MULT
# SImode sub-table, generic vs cortex-a8, in COSTS_N_INSNS units:
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


# LD/ST sub-table, generic (ARMv5TE) vs cortex-a8 (arm.cc L1214):
#   ldr/ldrb/ldrh/ldrsb/ldrsh/str/strb/strh   2 vs 1   -> 2x
#   ldrd/strd                                   3 vs 1   -> 3x
# ldm/stm is MIXED (first reg 2x, subsequent 0.5x) so not flagged here;
# the ldrh-run rule already targets the 4+ ldrh → ldm rewrite.
LDST_2X = ("ldr", "ldrb", "ldrh", "ldrsb", "ldrsh",
           "str", "strb", "strh")
LDST_3X = ("ldrd", "strd")


def mul_family_rule():
    """Category 2 (cost-delta) read from gcc's own cost table, not
    memory: every ARMv5TE core tunes to generic_extra_costs and the
    ARMv7-A baseline is cortex-a8 (now in arm.cc L1214), so the multiply
    family is 2-3x pricier on ARMv5TE than on the ARMv7-A baseline.
    muls is 1 vs 1 (equal), so it is deliberately not flagged."""
    def apply(name, instrs):
        out = []
        for m, ratio in MUL_FAMILY:
            n = sum(1 for x, _ in instrs if x == m)
            if n:
                out.append(("mul-family %s %d (%s vs armv7-a)"
                            % (m, n, ratio), name))
        return out
    return Rule(2, "mul-family", apply)


def ldst_cost_rule():
    """Category 2 (cost-delta): every single ldr/str is 2x and every
    ldrd/strd is 3x on ARMv5TE vs Cortex-A8 (arm.cc L1214). This is a
    census flag — consecutive loads that could be grouped into ldm are
    the rewrite (see ldrh-run)."""
    def apply(name, instrs):
        out = []
        n2 = sum(1 for m, _ in instrs if m in LDST_2X)
        if n2:
            out.append(("ldst-single %d (2x vs armv7-a)" % n2, name))
        n3 = sum(1 for m, _ in instrs if m in LDST_3X)
        if n3:
            out.append(("ldst-double %d (3x vs armv7-a)" % n3, name))
        return out
    return Rule(2, "ldst-cost", apply)


# Register-specified shift in a data-processing operand. gcc's cost table
# (generic_extra_costs, which every ARMv5TE core tunes to) charges
# COSTS_N_INSNS(1) for a shift_reg operand and 0 for a constant shift:
# 'add r0,r1,r2,lsl r3' costs an extra insn vs 'add r0,r1,r2,lsl #3'.
# The spelling is lsl/lsr/asr/ror followed by a register, never '#' (rrx
# carries no register). Loads/stores ALSO carry a register shift, but as a
# bracketed addressing offset (ldr r0,[r1,r2,lsl r3]), not the ALU
# shift_reg cost - so strip [...] before searching.
SHIFT_REG_RE = re.compile(r"\b(?:lsl|lsr|asr|ror)\s+r\d+")
BRACKET_RE = re.compile(r"\[[^\]]*\]")


def alu_shift_reg_rule():
    def apply(name, instrs):
        n = 0
        for _, o in instrs:
            if SHIFT_REG_RE.search(BRACKET_RE.sub("", o)):
                n += 1
        if n:
            return [("alu-shift-reg %d (1 insn, const shift is 0)" % n, name)]
        return []
    return Rule(2, "alu-shift-reg", apply)


# FP load/store. generic_extra_costs (every ARMv5TE core) charges
# loadf/storef (single, vldr/vstr .32) = 2 and loadd/stored (double,
# vldr/vstr .64) = 3. The .32/.64 suffix is how the cross-objdump spells
# single vs double; a bare vldr/vstr (older objdump) is left uncounted.
FP_2X = ("vldr.32", "vstr.32")
FP_3X = ("vldr.64", "vstr.64")


def fp_ldst_rule():
    def apply(name, instrs):
        out = []
        n2 = sum(1 for m, _ in instrs if m in FP_2X)
        if n2:
            out.append(("fp-ldst-single %d (2 insns)" % n2, name))
        n3 = sum(1 for m, _ in instrs if m in FP_3X)
        if n3:
            out.append(("fp-ldst-double %d (3 insns)" % n3, name))
        return out
    return Rule(2, "fp-ldst", apply)


# FP arithmetic. generic_extra_costs charges SF div=7 / DF div=15 (the
# standouts); DF mult=5, mult_addsub=fma=7 are also non-zero. The integer
# target is softfloat so this fires only on VFP builds, but division is
# expensive enough to key regardless. The .f32/.f64 suffix is the
# cross-objdump spelling.
FP_DIV_7 = ("vdiv.f32",)   # SF div = 7 insns
FP_DIV_15 = ("vdiv.f64",)  # DF div = 15 insns


def fp_arith_rule():
    def apply(name, instrs):
        out = []
        n7 = sum(1 for m, _ in instrs if m in FP_DIV_7)
        if n7:
            out.append(("fp-div-single %d (7 insns)" % n7, name))
        n15 = sum(1 for m, _ in instrs if m in FP_DIV_15)
        if n15:
            out.append(("fp-div-double %d (15 insns)" % n15, name))
        return out
    return Rule(2, "fp-arith", apply)


RULES = [ldrh_run_rule(), smlabb_alone_rule(), mul_family_rule(),
         ldst_cost_rule(), alu_shift_reg_rule(), fp_ldst_rule(),
         fp_arith_rule()]


def lint(text):
    """Return [(finding, func)] across all rules and functions."""
    out = []
    for name, instrs in parse_funcs(text):
        for rule in RULES:
            out.extend(rule.apply(name, instrs))
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
