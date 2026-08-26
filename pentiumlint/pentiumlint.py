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

# Processor priority for resolving CONFLICTING rules. p5/k6 are the
# primary targets; cyrix/p4 are secondary. A rewrite that is good on p4
# but bad on p5/k6 is NOT recommended - the primary targets win. The
# output below sorts findings by this order.
PROC_PRIORITY = ["p5", "p55c", "k6", "k6-2", "cyrix", "winchip", "p4"]

# The seed table, keyed to the engine's target family. grow this from
# gcc's x86-tune.def (category 1), the instruction cost table (category
# 2), and branch-predictability analysis (category 3), not from memory.
RULES = [
    Rule(1, ("inc", "dec"), "p4",
         "trace cache stores 2 uops per inc/dec (read-modify-write); "
         "add/sub wins. NOT p5/p6/k6: gcc X86_TUNE_USE_INCDEC keeps "
         "inc/dec there (stall negligible vs 1-byte encoding)"),
    Rule(2, ("movsx",), "p5,p55c,k6,k6-2",
         "sign-extension: 3 insns on P5 / 2 insns on K6 vs 1 on "
         "PentiumPro - a 3x/2x cost delta (gcc x86-tune-costs.h:"
         "415,846,632)."),
    Rule(2, ("movzx",), "p5,p55c,k6,k6-2",
         "zero-extension: 2 insns on P5 / 2 insns on K6 vs 1 on "
         "PentiumPro - a 2x cost delta (gcc x86-tune-costs.h:"
         "416,847,633). On P5 gcc prefers AND for zero-extend "
         "(X86_TUNE_ZERO_EXTEND_WITH_AND = m_486|m_PENT)."),
    Rule(2, ("imul", "mul"), "p5,p55c,p4",
         "11 insns on P5 / 15 insns on P4 vs 4 on PentiumPro "
         "(2.75x / 3.75x, gcc x86-tune-costs.h:404,2476). "
         "K6 runs imul in 3 insns (cheaper than PPro), so it is "
         "excluded. Cyrix/WinChip have no gcc cost table."),
    Rule(2, ("idiv",), "p4",
         "56 insns on P4 vs 17 on PentiumPro (3.29x, gcc "
         "x86-tune-costs.h:2482). P5 (25 insns, 1.47x) and K6 "
         "(18 insns, 1.06x) are both under 1.5x and excluded. "
         "A constant divisor should be a reciprocal multiply when "
         "exactness is proven."),
    Rule(2, ("movs", "stos", "lods", "scas", "cmps"), "p5,p55c",
         "microcoded, non-pairable on P5 (P5 optimization manual "
         "Table 27-4; gcc models via pentium_memcpy/memset "
         "stringop_algs in x86-tune-costs.h:359). Short fixed-count "
         "copies/scans should be explicit load/store loops."),
    Rule(3, ("jcc", "jmp"), "p4",
         "behavioral: P4's ~20-stage pipeline makes a branch mispredict "
         "cost ~20 cycles. Prefer branchless selects in hot loops where "
         "the branch is a coin flip."),
    # --- category-1 quirks from gcc x86-tune.def ---
    Rule(1, ("not",), "p5,p55c",
         "not pairable on P5 (X86_TUNE_NOT_UNPAIRABLE, x86-tune.def:726). "
         "Rewrite: xor reg,-1."),
    Rule(1, ("cdq", "cltd"), "p5,p55c,k6,k6-2",
         "P5/K6 avoid cdq/cltd (X86_TUNE_USE_CLTD excludes m_PENT and "
         "m_K6, x86-tune.def:355). Rewrite: mov edx,eax; sar edx,31."),
    Rule(1, ("stc", "clc", "cmc"), "p4",
         "slow on NetBurst (X86_TUNE_SLOW_STC, x86-tune.def:811). "
         "Rewrite: use sbb idiom."),
    Rule(1, ("sahf",), "p5,p55c",
         "P5 avoids sahf (X86_TUNE_USE_SAHF lists m_PPRO/m_P4_NOCONA/"
         "m_K6_GEODE but not m_PENT, x86-tune.def:348)."),
    # TODO(cat1, needs operand inspection): xor reg,reg (K6 USE_MOV0),
    # add mem,reg (P5 READ_MODIFY_WRITE), mov reg,-1 (P5 MOVE_M1_VIA_OR),
    # push [mem] (P5 PUSH_MEMORY), not mem (K6 NOT_VECTORMODE),
    # rol/ror reg,8 (P4 USE_XCHGB).
    Rule(2, ("shl", "shr", "sar"), "p4",
         "4 insns on P4 vs 1 on PentiumPro (4x, gcc x86-tune-costs.h:2474-"
         "2475). Both variable (reg,cl) and constant (reg,imm) shifts are "
         "expensive on P4's NetBurst pipeline. P5 variable_shift is also "
         "4x (x86-tune-costs.h:402) but needs operand inspection to "
         "distinguish from constant shifts."),
    Rule(2, ("lea",), "p4,k6",
         "3 insns on P4 / 2 insns on K6 vs 1 on PentiumPro (3x/2x, gcc "
         "x86-tune-costs.h:2473,832). P4 treats lea as a complex operation; "
         "break into add/mov where possible."),
]

# Two disassembly formats are supported:
#   Watcom wdis:     ADDR  HEXBYTES  MNEMONIC  tab-separated
#   gcc objdump -M intel:   ADDR:\tHEXBYTES   MNEMONIC  (Intel syntax)
_WATCOM_INST_RE = re.compile(r"^[0-9A-Fa-f]{4}\s+[0-9A-Fa-f ]{1,40}\t+(\w+)")
_GCC_INST_RE = re.compile(r"^\s+[0-9a-fA-F]+:\t[0-9a-fA-F ]+\s{2,}(\w+)")

# Function boundary regexes.
#   gcc objdump: "00000000 <_forward_chunk>:"
_GCC_FUNC_RE = re.compile(r'^[0-9a-fA-F]+ <([^>]+)>:$', re.MULTILINE)
#   Watcom wdis: "0000\t\t\tlz_row_pick_:" (NOT local labels like L$1:)
_WATCOM_FUNC_RE = re.compile(
    r'^[0-9A-Fa-f]{4}\t+([A-Za-z_][A-Za-z0-9_]*):$', re.MULTILINE)


def mnemonics(text):
    """Yield the mnemonic of each instruction line (Watcom or gcc format)."""
    for line in text.splitlines():
        m = _WATCOM_INST_RE.match(line) or _GCC_INST_RE.match(line)
        if m:
            yield m.group(1)


def _split_funcs(text):
    """Split text into (name, body) by function boundaries. Falls back to
    one unnamed block if no boundaries found."""
    boundaries = list(_GCC_FUNC_RE.finditer(text))
    if not boundaries:
        boundaries = list(_WATCOM_FUNC_RE.finditer(text))
    if not boundaries:
        yield "(whole-file)", text
        return
    pos = 0
    fname = None
    for m in boundaries:
        if fname is not None:
            yield fname, text[pos:m.start()]
        fname = m.group(1)
        pos = m.end()
    if fname is not None:
        yield fname, text[pos:]


def lint(text):
    """Return [(category, mnems, procs, reason, count, func_name)] per
    fired rule, sorted by processor priority (p5/k6 first)."""
    out = []
    for fname, body in _split_funcs(text):
        counts = {}
        for m in mnemonics(body):
            counts[m] = counts.get(m, 0) + 1
        for rule in RULES:
            total = sum(counts.get(m, 0) for m in rule.mnems)
            if total > 0:
                out.append((rule.category, "+".join(rule.mnems),
                            rule.procs, rule.reason, total, fname))
    out.sort(key=lambda r: _procs_rank(r[2]))
    return out


def _procs_rank(procs):
    """Rank a comma-separated processor set by PROC_PRIORITY; the first
    named processor determines the rank (p5/k6 outrank cyrix/p4)."""
    for p in PROC_PRIORITY:
        if p in procs.split(","):
            return PROC_PRIORITY.index(p)
    return len(PROC_PRIORITY)


def main(argv):
    if not argv:
        print("usage: pentiumlint.py <dis-output> ...", file=sys.stderr)
        return 2
    for path in argv:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        basename = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        for cat, mn, procs, reason, n, fname in lint(text):
            print("%-20s %-28s cat%d %-10s %-12s %d  %s"
                  % (basename, fname, cat, mn, procs, n, reason))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
