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

# The full shift/rotate family. gcc's cost table charges one
# shift_var/shift_const for all of them (pentium4_cost = 4/4,
# pentium_cost = 4/1), so the P4 shift rule and the P5 var-shift rule
# both key on this single tuple.
_SHIFT_OPS = ("shl", "sal", "shr", "sar", "rol", "ror", "rcl", "rcr")

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
         "PentiumPro - a 3x/2x cost delta (pentium_cost.movsx=3, "
         "k6_cost.movsx=2)."),
    Rule(2, ("movzx",), "p5,p55c,k6,k6-2",
         "zero-extension: 2 insns on P5 / 2 insns on K6 vs 1 on "
         "PentiumPro - a 2x cost delta (pentium_cost.movzx=2, "
         "k6_cost.movzx=2). On P5 gcc prefers AND for zero-extend "
         "(X86_TUNE_ZERO_EXTEND_WITH_AND = m_486|m_PENT)."),
    Rule(2, ("imul", "mul"), "p5,p55c,p4",
         "11 insns on P5 / 15 insns on P4 vs 4 on PentiumPro "
         "(2.75x / 3.75x, pentium_cost.mul=11, pentium4_cost.mul=15). "
         "K6 runs imul in 3 insns (cheaper than PPro), so it is "
         "excluded. Cyrix/WinChip have no gcc cost table."),
    Rule(2, ("idiv",), "p4",
         "56 insns on P4 vs 17 on PentiumPro (3.29x, "
         "pentium4_cost.div=56). P5 (25 insns, 1.47x) and K6 "
         "(18 insns, 1.06x) are both under 1.5x and excluded. "
         "A constant divisor should be a reciprocal multiply when "
         "exactness is proven."),
    Rule(1, ("movs", "stos"), "p5,p55c,k6,k6-2",
         "gcc does not emit a single rep movs/rep stos on these chips "
         "(X86_TUNE_SINGLE_STRINGOP = m_386|m_P4_NOCONA, so p5/k6 and "
         "friends expand to an explicit loop). Short fixed-count copies "
         "should be explicit load/store loops. lods/scas/cmps have no "
         "gcc flag, so they are not flagged."),
    Rule(3, ("jcc", "jmp"), "p4",
         "behavioral (NetBurst pipeline depth, not P4's own cost table - "
         "pentium4_cost branch_mispredict_scale=2 equals PPro/K6/P5): P4's "
         "~20-stage pipeline makes a branch mispredict cost ~20 cycles. "
         "Prefer branchless selects in hot loops where the branch is a coin "
         "flip. gcc r17-1797 recently raised the GENERIC target's "
         "branch_mispredict_scale from COSTS_N_INSNS(2) to +3 (~12% on "
         "modern CPUs), confirming mispredict cost is real and modeled."),
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
    # (Operand-inspection rules — xor reg,reg, rmw-mem, mov reg,-1,
    # push/not mem, rol/ror reg,8, var-shift, double-by-add, partial-reg,
    # partial-mem-read, simode-fiop — live in OP_RULES below.)
    Rule(2, _SHIFT_OPS, "p4",
         "4 insns on P4 vs 1 on PentiumPro (4x, pentium4_cost.shift_var/"
         "shift_const=4). Both variable (reg,cl) and constant (reg,imm) "
         "counts are expensive on P4's NetBurst pipeline; gcc's 4 covers "
         "the whole shift/rotate family. P5 variable_shift is also 4x "
         "(pentium_cost.shift_var=4 vs shift_const=1); that is the "
         "separate var-shift rule in OP_RULES."),
    Rule(2, ("lea",), "p4,k6",
         "3 insns on P4 / 2 insns on K6 vs 1 on PentiumPro (3x/2x, "
         "pentium4_cost.lea=3, k6_cost.lea=2). P4 treats lea as a complex "
         "operation; break into add/mov where possible."),
]

# --- operand-inspection rules -------------------------------------------------
# The mnemonic rules above see only the mnemonic; these need the operands.
# Each is a predicate over the operand list, pre-filtered by a mnemonic set.
# Sources, as always, are gcc's x86-tune.def (category 1) and
# x86-tune-costs.h (category 2).

# x86 register-name sets for operand classification.
_REG8 = {"al", "cl", "dl", "bl", "ah", "ch", "dh", "bh"}
_REG16 = {"ax", "cx", "dx", "bx", "sp", "bp", "si", "di"}
_REG32 = {"eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"}
_ALL_REGS = _REG8 | _REG16 | _REG32

# Read-modify-write ALU/shift ops. gcc's X86_TUNE_READ_MODIFY excludes
# m_PENT, so P5 avoids their memory-destination form (load into a reg,
# operate, store back). cmp/test are not here: they do not write back.
_RMW_OPS = ("add", "adc", "sub", "sbb", "and", "or", "xor",
            "neg", "not", "inc", "dec",
            "shl", "sal", "shr", "sar", "rol", "ror", "rcl", "rcr")

# Variable-count shifts (count in CL) cost 4 insns on P5 vs 1 for a
# constant count; _SHIFT_OPS is defined above, next to RULES.


def _is_mem(op):
    return "[" in op


def _is_all_ones(op):
    """True if `op` is an all-ones immediate: -1, 0xffffffff, 0ffffffffh,
    0xffff, 0xff, ... in any case/prefix/suffix."""
    if op == "-1":
        return True
    s = op.lower()
    if s.startswith("0x"):
        s = s[2:]
    if s.endswith("h"):
        s = s[:-1]
    return bool(s) and all(c == "f" for c in s)


def _imm_is(op, value):
    """True if `op` is the immediate `value` in any spelling (1, 1h, 0x1,
    01h). Strips 0x/hex-h and leading zeros before comparing."""
    s = op.lower()
    if s.startswith("0x"):
        s = s[2:]
    if s.endswith("h"):
        s = s[:-1]
    s = s.lstrip("0") or "0"
    return s == str(value)


def _same_reg(op0, op1):
    return op0.lower() == op1.lower() and op0.lower() in _ALL_REGS


OpRule = namedtuple("OpRule", ["category", "name", "mnems", "procs",
                               "reason", "pred"])

OP_RULES = [
    OpRule(2, "xor-r,r", ("xor",), "k6",
           "K6 clears a register faster with 'mov $0,reg' than 'xor reg,reg' "
           "(X86_TUNE_USE_MOV0 = m_K6, x86-tune.def). Rewrite: mov reg,0.",
           lambda o: len(o) == 2 and _same_reg(o[0], o[1])),
    OpRule(1, "rmw-mem", _RMW_OPS, "p5",
           "read-modify-write with a memory destination (e.g. add mem,reg) "
           "is avoided on P5 (X86_TUNE_READ_MODIFY excludes m_PENT, "
           "x86-tune.def). Load into a register, operate, store back.",
           lambda o: bool(o) and _is_mem(o[0])),
    OpRule(1, "push-mem", ("push",), "p5",
           "P5 runs a separate load + push faster than 'push mem' "
           "(X86_TUNE_PUSH_MEMORY excludes m_PENT, x86-tune.def).",
           lambda o: bool(o) and _is_mem(o[0])),
    OpRule(1, "not-mem", ("not",), "k6",
           "on K6, NOT with a memory operand is vector-decoded without a "
           "modRM byte; XOR is long-decoded, which helps here "
           "(X86_TUNE_NOT_VECTORMODE = m_K6, x86-tune.def).",
           lambda o: bool(o) and _is_mem(o[0])),
    OpRule(1, "rol/ror-8", ("rol", "ror"), "p4",
           "a 16-bit rotate by 8 is a byte swap; on P4 use xchgb %rh,%rl "
           "instead (X86_TUNE_USE_XCHGB = m_PENT4, x86-tune.def).",
           lambda o: len(o) == 2 and o[0].lower() in _REG16 and _imm_is(o[1], 8)),
    OpRule(1, "mov-r,-1", ("mov",), "p5",
           "on P5, loading -1 via OR is faster than MOV "
           "(X86_TUNE_MOVE_M1_VIA_OR = m_PENT|m_LAKEMONT, x86-tune.def). "
           "Rewrite: or reg,-1.",
           lambda o: len(o) == 2 and o[0].lower() in _ALL_REGS
           and _is_all_ones(o[1])),
    OpRule(2, "var-shift", _SHIFT_OPS, "p5",
           "variable-count shift (count in CL) is 4 insns vs 1 for a "
           "constant count on P5 (x86-tune-costs.h pentium_cost "
           "shift_var/shift_const). K6 is 1 vs 1, so it is excluded.",
           lambda o: len(o) >= 2 and o[1].lower() == "cl"),
    OpRule(1, "double-by-add", ("sal", "shl"), "p5,p55c,k6,k6-2,p4",
           "gcc doubles an integer register with add reg,reg, not sal reg,1 "
           "(X86_TUNE_DOUBLE_WITH_ADD = ~m_386, x86-tune.def; ~m_386 also "
           "covers i486/cyrix/winchip, not keyed here). Rewrite: add reg,reg.",
           lambda o: len(o) == 2 and o[0].lower() in _ALL_REGS
           and _imm_is(o[1], 1)),
    OpRule(1, "partial-reg", ("mov",), "p4",
           "a store to an 8/16-bit register leaves the upper bits stale and "
           "creates a partial-register dependency on P4 "
           "(X86_TUNE_PARTIAL_REG_DEPENDENCY = m_P4_NOCONA|..., x86-tune.def). "
           "Rewrite: movzx/movzbl to the full register.",
           lambda o: bool(o) and o[0].lower() in (_REG8 | _REG16)),
    OpRule(3, "partial-mem-read", ("mov",), "p5,p55c,k6,k6-2,p4",
           "behavioral: reading part of a memory location can hit a "
           "store-to-load forwarding stall after a large write to the same "
           "address (X86_TUNE_PARTIAL_MEMORY_READ_STALL, x86-tune.def). "
           "Census hint - the stall needs a preceding large write; the "
           "reader proves it.",
           lambda o: len(o) == 2 and o[0].lower() in (_REG8 | _REG16)
           and _is_mem(o[1])),
    OpRule(1, "simode-fiop", ("fild", "fist", "fistp", "fisttp"),
           "p5,p55c",
           "P5 avoids x87 integer load/store with a 32-bit (dword) operand "
           "(X86_TUNE_USE_SIMODE_FIOP excludes m_PENT, x86-tune.def). "
           "x87 domain - rarely hit on the integer target.",
           lambda o: any("dword" in x.lower() for x in o)),
]

# Two disassembly formats are supported:
#   Watcom wdis:     ADDR  HEXBYTES  MNEMONIC  tab-separated
#   gcc objdump -M intel:   ADDR:\tHEXBYTES   MNEMONIC  (Intel syntax)
_WATCOM_INST_RE = re.compile(r"^[0-9A-Fa-f]{4}\s+[0-9A-Fa-f ]{1,40}\t+(\w+)\s*(.*)$")
_GCC_INST_RE = re.compile(r"^\s+[0-9a-fA-F]+:\t[0-9a-fA-F ]+\s{2,}(\w+)\s*(.*)$")

# Function boundary regexes.
#   gcc objdump: "00000000 <_forward_chunk>:"
_GCC_FUNC_RE = re.compile(r'^[0-9a-fA-F]+ <([^>]+)>:$', re.MULTILINE)
#   Watcom wdis: "0000\t\t\tlz_row_pick_:" (NOT local labels like L$1:)
_WATCOM_FUNC_RE = re.compile(
    r'^[0-9A-Fa-f]{4}\t+([A-Za-z_][A-Za-z0-9_]*):$', re.MULTILINE)


def _operands(opstr):
    """Split an operand string into a stripped list. Intel syntax never
    puts a comma inside a memory operand ([base+idx*scale+disp]), so a
    top-level comma always separates operands."""
    return [o.strip() for o in opstr.split(",") if o.strip()]


def _instructions(text):
    """Yield (mnemonic, [operand...]) per instruction line (either format)."""
    for line in text.splitlines():
        m = _WATCOM_INST_RE.match(line) or _GCC_INST_RE.match(line)
        if m:
            yield m.group(1), _operands(m.group(2))


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
        instrs = list(_instructions(body))
        counts = {}
        for m, _ in instrs:
            counts[m] = counts.get(m, 0) + 1
        for rule in RULES:
            total = sum(counts.get(m, 0) for m in rule.mnems)
            if total > 0:
                out.append((rule.category, "+".join(rule.mnems),
                            rule.procs, rule.reason, total, fname))
        for rule in OP_RULES:
            n = sum(1 for m, o in instrs if m in rule.mnems and rule.pred(o))
            if n > 0:
                out.append((rule.category, rule.name, rule.procs,
                            rule.reason, n, fname))
    out.sort(key=lambda r: _procs_rank(r[2]))
    return out


def _procs_rank(procs):
    """Rank a comma-separated processor set by PROC_PRIORITY; the first
    named processor determines the rank (p5/k6 outrank cyrix/p4)."""
    for p in PROC_PRIORITY:
        if p in procs.split(","):
            return PROC_PRIORITY.index(p)
    return len(PROC_PRIORITY)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
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
