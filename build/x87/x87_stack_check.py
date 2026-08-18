#!/usr/bin/env python3
"""x87 register-stack depth verifier.

WHY THIS EXISTS: docs/x87-gcc-reg-stack-leak.md. gcc -m32 -mfpmath=387
-O2 was found to leave the x87 FP stack non-empty on return from some
function, which later faults an unrelated push elsewhere in the same
process and delivers the hardware's masked "real indefinite" QNaN
(0xFFC00000). Every attempt to catch it with a runtime probe (inline
fnstenv reads, gdb breakpoints and single-stepping) either changed gcc's
own register allocation enough to make the bug stop reproducing, or
could not reliably correlate a loop iteration with a tag-word reading in
optimized code. This tool reads the SHIPPED BYTES instead: no
recompilation, no execution, no probe of any kind to perturb.

WHAT IT CHECKS. gcc's own reg-stack pass has exactly one invariant to
keep: every control-flow edge into a join point must arrive at the same
x87 stack depth, and every `ret` must be reached at the depth the x87
ABI promises for that function's return type (0 for void/int/pointer,
1 for float/double - the value sits in %st(0) at the point of return).
A violation of either is visible directly in the disassembly: it is not
a claim about values, only about how many slots are live.

USAGE:
    python3 x87_stack_check.py <binary-or-.dis-file> [--function NAME]
    python3 x87_stack_check.py --self-test          # see below

--self-test hand-assembles a tiny function with one unmatched fld and
confirms this checker flags it, then a balanced twin and confirms it
does not - the two halves the project asked for: prove the checker can
see a real violation, and prove it does not cry wolf on ordinary code.

LIMITS, STATED RATHER THAN DISCOVERED LATER:
  - Indirect jumps/calls (jump tables, function pointers) end a block
    with an UNKNOWN target; this checker cannot follow them and reports
    them separately rather than silently assuming a depth. A function
    that relies entirely on a jump table for its control flow will read
    as "not fully verified", not as clean.
  - The call-balance table (FLOAT_RETURNING below) is derived from this
    project's own source plus known libm entry points. A call to an
    UNLISTED function that returns float/double through %st(0) would be
    mis-scored as depth-neutral, which could show a phantom join
    mismatch downstream of that one call. New float/double-returning
    functions must be added to the table or explicitly passed via
    --extra-float-fn.
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict

# ---- instruction depth-delta table ------------------------------------
# team-lead's original table, extended with every mnemonic actually
# observed in this project's 32-bit x87 build (see the shell census this
# script's history was developed against) plus the remaining standard
# x87 opcodes for completeness.

PUSH1 = {
    'fld', 'fld1', 'fldz', 'fldpi', 'fldl2e', 'fldl2t', 'fldlg2', 'fldln2',
    'flds', 'fldl', 'fldt',
    'fild', 'filds', 'fildl', 'fildll', 'fildq',
}
# fdecstp moves TOP as if a slot were pushed without loading one - rare
# in compiler output, included for completeness.
DECSTP = {'fdecstp'}
INCSTP = {'fincstp'}

POP1 = {
    'fstp', 'fstps', 'fstpl', 'fstpt',
    'fistp', 'fistpl', 'fistpll', 'fistpq',
    'faddp', 'fsubp', 'fsubrp', 'fmulp', 'fdivp', 'fdivrp',
    'fcomp', 'fcomps', 'fcompl', 'fucomp',
    'fcomip', 'fucomip',
}
POP2 = {'fcompp', 'fucompp'}

NEUTRAL = {
    'fst', 'fsts', 'fstl',
    'fist', 'fistl',
    'fxch', 'fabs', 'fchs', 'frndint', 'fnstsw', 'fnstcw', 'fldcw',
    'ftst', 'fxam',
    'fcom', 'fcoms', 'fcoml', 'fucom',
    'fadd', 'fadds', 'faddl',
    'fsub', 'fsubs', 'fsubl',
    'fsubr', 'fsubrs', 'fsubrl',
    'fmul', 'fmuls', 'fmull',
    'fdiv', 'fdivs', 'fdivl',
    'fdivr', 'fdivrs', 'fdivrl',
    'fcomi', 'fucomi',
    'fcmovb', 'fcmove', 'fcmovbe', 'fcmovu',
    'fcmovnb', 'fcmovne', 'fcmovnbe', 'fcmovnu',
    'ffree', 'ffreep',
    'fsqrt', 'fscale', 'fprem', 'fprem1', 'f2xm1', 'fyl2x', 'fyl2xp1',
    'fpatan', 'fnop', 'fwait',
}
# NOT here: bare 'nop' (0x90) and 'wait' - both are general x86
# instructions, not x87-family opcodes, and including them in ALL_X87
# made has_x87() misfire on almost any function (nop is used for
# alignment padding everywhere), triggering full CFG analysis - with a
# block-builder not hardened for large non-numeric control flow like
# switch/jump-table-heavy code - on functions with zero real x87
# content. That produced hundreds of phantom join-mismatch findings in
# `main` on the x86-64 control build, which is exactly the check this
# tool's own x86-64/Watcom control run exists to catch.
# emms/femms/finit/fninit reset the tag word to empty absolutely, not a
# delta - modelled as "depth := 0" wherever they appear.
RESET_ABS = {'emms', 'femms', 'finit', 'fninit'}

ALL_X87 = (PUSH1 | DECSTP | INCSTP | POP1 | POP2 | NEUTRAL | RESET_ABS)

# ---- known float/double-returning callees ------------------------------
# Built from grepping every src/*.c and common/*.c for
# "^float |^double |^static float |^static double " and hand-verifying
# each match's actual signature - several greps came back with `float *`
# (pointer-returning: lz_forward*, lz_t_f32, gdn_shadow) or a bare
# variable declaration (lz_prof_us, lz_moe_tau, g_xg) that pattern-match
# the same grep but do NOT return a value through %st(0) and must NOT be
# in this set. Plus libm entry points this codebase's x87 float path
# calls through lz_softplus. Extend via --extra-float-fn, not by editing
# this set silently - an unlisted callee is a correctness gap in THIS
# tool, not in the code being checked, and should be visible as one.
#
# GCC clone suffixes (.part.N, .constprop.N, .isra.N, .cold[.N]) are
# stripped by strip_clone() before matching against this table - do NOT
# add suffixed variants here, add the base name once.
FLOAT_RETURNING = {
    'lz_f32_class_get', 'lz_f32_count_get',       # double
    'lz_json_get_num',                            # double
    'epi_fixed', 'epi_q8', 'epi_q41',
    'pow2f', 'norm_ss_fixed', 'lz_i32f',
    'lz_sigmoid', 'lz_silu', 'lz_softplus', 'lz_rsqrt', 'lz_exp',
    'gdn_pass1_fixed_head',
    'lz_time_ms',
    'lz_look_lse', 'lz_random_f32', 'lz_sample_eff_temp',
    'lz_session_last_ms',
    # NOT lz_mtp_draft_step: forward.h:714 is `float *lz_mtp_draft_step(...)`
    # - a pointer return (EAX, not %st(0)). An early version of this table
    # included it from an unchecked grep match; the false BAD-RET-DEPTH
    # findings it produced are why every entry here now gets its
    # signature verified against the header, not just the grep hit.
    # libm, called from lz_softplus's slow path (x <= 20.0f):
    'exp', 'log1p', 'log', 'pow', 'sqrt', 'expf', 'log1pf', 'logf',
    'log10', 'log2', 'strtod', 'nanf', 'nanl', 'ldexp',
    # Watcom's own math387r CRT internals - not in any header this
    # project ships, named from the disassembly's own strong convention
    # (every one of these showed as "stack depth 1/2, expected 0" before
    # being added here, i.e. this is a HEURISTIC classification from
    # naming + observed behaviour, not a verified prototype - flagged
    # in case a future Watcom runtime renames or restructures them).
    '__fdiv_chk', '__fdiv_m32', '__fdiv_m64', '__fdiv_m32r', '__fdiv_m64r',
    '__pow87_err', '__math87_err', '__math2err',
    # NOT ___f_infinity: its finding was depth -1 (an underflow), not
    # +1 like every genuine misclassified-float-function case here - the
    # opposite signature, so adding it to "fix" the report would be
    # papering over something not yet understood rather than correcting
    # a table gap. Left as an open finding; likely a data constant
    # sitting inside the code segment's address range, not a real
    # function, given the map only distinguishes symbols by address
    # range, not by whether they are code.
}

# .part.0, .constprop.3, .isra.2, .cold, .cold.1, .lto_priv.0 - gcc
# clone/split suffixes that preserve the ORIGINAL function's signature
# (same return type, same ABI contract) but do not appear in source.
# Stripping these is required for both (a) a function checking its own
# ret-depth against its true return type, and (b) a call site scoring
# its callee correctly - missing this made an early run of this tool
# misclassify every call to lz_exp.part.0 as void-returning, which
# cascaded into phantom join-mismatch reports throughout lz_sigmoid.
# That failure mode is exactly why the self-test's "balanced code stays
# clean" half exists; the whole-binary run is what actually caught it.
CLONE_SUFFIX_RE = re.compile(
    r'\.(part|constprop|isra|cold|lto_priv)(\.\d+)?$')


def strip_clone(name):
    while True:
        m = CLONE_SUFFIX_RE.search(name)
        if not m:
            return name
        name = name[:m.start()]


def in_float_returning(name, extra):
    """Membership test tolerant of Watcom's own name-mangling: cdecl
    exports get exactly one trailing underscore appended (lz_sigmoid_,
    strtod_, log1p_ - none of which are in source or gcc's symbol
    table), so a bare set-membership check against FLOAT_RETURNING
    passed on every such symbol as false, which is what turned an
    entire wave of genuinely-correct float-returning functions into
    phantom BAD-RET-DEPTH findings on the first Watcom run. Try the
    name as given, then with one trailing underscore stripped - do not
    strip more than one, or a real double-underscore-prefixed CRT
    internal (__fdiv_m32) would silently match something unintended
    on the FRONT instead (this function only ever strips the back)."""
    base = strip_clone(name.split('@')[0])
    if base in FLOAT_RETURNING or base in extra:
        return True
    if base.endswith('_'):
        base2 = base[:-1]
        if base2 in FLOAT_RETURNING or base2 in extra:
            return True
    return False


JCC = re.compile(
    r'^j(mp|e|ne|z|nz|a|ae|nb|b|be|na|nbe|g|ge|nl|l|le|ng|nle|o|no|s|ns'
    r'|p|np|pe|po|c|nc|cxz|ecxz|rcxz)$')
LOOP = re.compile(r'^loop(e|z|ne|nz)?$')

LINE_RE = re.compile(
    r'^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2}\s)+\s*'
    r'([A-Za-z][A-Za-z0-9]*)\s*(.*)$')
FUNC_HDR_RE = re.compile(r'^([0-9a-fA-F]+)\s+<([^>]+)>:$')
# "call   80abcde <foo>" or "call   80abcde <foo+0x10>" or "jmp 80abcd <foo>"
TARGET_RE = re.compile(r'<([^+>]+)(?:\+0x[0-9a-fA-F]+)?>')


class Insn:
    __slots__ = ('addr', 'mnem', 'ops', 'raw')

    def __init__(self, addr, mnem, ops, raw):
        self.addr = addr
        self.mnem = mnem
        self.ops = ops
        self.raw = raw

    def target_name(self):
        m = TARGET_RE.search(self.ops)
        return m.group(1) if m else None

    def target_addr(self):
        # ELF/gcc objdump prints bare hex ("8f70"); PE/Watcom objdump
        # prints 0x-prefixed hex ("0x4012ff"). Both must resolve, or
        # every branch on a Watcom binary looks indirect.
        m = re.match(r'^\s*(?:0x)?([0-9a-fA-F]+)\b', self.ops)
        return int(m.group(1), 16) if m else None


def run_objdump(path):
    out = subprocess.run(['objdump', '-d', path], capture_output=True,
                          text=True, check=True)
    return out.stdout


def parse_functions(text):
    """-> dict[name] = list[Insn], in address order."""
    funcs = {}
    cur = None
    cur_name = None
    for line in text.splitlines():
        m = FUNC_HDR_RE.match(line)
        if m:
            cur_name = m.group(2)
            cur = []
            funcs[cur_name] = cur
            continue
        if cur is None:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        mnem = m.group(2)
        ops = m.group(3).strip()
        cur.append(Insn(addr, mnem, ops, line))
    return funcs


def depth_delta(insn, extra_float_fns, addr_to_name=None, is64=False):
    """-> (delta, is_reset, is_unknown_call, is_indirect_branch).

    addr_to_name: optional dict, function-entry address -> name, used
    when the disassembly text has no `<name>` annotation on the call
    operand at all - true of PE/Watcom output, which objdump prints
    with no symbol table for a stripped binary regardless of what
    synthetic function-boundary labels were inserted around it (see
    annotate_watcom_map.py's caveat). Without this fallback every call
    in a Watcom binary scores as an unresolved indirect call, which is
    a coverage gap in THIS TOOL, not evidence about the binary.

    is64: on x86-64, float/double returns go through %xmm0, not the x87
    stack, REGARDLESS of what FLOAT_RETURNING says - a call to a
    genuinely float-returning function on x86-64 pushes NOTHING onto
    the x87 stack. Without this, `ldexp` (real x87 use internally, but
    an SSE2 return on this ABI) read as a phantom BAD-RET-DEPTH finding
    on the x86-64 control build - the same class of ABI assumption
    error as the Watcom trailing-underscore one, just on the other
    axis (WHICH register class carries the return, not what the
    symbol is named). """
    mnem = insn.mnem
    if mnem in PUSH1 or mnem in DECSTP:
        return 1, False, False, False
    if mnem in POP1 or mnem in INCSTP:
        return -1, False, False, False
    if mnem in POP2:
        return -2, False, False, False
    if mnem in RESET_ABS:
        return 0, True, False, False
    if mnem in NEUTRAL:
        return 0, False, False, False
    if mnem == 'call':
        if is64:
            return 0, False, False, False  # x86-64: never %st(0)
        name = insn.target_name()
        if name is None and addr_to_name:
            t = insn.target_addr()
            if t is not None:
                name = addr_to_name.get(t)
        if name is None:
            return 0, False, True, False   # indirect call, unknown ABI
        if in_float_returning(name, extra_float_fns):
            return 1, False, False, False
        return 0, False, False, False
    # Not an x87 instruction (ordinary integer/SSE/MMX/control-flow) -
    # zero effect on the x87 stack.
    return 0, False, False, False


def is_block_ender(insn):
    if insn.mnem == 'ret' or insn.mnem.startswith('ret'):
        return True, 'ret'
    if insn.mnem == 'jmp':
        return True, 'jmp'
    if JCC.match(insn.mnem) or LOOP.match(insn.mnem):
        return True, 'jcc'
    if insn.mnem in ('hlt', 'ud2'):
        return True, 'hlt'
    return False, None


def build_blocks(insns):
    """-> (blocks: dict[addr]->list[Insn], order: list[addr])."""
    addr_of = {i.addr: idx for idx, i in enumerate(insns)}
    leaders = {insns[0].addr} if insns else set()
    for i in insns:
        ender, kind = is_block_ender(i)
        if not ender:
            continue
        if kind in ('jmp', 'jcc'):
            t = i.target_addr()
            if t is not None and t in addr_of:
                leaders.add(t)
        # instruction after any block-ender is a new leader if it
        # exists in this function's own instruction stream.
        idx = addr_of[i.addr]
        if idx + 1 < len(insns):
            leaders.add(insns[idx + 1].addr)
    leaders_sorted = sorted(leaders)
    blocks = {}
    order = []
    for k, start in enumerate(leaders_sorted):
        end = leaders_sorted[k + 1] if k + 1 < len(leaders_sorted) else None
        chunk = [i for i in insns
                 if i.addr >= start and (end is None or i.addr < end)]
        if chunk:
            blocks[start] = chunk
            order.append(start)
    return blocks, order


def successors(block_insns, blocks):
    """-> (list of successor addrs, indirect_flag)."""
    last = block_insns[-1]
    ender, kind = is_block_ender(last)
    if not ender:
        # fell off the end of the function's listed instructions with no
        # explicit control transfer - nothing to chain to.
        return [], False
    if kind == 'ret' or kind == 'hlt':
        return [], False
    if kind == 'jmp':
        t = last.target_addr()
        if t is not None and t in blocks:
            return [t], False
        return [], True   # indirect / out-of-function jump
    if kind == 'jcc':
        succ = []
        t = last.target_addr()
        if t is not None and t in blocks:
            succ.append(t)
        elif t is None:
            return [], True
        # fall-through
        idx_addrs = sorted(blocks.keys())
        pos = idx_addrs.index(block_insns[0].addr)
        if pos + 1 < len(idx_addrs):
            succ.append(idx_addrs[pos + 1])
        return succ, False
    return [], False


def float_return_expected(func_name, extra, is64=False):
    if is64:
        return False   # returns via %xmm0, not %st(0), on this ABI
    return in_float_returning(func_name, extra)


def call_target_name(insn, addr_to_name=None):
    """The callee's symbol for a direct call, else None."""
    if insn.mnem != 'call':
        return None
    name = insn.target_name()
    if name is None and addr_to_name:
        t = insn.target_addr()
        if t is not None:
            name = addr_to_name.get(t)
    return name


def function_peak(insns, extra_float_fns, addr_to_name=None, is64=False):
    """Highest depth this function reaches, walked straight through.

    Straight-line rather than CFG-aware, so it over-counts a branchy
    function by summing alternative paths. That direction is the safe
    one for the caller check below: it can report a call site that is
    actually fine, never miss one that is not.
    """
    depth = 0
    peak = 0
    for insn in insns:
        delta, is_reset, _unk, _ = depth_delta(
            insn, extra_float_fns, addr_to_name, is64)
        if is_reset:
            depth = 0
            continue
        depth += delta
        if depth > peak:
            peak = depth
    return peak


def check_function(name, insns, extra_float_fns, addr_to_name=None, is64=False,
                   peaks=None):
    """-> list of finding dicts."""
    findings = []
    blocks, order = build_blocks(insns)
    if not order:
        return findings

    entry = order[0]
    incoming = {entry: 0}
    visited_from = defaultdict(dict)   # block -> {pred: depth_in}
    worklist = [entry]
    seen_edges = set()
    unresolved = []

    while worklist:
        addr = worklist.pop()
        depth_in = incoming[addr]
        block = blocks[addr]
        depth = depth_in
        for insn in block:
            delta, is_reset, unk_call, _ = depth_delta(
                insn, extra_float_fns, addr_to_name, is64)
            if is_reset:
                depth = 0
                continue
            if unk_call:
                unresolved.append((insn.addr, insn.raw.strip()))
            depth += delta
            # The hardware has eight slots. A push past the eighth raises
            # a masked stack overflow and delivers real indefinite
            # (0xFFC00000) instead of a value - which is a NaN nothing
            # downstream distinguishes from a computed one. Balance alone
            # does not catch this: the pushes and pops still pair, the
            # peak simply does not fit. This is a separate finding kind
            # from a join mismatch and must not be folded into one.
            if depth > 8:
                findings.append({
                    'kind': 'depth-overflow', 'func': name,
                    'addr': insn.addr, 'depth': depth,
                    'text': '%s: depth %d exceeds the 8 physical '
                            'registers' % (insn.raw.strip(), depth)})
            # The callee's own peak rides on top of whatever the caller
            # is still holding. Both can be individually balanced and
            # individually under eight and still overflow together - the
            # x87 stack is one fixed eight-slot resource shared across
            # the call, and nothing in the ABI reserves any of it. A
            # per-function check that assumes entry depth zero cannot
            # see this, which is why it was missed here.
            if peaks is not None:
                callee = call_target_name(insn, addr_to_name)
                if callee and callee in peaks and depth + peaks[callee] > 8:
                    findings.append({
                        'kind': 'call-overflow', 'func': name,
                        'addr': insn.addr, 'callee': callee,
                        'depth': depth, 'peak': peaks[callee],
                        'text': '%s: %d live across the call, %s needs up '
                                'to %d more' % (insn.raw.strip(), depth,
                                                callee, peaks[callee])})
        succs, indirect = successors(block, blocks)
        if indirect:
            unresolved.append((block[-1].addr, block[-1].raw.strip()))
        ender, kind = is_block_ender(block[-1])
        if kind == 'ret':
            expect = 1 if float_return_expected(name, extra_float_fns, is64) else 0
            if depth != expect and depth != expect + 0:
                # allow depth == expect exactly; anything else is a find
                if depth != expect:
                    findings.append({
                        'kind': 'ret-depth',
                        'func': name,
                        'addr': block[-1].addr,
                        'got': depth,
                        'expect': expect,
                        'insn': block[-1].raw.strip(),
                    })
        for s in succs:
            edge = (addr, s, depth)
            if s in incoming:
                if incoming[s] != depth:
                    findings.append({
                        'kind': 'join-mismatch',
                        'func': name,
                        'addr': s,
                        'from_block': addr,
                        'existing_depth': incoming[s],
                        'new_depth': depth,
                    })
                    continue
            else:
                incoming[s] = depth
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            worklist.append(s)

    if unresolved:
        findings.append({
            'kind': 'unresolved',
            'func': name,
            'sites': unresolved,
        })
    return findings


def format_findings(findings):
    lines = []
    for f in findings:
        if f['kind'] == 'join-mismatch':
            lines.append(
                f"JOIN MISMATCH  {f['func']}  block 0x{f['addr']:x} "
                f"reached at depth {f['existing_depth']} earlier, "
                f"depth {f['new_depth']} from 0x{f['from_block']:x}")
        elif f['kind'] == 'ret-depth':
            lines.append(
                f"BAD RET DEPTH  {f['func']}  ret at 0x{f['addr']:x}: "
                f"stack depth {f['got']}, expected {f['expect']}  "
                f"[{f['insn']}]")
        elif f['kind'] == 'depth-overflow':
            lines.append(
                f"DEPTH OVERFLOW {f['func']}  0x{f['addr']:x}: "
                f"{f['text']}")
        elif f['kind'] == 'call-overflow':
            lines.append(
                f"CALL OVERFLOW  {f['func']}  0x{f['addr']:x}: "
                f"{f['text']}")
        elif f['kind'] == 'unresolved':
            for addr, raw in f['sites']:
                lines.append(
                    f"UNRESOLVED     {f['func']}  0x{addr:x}: {raw}  "
                    f"(indirect branch/call - not verified)")
    return lines


def has_x87(insns):
    return any(i.mnem in ALL_X87 for i in insns)


def run_self_test():
    """Hand-assembled proof the checker can see a real violation, and
    does not flag balanced code. Both twins compiled from raw bytes
    written as an objdump-shaped text block, so this needs neither gcc
    nor an assembler - the checker's input format IS objdump text."""
    broken = """
0000abcd <broken_leak>:
    abcd:\t55                   \tpush   %ebp
    abce:\td9 05 00 00 00 00    \tflds   0x0
    abd4:\td8 05 00 00 00 04    \tfadds  0x4
    abda:\td9 1d 08 00 00 00    \tfstps  0x8
    abe0:\td9 05 00 00 00 0c    \tflds   0xc
    abe6:\t5d                   \tpop    %ebp
    abe7:\tc3                   \tret
"""
    balanced = """
0000dead <balanced_twin>:
    dead:\t55                   \tpush   %ebp
    deae:\td9 05 00 00 00 00    \tflds   0x0
    deb4:\td8 05 00 00 00 04    \tfadds  0x4
    deba:\td9 1d 08 00 00 00    \tfstps  0x8
    dec0:\t5d                   \tpop    %ebp
    dec1:\tc3                   \tret
"""
    ok = True
    funcs = parse_functions(broken)
    findings = check_function('broken_leak', funcs['broken_leak'], set())
    if not findings:
        print("SELF-TEST FAIL: unmatched fld was not flagged")
        ok = False
    else:
        print("SELF-TEST ok: unmatched fld flagged ->")
        for line in format_findings(findings):
            print("  " + line)

    funcs = parse_functions(balanced)
    findings = check_function('balanced_twin', funcs['balanced_twin'], set())
    if findings:
        print("SELF-TEST FAIL: balanced code flagged as dirty ->")
        for line in format_findings(findings):
            print("  " + line)
        ok = False
    else:
        print("SELF-TEST ok: balanced code reported clean")

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', nargs='?', help='binary or objdump -d text file')
    ap.add_argument('--function', help='only check this function')
    ap.add_argument('--extra-float-fn', action='append', default=[],
                     help='additional callee name known to return float/double')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--quiet-clean', action='store_true',
                     help='suppress per-function OK lines')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)

    if not args.target:
        ap.error('target binary or .dis file required (or --self-test)')

    extra = set(args.extra_float_fn)

    try:
        with open(args.target, 'r', errors='replace') as fh:
            head = fh.read(4)
        is_text = True
    except UnicodeDecodeError:
        is_text = False
    if is_text:
        with open(args.target, 'r', errors='replace') as fh:
            text = fh.read()
    else:
        text = run_objdump(args.target)

    funcs = parse_functions(text)
    # Function-entry address -> name, for resolving calls on
    # disassembly formats (Watcom/PE) that never annotate a call
    # operand with <name> at all - see depth_delta's docstring.
    addr_to_name = {insns[0].addr: nm for nm, insns in funcs.items() if insns}
    # 64-bit detection: any function-entry address needing more than 32
    # bits. Determines whether a call/return can carry a value in
    # %st(0) at all - see depth_delta's and float_return_expected's
    # is64 docstrings. A file with no functions at all is called 32-bit
    # by default, harmlessly (nothing to check either way).
    is64 = any(a > 0xFFFFFFFF for a in addr_to_name)
    if is64:
        print("(detected 64-bit addresses: float/double returns go "
              "through %xmm0, not %st(0), for this whole file)")
    total_findings = 0
    x87_funcs = 0
    checked = 0
    # Every function's peak, needed before any of them can be checked:
    # a call site's safety depends on the callee's high-water mark, not
    # on its net effect.
    peaks = {nm: function_peak(ins, extra, addr_to_name, is64)
             for nm, ins in funcs.items() if ins}
    for name, insns in funcs.items():
        if args.function and name != args.function:
            continue
        if not has_x87(insns):
            continue
        x87_funcs += 1
        findings = check_function(name, insns, extra, addr_to_name, is64,
                                  peaks)
        checked += 1
        if findings:
            total_findings += len(findings)
            for line in format_findings(findings):
                print(line)
        elif not args.quiet_clean:
            print(f"ok  {name}")

    print(f"\n{x87_funcs} function(s) contain x87 instructions, "
          f"{checked} checked, {total_findings} finding(s).")
    sys.exit(1 if total_findings else 0)


if __name__ == '__main__':
    main()
