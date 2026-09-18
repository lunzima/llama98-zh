# -*- coding: utf-8 -*-
"""The s->wscr borrow rule, counted mechanically.

lz_t_f32(t, SCRATCH) BORROWS. It returns t->f for LZ_FMT_F32, and for every
other format it widens the tensor INTO SCRATCH and hands SCRATCH back. So a
second borrow of the same scratch does not sit next to the first, it
REPLACES it: the earlier pointer keeps its address and starts reading the
later tensor's bytes. This export carries no F32 tensor at all (Q8_0 483 /
BF16 183 / Q4_1 24 / Q6_1 8 / Q16_0 8), so on every shipped checkpoint the
scratch path is the one that runs and an F32 test is what hides the bug -
`A_log/dt_bias are F32, so wscr is free here` held on safetensors, failed on
a bin, and the decay gate saturated at (-5, 0) because of it.

THE RULE, stated once (forward.h's kda_decay field comment) and counted here:

  A borrow handed to a POINTER variable must not be alive across another
  borrow of the SAME scratch.

  "alive" is from the assignment to EACH later use of that variable in the
  enclosing function, not just the next one - an earlier version stopped at
  the next use and would have missed a borrow landing between the second and
  third uses of a variable that outlives a mutually exclusive arm.
  When the variable has no use in this function at all - the consumer took
  its address, as LZMoeRouteParams.rp.bias does - the whole enclosing brace
  block is the live range instead, which is the conservative reading.

WHAT IS NOT A BORROW, and is skipped on purpose

  * Unnamed: `lz_rmsnorm(..., lz_t_f32(&L->q_norm, s->wscr), ...)`. Dead at
    the end of its own full expression. This is the correct shape.
  * Element read: `lz_t_f32(&L->dt_bias, s->wscr)[h]` or an `[i]` on the
    left. What escapes is a float, not a pointer.
  * Value copy: `float dt_bias_h = lz_t_f32(...)[h]`. The `*` test below is
    what separates this from a pointer borrow.

WHAT IT CANNOT SEE. Each miss is written down rather than papered over:

  * A live range narrowed by an arm test is invisible. The range is taken
    from the assignment to the LAST use in the function, and a use sitting
    in a different arm still counts, so a borrow landing after this arm's
    own last use reads as a violation. That is what EXEMPT below exists
    for, and why an exemption carries the line it was granted for. It is a
    DIFFERENT class from two borrows in mutually exclusive arms - which
    this checker also cannot tell apart - and an exemption granted for one
    is not an argument about the other.
  * Scratches reached through a macro, or recomputed at run time, are
    invisible. This is a text-level check; it supplements reading the call
    site, it does not replace it.
  * Two unnamed borrows into two arguments of ONE full expression are not
    checked: C leaves that evaluation order unspecified and no site in this
    tree does it.
  * A file with no borrows at all exits 2, not 0 - the failure that would
    otherwise read as a clean tree.

Exit: 0 clean, 1 violations (or a stale exemption), 2 the check itself could
not run.
"""
import os
import re
import sys

# (file, variable) -> (line, why)
EXEMPT = {
    ("forward_kda.c", "kda_o_norm_w"): (
        898,
        "The two borrows are NOT in mutually exclusive arms: :898 and :916 "
        "are both inside the same `if (c->use_subn)` block and run in "
        "order. What makes :916 safe is where this variable's USE range "
        "ends on that path - its last use there is :911, and the uses at "
        ":955/:968/:980 that stretch the range to the end of the function "
        "sit in the `else if` ladder `use_subn` excludes. The checker "
        "cannot narrow a live range by an arm test, which is the whole of "
        "what this exemption is granted for."),
}


def strip(src):
    """Blank comments and literals, preserving every offset."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = src.find('\n', i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = ' '
            i = j
        elif c == '/' and i + 1 < n and src[i + 1] == '*':
            j = src.find('*/', i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, min(j, n)):
                if out[k] != '\n':
                    out[k] = ' '
            i = j
        elif c == '"' or c == "'":
            q = c
            j = i + 1
            while j < n:
                if src[j] == '\\':
                    j += 2
                    continue
                if src[j] == q:
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, n)):
                if out[k] != '\n':
                    out[k] = ' '
            i = j
        else:
            i += 1
    return ''.join(out)


def lines_of(src):
    off = [0]
    for ln in src.split('\n'):
        off.append(off[-1] + len(ln) + 1)
    return off


def lineno(offsets, pos):
    lo, hi = 0, len(offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offsets[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def brace_pairs(src):
    stack, out = [], []
    for i, c in enumerate(src):
        if c == '{':
            stack.append(i)
        elif c == '}' and stack:
            out.append((stack.pop(), i))
    return out


def enclosing(pairs, pos):
    best = None
    for (a, b) in pairs:
        if a < pos < b and (best is None or (b - a) < (best[1] - best[0])):
            best = (a, b)
    return best


def find_calls(src, name):
    """[(start, end, argstart, argend)] for `name(...)`, parens balanced."""
    out = []
    for m in re.finditer(r'(?<![A-Za-z0-9_])' + name + r'\s*\(', src):
        i = src.index('(', m.start())
        depth, j = 0, i
        while j < len(src):
            if src[j] == '(':
                depth += 1
            elif src[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= len(src):
            continue
        out.append((m.start(), j + 1, i + 1, j))
    return out


def split_args(src, argstart, argend):
    args, depth, cur = [], 0, argstart
    for i in range(argstart, argend):
        c = src[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == ',' and depth == 0:
            args.append(src[cur:i])
            cur = i + 1
    args.append(src[cur:argend])
    return [a.strip() for a in args]


def assignment_lhs(src, callstart):
    """Full left-hand side text if the call is the right of an `=`, else None.

    The whole declarator is returned, `const float *` included: whether the
    result is a pointer borrow or a value copy is decided from this text.
    """
    i = callstart - 1
    while i >= 0 and src[i] in ' \t\r\n':
        i -= 1
    if i < 0 or src[i] != '=':
        return None
    i -= 1
    while i >= 0 and src[i] in ' \t\r\n':
        i -= 1
    if i < 0 or src[i] in '=!<>+-*/%&|^':
        return None                      # ==, !=, <=, >=, +=, ...
    end = i + 1
    j = i
    while j >= 0 and src[j] not in ';{}':
        j -= 1
    text = src[j + 1:end].strip()
    return text or None


def lhs_var(text):
    """The named object in an assignment's LHS, subscripts stripped."""
    m = re.search(r'([A-Za-z_][A-Za-z0-9_]*'
                  r'(?:\s*(?:->|\.)\s*[A-Za-z_][A-Za-z0-9_]*)*)'
                  r'\s*(?:\[[^\]]*\])*\s*$', text)
    if not m:
        return None
    return re.sub(r'\s+', '', m.group(1))


def lhs_is_pointer(src, fstart, fend, text):
    """False for a scalar copy (`float dt_bias_h = ...`), true otherwise.

    An array element on the left is a scalar too. A struct member is taken
    as a pointer: the member's type is not in this translation unit. A bare
    name carries no type at all - `a_log_arr = lz_t_f32(...)` assigns to
    something declared elsewhere - so the function's own declarations are
    searched for `*name`. Missing that search was this checker's second
    bug: forward_kda's a_log_arr and dt_bias_arr are declared `const float
    *` above and assigned here, and a check that read only the assignment
    line called them value copies.
    """
    if re.search(r'\[[^\]]*\]\s*$', text):
        return False
    if ('*' in text) or ('->' in text) or ('.' in text):
        return True
    var = lhs_var(text)
    if not var:
        return False
    return bool(re.search(r'\*\s*' + re.escape(var) + r'(?![A-Za-z0-9_])',
                          src[fstart:fend]))


def func_span(src, pos):
    """[start, end) of the function enclosing pos, by column-0 braces.

    K&R layout puts a function's closing brace in column 0 and a nested
    block's does not. That is the whole of the heuristic; an unusual brace
    style would change the answer, which is why the span is used only to
    keep one function's same-named locals from being read as another's.
    """
    start = 0
    for m in re.finditer(r'\n\}', src):
        if m.start() < pos:
            start = m.start() + 1
        else:
            break
    m = re.search(r'\n\}', src[pos:])
    end = pos + m.start() + 1 if m else len(src)
    return start, end


def collect(src):
    """Every borrow of a scratch in one translation unit.

    `var` is set only on the ones a POINTER is handed to - those are the
    borrows this check watches. The others are still borrows, and still
    overwrite whatever was in the scratch, so they stay in the list as the
    second half of every pair. Dropping them here was this checker's first
    bug: forward_kda.c's kda_o_norm_w is overwritten by an UNNAMED borrow
    one arm over, and a list that only held named ones saw a group of one.
    """
    offsets = lines_of(src)
    out = []
    for (cs, ce, as_, ae) in find_calls(src, 'lz_t_f32'):
        after = src[ce:ce + 8].lstrip()
        if after.startswith('{'):         # the definition itself
            continue
        args = split_args(src, as_, ae)
        if len(args) < 2:
            continue
        rec = {'var': None, 'scratch': re.sub(r'\s+', '', args[1]),
               'call': (cs, ce), 'line': lineno(offsets, cs)}
        lhs = assignment_lhs(src, cs)
        if lhs is not None and not after.startswith('['):
            fs, fe = func_span(src, cs)
            if lhs_is_pointer(src, fs, fe, lhs):
                v = lhs_var(lhs)
                if v:
                    rec['var'] = v
        out.append(rec)
    return out


def check_file(path):
    with open(path, 'r') as fh:
        src = strip(fh.read())
    name = os.path.basename(path)
    calls = collect(src)
    named = [c for c in calls if c['var']]
    if not calls:
        return [], 0, 0, 0
    pairs = brace_pairs(src)
    by_scratch = {}
    for c in calls:
        by_scratch.setdefault(c['scratch'], []).append(c)

    raw = []
    for b in named:
        group = by_scratch[b['scratch']]
        if len(group) < 2:
            continue
        fstart, fend = func_span(src, b['call'][0])
        uses = [m.start() for m in
                re.finditer(r'(?<![A-Za-z0-9_>!.])' + re.escape(b['var']) +
                            r'(?![A-Za-z0-9_])', src)
                if fstart <= m.start() < fend and m.start() >= b['call'][1]]
        for b2 in group:
            if b2 is b:
                continue
            hit = None
            for u in uses:
                if b['call'][1] < b2['call'][0] < u:
                    hit = 'use at :%d' % lineno(lines_of(src), u)
                    break
            if hit is None and not uses:
                blk = enclosing(pairs, b['call'][0])
                if blk is not None and blk[0] < b2['call'][0] < blk[1]:
                    hit = 'no use in this file; block ends at :%d' % \
                          lineno(lines_of(src), blk[1])
            if hit:
                raw.append((name, b, b2, hit))
                break

    real, exempted = [], 0
    offsets = lines_of(src)
    for (f, b, b2, why) in raw:
        key = (f, b['var'])
        if key in EXEMPT:
            if EXEMPT[key][0] == b['line']:
                exempted += 1
                continue
            real.append((f, b['line'], b['var'], b['scratch'], b2['line'],
                         'STALE EXEMPTION: recorded for line %d, found here'
                         % EXEMPT[key][0]))
            continue
        real.append((f, b['line'], b['var'], b['scratch'], b2['line'], why))
    return real, len(calls), len(named), exempted


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    srcdir = os.path.join(root, 'src')
    if not os.path.isdir(srcdir):
        sys.stderr.write('wscr_borrow_check: no %s\n' % srcdir)
        return 2
    files = sorted(f for f in os.listdir(srcdir) if f.endswith('.c'))
    allv, total, named, exempt_n = [], 0, 0, 0
    for f in files:
        v, n, nn, e = check_file(os.path.join(srcdir, f))
        allv += v
        total += n
        named += nn
        exempt_n += e
    print('wscr_borrow_check: %d scratch borrow(s), %d of them handing a '
          'pointer to a variable, over %d file(s)'
          % (total, named, len(files)))
    if named == 0:
        print('FAIL: not one borrow was found - the scan is not seeing the '
              'tree it thinks it is')
        return 2
    print('                   %d exemption(s) matched their recorded line'
          % exempt_n)
    if not allv:
        print('PASS')
        return 0
    for (f, ln, var, scratch, ln2, why) in sorted(allv):
        print('  %s:%d: %s <- %s is still live when %s is borrowed at :%d (%s)'
              % (f, ln, var, scratch, scratch, ln2, why))
    print('FAIL: %d borrow(s) outlive a second borrow of the same scratch'
          % len(allv))
    return 1


if __name__ == '__main__':
    sys.exit(main())
