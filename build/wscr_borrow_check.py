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

THE STORAGE-POINT RULE, which is this rule's unbounded case. The live range
above is a local variable inside one function, so it ends. A borrow STORED
INTO A `static` OBJECT ends when the process does, and no call site can
reason about it:

  A `const T *` assigned into a `static` object is an unbounded live range.
  It is admissible only with content verification at the point of use, or a
  line-anchored exemption saying what makes it bounded instead.

  Content verification is the shape that works: the address identifies
  nothing, but the bytes do, so the consumer compares the whole row before
  it trusts anything derived from it. What the rule rejects is the
  pointer-keyed cache - `if (slot[i].w == w) return slot[i].p;` - which is
  an unbounded identity claim about a pointer that a scratch reuse makes
  false. src/ops_norm.c's two pools and src/ops_gdn.c's gdn_shadow are the
  three instances; the checker reports each store and each is exempted by
  line with its own argument. This is the class the live-range rule above
  cannot express, because its model is a variable inside one function and
  this one spans a function boundary with no end.

  THE SCAN IS TEXT-LEVEL AND THE PREPROCESSOR DOES NOT RUN, so an instance
  under `#if` is reported by the build that compiles it away. That is the
  point rather than a side effect: src/ops_gdn.c's g_shk store sits under
  `#if LZ_GDN_STATE_F32`, whose default is 0, and the rule sees it in the
  DEFAULT build - a claim the gate could not make if it keyed on what the
  compiler emitted. A latent instance that only a switch-matrix run can
  reach is exactly the one that would otherwise be added and forgotten.

WHAT IT CANNOT SEE. Each miss is written down rather than papered over:

  * A store THROUGH a pointer - `ps->j->nodes[i].key = k`, where `ps` is a
    caller's structure and only its name collides with a static's - is not
    reported. That is the one of the three detector false positives this
    rule produced on its first run that is not a regex bound but ALIASING,
    and aliasing is what a text-level check cannot resolve: it can only
    MISS here, never misreport. The list of stores below is therefore
    sound in the direction that matters (nothing reported is wrong) and
    NOT exhaustive (something stored through a pointer is invisible).
  * A borrow stored into a non-`static` object that nonetheless outlives
    the scratch - a field of a heap struct that lives as long as the model,
    say - is not reported. The rule keys on `static` because that is the
    only unbounded lifetime a text check can see; the rest needs the reader.
  * A pointer whose declaration carries no `const` on the pointee is not
    treated as a borrow. `static float *p = our_own_pool + off;` is owning,
    not borrowing, and the rule is silent about it - which is also why a
    non-const borrowed pointer escaping into a static would be missed.

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

# (file, line) -> why the store at that line is not an unbounded identity
# claim. Keyed on the LINE, not on a name: move the store and this goes
# stale, which is reported rather than silently re-matched at the new site.
STORE_EXEMPT = {
    ("ops_norm.c", 185): (
        "The stored w is an ADDRESS USED ONLY AS A PRE-FILTER. What admits "
        "the table is w_row_same, which compares all n elements of the row "
        "bit for bit before any value derived from it is consumed; a "
        "pointer that matches and a row that does not is a miss, not a "
        "hit. So the unbounded lifetime of this stored pointer carries no "
        "identity claim - moving the store or reusing the address changes "
        "nothing observable."),
    ("ops_norm.c", 401): (
        "Same as :185: build_qw's slot holds the address as a pre-filter "
        "and asks w_row_same for the answer. The two pools are keyed on the "
        "same quantity by the same function, deliberately."),
    ("ops_gdn.c", 1373): (
        "KNOWN DEFECT, and this exemption records it rather than clearing "
        "it. gdn_shadow's lookup is `if (g_shk[i] == sq) return g_shv[i];` "
        "- a pointer-keyed cache of ring pointers whose addresses are "
        "reused, which is exactly the shape this rule rejects. It is latent "
        "rather than live: the whole block sits under #if LZ_GDN_STATE_F32, "
        "whose default is 0, and reaching it also needs --spec K. The site "
        "is annotated at its own comment (ops_gdn.c's gdn_shadow header) as "
        "flagged rather than fixed. Remove this exemption when it is fixed."),
    ("ops_kernel.c", 1403): (
        "The stored `where` is a caller-supplied diagnostic string. Its "
        "pointee is a string literal, whose lifetime is the program's, so "
        "the unbounded store is the correct lifetime for it - there is no "
        "scratch to outlive and the value is never consumed as data."),
}

# (file, variable) -> (line, why)
EXEMPT = {
    ("forward_kda.c", "kda_o_norm_w"): (
        925,
        "The two borrows are NOT in mutually exclusive arms: :925 and :943 "
        "are both inside the same `if (c->use_subn)` block and run in "
        "order. What makes :943 safe is where this variable's USE range "
        "ends on that path - its last use there is :938, and the uses at "
        ":982/:995/:1007 that stretch the range to the end of the function "
        "sit in the `else if` ladder `use_subn` excludes. The checker "
        "cannot narrow a live range by an arm test, which is the whole of "
        "what this exemption is granted for."),
}


def _decl_end(src, i):
    """Offset of the `;` closing a declaration started before i, or -1.

    A struct/union/enum body is stepped over as a unit; a `{` that is not
    one of those ends a function HEAD, which is not a data object and is
    reported as -2 so the caller can skip it.
    """
    d = 0
    while i < len(src):
        c = src[i]
        if c == '{':
            if src[:i].rstrip().endswith(('struct', 'union', 'enum')):
                d2 = 0
                while i < len(src):
                    if src[i] == '{':
                        d2 += 1
                    elif src[i] == '}':
                        d2 -= 1
                        if d2 == 0:
                            break
                    i += 1
                i += 1
                continue
            return -2
        if c in '([':
            d += 1
        elif c in ')]}':
            d -= 1
        elif c == ';' and d == 0:
            return i
        elif c == '=' and d == 0:
            return i
        i += 1
    return -1


def static_objects(src):
    """{name: declarator} for every `static` object that can hold a pointer.

    Both scopes: a function-local static has the same process lifetime as a
    file-scope one, which is the only thing this rule keys on.

    Two exclusions, each paid for by a false positive this rule produced on
    its first run and each a different shape:

      * A `(` in the declarator is a FUNCTION, not an object. `static int
        jparse_value(LZJsonParser *ps);` is a prototype, and reading its
        last identifier declared `ps` - a name every one of json.c's parser
        functions also has as its parameter - which made two stores through
        a caller's pointer look like stores into a static.
      * A declarator with no `*` cannot hold a pointer. `static union {float
        f; uint32_t u;} b;` and `static int32_t g_epi_zg[..][..];` are
        scalar storage, and reading `b.f = x` and `g_epi_zg[t][g] = a` as
        pointer stores made every `const float *x` in the file a borrowed
        pointer being cached.
    """
    out = {}
    for m in re.finditer(r'\bstatic\b', src):
        e = _decl_end(src, m.end())
        if e < 0:
            continue
        decl = src[m.end():e]
        if '(' in decl or '*' not in decl:
            continue
        head = re.sub(r'\b(?:struct|union|enum)\b\s*\{[^{}]*\}', ' ', decl)
        head = re.split(r'[\[=]', head, maxsplit=1)[0]
        ids = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', head)
        if ids:
            out[ids[-1]] = decl
    return out


def const_pointer_names(src):
    """Names declared `const T *name` anywhere in the unit.

    The pointee's `const` is what makes the pointer a BORROW rather than
    owned storage, and it is the cut this rule uses. It covers both a
    parameter (`const float *w`) and a widened local (`const float *q_norm_w
    = lz_t_f32(...)`), which is the pair a store into a static is reached
    through.
    """
    return set(re.findall(r'\bconst\b[^;,(){}]*?\*\s*([A-Za-z_][A-Za-z0-9_]*)',
                          src))


def collect_stores(src):
    """[(line, lhs_text, rhs_name, static_name)] for `static[...] = borrow;`.

    Only a bare identifier on the right is reported. A computed pointer -
    `our_pool + off`, a calloc result - is the store site's own storage or
    its own allocation, and neither is a borrow of somebody else's scratch.
    """
    objs = static_objects(src)
    consts = const_pointer_names(src)
    out = []
    off = lines_of(src)
    for m in re.finditer(r'[^;{}]*;', src):
        stmt = m.group(0)
        if stmt.endswith(';'):
            stmt = stmt[:-1]
        eq = None
        d = 0
        for k, c in enumerate(stmt):
            if c in '([{':
                d += 1
            elif c in ')]}':
                d -= 1
            elif c == '=' and d == 0 and k + 1 < len(stmt) \
                    and stmt[k + 1] not in '=' \
                    and (k == 0 or stmt[k - 1] not in '=!<>+-*/%&|^'):
                eq = k
                break
        if eq is None:
            continue
        lhs = stmt[:eq].strip()
        rhs = stmt[eq + 1:].strip()
        root = re.match(r'[A-Za-z_][A-Za-z0-9_]*', lhs)
        if not root or root.group(0) not in objs:
            continue
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', rhs):
            continue
        if rhs not in consts:
            continue
        out.append((lineno(off, m.start() + eq), ' '.join(lhs.split()),
                    rhs, root.group(0)))
    return out


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
    stores = collect_stores(src)
    storev, store_exempt_n = [], 0
    for (ln, lhs, rhs, obj) in stores:
        key = (name, ln)
        if key in STORE_EXEMPT:
            store_exempt_n += 1
            continue
        storev.append((name, ln, lhs, rhs, obj, ''))
    found = set(s[0] for s in stores)
    for key in STORE_EXEMPT:
        if key[0] == name and key[1] not in found:
            storev.append((name, key[1], '(no store here)', '', '',
                           'STALE EXEMPTION: recorded for line %d, but this '
                           'line stores no borrowed pointer' % key[1]))
    if not calls:
        return [], 0, 0, 0, storev, store_exempt_n, len(stores)
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
    return real, len(calls), len(named), exempted, storev, store_exempt_n, \
        len(stores)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    srcdir = os.path.join(root, 'src')
    if not os.path.isdir(srcdir):
        sys.stderr.write('wscr_borrow_check: no %s\n' % srcdir)
        return 2
    files = sorted(f for f in os.listdir(srcdir) if f.endswith('.c'))
    allv, total, named, exempt_n = [], 0, 0, 0
    alls, nstore, store_exempt_n = [], 0, 0
    for f in files:
        v, n, nn, e, sv, se, ns = check_file(os.path.join(srcdir, f))
        allv += v
        total += n
        named += nn
        exempt_n += e
        alls += sv
        nstore += ns
        store_exempt_n += se
    print('wscr_borrow_check: %d scratch borrow(s), %d of them handing a '
          'pointer to a variable, over %d file(s)'
          % (total, named, len(files)))
    if named == 0:
        print('FAIL: not one borrow was found - the scan is not seeing the '
              'tree it thinks it is')
        return 2
    print('                   %d exemption(s) matched their recorded line'
          % exempt_n)
    print('                   %d borrow(s) stored into a static object, '
          '%d of them exempt' % (nstore, store_exempt_n))
    if nstore == 0:
        print('FAIL: not one store into a static object was found - the '
              'storage-point scan is not seeing the tree it thinks it is')
        return 2
    if not allv and not alls:
        print('PASS')
        return 0
    for (f, ln, var, scratch, ln2, why) in sorted(allv):
        print('  %s:%d: %s <- %s is still live when %s is borrowed at :%d (%s)'
              % (f, ln, var, scratch, scratch, ln2, why))
    for t in sorted(alls):
        extra = ' ' + t[5] if t[5] else ''
        print('  %s:%d: %s = %s, stored into the static `%s` - an unbounded '
              'live range needs content verification at the point of use, '
              'or a line-anchored exemption.%s'
              % (t[0], t[1], t[2], t[3], t[4], extra))
    print('FAIL: %d borrow(s) outlive a second borrow of the same scratch, '
          '%d store(s) put a borrowed pointer into a static object'
          % (len(allv), len(alls)))
    return 1


if __name__ == '__main__':
    sys.exit(main())
