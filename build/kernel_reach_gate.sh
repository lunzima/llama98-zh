#!/bin/bash
# Every hand-written kernel can be reached by a probe.
#
# WHY THIS EXISTS. A Watcom `#pragma aux` body has NO ADDRESS. It is
# expanded at each call site, so a kernel that is only ever named inside
# its own translation unit cannot be called, timed or compared from
# anywhere else - and a differential probe is by definition somewhere
# else. The kernel is not untested because anyone decided it did not
# need testing; it is untested because it is unreachable, and nothing
# says so.
#
# FOUND BY HAND FIVE TIMES before this was written, each time only
# because a probe was being built for that one kernel:
#
#   dot32_x16_mmx_2         declared in ops_mmx.h, defined only by gcc
#   dot32_q16_mmx           `static`, contradicting the same header
#   dot32_q41_mmx           same
#   lz_dot128_x16_sse2_asm  no wrapper at all
#   lz_dot128_q41_sse2_asm  no wrapper at all
#
# Nothing linked against any of them, so nothing complained. The two
# gcc/Watcom pair gates did not see it either: both sides can be
# present and both be unreachable.
#
# WHAT IS ASSERTED. For each `#pragma aux NAME` in the kernel files,
# some NON-STATIC function in that file must mention NAME in its body.
# That function is the address a probe can link against. Static
# wrappers do not count - that is exactly the dot32_q16_mmx case, where
# a wrapper existed and was invisible outside the file.
#
# THE RATCHET is FLOOR: the number of reachable kernels may not fall.
# A new kernel arrives with a wrapper, or the number moves the wrong
# way and this reddens. There is no exemption list, deliberately -
# "this one does not need to be testable" is a claim worth arguing for
# in a commit rather than parking in a variable.
#
# Exit codes:
#   0  PASS
#   1  FAIL - an unreachable kernel, or the detector cannot see one
set -u
cd "$(dirname "$0")/.."

FILES="src/ops_mmx.c src/ops_sse2.c src/ops_mmx_sse.c"
FLOOR=53

# SIX ARE STILL UNREACHABLE, and none of them is a dot product:
#
#   ops_mmx.c       wsum_pair, p2_split32
#   ops_sse2.c      p2_mul32, p2_split32, i32f_acc32
#   ops_mmx_sse.c   p2_split32
#
# THIS LIST SAID FOURTEEN AN HOUR AGO, and five of those were the
# gate's fault rather than the code's - see the note on the signature
# test below. A wrong list is worse than a shorter true one: each name
# on it is a claim that somebody should go add a wrapper.
#
# EVERY DOT KERNEL IS NOW REACHABLE AND PROBED - Q8_0, Q4_1, Q6_1,
# Q16_0 and T2, in group, per-32 and paired form, each holding its MMX
# tier against its SSE2 tier. Not one of the paired forms could be
# called from outside its own file before this round.
#
# WHAT IS LEFT IS NOT A DOT PRODUCT: p2_split32 (three tiers),
# p2_mul32, i32f_acc32, wsum_pair. Each needs its own oracle rather
# than a sixth copy of the dot probe - a split has no single return
# value to compare, an accumulate is float-valued, and wsum_pair reads
# two rows. That is why they are last, not because they matter less.
#
# THE _2 FORMS DOMINATE what is left, and that is a finding rather than
# a coincidence: the paired kernels were added later, each beside a
# single-token twin that already had a wrapper, and none of them got
# one. lz_dot32_x16_asm_2's probe coverage arrived only when its
# wrapper did.
#
# TWO KINDS OF UNREACHABLE, both real:
#   - a wrapper exists but is `static`, so it has no linkage
#   - the kernel is named only inside a row function or a GROUP4 macro,
#     where reaching it means driving a whole row rather than asking
#     the kernel a question
#
# THE FIRST VERSION OF THIS GATE COUNTED MENTIONS and reported 52 of 59
# reachable, which felt like good news. It was not: row_t2_mmx_asm
# names lz_dot128_t2_asm inside its row loop, and that mention was
# being counted. The mutation is what said so - making a real wrapper
# `static` did not move the number at all, because the row function was
# holding it up. A gate that cannot notice its own subject being
# removed is measuring something else.

command -v python >/dev/null 2>&1 || {
    echo "kernel_reach_gate: SKIP - no python"; exit 2; }

OUT=$(python - $FILES <<'PY'
import re, sys

def funcs(src):
    """(start, end) of every non-static function body in the file.

    A definition starts at column zero, is not a directive and not
    `static`/`extern`, and has a parameter list; the opening brace may
    be up to four lines further down, because these signatures wrap.
    The first version of this required the brace on the SAME line and
    reported 39 of 59 kernels unreachable - every multi-line signature,
    including lz_dot128_t2_mmx_w, which plainly is a wrapper. A
    parser's own blind spot reads exactly like a finding."""
    out, lines = [], src.split('\n')
    i = 0
    while i < len(lines):
        ln = lines[i]
        if (not ln or ln[0] in ' \t#/*}'
                or ln.startswith('static') or ln.startswith('extern')
                or '(' not in ln):
            i += 1
            continue
        # THE HEAD IS WHERE THE PARAMETER LIST CLOSES AND A BRACE
        # FOLLOWS - not "some line within five ends in {". That earlier
        # rule let a one-line function adopt a LATER function's brace:
        # `void wrap(...) { call(); }` does not end in `{`, so the
        # search walked forward and attached it to the next definition,
        # swallowing everything between. The planted selftest caught it
        # by reporting all three of its kernels reachable.
        blob = '\n'.join(lines[i:i + 6])
        p = blob.find('(')
        d, k = 0, p
        while k < len(blob):
            if blob[k] == '(':
                d += 1
            elif blob[k] == ')':
                d -= 1
                if d == 0:
                    break
            k += 1
        if k >= len(blob) or not blob[k + 1:].lstrip().startswith('{'):
            i += 1
            continue
        head = i + blob[:k].count('\n')
        depth, j = 0, head
        while j < len(lines):
            depth += lines[j].count('{') - lines[j].count('}')
            # `j > head` was here, and a one-line function never
            # terminated: its braces balance on the head line, so the
            # span ran on and swallowed whatever followed. The planted
            # selftest below is what said so - every one of its three
            # kernels read as reachable because the first wrapper had
            # eaten the other two functions.
            if depth <= 0:
                break
            j += 1
        out.append((i, j))
        i = j + 1
    return out, lines

def plist(text):
    """Parameter count of the first (...) in text, 0 for (void)."""
    i = text.find('(')
    if i < 0:
        return -1
    depth, j = 0, i
    while j < len(text):
        if text[j] == '(':
            depth += 1
        elif text[j] == ')':
            depth -= 1
            if depth == 0:
                break
        j += 1
    inner = text[i + 1:j].strip()
    if not inner or inner == 'void':
        return 0
    d, cnt = 0, 1
    for ch in inner:
        if ch in '([':
            d += 1
        elif ch in ')]':
            d -= 1
        elif ch == ',' and d == 0:
            cnt += 1
    return cnt

def nparams(src, name):
    m = re.search(r'\bextern\b[^;]*?\b' + re.escape(name) + r'\s*\(',
                  src, re.S)
    if not m:
        return -1
    return plist(src[m.end() - 1:])

def nparams_at(lines, a):
    return plist('\n'.join(lines[a:a + 6]))

total = reach = 0
bad = []
for path in sys.argv[1:]:
    try:
        src = open(path, encoding='utf-8', errors='replace').read()
    except IOError:
        continue
    spans, lines = funcs(src)
    for m in re.finditer(r'^#pragma aux\s+([A-Za-z_0-9]+)', src, re.M):
        name = m.group(1)
        total += 1
        # A WRAPPER, not merely a mention, and the test is the SIGNATURE
        # rather than the length. row_t2_mmx_asm names lz_dot128_t2_asm
        # inside its own row loop, but it takes a ROW's arguments - a
        # probe driving it cannot choose what the kernel sees. A wrapper
        # takes the kernel's own arguments, so its parameter count
        # matches within one (the odd-element tail some of them carry
        # adds no parameter; a length-n form against a fixed-width
        # kernel adds exactly one).
        #
        # A LINE-COUNT THRESHOLD WAS TRIED FIRST and called
        # lz_exp_q20_mmx unreachable - a genuine wrapper that
        # build/exp_q20_probe.c genuinely drives, thirteen lines long
        # because it handles the odd tail in C. Length is a proxy for
        # the wrong thing.
        want_np = nparams(src, name)
        hit = False
        for a, b in spans:
            if abs(nparams_at(lines, a) - want_np) > 1:
                continue
            body = '\n'.join(lines[a:b + 1])
            if re.search(r'\b' + re.escape(name) + r'\s*\(', body):
                hit = True
                break
        if hit:
            reach += 1
        else:
            bad.append(path + ':' + name)

print('TOTAL %d' % total)
print('REACH %d' % reach)
for b in bad:
    print('MISS %s' % b)

# SELFTEST, run by the SAME code above rather than a copy of it. The
# first version of this control was a hand-written duplicate of the
# scan, and it went stale the moment the predicate changed from a line
# count to a signature match - it kept passing while testing logic that
# no longer ran. Three synthetic kernels: one with a real wrapper, one
# with only a static wrapper, one named only by a function that takes a
# row's worth of arguments.
CTL = (
 'extern void k_wrapped(int a, int b);\n'
 '#pragma aux k_wrapped = "x"\n'
 'extern void k_static(int a, int b);\n'
 '#pragma aux k_static = "x"\n'
 'extern void k_rowonly(int a, int b);\n'
 '#pragma aux k_rowonly = "x"\n'
 'void wrap(int a, int b) { k_wrapped(a, b); }\n'
 'static void swrap(int a, int b) { k_static(a, b); }\n'
 'void rowfn(int a, int b, int c, int d, int e, int f) {\n'
 '    k_rowonly(a, b);\n'
 '}\n')
cspans, clines = funcs(CTL)
verdict = []
for m in re.finditer(r'^#pragma aux\s+([A-Za-z_0-9]+)', CTL, re.M):
    nm = m.group(1)
    np_ = nparams(CTL, nm)
    ok = False
    for a, b in cspans:
        if abs(nparams_at(clines, a) - np_) > 1:
            continue
        if re.search(r'\b' + nm + r'\s*\(', '\n'.join(clines[a:b + 1])):
            ok = True
            break
    verdict.append(nm + ('=reach' if ok else '=miss'))
print('CTL %s' % ' '.join(verdict))
PY
)

TOTAL=$(printf '%s\n' "$OUT" | sed -n 's/^TOTAL //p')
REACH=$(printf '%s\n' "$OUT" | sed -n 's/^REACH //p')
MISS=$(printf '%s\n' "$OUT" | sed -n 's/^MISS //p')

if [ -z "$TOTAL" ] || [ -z "$REACH" ]; then
    echo "kernel_reach_gate: FAIL - the scan produced no counts, so its"
    echo "  silence means nothing"
    exit 1
fi
if [ "$TOTAL" -lt 40 ]; then
    echo "kernel_reach_gate: FAIL - only $TOTAL kernel(s) found; the"
    echo "  pragma pattern has stopped matching and everything below"
    echo "  would read as clean"
    exit 1
fi

BAD=0
if [ "$REACH" -lt "$FLOOR" ]; then
    echo "kernel_reach_gate: $REACH of $TOTAL kernels reachable, floor is"
    echo "  $FLOOR. A kernel with no non-static caller in its own file"
    echo "  cannot be called from a probe - a #pragma aux body has no"
    echo "  address. Add a non-static wrapper."
    printf '  unreachable: %s\n' $MISS
    BAD=1
fi

CTL=$(printf '%s\n' "$OUT" | sed -n 's/^CTL //p')
WANT_CTL="k_wrapped=reach k_static=miss k_rowonly=miss"
if [ "$CTL" != "$WANT_CTL" ]; then
    echo "kernel_reach_gate: FAIL - on the planted set the scan said"
    echo "    $CTL"
    echo "  expected"
    echo "    $WANT_CTL"
    echo "  A static wrapper must NOT count (that was dot32_q16_mmx),"
    echo "  and neither must a function that names the kernel while"
    echo "  taking a row's arguments (that was row_t2_mmx_asm)."
    BAD=1
fi

if [ "$BAD" -eq 0 ]; then
    echo "kernel_reach_gate: PASS ($REACH of $TOTAL kernels reachable,"
    echo "  floor $FLOOR; on the planted set a static wrapper and a"
    echo "  row-signature caller both read as unreachable)"
    [ -n "$MISS" ] && printf '  still unreachable: %s\n' $MISS
    exit 0
fi
echo "kernel_reach_gate: FAIL"
exit 1
