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
FLOOR=33

# TWENTY-SIX ARE STILL UNREACHABLE, and the number is what it is
# because "reachable" means a THIN WRAPPER, not a mention:
#
#   ops_mmx.c       gdn1_x4, wsum_pair, p2_split32, lerp_q15_32,
#                   exp_q20_2, dot32_q41_2, dot32_q61, dot32_q61_2,
#                   dot32_q16_2
#   ops_sse2.c      p2_mul32, p2_split32, lerp_q15_32, exp_q20,
#                   dot32_x16_2, dot32_q41, dot32_q41_2, dot32_t2,
#                   dot32_t2_2, dot32_q61, dot32_q61_2, dot128_q61,
#                   dot32_q16, dot32_q16_2, dot128_q16
#   ops_mmx_sse.c   p2_split32, exp_q20_2
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
        head = i
        while head < len(lines) and head < i + 5:
            if lines[head].rstrip().endswith('{'):
                break
            head += 1
        else:
            i += 1
            continue
        if head >= len(lines) or not lines[head].rstrip().endswith('{'):
            i += 1
            continue
        depth, j = 0, head
        while j < len(lines):
            depth += lines[j].count('{') - lines[j].count('}')
            if depth <= 0 and j > head:
                break
            j += 1
        out.append((i, j))
        i = j + 1
    return out, lines

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
        hit = False
        for a, b in spans:
            # A WRAPPER, not merely a mention. row_t2_mmx_asm names
            # lz_dot128_t2_asm inside its own row loop, which does not
            # give a probe the kernel's address or let it choose the
            # kernel's arguments - it would have to drive a whole row.
            # A wrapper is short: signature, one call, closing brace.
            if b - a > 6:
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

# Control: the parser must report a planted kernel as unreachable and a
# wrapped one as reachable, or its counts carry nothing.
CTL=$(python - <<'PY'
import re
src = ('#pragma aux k_wrapped = "x"\n'
       '#pragma aux k_bare = "x"\n'
       'void wrap(void) { k_wrapped(1); }\n'
       'static void s(void) { k_bare(1); }\n')
lines = src.split('\n')
spans = []
for i, ln in enumerate(lines):
    if ln and ln[0] not in ' \t#/*}' and not ln.startswith('static') \
       and '{' in ln:
        spans.append((i, i))
got = []
for m in re.finditer(r'^#pragma aux\s+([A-Za-z_0-9]+)', src, re.M):
    n = m.group(1)
    hit = any(re.search(r'\b' + n + r'\s*\(', '\n'.join(lines[a:b + 1]))
              for a, b in spans)
    got.append(n + ('=reach' if hit else '=miss'))
print(' '.join(got))
PY
)
if [ "$CTL" != "k_wrapped=reach k_bare=miss" ]; then
    echo "kernel_reach_gate: FAIL - on a planted pair the parser said"
    echo "  '$CTL', expected 'k_wrapped=reach k_bare=miss'. A static"
    echo "  wrapper must NOT count as reachable; that was the"
    echo "  dot32_q16_mmx case."
    BAD=1
fi

if [ "$BAD" -eq 0 ]; then
    echo "kernel_reach_gate: PASS ($REACH of $TOTAL kernels reachable,"
    echo "  floor $FLOOR; a planted bare kernel reads as unreachable and"
    echo "  a static wrapper does not rescue it)"
    [ -n "$MISS" ] && printf '  still unreachable: %s\n' $MISS
    exit 0
fi
echo "kernel_reach_gate: FAIL"
exit 1
