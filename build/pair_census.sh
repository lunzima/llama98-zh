#!/bin/bash
# How much of each hand-written MMX kernel can dual-issue.
#
# THE TARGET FAMILY IS IN-ORDER AND DUAL-ISSUE. A Pentium retires two
# instructions per cycle only when the pair is independent; if the
# second reads or overwrites what the first just wrote, it waits. So the
# fraction of ADJACENT PAIRS that are dependent is an upper bound on how
# much of the second pipe the kernel can never use, and it is a property
# of the instruction ORDER rather than of the machine measuring it.
#
# WHY NOT THE OTHER CENSUSES. branch_census answers the predictor,
# move_census answers register pressure, and neither says anything about
# issue width. This is the third lens and it was the one missing: a
# kernel can have no branches and no spills and still run at half rate
# because every instruction feeds the next.
#
# WHAT IT DOES NOT MODEL, and the number is an upper bound because of
# it: the real U/V rules also bar two multiplies in one cycle, restrict
# shifts and packs to one pipe, and stall on address-generation
# interlocks. A pair this counts as independent may still not issue
# together. A pair it counts as DEPENDENT, though, definitely cannot -
# which is the direction that makes the number useful.
#
# THE FIRST LOOK, and it found one kernel unlike the others:
#
#   lz_exp_q20_2_mmx_asm      69%
#   lz_dot32_t2_asm           75%
#   lz_dot128_t2_asm          72%
#   lz_dot128_x16_asm         70%
#   lz_lerp_q15_32_mmx_asm    20%   <-- the outlier
#
# Four sit near seventy and one sits at twenty, so seventy is NOT "what
# MMX code looks like" - it is what a single dependency chain looks
# like. The lerp kernel is the existence proof: it carries two
# independent halves at once (mm0/mm1 holding one, mm2/mm3 the other,
# mm4/mm5 the third operand) and interleaves them, so most adjacent
# pairs touch different registers.
#
# THE FOUR ABOVE ARE ALL ONE CHAIN. exp_q20's eight-chunk body is the
# clearest: pxor, punpck, psllw, psrlw, psllw, pmaddwd, paddd is seven
# instructions each reading what the last wrote. Interleaving two chunks
# would halve that, cost no extra registers in three of the four, and
# change no arithmetic - the accumulators are separate and integer
# addition is associative and exact.
#
# AND --pair ALREADY DOES THIS, which was not why it was built. Its
# stated job is sharing one weight unpack across two tokens - a memory
# argument. But every _2 kernel scores lower than its single-token twin,
# by the same margin, in all five weight formats:
#
#   dot32_x16_sse2   66% -> 57%      dot32_q61_sse2   70% -> 62%
#   dot32_q41_sse2   66% -> 57%      dot32_q16_sse2   65% -> 56%
#   dot32_t2_sse2    69% -> 61%
#
# Two tokens means two independent chains to interleave, so the pairing
# knob buys issue width as well as bandwidth. Worth knowing before
# anyone measures --pair and attributes all of it to the unpack.
#
# Whole set: 4791 adjacent register pairs, 59% dependent.
#
# NOT ATTEMPTED HERE. Interleaving the single-token kernels by hand is a
# real edit to each, needing its probe re-run on both toolchains; it is
# named, measured and left. What this script exists for is that the next
# person does not have to re-derive which kernels are worth it - and the
# two low scorers say what the target looks like when it is reached.
#
# Usage: bash build/pair_census.sh [file ...]   (default: src/ops_mmx.c)
set -u
cd "$(dirname "$0")/.."

command -v python >/dev/null 2>&1 || { echo "pair_census: SKIP - no python"; exit 2; }

FILES="$*"
[ -n "$FILES" ] || FILES="src/ops_mmx.c src/ops_mmx_sse.c src/ops_sse2.c"

python - $FILES <<'PY'
import re, sys

def kernels(path):
    try:
        s = open(path, encoding='utf-8', errors='replace').read()
    except IOError:
        return
    # Each #pragma aux body: the name, then the quoted instructions up to
    # __parm. Matching on the pragma rather than on quotes anywhere keeps
    # ordinary string literals out.
    for m in re.finditer(r'#pragma aux\s+([A-Za-z_0-9]+)\s*=', s):
        name = m.group(1)
        end = s.find('__parm', m.end())
        if end < 0:
            continue
        body = s[m.end():end]
        ins = re.findall(r'"([a-z][a-z0-9]*)\s+([^"]*)"', body)
        yield name, ins

TOTAL = 0
DEP = 0
for path in sys.argv[1:]:
    rows = []
    for name, ins in kernels(path):
        tot = dep = 0
        for k in range(1, len(ins)):
            pd = ins[k-1][1].split(',')[0].strip()
            # Register destinations only: a store's destination is memory
            # and the next instruction cannot depend on it through a
            # register file.
            if not (pd.startswith('mm') or pd.startswith('xmm')):
                continue
            cur = [x.strip() for x in ins[k][1].split(',')]
            tot += 1
            if pd in cur:
                dep += 1
        if tot >= 8:
            rows.append((name, tot, dep))
    if not rows:
        continue
    print(path)
    for name, tot, dep in rows:
        print('  %-26s %4d pairs %4d dependent (%2.0f%%)'
              % (name, tot, dep, 100.0 * dep / tot))
        TOTAL += tot
        DEP += dep

if TOTAL == 0:
    print()
    print("pair_census: no kernel produced a row - every #pragma aux was")
    print("  skipped or had under eight register-destination pairs. An")
    print("  empty table measures nothing.")
    sys.exit(1)
print()
print('%d adjacent register pairs, %d dependent (%.0f%% overall)'
      % (TOTAL, DEP, 100.0 * DEP / TOTAL))
print('lower is better; a dependent pair cannot dual-issue on an in-order core')
PY
