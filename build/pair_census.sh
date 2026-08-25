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
#   lz_dot32_t2_asm           75%   <-- since interleaved, now 7%
#   lz_dot128_t2_asm          72%   <-- since interleaved, now 14%
#   lz_dot128_x16_asm         70%   <-- since interleaved, now 20%
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
# Whole set: 4791 adjacent register pairs, 59% dependent; ops_mmx.c
# alone is 2404 and 51% after the four edits below.
#
# FOUR HAVE BEEN DONE, so the technique is not hypothetical here.
#
# lz_dot32_t2_asm was the worst in the file at 75% and is now 7% - the
# best - by running the low and high nibble of each weight byte side by
# side on separate scratch into separate accumulators.
#
# lz_dot128_t2_asm, 72% -> 14%, needed a different split and the reason
# is worth keeping. Its four accumulators plus the weight quad plus one
# activation register leave exactly ONE free, so two chunks cannot each
# hold their own activation - but they do not have to. The chunks
# already pair off: same shift level, punpcklbw against punpckhbw of the
# same mm4. So the pair takes two SCRATCH registers (mm2 and the spare
# mm7) and SHARES one activation register, reloading it between the two
# pmaddwd. That reload is the one remaining dependent pair inside a
# chunk pair; the other is the second paddd into the accumulator.
# Instruction count is unchanged at 274.
#
# BOTH ARE BIT-IDENTICAL, and not by an argument about associativity -
# the same products reach the same accumulator in the same order.
# build/dot128_t2_probe.c passes 16,388 values with 0 mismatches and
# prints the same four lane sums as before either edit; end to end on
# kunmoe-v2-t2-probe the generated text is byte-for-byte what a build of
# the pre-edit kernel produces.
#
# lz_dot128_x16_asm, 70% -> 20%, was the easy one: mm0 mm3 mm4 mm5 were
# all free, so the lo and hi halves of each weight quad just take mm2
# and mm3 and run side by side. Instruction count unchanged at 167.
#
# WHAT THE REMAINING 20% IS, and it is not chunk work: 16 of the 32
# dependent pairs are the four horizontal folds, four each. A fold is
# movq/psrlq/paddd/movd and every step reads the last - that is what a
# horizontal reduction IS. The t2 kernel avoids it by keeping four
# accumulators and transposing once at the end, two 8-byte stores
# instead of four movd, and this kernel has the registers to do the
# same. Not done here: it is a change to the STORE shape, not to the
# chunk order, and mixing the two would muddy which one a regression
# came from.
#
# lz_dot32_x16_asm, 68% -> 20%, the same edit again - but it is the
# REFERENCE build/dot128_x16_probe.c holds the group kernel against, so
# changing it needed a different oracle rather than a different edit.
# Two kernels in one TU with the same body, changed the same way, would
# agree with each other and the probe would say nothing. The probe now
# also runs lz_dot32_x16_sse2 over every case: it shares no code with
# either, and cross-tier bit-identity is a requirement in its own
# right. 32,772 values, 0 mismatches. Mutation (the two pmaddwd
# operands swapped) fires all three detectors - the group check, the
# SSE2 check, and the absolute lane sums.
#
# THE REST ARE NOT DONE: lz_exp_q20_2_mmx_asm at 69%,
# lz_dot32_x16_asm_2 at 55%, and the SSE2 single-token kernels at
# 65-74%. Each is a real edit needing its own probe re-run on both
# toolchains.
#
# THE GENERAL LESSON FROM THE x16 PAIR, since it will come up again:
# when two kernels in one file share a body and one is the other's
# reference, the probe's agreement arm goes blind the moment you touch
# both. Reach for a tier that shares no code before reaching for the
# edit.
#
# Named and measured; what this script is for is that the next person
# does not re-derive which kernels are worth it.
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
