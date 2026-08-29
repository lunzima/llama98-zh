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
#   lz_dot128_x16_asm         70%   <-- since interleaved+folded, now 14%
#   lz_dot128_q16_asm         71%   <-- since four-streamed, now 6%
#   lz_dot128_q41_asm         74%   <-- since interleaved, now 26%
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
# --pair HELPS, BUT NOT AS MUCH AS IT FIRST LOOKED, and the correction
# is worth keeping because the first reading of this table was mine.
# Its stated job is sharing one weight unpack across two tokens - a
# memory argument. Every _2 kernel does score lower than its
# single-token twin, by a similar margin, in all five weight formats:
#
#   dot32_x16_sse2   66% -> 57%      dot32_q61_sse2   70% -> 62%
#   dot32_q41_sse2   66% -> 57%      dot32_q16_sse2   65% -> 56%
#   dot32_t2_sse2    69% -> 61%
#
# I read that as "pairing already interleaves, so it buys issue width
# as well as bandwidth". It does not. CARRYING two independent chains
# is not INTERLEAVING them: lz_dot32_x16_asm_2 held two accumulators
# and still scored 55%, because it finished both tokens against one
# widened half before it touched the other. Hand-interleaving it took
# it to 20% with the same instruction count - so the ~9 points pairing
# gives is the unpack being shared, and the other 35 were still on the
# floor.
#
# The right statement: pairing buys bandwidth, and leaves most of the
# issue width for whoever reorders the body.
#
# Whole set: 4791 adjacent register pairs, 59% dependent; ops_mmx.c
# alone is 2396 and 43% after the edits below; ops_sse2.c is 52%.
#
# EIGHT HAVE BEEN DONE, so the technique is not hypothetical here.
#
# THE "NOW" NUMBERS ARE AS OF EACH EDIT, NOT AS OF THIS RUN - the
# latency-oriented reschedule (4f5c80b, aux_sched.py) reordered several
# of these bodies afterwards, and bd4c1e7 (Q4_1 relocate) re-typed the
# q41 ones, so the "now X%" claims below read stale against the table.
# Read the rates off the run output, not the prose. The q41 dot32/dot128
# sit at 44% now (the claim said 26-27%); they are still interleaved
# (mm6/mm7 alternate), so the drift is a scheduling state, not the
# sequential-block bug. The one that drifted furthest was
# lz_dot32_t2_asm: its two nibble
# chains ran one whole then the other (sequential blocks, not
# instruction-interleaved) at 62%, while the claim said 7%. Re-interleaved
# 2026-08-29 (same register set mm0-mm6, the mm5/mm6 chains alternating
# instruction by instruction, no arithmetic change) - back to 7%, and
# bit-identical: dot128_t2_probe 49,156 x 3 tiers 0 mismatches, and the
# MMX-tier engine logits match gcc's twin exactly.
#
# lz_dot32_t2_asm was the worst in the file at 75%; the two-nibble
# split (low nibble mm5, high nibble mm6, into separate accumulators)
# brought it to 62%, and instruction-level interleaving of the two
# chains to 7%. The reschedule had paid the dependency back for latency
# in the other direction; the re-interleave did not - see 0b197de/d163851,
# both interleaves reverted for clobbering a live register, neither of
# which this edit reproduces (no register added, none re-ordered across
# a live boundary).
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
# AND THEN ITS FOLD, 20% -> 14%, in its own commit because it changes
# the STORE shape rather than the chunk order. Sixteen of the thirty-two
# dependent pairs were the four horizontal reductions, four each -
# movq/psrlq/paddd/movd, every step reading the last, which is what a
# horizontal reduction IS. Keeping four accumulators (mm6 mm5 mm4 mm0,
# exactly the free set) and transposing once at the end is the t2
# kernel's trick: six fewer instructions, and two 8-byte stores instead
# of four 4-byte ones.
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
# lz_dot32_x16_asm_2, 55% -> 20%. It ALREADY carried two independent
# chains - that is what --pair means - and still scored 55%, because
# carrying two chains is not the same as interleaving them: it fed both
# tokens from one widened half before touching the other. mm0 and mm4
# were free, so the lo and hi halves now go side by side and each token
# takes its two products in the order it took them before.
#
# That kernel had no probe at all, only a commit-message claim of
# "200000 random trials". It has one now, and reaching it needed
# dot32_x16_mmx_2 to exist on the Watcom side - src/ops_mmx.h declared
# it and only the gcc build defined it.
#
# AND THE SSE2 GROUP TWIN, lz_dot128_x16_sse2_asm: 65% -> 31% by the
# interleave, then -> 17% by the fold, 122 instructions to 110, four
# 4-byte stores to two 8-byte ones. Two accumulators are enough there:
# fold after every SECOND sub-block, transpose the pair with
# punpckldq/punpckhdq and store both sums at once. Four would not fit -
# SSE2 needs a register for each of the two activation loads, and
# giving one up costs more dependent pairs in the body than the extra
# accumulators save in the fold.
#
# THAT KERNEL HAD NO PROBE COVERAGE AT ALL until this round. It is a
# #pragma aux called by name inside row_q8_sse2_asm, so it had no
# address; lz_dot128_x16_sse2_w is one, and the probe now holds MMX
# GROUP against SSE2 GROUP. That arm is also the only one that covers
# a FOLD - no per-32 comparison ever reduces four sub-blocks.
#
# lz_dot128_q16_asm, 71% -> 6%, and it is the one that shows what the
# ceiling actually is. Its body was movq/pmaddwd/paddd eight times over,
# three instructions each reading the last, and only mm1 and mm6 were
# ever live - six registers idle. The four sub-blocks are four
# INDEPENDENT streams in memory, so the body now walks chunk i of every
# block before chunk i+1 of any, each stream into its own accumulator.
# Zero dependent pairs in the body; the six that remain are the
# transposing fold. 118 instructions to 112, two 8-byte stores instead
# of four 4-byte ones.
#
# It is also the only one of these that needed NO associativity
# argument: each accumulator still receives its own block's chunks in
# their original order. What changed is the order the streams are
# VISITED, not the order anything is summed.
#
# STILL NOT DONE: the q61 group kernels only. The exp pair and the q41
# group were interleaved after this list was written - the entries below
# carry the before/after (exp_q20_2_mmx 69%->55%, exp_q20_sse2 74%->43%,
# q41 both tiers).
#
# TWO OF THEM HAVE NO SPARE REGISTER, measured rather than assumed:
# lz_dot128_q61_asm uses all eight mm and lz_dot128_q61_sse2_asm all
# eight xmm - Q6_1 reconstructs 6-bit values from two planes and holds
# two mask constants while doing it. Freeing one means spilling a mask
# to memory and paying a load per use, which trades register pressure
# for memory traffic in a direction this project's requirements argue
# against. Not "hard": a different change from the ones above.
#
# THE MASK CANNOT BE ELIMINATED, which is the first thing a reader
# checks. The 0x03 plane is fully packed: 32 values x 2 bits = 8 bytes,
# four 2-bit groups per byte, and the kernel's four chunks extract them
# with psrlw 0 / 4 / 2 / 6. Every bit of the plane is a value, so
# `w2 << 4` without the 0x03 mask reads the neighbouring group's bits
# into bits [7:6] of the reconstruction - not zero by construction.
# Verified from src/ops_mmx.c's shift sequence and the plane sizing in
# row_q61_mmx_intrin, not assumed.
#
# NOR DOES THE dot32 VARIANT HAVE THE SPARE REGISTER, which the "group
# kernels only" phrasing below understates: lz_dot32_q61_asm reads 64%
# dependent too, and its live set is also all eight mm (mm1 accumulator,
# mm2/mm3 the two masks, mm4 the w2 qword, mm5 the w4 qword, mm6/mm7/mm0
# the reconstruction and the two widen temps). An interleave would need
# a register to hold one chunk's pmaddwd product while the next chunk
# reuses the widen temps - mm7 and mm0 are both live through their
# paddd, so there is none. Same verdict, same evidence, one size down.
# lz_dot128_q41_asm has mm0 and mm4 free and IS the same edit.
#
# THE GENERAL LESSON FROM THE x16 PAIR, since it will come up again:
# when two kernels in one file share a body and one is the other's
# reference, the probe's agreement arm goes blind the moment you touch
# both. Reach for a tier that shares no code before reaching for the
# edit.
#
# lz_dot128_q41_sse2_asm, 65% -> 47%, instruction count unchanged at 137.
# The same interleave as the MMX twin: xmm5 was free, so the two nibble
# planes of each chunk run their lo halves side by side and their hi
# halves side by side. Four movdqu+pmaddwd pairs become two per half,
# so the instruction count stays the same - each half's second pmaddwd
# takes its own movdqu rather than sharing one.
#
# THE SAME REFERENCE TRAP AS x16, hit again: lz_dot32_q41_asm's body is
# lz_dot128_q41_asm's first block, character for character, and the
# Edit refused with "2 matches" before it could do any harm. That is
# twice now, so it is a property of this file rather than an accident:
# a per-32 kernel and its group kernel are written to be identical, and
# that is exactly what makes one useless as the other's oracle the
# moment you touch both.
#
# lz_dot32_q41_asm, 73% -> 27%, the same edit as its group twin: mm0 and
# mm4 were free, so each qword extracts its lo and hi nibbles together
# (mm1/mm4) and widens them side by side (mm2/mm0). The paddd order into
# mm6 is unchanged, so it is bit-identical; the probe holds it against
# the already-verified group kernel rather than against itself.
#
# lz_dot32_q16_asm, 71% -> 15%, the group twin's four-stream trick but
# with two rounds instead of eight: Q16_0 has no nibble split, so each
# of the eight MACs is already independent and only the single mm6
# accumulator serialised them. Four accumulators (mm0/mm5/mm6/mm7) over
# four 8-element streams, folded at the end with three paddd plus the
# horizontal one. Integer addition is associative mod 2^32, so the split
# is bit-identical; the probe holds it against the SSE2 per-32 oracle
# directly (dot32_q16_sse2a), 81924 values, 0 mismatches.
#
# lz_dot32_q16_asm_2, 72% -> 22%, the x16_2 lesson a third time: the
# paired kernel ALREADY carried two independent accumulators (mm6/mm7)
# and still serialised them, feeding token A's whole product chain before
# token B's. Interleaving the two tokens' pmaddwd/paddd pairs side by
# side - both mm1/mm2 temporaries copied from mm0 up front - leaves each
# accumulator's add order untouched, so it is bit-identical. The probe's
# paired arm holds it against the SSE2 paired oracle, 81924 values.
#
# lz_dot32_q16_sse2_asm, 65% -> 32%, two streams where the MMX twin took
# four: Q16_0 int16 weights need no unpack, so the SSE2 form has xmm3/5/6/7
# free and each MAC already moves independent loads. Two accumulators
# (xmm4/xmm3) over two weight+activation pairs (xmm0/xmm5 and xmm1/xmm6)
# break the single-accumulator chain. Integer addition is associative mod
# 2^32, so it is bit-identical; the probe's cross-tier arm holds it
# against dot32_q16_mmx, 81924 values, 0 mismatches.
#
# lz_dot128_q16_sse2_asm, 65% -> 32%, the one q16 gap this census missed:
# the SSE2 GROUP twin ran its four sub-blocks one whole then the next into
# a single xmm4 accumulator, while the MMX group twin had already walked
# them as four concurrent streams. Same edit: four accumulators
# (xmm4/xmm5/xmm6/xmm7) over four streams, weights movdqa'd and
# activations taken as pmaddwd memory operands (no separate movdqu, so the
# body is also shorter), folded at the end. No associativity argument is
# needed - each stream reaches its own store in its original order - and
# build/dot128_q16_probe.c holds it against the MMX group kernel and the
# per-32 oracle, 81924 values, 0 mismatches under Watcom.
#
# lz_exp_q20_sse2_asm, 74% -> 43%, the two-lane split a fourth time. Each
# of the three multiplier stages of the Q20 Taylor computes a LOW lane
# (pmuludq against the unshifted half) and a HIGH lane (psrldq, pmuludq,
# pslldq) that recombine with por; the two write xmm2 and xmm3, read only
# the constants and the shared source, and stay independent until the por.
# So each stage was two serial chains that could be interleaved: moving
# the high lane's movdqa xmm3, xmm0 up beside the low lane's movdqa xmm2,
# xmm0 and alternating the two lanes' pmuludq/paddq/psrlq breaks every
# chain with no extra register and no arithmetic change - each lane still
# reaches its por in its original order. The fourth stage differs only in
# that its source is tab/cq, so its two shifts (psrldq xmm0 and psrldq
# xmm1) both sit inside the high lane and the low lane's read of the
# unshifted cq must stay ahead of psrldq xmm1. build/exp_q20_probe.c holds
# it against lz_exp_fixed's ORIGINAL association, 1,282,566 values,
# 0 mismatches under Watcom.
#
# lz_exp_q20_2_mmx_asm, 69% -> 55%, the paired MMX exp twin of the SSE2
# one above. The Q20 square term splits into three 16-bit limbs
# (mm6/mm7/mm5) that ran one whole then the next, each a serial
# movq/pmaddwd/pmaddwd/pslld/paddd chain; mm0 and mm1 are dead here (mm0
# is the consumed s, mm1 the consumed s>>1), so the three limbs get
# disjoint accumulators (mm3/mm0/mm1) and retire side by side. No
# arithmetic change - each limb still reaches its combine in its original
# order - and build/exp_q20_probe.c holds it against lz_exp_fixed's
# ORIGINAL association, 1,282,566 values, 0 mismatches under Watcom.
#
# lz_wsum_pair_sse2_asm, 68% -> 8%, three independent changes in one body.
# The attention pair MAC has four groups; each group sign-extends rowA
# (xmm2) and rowB (xmm3) in two serial chains that never touch, then folds
# the low and high words into two products accumulated at two disjoint
# memory offsets. Interleaving the two sign-extension chains (movdqa,
# punpcklbw/punpckhbw, psraw alternating xmm2/xmm3), moving the high
# punpckhwd up beside the low punpcklwd, and giving the second product its
# own accumulator (xmm6 was free) breaks every chain: the two rows, the
# two products, and the two accumulate-lanes all retire side by side. Each
# lane still reaches its own store in its original order and the two
# accumulator offsets are disjoint, so it is bit-identical;
# build/wsum_pair_probe.c holds it against the MMX tier and the declared C
# contract, 524288 values, 0 mismatches under Watcom.
#
# lz_wsum_pair_asm (the MMX twin), 56% -> 6%, the same three changes as the
# SSE2 twin above: the two sign-extension chains interleaved, the high
# punpckhwd moved up beside the low punpcklwd, and mm6 as the second
# accumulator (mm6 was free). MMX processes four elements a move, so there
# are eight groups instead of four and the same edit repeated. It is
# bit-identical; build/wsum_pair_probe.c holds it against the SSE2 tier and
# the declared C contract, 524288 values, 0 mismatches under Watcom.
#
# lz_matmul_row_sse_asm, 33% -> 0%, and it sat hidden behind a measurement
# bug THIS SCRIPT carried until it was found here: the terminator search
# looked for __parm, but eleven kernels (this one, lz_fwht_stage_f32_sse_asm,
# the p2 split kernels) spell it bare `parm`, so their bodies over-extended
# into the next kernel and every one of their numbers was wrong. Now fixed
# to accept either spelling. The F32 matmul row held xmm4-xmm7 free yet
# reused xmm2/xmm3 for the second 4-float half, a WAR serialisation; giving
# that half its own xmm4/xmm5 retires the two mulps/addps chains side by
# side. The halves write separate accumulators (xmm0, xmm1) in the same
# order, so it is bit-identical; build/matmul_row_probe.c holds it against
# the scalar C contract compiled -fp6 (volatile float keeps the product
# 32-bit, matching mulps), 160000 values, 0 mismatches under Watcom.
#
# lz_rmsnorm_out_sse_asm, 67% -> 44%, the element-wise RMS-norm output
# o = (x*scale)*(w+offset). Its two chains - mulps x*scale and addps
# w+offset - are independent until the final mulps, but they ran one whole
# then the other; moving the w load up beside the x load interleaves them
# and cuts the loop body's dependent pairs from four to two. No arithmetic
# change; build/rmsnorm_out_probe.c holds it against the scalar C contract
# compiled -fp6, 1280000 values, 0 mismatches under Watcom.
#
# THE SMALL SCANS, checked and found not worth it. lz_vmax_sse_asm (78%)
# is dominated by its six-instruction horizontal-max fold, not by its
# loop; a two-accumulator loop interleave DOES break the cross-iteration
# maxps chain, but that chain is non-adjacent so this adjacent-only
# metric barely moves (78%->73%) and the out-of-order SSE1 target hides
# it anyway - the edit stays (it is bit-identical, held by a 132000-value
# probe) but is not a measured win. lz_amax32_mmx/sse2 (67%/65%) and
# lz_norm_ss_sse_asm (72%) are all quantization (src/ops_quant.c -
# q8_amax, q8_group_scale, norm_ss_fixed), so their single-accumulator
# scans run once at conversion time and never in the inference loop; a
# dual-issue edit there buys nothing on the in-order target.
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
    # its parm specifier. Watcom spells that __parm in most kernels and
    # bare `parm` in eleven of them (e.g. lz_matmul_row_sse_asm), so the
    # terminator is either spelling at the start of a line; matching on
    # the pragma rather than on quotes anywhere keeps ordinary string
    # literals out.
    for m in re.finditer(r'#pragma aux\s+([A-Za-z_0-9]+)\s*=', s):
        name = m.group(1)
        mm = re.search(r'\n[ \t]*(?:__)?parm\b', s[m.end():])
        if not mm:
            continue
        end = m.end() + mm.start()
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
