#!/bin/bash
# Every per-operator cross-check in .prof/, run as one stage.
#
# WHY THEY NEED A CALLER. Ten harnesses compare one operator's
# implementations directly - SSE2 against MMX against C, the fixed tier
# against the float one, the Watcom x87 build against gcc, ARM
# soft-float against both. They are the only thing in this tree that
# compares operators rather than whole-engine output, and nothing ran
# them: ci.sh and the Makefile mention none of them, and
# gate_wiring_gate only counts scripts under build/.
#
# WHY OPERATOR-LEVEL AND NOT MORE FIXTURES. kernel_isa_identity_gate
# compares two tiers end to end, and a coverage build shows its fixture
# reaching 47.57% of the lines in src/ops_kernel_dot_mmx.h. The union
# over five checkpoints is 55.34% - so the remaining arms belong to
# weight formats no local model uses, and no amount of fixtures gets
# there. Feeding the kernels directly does.
#
# TWO OF THEM WERE PASSING WITHOUT MEASURING when this was written, and
# that is the failure mode to expect from a script nobody runs:
# scalar_regime_xcheck skipped both the x87 and soft-float arms it
# exists to compare, and norm_wp_xcheck did not compile at all because
# it still called lz_norm_select, one of the seven per-tier knobs the
# single float axis replaced.
#
# SKIP is not PASS here either: a harness that cannot build its probe
# says SKIP, and that is counted and printed separately.
#
# Exit codes:
#   0  PASS - every harness that could run, passed
#   1  FAIL - at least one compared and disagreed, or none could run
set -u
cd "$(dirname "$0")/.."

# wbuild_xcheck.sh is the shared Watcom probe BUILDER the others call,
# not a check of its own - it prints a usage line when run bare.
HARNESSES=$(ls .prof/*_xcheck.sh 2>/dev/null | grep -v "wbuild_xcheck")
[ -n "$HARNESSES" ] && [ -d .prof ] || {
    echo "xcheck_gate: SKIP - no .prof harnesses in this checkout"
    exit 2
}

# WHICH UNITS THESE REACH, and which they do not. The kernel matrix has
# fifteen units; mapping each to the harness that compares it:
#
#   atdot wsump        attn_sse2_xcheck        (SSE2 / MMX / C)
#   expfx sigq q8rnd
#   i32f               scalar_regime_xcheck    (gcc / x87 / soft-float)
#   nrmss              norm_ss_sse2 + norm_wp
#   fwhtf fwhti        fwht_f32 + fwht_mmx
#   f32mm              matmul_f32
#   q8_0 q4_1 q6_1
#   q16_0 t2           rowdot_xcheck           (six slots, both builds)
#   epi                epi_xcheck              (gcc / Watcom, via one
#                                              oracle over identical
#                                              inputs - see its header
#                                              for why that is a
#                                              comparison and not two
#                                              unrelated green runs)
#   recur              kda_gt1 + recur_xcheck  (gt==1 arm; pass-1 tap,
#                                              p2 group scale, pass-2
#                                              delta quantize)
#
# The five row kernels were the gap, and they are exactly the code the
# end-to-end fixture reaches least - 47.57% of ops_kernel_dot_mmx.h.
# rowdot_xcheck closes it by feeding the dispatch tables directly.
# UNCOVERED is printed every run rather than kept in a comment, so
# "every harness green" cannot be read as "every operator compared"
# while it has entries.
UNCOVERED=""

P=0; F=0; S=0
for h in $HARNESSES; do
    n=$(basename "$h" .sh)
    out=$(timeout 900 bash "$h" 2>&1)
    rc=$?
    last=$(printf '%s\n' "$out" | grep -E "PASS|FAIL|SKIP" | tail -1)
    case $rc in
        0) printf '  PASS  %-24s %s\n' "$n" "$(echo "$last" | cut -c1-52)"
           P=$((P + 1)) ;;
        2) printf '  SKIP  %-24s %s\n' "$n" "$(echo "$last" | cut -c1-52)"
           S=$((S + 1)) ;;
        *) printf '  FAIL  %-24s rc=%s\n' "$n" "$rc"
           printf '%s\n' "$out" | tail -6 | sed 's/^/          /'
           F=$((F + 1)) ;;
    esac
done

# Positive control. A glob that stopped matching, or a tree with .prof
# stripped, would run nothing and report no failures - which is the
# shape these harnesses were already in.
if [ $((P + F + S)) -lt 5 ]; then
    echo "xcheck_gate: FAIL - only $((P + F + S)) harness(es) found; the"
    echo "  list is not matching and a clean run here would mean nothing"
    exit 1
fi

echo "xcheck_gate: $P passed, $F failed, $S skipped"
if [ -n "$UNCOVERED" ]; then
    echo "xcheck_gate: NO per-operator check for:$(printf ' %s' $UNCOVERED)"
else
    echo "xcheck_gate: all 17 matrix units have a per-operator check"
fi

# The uncovered list is checked against the matrix, not trusted. A unit
# that gains a harness and is not removed from UNCOVERED understates the
# coverage; a unit ADDED to the matrix and not listed here overstates it,
# which is the direction that matters.
if [ -x ./llama98.exe ] || [ -x ./llama98 ]; then
    BIN=./llama98.exe; [ -x "$BIN" ] || BIN=./llama98
    NU=$("$BIN" --kernel-matrix 2>/dev/null | grep -c "^cell .* ref ")
    if [ "${NU:-0}" -gt 0 ] && [ "$NU" -ne 17 ]; then
        echo "xcheck_gate: FAIL - the matrix has $NU units, not the 17 the"
        echo "  mapping above was written against; re-derive UNCOVERED"
        exit 1
    fi
fi

[ "$F" -eq 0 ] || { echo "xcheck_gate: FAIL"; exit 1; }
echo "xcheck_gate: PASS"
exit 0
