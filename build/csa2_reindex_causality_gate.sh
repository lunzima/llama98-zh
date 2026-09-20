#!/bin/bash
# forward_csa2.c's causal block mask - the single n_vis line that says
# which compressed blocks a query at position pos may score.
# build/csa2_reindex_causality_probe.c's own header comment has the
# boundary, the reference it comes from, and why it reads s->xb2 rather
# than s->attn_out.
#
# THE CONTROL is TWO MUTATIONS of that line, not a git-history diff,
# same reason every other gate on this plan gives: a git-ref-pinned
# "before" copy goes stale the day a fix lands and HEAD catches up to
# the working tree. The two are the two distinct ways this quantity has
# been wrong, and the probe's own check list separates them - a mutant
# whose expected-to-fail lines stay PASS means the probe cannot see that
# defect; a mutant where any OTHER line changes means the checks are not
# independent. Both are reported as a gate FAILURE.
#
#   MUTATION 1, "chunk-end count": n_vis becomes (pos0 + nt) / cm - the
#   number of blocks finalized by the END of this chunk, which is
#   exactly what a counter living in the ring's own shared state hands a
#   REINDEX layer (its ci is aliased onto a source whose whole call
#   already finished). Correct at nt=1 by coincidence, leaky at every
#   wider chunk. Expected fallout: ALL FIVE checks turn FAIL - the two
#   boundary checks and both ratio-1 checks because the count no longer
#   tracks pos at all, and check3 because a per-chunk quantity cannot
#   be batch-invariant.
#
#   MUTATION 2, "one block tight": n_vis becomes pos / cm - conservative
#   (never a future block), so batch invariance survives untouched, but
#   one block short of the reference at every position and blind to the
#   query's own block entirely at cm=1. Expected fallout: check1,
#   check2, check4 and check5 turn FAIL; check3 STAYS PASS, which is
#   what makes these two mutants different tests rather than two names
#   for one.
#
# The link line is build/csa2_mode_dispatch_gate.sh's own, unchanged:
# lz_state_alloc's helpers reach ops_quant.c/ops_norm.c/ops_gdn.c/
# fwht.c, whose dispatch references the MMX/SSE2/AVX2 symbols
# unconditionally regardless of which kernel tier is selected, so the
# whole engine body plus those three objects is the minimum that links.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_reindex_causality_gate: SKIP - no C compiler"
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_reindex_causality_gate: SKIP - no python3"
    exit 2
}

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2reindexcausalitygate.$$)
mkdir -p "$W/fixed" "$W/mut1" "$W/mut2"
trap 'rm -rf "$W"' EXIT

for d in fixed mut1 mut2; do
    cp src/*.c src/*.h "$W/$d/" 2>/dev/null
done

# Sanity: the line both mutations target must exist exactly once in the
# current source, or this gate no longer matches the code it names.
NEEDLE='    int n_vis = (pos + 1) / cm;'
N=$(grep -Fc "$NEEDLE" "$W/fixed/forward_csa2.c")
if [ "$N" != "1" ]; then
    echo "csa2_reindex_causality_gate: FAIL - expected exactly 1 occurrence of" \
         "the n_vis line in forward_csa2.c, found $N (the gate no longer" \
         "matches the source)"
    exit 1
fi

mutate() {
    python3 - "$1" "$2" <<'PYEOF' || exit 1
import sys
path, new = sys.argv[1], sys.argv[2] + "\n"
old = "    int n_vis = (pos + 1) / cm;\n"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
if src.count(old) != 1:
    print("csa2_reindex_causality_gate: FAIL - mutation could not find "
          "exactly one copy of the n_vis line", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src.replace(old, new, 1))
PYEOF
}

mutate "$W/mut1/forward_csa2.c" '    int n_vis = (pos0 + nt) / cm;'
mutate "$W/mut2/forward_csa2.c" '    int n_vis = pos / cm;'

# A needle that is gone means the mutation took. The three outcomes are
# kept apart rather than collapsed into `if grep -q`: grep exits 2 when
# it cannot read the file at all, and `if` reads that as "not found" -
# i.e. as a successful mutation. The arms below are still caught by the
# compile step (a genuinely missing source fails the link with rc 2,
# which they treat as fatal), so this is a self-check reporting a read
# error as a pass rather than a hole in the gate - but a self-check
# should fail closed on its own account too, for the reason
# build/csa2_mode_dispatch_gate.sh's own check_mutated gives.
check_mutated() {
    local dir="$1" file="$2" needle="$3" label="$4"
    local rc
    grep -qF "$needle" "$dir/$file"
    rc=$?
    case $rc in
    0)  echo "csa2_reindex_causality_gate: FAIL - could not mutate $label" \
             "($dir/$file still has '$needle')"
        exit 1 ;;
    1)  : ;;
    *)  echo "csa2_reindex_causality_gate: FAIL - could not read $dir/$file to" \
             "check $label (grep exited $rc)"
        exit 1 ;;
    esac
}
check_mutated "$W/mut1" forward_csa2.c "$NEEDLE" "mutant 1"
check_mutated "$W/mut2" forward_csa2.c "$NEEDLE" "mutant 2"

ENGX="err compat lfn json safetensors model ops ops_kernel ops_rope ops_moe ops_epi ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn ops_conv1d fwht forward forward_attn forward_moe forward_ssm forward_kda forward_engram forward_csa2 forward_mtp forward_rope forward_ckpt forward_state sampler generate lz_mathf lz_bf16 lz4d tokenizer unicode chat gbk cpucheck"
ENGX_MMX="ops_mmx ops_mmx_sse ops_sse2"
ENGX_AVX2="ops_avx2"

build_and_run() {
    local dir="$1" label="$2"
    local f
    for f in $ENGX_MMX; do
        $CC -O2 -std=c99 -w -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mmmx -msse -msse2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_reindex_causality_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    for f in $ENGX_AVX2; do
        $CC -O2 -std=c99 -w -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mavx2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_reindex_causality_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    if ! $CC -O0 -std=c99 -w -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            $(for f in $ENGX_MMX $ENGX_AVX2; do echo "$dir/$f.o"; done) \
            build/csa2_reindex_causality_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_reindex_causality_gate: FAIL - probe did not build/link against $label"
        tail -n 30 "$dir/cc.log"
        return 2
    fi
    ( cd "$dir" && ./probe > run.log 2>&1 )
    local rc=$?
    cat "$dir/run.log"
    return $rc
}

# The probe's own total ok() call count. Every mutant here runs it to
# completion (none crashes), so each must account for exactly this many
# PASS+FAIL lines - a check that stopped running altogether would
# otherwise be invisible against an expected-FAIL list that never names
# it.
PROBE_TOTAL_LINES=5

# check_lines LOGFILE TAG PASS|FAIL LINE...
check_lines() {
    local logfile="$1" tag="$2" mode="$3"
    shift 3
    local line missing=0
    for line in "$@"; do
        grep -qF "$mode $line" "$logfile" || {
            echo "csa2_reindex_causality_gate: FAIL - $tag did not turn $mode on: $line"
            missing=1
        }
    done
    return $missing
}

C1="c2/check1: the REINDEX layer's own k_eff is min(topk, (pos+1)/cm) for every token"
C2="c2/check2: the FULL layer's own k_eff is min(topk, (pos+1)/cm) -"
C3="c2/check3: every token's o_proj output is bit-identical between"
C4="c2/check4: at cm==1 (the report's own uncompressed-main-KV special case,"
C5="c2/check5: the same at cm==1 for a REINDEX layer,"

RC=0

echo "-- current tree (must PASS) --"
build_and_run "$W/fixed" "the current (fixed) tree"
FIXED_RC=$?
if [ "$FIXED_RC" -eq 2 ]; then
    exit 1
elif [ "$FIXED_RC" -ne 0 ]; then
    echo "csa2_reindex_causality_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi
if [ "$(grep -c '^PASS\|^FAIL' "$W/fixed/run.log")" != "$PROBE_TOTAL_LINES" ]; then
    echo "csa2_reindex_causality_gate: FAIL - the current tree did not print all" \
         "$PROBE_TOTAL_LINES lines the probe is expected to print"
    RC=1
fi

echo ""
echo "-- mutant 1: n_vis is the chunk-end count, not this position's --"
build_and_run "$W/mut1" "mutant 1 (chunk-end count)"
MUT1_RC=$?
if [ "$MUT1_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT1_RC" -eq 0 ]; then
    echo "csa2_reindex_causality_gate: FAIL - mutant 1 PASSED; the probe cannot see this defect"
    RC=1
else
    missing=0
    check_lines "$W/mut1/run.log" "mutant 1" "FAIL" "$C1" "$C2" "$C3" "$C4" "$C5" || missing=1
    if [ "$(grep -c '^PASS\|^FAIL' "$W/mut1/run.log")" != "$PROBE_TOTAL_LINES" ]; then
        echo "csa2_reindex_causality_gate: FAIL - mutant 1 did not print all" \
             "$PROBE_TOTAL_LINES lines the probe normally prints"
        missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        RC=1
    else
        echo "-- mutant 1 failed every check, as expected --"
    fi
fi

echo ""
echo "-- mutant 2: n_vis one block tighter than the reference --"
build_and_run "$W/mut2" "mutant 2 (one block tight)"
MUT2_RC=$?
if [ "$MUT2_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT2_RC" -eq 0 ]; then
    echo "csa2_reindex_causality_gate: FAIL - mutant 2 PASSED; the probe cannot see this defect"
    RC=1
else
    missing=0
    check_lines "$W/mut2/run.log" "mutant 2" "FAIL" "$C1" "$C2" "$C4" "$C5" || missing=1
    check_lines "$W/mut2/run.log" "mutant 2" "PASS" "$C3" || missing=1
    if [ "$(grep -c '^PASS\|^FAIL' "$W/mut2/run.log")" != "$PROBE_TOTAL_LINES" ]; then
        echo "csa2_reindex_causality_gate: FAIL - mutant 2 did not print all" \
             "$PROBE_TOTAL_LINES lines the probe normally prints"
        missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        RC=1
    else
        echo "-- mutant 2 failed exactly the four boundary checks, leaving batch invariance green --"
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_reindex_causality_gate: FAIL"
    exit 1
fi
echo "csa2_reindex_causality_gate: PASS"
exit 0
