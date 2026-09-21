#!/bin/bash
# Which dims the indexer key's RoPE rotates - the slice src/forward_csa2.c
# hands src/ops_rope.c's lz_rope, read off the engine's own product.
# build/csa2_indexer_rope_slice_probe.c's own header comment has the two
# conventions (the V4.1 reference's trailing `k[..., -rd:]` against
# lz_rope's leading rotary_dim), the identity arm, and why the pre-rotation
# row comes from the same model at position 0 rather than from a second
# implementation of the compressor.
#
# THE CONTROLS are TWO MUTATIONS of the two lz_rope call sites, not a
# git-history diff (same reason every other gate on this plan gives: a
# git-ref-pinned "before" copy goes stale the day a fix lands). The probe
# reads the rows after the calls, so these are the two ways the dims
# actually move, and the check list separates them:
#
#   MUTATION 1, "leading slice": the offset is dropped from the indexer
#   K call, so lz_rope rotates the row's first rotary_dim dims again.
#   Expected fallout: check1, check2 and check3 turn FAIL; check4 STAYS
#   PASS (at rotary_dim == indexer_head_dim the offset is 0 either way -
#   the identity arm is exactly the case this mutation cannot move) and
#   check5 STAYS PASS (the main KV's own call is untouched).
#
#   MUTATION 2, "offset on the wrong call": the same offset is added to
#   the MAIN KV call instead. Expected fallout: check5 turns FAIL and
#   every indexer check STAYS PASS - which is what makes this a second
#   test rather than a second name for mutation 1.
#
# The link line is build/csa2_reindex_causality_gate.sh's own, unchanged:
# lz_state_alloc's helpers reach ops_quant.c/ops_norm.c/ops_gdn.c/fwht.c,
# whose dispatch references the MMX/SSE2/AVX2 symbols unconditionally
# regardless of which kernel tier is selected, so the whole engine body
# plus those three objects is the minimum that links.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler / no python3).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_indexer_rope_slice_gate: SKIP - no C compiler"
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_indexer_rope_slice_gate: SKIP - no python3"
    exit 2
}

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2idxropeslicegate.$$)
mkdir -p "$W/fixed" "$W/mut1" "$W/mut2"
trap 'rm -rf "$W"' EXIT

for d in fixed mut1 mut2; do
    cp src/*.c src/*.h "$W/$d/" 2>/dev/null
done

# Sanity: the two lines both mutations target must exist exactly once in
# the current source, or this gate no longer matches the code it names.
NEEDLE_IDX='(size_t)blk * idx_hd + (idx_hd - rotary_dim)'
NEEDLE_KV='lz_rope(ring + (size_t)blk * hd, 1, hd, rotary_dim'
N=$(grep -Fc "$NEEDLE_IDX" "$W/fixed/forward_csa2.c")
if [ "$N" != "1" ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - expected exactly 1 occurrence" \
         "of '$NEEDLE_IDX' in forward_csa2.c, found $N (the gate no longer" \
         "matches the source)"
    exit 1
fi
N=$(grep -Fc "$NEEDLE_KV" "$W/fixed/forward_csa2.c")
if [ "$N" != "1" ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - expected exactly 1 occurrence" \
         "of '$NEEDLE_KV' in forward_csa2.c, found $N (the gate no longer" \
         "matches the source)"
    exit 1
fi

mutate() {
    python3 - "$1" "$2" "$3" <<'PYEOF' || exit 1
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
if src.count(old) != 1:
    print("csa2_indexer_rope_slice_gate: FAIL - mutation could not find "
          "exactly one copy of %r" % (old,), file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src.replace(old, new, 1))
PYEOF
}

mutate "$W/mut1/forward_csa2.c" \
    '(size_t)blk * idx_hd + (idx_hd - rotary_dim)' '(size_t)blk * idx_hd'
mutate "$W/mut2/forward_csa2.c" \
    'lz_rope(ring + (size_t)blk * hd, 1, hd, rotary_dim' \
    'lz_rope(ring + (size_t)blk * hd + (hd - rotary_dim), 1, hd, rotary_dim'

# A needle that is gone means the mutation took. The three outcomes are
# kept apart rather than collapsed into `if grep -q`: grep exits 2 when it
# cannot read the file at all, and `if` reads that as "not found" - i.e. as
# a successful mutation. (build/csa2_reindex_causality_gate.sh's own
# check_mutated gives the same reason.)
check_mutated() {
    local dir="$1" needle="$2" label="$3"
    local rc
    grep -qF "$needle" "$dir/forward_csa2.c"
    rc=$?
    case $rc in
    0)  echo "csa2_indexer_rope_slice_gate: FAIL - could not mutate $label" \
             "($dir/forward_csa2.c still has '$needle')"
        exit 1 ;;
    1)  : ;;
    *)  echo "csa2_indexer_rope_slice_gate: FAIL - could not read" \
             "$dir/forward_csa2.c to check $label (grep exited $rc)"
        exit 1 ;;
    esac
}
check_mutated "$W/mut1" '(idx_hd - rotary_dim)' "mutant 1"
check_mutated "$W/mut2" "$NEEDLE_KV" "mutant 2"

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
            echo "csa2_indexer_rope_slice_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    for f in $ENGX_AVX2; do
        $CC -O2 -std=c99 -w -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mavx2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_indexer_rope_slice_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    if ! $CC -O0 -std=c99 -w -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            $(for f in $ENGX_MMX $ENGX_AVX2; do echo "$dir/$f.o"; done) \
            build/csa2_indexer_rope_slice_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_indexer_rope_slice_gate: FAIL - probe did not build/link against $label"
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
# otherwise be invisible against an expected-FAIL list that never names it.
PROBE_TOTAL_LINES=5

check_lines() {
    local logfile="$1" tag="$2" mode="$3"
    shift 3
    local line missing=0
    for line in "$@"; do
        grep -qF "$mode $line" "$logfile" || {
            echo "csa2_indexer_rope_slice_gate: FAIL - $tag did not turn $mode on: $line"
            missing=1
        }
    done
    return $missing
}

C1="idx/check1: the indexer key row's leading idx_hd - rotary_dim dims"
C2="idx/check2: the trailing rotary_dim dims are the frequency table's"
C3="idx/check3: control - the trailing dims did move"
C4="idx/check4: identity arm - at rotary_dim == indexer_head_dim the"
C5="idx/check5: control - the main KV entry still rotates its LEADING"

RC=0

echo "-- current tree (must PASS) --"
build_and_run "$W/fixed" "the current (fixed) tree"
FIXED_RC=$?
if [ "$FIXED_RC" -eq 2 ]; then
    exit 1
elif [ "$FIXED_RC" -ne 0 ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi
if [ "$(grep -c '^PASS\|^FAIL' "$W/fixed/run.log")" != "$PROBE_TOTAL_LINES" ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - the current tree did not print all" \
         "$PROBE_TOTAL_LINES lines the probe is expected to print"
    RC=1
fi
check_lines "$W/fixed/run.log" "the current tree" PASS "$C1" "$C2" "$C3" "$C4" "$C5" || RC=1

echo ""
echo "-- mutant 1: the indexer K call rotates the leading slice again --"
build_and_run "$W/mut1" "mutant 1 (leading slice)"
MUT1_RC=$?
if [ "$MUT1_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT1_RC" -eq 0 ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - mutant 1 PASSED; the probe cannot see this defect"
    RC=1
fi
if [ "$(grep -c '^PASS\|^FAIL' "$W/mut1/run.log")" != "$PROBE_TOTAL_LINES" ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - mutant 1 did not print all" \
         "$PROBE_TOTAL_LINES lines the probe is expected to print"
    RC=1
fi
check_lines "$W/mut1/run.log" "mutant 1" FAIL "$C1" "$C2" "$C3" || RC=1
check_lines "$W/mut1/run.log" "mutant 1" PASS "$C4" "$C5" || RC=1

echo ""
echo "-- mutant 2: the offset lands on the main KV call instead --"
build_and_run "$W/mut2" "mutant 2 (offset on the main KV call)"
MUT2_RC=$?
if [ "$MUT2_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT2_RC" -eq 0 ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - mutant 2 PASSED; the probe cannot see this defect"
    RC=1
fi
if [ "$(grep -c '^PASS\|^FAIL' "$W/mut2/run.log")" != "$PROBE_TOTAL_LINES" ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - mutant 2 did not print all" \
         "$PROBE_TOTAL_LINES lines the probe is expected to print"
    RC=1
fi
check_lines "$W/mut2/run.log" "mutant 2" FAIL "$C5" || RC=1
check_lines "$W/mut2/run.log" "mutant 2" PASS "$C1" "$C2" "$C3" "$C4" || RC=1

[ "$RC" -eq 0 ] && { echo ""
    echo "csa2_indexer_rope_slice_gate: PASS"; exit 0; }
echo ""
echo "csa2_indexer_rope_slice_gate: FAIL"
exit 1
