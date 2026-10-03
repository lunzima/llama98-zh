#!/bin/bash
# The indexer key's RoPE - which dims it rotates AND which elements it
# pairs - read off the engine's own product.
# build/csa2_indexer_rope_slice_probe.c's own header comment has the two
# conventions (the V4.1 reference's trailing `k[..., -rd:]` with adjacent
# element pairs against lz_rope's leading rotary_dim dims with
# base[i]/base[i + rotary_dim/2]), the identity arm, and why the
# pre-rotation row comes from the same model at position 0 rather than
# from a second implementation of the compressor.
#
# THE CONTROLS are THREE MUTATIONS, not a git-history diff (same reason
# every other gate on this plan gives: a git-ref-pinned "before" copy goes
# stale the day a fix lands). The probe reads the rows after the rotation
# call, so these are the three ways those dims can move wrongly, and the
# check list separates them:
#
#   MUTATION 1, "pairing reverted": the helper keeps the trailing slice
#   but pairs base[i] with base[i + rotary_dim/2] again, which is what
#   lz_rope does. Expected fallout: check2 and check6 turn FAIL (the
#   trailing dims are the host's rotation, not the reference's) and
#   check4 turns FAIL (at equal widths that is the whole row); check1,
#   check3 and check5 STAY PASS - the slice and the main KV entry are
#   untouched by this one.
#
#   MUTATION 2, "slice reverted": the helper keeps the reference's
#   adjacent pairing but takes the LEADING rotary_dim dims instead
#   (base = 0). Expected fallout: check1, check2 and check3 turn FAIL;
#   check4 STAYS PASS (at rotary_dim == indexer_head_dim the two offsets
#   are the same, which is exactly what the identity arm is there to say)
#   and check5 and check6 stay PASS.
#
#   MUTATION 3, "both rows through the indexer helper": the MAIN KV entry
#   takes the indexer's rotation instead of lz_rope. Expected fallout:
#   check5 turns FAIL and every indexer check STAYS PASS - which is what
#   makes this a third test rather than a second name for the first two.
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
mkdir -p "$W/fixed" "$W/mut1" "$W/mut2" "$W/mut3"
trap 'rm -rf "$W"' EXIT

for d in fixed mut1 mut2 mut3; do
    cp src/*.c src/*.h "$W/$d/" 2>/dev/null
done

# Sanity: the call and the line each mutation targets must exist exactly
# once in the current source, or this gate no longer matches the code it
# names.
NEEDLE_CALL='indexer_rope_adjacent(idxl, idx_hd,'
NEEDLE_BASE='    int half = rotary_dim / 2, base = width - rotary_dim, i;'
NEEDLE_KV='lz_rope(lat, 1, hd, rotary_dim, pos, s->rope_cs);'
for pair in "call:$NEEDLE_CALL" "base:$NEEDLE_BASE" "kv:$NEEDLE_KV"; do
    label=${pair%%:*}
    needle=${pair#*:}
    N=$(grep -Fc "$needle" "$W/fixed/forward_csa2.c")
    if [ "$N" != "1" ]; then
        echo "csa2_indexer_rope_slice_gate: FAIL - expected exactly 1 occurrence" \
             "of '$needle' in forward_csa2.c, found $N (the gate no longer" \
             "matches the source)"
        exit 1
    fi
done

# Each mutation is applied to its own copy with the exact text it replaces
# written out here, so a source that has moved on fails loudly instead of
# mutating something else.
mutate() {
    python3 - "$1" "$2" <<'PYEOF' || exit 1
import sys
path, which = sys.argv[1], sys.argv[2]
OLD = {
 'pairing': '''        float x0 = row[base + 2 * i];
        float x1 = row[base + 2 * i + 1];
        row[base + 2 * i]     = x0 * c - x1 * sn;
        row[base + 2 * i + 1] = x1 * c + x0 * sn;
''',
 'slice': '''    int half = rotary_dim / 2, base = width - rotary_dim, i;
''',
 'kv': '''            lz_rope(lat, 1, hd, rotary_dim, pos, s->rope_cs);
''',
}
NEW = {
 'pairing': '''        float x0 = row[base + i];
        float x1 = row[base + half + i];
        row[base + i]        = x0 * c - x1 * sn;
        row[base + half + i] = x1 * c + x0 * sn;
''',
 'slice': '''    int half = rotary_dim / 2, base = 0, i;
''',
 'kv': '''            indexer_rope_adjacent(lat, hd, rotary_dim, pos, s->rope_cs);
''',
}
with open(path, 'r', encoding='utf-8') as f:
    src = f.read()
if src.count(OLD[which]) != 1:
    print("csa2_indexer_rope_slice_gate: FAIL - mutation %r could not find "
          "exactly one copy of its target" % which, file=sys.stderr)
    sys.exit(1)
with open(path, 'w', encoding='utf-8') as f:
    f.write(src.replace(OLD[which], NEW[which], 1))
PYEOF
}

mutate "$W/mut1/forward_csa2.c" pairing
mutate "$W/mut2/forward_csa2.c" slice
mutate "$W/mut3/forward_csa2.c" kv

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
check_mutated "$W/mut1" 'float x0 = row[base + 2 * i];' "mutant 1"
check_mutated "$W/mut2" 'base = width - rotary_dim, i;' "mutant 2"
check_mutated "$W/mut3" "$NEEDLE_KV" "mutant 3"

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
PROBE_TOTAL_LINES=6

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

# Labels are prefixes of what the probe prints; each one is a full enough
# phrase that a renumbered check cannot match a different line.
C1="idx/check1: the indexer key row's leading idx_hd - rotary_dim dims"
C2="idx/check2: the trailing rotary_dim dims are the reference's"
C3="idx/check3: control - the trailing dims did move"
C4="idx/check4: identity arm - at rotary_dim == indexer_head_dim the"
C5="idx/check5: control - the main KV entry still rotates its LEADING"
C6="idx/check6: control - the trailing dims are NOT the host's"

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
check_lines "$W/fixed/run.log" "the current tree" PASS \
    "$C1" "$C2" "$C3" "$C4" "$C5" "$C6" || RC=1

echo ""
echo "-- mutant 1: the pairing is the host's half-split again --"
build_and_run "$W/mut1" "mutant 1 (half-split pairing)"
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
check_lines "$W/mut1/run.log" "mutant 1" FAIL "$C2" "$C4" "$C6" || RC=1
check_lines "$W/mut1/run.log" "mutant 1" PASS "$C1" "$C3" "$C5" || RC=1

echo ""
echo "-- mutant 2: the slice is the leading dims again --"
build_and_run "$W/mut2" "mutant 2 (leading slice)"
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
check_lines "$W/mut2/run.log" "mutant 2" FAIL "$C1" "$C2" "$C3" || RC=1
check_lines "$W/mut2/run.log" "mutant 2" PASS "$C4" "$C5" "$C6" || RC=1

echo ""
echo "-- mutant 3: the main KV entry through the indexer's own rotation --"
build_and_run "$W/mut3" "mutant 3 (main KV through the indexer helper)"
MUT3_RC=$?
if [ "$MUT3_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT3_RC" -eq 0 ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - mutant 3 PASSED; the probe cannot see this defect"
    RC=1
fi
if [ "$(grep -c '^PASS\|^FAIL' "$W/mut3/run.log")" != "$PROBE_TOTAL_LINES" ]; then
    echo "csa2_indexer_rope_slice_gate: FAIL - mutant 3 did not print all" \
         "$PROBE_TOTAL_LINES lines the probe is expected to print"
    RC=1
fi
check_lines "$W/mut3/run.log" "mutant 3" FAIL "$C5" || RC=1
check_lines "$W/mut3/run.log" "mutant 3" PASS "$C1" "$C2" "$C3" "$C4" "$C6" || RC=1

[ "$RC" -eq 0 ] && { echo ""
    echo "csa2_indexer_rope_slice_gate: PASS"; exit 0; }
echo ""
echo "csa2_indexer_rope_slice_gate: FAIL"
exit 1
