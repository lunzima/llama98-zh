#!/bin/bash
# --recur-write / --fixed off must be a COMPLETE float arm.
#
# WHY THIS EXISTS: the GDN/KDA recurrence has two passes. Pass 2's tier had
# a runtime selector (--recur-write); pass 1's did not. Its "knob",
# g_gdn_fixed, was initialised to 1 and assigned by nothing in the tree - a
# compile-time constant wearing a knob's name, with a comment next to the
# dispatch calling it "the runtime knob". So `--fixed off`, this project's
# float oracle - the baseline row of build/fixed_tier_xcompiler_gate.sh,
# the reference arm of every paired-NLL check, the denominator of every
# "is this tier worth it" marginal - ran int16 fixed-point arithmetic for
# the LARGER of the two passes, and no gate could see it: every consistency
# check compared two configurations that were both running the fixed head.
#
# THE CHECK. If a runtime float arm is complete, then additionally removing
# the fixed path AT COMPILE TIME must change nothing: the code deleted was
# code the runtime flag had already stopped executing. So build the tree
# twice, once ordinary and once -DLZ_GDN_FIXED=0, and require byte-identical
# --dump-all-logits. With the pass-1 defect present all four rows below
# differ; the difference is the whole recurrence's pass-1 contraction.
#
# WHY ROWS 3 AND 4 ALSO PASS --norm-write float. -DLZ_GDN_FIXED=0 is not
# only "no fixed recurrence": ops_quant.c's norm_ss_fixed/expof and
# ops_norm.c's three fixed output loops sit inside the same #if, and
# lz_norm_can_fixed() returns 0 in that build whatever --norm-write says.
# So the compile-time build has NO fixed norm either, and --recur-write
# float alone cannot match it - the residual is the norms tier, not the
# recurrence. Floating the norms too isolates the axis under test. Rows 1
# and 2 need no such flag because --fixed off already floats every axis.
#
# ROW 5 IS A ZERO CONTROL. Four rows that must be IDENTICAL are exactly the
# shape that passes when the runner is broken - a --dump-all-logits that
# never varies, a flag the binary ignores, two cmp calls against the same
# file. Row 5 requires the ordinary build's --fixed off to DIFFER from its
# own shipping default, i.e. that this instrument can see a difference at
# all. Without it rows 1-4 could be vacuous from birth.
#
# Builds are derived from `make -Bn llama98` and written to a temp dir, so
# this gate never touches build/native/ and cannot leave differently-flagged
# objects behind for the next build to link.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP (toolchain or model fixture missing)
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
MAKE=${MAKE:-make}

if ! command -v gcc >/dev/null 2>&1; then
    echo "recur_write_float_arm_gate: SKIP - gcc not found"
    exit 2
fi
MODELS_ROOT=${MODELS_ROOT:-/e/LLM/models}
MODEL=${LZ_NORM_INT_GATE_MODEL:-$MODELS_ROOT/kmr20}
if [ ! -d "$MODEL" ]; then
    echo "recur_write_float_arm_gate: SKIP - model not found: $MODEL" \
         "(set LZ_NORM_INT_GATE_MODEL or MODELS_ROOT)"
    exit 2
fi
TOK_ZH="$ROOT/build/norm_int_gate_tokens_zh.txt"
TOK_EN="$ROOT/build/norm_int_gate_tokens_en.txt"
for f in "$TOK_ZH" "$TOK_EN"; do
    if [ ! -f "$f" ]; then
        echo "recur_write_float_arm_gate: SKIP - fixture missing: $f"
        exit 2
    fi
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ---- two builds of the SAME working tree, from make's own recipe --------
# Every gcc line of `make -Bn llama98`, with build/native/ redirected into
# $WORK and the extra define appended. Taking the recipe from make rather
# than restating the source list means a new TU cannot silently drop out of
# what this gate compiles.
build_one() {
    local name="$1" extra="$2" line rc
    "$MAKE" -Bn llama98 2>/dev/null | grep '^gcc ' > "$WORK/$name.cmds"
    if [ ! -s "$WORK/$name.cmds" ]; then
        echo "recur_write_float_arm_gate: ERROR - 'make -Bn llama98' produced no gcc lines"
        return 2
    fi
    while IFS= read -r line; do
        line=${line//build\/native\//$WORK/}
        line=${line/ -o llama98 / -o $WORK/$name.exe }
        eval "$line $extra" >>"$WORK/$name.build.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "recur_write_float_arm_gate: ERROR - build '$name' failed"
            tail -n 15 "$WORK/$name.build.log"
            return 2
        fi
    done < "$WORK/$name.cmds"
    if [ ! -x "$WORK/$name.exe" ]; then
        echo "recur_write_float_arm_gate: ERROR - build '$name' produced no binary"
        return 2
    fi
    return 0
}

echo "-- build: ordinary --"
build_one on "" || exit 2
echo "-- build: -DLZ_GDN_FIXED=0 --"
build_one off "-DLZ_GDN_FIXED=0" || exit 2

FAIL=0
# run <binary-tag> <outfile-tag> <corpus> <flags...>
run_dump() {
    local bin="$1" tag="$2" tokfile="$3"; shift 3
    if ! "$WORK/$bin.exe" "$MODEL" --tokens-file "$tokfile" "$@" \
            --dump-all-logits "$WORK/$tag.bin" >"$WORK/$tag.log" 2>&1; then
        echo "FAIL: run $tag failed:"; tail -n 8 "$WORK/$tag.log"; FAIL=1; return 1
    fi
    if [ ! -s "$WORK/$tag.bin" ]; then
        echo "FAIL: run $tag produced an empty dump"; FAIL=1; return 1
    fi
    return 0
}

# check_same/check_diff take two tags already produced by run_dump. Both
# refuse to answer when a file is missing rather than reporting the cmp
# result for a file that was never written.
check_same() {
    local label="$1" a="$2" b="$3"
    if [ ! -s "$WORK/$a.bin" ] || [ ! -s "$WORK/$b.bin" ]; then
        echo "FAIL: $label - dump missing, nothing was compared"; FAIL=1; return
    fi
    if cmp -s "$WORK/$a.bin" "$WORK/$b.bin"; then
        echo "PASS: $label - identical ($(wc -c < "$WORK/$a.bin") bytes)"
    else
        echo "FAIL: $label - differ ($(cmp -l "$WORK/$a.bin" "$WORK/$b.bin" | wc -l)" \
             "of $(wc -c < "$WORK/$a.bin") bytes)"
        FAIL=1
    fi
}
check_diff() {
    local label="$1" a="$2" b="$3"
    if [ ! -s "$WORK/$a.bin" ] || [ ! -s "$WORK/$b.bin" ]; then
        echo "FAIL: $label - dump missing, nothing was compared"; FAIL=1; return
    fi
    if cmp -s "$WORK/$a.bin" "$WORK/$b.bin"; then
        echo "FAIL: $label - identical, so this instrument cannot see a" \
             "difference and the rows above prove nothing"
        FAIL=1
    else
        echo "PASS: $label - differ, as they must"
    fi
}

echo "-- rows 1-2: --fixed off, runtime-off vs compile-time-off --"
run_dump on  fo_on_zh  "$TOK_ZH" --fixed off
run_dump off fo_off_zh "$TOK_ZH" --fixed off
check_same "--fixed off / zh" fo_on_zh fo_off_zh
run_dump on  fo_on_en  "$TOK_EN" --fixed off
run_dump off fo_off_en "$TOK_EN" --fixed off
check_same "--fixed off / en" fo_on_en fo_off_en

echo "-- rows 3-4: --recur-write float --norm-write float, same pair --"
run_dump on  rw_on_zh  "$TOK_ZH" --recur-write float --norm-write float
run_dump off rw_off_zh "$TOK_ZH" --recur-write float --norm-write float
check_same "--recur-write float --norm-write float / zh" rw_on_zh rw_off_zh
run_dump on  rw_on_en  "$TOK_EN" --recur-write float --norm-write float
run_dump off rw_off_en "$TOK_EN" --recur-write float --norm-write float
check_same "--recur-write float --norm-write float / en" rw_on_en rw_off_en

echo "-- row 5: zero control --"
run_dump on def_on_zh "$TOK_ZH"
check_diff "ordinary build: --fixed off vs shipping default / zh" fo_on_zh def_on_zh

if [ "$FAIL" -ne 0 ]; then
    echo "recur_write_float_arm_gate: FAIL"
    exit 1
fi
echo "recur_write_float_arm_gate: PASS"
exit 0
