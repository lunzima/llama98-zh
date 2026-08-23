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
# three times - ordinary, -DLZ_GDN_FIXED=0, -DLZ_NORM_FIXED=0 - and require
# byte-identical --dump-all-logits. With the pass-1 defect present rows 1-2
# and 5-6 differ; the difference is the whole recurrence's pass-1
# contraction.
#
# ONE SWITCH, ONE AXIS. Each -D here must remove exactly the tier it
# names, so each row pairs a compile-time off switch with the runtime
# flag for that same tier and nothing else. -DLZ_GDN_FIXED=0 used to
# compile out the fixed norms as well - norm_ss_fixed and ops_norm.c's
# three output loops sat inside its #if - so rows 5 and 6 had to pass
# --norm-write float on top of --recur-write float, and the norms tier
# had no control arm of its own at all. LZ_NORM_FIXED now guards the
# norms; --recur-write float alone matches the recurrence build, and
# rows 7-8 run the same argument on the norms axis.
#
# WHY --fixed off IS CHECKED AGAINST BOTH BUILDS (rows 1-2 AND 3-4). While
# the two tiers shared a macro, a --fixed off that reached the recurrence
# but not the norms showed up in rows 1-2, because the one -D build had
# neither tier. Decoupled, that build still has the fixed norms and rows
# 1-2 would pass such a --fixed off. Each compile-time off switch
# therefore gets its own --fixed off pair: the group setter has to be
# complete for every axis separately, not for their union.
#
# ROWS 9-10 ARE ZERO CONTROLS. Rows that must be IDENTICAL are exactly the
# shape that passes when the runner is broken - a --dump-all-logits that
# never varies, a flag the binary ignores, two cmp calls against the same
# file. They require each runtime flag to DIFFER from the shipping default
# on the ordinary build, i.e. that this instrument can see a difference at
# all. Without them rows 1-8 could be vacuous from birth.
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
echo "-- build: -DLZ_NORM_FIXED=0 --"
build_one nof "-DLZ_NORM_FIXED=0" || exit 2
echo "-- build: -DLZ_CONV_FIXED=0 --"
build_one cof "-DLZ_CONV_FIXED=0" || exit 2
echo "-- build: -DLZ_EPI_FIXED=0 --"
build_one eof "-DLZ_EPI_FIXED=0" || exit 2
# The exact-float-state scaffold. Built and RUN here, not asserted for
# byte-identity - it deliberately computes different numbers, it is the
# reference the int8 state is measured against. What it needs is a
# caller: it compiled, it linked, and it segfaulted on every KunMoe
# model, because gdn_step_impl grew an int16 `v` entry whose `v` is NULL
# and the scaffold read v[] anyway. Then, once it ran, it ignored gvec
# and reported +1.29 NLL at t=27.9 - the number a scaffold gives when it
# is measuring an undecayed recurrence. Neither a compile check nor the
# switch-matrix gate's all-at-once link can see either failure; only a
# run can, and this is the only gate that already has a model and the
# build machinery in the same place.
echo "-- build: -DLZ_GDN_STATE_F32=1 (exact-state scaffold) --"
build_one sf32 "-DLZ_GDN_STATE_F32=1" || exit 2

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

echo "-- rows 1-2: --fixed off, runtime-off vs -DLZ_GDN_FIXED=0 --"
run_dump on  fo_on_zh  "$TOK_ZH" --fixed off
run_dump off fo_off_zh "$TOK_ZH" --fixed off
check_same "--fixed off vs GDN compiled out / zh" fo_on_zh fo_off_zh
run_dump on  fo_on_en  "$TOK_EN" --fixed off
run_dump off fo_off_en "$TOK_EN" --fixed off
check_same "--fixed off vs GDN compiled out / en" fo_on_en fo_off_en

echo "-- rows 3-4: --fixed off, runtime-off vs -DLZ_NORM_FIXED=0 --"
run_dump nof fo_nof_zh "$TOK_ZH" --fixed off
check_same "--fixed off vs norms compiled out / zh" fo_on_zh fo_nof_zh
run_dump nof fo_nof_en "$TOK_EN" --fixed off
check_same "--fixed off vs norms compiled out / en" fo_on_en fo_nof_en

echo "-- rows 5-6: --fixed off, runtime-off vs -DLZ_CONV_FIXED=0 --"
run_dump cof fo_cof_zh "$TOK_ZH" --fixed off
check_same "--fixed off vs conv1d compiled out / zh" fo_on_zh fo_cof_zh
run_dump cof fo_cof_en "$TOK_EN" --fixed off
check_same "--fixed off vs conv1d compiled out / en" fo_on_en fo_cof_en

echo "-- rows 7-8: --fixed off, runtime-off vs -DLZ_EPI_FIXED=0 --"
run_dump eof fo_eof_zh "$TOK_ZH" --fixed off
check_same "--fixed off vs epilogue compiled out / zh" fo_on_zh fo_eof_zh
run_dump eof fo_eof_en "$TOK_EN" --fixed off
check_same "--fixed off vs epilogue compiled out / en" fo_on_en fo_eof_en

# ROWS 5-8 USED TO PAIR EACH -D WITH ITS OWN RUNTIME FLAG - --recur-write
# float against -DLZ_GDN_FIXED=0, --norm-write float against
# -DLZ_NORM_FIXED=0 - because "each -D removes exactly the tier it names"
# needed a runtime arm that named the same tier and nothing else. The
# seven per-tier flags are gone (ops.h, at lz_float_ref_select), so those
# pairings are unspellable and, with one knob, redundant: rows 1-4
# already run --fixed off against both builds. The rows were not
# deleted, they were SPENT - the two freed slots now run the same
# argument against the two compile switches that never had a row,
# LZ_CONV_FIXED and LZ_EPI_FIXED. Four compile switches covered, up from
# two.
#
# What the collapse does cost: a per-tier runtime arm was an independent
# way to reach one tier, so a --fixed off that missed exactly one axis
# used to be visible as "rows 1-2 pass, rows 5-6 fail". Now the only
# instrument for that is the compile switch itself, which is why every
# switch needs its own row and why adding a switch without adding one
# here leaves it uncovered.

echo "-- row 9: zero control --"
run_dump on def_on_zh "$TOK_ZH"
check_diff "ordinary build: --fixed off vs shipping default / zh" fo_on_zh def_on_zh

echo "-- rows 10-11: the exact-state scaffold runs, and differs --"
# run_dump already FAILs on a non-zero exit or an empty dump, which is
# the segfault check. check_diff against the shipping default is the
# second half: a scaffold that produced the SAME bytes as the int8 state
# would not be an independent reference, it would be the thing it is
# supposed to be measuring.
run_dump sf32 sf_zh "$TOK_ZH"
check_diff "LZ_GDN_STATE_F32=1 vs shipping default / zh" sf_zh def_on_zh
run_dump sf32 sf_en "$TOK_EN"
run_dump on   def_on_en "$TOK_EN"
check_diff "LZ_GDN_STATE_F32=1 vs shipping default / en" sf_en def_on_en

echo "-- row 12: the scaffold is NEAR the default, not merely different --"
# ROWS 10-11 CANNOT SEE THE SECOND DEFECT THIS SCAFFOLD HAD, and saying
# so is the point of this row. "Differs from the default" is true of a
# correct exact-state arm AND of the gvec-ignoring one that reported
# +1.29 NLL at t=27.9 - an undecayed recurrence differs too. The
# assertion that separates them is a BOUND: a better state cannot move
# NLL by more than a whisker, because it is the same recurrence with
# less rounding. Measured after the fix, over three models x two
# fixtures (n=3786): -4.1e-04. Broken: +1.29, four orders up.
#
# 0.05 is deliberately loose - three orders above the real value and
# still 25x below the defect it exists to catch. A tight bound here
# would go red on an ordinary model change; this one only goes red when
# the scaffold has stopped computing the recurrence.
SF_TOL=0.05
if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: row 12 - no python3 to average the NLL dumps"
else
    for fx in zh en; do
        tok=$TOK_ZH; [ "$fx" = en ] && tok=$TOK_EN
        "$WORK/sf32.exe" "$MODEL" --tokens-file "$tok" \
            --dump-nll "$WORK/sfn_$fx.bin" >/dev/null 2>&1
        "$WORK/on.exe" "$MODEL" --tokens-file "$tok" \
            --dump-nll "$WORK/den_$fx.bin" >/dev/null 2>&1
        if [ ! -s "$WORK/sfn_$fx.bin" ] || [ ! -s "$WORK/den_$fx.bin" ]; then
            echo "FAIL: row 12 / $fx - an NLL dump is missing"; FAIL=1; continue
        fi
        d=$(python3 -c "
import struct,sys
def rd(p):
    b=open(p,'rb').read(); return struct.unpack('<%df'%(len(b)//4),b)
a=rd(sys.argv[1]); b=rd(sys.argv[2])
print('%.6f'%(sum(x-y for x,y in zip(a,b))/len(a)))
" "$WORK/sfn_$fx.bin" "$WORK/den_$fx.bin")
        if python3 -c "import sys; sys.exit(0 if abs(float(sys.argv[1]))<=float(sys.argv[2]) else 1)" \
                "$d" "$SF_TOL"; then
            echo "PASS: exact-state NLL delta / $fx - $d, within +-$SF_TOL"
        else
            echo "FAIL: exact-state NLL delta / $fx - $d, outside +-$SF_TOL;" \
                 "the scaffold is not computing this recurrence"
            FAIL=1
        fi
    done
fi

if [ "$FAIL" -ne 0 ]; then
    echo "recur_write_float_arm_gate: FAIL"
    exit 1
fi
echo "recur_write_float_arm_gate: PASS"
exit 0
