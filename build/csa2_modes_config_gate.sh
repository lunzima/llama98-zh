#!/bin/bash
# config.json schema/validation for csa2_kv_mode/csa2_kv_source - the
# engine-side half of DeepSeek-V4.1 report Sec2.3.1's Full/Reindex/
# Reuse cross-layer KV reuse axis (model.h's own csa2_mode[]/
# csa2_source[] comment on LZModelConfig). The training side already
# landed the three-mode forward pass in a DIFFERENT repository
# (E:\LLM\llama98-zh, KunMoeConfig.csa2_kv_mode/csa2_kv_source,
# engine/csa2.py) - that tree shares no git history with this one, so
# its validation does not reach this file by any mechanism except being
# written again here.
#
# No model dir needed - build/csa2_modes_config_probe.c writes its own
# synthetic config.json files (a 6-layer fixture, no safetensors) and
# calls the real lz_load_config straight out of src/model.c - so this
# sits with the other fast, build-and-run gates (kv-share-ced-boundary-
# gate, ced-proj-arm-gate) rather than the model-dir-gated ones.
#
# THE CONTROL is a MUTATION, not a git-history diff, for the same
# reason kv_share_ced_boundary_gate.sh gives: a git-ref-pinned "before"
# copy goes stale the day this fix lands and HEAD catches up to the
# working tree. The mutation deletes the cross-check that a REUSE
# layer's source must not itself be REUSE mode (the closing bracket of
# lz_load_config's csa2_kv_mode/csa2_kv_source cross-check loop, second
# `if` block) from a scratch copy of model.c and re-runs the SAME probe
# against it. Per the probe's own case design:
#   cases 1-5, 7a, 7b, 8   must stay PASS - untouched by this check
#   case 6 (reuse-of-reuse) must turn FAIL - this is what the check guards
# A mutant where case 6 stays PASS means the probe cannot see this
# defect at all. A mutant where any other case also turns FAIL means the
# probe's cases are not independent, which is a probe defect in its own
# right - reported as a gate FAILURE either way, not folded into "the
# mutation worked".
#
# Every source file is copied into the scratch dir (not just model.c/
# model.h), same quote-include resolution reason json_offset_gate.sh
# documents: `#include "model.h"` resolves against model.c's own
# directory before -I is ever consulted.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_modes_config_gate: SKIP - no C compiler"
    exit 2
}

# The real function this gate protects, linked whole (not transcribed)
# from src/model.c - json_offset_gate.sh's own proven link line for
# standing model.c up on its own outside the full engine build.
ENGX="err compat lfn json safetensors model ops_sched cpucheck lz4d"
ANCHOR='        if (c->csa2_mode[i] == LZ_CSA2_REUSE &&'

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2modesgate.$$)
mkdir -p "$W/fixed" "$W/mutant"
trap 'rm -rf "$W"' EXIT

cp src/*.c src/*.h "$W/fixed/" 2>/dev/null
cp src/*.c src/*.h "$W/mutant/" 2>/dev/null

N=$(grep -Fc "$ANCHOR" "$W/fixed/model.c")
if [ "$N" != "1" ]; then
    echo "csa2_modes_config_gate: FAIL - expected exactly 1 occurrence" \
         "of the reuse-of-reuse check's anchor line in model.c, found $N" \
         "(the gate no longer matches the source)"
    exit 1
fi
# The anchor line plus the six that complete the if-block (condition's
# second line + brace, qerr's two lines, lz_json_free, return, closing
# brace) - seven lines total.
sed -i "/$(printf '%s' "$ANCHOR" | sed 's/[.[\*^$/]/\\&/g')/,+6d" "$W/mutant/model.c"
if grep -qF "$ANCHOR" "$W/mutant/model.c"; then
    echo "csa2_modes_config_gate: FAIL - could not delete the reuse-of-reuse" \
         "check from the mutant's model.c"
    exit 1
fi

RC=0

build_and_run() {
    local dir="$1" label="$2"
    if ! $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            build/csa2_modes_config_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_modes_config_gate: FAIL - probe did not build against $label"
        tail -n 20 "$dir/cc.log"
        return 2
    fi
    ( cd "$dir" && ./probe > run.log 2>&1 )
    local rc=$?
    cat "$dir/run.log"
    return $rc
}

echo "-- current tree (all cases must PASS) --"
build_and_run "$W/fixed" "the current (fixed) tree"
FIXED_RC=$?
if [ "$FIXED_RC" -eq 2 ]; then
    exit 1
elif [ "$FIXED_RC" -ne 0 ]; then
    echo "csa2_modes_config_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi

echo ""
echo "-- mutant tree: reuse-of-reuse check deleted (case 6 only must FAIL) --"
build_and_run "$W/mutant" "the reuse-of-reuse-check mutant"
MUT_RC=$?
if [ "$MUT_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT_RC" -eq 0 ]; then
    echo "csa2_modes_config_gate: FAIL - the mutant PASSED; the probe" \
         "cannot see this defect"
    RC=1
else
    ROWS_FAILED=$(grep -c '^FAIL' "$W/mutant/run.log")
    if [ "$ROWS_FAILED" -ne 2 ] || \
       ! grep -q '^FAIL case 6: reuse-of-reuse is refused' "$W/mutant/run.log" || \
       ! grep -q '^FAIL case 6: refusal names csa2_kv_source' "$W/mutant/run.log"; then
        echo "csa2_modes_config_gate: FAIL - the mutant failed $ROWS_FAILED" \
             "row(s) instead of exactly case 6's two assertions - the" \
             "probe's cases are not independent"
        RC=1
    else
        echo "-- mutant correctly failed exactly case 6, every other case stayed PASS --"
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_modes_config_gate: FAIL"
    exit 1
fi
echo "csa2_modes_config_gate: PASS"
exit 0
