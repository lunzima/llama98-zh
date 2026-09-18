#!/bin/bash
# lz_kv_share_target vs CED's decoder region.
#
# Same collision the training side found and fixed first, in the
# separate E:\LLM\llama98-zh tree - a tree that shares no git history
# with this one, so its fix does not reach this file by any mechanism
# except being written again here: attn_kv_share claimed every
# full-attention layer past the shared source as a valid target, with
# no exception for a layer inside CED's decoder region, where report
# Sec2.2 Eq(1) is unconditional and attn_kv_share (a project-invented
# generalization, not an Eq(1) rule) has no basis to override it.
#
# No model dir needed - build/kv_share_ced_boundary_probe.c builds its
# own LZModelConfig fixture and calls the real lz_kv_share_src/
# lz_kv_share_target straight out of src/model.c - so this sits with
# the other fast, build-and-run gates (json-offset-gate, mm-alias-gate)
# rather than the model-dir-gated ones (kv-share-gate, ced-gate).
#
# THE CONTROL is a MUTATION, not a git-history diff, for the same
# reason json_offset_gate.sh gives: a git-ref-pinned "before" copy goes
# stale the day this fix lands and HEAD catches up to the working tree.
# The mutation deletes the fix's own if-block (the ced_split check and
# its return 0) from a scratch copy of model.c and re-runs the SAME
# probe against it. Per the probe's own three-row design:
#   row 1 (ced off)                must stay PASS - untouched by the fix
#   row 2 (ced on, split=1)        must turn FAIL - this is what the fix guards
#   row 3 (ced_split == n_layers)  must stay PASS - never enters the
#                                   deleted branch either way
# A mutant where row 2 stays PASS means the probe cannot see this
# defect at all. A mutant where row 3 also turns FAIL means the probe's
# row 3 is not independent of row 2's branch, which is a probe defect
# in its own right - reported as a gate FAILURE either way, not folded
# into "the mutation worked".
#
# Every source file is copied into the scratch dir (not just model.c/
# model.h) for the same quote-include resolution reason json_offset_
# gate.sh documents: `#include "model.h"` resolves against model.c's
# own directory before -I is ever consulted, so a mixed-directory build
# would silently link a fixed model.o against a stale model.h or vice
# versa.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "kv_share_ced_boundary_gate: SKIP - no C compiler"
    exit 2
}

# The real function this gate protects, linked whole (not transcribed)
# from src/model.c - see this project's own note on wfwd.c/wconv2.c:
# measuring the engine, not a copy of it. This exact file set is
# json_offset_gate.sh's own proven link line for standing model.c up
# on its own outside the full engine build.
ENGX="err compat lfn json safetensors model ops_sched cpucheck lz4d"
ANCHOR='if (c->ced_split > 0 && c->ced_split < c->n_layers && li >= c->ced_split)'

W=$(mktemp -d 2>/dev/null || echo /tmp/kvcedgate.$$)
mkdir -p "$W/fixed" "$W/mutant"
trap 'rm -rf "$W"' EXIT

cp src/*.c src/*.h "$W/fixed/" 2>/dev/null
cp src/*.c src/*.h "$W/mutant/" 2>/dev/null

N=$(grep -Fc "$ANCHOR" "$W/fixed/model.c")
if [ "$N" != "1" ]; then
    echo "kv_share_ced_boundary_gate: FAIL - expected exactly 1 occurrence" \
         "of the fix's anchor line in model.c, found $N (the gate no longer" \
         "matches the source)"
    exit 1
fi
sed -i "/$(printf '%s' "$ANCHOR" | sed 's/[.[\*^$/]/\\&/g')/,+1d" "$W/mutant/model.c"
if grep -qF "$ANCHOR" "$W/mutant/model.c"; then
    echo "kv_share_ced_boundary_gate: FAIL - could not delete the ced_split" \
         "exclusion from the mutant's model.c"
    exit 1
fi

RC=0

build_and_run() {
    local dir="$1" label="$2"
    if ! $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            build/kv_share_ced_boundary_probe.c -lm 2>"$dir/cc.log"; then
        echo "kv_share_ced_boundary_gate: FAIL - probe did not build against $label"
        tail -n 20 "$dir/cc.log"
        return 2
    fi
    "$dir/probe" > "$dir/run.log" 2>&1
    local rc=$?
    cat "$dir/run.log"
    return $rc
}

echo "-- current tree (all three rows must PASS) --"
build_and_run "$W/fixed" "the current (fixed) tree"
FIXED_RC=$?
if [ "$FIXED_RC" -eq 2 ]; then
    exit 1
elif [ "$FIXED_RC" -ne 0 ]; then
    echo "kv_share_ced_boundary_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi

echo ""
echo "-- mutant tree: ced_split exclusion deleted (row 2 only must FAIL) --"
build_and_run "$W/mutant" "the ced_split-exclusion mutant"
MUT_RC=$?
if [ "$MUT_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT_RC" -eq 0 ]; then
    echo "kv_share_ced_boundary_gate: FAIL - the mutant PASSED; the probe" \
         "cannot see this defect"
    RC=1
else
    ROWS_FAILED=$(grep -c '^FAIL' "$W/mutant/run.log")
    if [ "$ROWS_FAILED" -ne 1 ] || ! grep -q '^FAIL ced on (split=1)' "$W/mutant/run.log"; then
        echo "kv_share_ced_boundary_gate: FAIL - the mutant failed $ROWS_FAILED" \
             "row(s) instead of exactly row 2 (ced on, split=1) - the probe's" \
             "rows are not independent"
        RC=1
    else
        echo "-- mutant correctly failed exactly row 2, rows 1 and 3 stayed PASS --"
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "kv_share_ced_boundary_gate: FAIL"
    exit 1
fi
echo "kv_share_ced_boundary_gate: PASS"
exit 0
