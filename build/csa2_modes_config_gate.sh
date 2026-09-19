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
# THE CONTROL is TWO INDEPENDENT MUTATIONS, not a git-history diff, for
# the same reason kv_share_ced_boundary_gate.sh gives: a git-ref-pinned
# "before" copy goes stale the day this fix lands and HEAD catches up
# to the working tree.
#
# MUTANT 1 disables lz_load_config's own call into
# lz_csa2_last_writer_valid (a REPORT SEC2.3.1 code-review addendum
# landed after this gate itself: the narrower, explicit "a REUSE
# layer's source must not itself be REUSE mode" check this gate
# originally targeted was REPLACED by that whole-config predicate,
# which subsumes it - model.c's own comment on lz_csa2_last_writer_valid
# has the proof: a REUSE layer never becomes anyone's last writer, so
# any layer naming one as its own source always fails the same way.
# This gate's own anchor and mutation moved with it, in the same
# commit, rather than being left to test code that no longer exists) in
# a scratch copy of model.c and re-runs the SAME probe against it. Per
# the probe's own case design:
#   cases 1-5, 7a, 7b, 8, 11   must stay PASS - untouched by this check
#   case 6 (reuse-of-reuse) must turn FAIL - this is what the check guards
#
# MUTANT 2 (I3, fifth review round: this refusal previously had NO gate
# at all anywhere in the repo - a reviewer had to verify it by hand)
# disables lz_load_config's own S4+CSA2 refusal (the `if (c->
# mla_q_lora_rank > 0)` block in the csa2_layers loop, model.c). Only
# case 11's own two lines may turn FAIL.
#
# A mutant where its own target case stays PASS means the probe cannot
# see that defect at all. A mutant where any OTHER case also turns FAIL
# means the probe's cases are not independent, which is a probe defect
# in its own right - reported as a gate FAILURE either way, not folded
# into "the mutation worked".
#
# Every source file is copied into the scratch dir (not just model.c/
# model.h), same quote-include resolution reason json_offset_gate.sh
# documents: `#include "model.h"` resolves against model.c's own
# directory before -I is ever consulted.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler, no python3).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_modes_config_gate: SKIP - no C compiler"
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_modes_config_gate: SKIP - no python3"
    exit 2
}

# The real function this gate protects, linked whole (not transcribed)
# from src/model.c - json_offset_gate.sh's own proven link line for
# standing model.c up on its own outside the full engine build.
ENGX="err compat lfn json safetensors model ops_sched cpucheck lz4d"
ANCHOR1='        if (!lz_csa2_last_writer_valid(c, &bad)) {'
ANCHOR2_OLD="            if (c->mla_q_lora_rank > 0) {
                qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_MLA, idx);
                lz_json_free(&j);
                return 1;
            }"
ANCHOR2_NEW="            if (0) {
                qerr(errbuf, errlen, LZ_ERR_CFG_CSA2_MLA, idx);
                lz_json_free(&j);
                return 1;
            }"

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2modesgate.$$)
mkdir -p "$W/fixed" "$W/mut1" "$W/mut2"
trap 'rm -rf "$W"' EXIT

cp src/*.c src/*.h "$W/fixed/" 2>/dev/null
cp src/*.c src/*.h "$W/mut1/" 2>/dev/null
cp src/*.c src/*.h "$W/mut2/" 2>/dev/null

N=$(grep -Fc "$ANCHOR1" "$W/fixed/model.c")
if [ "$N" != "1" ]; then
    echo "csa2_modes_config_gate: FAIL - expected exactly 1 occurrence" \
         "of the last-writer check's anchor line in model.c, found $N" \
         "(the gate no longer matches the source)"
    exit 1
fi
python3 - "$W/mut1/model.c" "$ANCHOR1" <<'PYEOF' || exit 1
import sys
path, anchor = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = anchor + "\n"
new = "        if (0) {\n"
if src.count(old) != 1:
    print("csa2_modes_config_gate: FAIL - could not find exactly one copy "
          "of the last-writer check to mutate", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF
if grep -qF "$ANCHOR1" "$W/mut1/model.c"; then
    echo "csa2_modes_config_gate: FAIL - could not disable the last-writer" \
         "check in mutant 1's model.c"
    exit 1
fi

python3 - "$W/mut2/model.c" <<PYEOF || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = """$ANCHOR2_OLD"""
new = """$ANCHOR2_NEW"""
if src.count(old) != 1:
    print("csa2_modes_config_gate: FAIL - could not find exactly one copy "
          "of the S4+CSA2 refusal to mutate", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
if old in src:
    print("csa2_modes_config_gate: FAIL - a second copy of the S4+CSA2 "
          "refusal survived the replace (should be impossible after the "
          "count(old) != 1 check above)", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF
# No separate bash-level grep verification here, unlike mutant 1's own
# post-check just above: ANCHOR2_OLD/ANCHOR2_NEW are MULTI-LINE strings,
# and `grep -F` given one as a single argument does not search for the
# literal multi-line block - it splits on the embedded newlines and
# matches if ANY single resulting line (e.g. bare "return 1;" or "}",
# both common throughout model.c) appears ANYWHERE in the file, which
# is true almost by construction and would make this check pass no
# matter what the mutation actually did. The python script's own exact,
# whole-string count(old) (before) and `old in src` (after) checks
# above are what actually verify this mutation - not a redundant bash
# recheck that cannot express the same question correctly.

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
echo "-- mutant 1: reuse-of-reuse check deleted (case 6 only must FAIL) --"
build_and_run "$W/mut1" "the reuse-of-reuse-check mutant"
MUT1_RC=$?
if [ "$MUT1_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT1_RC" -eq 0 ]; then
    echo "csa2_modes_config_gate: FAIL - mutant 1 PASSED; the probe" \
         "cannot see this defect"
    RC=1
else
    ROWS_FAILED=$(grep -c '^FAIL' "$W/mut1/run.log")
    if [ "$ROWS_FAILED" -ne 2 ] || \
       ! grep -q '^FAIL case 6: reuse-of-reuse is refused' "$W/mut1/run.log" || \
       ! grep -q '^FAIL case 6: refusal names csa2_kv_source' "$W/mut1/run.log"; then
        echo "csa2_modes_config_gate: FAIL - mutant 1 failed $ROWS_FAILED" \
             "row(s) instead of exactly case 6's two assertions - the" \
             "probe's cases are not independent"
        RC=1
    else
        echo "-- mutant 1 correctly failed exactly case 6, every other case stayed PASS --"
    fi
fi

echo ""
echo "-- mutant 2 (I3): S4+CSA2 refusal deleted (case 11 only must FAIL) --"
build_and_run "$W/mut2" "the S4+CSA2-refusal mutant"
MUT2_RC=$?
if [ "$MUT2_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT2_RC" -eq 0 ]; then
    echo "csa2_modes_config_gate: FAIL - mutant 2 PASSED; the probe" \
         "cannot see this defect"
    RC=1
else
    ROWS_FAILED=$(grep -c '^FAIL' "$W/mut2/run.log")
    if [ "$ROWS_FAILED" -ne 2 ] || \
       ! grep -q '^FAIL case 11: mla_q_lora_rank>0 with a csa2_on layer' "$W/mut2/run.log" || \
       ! grep -q '^FAIL case 11: refusal names the csa2 layer' "$W/mut2/run.log"; then
        echo "csa2_modes_config_gate: FAIL - mutant 2 failed $ROWS_FAILED" \
             "row(s) instead of exactly case 11's two assertions - the" \
             "probe's cases are not independent"
        RC=1
    else
        echo "-- mutant 2 correctly failed exactly case 11, every other case stayed PASS --"
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_modes_config_gate: FAIL"
    exit 1
fi
echo "csa2_modes_config_gate: PASS"
exit 0
