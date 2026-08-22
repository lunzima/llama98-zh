#!/bin/bash
# Performance-knob invariance gate.
#
#   perf_knob_gate.sh <model-dir> [tokens]
#
# --pair and --prefetch are the two knobs in this engine that are PURELY
# about speed: pairing picks how many tokens a row kernel folds together
# and prefetch picks which cache hint the MMX kernels emit. Neither is a
# tier - there is no float arm and no fixed arm - so the whole of their
# contract is "the output does not change". Nothing asserted that.
#
# WHY IT IS WORTH A GATE RATHER THAN A COMMENT. Every other knob in the
# engine is a TIER, and a tier that changes the numbers is doing its job;
# a gate for those has to know what the right numbers are. These two are
# the opposite: any difference at all is a bug, so the assertion is free
# to write and impossible to argue with. That is also why they are the
# two easiest knobs to break silently - a kernel selected by --pair that
# rounds differently from its sibling produces plausible output, and the
# tier gates would never look at it. build/ab_control_builds have that
# defect class on record.
#
# --kv-rot is deliberately NOT here. It is a real tier (it rotates Q/K/V
# before the KV cache quantizes them, so the numbers MUST move) and it
# has no gate of its own; asserting invariance for it would be asserting
# something false. That gap is recorded in docs/arm-asm-audit.md rather
# than papered over here.
#
# Exit: 0 PASS, 1 FAIL, 2 SKIP (no binary or no model).
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
TOKENS=${2:-1,2,3,4,5,6,7,8}

BIN=${BIN:-}
if [ -z "$BIN" ]; then
    for c in "$ROOT/llama98.exe" "$ROOT/llama98"; do
        [ -x "$c" ] && BIN=$c && break
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "perf_knob_gate: SKIP - no native binary (make native, or set BIN)"
    exit 2
fi
if [ -z "$MODEL" ] || [ ! -d "$MODEL" ]; then
    echo "perf_knob_gate: SKIP - model dir not given or not found: '$MODEL'"
    exit 2
fi

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

run() {   # run TAG ARGS...
    local tag="$1"; shift
    if ! "$BIN" "$MODEL" --tokens "$TOKENS" \
            --dump-logits "$W/$tag.bin" "$@" >"$W/$tag.log" 2>&1; then
        echo "FAIL: run '$tag' exited nonzero"
        tail -5 "$W/$tag.log"
        return 1
    fi
    [ -s "$W/$tag.bin" ] || { echo "FAIL: run '$tag' wrote no logits"; return 1; }
    return 0
}

run base || exit 1
BASE_SZ=$(wc -c < "$W/base.bin")
echo "perf_knob_gate: baseline $BASE_SZ bytes"

FAILED=0
# Every value the SELECTORS accept, taken from lz_pair_select and
# lz_prefetch_select in src/ops_sched.c rather than from the help text.
# The first draft of this list carried a `--prefetch t0` that does not
# exist; it FAILED rather than being skipped, which is the behaviour
# below that matters - a value the parser rejects means either the
# parser or this list is wrong, and both are bugs worth stopping for.
for K in "--pair off" "--pair on" "--pair auto" \
         "--prefetch none" "--prefetch load" "--prefetch nta" \
         "--prefetch 3dnow" "--prefetch auto"; do
    TAG=$(echo "$K" | tr -d ' -')
    if ! run "$TAG" $K; then
        # A value this build does not accept is a FAIL, not a skip: the
        # help text lists it, so either the parser or the help is wrong.
        echo "FAIL: '$K' did not run"
        FAILED=1
        continue
    fi
    if cmp -s "$W/base.bin" "$W/$TAG.bin"; then
        echo "  PASS  $K"
    else
        N=$(cmp -l "$W/base.bin" "$W/$TAG.bin" | wc -l)
        echo "  FAIL  $K changed the output ($N of $BASE_SZ bytes)"
        FAILED=1
    fi
done

[ $FAILED -eq 0 ] || { echo "perf_knob_gate: FAIL"; exit 1; }
echo "perf_knob_gate: PASS"
exit 0
