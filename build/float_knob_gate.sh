#!/bin/bash
# One float knob, and the retired ones stay retired.
#
# --fixed off|all is the whole float axis. It replaced seven per-tier
# flags - --recur-write, --conv-write, --attn-write, --epi-write,
# --sig-write, --norm-write, --scalar-fixed - each with its own global
# and selector, because seven axes could only buy MIXED settings and
# mixing measured worse than either end (the paired NLL is at
# lz_float_ref_select in ops_sched.c).
#
# WHY A GATE AND NOT A MEMORY. A retired flag comes back by accident:
# somebody re-adds a parse arm for one tier while chasing a bug, and
# nothing says a word - the binary still runs, the logits still match,
# and the axis count has gone from one to two. Nothing else here counts
# the knobs.
#
# WHAT IS ASSERTED
#   1. --fixed off and --fixed all are both accepted.
#   2. Each of the seven retired names is REJECTED.
#   3. --pair and --prefetch are still accepted. They are performance
#      knobs, not FPU tiers, and the requirement keeps them; asserting
#      it here stops this gate being read as "no knobs at all".
#   4. ops_sched.c defines exactly one float-selection global.
#
# Exit codes:
#   0  PASS
#   1  FAIL - a retired knob answered, a kept one did not, or the
#      rejection detector itself is broken
#   2  SKIP - no binary or no model
set -u
cd "$(dirname "$0")/.."

BIN=./llama98.exe
[ -x "$BIN" ] || BIN=./llama98
[ -x "$BIN" ] || { echo "float_knob_gate: SKIP - no llama98; make native first"; exit 2; }

MODEL=${1:-/e/LLM/models/kmr20}
[ -d "$MODEL" ] || { echo "float_knob_gate: SKIP - no $MODEL"; exit 2; }

RETIRED="--recur-write --conv-write --attn-write --epi-write --sig-write --norm-write --scalar-fixed"
KEPT="--pair --prefetch"

# Is this option unknown to the binary? One place, so the two loops
# below cannot drift into asking different questions.
rejected() {
    "$BIN" "$MODEL" "$1" "$2" --check 2>&1 \
        | grep -qiE "unknown option|unrecognis|unrecogniz"
}

FAIL=0

# Control FIRST, and it is the one that matters: if the phrase this
# grep looks for ever changes, every retired knob reads as rejected and
# the gate passes while measuring nothing. A name that cannot exist
# must be reported as unknown.
if ! rejected --zzz-not-a-real-flag off; then
    echo "FAIL: the binary does not report --zzz-not-a-real-flag as an"
    echo "  unknown option, so this gate cannot tell a retired knob from"
    echo "  a live one and its PASS would mean nothing"
    exit 1
fi
# And the inverse control: a name that DOES exist must not be reported
# as unknown, or everything would look retired.
if rejected --fixed all; then
    echo "FAIL: --fixed all reads as an unknown option; the detector is"
    echo "  matching something other than the rejection message"
    exit 1
fi

for k in $RETIRED; do
    if ! rejected "$k" off; then
        echo "FAIL: $k is still accepted - the float axis is not one knob"
        FAIL=1
    fi
done

for v in off all; do
    if rejected --fixed "$v"; then
        echo "FAIL: --fixed $v is not accepted"
        FAIL=1
    fi
done

for k in $KEPT; do
    a=off
    [ "$k" = "--prefetch" ] && a=none
    if rejected "$k" "$a"; then
        echo "FAIL: $k was rejected; it is a performance knob and stays"
        FAIL=1
    fi
done

# The source side. One global, so a second axis cannot be added without
# this number moving.
N=$(grep -cE "^int lz_float_ref_g = " src/ops_sched.c)
if [ "$N" -ne 1 ]; then
    echo "FAIL: ops_sched.c defines $N float-selection global(s), expected 1"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo "float_knob_gate: PASS (1 axis, 7 retired names rejected,"
    echo "  2 performance knobs kept)"
    exit 0
fi
echo "float_knob_gate: FAIL"
exit 1
