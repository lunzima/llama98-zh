#!/bin/bash
# Every gate has a caller.
#
# A gate is three things: an implementation, a caller, and a mutation
# proving it can fail. This repo keeps re-learning the middle one. Six
# gate scripts were written, committed, mutation-tested, and depended on
# by no target - c89_floor, comment_hygiene, vc4_limits, dup_body,
# twin_pair and gcc_watcom_pair. All six passed every time anyone ran
# them by hand, and none could have reddened a build.
#
# Two of those six are the criteria for standing project requirements:
# every x86 kernel written for both toolchains (twin_pair), and no
# duplicated helper bodies (dup_body). Those requirements had nothing
# checking them.
#
# Wiring them was a one-off sweep. This makes the sweep a gate, because
# the seventh unwired script will look exactly like the first six: green
# when run by hand, invisible otherwise.
#
# WHAT COUNTS AS A CALLER: a mention in the Makefile or in ci.sh. Not a
# mention in another gate - that would let two unwired gates cite each
# other. Scripts under build/arm, build/mips and build/x87 are exempt
# and listed below: they are build helpers their sibling gates invoke,
# and the sibling gates are themselves checked here.
#
# Exit codes:
#   0  PASS
#   1  FAIL - a gate script no target depends on
set -u
cd "$(dirname "$0")/.."

# Helpers invoked by a sibling gate rather than by a target. Each is
# named with the gate that runs it, so an entry that stops being true
# stands out.
#   arm/build_engine.sh, arm/run_qemu.sh   <- arm/parity_gate.sh
#   mips/build_engine.sh                   <- mips/endian_gate.sh
EXEMPT="arm/build_engine.sh arm/run_qemu.sh mips/build_engine.sh"

FAIL=0
UNWIRED=""
for g in build/*_gate.sh; do
    b=$(basename "$g")
    [ "$b" = "gate_wiring_gate.sh" ] && continue
    if ! grep -q "$b" Makefile build/ci.sh 2>/dev/null; then
        UNWIRED="$UNWIRED $b"
        FAIL=1
    fi
done

for g in build/arm/*.sh build/mips/*.sh build/x87/*.sh; do
    [ -f "$g" ] || continue
    rel=${g#build/}
    case " $EXEMPT " in *" $rel "*) continue ;; esac
    if ! grep -q "$rel" Makefile build/ci.sh 2>/dev/null; then
        UNWIRED="$UNWIRED $rel"
        FAIL=1
    fi
done

# Positive control. If the search stopped matching, every script would
# look unwired rather than wired - so the control is a name that MUST be
# found, not one that must not.
if ! grep -q "kernel_matrix_gate.sh" Makefile build/ci.sh 2>/dev/null; then
    echo "gate_wiring_gate: FAIL - kernel_matrix_gate.sh reads as unwired,"
    echo "  which it is not; the search is broken and every PASS this gate"
    echo "  gives is meaningless"
    exit 1
fi

N=$(ls build/*_gate.sh 2>/dev/null | wc -l)
if [ "$FAIL" -eq 0 ]; then
    echo "gate_wiring_gate: PASS ($N gate script(s), every one reachable"
    echo "  from make or ci.sh)"
    exit 0
fi
echo "gate_wiring_gate: FAIL - script(s) no target depends on:"
for u in $UNWIRED; do echo "    $u"; done
echo "  A gate nothing invokes is a script. Add it to \`check\` in the"
echo "  Makefile, or to ci.sh, or delete it."
exit 1
