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

# AND PRESENT IN A FRESH CLONE, which is the same question one layer
# out. .gitignore ignores /build/* and names each keeper back in by
# hand, so a new gate is wired and invisible at the same time: `make
# check` runs it here and the file does not exist for anyone who clones.
# dedupe_gate.sh already did this once - the .gitignore comment says so,
# and until now nothing but that comment held it.
UNTRACKED=""
for g in build/*_gate.sh build/*.awk; do
    [ -e "$g" ] || continue
    git ls-files --error-unmatch "$g" >/dev/null 2>&1 || UNTRACKED="$UNTRACKED $g"
done
if [ -n "$UNTRACKED" ]; then
    echo "gate_wiring_gate: FAIL - wired but not tracked:"
    for u in $UNTRACKED; do echo "    $u"; done
    echo "  .gitignore ignores /build/* by name. Add a \`!\` line for each,"
    echo "  or the gate runs here and is absent from every clone."
    FAIL=1
fi

# Positive control for that half, pointed the same way as the one above:
# a name that MUST come back tracked. If git stopped answering, every
# script would read as untracked and the FAIL would be the instrument.
if ! git ls-files --error-unmatch build/ci.sh >/dev/null 2>&1; then
    echo "gate_wiring_gate: FAIL - build/ci.sh reads as untracked, which it"
    echo "  is not; the tracking check is broken"
    exit 1
fi

N=$(ls build/*_gate.sh 2>/dev/null | wc -l)
if [ "$FAIL" -eq 0 ]; then
    echo "gate_wiring_gate: PASS ($N gate script(s), every one reachable"
    echo "  from make or ci.sh, and every one tracked)"
    exit 0
fi
[ -n "$UNWIRED" ] || exit 1
echo "gate_wiring_gate: FAIL - script(s) no target depends on:"
for u in $UNWIRED; do echo "    $u"; done
echo "  A gate nothing invokes is a script. Add it to \`check\` in the"
echo "  Makefile, or to ci.sh, or delete it."
exit 1
