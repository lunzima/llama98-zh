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
# HAVING A CALLER IS NOT THE SAME AS BEING REACHED. gcc_warning_gate had
# one - ci.sh - and this gate was green on it for as long as it existed,
# while `make check` reported PASS over a dropped comment terminator that
# gcc warned about on every single build. `check` is the target people
# run; ci.sh is the one that runs on a schedule someone else owns. So the
# second half below splits the two and requires the ci.sh-only list to be
# WRITTEN DOWN with a reason. Being slow, or being a census that prints
# instead of judging, are good reasons; nobody having thought about it is
# not, and that is the state the split exists to make visible.
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

# ---- ci.sh-only, by decision rather than by accident -------------------
#
# Each name here is a gate `make check` deliberately does not run, with
# the reason. Anything ci.sh-only and NOT listed is the hole this half
# exists to find. Removing a gate from `check` therefore costs a line
# here, which is the point: it makes the shallower target's coverage a
# thing someone chose.
#
# CENSUSES, which print and do not judge - there is no PASS to assert:
#   div_census div_hotness move_census branch_census nest_census
#   o3_probe xcc_codesize dev_timing
# Those are not *_gate.sh and so never reach the loop above; named here
# only so the next reader does not go looking for them. cov_run.sh is
# not a census either - it is the coverage build the last two share.
#
# GATES THAT NEED A MODEL, a cross-toolchain build, or minutes of wall
# clock. `check` is meant to be runnable before every commit.
CI_ONLY="
prefetch_parity_gate.sh    # two full Watcom engine builds, and the arms it
                           #   checks are compiled by no other toolchain
align_gate.sh              # 5489 compiled checks, whole-tree instrumented build
codesize_gate.sh           # builds three toolchains to compare .text
native_obj_gate.sh         # pre-link hook, silent by design; ci.sh is where
                           #   it has objects to look at
kernel_isa_identity_gate.sh # disassembles both toolchains' objects
switch_matrix_gate.sh      # builds the knob matrix, many configurations
float_knob_gate.sh         # instrumented build plus two planted axes
str_table_order_gate.sh    # needs a model to walk the string tables
changelog_comment_gate.sh  # ratchet over the whole tree, slow
subn_gate.sh               # needs a model and a long token run
kv_rot_gate.sh             # needs a model and a long token run
perf_knob_gate.sh          # needs a model, and measures wall clock
tier_reach_gate.sh         # a coverage build plus a real generation, and
                           #   the answer is a property of the CHECKPOINT
                           #   as much as of the code
xcheck_gate.sh             # needs a model
fixed_tier_xcompiler_gate.sh # needs a model and both toolchains
recur_write_float_arm_gate.sh # needs the ARM cross toolchain
int_pipeline_arm_gate.sh   # needs the ARM cross toolchain
caption_symmetry_gate.sh   # needs the gui build and a screen readback
"
# REACHED, not mentioned. `make -n check` expands the whole prerequisite
# tree and prints the recipes without running them, so what comes out is
# what `make check` would actually invoke. Grepping the Makefile instead
# answers a different question and answers it wrong: gcc_flag_parity_gate
# and native_obj_gate both have rules, both are named in the file, and
# neither is a prerequisite of check - they are precisely the case this
# half exists to catch, and a mention-grep waves them through. Same
# substitution the whole gate is about, one level down.
REACHED=$(make -n check 2>/dev/null | grep -oE '[a-z0-9_]+_gate\.sh' | sort -u)

CIONLY_UNDECLARED=""
for g in build/*_gate.sh; do
    b=$(basename "$g")
    [ "$b" = "gate_wiring_gate.sh" ] && continue
    case "
$REACHED" in *"
$b"*) continue ;; esac                             # reached by make check
    grep -q "$b" build/ci.sh 2>/dev/null || continue   # already counted above
    case "$CI_ONLY" in
        *"$b"*) ;;
        *) CIONLY_UNDECLARED="$CIONLY_UNDECLARED $b" ;;
    esac
done
if [ -n "$CIONLY_UNDECLARED" ]; then
    echo "gate_wiring_gate: FAIL - ci.sh-only and not declared:"
    for u in $CIONLY_UNDECLARED; do echo "    $u"; done
    echo "  \`make check\` does not run these. Either add a target, or add"
    echo "  the name to CI_ONLY with the reason it is too slow or too"
    echo "  environment-dependent to run before a commit."
    FAIL=1
fi

# Positive control, pointed the way the other two are: a name that MUST
# read as ci.sh-only. If the Makefile grep started matching everything,
# the list above would go empty and this half would pass by measuring
# nothing.
if grep -q "subn_gate.sh" Makefile 2>/dev/null; then
    echo "gate_wiring_gate: FAIL - subn_gate.sh reads as reachable from a"
    echo "  Makefile target, which it is not; the depth split is broken and"
    echo "  its PASS means nothing"
    exit 1
fi

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
#
# EVERY SHELL SCRIPT UNDER build/, not only the gates. The first version
# looked at *_gate.sh and *.awk, and the very next script added here was
# a census - div_hotness.sh - which `git add` refused and which would
# have been committed as a working-tree-only file if the refusal had
# gone unread. A census that is not in the clone is as absent as a gate
# that is not.
UNTRACKED=""
for g in build/*.sh build/*/*.sh build/*.awk; do
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
# Counted, not just asserted: a PASS that does not say how the N split
# between the two depths is the same shape of silence this half was
# added to remove.
NMK=0
for g in build/*_gate.sh; do
    b=$(basename "$g")
    [ "$b" = "gate_wiring_gate.sh" ] && continue
    case "
$REACHED" in *"
$b"*) NMK=$((NMK + 1)) ;; esac
done
NCI=$((N - 1 - NMK))
if [ "$FAIL" -eq 0 ]; then
    echo "gate_wiring_gate: PASS ($N gate script(s), every one reachable"
    echo "  from make or ci.sh, and every one tracked;"
    echo "  $NMK reached by \`make check\`, $NCI ci.sh-only and each declared)"
    exit 0
fi
[ -n "$UNWIRED" ] || exit 1
echo "gate_wiring_gate: FAIL - script(s) no target depends on:"
for u in $UNWIRED; do echo "    $u"; done
echo "  A gate nothing invokes is a script. Add it to \`check\` in the"
echo "  Makefile, or to ci.sh, or delete it."
exit 1
