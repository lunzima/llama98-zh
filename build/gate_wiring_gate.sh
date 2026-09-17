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
# WHAT COUNTS AS A CALLER: a mention in the Makefile, in gates.mk or in
# ci.sh. Not a mention in another gate - that would let two unwired gates
# cite each other. gates.mk is where the gate targets live now: the
# shipped Makefile carries none of them and pulls it in with `-include`,
# so a gate wired only there is wired. Scripts under build/arm,
# build/mips and build/x87 are exempt
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
    if ! grep -q "$b" Makefile gates.mk build/ci.sh 2>/dev/null; then
        UNWIRED="$UNWIRED $b"
        FAIL=1
    fi
done

for g in build/arm/*.sh build/mips/*.sh build/x87/*.sh; do
    [ -f "$g" ] || continue
    rel=${g#build/}
    case " $EXEMPT " in *" $rel "*) continue ;; esac
    if ! grep -q "$rel" Makefile gates.mk build/ci.sh 2>/dev/null; then
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
norm_sse_parity_gate.sh    # a full Watcom engine build; the #pragma aux
                           #   twins it compares exist in no gcc build
i32f_acc_gate.sh           # same, for the two int32->float accumulate
                           #   tiers - two engine builds per run
align_gate.sh              # 5489 compiled checks, whole-tree instrumented build
codesize_gate.sh           # builds three toolchains to compare .text
native_obj_gate.sh         # pre-link hook, silent by design; ci.sh is where
                           #   it has objects to look at
kernel_isa_identity_gate.sh # disassembles both toolchains' objects
switch_matrix_gate.sh      # builds the knob matrix, many configurations
float_knob_gate.sh         # instrumented build plus two planted axes
str_table_order_gate.sh    # needs a model to walk the string tables
ced_gate.sh                # four engine builds: the off and on arms, plus a
                           #   mutant each for the skip and for the final-chunk
                           #   condition, plus the previous source rebuilt from
                           #   git for the off-vs-HEAD arm
ced_replay_gate.sh         # a model dir copied three times plus a mutant
                           #   engine build
mla_gate.sh                # needs python3 numpy+safetensors and rewrites a
                           #   ~500MB model.safetensors three times
csa2_gate.sh                # needs python3 numpy+safetensors and rewrites a
                           #   model.safetensors twice
kv_share_gate.sh           # builds its own fixture out of the given
                           #   checkpoint's weights, needs python3
                           #   numpy+safetensors, and skips on every
                           #   checkpoint here that this engine can load
attn_sink_learned_gate.sh  # builds its own fixtures from a checkpoint's
                           #   weights (four copies of a ~550MB
                           #   model.safetensors), needs python3
                           #   numpy+safetensors, and builds two mutant
                           #   engines
gr_gate.sh                 # three copies of a ~550MB model.safetensors,
                           #   needs python3 numpy+safetensors, builds the
                           #   pre-change engine from git and four more
                           #   mutant engines
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
endian_gate.sh             # needs WSL and the MIPS cross toolchain, and
                           #   runs the engine big-endian under qemu
"
# REACHED, not mentioned. `make -n check` expands the whole prerequisite
# tree and prints the recipes without running them, so what comes out is
# what `make check` would actually invoke. Grepping the Makefile instead
# answers a different question and answers it wrong: gcc_flag_parity_gate
# and native_obj_gate both have rules, both are named in the file, and
# neither is a prerequisite of check - they are precisely the case this
# half exists to catch, and a mention-grep waves them through. Same
# substitution the whole gate is about, one level down.
#
# THE DEFAULT GOAL COUNTS TOO, and leaving it out made this half wrong in
# the accusing direction. native_obj_gate.sh is not a prerequisite of
# `check`; it is a line in the `llama98` link recipe, so it runs on every
# native build and `make -n check` never mentions it. Reading only check
# would report a gate that runs constantly as one that never runs.
REACHED=$( { make -n check 2>/dev/null; make -n 2>/dev/null; } \
           | grep -oE '[a-z0-9_]+_gate\.sh' | sort -u)

CIONLY_UNDECLARED=""
NEVER_RUN=""
for g in build/*_gate.sh build/*/*_gate.sh; do
    b=$(basename "$g")
    [ "$b" = "gate_wiring_gate.sh" ] && continue
    case "
$REACHED" in *"
$b"*) continue ;; esac                             # reached by make check
    if ! grep -q "$b" build/ci.sh 2>/dev/null; then
        # THE THIRD CATEGORY, and it used to `continue` past here with
        # the comment "already counted above" - which was not true of
        # it. The first half asks only whether the name is MENTIONED in
        # the Makefile, so a .PHONY target that nothing depends on and
        # nothing invokes reads as wired; this half then skipped exactly
        # those, because they are not in ci.sh either. Between the two,
        # a gate could be reachable, tracked, declared nowhere, and run
        # by nothing - which is the substitution this whole gate exists
        # to refuse, reintroduced one level up from where it refuses it.
        # gcc_watcom_output_gate.sh sat here for two days.
        NEVER_RUN="$NEVER_RUN $b"
        FAIL=1
        continue
    fi
    case "$CI_ONLY" in
        *"$b"*) ;;
        *) CIONLY_UNDECLARED="$CIONLY_UNDECLARED $b" ;;
    esac
done
if [ -n "$NEVER_RUN" ]; then
    echo "gate_wiring_gate: FAIL - has a Makefile target, but nothing runs it:"
    for u in $NEVER_RUN; do echo "    $u"; done
    echo "  Not a prerequisite of \`check\` and not invoked by ci.sh, so"
    echo "  the only way it ever runs is somebody typing its target by"
    echo "  hand. Make it a prerequisite of \`check\`, or invoke it from"
    echo "  ci.sh and declare it in CI_ONLY, or delete it - but a gate"
    echo "  that never runs is worse than no gate, because the wiring"
    echo "  census counts it."
fi
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
if grep -q "subn_gate.sh" Makefile gates.mk 2>/dev/null; then
    echo "gate_wiring_gate: FAIL - subn_gate.sh reads as reachable from a"
    echo "  Makefile target, which it is not; the depth split is broken and"
    echo "  its PASS means nothing"
    exit 1
fi

# Positive control. If the search stopped matching, every script would
# look unwired rather than wired - so the control is a name that MUST be
# found, not one that must not.
if ! grep -q "kernel_matrix_gate.sh" Makefile gates.mk build/ci.sh 2>/dev/null; then
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

# ALL OF THEM, subdirectories included. This counted only
# build/*_gate.sh, so the census reported 53 while the tree held 56 -
# and a count that reads as complete is the same lie this gate exists
# to refuse, told about itself. arm/parity_gate.sh, arm/softfp_gate.sh
# and mips/endian_gate.sh were wired (the loop above checks that) but
# never classified: neither the ci.sh-only declaration nor the
# nothing-runs-it check reached them.
N=$(ls build/*_gate.sh build/*/*_gate.sh 2>/dev/null | wc -l)
# Counted, not just asserted: a PASS that does not say how the N split
# between the two depths is the same shape of silence this half was
# added to remove.
NMK=0
for g in build/*_gate.sh build/*/*_gate.sh; do
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
echo "  A gate nothing invokes is a script. Add it to \`check\` in"
echo "  gates.mk, or to ci.sh, or delete it."
exit 1
