#!/bin/bash
# Every fixed seed in the harness is the SAME fixed seed.
#
# The project's rule is that where a fixed random seed is needed it is
# 1145141919. Today every place that needs one uses it - branch_census.sh
# passes --seed 1145141919, and lerp_q15_probe.c and exp_q20_probe.c both
# start their LCG there - so this gate does not fix anything. It keeps
# it fixed, which is a different job and the one that was missing: a
# convention with no check is a convention that holds until someone
# writes `--seed 42` because 42 was what they typed.
#
# WHY IT MATTERS AND IT IS NOT TIDINESS. A seed that differs between two
# harnesses makes their numbers incomparable while both look reproducible
# - each is stable on its own, and the disagreement reads as a real
# effect. That is the expensive failure, not the one where a seed is
# missing outright.
#
# WHAT IS NOT CHECKED, deliberately. The engine's own --seed default is
# 1, and that is a user-facing default rather than a measurement seed:
# `--seed 0` means "from the clock" and 1 means "reproducible for
# whoever runs it". Changing it would move behaviour nobody asked to
# move. This gate is about the harness - build/ - where a seed exists
# only to make a measurement repeatable.
#
# NOT SCANNED FOR MISSING SEEDS, also deliberately. "This invocation
# samples and passes no seed" needs to know which invocations sample,
# and the ones here mostly do not: perf_knob_gate and subn_gate drive
# --tokens/--dump-all-logits, which is deterministic prefill with no
# sampler in the path. A predicate that guessed would fire on those and
# be switched off. Wrong seeds are checkable exactly; missing ones are
# not, and half a check that runs beats a whole one that gets disabled.
#
# Exit codes:
#   0  PASS
#   1  FAIL - a fixed seed that is not the project's
set -u
cd "$(dirname "$0")/.."

SEED=1145141919
FAIL=0
HITS=0

FILES=$(ls build/*.sh build/*.c build/*.awk build/*.py \
           build/arm/*.sh build/mips/*.sh build/x87/*.sh 2>/dev/null \
        | grep -v "seed_gate.sh")

# --seed N on a command line, and the probe idiom `= <digits>UL`. Both
# are matched with the number captured so the value can be compared
# rather than merely detected - a grep for "1145141919" would pass a
# file that mentions it in a comment and seeds itself with something
# else two lines down.
for f in $FILES; do
    while read -r n; do
        [ -n "$n" ] || continue
        HITS=$((HITS + 1))
        if [ "$n" != "$SEED" ]; then
            echo "FAIL: $f - fixed seed $n, expected $SEED"
            grep -n -- "--seed[= ]*$n" "$f" | head -2 | sed 's/^/        /'
            FAIL=1
        fi
    done <<EOF
$(grep -oE -- "--seed[= ]+[0-9]+" "$f" 2>/dev/null | grep -oE "[0-9]+")
EOF
    while read -r n; do
        [ -n "$n" ] || continue
        HITS=$((HITS + 1))
        if [ "$n" != "$SEED" ]; then
            echo "FAIL: $f - LCG seeded at $n, expected $SEED"
            grep -n "$n" "$f" | head -2 | sed 's/^/        /'
            FAIL=1
        fi
    done <<EOF
$(grep -oE "^static unsigned long rs = [0-9]+UL" "$f" 2>/dev/null \
  | grep -oE "[0-9]+")
EOF
done

# POSITIVE CONTROL, pointed the way this gate needs: a value that MUST be
# found. If the extraction broke, every file would read as having no
# seed at all and the gate would pass by measuring nothing - the same
# shape as "0 packed instructions" being indistinguishable from "symbol
# not found" in o3_probe.
if [ "$HITS" -eq 0 ]; then
    echo "seed_gate: FAIL - no fixed seed found anywhere in build/,"
    echo "  which is not true of this tree; the extraction is broken and"
    echo "  a PASS from it would mean nothing"
    exit 1
fi

[ "$FAIL" -eq 0 ] || { echo "seed_gate: FAIL"; exit 1; }
echo "seed_gate: PASS ($HITS fixed seed(s) in the harness, all $SEED)"
exit 0
