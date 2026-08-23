#!/bin/bash
# Wall clock on THIS machine, for the one question the rest of the
# measurement stack cannot answer.
#
#   dev_timing.sh [binary] [runs] [model] [token-file]
#
# The tree's performance criteria are static: f32 operation counts
# (-DLZ_F32COUNT=1), branch evaluations (gcov -b -c), instruction counts
# on ARM. All of them are about a machine that issues in order and
# predicts badly, and all of them are BLIND to an out-of-order core -
# they cannot see a dependency chain shortening, a cache line being
# reused, or a branch that costs nothing because it was predicted. So
# "must not be too slow on a modern box" had no criterion at all.
#
# IRON LAW 3 DOES NOT FORBID THIS, and the distinction is the whole
# reason this script can exist: what that law rules out is QEMU wall
# clock, because QEMU models no cache and no pipeline, so its timing
# describes the host translating instructions. Timing a real core on
# itself describes that core. The number here transfers to nothing else
# - it is not a claim about a Pentium - and that is stated rather than
# implied.
#
# BEST OF N, not the mean: scheduling noise, turbo and another process
# waking up all make a run LONGER, never shorter, so the minimum is the
# estimate with the noise removed rather than averaged in.
#
# The TOTAL seconds field is read rather than s/token: the latter prints
# three decimals on a ~0.010 value, 5% resolution, which cannot see the
# size of regression this is meant to catch. Total is two decimals on
# several seconds.
#
# NO THRESHOLD, and that is deliberate. A number here means nothing
# without a second build to compare against, and this machine is not the
# target - a pass/fail line would be inventing a budget for hardware
# nobody ships on. Run it on two revisions and read the pair.
set -u

BIN=${1:-}
if [ -z "$BIN" ]; then
    for c in ./llama98.exe ./llama98; do [ -x "$c" ] && BIN=$c && break; done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || {
    echo "dev_timing: SKIP - no binary (make native, or pass one)"; exit 2; }

N=${2:-4}
MODEL=${3:-${MODELS_ROOT:-/e/LLM/models}/kmr20}
TOKENS=${4:-build/norm_int_gate_tokens_zh.txt}
[ -d "$MODEL" ]  || { echo "dev_timing: SKIP - no model: $MODEL"; exit 2; }
[ -f "$TOKENS" ] || { echo "dev_timing: SKIP - no token file: $TOKENS"; exit 2; }

echo "dev_timing: $BIN, $(basename "$MODEL"), $(basename "$TOKENS"), best of $N"
best=""
for i in $(seq 1 "$N"); do
    t=$("$BIN" "$MODEL" --tokens-file "$TOKENS" 2>/dev/null \
        | grep -m1 '^Time' | awk '{print $2}')
    if [ -z "$t" ]; then echo "  run $i: RUNFAIL"; continue; fi
    echo "  run $i: $t s"
    if [ -z "$best" ]; then best=$t
    else best=$(awk -v a="$best" -v b="$t" 'BEGIN{print (b<a)?b:a}'); fi
done
[ -n "$best" ] || { echo "dev_timing: SKIP - no run produced a time"; exit 2; }
echo "dev_timing: best $best s"
exit 0
