#!/bin/bash
# How often each kernel-tier dispatch actually executes.
#
# The engine picks a SIMD tier two different ways:
#
#   a six-slot function-pointer table plus a pick (LZ_DEFINE_PICK in
#   ops_kernel_shared.h, and lz_row_pick in ops_matmul.c) - 5 sites; and
#
#   a direct branch on the global enum, `if (g_kernel == LZ_KERNEL_SSE2)`
#   - 48 sites.
#
# WHICH IS FASTER IS NOT A PROPERTY OF THE MECHANISM. It is a property of
# where the site sits:
#
#   outside the hot loop   both are free. A function pointer read once
#                          per matmul costs one load; a branch on a
#                          global that never changes is predicted every
#                          time.
#   inside the hot loop    they diverge, and NOT in the direction the
#                          question usually assumes. An indirect call
#                          per element cannot be inlined and cannot be
#                          scheduled across; a correctly-predicted
#                          branch costs approximately nothing on any
#                          core with a predictor, and on the in-order
#                          ARM926 target it costs the branch itself.
#
# So refactoring "to the better mechanism" wholesale is not a decision
# this repository can make from the source. It needs the execution count
# of each site, which is what this measures - the same method
# div_hotness.sh uses for divides, and for the same reason: where a
# thing sits is the whole reading, and a static count cannot see it.
#
# COLD AND UNATTRIBUTED ARE DIFFERENT ANSWERS. gcov writes "#####" for a
# line it instrumented that nobody reached - that is a count, and the
# count is zero. It writes "-" for a line it holds no code for, which
# means attribution failed (usually the optimiser moved or folded the
# test). Folding the two together turns "this dispatch is cold" into
# "this dispatch is unaccounted for", and they call for opposite
# actions: the first is fine where it is, the second is not measured at
# all.
#
# A CENSUS, NOT A GATE. It prints; it does not judge.
#
# Usage: bash build/dispatch_hotness.sh [MODEL] [TOP]
set -u
cd "$(dirname "$0")/.."

# THE SAME DEFAULT AS ITS SIBLINGS, and that is not cosmetic.
# div_hotness.sh, mul_hotness.sh and branch_census.sh all default to
# kmr20; this one shipped defaulting to the pilot model, and the first
# thing that happened was a cross-census comparison between a count
# taken here and a count taken there - two different generations, two
# different sets of code paths, presented as one ratio. cov_run.sh's own
# header says it: "A count is only comparable to another count from the
# same prompt, the same token budget and the same seed." A default that
# differs from the family's makes that mistake the easy one to make.
MODEL=${1:-/e/LLM/models/kmr20}
TOP=${2:-30}

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

. build/cov_run.sh
cov_run "$W" "$MODEL" 8 || exit 2

# The dispatch shapes, as they are written in the source. `pick` covers
# both LZ_DEFINE_PICK's generated functions and lz_row_pick.
#
# `cpu` is here because it is the same question asked a different way -
# "which machine is this" is as invariant as "which tier is this", and
# it is asked in the same places. It is worth its own row rather than
# being folded into `branch`: lz_cpu_has_mmx has ALREADY been through
# this treatment once (it was an extern function called 75,080,736
# times, now a static inline reading a cached global - see ops_quant.h),
# so its count answers a different question. A high number here does not
# mean "hoist it", it means "the inline is not being folded", which is a
# different problem with a different fix.
BRANCH_RE='g_kernel[[:space:]]*==|lz_kernel_sel\(\)[[:space:]]*=='
PICK_RE='_pick\('
CPU_RE='lz_cpu_has_[a-z0-9_]*\(\)|lz_soft_float_target\(\)'

: > "$W/sites.txt"
for f in src/*.c src/*.h; do
    [ -f "$f" ] || continue
    grep -nE "$BRANCH_RE" "$f" 2>/dev/null | while IFS=: read -r ln rest; do
        printf '%s\t%s\tbranch\n' "$(basename "$f")" "$ln"
    done >> "$W/sites.txt"
    grep -nE "$PICK_RE" "$f" 2>/dev/null \
      | grep -vE 'LZ_DEFINE_PICK|static .* _pick|^\s*[0-9]+:\s*/\*' \
      | while IFS=: read -r ln rest; do
        printf '%s\t%s\tpick\n' "$(basename "$f")" "$ln"
    done >> "$W/sites.txt"
    grep -nE "$CPU_RE" "$f" 2>/dev/null \
      | grep -vE 'static int lz_cpu_has|^\s*[0-9]+:\s*\*' \
      | while IFS=: read -r ln rest; do
        printf '%s\t%s\tcpu\n' "$(basename "$f")" "$ln"
    done >> "$W/sites.txt"
done

# Join: sites from the source, counts from the run. gcov's line format
# is "count:line:text", with counts carrying thousands separators on
# some locales - strip anything that is not a digit before comparing.
: > "$W/joined.txt"
while IFS=$'\t' read -r base ln kind; do
    g="$W/$base.gcov"
    [ -f "$g" ] || { printf '%s\t%s\t%s\tNOGCOV\n' "$base" "$ln" "$kind" \
        >> "$W/joined.txt"; continue; }
    cnt=$(awk -F: -v want="$ln" '
        { l=$2; gsub(/ /,"",l); if (l==want) { c=$1; gsub(/ /,"",c); print c; exit } }
    ' "$g")
    case "$cnt" in
        "")      cnt=NOLINE ;;
        \#####)  cnt=0 ;;
        -)       cnt=UNATTR ;;
        *)       cnt=$(printf '%s' "$cnt" | tr -cd '0-9') ;;
    esac
    printf '%s\t%s\t%s\t%s\n' "$base" "$ln" "$kind" "$cnt" >> "$W/joined.txt"
done < "$W/sites.txt"

echo
echo "dispatch sites by execution count (model $(basename "$MODEL"), 8 tokens)"
echo
printf '%12s  %6s  %-7s  %s\n' COUNT LINE KIND FILE
awk -F'\t' '$4 ~ /^[0-9]+$/ {printf "%12d\t%s\t%s\t%s\n", $4, $2, $3, $1}' \
    "$W/joined.txt" | sort -rn | head -"$TOP" \
  | while IFS=$'\t' read -r c ln kind base; do
        printf '%12s  %6s  %-7s  %s\n' "$c" "$ln" "$kind" "$base"
    done

echo
for k in branch pick cpu; do
    hot=$(awk -F'\t' -v k=$k '$3==k && $4 ~ /^[0-9]+$/ && $4+0>0' "$W/joined.txt" | wc -l)
    cold=$(awk -F'\t' -v k=$k '$3==k && $4=="0"' "$W/joined.txt" | wc -l)
    un=$(awk -F'\t' -v k=$k '$3==k && ($4=="UNATTR"||$4=="NOLINE"||$4=="NOGCOV")' \
        "$W/joined.txt" | wc -l)
    tot=$(awk -F'\t' -v k=$k '$3==k {s+=($4 ~ /^[0-9]+$/ ? $4 : 0)} END{print s+0}' \
        "$W/joined.txt")
    printf '%-7s sites: %3d executed, %3d cold, %3d unattributed; %d executions\n' \
        "$k" "$hot" "$cold" "$un" "$tot"
done
echo
echo "A site executed once per model load costs nothing whichever way it"
echo "is written. Only the ones near the top of this table are a question."
