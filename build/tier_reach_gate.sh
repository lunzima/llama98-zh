#!/bin/bash
# Every quantised matmul must reach its tiered kernel, not the scalar
# reference.
#
# WHAT THIS PINS AND WHY NOTHING ELSE DOES. The dispatch in
# lz_matmul_wq guards each format with
#
#     gs >= 32 && gs % 32 == 0 && in_dim % gs == 0
#
# plus in_dim <= LZ_MM_WIDEN_MAX for the four non-Q8 formats. A tensor
# failing any clause is still CORRECT - it takes matmul_scalar_ref - so
# a routing regression is invisible to every numeric gate in the tree.
# kernel_isa_identity_gate compares tiers against each other and passes
# whether or not the SIMD kernel ran; the parity gates compare x86
# against ARM and do the same. "Agrees" and "agrees and was reached" are
# different statements, and only an execution count separates them.
#
# ASSERTS, on the shipping dispatch with no --kernel pin:
#   1. matmul_scalar_ref executes ZERO times
#   2. at least one tiered kernel executes, so a build where nothing
#      runs cannot pass vacuously
#
# NOT asserted: that every impl is non-zero. matmul_t2_impl is 0 on
# kmr20 because that checkpoint has no T2 tensor - a correct 0 and an
# unreachable 0 look identical from here, so the gate does not pretend
# to tell them apart. The per-impl counts are printed for the reader.
#
# FUNCTION COUNTS, NOT LINE COUNTS. An earlier reading of this same
# question used the first line gcov held code for at or below the name's
# first grep hit - and ops_matmul.c:79 is a forward DECLARATION of
# matmul_scalar_ref, so the walk landed in lz_row_pick's body and
# reported its 3,832 calls as scalar fallbacks. `gcov -f` answers "how
# many times was this function entered" directly and cannot be pointed
# at the wrong body.
#
# Exit: 0 pass, 1 fail, 2 skip (no coverage build / no model).
#
# Usage: bash build/tier_reach_gate.sh [MODEL] [TOKENS]
set -u
cd "$(dirname "$0")/.."

MODEL=${1:-/e/LLM/models/kmr20}
TOKENS=${2:-8}

[ -d "$MODEL" ] || { echo "tier-reach: SKIP - no model at $MODEL"; exit 2; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

. build/cov_run.sh
cov_run "$W" "$MODEL" "$TOKENS" || exit 2

# gcov -f prints, for each function:
#     Function 'matmul_q8_impl'
#     Lines executed:NN.NN% of N
# and, when it was never entered, "No executions for this function".
gcov -f -o "$W" "$W"/*.gcda >"$W/f.log" 2>&1 || {
    echo "tier-reach: SKIP - gcov -f failed"; tail -3 "$W/f.log"; exit 2; }

# awk over the pairs: a function is ENTERED unless the following line
# says otherwise. Reported as entered/not, since gcov -f gives a line
# percentage rather than a call count.
entered() {  # $1 = function name -> "yes" | "no" | "absent"
    awk -v want="$1" '
        /^Function/ {
            name = $0
            sub(/^Function ./, "", name)     # gcov writes: Function '\''name'\''
            sub(/.$/, "", name)
            cur = (name == want)
            next
        }
        cur && /No executions/ { print "no"; found = 1; exit }
        cur && /Lines executed:/ {
            pct = $0; sub(/.*:/, "", pct); sub(/%.*/, "", pct)
            print (pct + 0 > 0) ? "yes" : "no"; found = 1; exit
        }
        END { if (!found) print "absent" }
    ' "$W/f.log"
}

echo "tier-reach: model $(basename "$MODEL"), $TOKENS tokens"
TIERED="matmul_q8_impl matmul_q41_impl matmul_q61_impl matmul_t2_impl \
matmul_q16_impl"
any=0
for fn in $TIERED; do
    st=$(entered "$fn")
    printf '  %-20s %s\n' "$fn" "$st"
    [ "$st" = yes ] && any=1
done

ref=$(entered matmul_scalar_ref)
printf '  %-20s %s\n' "matmul_scalar_ref" "$ref"

rc=0
if [ "$ref" = absent ]; then
    echo "FAIL: matmul_scalar_ref not found in the coverage data - the"
    echo "      gate did not measure what it claims to."
    rc=1
elif [ "$ref" = yes ]; then
    echo "FAIL: a quantised matmul fell through to the scalar reference."
    echo "      Correct but untiered: check the dispatch guards in"
    echo "      lz_matmul_wq against this checkpoint's (dtype, gs)."
    rc=1
fi
if [ "$any" -ne 1 ]; then
    echo "FAIL: no tiered kernel ran at all - a pass here would be vacuous."
    rc=1
fi
[ "$rc" -eq 0 ] && echo "PASS: every quantised matmul reached a tiered kernel"
exit $rc
