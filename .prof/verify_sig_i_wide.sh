#!/bin/bash
# Build and run .prof/verify_sig_i_wide.c, AND the four mutations that
# show each of its four rows can fail.
#
#   bash .prof/verify_sig_i_wide.sh
#
# WHY THE MUTATIONS RUN EVERY TIME. Four rows, four claims, and a green
# run says nothing about any of them unless a broken version of each is
# red in that row AND ONLY THAT ROW - a row that goes red whenever any
# other one does is not a fourth observation, it is the same one written
# out again. Each mutation below is chosen to be surgical in exactly that
# sense: it perturbs one branch of sigmoid_q15_i on one side of one
# input split, and the script requires the red set to be that row alone.
#
# A0-REFS is not one of the four. It is this probe's own precondition
# (its two independent references agree), upstream of every row, so a
# mutation of either reference reddens the rows that use it as well -
# said here rather than pretended otherwise.
#
# Runs on the host compiler: every claim is about the two C bodies in
# ops_quant.c. No target toolchain needed.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/e/LLM/_cepi/sigwide}
CC=${CC:-gcc}
F="-O2 -std=gnu99"
SRC="ops_quant ops_sched err compat cpucheck"

mkdir -p "$O"

build() {   # build DIR TAG - compile the probe against the sources in DIR
    local d="$1" tag="$2" f list=""
    for f in $SRC; do list="$list $d/src/$f.c"; done
    $CC $F -I"$d/src" -o "$O/$tag.exe" "$R/.prof/verify_sig_i_wide.c" \
        $list -lm 2>"$O/$tag.build.log"
}

if ! build "$R" gate; then
    echo "verify_sig_i_wide: SKIP - probe did not build"
    tail -5 "$O/gate.build.log"
    exit 2
fi
echo "-- gate --"
"$O/gate.exe"
RC=$?
[ $RC -eq 0 ] || { echo "FAIL: the gate itself is red"; exit 1; }

FAILED=0
for M in neg pos narrow out; do
    rm -rf "$O/tree"
    mkdir -p "$O/tree"
    cp -r "$R/src" "$O/tree/src"
    Q="$O/tree/src/ops_quant.c"
    case $M in
    # The biased-unsigned floor, moved by one unit of 2^-15 of a table
    # step for NEGATIVE v only. Only the sign correction can be wrong
    # for one sign and right for the other, which is what makes the
    # two wide rows two rows.
    neg)
        ROW=WIDE-NEG
        sed -i 's/return q + LZ_SIG_T15;/return q + LZ_SIG_T15 + (v < 0 ? 1 : 0);/' "$Q"
        CHK='(v < 0 ? 1 : 0)' ;;
    pos)
        ROW=WIDE-POS
        sed -i 's/return q + LZ_SIG_T15;/return q + LZ_SIG_T15 + (v >= 0 ? 1 : 0);/' "$Q"
        CHK='(v >= 0 ? 1 : 0)' ;;
    # The pre-existing narrow body's shift. Reaches no e >= 20, so the
    # two wide rows must stay green under it.
    narrow)
        ROW=NARROW
        sed -i 's/sh = e - 4;/sh = e - 3;/' "$Q"
        CHK='sh = e - 3;' ;;
    # The out-of-domain fallback's scale. Reaches no e in [0, 50].
    out)
        ROW=OUTSIDE
        sed -i 's/pow2f(-e))/pow2f(-e - 1))/' "$Q"
        CHK='pow2f(-e - 1))' ;;
    esac
    if ! grep -qF "$CHK" "$Q"; then
        echo "FAIL: mutation '$M' did not apply - the line it edits moved"
        FAILED=1
        continue
    fi
    if ! build "$O/tree" "mut_$M"; then
        echo "FAIL: mutation '$M' did not build"
        tail -5 "$O/mut_$M.build.log"
        FAILED=1
        continue
    fi
    OUT=$("$O/mut_$M.exe" 2>&1)
    RED=$(echo "$OUT" | awk '/ FAIL$/ {printf "%s ", $1}')
    RED=$(echo $RED)
    if [ "$RED" = "$ROW" ]; then
        echo "mutation $M: red rows [$RED] - correct, and no other row"
    else
        echo "FAIL: mutation '$M' reddened [$RED], expected exactly [$ROW]"
        echo "$OUT" | sed 's/^/      /'
        FAILED=1
    fi
done

echo
if [ $FAILED -eq 0 ]; then
    echo "verify_sig_i_wide: PASS (gate green, 4 mutations each red in one row)"
else
    echo "verify_sig_i_wide: FAIL"
fi
exit $FAILED
