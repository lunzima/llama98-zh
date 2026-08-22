#!/bin/bash
# Build and run the conv1d epilogue's cross-entry gate, AND the two
# mutations that show it can fail.
#
#   bash .prof/conv_epi_xentry.sh
#
# WHY THE MUTATIONS RUN EVERY TIME. The gate's claim is that
# sigmoid_q15(x) and sigmoid_q15_t(16*fl(x+12)) are the same function
# over the conv accumulator's domain - a claim with two independent
# halves (the table index, and the two entries' upper clamps agreeing
# only because g_sigtab[384] happens to be 32768). A green run says
# nothing about either half unless a broken version of each is red, and
# a mutation run by hand once is a fact about that afternoon. So the
# script breaks each half in a scratch copy of the tree and requires
# both to fail, in the region they belong to.
#
# Runs on the host compiler: the identity is a fact about the two C
# entry points in ops_quant.c, and it is --kernel arm-asm's conv
# epilogue (src/ops.c) that consumes it. No ARM toolchain needed.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/cvxentry}
CC=${CC:-gcc}
F="-O2 -std=c99 -I$R/src"
SRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     fwht cpucheck err"

mkdir -p "$O"

# build DIR TAG - compile the probe against the sources in DIR
build() {
    local d="$1" tag="$2" f list=""
    for f in $SRC; do list="$list $d/src/$f.c"; done
    $CC $F -I"$d/src" -o "$O/$tag" "$R/.prof/conv_epi_xentry.c" $list -lm \
        2>"$O/$tag.build.log"
}

if ! build "$R" gate; then
    echo "conv_epi_xentry: SKIP - probe did not build"
    tail -5 "$O/gate.build.log"
    exit 2
fi
echo "-- gate --"
"$O/gate"
RC=$?
[ $RC -eq 0 ] || { echo "FAIL: the gate itself is red"; exit 1; }

# ---- mutation 1: the upper clamp's table value -------------------------
# sigmoid_q15 returns a literal 32768 for x >= 12; sigmoid_q15_t returns
# g_sigtab[384]. Only the table makes those equal. Break the literal and
# ONLY the x >= 12 region may go red.
#
# ---- mutation 2: the folded coordinate's offset ------------------------
# t = acc*k1 + lz_sig_q15_t_offset(). Move the offset by one table step
# and the two entries index different rows: the INSIDE region goes red.
FAILED=0
for M in clamp offset; do
    rm -rf "$O/tree"
    mkdir -p "$O/tree"
    cp -r "$R/src" "$O/tree/src"
    case $M in
    clamp)
        sed -i 's/if (x >= -LZ_SIG_LO) return 32768;/if (x >= -LZ_SIG_LO) return 32767;/' \
            "$O/tree/src/ops_quant.c" ;;
    offset)
        sed -i 's/return -LZ_SIG_LO \* LZ_SIG_STEP; }/return -LZ_SIG_LO * LZ_SIG_STEP + 1.0f; }/' \
            "$O/tree/src/ops_quant.c" ;;
    esac
    if ! grep -q "$([ $M = clamp ] && echo 32767 || echo '+ 1.0f')" \
              "$O/tree/src/ops_quant.c"; then
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
    OUT=$("$O/mut_$M" 2>&1)
    if echo "$OUT" | grep -q '^PASS'; then
        echo "FAIL: mutation '$M' left the gate GREEN - it does not test that half"
        FAILED=1
    else
        echo "-- mutation $M: red as required --"
        echo "$OUT" | grep -E '^conv-epi|^  regions'
    fi
done
[ $FAILED -eq 0 ] || exit 1
echo "conv_epi_xentry: PASS (gate green, both mutations red)"
exit 0
