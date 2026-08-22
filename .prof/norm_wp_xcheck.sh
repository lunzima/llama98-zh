#!/bin/bash
# Build and run the (1 + w) table cross-check, plus the mutation that
# shows it can fail AND measures which arm ran.
#
#   bash .prof/norm_wp_xcheck.sh
#
# The mutation drops the `+ 1.0f` from build_wp. Under it the cached
# rows compute x*inv*w while the rows past the pool's slots still
# compute x*inv*(1+w), so the number of differing rows IS the number of
# rows that took the inline path - a green check with zero differing
# rows under mutation would mean the table was never consulted and the
# whole check was measuring one arm twice.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/normwp}
CC=${CC:-gcc}
F="-O2 -std=c99"
SRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     fwht cpucheck err"

mkdir -p "$O"

build() {   # build DIR TAG
    local d="$1" tag="$2" f list=""
    for f in $SRC; do list="$list $d/src/$f.c"; done
    $CC $F -I"$d/src" -o "$O/$tag" "$R/.prof/norm_wp_xcheck.c" $list -lm \
        2>"$O/$tag.build.log"
}

if ! build "$R" gate; then
    echo "norm_wp_xcheck: SKIP - probe did not build"
    tail -5 "$O/gate.build.log"
    exit 2
fi
echo "-- gate --"
"$O/gate" || { echo "FAIL: the gate itself is red"; exit 1; }

# One mutation per table. Each drops the `+ 1.0f` from its own fill
# loop, so the differing-row count is that table's cached/inline split.
FAILED=0
for M in wp qw; do
    rm -rf "$O/tree"; mkdir -p "$O/tree"; cp -r "$R/src" "$O/tree/src"
    case $M in
    wp) sed -i '0,/for (i = 0; i < n; i++) p\[i\] = 1.0f + w\[i\];/s//for (i = 0; i < n; i++) p[i] = w[i];/' \
            "$O/tree/src/ops_norm.c"
        PAT='p\[i\] = w\[i\];' ;;
    qw) sed -i '0,/for (i = 0; i < n; i++) wp\[i\] = 1.0f + w\[i\];/s//for (i = 0; i < n; i++) wp[i] = w[i];/' \
            "$O/tree/src/ops_norm.c"
        PAT='wp\[i\] = w\[i\];' ;;
    esac
    if ! grep -q "$PAT" "$O/tree/src/ops_norm.c"; then
        echo "FAIL: mutation '$M' did not apply - its fill loop moved"
        FAILED=1
        continue
    fi
    if ! build "$O/tree" "mut_$M"; then
        echo "FAIL: mutation '$M' did not build"
        tail -5 "$O/mut_$M.build.log"
        FAILED=1
        continue
    fi
    echo "-- mutation $M (its fill loop forgets the +1) --"
    OUT=$("$O/mut_$M" 2>&1)
    echo "$OUT" | grep -E 'rows|elem'
    if ! echo "$OUT" | grep -q '^FAIL'; then
        echo "FAIL: mutation '$M' left the check GREEN - that table is never"
        echo "      consulted, so the check compares one arm to itself"
        FAILED=1
    fi
done
[ $FAILED -eq 0 ] || exit 1
echo "norm_wp_xcheck: PASS (gate green, both mutations red)"
exit 0
