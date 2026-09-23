#!/bin/bash
# Build and run the (1 + w) table cross-check, with one mutation per
# thing the check claims.
#
#   bash .prof/norm_wp_xcheck.sh
#
# TWO KINDS OF MUTATION, because the check makes two claims. The FILL
# mutations drop the `+ 1.0f` from a pool's build: the cached rows then
# compute x*inv*w while the rows past the pool's slots still compute
# x*inv*(1+w), so the number of differing rows IS the number of rows
# that took the inline path - a green check with zero differing rows
# under mutation would mean the table was never consulted and the whole
# check was measuring one arm twice. The KEY mutations replace a pool's
# full row compare with a two-word endpoint sample: the reuse fixture
# presents one address with a row that differs only in the middle, so
# the sampled words still agree and the stale table is served. That
# fixture going red under the key mutations and staying green without
# them is the whole evidence that the key is compared and not assumed.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/normwp}
CC=${CC:-gcc}
F="-O2 -std=c99"
SRC="ops ops_kernel ops_epi ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
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
echo "-- gate (gated arm) --"
"$O/gate" gated || { echo "FAIL: the gated arm is red"; exit 1; }

# One mutation per table, and the ARGV column says which invocation the
# table lives in - the two Q15 tables share one pool, so only the arm
# that runs first gets a cached/inline split (see the probe's header).
#
# wp and qw drop the `+ 1.0f` from their own fill loop, so the
# differing-row count is that table's split. The gated table has no
# `+ 1.0f` to drop - it quantizes w itself - so its mutation is an
# off-by-one in the same fill, applied inside build_qw's line range
# because the identical expression also appears in the two callers'
# cache-miss arms, which are the very thing it must NOT change.
FAILED=0
for M in wp wpkey qw qwkey gated; do
    rm -rf "$O/tree"; mkdir -p "$O/tree"; cp -r "$R/src" "$O/tree/src"
    ARGV=""
    case $M in
    wp) sed -i '0,/for (i = 0; i < n; i++) p\[i\] = 1.0f + w\[i\];/s//for (i = 0; i < n; i++) p[i] = w[i];/' \
            "$O/tree/src/ops_norm.c"
        PAT='p\[i\] = w\[i\];' ;;
    wpkey)
        sed -i 's/w_row_same(wp_slot\[i\]\.s, w, n)/(wp_slot[i].s[0] == w[0] \&\& wp_slot[i].s[n - 1] == w[n - 1])/' \
            "$O/tree/src/ops_norm.c"
        PAT='wp_slot\[i\]\.s\[0\] == w\[0\]' ;;
    qw) sed -i '0,/for (i = 0; i < n; i++) wp\[i\] = 1.0f + w\[i\];/s//for (i = 0; i < n; i++) wp[i] = w[i];/' \
            "$O/tree/src/ops_norm.c"
        PAT='wp\[i\] = w\[i\];' ;;
    qwkey)
        sed -i 's/w_row_same(qw_slot\[i\]\.s, w, n)/(qw_slot[i].s[0] == w[0] \&\& qw_slot[i].s[n - 1] == w[n - 1])/' \
            "$O/tree/src/ops_norm.c"
        PAT='qw_slot\[i\]\.s\[0\] == w\[0\]' ;;
    gated)
        sed -i '/^static const short \*build_qw/,/^    qw_used += n;/ s/int v = q8_round(w\[i\] \* sw);/int v = q8_round(w[i] * sw) + 1;/' \
            "$O/tree/src/ops_norm.c"
        PAT='q8_round(w\[i\] \* sw) + 1;'
        ARGV=gated ;;
    esac
    if ! grep -q "$PAT" "$O/tree/src/ops_norm.c"; then
        echo "FAIL: mutation '$M' did not apply - its target moved"
        FAILED=1
        continue
    fi
    if ! build "$O/tree" "mut_$M"; then
        echo "FAIL: mutation '$M' did not build"
        tail -5 "$O/mut_$M.build.log"
        FAILED=1
        continue
    fi
    echo "-- mutation $M --"
    OUT=$("$O/mut_$M" $ARGV 2>&1)
    echo "$OUT" | grep -E 'rows|elem|key:'
    if ! echo "$OUT" | grep -q '^FAIL'; then
        echo "FAIL: mutation '$M' left the check GREEN - that table is never"
        echo "      consulted, or its key is never compared, so the check"
        echo "      compares one arm to itself"
        FAILED=1
    fi
done
[ $FAILED -eq 0 ] || exit 1
echo "norm_wp_xcheck: PASS (gate green, all five mutations red)"
exit 0
