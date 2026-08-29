#!/bin/bash
# Does select_topk_exp change the candidate distribution? Builds the
# CURRENT sampler and the one from HEAD~ side by side and compares the
# distributions lz_target_dist returns.
#
#   bash .prof/sampler_topk_xcheck.sh [old-rev]
#
# ONE SNAPSHOT, TWO SAMPLERS. Both binaries are compiled from the same
# copy of src/, with only sampler.c swapped. This is not tidiness - it
# is the whole reason the check means anything. The first two attempts
# at this comparison were worthless: one built the control from HEAD
# while the new binary came from the working tree, which also carried
# another session's uncommitted forward.c, and the other rebuilt the two
# arms several minutes apart while that session kept editing. Both
# reported EVERY seed differing, including --temp 0, which the sampler
# change cannot reach.
#
# WHAT IT ASSERTS
#   candidate-set differences  must be 0 - the surviving token set is
#                              the same set, which is the claim that
#                              matters. exp is monotone, so top_k on
#                              logits picks what top_k on probabilities
#                              picks.
#   max relative difference    must stay under LIMIT (a few float32
#                              ulps). The normalizer cancels in exact
#                              arithmetic; what is left is accumulated
#                              rounding, and this is where it is
#                              bounded rather than asserted.
#
# It is NOT a bit-identity check, and it must not be turned into one:
# the change is documented as value-changing. Measured 4.8e-07, about
# four ulps, over 24 cases and 571 surviving candidates.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
OLDREV=${1:-HEAD~1}
O=${O:-/tmp/smtopk}
CC=${CC:-gcc}
LIMIT=${LIMIT:-2e-06}
# lz_mathf carries lz_powf, which sampler.c calls for the temperature
# curve. Without it the link fails on an undefined reference and this
# harness reports SKIP - "new arm did not build" - which reads as an
# environment problem and is a missing name in this list. A SKIP is not
# a PASS, and a SKIP whose cause is the harness is worse than a FAIL:
# nobody goes looking.
SRC="sampler ops ops_kernel ops_epi ops_quant ops_sched ops_norm ops_gdn ops_matmul
     ops_t2_scalar fwht cpucheck err lz_mathf"

command -v python3 >/dev/null 2>&1 || {
    echo "sampler_topk_xcheck: SKIP - needs python3 for the float compare"
    exit 2; }

rm -rf "$O"; mkdir -p "$O/new" "$O/old"
cp -r "$R/src" "$O/new/src"
cp -r "$R/src" "$O/old/src"
if ! git -C "$R" show "$OLDREV:src/sampler.c" > "$O/old/src/sampler.c" 2>/dev/null; then
    echo "sampler_topk_xcheck: SKIP - no src/sampler.c at $OLDREV"
    exit 2
fi

build() {   # build DIR TAG
    local d="$1" tag="$2" f list=""
    for f in $SRC; do list="$list $d/src/$f.c"; done
    $CC -O2 -std=c99 -I"$d/src" -o "$O/bin_$tag" \
        "$R/.prof/sampler_topk_xcheck.c" $list -lm 2>"$O/$tag.log"
}
for T in new old; do
    build "$O/$T" "$T" || {
        echo "sampler_topk_xcheck: SKIP - $T arm did not build"
        tail -5 "$O/$T.log"; exit 2; }
done

"$O/bin_new" "$O/new.bin" > "$O/new.txt" || exit 1
"$O/bin_old" "$O/old.bin" > "$O/old.txt" || exit 1

python3 - "$O/old.bin" "$O/new.bin" "$LIMIT" <<'PY'
import struct, sys
a = open(sys.argv[1], 'rb').read()
b = open(sys.argv[2], 'rb').read()
limit = float(sys.argv[3])
if len(a) != len(b):
    print("FAIL: dumps differ in size"); sys.exit(1)
n = len(a) // 4
A = struct.unpack('<%df' % n, a)
B = struct.unpack('<%df' % n, b)
mx = 0.0; nz = 0; setdiff = 0
for x, y in zip(A, B):
    if (x != 0.0) != (y != 0.0):
        setdiff += 1
    if x != 0.0 or y != 0.0:
        nz += 1
        d = abs(x - y) / max(abs(x), abs(y), 1e-30)
        mx = max(mx, d)
print("surviving candidates compared: %d" % nz)
print("candidate-set differences:     %d" % setdiff)
print("max relative difference:       %.3e  (limit %.1e)" % (mx, limit))
if nz < 100:
    print("FAIL: fewer than 100 candidates survived across all cases - the")
    print("      fixture is too peaked to be testing the nucleus at all")
    sys.exit(1)
if setdiff:
    print("FAIL: the two paths keep different tokens")
    sys.exit(1)
if mx > limit:
    print("FAIL: the distributions drifted past the rounding budget")
    sys.exit(1)
print("PASS")
PY
