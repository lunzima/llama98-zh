#!/bin/bash
# epi's int32 x int16 -> int64 MAC, gcc against Watcom.
#
#   bash .prof/epi_xcheck.sh
#
# HOW THIS COMPARES THEM, because it is transitive and that is only
# sound under a condition this script checks. Neither side's output is
# diffed against the other's directly; each is run against
# .prof/epi_mac_oracle.c, which recomputes every sum three more ways -
# a 32x32 partial-product form, the carry-free 16-bit split, and the
# straight int64 loop. The inputs are a fixed xorshift seed, so both
# sides see the same sequence, and agreement with one predicate over
# the same inputs is agreement with each other.
#
# THE CONDITION IS THE CASE COUNT. If one side ran fewer cases the
# transitive step is invalid - it would be two green runs over
# different input sets, which is what "both passed" would otherwise be
# read as. The counts are compared below and a mismatch is a FAIL even
# when both sides say PASS.
#
# The ARM kernel is the third implementation of the same operator and
# meets the same predicate; it needs WSL and QEMU, so it stays in
# .prof/epi_arm_gate.sh rather than here, where the other harnesses are
# all gcc against Watcom.
set -u
cd "$(dirname "$0")/.."

GCC=${GCC:-gcc}
O=${O:-/tmp/epixchk}
mkdir -p "$O"

count_of() { printf '%s\n' "$1" | sed -n 's/^epi_mac_oracle: \([0-9]*\) cases.*/\1/p'; }

# gcc, linking the shipped ops_sse2.c body rather than the harness copy.
if ! $GCC -O2 -std=c99 -Wall -Wextra -msse2 -Isrc \
        -DEPI_LINK_SHIPPED -DLZ_SSE2_TU=1 \
        .prof/epi_mac_oracle.c src/ops_sse2.c src/ops_sched.c \
        src/cpucheck.c src/err.c -o "$O/gcc.exe" 2>"$O/gcc.log"; then
    echo "epi_xcheck: FAIL - gcc build:"
    tail -5 "$O/gcc.log"
    exit 1
fi
GOUT=$("$O/gcc.exe" 2>&1); GRC=$?
GN=$(count_of "$GOUT")

WOUT=$(bash .prof/epi_watcom_gate.sh 2>&1); WRC=$?
WN=$(count_of "$WOUT")

if [ "$WRC" = 2 ]; then
    echo "epi_xcheck: SKIP - no Watcom toolchain"
    exit 2
fi

echo "  gcc     rc=$GRC  $(printf '%s\n' "$GOUT" | tail -2 | head -1)"
echo "  watcom  rc=$WRC  $(printf '%s\n' "$WOUT" | tail -2 | head -1)"

if [ "$GRC" != 0 ] || [ "$WRC" != 0 ]; then
    echo "epi_xcheck: FAIL - a side did not pass the oracle"
    exit 1
fi
if [ -z "$GN" ] || [ -z "$WN" ] || [ "$GN" != "$WN" ]; then
    echo "epi_xcheck: FAIL - case counts differ (gcc=$GN watcom=$WN);"
    echo "  two green runs over different inputs are not a comparison"
    exit 1
fi
echo "epi_xcheck: PASS - gcc and Watcom SSE2 meet one predicate over $GN cases"
exit 0
