#!/bin/bash
# Every bilingual string table lists its codes in the same order as the
# enum that indexes it.
#
# WHY THIS EXISTS. The tables were written with designated initialisers,
# which bind by index - so order never reached the compiler and a
# misordered file still produced the right strings. err.c had one code
# two slots out of place for as long as it did for exactly that reason.
# That construct is C99 and Visual C++ 4.0, which is in the target
# family, cannot compile it; the conversion is to positional
# initialisers, where order is the ONLY thing binding a string to a
# code. This gate is the safety the designated form gave, restated so it
# survives losing it - and it is stronger, because it reports the drift
# the designated form tolerated.
#
# ONE SCRIPT, TWO TABLES. src/err.c and gui/localized_strings.c have the
# same shape and the same hazard; a second copy of this logic would be a
# second thing to keep in step. Each row below is
# <prefix> <header> <source>.
#
# Exit codes:
#   0  PASS - every table matches its enum, element for element
#   1  FAIL - an order or count mismatch, or an extraction found nothing
set -u
cd "$(dirname "$0")/.."

TABLES="LZ_ERR:src/err.h:src/err.c
LZ_STR:gui/localized_strings.h:gui/localized_strings.c"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
FAIL=0
TOTAL=0

for row in $TABLES; do
    PFX=${row%%:*}
    rest=${row#*:}
    HDR=${rest%%:*}
    SRC=${rest#*:}
    [ -f "$HDR" ] && [ -f "$SRC" ] || {
        echo "FAIL: $HDR or $SRC missing"; FAIL=1; continue; }

    grep -oE "^ *${PFX}_[A-Z0-9_]+" "$HDR" | tr -d ' ' \
        | grep -v "^${PFX}_COUNT\$" > "$W/enum.txt"
    # The tag comment on each row. That comment IS the binding for a
    # reader, the way position is for the compiler, so this reads
    # exactly what a human reads and nothing else.
    grep -oE "^ *\/\* ${PFX}_[A-Z0-9_]+" "$SRC" \
        | grep -oE "${PFX}_[A-Z0-9_]+" > "$W/tbl.txt"

    N=$(wc -l < "$W/enum.txt")
    T=$(wc -l < "$W/tbl.txt")

    # Positive control. An empty-vs-empty compare is equal and means
    # nothing; a regex that has stopped matching one side would
    # otherwise read as a clean bill of health.
    if [ "$N" -lt 50 ]; then
        echo "FAIL: only $N enum member(s) extracted from $HDR - the"
        echo "      pattern has stopped matching and any comparison here"
        echo "      is empty"
        FAIL=1; continue
    fi
    if [ "$T" -ne $((N * 2)) ]; then
        echo "FAIL: $SRC has $T tag comment(s), expected $((N * 2))"
        echo "      ($N codes x 2 tables) - a table is short, long, or"
        echo "      one of the two is not being seen at all"
        FAIL=1; continue
    fi

    head -n "$N" "$W/tbl.txt" > "$W/a.txt"
    tail -n "$N" "$W/tbl.txt" > "$W/b.txt"
    for t in a b; do
        if ! diff -q "$W/enum.txt" "$W/$t.txt" >/dev/null; then
            echo "FAIL: $SRC table $t is not in enum order:"
            diff "$W/enum.txt" "$W/$t.txt" | head -8 | sed 's/^/        /'
            FAIL=1
        fi
    done
    [ "$FAIL" -eq 0 ] && echo "  $SRC: $N codes, both tables in enum order"
    TOTAL=$((TOTAL + N))
done

if [ "$FAIL" -eq 0 ]; then
    echo "str_table_order_gate: PASS ($TOTAL codes across 2 files)"
    exit 0
fi
echo "str_table_order_gate: FAIL"
exit 1
