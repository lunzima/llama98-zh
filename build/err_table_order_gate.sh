#!/bin/bash
# The two message tables in err.c list their codes in the same order as
# the LZErr enum in err.h.
#
# WHY IT MATTERS EVEN THOUGH THE CODE IS CORRECT TODAY. The tables use
# designated initialisers, so order is irrelevant to the compiler and a
# misordered file still produces the right messages - which is how
# LZ_ERR_ST_READ came to sit two slots early in both tables with nothing
# noticing. That construct is C99. The target family includes Visual C++
# 4.0, which cannot compile it at all, and the conversion it needs is to
# positional initialisers - where order is the ONLY thing that binds a
# string to a code. A file that is silently misordered today becomes a
# file that silently swaps two error messages the moment it is converted.
#
# So this asserts the property that conversion will depend on, before
# the conversion, and keeps it true afterwards. It is the whole safety
# net that designated initialisers provide, restated in a form that
# survives losing them.
#
# Exit codes:
#   0  PASS - both tables match the enum, element for element
#   1  FAIL - an order or count mismatch, or the extraction found nothing
set -u
cd "$(dirname "$0")/.."

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

# The enum, minus its count sentinel.
grep -oE "^ *LZ_ERR_[A-Z0-9_]+" src/err.h | tr -d ' ' \
    | grep -v '^LZ_ERR_COUNT$' > "$W/enum.txt"
# Every row's tag comment, in file order. Two tables, so twice the enum.
# The comment IS the binding now - the compiler sees position only - so
# this reads exactly what a human reads and nothing else.
grep -oE "^ *\/\* LZ_ERR_[A-Z0-9_]+" src/err.c | grep -oE "LZ_ERR_[A-Z0-9_]+" > "$W/tbl.txt"

N=$(wc -l < "$W/enum.txt")
T=$(wc -l < "$W/tbl.txt")

# Positive control. Both extractions have to find something, and the
# table has to be exactly two tables' worth - an empty-vs-empty compare
# is equal and means nothing, and a regex that has stopped matching one
# side would otherwise read as a clean bill of health.
if [ "$N" -lt 50 ]; then
    echo "FAIL: only $N enum member(s) extracted from err.h - the pattern"
    echo "      has stopped matching and any comparison here is empty"
    exit 1
fi
if [ "$T" -ne $((N * 2)) ]; then
    echo "FAIL: err.c has $T initialiser tag(s), expected $((N * 2))"
    echo "      ($N codes x 2 tables) - a table is short, long, or one of"
    echo "      the two is not being seen at all"
    exit 1
fi

head -n "$N" "$W/tbl.txt" > "$W/en.txt"
tail -n "$N" "$W/tbl.txt" > "$W/zh.txt"

FAIL=0
for t in en zh; do
    if ! diff -q "$W/enum.txt" "$W/$t.txt" >/dev/null; then
        echo "FAIL: the ${t} table is not in enum order:"
        diff "$W/enum.txt" "$W/$t.txt" | head -8 | sed 's/^/        /'
        FAIL=1
    fi
done

if [ "$FAIL" -eq 0 ]; then
    echo "err_table_order_gate: PASS ($N codes, both tables in enum order)"
    exit 0
fi
echo "err_table_order_gate: FAIL"
exit 1
