#!/bin/bash
# pentiumlint / cyrixlint: pre-P6 x86 inefficiency census.
#
# Reads build/pre_p6_ineff.txt (pattern, processor set, reason) and
# scans the Watcom shipping objects for those patterns, reporting which
# processor(s) each finding bites on. The per-processor split is the
# value: inc/dec stall on P5's partial flags but not on K6, so a flat
# rule would be wrong on half the family.
#
# THIS IS A CENSUS, NOT A GATE. It prints; it does not judge. A finding
# means "on this processor this sequence is slower than the rewrite",
# and the rewrite is only correct when the flags it changes are dead -
# the reader proves that, this does not.
#
# WHY WATCOM OBJECTS. The shipping build is 32-bit Watcom for the
# pre-P6 family; gcc x86-64 is a different target (more registers, a
# different scheduler) and its numbers do not transfer (see the
# rulebook's cross-toolchain note). wdis reads Watcom's OMF.
#
# Usage: bash build/pentiumlint.sh [OBJ...]
set -u
cd "$(dirname "$0")/.."

TABLE="build/pre_p6_ineff.txt"
[ -f "$TABLE" ] || { echo "pentiumlint: missing $TABLE"; exit 2; }

WDIS=""
for w in /c/WATCOM/binnt64/wdis.exe /c/WATCOM/binnt/wdis.exe; do
    [ -x "$w" ] && WDIS="$w" && break
done
[ -n "$WDIS" ] || { echo "pentiumlint: SKIP - no wdis"; exit 2; }

OBJS=${*:-}
if [ -z "$OBJS" ]; then
    OBJS=$(ls build/watcom/cli/ops_*.obj build/watcom/cli/*_mmx.obj 2>/dev/null)
fi
[ -n "$OBJS" ] || { echo "pentiumlint: no Watcom objects (run make watcom-cli first)"; exit 2; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

for o in $OBJS; do
    [ -f "$o" ] || continue
    base=$(basename "$o" .obj)
    "$WDIS" "$o" > "$W/$base.dis" 2>/dev/null
done

echo "pentiumlint: pre-P6 inefficiencies (census)"
echo
echo "  finding            processors                 object:function-ish"
echo

# Each rule: pattern (comma mnemonics), processors, reason. For every
# mnemonic, grep the disassemblies and report one line per (rule, object)
# with the total hit count, so a reader sees WHERE, not just WHICH.
while IFS=$'\t' read -r pat procs reason; do
    case "$pat" in ''|\#*) continue ;; esac
    for o in $OBJS; do
        [ -f "$o" ] || continue
        base=$(basename "$o" .obj)
        hits=0
        for m in $(echo "$pat" | tr ',' ' '); do
            n=$(grep -ciE "\\b$m\\b" "$W/$base.dis" 2>/dev/null)
            hits=$((hits + n))
        done
        [ "$hits" -gt 0 ] || continue
        printf "  %-18s %-25s %s (%d)\n" "$pat" "$procs" "$base" "$hits"
    done
done < "$TABLE"

echo
echo "procs legend: p5 p55c cyrix winchip k6 k6-2 (the pre-P6 Socket 7"
echo "family the rulebook targets). A rule fires only on the processors it"
echo "names; the rewrite is only correct when the changed flags are dead."
