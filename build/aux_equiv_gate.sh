#!/bin/bash
# Did the last commit change what a hand-written kernel COMPUTES?
#
# The reorder passes on the #pragma aux bodies are meant to move
# instructions, not values: iron law two says the assembly tier and its
# intrinsics twin agree bit for bit, so a reorder that changes a result
# is a defect however plausible the new number looks.
#
# WHY THIS EXISTS. lz_dot128_t2_asm was reordered to hide pmaddwd
# latency and the pass hoisted `movq mm4, [edx]` above the `add edx, 8`
# that defines its address. Chunks 1-3 then read the previous chunk's
# weights; chunk 0 stayed correct, so anything that spot-checked the
# first sub-block agreed. build/dot128_t2_probe.c counted 11523 of 49156
# values wrong - and it is hand-run, so nothing said so for a day.
#
# pentiumlint's aux_equiv ALREADY CATCHES THIS. Run on that commit's
# before and after it reports "lz_dot128_t2_asm: store 0 computes a
# different value". The tool was not missing. It was not run. That is
# what a gate is for.
#
# Exit: 0 pass, 1 fail, 2 skip (no pentiumlint, or no git history).
set -u
cd "$(dirname "$0")/.."

L=${PENTIUMLINT:-}
for c in "$L" ../kunkun-linter/pentiumlint /e/kunkun-linter/pentiumlint \
         /mnt/e/kunkun-linter/pentiumlint; do
    [ -n "$c" ] && [ -f "$c/aux_equiv.py" ] && L=$c && break
done
if [ -z "$L" ] || [ ! -f "$L/aux_equiv.py" ]; then
    echo "aux_equiv_gate: SKIP - pentiumlint not found (set PENTIUMLINT)"
    exit 2
fi

PY=python; command -v python >/dev/null 2>&1 || PY=python3
FILES="src/ops_mmx.c src/ops_mmx_sse.c src/ops_sse2.c"
T=$(mktemp -d "$(pwd)/../.auxequiv-XXXXXX") || exit 1
trap 'rm -rf "$T"' EXIT

compare() {   # $1 = old text file, $2 = new text file, $3 = label
    "$PY" - "$L" "$1" "$2" "$3" <<'PYEOF'
import sys, os
sys.path.insert(0, sys.argv[1])
import aux_equiv
old = open(sys.argv[2], encoding="utf-8", errors="replace").read()
new = open(sys.argv[3], encoding="utf-8", errors="replace").read()
declared = set(os.environ.get("DECLARED", "").split())
seen = os.environ.get("SEEN_LOG", "")
bad = 0
n = 0
for name, ok, detail in aux_equiv.compare_files(old, new):
    n += 1
    if ok:
        continue
    if seen:
        open(seen, "a", encoding="utf-8").write("CHANGED %s \n" % name)
    if name in declared:
        print("  changed %s (%s): %s - DECLARED" % (name, sys.argv[4], detail))
        continue
    print("  CHANGED %s (%s): %s" % (name, sys.argv[4], detail))
    bad += 1
print("  %s: %d kernel(s) compared, %d undeclared change(s)"
      % (sys.argv[4], n, bad))
sys.exit(1 if bad else 0)
PYEOF
}

OLD_REV=${OLD_REV:-HEAD~1}
NEW_REV=${NEW_REV:-HEAD}
if ! git rev-parse --verify -q "$OLD_REV" >/dev/null 2>&1; then
    echo "aux_equiv_gate: SKIP - no $OLD_REV to compare against"
    exit 2
fi

# A VALUE CHANGE IS SOMETIMES THE POINT - the commit that fixed
# lz_dot128_t2_asm changed what it computes, which is what fixing it
# means. So the gate takes a declaration in the commit message:
#
#   kernel-value-change: lz_dot128_t2_asm  why, in a clause
#
# Declared and changed is a pass. Changed and undeclared is the failure.
# Declared and unchanged is a warning, not a pass-through: a marker that
# outlives its commit would quietly whitelist the kernel forever.
DECLARED=$(git log -1 --format=%B "$NEW_REV" 2>/dev/null |
           sed -n 's/^[[:space:]]*kernel-value-change:[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*\).*/\1/p')
[ -n "$DECLARED" ] && echo "-- declared value change(s):" $DECLARED " --"

echo "-- $OLD_REV -> $NEW_REV --"
rc=0
compared=0
for f in $FILES; do
    b=$(basename "$f")
    git show "$OLD_REV:$f" > "$T/old-$b" 2>/dev/null || continue
    git show "$NEW_REV:$f" > "$T/new-$b" 2>/dev/null || continue
    compared=$((compared + 1))
    DECLARED="$DECLARED" SEEN_LOG="$T/seen.log" \
        compare "$T/old-$b" "$T/new-$b" "$b" || rc=1
done
for d in $DECLARED; do
    grep -q "CHANGED $d " "$T/seen.log" 2>/dev/null || \
        echo "  warn: '$d' is declared changed but did not change -" \
             "a stale marker whitelists it for good"
done
if [ "$compared" -eq 0 ]; then
    # Not a pass. Zero comparisons prints the same as everything agreeing.
    echo "aux_equiv_gate: FAIL - none of the asm TUs could be read from git"
    exit 1
fi

# ---- the positive control ----------------------------------------------
# The historical defect itself, replanted: lift the two loads at a chunk
# boundary above the two pointer increments. A checker blind to the GP
# address dependence calls that a pure reorder, which is exactly the
# false negative this gate would be worthless with.
# The WORKING TREE copy, not HEAD's: the control asks whether the checker
# can see this class of change at all, which is a property of the checker
# and not of what happens to be committed.
cp src/ops_mmx.c "$T/mut.c"
"$PY" - "$T/mut.c" <<'PYEOF'
import sys, re
p = sys.argv[1]
s = open(p, encoding="utf-8", errors="replace").read()
pat = ('    "add       ecx, 64" \\\n'
       '    "add       edx, 8" \\\n'
       '    "movq      mm4, [edx]" \\\n'
       '    "movq      mm5, [ecx]" \\\n')
rep = ('    "movq      mm4, [edx]" \\\n'
       '    "movq      mm5, [ecx]" \\\n'
       '    "add       ecx, 64" \\\n'
       '    "add       edx, 8" \\\n')
if pat not in s:
    print("  FAIL - the mutation site is gone; this gate is unproven")
    sys.exit(2)
open(p, "w", encoding="utf-8", newline="").write(s.replace(pat, rep, 1))
PYEOF
mrc=$?
if [ $mrc -ne 0 ]; then
    echo "aux_equiv_gate: FAIL - could not plant the control mutation"
    exit 1
fi

echo "-- control: a load hoisted above the pointer increment --"
if compare src/ops_mmx.c "$T/mut.c" "mutation" > "$T/mut.log" 2>&1; then
    cat "$T/mut.log"
    echo "aux_equiv_gate: FAIL - the checker called the planted hoist a pure"
    echo "  reorder. Its verdicts above mean nothing."
    exit 1
fi
grep -m1 "CHANGED" "$T/mut.log" | sed 's/^/  /'

if [ $rc -ne 0 ]; then
    echo "aux_equiv_gate: FAIL - a kernel body's VALUE changed in the last"
    echo "  commit. If that was intended, say so in the commit message and"
    echo "  re-run the format's cross-check probe; if it was a reorder, it"
    echo "  moved something across a dependence."
    exit 1
fi
echo "aux_equiv_gate: PASS (mutation caught)"
exit 0
