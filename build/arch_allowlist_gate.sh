#!/bin/bash
# The two-generation allowlist: this engine loads Qwen3.5 (GDN) and
# KunMoe (KDA) and refuses every other checkpoint shape, rather than
# half-binding it against one of their specs. The verdict comes from the
# checkpoint's OWN linear-layer tensor names - src/model.c's arch_probe
# - never from its model_type string.
#
#   arch_allowlist_gate.sh [models-root] [engine-binary]
#
# WHAT EACH ARM IS FOR. None of them is "it exited 1", and the reason is
# that exit code alone cannot tell a refusal by this rule apart from any
# of the dozen other ways a load fails.
#
#   control   kunkun98-dense must LOAD. "The engine refused X" is
#             satisfied by an engine that refuses everything, so without
#             this arm the gate stays green on a build that loads
#             nothing at all.
#
#   refused   three checkpoints that are neither generation, and whose
#             SHAPES differ from one another - 311 tensors with no
#             linear_attn name anywhere, 105 of them the same, and a
#             15-tensor MTP slice. Each must fail with the allowlist
#             message as its ENTIRE output. One fixture alone would
#             leave open that only that one reaches the predicate; the
#             exit code alone would accept an unrelated failure.
#
#   table     the expected text is READ OUT of src/err.c rather than
#             retyped here, so the gate follows the message table (iron
#             law one) instead of drifting from it. Two rows must carry
#             the marker - the EN/ZH pair, so neither language has lost
#             the message - and the English row must be unique, or the
#             text no longer identifies one error code and matching it
#             stops meaning anything.
#
#   mutant    an engine built from a copy of src/ with arch_probe's
#             "-1: neither generation" changed to "0: GDN". On it the
#             allowlist message must disappear from all three fixtures
#             WHILE the control still loads. That is what turns "the
#             message appeared" into "the message appeared BECAUSE the
#             probe answered neither": same checkpoints, same rest of
#             the loader, one line changed. Point the gate at a mutant
#             binary (the second argument) and the refused arms go red -
#             that is this gate's own positive control, and the reason
#             the argument exists.
#
# The mutation is built in a temp copy rather than edited into the
# tracked source, which on a shared worktree would outlive the run.
#
# Exit: 0 pass, 1 fail, 2 skip (no binary, no models root, no fixture).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${1:-E:/LLM/models}
BIN=${2:-$ROOT/llama98.exe}

CONTROL=kunkun98-dense
REFUSED="Qwen3-0.6B router-bert qwen3.5-0.8b-mtp"

[ -x "$BIN" ] || { echo "arch_allowlist_gate: SKIP - no $BIN (run: make native)"; exit 2; }
[ -d "$MODELS" ] || { echo "arch_allowlist_gate: SKIP - no models root '$MODELS'"; exit 2; }
for d in $CONTROL $REFUSED; do
    [ -d "$MODELS/$d" ] || {
        echo "arch_allowlist_gate: SKIP - no fixture dir '$MODELS/$d'"; exit 2; }
done

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

FAILED=0
fail() { echo "  FAIL: $*"; FAILED=1; }

# The message, from the table that owns it. The sed picks the row whose
# literal carries no backslash escape, which is the English one: the
# Chinese row spells its text as \xNN bytes and cannot match.
MARK=LZ_ERR_CFG_UNSUPPORTED_ARCH
NMARK=$(grep -c "/\* $MARK \*/" "$ROOT/src/err.c")
EN_ROW=$(sed -n 's|^    /\* '"$MARK"' \*/ "\([^"\\]*\)",[[:space:]]*$|\1|p' "$ROOT/src/err.c")
NEN=$(printf '%s\n' "$EN_ROW" | grep -c .)
if [ "$NMARK" -ne 2 ] || [ "$NEN" -ne 1 ]; then
    echo "  FAIL: src/err.c carries the $MARK marker $NMARK time(s) (want 2,"
    echo "  the EN/ZH pair) and $NEN English row(s) (want 1) - the expected"
    echo "  message cannot be read out of the table, so nothing below would"
    echo "  be comparing against it"
    echo "arch_allowlist_gate: FAIL"
    exit 1
fi
WANT="Failed to load: $EN_ROW"

echo "arch_allowlist_gate: $MODELS (engine $BIN)"
echo "  message under test: $EN_ROW"

# run <bin> <dir> -> RC, OUT (whole output, CR-stripped)
RC=0
OUT=
run() {
    "$1" "$2" --check >"$W/out" 2>&1
    RC=$?
    OUT=$(tr -d '\r' <"$W/out")
}

run "$BIN" "$MODELS/$CONTROL"
if [ "$RC" -eq 0 ]; then
    echo "  control  $CONTROL: rc=0 (loads)"
else
    fail "control $CONTROL exited $RC - a build that refuses this one too"
    echo "         makes every refusal below say nothing about the allowlist"
    printf '%s\n' "$OUT" | tail -3 | sed 's/^/        | /'
fi

for d in $REFUSED; do
    run "$BIN" "$MODELS/$d"
    rc=$RC
    out=$OUT
    if [ "$rc" -eq 0 ]; then
        fail "$d LOADED (rc=0) - not refused at all"
    elif [ "$out" = "$WANT" ]; then
        echo "  refused  $d: rc=$rc, the allowlist message and nothing else"
    else
        fail "$d exited $rc, but that is a DIFFERENT failure:"
        printf '%s\n' "$out" | sed 's/^/        | /'
    fi
done

# The three refused checkpoints must not be three copies of one shape: a
# predicate shown to answer -1 for one file has not been shown to answer
# -1 for another. Printed so the report can see them, and asserted
# distinct so a fixture swapped for a duplicate is caught.
shape() {
    local f n
    f=$(ls "$1"/*.safetensors 2>/dev/null | head -1)
    [ -n "$f" ] || { echo "no-safetensors"; return; }
    n=$(head -c 4 "$f" | od -An -tu4 | tr -d ' ')
    [ -n "$n" ] || n=0
    printf '%s-tensors/linear_attn=%s' \
        "$(head -c $((8 + n)) "$f" | grep -ao '"dtype"' | wc -l | tr -d ' ')" \
        "$(head -c $((8 + n)) "$f" | grep -ao 'linear_attn\.' | wc -l | tr -d ' ')"
}
SHAPES=""
for d in $REFUSED; do
    s=$(shape "$MODELS/$d")
    SHAPES="$SHAPES $s"
    echo "  shape    $d: $s"
done
UNIQ=$(for s in $SHAPES; do echo "$s"; done | sort -u | wc -l | tr -d ' ')
[ "$UNIQ" -eq 3 ] || fail "the three refused fixtures are not three distinct shapes ($UNIQ distinct) - 'all three reach the same predicate' would then be one measurement taken three times"

# Mutation. The guard runs first: if the line moved or was rewritten,
# the substitution below would match nothing and the gate would go on
# testing the unmutated engine while believing it had a mutant.
N1=$(grep -c 'seen_kda == seen_gdn) return -1;' "$ROOT/src/model.c")
if [ "$N1" -ne 1 ]; then
    fail "src/model.c has $N1 occurrence(s) of arch_probe's neither-generation return, expected 1 - the line this arm mutates is gone, and the arm would otherwise test an unmutated build"
else
    MUT=$W/mut
    mkdir -p "$MUT"
    cp -r "$ROOT/src" "$ROOT/common" "$MUT/"
    cp "$ROOT/Makefile" "$MUT/Makefile"
    sed -i 's/seen_kda == seen_gdn) return -1;/seen_kda == seen_gdn) return 0;/' "$MUT/src/model.c"
    if [ "$(grep -c 'seen_kda == seen_gdn) return 0;' "$MUT/src/model.c")" -ne 1 ]; then
        fail "the mutation did not land in the copy of src/model.c"
    elif ( cd "$MUT" && make native ) >"$MUT/build.log" 2>&1; then
        echo "  mutation arch_probe's -1 -> 0, built from a copy of src/:"
        run "$MUT/llama98.exe" "$MODELS/$CONTROL"
        if [ "$RC" -eq 0 ]; then
            echo "  mutant   $CONTROL: still rc=0 (the mutation is confined to the refusal)"
        else
            fail "with the mutant engine the control $CONTROL stopped loading (rc=$RC) - the mutation is not confined to the refusal"
            printf '%s\n' "$OUT" | tail -3 | sed 's/^/        | /'
        fi
        for d in $REFUSED; do
            run "$MUT/llama98.exe" "$MODELS/$d"
            rc=$RC
            out=$OUT
            if [ "$out" = "$WANT" ]; then
                fail "the allowlist message SURVIVED the mutation on $d - it does not come from the branch this gate is about"
            else
                echo "  mutant   $d: rc=$rc, $(printf '%s' "$out" | head -1)"
            fi
            [ "$rc" -ne 0 ] || fail "with the mutant engine $d loaded (rc=0) - the refusal vanished entirely, not just its message"
        done
    else
        fail "the mutant engine did not build"
        tail -5 "$MUT/build.log" | sed 's/^/        | /'
    fi
fi

[ "$FAILED" -eq 0 ] && { echo "arch_allowlist_gate: PASS"; exit 0; }
echo "arch_allowlist_gate: FAIL"
exit 1
