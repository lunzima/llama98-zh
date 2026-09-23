#!/bin/bash
# forward_csa2.c bills the f32 census.
#
# Until these three sites existed, a -DLZ_F32COUNT=1 build reported the
# whole compressed-attention layer as free: forward_csa2.c carried no
# LZ_FCX at all. A row that reads zero because nothing bills it and a row
# that reads zero because the code does no work are the same three
# characters, which is the reading this gate exists to separate - and the
# census is the instrument the whole soft-float reduction plan is
# ordered by, so a layer missing from it is a layer missing from the
# plan.
#
# WHAT EACH ARM PROVES
#   identity   the counting build and the plain build produce
#              byte-identical logits on the same tokens. This is the
#              census's own standing contract (ops.h: the counting build
#              is bit-identical to the shipping one); without it every
#              number below could be a number about a different program.
#   reached    on a model with a csa2 layer, all three sites report a
#              non-zero count AND lz_f32_count_reached - the row has to
#              exist in the table, not merely be absent.
#   scoped     on a model with no csa2 layer, all three report "not run",
#              which is the reached flag's own signal, not a zero. A site
#              that fires on a model that cannot reach it would mean the
#              billing sits on a shared path rather than in
#              forward_csa2.c.
#   mutation   one site's billing renamed onto another row: the indexer
#              must fall back to "- (not run)". The census prints every
#              site whatever happened to it (cli_main's own else branch),
#              so "not run" is a reading this table can produce - and the
#              one the reached arm above must not be satisfied by. That is
#              what makes the arm sensitive to the source rather than to
#              the -DLZ_F32COUNT flag alone.
#
# WHAT THIS GATE DOES NOT COVER, and cannot: the selection scan's integer
# dedup (k_eff * n_vis / 2 compares at the top-k this project ships) is
# the largest single cost of a csa2 layer and is deliberately unbilled -
# ops.h's LZ_FC_CSA2_IDX comment has the reason (an integer compare may
# not share a row with soft-float operations). Its cost is visible in the
# operation counts written at that loop, not in this table.
#
#   csa2_f32count_gate.sh [csa2-model] [control-model]
#
# Exit: 0 pass, 1 fail, 2 skip (no gcc, no model, no token list).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-E:/LLM/models/kunkun98}
CONTROL=${2:-E:/LLM/models/kmr20}
W=$(mktemp -d)
FAILED=0
# Long enough that the csa2 layer sees blocks: n_vis is (pos + 1)/cm, so a
# five-token prompt leaves every site at a handful of executions and a
# "non-zero" assertion would be testing rounding rather than billing.
LINE="人工智能的发展历史可以追溯到二十世纪五十年代，当时的科学家们开始思考机器是否能够模拟人类的思维过程。"
PROMPT=""
i=0
while [ $i -lt 8 ]; do PROMPT="$PROMPT$LINE"; i=$((i + 1)); done
SITES="csa2 indexer
csa2 top-k select
csa2 core attn"

command -v gcc >/dev/null 2>&1 || { echo "SKIP: no gcc"; exit 2; }
[ -d "$MODEL" ] || { echo "SKIP: no model at $MODEL"; exit 2; }
[ -d "$CONTROL" ] || { echo "SKIP: no control model at $CONTROL"; exit 2; }

# build <dir> <extra-cflags> [<sed-expr> <marker>]
build() {
    local dir="$1" extra="$2" expr="${3:-}" marker="${4:-}" s="" f
    mkdir -p "$dir"
    cp -r "$ROOT/src" "$ROOT/common" "$dir/"
    cp "$ROOT/Makefile" "$dir/Makefile"
    if [ -n "$expr" ]; then
        sed -i "$expr" "$dir/src/forward_csa2.c"
        if ! grep -q "$marker" "$dir/src/forward_csa2.c"; then
            echo "  FAIL: mutation did not apply"; return 1
        fi
    fi
    # The SIMD translation units link as prebuilt objects and carry no
    # billing sites (cov_run.sh's own reason), so build/native is reused.
    for f in $(cd "$ROOT" && make -s print-ENG); do s="$s $dir/src/$f.c"; done
    for f in $(cd "$ROOT" && make -s print-COM); do s="$s $dir/common/$f.c"; done
    gcc -O2 -march=x86-64 -std=c99 -I"$dir/src" -I"$dir/common" \
        -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 $extra \
        -o "$dir/e.exe" $s "$dir/src/cli_attr.c" "$dir/src/cli_main.c" \
        "$ROOT/build/native/ops_mmx.o" "$ROOT/build/native/ops_mmx_sse.o" \
        "$ROOT/build/native/ops_sse2.o" -lm > "$dir/build.log" 2>&1 || {
        echo "  FAIL: build failed in $dir"; tail -3 "$dir/build.log"; return 1; }
    return 0
}

# The census table, as "site|count" lines, from a --tokens-file run. The
# site name is matched as LETTER-INITIAL words rather than "anything up to
# two spaces": the count is itself word-shaped, so a greedy character
# class over [a-z0-9 -] swallows the number and reports the percentage's
# leading digit as the count.
census() {
    "$1" "$2" --tokens-file "$3" 2>/dev/null \
        | sed -n 's/^  \(csa2 [a-z][a-z0-9-]*\( [a-z][a-z0-9-]*\)*\)  *\(-\|[0-9][0-9]*\).*/\1|\3/p'
}

build "$W/plain" "" || exit 1
build "$W/count" "-DLZ_F32COUNT=1" || exit 1
PLAIN=$W/plain/e.exe
COUNT=$W/count/e.exe

"$COUNT" "$MODEL" --prompt "$PROMPT" --dump-prompt-tokens "$W/toks" -n 1 >/dev/null 2>&1
if [ ! -s "$W/toks" ]; then
    echo "SKIP: could not produce a token list for $MODEL"
    exit 2
fi

# ---- identity ---------------------------------------------------------
r=0
"$COUNT" "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/count.lg" >/dev/null 2>&1 || r=1
"$PLAIN" "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/plain.lg" >/dev/null 2>&1 || r=1
if [ "$r" -ne 0 ]; then
    echo "  FAIL: an identity arm exited non-zero"
    FAILED=1
elif cmp -s "$W/count.lg" "$W/plain.lg"; then
    echo "  identity: the counting build is bit-identical to the plain one ($(wc -c < "$W/plain.lg") bytes)"
else
    echo "  FAIL: the counting build changed the logits"
    FAILED=1
fi

# ---- reached, on a model that has a csa2 layer ------------------------
C=$(census "$COUNT" "$MODEL" "$W/toks")
if [ -z "$C" ]; then
    echo "  FAIL: the census printed no csa2 row at all"
    FAILED=1
else
    while IFS= read -r site; do
        got=$(printf '%s\n' "$C" | sed -n "s/^$site|//p")
        if [ -z "$got" ]; then
            echo "  FAIL: $site is missing from the census table"
            FAILED=1
        elif [ "$got" = "-" ]; then
            echo "  FAIL: $site exists but was never reached on $MODEL"
            FAILED=1
        elif [ "$got" -gt 0 ]; then
            echo "  reached: $site = $got operations per token"
        else
            echo "  FAIL: $site was reached and billed zero"
            FAILED=1
        fi
    done <<< "$SITES"
fi

# ---- scoped, on a model with no csa2 layer ---------------------------
"$COUNT" "$CONTROL" --prompt "$PROMPT" --dump-prompt-tokens "$W/ctoks" -n 1 >/dev/null 2>&1
if [ -s "$W/ctoks" ]; then
    C2=$(census "$COUNT" "$CONTROL" "$W/ctoks")
    while IFS= read -r site; do
        got=$(printf '%s\n' "$C2" | sed -n "s/^$site|//p")
        if [ "$got" = "-" ] || [ -z "$got" ]; then
            : # not run, or not in the table at all - both are "never reached"
        else
            echo "  FAIL: $site billed $got on a model with no csa2 layer"
            FAILED=1
        fi
    done <<< "$SITES"
    echo "  scoped: no csa2 row fires on $CONTROL (no csa2 layer to run)"
else
    echo "  SKIP: control model produced no token list"
fi

# ---- mutation: the indexer's billing renamed onto another row ---------
if build "$W/mut" "-DLZ_F32COUNT=1" \
       's/LZ_FC_CSA2_IDX/LZ_FC_CSA2_CORE/' 'LZ_FC_CSA2_CORE, n_idx_heads'; then
    M=$(census "$W/mut/e.exe" "$MODEL" "$W/toks")
    # The census prints EVERY site, "- (not run)" for the ones nothing
    # billed, so the mutant's indexer row is present and reads "-" rather
    # than being absent.
    if printf '%s\n' "$M" | grep -q "^csa2 indexer|-"; then
        echo "  mutation (indexer billing renamed onto the core row): the indexer reads '- (not run)', as the header claims"
    else
        echo "  FAIL: the indexer row still carries a count after its billing was renamed away"
        FAILED=1
    fi
else
    FAILED=1
fi

rm -rf "$W"
if [ "$FAILED" -eq 0 ]; then
    echo "PASS csa2_f32count_gate"
    exit 0
fi
echo "FAIL csa2_f32count_gate"
exit 1
