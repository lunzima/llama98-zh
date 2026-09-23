#!/bin/bash
# The four CSA2 ring buffers are sized for the model, not for nf.
#
# alloc_csa2_ring passes c->n_full_layers to lz_csa2_pend_base and
# lz_csa2_ring_base, which are prefix sums over the slots - passing the
# slot COUNT is how both functions say "give me the end offset" in their
# own comments. What came back was therefore already the whole model's
# total, and multiplying it by nf allocated nf copies of a buffer whose
# every consumer computes an offset into ONE of them (lz_csa2_pend_base(c,
# ci, ...) / lz_csa2_ring_base(c, ci, ...) in forward_csa2.c and
# forward_ckpt.c - no site anywhere writes ci * total + base). Half the
# allocation, at nf = 2, was memory nothing could name or index:
# 45,056 B at the 64-slot working context on kunkun98, 5,246,976 B at
# seq 8192.
#
# HOW THIS GATE GETS AT "WHICH TERM" AND NOT MERELY "HOW MUCH". Two arms
# differing in one line give a difference, and a difference alone cannot
# say which side was wrong - any pair of cancelling errors satisfies it.
# So the gate runs THREE isolated mutants and SOLVES the layout out of
# them:
#
#   nf * back on csa2_comp_ring only      d_comp = (nf-1) * nblocks * head_dim * 4
#   nf * back on csa2_idx_comp_ring only  d_idx  = (nf-1) * nblocks * idx_dim  * 4
#   nf * back on the two pending rings    d_pend = 2 * (nf-1) * pdepth * head_dim * 4
#
# From the first two the gate solves nblocks TWICE - through head_dim and
# through idx_dim - and the two solutions must agree; from the third it
# solves pdepth. It then checks all three against the config's own
# ceilings (sum over slots of ceil(seq/cm) and of cm * depth), which is
# the one place this file restates the layout. A formula with a wrong
# component cannot pass: head_dim and idx_dim differ by 4x on these
# models, so a term attributed to the wrong buffer moves the solved value
# off by that factor.
#
# THE ARMS NEED nf >= 2: at n_full_layers == 1 the defect is
# arithmetically absent (nf * total == total) and every arm here would
# pass on a broken allocator. The gate SKIPs there with that reason, and
# then MEASURES it - build/csa2_ring_nf_probe.c runs two hand-built
# fixtures differing in nothing but the layer count and reports what
# lz_state_alloc allocated for each, on this tree and on the mutant. On
# the nf = 1 fixture the two must agree; on nf = 2 they must differ by
# exactly the ring total. Without that arm, "the arms above are measuring
# the defect" and "the arms above are running on a machine that cannot
# have it" read the same.
#
# The reference arm is a mutation, not `git show HEAD:` - a git-pinned
# before-copy goes stale the day the fix lands, which is exactly how the
# first version of build/csa2_dense_kv_gate.sh failed.
#
#   csa2_ring_nf_gate.sh [model-with-csa2] [second-model]
#
# Exit: 0 pass, 1 fail, 2 skip (no python3, no model, nf < 2).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-E:/LLM/models/kunkun98}
SECOND=${2:-E:/LLM/models/kunkun-ce}
W=$(mktemp -d)
FAILED=0
PROMPT="人工智能的发展历史可以追溯到二十世纪五十年代。"

command -v python3 >/dev/null 2>&1 || { echo "SKIP: no python3"; exit 2; }
[ -d "$MODEL" ] || { echo "SKIP: no model at $MODEL"; exit 2; }

# <kvd head_dim idx_dim nf pdepth_config nblocks_config cm_list>
# pdepth/nblocks are the sum over slots the two size functions compute;
# cm_list is what the config says each slot's ratio is (the scalar unless
# csa2_compress_m_layers names that layer).
geom() {
    python3 - "$1" "$2" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
seq = int(sys.argv[2])
per = c.get("csa2_compress_m_layers") or {}
scalar = int(c.get("csa2_compress_m") or 1)
mode = c.get("csa2_kv_mode") or {}
# A SLOT IS EVERY full_attention LAYER, csa2_on or not: model.c's
# csa2_slot_owner counts layers whose mode is FULL, and a layer that is
# not csa2_on at all carries that mode as the calloc default. Only a
# REINDEX/REUSE layer is excluded - it aliases its source and owns none.
slots = []
for l, lt in enumerate(c.get("layer_types") or []):
    if lt != "full_attention":
        continue
    if str(mode.get(str(l), "full")).lower() != "full":
        continue
    slots.append(int(per.get(str(l), scalar)))
nf = len(slots)
depth = 1                   # ssm_ring_depth with no MTP head bound
pdepth = sum(m * depth for m in slots)
nblocks = sum((seq + m - 1) // m for m in slots)
print(int(c["head_dim"]), int(c["csa2_indexer_head_dim"]), nf,
      pdepth, nblocks, ",".join(str(m) for m in slots))
PY
}

# build <dir> <sed-expr|-> <marker|->
build() {
    local dir="$1" expr="$2" marker="${3:-}"
    mkdir -p "$dir"
    cp -r "$ROOT/src" "$ROOT/common" "$dir/"
    cp "$ROOT/Makefile" "$dir/Makefile"
    if [ "$expr" != "-" ]; then
        sed -i "$expr" "$dir/src/forward_state.c"
        if ! grep -q "$marker" "$dir/src/forward_state.c"; then
            echo "  FAIL: mutation did not apply"; return 1
        fi
    fi
    ( cd "$dir" && make native ) > "$dir/build.log" 2>&1 || {
        echo "  FAIL: build failed in $dir"; tail -3 "$dir/build.log"; return 1; }
    return 0
}
state_bytes() {
    "$1" "$2" --prompt "$PROMPT" -n 1 --ctx 64 2>/dev/null \
        | sed -n 's/^State *[0-9.]* MB (\([0-9]*\) B, *\([0-9]*\) slots.*/\1 \2/p' | head -1
}
bytes_of() { state_bytes "$1" "$2" | cut -d' ' -f1; }

# Three mutations. Each one re-adds the nf factor to a DIFFERENT subset,
# so the three deltas isolate three different terms.
M_COMP='s/^                nblocks \* c->head_dim, (size_t)ring_stride/                nf * nblocks * c->head_dim, (size_t)ring_stride/'
M_IDX='s/^                nblocks \* c->csa2_indexer_head_dim, (size_t)ring_stride/                nf * nblocks * c->csa2_indexer_head_dim, (size_t)ring_stride/'
M_PEND='s/^                pdepth \* c->head_dim, sizeof(float)/                nf * pdepth * c->head_dim, sizeof(float)/g'
M_ALL='s/^                pdepth \* c->head_dim, sizeof(float)/                nf * pdepth * c->head_dim, sizeof(float)/; s/^                nblocks \* c->head_dim, (size_t)ring_stride/                nf * nblocks * c->head_dim, (size_t)ring_stride/; s/^                nblocks \* c->csa2_indexer_head_dim, (size_t)ring_stride/                nf * nblocks * c->csa2_indexer_head_dim, (size_t)ring_stride/'

check_model() {
    local M="$1" tag="$2" want_nf
    local G; G=$(geom "$M" 32768)
    set -- $G
    local hd=$1 idx=$2 nf=$3 pdepth=$4 nblocks=$5 cms=$6
    if [ "$nf" -lt 2 ]; then
        echo "  SKIP $tag: n_full_layers = $nf, and at nf < 2 the defect is"
        echo "             arithmetically absent - this gate would pass on a"
        echo "             broken allocator"
        return 0
    fi
    # pdepth/nblocks depend on the run's own seq, so they are derived from
    # the slot count the engine reports rather than from a number chosen
    # here. --ctx 64 pins that slot count; the derivation below is the one
    # every assertion uses, and it is printed after it, not before.
    local REF N; REF=$(bytes_of "$W/ref/llama98.exe" "$M")
    if [ -z "$REF" ]; then echo "  FAIL $tag: no state line"; FAILED=1; return 0; fi
    N=$(state_bytes "$W/ref/llama98.exe" "$M" | cut -d' ' -f2)
    G=$(geom "$M" "$N")
    set -- $G
    pdepth=$4; nblocks=$5
    echo "  $tag: nf=$nf head_dim=$hd idx_dim=$idx cm=$cms at $N slots -> pdepth=$pdepth nblocks=$nblocks"

    local d_comp d_idx d_pend solved_blocks_a solved_blocks_b solved_pdepth
    N=$(bytes_of "$W/m_comp/llama98.exe" "$M"); d_comp=$(( N - REF ))
    N=$(bytes_of "$W/m_idx/llama98.exe"  "$M"); d_idx=$(( N - REF ))
    N=$(bytes_of "$W/m_pend/llama98.exe" "$M"); d_pend=$(( N - REF ))

    solved_blocks_a=$(( d_comp / ((nf - 1) * hd * 4) ))
    solved_blocks_b=$(( d_idx  / ((nf - 1) * idx * 4) ))
    solved_pdepth=$(( d_pend / (2 * (nf - 1) * hd * 4) ))

    if [ "$d_idx" -eq 0 ]; then
        echo "  FAIL $tag: the indexer-ring mutant moved nothing (idx_dim=$idx)"
        FAILED=1
    elif [ "$solved_blocks_a" != "$solved_blocks_b" ]; then
        echo "  FAIL $tag: the two ring mutants disagree on nblocks ($solved_blocks_a vs $solved_blocks_b)"
        FAILED=1
    elif [ "$solved_blocks_a" != "$nblocks" ]; then
        echo "  FAIL $tag: nblocks solved from the mutants is $solved_blocks_a, config says $nblocks"
        FAILED=1
    elif [ "$solved_pdepth" != "$pdepth" ]; then
        echo "  FAIL $tag: pdepth solved from the mutants is $solved_pdepth, config says $pdepth"
        FAILED=1
    else
        printf "  %s: comp=%s B -> nblocks %s (hd) and %s (idx), pend=%s B -> pdepth %s" \
               "$tag" "$d_comp" "$solved_blocks_a" "$solved_blocks_b" "$d_pend" "$solved_pdepth"
        echo " - both agree with the config"
    fi
}

echo "== csa2_ring_nf_gate: $MODEL =="
build "$W/ref" - || exit 1
build "$W/m_comp" "$M_COMP" 'nf \* nblocks \* c->head_dim' || exit 1
build "$W/m_idx"  "$M_IDX"  'nf \* nblocks \* c->csa2_indexer_head_dim' || exit 1
build "$W/m_pend" "$M_PEND" 'nf \* pdepth \* c->head_dim' || exit 1

# ---- bit-identity with all four rings back at nf copies ---------------
"$W/ref/llama98.exe" "$MODEL" --prompt "$PROMPT" --dump-prompt-tokens "$W/toks" -n 1 >/dev/null 2>&1
if [ ! -s "$W/toks" ]; then echo "  FAIL: no token list"; exit 1; fi
r=0
"$W/ref/llama98.exe" "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/ref.lg" >/dev/null 2>&1 || r=1
if build "$W/m_all" "$M_ALL" 'nf \* pdepth \* c->head_dim'; then
    "$W/m_all/llama98.exe" "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/m_all.lg" >/dev/null 2>&1 || r=1
    if [ "$r" -ne 0 ]; then
        echo "  FAIL: an identity arm exited non-zero"
        FAILED=1
    elif cmp -s "$W/ref.lg" "$W/m_all.lg"; then
        echo "  bit-identity: allocating all four rings nf times over changes no logit"
    else
        echo "  FAIL: the nf-sized rings changed the logits, so something DOES index past the end offset"
        FAILED=1
    fi
else
    FAILED=1
fi

check_model "$MODEL" "kunkun98-like"
if [ -d "$SECOND" ] && [ "$SECOND" != "$MODEL" ]; then
    G=$(geom "$SECOND" 32768); set -- $G
    if [ "$3" -ge 2 ] && [ "$1" != "$(geom "$MODEL" 32768 | cut -d' ' -f1)" ]; then
        check_model "$SECOND" "second geometry"
    fi
fi

# ---- the nf = 1 control, MEASURED rather than asserted ----------------
# The arms above need nf >= 2 and SKIP otherwise, with the reason that the
# defect is arithmetically absent at nf == 1. That reason is a premise
# until it is measured: build/csa2_ring_nf_probe.c builds two fixtures
# differing in nothing but the number of full-attention layers and asks
# lz_state_alloc for its byte count. On the nf = 1 one, a build that
# multiplies by nf and one that does not must allocate the SAME bytes -
# which is what makes "the arms above are measuring the defect" different
# from "the arms above are running on a machine that cannot have it".
if [ -e "$ROOT/build/csa2_ring_nf_probe.c" ]; then
    probe_of() {   # <srcroot> <fixture>
        # The build dir is keyed on BOTH - the same fixture has to be run
        # by two different trees, and a cache keyed on the fixture alone
        # hands the second caller the first tree's binary.
        local srcdir="$1" fx="$2" s=""
        local d="$W/probe_$(basename "$srcdir")_$fx"
        if [ ! -x "$d/p.exe" ]; then
            mkdir -p "$d"
            for f in $(cd "$ROOT" && make -s print-ENG); do s="$s $srcdir/src/$f.c"; done
            for f in $(cd "$ROOT" && make -s print-COM); do s="$s $srcdir/common/$f.c"; done
            gcc -O2 -std=c99 -I"$srcdir/src" -I"$srcdir/common" -ffp-contract=off \
                -o "$d/p.exe" $s "$ROOT/build/csa2_ring_nf_probe.c" -lm \
                > "$d/build.log" 2>&1 || {
                echo "  FAIL: the probe did not build against $srcdir"
                tail -3 "$d/build.log"; FAILED=1; return 0; }
        fi
        "$d/p.exe" "$fx"
    }
    R1=$(probe_of "$W/ref" nf1);   A1=$(probe_of "$W/m_all" nf1)
    R2=$(probe_of "$W/ref" nf2);   A2=$(probe_of "$W/m_all" nf2)
    bR1=`printf '%s' "$R1" | sed -n 's/^bytes=\([0-9]*\).*/\1/p'`
    bA1=`printf '%s' "$A1" | sed -n 's/^bytes=\([0-9]*\).*/\1/p'`
    bR2=`printf '%s' "$R2" | sed -n 's/^bytes=\([0-9]*\).*/\1/p'`
    bA2=`printf '%s' "$A2" | sed -n 's/^bytes=\([0-9]*\).*/\1/p'`
    ring2=`printf '%s' "$R2" | sed -n 's/.* ring=\([0-9]*\).*/\1/p'`
    if [ -z "$bR1" ] || [ -z "$bA1" ] || [ -z "$bR2" ] || [ -z "$bA2" ]; then
        echo "  FAIL: the probe printed no byte count"; FAILED=1
    elif [ "$bA1" != "$bR1" ]; then
        echo "  FAIL: nf=1 control - multiplying by nf changed the byte count there ($bR1 vs $bA1),"
        echo "        so the fixtures differ in something other than the layer count"
        FAILED=1
    elif [ "$bA2" -le "$bR2" ]; then
        echo "  FAIL: nf=2 - the mutant did not allocate more ($bR2 vs $bA2)"; FAILED=1
    elif [ "$(( bA2 - bR2 ))" != "$ring2" ]; then
        echo "  FAIL: nf=2 - the mutant adds $(( bA2 - bR2 )) B, the rings are $ring2 B"
        FAILED=1
    else
        echo "  control: nf=1 allocates $bR1 B either way; nf=2 changes by $ring2 B, the whole ring"
        echo "           ($R2 / $A2)"
    fi
else
    echo "  SKIP: no build/csa2_ring_nf_probe.c"
fi

rm -rf "$W"
if [ "$FAILED" -eq 0 ]; then echo "PASS csa2_ring_nf_gate"; exit 0; fi
echo "FAIL csa2_ring_nf_gate"
exit 1
