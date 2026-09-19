#!/bin/bash
# F11 (code review, third round): model_walk's CSA2_SPECS tensor walk
# never gated on L->csa2_mode - every csa2_on layer, REINDEX and REUSE
# included, was walked as if it owned all seven csa2_w_* tensors, the
# same way a FULL-mode layer does. forward_csa2.c's own dispatch
# (Task 5.4) already reads a strict per-mode subset (steps 2/3's own
# csa2_mode guards), so this was purely a LOADING-time bug: the
# training side's export_q8.py (a DIFFERENT repository, no shared git
# history - E:\LLM\llama98-zh, commit efd2d34's csa2_suffixes_for) was
# fixed first, to only WRITE the tensors a REINDEX/REUSE layer actually
# owns, and that made this engine's own unconditional reader-side
# assumption visible: a real REINDEX-mode model.bin now fails to load
# at all ("tensor layers.N.self_attn.csa2_w_kv.weight elem count
# mismatch: want <full-shape>, file <a smaller, unrelated tensor's
# bytes, misread>").
#
# THE CONTROL is a MUTATION (src/model.c's CSA2_SPECS walk reverted to
# unconditional), not a git-history diff, same reason every other gate
# on this plan gives. A lightweight, model.c-only link (not the whole
# engine, not the full CLI) is enough - lz_open/lz_open_bin need no
# forward-pass machinery at all, the same fact build/csa2_modes_config_
# gate.sh already established for lz_load_config alone; this is the
# same ENGX list plus lz_open/lz_open_bin's own two extra objects
# (safetensors.c was already there).
#
# THE FIXTURES are REAL exports, not a hand-built LZModel/LZLayer the
# way build/csa2_mode_dispatch_probe.c's own fixtures are (that probe
# calls forward_csa2() directly, bypassing model_walk entirely, and so
# cannot see this bug at all - independent code review found the same
# thing this gate now guards). Built via the TRAINING tree's own real
# export_q8.export(), reusing its existing test fixtures
# (tests/test_csa2_export_modes.py's _cfg/_write_model/_reindex_shapes)
# rather than re-deriving the CSA2 tensor-shape table a third time -
# E:\LLM\llama98-zh is a different repository with no shared git
# history, so this gate's own python embeds the exact same sys.path
# insert build/mut_kernel_matrix.py already uses to reach it.
#
# ROUND 4 (F4): loading is only half the pipeline. Once the mutation
# above is confirmed, the shipped llama98.exe CLI (built separately by
# `make native` - not rebuilt or mutated here, since the defect this
# gate guards is purely a loading-time one and running a forward pass
# through a wrongly-loaded model is not a second, independent question)
# runs a real forward pass over each of the three real fixtures and
# checks every dumped logit is finite, confirming forward_csa2.c's own
# per-mode dispatch actually reads what model_walk actually loaded -
# not merely that loading itself completed.
#
# NOT automated here, and not claimed to be: a one-time manual check,
# during this task's own development, that a real, exported FULL-mode-
# only CSA2 fixture's forward pass is byte-identical before and after
# every round of this task's fixes (git worktree at ca1a594, the commit
# immediately before Part 5 Task 5.4 started) - the correct control,
# unlike an earlier round's mistaken use of Qwen3.5-0.8B, which carries
# no CSA2 keys at all and so never reaches forward_csa2.c either way.
# An EARLIER version of this comment described that comparison as if
# this script reproduced it on every run; it does not - re-running it
# would need a checked-out ca1a594 worktree this gate does not build or
# keep, and the finite-logits check just below is what actually runs
# here, every time.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler, no python3, training
# tree not found at TRAIN_ROOT). The round-4 forward-pass step SKIPs on
# its own (a line, not the whole gate) if llama98.exe/llama98 has not
# been built - loading is still fully verified either way.
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_mode_export_load_gate: SKIP - no C compiler"
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_mode_export_load_gate: SKIP - no python3"
    exit 2
}
TRAIN_ROOT="${TRAIN_ROOT:-E:/LLM/llama98-zh}"
[ -f "$TRAIN_ROOT/tools/export_q8.py" ] || {
    echo "csa2_mode_export_load_gate: SKIP - training tree not found at" \
         "TRAIN_ROOT ($TRAIN_ROOT/tools/export_q8.py missing) - set TRAIN_ROOT"
    exit 2
}

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2modeexportloadgate.$$)
mkdir -p "$W/fixed" "$W/mutant" "$W/fixtures"
trap 'rm -rf "$W"' EXIT

cp src/*.c src/*.h "$W/fixed/" 2>/dev/null
cp src/*.c src/*.h "$W/mutant/" 2>/dev/null

# Sanity: the exact lines the mutation targets must exist once, or this
# gate no longer matches the code it names.
N=$(grep -Fc "            if (L->csa2_mode != LZ_CSA2_REUSE) {" "$W/fixed/model.c")
if [ "$N" != "1" ]; then
    echo "csa2_mode_export_load_gate: FAIL - expected exactly 1 occurrence of" \
         "the CSA2_SPECS mode guard in model.c, found $N (the gate no longer" \
         "matches the source)"
    exit 1
fi

python3 - "$W/mutant/model.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = ("            int k;\n"
       "            if (L->csa2_mode != LZ_CSA2_REUSE) {\n"
       "                int want = (L->csa2_mode == LZ_CSA2_FULL) ?\n"
       "                           CSA2_OWN_FULL : CSA2_OWN_REINDEX;\n"
       "                for (k = 0; k < (int)(sizeof(CSA2_SPECS) / sizeof(CSA2_SPECS[0])); k++) {\n"
       "                    LZTensor *field;\n"
       "                    if (!(CSA2_SPECS[k].own & want)) continue;\n"
       "                    snprintf(name, sizeof(name), \"%slayers.%d.%s\",\n"
       "                             m->prefix, li, CSA2_SPECS[k].suf);\n"
       "                    field = (LZTensor *)((char *)L + CSA2_SPECS[k].field);\n"
       "                    if (model_walk_one(m, name, CSA2_SPECS[k].dims, field,\n"
       "                                       visit, ctx, errbuf, errlen) != 0) return 1;\n"
       "                }\n"
       "            }\n")
new = ("            int k;\n"
       "            for (k = 0; k < (int)(sizeof(CSA2_SPECS) / sizeof(CSA2_SPECS[0])); k++) {\n"
       "                LZTensor *field;\n"
       "                snprintf(name, sizeof(name), \"%slayers.%d.%s\",\n"
       "                         m->prefix, li, CSA2_SPECS[k].suf);\n"
       "                field = (LZTensor *)((char *)L + CSA2_SPECS[k].field);\n"
       "                if (model_walk_one(m, name, CSA2_SPECS[k].dims, field,\n"
       "                                   visit, ctx, errbuf, errlen) != 0) return 1;\n"
       "            }\n")
if src.count(old) != 1:
    print("csa2_mode_export_load_gate: FAIL - could not find exactly one copy "
          "of the CSA2_SPECS mode-gated walk to mutate", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF
if grep -qF "if (L->csa2_mode != LZ_CSA2_REUSE) {" "$W/mutant/model.c"; then
    echo "csa2_mode_export_load_gate: FAIL - could not mutate the mutant's model.c"
    exit 1
fi

# The lightweight, model.c-only link - lz_open/lz_open_bin need no
# forward-pass machinery, same fact build/csa2_modes_config_gate.sh
# already established for lz_load_config alone. safetensors.c/lz4d.c
# cover both loading paths (safetensors and bin, plain or lz4-framed).
ENGX="err compat lfn json safetensors model ops_sched cpucheck lz4d"

build_probe() {
    local dir="$1" label="$2"
    if ! $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            build/csa2_mode_export_load_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_mode_export_load_gate: FAIL - probe did not build against $label"
        tail -n 30 "$dir/cc.log"
        return 2
    fi
    return 0
}

echo "-- building real export fixtures via the training tree's own export_q8.export() --"
( cd "$W/fixtures" && python3 - "$TRAIN_ROOT" <<'PYEOF'
# -*- coding: utf-8 -*-
import sys, os, pathlib
TRAIN = sys.argv[1]
sys.path.insert(0, TRAIN)
sys.path.insert(0, os.path.join(TRAIN, "tests"))

from tools import export_q8
from tests.test_csa2_export_modes import _cfg, _write_model, _reindex_shapes

def build(tag, csa2_layers, csa2_kv_mode, csa2_kv_source):
    cfg = _cfg(csa2_layers=csa2_layers, csa2_kv_mode=csa2_kv_mode,
              csa2_kv_source=csa2_kv_source)
    c = export_q8.resolve_config(cfg)
    h, hd = c["hidden_size"], c["head_dim"]
    idx_hd, idx_heads = c["csa2_indexer_head_dim"], c["csa2_indexer_heads"]
    idx_r = c["csa2_indexer_q_lora_rank"]
    # Layer 1 always stays full mode (the chain's own root) - needs all seven.
    extra = {
        "layers.1.self_attn.csa2_w_kv.weight": [hd, h],
        "layers.1.self_attn.csa2_w_gate.weight": [hd, h],
        "layers.1.self_attn.csa2_w_kv_idx.weight": [idx_hd, h],
        "layers.1.self_attn.csa2_w_gate_idx.weight": [idx_hd, h],
        "layers.1.self_attn.csa2_w_weights.weight": [idx_heads, h],
        "layers.1.self_attn.csa2_w_dq_idx.weight": [idx_r, h],
        "layers.1.self_attn.csa2_w_uq_idx.weight": [idx_heads * idx_hd, idx_r],
    }
    if 2 in csa2_layers and csa2_kv_mode.get(2) == "reindex":
        extra.update(_reindex_shapes(c, 2))
    # A reuse-mode layer (3, when present) needs no tensors of its own at all.
    dirpath = pathlib.Path(".") / tag
    _write_model(dirpath, cfg, extra)
    export_q8.export(str(dirpath), str(dirpath / "model.bin"))
    print("built", tag)

build("control_full", [1], {}, {})
build("reindex", [1, 2], {2: "reindex"}, {2: 1})
build("reuse", [1, 2, 3], {2: "reindex", 3: "reuse"}, {2: 1, 3: 2})
PYEOF
) || { echo "csa2_mode_export_load_gate: FAIL - fixture export failed"; exit 1; }

echo ""
echo "-- current tree (must load all three) --"
build_probe "$W/fixed" "the current (fixed) tree" || exit 1
RC=0
for tag in control_full reindex reuse; do
    out=$("$W/fixed/probe" "$W/fixtures/$tag" 2>&1)
    echo "  $tag: $out"
    case "$out" in
        PASS:*) ;;
        *) echo "csa2_mode_export_load_gate: FAIL - fixed tree did not load $tag"; RC=1 ;;
    esac
done

echo ""
echo "-- mutant tree: CSA2_SPECS walk reverted to unconditional (reindex/reuse must FAIL, control_full must still PASS) --"
build_probe "$W/mutant" "the reverted-CSA2_SPECS mutant" || exit 1
out=$("$W/mutant/probe" "$W/fixtures/control_full" 2>&1)
echo "  control_full: $out"
case "$out" in
    PASS:*) ;;
    *) echo "csa2_mode_export_load_gate: FAIL - mutant lost an unrelated PASS (control_full)"; RC=1 ;;
esac
for tag in reindex reuse; do
    out=$("$W/mutant/probe" "$W/fixtures/$tag" 2>&1)
    echo "  $tag: $out"
    case "$out" in
        FAIL:*csa2_w_kv*elem\ count\ mismatch*) ;;
        FAIL:*) echo "csa2_mode_export_load_gate: FAIL - mutant failed $tag for the wrong reason: $out"; RC=1 ;;
        *) echo "csa2_mode_export_load_gate: FAIL - mutant PASSED $tag; the probe cannot see this defect"; RC=1 ;;
    esac
done

# Code-review round 4 (F4): loading is only half the pipeline
# build/csa2_mode_dispatch_probe.c's own hand-built LZModel/LZLayer
# fixtures never exercise - the REAL, shipped llama98.exe CLI (already
# built, not rebuilt here: this step needs no mutant, since the
# mutation above is purely a LOADING-time defect and a forward pass
# through a wrongly-loaded model is not a meaningful second question)
# runs a real forward pass over each of the three real, exported
# fixtures and checks for finite logits - the loader's own tensors,
# actually read by forward_csa2.c's real per-mode dispatch, not merely
# structurally present.
BIN="$(pwd)/llama98.exe"
[ -x "$BIN" ] || BIN="$(pwd)/llama98"
if [ ! -x "$BIN" ]; then
    echo ""
    echo "csa2_mode_export_load_gate: SKIP - no $BIN (run: make native) -" \
         "loading is verified above regardless; only the real forward-pass" \
         "step needs the shipped CLI"
else
    echo ""
    echo "-- real forward pass through the shipped CLI (loading is not the whole pipeline) --"
    for tag in control_full reindex reuse; do
        # cd into $W and use RELATIVE paths from here on, not $W/... -
        # this task's own round-4 history (build/csa2_mode_export_load_
        # gate.sh's git log) already hit native-Windows-python
        # interpreting a bash-resolved /tmp/... string as drive-relative
        # to cwd instead of MSYS's own mapping; the python3 call two
        # lines below is the SAME trap, reached again by embedding $W
        # directly into a python string instead of cd-ing first.
        ( cd "$W" && "$BIN" "fixtures/$tag" --tokens 1,2,3 --dump-all-logits "$tag.logits" >"$tag.log" 2>&1 )
        if [ ! -s "$W/$tag.logits" ]; then
            echo "csa2_mode_export_load_gate: FAIL - $tag: forward pass produced no logits"
            tail -10 "$W/$tag.log"
            RC=1
        # Reads the DUMPED LOGITS FILE itself, not the CLI's own stdout/
        # stderr text - an earlier version of this check grepped that
        # text for the literal substrings "nan"/"inf", which only a
        # diagnostic PRINTED BY THE CLI could ever contain; a non-finite
        # float32 in the binary dump has no ASCII "nan" in it anywhere,
        # so that version could never actually fail on the one thing its
        # own message claimed to check. nll_paired.c's own slurp() is
        # the same raw-float32-array read, in C rather than Python -
        # reused here as a python3 one-liner since this script already
        # requires python3 (the fixture-export step above) and does not
        # otherwise need a C compile step just for this.
        elif ! ( cd "$W" && python3 -c "
import struct, sys, math
with open('$tag.logits', 'rb') as f:
    data = f.read()
n = len(data) // 4
vals = struct.unpack('<%df' % n, data[:n * 4])
sys.exit(1 if any(not math.isfinite(v) for v in vals) else 0)
" ); then
            echo "csa2_mode_export_load_gate: FAIL - $tag: non-finite value in the dumped logits file"
            RC=1
        else
            echo "  $tag: PASS (forward pass ran, every dumped logit finite)"
        fi
    done
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_mode_export_load_gate: FAIL"
    exit 1
fi
echo "csa2_mode_export_load_gate: PASS"
exit 0
