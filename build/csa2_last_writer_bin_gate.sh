#!/bin/bash
# C1 (report Sec2.3.1 mode-dispatch task, fifth review round) AND ITS
# OWN SEQUEL, C-NEW-1/C-NEW-2 (sixth review round): three SEPARATE
# `for (i = 0; i < c->n_layers; i++)` loops in lz_open_bin (model.c) -
# lz_csa2_last_writer_valid's own last_writer[]/csa2_on[]/csa2_mode[]/
# csa2_source[] walk (C1), the n_full_layers/n_linear_layers cross-
# check against layer_types[] (C-NEW-1, added THIS TASK to close a
# Minor item), and the S4+CSA2 csa2_on[] scan (C-NEW-2, added THIS TASK
# to close I3) - each trusted c->n_layers as a safe loop bound before it
# had ever been range-checked. C-NEW-1/C-NEW-2 are proof this shape of
# bug does not go away one instance at a time: fixing C1 by relocating
# ONE call site left the underlying hazard (nothing stops n_layers from
# being arbitrary at the top of this function) in place for the very
# next loop of this shape to fall into, twice, in the immediately
# following review round.
#
# THE FIX IS STRUCTURAL, not three more individually-relocated checks:
# `c->n_layers <= 0 || c->n_layers > LZ_MAX_LAYERS` is now refused the
# instant RD(&c->n_layers, 4) returns - model.c's own comment at that
# read site has the full argument for why the check belongs THERE
# rather than near any one of the loops it protects. Every c->n_layers-
# indexed loop in the rest of this function, present or future,
# inherits the bound for free; nothing downstream has to remember its
# own position relative to the check, which lives at the read site and
# stays there.
#
# THE TEST VALUE is n_layers=0x7FFFFFFF, not a small-looking overflow
# like 200: both this round's spec-compliance and code-quality
# reviewers used it independently (one via constructed-header/gdb
# backtrace, the other via differential binary comparison) and it is
# what actually walks OUT of the whole LZModelConfig struct, not merely
# past one 128-wide array into another one still inside it - the
# earlier version of this gate used n_layers=200 paired with a
# csa2_mode[]/csa2_source[] aliasing trick (the two arrays are
# adjacent in LZModelConfig) to reach exactly one of the three loops
# above; 0x7FFFFFFF needs no such trick and is refused by the single
# early check regardless of any other field's value, which is the
# point - the fix protects all three loops (and any future one) with
# ONE check, not three still requiring their own bespoke exploit.
#
# I4 (fifth review round, "third time this exact pattern has appeared"):
# build/csa2_mode_dispatch_probe.c's own check_bin_validation calls
# lz_csa2_mode_source_valid/lz_csa2_last_writer_valid directly against a
# hand-built LZModelConfig - never through lz_open_bin's own byte-stream
# parsing, which is exactly where C1/C-NEW-1/C-NEW-2 all lived and
# exactly why that probe could not see any of them. THE SECOND CHECK
# below closes that gap with a real byte stream through the real
# loader, for a DIFFERENT (safely in-bounds) corruption; THE FIRST
# CHECK is C1/C-NEW-1/C-NEW-2's own shared regression test.
#
# THIS RUNS OUT OF PROCESS, not in-process: the malformed header this
# gate exists to catch, before its own fix, corrupts memory arbitrarily
# (not merely "crashes cleanly") - running it through a hand-built
# LZModelConfig call from WITHIN this project's own probe binary would
# risk corrupting the probe's own test harness state rather than
# producing a clean, isolated failure. The real llama98.exe CLI, run as
# a CHILD PROCESS and inspected only by exit code/output text, is what
# makes a still-present defect a reported FAIL instead of a crashed
# gate script.
#
# THE FIXTURE is a REAL export (export_q8.export(), the training tree's
# actual production exporter - E:\LLM\llama98-zh, a different
# repository with no shared git history), not a hand-packed byte
# buffer: a 4-layer FULL/REINDEX/REUSE chain
# (csa2_layers=[1,2,3], csa2_kv_mode={2:"reindex",3:"reuse"},
# csa2_kv_source={2:1,3:2}), reusing tests/test_csa2_export_modes.py's
# own _cfg/_write_model/_reindex_shapes fixture helpers rather than
# re-deriving the CSA2 tensor-shape table a third time.
#
# csa2_mode[]/csa2_source[] (the I4 check's own target field lives in
# csa2_mode[]) are located by SIGNATURE SEARCH, not a hardcoded byte
# offset - csa2_source[]'s own 24-byte signature ([-1,-1,1,2,-1,-1],
# unique to this fixture) is found first, and csa2_mode[]'s array is
# exactly the LZ_MAX_LAYERS*4 (512) bytes immediately before it in the
# file (model.c's own RD() call order: csa2_mode written immediately
# before csa2_source, both LZ_MAX_LAYERS wide, no gap). A header-layout
# change that moves either array is a loud failure here (the signature
# search reports how many times it matched), not a silent patch of the
# wrong field.
#
# Exit: 0 both checks pass (fixed), 1 either reproduces, 2 skip (no
# python3, no training tree, no built CLI).
set -u
cd "$(dirname "$0")/.." || exit 1

eval "$(LZ_TREE_LOCK_NAME="$(basename "$0")" LZ_TREE_LOCK_WHAT="the last-writer bin gate" bash build/tree_lock.sh acquire --print-env)" || exit 2
ENG="$(pwd)"
TRAIN="${1:-E:/LLM/llama98-zh}"

command -v python3 >/dev/null 2>&1 || {
    echo "csa2_last_writer_bin_gate: SKIP - no python3"; exit 2; }
[ -f "$TRAIN/tools/export_q8.py" ] || {
    echo "csa2_last_writer_bin_gate: SKIP - no training tree at $TRAIN"; exit 2; }
BIN="$ENG/llama98.exe"; [ -x "$BIN" ] || BIN="$ENG/llama98"
[ -x "$BIN" ] || {
    echo "csa2_last_writer_bin_gate: SKIP - no $BIN (run: make native)"; exit 2; }
CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_last_writer_bin_gate: SKIP - no C compiler (needed for the" \
         "I3 bin-path mutation arm)"; exit 2; }

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2lastwriterbin.$$)
trap 'bash build/tree_lock.sh release; rm -rf "$W"' EXIT

# Shared by every mutation arm below: cp src/*.c/*.h + common/*.c/*.h
# into $1 (the caller has already mutated $1/src/model.c before calling
# this), then link exactly `make native`'s own line - the COLD sources
# (compiled fresh every build, per the Makefile's own COLD_ENG_SRCS/
# COLD_CLI_COM_SRCS split) from $1, the HOT objects from build/native/*.o
# reused as-is, since a model.c-only mutation never touches them.
build_mutant_cli() {
    local dir="$1" label="$2"
    if $CC -Os -march=x86-64 -mtune=znver3 -std=c99 -w -I"$dir/src" \
            -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -I"$dir/common" -o "$dir/llama98" \
            "$dir/src/err.c" "$dir/src/compat.c" "$dir/src/lfn.c" "$dir/src/json.c" \
            "$dir/src/safetensors.c" "$dir/src/model.c" "$dir/src/ops_kernel.c" \
            "$dir/src/ops_t2_arm.c" "$dir/src/ops_arm.c" "$dir/src/ops_t2_scalar.c" \
            "$dir/src/fwht.c" "$dir/src/lz_mathf.c" "$dir/src/lz_bf16.c" \
            "$dir/src/lz4d.c" "$dir/src/tokenizer.c" "$dir/src/unicode.c" \
            "$dir/src/chat.c" "$dir/src/gbk.c" "$dir/src/cpucheck.c" \
            "$dir/common/session.c" "$dir/common/beep.c" \
            "$dir/src/cli_attr.c" "$dir/src/cli_main.c" \
            build/native/ops.o build/native/ops_epi.o build/native/ops_matmul.o \
            build/native/ops_quant.o build/native/ops_gdn.o build/native/ops_norm.o \
            build/native/ops_sched.o build/native/ops_conv1d.o build/native/ops_moe.o \
            build/native/ops_rope.o build/native/forward.o build/native/forward_attn.o \
            build/native/forward_kda.o build/native/forward_moe.o build/native/forward_ssm.o \
            build/native/forward_engram.o build/native/forward_csa2.o build/native/forward_mtp.o \
            build/native/forward_rope.o build/native/forward_ckpt.o build/native/forward_state.o \
            build/native/sampler.o build/native/generate.o build/native/stream.o \
            build/native/ops_mmx.o build/native/ops_mmx_sse.o build/native/ops_sse2.o \
            build/native/ops_avx2.o -lm 2>"$dir/cc.log"; then
        return 0
    fi
    echo "csa2_last_writer_bin_gate: FAIL - mutant llama98 ($label) did not" \
         "build (build/native/*.o must already exist - run 'make native' first)"
    tail -20 "$dir/cc.log"
    return 1
}

( cd "$W" && python3 - "$TRAIN" <<'PYEOF'
import sys, os, pathlib
TRAIN = sys.argv[1]
sys.path.insert(0, TRAIN); sys.path.insert(0, os.path.join(TRAIN, "tests"))
from tools import export_q8
from tests.test_csa2_export_modes import _cfg, _write_model, _reindex_shapes
cfg = _cfg(csa2_layers=[1,2,3], csa2_kv_mode={2:"reindex",3:"reuse"},
           csa2_kv_source={2:1,3:2})
c = export_q8.resolve_config(cfg)
h, hd = c["hidden_size"], c["head_dim"]
idx_hd, idx_heads, idx_r = (c["csa2_indexer_head_dim"], c["csa2_indexer_heads"],
                            c["csa2_indexer_q_lora_rank"])
extra = {
    "layers.1.self_attn.csa2_w_kv.weight":       [hd, h],
    "layers.1.self_attn.csa2_w_gate.weight":     [hd, h],
    "layers.1.self_attn.csa2_w_kv_idx.weight":   [idx_hd, hd],
    "layers.1.self_attn.csa2_k_norm_idx.weight": [idx_hd],
    "layers.1.self_attn.csa2_w_weights.weight":  [idx_heads, h],
    "layers.1.self_attn.csa2_w_dq_idx.weight":   [idx_r, h],
    "layers.1.self_attn.csa2_w_uq_idx.weight":   [idx_heads*idx_hd, idx_r],
}
extra.update(_reindex_shapes(c, 2))
d = pathlib.Path("good"); _write_model(d, cfg, extra)
export_q8.export(str(d), str(d/"model.bin"))

# s4csa2 (I3's OWN bin-path fixture): mla_q_lora_rank=32 + csa2_layers=[1]
# (FULL mode, the default - I3's refusal is mode-independent) is a
# combination the TRAINING side supports deliberately (tests/test_
# csa2_export_modes.py's own test_full_with_s4_composition_also_drops_
# dq_idx pins the exact tensor list this drops csa2_w_dq_idx from) -
# this is a REAL, legitimately-exportable checkpoint on that side, not
# a hand-patched byte pattern, which is exactly why this engine's own
# refusal has to be explicit: nothing about the bytes themselves looks
# malformed, only the COMBINATION is unsupported here.
cfg2 = _cfg(csa2_layers=[1], mla_q_lora_rank=32)
c2 = export_q8.resolve_config(cfg2)
h2 = c2["hidden_size"]
idx_hd2, idx_heads2 = c2["csa2_indexer_head_dim"], c2["csa2_indexer_heads"]
idx_r2 = c2["csa2_indexer_q_lora_rank"]
qgd2, qr2 = c2["attn_qgate_dim"], 32
extra2 = {
    "layers.1.self_attn.csa2_w_kv.weight":       [c2["head_dim"], h2],
    "layers.1.self_attn.csa2_w_gate.weight":     [c2["head_dim"], h2],
    "layers.1.self_attn.csa2_w_kv_idx.weight":   [idx_hd2, c2["head_dim"]],
    "layers.1.self_attn.csa2_k_norm_idx.weight": [idx_hd2],
    "layers.1.self_attn.csa2_w_weights.weight":  [idx_heads2, h2],
    # csa2_w_uq_idx's second dim is csa2_indexer_q_lora_rank - the
    # INDEXER's own down-projection rank, unrelated to mla_q_lora_rank
    # (S4's shared query rank, used below for q_proj.down/up only).
    "layers.1.self_attn.csa2_w_uq_idx.weight":   [idx_heads2*idx_hd2, idx_r2],
    # NO csa2_w_dq_idx - S4 composition drops it on the training side;
    # this engine never reads that field for this fixture either way,
    # since the refusal this fixture exists to test happens before any
    # tensor is ever walked.
}
# mla_q_lora_rank is a GLOBAL config field (model.c's present_mla_q),
# not per-layer - EVERY full_attention layer (1, 2, 3 in this 4-layer
# fixture, per _cfg's own MINI_CONFIG-derived layer_types), csa2_on or
# not, gets q_proj.down/up instead of q_proj. _write_model's own
# default synthesis has no S4 awareness, so all three need their shapes
# named explicitly here or export_q8.export()'s own shape_of() lookup
# fails on the first one it walks.
for li in (1, 2, 3):
    extra2["layers.%d.self_attn.q_proj.down.weight" % li] = [qr2, h2]
    extra2["layers.%d.self_attn.q_proj.up.weight" % li] = [qgd2, qr2]
# _write_model()'s own shapes dict is _mini_shapes(cfg) (no S4
# awareness - unconditionally writes q_proj.weight for every full_
# attention layer) .update()'d with csa2_extra_shapes - update() ADDS
# q_proj.down/up.weight, it does not REMOVE the now-orphaned plain
# q_proj.weight the base dict already put there, which export_q8's own
# "every tensor must be claimed" check then refuses. Built here by
# hand instead of through _write_model for exactly that reason - the
# three plain q_proj.weight entries are deleted before writing, not
# left to become unclaimed tensors export needs --allow-unexported to
# ignore.
import json as _json, struct as _struct
import numpy as _np
from tests.test_export_pad import _mini_shapes
d2 = pathlib.Path("s4csa2")
d2.mkdir(parents=True, exist_ok=True)
(d2 / "config.json").write_text(_json.dumps(cfg2), encoding="utf-8")
shapes2 = dict(_mini_shapes(cfg2))
for li in (1, 2, 3):
    del shapes2["layers.%d.self_attn.q_proj.weight" % li]
shapes2.update(extra2)
rng2 = _np.random.default_rng(11)
head2, blob2, off2 = {}, bytearray(), 0
for name, shape in shapes2.items():
    a = rng2.normal(size=shape).astype(_np.float32)
    raw = a.tobytes()
    head2["model." + name] = {"dtype": "F32", "shape": list(shape),
                              "data_offsets": [off2, off2 + len(raw)]}
    blob2 += raw
    off2 += len(raw)
hj2 = _json.dumps(head2).encode("utf-8")
with open(d2 / "model.safetensors", "wb") as f:
    f.write(_struct.pack("<Q", len(hj2)))
    f.write(hj2)
    f.write(blob2)
export_q8.export(str(d2), str(d2/"model.bin"))
PYEOF
) >"$W/export.log" 2>&1 || {
    echo "csa2_last_writer_bin_gate: FAIL - fixture export failed"
    tail -5 "$W/export.log"; exit 1; }
rm -f "$W/s4csa2/model.safetensors"

cp -r "$W/good" "$W/bad_overflow" && rm -f "$W/bad_overflow/model.safetensors"
cp -r "$W/good" "$W/bad_mode" && rm -f "$W/bad_mode/model.safetensors"
cp -r "$W/good" "$W/bad_lastwriter" && rm -f "$W/bad_lastwriter/model.safetensors"

python3 - "$W/bad_overflow/model.bin" "$W/bad_mode/model.bin" "$W/bad_lastwriter/model.bin" <<'PYEOF' || exit 1
import re, struct, sys
p_overflow, p_mode, p_lastwriter = sys.argv[1], sys.argv[2], sys.argv[3]

def find_csa2_source(d):
    # csa2_source[] signature for this fixture: [-1,-1,1,2,-1,-1]
    sig = (b"\xff\xff\xff\xff\xff\xff\xff\xff\x01\x00\x00\x00\x02\x00\x00\x00"
           b"\xff\xff\xff\xff\xff\xff\xff\xff")
    locs = [m.start() for m in re.finditer(re.escape(sig), bytes(d))]
    if len(locs) != 1:
        print("csa2_last_writer_bin_gate: FAIL - csa2_source[] signature "
              "matched %d times, not 1 (header layout changed)" % len(locs),
              file=sys.stderr)
        sys.exit(1)
    return locs[0]

LZ_MAX_LAYERS = 128

# bad_overflow (C1/C-NEW-1/C-NEW-2's shared regression test): n_layers
# set to 0x7FFFFFFF (byte 16) - no other field needs to change. Before
# the structural fix this walked OUT of the whole LZModelConfig struct
# in any of three separate for-loops (see this script's own header);
# after it, RD(&c->n_layers, 4)'s own immediate range check refuses the
# header before any of those loops - or anything else in this function
# - ever runs.
d = bytearray(open(p_overflow, "rb").read())
if struct.unpack("<i", bytes(d[16:20]))[0] != 4:
    print("csa2_last_writer_bin_gate: FAIL - n_layers is not at byte 16",
          file=sys.stderr); sys.exit(1)
d[16:20] = struct.pack("<i", 0x7FFFFFFF)
open(p_overflow, "wb").write(bytes(d))

# bad_mode (I4): csa2_mode[2] set to 47, a value outside {0,1,2} - well
# within n_layers=4, refused by lz_csa2_mode_source_valid's own per-
# layer loop (model.c), the SAME function check_bin_validation's own
# "csa2_mode outside {0,1,2} is refused" line already covers against a
# hand-built LZModelConfig - this is the same claim, through the real
# lz_open_bin byte-stream parsing that function itself never touches.
# csa2_mode[] is NOT located by its own signature search: 0/1/2 values
# are common (padding is 0 too), so a search for [0,1,2,...] would risk
# multiple matches. It does not need one - model.c's own RD() call order
# writes csa2_mode[] immediately before csa2_source[] (both LZ_MAX_LAYERS
# wide, no gap), so csa2_mode[]'s base is exactly 512 bytes before the
# csa2_source signature already found above.
d = bytearray(open(p_mode, "rb").read())
src0 = find_csa2_source(d)
mode0 = src0 - LZ_MAX_LAYERS * 4
if struct.unpack("<i", bytes(d[mode0 + 8:mode0 + 12]))[0] != 1:
    print("csa2_last_writer_bin_gate: FAIL - csa2_mode[2] is not the "
          "expected REINDEX(1) - csa2_mode[]/csa2_source[] adjacency "
          "assumption is stale", file=sys.stderr)
    sys.exit(1)
d[mode0 + 8:mode0 + 12] = struct.pack("<i", 47)  # csa2_mode[2] = 47 (garbage)
open(p_mode, "wb").write(bytes(d))

# bad_lastwriter (Minor-2, sixth review round): csa2_source[3] set to 1
# instead of 2 - layer 3 (REUSE) declaring layer 1 (the FULL root) as
# its source instead of layer 2 (REINDEX, the ACTUAL last writer of
# root 1's shared physical slot, since layer 2's own call overwrites it
# before layer 3's ever runs). Passes lz_csa2_mode_source_valid's own
# per-layer question outright - 1 IS a strictly earlier, csa2_on,
# full_attention layer - which is exactly the gap lz_csa2_last_writer_
# valid's own whole-config question exists to close (model.c's own
# comment on that function has this exact scenario). csa2_source[3] is
# 12 bytes past csa2_source[]'s own base (src0), found by the same
# signature search as csa2_mode[2] above.
d = bytearray(open(p_lastwriter, "rb").read())
src0 = find_csa2_source(d)
if struct.unpack("<i", bytes(d[src0 + 12:src0 + 16]))[0] != 2:
    print("csa2_last_writer_bin_gate: FAIL - csa2_source[3] is not the "
          "expected 2 - fixture layout is stale", file=sys.stderr)
    sys.exit(1)
d[src0 + 12:src0 + 16] = struct.pack("<i", 1)  # csa2_source[3] = 1, not 2
open(p_lastwriter, "wb").write(bytes(d))
PYEOF

RC=0

echo "-- control: the unpatched fixture must load --"
"$BIN" "$W/good" --tokens 1,2,3 >"$W/good.out" 2>&1
GRC=$?
if [ "$GRC" -ne 0 ]; then
    echo "csa2_last_writer_bin_gate: FAIL - the CONTROL fixture did not load" \
         "(rc=$GRC); nothing below this means anything until it does"
    tail -3 "$W/good.out"
    exit 1
fi
echo "   control loads (rc=0)"

echo ""
echo "-- C1/C-NEW-1/C-NEW-2: malformed header (n_layers=0x7FFFFFFF) --"
"$BIN" "$W/bad_overflow" --tokens 1,2,3 >"$W/bad_overflow.out" 2>&1
RC1=$?
echo "   exit=$RC1"; tail -2 "$W/bad_overflow.out"
if [ "$RC1" -ge 128 ] || [ "$RC1" -eq 139 ]; then
    echo "csa2_last_writer_bin_gate: FAIL - REPRODUCES - crashed (signal)," \
         "not refused"
    RC=1
elif [ "$RC1" -eq 0 ]; then
    echo "csa2_last_writer_bin_gate: FAIL - REPRODUCES (differently) - a" \
         "header with n_layers=0x7FFFFFFF was ACCEPTED; it must be refused"
    RC=1
elif ! grep -q "Failed to load" "$W/bad_overflow.out"; then
    echo "csa2_last_writer_bin_gate: FAIL - nonzero exit but no 'Failed to" \
         "load' line"
    RC=1
else
    echo "   OK - refused cleanly, no crash (the early n_layers check" \
         "model.c's own RD(&c->n_layers, 4) site now runs)"
fi

echo ""
echo "-- I4: malformed header (csa2_mode[2]=47, real lz_open_bin path) --"
"$BIN" "$W/bad_mode" --tokens 1,2,3 >"$W/bad_mode.out" 2>&1
RC2=$?
echo "   exit=$RC2"; tail -2 "$W/bad_mode.out"
if [ "$RC2" -ge 128 ] || [ "$RC2" -eq 139 ]; then
    echo "csa2_last_writer_bin_gate: FAIL - I4: crashed (signal) on an" \
         "in-bounds per-layer csa2_mode corruption - should refuse cleanly"
    RC=1
elif [ "$RC2" -eq 0 ]; then
    echo "csa2_last_writer_bin_gate: FAIL - I4: csa2_mode=47 was ACCEPTED;" \
         "lz_csa2_mode_source_valid's per-layer loop should have refused it"
    RC=1
elif ! grep -q "Failed to load" "$W/bad_mode.out"; then
    echo "csa2_last_writer_bin_gate: FAIL - I4: nonzero exit but no 'Failed" \
         "to load' line"
    RC=1
else
    echo "   I4: OK - refused cleanly through the real lz_open_bin path"
fi

echo ""
echo "-- F9/Minor-2: malformed header (csa2_source[3]=1, not the actual last writer 2) --"
"$BIN" "$W/bad_lastwriter" --tokens 1,2,3 >"$W/bad_lastwriter.out" 2>&1
RC4=$?
echo "   exit=$RC4"; tail -2 "$W/bad_lastwriter.out"
if [ "$RC4" -ge 128 ] || [ "$RC4" -eq 139 ]; then
    echo "csa2_last_writer_bin_gate: FAIL - F9: crashed (signal) on a" \
         "last-writer violation - should refuse cleanly"
    RC=1
elif [ "$RC4" -eq 0 ]; then
    echo "csa2_last_writer_bin_gate: FAIL - F9: csa2_source[3]=1 (not the" \
         "actual last writer) was ACCEPTED; lz_csa2_last_writer_valid" \
         "should have refused it"
    RC=1
elif ! grep -q "Failed to load" "$W/bad_lastwriter.out"; then
    echo "csa2_last_writer_bin_gate: FAIL - F9: nonzero exit but no 'Failed" \
         "to load' line"
    RC=1
else
    echo "   F9: OK - refused cleanly through the real lz_open_bin path"
fi

echo ""
echo "-- I3 (bin path): real S4+CSA2 export (mla_q_lora_rank=32, csa2_layers=[1]) --"
"$BIN" "$W/s4csa2" --tokens 1,2,3 >"$W/s4csa2.out" 2>&1
RC3=$?
echo "   exit=$RC3"; tail -2 "$W/s4csa2.out"
if [ "$RC3" -ge 128 ] || [ "$RC3" -eq 139 ]; then
    echo "csa2_last_writer_bin_gate: FAIL - I3 (bin): crashed (signal) on a" \
         "real S4+CSA2 export - should refuse cleanly"
    RC=1
elif [ "$RC3" -eq 0 ]; then
    echo "csa2_last_writer_bin_gate: FAIL - I3 (bin): a real S4+CSA2" \
         "checkpoint was ACCEPTED; it must be refused"
    RC=1
elif ! grep -q "Failed to load" "$W/s4csa2.out"; then
    echo "csa2_last_writer_bin_gate: FAIL - I3 (bin): nonzero exit but no" \
         "'Failed to load' line"
    RC=1
else
    echo "   I3 (bin): OK - refused cleanly through the real lz_open_bin path"
fi

# MUTATION ARM for I3 (bin path): the check above is a repro, not yet
# proof the specific check matters. Rebuilds llama98 (build_mutant_cli,
# this script's own header comment) from a scratch copy of model.c with
# the bin-path mla_q_lora_rank refusal disabled, and confirms its own
# message is GONE from the same s4csa2 fixture's output - NOT that the
# fixture now loads (rc=0): model_walk's CSA2_SPECS always expects
# csa2_w_dq_idx regardless of mode, and this fixture's tensor stream
# genuinely lacks it (the training side drops it under this exact
# composition), so with the refusal gone the .bin loader's POSITIONAL
# reads still desync one tensor later and the load still fails - just
# on a confusing "elem count mismatch" far from the real cause instead
# of a clean "mla_q_lora_rank out of range", PRECISELY the degradation
# model.c's own comment on this refusal describes ("silently misaligned
# past this walk's dq_idx read, not refused"). What proves the specific
# check is gone is the ABSENCE of its own message, not the exit code.
MUT="$W/mut_i3"
mkdir -p "$MUT/src" "$MUT/common"
cp src/*.c src/*.h "$MUT/src/" 2>/dev/null
cp common/*.c common/*.h "$MUT/common/" 2>/dev/null
ANCHOR_BIN_OLD="        if (c->mla_q_lora_rank > 0) {
            for (i = 0; i < c->n_layers; i++) {
                if (c->csa2_on[i]) {
                    qerr(errbuf, errlen, LZ_ERR_BIN_FIELD_RANGE, \"mla_q_lora_rank\");
                    goto fail;
                }
            }
        }"
python3 - "$MUT/src/model.c" <<PYEOF || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = """$ANCHOR_BIN_OLD"""
if src.count(old) != 1:
    print("csa2_last_writer_bin_gate: FAIL - could not find exactly one copy "
          "of the bin-path S4+CSA2 refusal to mutate", file=sys.stderr)
    sys.exit(1)
new = old.replace("if (c->mla_q_lora_rank > 0) {",
                  "if (0 && c->mla_q_lora_rank > 0) {", 1)
src = src.replace(old, new, 1)
if old in src:
    print("csa2_last_writer_bin_gate: FAIL - a second copy survived the "
          "replace", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

if build_mutant_cli "$MUT" "I3 bin-path"; then
    "$MUT/llama98" "$W/s4csa2" --tokens 1,2,3 >"$W/s4csa2_mut.out" 2>&1
    MRC=$?
    if grep -q "mla_q_lora_rank out of range" "$W/s4csa2_mut.out"; then
        echo "csa2_last_writer_bin_gate: FAIL - I3 (bin) mutant still prints" \
             "the mla_q_lora_rank refusal (rc=$MRC) - the mutation had no effect"
        RC=1
    else
        if [ "$MRC" -eq 0 ]; then
            WHAT="loaded"
        else
            WHAT="failed on an unrelated, far-from-the-cause tensor mismatch instead"
        fi
        echo "-- I3 (bin) mutant: the explicit refusal is gone (rc=$MRC, $WHAT" \
             "- see this script's own comment) --"
        tail -2 "$W/s4csa2_mut.out"
    fi
else
    RC=1
fi

# MUTATION ARM for Minor-1 (sixth review round): the bad_overflow check
# above proves n_layers=0x7FFFFFFF is refused, but not THIS check in
# particular - the "derived quantities self-consistent" block further
# down in lz_open_bin tests the exact same condition, with the exact
# same LZ_ERR_BIN_CFG error code, so a passing bad_overflow case cannot
# by itself tell which of the two actually fired. Deletes ONLY the
# early check (model.c's own RD(&c->n_layers, 4) site) and rebuilds;
# the n_full_layers/n_linear_layers cross-check loop and the S4+CSA2
# csa2_on[] scan both run BEFORE the later block ever would, so if the
# early check is truly what stood between them and an unbounded
# n_layers, deleting it alone must still crash - proving it carries
# real weight, not merely duplicating a check that would have caught
# this anyway.
MUT="$W/mut_nlayers"
mkdir -p "$MUT/src" "$MUT/common"
cp src/*.c src/*.h "$MUT/src/" 2>/dev/null
cp common/*.c common/*.h "$MUT/common/" 2>/dev/null
python3 - "$MUT/src/model.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = ("    if (c->n_layers <= 0 || c->n_layers > LZ_MAX_LAYERS) {\n"
       "        qerr(errbuf, errlen, LZ_ERR_BIN_CFG);\n"
       "        goto fail;\n"
       "    }\n"
       "    RD(&c->seq_len, 4);\n")
new = "    RD(&c->seq_len, 4);\n"
if src.count(old) != 1:
    print("csa2_last_writer_bin_gate: FAIL - could not find exactly one copy "
          "of the early n_layers check to mutate", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
if old in src:
    print("csa2_last_writer_bin_gate: FAIL - a second copy survived the "
          "replace", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

if build_mutant_cli "$MUT" "Minor-1 early n_layers check"; then
    "$MUT/llama98" "$W/bad_overflow" --tokens 1,2,3 >"$W/bad_overflow_mut.out" 2>&1
    MRC=$?
    if [ "$MRC" -ge 128 ] || [ "$MRC" -eq 139 ]; then
        echo "-- Minor-1 mutant: correctly CRASHED (rc=$MRC) on the same" \
             "n_layers=0x7FFFFFFF header - the early check is load-bearing --"
    else
        echo "csa2_last_writer_bin_gate: FAIL - Minor-1 mutant did not crash" \
             "(rc=$MRC) - the later 'derived quantities' block alone would" \
             "have caught this, so the early check's own removal is invisible"
        tail -2 "$W/bad_overflow_mut.out"
        RC=1
    fi
else
    RC=1
fi

# MUTATION ARM for Minor-2 (sixth review round): F9's own whole-config
# check (model.c's `if (ver == 5 && !lz_csa2_last_writer_valid(c,
# NULL))`) had no mutation arm anywhere in this task's own gates -
# check_bin_validation's own F9 lines (build/csa2_mode_dispatch_probe.c)
# call the predicate directly, the same I4 gap this whole file exists
# to close for the OTHER two predicates. Neuters the call and rebuilds;
# the bad_lastwriter fixture above must then load AND complete a full
# forward pass (rc=0) - the violation itself is memory-safe (layer 3
# just aliases the wrong-but-in-bounds physical slot, layer 1's instead
# of layer 2's), so nothing else stops it once this one check is gone.
MUT="$W/mut_lastwriter"
mkdir -p "$MUT/src" "$MUT/common"
cp src/*.c src/*.h "$MUT/src/" 2>/dev/null
cp common/*.c common/*.h "$MUT/common/" 2>/dev/null
python3 - "$MUT/src/model.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = "    if (ver == 5 && !lz_csa2_last_writer_valid(c, NULL)) {\n"
new = "    if (0 && ver == 5 && !lz_csa2_last_writer_valid(c, NULL)) {\n"
if src.count(old) != 1:
    print("csa2_last_writer_bin_gate: FAIL - could not find exactly one copy "
          "of the F9 whole-config check to mutate", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
if old in src:
    print("csa2_last_writer_bin_gate: FAIL - a second copy survived the "
          "replace", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

if build_mutant_cli "$MUT" "Minor-2 F9 whole-config check"; then
    "$MUT/llama98" "$W/bad_lastwriter" --tokens 1,2,3 >"$W/bad_lastwriter_mut.out" 2>&1
    MRC=$?
    if [ "$MRC" -eq 0 ]; then
        echo "-- Minor-2 mutant: correctly ACCEPTED and ran the same" \
             "last-writer violation to completion (rc=0) --"
    else
        echo "csa2_last_writer_bin_gate: FAIL - Minor-2 mutant still refused" \
             "or crashed (rc=$MRC) - the probe cannot see this defect"
        tail -2 "$W/bad_lastwriter_mut.out"
        RC=1
    fi
else
    RC=1
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_last_writer_bin_gate: FAIL"
    exit 1
fi
echo "csa2_last_writer_bin_gate: PASS"
exit 0
