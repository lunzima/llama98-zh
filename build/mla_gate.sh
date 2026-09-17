#!/bin/bash
# S4 MLA (low-rank Q/O) gate. model.h's own mla_q_lora_rank/
# mla_o_lora_rank comment has the mechanism.
#
#   mla_gate.sh <model-dir> [prompt]
#
# No real MLA-trained checkpoint exists (engine/kunmoe_modeling.py's
# _wire_mla is training-side-only so far), so this gate builds one:
# every full_attention layer's q_proj/o_proj weight is REPLACED with a
# random matrix that is EXACTLY rank r by construction (up @ down for
# random up/down of rank r), then written two ways -
#
#   off   the product up@down as a single q_proj/o_proj tensor, no
#         mla_*_lora_rank in config.json - the plain single-matmul path,
#         computing the exact same linear map as `on` mathematically.
#   on    the same up/down matrices, unmultiplied, as q_proj.down/
#         q_proj.up (mla_q_lora_rank/mla_o_lora_rank set) - the chained
#         two-matmul path.
#   wrong same shapes as `on`, but up/down are a SECOND, independent
#         random low-rank factorization - not a factorization of off's
#         matrix at all. The mutation/positive-control arm.
#
# WHY NOT BIT-IDENTICAL. Unlike --ced/--ced-replay-w, chaining two
# matmuls through an int8 intermediate genuinely introduces one more
# quantization step than the single matmul off takes - off and on
# compute the same linear map in EXACT float arithmetic but not in this
# engine's actual (quantized) arithmetic, so a real difference is
# expected, and is not a bug. The assertion is therefore top-1 argmax
# agreement and a small KL against `off`, not cmp -s - the same
# argmax/top-5 methodology CLAUDE.md's own judging-criteria section
# prescribes for exactly this situation ("compare against a precise
# reference with argmax/top-5, not |logits diff|, when a lossy step is
# expected to move the raw numbers").
#
# `wrong` is the discriminator that a passing on-vs-off comparison on
# its own cannot provide: if the engine silently ignored q_proj_down/
# q_proj_up and fell through to some stale or zeroed q_proj, on and
# wrong would read IDENTICAL weights (both absent) and produce
# identical output despite encoding different linear maps - so `wrong`
# must diverge from `on` by much more than on diverges from off.
#
# NOT TESTED HERE: interaction with the MTP head's own attention
# block. m->mtp->blk is a separate LZLayer that setup_layers never
# touches (model.h's own comment on why), so mla_q_on/mla_o_on there
# stay at their calloc default of 0 by construction, but attn_impl only
# ever runs against it via lz_debug_mtp_attn_t1_slow, an internal
# debug global with no CLI/env exposure to flip from a shell script -
# the DSpark T=1 fast path (mtp_attn_t1) is what actually runs in every
# reachable configuration and does not call attn_impl at all.
#
# Exit: 0 pass, 1 fail, 2 skip (no binary / no model / no full_attention
# layer / no numpy+safetensors).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
PROMPT=${2:-"人工智能的发展历史可以追溯到二十世纪五十年代，当时的科学家们开始思考机器是否能够模拟人类的思维过程。"}
RANK=64

BIN=$ROOT/llama98.exe
[ -x "$BIN" ] || { echo "mla_gate: SKIP - no $BIN (run: make native)"; exit 2; }
[ -n "$MODEL" ] && [ -d "$MODEL" ] || {
    echo "mla_gate: SKIP - model dir not given or not found: '$MODEL'"; exit 2; }
[ -f "$MODEL/config.json" ] && [ -f "$MODEL/model.safetensors" ] || {
    echo "mla_gate: SKIP - no config.json/model.safetensors under '$MODEL'"; exit 2; }
python3 -c "import safetensors, numpy" >/dev/null 2>&1 || {
    echo "mla_gate: SKIP - no python3 safetensors/numpy"; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
FAILED=0

BUILD_LOG="$W/build.log"
python3 - "$MODEL" "$W" "$RANK" > "$BUILD_LOG" 2>&1 <<'PY'
import json, os, sys
import numpy as np
from safetensors.numpy import load_file, save_file

model_dir, w, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
tc = cfg.get("text_config", cfg)
if tc.get("use_subn"):
    print("SKIP: use_subn=1 - mla_q_lora_rank/mla_o_lora_rank refuses that "
          "combination (LZ_ERR_CFG_MLA_SUBN), so this fixture cannot host "
          "the gate")
    sys.exit(2)
lt = tc["layer_types"]
full_layers = [i for i, t in enumerate(lt) if t == "full_attention"]
if not full_layers:
    print("SKIP: no full_attention layer in this fixture")
    sys.exit(2)

hidden = tc["hidden_size"]
tensors = load_file(os.path.join(model_dir, "model.safetensors"))
prefix = "model."
q_name = prefix + "layers.%d.self_attn.q_proj.weight" % full_layers[0]
if q_name not in tensors:
    prefix = ""
    q_name = "layers.%d.self_attn.q_proj.weight" % full_layers[0]
qgd, h1 = tensors[q_name].shape
assert h1 == hidden
o_name = prefix + "layers.%d.self_attn.o_proj.weight" % full_layers[0]
h2, qd = tensors[o_name].shape
assert h2 == hidden

def lowrank(rng, out_dim, in_dim, r):
    # Small init (matches nn.Linear's own default order of magnitude) so
    # the product's row norms stay in the range every other tensor in
    # this checkpoint is already at - an oversized synthetic q_proj
    # would saturate the activation quantizer's scale on every OTHER
    # weight sharing the same xq/xqs buffer this token, and that failure
    # has nothing to do with MLA.
    scale = 1.0 / np.sqrt(in_dim)
    down = (rng.standard_normal((r, in_dim)) * scale).astype(np.float32)
    up = (rng.standard_normal((out_dim, r)) * scale).astype(np.float32)
    return down, up

rng = np.random.default_rng(1145141919)
q_down, q_up = lowrank(rng, qgd, hidden, rank)
o_down, o_up = lowrank(rng, hidden, qd, rank)
q_synth = q_up @ q_down
o_synth = o_up @ o_down
# independent second factorization - the `wrong` arm's positive control
rng2 = np.random.default_rng(271828182)
q_down2, q_up2 = lowrank(rng2, qgd, hidden, rank)
o_down2, o_up2 = lowrank(rng2, hidden, qd, rank)

def write_variant(tag, cfg_extra, per_layer):
    """per_layer(li) -> dict of {suffix: array} replacing/adding tensors
    at layer li; suffixes without the 'layers.li.' prefix."""
    out_dir = os.path.join(w, tag)
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(model_dir):
        # model.safetensors gets rebuilt below with the patched tensors.
        # model.bin/model.bin.lz4 are a PRE-EXPORTED snapshot lz_open
        # prefers over model.safetensors whenever either is present
        # (src/model.c's lz_open dir-probe) - copying either stale
        # sibling in unmodified would make every arm silently load the
        # SAME unpatched weights regardless of what this script writes
        # to model.safetensors, which is exactly the failure this
        # gate's `wrong` arm exists to catch and very nearly did not.
        if name in ("model.safetensors", "model.bin", "model.bin.lz4"):
            continue
        src = os.path.join(model_dir, name)
        if os.path.isfile(src):
            import shutil
            shutil.copyfile(src, os.path.join(out_dir, name))
    new_cfg = json.loads(json.dumps(cfg))
    new_tc = new_cfg.get("text_config", new_cfg)
    new_tc.update(cfg_extra)
    json.dump(new_cfg, open(os.path.join(out_dir, "config.json"), "w",
                            encoding="utf-8"))
    out = dict(tensors)
    for li in full_layers:
        base = "layers.%d.self_attn." % li
        drop, add = per_layer(li)
        for suf in drop:
            del out[prefix + base + suf]
        for suf, arr in add.items():
            out[prefix + base + suf] = arr
    save_file(out, os.path.join(out_dir, "model.safetensors"))

write_variant("off", {}, lambda li: (
    [], {"q_proj.weight": q_synth, "o_proj.weight": o_synth}))
write_variant("on", {"mla_q_lora_rank": rank, "mla_o_lora_rank": rank},
             lambda li: (["q_proj.weight", "o_proj.weight"],
                        {"q_proj.down.weight": q_down, "q_proj.up.weight": q_up,
                         "o_proj.down.weight": o_down, "o_proj.up.weight": o_up}))
write_variant("wrong", {"mla_q_lora_rank": rank, "mla_o_lora_rank": rank},
             lambda li: (["q_proj.weight", "o_proj.weight"],
                        {"q_proj.down.weight": q_down2, "q_proj.up.weight": q_up2,
                         "o_proj.down.weight": o_down2, "o_proj.up.weight": o_up2}))
print("built off/on/wrong under %s, full_attention layers %s, rank %d"
      % (w, full_layers, rank))
PY
BUILD_RC=$?
cat "$BUILD_LOG"
if [ "$BUILD_RC" -eq 2 ]; then exit 2; fi
if [ "$BUILD_RC" -ne 0 ]; then
    echo "mla_gate: FAIL - fixture build failed"
    exit 1
fi

arm() {
    local tag="$1"
    "$BIN" "$W/$tag" --prompt "$PROMPT" -n 1 --seed 1145141919 \
        --dump-prefill "$W/$tag.pf" >"$W/$tag.log" 2>&1 || {
        echo "  FAIL: '$tag' exited non-zero"; tail -8 "$W/$tag.log"; return 1; }
    [ -s "$W/$tag.pf" ] || { echo "  FAIL: '$tag' wrote no prefill logits"; return 1; }
    return 0
}
arm off   || exit 1
arm on    || exit 1
arm wrong || exit 1

python3 - "$W" "$MODEL" > "$W/compare.log" 2>&1 <<'PY'
import json, os, sys
import numpy as np

w, model_dir = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
tc = cfg.get("text_config", cfg)
vocab = tc["vocab_size"]

def load(tag):
    return np.fromfile(os.path.join(w, tag + ".pf"), dtype=np.float32,
                       count=vocab)

off, on, wrong = load("off"), load("on"), load("wrong")

def logsoftmax(x):
    x = x - x.max()
    e = np.exp(x)
    return x - np.log(e.sum())

def kl(p_logits, q_logits):
    lp, lq = logsoftmax(p_logits), logsoftmax(q_logits)
    return float((np.exp(lp) * (lp - lq)).sum())

top1_off, top1_on, top1_wrong = off.argmax(), on.argmax(), wrong.argmax()
kl_on = kl(off, on)
kl_wrong = kl(off, wrong)
print("top1 off=%d on=%d wrong=%d" % (top1_off, top1_on, top1_wrong))
print("KL(off||on)=%.6g  KL(off||wrong)=%.6g" % (kl_on, kl_wrong))

ok = True
if top1_on != top1_off:
    print("FAIL: on's argmax disagrees with off's exact-linear-map "
          "reference (quantization noise should not flip the top token "
          "on a well-conditioned random fixture)")
    ok = False
if kl_on > 1.0:
    print("FAIL: KL(off||on)=%.6g is too large for a chain computing the "
          "exact same linear map up to one extra int8 quantization step"
          % kl_on)
    ok = False
if not (kl_wrong > 100 * max(kl_on, 1e-6)):
    print("FAIL: wrong (an unrelated low-rank factorization) did not "
          "diverge far more than on did - the on/off comparison above "
          "cannot then be trusted to notice a broken wire (e.g. "
          "q_proj_down/q_proj_up silently unread)")
    ok = False
else:
    print("mutation (independent low-rank factorization): red as it "
          "should be (KL(off||wrong)=%.6g >> KL(off||on)=%.6g)"
          % (kl_wrong, kl_on))
sys.exit(0 if ok else 1)
PY
CMP_RC=$?
cat "$W/compare.log"
[ "$CMP_RC" -eq 0 ] || FAILED=1

[ "$FAILED" -eq 0 ] && { echo "mla_gate: PASS"; exit 0; }
echo "mla_gate: FAIL"
exit 1
