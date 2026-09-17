#!/bin/bash
# Gated residual (config.json "gated_residual", nr branches). model.h's own
# LZModelConfig.gated_residual comment has the field and the span rule,
# LZLayer's gr_attn/gr_mlp comment has the tensor list, forward.c's gr_read /
# gr_write / gr_expand / gr_collapse have the mechanism, and the training
# side's engine/kunmoe_modeling.py::GatedResidual4 (+ GrDecoderLayer.forward)
# is the reference this is a port of:
#
#   read   Rhat_i = RMSNorm_i(R_i);  G = unvec(sigmoid(W_u SiLU((1/n_r) W_d vec(Rhat))))
#          x    = (1/n_r) * sum_i (G_i * Rhat_i)
#   write  s    = 2*sigmoid((1/n_r) W_w vec(Rhat))   [from the read's Rhat,
#          R'_i = R_i + s_i*y                         not a recomputed one]
#   gr_first expands [B,S,D] into nr identical branches, gr_last takes the
#   MEAN back. The branches ride inside s->x, exactly as on the training side.
#
#   gr_gate.sh <model-dir> [second-model-dir]
#
# WHAT EACH ARM PROVES
#
#   pre/head   The pair that pins this task's milestone. `pre` is the engine
#              built from the commit BEFORE the one that introduced
#              LZLayer.gr_on, located by searching history for that line
#              rather than by a hash written down here - a pinned hash rots
#              into a skip the first time anyone rewrites the file, and a
#              skipped arm is the shape this repo keeps re-learning. On a GR
#              checkpoint `pre` must FAIL, and its message must name an
#              unclaimed gr_* tensor: before this change those tensors had no
#              spec at all, which is exactly the rc=1 the milestone is about.
#              `head` must PASS.
#
#   off        The same weights with the KEY absent and the gr_* tensors
#              REMOVED - a checkpoint that never carried GR. It must load,
#              AND its logits must be BITWISE IDENTICAL between `pre` and
#              `head`. That is the whole "off means off" contract: with the
#              switch clear, every line this task added is dead, and the
#              cheapest way to say that is that the binary before the change
#              and the binary after it agree on every byte.
#
#   keyonly    The same weights with the KEY absent but the tensors KEPT.
#              Must be REFUSED. Without this arm the `off` arm would also be
#              green on an engine that simply ignores the switch and reads
#              whatever tensors the file happens to hold - the key, not the
#              file's contents, is what has to decide.
#
#   on         The checkpoint as shipped. Must load, and must carry exactly
#              as many more parameters than `off` as its gr_* tensors hold -
#              the count is how the gate knows they are CLAIMED and not
#              merely tolerated.
#
#   raw        THE ARITHMETIC ARM, and the only one here that is not about
#              wiring. A mutant that neutralises the gates (G→1, s→1) and
#              the read normalisation must reproduce the UNMUTATED engine's
#              GR-off logits BIT FOR BIT. That identity is not invented
#              here: it is the training side's own measured `raw` arm, which
#              reproduces its GR-off arm exactly (0 of 32768 logits bytes).
#              It is what forces the branch sum to be a pairwise tree rather
#              than a running total (gr_tree_sum's own comment) - with a
#              running total the same arm missed by 1-cos 1.2e-7, measured.
#
#   batch      The batched prefill must equal the one-token-at-a-time
#              prefill BIT FOR BIT, at every width. That is the engine's
#              standing contract (test_batch_parity), and GR is the one
#              mechanism that changes a per-token STRIDE mid-stack: s->x is
#              nr times wider between gr_first and gr_last, so every index
#              into it is batch-width-visible in a way no other buffer's
#              is. It is not hypothetical - the first version of gr_expand
#              copied only branches 1..nr-1, which is right for token 0 (its
#              source is already at its destination) and wrong for every
#              later token, and batch 1 can never see it. This arm is what
#              catches that class directly; `raw` caught it first, by
#              accident.
#
#   sens       The checkpoint with layers.0.gr_attn.wu.weight offset by
#              +0.05. BOTH its prefill logits and its greedy continuation
#              must differ from `on`'s. The offset comes from a measured
#              curve, not from taste, and the two criteria do not have the
#              same shape - which is the whole reason for measuring:
#
#                offset   prefill logits   greedy text
#                1e-6     DIFF             DIFF
#                1e-5     DIFF             DIFF
#                1e-4     DIFF             DIFF
#                0.001    DIFF             DIFF
#                0.01     DIFF             SAME
#                0.05     DIFF             DIFF
#                0.2      DIFF             SAME
#                1.0      DIFF             DIFF
#
#              The LOGIT criterion is the tight and monotone one: this
#              checkpoint's GR gates are not inert the way its sinks are
#              (Task 2 lost a round to the opposite case - kunkun98's sinks
#              sit at their init value, so every offset below +20 was
#              exactly zero and "a small perturbation must move the text"
#              could not fail there). It fires at 1e-6, i.e. 1e-5 of the
#              tensor's own std.
#
#              THE TEXT CRITERION IS DISCRETE AND NOT MONOTONE, and the
#              table above is the evidence: 0.01 and 0.2 leave the 16-token
#              continuation unchanged while both their neighbours move it.
#              A greedy continuation is a chain of argmaxes, so a
#              perturbation either moves the first one or does not, and a
#              moved one can walk back onto the same tokens. +0.05 is
#              picked from this table as a value where it does fire; the
#              assertion's weight is still carried by the logits, and a
#              future prompt change that makes +0.05 stop moving the text
#              is this arm telling the truth rather than a defect.
#
# MUTATIONS, each aimed at a different arm, because the halves fail
# independently:
#
#   M1  setup_layers' gr_on derivation pinned to 0. The load must fail: the
#      gr_* tensors go unclaimed. This is the arm that catches "the flag is
#      never set", and it is also `pre`'s failure mode reproduced without
#      needing an old revision to exist.
#   M2  gr_collapse drops the division - mean becomes sum. Caught by `raw`,
#      which is an identity claim about exactly that arithmetic.
#   M3  THE ONE THIS TASK IS MOST LIKELY TO GET WRONG: the read gate is
#      skipped when use_subn is set (the reading that says "GR replaces the
#      pre-norm, SubLN deleted the pre-norm, so there is nothing to do").
#      It is wrong because SubLN moved the normalisation into each
#      PROJECTION; the residual stream reaches GR raw, so GR's read is the
#      only normalisation in front of the gate. Caught by `sens`: with the
#      read gone the read gate is never computed, so perturbing W_u cannot
#      move anything.
#
# Exit: 0 pass, 1 fail, 2 skip (no binary / no model dir / no python3 with
# numpy+safetensors / no git revision to build `pre` from).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
MODEL2=${2:-}
PROMPT="人工智能的发展历史可以追溯到二十世纪五十年代，当时的科学家们开始思考机器是否能够模拟人类的思维过程。"
PERT_DELTA=0.05
NGEN=16

BIN=$ROOT/llama98.exe
[ -x "$BIN" ] || { echo "gr_gate: SKIP - no $BIN (run: make native)"; exit 2; }
[ -n "$MODEL" ] && [ -d "$MODEL" ] || {
    echo "gr_gate: SKIP - model dir not given or not found: '$MODEL'"; exit 2; }
[ -f "$MODEL/config.json" ] || {
    echo "gr_gate: SKIP - no config.json under '$MODEL'"; exit 2; }
[ -s "$MODEL/model.safetensors" ] || {
    echo "gr_gate: SKIP - '$MODEL' has no model.safetensors (this gate rewrites one)"; exit 2; }
python3 -c 'import numpy, safetensors' 2>/dev/null || {
    echo "gr_gate: SKIP - needs python3 with numpy and safetensors"; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
FAILED=0

# ---- fixtures --------------------------------------------------------
# One python pass writes all three, because the decision they share (which
# tensors are GR, which key turns it on) has to be made once or it drifts
# between arms.
python3 - "$MODEL" "$W" "$PERT_DELTA" <<'PY'
import json, os, shutil, sys

model_dir, w, pert = sys.argv[1], sys.argv[2], float(sys.argv[3])

def bail(msg):
    print("SKIP: " + msg)
    sys.exit(2)

try:
    import numpy as np
    from safetensors.numpy import load_file, save_file
except Exception as exc:
    bail("python3 numpy/safetensors unavailable (%s)" % exc)

cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
tc = cfg.get("text_config", cfg)
nr = int(tc.get("gated_residual") or 0)
ced = int(tc.get("ced_split") or 0)
n = int(tc.get("num_hidden_layers") or len(tc["layer_types"]))
if nr < 2:
    bail("'%s' does not set gated_residual (got %r) - nothing for this gate"
         " to exercise" % (model_dir, tc.get("gated_residual")))

tensors = load_file(os.path.join(model_dir, "model.safetensors"))
gr_names = [k for k in tensors if ".gr_attn." in k or ".gr_mlp." in k]
if not gr_names:
    bail("no gr_attn/gr_mlp tensors in '%s'" % model_dir)

# the span rule, evaluated here from the config too, so the shape of what
# this gate exercises is reported rather than assumed
span = ced if 0 < ced < n else n
print("  fixture: nr=%d, layers [0,%d), %d gr_* tensor(s), %d parameter(s)"
      % (nr, span, len(gr_names), sum(int(tensors[k].size) for k in gr_names)))
print("  fixture: perturbation -> model.layers.0.gr_attn.wu.weight +%g" % pert)

SKIP_FILES = ("model.safetensors", "model.bin", "model.bin.lz4")

def scaffold(out):
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(model_dir):
        if name in SKIP_FILES or "safetensors" in name:
            continue
        src = os.path.join(model_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out, name))

def emit(tag, keep_gr, perturb, drop_key):
    out = os.path.join(w, tag)
    scaffold(out)
    c = json.loads(json.dumps(cfg))
    if drop_key:
        ctc = c.get("text_config") or c
        ctc.pop("gated_residual", None)
    json.dump(c, open(os.path.join(out, "config.json"), "w", encoding="utf-8"))
    tt = {}
    for k, v in tensors.items():
        if not keep_gr and (".gr_attn." in k or ".gr_mlp." in k):
            continue
        if perturb and k == "model.layers.0.gr_attn.wu.weight":
            v = v.astype(np.float32) + np.float32(pert)
        tt[k] = v
    save_file(tt, os.path.join(out, "model.safetensors"))

# off: never carried the switch. keyonly: the key says off, the file says
# otherwise. pert: the switch on, one gate tensor moved.
emit("off", keep_gr=False, perturb=False, drop_key=True)
emit("keyonly", keep_gr=True, perturb=False, drop_key=True)
emit("pert", keep_gr=True, perturb=True, drop_key=False)
PY
PYRC=$?
[ "$PYRC" -eq 2 ] && exit 2
[ "$PYRC" -ne 0 ] && { echo "gr_gate: FAIL - fixture build"; exit 1; }

GRPARAMS=$(python3 -c '
import sys
from safetensors.numpy import load_file
t = load_file(sys.argv[1] + "/model.safetensors")
print(sum(int(v.size) for k, v in t.items()
          if ".gr_attn." in k or ".gr_mlp." in k))' "$MODEL")

echo "gr_gate: $MODEL${MODEL2:+ and $MODEL2} (pert +$PERT_DELTA, gr params $GRPARAMS)"

# ---- helpers ---------------------------------------------------------
check() { "$1" "$2" --check >"$3" 2>&1; }
params() { sed -n 's/^Params *\([0-9][0-9]*\).*/\1/p' "$1" | head -1; }
why() { sed -n 's/^Failed to load: \(.*\)/\1/p' "$1" | head -1; }

# run BINARY TAG DIR - one prefill, then NGEN greedy tokens; .pf .txt .log
run() {
    local bin="$1" tag="$2" dir="$3"
    "$bin" "$dir" --prompt "$PROMPT" -n "$NGEN" --temp 0 --seed 1145141919 \
        --dump-prefill "$W/$tag.pf" >"$W/$tag.log" 2>&1 || {
        echo "  FAIL: '$tag' exited non-zero"; tail -5 "$W/$tag.log"; return 1; }
    [ -s "$W/$tag.pf" ] || { echo "  FAIL: '$tag' wrote no prefill logits"; return 1; }
    sed -n '/generation start/,/generation end/p' "$W/$tag.log" \
        | sed '1d;$d' >"$W/$tag.txt"
    [ -s "$W/$tag.txt" ] || { echo "  FAIL: '$tag' generated no text"; return 1; }
    return 0
}

# ---- the pre-change binary -------------------------------------------
# Located by searching history for the line, not by a hash written down
# here: a pinned revision rots into a skip the first time anyone rewrites
# model.c, and a skipped arm looks exactly like a passing one.
GR_INTRO=$(git -C "$ROOT" log --format=%H --reverse \
             -S'L->gr_on = c->gated_residual' -- src/model.c | head -1)
if [ -z "$GR_INTRO" ]; then
    echo "gr_gate: SKIP - no commit in this repo introduces LZLayer.gr_on, so"
    echo "  the pre/head pair cannot be built. If that line was legitimately"
    echo "  renamed, point this search at the new spelling."
    exit 2
fi
PRE_REV=$(git -C "$ROOT" rev-parse "$GR_INTRO^" 2>/dev/null) || {
    echo "gr_gate: SKIP - $GR_INTRO has no parent"; exit 2; }
mkdir -p "$W/pre"
if ! git -C "$ROOT" archive "$PRE_REV" | tar -x -C "$W/pre" 2>/dev/null; then
    echo "gr_gate: SKIP - cannot export $PRE_REV"; exit 2
fi
if ! ( cd "$W/pre" && make native ) >"$W/pre/build.log" 2>&1; then
    echo "gr_gate: SKIP - the pre-change tree ($PRE_REV) does not build here"
    tail -3 "$W/pre/build.log"
    exit 2
fi
PREBIN="$W/pre/llama98.exe"
echo "  pre-change engine: $PRE_REV"

# ---- arms ------------------------------------------------------------
for d in "$MODEL" ${MODEL2:+"$MODEL2"}; do
    if check "$BIN" "$d" "$W/on.chk"; then
        echo "  on   $(basename "$d") loads (params $(params "$W/on.chk"))"
    else
        echo "  FAIL: $(basename "$d") does not load: $(why "$W/on.chk")"
        exit 1
    fi
    if "$PREBIN" "$d" --check >"$W/pre.chk" 2>&1; then
        echo "  FAIL: the pre-change engine ACCEPTED $(basename "$d") - this"
        echo "    checkpoint has gr_* tensors no spec there claims, so the"
        echo "    milestone this gate pins did not happen"
        FAILED=1
    elif grep -q 'gr_attn\|gr_mlp' "$W/pre.chk"; then
        echo "  pre  $(basename "$d") refused, naming a gr_* tensor: $(why "$W/pre.chk")"
    else
        echo "  FAIL: the pre-change engine refused $(basename "$d") for some"
        echo "    OTHER reason, so this pair does not measure the GR gap:"
        echo "    $(why "$W/pre.chk")"
        FAILED=1
    fi
done

check "$BIN" "$MODEL" "$W/on.chk" || { echo "  FAIL: on arm does not load"; exit 1; }
check "$BIN" "$W/off" "$W/off.chk" || {
    echo "  FAIL: off arm does not load: $(why "$W/off.chk")"; exit 1; }
if check "$BIN" "$W/keyonly" "$W/keyonly.chk"; then
    echo "  FAIL: the keyless config with the gr_* tensors still in the file"
    echo "    was accepted; the key is not what decides"
    FAILED=1
else
    echo "  keyonly refused ($(why "$W/keyonly.chk"))"
fi

POFF=$(params "$W/off.chk")
PON=$(params "$W/on.chk")
if [ "$((PON - POFF))" -eq "$GRPARAMS" ]; then
    echo "  on - off params = $((PON - POFF)) = the checkpoint's gr_* parameters (claimed)"
else
    echo "  FAIL: on - off params = $((PON - POFF)), expected $GRPARAMS"
    FAILED=1
fi

# off is dead: the pre-change binary and this one must agree byte for byte
run "$PREBIN" off_pre "$W/off" || exit 1
run "$BIN" off_head "$W/off" || exit 1
if cmp -s "$W/off_pre.pf" "$W/off_head.pf"; then
    echo "  off  pre vs head prefill logits BITWISE IDENTICAL (the GR path is dead when the key is clear)"
else
    echo "  FAIL: the off arm's prefill logits moved between the pre-change"
    echo "    engine and this one - a clear key is not leaving this code dead"
    FAILED=1
fi

run "$BIN" on "$MODEL" || exit 1
run "$BIN" pert "$W/pert" || exit 1
if cmp -s "$W/on.pf" "$W/pert.pf"; then
    echo "  FAIL: sens (+$PERT_DELTA) prefill logits equal on's - the gate is not reading gr_attn.wu"
    FAILED=1
else
    echo "  sens (+$PERT_DELTA): prefill logits differ from on (the tensor is read)"
fi
if cmp -s "$W/on.txt" "$W/pert.txt"; then
    echo "  FAIL: sens (+$PERT_DELTA) generated the same $NGEN tokens as on"
    FAILED=1
else
    echo "  sens (+$PERT_DELTA): greedy continuation differs from on"
fi

# batch parity - see this file's header. Width 1 is batching off, 4 is
# LZ_BATCH_MAX and the shipping default.
"$BIN" "$MODEL" --prompt "$PROMPT" -n 1 --temp 0 --batch 1 \
    --dump-prefill "$W/b1.pf" >/dev/null 2>&1 || { echo "  FAIL: --batch 1 run failed"; exit 1; }
"$BIN" "$MODEL" --prompt "$PROMPT" -n 1 --temp 0 --batch 4 \
    --dump-prefill "$W/b4.pf" >/dev/null 2>&1 || { echo "  FAIL: --batch 4 run failed"; exit 1; }
if cmp -s "$W/b1.pf" "$W/b4.pf"; then
    echo "  batch  width 1 vs 4 prefill logits BITWISE IDENTICAL"
else
    echo "  FAIL: batched prefill differs from one-token-at-a-time on a GR"
    echo "    model - an index into the widened s->x is batch-width dependent"
    FAILED=1
fi

# ---- mutations -------------------------------------------------------
MUTBIN=
# mutant TAG FILE PROBE [-z] -e EXPR ... : each on a private copy of src/,
# never the live tree, and each verified to have actually matched (the
# probe must survive) before it is trusted - a sed that silently found
# nothing would build the unmutated engine and this gate would report a
# green mutation.
mutant() {
    local tag="$1" file="$2" probe="$3"; shift 3
    local d="$W/$tag" zflag=()
    MUTBIN=""
    mkdir -p "$d"
    cp -r "$ROOT/src" "$ROOT/common" "$d/"
    cp "$ROOT/Makefile" "$d/Makefile"
    if [ "$1" = "-z" ]; then zflag=(-z); shift; fi
    local args=()
    while [ "$#" -gt 0 ]; do args+=(-e "$1"); shift; done
    sed -i "${zflag[@]}" "${args[@]}" "$d/$file"
    if ! grep -qF "$probe" "$d/$file"; then
        echo "  FAIL: mutation '$tag' did not match $file - this gate no longer matches the source"
        return 1
    fi
    if ! ( cd "$d" && make native ) >"$d/build.log" 2>&1; then
        echo "  FAIL: mutant '$tag' failed to build"; tail -5 "$d/build.log"
        return 1
    fi
    MUTBIN="$d/llama98.exe"
    return 0
}

# The three substitutions that neutralise the read and write gates. Shared
# by the `raw` arm and by M2, which is only observable as arithmetic once
# the gates are out of the way.
RAW_READ='s/lz_t_f32(&norms\[b\], s->wscr), dim, c->rms_norm_eps);/lz_t_f32(\&norms[b], s->wscr), dim, c->rms_norm_eps), memcpy(RH + (size_t)b * dim, R + (size_t)b * dim, (size_t)dim * sizeof(float));/'
RAW_G='s/for (i = 0; i < flat; i++) G\[i\] = lz_sigmoid(G\[i\]);/for (i = 0; i < flat; i++) G[i] = 1.0f;/'
RAW_S='s/for (i = 0; i < nr; i++) sg\[i\] = 2.0f \* lz_sigmoid(sg\[i\] \/ nrf);/for (i = 0; i < nr; i++) sg[i] = 1.0f;/'

run "$BIN" on "$MODEL" || exit 1

if mutant raw src/forward.c 'memcpy(RH + (size_t)b * dim, R + (size_t)b * dim' \
        "$RAW_READ" "$RAW_G" "$RAW_S"; then
    run "$MUTBIN" raw "$MODEL" || exit 1
    if cmp -s "$W/raw.pf" "$W/off_head.pf"; then
        echo "  raw  gate-neutralised GR-on == GR-off BITWISE (the arithmetic identity holds)"
    else
        echo "  FAIL: with the gates neutralised and the read norm removed, GR-on"
        echo "    does not reproduce GR-off; the branch arithmetic is not the"
        echo "    identity the training side's own raw arm reproduces exactly"
        FAILED=1
    fi
else
    FAILED=1
fi

# M1: the derivation never fires, so the tensors are never claimed.
if mutant m1 src/model.c 'L->gr_on = 0 && c->gated_residual' \
        's/L->gr_on = c->gated_residual >= 2 \&\& i < lz_gr_span(c)/L->gr_on = 0 \&\& c->gated_residual >= 2 \&\& i < lz_gr_span(c)/'; then
    if "$MUTBIN" "$MODEL" --check >"$W/m1.chk" 2>&1; then
        echo "  FAIL: mutation M1 (gr_on pinned to 0) still accepted the checkpoint"
        FAILED=1
    else
        echo "  mutation M1 (gr_on pinned to 0): red as it should be ($(why "$W/m1.chk"))"
    fi
else
    FAILED=1
fi

# M2: mean becomes sum in gr_last's collapse. Only the raw identity sees it.
if mutant m2 src/forward.c 'dst[i] = base[i];' \
        "$RAW_READ" "$RAW_G" "$RAW_S" \
        's/for (i = 0; i < dim; i++) dst\[i\] = base\[i\] \/ (float)nr;/for (i = 0; i < dim; i++) dst[i] = base[i];/'; then
    run "$MUTBIN" m2 "$MODEL" || exit 1
    if cmp -s "$W/m2.pf" "$W/off_head.pf"; then
        echo "  FAIL: mutation M2 (gr_last collapses with sum, not mean) was not"
        echo "    caught - the raw arm cannot see the collapse's arithmetic"
        FAILED=1
    else
        echo "  mutation M2 (collapse = sum, not mean): red as it should be - the raw arm is watching it"
    fi
else
    FAILED=1
fi

# M3: the read gate is skipped under SubLN. That removes the read gate
# entirely, so perturbing W_u cannot move anything. -z: the substitution
# spans a line break.
if mutant m3 src/forward.c 'L->gr_on && !c->use_subn' -z \
        's/if (L->gr_on) {\n            gr_read(m, s, L->gr_attn_norms,/if (L->gr_on \&\& !c->use_subn) {\n            gr_read(m, s, L->gr_attn_norms,/;s/if (L->gr_on) {\n            gr_read(m, s, L->gr_mlp_norms,/if (L->gr_on \&\& !c->use_subn) {\n            gr_read(m, s, L->gr_mlp_norms,/'; then
    "$MUTBIN" "$MODEL" --prompt "$PROMPT" -n "$NGEN" --temp 0 \
        --seed 1145141919 --dump-prefill "$W/m3_on.pf" >/dev/null 2>&1
    "$MUTBIN" "$W/pert" --prompt "$PROMPT" -n "$NGEN" --temp 0 \
        --seed 1145141919 --dump-prefill "$W/m3_pert.pf" >/dev/null 2>&1
    if cmp -s "$W/m3_on.pf" "$W/m3_pert.pf"; then
        echo "  mutation M3 (read gate skipped under use_subn): red as it should be - the sensitivity arm collapses onto on"
    else
        echo "  FAIL: mutation M3 (read gate skipped under use_subn) was not"
        echo "    caught; the read gate is not doing the normalisation the"
        echo "    SubLN body does not do"
        FAILED=1
    fi
else
    FAILED=1
fi

# M4: gr_expand stops copying branch 0. Correct for token 0, wrong for every
# later token - the exact defect the batch arm was added for.
if mutant m4 src/forward.c 'for (b = nr - 1; b >= 1; b--)' \
        's/for (b = nr - 1; b >= 0; b--)/for (b = nr - 1; b >= 1; b--)/'; then
    "$MUTBIN" "$MODEL" --prompt "$PROMPT" -n 1 --temp 0 --batch 4 \
        --dump-prefill "$W/m4.pf" >/dev/null 2>&1
    if cmp -s "$W/m4.pf" "$W/b4.pf"; then
        echo "  FAIL: mutation M4 (gr_expand skips branch 0) was not caught -"
        echo "    the batch arm is not looking at the widened stride"
        FAILED=1
    else
        echo "  mutation M4 (gr_expand skips branch 0): red as it should be - the batch arm is watching the stride"
    fi
else
    FAILED=1
fi

[ "$FAILED" -eq 0 ] && { echo "gr_gate: PASS"; exit 0; }
echo "gr_gate: FAIL"
exit 1
