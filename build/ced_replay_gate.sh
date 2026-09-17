#!/bin/bash
# ced_split/ced_replay_w gate (model.h's own field comment has the
# mechanism; docs/dsv41-optimization-menu.md's E3 section has the
# measurement it implements).
#
#   ced_replay_gate.sh <model-dir> [prompt]
#
# Unlike --ced (E1), this knob's claim is NOT "output unchanged" - a
# reset genuinely discards recurrent state, so it is EXPECTED to move
# the prefill logits whenever it actually fires. Three arms instead of
# two, and each one is checking a different failure mode:
#
#   off   ced_split/ced_replay_w absent from config.json entirely - the
#         "never carried these keys" contract every other optional
#         field in this struct already has. Baseline.
#   on    ced_split = num_hidden_layers/2 (the report's own 50/50 split,
#         model.h's own convention), ced_replay_w small enough that the
#         prompt is longer than it, so the reset MUST fire mid-prefill.
#   wide  same split, ced_replay_w far larger than the prompt - the
#         built-in correctness control: T > W is false, so
#         lz_forward_batch's own replay_boundary computation must come
#         out -1 and the reset must never run at all, reproducing OFF
#         bit for bit. Same control shape as
#         .prof/ced_replay_w_probe.py's own W=N assertion on the
#         training side.
#
# THE COUNTER IS THE OTHER HALF, same reasoning as ced_gate.sh's own
# skip counter: "on" moving the logits is not on its own evidence the
# reset ran - a model with zero KDA layers at or above the split would
# also produce different-by-coincidence logits from an unrelated cause
# and this gate would not know why. [ced-replay: N layers reset] must
# read the exact count of KDA layers at index >= split on the on arm,
# and 0 on both off and wide.
#
# MUTATION: the loop condition guarding which layers qualify
# (`l < c->ced_split`) is the one line whose breakage is invisible to a
# passing build - flip it to `>=` and the reset fires on the WRONG
# layers (the encoder half instead of the decoder half). The on-vs-off
# comparison still differs (some layers still get reset, just not the
# right ones), so the mutant's positive control is the layer COUNT
# reported on the on arm: with `>=` flipped in, on an 8-layer/split-4
# fixture the count becomes 4 (layers 0-3, all KDA-typed layers below
# the split) instead of the correct 3 (layers 4-6, the KDA layers at or
# above the split) - a mutation this gate must catch by comparing that
# number, not just by comparing logits.
#
# Exit: 0 pass, 1 fail, 2 skip (no binary / no model / no KDA layers at
# or above the split, which can't exercise anything here).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
PROMPT=${2:-"人工智能的发展历史可以追溯到二十世纪五十年代，当时的科学家们开始思考机器是否能够模拟人类的思维过程。这是一段用于测试的更长提示词，包含更多汉字以确保提示长度超过窗口大小。"}

BIN=$ROOT/llama98.exe
[ -x "$BIN" ] || { echo "ced_replay_gate: SKIP - no $BIN (run: make native)"; exit 2; }
[ -n "$MODEL" ] && [ -d "$MODEL" ] || {
    echo "ced_replay_gate: SKIP - model dir not given or not found: '$MODEL'"; exit 2; }
[ -f "$MODEL/config.json" ] || {
    echo "ced_replay_gate: SKIP - no config.json under '$MODEL'"; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
FAILED=0

# Split point and expected reset-layer count, both derived from the
# SAME config.json a Python one-liner reads once - not hand-entered,
# so the gate stays correct against whatever fixture $MODEL names.
read -r SPLIT EXPECT <<EOF
$(python3 - "$MODEL/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
lt = c.get("layer_types") or []
n = c.get("num_hidden_layers") or len(lt)
s = n // 2
expect = sum(1 for t in lt[s:] if t == "linear_attention")
print(s, expect)
PY
)
EOF

[ "$EXPECT" -gt 0 ] || {
    echo "ced_replay_gate: SKIP - '$MODEL' has no linear_attention layer at or above split $SPLIT"
    exit 2; }

# config <tag> <ced_split> <ced_replay_w-or-empty>
config() {
    local tag="$1" split="$2" w="$3"
    cp -r "$MODEL" "$W/$tag"
    python3 - "$W/$tag/config.json" "$split" "$w" <<'PY'
import json, sys
path, split, w = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(path))
if w != "":
    c["ced_split"] = int(split)
    c["ced_replay_w"] = int(w)
json.dump(c, open(path, "w"))
PY
}

config off "$SPLIT" ""
config on  "$SPLIT" 8
config wide "$SPLIT" 1000000

echo "ced_replay_gate: $MODEL (split=$SPLIT, expect $EXPECT layer(s) reset on 'on')"

# arm TAG [BINARY] - run one prefill and write $W/TAG.pf + $W/TAG.log
arm() {
    local tag="$1" bin=${2:-$BIN}
    "$bin" "$W/$tag" --prompt "$PROMPT" -n 1 --seed 1145141919 \
        --dump-prefill "$W/$tag.pf" >"$W/$tag.log" 2>&1 || {
        echo "  FAIL: '$tag' exited non-zero"; tail -5 "$W/$tag.log"; return 1; }
    [ -s "$W/$tag.pf" ] || { echo "  FAIL: '$tag' wrote no prefill logits"; return 1; }
    return 0
}
arm off || exit 1
arm on  || exit 1
arm wide || exit 1

layers() { sed -n 's/.*\[ced-replay: \([0-9]*\) layers reset\].*/\1/p' "$1" | head -1; }
NOFF=$(layers "$W/off.log"); NON=$(layers "$W/on.log"); NWIDE=$(layers "$W/wide.log")
[ -n "$NOFF" ] && [ -n "$NON" ] && [ -n "$NWIDE" ] || {
    echo "  FAIL: no [ced-replay: ...] line - the counter this gate leans on is gone"
    exit 1; }
printf '  off layers=%s  on layers=%s  wide layers=%s\n' "$NOFF" "$NON" "$NWIDE"

[ "$NOFF" -eq 0 ] || { echo "  FAIL: off reset $NOFF layer(s) with no ced_replay_w set"; FAILED=1; }
[ "$NWIDE" -eq 0 ] || { echo "  FAIL: wide (window >= prompt) reset $NWIDE layer(s), should reset none"; FAILED=1; }
[ "$NON" -eq "$EXPECT" ] || {
    echo "  FAIL: on reset $NON layer(s), expected exactly $EXPECT (config.json's own layer_types)"
    FAILED=1; }

# Preserved before the mutation section below reuses the "on" tag with
# a different binary - that overwrites on.pf/on.log in place.
cp "$W/on.pf" "$W/on_orig.pf"

if cmp -s "$W/off.pf" "$W/on.pf"; then
    echo "  FAIL: on's logits equal off's, but $EXPECT layer(s) reportedly reset - the reset ran but changed nothing observable"
    FAILED=1
else
    echo "  on vs off: logits differ (expected - the reset discards real state)"
fi
if cmp -s "$W/off.pf" "$W/wide.pf"; then
    echo "  wide vs off: bitwise identical (expected - window >= prompt inserts no boundary)"
else
    echo "  FAIL: wide's logits differ from off's, but 0 layers were reported reset"
    FAILED=1
fi

# Mutation: flip the qualifying condition so the WRONG half resets.
# lz_ced_replay_reset's guard is `c->layer_types[l] != LZ_LT_KDA || l < c->ced_split`;
# replacing `l < c->ced_split` with `l >= c->ced_split` reset the layers
# BELOW the split instead of at-or-above it - same shape of bug as
# ced_gate.sh's M2 (skip in the wrong place, not a missing skip). The
# discriminator is the LOGITS, not the layer count: a fixture whose two
# halves happen to carry the same number of linear_attention layers
# (this repo's own _kdvd_gate fixture does, 3 and 3) would make a
# count-only comparison blind to exactly this bug, even though the
# WRONG three layers were reset.
MUT=$W/mut
mkdir -p "$MUT"
cp -r "$ROOT/src" "$ROOT/common" "$MUT/"
cp "$ROOT/Makefile" "$MUT/Makefile"
sed -i 's/c->layer_types\[l\] != LZ_LT_KDA || l < c->ced_split/c->layer_types[l] != LZ_LT_KDA || l >= c->ced_split/' \
    "$MUT/src/forward.c"
if grep -q 'l >= c->ced_split' "$MUT/src/forward.c"; then
    if ( cd "$MUT" && make native ) >"$MUT/build.log" 2>&1; then
        arm on "$MUT/llama98.exe" >/dev/null 2>&1
        mv "$W/on.log" "$W/mut_on.log" 2>/dev/null
        NMUT=$(layers "$W/mut_on.log")
        if cmp -s "$W/on_orig.pf" "$W/on.pf"; then
            echo "  FAIL: mutation was not caught (mutant's logits equal the correct on arm's)"
            FAILED=1
        else
            echo "  mutation (l>=split instead of l<split): red as it should be (logits differ, mutant reset $NMUT layer(s) on the wrong side)"
        fi
    else
        echo "  FAIL: mutant failed to build (sed did not match, or an unrelated build break)"
        tail -5 "$MUT/build.log"
        FAILED=1
    fi
else
    echo "  FAIL: sed did not find the guard line to mutate - this gate no longer matches the source"
    FAILED=1
fi

[ "$FAILED" -eq 0 ] && { echo "ced_replay_gate: PASS"; exit 0; }
echo "ced_replay_gate: FAIL"
exit 1
