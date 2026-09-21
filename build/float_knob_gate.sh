#!/bin/bash
# One float knob, and the retired ones stay retired.
#
# --fixed off|all is the whole float axis. It replaced seven per-tier
# flags - --recur-write, --conv-write, --attn-write, --epi-write,
# --sig-write, --norm-write, --scalar-fixed - each with its own global
# and selector, because seven axes could only buy MIXED settings and
# mixing measured worse than either end (the paired NLL is at
# lz_float_ref_select in ops_sched.c).
#
# WHY A GATE AND NOT A MEMORY. A retired flag comes back by accident:
# somebody re-adds a parse arm for one tier while chasing a bug, and
# nothing says a word - the binary still runs, the logits still match,
# and the axis count has gone from one to two. Nothing else here counts
# the knobs.
#
# WHAT IS ASSERTED
#   1. --fixed off and --fixed all are both accepted.
#   2. Each of the seven retired names is REJECTED.
#   3. --pair and --prefetch are still accepted. They are performance
#      knobs, not FPU tiers, and the requirement keeps them; asserting
#      it here stops this gate being read as "no knobs at all".
#   4. ops_sched.c defines exactly one float-selection global.
#
# WHAT THE AXIS COSTS, measured rather than assumed, so the next reader
# asking "how much precision does the integer default give up" has a
# number instead of a guess. kmr20 on .prof/tz.bin, 152 positions,
# vocab 32768, --kernel sse2, --fixed all against --fixed off:
#
#   max abs diff   1.742
#   rms diff       0.107
#   top-1 agree    148 / 152  (97.37%)
#
# THE FOUR DISAGREEMENTS ARE ALL NEAR-TIES, which is the part that
# decides whether there is anything to fix. At each of them the FLOAT
# reference's own top-1/top-2 margin is 0.017, 0.034, 0.053 and 0.067 -
# every one below the 0.107 rms difference between the two tiers. So
# the integer tier diverges only where the float model was already
# close to indifferent, and there is no position where it overturns a
# confident float decision.
#
# THE CRITERION FOR NEXT TIME, since "97%" alone does not say whether
# that is good: a disagreement at a float margin WELL ABOVE the rms
# difference is a real defect and worth chasing. One inside it is the
# two tiers disagreeing about a coin flip, and chasing it would be
# chasing noise.
#
# AND WHAT THE AXIS IS FOR: comparing this engine against another
# implementation. Done, against HF transformers 5.5 on Qwen3.5-0.8B -
# the same safetensors directory both read, no conversion - with the
# same 153 token ids fed to each, llama98 at --fixed off and torch at
# float32:
#
#   rms diff      0.032   over 37.7M logits
#   max abs diff  0.49
#   top-1 agree   149 / 152  (98.0%)
#
# THE ALIGNMENT IS ITS OWN CONTROL. llama98 emits 152 positions for 153
# ids and transformers emits 153, so one of them has to be offset, and
# picking wrong would have been reported as poor agreement. Offset by
# one the rms is 1.68 - fifty times worse - so the head alignment is
# not a lucky guess.
#
# WHAT THIS IS NOT: a gate. It needs torch, CUDA (Qwen3.5's fused
# activation refuses to run on CPU) and a 1.7GB checkpoint, none of
# which belong in `check`. It is the measurement the knob exists to
# make possible, recorded so the next reader knows it has been made and
# what it said.
#
# NOT ATTRIBUTABLE PER OPERATOR, and that is a consequence of this
# gate's own requirement rather than an oversight: --fixed off floats
# every operator at once, so a difference cannot be split between the
# norms, the attention MACs and the transcendentals. The per-tier knobs
# that could have bisected it are the seven this gate exists to keep
# retired. Attribution would need a temporary build, not a shipped
# option.
#
# Exit codes:
#   0  PASS
#   1  FAIL - a retired knob answered, a kept one did not, or the
#      rejection detector itself is broken
#   2  SKIP - no binary or no model
set -u
cd "$(dirname "$0")/.."

BIN=./llama98.exe
[ -x "$BIN" ] || BIN=./llama98
[ -x "$BIN" ] || { echo "float_knob_gate: SKIP - no llama98; make native first"; exit 2; }

MODEL=${1:-/e/LLM/models/kmr20}
[ -d "$MODEL" ] || { echo "float_knob_gate: SKIP - no $MODEL"; exit 2; }

RETIRED="--recur-write --conv-write --attn-write --epi-write --sig-write --norm-write --scalar-fixed"
KEPT="--pair --prefetch"

# Is this option unknown to the binary? One place, so the two loops
# below cannot drift into asking different questions.
rejected() {
    "$BIN" "$MODEL" "$1" "$2" --check 2>&1 \
        | grep -qiE "unknown option|unrecognis|unrecogniz"
}

FAIL=0

# Control FIRST, and it is the one that matters: if the phrase this
# grep looks for ever changes, every retired knob reads as rejected and
# the gate passes while measuring nothing. A name that cannot exist
# must be reported as unknown.
if ! rejected --zzz-not-a-real-flag off; then
    echo "FAIL: the binary does not report --zzz-not-a-real-flag as an"
    echo "  unknown option, so this gate cannot tell a retired knob from"
    echo "  a live one and its PASS would mean nothing"
    exit 1
fi
# And the inverse control: a name that DOES exist must not be reported
# as unknown, or everything would look retired.
if rejected --fixed all; then
    echo "FAIL: --fixed all reads as an unknown option; the detector is"
    echo "  matching something other than the rejection message"
    exit 1
fi

for k in $RETIRED; do
    if ! rejected "$k" off; then
        echo "FAIL: $k is still accepted - the float axis is not one knob"
        FAIL=1
    fi
done

for v in off all; do
    if rejected --fixed "$v"; then
        echo "FAIL: --fixed $v is not accepted"
        FAIL=1
    fi
done

for k in $KEPT; do
    a=off
    [ "$k" = "--prefetch" ] && a=none
    if rejected "$k" "$a"; then
        echo "FAIL: $k was rejected; it is a performance knob and stays"
        FAIL=1
    fi
done

# The source side, as the WHOLE SET rather than one member.
#
# This counted `^int lz_float_ref_g = ` and said a second axis could not
# be added without the number moving. It could: planting
# `int lz_attn_float_g = 0;` next to it leaves the count at 1, because
# the pattern names one variable and a new axis has a new name. The
# blacklist above has the same hole one layer up - it rejects seven
# names and cannot see an eighth.
#
# So assert the set. ops_sched.c is where a tier selector lives, and a
# non-static global here IS an axis:
#
#   g_kernel          the ISA selector - --kernel. Kept: MMX/SSE/SSE2
#                     stay by requirement.
#   g_pair            --pair. A performance knob; perf_knob_gate holds
#                     it to not changing any number.
#   lz_bf16_store_g   --bf16-store. Listed TWICE on purpose, and the
#                     duplication is not a typo to tidy up: ops_sched.c
#                     declares it under `#if`/`#else` (`:383` = 1 for one
#                     side, `:385` = 0 for the other), and the comparison
#                     below is over the MATCHED SOURCE TEXT - it does not
#                     dedupe. Naming it once leaves the gate red with a
#                     diff that looks like a misspelling.
#   lz_fastfp_g       --fastfp.
#   lz_float_ref_g    --fixed. The float axis, and the only one.
#
# A fourth of any name reddens. That is the safe direction: a new knob
# has to be argued for here rather than arriving silently.
EXPECT_GLOBALS="g_kernel g_pair lz_bf16_store_g lz_bf16_store_g lz_fastfp_g lz_float_ref_g"
GOT=$(grep -oE "^(int|unsigned|long|float|double)[a-zA-Z0-9_ ]*\**([a-zA-Z_][a-zA-Z0-9_]*) *=" \
      src/ops_sched.c | grep -oE "[a-zA-Z_][a-zA-Z0-9_]* *=$" | sed 's/ *=$//' | sort | tr '\n' ' ')
WANT=$(printf '%s\n' $EXPECT_GLOBALS | sort | tr '\n' ' ')
if [ "$GOT" != "$WANT" ]; then
    echo "FAIL: ops_sched.c's non-static globals are"
    echo "        $GOT"
    echo "      expected"
    echo "        $WANT"
    echo "      Each one is a runtime axis. A new one is a new knob and"
    echo "      needs a line in this gate saying which requirement keeps it."
    FAIL=1
fi

# The CLI side, by SHAPE rather than by name, for the same reason. The
# seven retired flags were --*-write and --scalar-fixed; an eighth would
# be spelled the same way and the blacklist would not see it.
STRAY=$(grep -ohE '"--[a-z0-9-]+"' src/cli_main.c | tr -d '"' | sort -u \
        | grep -E -- "-write$|fixed" | grep -v "^--fixed$" | tr '\n' ' ')
if [ -n "$STRAY" ]; then
    echo "FAIL: option(s) shaped like a per-tier float knob: $STRAY"
    echo "      --fixed is the whole axis; these are the shape the seven"
    echo "      retired ones had."
    FAIL=1
fi

# Mutation for both, because each was just found to be a rule that could
# not fail. A planted global and a planted option must each redden.
MG=$(printf 'int lz_float_ref_g = 0;\nint lz_attn_float_g = 0;\n' \
     | grep -oE "^(int|unsigned|long|float|double)[a-zA-Z0-9_ ]*\**([a-zA-Z_][a-zA-Z0-9_]*) *=" \
     | grep -c "lz_attn_float_g")
MO=$(printf '"--attn-write"\n"--fixed"\n' | tr -d '"' \
     | grep -E -- "-write$|fixed" | grep -vc "^--fixed$")
if [ "$MG" -ne 1 ] || [ "$MO" -ne 1 ]; then
    echo "FAIL: the planted second axis reports global=$MG option=$MO;"
    echo "      a zero means the rule above cannot fail and its silence"
    echo "      carries nothing"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo "float_knob_gate: PASS (1 axis, 7 retired names rejected,"
    echo "  2 performance knobs kept; ops_sched.c's global set is exactly"
    echo "  {${WANT% }}, no option is shaped like a per-tier float knob,"
    echo "  and both planted axes redden)"
    exit 0
fi
echo "float_knob_gate: FAIL"
exit 1
