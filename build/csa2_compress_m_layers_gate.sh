#!/bin/bash
# Per-layer csa2_compress_m - config.json's "csa2_compress_m_layers"
# object (model.h's own csa2_compress_m comment on LZModelConfig has the
# schema; build/csa2_compress_m_layers_probe.c has the fixture and the
# full statement of what the criterion does and does not prove).
#
# WHAT THIS GATE IS FOR. The compression ratio is a training-time
# decision, and the selection decision this project is acting on
# ("make the ratio conservative") only becomes executable per layer:
# prefill is input-dense and can be compressed, decode has to locate
# history token by token and cannot. DeepSeek-V4.1-Flash is itself
# per-layer - 18 encoder layers at ratio 2, 20 decoder layers at ratio 1
# - and this engine carried that whole axis as one int. The criterion is
# therefore "can two layers run at two ratios at once", not "is either
# ratio better"; the second question needs training, which no checkpoint
# in this tree has.
#
# THE ARMS, AND WHY THE TWO CONTROLS ARE PART OF THE CRITERION:
#
#   fixed arm    build/csa2_compress_m_layers_probe.c against the current
#                src/. Its check3 is the criterion (scalar 2, layer 2
#                named 4, both must publish their own visibility
#                sequence). check1 (scalar 2 alone) and check2 (scalar 4
#                alone) are controls: check2 requires the SAME observable
#                to move when cm moves, so a red check3 that comes with a
#                green check2 is the missing surface rather than a blind
#                instrument, and a red check2 is reported as a broken
#                fixture. check4's nine lines are the load-time refusals
#                (a value <= 0, a value past seq_len, a layer index past
#                n_layers, a layer that is not csa2_on, a non-numeric
#                value, and a ratio that disagrees with the one an
#                aliased layer shares its ring with).
#
#   MUTATION 1, "one layer's ratio for all of them": forward_csa2.c's own
#   `int cm = lz_csa2_cm(c, layer);` becomes `int cm = lz_csa2_cm(c, 0);`
#   - every layer consumes the ratio of layer 0. This is the smallest
#   change that keeps the config surface intact and the resolution
#   correct (model.c still answers per layer, and the allocator still
#   sizes each ci for its own ratio), so it isolates the one thing the
#   criterion claims: that each layer CONSUMES its own value. Expected
#   fallout: exactly two lines turn FAIL - check3's "layer 2 (named 4)
#   runs at its own ratio" and check3's "the two layers' published
#   sequences differ within one run". check3's layer-0 line stays PASS
#   (layer 0's own ratio IS the scalar), and every check1/check2/check4
#   line stays PASS (their configs name no layer, so layer 0's ratio is
#   every layer's ratio there). A mutant where those two lines stay PASS
#   means the criterion cannot see the defect at all; a mutant where any
#   OTHER line also changes means the probe's checks are not independent
#   - both are gate FAILURES, reported as such.
#
#   The mutation needs the per-layer consumption site to exist, and this
#   gate is written to run BEFORE that site lands - that is what the red
#   run is. When the substitution's own needle is absent the gate prints
#   a single SKIP line for the mutant arm and still runs the fixed arm,
#   so the pre-landing tree reports the failing criterion rather than a
#   shell error about a missing needle.
#
#   The mutant arm also asserts its own fallout by NAME rather than by
#   count alone, and requires the mutant's exit status to be nonzero: a
#   mutation that quietly did nothing would otherwise look exactly like a
#   mutation that worked.
#
# THE IDENTITY CASE is not in this probe. "A config with no per-layer
# object reproduces today's output bit for bit" is what the existing
# end-to-end gates (build/csa2_gate.sh, and any of the checkpoint parity
# gates) already assert, and a second, weaker copy of that assertion here
# would be a number to keep in step rather than a signal.
#
# No model dir needed: build/csa2_compress_m_layers_probe.c writes its
# own config.json fixtures and calls the real lz_load_config/
# lz_state_alloc/forward_csa2, so, like csa2_mode_dispatch_gate.sh and
# ced_proj_arm_gate.sh, this needs the whole engine body (lz_state_alloc's
# helpers span ops_quant.c/ops_norm.c/fwht.c etc.), not one translation
# unit. Every source file is copied into the scratch dir for the
# quote-include resolution reason json_offset_gate.sh documents:
# `#include "forward.h"` resolves against the including file's own
# directory before -I is ever consulted.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler, no python3).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_compress_m_layers_gate: SKIP - no C compiler"
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_compress_m_layers_gate: SKIP - no python3"
    exit 2
}

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2cmlayersgate.$$)
mkdir -p "$W/fixed" "$W/mut1"
trap 'rm -rf "$W"' EXIT

for d in fixed mut1; do
    cp src/*.c src/*.h "$W/$d/" 2>/dev/null
done

# The mutant arm's own precondition, checked as a count rather than a
# `grep -q`: grep exits 2 when it cannot read the file at all, and `if`
# reads that as "not found" - i.e. as a source that has no consumption
# site, which is the one case that is deliberately tolerated here.
CM_LINE='    int cm = lz_csa2_cm(c, layer);'
CM_N=$(grep -cF "$CM_LINE" "$W/mut1/forward_csa2.c")
MUT_ARM=1
case $CM_N in
0)  echo "csa2_compress_m_layers_gate: mutant 1 SKIP - forward_csa2.c has no" \
         "per-layer consumption site yet (pre-landing tree); running the" \
         "fixed arm only"
    MUT_ARM=0 ;;
1)  : ;;
*)  echo "csa2_compress_m_layers_gate: FAIL - expected exactly 1 occurrence of" \
         "'$CM_LINE' in forward_csa2.c, found $CM_N (the gate no longer matches" \
         "the source)"
    exit 1 ;;
esac

if [ "$MUT_ARM" = "1" ]; then
python3 - "$W/mut1/forward_csa2.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = "    int cm = lz_csa2_cm(c, layer);\n"
new = "    int cm = lz_csa2_cm(c, 0);\n"
if src.count(old) != 1:
    print("csa2_compress_m_layers_gate: FAIL - mutation 1 could not find "
          "exactly one copy of %r" % old, file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src.replace(old, new, 1))
PYEOF
fi

# The whole engine body (Makefile's $(ENG)) plus the MMX/SSE2/AVX2
# objects every gcc build links unconditionally - the same list
# csa2_mode_dispatch_gate.sh/csa2_cache_idx_alias_gate.sh use, for the
# same reason (lz_state_alloc's helpers reach ops_quant.c/ops_norm.c/
# ops_gdn.c/fwht.c, whose dispatch code references those symbols
# regardless of which kernel tier is selected).
ENGX="err compat lfn json safetensors model ops ops_kernel ops_rope ops_moe ops_epi ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn ops_conv1d fwht forward forward_attn forward_moe forward_ssm forward_kda forward_engram forward_csa2 forward_mtp forward_rope forward_ckpt forward_state sampler generate lz_mathf lz_bf16 lz4d tokenizer unicode chat gbk cpucheck"
ENGX_MMX="ops_mmx ops_mmx_sse ops_sse2"
ENGX_AVX2="ops_avx2"

build_and_run() {
    local dir="$1" label="$2"
    local f
    for f in $ENGX_MMX; do
        $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mmmx -msse -msse2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_compress_m_layers_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    for f in $ENGX_AVX2; do
        $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mavx2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_compress_m_layers_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    if ! $CC -O0 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            $(for f in $ENGX_MMX $ENGX_AVX2; do echo "$dir/$f.o"; done) \
            build/csa2_compress_m_layers_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_compress_m_layers_gate: FAIL - probe did not build/link against $label"
        tail -n 30 "$dir/cc.log"
        return 2
    fi
    ( cd "$dir" && ./probe > run.log 2>&1 )
    local rc=$?
    cat "$dir/run.log"
    return $rc
}

# ok() calls in build/csa2_compress_m_layers_probe.c: check1 4, check2 3,
# check3 4, check4 9. Counted off the control run's own output below
# rather than trusted, so a probe that stops early is a gate FAILURE
# rather than a shorter list of passing lines.
PROBE_TOTAL_LINES=20

echo "-- current tree (must PASS) --"
build_and_run "$W/fixed" "the current (fixed) tree"
FIXED_RC=$?
RC=0
FIXED_N=$(grep -cE '^(PASS|FAIL) ' "$W/fixed/run.log")
if [ "$FIXED_RC" -eq 2 ]; then
    exit 1
fi
if [ "$FIXED_N" != "$PROBE_TOTAL_LINES" ]; then
    echo "csa2_compress_m_layers_gate: FAIL - the current tree printed" \
         "$FIXED_N line(s), expected $PROBE_TOTAL_LINES - a probe that stopped" \
         "early passes a per-line check for free"
    RC=1
fi
if [ "$FIXED_RC" -ne 0 ]; then
    echo "csa2_compress_m_layers_gate: FAIL - the current tree did not pass its" \
         "own probe (see the lines above)"
    RC=1
fi

if [ "$MUT_ARM" = "1" ]; then
    echo ""
    echo "-- mutant 1: every layer consumes layer 0's ratio --"
    build_and_run "$W/mut1" "mutant 1"
    MUT_RC=$?
    if [ "$MUT_RC" -eq 2 ]; then
        exit 1
    fi
    MUT_N=$(grep -cE '^(PASS|FAIL) ' "$W/mut1/run.log")
    if [ "$MUT_N" != "$PROBE_TOTAL_LINES" ]; then
        echo "csa2_compress_m_layers_gate: FAIL - mutant 1 printed $MUT_N line(s)," \
             "expected $PROBE_TOTAL_LINES"
        RC=1
    fi
    if [ "$MUT_RC" -eq 0 ]; then
        echo "csa2_compress_m_layers_gate: FAIL - the mutant passed: the" \
             "criterion does not depend on each layer consuming its own ratio"
        RC=1
    fi
    # By NAME, not by count: a mutant that reddens some other line is a
    # probe whose checks are not independent.
    for line in \
        "check3: layer 2 (named 4) runs at its own ratio, 0,0,0,1,1,1,1,2" \
        "check3: the two layers' published sequences differ within one run"
    do
        grep -qF "FAIL $line" "$W/mut1/run.log" || {
            echo "csa2_compress_m_layers_gate: FAIL - mutant 1 did not turn FAIL" \
                 "on: $line"
            RC=1
        }
    done
    # Every other line, including check1/check2, must still be a PASS.
    if grep -E '^FAIL ' "$W/mut1/run.log" | grep -vF \
        -e "check3: layer 2 (named 4) runs at its own ratio" \
        -e "check3: the two layers' published sequences differ" | grep -q .; then
        echo "csa2_compress_m_layers_gate: FAIL - mutant 1 reddened lines it should" \
             "not have (listed above); the probe's checks are not independent"
        RC=1
    fi
    if [ "${RC:-0}" = "0" ]; then
        echo ""
        echo "csa2_compress_m_layers_gate: PASS - the criterion is red under the" \
             "mutation and green on the tree"
    fi
else
    echo ""
    echo "csa2_compress_m_layers_gate: mutant 1 SKIPPED (no consumption site in" \
         "this tree) - the fixed arm's own verdict stands above"
fi

exit "${RC:-0}"
