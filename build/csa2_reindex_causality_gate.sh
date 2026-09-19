#!/bin/bash
# C2 (report Sec2.3.1 mode-dispatch task, fifth review round): a
# REINDEX layer's n_vis leaked causal information from its already-
# finished source's whole call (REUSE's own n_vis is dead code - see
# build/csa2_reindex_causality_probe.c's own header comment for why a
# downstream REUSE layer's OUTPUT still moved anyway). That probe's
# header comment has the full argument, why it reads s->xb2 rather than
# s->attn_out, and what changed alongside the fix in build/csa2_mode_
# dispatch_probe.c.
#
# THE CONTROL is a MUTATION (forward_csa2.c's n_vis line reverted to the
# pre-fix shared-counter read), not a git-history diff, same reason
# every other gate on this plan gives.
#
# The link line is build/csa2_mode_dispatch_gate.sh's own, unchanged:
# lz_state_alloc's helpers reach ops_quant.c/ops_norm.c/ops_gdn.c/
# fwht.c, whose dispatch references the MMX/SSE2/AVX2 symbols
# unconditionally regardless of which kernel tier is selected, so the
# whole engine body plus those three objects is the minimum that links.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_reindex_causality_gate: SKIP - no C compiler"
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_reindex_causality_gate: SKIP - no python3"
    exit 2
}

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2reindexcausalitygate.$$)
mkdir -p "$W/fixed" "$W/mut"
trap 'rm -rf "$W"' EXIT

for d in fixed mut; do
    cp src/*.c src/*.h "$W/$d/" 2>/dev/null
done

# Sanity: the line the mutation targets must exist exactly once in the
# current source, or this gate no longer matches the code it names.
NEEDLE='    int n_vis = mode == LZ_CSA2_FULL ? s->csa2_n_blocks[ci] : pos / cm;'
N=$(grep -Fc "$NEEDLE" "$W/fixed/forward_csa2.c")
if [ "$N" != "1" ]; then
    echo "csa2_reindex_causality_gate: FAIL - expected exactly 1 occurrence of" \
         "the n_vis line in forward_csa2.c, found $N (the gate no longer" \
         "matches the source)"
    exit 1
fi

python3 - "$W/mut/forward_csa2.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = "    int n_vis = mode == LZ_CSA2_FULL ? s->csa2_n_blocks[ci] : pos / cm;\n"
new = "    int n_vis = s->csa2_n_blocks[ci];\n"
if src.count(old) != 1:
    print("csa2_reindex_causality_gate: FAIL - mutation could not find "
          "exactly one copy of the n_vis line", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

if grep -qF 'mode == LZ_CSA2_FULL ? s->csa2_n_blocks[ci] : pos / cm' "$W/mut/forward_csa2.c"; then
    echo "csa2_reindex_causality_gate: FAIL - could not mutate forward_csa2.c" \
         "($W/mut/forward_csa2.c still has the fixed n_vis line)"
    exit 1
fi

ENGX="err compat lfn json safetensors model ops ops_kernel ops_rope ops_moe ops_epi ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn ops_conv1d fwht forward forward_attn forward_moe forward_ssm forward_kda forward_engram forward_csa2 forward_mtp forward_rope forward_ckpt forward_state sampler generate lz_mathf lz_bf16 lz4d tokenizer unicode chat gbk cpucheck"
ENGX_MMX="ops_mmx ops_mmx_sse ops_sse2"
ENGX_AVX2="ops_avx2"

build_and_run() {
    local dir="$1" label="$2"
    local f
    for f in $ENGX_MMX; do
        $CC -O2 -std=c99 -w -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mmmx -msse -msse2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_reindex_causality_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    for f in $ENGX_AVX2; do
        $CC -O2 -std=c99 -w -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mavx2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_reindex_causality_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    if ! $CC -O0 -std=c99 -w -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            $(for f in $ENGX_MMX $ENGX_AVX2; do echo "$dir/$f.o"; done) \
            build/csa2_reindex_causality_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_reindex_causality_gate: FAIL - probe did not build/link against $label"
        tail -n 30 "$dir/cc.log"
        return 2
    fi
    ( cd "$dir" && ./probe > run.log 2>&1 )
    local rc=$?
    cat "$dir/run.log"
    return $rc
}

RC=0

echo "-- current tree (must PASS) --"
build_and_run "$W/fixed" "the current (fixed) tree"
FIXED_RC=$?
if [ "$FIXED_RC" -eq 2 ]; then
    exit 1
elif [ "$FIXED_RC" -ne 0 ]; then
    echo "csa2_reindex_causality_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi

echo ""
echo "-- mutant: n_vis reverted to the shared/aliased counter read --"
build_and_run "$W/mut" "the mutant (n_vis reverted)"
MUT_RC=$?
if [ "$MUT_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT_RC" -eq 0 ]; then
    echo "csa2_reindex_causality_gate: FAIL - the mutant PASSED; the probe cannot see this defect"
    RC=1
else
    if grep -qF "FAIL c2/check1:" "$W/mut/run.log" && grep -qF "FAIL c2/check2:" "$W/mut/run.log"; then
        echo "-- mutant failed both checks, as expected --"
    else
        echo "csa2_reindex_causality_gate: FAIL - the mutant did not fail both checks"
        RC=1
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_reindex_causality_gate: FAIL"
    exit 1
fi
echo "csa2_reindex_causality_gate: PASS"
exit 0
