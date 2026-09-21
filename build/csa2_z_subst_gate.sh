#!/bin/bash
# A-2: CED's substitution reaches BOTH compressor inputs, not just C.
#
# Report Eq(1) (SS2.2, PDF p.9): `C_l = H_{L/2} W^{KV}_l , Z_l =
# H_{L/2} W^{Z}_l , l > L/2`. The reference implementation says the same
# structurally - .prof/dsv41_ref/model.py's Compressor.forward does
# `kv, score = self.wkv(x), self.wgate(x)` off ONE `x` - and the training
# side already did it (engine/csa2.py's csa2_compress takes one H and
# feeds both projections). The engine fed `Z` this position's own hidden
# while feeding `C` the substituted one.
#
# WHY THIS NEEDS ITS OWN FIXTURE, AND WHY IT RAN UNSEEN. At
# csa2_compress_m == 1 the softmax over a block's single position is 1
# whatever the gate logits are, so Z drops out of the pooled entry
# entirely - and every other fixture in this tree's gates runs at m == 1.
# The probe therefore runs at m=2, which is the point of it rather than a
# detail: change that back to 1 and the gate becomes blind while staying
# green. csa2_mode_dispatch_gate.sh's check4a asks the neighbouring
# question (does this layer read kv_in at all) at m=1 and cannot see this
# one.
#
# THE CRITERION is intrinsic - no reference value is compared against.
# With kv_in non-NULL, the finalized compressed entry must not move when
# the position's own hidden moves, and must move when kv_in moves. The
# second arm is the control: without it the first would also pass for a
# layer whose compression read nothing at all. Bit-exact comparison, not
# a tolerance - the arms feed identical weights and identical kv_in, so
# any difference is the input leaking through, and a tolerance would only
# make that harder to see.
#
# THE MUTANT is the one-line substitution this commit is about, reversed:
# `kvh` back to `h`. It must redden the a2 assertion and nothing else -
# the setup and control arms are asserted to stay green, so a mutant that
# broke the fixture is reported as not clean rather than counted.
#
# Exit: 0 pass, 1 fail, 2 skip (no compiler / no python3).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_z_subst_gate: SKIP - no C compiler"; exit 2; }
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_z_subst_gate: SKIP - no python3"; exit 2; }

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2zsubstgate.$$)
mkdir -p "$W/fixed" "$W/mut1"
trap 'rm -rf "$W"' EXIT

for d in fixed mut1; do
    cp src/*.c src/*.h "$W/$d/" 2>/dev/null
done

NEEDLE='            lz_matmul_w(s->csa2_z_j, kvh, &L->csa2_w_gate, hdim, hd, s->xq, s->xqs);'
N=$(grep -Fc "$NEEDLE" "$W/fixed/forward_csa2.c")
if [ "$N" != "1" ]; then
    echo "csa2_z_subst_gate: FAIL - expected exactly 1 occurrence of the Z" \
         "projection line, found $N (the gate no longer matches the source)"
    exit 1
fi
# The C line next to it must already read kvh: if someone reverts the pair
# together, the mutation below would no longer isolate anything.
N=$(grep -Fc 'lz_matmul_w(s->csa2_c_j, kvh, &L->csa2_w_kv' "$W/fixed/forward_csa2.c")
if [ "$N" != "1" ]; then
    echo "csa2_z_subst_gate: FAIL - the C projection is not reading kvh either," \
         "so this gate's mutation would not isolate Z (found $N)"
    exit 1
fi

echo "-- mutant: the Z projection's input put back to this position's own h --"
python3 - "$W/mut1/forward_csa2.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = "            lz_matmul_w(s->csa2_z_j, kvh, &L->csa2_w_gate, hdim, hd, s->xq, s->xqs);\n"
new = "            lz_matmul_w(s->csa2_z_j, h, &L->csa2_w_gate, hdim, hd, s->xq, s->xqs);\n"
if src.count(old) != 1:
    print("csa2_z_subst_gate: FAIL - the mutation could not find exactly one "
          "copy of the Z projection line", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src.replace(old, new, 1))
PYEOF
grep -qF "lz_matmul_w(s->csa2_z_j, kvh," "$W/mut1/forward_csa2.c" && {
    echo "csa2_z_subst_gate: FAIL - the mutation did not take"; exit 1; }

ENGX="err compat lfn json safetensors model ops ops_kernel ops_rope ops_moe ops_epi ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn ops_conv1d fwht forward forward_attn forward_moe forward_ssm forward_kda forward_engram forward_csa2 forward_mtp forward_rope forward_ckpt forward_state sampler generate lz_mathf lz_bf16 lz4d tokenizer unicode chat gbk cpucheck"
ENGX_MMX="ops_mmx ops_mmx_sse ops_sse2"
ENGX_AVX2="ops_avx2"

build_and_run() {
    local dir="$1" label="$2" f
    for f in $ENGX_MMX; do
        $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mmmx -msse -msse2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_z_subst_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"; return 2; }
    done
    $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
        -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mavx2 \
        -c -o "$dir/ops_avx2.o" "$dir/ops_avx2.c" 2>"$dir/cc_avx2.log" || {
        echo "csa2_z_subst_gate: FAIL - ops_avx2.c did not build against $label"
        tail -n 20 "$dir/cc_avx2.log"; return 2; }
    if ! $CC -O0 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            $(for f in $ENGX_MMX $ENGX_AVX2; do echo "$dir/$f.o"; done) \
            build/csa2_z_subst_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_z_subst_gate: FAIL - probe did not build/link against $label"
        tail -n 30 "$dir/cc.log"; return 2
    fi
    ( cd "$dir" && ./probe > run.log 2>&1 )
    local rc=$?
    cat "$dir/run.log"
    return $rc
}

RC=0

echo ""
echo "-- current tree (both arms must hold) --"
build_and_run "$W/fixed" "the current tree"
FIXED_RC=$?
if [ "$FIXED_RC" -eq 2 ]; then exit 1; fi
if [ "$FIXED_RC" -ne 0 ]; then
    echo "csa2_z_subst_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi

echo ""
echo "-- mutant: Z reads this position's own hidden again --"
build_and_run "$W/mut1" "the mutant"
M1_RC=$?
if [ "$M1_RC" -eq 2 ]; then exit 1; fi
# THE ASSERTION: it must redden, and specifically on the a2 line.
if grep -qF "PASS a2: the compressed entry does NOT move" "$W/mut1/run.log"; then
    echo "csa2_z_subst_gate: FAIL - the mutant still produced a position-" \
         "independent entry; this probe cannot see the defect it names"
    RC=1
else
    echo "  the a2 assertion failed under the mutant (expected)"
fi
# ...and nothing else may have moved. A mutant that also broke the fixture
# would produce that red for the wrong reason.
if ! grep -qF "PASS setup: the compressed entry is not the zero vector" \
        "$W/mut1/run.log"; then
    echo "csa2_z_subst_gate: FAIL - the mutant broke the fixture itself (no" \
         "block closed), so the red above proves nothing"; RC=1
fi
if ! grep -qF "PASS a2 control: ...and it DOES move when kv_in moves" \
        "$W/mut1/run.log"; then
    echo "csa2_z_subst_gate: FAIL - the mutant reddened the CONTROL arm too," \
         "so its red is indistinguishable from a compression that reads nothing"
    RC=1
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_z_subst_gate: FAIL"
    exit 1
fi
echo "csa2_z_subst_gate: PASS"
exit 0
