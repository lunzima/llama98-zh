#!/bin/bash
# A-3: the inverse RoPE on CSA2's attention output.
#
# Step 5 of forward_csa2.c scores and then accumulates the SAME ring
# pointer (one shared K=V entry, Eq 19), and step 3 rotated every ring
# entry - so the attention output is a weighted sum of rotated values.
# The reference un-rotates it at the query's own position
# (.prof/dsv41_ref/model.py:781, `apply_rotary_emb(o[..., -rd:],
# freqs_cis, True)`), and this tree now does the same by handing lz_rope
# the conjugate table, LZRunState's own rope_cs_inv.
#
# WHAT THIS GATE CAN AND CANNOT PROVE - the distinction is the whole
# point of it, so it is here rather than in the probe:
#
#   The TABLE check has a real oracle that does not come from the
#   reference: R(a)^-1 R(a) = I, so cs-then-cs_inv must return its input.
#   Its control arm (a NON-conjugated copy of the same table) must miss
#   by orders of magnitude; a check that passed for both tables would be
#   measuring nothing. This is what catches a dropped negation, or the
#   cosines negated instead of the sines.
#
#   The WIRING check has NO oracle here. The report gives no formula for
#   the output-side rotation (only the fused-kernel mention at
#   .prof/dsv41_report.txt:969-970); .prof/dsv41_ref/model.py:781 is the
#   whole of the available specification, and it is the code this side is
#   being compared against, so it cannot also be the independent check.
#   So this gate does not assert what the output should be. It asserts
#   that the output MOVES when the un-rotation is deleted - the probe
#   prints its attention output and the fixed build's block is diffed
#   against the mutant's. "It is wired in" is the claim; "it is correct"
#   is not, and a green run here must not be read as more than that.
#
# THE MUTANTS, both text substitutions (no git refs - this tree's other
# CSA2 gates explain why):
#   mut1  the step-5b lz_rope call deleted -> the wiring comparison must
#         differ. The probe's own two table checks must STILL pass, and
#         that is asserted too: a mutation that reddened them would mean
#         it broke something other than the wiring.
#   mut2  lz_rope_conjugate_table copies instead of negating - the table
#         stops being an inverse. Its round-trip check must FAIL, and
#         specifically THAT line: the non-conjugated control must still
#         pass, which is what separates "the inverse is wrong" from "the
#         check is broken".
#
# Exit: 0 pass, 1 fail, 2 skip (no compiler / no python3).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_out_rope_gate: SKIP - no C compiler"; exit 2; }
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_out_rope_gate: SKIP - no python3"; exit 2; }

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2outropegate.$$)
mkdir -p "$W/fixed" "$W/mut1" "$W/mut2"
trap 'rm -rf "$W"' EXIT

for d in fixed mut1 mut2; do
    cp src/*.c src/*.h "$W/$d/" 2>/dev/null
done

# Every line a mutation targets must exist exactly once, or the gate no
# longer matches the source it names and would report a stale tree as a
# passed mutation.
for needle_file in \
    "forward_csa2.c:        lz_rope(s->attn_out, nh, hd, rotary_dim, pos, s->rope_cs_inv);" \
    "forward_rope.c:        dst[i] = (i & 1u) ? -src[i] : src[i];"
do
    file=${needle_file%%:*}
    needle=${needle_file#*:}
    N=$(grep -Fc "$needle" "$W/fixed/$file")
    if [ "$N" != "1" ]; then
        echo "csa2_out_rope_gate: FAIL - expected exactly 1 occurrence of" \
             "'$needle' in $file, found $N (the gate no longer matches the source)"
        exit 1
    fi
done

echo "-- mutant 1: the step-5b inverse-rotation call deleted --"
python3 - "$W/mut1/forward_csa2.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = "        lz_rope(s->attn_out, nh, hd, rotary_dim, pos, s->rope_cs_inv);\n"
if src.count(old) != 1:
    print("csa2_out_rope_gate: FAIL - mutation 1 could not find exactly one "
          "copy of the call", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src.replace(old, "", 1))
PYEOF
grep -qF "lz_rope(s->attn_out" "$W/mut1/forward_csa2.c" && {
    echo "csa2_out_rope_gate: FAIL - mutation 1 did not take"; exit 1; }

echo "-- mutant 2: the conjugate table copies instead of negating --"
python3 - "$W/mut2/forward_rope.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = "        dst[i] = (i & 1u) ? -src[i] : src[i];\n"
new = "        dst[i] = src[i];\n"
if src.count(old) != 1:
    print("csa2_out_rope_gate: FAIL - mutation 2 could not find exactly one "
          "copy of the negation", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src.replace(old, new, 1))
PYEOF
grep -qF "(i & 1u) ? -src[i]" "$W/mut2/forward_rope.c" && {
    echo "csa2_out_rope_gate: FAIL - mutation 2 did not take"; exit 1; }

# The whole engine body (Makefile's $(ENG)) plus the MMX/SSE2/AVX2
# objects every gcc build links unconditionally - the same list
# build/csa2_mode_dispatch_gate.sh uses, for the reasons its own header
# gives (lz_state_alloc's helpers reach ops_quant/ops_norm/ops_gdn/fwht,
# whose dispatch code references those symbols regardless of the kernel
# tier selected).
ENGX="err compat lfn json safetensors model ops ops_kernel ops_rope ops_moe ops_epi ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn ops_conv1d fwht forward forward_attn forward_moe forward_ssm forward_kda forward_engram forward_csa2 forward_mtp forward_rope forward_ckpt forward_state sampler generate lz_mathf lz_bf16 lz4d tokenizer unicode chat gbk cpucheck"
ENGX_MMX="ops_mmx ops_mmx_sse ops_sse2"
ENGX_AVX2="ops_avx2"

build_and_run() {
    local dir="$1" label="$2" f
    for f in $ENGX_MMX; do
        $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mmmx -msse -msse2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_out_rope_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"; return 2; }
    done
    $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
        -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mavx2 \
        -c -o "$dir/ops_avx2.o" "$dir/ops_avx2.c" 2>"$dir/cc_avx2.log" || {
        echo "csa2_out_rope_gate: FAIL - ops_avx2.c did not build against $label"
        tail -n 20 "$dir/cc_avx2.log"; return 2; }
    if ! $CC -O0 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            $(for f in $ENGX_MMX $ENGX_AVX2; do echo "$dir/$f.o"; done) \
            build/csa2_out_rope_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_out_rope_gate: FAIL - probe did not build/link against $label"
        tail -n 30 "$dir/cc.log"; return 2
    fi
    ( cd "$dir" && ./probe > run.log 2>&1 )
    local rc=$?
    cat "$dir/run.log"
    return $rc
}

RC=0

echo ""
echo "-- current tree (both checks must pass) --"
build_and_run "$W/fixed" "the current tree"
FIXED_RC=$?
if [ "$FIXED_RC" -eq 2 ]; then exit 1; fi
if [ "$FIXED_RC" -ne 0 ]; then
    echo "csa2_out_rope_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi

echo ""
echo "-- mutant 1: inverse-rotation call deleted (wiring must move) --"
build_and_run "$W/mut1" "mutant 1"
M1_RC=$?
if [ "$M1_RC" -eq 2 ]; then exit 1; fi
# The table checks are about the TABLE; deleting the call must not touch
# them. If they redden, the mutation broke something else and its wiring
# evidence would be worthless.
if ! grep -qF "PASS table: rotating by rope_cs then rope_cs_inv" "$W/mut1/run.log"; then
    echo "csa2_out_rope_gate: FAIL - mutant 1 also broke the table round-trip;"
    echo "                      it is not a clean wiring-only mutation"
    RC=1
fi
if ! grep -qF "PASS wiring: the layer ran and its attention output is finite" \
        "$W/mut1/run.log"; then
    echo "csa2_out_rope_gate: FAIL - mutant 1 did not run the layer at all"
    RC=1
fi
# THE ASSERTION: same probe, same fixture, different attention output.
if diff -q <(sed -n '/^OUT-BEGIN$/,/^OUT-END$/p' "$W/fixed/run.log") \
           <(sed -n '/^OUT-BEGIN$/,/^OUT-END$/p' "$W/mut1/run.log") >/dev/null; then
    echo "csa2_out_rope_gate: FAIL - deleting the inverse rotation did not move"
    echo "                      the attention output; the step is not wired in"
    RC=1
else
    echo "  attention output differs between the fixed build and mutant 1 (expected)"
fi

echo ""
echo "-- mutant 2: conjugation dropped (the table stops being an inverse) --"
build_and_run "$W/mut2" "mutant 2"
M2_RC=$?
if [ "$M2_RC" -eq 2 ]; then exit 1; fi
# It MUST fail, and it must be THAT line: the control arm proves the check
# can still see a non-inverse table, so a red there means the table is
# wrong rather than the check being dead.
if grep -qF "PASS table: rotating by rope_cs then rope_cs_inv" "$W/mut2/run.log"; then
    echo "csa2_out_rope_gate: FAIL - a NON-inverse table still round-tripped;"
    echo "                      the table check cannot see the difference"
    RC=1
else
    echo "  the inverse round-trip failed under mutant 2 (expected)"
fi
if ! grep -qF "PASS table: the NON-conjugated control does NOT round-trip" \
        "$W/mut2/run.log"; then
    echo "csa2_out_rope_gate: FAIL - mutant 2 reddened the CONTROL arm too, so"
    echo "                      the red above is indistinguishable from a dead check"
    RC=1
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_out_rope_gate: FAIL"
    exit 1
fi
echo "csa2_out_rope_gate: PASS"
exit 0
