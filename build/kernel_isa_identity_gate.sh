#!/bin/bash
# Kernel-ISA numeric identity gate (iron law 2, end to end).
#
# Iron law 2 says every operator's twins are bit-identical. Nothing
# checked it at the whole-engine level. kernel_matrix_gate asserts
# COVERAGE - which cells have a hand-written kernel and which are
# honestly empty - and fixed_tier_xcompiler_gate asserts gcc vs Watcom
# on the DEFAULT tier. Neither compares one ISA tier against another in
# the same binary, which is where a twin that drifted would show.
#
# That gap mattered before this gate and matters more now: the operators
# landed in the int-pipeline work (integer SwiGLU, the integer sigmoid
# and exp entries, the integer requantize) all justify themselves partly
# with "integer arithmetic is identical across ISAs by construction".
# By construction is an argument. This is the measurement.
#
# ASSERTS: --kernel mmx and --kernel sse2 produce byte-identical
# --dump-all-logits on both fixtures. One binary, one model, two tiers;
# any difference is a twin that diverged.
#
# ALSO ASSERTS: --kernel ref == --kernel sse2 under --epi-write float.
# ref differs from the SIMD tiers on the shipping default - 57,975,565
# of 78,905,344 bytes - and that reads like a diverged kernel. It is not.
# lz_matmul_xq_nt routes LZ_KERNEL_REF to matmul_scalar_ref BEFORE any
# epilogue is selected (mm_i16_dispatch refuses only that tier), so ref
# structurally has no fixed epilogue. Turning off that ONE axis makes the
# two byte-identical, which localises the whole difference to it:
#
#   --epi-write float   ref == sse2      (measured, both fixtures)
#   shipping default    ref != sse2      (the epilogue ref does not have)
#
# So the row that belongs in a gate is the first one. It covers conv,
# norms, sigmoid, recurrence, attention and the scalar tier against their
# C reference and would catch a SIMD cell that drifted in any of them;
# only the epilogue is excluded, and excluded for a reason that is
# measured rather than asserted. Dropping ref entirely - the first
# version of this gate did - throws that coverage away to avoid one
# known difference.
#
# Runs on whatever kernels the host offers: if the CPU has no SSE2 the
# tier selector refuses and the gate SKIPs rather than comparing a tier
# against itself, which would pass while measuring nothing.
#
# Exit codes: 0 PASS, 1 FAIL, 2 ERROR/SKIP.
set -u
cd "$(dirname "$0")/.." || exit 1

MODELS_ROOT=${MODELS_ROOT:-/e/LLM/models}
MODEL=${LZ_NORM_INT_GATE_MODEL:-$MODELS_ROOT/kmr20}
if [ ! -d "$MODEL" ]; then
    echo "kernel_isa_identity_gate: SKIP - model not found: $MODEL"
    exit 2
fi

BIN=llama98.exe
[ -x "$BIN" ] || BIN=llama98
if [ ! -x "$BIN" ]; then
    echo "kernel_isa_identity_gate: ERROR - no native binary; run make llama98"
    exit 2
fi

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

# Prove the two tiers are actually distinct on this host before trusting
# an "identical" result: a selector that silently resolves both names to
# the same kernel would make this gate green while measuring nothing.
SEL_MMX=$("./$BIN" "$MODEL" --kernel mmx  --check 2>&1 | grep -m1 'tier ')
SEL_SSE=$("./$BIN" "$MODEL" --kernel sse2 --check 2>&1 | grep -m1 'tier ')
if [ -z "$SEL_MMX" ] || [ -z "$SEL_SSE" ]; then
    echo "kernel_isa_identity_gate: SKIP - cannot read the selected tier"
    exit 2
fi
if [ "$SEL_MMX" = "$SEL_SSE" ]; then
    echo "kernel_isa_identity_gate: SKIP - both names resolve to the same"
    echo "  tier on this host ($SEL_MMX); nothing to compare"
    exit 2
fi
echo "mmx  -> $SEL_MMX"
echo "sse2 -> $SEL_SSE"

# FLAG SETS, not just fixtures. Default flags do not reach every kernel
# in the matrix: the float Hadamard (fwhtf, lz_fwht) only runs when
# --kv-rot is on, so with one flag set this gate never executed that
# operator once. A gate has to sample where the kernel lives, not merely
# where the model does.
#
# WHAT THE `rot` SET DOES NOT CATCH, measured rather than assumed: with
# lz_fwht's SSE body mutated (second half's pointer +1) this gate still
# reports all four combinations identical. It has to - fwhtf fills its
# sseI/sse2I cells from ONE body, so mmx and sse2 run the same
# instructions and comparing them is blind to that body by construction.
# What proves fwhtf is .prof/fwht_f32_xcheck.sh, which compares the
# kernel against the C in one binary; the same mutation moves it.
#
# The set is here for the kernels --kv-rot SWITCHES rather than the one
# it enables: s->kv_rot_v != 0 takes the paired wsum kernel out of the
# attention epilogue and folds kv_rot_k into the softmax scale, so a
# different selection of twins runs under it than under the default.
#
# One LABEL:TIER_A:TIER_B:FLAGS per LINE, and read with `while read`, not
# `for` - FLAGS contains spaces, and word-splitting a space-separated list
# turned "--kv-rot on" into a flagset named "on" that the loader then
# tried to open as a model directory. It failed loudly here; it would not
# have in a gate whose runs are allowed to be empty.
#
# The tier PAIR is per line because the `epi` row compares ref against
# sse2 while the others compare mmx against sse2 - see the header for why
# ref is only comparable once the epilogue axis is off, and why dropping
# it entirely would throw away the coverage it does bring.
#
# Adding a flag set here is the cheap half of covering a new operator end
# to end; the expensive half (SIMD arm against the same binary's C arm,
# byte for byte) is the per-operator .prof/*_xcheck.sh probe.
FLAGSETS="dflt:mmx:sse2:
rot:mmx:sse2:--kv-rot on
epi:ref:sse2:--epi-write float"

FAIL=0
PASS=0
for fx in zh en; do
    TOK="build/norm_int_gate_tokens_$fx.txt"
    [ -f "$TOK" ] || { echo "SKIP: fixture missing: $TOK"; continue; }
    while IFS= read -r fs; do
        [ -n "$fs" ] || continue
        lbl=${fs%%:*};   rest=${fs#*:}
        ta=${rest%%:*};  rest=${rest#*:}
        tb=${rest%%:*}
        flags=${rest#*:}
        ok=1
        for k in "$ta" "$tb"; do
            if ! "./$BIN" "$MODEL" --tokens-file "$TOK" --kernel "$k" \
                    $flags --dump-all-logits "$TMPD/$k.bin" \
                    >"$TMPD/$k.log" 2>&1; then
                echo "FAIL: --kernel $k / $fx / $lbl did not run"
                tail -n 3 "$TMPD/$k.log" | sed 's/^/      /'
                FAIL=1
                ok=0
                break
            fi
        done
        [ "$ok" = 1 ] || continue
        SZ=$(wc -c <"$TMPD/$ta.bin")
        if [ "$SZ" -eq 0 ]; then
            echo "FAIL: $fx / $lbl - empty dump"
            FAIL=1
            continue
        fi
        if cmp -s "$TMPD/$ta.bin" "$TMPD/$tb.bin"; then
            echo "PASS: $fx / $lbl - $ta == $tb ($SZ bytes)"
            PASS=$((PASS + 1))
        else
            ND=$(cmp -l "$TMPD/$ta.bin" "$TMPD/$tb.bin" 2>/dev/null | wc -l)
            echo "FAIL: $fx / $lbl - $ta and $tb differ in $ND of $SZ bytes"
            FAIL=1
        fi
    done <<EOF
$FLAGSETS
EOF
done

echo "kernel_isa_identity_gate: $PASS fixture x flagset combination(s) identical"
# FAIL is checked FIRST. An earlier order asked "did anything pass?"
# before "did anything fail?", so a run where BOTH fixtures diverged
# reported SKIP and exited 2 - and ci.sh counts SKIP as not-a-failure.
# The mutation that proves this gate has teeth is exactly the run that
# exposed it: a gate whose total failure reads as "nothing ran" is worse
# than no gate.
if [ "$FAIL" -ne 0 ]; then
    echo "kernel_isa_identity_gate: FAIL"
    exit 1
fi
if [ "$PASS" -gt 0 ]; then
    echo "kernel_isa_identity_gate: PASS"
    exit 0
fi
echo "kernel_isa_identity_gate: SKIP - no fixture ran"
exit 2
