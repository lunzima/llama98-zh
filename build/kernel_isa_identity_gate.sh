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
# The BIN column is there for the same row. Turning the epilogue off
# used to be `--epi-write float`, one of seven per-tier runtime flags;
# those collapsed into the single --fixed knob, and --fixed off would
# take conv, norms, sigmoid, recurrence and attention out of the
# comparison too - which is the coverage this row exists for. So the
# axis moved to a COMPILE switch, LZ_EPI_FIXED, and the row runs against
# a second binary built with it at 0. Measured, not assumed:
# LZ_EPI_FIXED=0 makes ref and sse2 byte-identical on kmr20/zh, and the
# ordinary build (the control) leaves them differing - the same pair of
# results --epi-write float used to give.
#
# Adding a flag set here is the cheap half of covering a new operator end
# to end; the expensive half (SIMD arm against the same binary's C arm,
# byte for byte) is the per-operator .prof/*_xcheck.sh probe.
RBIN="./$BIN"

# The LZ_EPI_FIXED=0 control build. Made here rather than kept in the
# tree because it is one define away from the ordinary build and a
# checked-in second binary is a stale-artifact problem waiting to
# happen. EXTRA_DEFS is the Makefile's hook, so this does not restate
# -O3 -march=x86-64 and the TU defines and cannot drift from them.
# If it will not build, the row SKIPs with its reason rather than
# quietly leaving ref uncompared.
EPI0_BIN=""
EPI0_WBIN=""
EPI0_DIR="$TMPD/epi0"
mkdir -p "$EPI0_DIR"
if tar -cf - --exclude=.git --exclude=.prof --exclude='*.o' \
        --exclude='*.obj' --exclude='*.exe' . 2>/dev/null \
        | tar -x -C "$EPI0_DIR" 2>/dev/null; then
    if command -v gcc >/dev/null 2>&1; then
        if ( cd "$EPI0_DIR" && rm -rf build/native \
             && make EXTRA_DEFS=-DLZ_EPI_FIXED=0 -j4 ) \
                >"$TMPD/epi0.gcc.log" 2>&1; then
            for c in "$EPI0_DIR/llama98.exe" "$EPI0_DIR/llama98"; do
                [ -x "$c" ] && EPI0_BIN=$c && break
            done
        fi
        [ -n "$EPI0_BIN" ] || {
            echo "note: gcc LZ_EPI_FIXED=0 build failed, epi row will SKIP"
            tail -n 3 "$TMPD/epi0.gcc.log" 2>/dev/null | sed 's/^/      /'; }
    else
        echo "note: no gcc, epi row will SKIP"
    fi
    # Watcom control, for the cross-compiler block's x87 rows only.
    if [ -x /c/WATCOM/binnt64/wcc386.exe ]; then
        if ( cd "$EPI0_DIR" && rm -rf build/watcom/cli \
             && make watcom-cli EXTRA_DEFS=-DLZ_EPI_FIXED=0 ) \
                >"$TMPD/epi0.wat.log" 2>&1; then
            for c in "$EPI0_DIR/llama98-w.exe" "$EPI0_DIR/llama98-w"; do
                [ -x "$c" ] && EPI0_WBIN=$c && break
            done
        fi
        [ -n "$EPI0_WBIN" ] || {
            echo "note: Watcom LZ_EPI_FIXED=0 build failed, x87 rows will SKIP"
            tail -n 3 "$TMPD/epi0.wat.log" 2>/dev/null | sed 's/^/      /'; }
    fi
else
    echo "note: could not copy the tree, epi rows will SKIP"
fi

FLAGSETS="dflt:mmx:sse2:$RBIN:
rot:mmx:sse2:$RBIN:--kv-rot on
epi:ref:sse2:$EPI0_BIN:"

FAIL=0
PASS=0
for fx in zh en; do
    TOK="build/norm_int_gate_tokens_$fx.txt"
    [ -f "$TOK" ] || { echo "SKIP: fixture missing: $TOK"; continue; }
    while IFS= read -r fs; do
        [ -n "$fs" ] || continue
        lbl=${fs%%:*};   rest=${fs#*:}
        ta=${rest%%:*};  rest=${rest#*:}
        tb=${rest%%:*};  rest=${rest#*:}
        rbin=${rest%%:*}
        flags=${rest#*:}
        if [ -z "$rbin" ]; then
            echo "SKIP: $fx / $lbl - its binary was not built"
            continue
        fi
        ok=1
        for k in "$ta" "$tb"; do
            if ! "$rbin" "$MODEL" --tokens-file "$TOK" --kernel "$k" \
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

# ---------------------------------------------------------------------
# CROSS-COMPILER, PER TIER. The loop above compares tiers inside ONE
# binary; fixed_tier_xcompiler_gate.sh compares gcc against Watcom but
# never pins --kernel (the string does not appear in it), so it has only
# ever compared whatever `auto` resolved to on each side - sse2 on both.
# That leaves the pair iron law 2 is actually about untested: the MMX
# and SSE2 kernels are written TWICE, once as gcc intrinsics and once as
# wcc386 #pragma aux, and nothing compared intrinsic against assembly
# per tier.
#
# It also leaves out x87 entirely. gcc on x86-64 compiles scalar float
# to SSE; the only x87 in this tree is the Watcom build's `ref` tier, so
# "does the C reference give the same answer on an 80-bit stack as on
# SSE" was a question no gate asked. It is the question
# lz_fpu_float_begin/end (PC=24) exists to answer, and the answer should
# be yes - but should-be is not a measurement.
#
# MEASURED, kmr20, zh corpus, 6 arms:
#   shipping default   gcc/wat ref  agree     (x87 == SSE scalar)
#                      gcc/wat mmx  agree     )  and all four SIMD arms
#                      gcc/wat sse2 agree     )  share one hash
#                      ref != the SIMD four   (the epilogue ref has not)
#   --epi-write float  ALL SIX collapse to one hash
#
# The soft-float fourth architecture is NOT here: ARMv5TE has no FPU and
# its arm is build/arm/parity_gate.sh, which pins `ref` on both sides.
# ref is in the six above, so the chain closes transitively - soft-float
# == x86 ref == every other x86 tier - but only while both gates use the
# same pin. If either changes its pins, that chain breaks silently.
#
# MUTATION-TESTED, and both mutations were surgical - one arm's flags
# changed, only the assertions fed by that arm went red:
#   --kv q4 on the gcc ref arm  -> x87-vs-SSE and all-six FAIL,
#                                  both twin pairs stay green
#   --kv q4 on the wat sse2 arm -> the sse2 twin pair FAILs alone, the
#                                  mmx pair, x87 and all-six stay green
# The six intra-binary rows stayed green through both, which is what
# says these four assertions are new coverage rather than a restatement.
WBIN=""
for c in llama98-w.exe llama98-w; do [ -x "$c" ] && WBIN=$c && break; done
if [ -z "$WBIN" ] && [ -x /c/WATCOM/binnt64/wcc386.exe ]; then
    if make watcom-cli >"$TMPD/wat.build.log" 2>&1; then
        for c in llama98-w.exe llama98-w; do [ -x "$c" ] && WBIN=$c && break; done
    else
        echo "  cross-compiler: SKIP - 'make watcom-cli' failed"
        tail -n 3 "$TMPD/wat.build.log" | sed 's/^/      /'
    fi
fi
if [ -n "$WBIN" ] && ! "./$WBIN" "$MODEL" --check >"$TMPD/wchk.log" 2>&1; then
    echo "  cross-compiler: SKIP - $WBIN does not run on this host"
    WBIN=""
fi
if [ -z "$WBIN" ]; then
    echo "  cross-compiler: SKIP - no Watcom binary (wcc386 absent, or a"
    echo "      host that cannot execute a 32-bit Windows image)"
else
    # ONE FIXTURE, and it is a budget decision with a stated cost: the
    # Watcom `ref` tier takes 89 s on this host against gcc's 30 s and
    # 14 s for its own SIMD, so the full 2x2x3 is twelve minutes. The en
    # corpus is not uncovered - fixed_tier_xcompiler_gate.sh runs BOTH
    # corpora across every write axis, on the auto tier - it is the
    # per-TIER cross-compiler pairing that runs on zh only.
    #
    # FIVE RUNS PER COMPILER, NOT SIX. `ref` is run once, under
    # --epi-write float, and its result is used for both rows: ref
    # reaches matmul_scalar_ref before any epilogue is selected, so that
    # axis cannot move it. Measured rather than assumed - gcc ref and
    # Watcom ref give the same hash with the flag and without it - and
    # the structural reason is in this gate's header.
    fx=zh
    TOK="build/norm_int_gate_tokens_$fx.txt"
    if [ ! -f "$TOK" ]; then
        echo "  cross-compiler: SKIP - fixture missing: $TOK"
    else
        n=0
        # Two binaries per compiler now, not two flagsets on one: the
        # `epi` rows turn the epilogue off through LZ_EPI_FIXED, a
        # compile switch, because the runtime flag they used to use
        # (--epi-write float) is gone with the other six per-tier flags.
        # A missing control build SKIPs its rows rather than silently
        # dropping the count check below.
        for cc in gcc wat; do
            b=$RBIN; e=$EPI0_BIN
            [ "$cc" = wat ] && { b="./$WBIN"; e=$EPI0_WBIN; }
            for spec in "ref:epi" "mmx:dflt" "sse2:dflt" \
                        "mmx:epi" "sse2:epi"; do
                k=${spec%%:*};   lbl=${spec#*:}
                rb=$b; [ "$lbl" = epi ] && rb=$e
                if [ -z "$rb" ]; then
                    echo "  cross-compiler: SKIP $cc / $k / $lbl -" \
                         "no LZ_EPI_FIXED=0 build"
                    continue
                fi
                if ! "$rb" "$MODEL" --tokens-file "$TOK" --kernel "$k" \
                        --dump-all-logits "$TMPD/x.$cc.$k.$lbl.bin" \
                        >"$TMPD/x.$cc.$k.$lbl.log" 2>&1; then
                    echo "FAIL: $cc / --kernel $k / $lbl did not run"
                    tail -n 3 "$TMPD/x.$cc.$k.$lbl.log" | sed 's/^/      /'
                    FAIL=1; continue
                fi
                n=$((n + 1))
            done
        done
        if [ "$n" -ne 10 ]; then
            echo "  cross-compiler: incomplete ($n/10 runs), nothing compared"
        else
            # Twin pairs as SHIPPED: gcc intrinsics against wcc386
            # #pragma aux, per tier, on the default flags.
            for k in mmx sse2; do
                if cmp -s "$TMPD/x.gcc.$k.dflt.bin" "$TMPD/x.wat.$k.dflt.bin"; then
                    echo "PASS: $fx / xcc - gcc $k == watcom $k (intrinsics vs asm)"
                    PASS=$((PASS + 1))
                else
                    ND=$(cmp -l "$TMPD/x.gcc.$k.dflt.bin" \
                                "$TMPD/x.wat.$k.dflt.bin" | wc -l)
                    echo "FAIL: $fx / xcc - gcc $k and watcom $k differ in $ND bytes"
                    FAIL=1
                fi
            done
            # x87 against SSE scalar: the same C, two float units.
            if cmp -s "$TMPD/x.gcc.ref.epi.bin" "$TMPD/x.wat.ref.epi.bin"; then
                echo "PASS: $fx / xcc - gcc ref == watcom ref (SSE scalar vs x87)"
                PASS=$((PASS + 1))
            else
                ND=$(cmp -l "$TMPD/x.gcc.ref.epi.bin" \
                            "$TMPD/x.wat.ref.epi.bin" | wc -l)
                echo "FAIL: $fx / xcc - SSE scalar and x87 differ in $ND bytes"
                FAIL=1
            fi
            # And with the epilogue axis off, the six are ONE value.
            allsame=1
            for f in "$TMPD"/x.gcc.ref.epi.bin "$TMPD"/x.wat.ref.epi.bin \
                     "$TMPD"/x.gcc.mmx.epi.bin "$TMPD"/x.wat.mmx.epi.bin \
                     "$TMPD"/x.gcc.sse2.epi.bin "$TMPD"/x.wat.sse2.epi.bin; do
                cmp -s "$TMPD/x.gcc.ref.epi.bin" "$f" || allsame=0
            done
            if [ "$allsame" = 1 ]; then
                echo "PASS: $fx / xcc - all six arms (SSE, MMX, x87; two" \
                     "independent implementations) are one value"
                PASS=$((PASS + 1))
            else
                echo "FAIL: $fx / xcc - the six arms are not one value"
                FAIL=1
            fi
        fi
    fi
fi

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
