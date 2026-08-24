#!/bin/bash
# Kernel-ISA numeric identity gate, end to end.
#
# Every operator's twins must be bit-identical. Nothing
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
# HOW MUCH OF THE TWINS THAT REACHES, measured rather than assumed: a
# coverage build run at --kernel mmx over the zh fixture executes 47.57%
# of the 103 lines in src/ops_kernel_dot_mmx.h. The rest is the arms for
# weight formats this model does not use. So a PASS here means "the
# kernels this fixture exercises agree", and the other half is covered
# by registration (kernel_matrix_gate) and by having a twin at all
# (twin_pair_gate) - neither of which compares numbers.
#
# Closing that needs fixtures whose weights span the formats, not a
# change to this script. Written down because "12 combinations
# identical" reads like every kernel was compared, and it is not what
# was measured.
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

# IS THAT BINARY THIS SOURCE? Every row below compares it against a
# binary this script BUILDS from the working tree, so the two differ by
# the switch under test only if the pre-built one came from the same
# sources. Nothing checked that, and the failure sentence names a cause
# the script never established: "SSE1 and SSE2 differ in 57813945 of
# 78905344 bytes" on a run where the SSE1 side was built minutes ago and
# ./llama98.exe was two hours and several commits old. The same byte
# count appeared in three rows, which is the tell - one stale binary,
# not three ISA defects.
#
# mtime, because that is the relation make itself uses within one tree,
# and the direction is the safe one: refuse to report rather than report
# a wrong cause. A source newer than the binary is a SKIP with the
# command to run, never a FAIL.
#
# Positive control, in the direction this can go quiet: an empty result
# has to mean "nothing is newer", not "find -newer answered nothing".
# A marker dated 2000 must come back with sources newer than it.
OLDMARK=$(mktemp) && touch -t 200001010000 "$OLDMARK" 2>/dev/null
if [ -n "${OLDMARK:-}" ] && \
   [ -z "$(find src -name '*.c' -newer "$OLDMARK" -print 2>/dev/null | head -1)" ]; then
    echo "kernel_isa_identity_gate: ERROR - 'find -newer' reports nothing"
    echo "  newer than a marker dated 2000, so the freshness check below"
    echo "  cannot fail and its silence would carry nothing"
    rm -f "$OLDMARK"
    exit 2
fi
rm -f "$OLDMARK"

NEWER=$(find src common -name '*.c' -newer "$BIN" -print 2>/dev/null | head -4)
if [ -n "$NEWER" ]; then
    echo "kernel_isa_identity_gate: SKIP - $BIN is older than its sources,"
    echo "  so every row below would compare two different programs and"
    echo "  report the difference as an ISA divergence. Newer:"
    printf '%s\n' "$NEWER" | sed 's/^/      /'
    echo "  Run 'make llama98' first."
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

# THE SSE1 ROWS NEED A DIFFERENT OBSERVABLE, and the one above would lie
# about them. SSE1 adds no integer ROW kernel, so `--kernel sse` and
# `--kernel mmx` both report "tier mmx-intrin" - identical on the line
# checked above, while the operator bodies underneath differ. The
# recurrence line is where the three separate, so that is what the
# distinctness check reads for them. Without it the sse/ssem/ssef rows
# would pass on a host that clamped sse away, measuring nothing.
imp_of() { "./$BIN" "$MODEL" --kernel "$1" --check 2>&1 | \
           grep -m1 -o 'recurrence impl [a-z0-9-]*'; }
IMP_MMX=$(imp_of mmx); IMP_SSE1=$(imp_of sse); IMP_SSE2=$(imp_of sse2)
SSE1_ROWS=1
if [ -z "$IMP_MMX" ] || [ -z "$IMP_SSE1" ] || [ -z "$IMP_SSE2" ] ||
   [ "$IMP_MMX" = "$IMP_SSE1" ] || [ "$IMP_SSE1" = "$IMP_SSE2" ]; then
    echo "note: SSE1 rows will SKIP - mmx/sse/sse2 do not resolve to three"
    echo "  distinct bodies here (mmx=$IMP_MMX sse=$IMP_SSE1 sse2=$IMP_SSE2)"
    SSE1_ROWS=0
else
    echo "sse  -> $IMP_SSE1 (mmx=$IMP_MMX sse2=$IMP_SSE2)"
fi

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
    # THE SSE1 ARM. SSE1 is not a --kernel tier and cannot be: the row
    # kernels' sseI/sseA cells are `i`, impossible, with the
    # instruction-by-instruction reason registered in the matrix (no
    # signed 16x16->32 MAC, no integer %xmm). But ONE operator does have
    # a real SSE1 body - the q8 rounding kernel, lz_q8round32_sse - and
    # lz_q8r_tier picks it on a machine with SSE and no SSE2. That tier
    # had no comparison: switch_matrix_gate compiles LZ_Q8R_FORCE_SSE=1
    # and nothing ran it, the same gap LZ_GDN_STATE_F32 sat in.
    #
    # Compared against SSE2 with --kernel sse2 on BOTH sides, so the q8r
    # tier is the only thing that moves - forcing it via the compile
    # switch rather than via --kernel ref, which would change the matmul
    # kernel too and blur what the row measures.
    if [ -n "$EPI0_BIN" ]; then
        SSE1_DIR="$TMPD/sse1"
        mkdir -p "$SSE1_DIR"
        if tar -cf - --exclude=.git --exclude=.prof --exclude='*.o' \
                --exclude='*.obj' --exclude='*.exe' . 2>/dev/null \
                | tar -x -C "$SSE1_DIR" 2>/dev/null \
           && ( cd "$SSE1_DIR" && rm -rf build/native \
                && make EXTRA_DEFS=-DLZ_Q8R_FORCE_SSE=1 -j4 ) \
                   >"$TMPD/sse1.log" 2>&1; then
            for c in "$SSE1_DIR/llama98.exe" "$SSE1_DIR/llama98"; do
                [ -x "$c" ] && SSE1_BIN=$c && break
            done
        fi
        [ -n "${SSE1_BIN:-}" ] || {
            echo "note: LZ_Q8R_FORCE_SSE=1 build failed, q8r row will SKIP"
            tail -n 3 "$TMPD/sse1.log" 2>/dev/null | sed 's/^/      /'; }
    fi
else
    echo "note: could not copy the tree, epi rows will SKIP"
fi
SSE1_BIN=${SSE1_BIN:-}

# sse and ssem are SSE1 as a real tier. It used to be reachable only by
# CPUID behind g_kernel's back, which is why --kernel mmx measured as
# "recurrence impl sse-intrin" before LZ_KERNEL_SSE existed - the mmx row
# above was comparing SSE1 against SSE2 and calling it MMX. ssem is the
# comparison the deleted P2 and EXP_Q20 control builds were standing in
# for; ssef reaches the SSE1 float matmul row, which the fixed default
# does not.
FLAGSETS="dflt:mmx:sse2:$RBIN:
rot:mmx:sse2:$RBIN:--kv-rot on
sse:sse:sse2:$RBIN:
ssem:mmx:sse:$RBIN:
ssef:mmx:sse:$RBIN:--fixed off
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
        case "$lbl" in
        sse|ssem|ssef)
            [ "$SSE1_ROWS" = 1 ] || {
                echo "SKIP: $fx / $lbl - sse is not distinct on this host"
                continue
            } ;;
        esac
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
# THE q8-ROUNDING TIER, SSE1 against SSE2. Two BINARIES at one --kernel,
# not two --kernels in one binary, so it does not fit the loop above:
# the tier is chosen by lz_q8r_tier, and the only way to pin its SSE1
# value on an SSE2 host is the compile switch.
if [ -z "$SSE1_BIN" ]; then
    echo "  q8r: SKIP - no LZ_Q8R_FORCE_SSE=1 build"
else
    for fx in zh en; do
        TOK="build/norm_int_gate_tokens_$fx.txt"
        [ -f "$TOK" ] || { echo "  q8r: SKIP - fixture missing: $TOK"; continue; }
        ok=1
        for side in sse1 sse2; do
            b=$RBIN; [ "$side" = sse1 ] && b=$SSE1_BIN
            if ! "$b" "$MODEL" --tokens-file "$TOK" --kernel sse2 \
                    --dump-all-logits "$TMPD/q8r.$side.$fx.bin" \
                    >"$TMPD/q8r.$side.$fx.log" 2>&1; then
                echo "FAIL: q8r / $side / $fx did not run"
                tail -n 3 "$TMPD/q8r.$side.$fx.log" | sed 's/^/      /'
                FAIL=1; ok=0; break
            fi
        done
        [ "$ok" = 1 ] || continue
        SZ=$(wc -c <"$TMPD/q8r.sse2.$fx.bin")
        if cmp -s "$TMPD/q8r.sse1.$fx.bin" "$TMPD/q8r.sse2.$fx.bin"; then
            echo "PASS: $fx / q8r - SSE1 == SSE2 rounding tier ($SZ bytes)"
            PASS=$((PASS + 1))
        else
            ND=$(cmp -l "$TMPD/q8r.sse1.$fx.bin" "$TMPD/q8r.sse2.$fx.bin" \
                 2>/dev/null | wc -l)
            echo "FAIL: $fx / q8r - SSE1 and SSE2 differ in $ND of $SZ bytes"
            FAIL=1
        fi
    done
    # THE ROW ABOVE IS THE SHAPE THAT LIES IF THE SWITCH IS INERT, so
    # this is not left to the reader: mutating p127 from 127 to 126 in
    # lz_q8round32_sse (the SSE1 body, src/ops_mmx_sse.c) changes the
    # LZ_Q8R_FORCE_SSE=1 binary's output and leaves the ordinary build's
    # byte-identical - measured on kmr20/zh, both directions. So the
    # SSE1 tier executes, and "identical" above means the two roundings
    # agree rather than that only one of them ran.
    :
fi

# ---------------------------------------------------------------------
# CROSS-COMPILER, PER TIER. The loop above compares tiers inside ONE
# binary; fixed_tier_xcompiler_gate.sh compares gcc against Watcom but
# never pins --kernel (the string does not appear in it), so it has only
# ever compared whatever `auto` resolved to on each side - sse2 on both.
# That leaves the pair the requirement is actually about untested: the MMX
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
