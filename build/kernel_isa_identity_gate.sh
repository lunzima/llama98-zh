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
# CROSS-COMPILER (gcc vs Watcom) IS REPORTED, NOT ASSERTED, and the
# reason is measured. gcc's x86-64 build does scalar float in SSE, which
# rounds every instruction to 24 bits. Watcom emits x87, which stores
# each STATEMENT back to 24 bits but keeps a whole EXPRESSION on the
# 80-bit stack - classic double rounding. The two therefore disagree by
# 1 ulp on ordinary float arithmetic, before any kernel is involved:
#
#   volatile float chain of + - * /   SSE 41412422  x87 41412423
#   lz_sinf(-55.0f)                   SSE 3f7feff6  x87 3f7feff5
#   lz_logf(6.27778e-26f)             SSE c2681eec  x87 c2681eeb
#
# That is NOT a Watcom defect and not a kernel that drifted: the SAME
# source compiled by gcc with -mfpmath=387 reproduces the Watcom bytes
# exactly (3f7feff5 / c2681eeb), so the split is the float unit, not the
# compiler. lz_fpu_float_begin's PC=24 does not close it either - the
# control word sets the mantissa width, it does not force a store inside
# an expression.
#
# One ulp at the input, amplified through 24 transformer layers, ends up
# changing almost every logit byte (76.0 of 78.9 MB on the zh fixture),
# so a byte count here says nothing about severity. The cross-compiler
# rows print their measurement and do not set FAIL.
#
# What DOES stay a hard assert: gcc across all three tiers, Watcom across
# ref == mmx. Both hold on every run to date and both catch a kernel that
# drifted from its own scalar reference.
#
# Watcom's sse2 tier is reported rather than asserted, and the reason is
# narrower than "x87 vs SSE". ONE WATCOM BINARY CONTAINS BOTH ROUNDINGS:
# its scalar code is x87, its mmx twin is integer SIMD (inheriting x87
# around it), but its sse2 twin issues real mulps/addps/cvtps2dq
# (src/ops_sse2.c) - SSE rounding inside an x87 binary. Whether that
# reaches the logits depends on how much of the float result the
# subsequent quantisation absorbs, and the measurements disagree with
# each other: one run had watcom sse2 standing apart from watcom ref
# (gcc-sse2-vs-watcom-sse2 74,958,947 bytes against 75,950,881 for the
# other two tiers), a later clean rebuild had watcom ref == watcom sse2
# exactly and all three cross-compiler rows at the same 75,950,881.
#
# The second is the trustworthy one - the first ran against a tree whose
# X86_64_CFLAGS still carried an -mfpmath=387 experiment - but the row is
# left informational rather than promoted to an assert on one clean run.
# Every gcc-vs-Watcom pairing is informational for the reason above.
#
# BUILDING gcc WITH -mfpmath=387 TO CLOSE THE GAP DOES NOT WORK, and the
# attempt is on record because it looks like the obvious fix. It does
# make the gcc scalar path match Watcom - but the SIMD tiers keep
# emitting real SSE2 instructions, so the SAME binary then disagrees
# with ITSELF: this gate's `epi` row went from PASS to
#
#   FAIL: zh / epi - ref and sse2 differ in 57306064 of 78905344 bytes
#
# because ref had become x87 while sse2 stayed SSE. That trades one
# cross-compiler difference for a new intra-binary one, which is the
# worse of the two - the hard asserts above are the ones that catch a
# diverged kernel.
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
else
    echo "note: could not copy the tree, epi rows will SKIP"
fi

# sse and ssem are SSE1 as a real tier. It used to be reachable only by
# CPUID behind g_kernel's back, which is why --kernel mmx measured as
# "recurrence impl sse-intrin" before LZ_KERNEL_SSE existed - the mmx row
# above was comparing SSE1 against SSE2 and calling it MMX. ssem is the
# comparison the deleted P2 and EXP_Q20 control builds were standing in
# for; ssef reaches the SSE1 float matmul row, which the fixed default
# does not.
#
# `sse` ALSO CARRIES THE q8-ROUNDING TIER, which used to be a row of its
# own fed by an LZ_Q8R_FORCE_SSE=1 control build - a whole second tree
# and second `make` to reach a body that --kernel sse now selects at run
# time. lz_q8r_tier reads g_kernel, so sse against sse2 moves it from 1
# to 2 in one binary.
#
# THAT ROW IS THE SHAPE THAT LIES IF THE TIER IS INERT, so it is not
# left to the reader: mutating p127 from 127 to 126 in lz_q8round32_sse
# (the SSE1 body, src/ops_mmx_sse.c) reddens `sse` and `ssem` and leaves
# dflt, rot and epi green - measured on kmr20, both fixtures. So the
# SSE1 tier executes, and "identical" means the two roundings agree
# rather than that only one of them ran.
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
            # HARD ASSERT, grouped by FLOAT UNIT rather than by compiler,
            # because the grouping that matters is which tiers round the
            # same way - and one Watcom binary contains both.
            #
            #   gcc:  scalar is SSE (x86-64 default) and the SIMD tiers
            #         are SSE2, so all three tiers are one rounding.
            #   wat:  scalar is x87; the mmx twin is INTEGER SIMD, so it
            #         inherits x87's rounding for everything around it -
            #         but the sse2 twin issues real mulps/addps/cvtps2dq
            #         (src/ops_sse2.c), i.e. SSE rounding inside an x87
            #         binary. wat ref == wat sse2 is therefore structurally
            #         unreachable and says nothing about the kernel.
            #
            # So gcc gets all three tiers; Watcom gets ref == mmx, which
            # is the row that still catches a #pragma aux integer kernel
            # drifting from its own scalar reference.
            if cmp -s "$TMPD/x.gcc.ref.epi.bin" "$TMPD/x.gcc.mmx.epi.bin" \
               && cmp -s "$TMPD/x.gcc.ref.epi.bin" "$TMPD/x.gcc.sse2.epi.bin"; then
                echo "PASS: $fx / xcc - gcc ref == gcc mmx == gcc sse2" \
                     "(one binary, one rounding, three tiers)"
                PASS=$((PASS + 1))
            else
                echo "FAIL: $fx / xcc - gcc's own tiers are not one value"
                FAIL=1
            fi
            if cmp -s "$TMPD/x.wat.ref.epi.bin" "$TMPD/x.wat.mmx.epi.bin"; then
                echo "PASS: $fx / xcc - watcom ref == watcom mmx" \
                     "(x87 scalar vs its integer-SIMD twin)"
                PASS=$((PASS + 1))
            else
                echo "FAIL: $fx / xcc - watcom's x87 tiers are not one value"
                FAIL=1
            fi
            if cmp -s "$TMPD/x.wat.ref.epi.bin" "$TMPD/x.wat.sse2.epi.bin"; then
                echo "INFO: $fx / xcc - watcom ref == watcom sse2" \
                     "(x87 and SSE agreed here)"
            else
                ND=$(cmp -l "$TMPD/x.wat.ref.epi.bin" \
                            "$TMPD/x.wat.sse2.epi.bin" | wc -l)
                NT=$(wc -c < "$TMPD/x.wat.ref.epi.bin")
                echo "INFO: $fx / xcc - watcom ref vs watcom sse2 differ in" \
                     "$ND of $NT bytes (known: x87 scalar vs the sse2 twin's" \
                     "own mulps/addps rounding, same cause as the rows below)"
            fi
            # REPORTED, NOT ASSERTED: gcc (SSE, 24-bit every instruction)
            # against Watcom (x87, 80-bit inside an expression). The
            # header measures why these cannot match and why the byte
            # count is not a severity signal. Printed so a change in the
            # relationship is visible in the log; it does not set FAIL.
            for k in ref mmx sse2; do
                if cmp -s "$TMPD/x.gcc.$k.epi.bin" "$TMPD/x.wat.$k.epi.bin"; then
                    echo "INFO: $fx / xcc - gcc $k == watcom $k" \
                         "(SSE and x87 agreed here)"
                else
                    ND=$(cmp -l "$TMPD/x.gcc.$k.epi.bin" \
                                "$TMPD/x.wat.$k.epi.bin" | wc -l)
                    NT=$(wc -c < "$TMPD/x.gcc.$k.epi.bin")
                    echo "INFO: $fx / xcc - gcc $k vs watcom $k differ in" \
                         "$ND of $NT bytes (known: SSE 24-bit vs x87 80-bit," \
                         "see this script's header)"
                fi
            done
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
