#!/bin/bash
# Control-switch matrix gate (build/ci.sh stage 5).
#
# The durable fix for this project's recurring failure mode - "a
# configuration nobody happens to build": a switch's non-default value
# compiles to a hard-coded sibling that silently diverges, and no gate
# sees it because no build passes that flag. This gate compiles the
# engine sources with EVERY control switch's non-default value on BOTH
# toolchains (gcc x86-64 and Open Watcom at the -4r floor, mirroring
# make watcom-cli), once per commit, so a contradiction becomes a
# compile error.
#
# Per entry, on both toolchains:
#   1. src/ops.c and src/forward.c compile with the -D flags. A
#      non-default value that cannot compile is either a genuinely
#      contradictory combination or a latent bug - both are FAILs,
#      reported not papered over. For LZ_SSE2_GROUP the guarded code
#      lives in src/ops_sse2.c (the format group kernels), so that file
#      is compiled as well, with the SIMD-capable tiers the Makefile
#      gives it.
#   2. The flags actually took effect - a misspelled or overridden -D
#      compiles clean and proves nothing:
#        gcc:    -E -dM on src/ops.c; each macro must appear with its
#                non-default value under the flags AND not appear with
#                that value without them (the header's own #ifndef
#                default, or nothing for the define-style switches).
#        Watcom: no macro-dump option, so each entry also compiles a
#                small generated probe (#ifndef/#if != #error). A
#                #error is a nonzero wcc386 exit, so the probe cannot
#                pass unless every flag reached the preprocessor. Float
#                values are skipped in the generated probe (#if cannot
#                compare floats) and covered by the committed probe
#                below instead, which is also the independent record of
#                the expected values that catches table typos the
#                differ-from-default check cannot.
#
# WHAT THIS GATE DOES NOT DO, and a PASS must not be read as: it never
# LINKS and never RUNS. Every check above is per translation unit, so a
# switch whose non-default value leaves a definition missing behind its
# own guard compiles clean here and fails only when someone links - and
# for a control switch, nobody does. Measured: of the switches below,
# exactly two (LZ_GDN_FIXED=0, LZ_NORM_FIXED=0) have a standing gate that
# builds a binary and runs it, in recur_write_float_arm_gate.sh. The rest
# were run once, by hand, on the day they were written.
#
# build/int_pipeline_arm_gate.sh closes that for the int-pipeline family
# (links and runs each =0 arm) and FAILS if a switch is added to SWITCHES
# below without being added there - so the two tables cannot drift apart
# silently the way this gate and fixed_tier_xcompiler_gate did over
# --norm-write int.
#
# The switches and their non-default values (reconciled against the
# #ifndef sites in src/ops.h / src/ops.c / src/ops_kernel_shared.h):
#
#   LZ_GDN_FIXED        0        #ifndef default 1
#   LZ_GDN_STATE_2PLANE 0        #ifndef default 1
#   LZ_ATTN_FIXED       0        #ifndef default 3
#   LZ_CONV_FIXED       0        #ifndef default 1
#   LZ_NORM_FIXED       0        #ifndef default 1 - the fixed norms tier,
#      an axis of its own. It used to ride on LZ_GDN_FIXED, which made
#      the recurrence's control arm float the norms too and left the
#      norms with no control arm at all.
#   LZ_SSE2_GROUP       0        #ifndef default 1 (guarded code in ops_sse2.c)
#   LZ_Q8_MIN_SCALE     0.0f     #ifndef default 1.0e-30f
#   LZ_EXACT_MATH       1        defined-style (#if defined)
#   LZ_CPUID_NOCACHE    1        defined-style (#ifdef)
#   LZ_KV_2PLANE        1        #ifndef default 0 - two-plane KV cache
#   LZ_GDN_STATE_F32    1        #ifndef default 0 - debug f32 state probe
#   LZ_BATCH_MAX        1        #ifndef default 8 - batching off, PAIR (below)
#   LZ_SPEC_K           1        #ifndef default 0 - runtime spec_k default
#   LZ_PF_DIST          0        #ifndef default 4 - prefetch lookahead sweep
# LZ_P2_FORCE_MMX and LZ_Q8R_FORCE_SSE are gone: the pure-MMX p2 tier
# and the SSE1 q8 rounding tier are `--kernel mmx` and `--kernel sse`
# now that LZ_KERNEL_SSE exists, so the switches changed nothing and a
# switch that changes nothing passes this gate for the wrong reason.
#
# Plus one PAIR that is not a switch's single off value:
#
#   LZ_GDN_FIXED=2 + LZ_GDN_STATE_2PLANE=0   the fixed pass-2 tier. The
#      lone LZ_GDN_FIXED=2 is a genuine contradiction: src/ops_gdn.c's
#      own #error rejects it against the default LZ_GDN_STATE_2PLANE=1
#      ("would be four kernel passes"). The pair is the legal build,
#      verified to compile on both toolchains.
#
# Plus a CROSS list (below): multi-switch combinations where an #error
# only fires when two non-default values interact. Same check_one, same
# macro-assertion - the point is the combination compiles, not just each
# member alone.
#
# Two entries are contradictory AS GIVEN and the gate builds the legal
# pair instead of the lone flag:
#   LZ_BATCH_MAX=1       ops.h's guard "#if LZ_BATCH_DEFAULT >
#                        LZ_BATCH_MAX #error" fires unless the control
#                        is the documented pair -DLZ_BATCH_MAX=1
#                        -DLZ_BATCH_DEFAULT=1.
#   LZ_GDN_FIXED=2       the #error above; needs 2PLANE=0.
#
# Deliberately NOT here: include guards, LZ_MMX_TU/LZ_SSE2_TU (build
# plumbing, always on), LZ_SPEC_K_MAX (a constant, not a switch),
# LZ_BATCH_DEFAULT (a runtime default, not a code-path switch),
# LZ_GDN_LO_SCALE and LZ_CONV_ES (calibration constants; GDN_LO_SCALE's
# own comment declares the alternative a settled question), LZ_TAP (a
# function-like debug macro, not a numeric switch), and
# LZ_WINVER/LZ_WINNT (the Windows API floor, exercised at 0x0351 by the
# watcom-gui/dll stages). LZ_ATTN_FIXED's intermediate bitmask values
# (1 = scoring-only, 2 = wsum-only) are legal but are refinements of the
# off value 0 this gate builds.
#
# Exit codes: 0 PASS, 1 FAIL, 2 ERROR (missing toolchain).
set -u
cd "$(dirname "$0")/.."

if ! command -v gcc >/dev/null 2>&1; then
    echo "switch_matrix_gate: ERROR - gcc not found"
    exit 2
fi
if [ -x /c/WATCOM/binnt64/wcc386.exe ]; then
    WCC=/c/WATCOM/binnt64/wcc386.exe
    WATCOM_ROOT=/c/WATCOM
elif [ -x /mnt/c/WATCOM/binl64/wcc386 ]; then
    WCC=/mnt/c/WATCOM/binl64/wcc386
    WATCOM_ROOT=/mnt/c/WATCOM
else
    echo "switch_matrix_gate: ERROR - wcc386 not found (checked binnt64/binl64)"
    exit 2
fi

# Same flags as make native / make watcom-cli's ordinary floor (see the
# Makefile's CFLAGS / CLI_FLAGS). -we is Watcom's "warnings are errors";
# a non-default value that warns under -we fails the real build too, so
# it fails here.
GCC_BASE="-O2 -std=c99 -Wall -Wextra -Isrc -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
WCC_BASE="-za99 -otexan -zp4 -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -4r -bt=nt -we -DWIN32_LEAN_AND_MEAN -DWINVER=0x0351 -D_WIN32_WINNT=0x0351"
# ops_sse2.c's own tier (the Makefile's HIGH_CPU_FLAGS: -6r -D__MMX__=1).
WCC_MMX_BASE="-za99 -otexan -zp4 -D__MMX__=1 -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -6r -bt=nt -we -DWIN32_LEAN_AND_MEAN -DWINVER=0x0351 -D_WIN32_WINNT=0x0351"
WCC_INC="-I$WATCOM_ROOT/h -I$WATCOM_ROOT/h/nt -Isrc"

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

PASS=0
FAIL=0

check_one() {
    # $1 label  $2 -D flags  $3 proves (NAME=VALUE list)  $4 also ops_sse2.c
    local label="$1" defines="$2" proves="$3" sse2="$4"
    local ok=1 src p pname pval line int_only probe_src
    local safe
    safe=$(printf '%s' "$label" | tr ' =+' '___')

    for src in ops.c forward.c; do
        if ! gcc $GCC_BASE $defines -c -o /dev/null "src/$src" 2>"$TMPD/gcc-$safe-${src%.c}.log"; then
            echo "FAIL: $label  gcc src/$src (compile)"
            grep -E 'error:' "$TMPD/gcc-$safe-${src%.c}.log" | head -3 | sed 's/^/      /'
            ok=0
        fi
    done

    if [ "$sse2" = "1" ]; then
        if ! gcc $GCC_BASE $defines -c -o /dev/null src/ops_sse2.c 2>"$TMPD/gcc-$safe-sse2.log"; then
            echo "FAIL: $label  gcc src/ops_sse2.c (compile)"
            grep -E 'error:' "$TMPD/gcc-$safe-sse2.log" | head -3 | sed 's/^/      /'
            ok=0
        fi
        if ! "$WCC" $WCC_MMX_BASE $WCC_INC $defines -fo="$TMPD/wc-$safe-sse2.obj" src/ops_sse2.c >"$TMPD/wcc-$safe-sse2.log" 2>&1; then
            echo "FAIL: $label  watcom src/ops_sse2.c (compile)"
            grep -E 'Error' "$TMPD/wcc-$safe-sse2.log" | head -3 | sed 's/^/      /'
            ok=0
        fi
    fi

    # gcc -dM proof: every asserted macro appears with its non-default
    # value under the flags, and not with that value without them.
    #
    # The "with" side is checked PER TRANSLATION UNIT, not unioned. A -D
    # on the command line shows up in the -dM of every unit that does not
    # touch the macro, so a union lets the one file that DOES override the
    # switch hide behind the other one - which is this gate's whole
    # subject (LZ_NORM_FIXED collided with an ops_sched.c enum constant of
    # the same name, and the ops.c proof stayed green). Measured, not
    # argued: an unconditional `#define LZ_MLAT_I16 1` in forward.c leaves
    # ops.c reading 0 from the command line and the union green, while the
    # per-unit form reports forward.c and only forward.c. A switch whose
    # #ifndef site is in either file still satisfies both units, because
    # the other one simply inherits the command line.
    #
    # The "without" side stays a UNION: a default that already equals the
    # asserted value in EITHER unit means the switch proves nothing there.
    dm_with_ops=$(gcc $GCC_BASE $defines -E -dM src/ops.c 2>/dev/null)
    dm_with_fwd=$(gcc $GCC_BASE $defines -E -dM src/forward.c 2>/dev/null)
    dm_with="$dm_with_ops
$dm_with_fwd"
    dm_without=$(gcc $GCC_BASE -E -dM src/ops.c 2>/dev/null
                 gcc $GCC_BASE -E -dM src/forward.c 2>/dev/null)
    for p in $proves; do
        pname="${p%%=*}"
        pval="${p#*=}"
        line="#define $pname $pval"
        if ! echo "$dm_with_ops" | grep -Fxq "$line"; then
            echo "FAIL: $label  -dM src/ops.c: '$line' not in preprocessed output with $defines"
            ok=0
        fi
        if ! echo "$dm_with_fwd" | grep -Fxq "$line"; then
            echo "FAIL: $label  -dM src/forward.c: '$line' not in preprocessed output with $defines"
            ok=0
        fi
        if echo "$dm_without" | grep -Fxq "$line"; then
            echo "FAIL: $label  -dM: '$line' present WITHOUT the -D flags (overridden by a header define)"
            ok=0
        fi
    done

    for src in ops.c forward.c; do
        if ! "$WCC" $WCC_BASE $WCC_INC $defines -fo="$TMPD/wc-$safe-${src%.c}.obj" "src/$src" >"$TMPD/wcc-$safe-${src%.c}.log" 2>&1; then
            echo "FAIL: $label  watcom src/$src (compile)"
            grep -E 'Error' "$TMPD/wcc-$safe-${src%.c}.log" | head -3 | sed 's/^/      /'
            ok=0
        fi
    done

    # Watcom proof for THIS entry, generated from the same proves list
    # (the committed probe below is the canonical-set cross-check).
    # Float values are skipped - #if cannot compare floats; the one
    # float macro (LZ_Q8_MIN_SCALE) is asserted by the committed probe.
    int_only=1
    for p in $proves; do
        case "${p#*=}" in *[.fF]*) int_only=0 ;; esac
    done
    if [ "$int_only" -eq 1 ]; then
        probe_src="$TMPD/probe-$safe.c"
        : > "$probe_src"
        for p in $proves; do
            pname="${p%%=*}"
            pval="${p#*=}"
            printf '#ifndef %s\n#error "%s not defined (expected %s)"\n#endif\n#if %s != %s\n#error "%s != %s"\n#endif\n' \
                "$pname" "$pname" "$pval" "$pname" "$pval" "$pname" "$pval" >> "$probe_src"
        done
        if ! "$WCC" $WCC_BASE $WCC_INC $defines -fo="$TMPD/probe-$safe.obj" "$probe_src" >"$TMPD/probe-$safe.log" 2>&1; then
            echo "FAIL: $label  watcom probe - a -D flag did not take effect:"
            grep -E 'Error' "$TMPD/probe-$safe.log" | head -3 | sed 's/^/      /'
            ok=0
        fi
    fi

    if [ "$ok" -eq 1 ]; then
        proof=""
        for p in $proves; do
            pname="${p%%=*}"; pval="${p#*=}"
            proof="$proof [$(echo "$dm_with" | grep -Fx "#define $pname $pval")]"
        done
        echo "PASS: $label$proof"
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
}

# name=value pairs, whitespace-separated. The two pair-entries are
# handled in the loop below.
# LZ_BVEC_I16 / LZ_MOEGATE_I16 / LZ_KLAT_I16 / LZ_KGATE_I16 are the four
# q16_0 int16 exits and LZ_MLAT_I16 the MoE latent's (forward.c). Each
# defaults to 1 and each =0 build is the control arm its measurement was
# taken against, which is exactly the kind of build nobody runs again
# unless a gate does.
# LZ_MLAT_REUSE is the expert loop's quantize-reuse, an axis of its own:
# it landed in the same commit as LZ_MLAT_I16 and is worth more, and
# with the two on one switch neither half's contribution could be
# rebuilt. All four arms now close: both off reproduces the pre-change
# census exactly (1,127,814 on kmr20/zh) and byte-identical logits on
# both fixtures; the exit alone is +512, the reuse alone -4,224.
# LZ_VPROJ_I16 is the V projection's int16 exit (milestone 7). One
# switch, not two: unlike the MoE latent's pair the exit and the
# requantize it unblocks cannot be measured apart - with the exit taken
# there is no float row left for the float quantize to read. Its =0 arm
# reproduces the pre-change census exactly (1,121,798 on kmr20/zh) and
# byte-identical logits on both fixtures and both gate models.
# LZ_Q8I16_INT is lz_quantize_q8_i16's integer element loop (ops.c), the
# consumer side of those exits: with it off the loop converts every
# element back to float, which is what made the exits above move a
# conversion instead of removing one. Its =0 arm is the control the
# saving was measured against - byte-identical logits and a census that
# reproduces 1,121,350 exactly on kmr20/zh.
# LZ_Q8INT_INT is lz_quantize_q8_int's integer element loop (ops.c), the
# same rewrite one width up: a per-group normalizing pre-shift feeds the
# shared reciprocal and the correction test is evaluated mod 2^32, so an
# amax of 4.6e8 rounds exactly without int64. Its =0 arm restores the
# float loop, reproduces the pre-change census exactly (740,646 on
# kmr20/zh) and gives byte-identical logits on both fixtures and both
# gate models.
# LZ_SWIGLU_I16 is the integer SwiGLU (forward.c), one switch for both
# MoE sites: they are the same mechanism reached by the same helper, so
# neither half's contribution could be rebuilt separately anyway. Its =0
# arm reproduces the pre-change census exactly (1,108,134 on kmr20/zh)
# and byte-identical logits on both fixtures and both gate models.
# LZ_CONV_SIG_I is the fixed conv epilogue's integer-coordinate sigmoid
# (ops.h, so ops.c and forward.c see the same value - forward.c builds
# the exponent table the kernel reads). Its =0 arm reproduces the
# pre-change census exactly (888,102 on kmr20/zh) and byte-identical
# logits on both fixtures and both gate models.
# LZ_CONVO_I16 is the fixed conv's int16 OUTPUT for q and k
# (forward.c), a different axis from LZ_CONV_SIG_I's input-side entry -
# either can be on without the other. Its =0 arm reproduces the
# pre-change census exactly (858,286 on kmr20/zh) and byte-identical
# logits against a tree with the change removed altogether, which is a
# stronger control than the =0 build alone: the wiring is guarded in
# five places and a missed guard would show up here and nowhere else.
# LZ_CONVO_ES is its exponent and is NOT a control arm - a value, chosen
# from .prof/convqk_range.c's clamp column, whose wrongness shows up in
# lz_conv_o_clamped rather than in a build.
# LZ_FDIV_RECIP is the integer-reciprocal divide (ops_quant.c). Its
# non-default value is 1, not 0 - the default is off on a host with a
# hardware divider and on where -DLZ_NO_FPU is passed, so the matrix's
# job here is to prove the ON arm builds and runs everywhere rather than
# the off one. BIT-IDENTICAL both ways, which is unusual in this table
# and is why the switch carries a positive control (lz_debug_fdiv_recip,
# 713,568 taken on kmr20/zh against 0 in the off arm): a logits cmp
# reads rc=0 whichever arm compiled, so it cannot tell them apart.
# LZ_Q8R_BITS is q8_round done on the bit pattern instead of through the
# magic add (ops_quant.h). Same disposition as LZ_FDIV_RECIP and for the
# same reason - default off where an FPU makes the add three cycles, on
# under -DLZ_NO_FPU - so the non-default arm to prove here is 1. Also
# bit-identical both ways, so it carries the same kind of positive
# control: lz_debug_q8r_bits, 59,316,632 taken and 0 declined on
# kmr20/zh against 0 taken in the off arm.
# LZ_GATE_SIG_I and LZ_GATE_QW are the gated norm's two savings
# (ops_norm.c), TWO switches and not one because they are two different
# removals in one loop and neither number could be rebuilt from a single
# flag: the sigmoid's integer coordinate is 18,432 hardware converts a
# token and value-changing, the weight table is 6,144 magic ones and
# bit-identical. Each =0 arm and the both-off arm were all measured;
# both off reproduces the pre-change census exactly (888,102 on kmr20/zh)
# and byte-identical logits on both fixtures and both gate models.
# LZ_KGATE_EXP_I is the KDA decay gate's two transcendentals in the
# integer domain (forward.c), on top of LZ_KGATE_I16's int16 exit. ONE
# switch and not two, unlike the gated norm's pair: sigmoid_q15_ti's
# output is lz_exp_t's input, so with only one of them integer the
# coordinate has to be rebuilt in float between them and neither half's
# saving could be measured alone. Its =0 arm reproduces the pre-change
# census exactly (808,230 on kmr20/zh, 828,487 en) and byte-identical
# logits on both fixtures and both gate models.
# LZ_SOFTPLUS_POLY is lz_softplus's table+poly float body (ops.h, so
# ops.c and ops_quant.c see the same value). Its =0 arm is the libm
# double pair it replaced, and it is a control arm in the strongest
# sense available: softplus does not execute at all on either shipping
# KDA checkpoint, so =0 reproduces the pre-change census exactly on both
# (720,654 zh on kmr20) AND byte-identical logits on both fixtures and
# both gate models, while on the GDN checkpoint _armgate2, where it does
# execute, =0 reproduces 981,938 exactly.
# The _ES exponents are calibration constants, not switches, and are
# excluded for the same reason LZ_CONV_ES is.
SWITCHES="LZ_GDN_FIXED=0 LZ_GDN_STATE_2PLANE=0 LZ_ATTN_FIXED=0 LZ_CONV_FIXED=0 LZ_NORM_FIXED=0 LZ_EPI_FIXED=0 LZ_SSE2_GROUP=0 LZ_Q8_MIN_SCALE=0.0f LZ_EXACT_MATH=1 LZ_CPUID_NOCACHE=1 LZ_NO_FPU=1 LZ_KV_2PLANE=1 LZ_GDN_STATE_F32=1 LZ_BATCH_MAX=1 LZ_SPEC_K=1 LZ_PF_DIST=0 LZ_BVEC_I16=0 LZ_MOEGATE_I16=0 LZ_KLAT_I16=0 LZ_KGATE_I16=0 LZ_KGATE_EXP_I=0 LZ_MLAT_I16=0 LZ_MLAT_REUSE=0 LZ_VPROJ_I16=0 LZ_Q8I16_INT=0 LZ_Q8INT_INT=0 LZ_SWIGLU_I16=0 LZ_CONV_SIG_I=0 LZ_CONVO_I16=0 LZ_GATE_SIG_I=0 LZ_GATE_QW=0 LZ_SOFTPLUS_POLY=0 LZ_FDIV_RECIP=1 LZ_Q8R_BITS=1"

for entry in $SWITCHES; do
    name="${entry%%=*}"
    case "$name" in
        LZ_BATCH_MAX)
            check_one "LZ_BATCH_MAX=1(+LZ_BATCH_DEFAULT=1)" \
                "-DLZ_BATCH_MAX=1 -DLZ_BATCH_DEFAULT=1" \
                "LZ_BATCH_MAX=1 LZ_BATCH_DEFAULT=1" 0
            ;;
        LZ_SSE2_GROUP)
            check_one "$entry" "-D$entry" "$entry" 1
            ;;
        *)
            check_one "$entry" "-D$entry" "$entry" 0
            ;;
    esac
done

# The fixed pass-2 tier: legal only as the pair with the two-plane state
# off (src/ops_gdn.c's #error rejects it against the default), a distinct
# code path from LZ_GDN_FIXED=0.
check_one "LZ_GDN_FIXED=2(+LZ_GDN_STATE_2PLANE=0)" \
    "-DLZ_GDN_FIXED=2 -DLZ_GDN_STATE_2PLANE=0" \
    "LZ_GDN_FIXED=2 LZ_GDN_STATE_2PLANE=0" 0

# Cross combinations: multiple non-default switches that must hold
# TOGETHER. A single-switch build proves each alone; a combination can
# still contradict (an #error that only fires when two flags interact),
# so each cross is compiled and macro-asserted as a unit - the same
# check_one the singles use, which also proves every flag took effect.
# A cross naming LZ_SSE2_GROUP compiles ops_sse2.c as well (its guarded
# code lives there), matching the single-switch handling.
# LZ_GDN_FIXED=0,LZ_NORM_FIXED=0 is the both-tiers-off build. It only
# became a combination worth naming when the norms got their own switch:
# before that, -DLZ_GDN_FIXED=0 alone floated both, and this pair could
# not be expressed.
CROSS="LZ_SSE2_GROUP=0,LZ_GDN_FIXED=2,LZ_GDN_STATE_2PLANE=0
LZ_NO_FPU=1,LZ_GDN_FIXED=0
LZ_GDN_STATE_2PLANE=0,LZ_GDN_FIXED=0
LZ_GDN_FIXED=0,LZ_NORM_FIXED=0"

while IFS= read -r cross; do
    [ -z "$cross" ] && continue
    flags=""
    proves=""
    label=""
    IFS=, read -r -a xs <<< "$cross"
    for x in "${xs[@]}"; do
        flags="$flags -D$x"
        proves="$proves $x"
        [ -n "$label" ] && label="$label+"
        label="$label$x"
    done
    case "$cross" in
        *LZ_SSE2_GROUP=0*)
            check_one "$label" "$flags" "$proves" 1
            ;;
        *)
            check_one "$label" "$flags" "$proves" 0
            ;;
    esac
done <<< "$CROSS"

# --- all-at-once probe: the independent record of the expected values ---
# Compiled with every switch's primary non-default value. #error-asserts
# the canonical set, including the one float macro (#if cannot compare
# floats, so it is asserted via a C constant expression instead). This is
# what catches a table typo the differ-from-default check cannot see.
ALLDEFS=""
for entry in $SWITCHES; do
    ALLDEFS="$ALLDEFS -D$entry"
done
ALLDEFS="$ALLDEFS -DLZ_BATCH_DEFAULT=1"

if gcc -std=c99 $ALLDEFS -fsyntax-only build/switch_matrix_probe.c >"$TMPD/probe-gcc.log" 2>&1; then
    echo "PASS: probe (gcc - all flags took effect)"
else
    echo "FAIL: probe (gcc) - a -D flag did not take effect:"
    head -5 "$TMPD/probe-gcc.log" | sed 's/^/      /'
    FAIL=$((FAIL + 1))
fi
if "$WCC" $WCC_BASE $WCC_INC $ALLDEFS -fo="$TMPD/probe.obj" build/switch_matrix_probe.c >"$TMPD/probe.log" 2>&1; then
    echo "PASS: probe (watcom - all flags took effect)"
else
    echo "FAIL: probe (watcom) - a -D flag did not take effect:"
    grep -E 'Error' "$TMPD/probe.log" | head -5 | sed 's/^/      /'
    FAIL=$((FAIL + 1))
fi

# --- all-at-once LINK: the class check_one structurally cannot see ---
# check_one compiles ops.c and forward.c to /dev/null. A switch that
# compiles out a definition still referenced from a TU it does not
# compile - cli_main.c's --debug-counters block reads a dozen of these
# counters unconditionally - produces an object that is fine on its own
# and a binary that does not exist. That is not hypothetical:
# -DLZ_CONV_FIXED=0 would not link (lz_conv_o_clamped,
# lz_debug_convo_i16) and every gate above it was green, because none of
# them ever reached the linker.
#
# ONE link, not one per switch: 35 full builds would dominate this
# gate's runtime, and the probe above has already asserted that all 35
# non-default values are simultaneously legal, so one binary with every
# switch flipped reaches every conditional definition at once. What it
# cannot do is ATTRIBUTE - it names the missing symbol, not the switch
# that removed it - so on a failure, bisect with the four tier switches
# first (build/recur_write_float_arm_gate.sh links those five builds
# individually and is the finer instrument).
# LZ_GDN_STATE_2PLANE=0 is held OUT of the link set, and the reason is
# a finding rather than a convenience: LZ_KV_2PLANE=1 calls
# lz_gdn_quantize_2p, which LZ_GDN_STATE_2PLANE owns, so the two
# non-default values contradict each other. That is now an #error in
# ops.h (it was a link failure with no declaration, which is the same
# fact arriving later and less legibly). Both remain covered
# individually by check_one above; what this one link cannot do is hold
# both at once, and saying so beats a set quietly trimmed to whatever
# happened to build. Coverage here: 34 of the 35 switches.
# Counted, not written down. The message below used to say "34 of 35"
# in a literal, and removing one switch from the list above left the
# literal describing a list it no longer matched.
LINKDEFS=""
NSW=0
NLINK=0
for entry in $SWITCHES; do
    NSW=$((NSW + 1))
    [ "$entry" = "LZ_GDN_STATE_2PLANE=0" ] && continue
    NLINK=$((NLINK + 1))
    LINKDEFS="$LINKDEFS -D$entry"
done

LINKSRC=$(make -n llama98 2>/dev/null | grep -m1 -- '-o llama98 ')
if [ -z "$LINKSRC" ]; then
    echo "SKIP: all-at-once link - could not read the link command from make"
else
    # The link line names build/native/*.o, which make builds by a
    # separate rule. If they are absent - a fresh clone, a `make clean`,
    # or another session's build removing them mid-run in this shared
    # tree - ld reports "cannot find" and this stage would call it "a
    # switch removed a definition someone still references". That is an
    # environment fact wearing a code defect's words, and it was seen:
    # 41 of 42 passing with the one failure entirely missing objects.
    MISSING=""
    for o in $(printf '%s\n' $LINKSRC | grep -oE 'build/native/[a-z0-9_]+\.o'); do
        [ -f "$o" ] || MISSING="$MISSING $o"
    done
    if [ -n "$MISSING" ]; then
        echo "SKIP: all-at-once link - missing prerequisite object(s):$MISSING"
        echo "      Run 'make llama98' first. This is not a switch failure."
        LINKSRC=""
    fi
fi
if [ -n "$LINKSRC" ]; then
    LINKCMD=${LINKSRC/ -o llama98 / -o $TMPD/allsw.exe }
    if eval "$LINKCMD $LINKDEFS -DLZ_BATCH_DEFAULT=1" \
            >"$TMPD/allsw.log" 2>&1; then
        echo "PASS: all-at-once link ($NLINK of $NSW switches non-default, one binary)"
    else
        echo "FAIL: all-at-once link - a switch removed a definition someone still references:"
        # The SYMBOL, which is the whole content of this failure. ld
        # writes it with a long toolchain path in front and the phrase
        # broken across the line, so a grep for 'undefined reference'
        # catches the wrapped continuation and not the name - this
        # printed only "collect2.exe: error" and left the reader with a
        # failure that named nothing.
        SYMS=$(grep -oE "undefined reference to \`[^']+'" "$TMPD/allsw.log" \
               | sed "s/.*\`//;s/'//" | sort -u)
        [ -n "$SYMS" ] || SYMS=$(grep -oE "undefined symbol[: ]+[A-Za-z_][A-Za-z0-9_]*" \
                                 "$TMPD/allsw.log" | awk '{print $NF}' | sort -u)
        if [ -n "$SYMS" ]; then
            printf '      %s\n' $SYMS
        else
            echo "      (no symbol in the log - the link failed for some other"
            echo "       reason; the tail follows)"
            tail -6 "$TMPD/allsw.log" | sed 's/^/      /'
        fi
        FAIL=$((FAIL + 1))
    fi
fi

# ---------------------------------------------------------------------
# NO COMPILE-TIME TIER ESCAPES. Five LZ_*_FORCE_* macros existed, one
# per site that had an SSE1 or MMX body reachable from CPUID but from no
# knob: Q8R, I32FACC, P2, EXP_Q20 and the NORM_SS pair. Each was a
# separate control BUILD standing in for a missing selector value, and
# LZ_KERNEL_SSE is that value - `--kernel sse` and `--kernel mmx` now
# select those bodies at run time, out of one binary, which is what
# kernel_isa_identity_gate's sse and ssem flagsets compare.
#
# All five are deleted, so the count is zero and a ratchet can hold it
# there. The next one will look reasonable when it is written - it did
# all five times - and the objection is not that a control build is
# useless but that a second way to select a body can disagree with the
# first. A site that genuinely has no runtime tier keeps its A/B switch
# and is named in SWITCHES above (LZ_SSE2_GROUP, LZ_BATCH_MAX); this
# check is specifically about the FORCE spelling, which is what the five
# had in common.
FORCERE='^[[:space:]]*#[[:space:]]*(if|ifdef|ifndef).*LZ_[A-Z0-9_]*_FORCE_'
FORCERE="$FORCERE"'|defined[[:space:]]*\([[:space:]]*LZ_[A-Z0-9_]*_FORCE_'
FORCEUSE=$(grep -rlE "$FORCERE" src 2>/dev/null | sort -u)
if [ -n "$FORCEUSE" ]; then
    echo "FAIL: a compile-time tier escape is back:"
    for f in $FORCEUSE; do echo "    $f"; done
    echo "  LZ_*_FORCE_* in a preprocessor conditional means a body is"
    echo "  selectable two ways. Give it a --kernel tier instead, the way"
    echo "  LZ_KERNEL_SSE replaced the five that used to be here."
    FAIL=$((FAIL + 1))
else
    PASS=$((PASS + 1))
    echo "PASS: no LZ_*_FORCE_* tier escape in src/ (all five retired)"
fi

# Positive control over the WHOLE check, not just the pattern: a planted
# tree with one escape in it, run through the same grep. A zero from a
# search that cannot find anything is how a ratchet turns into a comment,
# and testing the regex alone would leave the recursive search untested.
#
# Two shapes planted, because the five used both, and one line of prose,
# because src/ops.h names all five in a sentence and a check that
# reddens on its own documentation is a check someone deletes.
FCTL="$TMPD/forcectl"
mkdir -p "$FCTL"
printf '#ifdef LZ_Q8R_FORCE_SSE\nint a;\n#endif\n'                  >"$FCTL/a.c"
printf '#if defined(LZ_NORM_SS_FORCE_MMX) && 1\nint b;\n#endif\n'   >"$FCTL/b.c"
printf '   LZ_NORM_SS_FORCE_MMX. Five separate control BUILDS.\n'   >"$FCTL/c.c"
FCTL_HIT=$(grep -rlE "$FORCERE" "$FCTL" 2>/dev/null | sed 's|.*/||' | sort | tr '\n' ' ')
if [ "$FCTL_HIT" != "a.c b.c " ]; then
    echo "FAIL: the FORCE search does not find what it is for."
    echo "  planted a.c (#ifdef) and b.c (defined()) plus c.c (prose only);"
    echo "  the search returned: ${FCTL_HIT:-nothing}"
    FAIL=$((FAIL + 1))
fi

echo
echo "switch_matrix_gate: pass=$PASS fail=$FAIL"
if [ "$FAIL" -eq 0 ]; then
    echo "switch_matrix_gate: PASS"
    exit 0
fi
echo "switch_matrix_gate: FAIL"
exit 1
