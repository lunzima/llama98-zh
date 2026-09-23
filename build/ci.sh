#!/bin/bash
# Minimal CI. No .github/ workflow exists yet - this is the script such
# a workflow would call, and it also runs by hand:
#
#   build/ci.sh
#
# THIS SCRIPT AND `make check` DO NOT CONTAIN EACH OTHER, so a green in
# one says nothing about the other. comment_hygiene_gate.sh is a
# prerequisite of `check` and appears below only in a comment - a grep
# for its name finds prose, not an invocation. The gates wired into
# `check` are, with codesize_gate.sh the exception, not stages here.
# "The tree is green" therefore has no single instrument; name the one
# that ran.
#
# Stages, each independently PASS / FAIL / SKIP, none silent:
#   1. make native             (gcc)
#   2. make watcom-cli          (Open Watcom, the target compiler)
#   3. llama98 <dir> --check on a curated set of KunMoe checkpoints
#   4. build/arm/parity_gate.sh (x86 <-> ARMv5TE bit-identity)
#   5. build/switch_matrix_gate.sh - every control switch's non-default
#      value compiled on gcc + Watcom, so a "configuration nobody
#      happens to build" fails here instead of silently diverging
#   6. build/fixed_tier_xcompiler_gate.sh - gcc/Watcom bit-identity of
#      BOTH arms of --fixed (all, the shipping engine; off, the float
#      control) over hundreds of tokens of real text. The one knob
#      replaced 12 per-tier values. This is the
#      case the norms tier's original 8/16-token verification could not see
#      (lz_quantize_q8_int's double-rounding hazard, see that script's
#      own header) and that a later audit's 153/211-token re-check of
#      the other ten tiers could not see either, for the same reason
#   7. build/recur_write_float_arm_gate.sh - runtime-off vs
#      compile-time-off of the fixed recurrence, the one comparison a
#      cross-compiler gate cannot make: two toolchains agreeing about a
#      fixed path they both take says nothing about whether the float
#      arm is reachable at all
#   8. make watcom-gui / watcom-dll / watcom-dos (the remaining Watcom
#      targets; DOS is left out of `make check` by design)
#   9. build/x87/build_engine.sh under WSL (32-bit x87 gcc, the fourth
#      toolchain shape)
#  10. build/mips/endian_gate.sh under WSL - the same source cross-built
#      big-endian AND little-endian and compared byte for byte. Every
#      other stage above runs on a little-endian machine, so a loader
#      that assumes little-endian byte order passes all nine of them;
#      this is the only one that can tell portable from untested
#  11. build/perf_knob_gate.sh - the PERFORMANCE-only knobs, where the
#      assertion is invariance: a knob that only trades speed for memory
#      must not move a single logit bit
#  12. build/kv_rot_gate.sh - --kv-rot, the one output-CHANGING knob,
#      so the assertion is a quality number rather than invariance
#  13. build/int_pipeline_arm_gate.sh - stage 5 compiles every control
#      switch's non-default value but never links or runs one, so a
#      definition missing behind a guard is invisible there. This links
#      and runs each int-pipeline =0 arm, and fails if stage 5 gains a
#      switch it does not list
#  14. build/kernel_matrix_gate.sh - every operator has a cell per tier
#      per build class and an empty one carries a reason. Built on all
#      three classes and cross-checked against each binary's own macros,
#      so neither an unregistered kernel nor a registration outliving
#      its kernel survives
#  15. build/kernel_isa_identity_gate.sh - every operator's twins are
#      bit-identical, end to end: two TIERS in one binary, the
#      comparison stage 6's two-COMPILER gate cannot make
#  16. build/subn_gate.sh - --subn-scale and --hadamard, the two knobs
#      whose gate could not run until a SubLN checkpoint that exports
#      was found
#  17. build/gcc_flag_parity_gate.sh - the gui and the CLI compile the
#      same engine sources with the same flags. The only stage here that
#      compares a BUILD RECIPE rather than an output, because this defect
#      does not change the output
#  18. build/caption_symmetry_gate.sh - the hand-drawn caption bar and
#      its buttons are symmetric. The only stage that checks the shipping
#      product's APPEARANCE, and it does so as arithmetic because the
#      two rectangles are computed from metrics rather than from each
#      other
#  19. build/changelog_comment_gate.sh - the phrasing half of the
#      comment rule: no file gains a changelog phrase. The STRICT and
#      LOOSE lists in build/changelog_comment_gate.sh are the authority
#      for which phrases those are - quoting them here would be a second
#      copy of the list, and it would drift. A ratchet, because some of
#      those sentences are load-bearing and the repair is to restate
#      them, not delete them
#  20. build/str_table_order_gate.sh - src/err.c and
#      gui/localized_strings.c list their codes in their enums' order.
#      Those tables are positional now, because Visual C++ 4.0 cannot
#      compile the designated form, so order is the only thing binding a
#      string to a code
#  21. build/align_gate.sh - UBSan over a real forward pass: does any
#      C dereference use a pointer not aligned for its type. An
#      unaligned load is legal on x86 and returns a ROTATED word on
#      ARMv5TE, so no x86 run can answer this and every run here is x86
#  22. build/codesize_gate.sh - the gui fits on a 1.44 MB floppy, and
#      the CLI's .text stays under a ceiling. Two criteria because the
#      floppy is at 68% and would not warn about anything
#  23. build/float_knob_gate.sh - one float axis. --fixed off|all is
#      accepted, the seven per-tier flags it replaced are rejected, the
#      two performance knobs are kept, and ops_sched.c has exactly one
#      float-selection global. Nothing else counts the knobs
#  24. build/xcheck_gate.sh - the ten per-OPERATOR cross-checks in
#      .prof/. Every other stage compares whole-engine output; these
#      compare one operator's implementations against each other, which
#      is the only way to reach the arms no local checkpoint exercises
#  25. build/gcc_warning_gate.sh - zero gcc warnings, asserted rather
#      than printed. The Watcom side has always been counted per file;
#      this is the same assertion for the toolchain whose warnings only
#      ever went to the console
#  26. build/tier_reach_gate.sh - every quantised matmul REACHES its
#      tiered kernel. A tensor whose (dtype, gs) fails a dispatch guard
#      takes matmul_scalar_ref and is still correct, so every numeric
#      stage above passes on it. Only an execution count separates
#      "agrees" from "agrees and was reached"
#  27. build/aux_parm_arity_gate.sh - every #pragma aux declaration's
#      parameter list matches the function it decorates. Watcom takes a
#      wrong arity silently and passes the wrong register
#  28. build/aux_equiv_gate.sh - no kernel body's VALUE changes without
#      the commit message declaring it. A reorder that crosses a
#      dependence looks like a cleanup in review
#  29. build/prefetch_parity_gate.sh - the --prefetch arms compute the
#      same thing. Two full Watcom engine builds, and those arms are
#      compiled by no other toolchain
#  30. build/norm_sse_parity_gate.sh - the norm #pragma aux twins against
#      their C, on the toolchain that has them; a gcc build contains
#      neither side
#  31. build/i32f_acc_gate.sh - the two int32->float accumulate tiers
#      agree, one engine build per arm
#
# THIS LIST DOCUMENTED 24 OF 31, and the omission is the kind this tree
# treats as a defect rather than untidiness: the list is what a reader
# uses to know what CI covers, and one that stops at 26 (having also
# skipped 25) under-reports by seven stages while reading as complete.
# The stage numbering in the code was right the whole time; only the map
# was short.
#
# BEFORE READING A RED: this tree is shared. A run started while another
# session has a file half-edited reports that session's state as this
# run's failure, and it does not report it once - one source file that
# does not compile takes every stage that builds the engine with it. A
# full run has been seen at 45 PASS / 53 FAIL where all 53 traced to two
# undeclared identifiers in src/forward.c, mid-edit elsewhere.
#
# So: check `git status --short src/` first. If a file is modified and
# not yours, the red is a snapshot, not a result - re-run when it
# settles. The stage list below cannot tell those apart, and neither can
# the exit code.
#
# Judging discipline (this repo has paid for getting this wrong before):
#   - Never grep bare 'error' - "0 errors" contains the substring.
#     Watcom's own per-file line ("N lines, included M, W warnings,
#     E errors") is parsed for the numbers, not matched as text.
#   - A build's exit code is trusted only after also confirming the
#     binary it was supposed to produce got a NEWER mtime than a
#     sentinel dropped just before the build ran. `make native` failing
#     with the target already up to date, or Windows refusing to
#     overwrite a locked/running .exe, both exit nonzero or leave the
#     old binary in place - either way this must not fall through to
#     testing yesterday's binary as if it were today's. -B forces make
#     to attempt a real rebuild every run so "up to date, skip" can
#     never masquerade as "verified".
#   - SKIP is not PASS. Every skip is printed with its reason and
#     counted separately in the summary; the exit code only looks at
#     FAIL.
set -u
cd "$(dirname "$0")/.." || exit 1

LZ_ENV=$(LZ_TREE_LOCK_NAME="$(basename "$0")" LZ_TREE_LOCK_WHAT="a full CI round" bash build/tree_lock.sh acquire --print-env) || exit 2
eval "$LZ_ENV"
trap 'bash build/tree_lock.sh release' EXIT
ROOT=$(pwd)

# The shared-tree warning above, made visible at the top of the run
# rather than left for whoever reads the summary. Printed, not enforced:
# a dirty tree is the normal way to test a change, and refusing would
# make this unusable. What it prevents is reading someone else's
# half-finished edit as this run's result.
DIRTY=$(git status --short -- src gui common 2>/dev/null | grep '^ M' | wc -l)
if [ "${DIRTY:-0}" -gt 0 ]; then
    echo "NOTE: $DIRTY modified source file(s) in the working tree:"
    git status --short -- src gui common 2>/dev/null | grep '^ M' \
        | sed 's/^/        /'
    echo "      A red below may be one of these mid-edit rather than a"
    echo "      defect. One file that does not compile fails every stage"
    echo "      that builds the engine."
    echo
fi

PASS=0
FAIL=0
SKIP=0
SUMMARY=()

note() { SUMMARY+=("$1"); }

freshness_sentinel() {
    # Echoes a path that is older than "right now" by construction, for
    # the `$exe -nt $sentinel` staleness check below. Git Bash's mtime
    # resolution is coarse enough that back-to-back mktemp calls can tie,
    # so this sleeps past it.
    local s
    s=$(mktemp)
    sleep 1
    echo "$s"
}

# ---------------------------------------------------------------------
# Stage 1: make native (gcc)
# ---------------------------------------------------------------------
echo "==== [1/31]make native ===="
NATIVE_EXE="$ROOT/llama98.exe"
[ -e "$NATIVE_EXE" ] || NATIVE_EXE="$ROOT/llama98"
LOG=build/ci-native.log
SENTINEL=$(freshness_sentinel)
native_ok=1
if ! make -B native >"$LOG" 2>&1; then
    echo "FAIL: make native (see $LOG)"
    tail -n 25 "$LOG"
    note "FAIL  make native"
    FAIL=$((FAIL + 1))
    native_ok=0
elif grep -q 'Permission denied' "$LOG"; then
    echo "FAIL: make native reported Permission denied - $NATIVE_EXE is likely locked by a running process; this is NOT a successful build"
    note "FAIL  make native (exe locked - Permission denied)"
    FAIL=$((FAIL + 1))
    native_ok=0
elif [ ! -e "$NATIVE_EXE" ]; then
    echo "FAIL: make native exited 0 but $NATIVE_EXE does not exist"
    note "FAIL  make native (binary missing)"
    FAIL=$((FAIL + 1))
    native_ok=0
elif [ ! "$NATIVE_EXE" -nt "$SENTINEL" ]; then
    echo "FAIL: make native exited 0 but $NATIVE_EXE was NOT rebuilt (stale artifact)"
    note "FAIL  make native (stale artifact)"
    FAIL=$((FAIL + 1))
    native_ok=0
else
    SZ=$(wc -c < "$NATIVE_EXE")
    echo "PASS: make native ($NATIVE_EXE, $SZ bytes)"
    note "PASS  make native"
    PASS=$((PASS + 1))
fi
rm -f "$SENTINEL"

# ---------------------------------------------------------------------
# Stage 2: make watcom-cli (Open Watcom, the target compiler)
# ---------------------------------------------------------------------
echo
echo "==== [2/31]make watcom-cli ===="
WATCOM_EXE="$ROOT/llama98-w.exe"
WATCOM_OBJDIR="$ROOT/build/watcom/cli"
LOG=build/ci-watcom-cli.log
SENTINEL=$(freshness_sentinel)
if ! make -B watcom-cli >"$LOG" 2>&1; then
    echo "FAIL: make watcom-cli (see $LOG)"
    tail -n 25 "$LOG"
    note "FAIL  make watcom-cli"
    FAIL=$((FAIL + 1))
elif [ ! -e "$WATCOM_EXE" ]; then
    echo "FAIL: make watcom-cli exited 0 but $WATCOM_EXE does not exist"
    note "FAIL  make watcom-cli (binary missing)"
    FAIL=$((FAIL + 1))
elif [ ! "$WATCOM_EXE" -nt "$SENTINEL" ]; then
    echo "FAIL: make watcom-cli exited 0 but $WATCOM_EXE was NOT rebuilt (stale artifact)"
    note "FAIL  make watcom-cli (stale artifact)"
    FAIL=$((FAIL + 1))
else
    # Watcom's own per-file line looks like:
    #   src/foo.c: 123 lines, included 4567, 0 warnings, 0 errors
    # Sum the numeric error/warning counts - never text-match 'error'.
    NFILES=$(grep -cE ': [0-9]+ lines, included [0-9]+, [0-9]+ warnings, [0-9]+ errors$' "$LOG")
    NERR=$(grep -oE '[0-9]+ errors$' "$LOG" | awk '{s+=$1} END{print s+0}')
    NWARN=$(grep -oE '[0-9]+ warnings,' "$LOG" | awk '{s+=$1} END{print s+0}')
    NOBJS=$(ls -1 "$WATCOM_OBJDIR"/*.obj 2>/dev/null | wc -l)
    if [ "$NERR" -gt 0 ]; then
        echo "FAIL: make watcom-cli compiled with $NERR error(s) across $NFILES file summaries"
        grep -E ', [0-9]+ warnings, [1-9][0-9]* errors$' "$LOG"
        note "FAIL  make watcom-cli ($NERR compile errors)"
        FAIL=$((FAIL + 1))
    elif [ "$NFILES" -eq 0 ]; then
        echo "FAIL: make watcom-cli - no per-file compile summary lines found in the log at all (wcc386 likely didn't run - check $LOG)"
        note "FAIL  make watcom-cli (no compiler output found)"
        FAIL=$((FAIL + 1))
    elif [ "$NFILES" -ne "$NOBJS" ]; then
        echo "FAIL: make watcom-cli - $NFILES compile summaries in the log but $NOBJS .obj files in $WATCOM_OBJDIR (one file likely crashed the compiler silently)"
        note "FAIL  make watcom-cli (summary/obj count mismatch: $NFILES vs $NOBJS)"
        FAIL=$((FAIL + 1))
    else
        SZ=$(wc -c < "$WATCOM_EXE")
        echo "PASS: make watcom-cli ($NFILES files, $NWARN warnings, 0 errors, $WATCOM_EXE: $SZ bytes)"
        note "PASS  make watcom-cli ($NFILES files, $NWARN warnings)"
        PASS=$((PASS + 1))
    fi
fi
rm -f "$SENTINEL"

# ---------------------------------------------------------------------
# Stage 3: llama98 <dir> --check on a curated set of KunMoe checkpoints
# ---------------------------------------------------------------------
echo
echo "==== [3/31]model --check sweep ===="
MODELS_ROOT=${MODELS_ROOT:-/e/LLM/models}
# Curated, not a blind sweep of every directory under MODELS_ROOT:
# a usable sweep is a handful of representative checkpoints, and
# E:/LLM/models/ also
# holds Qwen-family reference / abliteration / rotation-study checkpoints
# (heretic, Qwopus3.5-*, rot-a05, qwen3.5-0.8b-mtp, router-bert, the raw
# Qwen3.5-0.8B dumps) that were never meant to load through this engine -
# measured here: they fail --check with "missing tensor mtp.layers.0...",
# "embed_tokens.weight not found" or "config missing layer_types", which
# is a fact about those checkpoints' purpose, not an engine regression.
# Gating CI on them would make it permanently red for reasons unrelated
# to this engine. Override with CHECK_MODELS="a b c" to test others.
#
# kunkun98-pilot-20260806 WAS IN THIS LIST AND IS NOT ON DISK, so the
# sweep has been running two checkpoints while reading as three. Same
# name was ARM_MODEL below, where the cost was larger - see there.
# Replaced with kmr20, which every cross-compiler gate already uses.
# build/fixture_exists_gate.sh now refuses this class.
CHECK_MODELS=${CHECK_MODELS:-"s1v3 kmr20 kunmoe-v2"}

if [ "$native_ok" -ne 1 ]; then
    echo "SKIP: model --check sweep (native build failed in stage 1, no binary to run)"
    note "SKIP  model --check sweep (no native binary)"
    SKIP=$((SKIP + 1))
else
    for m in $CHECK_MODELS; do
        dir="$MODELS_ROOT/$m"
        if [ ! -d "$dir" ]; then
            echo "SKIP: model '$m' not found at $dir"
            note "SKIP  model --check: $m (directory not found)"
            SKIP=$((SKIP + 1))
            continue
        fi
        OUT=$("$NATIVE_EXE" "$dir" --check 2>&1)
        RC=$?
        if [ "$RC" -eq 0 ] && echo "$OUT" | grep -q 'Structure OK'; then
            echo "PASS: $m --check"
            note "PASS  model --check: $m"
            PASS=$((PASS + 1))
        else
            echo "FAIL: $m --check (exit $RC)"
            echo "$OUT" | tail -5
            note "FAIL  model --check: $m"
            FAIL=$((FAIL + 1))
        fi
    done
fi

# ---------------------------------------------------------------------
# Stage 4: x86 <-> ARMv5TE parity gate
# ---------------------------------------------------------------------
echo
echo "==== [4/31]ARM parity gate ===="
# THIS NAMED A CHECKPOINT THAT IS NOT ON DISK, and the consequence was
# the worst one available: parity_gate.sh SKIPs on a missing model dir,
# ci.sh counts SKIP as not-a-failure, and so the x86-against-ARM
# bit-identity contract - the hardest promise this engine makes - was
# not being checked by CI at all, while the run reported no failure.
# Nothing distinguishes "the ARM toolchain is absent on this machine",
# which is a legitimate skip, from "the fixture name rotted", which is
# not. _armgate2 is on disk and is named for this gate; verified PASS on
# all four rows (x86 vs ARM identical over 20,840,448 bytes, and arm-c /
# arm-asm / arm-asm-fixed each identical to ref).
#
# THOSE FOUR ROWS WERE VERIFIED ON THE PREVIOUS FIXTURE, _armgate2, and
# that model is gone - it is not restorable, see
# docs/dsv41-optimization-menu.md section 6. _armgate3 replaces it: a
# DIFFERENT model (8 layers, 8 heads / 2 KV, dense FFN 3584, from
# kunmoe-v2-dense) chosen because it preserves what _armgate2 uniquely
# covered - forward_ssm (GDN) and the dense FFN - which kmr20 does not,
# being KDA and MoE. It exports with all four formats and its 160-token
# logits dump is the same 20,840,448 bytes, because the vocabulary is the
# same 32768.
#
# THE ROWS HAVE NOT BEEN RE-RUN ON IT. The byte counts above still
# describe the dump size and nothing else; whether arm-c, arm-asm and
# arm-asm-fixed are still identical to ref ON THIS MODEL is unknown until
# someone runs it with the ARM toolchain. Say so when reporting it, or
# run it first.
ARM_MODEL="$MODELS_ROOT/_armgate3"
LOG=build/ci-arm.log
if [ "$native_ok" -ne 1 ]; then
    echo "SKIP: ARM parity gate (native build failed in stage 1, nothing to compare ARM against)"
    note "SKIP  ARM parity gate (no native binary)"
    SKIP=$((SKIP + 1))
else
    X86_BIN="$NATIVE_EXE" build/arm/parity_gate.sh "$ARM_MODEL" >"$LOG" 2>&1
    ARM_RC=$?
    cat "$LOG"
    case $ARM_RC in
        0)
            echo "PASS: ARM parity gate"
            note "PASS  ARM parity gate"
            PASS=$((PASS + 1))
            ;;
        2)
            REASON=$(grep -m1 '^SKIP:' "$LOG")
            echo "SKIP: ARM parity gate ($REASON)"
            note "SKIP  ARM parity gate ($REASON)"
            SKIP=$((SKIP + 1))
            ;;
        *)
            echo "FAIL: ARM parity gate"
            note "FAIL  ARM parity gate"
            FAIL=$((FAIL + 1))
            ;;
    esac
fi

# ---------------------------------------------------------------------
# Stage 5: control-switch matrix (every non-default value, gcc + Watcom)
# ---------------------------------------------------------------------
echo
echo "==== [5/31]control-switch matrix ===="
LOG=build/ci-switch-matrix.log
build/switch_matrix_gate.sh >"$LOG" 2>&1
MATRIX_RC=$?
cat "$LOG"
case $MATRIX_RC in
    0)
        echo "PASS: control-switch matrix"
        note "PASS  control-switch matrix"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 '^switch_matrix_gate: ERROR' "$LOG")
        echo "SKIP: control-switch matrix ($REASON)"
        note "SKIP  control-switch matrix ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: control-switch matrix"
        note "FAIL  control-switch matrix"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 6: every fixed-point write tier's gcc<->Watcom bit-identity
# ---------------------------------------------------------------------
echo
echo "==== [6/31]fixed-tier cross-compiler gate ===="
LOG=build/ci-fixed-tier-xcompiler.log
build/fixed_tier_xcompiler_gate.sh >"$LOG" 2>&1
NIX_RC=$?
cat "$LOG"
case $NIX_RC in
    0)
        echo "PASS: fixed-tier cross-compiler gate"
        note "PASS  fixed-tier cross-compiler gate"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 '^fixed_tier_xcompiler_gate: SKIP' "$LOG")
        echo "SKIP: fixed-tier cross-compiler gate ($REASON)"
        note "SKIP  fixed-tier cross-compiler gate ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: fixed-tier cross-compiler gate"
        note "FAIL  fixed-tier cross-compiler gate"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 7: --fixed off has to be a COMPLETE float arm, not a float arm
# for the axes that happen to have a runtime selector. Stage 6 compares
# two compilers against each other, so an axis both of them run in fixed
# point is invisible there; this one compares runtime-off against
# compile-time-off in the same compiler, where it is not.
# ---------------------------------------------------------------------
echo
echo "==== [7/31]recurrence float-arm gate ===="
LOG=build/ci-recur-write-float-arm.log
build/recur_write_float_arm_gate.sh >"$LOG" 2>&1
NIX_RC=$?
cat "$LOG"
case $NIX_RC in
    0)
        echo "PASS: recurrence float-arm gate"
        note "PASS  recurrence float-arm gate"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 '^recur_write_float_arm_gate: SKIP' "$LOG")
        echo "SKIP: recurrence float-arm gate ($REASON)"
        note "SKIP  recurrence float-arm gate ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: recurrence float-arm gate"
        note "FAIL  recurrence float-arm gate"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 8: the remaining Watcom targets. `make check` builds gui and dll
# but deliberately leaves out dos (it needs the extender stub and is a
# separate delivery decision); none of the three is in ci.sh's stage 2.
# ---------------------------------------------------------------------
echo
echo "==== [8/31]watcom gui/dll/dos targets ===="
for t in watcom-gui watcom-dll watcom-dos; do
    case $t in
        watcom-gui) EXE="$ROOT/kunkun98-w.exe" ;;
        watcom-dll) EXE="$ROOT/llama98.dll" ;;
        watcom-dos) EXE="$ROOT/llama98-dos.exe" ;;
    esac
    LOG="build/ci-$t.log"
    SENTINEL=$(freshness_sentinel)
    if ! make -B "$t" >"$LOG" 2>&1; then
        echo "FAIL: make $t (see $LOG)"
        tail -n 25 "$LOG"
        note "FAIL  make $t"
        FAIL=$((FAIL + 1))
    elif [ ! -e "$EXE" ]; then
        echo "FAIL: make $t exited 0 but $EXE does not exist"
        note "FAIL  make $t (binary missing)"
        FAIL=$((FAIL + 1))
    elif [ ! "$EXE" -nt "$SENTINEL" ]; then
        echo "FAIL: make $t exited 0 but $EXE was NOT rebuilt (stale artifact)"
        note "FAIL  make $t (stale artifact)"
        FAIL=$((FAIL + 1))
    else
        SZ=$(wc -c < "$EXE")
        echo "PASS: make $t ($EXE, $SZ bytes)"
        note "PASS  make $t"
        PASS=$((PASS + 1))
    fi
    rm -f "$SENTINEL"
done

# ---------------------------------------------------------------------
# Stage 9: 32-bit x87 gcc build under WSL (Windows gcc has no 32-bit
# libs). The fourth toolchain shape - a Socket-7-class target with x87
# as its only float unit. SKIP when WSL is not usable, never PASS.
# ---------------------------------------------------------------------
echo
echo "==== [9/31]32-bit x87 gcc build (WSL) ===="
LOG=build/ci-x87.log
# WHICH compiler, not whether `gcc` happens to be the one. Ubuntu ships
# gcc-N-multilib per version, so a box can have working 32-bit support
# under gcc-14 while plain `gcc` is gcc-13 and fails at the LINK with
# "skipping incompatible libgcc.a". That is what this stage did: it
# skipped for want of multilib on a machine that had it, and the skip
# hid a build/x87 source list that had gone nine files stale.
#
# Probes with a LINK, not -c: the missing piece is 32-bit libgcc and a
# compile-only check passes without it.
X87CC=""
if command -v wsl >/dev/null 2>&1; then
    for cand in gcc gcc-14 gcc-13 gcc-12; do
        if wsl bash -lc "printf 'int main(void){return 0;}\n' >/tmp/_m32t.c && $cand -m32 /tmp/_m32t.c -o /tmp/_m32t" >/dev/null 2>&1; then
            X87CC=$cand
            break
        fi
    done
fi
if ! command -v wsl >/dev/null 2>&1; then
    echo "SKIP: 32-bit x87 build (wsl not installed)"
    note "SKIP  32-bit x87 build (wsl not installed)"
    SKIP=$((SKIP + 1))
elif [ -z "$X87CC" ]; then
    echo "SKIP: 32-bit x87 build (WSL present but no gcc that can link -m32;"
    echo "      tried gcc gcc-14 gcc-13 gcc-12 - apt install gcc-N-multilib)"
    note "SKIP  32-bit x87 build (no -m32 multilib in WSL)"
    SKIP=$((SKIP + 1))
else
    # $ROOT is Git Bash shaped (/e/llama98-zh); the WSL cd needs the
    # /mnt/e/... shape. The build script finds the repo root itself.
    WSL_ROOT=$ROOT
    case "$ROOT" in
        /[a-z]/*) WSL_ROOT="/mnt/$(echo "$ROOT" | cut -c2)/$(echo "$ROOT" | cut -c4-)" ;;
    esac
    if wsl bash -lc "cd '$WSL_ROOT' && CC=$X87CC bash build/x87/build_engine.sh" >"$LOG" 2>&1; then
        if grep -q '^linked:' "$LOG"; then
            echo "PASS: 32-bit x87 build (WSL, CC=$X87CC)"
            note "PASS  32-bit x87 build (WSL, CC=$X87CC)"
            PASS=$((PASS + 1))
        else
            echo "FAIL: 32-bit x87 build exited 0 but no 'linked:' line in $LOG"
            tail -n 10 "$LOG"
            note "FAIL  32-bit x87 build (no linked binary)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL: 32-bit x87 build (WSL) - see $LOG"
        tail -n 15 "$LOG"
        note "FAIL  32-bit x87 build (WSL)"
        FAIL=$((FAIL + 1))
    fi
fi

# ---------------------------------------------------------------------
# Stage 10: big-endian vs little-endian bit-identity (WSL + MIPS cross).
# The ONLY stage that can see a byte-order bug: every other target here
# is little-endian, so a loader that assumes little-endian byte order is
# bit-identical to a correct one in all nine stages above. SKIP when the
# MIPS cross toolchain is absent, never PASS.
# ---------------------------------------------------------------------
echo
echo "==== [10/31]big-endian parity gate (WSL) ===="
LOG=build/ci-endian.log
if ! command -v wsl >/dev/null 2>&1; then
    echo "SKIP: big-endian parity gate (wsl not installed)"
    note "SKIP  big-endian parity gate (wsl not installed)"
    SKIP=$((SKIP + 1))
else
    WSL_ROOT=$ROOT
    case "$ROOT" in
        /[a-z]/*) WSL_ROOT="/mnt/$(echo "$ROOT" | cut -c2)/$(echo "$ROOT" | cut -c4-)" ;;
    esac
    wsl bash -lc "cd '$WSL_ROOT' && bash build/mips/endian_gate.sh" >"$LOG" 2>&1
    END_RC=$?
    case $END_RC in
        0)
            echo "PASS: big-endian parity gate"
            note "PASS  big-endian parity gate"
            PASS=$((PASS + 1))
            ;;
        2)
            REASON=$(grep -m1 'endian-gate: SKIP' "$LOG")
            echo "SKIP: big-endian parity gate ($REASON)"
            echo "  Byte-order portability goes UNVERIFIED - no other stage"
            echo "  in this script runs on a big-endian machine."
            note "SKIP  big-endian parity gate ($REASON)"
            SKIP=$((SKIP + 1))
            ;;
        *)
            echo "FAIL: big-endian parity gate - see $LOG"
            tail -n 15 "$LOG"
            note "FAIL  big-endian parity gate"
            FAIL=$((FAIL + 1))
            ;;
    esac
fi

# ---------------------------------------------------------------------
# Stage 11: the PERFORMANCE-only knobs. Every other knob in this
# engine is a tier, and a tier that moves the numbers is doing its job -
# which is why the tier gates cannot say anything about --pair and
# --prefetch, whose entire contract is that they move NOTHING. That made
# them the two easiest knobs to break silently.
#
# --ced belongs to the same class and is the most dangerous of the three,
# because unlike --pair and --prefetch it cannot move the output even
# when it is broken: a knob wired to nothing also produces identical
# numbers. Its gate carries a skip counter and two mutants for that
# reason; see build/ced_gate.sh's header.
# ---------------------------------------------------------------------
echo
echo "==== [11/31]performance-knob invariance ===="
LOG=build/ci-perf-knob.log
build/perf_knob_gate.sh "$ARM_MODEL" >"$LOG" 2>&1
PK_RC=$?
cat "$LOG"
case $PK_RC in
    0)
        echo "PASS: performance-knob invariance"
        note "PASS  performance-knob invariance"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'perf_knob_gate: SKIP' "$LOG")
        echo "SKIP: performance-knob invariance ($REASON)"
        note "SKIP  performance-knob invariance ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: performance-knob invariance"
        note "FAIL  performance-knob invariance"
        FAIL=$((FAIL + 1))
        ;;
esac

# --ced, same class, own script: it needs a git tree (to rebuild the
# previous source for the off-vs-HEAD arm) and three engine builds for
# its mutants, neither of which fits perf_knob_gate.sh.
echo
echo "==== [11b/31]--ced invariance ===="
LOG=build/ci-ced.log
# Pointed at kunkun98 rather than ARM_MODEL, for the same reason the sink
# gate below is (its own note): ARM_MODEL does not carry ced_split, and this
# gate builds no CED config of its own - it reads whatever config.json the
# model dir has. On ARM_MODEL every CED arm is therefore inert, which is
# what "the mutant matched the off arm" beside "off skipped 0, on skipped
# 16, logits identical" was saying: the path was being checked by nothing.
build/ced_gate.sh "$MODELS_ROOT/kunkun98" >"$LOG" 2>&1
CED_RC=$?
cat "$LOG"
case $CED_RC in
    0)
        note "PASS  --ced invariance"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'ced_gate: SKIP' "$LOG")
        note "SKIP  --ced invariance ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  --ced invariance"
        FAIL=$((FAIL + 1))
        ;;
esac

# ced_split/ced_replay_w (E3): NOT an invariance claim like the two
# above - a fired reset is supposed to move the prefill logits, so this
# gate's arms and its mutation both compare logits between two ON
# configurations, never against an unmodified baseline alone. Own
# script for the same reason as --ced: a mutant engine build per run.
#
# Pointed at kunkun98 rather than ARM_MODEL, like the --ced stage above
# and for the same reason: the checkpoint has to carry ced_split for the
# reset to have layers to fire on. ARM_MODEL does not, so the gate read
# "off layers=0 on layers=0" and failed on a fixture that could not
# exercise it - which says nothing about the engine.
echo
echo "==== [11c/31]--ced-replay-w invariance ===="
LOG=build/ci-ced-replay.log
build/ced_replay_gate.sh "$MODELS_ROOT/kunkun98" >"$LOG" 2>&1
CEDR_RC=$?
cat "$LOG"
case $CEDR_RC in
    0)
        note "PASS  --ced-replay-w invariance"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'ced_replay_gate: SKIP' "$LOG")
        note "SKIP  --ced-replay-w invariance ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  --ced-replay-w invariance"
        FAIL=$((FAIL + 1))
        ;;
esac

# S4 MLA low-rank Q/O: own script for the same reason as the two above -
# it builds three synthetic model dirs (safetensors surgery, needs
# python3 numpy+safetensors) rather than reusing ARM_MODEL as-is.
echo
echo "==== [11d/31]MLA low-rank Q/O ===="
LOG=build/ci-mla.log
build/mla_gate.sh "$ARM_MODEL" >"$LOG" 2>&1
MLA_RC=$?
cat "$LOG"
case $MLA_RC in
    0)
        note "PASS  MLA low-rank Q/O"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'mla_gate: SKIP' "$LOG")
        note "SKIP  MLA low-rank Q/O ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  MLA low-rank Q/O"
        FAIL=$((FAIL + 1))
        ;;
esac

# S2 CSA2 engine-side scalar reference: own script for the same reason
# as MLA/ced_replay above - synthetic model dirs, python3 numpy+
# safetensors.
echo
echo "==== [11e/31]CSA2 engine scalar reference ===="
LOG=build/ci-csa2.log
build/csa2_gate.sh "$ARM_MODEL" >"$LOG" 2>&1
CSA2_RC=$?
cat "$LOG"
case $CSA2_RC in
    0)
        note "PASS  CSA2 engine scalar reference"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'csa2_gate: SKIP' "$LOG")
        note "SKIP  CSA2 engine scalar reference ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  CSA2 engine scalar reference"
        FAIL=$((FAIL + 1))
        ;;
esac

# S1 attn_kv_share: own script for the same reason as MLA/CSA2 above -
# it BUILDS its fixture from ARM_MODEL's weights rather than using it
# as-is, since no checkpoint here that this engine can load carries the
# key. Needs python3 safetensors+numpy for the safetensors arms.
echo
echo "==== [11f/31]attn_kv_share ===="
LOG=build/ci-kv-share.log
build/kv_share_gate.sh "$ARM_MODEL" >"$LOG" 2>&1
KVS_RC=$?
cat "$LOG"
case $KVS_RC in
    0)
        note "PASS  attn_kv_share"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'kv_share_gate: SKIP' "$LOG")
        note "SKIP  attn_kv_share ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  attn_kv_share"
        FAIL=$((FAIL + 1))
        ;;
esac

# attn_sink (learned per-head attention sink). Own script for the same
# reason as MLA/CSA2/kv_share above - it BUILDS its fixtures from a
# checkpoint's weights, because the key and the tensor are what is under
# test, and it needs python3 numpy+safetensors. It is pointed at
# kunkun98 rather than ARM_MODEL on purpose: ARM_MODEL does not set the
# key, so the gate would SKIP there and the sink path would be checked by
# nothing. Also builds two mutants (one per half of the wiring).
echo
echo "==== [11g/31]attn_sink (learned) ===="
LOG=build/ci-attn-sink.log
build/attn_sink_learned_gate.sh "$MODELS_ROOT/kunkun98" >"$LOG" 2>&1
ASINK_RC=$?
cat "$LOG"
case $ASINK_RC in
    0)
        note "PASS  attn_sink (learned)"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'attn_sink_learned_gate: SKIP' "$LOG")
        note "SKIP  attn_sink (learned) ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  attn_sink (learned)"
        FAIL=$((FAIL + 1))
        ;;
esac

# gated_residual. Own script for the same reason as the sink gate just
# above - it BUILDS its fixtures from a checkpoint's weights (the key and
# the tensors are what is under test), needs python3 numpy+safetensors,
# and builds five more engines (the pre-change one from git plus four
# mutants). Pointed at kunkun98 and kunkun-ce, the only two checkpoints
# that set the key: ARM_MODEL does not, so the gate would SKIP there and
# the GR path would be checked by nothing.
echo
echo "==== [11h/31]gated residual ===="
LOG=build/ci-gr.log
build/gr_gate.sh "$MODELS_ROOT/kunkun98" "$MODELS_ROOT/kunkun-ce" >"$LOG" 2>&1
GR_RC=$?
cat "$LOG"
case $GR_RC in
    0)
        note "PASS  gated residual"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'gr_gate: SKIP' "$LOG")
        note "SKIP  gated residual ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  gated residual"
        FAIL=$((FAIL + 1))
        ;;
esac

# S2 CSA2's dense-plane accounting: a csa2_on layer owns no per-position
# K/V, so it must be charged for no plane. Own script (it builds HEAD and
# the working tree into scratch dirs) for the same reason as 11e/11f.
echo
echo "==== [11i/31]CSA2 dense KV planes ===="
LOG=build/ci-csa2-dense-kv.log
build/csa2_dense_kv_gate.sh >"$LOG" 2>&1
C2DK_RC=$?
cat "$LOG"
case $C2DK_RC in
    0)
        note "PASS  CSA2 dense KV planes"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        note "SKIP  CSA2 dense KV planes ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  CSA2 dense KV planes"
        FAIL=$((FAIL + 1))
        ;;
esac

# S2 CSA2's census billing: the layer the soft-float plan is most likely
# to be about was the one layer missing from the table that plan is
# ordered by. Own script (it builds the counting engine gcc-direct, which
# is no Makefile target) for the same reason as 11e/11i.
echo
echo "==== [11j/31]CSA2 f32 census billing ===="
LOG=build/ci-csa2-f32count.log
build/csa2_f32count_gate.sh >"$LOG" 2>&1
C2FC_RC=$?
cat "$LOG"
case $C2FC_RC in
    0)
        note "PASS  CSA2 f32 census billing"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        note "SKIP  CSA2 f32 census billing ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  CSA2 f32 census billing"
        FAIL=$((FAIL + 1))
        ;;
esac

# S2 CSA2's ring sizing: four buffers were allocated nf times the prefix
# sum that already spans every slot. Own script (five engine builds, one
# per isolated mutant) for the same reason as the rest of the pair.
echo
echo "==== [11k/31]CSA2 ring sizing ===="
LOG=build/ci-csa2-ring-nf.log
build/csa2_ring_nf_gate.sh >"$LOG" 2>&1
C2RN_RC=$?
cat "$LOG"
case $C2RN_RC in
    0)
        note "PASS  CSA2 ring sizing"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        note "SKIP  CSA2 ring sizing ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        note "FAIL  CSA2 ring sizing"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 12: --kv-rot, the one output-changing knob that had no gate. Not
# invariance (stage 11's assertion) - a QUALITY claim with a number:
# rotating before the KV cache quantizes is supposed to reduce the
# error, so the rotated run must land closer to the --kv f32 reference.
# Two real corpora, whole-sequence logits, and a per-arm check that the
# int16 exits did not move underneath the comparison; the gate's header
# gives the three ways it has been wrong before and the measurements
# that closed each one.
#
# ARM_MODEL is an F32 safetensors checkpoint, so lz_matmul_xq_i16_ok
# refuses every int16 exit on it (v_proj, swiglu, kda gate, moe router:
# all read 0). That is what makes the arms matched here and lets q8 take
# the direction assertion - and it also means stages 4, 11 and 12 do not
# exercise the int pipeline at all. Point the gate at a quantized model
# (build/kv_rot_gate.sh <model>) to cover that side.
# ---------------------------------------------------------------------
echo
echo "==== [12/31]--kv-rot quality gate ===="
LOG=build/ci-kv-rot.log
build/kv_rot_gate.sh "$ARM_MODEL" >"$LOG" 2>&1
KR_RC=$?
cat "$LOG"
case $KR_RC in
    0)
        echo "PASS: --kv-rot quality gate"
        note "PASS  --kv-rot quality gate"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'kv_rot_gate: SKIP' "$LOG")
        echo "SKIP: --kv-rot quality gate ($REASON)"
        note "SKIP  --kv-rot quality gate ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: --kv-rot quality gate"
        note "FAIL  --kv-rot quality gate"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 13: stage 5 proves every control switch compiles at its
# non-default value; it never links or runs one. For the int-pipeline
# family that is the whole risk - each =0 arm keeps a replaced float
# path alive that nothing else executes, so a definition missing behind
# the guard compiles clean and shows up only when someone links, and
# nobody does. Measured before this stage existed: exactly two of the
# matrix's switches had a standing gate that ran their build.
# ---------------------------------------------------------------------
echo
echo "==== [13/31] int-pipeline control arms ===="
LOG=build/ci-int-pipeline-arms.log
build/int_pipeline_arm_gate.sh >"$LOG" 2>&1
IPA_RC=$?
cat "$LOG"
case $IPA_RC in
    0)
        echo "PASS: int-pipeline control arms"
        note "PASS  int-pipeline control arms"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 '^int_pipeline_arm_gate: SKIP' "$LOG")
        echo "SKIP: int-pipeline control arms ($REASON)"
        note "SKIP  int-pipeline control arms ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: int-pipeline control arms"
        note "FAIL  int-pipeline control arms"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 14: the kernel-coverage matrix. Every operator has a cell per
# tier per build class and an EMPTY cell carries a
# reason; nothing enforced that from CI, so the matrix could go stale in
# either direction - a kernel written and never registered, or a
# registration outliving the kernel it names. The gate builds all three
# classes (gcc, Watcom, ARM cross) and cross-checks each binary's own
# --kernel-matrix against what its macros actually define.
# ---------------------------------------------------------------------
echo
echo "==== [14/31]kernel coverage matrix ===="
LOG=build/ci-kernel-matrix.log
build/kernel_matrix_gate.sh >"$LOG" 2>&1
KMX_RC=$?
cat "$LOG"
case $KMX_RC in
    0)
        echo "PASS: kernel coverage matrix"
        note "PASS  kernel coverage matrix"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: kernel coverage matrix ($REASON)"
        note "SKIP  kernel coverage matrix ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: kernel coverage matrix"
        note "FAIL  kernel coverage matrix"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 15: twin bit-identity END TO END. Stage 6 compares two COMPILERS on one
# tier; this compares two TIERS in one binary, which is where a twin
# that drifted from its sibling shows and where a cross-compiler gate is
# blind by construction. Both gates existed on disk before this stage
# and neither was called from here - stage 14's was uncommitted
# entirely, so a clean checkout did not have it.
#
# It now also closes the third side of that square: the same TIER across
# both compilers, which neither gate did (stage 6 never pins --kernel,
# so it has only ever compared auto-vs-auto). That is where the MMX and
# SSE2 kernels' two hand-written implementations meet, and where x87
# lives at all - gcc on x86-64 compiles scalar float to SSE, so the only
# x87 in this tree is the Watcom build's `ref` tier. Adds a Watcom build
# and ten runs; the Watcom `ref` tier alone is 89 s, which is why that
# half runs on one fixture.
# ---------------------------------------------------------------------
echo
echo "==== [15/31]kernel ISA identity ===="
LOG=build/ci-kernel-isa.log
build/kernel_isa_identity_gate.sh >"$LOG" 2>&1
KID_RC=$?
cat "$LOG"
case $KID_RC in
    0)
        echo "PASS: kernel ISA identity"
        note "PASS  kernel ISA identity"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'kernel_isa_identity_gate: SKIP' "$LOG")
        echo "SKIP: kernel ISA identity ($REASON)"
        note "SKIP  kernel ISA identity ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: kernel ISA identity"
        note "FAIL  kernel ISA identity"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 16: the SubLN activation path. --subn-scale and --hadamard were
# the only two knobs in docs/knob-audit.md whose gate could not run, and
# the recorded reason was that no SubLN checkpoint exports. That was
# generalised from the ones that predate the attention projection norms;
# kunkun-ce exports clean.
#
# IT STILL SKIPS AT HEAD, and the honest reason is in the gate's own SKIP
# text: the five assertions were verified on a build carrying in-flight
# loader work, and HEAD's tensor spec list does not yet claim the
# per-projection SubLN norms. Wired here anyway rather than held back -
# a stage that says exactly why it is skipping is worth more than one
# nobody has written, and it turns green by itself when the loader
# lands.
# ---------------------------------------------------------------------
echo
echo "==== [16/31]SubLN activation path ===="
LOG=build/ci-subn.log
build/subn_gate.sh >"$LOG" 2>&1
SUBN_RC=$?
cat "$LOG"
case $SUBN_RC in
    0)
        echo "PASS: SubLN activation path"
        note "PASS  SubLN activation path"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: SubLN activation path ($REASON)"
        note "SKIP  SubLN activation path ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: SubLN activation path"
        note "FAIL  SubLN activation path"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 17: the gui and the CLI compile the engine with the same flags.
#
# Cheap - `make -n`, no compile - and it catches a class nothing else
# here can. Both link the same $(ENG_SRCS), so a flag difference ships
# two different engines under one version number, and the OUTPUT is
# identical either way: -O2 and -O3 gave byte-equal --dump-all-logits on
# two models. Every other stage compares numbers, so every other stage
# would have passed through the whole period the gui was building at -O2
# against the CLI's -O3.
# ---------------------------------------------------------------------
echo
echo "==== [17/31]gui/CLI engine flag parity ===="
LOG=build/ci-flagparity.log
build/gcc_flag_parity_gate.sh >"$LOG" 2>&1
FLAGP_RC=$?
cat "$LOG"
case $FLAGP_RC in
    0)
        echo "PASS: gui/CLI engine flag parity"
        note "PASS  gui/CLI engine flag parity"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: gui/CLI engine flag parity ($REASON)"
        note "SKIP  gui/CLI engine flag parity ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: gui/CLI engine flag parity"
        note "FAIL  gui/CLI engine flag parity"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 18: the caption bar and its buttons are symmetric.
#
# The gui ships a hand-drawn title bar, and nothing above this line
# looks at the product's appearance at all - every other stage compares
# numbers out of the engine. This one is mechanical anyway: bar_rect and
# btn_rect read system metrics independently, so the gaps above and
# below a button are an expression in three of them and either match or
# do not. The defect it was written after - a missing -1 that made the
# bar a pixel taller than the reference, with the whole difference
# landing under the buttons - was reported by eye, because there was no
# criterion that could see it.
# ---------------------------------------------------------------------
echo
echo "==== [18/31]caption bar symmetry ===="
LOG=build/ci-capsym.log
build/caption_symmetry_gate.sh >"$LOG" 2>&1
CAPSYM_RC=$?
cat "$LOG"
case $CAPSYM_RC in
    0)
        echo "PASS: caption bar symmetry"
        note "PASS  caption bar symmetry"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: caption bar symmetry ($REASON)"
        note "SKIP  caption bar symmetry ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: caption bar symmetry"
        note "FAIL  caption bar symmetry"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 19: comments describe the code, not the edits that made it.
#
# The phrasing half of the rule comment_hygiene_gate covers by hash,
# date and script. A ratchet, per file: some of these sentences carry
# knowledge that stops a removed design being re-added, so the repair is
# to restate them in the present tense, not to delete them, and doing
# that everywhere at once is a large edit across files other sessions
# hold open. Lowering a baseline number is the work; raising one is the
# regression this catches.
# ---------------------------------------------------------------------
echo
echo "==== [19/31]comment phrasing ratchet ===="
LOG=build/ci-changelog.log
build/changelog_comment_gate.sh >"$LOG" 2>&1
CHLOG_RC=$?
cat "$LOG"
case $CHLOG_RC in
    0)
        echo "PASS: comment phrasing ratchet"
        note "PASS  comment phrasing ratchet"
        PASS=$((PASS + 1))
        ;;
    *)
        echo "FAIL: comment phrasing ratchet"
        note "FAIL  comment phrasing ratchet"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 20: the bilingual string tables are in their enums' order.
#
# src/err.c and gui/localized_strings.c are positional, because Visual
# C++ 4.0 cannot compile the designated form - so position is what binds
# a string to a code, and the tag comment on each row is the only thing
# that says which. Nothing else here would notice a transposed pair: the
# strings are all still present, and the engine's numeric output does not
# touch them.
# ---------------------------------------------------------------------
echo
echo "==== [20/31]string table order ===="
LOG=build/ci-strtbl.log
build/str_table_order_gate.sh >"$LOG" 2>&1
STRTBL_RC=$?
cat "$LOG"
case $STRTBL_RC in
    0)
        echo "PASS: string table order"
        note "PASS  string table order"
        PASS=$((PASS + 1))
        ;;
    *)
        echo "FAIL: string table order"
        note "FAIL  string table order"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 21: unaligned access, asked of the machine.
#
# -Wcast-align=strict is blind here: the buffers reach the kernels
# through void *, so the cast it would check is not written anywhere the
# compiler sees. UBSan checks the access. It matters because an
# unaligned load is legal on x86 and merely slow, while on ARMv5TE it
# returns a rotated word - the right bytes in the wrong order, no fault,
# a wrong number - and every stage above this one runs on x86.
# ---------------------------------------------------------------------
echo
echo "==== [21/31]unaligned access (UBSan, WSL) ===="
LOG=build/ci-align.log
build/align_gate.sh >"$LOG" 2>&1
ALIGN_RC=$?
cat "$LOG"
case $ALIGN_RC in
    0)
        echo "PASS: unaligned access"
        note "PASS  unaligned access"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: unaligned access ($REASON)"
        note "SKIP  unaligned access ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: unaligned access"
        note "FAIL  unaligned access"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 22: size. The gui on one 1.44 MB floppy, and the CLI's .text
# under a ceiling - the second because the first is at 68% and would not
# redden until something had gone badly wrong.
# ---------------------------------------------------------------------
echo
echo "==== [22/31]code size ===="
LOG=build/ci-codesize.log
build/codesize_gate.sh >"$LOG" 2>&1
SIZE_RC=$?
cat "$LOG"
case $SIZE_RC in
    0)
        echo "PASS: code size"
        note "PASS  code size"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: code size ($REASON)"
        note "SKIP  code size ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: code size"
        note "FAIL  code size"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 23: one float axis.
#
# --fixed off|all replaced seven per-tier flags, and a retired flag
# comes back by accident - somebody re-adds a parse arm for one tier
# while chasing something else, and nothing says a word: the binary
# runs, the logits match, and the axis count has quietly gone from one
# to two. No other stage counts the knobs.
# ---------------------------------------------------------------------
echo
echo "==== [23/31]float knob count ===="
LOG=build/ci-floatknob.log
build/float_knob_gate.sh >"$LOG" 2>&1
FKNOB_RC=$?
cat "$LOG"
case $FKNOB_RC in
    0)
        echo "PASS: float knob count"
        note "PASS  float knob count"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: float knob count ($REASON)"
        note "SKIP  float knob count ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: float knob count"
        note "FAIL  float knob count"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 24: the per-operator cross-checks.
#
# Every stage above compares whole-engine output. These compare one
# operator's implementations against each other, and that is not the
# same question: a coverage build shows stage 15's fixture reaching
# 47.57% of src/ops_kernel_dot_mmx.h, and the union over five
# checkpoints only 55.34%, so the rest belongs to weight formats no
# local model uses. Feeding the kernels directly is the only route
# there.
#
# They existed and nobody called them. Two were passing without
# measuring anything when this stage was written.
# ---------------------------------------------------------------------
echo
echo "==== [24/31]per-operator cross-checks ===="
LOG=build/ci-xcheck.log
build/xcheck_gate.sh >"$LOG" 2>&1
XCHK_RC=$?
cat "$LOG"
case $XCHK_RC in
    0)
        echo "PASS: per-operator cross-checks"
        note "PASS  per-operator cross-checks"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: per-operator cross-checks ($REASON)"
        note "SKIP  per-operator cross-checks ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: per-operator cross-checks"
        note "FAIL  per-operator cross-checks"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 25: the gcc builds emit no warnings, ASSERTED.
#
# Stage 2 reads Watcom's own "0 warnings, 0 errors" per file and fails
# on a non-zero. Nothing did that for gcc: warnings went into the build
# log, make exited 0, and the stage printed PASS. forward.c's unused
# `lslot` sat in every `make native` for as long as that was true, and
# was finally noticed by the 32-bit x87 build - a third toolchain,
# looked at by hand.
# ---------------------------------------------------------------------
echo
echo "==== [25/31]gcc build warnings ===="
LOG=build/ci-gccwarn.log
build/gcc_warning_gate.sh >"$LOG" 2>&1
GCCW_RC=$?
cat "$LOG"
case $GCCW_RC in
    0)
        echo "PASS: gcc build warnings"
        note "PASS  gcc build warnings"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: gcc build warnings ($REASON)"
        note "SKIP  gcc build warnings ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: gcc build warnings"
        note "FAIL  gcc build warnings"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 26: every quantised matmul reaches a tiered kernel
# ---------------------------------------------------------------------
echo
echo "==== [26/31]tier reach ===="
LOG=build/ci-tierreach.log
build/tier_reach_gate.sh "$MODELS_ROOT/kmr20" >"$LOG" 2>&1
TIER_RC=$?
cat "$LOG"
case $TIER_RC in
    0)
        echo "PASS: tier reach"
        note "PASS  tier reach"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: tier reach ($REASON)"
        note "SKIP  tier reach ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: tier reach"
        note "FAIL  tier reach"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 27: #pragma aux argument registers match what the body reads
# ---------------------------------------------------------------------
echo
echo "==== [27/31]aux parm ===="
LOG=build/ci-auxparm.log
build/aux_parm_arity_gate.sh >"$LOG" 2>&1
APARM_RC=$?
cat "$LOG"
if [ $APARM_RC -eq 0 ]; then
    echo "PASS: aux parm"
    note "PASS  aux parm"
    PASS=$((PASS + 1))
else
    echo "FAIL: aux parm"
    note "FAIL  aux parm"
    FAIL=$((FAIL + 1))
fi

# ---------------------------------------------------------------------
# Stage 28: the last commit did not change what a kernel computes
# ---------------------------------------------------------------------
echo
echo "==== [28/31]aux equiv ===="
LOG=build/ci-auxequiv.log
build/aux_equiv_gate.sh >"$LOG" 2>&1
AEQ_RC=$?
cat "$LOG"
case $AEQ_RC in
    0)
        echo "PASS: aux equiv"
        note "PASS  aux equiv"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: aux equiv ($REASON)"
        note "SKIP  aux equiv ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: aux equiv"
        note "FAIL  aux equiv"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 29: the prefetch knob changes timing, not values
# ---------------------------------------------------------------------
echo
echo "==== [29/31]prefetch parity ===="
LOG=build/ci-pfparity.log
build/prefetch_parity_gate.sh >"$LOG" 2>&1
PFP_RC=$?
cat "$LOG"
case $PFP_RC in
    0)
        echo "PASS: prefetch parity"
        note "PASS  prefetch parity"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: prefetch parity ($REASON)"
        note "SKIP  prefetch parity ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: prefetch parity"
        note "FAIL  prefetch parity"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 30: the SSE1 norm helpers, gcc twin against Watcom twin
# ---------------------------------------------------------------------
echo
echo "==== [30/31]norm sse parity ===="
LOG=build/ci-normsse.log
build/norm_sse_parity_gate.sh >"$LOG" 2>&1
NSP_RC=$?
cat "$LOG"
case $NSP_RC in
    0)
        echo "PASS: norm sse parity"
        note "PASS  norm sse parity"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: norm sse parity ($REASON)"
        note "SKIP  norm sse parity ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: norm sse parity"
        note "FAIL  norm sse parity"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Stage 31: the int32->float accumulate tiers, gcc against Watcom
# ---------------------------------------------------------------------
echo
echo "==== [31/31]i32f acc ===="
LOG=build/ci-i32facc.log
build/i32f_acc_gate.sh >"$LOG" 2>&1
I32F_RC=$?
cat "$LOG"
case $I32F_RC in
    0)
        echo "PASS: i32f acc"
        note "PASS  i32f acc"
        PASS=$((PASS + 1))
        ;;
    2)
        REASON=$(grep -m1 'SKIP' "$LOG")
        echo "SKIP: i32f acc ($REASON)"
        note "SKIP  i32f acc ($REASON)"
        SKIP=$((SKIP + 1))
        ;;
    *)
        echo "FAIL: i32f acc"
        note "FAIL  i32f acc"
        FAIL=$((FAIL + 1))
        ;;
esac

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
echo
echo "==== summary ===="
for line in "${SUMMARY[@]:-}"; do
    [ -n "$line" ] && echo "$line"
done
echo
echo "pass=$PASS fail=$FAIL skip=$SKIP"
if [ "$FAIL" -eq 0 ]; then
    VERDICT="CI: PASS (skips do not count as failures, but are listed above)"
    RC=0
else
    VERDICT="CI: FAIL"
    RC=1
fi
echo "$VERDICT"
# The same two lines, to one fixed path, and that path printed. A round's
# verdict landing wherever the runner's command line pointed it
# (`> /tmp/...` one round, `> E:/tmp/...` the next) made "where is this
# round's CI: PASS" a question about somebody's memory - the file was on
# disk, complete, and still not findable. This does not change the
# per-stage logs (build/ci-*.log); it adds the summary line the project
# quotes, beside the counts. build/ is gitignored, so this is a local
# reading rather than an artifact.
{ echo "pass=$PASS fail=$FAIL skip=$SKIP"; echo "$VERDICT"; } > build/ci-summary.txt
echo "ci.sh: summary also written to build/ci-summary.txt"
# Criterion: after a full run, build/ci-summary.txt holds the TWO VERDICT
# LINES this run printed - `pass=... fail=... skip=...` and the CI:
# PASS/FAIL line. Not "the last two lines of stdout": the line above
# announcing the path is printed after them, so the file matches stdout's
# last two lines only by accident of where that announcement sits.
exit "$RC"
