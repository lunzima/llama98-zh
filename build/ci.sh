#!/bin/bash
# Minimal CI. No .github/ workflow exists yet - this is the script such
# a workflow would call, and it also runs by hand:
#
#   build/ci.sh
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
#      EVERY fixed-point write tier (all 12 --*-write values plus
#      --fixed all, plus the float baseline as a control) over hundreds
#      of tokens of real text, the case the --norm-write int tier's
#      original 8/16-token verification could not see
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
ROOT=$(pwd)

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
echo "==== [1/11] make native ===="
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
echo "==== [2/11] make watcom-cli ===="
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
echo "==== [3/11] model --check sweep ===="
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
CHECK_MODELS=${CHECK_MODELS:-"s1v3 kunkun98-pilot-20260806 kunmoe-v2"}

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
echo "==== [4/11] ARM parity gate ===="
ARM_MODEL="$MODELS_ROOT/kunkun98-pilot-20260806"
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
echo "==== [5/11] control-switch matrix ===="
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
echo "==== [6/11] fixed-tier cross-compiler gate ===="
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
echo "==== [7/11] recurrence float-arm gate ===="
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
echo "==== [8/11] watcom gui/dll/dos targets ===="
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
# Stage 8: 32-bit x87 gcc build under WSL (Windows gcc has no 32-bit
# libs). The fourth toolchain shape - a Socket-7-class target with x87
# as its only float unit. SKIP when WSL is not usable, never PASS.
# ---------------------------------------------------------------------
echo
echo "==== [9/11] 32-bit x87 gcc build (WSL) ===="
LOG=build/ci-x87.log
if ! command -v wsl >/dev/null 2>&1; then
    echo "SKIP: 32-bit x87 build (wsl not installed)"
    note "SKIP  32-bit x87 build (wsl not installed)"
    SKIP=$((SKIP + 1))
elif ! wsl bash -lc 'printf "int main(void){return 0;}\n" >/tmp/_m32t.c && gcc -m32 /tmp/_m32t.c -o /tmp/_m32t 2>/dev/null' >/dev/null 2>&1; then
    echo "SKIP: 32-bit x87 build (WSL present but no gcc -m32 multilib)"
    note "SKIP  32-bit x87 build (no -m32 multilib in WSL)"
    SKIP=$((SKIP + 1))
else
    # $ROOT is Git Bash shaped (/e/llama98-zh); the WSL cd needs the
    # /mnt/e/... shape. The build script finds the repo root itself.
    WSL_ROOT=$ROOT
    case "$ROOT" in
        /[a-z]/*) WSL_ROOT="/mnt/$(echo "$ROOT" | cut -c2)/$(echo "$ROOT" | cut -c4-)" ;;
    esac
    if wsl bash -lc "cd '$WSL_ROOT' && bash build/x87/build_engine.sh" >"$LOG" 2>&1; then
        if grep -q '^linked:' "$LOG"; then
            echo "PASS: 32-bit x87 build (WSL)"
            note "PASS  32-bit x87 build (WSL)"
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
echo "==== [10/11] big-endian parity gate (WSL) ===="
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
# Stage 11: the two PERFORMANCE-only knobs. Every other knob in this
# engine is a tier, and a tier that moves the numbers is doing its job -
# which is why the tier gates cannot say anything about --pair and
# --prefetch, whose entire contract is that they move NOTHING. That made
# them the two easiest knobs to break silently.
# ---------------------------------------------------------------------
echo
echo "==== [11/11] performance-knob invariance ===="
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
    echo "CI: PASS (skips do not count as failures, but are listed above)"
    exit 0
else
    echo "CI: FAIL"
    exit 1
fi
