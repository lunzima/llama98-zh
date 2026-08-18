#!/bin/bash
# Minimal CI. No .github/ workflow exists yet - this is the script such
# a workflow would call, and it also runs by hand:
#
#   build/ci.sh
#
# Stages, each independently PASS / FAIL / SKIP, none silent:
#   1. make native      (gcc)
#   2. make watcom-cli   (Open Watcom, the target compiler)
#   3. llama98 <dir> --check on a curated set of KunMoe checkpoints
#   4. build/arm/parity_gate.sh (x86 <-> ARMv5TE bit-identity)
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
echo "==== [1/4] make native ===="
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
echo "==== [2/4] make watcom-cli ===="
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
echo "==== [3/4] model --check sweep ===="
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
echo "==== [4/4] ARM parity gate ===="
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
