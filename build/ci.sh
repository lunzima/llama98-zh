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
#      BOTH arms of --fixed (all, the shipping engine; off, the float
#      control) over hundreds of tokens of real text - it used to be 12
#      per-tier values, which is what the one knob replaced. This is the
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
#      comment rule: no file gains an "it used to be" / "no longer" /
#      "the old loop". A ratchet, because some of those sentences are
#      load-bearing and the repair is to restate them, not delete them
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
#  26. build/tier_reach_gate.sh - every quantised matmul REACHES its
#      tiered kernel. A tensor whose (dtype, gs) fails a dispatch guard
#      takes matmul_scalar_ref and is still correct, so every numeric
#      stage above passes on it. Only an execution count separates
#      "agrees" from "agrees and was reached"
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
echo "==== [1/26]make native ===="
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
echo "==== [2/26]make watcom-cli ===="
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
echo "==== [3/26]model --check sweep ===="
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
echo "==== [4/26]ARM parity gate ===="
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
echo "==== [5/26]control-switch matrix ===="
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
echo "==== [6/26]fixed-tier cross-compiler gate ===="
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
echo "==== [7/26]recurrence float-arm gate ===="
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
echo "==== [8/26]watcom gui/dll/dos targets ===="
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
echo "==== [9/26]32-bit x87 gcc build (WSL) ===="
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
echo "==== [10/26]big-endian parity gate (WSL) ===="
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
echo "==== [11/26]performance-knob invariance ===="
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
echo "==== [12/26]--kv-rot quality gate ===="
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
echo "==== [13/26] int-pipeline control arms ===="
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
echo "==== [14/26]kernel coverage matrix ===="
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
echo "==== [15/26]kernel ISA identity ===="
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
echo "==== [16/26]SubLN activation path ===="
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
echo "==== [17/26]gui/CLI engine flag parity ===="
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
echo "==== [18/26]caption bar symmetry ===="
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
echo "==== [19/26]comment phrasing ratchet ===="
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
echo "==== [20/26]string table order ===="
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
echo "==== [21/26]unaligned access (UBSan, WSL) ===="
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
echo "==== [22/26]code size ===="
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
echo "==== [23/26]float knob count ===="
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
echo "==== [24/26]per-operator cross-checks ===="
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
echo "==== [25/26]gcc build warnings ===="
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
echo "==== [26/26]tier reach ===="
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
