#!/bin/bash
# The gcc builds emit zero warnings, asserted rather than printed.
#
# WHY THIS EXISTS. `make watcom-cli` reports "N lines, 0 warnings, 0
# errors" per file and ci.sh checks that number. The gcc side prints
# warnings into the build log and nothing reads them, so make succeeds
# and the stage says PASS. One lived there long enough to be found by a
# THIRD toolchain: forward.c's `lslot`, unused in every build where
# LZ_KV_2PLANE is off, which is every shipping build. It was visible in
# `make native` output the whole time.
#
# WHAT IS ASSERTED
#   1. The gcc CLI builds with zero warnings.
#   2. The gcc gui builds with zero warnings - it compiles the same
#      engine sources plus gui/, and gcc_flag_parity_gate holds the two
#      to the same flags, so a warning in gui/ is the half the CLI
#      cannot see.
#   3. The compile is a REAL one. A build that was already up to date
#      compiles nothing and reports zero warnings for that reason, which
#      is the failure mode this gate would otherwise have: green because
#      it measured nothing. Both targets are forced.
#
# NOT a ratchet with a baseline, unlike changelog_comment_gate: the
# count is zero today, and a warning is a defect report from the
# compiler rather than a style preference. If a warning ever has to be
# accepted, silence it AT THE SITE with a comment saying why - that puts
# the reason next to the code instead of in a number here.
#
# Exit codes:
#   0  PASS - both targets built, zero warnings
#   1  FAIL - a warning, or a build failed, or nothing was compiled
#   2  SKIP - no gcc
set -u
cd "$(dirname "$0")/.."

command -v gcc >/dev/null 2>&1 || {
    echo "gcc_warning_gate: SKIP - no gcc"; exit 2; }

MAKE=${MAKE:-make}
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

FAIL=0

# Force a real compile of both. `rm -rf build/native build/gui` is what
# makes llama98 and kunkun98-gui rebuild from source rather than relink;
# see the "measured nothing" note above.
#
# The wipe and the make are separated by a settle, because on this host
# a `rm -rf build/native` immediately followed by make fails with
# "cannot find build/native/ops_mmx.o" often enough to matter: the
# directory delete has not landed when the pattern rule's mkdir runs,
# and the object is written into a directory that is then removed.
rm -rf build/native build/gui
sleep 2
mkdir -p build/native build/gui

for tgt in llama98 kunkun98-gui; do
    LOG="$TMPD/$tgt.log"
    if ! "$MAKE" "$tgt" >"$LOG" 2>&1; then
        echo "FAIL: make $tgt did not build"
        grep -E 'error:|Error' "$LOG" | head -5 | sed 's/^/      /'
        FAIL=1
        continue
    fi
    # Proof that something was actually compiled. A relink emits no gcc
    # command line with -o pointing at the target, and an up-to-date
    # tree emits none at all.
    if ! grep -q 'gcc .*-o ' "$LOG"; then
        echo "FAIL: make $tgt ran no compile, so its 0 warnings mean nothing"
        FAIL=1
        continue
    fi
    n=$(grep -c 'warning:' "$LOG")
    if [ "$n" -ne 0 ]; then
        echo "FAIL: $tgt - $n gcc warning(s)"
        grep -E 'warning:' "$LOG" | head -6 | sed 's/^/      /'
        FAIL=1
    else
        echo "  $tgt: 0 warnings"
    fi
done

# MUTATION, because "zero warnings" is also what a broken grep reports.
# A file that is guaranteed to warn must be seen to warn, through the
# same pattern the loop above uses.
MUT="$TMPD/mut.c"
printf 'int f(void) { int unused_on_purpose; return 0; }\n' >"$MUT"
MN=$(gcc -c -O2 -std=c99 -Wall -Wextra -o "$TMPD/mut.o" "$MUT" 2>&1 \
     | grep -c 'warning:')
if [ "$MN" -eq 0 ]; then
    echo "FAIL: a deliberately-warning file reported 0 warnings, so this"
    echo "  gate cannot tell a clean build from a broken detector"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo "gcc_warning_gate: PASS (llama98 and kunkun98-gui, real compiles,"
    echo "  0 warnings; the planted warning is detected)"
    exit 0
fi
echo "gcc_warning_gate: FAIL"
exit 1
