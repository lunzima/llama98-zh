#!/bin/bash
# The hand-drawn caption bar and its buttons, checked as arithmetic.
#
# captionwnd.c builds bar_rect and btn_rect from raw system metrics
# independently - the buttons are not derived from the bar - so the gap
# above and below a button is a closed-form expression and can be
# asserted without drawing anything. build/capmetrics.c holds the same
# arithmetic and the equality.
#
# WHY NOT A SCREENSHOT. Appearance has no automated criterion and a
# screen readback grabs whatever window is on top; both were tried
# elsewhere in this tree and both produced confident wrong answers. This
# checks the one property that IS mechanical.
#
# Exit codes:
#   0  PASS
#   1  FAIL - the paddings differ, or the probe will not build
#   2  SKIP - not a Windows host, or no gcc
set -u
cd "$(dirname "$0")/.."

command -v gcc >/dev/null 2>&1 || { echo "caption_symmetry_gate: SKIP - no gcc"; exit 2; }
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) ;;
    *) echo "caption_symmetry_gate: SKIP - GetSystemMetrics needs a Windows host"; exit 2 ;;
esac

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
if ! gcc -O2 -o "$T/capmetrics.exe" build/capmetrics.c -lgdi32 -luser32 2>"$T/err"; then
    echo "caption_symmetry_gate: FAIL - probe does not build"
    sed 's/^/      /' "$T/err" | head -10
    exit 1
fi
"$T/capmetrics.exe"
RC=$?

# Mutation, run every time rather than trusted from a past session: put
# back the missing -1 that made the bar a pixel too tall and confirm the
# equality above actually reddens. A criterion whose only evidence is
# that it once printed PASS is not evidence.
sed 's/GetSystemMetrics(SM_CYCAPTION) - 1; }/GetSystemMetrics(SM_CYCAPTION); }/' \
    build/capmetrics.c > "$T/mut.c"
if cmp -s build/capmetrics.c "$T/mut.c"; then
    echo "caption_symmetry_gate: FAIL - the mutation edited nothing, so it"
    echo "      proves nothing; bar_bottom's shape in capmetrics.c changed"
    exit 1
fi
if gcc -O2 -o "$T/mut.exe" "$T/mut.c" -lgdi32 -luser32 2>/dev/null; then
    if "$T/mut.exe" >/dev/null 2>&1; then
        echo "caption_symmetry_gate: FAIL - the taller bar still passes, so"
        echo "      this gate cannot see the defect it was written for"
        exit 1
    fi
fi

[ "$RC" -eq 0 ] || exit 1
exit 0
