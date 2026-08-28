#!/bin/bash
# The prefetch knob must not change any result, and this proves the check
# can see it when one does.
#
# WATCOM IS NOT OPTIONAL HERE. LZ_Q41_PFSEL and its siblings live behind
# `#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)`; an ordinary
# gcc build compiles none of them, so a gcc-only run of this probe is
# green without having executed a single prefetch arm. The mutation
# below was planted, built with gcc, and changed nothing at all before
# that was understood - which is the whole reason this driver exists
# rather than a bare probe.
#
# THE MUTATION is one arm differing by one: the LZ_PF_NONE arm of
# LZ_Q41_PFSEL adds 1 to acc[0]. That is the defect shape the property
# guards against - not the prefetch instruction, which has no
# architectural effect, but an ARM whose loop bound, kernel or guard
# differs from its siblings' and which therefore only one --prefetch
# setting would ever show.
#
# Exit: 0 pass, 1 fail, 2 skip (no Watcom).
set -u
cd "$(dirname "$0")/.."

if [ ! -x /c/WATCOM/binnt64/wcc386.exe ] && \
   [ ! -x /mnt/c/WATCOM/binl64/wcc386 ]; then
    echo "prefetch_parity_gate: SKIP - no C:\\WATCOM, and the prefetch"
    echo "  arms are not compiled by any other toolchain"
    exit 2
fi
[ -f .prof/wbuild_xcheck.sh ] || {
    echo "prefetch_parity_gate: SKIP - .prof/wbuild_xcheck.sh absent"; exit 2; }

echo "-- as shipped --"
bash .prof/wbuild_xcheck.sh prefetch_parity_probe > /tmp/pfg.log 2>&1
[ -x .prof/_wxc/probe.exe ] || {
    echo "prefetch_parity_gate: SKIP - probe did not build"
    tail -6 /tmp/pfg.log; exit 2; }
.prof/_wxc/probe.exe > /tmp/pfg.run 2>&1
rc=$?
tail -6 /tmp/pfg.run
[ $rc -eq 0 ] || { echo "prefetch_parity_gate: FAIL"; exit 1; }

# ---- the control -------------------------------------------------------
W=$(mktemp -d "$(pwd)/../.pfgate-XXXXXX") || exit 1
trap 'rm -rf "$W"' EXIT
cp -r src build .prof "$W/" 2>/dev/null || true
PY=python; command -v python >/dev/null 2>&1 || PY=python3
"$PY" - "$W/src/ops_kernel_dot_shared.h" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8", errors="replace").read()
old = "else if (pf_m_ == LZ_PF_NONE) { LZ_Q41_ACC(DOT32, LZ_PFI_NONE); }"
new = ("else if (pf_m_ == LZ_PF_NONE) { LZ_Q41_ACC(DOT32, LZ_PFI_NONE);"
       " acc[0] += 1; }")
if s.count(old) != 1:
    print("  the mutation site is gone; this gate is unproven")
    sys.exit(2)
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new, 1))
PYEOF
[ $? -eq 0 ] || { echo "prefetch_parity_gate: FAIL - could not plant the"
                  echo "  control mutation"; exit 1; }

echo "-- control: the --prefetch none arm differs by one --"
R="$W" bash .prof/wbuild_xcheck.sh prefetch_parity_probe > "$W/build.log" 2>&1
if [ ! -x "$W/.prof/_wxc/probe.exe" ]; then
    echo "prefetch_parity_gate: FAIL - the mutated tree did not build, so"
    echo "  the control did not run"
    tail -6 "$W/build.log"; exit 1
fi
"$W/.prof/_wxc/probe.exe" > "$W/run.log" 2>&1
mrc=$?
if [ $mrc -eq 0 ]; then
    echo "prefetch_parity_gate: FAIL - a mode-dependent difference did NOT"
    echo "  redden the probe. Its green above means the sweep ran, not that"
    echo "  the arms agree."
    exit 1
fi
grep -m1 "MISMATCH" "$W/run.log" | sed 's/^/  /'
echo "prefetch_parity_gate: PASS (mutation caught)"
exit 0
