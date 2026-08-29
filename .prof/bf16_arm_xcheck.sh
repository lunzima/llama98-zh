#!/bin/bash
# lz_bf16's ARM fast paths against its own C core, on ARM under QEMU.
#
#   bash .prof/bf16_arm_xcheck.sh
#
# WHY IT EXISTS. build/kernel_probe_census.sh's ARM half reported
# bf16_addsub_fast_asm and bf16_mul_fast_asm as reached by no
# ARM-cross-building driver, and unlike the dot kernels there is no
# end-to-end gate that could be covering them: the Makefile says of
# lz_bf16.c that NOTHING CALLS IT TODAY. The file is compiled by every
# toolchain to keep its portability honest, which is exactly the state
# in which an arithmetic defect waits without anyone noticing.
#
# THE REFERENCE IS THE FILE'S OWN C CORE. src/ops_arm_prim.h states the
# rule: a primitive owes its C tier bit-identity, and DECLINES (returns
# 0) where it cannot promise that, so the caller runs the C. So the
# comparison is two builds of the same source - one where the fast path
# runs, one where every call site is forced to decline - over the same
# inputs. No oracle is invented; the C core is the oracle, and this
# script only has to make it answer.
#
# THE THIRD RUN IS WHAT MAKES THE OTHER TWO MEAN ANYTHING. If the fast
# path never accepted, both builds would take the C path and agree
# perfectly while testing nothing. The instrumented build counts
# acceptances: 88,152 of 200,000 today, and the check fails if it is
# zero.
#
# Exit: 0 pass, 1 fail, 2 skip (no cross-compiler or no QEMU).
set -u
cd "$(dirname "$0")/.."

FLAGS="-march=armv5te -mtune=arm926ej-s -mfloat-abi=soft -DLZ_NO_FPU=1"
FLAGS="$FLAGS -O2 -std=c99 -Isrc"
PY=python; command -v python >/dev/null 2>&1 || PY=python3

if ! command -v wsl >/dev/null 2>&1 || \
   ! wsl -e bash -lc 'command -v arm-linux-gnueabi-gcc-14 >/dev/null' 2>/dev/null; then
    echo "bf16-arm-xcheck: SKIP - no ARM cross-compiler or QEMU"
    exit 2
fi

# Beside the repo: Git Bash's /tmp and WSL's are different directories.
W=$(mktemp -d "$(pwd)/../.bf16arm-XXXXXX"); trap 'rm -rf "$W"' EXIT

wslpath_of() {
    case "$1" in
        /[a-z]/*) printf '/mnt%s' "$1" ;;
        *)        printf '%s' "$1" ;;
    esac
}

build_and_run() {   # $1 = tree root, $2 = tag
    local d
    d=$(wslpath_of "$1")
    wsl -e bash -lc "cd '$d' && arm-linux-gnueabi-gcc-14 $FLAGS \
        -o /tmp/bf16arm-$2 .prof/bf16_arm_xcheck.c src/lz_bf16.c 2>&1 && \
        qemu-arm-static -cpu arm926 -L /usr/arm-linux-gnueabi /tmp/bf16arm-$2"
}

hash_of() { printf '%s\n' "$1" | sed -n 's/.*hash \([0-9A-F]*\).*/\1/p' | head -1; }

FAST=$(build_and_run "$(pwd)" fast 2>&1)
echo "$FAST"
HF=$(hash_of "$FAST")
[ -n "$HF" ] || { echo "bf16-arm-xcheck: FAIL - the fast build produced no hash"
                  exit 1; }

# ---- the declining arm -------------------------------------------------
cp -r src "$W/src" && cp -r .prof "$W/.prof" 2>/dev/null
"$PY" - "$W/src/lz_bf16.c" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8", errors="replace").read()
n = 0
for fn in ("bf16_mul_fast_asm", "bf16_addsub_fast_asm"):
    key = "if (" + fn + "("
    n += s.count(key)
    s = s.replace(key, "if (0 && " + fn + "(")
if n < 2:
    print("  the decline sites are gone; this check is unproven")
    sys.exit(2)
open(p, "w", encoding="utf-8", newline="").write(s)
PYEOF
[ $? -eq 0 ] || { echo "bf16-arm-xcheck: FAIL - could not build the"
                  echo "  declining arm"; exit 1; }

SLOW=$(build_and_run "$W" slow 2>&1)
HS=$(hash_of "$SLOW")
[ -n "$HS" ] || { echo "$SLOW"
                  echo "bf16-arm-xcheck: FAIL - the declining build produced"
                  echo "  no hash, so nothing was compared"; exit 1; }

echo "  fast $HF   C-core $HS"
if [ "$HF" != "$HS" ]; then
    echo "bf16-arm-xcheck: FAIL - the ARM fast path disagrees with the C core"
    exit 1
fi

# ---- did the fast path ever accept? -----------------------------------
cp -r src "$W/src2" && mkdir -p "$W/.prof2" && cp -r .prof/. "$W/.prof2/"
"$PY" - "$W/src2/lz_bf16.c" "$W/.prof2/bf16_arm_xcheck.c" <<'PYEOF'
import sys
c, probe = sys.argv[1], sys.argv[2]
s = open(c, encoding="utf-8", errors="replace").read()
s = s.replace("static uint32_t bf16_binop_c(uint32_t a, uint32_t b, int op) {",
              "long g_bf16_fast_taken = 0;\n"
              "static uint32_t bf16_binop_c(uint32_t a, uint32_t b, int op) {",
              1)
hit = 0
for fn in ("bf16_mul_fast_asm", "bf16_addsub_fast_asm"):
    old = "if (" + fn + "(a, b, &q)) return q;"
    hit += s.count(old)
    s = s.replace(old, "if (" + fn + "(a, b, &q)) { g_bf16_fast_taken++;"
                       " return q; }")
if hit < 2:
    print("  the accept branches are gone; acceptance is unmeasured")
    sys.exit(2)
open(c, "w", encoding="utf-8", newline="").write(s)
t = open(probe, encoding="utf-8", errors="replace").read()
t = t.replace('#include "lz_bf16.h"',
              '#include "lz_bf16.h"\nextern long g_bf16_fast_taken;', 1)
t = t.replace('    printf("bf16-arm-xcheck: hash',
              '    printf("  ACCEPTED %ld\\n", g_bf16_fast_taken);\n'
              '    printf("bf16-arm-xcheck: hash', 1)
open(probe, "w", encoding="utf-8", newline="").write(t)
PYEOF
[ $? -eq 0 ] || { echo "bf16-arm-xcheck: FAIL - could not instrument"; exit 1; }

d=$(wslpath_of "$W")
CNT=$(wsl -e bash -lc "cd '$d' && arm-linux-gnueabi-gcc-14 $FLAGS -Isrc2 \
    -o /tmp/bf16arm-cnt .prof2/bf16_arm_xcheck.c src2/lz_bf16.c 2>&1 && \
    qemu-arm-static -cpu arm926 -L /usr/arm-linux-gnueabi /tmp/bf16arm-cnt" 2>&1)
TAKEN=$(printf '%s\n' "$CNT" | sed -n 's/.*ACCEPTED \([0-9]*\).*/\1/p' | head -1)
if [ -z "$TAKEN" ]; then
    printf '%s\n' "$CNT" | tail -4
    echo "bf16-arm-xcheck: FAIL - could not count acceptances, so the"
    echo "  agreement above may be two C paths agreeing"
    exit 1
fi
echo "  fast path accepted $TAKEN time(s)"
if [ "$TAKEN" -eq 0 ]; then
    echo "bf16-arm-xcheck: FAIL - the fast path never accepted, so both"
    echo "  builds ran the C core and the match means nothing"
    exit 1
fi
echo "bf16-arm-xcheck: PASS (hash $HF, $TAKEN acceptance(s))"
exit 0
