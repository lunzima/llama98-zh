#!/bin/bash
# Does anything in this tree spell a 64-bit printf modifier by hand?
#
# lz_int.h exists so the 64-bit type and its length modifier are named once
# each: `lz_i64`/`lz_u64` and `LZ_PRI64` ("I64" before MSVC 2013, "ll"
# elsewhere). The tree was swept for the TYPE (213 sites) and the sweep left
# no gate behind, so the 214th was written by hand - src/model.c:1893, which
# is what turned vc4-compile-gate red. The FORMAT half of the same rule had
# zero call sites anywhere, which is what this gate is for: a bare %ll is
# wrong on any toolchain whose lz_i64 is not `long long`, and NOTHING ELSE
# CAN SEE IT. vc4-compile-gate compiles and does not run, and format strings
# are not type-checked, so err.c passes it while carrying 30 of these.
#
# SCOPE, AND WHY IT DIFFERS FROM vc4-compile-gate's. This question is a
# RUNTIME one - a format string handed to a CRT that does not understand
# `ll` prints garbage - so its scope is "every format string this tree can
# reach", not "every file VC++ 4.0 compiles". It therefore covers
# src/ gui/ common/, the same three directories c89_floor_gate.sh scans.
# Two gates disagreeing about a file is not a defect; an UNSTATED scope is,
# because the next person reads the disagreement as one of them being wrong.
#
# WHY THE SCAN IS LEXICAL, NOT LINE-BASED. src/err.c's tables are lines of
# the form `/* LZ_ERR_X */ "template"` - the match is in the DATA, on a line
# whose first token is a comment. src/err.h's enum documents the same
# templates in trailing comments - the match is in the COMMENT, on a line
# whose first token is code. A rule keyed on the line's first token gets one
# of those wrong in each direction, and either wrong answer still runs: the
# first direction hides real sites, the second reports phantom ones, and the
# gate stays green or red for a reason nobody can see. So the scan strips
# /* */ and // before matching - measured, that is err.c 30 (kept, data) and
# err.h 0 (dropped, comments).
#
# THE BASELINE, AND WHERE IT IS NARROW. This is a per-file count, taken at
# the commit named below. It forbids NEW bare modifiers and permits the
# existing ones, so it holds the ratchet without asking anyone to clean 78
# sites first. It cannot see an exchange WITHIN one file (remove one, add
# one, net zero); it DOES see an exchange ACROSS files, because both counts
# move. That is the whole of the weakness - do not build something more
# elaborate to cover the half it already covers.
#
# Exit: 0 pass, 1 fail, 2 skip (no python3).
set -u
cd "$(dirname "$0")/.."

command -v python3 >/dev/null 2>&1 || { echo "pri64_scan_gate: SKIP - no python3"; exit 2; }

BASELINE_SHA=2270867

OUT=$(python3 - <<'PY'
import glob, re, sys

BS = chr(92)

def strip_comments(src):
    out, i, n = [], 0, len(src)
    while i < n:
        if src.startswith('/*', i):
            j = src.find('*/', i + 2)
            i = n if j < 0 else j + 2
            out.append(' ')
        elif src.startswith('//', i):
            j = src.find(chr(10), i)
            i = n if j < 0 else j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == BS else 1
            out.append(src[i:j + 1])
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return ''.join(out)

PAT = re.compile(r'%ll[du]')
BASE = {"src/cli_main.c": 46, "src/err.c": 30, "gui/aboutdlg.c": 1, "src/model.c": 1}

files = sorted(glob.glob('src/*.[ch]') + glob.glob('gui/*.[ch]') + glob.glob('common/*.[ch]'))
bad = 0
for f in files:
    f = f.replace(BS, '/')
    n = len(PAT.findall(strip_comments(open(f, encoding='utf-8', errors='replace').read())))
    exp = BASE.get(f, 0)
    if n > exp:
        print("FAIL: %s has %d, baseline %d" % (f, n, exp))
        bad = 1
    elif n < exp:
        print("note: %s is at %d, below its baseline of %d - tighten the baseline" % (f, n, exp))
print("scanned %d file(s); baseline taken at %s" % (len(files), "2270867"))
sys.exit(bad)
PY
)
rc=$?
printf '%s\n' "$OUT" | grep -v '^scanned'

# The gate's own control: a scan that stopped matching would report zero
# everywhere and read as clean. A file with one hand-spelled modifier must
# be found, or this gate has never been shown to see anything.
CTL=$(python3 - <<'PY'
import re
src = 'const char *s = "want %lld here";\n'
print("1" if re.search(r'%ll[du]', src) else "0")
PY
)
if [ "$CTL" != "1" ]; then
    echo "pri64_scan_gate: FAIL - the control was not detected; this scan sees nothing"
    exit 1
fi
[ "$rc" -eq 0 ] || { echo "pri64_scan_gate: FAIL"; exit 1; }
echo "pri64_scan_gate: PASS (0 new bare 64-bit modifiers; control detected)"
exit 0
