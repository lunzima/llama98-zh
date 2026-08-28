#!/bin/bash
# Every register a #pragma aux body dereferences must be one Watcom
# actually writes on entry.
#
# WHY THIS AND NOT ARITY. The defect that produced this gate was an extra
# [__eax] at the FRONT of lz_dot32_t2_asm's __parm list. Watcom assigns
# arguments to the list left to right and IGNORES registers past the last
# argument, so the two pointers landed in eax and edx while the body's
# second instruction was `movq mm3, [ecx]` - reading whatever the caller
# happened to leave in ecx. .prof/rowdot_xcheck.c died on ECX=0. The same
# spare register at the END of the list is inert, and one had been sitting
# in lz_dot32_q41_asm doing nothing, which is exactly why "three registers
# for two parameters" did not look wrong to anyone reading it.
#
# So the count is not the property. The property is: for each register
# used as a memory BASE in the body, is it (a) among the first N entries
# of __parm, N being the prototype's argument count, or (b) written by
# the body before it is read? Anything else is an uninitialised base, and
# it does not announce itself - on a machine where the stale value is a
# valid address the kernel returns a plausible number.
#
# Arity is still reported, as a WARNING rather than a failure: a spare
# trailing register is legal and this tree has carried one.
#
# WHAT THIS CANNOT SEE: two parm registers SWAPPED. Both still appear in
# the effective set, so the membership test passes while the kernel reads
# its weights from the activation pointer. Only a cross-check that runs
# the kernel catches that - .prof/rowdot_xcheck.c for these, which is the
# probe the defect above was found by.
#
# Exit: 0 pass, 1 fail.
set -u
cd "$(dirname "$0")/.."

run_check() {   # $1 = tree root to scan
python - "$1" <<'PY'
import re, sys, glob, os

root = sys.argv[1]
GP = ("eax","ebx","ecx","edx","esi","edi","ebp")

def parse(path):
    s = open(path, encoding="utf-8", errors="replace").read()
    # Strip the line continuations so a pragma is one string, but keep
    # the pragma boundaries: a pragma ends at the first line with no
    # trailing backslash.
    out = []
    for m in re.finditer(r"#pragma aux\s+(\w+)\s*=", s):
        name = m.group(1)
        i = m.end()
        body = []
        for line in s[i:].split("\n"):
            body.append(line)
            if not line.rstrip().endswith("\\"):
                break
        out.append((name, "\n".join(body)))
    return s, out

def arg_count(src, name):
    d = re.search(r"\b" + re.escape(name) + r"\s*\(([^;{]*?)\)\s*;", src, re.S)
    if not d:
        return None
    a = " ".join(d.group(1).split())
    if a in ("", "void"):
        return 0
    # Splitting on commas is safe here: these prototypes take pointers
    # and scalars, no function-pointer or comma-bearing types.
    return a.count(",") + 1

fail = warn = seen = 0
files = sorted(glob.glob(os.path.join(root, "src", "*.c")) +
               glob.glob(os.path.join(root, "src", "*.h")))
for f in files:
    src, pragmas = parse(f)
    for name, body in pragmas:
        pm = re.search(r"__parm\s*((?:\[[^\]]*\]\s*)*)", body)
        if not pm:
            continue
        regs = [r.strip().replace("__", "") for r in
                re.findall(r"\[([^\]]*)\]", pm.group(1))]
        regs = [r for r in regs if r]          # __parm [] is zero, not one
        n = arg_count(src, name)
        if n is None:
            continue
        seen += 1
        effective = set(regs[:n])

        # Registers the body writes before any use: a base loaded inside
        # the body is legitimate even though no argument supplies it.
        written = set()
        for line in body.split("\n"):
            q = re.search(r'"\s*(\w+)\s+(\w+)\s*,', line)
            if q and q.group(2) in GP and q.group(1) not in ("cmp", "test"):
                written.add(q.group(2))

        bases = set()
        for line in body.split("\n"):
            for b in re.findall(r"\[\s*(e[a-z][a-z])", line):
                bases.add(b)
            # [eax+edx] style: the second register is a base too.
            for b in re.findall(r"\[\s*e[a-z][a-z]\s*\+\s*(e[a-z][a-z])", line):
                bases.add(b)

        bad = sorted(b for b in bases if b not in effective and b not in written)
        if bad:
            print("  FAIL %s (%s): dereferences %s, but __parm gives only %s"
                  % (name, os.path.basename(f), ", ".join(bad),
                     ", ".join(regs[:n]) or "nothing"))
            print("       %d parm register(s) listed for %d argument(s); "
                  "Watcom ignores the rest" % (len(regs), n))
            fail += 1
        elif len(regs) != n:
            print("  warn %s (%s): %d parm register(s) for %d argument(s) "
                  "- inert here, fatal if the spare moves left"
                  % (name, os.path.basename(f), len(regs), n))
            warn += 1

print("  %d pragma aux body/parm pair(s) checked, %d warning(s)" % (seen, warn))
if seen == 0:
    print("  FAIL - nothing was checked; the scan found no #pragma aux at all")
    sys.exit(1)
sys.exit(1 if fail else 0)
PY
}

echo "-- src/, as shipped --"
run_check "."
rc=$?
if [ $rc -ne 0 ]; then
    echo "aux_parm_arity_gate: FAIL"
    exit 1
fi

# ---- the mutation ------------------------------------------------------
# Plant the exact defect: a spare register at the FRONT of a two-argument
# kernel's list. If the gate cannot see that, its green above means the
# scan ran, not that the mapping is right.
W=$(mktemp -d "$(pwd)/../.auxparm-XXXXXX") || exit 1
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/src" && cp src/ops_mmx.c "$W/src/" || exit 1
sed -i 's/__parm \[__edx\] \[__ecx\] \\/__parm [__eax] [__edx] [__ecx] \\/' \
    "$W/src/ops_mmx.c"
if ! grep -q "__parm \[__eax\] \[__edx\] \[__ecx\]" "$W/src/ops_mmx.c"; then
    echo "aux_parm_arity_gate: FAIL - the mutation did not apply, so the"
    echo "  green above is unproven"
    exit 1
fi

echo "-- mutation (spare register at the head of a two-arg __parm) --"
if run_check "$W" > "$W/mut.log" 2>&1; then
    cat "$W/mut.log"
    echo "aux_parm_arity_gate: FAIL - the planted defect did NOT redden the"
    echo "  gate; it is not checking what it claims to"
    exit 1
fi
grep -m1 "FAIL lz_dot32_t2_asm" "$W/mut.log" | sed 's/^/  /'
echo "aux_parm_arity_gate: PASS (mutation caught)"
exit 0
