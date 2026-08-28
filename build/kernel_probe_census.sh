#!/bin/bash
# Which hand-written kernels has NOTHING ever run?
#
# A SIMD kernel is written twice - assembly and intrinsics - and the two
# must agree bit for bit. That claim is worth exactly what the check
# behind it is,
# and the checks are scattered across build/*_probe.c, .prof/*_xcheck.c
# and the end-to-end gates. Nothing added them up, so the answer was
# reached by remembering - and remembering got it wrong twice in one
# session, once naming a kernel as uncovered whose probe had been sitting
# in .prof/ the whole time.
#
# COVERED MEANS REACHED, NOT MENTIONED, and the difference is three
# things that have each bitten this tree:
#
#   comments and strings   every probe explains which kernel it is for,
#                          and prints its name. Both are stripped.
#   transitivity           a probe calls the OPERATOR; the operator picks
#                          a tier; the tier wrapper calls the #pragma aux
#                          body. Following one level called
#                          lz_fwht_stage_f32_sse_asm uncovered when
#                          .prof/fwht_f32_xcheck.c drives it three hops
#                          up. This walks the whole call graph.
#   #pragma aux            has no address: it is emitted at its call site
#                          and can only be reached through a named caller
#                          in its own TU. A pragma nothing calls is
#                          unreachable by any probe, whatever any probe
#                          says about it, and gets its own column.
#
# WHAT THIS DOES NOT ESTABLISH. Reached is not checked - a probe can call
# a kernel and compare nothing that would catch a defect in it. So the
# covered count is an UPPER bound and the uncovered list is the part to
# act on: those cannot be checked by anything, whatever the probes claim.
#
# THE ARM HALF ANSWERS THE SAME QUESTION for the other toolchain: a
# hand-written ARMv5TE kernel is a function whose body holds an __asm__
# block, no x86 build contains one, and the analogue of "built with
# Watcom" is "cross-built and run under QEMU". 36 of 36 today. Three
# edges had to exist before that number meant anything - a macro naming
# a macro, an ALL-CAPS macro INVOCATION, and a function whose address is
# taken in a ternary dispatch (`row = cond ? row_q41_arm_asm :
# row_q41_arm`) - and each was found by a kernel this file wrongly
# called uncovered, then covered for real rather than by argument.
#
# WHERE IT STANDS: 72 of 74 reached by a Watcom-built probe, 0 reached
# only on paper, 0 called by nothing. The two left are lz_cpucheck_brand
# and lz_cpucheck_maxext, which read the CPUID brand string - they have
# no second implementation to disagree with and no value a reference
# could predict, so they are the floor rather than a gap. Every kernel
# that computes something a twin also computes now has a driver that
# builds both halves and compares them.
#
# Exit: 0 always. This is a census, not a gate.
set -u
cd "$(dirname "$0")/.."

PY=python; command -v python >/dev/null 2>&1 || PY=python3
"$PY" - "$@" <<'PYEOF'
import re, os, glob, sys

def strip(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    s = re.sub(r'"(?:[^"\\]|\\.)*"', ' "" ', s)
    return s

SRC = sorted(glob.glob("src/*.c") + glob.glob("src/*.h"))
PROBES = sorted(glob.glob("build/*probe*.c") + glob.glob(".prof/*xcheck*.c"))

# WHICH PROBES ARE BUILT WITH WATCOM, which for a #pragma aux kernel is
# the only build that contains it at all: gcc compiles no pragma body,
# so a gcc-only probe "reaches" such a kernel on paper and executes
# nothing. That is not a hypothetical - lz_gdn1_x4_asm accumulated into
# an uninitialised mm7 while two probes drove lz_gdn_step straight
# through it, because both drivers build with $CC and neither has a
# Watcom arm.
#
# A .prof/X_xcheck.c counts if .prof/X_xcheck.sh runs wbuild_xcheck.sh;
# a build/*_probe.c counts if some gate or driver names it next to
# wbuild_xcheck.sh. Measured today: 4 of 15 xcheck drivers have that arm.
WATCOM = set()
drivers = glob.glob(".prof/*.sh") + glob.glob("build/*.sh")
for d in drivers:
    t = open(d, encoding="utf-8", errors="replace").read()
    if "wbuild_xcheck" not in t:
        continue
    stem = os.path.basename(d)[:-3]
    for p in PROBES:
        b = os.path.basename(p)[:-2]
        if b == stem or b in t:
            WATCOM.add(p)

KEYWORDS = set("if for while switch return sizeof do else case defined "
               "static inline void int float double char unsigned signed "
               "const struct union enum typedef extern".split())

# ---- kernels ----------------------------------------------------------
# From the STRIPPED text. Scanning the raw file for "#pragma aux NAME"
# picked up every comment that says "the #pragma aux twin", and invented
# kernels called `twin`, `above`, `says` and `with` - which then showed
# up in the uncalled column, where an invented name always will.
kernels = {}
for f in SRC:
    t = strip(open(f, encoding="utf-8", errors="replace").read())
    for m in re.finditer(r"#pragma\s+aux\s+([A-Za-z_]\w*)", t):
        kernels[m.group(1)] = (os.path.basename(f), "pragma")

# THE ARM HALF. A hand-written ARMv5TE kernel is a function whose body
# contains an __asm__ block, and it has the same property the pragmas
# do for the opposite toolchain: no x86 build contains one, so an x86
# probe reaches it on paper and executes nothing. The ARM analogue of
# "built with Watcom" is "built with the cross-compiler and run under
# QEMU", which is what build/arm/*.sh and the .prof/arm_* drivers do.
armk = {}

# ---- call graph -------------------------------------------------------
# A function definition is a header line followed by a body; take the body
# by brace balance. Anything that looks like `name(` inside it is a call.
calls = {}          # caller -> set(callee)
allbodies = {}      # caller -> its body text, for the address pass
owner = {}          # function -> file
DEF = re.compile(r"^[A-Za-z_][\w\s\*]*?\b([a-z_]\w*)\s*\([^;{]*\)\s*\{", re.M)
for f in SRC:
    t = strip(open(f, encoding="utf-8", errors="replace").read())
    for m in DEF.finditer(t):
        name = m.group(1)
        if name in KEYWORDS:
            continue
        i = t.index("{", m.start())
        depth, j = 0, i
        while j < len(t):
            if t[j] == "{": depth += 1
            elif t[j] == "}":
                depth -= 1
                if depth == 0: break
            j += 1
        bodytext = t[i:j]
        if "__asm__" in bodytext:
            armk[name] = (os.path.basename(f), "arm-asm")
        owner.setdefault(name, os.path.basename(f))
        s = calls.setdefault(name, set())
        for c in re.finditer(r"\b([a-z_]\w*)\s*\(", bodytext):
            if c.group(1) not in KEYWORDS:
                s.add(c.group(1))
        # A #pragma aux body is emitted here, so naming it is calling it.
        for c in re.finditer(r"\b([a-z_]\w*)\b", bodytext):
            if c.group(1) in kernels:
                s.add(c.group(1))
        # A KERNEL PASSED TO A MACRO is called by it, and the name is not
        # followed by '(' at the call site: row_q41_sse2_asm reaches
        # dot32_q41_sse2a only as LZ_Q41_PFSEL(dot32_q41_sse2a). Missing
        # that edge reported a shipping SSE2 dot kernel as unprobed while
        # .prof/rowdot_xcheck.c drives it. Restricted to arguments of an
        # ALL-CAPS macro so it does not turn every mention into an edge.
        for mm in re.finditer(r"\b([A-Z][A-Z0-9_]{2,})\s*\(([^;]*?)\)",
                              bodytext):
            # The MACRO ITSELF is a call edge, not only its arguments.
            # Recording only the arguments broke the chain
            # row_q41_sse2_asm -> LZ_Q41_PFSEL -> LZ_PFI_NTA -> lz_pf_nta
            # at the first hop, because the body scan above matches
            # `name(` only for a lower-case name - so the three prefetch
            # pragmas read as reached by nothing while
            # build/prefetch_parity_probe.c drives all four modes.
            s.add(mm.group(1))
            for c in re.finditer(r"\b([a-z_]\w*)\b", mm.group(2)):
                s.add(c.group(1))
        allbodies[name] = bodytext

# TAKING A FUNCTION'S ADDRESS IS A CALL EDGE, and it needs its own pass
# because the name table is only complete once every file is read. The
# tier dispatch is `row = (g_kernel == LZ_KERNEL_ARM_ASM)
# ? row_q41_arm_asm : row_q41_arm;` - a ternary, so the callee is not
# followed by '(' and neither the call scan nor the macro-argument scan
# sees it. build/arm/parity_gate.sh runs --kernel ref / arm-c / arm-asm
# in turn, so those really are executed; without this edge
# lz_dot32_q16/q41/q61_arm_asm read as reached by no ARM driver at all.
for caller, bodytext in allbodies.items():
    s = calls.setdefault(caller, set())
    for c in re.finditer(r"([a-z_]\w*)(?!\s*\()", bodytext):
        n = c.group(1)
        if n != caller and (n in owner or n in kernels or n in armk):
            s.add(n)

# MACROS ARE CALL EDGES. lz_mmx_empty_local is only ever invoked as
# _mm_empty(), and the three lz_pf_* prefetch pragmas only through their
# LZ_PF_* macros, so a graph built from function bodies alone reports
# four live kernels as called by nothing - which reads as dead code.
DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)(?:\([^)]*\))?(.*)$", re.M)
# TWICE, because a macro can name another macro. LZ_Q41_PFSEL's body
# names LZ_PFI_NTA, which is itself a #define, and on a single pass
# LZ_PFI_NTA is not in `calls` yet when PFSEL is read - so the chain
# probe -> row kernel -> PFSEL -> LZ_PFI_NTA -> lz_pf_nta broke in the
# middle and the three prefetch pragmas were reported as reached by
# nothing while build/prefetch_parity_probe.c drives all four modes
# through them.
for _pass in (0, 1):
    for f in SRC:
        t = strip(open(f, encoding="utf-8", errors="replace").read())
        # Rejoin line continuations first: a macro body split over lines
        # would otherwise lose everything after the backslash.
        t = t.replace("\\\n", " ")
        for m in DEFINE.finditer(t):
            name, repl = m.group(1), m.group(2)
            s = calls.setdefault(name, set())
            for c in re.finditer(r"\b(\w+)\b", repl):
                if c.group(1) in kernels or c.group(1) in calls:
                    if c.group(1) != name:
                        s.add(c.group(1))

# A kernel named in a dispatch-table initialiser is reachable through
# whatever consults that table; treat the enclosing file's operators as
# callers by adding table entries to every function in the same file.
TABLE = re.compile(r"=\s*\{[^;]*?\}", re.S)
for f in SRC:
    t = strip(open(f, encoding="utf-8", errors="replace").read())
    for m in TABLE.finditer(t):
        ents = [e for e in re.findall(r"\b([a-z_]\w*)\b", m.group(0))
                if e in kernels or e in calls]
        if not ents:
            continue
        for fn, fl in owner.items():
            if fl == os.path.basename(f):
                calls.setdefault(fn, set()).update(ents)

# ---- seeds: what the probes name -------------------------------------
seeds, wseeds = set(), set()
for p in PROBES:
    t = strip(open(p, encoding="utf-8", errors="replace").read())
    for m in re.finditer(r"\b([a-z_]\w*)\b", t):
        if m.group(1) not in KEYWORDS:
            seeds.add(m.group(1))
            if p in WATCOM:
                wseeds.add(m.group(1))

def closure(start):
    seen, stack = set(), list(start)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for c in calls.get(n, ()):
            if c not in seen:
                stack.append(c)
    return seen

wreach = closure(wseeds)

# ---- the ARM side -----------------------------------------------------
# Seeds are every name a driver that cross-compiles for ARM mentions,
# plus the whole engine entry point when that driver runs a real forward
# pass: build/arm/parity_gate.sh compares x86 against ARMv5TE over
# generation, so anything lz_forward reaches is executed there.
ARMDRV = [d for d in glob.glob(".prof/*.sh") + glob.glob("build/*.sh") +
          glob.glob("build/arm/*.sh")
          if re.search(r"arm-linux-gnueabi|qemu-arm",
                       open(d, encoding="utf-8", errors="replace").read())]
aseeds = set()
for d in ARMDRV:
    t = strip(open(d, encoding="utf-8", errors="replace").read())
    for m in re.finditer(r"\b([a-z_]\w*)\b", t):
        aseeds.add(m.group(1))
    # A driver that builds and RUNS the engine reaches whatever the
    # forward pass reaches; naming the entry point is what says so.
    if re.search(r"parity_gate|build_engine|run_qemu", d):
        aseeds |= {"lz_forward", "lz_generate", "lz_mtp_draft_step"}
    # A .prof driver's own probe source is where its names live.
    stem = os.path.basename(d)[:-3]
    for c in (".prof/%s.c" % stem, "build/%s.c" % stem):
        if os.path.exists(c):
            ct = strip(open(c, encoding="utf-8", errors="replace").read())
            for m in re.finditer(r"\b([a-z_]\w*)\b", ct):
                aseeds.add(m.group(1))
areach = closure(aseeds)

# ---- closure ----------------------------------------------------------
reach, stack = set(), [s for s in seeds]
while stack:
    n = stack.pop()
    if n in reach:
        continue
    reach.add(n)
    for c in calls.get(n, ()):
        if c not in reach:
            stack.append(c)

called_by_something = set()
for s in calls.values():
    called_by_something |= s

rows = []
for k, (f, kind) in sorted(kernels.items()):
    if k not in called_by_something:
        rows.append(("UNCALLED", k, f))
    elif k in wreach:
        rows.append(("reached", k, f))
    elif k in reach:
        # On paper only. gcc compiles no #pragma aux body, so a probe
        # that is never built with Watcom executes nothing here.
        rows.append(("GCCONLY", k, f))
    else:
        rows.append(("GAP", k, f))

cov = [r for r in rows if r[0] == "reached"]
gcconly = [r for r in rows if r[0] == "GCCONLY"]
gap = [r for r in rows if r[0] == "GAP"]
dead = [r for r in rows if r[0] == "UNCALLED"]

print("%d #pragma aux kernel(s) in src/, %d probe source(s), %d of them"
      " built with Watcom" % (len(rows), len(PROBES), len(WATCOM)))
print("  a WATCOM-built probe reaches : %d" % len(cov))
print("  only a gcc-built one does    : %d" % len(gcconly))
print("  no probe reaches             : %d" % len(gap))
print("  nothing calls at all         : %d" % len(dead))
if gcconly:
    print("\n-- REACHED ON PAPER ONLY: no Watcom build of any probe runs"
          " these --")
    for _, k, f in gcconly:
        who = sorted(c for c in calls if k in calls[c])[:2]
        print("  %-32s %-16s via: %s" % (k, f, ", ".join(who) or "(none)"))
if gap:
    print("\n-- NO PROBE REACHES THESE --")
    for _, k, f in gap:
        who = sorted(c for c in calls if k in calls[c])[:3]
        print("  %-32s %-16s via: %s" % (k, f, ", ".join(who) or "(none)"))
if dead:
    print("\n-- called by nothing: unreachable by construction --")
    for _, k, f in dead:
        print("  %-32s %s" % (k, f))

# ---- ARM report -------------------------------------------------------
acov = sorted(k for k in armk if k in areach)
agap = sorted(k for k in armk if k not in areach)
print("\n%d ARM inline-asm kernel(s), %d driver(s) cross-build for ARM"
      % (len(armk), len(ARMDRV)))
print("  an ARM-built driver reaches : %d" % len(acov))
print("  no ARM driver reaches       : %d" % len(agap))
if agap:
    print("\n-- NO ARM DRIVER REACHES THESE --")
    for k in agap:
        who = sorted(c for c in calls if k in calls[c])[:2]
        print("  %-32s %-18s via: %s"
              % (k, armk[k][0], ", ".join(who) or "(none)"))
if "-v" in sys.argv:
    print("\n-- reached --")
    for _, k, f in cov:
        print("  %-32s %s" % (k, f))
PYEOF
exit 0
