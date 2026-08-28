#!/bin/bash
# Structural organisation, measured: is a file's ORDER logical, does a
# function live in the right file, does a file's NAME say what it holds.
#
# build/nest_census.sh already ranks functions by how tangled their
# control flow is. That is one axis and not the one that makes a tree
# hard to navigate. These are the other three, and each is stated as
# something countable rather than as taste:
#
#   ORDER        a file that defines topic A, then topic B, then topic A
#                again forces a reader to hold two threads. The count is
#                RE-ENTRIES: how many times a topic resumes after another
#                topic intervened. Zero means the file is sorted by
#                topic, whatever order the topics come in.
#
#   PLACEMENT    a function belongs with its family. Family is the name
#                prefix it shares with others (lz_dot32_, row_q41_,
#                lz_epi_). If most of a function's family lives in
#                another file, this one is the odd member.
#
#   NAMING       a file's name should name what it mostly holds. If the
#                file's stem appears in no part of its dominant family's
#                prefix, the name is not describing the contents.
#
# WHAT IT CANNOT SEE, stated because a census that only prints hits reads
# as if it looked everywhere:
#   - a topic the naming does not encode. Two unrelated operators that
#     both start lz_ look like one family here.
#   - a function in the right file at the wrong ABSTRACTION level.
#   - headers. Only .c files are ordered documents; a header is an
#     interface and its order answers to the interface.
#   - deliberate placement. ops_mmx.c holds row kernels AND dot kernels
#     on purpose: a #pragma aux body has no address and has to be called
#     from its own TU. Hits are questions, not defects.
#   - ABBREVIATION. This tree shortens consistently - safetensors.c holds
#     lz_st*, lz_softfp.c holds the __aeabi* soft-float entry points -
#     and the NAMING axis reads both as mismatches. It has no dictionary
#     and cannot get those right; treat that column as the weakest.
#   - ORGANISING BY ROLE rather than by family. ops_sched.c holds
#     lz_epi_mode and lz_bf16_store_select because it is the knob TU, not
#     because they were misfiled. PLACEMENT cannot tell the two apart,
#     and a per-tier family (lz_dot32_* exists once per ISA) has no one
#     home for it to compare against.
#
# Exit: 0 always. This is a census.
set -u
cd "$(dirname "$0")/.."

PY=python; command -v python >/dev/null 2>&1 || PY=python3
"$PY" - "$@" <<'PYEOF'
import re, os, glob, sys, collections

def strip(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    return s

KEYWORDS = set("if for while switch return sizeof do else case defined "
               "static inline void int float double char unsigned signed "
               "const struct union enum typedef extern sizeof".split())
DEF = re.compile(r"^[A-Za-z_][\w\s\*]*?\b([a-z_]\w*)\s*\([^;{]*\)\s*\{", re.M)

FILES = sorted(glob.glob("src/*.c") + glob.glob("gui/*.c") +
               glob.glob("common/*.c"))

order = {}          # file -> [function names, in definition order]
bodies = {}         # file -> {name: body text}
for f in FILES:
    t = strip(open(f, encoding="utf-8", errors="replace").read())
    names, bd = [], {}
    for m in DEF.finditer(t):
        n = m.group(1)
        if n in KEYWORDS:
            continue
        i = t.index("{", m.start())
        depth, j = 0, i
        while j < len(t):
            if t[j] == "{": depth += 1
            elif t[j] == "}":
                depth -= 1
                if depth == 0: break
            j += 1
        names.append(n)
        bd[n] = t[i:j]
    order[f] = names
    bodies[f] = bd

# ---- families ---------------------------------------------------------
# The longest underscore-delimited prefix a name shares with at least two
# others, tree-wide. One-off names have no family and are not judged.
allnames = [(n, f) for f in FILES for n in order[f]]
def prefixes(n):
    p = n.split("_")
    return ["_".join(p[:i]) for i in range(len(p) - 1, 0, -1)]

count = collections.Counter()
for n, _ in allnames:
    for p in prefixes(n):
        count[p] += 1

family = {}
for n, f in allnames:
    for p in prefixes(n):
        if count[p] >= 3 and len(p) >= 4:
            family[(n, f)] = p
            break

famfiles = collections.defaultdict(collections.Counter)
for (n, f), p in family.items():
    famfiles[p][os.path.basename(f)] += 1

# ---- 1. order: topic re-entries --------------------------------------
# TWO GROUPINGS, and a file is only disordered if BOTH say so. The kernel
# files are grouped by weight FORMAT - q8's dot32, dot128 and row
# together, then q41's - which under a by-kind key looks like lz_dot128
# resuming four times. That is a sorted file read with the wrong key, and
# scoring it as tangled would have sent someone to reorder a file that is
# already organised.
FORMATS = ("q8", "q41", "q61", "q16", "t2", "x16", "f32", "p2")
def fmt_key(n):
    for x in FORMATS:
        if re.search(r"_" + x + r"(_|\b)", n):
            return x
    return None

def re_entries(seq, names=None, callers=None, pos=None, idx=None):
    """Count topic resumptions, EXCLUDING local-helper ordering.

    A helper defined immediately before its only caller is good C, not
    tangle: it is what lets the file have no forward declarations. It
    reads as a re-entry because the helper's name puts it in a different
    family - sample_residual between lz_spec_accept and
    lz_spec_accept_temp, verify_row_argmax before lz_spec_round.
    Measured over the whole tree, 93 of 103 re-entries are this, and
    without the filter this axis put generate.c third and would have sent
    someone to reorder a file that is already ordered.

    So an intervening run is forgiven when every function in it is called
    only from at or after the point it was introduced."""
    seen, last, c, where, runstart = set(), None, 0, [], None
    for k, s in enumerate(seq):
        if s != last:
            if s in seen and runstart is not None and names is not None:
                lo = idx[runstart]
                inter = [names[j] for j in range(lo, idx[k])]
                local = all(
                    callers[x] and all(pos[y] >= lo for y in callers[x])
                    for x in inter if callers.get(x))
                if not (inter and local):
                    c += 1
                    where.append(s)
            elif s in seen and names is None:
                c += 1
                where.append(s)
            seen.add(s)
            last = s
            runstart = k
    return c, where

print("== ORDER: a topic that resumes after another topic intervened ==")
rows = []
for f in FILES:
    names = order[f]
    pos = {n: i for i, n in enumerate(names)}
    callers = collections.defaultdict(set)
    for n in names:
        for c in re.finditer(r"\b([a-z_]\w*)\s*\(", bodies[f][n]):
            if c.group(1) in pos:
                callers[c.group(1)].add(n)
    fam_pairs = [(i, family.get((n, f)))
                 for i, n in enumerate(names) if family.get((n, f))]
    fmt_pairs = [(i, fmt_key(n)) for i, n in enumerate(names) if fmt_key(n)]
    byfam = [s for _, s in fam_pairs]
    byfmt = [s for _, s in fmt_pairs]
    a, wa = re_entries(byfam, names, callers, pos, [i for i, _ in fam_pairs])
    b, wb = re_entries(byfmt, names, callers, pos, [i for i, _ in fmt_pairs])
    # If the file is sorted under EITHER key it is sorted; report the
    # smaller count, and only when both keys agree there is interleaving.
    if a and (b or len(byfmt) < 3):
        n_, w_ = (a, wa) if a <= b or not b else (b, wb)
        rows.append((n_, os.path.basename(f), len(byfam), w_))
for r in sorted(rows, reverse=True)[:12]:
    ws = collections.Counter(r[3]).most_common(3)
    print("  %3d re-entries  %-22s %3d placed function(s)  %s"
          % (r[0], r[1], r[2], ", ".join("%s x%d" % w for w in ws)))
print("  %d of %d .c file(s) have at least one" % (len(rows), len(FILES)))

# ---- 2. placement: the odd family member -----------------------------
print("\n== PLACEMENT: family lives mostly elsewhere ==")
hits = 0
for (n, f), p in sorted(family.items()):
    here = os.path.basename(f)
    home, homecount = famfiles[p].most_common(1)[0]
    total = sum(famfiles[p].values())
    if here != home and famfiles[p][here] * 4 <= homecount and total >= 4:
        print("  %-34s in %-18s but %d/%d of %s* is in %s"
              % (n, here, homecount, total, p, home))
        hits += 1
if not hits:
    print("  none")

# ---- 3. naming: the file's stem against its dominant family ----------
print("\n== NAMING: the file's name against what it mostly holds ==")
hits = 0
for f in FILES:
    fams = collections.Counter(family.get((n, f)) for n in order[f])
    fams.pop(None, None)
    if not fams:
        continue
    dom, dc = fams.most_common(1)[0]
    tot = sum(fams.values())
    if dc * 2 < tot:
        continue                      # no dominant topic: nothing to claim
    # A family spread over many files - lz_gui, lz_common - says nothing
    # about THIS file, so a mismatch against it is not evidence. It named
    # aboutdlg.c and layout.c as misnamed for holding lz_gui*, which is
    # what every file in gui/ holds.
    if len(famfiles[dom]) >= 3:
        continue
    # BOTH DIRECTIONS. Testing only "is the file's stem inside the family
    # name" called inifile.c/lz_ini*, modelload.c/lz_gui_model* and
    # lz_mathf.c/lz_math* misnamed - every one of which says exactly what
    # it holds. The stem and the family have to share no token either way
    # before the name is failing to describe the contents.
    # len >= 2, not > 2: dropping two-character tokens threw away "t2",
    # so ops_t2_arm.c looked unrelated to lz_dot32_t2*.
    stem = os.path.basename(f)[:-2]
    parts = [x for x in re.split(r"[_]", stem) if len(x) >= 2]
    fam_parts = [x for x in re.split(r"[_]", dom) if len(x) >= 2]
    shared = any(x in dom for x in parts) or \
             any(y in stem for y in fam_parts) or \
             any(x[:4] == y[:4] for x in parts for y in fam_parts)
    if not shared:
        print("  %-22s %d/%d of its placed functions are %s*"
              % (os.path.basename(f), dc, tot, dom))
        hits += 1
if not hits:
    print("  none")
PYEOF
exit 0
