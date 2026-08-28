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
# NOTHING IS FILTERED OUT, only CLASSIFIED. Earlier versions of this file
# suppressed each pattern below as it was understood, and that is the
# instrument deciding for the reader which untidiness is acceptable -
# which is not its call. Every axis now prints its explained cases under
# "explained, listed anyway" with the reason attached, so a standard
# stricter than this file's applies to the whole list rather than to a
# pre-filtered one. A clean run means zero UNEXPLAINED, not zero output.
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
#   - A PER-ISA TU, by construction. Names here are
#     lz_<op>_<format>_<isa>[_asm]: the file stem carries the ISA field
#     and the family prefix carries the operator field, so ops_arm.c,
#     ops_sse2.c and ops_mmx.c can never share a token with their own
#     dominant family however the test is written. Counting declarations
#     as family spread (above) is what lets the >= 3 files escape fire
#     for them instead; if a future ops_<isa>.c appears here anyway,
#     that is the reason, not a defect in the file.
#   - A VERB PREFIX IS NOT A TOPIC, and a file organised by CONCERN will
#     spread one noun across several. gui/main.c runs menu, status bar,
#     layout, text emission, jobs, commands, wndproc, selftest in that
#     order; apply_settings sits in the settings stretch and apply_font
#     in the font one, which is right, and this axis scores the pair as
#     apply* resuming. Its four hits there are all of that shape.
#
# WHERE IT STANDS, so a future reader knows what today's output means:
# ORDER 6 unexplained across 3 files, PLACEMENT 0, NAMING 2. The first
# version of ORDER reported 103 across 25 files and ranked generate.c
# third; every one of those differences was this instrument being wrong
# rather than the tree changing.
#
# ALL SIX NAMING HITS HAVE BEEN JUDGED, one subagent per file, each
# reading only its own. Five were KEEP and the reasons are now rules
# above: common/stream.c exports six lz_stream_* and its ten scan_* are
# static arms of one dispatch chain; forward_kda.c is 950 of 1156 lines
# KDA and lost the vote only because `kda` was too short to be a family;
# ops_arm.c is 17-for-17 ARMv5TE; safetensors.c owns lz_st_ exclusively,
# 7 of 7; lz_softfp.c holds __aeabi_* names the ARM EABI fixes and gcc
# emits by name, which no filename could ever match. The sixth,
# cli_argv.c, was RENAME-FUNCTIONS: lz_gbk_argv_convert claims membership
# in llama98.def's exported GBK transcoding group and is neither
# exported nor allocation-free.
#
# The two that remain are the two that cannot be matched mechanically -
# an acronym and an externally mandated namespace - and they are left
# printing rather than suppressed, because a reader accepting them is
# not the same as this file deciding for them.
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
# `__attribute__((naked))` and friends sit between the type and the name,
# and \w\s\* cannot cross their parentheses: all nine naked entry points
# in src/lz_softfp.c were invisible, so a 17-function file was judged on
# a sample of 7. The alternation is spelled out rather than made general
# because a permissive prefix here starts matching call statements.
DEF = re.compile(r"^[A-Za-z_][\w\s\*]*?"
                 r"(?:__attribute__\s*\(\([^)]*\)\)\s*)?"
                 r"[\w\s\*]*?\b([a-z_]\w*)\s*\([^;{]*\)\s*\{", re.M)
# A DECLARATION is family membership too, and only definitions were
# counted. The x86 twins of lz_dot32_* are #pragma aux externs
# (src/ops_mmx.c) or intrinsics in ops_kernel_dot_*.h, so the family
# looked like 8 in ops_arm.c and 1 in ops_sse2.c - which defeated the
# "a family in >= 3 files says nothing about this file" escape and let
# ops_arm.c be reported for holding the ARM tier of a per-ISA family.
DECL = re.compile(r"^[A-Za-z_][\w\s\*]*?\b([a-z_]\w*)\s*\([^;{]*\)\s*;", re.M)

FILES = sorted(glob.glob("src/*.c") + glob.glob("gui/*.c") +
               glob.glob("common/*.c"))

order = {}          # file -> [function names, in definition order]
bodies = {}         # file -> {name: body text}
arm = {}            # file -> {name: which preprocessor branch it is in}
static_fn = set()   # (name, file) pairs with internal linkage
for f in FILES:
    t = strip(open(f, encoding="utf-8", errors="replace").read())
    # PARALLEL ARMS ARE NOT A SEQUENCE. src/ops_mmx.c is
    # #if defined(__WATCOMC__) ... #else /* gcc: intrinsics */ ... #endif
    # and each arm walks the same weight formats in the same order, so one
    # sequence over the whole file sees p2 resume at the bottom and calls
    # a perfectly ordered file tangled. Two arms of one conditional are
    # alternative texts; nobody reads them in succession.
    #
    # The full nesting PATH, not the depth: that file wraps everything in
    # #if LZ_MMX_TU, so its __WATCOMC__ conditional sits at depth two and
    # a depth-one test saw neither arm. Two functions are comparable only
    # when every conditional open at both of them is on the same branch.
    armat, stack, nextid = [], [], 0
    for line in t.split("\n"):
        s = line.lstrip()
        if re.match(r"#\s*if", s):
            stack.append([nextid, 0]); nextid += 1
        elif re.match(r"#\s*(else|elif)", s) and stack:
            stack[-1][1] += 1
        elif re.match(r"#\s*endif", s) and stack:
            stack.pop()
        armat.append(tuple(tuple(x) for x in stack))
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
        if m.group(0).lstrip().startswith("static"):
            static_fn.add((n, f))
        ln = t[:m.start()].count("\n")
        arm.setdefault(f, {})[n] = armat[ln] if ln < len(armat) else ()
    order[f] = names
    bodies[f] = bd
    arm.setdefault(f, {})

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
        # len >= 3, not >= 4. `kda` has six members, `gdn` twelve and
        # `epi` nineteen, and all three were invisible as families - so
        # forward_kda.c, 950 of whose 1156 lines are KDA, was judged to
        # be a subn_* file 4/7. This is the same off-by-one already fixed
        # for the stem tokeniser below ("len >= 2, not > 2, so t2 is not
        # thrown away"), never applied to the family KEY. Not 2: `lz`
        # would then become everyone's last-resort family.
        if count[p] >= 3 and len(p) >= 3:
            family[(n, f)] = p
            break

famfiles = collections.defaultdict(collections.Counter)
for (n, f), p in family.items():
    famfiles[p][os.path.basename(f)] += 1

# DECLARATIONS COUNT TOWARDS A FAMILY'S SPREAD, and only definitions did.
# A per-ISA family's members are not all C bodies: the x86 twins of
# lz_dot32_* are `extern` declarations followed by #pragma aux
# (src/ops_mmx.c) or static inline intrinsics in ops_kernel_dot_*.h, and
# neither matches DEF. lz_dot32 therefore looked like {ops_arm.c: 8,
# ops_sse2.c: 1} instead of five files, which defeated the "a family
# spread over >= 3 files says nothing about THIS file" escape and let
# ops_arm.c be reported for holding the ARM tier of a per-ISA family.
# Spread only - a declaration does not make the file its owner, so this
# does not feed `order`, ORDER or the NAMING counts.
for f in sorted(glob.glob("src/*.h") + glob.glob("gui/*.h") +
                glob.glob("common/*.h") + FILES):
    t = strip(open(f, encoding="utf-8", errors="replace").read())
    for m in DECL.finditer(t):
        n = m.group(1)
        if n in KEYWORDS:
            continue
        for p in prefixes(n):
            if count[p] >= 3 and len(p) >= 3:
                if os.path.basename(f) not in famfiles[p]:
                    famfiles[p][os.path.basename(f)] = 0
                break

# SOME FAMILIES ARE NAMED BY SUFFIX, NOT PREFIX, and PLACEMENT read only
# the prefix. ops_sched.c is the knob TU and its members are named for
# the operator they control - lz_attn_mode, lz_sig_mode, lz_conv_mode,
# lz_gdn_mode, lz_epi_mode, lz_bf16_store_select - so the family is the
# _mode/_select/_tier ending. Measured: 8 of 8 *_select in the tree are
# in that file, 9 of 11 *_mode, and the only *_sel. Reading the prefix
# alone reported three members of a fourteen-member family that lives
# together as the odd ones out, and moving them would have REDUCED the
# concentration the axis exists to measure.
suffix = {}
suffixfiles = collections.defaultdict(collections.Counter)
for n, f in allnames:
    if "_" not in n:
        continue
    s = n.rsplit("_", 1)[1]
    if len(s) < 3:
        continue
    suffix[(n, f)] = s
    suffixfiles[s][os.path.basename(f)] += 1

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
    """Every topic resumption, CLASSIFIED rather than filtered.

    Returns (unexplained, explained) as lists of (topic, reason). Nothing
    is dropped: a filter decides for the reader which patterns are
    acceptable, and that is the reader's call. Each entry carries why it
    is where it is, so a standard stricter than this file's can be
    applied to the same list.

    local-helper is the one large category. A helper defined immediately
    before its only caller is what lets a file have no forward
    declarations - sample_residual between lz_spec_accept and
    lz_spec_accept_temp, verify_row_argmax before lz_spec_round - and it
    reads as a re-entry because the helper's name puts it in another
    family. 93 of 103 tree-wide are this."""
    seen, last, bad, ok, runstart = set(), None, [], [], None
    for k, s in enumerate(seq):
        if s != last:
            if s in seen and runstart is not None and names is not None:
                lo = idx[runstart]
                inter = [names[j] for j in range(lo, idx[k])]
                local = all(
                    callers[x] and all(pos[y] >= lo for y in callers[x])
                    for x in inter if callers.get(x))
                if inter and local:
                    ok.append((s, "local-helper"))
                else:
                    bad.append((s, ""))
            elif s in seen and names is None:
                bad.append((s, ""))
            seen.add(s)
            last = s
            runstart = k
    return bad, ok

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
    # Per preprocessor branch, then summed: see the arm comment above.
    bad_f, ok_f = [], []
    bad_m, ok_m = [], []
    arms = sorted(set(arm[f].values()))
    for br in arms:
        sel = [i for i, n in enumerate(names) if arm[f].get(n, ()) == br]
        fam_pairs = [(i, family.get((names[i], f)))
                     for i in sel if family.get((names[i], f))]
        fmt_pairs = [(i, fmt_key(names[i])) for i in sel if fmt_key(names[i])]
        x, y = re_entries([s for _, s in fam_pairs], names, callers, pos,
                          [i for i, _ in fam_pairs])
        bad_f += x; ok_f += y
        x, y = re_entries([s for _, s in fmt_pairs], names, callers, pos,
                          [i for i, _ in fmt_pairs])
        bad_m += x; ok_m += y
    byfam = [s for s in (family.get((n, f)) for n in names) if s]
    byfmt = [s for s in (fmt_key(n) for n in names) if s]
    # Whole-file counts, so the arm split is reported and not just applied.
    whole_f, _ = re_entries(byfam, names, callers, pos,
                            [i for i, n in enumerate(names)
                             if family.get((n, f))])
    armsplit = len(whole_f) - len(bad_f) - len(ok_f)
    # Sorted under EITHER key is sorted. The by-format key is the kernel
    # TUs'; the by-family key is everything else's.
    if len(byfmt) >= 3 and len(bad_m) < len(bad_f):
        bad, ok = bad_m, ok_m
    else:
        bad, ok = bad_f, ok_f
    if bad or ok or armsplit > 0:
        rows.append((len(bad), f, len(byfam), bad,
                     len(ok), max(0, armsplit), len(arms)))

rows.sort(reverse=True)
for c, fn, placed, bad, nok, narm, narms in rows[:20]:
    ws = collections.Counter(s for s, _ in bad).most_common(3)
    extra = []
    if nok:  extra.append("%d local-helper" % nok)
    if narm: extra.append("%d across %d preprocessor arm(s)" % (narm, narms))
    print("  %3d unexplained  %-24s %3d placed  %-26s %s"
          % (c, fn, placed, ", ".join("%s x%d" % w for w in ws) or "-",
             "[" + "; ".join(extra) + "]" if extra else ""))
print("  %d unexplained across %d file(s); %d explained, listed above so"
      " that a stricter\n  standard than this file's can be applied to"
      " the same list rather than to a\n  filtered one"
      % (sum(r[0] for r in rows), len([r for r in rows if r[0]]),
         sum(r[4] + r[5] for r in rows)))

# ---- 2. placement: the odd family member -----------------------------
print("\n== PLACEMENT: family lives mostly elsewhere ==")
hits = 0
soft = []
for (n, f), p in sorted(family.items()):
    here = os.path.basename(f)
    home, homecount = famfiles[p].most_common(1)[0]
    total = sum(famfiles[p].values())
    if here != home and famfiles[p][here] * 4 <= homecount and total >= 4:
        # CLASSIFIED, NOT FILTERED. A static function cannot be called
        # from another file, so a move would have to make it extern
        # first - which is an argument, not an exemption, and the reader
        # gets to hear it. Each of the three that landed here is
        # correctly placed on inspection. The three that were here have
        # since been RENAMED rather than moved, which is why they no
        # longer appear: lz4d_read_void -> lz4_reader (model.c's adapter
        # to model.c's own rd pointer, identified by ADDRESS in
        # m_uses_lz4), utf8_argv -> win32_utf8_argv (reads the command
        # line, a CLI concern), utf8_extra -> gbk_utf8_extra (gbk.c's own
        # five-line lead-byte predicate). Each had shared a prefix with
        # another file's family and nothing else.
        # The SUFFIX family gets a vote. Home under either key is home.
        s = suffix.get((n, f))
        if s and suffixfiles[s]:
            shome, scount = suffixfiles[s].most_common(1)[0]
            stotal = sum(suffixfiles[s].values())
            if shome == here and stotal >= 4 and scount * 2 > stotal:
                soft.append("  %-34s in %-18s   [*_%s lives here: %d/%d]"
                            % (n, here, s, scount, stotal))
                continue
        # A PAIR IS NOT SPLIT. lz_bf16_store is the getter for the knob
        # lz_bf16_store_select sets, and that one is home here by the
        # suffix vote above; moving the getter to lz_bf16.c would put a
        # knob's two halves in two files, which is worse than either
        # placement the axis was choosing between. A sibling sharing a
        # LONGER prefix than the family key, in this same file, is that
        # relationship.
        sib = [m for (m, g) in family
               if g == f and m != n and m.startswith(n + "_")]
        if sib:
            soft.append("  %-34s in %-18s   [paired with %s, here]"
                        % (n, here, sorted(sib)[0]))
            continue
        line = ("  %-34s in %-18s but %d/%d of %s* is in %s"
                % (n, here, homecount, total, p, home))
        if (n, f) in static_fn:
            soft.append(line + "   [static: a move needs extern first]")
        elif len(famfiles[p]) >= 3:
            soft.append(line + "   [family spans %d files: per-tier, no"
                        " one home]" % len(famfiles[p]))
        else:
            print(line)
            hits += 1
if not hits:
    print("  none unexplained")
if soft:
    print("  -- explained, listed anyway --")
    for s in soft:
        print(s)

# ---- 3. naming: the file's stem against its dominant family ----------
print("\n== NAMING: the file's name against what it mostly holds ==")
hits = 0
soft2 = []
for f in FILES:
    # EXPORTED FUNCTIONS ONLY. A file's name promises what it OFFERS, and
    # a static helper is offered to nobody: common/stream.c exports six
    # lz_stream_* and holds ten static scan_* arms of one dispatch chain,
    # and counting all sixteen equally reported the file as misnamed for
    # its own internals. PLACEMENT already excludes statics for the
    # neighbouring reason; NAMING did not.
    fams = collections.Counter(family.get((n, f)) for n in order[f]
                               if (n, f) not in static_fn)
    fams.pop(None, None)
    if not fams:
        continue
    dom, dc = fams.most_common(1)[0]
    tot = sum(fams.values())
    # <=, not <: an exact tie is not a dominant topic. ops_arm.c's 8/16
    # was a plurality against seven other families, admitted as
    # "dominant" by a strict comparison.
    if dc * 2 <= tot:
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
        # An ABBREVIATION is a match this has no dictionary for, so it is
        # labelled rather than dropped: safetensors.c holding lz_st* is
        # the file saying exactly what it holds. The test is whether a
        # family token is a PREFIX of a stem token - st/safetensors, ini/
        # inifile - which catches the mechanical shortenings and none of
        # the semantic ones (lz_softfp.c holding __aeabi* stays
        # unexplained, and correctly: nothing in the name says EABI).
        bare = [y for y in fam_parts if y != "lz"]
        abbrev = any(x.startswith(y) and len(y) >= 2
                     for x in parts for y in bare)
        if abbrev:
            soft2.append("  %-22s %d/%d are %s*   [abbreviation]"
                         % (os.path.basename(f), dc, tot, dom))
            continue
        if tot < 3:
            soft2.append("  %-22s %d/%d are %s*   [%d placed function(s):"
                         " too few to have a dominant topic]"
                         % (os.path.basename(f), dc, tot, dom, tot))
            continue
        print("  %-22s %d/%d of its placed functions are %s*"
              % (os.path.basename(f), dc, tot, dom))
        hits += 1
if not hits:
    print("  none unexplained")
if soft2:
    print("  -- explained, listed anyway --")
    for s2 in soft2:
        print(s2)
PYEOF
exit 0
