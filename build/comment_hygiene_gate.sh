#!/bin/bash
# Comments say what is true and why, not when it changed, which commit
# changed it, or which rule says so. Four shapes this tree re-grows:
#
#   COMMIT HASHES      "(8017c83)", "the same trade 765e798 takes".
#                      git already answers "which commit"; a hash in a
#                      comment is a pointer that rots the moment the
#                      branch is rebased, and it reads as provenance
#                      while carrying none of the reasoning.
#   DATES              "Measured 2026-08-22", "the 2026-08-09 shape".
#                      A measurement's CONDITIONS (model, token count,
#                      fixture) are what let someone repeat it; the day
#                      it happened does not.
#   CHINESE            Source stays ASCII; this catches a comment
#                      slipping through. The user-visible string tables
#                      are the exception and are baselined below.
#   RULEBOOK CITES     "iron law 10", "iron law 8: a missing cell ...".
#                      A citation is not a reason: at the call site
#                      "iron law 10" says nothing, while "a 1995
#                      compiler is in the target family" says the whole
#                      thing. The rulebook can be renumbered or
#                      reorganised and every citation in the source
#                      then points at the wrong rule, silently.
#
# NOT CHECKED HERE: "overly long" and "misleading". Neither has a
# mechanical predicate, and a gate that guesses at them would fail on
# the long comments this project wants - the ones carrying a
# derivation. They stay a review question.
#
# A RATCHET, per file, for the reason c89_floor_gate is one: the
# pre-existing surface is real and a gate that reddens on all of it
# gets switched off. Baseline below is the state when this landed.
#
# Exit codes:
#   0  PASS - no file exceeds its baseline
#   1  FAIL - a file gained one, or the instrument found nothing at all
set -u
cd "$(dirname "$0")/.."

LZ_ENV=$(LZ_TREE_LOCK_NAME="$(basename "$0")" LZ_TREE_LOCK_WHAT="the comment hygiene ratchet" bash build/tree_lock.sh acquire --print-env) || exit 2
eval "$LZ_ENV"
trap 'bash build/tree_lock.sh release' EXIT

# TWO SPELLINGS, one predicate. This matched "iron law" only, and six
# citations in src/ops_mmx.c spelled it "rule 6 item 6" / "rule 6.6" and
# went straight through - the phrase the rulebook happens to use is not
# the thing being banned. What is banned is a citation standing in for a
# reason, whatever it is called.
# `rule N` needs the digit: "the rule is that modify tells the truth" is
# a sentence, not a cite, and must not redden.
RULECITE='iron law|\brule [0-9]'

# WHAT IS SCANNED. src/ is not the tree. Three of this session's
# findings - a date, a rulebook cite, a dead flag name - were in the
# Makefile, and every earlier report of "zero tree-wide" was a src/-only
# figure presented as a whole-tree one. The build files carry as much
# prose as the sources do, and gui/ is a third of the shipping product.
SCAN="Makefile src/*.c src/*.h gui/*.c gui/*.h common/*.c common/*.h"

# path:count. Only non-zero entries listed, and keyed by PATH rather
# than basename: a basename key means a future gui/model.c silently
# inherits src/model.c's allowance, which is the failure mode where a
# table keyed by name hands a newcomer someone else's excuse.
#
#   src/err.c:138            NOT comment debt: the user-visible Chinese
#   gui/localized_strings.c:100  message tables, which are the shape the
#                            project asks for - every user-visible
#                            string lives in a table, source stays ASCII
#                            everywhere else. Counted because the scan
#                            cannot tell a string from a comment;
#                            excused here rather than by weakening the
#                            scan, which would also stop catching a
#                            Chinese COMMENT in those same files.
#   The two message tables carry no CJK lines: Visual C++ 4.0 rejects raw
#   UTF-8 in a string literal, so build/utf8_escape.py rewrites every one
#   as \xNN and the source is ASCII throughout. They hold no allowance
#   either - an allowance would let a whole table come back without a
#   word, which is the shape this file exists to refuse.
#
#   src/forward.c and src/model.c were baselined at 1 each for the
#   lz_int.h include comment, which was corrected in the working tree
#   but shared a diff hunk with another session's include-block
#   reshuffle. That commit landed and both files read 0 under the gate's
#   own byte pattern, so the allowance is retired for exactly the reason
#   the message tables' was: an allowance outliving what it excused is a
#   standing permit for one Chinese comment in each of those two files.
#
#   THE TABLE IS EMPTY NOW. That is the interesting state, because an
#   empty baseline is the one that stops being exercised - nothing here
#   would notice if the lookup broke. The mutation to run when adding
#   the next entry is the one that was run when removing this one: plant
#   a CJK comment in a scratch copy of a baselined file and check it
#   reddens.
BASELINE=""

FAIL=0
TOTAL=0
for f in $SCAN; do
    [ -f "$f" ] || continue
    b=$f
    exp=0
    for e in $BASELINE; do
        [ "${e%%:*}" = "$b" ] && exp=$(echo "$e" | cut -d: -f2)
    done
    # A hash is 7-8 hex chars that is NOT all digits (1572864 and
    # 8388608 are element counts, this tree is full of them) and NOT a
    # bit pattern written as one (0084c390 is a float's smallest
    # nonzero, quoted next to its exponent field). The discriminator
    # that works on both: a commit hash in this tree's prose is
    # ALWAYS 7 chars, and a bit pattern is 8.
    h=$(grep -ohE "\b[0-9a-f]{7}\b" "$f" 2>/dev/null \
        | grep -vE "^[0-9]+$" | wc -l)
    d=$(grep -ohE "20[0-9]{2}-[0-9]{2}-[0-9]{2}" "$f" 2>/dev/null | wc -l)
    # BYTE range, not \x{4e00}: `grep -P` refuses a multibyte pattern
    # outside a UTF-8 locale and exits 2, and a `|| echo 0` then reads
    # that refusal as "no CJK found". Caught by mutation - a Chinese
    # comment planted in ops.c did not redden this gate. UTF-8 CJK
    # always starts 0xE4-0xE9, which LC_ALL=C matches as bytes on every
    # host.
    z=$(LC_ALL=C grep -c $'[\xe4-\xe9][\x80-\xbf][\x80-\xbf]' "$f" 2>/dev/null)
    [ -n "$z" ] || z=0
    r=$(grep -ciE "$RULECITE" "$f" 2>/dev/null)
    [ -n "$r" ] || r=0
    n=$((h + d + z + r))
    TOTAL=$((TOTAL + n))
    if [ "$n" -gt "$exp" ]; then
        echo "FAIL: $b - $h hash(es), $d date(s), $z CJK, $r rulebook cite(s) (baseline $exp)"
        grep -nohE "\b[0-9a-f]{7,8}\b|20[0-9]{2}-[0-9]{2}-[0-9]{2}" "$f" 2>/dev/null \
            | grep -vE "^[0-9]+:[0-9]+$" | head -4 | sed 's/^/        /'
        FAIL=1
    fi
done

# THE RULEBOOK CITE, over build/ as well - the one predicate here whose
# scope is the whole repository rather than the shipped sources. A
# citation is not a reason wherever it sits: "iron law 8" says nothing
# at the line it annotates, the rulebook can be renumbered, and every
# citation then points at the wrong rule silently. That argument does
# not stop at src/.
#
# Only this predicate widens. Commit hashes and dates in a gate header
# are derivations - "measured at 261,968 with -O2" needs the commit that
# measured it - and a gate that reddened on those would be switched off.
#
# THIS FILE IS THE EXEMPTION, and it has to be: naming the phrase is how
# the check finds it. Nothing else may.
#
# SUBDIRECTORIES TOO. This listed build/*.sh only, so build/arm,
# build/mips and build/x87 were outside a predicate whose stated scope is
# "the whole repository" - and build/mips/endian_gate.sh opened with a
# rulebook citation for as long as that was true. The scope in the
# comment above was right; the glob under it was one level short.
BSCAN=$(ls build/*.sh build/*.awk build/*.c build/*.py \
           build/arm/*.sh build/mips/*.sh build/x87/*.sh 2>/dev/null \
        | grep -v "comment_hygiene_gate.sh")
BR=0
for f in $BSCAN; do
    n=$(grep -ciE "$RULECITE" "$f" 2>/dev/null)
    [ -n "$n" ] || n=0
    if [ "$n" -gt 0 ]; then
        echo "FAIL: $f - $n rulebook cite(s); say the reason, not the number"
        grep -inE "$RULECITE" "$f" | head -3 | sed 's/^/        /'
        BR=$((BR + n))
        FAIL=1
    fi
done

# ---- THE SAME FOUR PREDICATES THE SCAN LOOP ABOVE APPLIES -------------
#
# This scope carried ONE of them (RULECITE) while SCAN carried four, so a
# commit hash, a date or a Chinese comment in build/ was counted by
# neither loop: the two loops read like the same thing and were not, and
# the counts they print make it look as though the whole tree had been
# checked. `comment_hygiene_gate.sh` stays out of the file list for the
# reason it always did - it defines these rules, and its own control arm
# below plants one instance of each form.
#
# BBASE IS PER PREDICATE (`path:h:d:z`), NOT A SUM. The SCAN loop adds
# its four counts into one allowance, and a sum is fungible: a new hash
# can ride in behind a CJK line that was deleted, the total is unchanged
# and all four predicates stay green. Four predicates that can cancel
# each other are not four predicates, so these are compared one by one.
#
# THE NUMBERS ARE THE DEBT, measured per file when this loop gained the
# predicates - a ratchet, not a licence. Every entry is a line someone
# can remove, and the note printed when a count falls below its baseline
# is the invitation to remove it. Two entries are NOT debt and are here
# for a reason written at the site instead: the Chinese PROMPT= literals
# (this scan cannot tell a string from a comment, the same treatment
# src/err.c's message tables get above) and the baseline revision printed
# by build/pri64_scan_gate.sh, which is the identity of a measurement
# rather than of a source line.
BBASE="
build/arm/parity_gate.sh:1:1:0
build/attn_sink_learned_gate.sh:0:0:1
build/bare_endif_gate.sh:1:0:0
build/c89_floor_gate.sh:4:0:0
build/ced_gate.sh:0:0:1
build/ced_proj_arm_probe.c:2:0:0
build/ced_replay_gate.sh:0:0:1
build/codesize_gate.sh:3:1:0
build/csa2_dense_kv_gate.sh:0:0:1
build/csa2_f32count_gate.sh:0:0:1
build/csa2_gate.sh:0:0:1
build/csa2_mode_dispatch_gate.sh:1:0:0
build/csa2_mode_export_load_gate.sh:3:0:0
build/csa2_ring_fmt_gate.sh:0:0:1
build/csa2_ring_nf_gate.sh:0:0:1
build/fixed_tier_xcompiler_gate.sh:4:1:0
build/gcc_watcom_output_gate.sh:1:0:0
build/gr_gate.sh:0:0:1
build/kernel_isa_identity_gate.sh:1:0:0
build/kernel_matrix_gate.sh:0:2:0
build/kv_share_gate.sh:0:0:1
build/lz4_model_gate.sh:0:1:0
build/mips/endian_gate.sh:0:1:0
build/mla_gate.sh:0:0:1
build/move_census.sh:0:1:0
build/pair_census.sh:4:1:0
build/restore_broken_reorders.py:6:0:0
build/restore_q61_sse2.py:4:0:0
build/subn_gate.sh:0:2:0
build/x87/build_engine.sh:0:1:0
"
BZ=0
BO=0
for f in $BSCAN; do
    hb=$(grep -ohE "\b[0-9a-f]{7}\b" "$f" 2>/dev/null \
         | grep -vE "^[0-9]+$" | wc -l)
    db=$(grep -ohE "20[0-9]{2}-[0-9]{2}-[0-9]{2}" "$f" 2>/dev/null | wc -l)
    zb=$(LC_ALL=C grep -c $'[\xe4-\xe9][\x80-\xbf][\x80-\xbf]' "$f" 2>/dev/null)
    [ -n "$zb" ] || zb=0
    exp=0:0:0
    for e in $BBASE; do
        [ "${e%%:*}" = "$f" ] && exp=$(echo "$e" | cut -d: -f2-)
    done
    eh=${exp%%:*}; er=${exp#*:}; ed=${er%%:*}; ez=${er#*:}
    [ -n "$ez" ] || ez=0
    if [ "$hb" -gt "$eh" ] || [ "$db" -gt "$ed" ] || [ "$zb" -gt "$ez" ]; then
        echo "FAIL: $f - $hb hash(es) [base $eh], $db date(s) [base $ed]," \
             "$zb CJK [base $ez]; say the reason, not the coordinate"
        grep -nohE "\b[0-9a-f]{7}\b|20[0-9]{2}-[0-9]{2}-[0-9]{2}" "$f" 2>/dev/null \
            | grep -vE "^[0-9]+$" | head -4 | sed 's/^/        /'
        BZ=$((BZ + 1))
        FAIL=1
    elif [ "$hb" -lt "$eh" ] || [ "$db" -lt "$ed" ] || [ "$zb" -lt "$ez" ]; then
        BO=$((BO + 1))
        echo "note: $f is below its baseline in at least one predicate" \
             "($hb/$db/$zb against $eh/$ed/$ez) - tighten it"
    fi
done

# ---- SOURCE-LINE CITATIONS, over the same two scopes ------------------
#
# `src/ops.c:1104` in a comment is a fact about a file that keeps being
# edited, and it rots without anyone touching the comment. Four of them
# were live when this check was added, and all four were wrong: two in
# ops_t2_scalar.{c,h} pointed at Q8 quantization instead of the T2
# unpack, and two in switch_matrix_gate.sh named ops.c for an #error
# that lives in ops_gdn.c - a reader following that one finds nothing
# and concludes the guard does not exist.
#
# A FUNCTION NAME IS THE FIX, not a better number: it survives edits,
# and grep finds it. Same argument as the rulebook cite above - say the
# thing, not the coordinate.
#
# EXTERNAL FILES ARE EXEMPT. `graph.h:903` and `common.h:260` cite
# llama.cpp, which this repo does not edit and cannot check; the checks
# below only fire on names that exist in this tree.
#
# OUT OF RANGE ONLY, which is the part that can be decided. `file.c:N`
# is also how this repo writes file:COUNT tables - c89_floor_gate's
# baseline reads `forward.c:16` for sixteen known violations, and
# vc4_limits_gate assigns STR_BASELINE="src/cli_main.c:2" for two
# over-long strings - so "looks like a citation" is not decidable, and
# two successively looser rules produced a false positive each. A line
# PAST THE END of the file it names is stale by arithmetic, whatever it
# meant: src/ops.c:5788 sat in build/x87/build_engine.sh with ops.c at
# 4706 lines.
#
# An in-range citation that points at the wrong thing is not
# mechanically checkable and is not attempted. Four were live when this
# was written - two in ops_t2_scalar.{c,h} aimed at Q8 quantization
# instead of the T2 unpack, two in switch_matrix_gate.sh naming ops.c
# for an #error that is in ops_gdn.c - and all four were fixed by hand.
# Half a check that survives beats a whole one that gets switched off,
# the same trade seed_gate makes about missing seeds.
CR=0
for f in $SCAN $BSCAN; do
    [ -f "$f" ] || continue
    while read -r cite; do
        [ -n "$cite" ] || continue
        num=${cite##*:}
        base=${cite%:*}; base=${base##*/}
        [ -f "src/$base" ] || continue
        len=$(grep -c '' "src/$base" 2>/dev/null || echo 0)
        [ "$num" -gt "$len" ] 2>/dev/null || continue
        echo "FAIL: $f cites $cite but src/$base has $len lines"
        echo "        - name the function; a line number rots unedited"
        CR=$((CR + 1))
        FAIL=1
    done <<EOF
$(grep -oE "\b(src/)?[a-z_0-9]+\.[ch]:[0-9]+" "$f" 2>/dev/null | sort -u)
EOF
done

# Positive control: the scan must be able to see one. Planting a hash
# in this script's own text would be caught by the src/ loop only if it
# lived in src/, so the control is a direct test of the three regexes
# against strings that must match.
CTL=$(printf 'see 8017c83\nmeasured 2026-08-24\nper iron law 10\n' \
      | grep -ciE "\b[0-9a-f]{7}\b|20[0-9]{2}-[0-9]{2}-[0-9]{2}|iron law")
CJK=$(printf '/* \xe4\xb8\xad */\n' \
      | LC_ALL=C grep -c $'[\xe4-\xe9][\x80-\xbf][\x80-\xbf]')
if [ "$CTL" -ne 3 ] || [ "$CJK" -ne 1 ]; then
    echo "FAIL: the patterns do not match their own examples"
    echo "      (ascii $CTL of 3, cjk $CJK of 1) - this gate is blind"
    echo "      and its PASS means nothing"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    # $TOTAL is every counted line INCLUDING the baselined ones, so
    # "$TOTAL over baseline: 0" read as one phrase said "1 over
    # baseline" on a run that passed. Two numbers, named.
    echo "comment_hygiene_gate: PASS ($TOTAL counted, 0 over baseline;"
    echo "  $(printf '%s\n' $BSCAN | wc -l) build script(s): 0 rulebook cite(s),"
    echo "  none over its hash/date/CJK baseline, $BO below it)"
    exit 0
fi
echo "comment_hygiene_gate: FAIL"
exit 1
