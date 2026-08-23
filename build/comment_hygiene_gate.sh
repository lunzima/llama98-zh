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
#   gui/localized_strings.c:99   message tables, which are the shape the
#                            project asks for - every user-visible
#                            string lives in a table, source stays ASCII
#                            everywhere else. Counted because the scan
#                            cannot tell a string from a comment;
#                            excused here rather than by weakening the
#                            scan, which would also stop catching a
#                            Chinese COMMENT in those same files.
#   src/forward.c:1 src/model.c:1
#                            The lz_int.h include comment, already
#                            corrected in the working tree but sharing a
#                            diff hunk with another session's
#                            include-block reshuffle. Their commit
#                            carries it; baselined rather than left
#                            silently red.
BASELINE="src/err.c:138 gui/localized_strings.c:99 src/forward.c:1 src/model.c:1"

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
    r=$(grep -ciE "iron law" "$f" 2>/dev/null)
    [ -n "$r" ] || r=0
    n=$((h + d + z + r))
    TOTAL=$((TOTAL + n))
    if [ "$n" -gt "$exp" ]; then
        echo "FAIL: $b - $h hash(es), $d date(s), $z CJK, $r rulebook cite(s) (baseline $exp)"
        grep -nohE "\b[0-9a-f]{7,8}\b|20[0-9]{2}-[0-9]{2}-[0-9]{2}" "$f" 2>/dev/null \
            | grep -vE "^[0-9]+$" | head -4 | sed 's/^/        /'
        FAIL=1
    fi
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
    echo "comment_hygiene_gate: PASS ($TOTAL over baseline: 0)"
    exit 0
fi
echo "comment_hygiene_gate: FAIL"
exit 1
