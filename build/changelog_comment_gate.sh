#!/bin/bash
# Comments describe the code as it is, not the sequence of edits that
# produced it. This is the phrasing half of that rule; the hash/date/CJK
# half is comment_hygiene_gate.sh.
#
# WHAT THIS CATCHES: "it used to be", "no longer", "the old loop", "as
# before this task". A reader who has to reconstruct a previous version
# of the file to understand the current one is reading a diff, not a
# comment - and the previous version is in git, where it can be read
# accurately.
#
# WHAT IT MUST NOT CATCH, and why this is a RATCHET rather than a ban:
# some of these sentences are load-bearing. "A `s == 0` fast path used
# to be here and its arm never fired" stops someone re-adding it; the
# repair is to rewrite it as a present-tense claim ("a `s == 0` fast
# path would never fire, because ...") rather than to delete the
# knowledge. Rewriting all of them at once is a large mechanical edit
# across files other sessions hold open, so the baseline below is the
# count at the time this landed and the assertion is only that no file
# gains one. Lowering a number here is the work; raising one is the
# regression.
#
# The phrase "the old" also has a legitimate non-historical use - "read
# the OLD state from sq_in" means the pre-step value, not a previous
# revision. Those sit inside the baseline; the gate cannot tell them
# apart and does not try, because a gate that guesses at intent fails on
# the cases the project wants to keep.
#
# Exit codes:
#   0  PASS - no file above its baseline
#   1  FAIL - a file gained one, or the scan matched nothing at all
set -u
cd "$(dirname "$0")/.."

# TWO PATTERNS, because the phrases are not equally guilty.
#
# STRICT is unambiguous: "it used to be", "previously", "formerly", "as
# before this task" have no reading in which they describe the code as
# it is. They are a ratchet toward zero and the baseline below is what
# is left.
#
# LOOSE catches "no longer", "the old" and "originally", which DO have
# legitimate present-tense uses and are full of them here - "once they
# can no longer participate" describes a state, "the OLD s->mtp_pos" is
# the pre-step value, "the old in-place update" names the other code
# path. Rewriting those to satisfy a regex would make the comments
# worse. So LOOSE is counted and printed, never failed on: it is a
# number to look at, not an assertion.
STRICT="\bused to be|\bused to |\bbefore this task|\bas before this|\bpreviously|\bformerly"
LOOSE="no longer|the old |originally"
PAT="$STRICT"

# THE DEBT, per file, keyed by PATH. src/ is at zero but for one file
# another session holds open. build/ joined the scope later and came in at
# 38 across seventeen files, which is the whole reason this stopped being
# "a ratchet with almost nothing left to ratchet" and became a real one:
# every number below is the count measured when the scope reached that
# file, and each is a sentence someone can rewrite as a present-tense
# claim - the note at the bottom asks for exactly that.
#
# KEYED BY PATH, NOT BASENAME. This table was keyed by basename until the
# scope reached build/, and the sibling gate states the reason that is
# wrong in as many words: a basename key lets a future gui/forward.c
# inherit src/forward.c's allowance - "a table keyed by name hands a
# newcomer someone else's excuse". The scope already holds two such pairs
# (session.c and session.h exist under both src/ and common/), so the
# collision was latent, not absent, and widening the scope is what would
# have made it live.
BASELINE="
src/forward.c:1
build/arm/parity_gate.sh:3
build/arm/softfp_gate.sh:1
build/c89_floor_gate.sh:1
build/ci.sh:3
build/comment_hygiene_gate.sh:1
build/csa2_last_writer_bin_gate.sh:1
build/csa2_modes_config_gate.sh:1
build/fixed_tier_xcompiler_gate.sh:8
build/gate_wiring_gate.sh:2
build/gcc_flag_parity_gate.sh:1
build/kernel_isa_identity_gate.sh:6
build/kernel_matrix_gate.sh:1
build/move_census.sh:1
build/o3_probe.sh:1
build/recur_write_float_arm_gate.sh:2
build/subn_gate.sh:1
build/switch_matrix_gate.sh:4
"

# WHAT IS SCANNED. src/ was the whole list, which left three sentences
# standing in gui/ and one in the Makefile that this gate would have
# failed on the day they were written - the rule is about comments, and
# the engine is not the only place comments live. common/, gui/ and the
# Makefile join it.
#
# WIDENING IT FOUND A FALSE POSITIVE WORTH KNOWING ABOUT: compat40.h's
# "Used to save and restore a thread's layout" is the purpose sense of
# the words, not the historical one, and STRICT cannot tell them apart.
# It was rewritten to "Callers save and restore" rather than exempted,
# because the ambiguity was in the comment, not in the regex.
#
# build/*.sh is DELIBERATELY OUT, and that is a decision rather than an
# oversight. Several of its matches are load-bearing in a way source
# comments are not - "ROWS 5-8 USED TO PAIR EACH -D WITH ITS OWN RUNTIME
# FLAG" is what stops the next reader re-adding seven retired knobs, and
# this file's own header has to quote the phrases to say what it catches.
# The count is MEASURED and printed at the end rather than written here,
# so it cannot go stale into a claim.
SCAN="src/*.c src/*.h common/*.c common/*.h gui/*.c gui/*.h Makefile
      build/*.sh build/arm/*.sh build/mips/*.sh build/x87/*.sh"

FAIL=0
TOTAL=0
BELOW=0
for f in $SCAN; do
    # THIS FILE IS EXCLUDED BY NAME, the way comment_hygiene_gate.sh
    # excludes itself from its build scope. Its header has to quote the
    # phrases to say which ones it catches, and its control arm below
    # feeds the patterns their own examples: a scan cannot tell a
    # quotation from a use. By NAME rather than by pattern, so the
    # exemption cannot spill onto anything else.
    [ "$f" = "build/changelog_comment_gate.sh" ] && continue
    exp=0
    for e in $BASELINE; do
        [ "${e%%:*}" = "$f" ] && exp=${e##*:}
    done
    # -o, not -c: grep -c counts LINES, so a second phrase appended to a
    # line that already has one is free. Matches is the quantity the
    # ratchet is about.
    n=$(grep -oiE "$PAT" "$f" 2>/dev/null | wc -l)
    [ -n "$n" ] || n=0
    TOTAL=$((TOTAL + n))
    if [ "$n" -gt "$exp" ]; then
        echo "FAIL: $f - $n changelog phrase(s), baseline $exp"
        grep -nioE "$PAT" "$f" | head -3 | sed 's/^/        /'
        FAIL=1
    elif [ "$n" -lt "$exp" ]; then
        BELOW=$((BELOW + 1))
        echo "note: $f is at $n, below its baseline of $exp - tighten it"
    fi
done

# Positive control, one per pattern. Without it a typo makes every file
# report zero, every file is at or under its baseline, and the gate
# passes while measuring nothing - the shape this repo has hit more than
# once. Two examples each, because a pattern that lost one alternative
# out of six would still match the other.
SCTL=$(printf 'it used to be four\nformerly the case\n' | grep -ciE "$STRICT")
LCTL=$(printf 'this is no longer true\nthe old loop\n' | grep -ciE "$LOOSE")
if [ "$SCTL" -ne 2 ] || [ "$LCTL" -ne 2 ]; then
    echo "FAIL: a pattern does not match its own examples"
    echo "      (strict $SCTL of 2, loose $LCTL of 2) - this gate is blind"
    FAIL=1
fi

# The loose phrases are COUNTED, not asserted on: "once they can no
# longer participate" describes a state and "the OLD s->mtp_pos" is the
# pre-step value. Printed so the number is visible if it starts growing.
LTOT=0
for f in $SCAN; do
    n=$(grep -oiE "$LOOSE" "$f" 2>/dev/null | wc -l)
    LTOT=$((LTOT + n))
done

# WHAT STAYS OUT, named and with its count, rather than left to be
# discovered. There is no excluded directory any more - the build scripts
# were the last one and they are in $SCAN above. The one file that stays
# out is the one that states the rule, and its count is printed here so
# that its exclusion cannot grow without this line changing.
SELF=$(grep -oiE "$PAT" build/changelog_comment_gate.sh 2>/dev/null | wc -l)

if [ "$FAIL" -eq 0 ]; then
    echo "changelog_comment_gate: PASS ($TOTAL strict, none above baseline;"
    echo "  $LTOT loose - counted, not asserted, see the header)"
    echo "  scanned: $SCAN"
    echo "  excluded by name: build/changelog_comment_gate.sh ($SELF strict"
    echo "    match(es), quoting the phrases it catches)"
    [ "$BELOW" -eq 0 ] || echo "  $BELOW file(s) below baseline - tighten them"
    exit 0
fi
echo "changelog_comment_gate: FAIL"
exit 1
