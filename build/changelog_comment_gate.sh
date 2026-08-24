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

PAT="used to be|used to |no longer|the old |originally|before this task|as before this|previously|formerly"

BASELINE="err.c:1 forward.c:3 generate.c:6 model.c:1 ops.c:8
          ops_gdn.c:1 ops_matmul.c:1 ops_mmx.c:2 ops_norm.c:5 ops_quant.c:4
          ops_sched.c:5 sampler.c:4 llama_zh.h:1 lz_int.h:1 net.h:1 ops.h:6
          ops_kernel_dot.h:2"

FAIL=0
TOTAL=0
for f in src/*.c src/*.h; do
    b=$(basename "$f")
    exp=0
    for e in $BASELINE; do
        [ "${e%%:*}" = "$b" ] && exp=${e##*:}
    done
    # -o, not -c: grep -c counts LINES, so a second phrase appended to a
    # line that already has one is free. Matches is the quantity the
    # ratchet is about.
    n=$(grep -oiE "$PAT" "$f" 2>/dev/null | wc -l)
    [ -n "$n" ] || n=0
    TOTAL=$((TOTAL + n))
    if [ "$n" -gt "$exp" ]; then
        echo "FAIL: $b - $n changelog phrase(s), baseline $exp"
        grep -nioE "$PAT" "$f" | head -3 | sed 's/^/        /'
        FAIL=1
    fi
done

# Positive control. Without it a typo in $PAT makes every file report
# zero, every file is at or under its baseline, and the gate passes
# while measuring nothing - the shape this repo has hit more than once.
CTL=$(printf 'it used to be four\nthis is no longer true\nthe old loop\n' \
      | grep -ciE "$PAT")
if [ "$CTL" -ne 3 ]; then
    echo "FAIL: the pattern does not match its own examples ($CTL of 3)"
    echo "      - this gate is blind and its PASS means nothing"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo "changelog_comment_gate: PASS ($TOTAL total, none above baseline)"
    exit 0
fi
echo "changelog_comment_gate: FAIL"
exit 1
