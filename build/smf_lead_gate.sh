#!/bin/bash
# common/smf.c's lead-in trim: does lz_smf_open() find the piece's true
# first sounding note across every track, or only the first track (by
# index) that happens to have one?
#
# lz_smf_open's pass 1 walks each track to completion, in track order,
# sharing one `first` accumulator across all of them (track_advance's
# own comment has the full account). The accumulator has to become the
# MINIMUM tick over every note-on in the file; before this fix it only
# ever got written once (on an "== sentinel" guard rather than a
# "smaller than what is there" one), so a part carried on a later track
# that sounds AHEAD of an earlier track's own first note lost the race -
# `first` locked onto the earlier track's later note, `lead` came out
# too large, and every real event before that point clamped to output
# tick 0 (lz_smf_next's `out->tick > lead ? out->tick - lead : 0`) -
# a burst of simultaneous notes rather than the intro it was written as.
# Reported against a real file: gadameiren_arr.mid's intro is carried on
# channels 1 and 2 (tracks 2 and 3), both sounding before channel 0's
# own first note (track 1) - exactly the shape this gate reproduces
# synthetically, in build/smf_lead_probe.c.
#
# THE CONTROL is a MUTATION, not a git-history diff, same reasoning as
# this project's other gates on this shape: a git-ref-pinned "before"
# copy goes stale the day this fix lands and HEAD catches up to the
# working tree. The mutation reverts track_advance's condition back to
# the "write once" guard in a scratch copy and re-runs the SAME probe.
#
# WHY smf.c ALONE, no other engine source: it depends only on
# common/smf.h and the two music.h constants it reads (LZ_MUSIC_VOL_MAX,
# LZ_MUSIC_AS_WRITTEN) - no model, no forward pass, no Win32.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "smf_lead_gate: SKIP - no C compiler"
    exit 2
}

W=$(mktemp -d 2>/dev/null || echo /tmp/smfleadgate.$$)
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/fixed" "$W/mutant"

cp common/smf.c common/smf.h common/music.h "$W/fixed/"
cp common/smf.c common/smf.h common/music.h "$W/mutant/"

# The exact token this fix introduced - a two-way OR on the guard - back
# to the original one-way sentinel check. If this sed finds nothing, the
# guard's own text has moved and this gate no longer matches the source.
sed -i \
    's/(\*first_note == (unsigned long)-1 || t->tick < \*first_note)/*first_note == (unsigned long)-1/' \
    "$W/mutant/smf.c"
if grep -q 't->tick < \*first_note' "$W/mutant/smf.c"; then
    echo "smf_lead_gate: FAIL - could not mutate track_advance's first_note guard" \
         "(the sed target text is gone - the gate no longer matches the source)"
    exit 1
fi

RC=0
run_one() {
    local dir="$1" label="$2" want_pass="$3"
    if ! $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -o "$dir/probe" \
            build/smf_lead_probe.c "$dir/smf.c" 2>"$dir/cc.log"; then
        echo "smf_lead_gate: FAIL - probe did not build against $label"
        tail -n 20 "$dir/cc.log"
        return 1
    fi
    "$dir/probe" > "$dir/run.log" 2>&1
    local rc=$?
    cat "$dir/run.log"
    if [ "$want_pass" = 1 ] && [ "$rc" -ne 0 ]; then
        echo "smf_lead_gate: FAIL - $label did not pass the probe (exit $rc)"
        return 1
    fi
    if [ "$want_pass" = 0 ] && [ "$rc" -eq 0 ]; then
        echo "smf_lead_gate: FAIL - $label PASSED the probe; it should have" \
             "failed (the reverted guard cannot be told apart from the fix)"
        return 1
    fi
    return 0
}

echo "-- current tree (must PASS) --"
run_one "$W/fixed" "the current (fixed) tree" 1 || RC=1

echo ""
echo "-- mutated tree: first_note guard reverted to write-once (must FAIL) --"
run_one "$W/mutant" "the write-once-guard mutant" 0 || RC=1
if [ "$RC" -eq 0 ]; then
    echo "-- mutant correctly failed - the probe is sensitive to this exact" \
         "defect class --"
fi

# ------------------------------------------------- real-file regression
# The file that reported this: gadameiren_arr.mid's intro runs on
# channels 1/2 (tracks 2/3), both ahead of channel 0's own first note.
# music/ is untracked (a separate CC BY 4.0 project, CLAUDE.md's own
# rule) so this is SKIPPED rather than required when it is not present -
# same posture json_offset_gate.sh takes for its own on-disk fixture.
MID="${LZ_SMF_GATE_MIDI:-music/gadameiren_arr.mid}"
echo ""
if [ ! -f "$MID" ]; then
    echo "smf_lead_gate: real-file regression SKIPPED - $MID not found" \
         "(set LZ_SMF_GATE_MIDI)"
else
    echo "-- real-file regression: $MID --"
    cp build/smf_lead_realfile_probe.c "$W/fixed/harness.c"
    cp build/smf_lead_realfile_probe.c "$W/mutant/harness.c"
    for dir_label in "$W/fixed fixed" "$W/mutant mutant"; do
        set -- $dir_label
        dir="$1"; label="$2"
        if ! $CC -O2 -std=c99 -I"$dir" -o "$dir/realfile" \
                "$dir/harness.c" "$dir/smf.c" 2>"$dir/cc3.log"; then
            echo "smf_lead_gate: FAIL - real-file harness did not build ($label)"
            tail -n 20 "$dir/cc3.log"
            RC=1
            continue
        fi
        "$dir/realfile" "$(pwd)/$MID" > "$dir/real.log" 2>&1
        cat "$dir/real.log"
    done
    FIXED_LEAD=$(grep -oE '^lead=[0-9]+' "$W/fixed/real.log" 2>/dev/null | cut -d= -f2)
    MUT_LEAD=$(grep -oE '^lead=[0-9]+' "$W/mutant/real.log" 2>/dev/null | cut -d= -f2)
    if [ -z "$FIXED_LEAD" ]; then
        echo "smf_lead_gate: FAIL - fixed tree did not report a lead value on $MID"
        RC=1
    elif [ -n "$MUT_LEAD" ] && [ "$FIXED_LEAD" -ge "$MUT_LEAD" ]; then
        echo "smf_lead_gate: FAIL - the fixed tree's lead ($FIXED_LEAD) is not" \
             "smaller than the mutant's ($MUT_LEAD) on the real file - this" \
             "file does not exercise the defect the way the report described"
        RC=1
    else
        echo "smf_lead_gate: real file's lead shrank from $MUT_LEAD (mutant) to" \
             "$FIXED_LEAD (fixed) - the intro on channels 1/2 is no longer" \
             "absorbed into the trim"
    fi
fi

if [ "$RC" -eq 0 ]; then
    echo "smf_lead_gate: PASS"
else
    echo "smf_lead_gate: FAIL"
fi
exit $RC
