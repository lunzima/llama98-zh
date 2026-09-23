#!/bin/bash
# One writer per tree, and a name attached to it.
#
# WHAT THIS IS FOR. Readings produced here have been invalidated by a
# second writer more than once: a `make check` whose objects vanished
# under it, a `ci.sh` overlapping another run's builds, and two sessions
# that each believed the tree was theirs. A process snapshot cannot
# answer "is the tree idle" - compilers are pulse-shaped, so "no gcc
# right now" is true between any two of them. This is the thing that
# CAN answer it: a claim, held by a name, that someone either has or has
# not taken.
#
# It is not only an exclusion. The holder's name, pid, start time, HEAD
# and purpose are IN the lock, because the other half of the problem is
# invisible to mutual exclusion: a session that starts work while another
# is still finishing cannot be blocked by a mutex - it can only be told,
# if the finisher's claim is visible.
#
#   tree_lock.sh acquire [--steal]   take the lock, or explain who has it
#   tree_lock.sh release             give it back (only your own)
#   tree_lock.sh status              who holds it, or that nobody does
#
# WHICH FORM A READING NEEDS. One entry point is protected by its own
# lock (the gates that touch a shared path take it themselves). A ROUND -
# anything that runs more than one of them, or whose result is a reading
# someone will quote - needs the outer form, build/with_tree_lock.sh,
# because between two gates there is otherwise a window for an outside
# writer to replace what the later stages read:
#
#   make locked-check                      a per-commit verification round
#   with_tree_lock.sh bash build/ci.sh     a full CI round, whose summary
#                                          line is quoted
#   with_tree_lock.sh <a census or an      anything whose number goes into
#                      A/B measurement>     a report
#
# NOT needed, deliberately: reading the tree (ls, grep, git status, a log),
# and the gates that build only in a scratch directory and share nothing.
# `make check` also remains available unlocked - it is the ordinary form,
# and its per-gate locks still keep two gates from colliding.
#
# WHAT IT DOES NOT COVER: editors. Taking the lock says no other RUNNER is
# in the tree; it says nothing about someone editing a file this run is
# about to read. An editor does not take the lock and cannot be excluded
# by one, and the reading comes back looking ordinary either way - so a
# run that overlapped an edit to a file it reads is a window, not a
# result, no matter how plausible its output. `comment_hygiene_gate.sh`
# and this file's own gate both scan build/*.sh, which is where that
# happens most easily.
#
# WHAT IT LOCKS: one claim for the whole tree, not one per directory.
# The shared things that have been lost to races live in different
# places - build/native/ and the .gcov files gcov drops in the tree root -
# and the property being protected is "one reading at a time", which is
# not a property of a directory.
#
# ATOMICITY. The lock IS a directory (build/.tree-lock.d): mkdir either
# creates it or fails, with no window in between, so two concurrent
# acquirers cannot both win. The owner file inside is written to a
# temporary name and moved into place, so a reader sees either the
# previous owner or the new one, never half of either. Stealing is
# `mv` the whole directory aside and then mkdir a fresh one - one of two
# concurrent stealers loses the mv (its source is gone) and refuses.
#
# STALE LOCKS. A holder that dies without releasing leaves the claim.
# `acquire` probes the pid it finds and prints WHICH of three states it
# is in - alive / gone / undecidable - and refuses in all three unless
# `--steal` is given. It never takes a lock on its own initiative:
# automatic takeover driven by a timeout would have to be either longer
# than the longest legitimate run (an hour here, so it would never fire)
# or short enough to steal a live holder's lock, which is the silent
# failure this exists to remove. `--steal` prints what it overrides.
#
# The directory a steal moves aside is named <LOCKDIR>.stale.<ts>.<pid>
# and nothing reclaims it: `status` reads the lock's own path, and a
# claim that is no longer there is not the lock it reports on, so these
# accumulate where neither this script nor `git status` looks. How many
# to keep, and who removes the rest, is a question that is open here -
# this script reclaims none of them.
#
# Exit codes: 0 acquired/released/read, 2 refused (someone else holds
# it), 3 refused (the lock is not yours to release), 4 usage/unusable.
set -u
cd "$(dirname "$0")/.." || exit 4

# The directory is overridable so the gate that tests this script can
# exercise it against a scratch path instead of the shipping one - a test
# that deletes lock directories must not be able to delete a live claim.
LOCKDIR=${LZ_TREE_LOCK_DIR:-build/.tree-lock.d}
OWNERFILE="$LOCKDIR/owner"

now_iso() { date +"%Y-%m-%dT%H:%M:%S"; }

token_new() {
    # pid + time + a random word: unique per acquisition, and readable.
    printf '%s-%s-%s' "$$" "$(date +%s)" "${RANDOM:-0}${RANDOM:-0}"
}

read_field() {                       # field name -> value on stdout
    [ -f "$OWNERFILE" ] || return 1
    sed -n "s/^$1=//p" "$OWNERFILE" | head -1
}

# alive / gone / unknown. Several probes because this shell has no single
# portable one: MSYS `kill -0` answers only for processes it can see, and
# `ps -W` lists Windows pids. A probe that cannot run is NOT evidence of
# death - it makes the state undecidable, and undecidable refuses.
probe_pid() {
    local pid=$1
    [ -n "$pid" ] || { echo unknown; return; }
    if kill -0 "$pid" 2>/dev/null; then echo alive; return; fi
    if command -v ps >/dev/null 2>&1 && ps -W 2>/dev/null | awk '{print $1}' | grep -qx "$pid"; then
        echo alive; return
    fi
    if kill -0 "$pid" 2>/dev/null || [ -d "/proc/$pid" ] 2>/dev/null; then
        echo alive; return
    fi
    # A pid that the kernel rejects outright is gone; a pid we cannot ask
    # about at all is unknown.
    if kill -0 "$pid" 2>&1 | grep -qi 'no such process'; then echo gone; return; fi
    echo unknown
}

describe_holder() {
    local o n p s t h w
    o=$(read_field owner)  || { echo "  (no owner file inside $LOCKDIR)"; return; }
    n=$(read_field name)   || n="?"
    p=$(read_field pid)    || p="?"
    s=$(read_field start_iso) || s="?"
    h=$(read_field head)   || h="?"
    w=$(read_field what)   || w="?"
    echo "  holder : $n (token $o)"
    echo "  pid    : $p"
    echo "  since  : $s"
    echo "  HEAD   : $h"
    echo "  doing  : $w"
    echo "  pid is : $(probe_pid "$p")"
}

cmd_status() {
    if [ ! -d "$LOCKDIR" ]; then
        echo "tree_lock: free (no $LOCKDIR)"
        return 0
    fi
    # An owner file that is missing, or that is missing any of the fields
    # a reader needs, is NEITHER "held by someone" nor "free". Reporting
    # the readable half as a holder would be the same substitution this
    # lock exists to refuse: two states, one report.
    local miss=""
    if [ ! -f "$OWNERFILE" ]; then
        miss="no owner file"
    else
        for f in owner name pid start_iso head; do
            [ -n "$(read_field "$f" 2>/dev/null)" ] || miss="$miss$f "
        done
        [ -n "$miss" ] && miss="missing field(s): $miss"
    fi
    if [ -n "$miss" ]; then
        echo "tree_lock: UNDECIDABLE - incomplete lock at $LOCKDIR ($miss)"
        echo "  this is neither 'held by someone' nor 'free'. Inspect it by hand:"
        echo "    ls -la $LOCKDIR; cat $OWNERFILE"
        return 0
    fi
    echo "tree_lock: held"
    describe_holder
    return 0
}

cmd_acquire() {
    # Flags in any order: `acquire --print-env --steal` and
    # `acquire --steal --print-env` are the same request, and a positional
    # read would silently treat the first one as neither.
    local steal="" printenv="" a
    for a in "$@"; do
        case "$a" in
            --steal)     steal=--steal ;;
            --print-env) printenv=--print-env ;;
        esac
    done
    # Re-entrant for the same holder: a runner that holds the lock calls
    # gates that also acquire, and those must not deadlock against it.
    if [ -n "${LZ_TREE_LOCK:-}" ] && [ -f "$OWNERFILE" ] &&
       [ "$(read_field owner)" = "$LZ_TREE_LOCK" ]; then
        [ "$printenv" = "--print-env" ] && echo "export LZ_TREE_LOCK=$LZ_TREE_LOCK"
        return 0
    fi
    if mkdir "$LOCKDIR" 2>/dev/null; then
        : > "$LOCKDIR/owner.tmp"
        {
            echo "owner=$(token_new)"
            echo "name=${LZ_TREE_LOCK_NAME:-$0}"
            # The OWNER is the process that asked for the lock, not this
            # script: this script exits immediately, and a liveness probe
            # pointed at its own pid would call every live holder dead.
            echo "pid=${LZ_TREE_LOCK_PID:-$PPID}"
            echo "pid_self=$$"
            echo "start=$(date +%s)"
            echo "start_iso=$(now_iso)"
            echo "head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
            echo "what=${LZ_TREE_LOCK_WHAT:-$(basename "$0")}"
        } > "$LOCKDIR/owner.tmp"
        mv -f "$LOCKDIR/owner.tmp" "$OWNERFILE"
        export LZ_TREE_LOCK="$(read_field owner)"
        if [ "$printenv" = "--print-env" ]; then
            echo "export LZ_TREE_LOCK=$LZ_TREE_LOCK"
        else
            echo "tree_lock: acquired by ${LZ_TREE_LOCK_NAME:-$0} (token $LZ_TREE_LOCK)"
        fi
        return 0
    fi
    # Somebody has it. Say who, and in which of the three states.
    if [ "$printenv" = "--print-env" ]; then
        # stdout carries ONLY the export line the caller evals, so every
        # refusal has to go to stderr: a partial capture would otherwise
        # be read as an acquisition.
        {
          echo "tree_lock: REFUSED - $LOCKDIR exists"
          describe_holder
        } >&2
    else
        echo "tree_lock: REFUSED - $LOCKDIR exists"
        describe_holder
    fi
    if [ "$steal" != "--steal" ]; then
        {
          echo "  If that holder is gone, take it over explicitly:"
          echo "    LZ_TREE_LOCK_NAME=<you> bash $0 acquire --steal"
        } >&2
        return 2
    fi
    local pid state aside
    pid=$(read_field pid) || pid=""
    state=$(probe_pid "$pid")
    if [ "$state" = "alive" ]; then
        echo "tree_lock: REFUSED - pid $pid is ALIVE; --steal does not take a live holder's lock" >&2
        return 2
    fi
    aside="$LOCKDIR.stale.$(date +%s).$$"
    if ! mv "$LOCKDIR" "$aside" 2>/dev/null; then
        echo "tree_lock: REFUSED - another process took the lock while this one decided to steal" >&2
        return 2
    fi
    {
      echo "tree_lock: OVERRIDING a lock whose pid is $state - moved to $aside"
      echo "  overridden holder was:"
      echo "    $(sed -n 's/^name=//p' "$aside/owner" 2>/dev/null | head -1) pid=$pid"
      echo "    HEAD=$(sed -n 's/^head=//p' "$aside/owner" 2>/dev/null | head -1) start=$(sed -n 's/^start_iso=//p' "$aside/owner" 2>/dev/null | head -1)"
    } >&2
    if ! mkdir "$LOCKDIR" 2>/dev/null; then
        echo "tree_lock: REFUSED - could not create $LOCKDIR after the move" >&2
        return 2
    fi
    {
        echo "owner=$(token_new)"
        echo "name=${LZ_TREE_LOCK_NAME:-$0}"
        echo "pid=${LZ_TREE_LOCK_PID:-$PPID}"
        echo "pid_self=$$"
        echo "start=$(date +%s)"
        echo "start_iso=$(now_iso)"
        echo "head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "what=${LZ_TREE_LOCK_WHAT:-$(basename "$0")} (stole from pid $pid)"
    } > "$LOCKDIR/owner.tmp"
    mv -f "$LOCKDIR/owner.tmp" "$OWNERFILE"
    export LZ_TREE_LOCK="$(read_field owner)"
    if [ "$printenv" = "--print-env" ]; then
        echo "export LZ_TREE_LOCK=$LZ_TREE_LOCK"
    else
        echo "tree_lock: acquired by ${LZ_TREE_LOCK_NAME:-$0} (token $LZ_TREE_LOCK)"
    fi
    return 0
}

cmd_release() {
    if [ ! -d "$LOCKDIR" ]; then
        echo "tree_lock: nothing to release (no $LOCKDIR)"
        return 0
    fi
    local cur opener mine
    cur=$(read_field owner) || cur=""
    if [ -z "${LZ_TREE_LOCK:-}" ] || [ "$cur" != "$LZ_TREE_LOCK" ]; then
        # A holder that was stolen from still has its token, and its late
        # release must not free the new holder's lock - the shape where an
        # action destroys someone else's state and nothing reports it.
        echo "tree_lock: REFUSED - not yours to release"
        echo "  this process token: ${LZ_TREE_LOCK:-<none>}"
        echo "  current lock     : ${cur:-<none>}"
        return 3
    fi
    # THE TOKEN IS INHERITED; THE OPENING IS NOT. Every process the holder
    # starts inherits LZ_TREE_LOCK, so a nested take passes through
    # silently (that is what makes a runner's own gates re-entrant) and a
    # nested process carries the same token as the holder. A release that
    # asked only whether the token is current therefore lets a nested gate
    # free a claim it never opened: the holder runs on, and the lock is
    # gone from under it - measured as free for the whole of a gate whose
    # `make` runs the build hooks. The opener is the shell that asked for
    # the lock, recorded in the owner file; a release belongs to that
    # shell, or to a process started with its pid.
    opener=$(read_field pid) || opener=""
    mine=${LZ_TREE_LOCK_PID:-$PPID}
    if [ -z "$opener" ] || [ "$opener" != "$mine" ]; then
        echo "tree_lock: REFUSED - not the process that took it"
        echo "  lock opened by pid: ${opener:-<none>}"
        echo "  this releaser pid : $mine"
        return 3
    fi
    rm -rf "$LOCKDIR"
    echo "tree_lock: released"
    return 0
}

case "${1:-}" in
    acquire) shift; cmd_acquire "$@" ;;
    release) cmd_release ;;
    status)  cmd_status ;;
    *) echo "usage: $0 acquire [--steal] [--print-env] | release | status" >&2; exit 4 ;;
esac
