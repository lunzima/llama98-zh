#!/bin/bash
# Run a command under the tree lock: take it, run, give it back.
#
# WHY THIS EXISTS ON TOP OF THE PER-GATE LOCKS. Each entry point that
# touches a shared path now takes the lock for itself, which stops two
# gates from colliding. It does not stop an OUTSIDE writer from slipping
# in BETWEEN two gates of a run - and from that moment the run is reading
# a tree its later stages did not start from. "Every gate is safe" and
# "this round is safe" are different statements; this is the second one.
#
#   with_tree_lock.sh make check
#   with_tree_lock.sh make ci
#   with_tree_lock.sh bash build/ci.sh
#
# The token is exported into this shell and therefore into the command and
# everything it starts, so the per-gate locks inside the round pass
# through silently instead of deadlocking against their own runner.
#
# A command that cannot take the lock does not run at all: a refused
# round is a reading nobody has to withdraw later.
#
# Exit codes: the command's own, or 2 if the lock is held by someone else.
set -u
cd "$(dirname "$0")/.." || exit 4

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command...>" >&2
    exit 4
fi

# --print-env keeps stdout for the eval below; refusals go to stderr.
eval "$(LZ_TREE_LOCK_NAME="${LZ_TREE_LOCK_NAME:-with_tree_lock.sh: $1}" \
        LZ_TREE_LOCK_WHAT="${LZ_TREE_LOCK_WHAT:-$*}" \
        bash build/tree_lock.sh acquire --print-env)" || exit 2
trap 'bash build/tree_lock.sh release' EXIT

"$@"
rc=$?
exit "$rc"
