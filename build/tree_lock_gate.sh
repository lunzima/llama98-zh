#!/bin/bash
# The tree lock's own criteria - nine arms, each one a state the lock has
# to get right, because a lock that is wrong in any of them is worse than
# no lock: it reads as mutual exclusion while the exclusion is not there.
#
#   A1  take twice                     -> the second refuses AND names the
#                                         holder (name, pid, since, HEAD)
#   A2  release, then take             -> succeeds
#   A3  holder killed, then take       -> refuses (stale) and prints which
#                                         of alive/gone/undecidable it saw;
#                                         the command it prints really works
#   A4  --steal                        -> succeeds, and the override print
#                                         names the holder it overrode
#   A5  nested take, matching token    -> silent success (a runner holding
#                                         the lock must be able to run gates
#                                         that also take it)
#   A6  nested take, wrong token       -> refuses
#   A7  late release after a steal     -> refuses, and the NEW holder's lock
#                                         is still there. This is the shape
#                                         where an action destroys someone
#                                         else's state and nothing reports
#                                         it; it fires only after a steal,
#                                         which is why it needs its own arm.
#   A8  two stealers at once           -> exactly one wins, the other
#                                         refuses (mkdir/mv are the atomic
#                                         primitives; the loser's mv finds
#                                         its source gone)
#   A9  half-written owner file        -> status says UNDECIDABLE, not
#                                         "free" and not "held by nobody"
#
# THE GATE NEVER TOUCHES THE SHIPPING LOCK. It points LZ_TREE_LOCK_DIR at a
# scratch path, because a test that deletes lock directories must not be
# able to delete a live claim; the last arm asserts the shipping default
# is still the path the script documents, so the override cannot quietly
# become the only path anybody exercises.
#
# No compiler, no model, no git write: this runs in seconds.
#
# Exit: 0 all arms pass, 1 an arm failed (printed), 2 skip (no bash).
set -u
cd "$(dirname "$0")/.." || exit 1

TL=build/tree_lock.sh
[ -f "$TL" ] || { echo "tree_lock_gate: SKIP - no $TL"; exit 2; }

W=$(mktemp -d); LD="$W/lock"
export LZ_TREE_LOCK_DIR="$LD"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$W"' EXIT

FAIL=0
ok() {
    printf '%s %s\n' "$1" "$2"
    fflush=1
    [ "$1" = "PASS" ] || FAIL=1
}

reset_lock() { rm -rf "$LD" "$LD".stale.* 2>/dev/null; unset LZ_TREE_LOCK 2>/dev/null || true; }

# A holder that STAYS ALIVE, so the liveness probe has something to find.
# The recorded owner pid is the subshell's (the acquiring script's PPID),
# which is what gets killed in A3/A7.
# Sets HOLD_PID instead of echoing it: `H3=$(hold_start X)` blocks for the
# fixture's whole lifetime, because a command substitution waits for the
# subshell's stdout to close and this subshell holds it open while it
# sleeps. Measured before the change: 61 s per call, four calls, in a 275 s
# gate where everything else is subsecond.
hold_start() {
    local name=$1
    ( LZ_TREE_LOCK_NAME="$name" LZ_TREE_LOCK_WHAT="gate fixture" \
      bash "$TL" acquire >"$W/hold.$name.log" 2>&1; sleep 60 ) &
    HOLD_PID=$!
}

# ---------------------------------------------------------------- A1/A2
reset_lock
LZ_TREE_LOCK_NAME=first bash "$TL" acquire >"$W/a1.first" 2>&1
rc1=$?
LZ_TREE_LOCK_NAME=second bash "$TL" acquire >"$W/a1.second" 2>&1
rc2=$?
ok "$([ "$rc1" -eq 0 ] && echo PASS || echo FAIL)" "A1: first acquire succeeds"
if grep -q "REFUSED" "$W/a1.second" &&
   grep -q "holder : first" "$W/a1.second" &&
   grep -qE "^  pid    : [0-9]+" "$W/a1.second" &&
   grep -qE "^  since  : " "$W/a1.second" &&
   grep -qE "^  HEAD   : " "$W/a1.second"; then
    ok PASS "A1: the second refuses and names the holder (name, pid, since, HEAD)"
    ok "$([ "$rc2" -eq 2 ] && echo PASS || echo FAIL)" "A1: refusal exit code is 2"
else
    ok FAIL "A1: the second refuses and names the holder (name, pid, since, HEAD)"
fi

T=$(sed -n 's/^owner=//p' "$LD/owner")
LZ_TREE_LOCK="$T" bash "$TL" release >"$W/a2.release" 2>&1
LZ_TREE_LOCK_NAME=after-release bash "$TL" acquire >"$W/a2.acquire" 2>&1
ok "$([ $? -eq 0 ] && ! grep -q REFUSED "$W/a2.acquire" && echo PASS || echo FAIL)" \
   "A2: after a release the lock can be taken again"

# ------------------------------------------------------------------- A3
reset_lock
hold_start A3holder; H3=$HOLD_PID
sleep 1
kill -9 "$H3" 2>/dev/null; wait "$H3" 2>/dev/null
sleep 1
LZ_TREE_LOCK_NAME=A3later bash "$TL" acquire >"$W/a3.acquire" 2>&1
rc3=$?
if grep -q "REFUSED" "$W/a3.acquire" && grep -qE "^  pid is : (gone|unknown|alive)" "$W/a3.acquire"; then
    ok PASS "A3: taking a dead holder's lock refuses, and prints the probe's verdict"
    ok "$([ "$rc3" -eq 2 ] && echo PASS || echo FAIL)" "A3: stale refusal exit code is 2"
else
    ok FAIL "A3: taking a dead holder's lock refuses, and prints the probe's verdict"
fi
# The command the refusal prints has to work - a diagnostic that cannot be
# acted on leaves the reader stuck with a lock nobody can clear.
grep -q "acquire --steal" "$W/a3.acquire" \
    && ok PASS "A3: the refusal prints the takeover command" \
    || ok FAIL "A3: the refusal prints the takeover command"

# ------------------------------------------------------------------- A4
reset_lock
hold_start A4holder; H4=$HOLD_PID
sleep 1
kill -9 "$H4" 2>/dev/null; wait "$H4" 2>/dev/null
sleep 1
LZ_TREE_LOCK_NAME=A4thief bash "$TL" acquire --steal >"$W/a4.steal" 2>&1
rc4=$?
if [ "$rc4" -eq 0 ] && grep -q "OVERRIDING" "$W/a4.steal" &&
   grep -q "A4holder" "$W/a4.steal"; then
    ok PASS "A4: --steal takes a dead holder's lock and names what it overrode"
else
    ok FAIL "A4: --steal takes a dead holder's lock and names what it overrode"
fi

# ------------------------------------------------------------------- A5
reset_lock
LZ_TREE_LOCK_NAME=A5outer bash "$TL" acquire >"$W/a5.outer" 2>&1
T5=$(sed -n 's/^owner=//p' "$LD/owner")
LZ_TREE_LOCK="$T5" LZ_TREE_LOCK_NAME=A5inner bash "$TL" acquire >"$W/a5.inner" 2>&1
rc5=$?
ok "$([ "$rc5" -eq 0 ] && [ ! -s "$W/a5.inner" ] && echo PASS || echo FAIL)" \
   "A5: a nested take with the matching token succeeds silently"

# ------------------------------------------------------------------- A6
LZ_TREE_LOCK="not-the-token" LZ_TREE_LOCK_NAME=A6inner bash "$TL" acquire >"$W/a6.inner" 2>&1
rc6=$?
ok "$([ "$rc6" -eq 2 ] && grep -q REFUSED "$W/a6.inner" && echo PASS || echo FAIL)" \
   "A6: a nested take with the wrong token refuses"

# ------------------------------------------------------------------- A7
# A holder is killed, its lock is stolen, and then the dead holder's
# wrapper comes back to release - holding a token that is no longer the
# lock's. The fixture has to be a KILLED holder for the steal to succeed
# at all (a live one refuses), which is also why this arm cannot be folded
# into A1/A2.
reset_lock
hold_start A7holder; H7=$HOLD_PID
sleep 1
T_A=$(sed -n 's/^owner=//p' "$LD/owner")
kill -9 "$H7" 2>/dev/null; wait "$H7" 2>/dev/null
sleep 1
LZ_TREE_LOCK_NAME=A7thief bash "$TL" acquire --steal >"$W/a7.steal" 2>&1
T_B=$(sed -n 's/^owner=//p' "$LD/owner")
LZ_TREE_LOCK="$T_A" bash "$TL" release >"$W/a7.late" 2>&1
rc7=$?
if [ "$rc7" -eq 3 ] && grep -q "not yours to release" "$W/a7.late" &&
   [ -d "$LD" ] && [ "$(sed -n 's/^owner=//p' "$LD/owner")" = "$T_B" ]; then
    ok PASS "A7: a stolen-from holder's late release is refused and leaves the new lock intact"
else
    ok FAIL "A7: a stolen-from holder's late release is refused and leaves the new lock intact"
fi

# ------------------------------------------------------------------- A8
reset_lock
hold_start A8holder; H8=$HOLD_PID
sleep 1
kill -9 "$H8" 2>/dev/null; wait "$H8" 2>/dev/null
sleep 1
( LZ_TREE_LOCK_NAME=A8one   bash "$TL" acquire --steal >"$W/a8.one"   2>&1; echo $? >"$W/a8.one.rc" ) &
P1=$!
( LZ_TREE_LOCK_NAME=A8two   bash "$TL" acquire --steal >"$W/a8.two"   2>&1; echo $? >"$W/a8.two.rc" ) &
P2=$!
# Wait for THESE two, not `wait`: the fixture holders above are background
# subshells sleeping out their lifetime, and a bare `wait` would sit on
# them - measured at 4 minutes of the gate's runtime for arms already
# finished.
wait "$P1" "$P2"
R1=$(cat "$W/a8.one.rc" 2>/dev/null || echo missing)
R2=$(cat "$W/a8.two.rc" 2>/dev/null || echo missing)
WINS=0
[ "$R1" = "0" ] && WINS=$((WINS + 1))
[ "$R2" = "0" ] && WINS=$((WINS + 1))
ok "$([ "$WINS" -eq 1 ] && echo PASS || echo FAIL)" \
   "A8: two simultaneous steals - exactly one wins (rc: one=$R1 two=$R2)"

# ------------------------------------------------------------------- A9
reset_lock
mkdir -p "$LD"
printf 'owner=half-written' > "$LD/owner"     # no newline, no other fields
bash "$TL" status >"$W/a9.status" 2>&1
if grep -qi "UNDECIDABLE" "$W/a9.status"; then
    ok PASS "A9: a half-written owner file reads as UNDECIDABLE"
elif grep -qi "free" "$W/a9.status"; then
    ok FAIL "A9: a half-written owner file read as FREE (a lock nobody can see)"
else
    ok FAIL "A9: a half-written owner file did not read as UNDECIDABLE"
fi
# And the shipping default is still the documented path: the scratch
# directory above must not have become the only path anybody runs.
reset_lock
grep -q '^LOCKDIR=${LZ_TREE_LOCK_DIR:-build/\.tree-lock\.d}$' "$TL" \
    && ok PASS "A9: the shipping default lock path is unchanged" \
    || ok FAIL "A9: the shipping default lock path is unchanged"
[ -d build/.tree-lock.d ] \
    && ok FAIL "A9: this gate created the SHIPPING lock" \
    || ok PASS "A9: this gate left the shipping lock untouched"

# ------------------------------------------------------------------- A11
# THE OUTER LOCK. Per-gate acquisition stops two gates colliding; it does
# not stop an outside writer slipping in BETWEEN two gates of a round -
# and from that point the round is reading a tree its later stages did not
# start from. So a locked round must refuse an outside acquire for its
# whole duration, and that is what this arm measures: a wrapper holds the
# lock, an outside acquire is refused WITH the holder named, and once the
# round ends the lock is free again.
reset_lock
LZ_TREE_LOCK_NAME="A11 round" bash build/with_tree_lock.sh sleep 3 >"$W/a11.round" 2>&1 &
R11=$!
# Wait for the ROUND to hold it, not for a fixed second: a sleep here is a
# race the machine's load decides, and this arm is about the lock, not
# about how fast the wrapper starts. Measured once: at 1 s the outside
# acquire beat the wrapper on a loaded machine and the arm went red.
for _i in $(seq 1 100); do
    [ -f "$LD/owner" ] && break
    sleep 0.1
done
LZ_TREE_LOCK_NAME=A11outside bash "$TL" acquire >"$W/a11.outside" 2>&1
rc11=$?
if [ "$rc11" -eq 2 ] && grep -q "REFUSED" "$W/a11.outside" &&
   grep -q "A11 round" "$W/a11.outside"; then
    ok PASS "A11: during a locked round an outside acquire is refused, and names the round"
else
    ok FAIL "A11: during a locked round an outside acquire is refused, and names the round (rc=$rc11)"
fi
wait "$R11"
bash "$TL" status >"$W/a11.after" 2>&1
grep -q "free" "$W/a11.after" \
    && ok PASS "A11: the wrapper releases when its command ends" \
    || ok FAIL "A11: the wrapper releases when its command ends"

# ------------------------------------------------------------------ A12
# THE CALLER SIDE, in two halves, because "the lock refused" and "the
# caller stopped" are different statements.
#
# Half one is the FORM the callers use. `eval "$(...)"` reports eval's
# own status: a refusal captures an empty string, eval of it returns 0,
# and `|| exit 2` never fires - the run proceeds with the lock held by
# nobody, which is the failure this script exists to prevent wearing the
# shape of an ordinary start. The capture form is asserted to stop AND
# the eval form is asserted to run, so the arm carries its own contrast:
# a predicate that both forms satisfy would measure nothing.
#
# Half two is WHICH CHANNEL the refusal's words arrive on. Under
# --print-env stdout is the line the caller evals, so a refusal puts
# nothing there; prose on stdout is a partial acquisition read as a
# whole one. Both refusal paths are checked - the first refusal, and the
# override, whose report is prose as well.
reset_lock
hold_start A12holder; H12=$HOLD_PID
sleep 1
LZ_TREE_LOCK_NAME=A12stdout bash "$TL" acquire --print-env >"$W/a12.out" 2>"$W/a12.err"
rc12s=$?
ok "$([ "$rc12s" -eq 2 ] && [ ! -s "$W/a12.out" ] && grep -q REFUSED "$W/a12.err" && echo PASS || echo FAIL)" \
   "A12: a refusal under --print-env writes nothing to stdout"
( LZ_ENV=$(LZ_TREE_LOCK_NAME=A12capture bash "$TL" acquire --print-env 2>/dev/null) || exit 2
  echo "RAN" ) >"$W/a12.capture" 2>&1
rc12c=$?
( eval "$(LZ_TREE_LOCK_NAME=A12eval bash "$TL" acquire --print-env 2>/dev/null)" || exit 2
  echo "RAN" ) >"$W/a12.eval" 2>&1
rc12e=$?
ok "$([ "$rc12c" -eq 2 ] && ! grep -q RAN "$W/a12.capture" && echo PASS || echo FAIL)" \
   "A12: a caller that captures the env line first stops on a refusal (rc=$rc12c)"
ok "$([ "$rc12e" -eq 0 ] && grep -q RAN "$W/a12.eval" && echo PASS || echo FAIL)" \
   "A12: the eval form is the one that does not stop (rc=$rc12e)"
kill -9 "$H12" 2>/dev/null; wait "$H12" 2>/dev/null
sleep 1
LZ_TREE_LOCK_NAME=A12thief bash "$TL" acquire --steal --print-env >"$W/a12.steal.out" 2>"$W/a12.steal.err"
if [ $? -eq 0 ] && [ "$(wc -l < "$W/a12.steal.out")" -eq 1 ] &&
   grep -q '^export LZ_TREE_LOCK=' "$W/a12.steal.out"; then
    ok PASS "A12: a steal under --print-env puts the export line alone on stdout"
else
    ok FAIL "A12: a steal under --print-env puts the export line alone on stdout"
fi

# ------------------------------------------------------------------ A13
# RELEASE BELONGS TO THE OPENER, unlike acquire. Every process the holder
# starts inherits LZ_TREE_LOCK, so a nested take passes through silently
# (A5) and the nested process carries the holder's token with it. A
# release that asked only "is this token current" therefore frees a claim
# the nested process never opened, and the holder runs on with the lock
# gone from under it - the shape measured on a gate whose `make` starts
# the build hooks, where the tree reads free for the whole build. A5
# covers the acquire half of re-entrancy; this arm is the release half,
# and it asserts both directions, because a predicate that refuses
# everyone would satisfy "the nested one is refused" while breaking the
# holder's own give-it-back.
#
# The fixture's own first spelling is worth recording, because it was red
# at the wrong arm: the holder redirected the acquire's stdout into a
# file rather than eval'ing the export line, so it held no token at all
# and the nested release was refused by the TOKEN arm. That refusal reads
# as this arm failing while what it tests - the opener check - had not
# been reached.
reset_lock
(
    # The holder takes the lock the way the call sites do, capture first:
    # a holder that never evals the export line has no token, and its
    # release is then refused by the token arm rather than by this one.
    LZ_ENV=$(LZ_TREE_LOCK_NAME=A13outer bash "$TL" acquire --print-env) || exit 2
    eval "$LZ_ENV"
    printf '%s\n' "$LZ_ENV" >"$W/a13.outer"
    # Hold until the nested attempt below is done, rather than for a fixed
    # second: how long the attempt takes is the machine's business, and a
    # release that beat it would test nothing.
    while [ ! -f "$W/a13.done" ]; do sleep 0.1; done
    bash "$TL" release >"$W/a13.holderRelease" 2>&1
) &
H13=$!
for _i in $(seq 1 100); do
    [ -f "$LD/owner" ] && break
    sleep 0.1
done
T13=$(sed -n 's/^owner=//p' "$LD/owner")
( LZ_TREE_LOCK="$T13" LZ_TREE_LOCK_NAME=A13nested bash "$TL" acquire >"$W/a13.nested" 2>&1 )
rc13n=$?
( LZ_TREE_LOCK="$T13" bash "$TL" release >"$W/a13.nestedRelease" 2>&1 )
rc13r=$?
ok "$([ "$rc13n" -eq 0 ] && [ ! -s "$W/a13.nested" ] && echo PASS || echo FAIL)" \
   "A13: a nested take inside a held lock still passes through (rc=$rc13n)"
if [ "$rc13r" -eq 0 ] && [ ! -s "$W/a13.nestedRelease" ] && [ -d "$LD" ] &&
   [ "$(sed -n 's/^owner=//p' "$LD/owner")" = "$T13" ]; then
    ok PASS "A13: a nested release gives nothing back, silently, and the holder keeps the lock"
else
    ok FAIL "A13: a nested release gives nothing back, silently, and the holder keeps the lock (rc=$rc13r)"
fi
touch "$W/a13.done"
wait "$H13"
if grep -q "released" "$W/a13.holderRelease" &&
   ! grep -q "REFUSED" "$W/a13.holderRelease" &&
   [ ! -d "$LD" ]; then
    ok PASS "A13: the holder's own release still gives the lock back"
else
    ok FAIL "A13: the holder's own release still gives the lock back"
fi

# ------------------------------------------------------------------ A10
# THE CLIENT LIST, and this arm is the reason the list is worth having: a
# lock honoured by some writers reads as mutual exclusion while the
# exclusion is not there - the unlisted writer runs, the listed one
# believes it is alone, and nothing reports the difference. So the entry
# points that touch a shared path are enumerated HERE, each either wired
# to the lock or carrying a reason it does not need one, and an entry that
# is neither fails this arm. The screen is deliberately the shared paths
# rather than "does it mention a compiler": a build in a scratch directory
# shares nothing and must not be serialized away.
SHARED_RE='build/native|build/gui|build/watcom|\./\*\.gcov'
# <file>: <reason>
# The fifteen that acquire are not listed here - this table is for the
# ones that must NOT, each with the reason it is safe without. A name
# here that starts acquiring is harmless (the census counts it as wired),
# but a name that STOPS and is still listed would go unnoticed, which is
# why the two directions are checked differently: wired is counted from
# the file, declared is read from this table.
CLIENTS='
arm/parity_gate.sh: runs through WSL, where this lock is not visible; its caller holds it
cov_run.sh: sourced by its callers - it runs in their shell and must not take one itself
'
UNLISTED=""
WIRED=0
DECLARED=0
for f in build/*.sh build/*/*.sh; do
    [ -f "$f" ] || continue
    rel=${f#build/}
    case "$rel" in tree_lock.sh|tree_lock_gate.sh) continue ;; esac
    grep -qE "$SHARED_RE" "$f" || continue
    if grep -q "tree_lock.sh acquire" "$f"; then
        WIRED=$((WIRED + 1))
        continue
    fi
    if printf '%s\n' "$CLIENTS" | grep -q "^$rel:"; then
        DECLARED=$((DECLARED + 1))
        continue
    fi
    UNLISTED="$UNLISTED $rel"
done
if [ -z "$UNLISTED" ]; then
    ok PASS "A10: every entry point touching a shared path is wired or declared"
else
    ok FAIL "A10: unlisted entry point(s) touching a shared path:$UNLISTED"
fi
echo "  A10 census: $WIRED wired, $DECLARED declared not-yet-wired, 0 unlisted"
printf '%s\n' "$CLIENTS" | grep -v '^$' | head -3 | sed 's/^/    e.g. /'

echo
if [ "$FAIL" -eq 0 ]; then
    echo "tree_lock_gate: PASS"
    exit 0
fi
echo "tree_lock_gate: FAIL"
exit 1
