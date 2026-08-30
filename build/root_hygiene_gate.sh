#!/bin/bash
# The repository root holds an enumerated set of entries and nothing
# else. Anything not on the list is a defect, whether git can see it or
# not.
#
# WHY THE ROOT AND NOT THE WHOLE TREE. A file that lands somewhere under
# src/ is obviously misplaced and gets noticed. A file that lands in the
# ROOT is indistinguishable from a top-level project file - it sits next
# to Makefile and README.md and reads as one of them. make_lz4.py, the
# model packer, sat there untracked and was read as "something the
# release has not committed yet"; it reached the public remote twice
# before anyone asked why a model-producing tool was in an inference
# engine. The list below is what CLAUDE.md's Iron law zero enumerates,
# plus the local-private entries and the build products.
#
# WHAT PUTS THINGS HERE is not carelessness with `mv`. It is cwd: an
# ad-hoc compile run from the root leaves foo.obj and foo.err beside the
# Makefile, a probe writes its dump to a relative path, a pytest run in
# the wrong tree drops .pytest_cache. Those are unavoidable while the
# work happens here; what is avoidable is their SURVIVING, which is what
# this gate is for.
#
# GIT STATUS IS NOT THE TEST, and .pytest_cache is why. pytest writes a
# .gitignore containing `*` INSIDE its own cache directory, so the
# directory is invisible to `git status` while being entirely untracked
# and entirely unowned. A gate that asked git what is untracked would
# have reported a clean root over it. This one reads the directory.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || { echo "root_hygiene_gate: FAIL - cannot cd to $ROOT"; exit 1; }

# The public release tree, per CLAUDE.md Iron law zero. A name is added
# here only when it is decided that the entry SHIPS - not when it
# appears.
SHIPPED="
.gitignore
LICENSE
Makefile
README.md
assets
build
common
contrib
gui
llama98.def
src
"

# Local-private: present in this working tree, excluded from the push.
# build/ and .prof/ hold the gates and probes; docs/ the design notes;
# gates.mk the verification targets; CLAUDE.md the working rules.
LOCAL="
.claude
.git
.prof
CLAUDE.md
docs
gates.mk
"

# Build products the Makefile writes here by design, and the runtime ini
# each binary writes beside itself. Both are in .gitignore.
PRODUCTS="
llama98
llama98.exe
llama98-w.exe
llama98-dos.exe
llama98.dll
kunkun98-gui
kunkun98-gui.exe
kunkun98-w.exe
kunkun98-serve
kunkun98-serve.exe
kunkun98.ini
kunkun98-gui.ini
kunkun98-w.ini
"

ALLOWED=$(printf '%s\n%s\n%s\n' "$SHIPPED" "$LOCAL" "$PRODUCTS" | grep -v '^$' | sort -u)

# ls -A, not git: see the .pytest_cache note above.
ACTUAL=$(ls -A | sort)

STRAY=$(comm -13 <(printf '%s\n' "$ALLOWED") <(printf '%s\n' "$ACTUAL"))

if [ -n "$STRAY" ]; then
    echo "root_hygiene_gate: FAIL - entries in the repository root that are on no list:"
    printf '  %s\n' $STRAY
    echo "Each is one of three things. A DROPPING (an .obj, .err, a cache"
    echo "directory) - delete it. A file that belongs somewhere else - move"
    echo "it to the directory whose code reads it. A new top-level entry"
    echo "that genuinely ships - add it to SHIPPED here AND to Iron law"
    echo "zero in CLAUDE.md, in that order, deliberately."
    exit 1
fi

# The other direction: a SHIPPED entry that has gone missing is a
# different failure with the same list. Not checked for LOCAL (a fresh
# clone has neither) or PRODUCTS (absent until built).
MISSING=""
for e in $(printf '%s\n' "$SHIPPED" | grep -v '^$'); do
    [ -e "$e" ] || MISSING="$MISSING $e"
done
if [ -n "$MISSING" ]; then
    echo "root_hygiene_gate: FAIL - shipped entries missing from the root:$MISSING"
    exit 1
fi

echo "root_hygiene_gate: PASS ($(printf '%s\n' "$ACTUAL" | grep -c .) entries, all enumerated)"
