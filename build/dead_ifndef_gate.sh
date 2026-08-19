#!/bin/bash
# Dead-#ifndef gate. A "#ifndef X / #define Y" whose X was already defined
# upstream is dead code: Y never takes effect, and reads as a lie about
# the default. Two forms, both caught by a cheap static check:
#
#   (a) a header already defines X  - the #ifndef in a .c that includes it
#       never fires. src/ops.h's "#ifndef LZ_ATTN_FIXED / #define 3" against
#       a stale src/ops.c "#ifndef LZ_ATTN_FIXED / #define 0" is the case
#       this gate exists for.
#   (b) one file #defines X twice  - the second #ifndef never fires.
#
# Both are grep-level: any header #define of X makes a same-named #ifndef
# elsewhere suspicious, and any file with two #define X lines is a duplicate
# regardless of value. Over-reporting is intended - a real duplicate costs
# a minute to confirm, a missed one silently lies about a default.
set -u
cd "$(dirname "$0")/.."

FAIL=0

# (a) header-defined macros vs #ifndef sites in .c files.
for c in src/*.c common/*.c; do
    [ -e "$c" ] || continue
    while IFS= read -r X; do
        case "$X" in *_H|*_H_) continue ;;  # include guards
        esac
        if grep -qE "^[[:space:]]*#define $X([[:space:]]|$)" src/*.h common/*.h 2>/dev/null; then
            echo "dead-ifndef: $c #ifndef $X, but a header already defines it"
            FAIL=1
        fi
    done < <(grep -oE '^[[:space:]]*#ifndef LZ_[A-Z_a-z0-9_]+' "$c" | awk '{print $2}')
done

# (b) a macro #ifndef'd twice in one file - the second #ifndef never fires.
# Keyed on #ifndef, NOT #define: an #ifdef/#else pair legitimately #defines
# the same name twice (one per branch), so counting #define over-reports.
for c in src/*.c common/*.c; do
    [ -e "$c" ] || continue
    for X in $(grep -oE '^[[:space:]]*#ifndef LZ_[A-Z_a-z0-9_]+' "$c" | awk '{print $2}' | sort | uniq -d); do
        echo "dead-ifndef (dup): $c has two #ifndef $X"
        FAIL=1
    done
done

if [ "$FAIL" -eq 0 ]; then
    echo "dead_ifndef_gate: PASS"
    exit 0
fi
echo "dead_ifndef_gate: FAIL"
exit 1
