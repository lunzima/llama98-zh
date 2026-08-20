#!/bin/bash
# Bare-#endif gate.
#
# Every #endif in the engine sources must carry a trailing comment naming
# the macro it closes. The annotation is a regression guard, not a style
# choice: ops.c's 49 bare #endif were annotated once (acef096), and the
# split into ops_*.c immediately re-introduced bare ones in the moved
# blocks - a #endif with no comment gives a reader no way to know which
# #if it closes without re-counting the whole nesting stack, and a
# re-indent or an inserted block makes the mismatch invisible to a
# compiler that only reports "unbalanced #endif".
#
# The check is the awk from the gate's own spec, asserted empty: any
# line that is nothing but an #endif (optionally surrounded by
# whitespace - [[:space:]] also covers the CR of the CRLF sources, so a
# bare CR-terminated #endif is still caught) is a FAIL. A commented
# #endif has non-whitespace after the directive and does not match.
#
# Needs no build, no disassembly - runs in milliseconds, before either
# compiler starts.
#
# Exit codes:
#   0  PASS - every #endif in src/*.c common/*.c has a trailing comment
#   1  FAIL - at least one bare #endif
set -u
cd "$(dirname "$0")/.."

HITS=$(awk '/^[[:space:]]*#endif[[:space:]]*$/{print FILENAME":"FNR": bare #endif"}' src/*.c common/*.c)
if [ -n "$HITS" ]; then
    echo "$HITS"
    echo "bare_endif_gate: FAIL"
    exit 1
fi
echo "bare_endif_gate: PASS"
exit 0
