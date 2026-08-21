#!/bin/bash
# Turns a raw-counter run's output into the rawfix table classify.awk and
# rollup.awk read. Slots the run never touched are written as 0, so a
# gcov block-sum cannot leak a count into a line that never ran.
# Reads and measures only.  $1 = raw-counter run stdout, $2 = output file
set -u
cd "$(dirname "$0")"
awk -v out="$2" '
FILENAME == ARGV[1] {
    if ($0 ~ /^#/ || $0 == "") next
    n = split($0, f, "\t")
    slot[f[1] + 0] = f[2] "\t" f[3]
    next
}
/^  rawc\[/ {
    s = $1; gsub(/[^0-9]/, "", s)
    v[s + 0] = $2 + 0
}
END {
    for (k in slot) printf "%s\t%d\n", slot[k], v[k] + 0 > out
}
' slotmap.tsv "$1"
echo "wrote $2"
