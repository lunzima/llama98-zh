#!/bin/bash
# f16 safetensors: the SUBNORMAL band is converted exactly.
#
# Why this gate exists: safetensors.c carried its own f16->f32 arithmetic
# and placed a subnormal's exponent at 127 - 15 - shift instead of
# 127 - 14 - shift, halving every value in [2^-24, 2^-14) at load with no
# error and no other symptom. Normal values, +-0, Inf and NaN were right,
# so nothing short of a value in that band could catch it.
#
# The criterion is the OUTCOME, not the source: a tiny real .safetensors
# file holding one F16 tensor whose entries span the subnormal band,
# loaded through the same lz_st_open/lz_st_read_f32 the engine's loader
# calls, compared bit for bit against Python's ieee754 decoding.
#
# Exit: 0 PASS, 1 FAIL, 2 SKIP (no gcc or no python).
set -u
cd "$(dirname "$0")/.."

command -v gcc >/dev/null 2>&1 || { echo "st_f16_gate: SKIP - no gcc"; exit 2; }
command -v python >/dev/null 2>&1 || { echo "st_f16_gate: SKIP - no python"; exit 2; }

TMP=build/_st_f16_gate
mkdir -p "$TMP"

python - "$TMP" <<'PY'
import json, struct, sys
d = sys.argv[1]
vals = [0x0000, 0x8000, 0x0001, 0x0002, 0x0007, 0x03FF, 0x0400,
        0x3555, 0xB555, 0x3C00, 0x7BFF, 0x7C00, 0xFC00, 0x7E00]
data = struct.pack('<%dH' % len(vals), *vals)
hdr = json.dumps({"w": {"dtype": "F16", "shape": [len(vals)],
                        "data_offsets": [0, len(data)]}}).encode()
open(d + '/f16.safetensors', 'wb').write(struct.pack('<Q', len(hdr)) + hdr + data)
want = []
for h in vals:
    f = struct.unpack('<e', struct.pack('<H', h))[0]
    want.append('%08x' % struct.unpack('<I', struct.pack('<f', f))[0])
open(d + '/want.txt', 'w').write('\n'.join(want) + '\n')
PY

cat > "$TMP/probe.c" <<'C'
#include <stdio.h>
#include <string.h>
#include "safetensors.h"

int main(int argc, char **argv) {
    LZSafetensors st;
    const LZStTensor *t;
    char err[256];
    float buf[64];
    int i;
    if (argc < 2) return 2;
    if (lz_st_open(&st, argv[1], err, (int)sizeof err) != 0) {
        printf("open failed: %s\n", err);
        return 2;
    }
    t = lz_st_find(&st, "w");
    if (!t) { printf("tensor w missing\n"); lz_st_close(&st); return 2; }
    if (lz_st_read_f32(&st, t, buf, t->n_elem, err, (int)sizeof err) != 0) {
        printf("read failed: %s\n", err);
        lz_st_close(&st);
        return 2;
    }
    for (i = 0; i < (int)t->n_elem; i++) {
        unsigned int u;
        memcpy(&u, &buf[i], 4);
        printf("%08x\n", u);
    }
    lz_st_close(&st);
    return 0;
}
C

gcc -O2 -Isrc -o "$TMP/probe.exe" "$TMP/probe.c" \
    src/safetensors.c src/json.c src/compat.c src/err.c 2>"$TMP/cc.log" || {
    echo "st_f16_gate: FAIL - probe did not compile"; sed 's/^/      /' "$TMP/cc.log"; exit 1; }

"$TMP/probe.exe" "$TMP/f16.safetensors" > "$TMP/got.txt" || {
    echo "st_f16_gate: FAIL - probe failed"; cat "$TMP/got.txt"; exit 1; }

if diff -u "$TMP/want.txt" "$TMP/got.txt" > "$TMP/diff.txt"; then
    echo "st_f16_gate: PASS ($(grep -c . "$TMP/want.txt") values, subnormals included)"
    exit 0
fi
echo "st_f16_gate: FAIL - f16 decode differs from ieee754"
sed 's/^/      /' "$TMP/diff.txt"
exit 1
