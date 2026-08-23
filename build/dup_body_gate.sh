#!/bin/bash
# Byte-identical function BODIES across src/, found mechanically.
#
# build/dedupe_gate.sh checks five names it was told about. This one
# does not need to be told: it hashes every function body in src/ and
# reports collisions, so a copy that arrives under a NEW name - the
# shape dedupe_gate's renamed-copy check was added for after the fact -
# reddens on the day it lands rather than when somebody notices.
#
# THE BODY IS HASHED, NEVER THE HEADER. Two identical bodies under
# different names must collide; including the name guarantees they never
# do, and that is exactly the case worth catching.
#
# THE INSTRUMENT CHECKS ITSELF FIRST. q8_fixed_epi/q16_fixed_epi are a
# known, deliberate, documented duplicate pair; if the scan stops
# finding them the gate fails as BLIND rather than passing as clean.
# Two earlier versions of this scanner passed while seeing nothing:
# one collapsed multi-line comments to a single space (which glues
# definitions onto the previous line in a tree this comment-dense, so
# almost no header matched), the other captured one letter of the name
# and hashed the header along with the body. A scan with no positive
# control is how both got as far as "no duplicates found".
#
# Exit codes:
#   0  PASS - the control was found and every collision is allowlisted
#   1  FAIL - a collision outside the allowlist, or the control missing
#   2  SKIP - no python3
set -u
cd "$(dirname "$0")/.."

command -v python3 >/dev/null 2>&1 || {
    echo "dup_body_gate: SKIP - no python3"; exit 2; }

# Allowlisted collisions, one "name+name" per line, names sorted. Each
# needs a reason here, not just an entry.
#
#   q16_fixed_epi+q8_fixed_epi
#       Deliberate and documented at epi_q41_fixed: the two predicates
#       are derived from different accumulator bounds (Q8_0's signed
#       ~2^27 against the unsigned tiers' <2^26) and must stay free to
#       diverge. Also this gate's positive control.
#   lz_epi_mac+lz_epi_mac_i16
#       Same body text, different PARAMETER TYPES (int32 vs int16 second
#       operand), so the bodies read different widths. A body-only hash
#       cannot see that; not a duplicate.
#   qerr+sterr
#       Module-local error formatters in model.c and safetensors.c.
#   p2_blk+p2_blk+p2_blk_sse2
#       NOT deliberate - carried here so this gate can land green while
#       the merge is done separately. p2_blk is defined twice in
#       ops_mmx.c, once in each arm of the Watcom/gcc split, and the
#       body is pure C with no intrinsic or asm in it: the split buys
#       nothing. p2_blk_sse2 is a third copy under another name, and
#       that one may have to stay - it is a SEPARATE static buffer, so
#       merging it changes which buffer a caller gets, which is
#       semantics rather than tidying.
ALLOW="q16_fixed_epi+q8_fixed_epi
lz_epi_mac+lz_epi_mac_i16
qerr+sterr
p2_blk+p2_blk+p2_blk_sse2"

python3 - "$ALLOW" <<'PY'
import re, sys, os, hashlib

allow = set(l.strip() for l in sys.argv[1].split('\n') if l.strip())
files = [os.path.join('src', f) for f in sorted(os.listdir('src'))
         if f.endswith('.c') or f.endswith('.h')]

def strip_comments(s):
    s = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'),
               s, flags=re.S)
    return re.sub(r'//[^\n]*', '', s)

hdr = re.compile(r'^[A-Za-z_][^;=]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*$')
bodies = {}
for path in files:
    lines = strip_comments(open(path, 'r', errors='replace').read()).split('\n')
    i = 0
    while i < len(lines):
        m = hdr.match(lines[i])
        if not m:
            i += 1; continue
        depth, j, started, buf = 0, i, False, []
        while j < len(lines) and j < i + 400:
            depth += lines[j].count('{') - lines[j].count('}')
            if '{' in lines[j] and not started:
                buf.append(lines[j][lines[j].index('{'):]); started = True
            elif started:
                buf.append(lines[j])
            if started and depth <= 0:
                break
            j += 1
        if not started or depth > 0:
            i += 1; continue
        norm = re.sub(r'\s+', ' ', '\n'.join(buf)).strip()
        if len(norm) > 100:
            k = hashlib.sha1(norm.encode()).hexdigest()[:12]
            bodies.setdefault(k, []).append((os.path.basename(path),
                                             m.group(1), len(norm)))
        i = j + 1

dups = {k: v for k, v in bodies.items()
        if len(set((f, n) for f, n, _ in v)) > 1 or len(v) > 1}
ctl = any(set(n for _, n, _ in v) == {'q8_fixed_epi', 'q16_fixed_epi'}
          for v in dups.values())
print('scanned %d files, %d bodies > 100 chars, %d collision group(s)'
      % (len(files), len(bodies), len(dups)))
fail = 0
if not ctl:
    print('FAIL: positive control q8_fixed_epi/q16_fixed_epi NOT FOUND -'
          ' the scan is blind, its "clean" means nothing')
    fail = 1
for k, v in sorted(dups.items(), key=lambda kv: -kv[1][0][2]):
    key = '+'.join(sorted(n for _, n, _ in v))
    if key in allow:
        print('OK:   %s (allowlisted)' % key)
    else:
        print('FAIL: %s - %d copies, %d chars, not allowlisted'
              % (key, len(v), v[0][2]))
        for f, n, _ in v:
            print('        %-22s %s' % (f, n))
        fail = 1
print('dup_body_gate: %s' % ('FAIL' if fail else 'PASS'))
sys.exit(1 if fail else 0)
PY
