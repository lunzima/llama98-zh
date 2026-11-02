#!/usr/bin/env python3
"""How far apart the velocity bands of one key zone sit, per bank.

WHY THIS IS A CRITERION AT ALL. gm.dls gives all 129 of its programs a
single 0-127 velocity band per key zone: the velocity response is the
engine's curve and nothing else. This bank inherits up to 22 bands per
zone from the source DLS and the calibration has been writing each one
its own gain, so the layer differences are the calibration's own work -
p75's keys 92-124 ran -287, +99, -181, -14, -77, -73, -72, -25, +107
in tenths, a 40 dB sawtooth where the reference has one number. The
listener hears exactly that: keys 118-127 at +14.5 dB on velocity 32
and -5 dB on velocity 110.

SO THE NUMBER TO WATCH IS THE BAND GAIN SPREAD, and it needs no
rendering: it is read straight off the bank, and the reference's value
is 0.0 by construction. A zone that shows anything is a zone where the
calibration wrote a layer the reference does not have.

Overlapping bands are skipped - two regions sounding at once are a
stacked instrument, and their gains are a mix decision rather than a
level one.

Usage:
  python3 tools/bandspread.py banks/picked.ksb banks/gm.ksb
"""
import collections
import struct
import sys


def zones(path):
    d = open(path, 'rb').read()
    n, nr, ns, ver, doff = struct.unpack_from('<HHHHI', d, 4)
    ro = 16 + n * 7
    by = collections.defaultdict(list)
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from('<BBBHH', d, 16 + i * 7)
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi, vlo, vhi = struct.unpack_from('<HHHH', d, o)
            g = struct.unpack_from('<i', d, o + 20)[0]
            by[(prog, msb, lsb, klo, min(khi, 127))].append(((vlo, vhi), g))
    out = []
    for (prog, msb, lsb, klo, khi), rs in by.items():
        bands = sorted(set(b for b, _ in rs))
        if len(bands) < 2:
            continue
        if not all(bands[t][1] < bands[t + 1][0] for t in range(len(bands) - 1)):
            continue
        gs = [g for _, g in rs]
        out.append(((max(gs) - min(gs)) / 10.0, prog, klo, khi, len(bands)))
    return out


def main(argv):
    paths = [a for a in argv[1:] if not a.startswith('-')]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        z = zones(p)
        z.sort(reverse=True)
        over = [t for t in z if t[0] > 1.0]
        name = p.replace('\\', '/').split('/')[-1]
        print('%-28s %d multi-band zone(s), %d over 1.0 dB, worst %s'
              % (name, len(z), len(over),
                 ('%.1f dB' % z[0][0]) if z else 'none'))
        for s, prog, klo, khi, nb in z[:5]:
            print('      p%-4d keys %3d-%3d  %d bands  spread %.1f dB'
                  % (prog, klo, khi, nb, s))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
