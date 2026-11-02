#!/usr/bin/env python3
"""Count how close a render runs to full scale, per piece.

WHY THIS EXISTS AS A TOOL. The bank's clipping was being reported as
"clip98/95 = 47/62 against gm's 62/84" from numbers added up by hand in
one session, with no script that could produce them again - so the
figure could not be re-measured after the next change, and the peak
ceiling in ksb_pick.level_trim_from_rows was being tuned against it.
A criterion nobody can re-run is a number, not a gate.

WHAT IS COUNTED. A frame is "clipped" when |sample| is within one of
full scale, and "near" when it is within 2% of it. The 1% figure is
the audible one - a run that touches 32767 once is not the problem,
a run that sits at 32700 for a second is - so both are printed and
the piece is judged on the pair, not on either alone.

THE COMPARISON IS AGAINST THE REFERENCE ON THE SAME PIECE. gm clipping
is the bound this bank is held to: being under it is headroom, being
over it is a regression. A count with nothing to compare it to cannot
say which.
"""
import os
import sys
import wave

import numpy as np

FULL = 32768.0
CLIP = FULL - 1.0          # touching the rail
NEAR = 0.98 * FULL         # within 2% of it


def stats(path):
    w = wave.open(path, 'rb')
    n, ch = w.getnframes(), w.getnchannels()
    a = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float64)
    w.close()
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    peak = float(np.abs(a).max()) if len(a) else 0.0
    return {
        'frames': len(a),
        'peak': peak,
        'clip': int((np.abs(a) >= CLIP).sum()),
        'near': int((np.abs(a) >= NEAR).sum()),
        'rms': float(np.sqrt((a * a).mean())) if len(a) else 0.0,
    }


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    ref = None
    paths = []
    for a in argv[1:]:
        if a.startswith('--ref='):
            ref = a[6:]
        else:
            paths.append(a)
    rs = stats(ref) if ref and os.path.exists(ref) else None
    print('%-28s %10s %8s %8s %8s' % ('piece', 'frames', 'peak', 'clip', 'near'))
    if rs:
        print('%-28s %10d %8.1f %8d %8d' % ('REF ' + os.path.basename(ref),
                                            rs['frames'], rs['peak'],
                                            rs['clip'], rs['near']))
    tot_c = tot_n = 0
    for p in paths:
        if not os.path.exists(p):
            print('%-28s MISSING' % os.path.basename(p))
            continue
        s = stats(p)
        tot_c += s['clip']
        tot_n += s['near']
        print('%-28s %10d %8.1f %8d %8d' % (os.path.basename(p), s['frames'],
                                            s['peak'], s['clip'], s['near']))
    print('total clip %d  near %d' % (tot_c, tot_n))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
