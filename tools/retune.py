"""Apply a measured per-program tuning correction to a KSB bank.

    python retune.py banks/picked.ksb 52=+11 58=+18 20=-18 84=-15

Each delta is in cents and is added to every region's fine-tune of
that program, so layered regions keep their relative detune. The
values come from rendering A4 through the bank and measuring the
result; the engine's semitone spacing is exact (note 68->69 measured
99.9 cents), so a whole-program offset is the sample's own pitch.
"""
import struct
import sys

REG = 104


def main(argv):
    if len(argv) < 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    path = argv[1]
    want = {}
    for a in argv[2:]:
        k, v = a.split('=')
        want[int(k)] = int(v)
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', d, 4)
    io = 16
    ro = io + n * 7
    hit = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from('<BBBHH', d, io + i * 7)
        if prog not in want:
            continue
        delta = want[prog]
        for j in range(cnt):
            o = ro + (first + j) * REG
            fine = struct.unpack_from('<h', d, o + 18)[0]
            struct.pack_into('<h', d, o + 18,
                             max(-32767, min(32767, fine + delta)))
            hit += 1
    open(path, 'wb').write(bytes(d))
    print('%s: %d region(s) retuned for %s' %
          (path, hit, ', '.join('p%d %+d' % kv for kv in sorted(want.items()))))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
