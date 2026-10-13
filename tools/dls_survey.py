"""Read a DLS bank and say what is in it, and what a subset would cost.

The point is to keep only the instruments actually played. A DLS is a
RIFF tree: `lins` holds one `ins` per instrument, each with an `insh`
naming its bank and program and a `lrgn` of regions; every region's
`wlnk` points into the wave pool `wvpl`, which is a flat list of `wave`
chunks. So the instruments wanted select a set of pool indices, and
those select the bytes that have to be kept.

    python dls_survey.py <bank.dls> [gm program ...]

With no programs it lists everything; with programs it reports what
they reference and what fraction of the file that is.
"""
import struct
import sys

# The menu's ten first, then the programs it has held before, because
# this file is also read while looking at what a slot used to be.
GM_NAMES = {0: 'acoustic grand', 4: 'electric piano', 6: 'harpsichord',
            7: 'clavinet', 12: 'marimba', 16: 'drawbar organ',
            25: 'steel guitar', 46: 'orchestral harp', 80: 'square lead',
            81: 'saw lead',
            38: 'synth bass', 45: 'pizzicato', 87: 'lead 8 bass+lead'}


def chunks(d, start, end):
    """Walk RIFF chunks in [start, end) -> (id, form, body_start, body_end)."""
    i = start
    while i + 8 <= end:
        cid = d[i:i + 4]
        n = struct.unpack('<I', d[i + 4:i + 8])[0]
        body = i + 8
        if cid in (b'LIST', b'RIFF'):
            yield cid, d[body:body + 4], body + 4, body + n
        else:
            yield cid, None, body, body + n
        i = body + n + (n & 1)


def find(d, start, end, want, form=None):
    for cid, f, a, b in chunks(d, start, end):
        if cid == want and (form is None or f == form):
            return a, b
    return None, None


def read(path):
    d = open(path, 'rb').read()
    if d[:4] != b'RIFF' or d[8:12] != b'DLS ':
        raise SystemExit('%s: not a DLS' % path)
    top_a, top_b = 12, 8 + struct.unpack('<I', d[4:8])[0]

    lins_a, lins_b = find(d, top_a, top_b, b'LIST', b'lins')
    wvpl_a, wvpl_b = find(d, top_a, top_b, b'LIST', b'wvpl')

    # the wave pool, in order, with each entry's extent
    pool = []
    if wvpl_a is not None:
        for cid, f, a, b in chunks(d, wvpl_a, wvpl_b):
            if cid == b'LIST' and f == b'wave':
                pool.append((a - 12, b))     # include the LIST header

    ins = []
    for cid, f, a, b in chunks(d, lins_a, lins_b):
        if cid != b'LIST' or f != b'ins ':
            continue
        ih_a, ih_b = find(d, a, b, b'insh')
        if ih_a is None:
            continue
        nreg, bank, prog = struct.unpack('<III', d[ih_a:ih_a + 12])
        used = []
        lr_a, lr_b = find(d, a, b, b'LIST', b'lrgn')
        if lr_a is not None:
            for c2, f2, r_a, r_b in chunks(d, lr_a, lr_b):
                if c2 != b'LIST' or f2 not in (b'rgn ', b'rgn2'):
                    continue
                wl_a, wl_b = find(d, r_a, r_b, b'wlnk')
                if wl_a is not None:
                    used.append(struct.unpack('<I', d[wl_a + 8:wl_a + 12])[0])
        name = ''
        inf_a, inf_b = find(d, a, b, b'LIST', b'INFO')
        if inf_a is not None:
            n_a, n_b = find(d, inf_a, inf_b, b'INAM')
            if n_a is not None:
                name = d[n_a:n_b].split(b'\x00')[0].decode('latin-1')
        ins.append((bank, prog & 0x7F, bool(prog & 0x80000000), name,
                    nreg, used))
    return d, ins, pool


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1].strip())
        return 2
    want = set(int(x) for x in argv[2:])
    d, ins, pool = read(argv[1])
    total = len(d)
    pool_bytes = sum(b - a for a, b in pool)
    print('%s: %.1f MB, %d instruments, %d waves (%.1f MB of pool)'
          % (argv[1], total / 1e6, len(ins), len(pool), pool_bytes / 1e6))
    if not want:
        for bank, prog, drum, name, nreg, used in ins[:40]:
            print('  bank %-5d prog %-3d %-24s %d regions%s'
                  % (bank, prog, name, nreg, ' [drums]' if drum else ''))
        if len(ins) > 40:
            print('  ... and %d more' % (len(ins) - 40))
        return 0

    # BANK 0 ONLY. This DLS carries a GM-compatible bank 0 and then a
    # bank per song - 21504, 22784 and so on - holding that song's own
    # voicings under the same program numbers. Matching on the program
    # alone returns seven different square leads and no way to choose.
    keep, waves = [], set()
    for bank, prog, drum, name, nreg, used in ins:
        if drum or bank != 0 or prog not in want:
            continue
        keep.append((prog, name, nreg, used))
        waves |= set(used)
    keep.sort()
    print('  %-3s %-24s %-8s %-6s %s'
          % ('GM', 'name in the bank', 'regions', 'waves', 'what we call it'))
    for prog, name, nreg, used in keep:
        print('  %-3d %-24s %-8d %-6d %s'
              % (prog, name, nreg, len(set(used)),
                 GM_NAMES.get(prog, '')))
    missing = want - set(p for p, _, _, _ in keep)
    for p in sorted(missing):
        print('  %-3d %-24s MISSING from this bank' % (p, GM_NAMES.get(p, '')))
    kept = sum(pool[i][1] - pool[i][0] for i in sorted(waves) if i < len(pool))
    print('  %d of %d waves, %.1f MB of %.1f MB (%.0f%% of the pool)'
          % (len(waves), len(pool), kept / 1e6, pool_bytes / 1e6,
             100.0 * kept / pool_bytes if pool_bytes else 0))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
