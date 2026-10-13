"""Render one note from a DLS instrument to a WAV.

    python dls_play.py <bank.dls> <gm program> <midi note> <ms> <out.wav>

This is the part of a sampler that decides which bytes to play and how
fast, and nothing else: region select, resample, loop. No envelope, no
filter, no velocity - so what comes out is the recorded sample at the
asked-for pitch, which is what has to be right before any of the rest
means anything.

PITCH COMES FROM THE REGION, NOT THE WAVE. Both carry a `wsmp` giving
the note the sample was recorded at, and the region's overrides the
wave's when it is there. Reading only the wave's puts multisampled
instruments out by whatever the two disagree by, which for this bank is
up to an octave on some regions and nothing on others - the kind of
error that sounds like a bad sample rather than a bad reader.
"""
import struct
import sys
import wave

from dls_survey import chunks, find

OUT_RATE = 44100


def wsmp_of(d, a, b):
    """-> (unity note, fine cents, gain dB, loop start, loop len) or None."""
    ws_a, ws_b = find(d, a, b, b'wsmp')
    if ws_a is None:
        return None
    cb, unity, fine, gain, opt, nloop = struct.unpack('<IHhiII',
                                                      d[ws_a:ws_a + 20])
    ls = ll = 0
    if nloop:
        _, _, ls, ll = struct.unpack('<IIII', d[ws_a + cb:ws_a + cb + 16])
    # lGain is in DLS relative gain units, a 655360th of a decibel -
    # NOT the 16.16 dB it looks like. Read as 16.16 the FF8 bank's
    # trims come out as -10 to -60 dB and its harp renders silent;
    # read correctly every one of them is a round -1.0, -2.0, -2.5,
    # -3.0, -5.0 or -6.0 dB, which is what a per-sample trim is.
    return unity, fine, gain / 655360.0, ls, ll


def load(path):
    d = open(path, 'rb').read()
    top = 8 + struct.unpack('<I', d[4:8])[0]
    lins = find(d, 12, top, b'LIST', b'lins')
    wvpl_a, _ = find(d, 12, top, b'LIST', b'wvpl')
    pt_a, _ = find(d, 12, top, b'ptbl')
    cb, ncue = struct.unpack('<II', d[pt_a:pt_a + 8])
    cues = struct.unpack('<%dI' % ncue, d[pt_a + cb:pt_a + cb + 4 * ncue])
    return d, lins, wvpl_a, cues


def region(d, lins, prog, note, vel=100):
    """The region of bank-0 program `prog` that covers `note`."""
    for cid, f, a, b in chunks(d, lins[0], lins[1]):
        if cid != b'LIST' or f != b'ins ':
            continue
        ih_a, _ = find(d, a, b, b'insh')
        if ih_a is None:
            continue
        _, bank, p = struct.unpack('<III', d[ih_a:ih_a + 12])
        if p & 0x80000000 or bank != 0 or (p & 0x7F) != prog:
            continue
        lr_a, lr_b = find(d, a, b, b'LIST', b'lrgn')
        for c2, f2, r_a, r_b in chunks(d, lr_a, lr_b):
            if c2 != b'LIST' or f2 not in (b'rgn ', b'rgn2'):
                continue
            rh_a, _ = find(d, r_a, r_b, b'rgnh')
            klo, khi, vlo, vhi = struct.unpack('<HHHH', d[rh_a:rh_a + 8])
            if klo <= note <= khi and (vlo > vhi or vlo <= vel <= vhi):
                wl_a, _ = find(d, r_a, r_b, b'wlnk')
                idx = struct.unpack('<I', d[wl_a + 8:wl_a + 12])[0]
                return idx, wsmp_of(d, r_a, r_b), (klo, khi)
    return None


def pcm(d, wvpl_a, cues, idx):
    """-> samples as floats, rate, the wave's own wsmp."""
    a = wvpl_a + cues[idx]
    b = a + 8 + struct.unpack('<I', d[a + 4:a + 8])[0]
    fa, _ = find(d, a + 12, b, b'fmt ')
    tag, ch, rate, _, _, bits = struct.unpack('<HHIIHH', d[fa:fa + 16])
    if tag != 1 or bits != 16:
        raise SystemExit('wave %d is not 16-bit PCM' % idx)
    da, db = find(d, a + 12, b, b'data')
    n = (db - da) // (2 * ch)
    raw = struct.unpack('<%dh' % (n * ch), d[da:da + n * ch * 2])
    return [raw[i * ch] / 32768.0 for i in range(n)], rate, \
        wsmp_of(d, a + 12, b)


def render(x, rate, ws, note, ms):
    unity, fine, gain, ls, ll = ws
    step = (rate / float(OUT_RATE)) * 2.0 ** ((note - unity + fine / 100.0)
                                              / 12.0)
    amp = 10.0 ** (gain / 20.0)
    n = int(OUT_RATE * ms / 1000.0)
    # Interpolating the last frame of a loop towards the frame after it
    # steps out of the loop once per cycle, and where the loop ends at
    # the buffer's end there is no such frame at all. Its neighbour is
    # the loop start.
    end = min(len(x), ls + ll) if ll > 1 else len(x)
    out, pos = [], 0.0
    for _ in range(n):
        if pos >= end:
            if ll <= 1:
                break
            pos = ls + (pos - ls) % ll
        i = int(pos)
        f = pos - i
        nxt = x[i + 1] if i + 1 < end else (x[ls] if ll > 1 else x[i])
        out.append((x[i] * (1.0 - f) + nxt * f) * amp)
        pos += step
    return out


def main(argv):
    if len(argv) < 6:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    prog, note, ms, out = int(argv[2]), int(argv[3]), float(argv[4]), argv[5]
    d, lins, wvpl_a, cues = load(argv[1])
    r = region(d, lins, prog, note)
    if r is None:
        print('program %d has no region covering note %d' % (prog, note))
        return 1
    idx, rws, keys = r
    x, rate, wws = pcm(d, wvpl_a, cues, idx)
    ws = rws if rws is not None else wws
    y = render(x, rate, ws, note, ms)
    w = wave.open(out, 'wb')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(OUT_RATE)
    w.writeframes(struct.pack('<%dh' % len(y),
                              *[max(-32768, min(32767, int(v * 32767)))
                                for v in y]))
    w.close()
    print('prog %d note %d -> wave %d, keys %d-%d, %d Hz, unity %d fine %+d'
          % (prog, note, idx, keys[0], keys[1], rate, ws[0], ws[1]))
    print('  %s: %.2f s' % (out, len(y) / float(OUT_RATE)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
