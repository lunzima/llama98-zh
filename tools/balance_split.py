"""Hold one instrument's loudness across its own keys and layers.

WHY THIS EXISTS. calib and balance_regions both write ONE gain per
region, and they can only be as good as the regions they were shown.
cal_keys samples a region's two ends and the end farthest from its
unity, which finds a boundary error and cannot see a dip in the middle:
p64's region 0-50 held keys 0 through 50 at one gain, key 0 matched gm
to 0.1 dB and keys 12 through 50 sat 7 to 21 dB under it. split_regions
then cut where a deviation crosses the loudness bound, but with only
the two ends measured the crossing was found at key 49 and the cut left
0-49 as one piece - the dip was still inside a single gain.

BOTH AXES ARE MEASURED, AND EACH REGION IS ITS OWN PROBE. A region is a
(key range, velocity range) pair with one gain; there is nothing to
learn from playing it at three velocities, and a lot to lose by playing
the whole range at three: p64's vel 55-68, 96-108 and 121-127 regions
matched none of 40, 85 and 120, so they were never measured and calib
left them where they stood. Here every region is played at the
midpoint of its own velocity range, at every sampled key it covers, so
no layer is invisible and the note count is close to the key count
instead of three times it.

THE WINDOW COMES FROM THE REFERENCE. A fixed slice is the same number
of seconds on both banks and not the same part of the note - a sample
played 29 semitones down is still in its attack at 0.45 s - so each
note is measured over gm's own span, capped so the next note of the
sweep cannot leak in.

NO NEW CORRECTION RULE. The grid goes to the pipeline's own
split_regions (which cuts where the deviation turns) and calib (which
levels each piece and keeps KEEP of any deviation inside
KEEP_BAND_DB, so character survives).

USAGE
    python tools/balance_split.py banks/picked.ksb [--gap 6] [--apply]

Without --apply it measures and reports only.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from note_window import note_window  # noqa: E402
import ksb_pick as kp  # noqa: E402

PLAYER = os.path.join(HERE, 'kunkun98-play.exe')
REF = os.path.join(HERE, 'banks', 'gm.ksb')

# NO TWO MEASURED KEYS WITHIN ONE REGION MORE THAN THIS APART. Six
# semitones is a fourth: a sample stretched that far moves about 1 dB
# in level on the slopes measured here, so the readings between them
# interpolate. It roughly doubles the grid over the three-point rule.
GAP = 6

# How far apart the notes of the sweep are, and the longest window a
# note may be measured over. The second must stay under the first.
SPACING = 0.8
WIN_CAP = 0.7

# How much of the previous note may still be sounding when the next one
# opens, as a fraction of the next note's own peak. Three per cent is
# -30 dB: audible as a difference between two readings and not as a
# difference between two instruments.
BLEED_OK = 0.03

# ksb_pick's own default for calib: this much of a deviation inside
# KEEP_BAND_DB survives, so an instrument keeps its character.
KEEP = 0.3

PASSES = 3


def bank_regions(path):
    """{program: [(klo, khi, vlo, vhi), ...]} in file order."""
    b = open(path, 'rb').read()
    n, nr, ns, ver, doff = struct.unpack_from('<HHHHI', b, 4)
    io, ro = 16, 16 + n * 7
    out = {}
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from(
            '<BBBHH', b, io + i * 7)
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi, vlo, vhi = struct.unpack_from('<HHHH', b, o)
            out.setdefault(prog, []).append((klo, min(khi, 127), vlo, vhi))
    return out


def dense_keys(regs, gap=GAP):
    """Every boundary plus a fixed stride, so no gap exceeds `gap`."""
    if not regs:
        return []
    ks = set()
    for klo, khi, vlo, vhi in regs:
        ks.add(klo)
        ks.add(khi)
        k = klo + gap
        while k < khi:
            ks.add(k)
            k += gap
    return sorted(k for k in ks if 0 <= k <= 127)


def assignments(regs, gap=GAP):
    """One note per (sampled key, region, loudness), from the region's
    own velocity band. The bank is asked for every region it holds.

    EVERY REGION IS HAD AT MORE THAN ONE LOUDNESS. One reading per key
    at a single velocity is a coin toss with a gain attached: the
    second reading is what says whether the first was about the
    instrument or about that note. A region with three or more sampled
    keys is measured at two points of its band; one with fewer keys is
    walked across all of it, because its keys cannot carry the
    evidence on their own.

    A REGION OF ONE OR TWO KEYS IS WALKED ACROSS ITS VELOCITY BAND
    INSTEAD. At a single velocity it has one or two notes of evidence,
    and one note is not a measurement: p21's single-key region could
    only be moved on that, and the step cap turned one bad reading into
    6 dB of error per pass. The keys are played at several loudnesses
    so the readings can agree with each other - which is the evidence -
    and balance_regions declines to move when they do not.
    """
    keys = set(dense_keys(regs, gap))
    notes = set()
    for klo, khi, vlo, vhi in regs:
        ks = [k for k in range(klo, khi + 1) if k in keys]
        span = vhi - vlo
        if len(ks) >= 3:
            # TWO LOUDNESSES, NOT ONE. A region with keys to spare
            # still has only one reading per key at a single velocity,
            # and the second one is what says whether the first was
            # about the instrument or about that note.
            vels = sorted({vlo + span // 3, vlo + 2 * span // 3}
                          if span >= 30 else {(vlo + vhi) // 2})
        else:
            vels = sorted({vlo + i * span // 6 for i in range(7)}
                          if span >= 30
                          else {vlo, vlo + span // 3, (vlo + vhi) // 2,
                                vlo + 2 * span // 3, vhi})
        for k in ks:
            for v in vels:
                notes.add((k, v))
    return sorted(notes)


def write_midi(path, prog, notes, spacing=SPACING):
    def vlq(v):
        o = [v & 0x7f]
        v >>= 7
        while v:
            o.append((v & 0x7f) | 0x80)
            v >>= 7
        return bytes(reversed(o))

    div = 96
    drum = (prog == 128)
    ch = 9 if drum else 0
    ev = [] if drum else [(0, bytes([0xC0, prog]))]
    ticks = int(round(spacing * 192))
    hold = int(round((spacing - 0.05) * 192))
    for k, (key, vel) in enumerate(notes):
        t = 72 + k * ticks
        ev.append((t, bytes([0x90 | ch, key, vel])))
        ev.append((t + hold, bytes([0x80 | ch, key, 0])))
    ev.sort(key=lambda x: x[0])
    body = b''
    last = 0
    for t, data in ev:
        body += vlq(t - last) + data
        last = t
    body += vlq(0) + bytes([0xFF, 0x2F, 0])
    tempo = bytes([0, 0xFF, 0x51, 3, 7, 0xA1, 0x20, 0, 0xFF, 0x2F, 0])
    open(path, 'wb').write(b'MThd' + struct.pack('>IHHH', 6, 1, 2, div) +
                           b'MTrk' + struct.pack('>I', len(tempo)) + tempo +
                           b'MTrk' + struct.pack('>I', len(body)) + body)


def render(bank, midi, wav):
    subprocess.run([PLAYER, '--bank', bank, '--multi', '--wav', wav, midi],
                   capture_output=True)


class Slots(object):
    """One rendered phrase, read a note at a time.

    THE RENDER IS NOT LOADED WHOLE. _note_rms in ksb_pick learned this
    the hard way - a 7400 s calibration grid, readframes and the
    float64 cast peaked near 3 GiB and the second pass raised
    MemoryError - and this file repeated it: p64's sweep is 3262 s of
    audio, which is 1.1 GB as one float64 array, and the pass died
    without a traceback. Every caller here wants a note's own slot and
    nothing else, so that is all that is read.
    """

    def __init__(self, path):
        import wave
        self.w = wave.open(path, 'rb')
        self.ch = self.w.getnchannels()
        self.rate = self.w.getframerate()
        self.n = self.w.getnframes()

    def at(self, t0, t1):
        a0 = max(0, int(t0 * self.rate))
        a1 = min(self.n, int(t1 * self.rate))
        if a1 <= a0:
            return np.zeros(0)
        self.w.setpos(a0)
        x = np.frombuffer(self.w.readframes(a1 - a0), dtype='<i2')
        if self.ch > 1:
            x = x.reshape(-1, self.ch).mean(axis=1)
        return np.asarray(x, dtype=np.float64)

    def close(self):
        self.w.close()


def note_spans(path, nnotes, spacing=SPACING):
    """gm's own span for each note, capped so the next note stays out."""
    s = Slots(path)
    out = []
    try:
        for k in range(nnotes):
            t0 = k * spacing
            x = s.at(t0, t0 + spacing)
            w = note_window(x, s.rate)
            if w is None:
                out.append(None)
                continue
            lo = w[0] / float(s.rate)
            hi = min(w[1] / float(s.rate), WIN_CAP)
            out.append((lo, hi) if hi - lo > 0.01 else None)
    finally:
        s.close()
    return out


def levels(path, spans, spacing=SPACING):
    s = Slots(path)
    out = []
    try:
        for k, sp in enumerate(spans):
            t0 = k * spacing
            lo, hi = sp if sp else (0.02, WIN_CAP)
            seg = s.at(t0 + lo, t0 + hi)
            if len(seg) < 256 or np.abs(seg).max() < 20:
                out.append(None)
            else:
                out.append(float(np.sqrt((seg * seg).mean())))
    finally:
        s.close()
    return out


def max_bleed(path, nnotes, spacing):
    """How much of the previous note is still sounding when the next
    one opens, as a fraction of the LOUDEST NOTE IN THE PHRASE.

    THE REFERENCE IS THE PHRASE, NOT THE NEXT NOTE. Against the next
    note's own peak this read 2.1 on p11 - more than one - because the
    notes are played at velocities 50 times apart and a soft note
    following a loud one is quieter than the loud one's tail no matter
    how far apart they are. What the spacing has to guarantee is that
    no tail is audible against the scale the whole phrase is measured
    in, so that is what it is compared to.
    """
    s = Slots(path)
    worst = 0.0
    try:
        peak = 0.0
        for k in range(nnotes):
            x = s.at(k * spacing, (k + 1) * spacing)
            if len(x):
                peak = max(peak, float(np.abs(x).max()))
        if peak <= 0:
            return 0.0
        for k in range(1, nnotes):
            t0 = k * spacing
            pre = s.at(t0 - 0.05, t0)
            if len(pre) < 8:
                continue
            worst = max(worst, float(np.abs(pre).max()) / peak)
    finally:
        s.close()
    return worst


def sweep(bank, prog, notes, tmp, spacing=SPACING):
    """{(key, vel): level} for the bank and for gm, same window.

    THE SPACING GROWS UNTIL THE NOTES STOP TOUCHING. Both banks are
    checked: whichever rings longer sets the pace, and the window is
    taken from gm as always.
    """
    sp = spacing
    wa = os.path.join(tmp, 'bs_a.wav')
    wg = os.path.join(tmp, 'bs_g.wav')
    mid = os.path.join(tmp, 'bs.mid')
    # WHICH SPACING THE RENDER ACTUALLY USED. `sp` is doubled AFTER the
    # check, so at the end of the loop it is the spacing that was NOT
    # written - and every window was then taken at that time instead of
    # at the note's own. Reading a window at the wrong time gives the
    # previous note's tail, or silence, and p11 came back with 33 keys
    # of 3488 notes measurable.
    used = sp
    for attempt in range(4):
        used = sp
        write_midi(mid, prog, notes, sp)
        render(bank, mid, wa)
        render(REF, mid, wg)
        if not (os.path.exists(wa) and os.path.exists(wg)):
            return {}, {}
        bleed = max(max_bleed(wa, len(notes), sp),
                    max_bleed(wg, len(notes), sp))
        if bleed < BLEED_OK:
            break
        sp *= 2.0
    spans = note_spans(wg, len(notes), used)
    ours = levels(wa, spans, used)
    ref = levels(wg, spans, used)
    for f in (mid, wa, wg):
        try:
            os.remove(f)
        except OSError:
            pass
    return (dict(zip(notes, ours)), dict(zip(notes, ref)))


def as_grid(pairs):
    """{(key, vel): level} -> {vel: {key: level}}, dropping silence."""
    out = {}
    for (k, v), lv in pairs.items():
        if lv:
            out.setdefault(v, {})[k] = lv
    return out


def measure_all(bank, regs, gap, tmp):
    grid, rgrid, before, ref_lv = {}, {}, {}, {}
    for prog in sorted(regs):
        notes = assignments(regs[prog], gap)
        ours, ref = sweep(bank, prog, notes, tmp)
        grid[prog] = as_grid(ours)
        rgrid[prog] = as_grid(ref)
        allv = [x for g in grid[prog].values() for x in g.values()]
        allr = [x for g in rgrid[prog].values() for x in g.values()]
        if allv and allr:
            before[prog] = sorted(allv)[len(allv) // 2]
            ref_lv[prog] = sorted(allr)[len(allr) // 2]
    return grid, rgrid, before, ref_lv


def span_report(regs, grid, rgrid):
    """The widest deviation one gain has to hold, per region."""
    worst = []
    for prog, rs in regs.items():
        dev = {}
        for vel, gv in grid.get(prog, {}).items():
            rv = rgrid.get(prog, {}).get(vel, {})
            for k, c in gv.items():
                r = rv.get(k)
                if r and c > 0:
                    dev.setdefault(k, []).append(20.0 * np.log10(c / r))
        med = {k: sorted(v)[len(v) // 2] for k, v in dev.items()}
        for klo, khi, vlo, vhi in rs:
            vals = [med[k] for k in range(klo, khi + 1) if k in med]
            if len(vals) >= 2:
                worst.append((max(vals) - min(vals), prog,
                              (klo, khi, vlo, vhi)))
    worst.sort(reverse=True)
    return worst


def main(argv):
    bank = argv[1] if len(argv) > 1 else os.path.join(HERE, 'banks',
                                                      'picked.ksb')
    gap = GAP
    apply_ = '--apply' in argv
    if '--gap' in argv:
        gap = int(argv[argv.index('--gap') + 1])

    regs = bank_regions(bank)
    tmp = tempfile.mkdtemp(prefix='bsplit')
    total = sum(len(assignments(r, gap)) for r in regs.values())
    print('sweeping %d programs, %d notes (gap %d, one velocity per region)'
          % (len(regs), total, gap), flush=True)
    grid, rgrid, before, ref_lv = measure_all(bank, regs, gap, tmp)

    w0 = span_report(regs, grid, rgrid)
    print('before: %d regions span >6 dB, worst %.1f dB (p%d %d-%d)'
          % (sum(1 for s, _, _ in w0 if s > 6), w0[0][0], w0[0][1],
             w0[0][2][0], w0[0][2][1]))
    if not apply_:
        print('(measure only; pass --apply to split and re-level)')
        return 0

    fallback = sorted(ref_lv.values())[len(ref_lv) // 2]
    for it in range(PASSES):
        cuts = kp.split_regions(bank, grid, rgrid)
        # THE GRID IS FRESH AT THIS POINT - split_regions does not move
        # a gain, so the readings behind it are still the bank's.
        moved = kp.balance_regions(bank, grid, rgrid)
        print('pass %d: %d cut(s), %d region(s) balanced'
              % (it + 1, cuts, moved), flush=True)
        if not cuts and not moved:
            break
        regs = bank_regions(bank)
        grid, rgrid, before, ref_lv = measure_all(bank, regs, gap, tmp)
        fallback = sorted(ref_lv.values())[len(ref_lv) // 2]
        # balance_regions moves each velocity band on its own, which is
        # how the bands came apart in the first place; the zone is
        # collapsed back to the reference's one gain per key zone after
        # it, and measured again so the next pass starts from the bank
        # as it stands.
        uni = kp.unify_velocity_gains(bank, grid, rgrid)
        if uni:
            print('   %d region(s) re-unified to one gain per zone'
                  % uni, flush=True)
            grid, rgrid, before, ref_lv = measure_all(bank, regs, gap, tmp)
            fallback = sorted(ref_lv.values())[len(ref_lv) // 2]

    w1 = span_report(regs, grid, rgrid)
    print('after : %d regions span >6 dB, worst %.1f dB (p%d %d-%d)'
          % (sum(1 for s, _, _ in w1 if s > 6), w1[0][0], w1[0][1],
             w1[0][2][0], w1[0][2][1]))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
