"""Rewrite DLS sources at the sample rate their content actually needs.

    python tools/dls_resample.py --out build/dls banks/*.dls

Each wave's content edge is the highest frequency standing `db` above
its own noise floor, measured on the attack and on the loudest window
as well as on the centre. The storage rate is
`margin*edge/WIDEST_CUT` (margin is 2*guard), the divisor being the
widest reconstruction row the player has: a rate of 2*edge would have
the player cut the top of the band the measurement just called content.
`storage_rate` is the one owner of that decision; ksb_build calls it.
This pass only shrinks, never above the source rate. Resampling is the
Kaiser-windowed sinc from ksb_build with an explicit cutoff, and the
tuning lost to loop rounding is written back into the wsmp fine tune.

The output is an ordinary DLS: the picker and ksb_build read it as they
read the originals, so selection and packing see the shipped signal and
the shipped size. build/dls/manifest.json caches by source hash,
parameters and TOOL_VERSION.
"""
import numpy as np
import struct
import sys
from collections import namedtuple

from gen_fir import WIDEST_CUT


def _frames(x, rate, frame=0.020):
    fl = max(16, int(frame * rate))
    n = len(x) // fl
    if n == 0:
        return np.zeros((0, fl)), fl
    return x[:n * fl].reshape(n, fl), fl


def noise_floor(x, rate, frame_db=-60.0):
    """Per-bin noise POWER DENSITY, robust to sustained tones.

    Three steps: a low percentile over time (content is intermittent,
    noise is not), a wide median across frequency (a tone occupies a few
    bins and is replaced by its neighbours), and a correction for the
    percentile's bias on exponentially distributed bin powers."""
    blocks, fl = _frames(x, rate)
    bins = fl // 2 + 1
    if len(blocks) == 0:
        return np.full(bins, 1e-12)
    rms = np.sqrt((blocks * blocks).mean(axis=1))
    keep = blocks[rms > rms.max() * 10 ** (frame_db / 20.0)]
    if len(keep) == 0:
        return np.full(bins, 1e-12)
    w = np.hanning(fl)
    norm = (w * w).sum() * rate
    spec = np.abs(np.fft.rfft(keep * w, axis=1)) ** 2 / norm
    low = np.percentile(spec, 5, axis=0) / 0.0513
    width = max(9, (len(low) // 64) | 1)
    half = width // 2
    smooth = np.array([np.median(low[max(0, i - half):i + half + 1])
                       for i in range(len(low))])
    return smooth


def _segment_edge(seg, rate, flr, ff, db):
    """The highest 3-bin run in `seg` standing `db` above the floor.

    The centre, the attack and the loudest window are all segments of
    the same wave, so they are measured here by the same rule."""
    n = len(seg)
    if n < 64:
        return None
    w = np.hanning(n)
    dens = np.abs(np.fft.rfft(seg * w)) ** 2 / ((w * w).sum() * rate)
    fr = np.fft.rfftfreq(n, 1.0 / rate)
    sm = dens.copy()
    if len(dens) >= 2:
        sm[0] = np.median(dens[:2])
        sm[-1] = np.median(dens[-2:])
    if len(dens) > 2:
        sm[1:-1] = np.median(np.vstack((dens[:-2], dens[1:-1], dens[2:])),
                             axis=0)
    floor = np.interp(fr, ff, flr)
    above = sm > floor * 10 ** (db / 10.0)
    # A RUN OF THREE BINS, not one. For stationary noise the chance a
    # single bin clears the floor by 10 dB is small, but over twenty
    # thousand bins the maximum does so by chance; content is wider than
    # the noise's own fluctuations.
    runs = []
    start = None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(above) - 1))
    good = [r for r in runs if r[1] - r[0] + 1 >= 3]
    return float(fr[good[-1][1]]) if good else None


def content_edge(x, rate, edge_floor=4000.0, edge_cap=16000.0, db=10.0):
    """The highest frequency standing `db` above the noise floor.

    MEASURED WHERE THE SOUND IS, NOT ONLY IN THE MIDDLE. The centre of a
    long wave is its decay, so a wave whose only high content is the
    attack reads low and is then stored too slowly for that attack: the
    marimba this was found on measured 4000 Hz from its 74 ms wave even
    though four fifths of its first 10 ms is 2.94-4.2 kHz. The first
    30 ms and the loudest 30 ms window are measured beside the centre
    and the largest edge wins. The transient is not made part of the
    floor: the floor is still the quiet frames' spectrum, and the
    transient is the content compared against it.

    The signal spectrum is smoothed with a 3-bin median so one noisy
    bin cannot end the search. Both sides are power per hertz."""
    flr = noise_floor(x, rate)
    ff = np.fft.rfftfreq(max(16, int(0.020 * rate)), 1.0 / rate)
    segs = []
    n = min(len(x), 65536)
    if n >= 64:
        a = (len(x) - n) // 2
        segs.append(x[a:a + n])
    na = min(len(x), max(64, int(0.030 * rate)))
    if na >= 64:
        segs.append(x[:na])
    hop = max(1, int(0.010 * rate))
    if len(x) > na:
        sq = np.concatenate(([0.0], np.cumsum(x * x)))
        starts = np.arange(0, len(x) - na + 1, hop)
        sums = sq[starts + na] - sq[starts]
        best = int(starts[int(np.argmax(sums))])
        if best:
            segs.append(x[best:best + na])
    edge = edge_floor
    for seg in segs:
        e = _segment_edge(seg, rate, flr, ff, db)
        if e is not None and e > edge:
            edge = e
    # THE FLOOR CANNOT EXCEED WHAT THE SOURCE CAN CARRY. A 3753 Hz
    # source has no bin above 1876 Hz; flooring its edge at 4 kHz and
    # then dividing by the player's cutoff would order a rate the wave
    # cannot use.
    return float(min(edge_cap, rate / 2.0, max(edge_floor, edge)))


# WHAT storage_rate() RETURNS: the rate to store at and the edge it was
# derived from. The edge is handed back because the resampler's own
# cutoff is stated against it; computing it twice would let the two
# uses drift.
Storage = namedtuple('Storage', 'rate edge')


def storage_rate(x, rate, edge_floor=4000.0, edge_cap=12000.0, edge_db=8.0,
                 margin=2.1, rate_lo=8000, rate_hi=36000):
    """THE ONE PLACE THAT DECIDES HOW FAST A WAVE IS STORED.

    rate >= margin*edge/WIDEST_CUT. margin is the old `2*guard` (or
    ksb_build's k): Nyquist must sit margin times above the edge, and
    the player's widest reconstruction row is -6 dB at WIDEST_CUT of the
    stored Nyquist, so storing at 2*edge alone loses the top 30 % of the
    band the measurement just called content. WIDEST_CUT comes from
    gen_fir.py, the producer of the table the player uses; the number is
    not written here a second time.

    The rate is a lower bound on what the PLAYER can be told, so it may
    exceed the source rate: an already band-limited wave stored above
    its own rate keeps the same bytes of band but moves them below the
    reconstruction cutoff. Callers that only downsample clamp it."""
    edge = content_edge(x, rate, edge_floor, edge_cap, edge_db)
    want = int(np.ceil(margin * edge / WIDEST_CUT))
    return Storage(int(min(rate_hi, max(rate_lo, want))), edge)


def pitch_shift_cents(rate, span, rate2, span2):
    """Pitch offset introduced by resampling, in cents.

    `span`/`span2` are the loop's total length (or the whole sample's
    length for an unlooped wave). The player hears rate2/span2 where it
    used to hear rate/span; the difference is what `fine` must undo."""
    return 1200.0 * np.log2((rate2 * span) / (rate * span2))

def _chunk(cid, data):
    b = cid + struct.pack('<I', len(data)) + data
    return b + (b'\0' if len(data) & 1 else b'')


def _children(buf, off, end):
    out = []
    while off + 8 <= end:
        cid = buf[off:off + 4]
        n = struct.unpack_from('<I', buf, off + 4)[0]
        out.append((cid, buf[off + 8:off + 8 + n]))
        off += 8 + n + (n & 1)
    return out


def _list(tag, children):
    body = tag + b''.join(_chunk(c, d) for c, d in children)
    return _chunk(b'LIST', body)


def rewrite_dls(buf, transform):
    """Rewrite the wave pool of a DLS, preserving everything else.

    `transform(children)` receives one wave LIST's [(id, data)] and
    returns the replacement list. The wave pool, the cue table and the
    RIFF/LIST sizes are rebuilt; instruments, articulations and INFO are
    copied byte for byte (wlnk refers to wave indices, not offsets)."""
    assert buf[:4] == b'RIFF' and buf[8:12] == b'DLS ', 'not a DLS'
    top = _children(buf, 12, len(buf))
    new_waves = []
    for cid, d in top:
        if cid == b'LIST' and d[:4] == b'wvpl':
            for wc, wd in _children(d, 4, len(d)):
                if wc == b'LIST' and wd[:4] == b'wave':
                    new_waves.append(_list(b'wave', transform(
                        list(_children(wd, 4, len(wd))))))
    if not new_waves:
        raise ValueError('no wave pool')
    cues, pos = [], 0
    for w in new_waves:
        cues.append(pos)
        pos += len(w)
    ptbl_data = struct.pack('<II', 8, len(cues)) + b''.join(
        struct.pack('<I', c) for c in cues)
    out = []
    for cid, d in top:
        if cid == b'LIST' and d[:4] == b'wvpl':
            out.append((b'LIST', b'wvpl' + b''.join(new_waves)))
        elif cid == b'ptbl':
            out.append((b'ptbl', ptbl_data))
        else:
            out.append((cid, d))
    if not any(c == b'ptbl' for c, d in out):
        out.append((b'ptbl', ptbl_data))
    body = b'DLS ' + b''.join(_chunk(c, d) for c, d in out)
    return b'RIFF' + struct.pack('<I', len(body)) + body


# THE PROJECT PROFILE, not whatever the tool's default happened to be
# when a source was processed. The parameters drifted across the
# session - FreeFont was built with edge_db 8 / guard 1.05, every later
# source with 10 / 1.10, and the caps were 12000 or 16000 depending on
# which command line was typed that day - because each new source
# arrived in its own invocation and took the tool's then-current
# default. One profile, applied by name, replaces that.
DEFAULT = dict(edge_floor=4000.0, edge_cap=12000.0, edge_db=8.0,
               guard=1.05, rate_lo=8000, rate_hi=36000)
# THE ONE DOCUMENTED EXCEPTION: FF8 is the bank this project is built
# around and keeps 16000 Hz. Everything else takes the profile above.
PROFILE = {'ff8.dls': dict(edge_cap=16000.0)}
TOOL_VERSION = 6


def process_wave(kids, params):
    """Resample one wave LIST uniformly. Returns (new_kids, record).

    THE WHOLE WAVE IS RESAMPLED WITH ONE MAPPING. An earlier version
    resampled the attack and the loop separately and unrolled the loop to
    reduce its length rounding; that made the map piecewise, and a region
    whose own loop lay outside the wave's processed span could not be
    scaled at all - it kept source-domain points in a resampled wave.
    A uniform mapping scales every loop, including region overrides,
    with the same factor, and fine compensation keeps the pitch exact
    without needing the loop length to carry it."""
    import ksb_build
    d = dict(kids)
    fmt = d.get(b'fmt ')
    pcm = d.get(b'data')
    ws = d.get(b'wsmp')
    rec = {'rate': 0, 'rate2': 0, 'edge': 0.0, 'delta': 0.0, 'frames': 0,
           'skip': None, 'map': None}
    if fmt is None or pcm is None or len(fmt) < 16:
        rec['skip'] = 'missing fmt/data'
        return kids, rec
    tag, ch, rate, avg, align, bits = struct.unpack('<HHIIHH', fmt[:16])
    if tag != 1 or bits != 16:
        rec['skip'] = 'not 16-bit PCM'
        return kids, rec
    frames = len(pcm) // (2 * ch)
    x = np.frombuffer(pcm[:frames * ch * 2], dtype='<i2').astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch)
    mono = x.mean(axis=1) if ch > 1 else x
    p = dict(DEFAULT)
    p.update(params)
    st = storage_rate(mono, rate, p['edge_floor'], p['edge_cap'],
                      p['edge_db'], 2.0 * p['guard'],
                      p['rate_lo'], p['rate_hi'])
    edge = st.edge
    # THIS PASS ONLY DOWNSAMPLES. It is the shrinker in front of the
    # picker; a wave already at or below its required rate keeps every
    # frame. ksb_build stores above the source when the player's
    # reconstruction cutoff asks for it.
    r2 = min(st.rate, rate)
    ratio = r2 / float(rate)
    cut = min(1.0, edge / (r2 / 2.0))
    rec.update(rate=rate, rate2=r2, edge=edge)
    # VERY SHORT SAMPLES KEEP THEIR RATE. Rounding a two-frame sample
    # to one frame is a 1200-cent transposition; below 64 frames the
    # rounding is worth more than the bandwidth the rate saves.
    if frames * ratio < 64:
        r2 = rate
        ratio = 1.0
        cut = 1.0
        rec['rate2'] = r2
    m = max(1, int(round(frames * ratio)))
    scale = m / float(frames)
    loop = None
    if ws is not None and len(ws) >= 36:
        lc = struct.unpack_from('<i', ws, 16)[0]
        ls = struct.unpack_from('<I', ws, 28)[0]
        ll = struct.unpack_from('<I', ws, 32)[0]
        if lc > 0 and ll > 0 and ls + ll <= frames:
            loop = (ls, ll)
    positions = np.arange(m) * (frames / float(m))
    def run(channel):
        return ksb_build.resample(channel, positions, ratio, cut=cut,
                                  wrap=loop)
    if ch > 1:
        ys = [run(x[:, c]) for c in range(ch)]
        n = min(len(v) for v in ys)
        y = np.stack([v[:n] for v in ys], axis=1)
        m = n
    else:
        y = run(x)
    rec['frames'] = m
    rec['delta'] = pitch_shift_cents(rate, frames, r2, m)
    rec['map'] = (scale, frames, m)
    new_ws = ws
    if ws is not None:
        b = bytearray(ws)
        fine = struct.unpack_from('<h', b, 6)[0]
        v = int(round(fine - rec['delta']))
        if v < -32768 or v > 32767:
            raise SystemExit('fine overflow on a wave')
        struct.pack_into('<h', b, 6, v)
        if loop is not None:
            ls2 = int(round(loop[0] * scale))
            ll2 = max(1, int(round(loop[1] * scale)))
            if ls2 + ll2 > m:
                ll2 = max(1, m - ls2)
            struct.pack_into('<I', b, 28, ls2)
            struct.pack_into('<I', b, 32, ll2)
        new_ws = bytes(b)
    new_pcm = np.clip(y, -32768, 32767).astype('<i2').tobytes()
    new_fmt = bytearray(fmt)
    struct.pack_into('<I', new_fmt, 4, r2)
    out = []
    for c, data in kids:
        if c == b'fmt ':
            out.append((c, bytes(new_fmt)))
        elif c == b'data':
            out.append((c, new_pcm))
        elif c == b'wsmp':
            out.append((c, new_ws))
        else:
            out.append((c, data))
    return out, rec


def _rewrite_region(rd, recs):
    """Correct a region's wsmp: fine by -delta, loop points by the scale
    the whole wave was resampled with. Every loop maps, including one
    that differs from the wave's."""
    reg = _children(rd, 4, len(rd))
    idx = None
    for c, d in reg:
        if c == b'wlnk':
            idx = struct.unpack_from('<I', d, 8)[0]
    if idx is None or idx >= len(recs) or recs[idx] is None:
        return rd
    rec = recs[idx]
    nr = []
    for c, d in reg:
        if c == b'wsmp':
            b = bytearray(d)
            fine = struct.unpack_from('<h', b, 6)[0]
            v = int(round(fine - rec['delta']))
            if v < -32768 or v > 32767:
                raise SystemExit('fine overflow on wave %d' % idx)
            struct.pack_into('<h', b, 6, v)
            if len(b) >= 36 and rec.get('map') is not None:
                scale, frames, m = rec['map']
                lc = struct.unpack_from('<i', b, 16)[0]
                rls = struct.unpack_from('<I', b, 28)[0]
                rll = struct.unpack_from('<I', b, 32)[0] if lc > 0 else 0
                if lc > 0 and rll > 0:
                    ls2 = int(round(rls * scale))
                    ll2 = max(1, int(round(rll * scale)))
                    if ls2 + ll2 > m:
                        ll2 = max(1, m - ls2)
                    struct.pack_into('<I', b, 28, ls2)
                    struct.pack_into('<I', b, 32, ll2)
            d = bytes(b)
        nr.append((c, d))
    return rd[:4] + b''.join(_chunk(a, b2) for a, b2 in nr)


def _patch_list(d, deltas):
    """Rebuild LIST lins/ins /lrgn with region wsmp fines corrected."""
    tag = d[:4]
    kids = _children(d, 4, len(d))
    if tag == b'lrgn':
        return tag, [(c, _rewrite_region(x, deltas))
                     if (c == b'LIST' and x[:4] in (b'rgn ', b'rgn2'))
                     else (c, x) for c, x in kids]
    if tag in (b'lins', b'ins '):
        nk = []
        for c, x in kids:
            if c == b'LIST' and x[:4] in (b'lins', b'ins ', b'lrgn'):
                t, kk = _patch_list(x, deltas)
                nk.append((b'LIST', t + b''.join(_chunk(a, b2)
                                                 for a, b2 in kk)))
            else:
                nk.append((c, x))
        return tag, nk
    return tag, kids


def process_dls(buf, params=None):
    """Resample every wave and correct region tuning. Returns (bytes, recs)."""
    params = params or {}
    top = _children(buf, 12, len(buf))
    waves, recs = [], []
    for c, d in top:
        if c == b'LIST' and d[:4] == b'wvpl':
            for wc, wd in _children(d, 4, len(d)):
                if wc == b'LIST' and wd[:4] == b'wave':
                    nk, rec = process_wave(list(_children(wd, 4, len(wd))),
                                           params)
                    waves.append(nk)
                    recs.append(rec)
    if not waves:
        raise ValueError('no wave pool')
    cues, pos = [], 0
    wchunks = [_list(b'wave', k) for k in waves]
    for w in wchunks:
        cues.append(pos)
        pos += len(w)
    ptbl_data = struct.pack('<II', 8, len(cues)) + b''.join(
        struct.pack('<I', c) for c in cues)
    out = []
    for c, d in top:
        if c == b'LIST' and d[:4] == b'wvpl':
            out.append((b'LIST', b'wvpl' + b''.join(wchunks)))
        elif c == b'LIST' and d[:4] == b'lins':
            t, kk = _patch_list(d, recs)
            out.append((b'LIST', t + b''.join(_chunk(a, b2) for a, b2 in kk)))
        elif c == b'ptbl':
            out.append((b'ptbl', ptbl_data))
        else:
            out.append((c, d))
    if not any(c == b'ptbl' for c, d in out):
        out.append((b'ptbl', ptbl_data))
    body = b'DLS ' + b''.join(_chunk(c, d) for c, d in out)
    return b'RIFF' + struct.pack('<I', len(body)) + body, recs


def _sha(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def main(argv):
    import json
    import os
    outdir = None
    params = dict(DEFAULT)
    caps = {}
    force = False
    srcs = []
    it = iter(argv[1:])
    for a in it:
        if a == '--out':
            outdir = next(it)
        elif a == '--edge-cap':
            params['edge_cap'] = float(next(it))
        elif a == '--cap':
            name, hz = next(it).rsplit('=', 1)
            caps[name.lower()] = float(hz)
        elif a == '--edge-floor':
            params['edge_floor'] = float(next(it))
        elif a == '--edge-db':
            params['edge_db'] = float(next(it))
        elif a == '--guard':
            params['guard'] = float(next(it))
        elif a == '--rate-lo':
            params['rate_lo'] = int(next(it))
        elif a == '--rate-hi':
            params['rate_hi'] = int(next(it))
        elif a == '--force':
            force = True
        else:
            srcs.append(a)
    if outdir is None or not srcs:
        raise SystemExit(__doc__ or 'usage: dls_resample.py --out DIR a.dls ...')
    os.makedirs(outdir, exist_ok=True)
    mpath = os.path.join(outdir, 'manifest.json')
    manifest = {}
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))
    for src in srcs:
        raw = open(src, 'rb').read()
        sha = _sha(raw)
        name = os.path.basename(src)
        local = dict(params)
        local.update(PROFILE.get(name.lower(), {}))
        if name.lower() in caps:
            local['edge_cap'] = caps[name.lower()]
        dst = os.path.join(outdir, name)
        key = name
        rec = manifest.get(key)
        sig = {'sha': sha, 'params': local, 'tool': TOOL_VERSION}
        if (not force and rec == dict(rec or {}, **sig)
                and os.path.exists(dst)):
            print('%-28s cached' % name)
            continue
        out, recs = process_dls(raw, local)
        open(dst, 'wb').write(out)
        manifest[key] = dict(sig, out_sha=_sha(out))
        json.dump(manifest, open(mpath, 'w'), indent=1)
        done = [r for r in recs if not r['skip']]
        skipped = [r for r in recs if r['skip']]
        downs = [r for r in done if r['rate2'] < r['rate']]
        edges = [r['edge'] for r in done] or [0]
        dd = [abs(r['delta']) for r in done] or [0]
        print('%-28s %4d waves, %3d downsampled, edge %.0f..%.0f Hz, '
              'max |delta| %.1f cents, %d skipped, %.1f -> %.1f MiB'
              % (name, len(done), len(downs), min(edges), max(edges),
                 max(dd), len(skipped), len(raw) / 1048576.0,
                 len(out) / 1048576.0))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
