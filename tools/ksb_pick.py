"""Pick one instrument per GM program from several DLS banks, then
settle each program's loudness against gm.dls.

    python ksb_pick.py --out banks/picked.ksb         banks/gm.dls banks/FF8.dls ... [--cap 37800] [--kac PATH]
        [--render-map MAP] [--air-map MAP] [--dirt-map MAP] [--retune-frac 0.5]
        [--calibrate [--keep 0.3] [--ref BANK]]

Selection works per GM program over the candidate banks:

  - a melodic program prefers candidates covering 90% of the 21..108
    range, and the drum kit must cover 27..87;
  - a candidate is dropped when a measured (key, velocity) lies below
    25% of gm (the engine cannot boost it there) or when its profile is
    wider than the 25%-200% window any single gain could fit;
  - a global bad-side veto removes the worst 5% on two or more
    continuous criteria, and a sub-period repeating fragment outright.

Each source is measured by rendering it through kunkun98-play at seven
velocities and eleven keys, and the selected programs are then rendered
again, one note at a time, for the readings a sample cannot give. The
criteria in CRITERIA - bandwidth, air (sample and rendered), noise,
loop seams in four axes (phase: step and slope; power and timbre across
the join; envelope flutter inside it; the rendered seam), clipping,
DC and sub-audio energy (sample and rendered), bits, gain, the measured
key profile, density, size, stretch, depth, the gated start click,
release, the rendered tuning, similarity to gm (ranked, middle wins),
similarity to FF8, and the join cost at key-zone and velocity-layer
boundaries with the onset slip between layers - are min-max normalised
across the candidates and weighted, so a large
loss costs in proportion. Criteria that measure the same property share
one weight cap, and a criterion a source does not carry is left out of
that source's average rather than scored as its worst. The source
preference (FF8 1.5 by default) is a tie-break inside PREFER_MARGIN,
never a multiplier.

The render maps are produced by tune_pick (--render-map: tuning, seam
and the gm/FF8 similarities), air_census (--air-map) and the rendered
dirt census (--dirt-map). --retune-frac writes half of each region's
measured detune back as fine-tune when it is given; with it absent no
pitch correction is applied at all.

Exposure is balanced after scoring, on two budgets. Each source's
target comes from its measured mean score (EXPOSURE_SLOPE) times its
preference, and a bank over target moves its cheapest programs to the
best alternative that is under its own target. The first pass budgets
PROGRAM COUNTS and the drum kit is excluded from it - one kit is one
channel, not a slot. The second budgets SAMPLE BYTES with the same
weights, because a count quota is blind to size and one source can hold
9% of the programs and 57% of the bytes; the kit IS in that pass, as
the largest single item in the bank. Both passes stop at MAX_MOVE_LOSS.

Loudness. The selected programs are rendered and compared with the same
matrix from gm.dls, the level a General MIDI file expects. Calibration
moves each region by the lower quartile of its measured (key, velocity)
deviations, keeps `keep` of any deviation inside KEEP_BAND_DB and
corrects the rest, repeats on the bank it actually wrote, then enforces
the 25%-200% program bound with one final shift. Region gains are
attenuation and boost (the engine's boost stops at +12 dB).

The output is an ordinary KSB bank. The tool is build-side and needs
the player binary for calibration.
"""
import collections
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import ksb_build as _ksb_build

from dls_play import chunks, find, load

# The work directory of the run in progress, for main() to remove.
_WORKDIR = None

# Every connection the synth does not play, weighted by how much it
# changes the sound. Filter is the largest; effect sends only lose the
# ambience; an absent vibrato LFO changes pitch.
PENALTY = {
    # The synth now plays these: filter with its five cutoff
    # modulations, envelope delay/hold, and a separate vibrato LFO.
    'filter': 0,
    'delayhold': 0,
    'viblfo': 0,
    # keyNum -> pitch is extracted and played; effect sends are carried
    # but there is no effect bus to receive them.
    'keypitch': 0,
    'sends': 0,
}

NOTE_KEYS = (24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104)
VEL = 100
# THE GRID IS MEASURED AT SEVEN VELOCITIES. A bank's velocity layers
# are separate regions, and the one selected at velocity 100 says
# nothing about the one selected at 40: a pad whose soft layer was never
# calibrated plays 12 dB quiet under gm in a real arrangement while the
# 100 grid is perfect. Three points missed the layer boundaries between
# them; seven span the arrangement's actual range.
GRID_VELS = (24, 40, 56, 72, 88, 100, 120)
SEG_SEC = 6.8
NOTE_DT = 0.6
# THE DRUM KIT'S NOTES DO NOT FIT IN 0.6 SECONDS. A cymbal rings for
# 1.2 to 1.8 s and is still 13 to 25 dB up when the next note's window
# opens, so the grid measured "this drum plus the last cymbals' tails"
# and read the open hi-hat as +7.7 dB against gm where an isolated
# render reads -31.3 - the calibration then moved the gain by 13 dB in
# the wrong direction. Measured tails: hi-hats 0.06 s, Ride 1 1.16 s,
# open hi-hat 1.53 s, Crash 1 1.81 s. 2.5 s clears all of them. The
# cost is 62 keys x 1.9 s more, about 118 s on a 7400-second grid.
KIT_NOTE_DT = 4.0


def note_dt(prog):
    return KIT_NOTE_DT if prog == 128 else NOTE_DT
NOTE_LEN = 0.4
SEG_TICKS = int(round(SEG_SEC * 192))
SEG_SEC_EFF = SEG_TICKS / 192.0
# CALIBRATION'S KEEP BAND, in dB, against gm.dls. A difference inside
# the band is an instrument's character and `keep` of it survives; a
# difference outside it is an error and is corrected in full. With
# keep=0.3 and a 4 dB band the largest surviving deviation is 1.2 dB -
# "similar to gm, still audible as itself" without a note that sticks
# out of the mix. A flat 30% of a 29 dB deficit is what left a reverse
# cymbal 17 dB down.
KEEP_BAND_DB = 4.0

# The most one balance pass may move a region, in dB. The measurement
# carries its own error; correcting a reading that is off by that error
# walks the wrong way by the same amount, and the next pass corrects
# that. Bounded, the walk converges instead of ringing.
#
# "Twelve is the largest error the shipped bank shows in one region"
# was the case for this number while the audit sampled six keys per
# program. Measuring every key the corpus plays found p49's region
# 44-49 sitting 23.7 dB under gm, so an error that size now takes two
# passes rather than one - which is exactly what the bound is for. It
# used to be six, and six was the reason the corrections never landed:
# two passes topped out at 12 dB on a 9.4 dB error and overshot it, so
# the keys that had been fixed became the new outliers. The number only
# needs to stop a runaway from a bad reading, and balance_from_audit
# drops the readings that disagree instead of acting on them - see
# TRUST_SPREAD_DB.
STEP_CAP_DB = 12.0

# How far the same key may read apart at two velocities before those
# readings are treated as noise rather than as a measurement. Four
# decibels is more than a velocity layer's own level change and less
# than the 9-22 dB errors being chased.
TRUST_SPREAD_DB = 4.0

# How far a key has to sit from its own region's median before the
# region is cut around it. Three decibels is a step a listener hears
# across a melody and is well under the 7-10 dB these keys measured;
# below it the difference is the region's own slope, which one gain is
# allowed to carry.
KEY_EDGE_DB = 3.0

# THE GAIN A CALIBRATION PASS MAY WRITE, in tenths of a decibel. The
# format allows +/-32767 (ksb_build clamps there) and gm.dls only ever
# uses -960..0. The old ceiling was +120 with no comment anywhere, and
# it was BINDING: p92's keys 120-127 read 2.7 to 12.6 dB under gm
# because the region holding key 127 asked for +246 and got +120, and
# the curve stopped at the ceiling exactly where the error started.
# The reading was a pure level error - spectral centroid ratio
# 0.98-1.00, hf ratio 0.99, duration ratio 1.00 - so gain is the right
# knob and the ceiling was the only thing in the way.
#
# THE NUMBER IS THE ENGINE'S, NOT THIS TOOL'S. common/wave.c holds the
# same ceiling in BOOST_MAX and its db_boost_whole table, and a gain
# written above it is silently clamped at play time - the calibration
# then reports success on a bank whose sound never moved. Keeping the
# two equal is what makes a written gain an audible one.
GAIN_MIN_TENTHS = -960
GAIN_MAX_TENTHS = 400


def content_signatures(path):
    """Key each wave by (frames, rate, loop) and keep a normalised
    excerpt. Two sources carrying the same recordings match key for key
    even when a format roundtrip changed the low bits."""
    d = load(path)[0]
    w = find(d, 12, len(d), b'LIST', b'wvpl')
    out = {}
    for c, f, a, b in chunks(d, w[0], w[1]):
        if c != b'LIST' or f != b'wave':
            continue
        fa, _ = find(d, a, b, b'fmt ')
        da, _ = find(d, a, b, b'data')
        rate = struct.unpack_from('<I', d, fa + 4)[0]
        n = struct.unpack_from('<I', d, da - 4)[0] // 2
        ws, _ = find(d, a, b, b'wsmp')
        ls = ll = 0
        if ws is not None:
            if struct.unpack_from('<i', d, ws + 16)[0] > 0:
                ls, ll = struct.unpack_from('<II', d, ws + 28)
        x = np.frombuffer(d[da:da + min(n, 1024) * 2],
                           dtype='<i2').astype(np.float32)
        if len(x) < 64:
            continue
        x = x - x.mean()
        nrm = np.linalg.norm(x)
        if nrm < 1e-6:
            continue
        out[(n, rate, ls, ll)] = (x / nrm).astype(np.float16)
    return out


def content_aliases(sources):
    """Union sources whose wave pools are the same recordings."""
    parent = {s['path']: s['path'] for s in sources}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(sources):
        A = a.get('fp')
        if not A:
            continue
        for b in sources[i + 1:]:
            B = b.get('fp')
            if not B:
                continue
            keys = set(A) & set(B)
            if not keys:
                continue
            same = 0
            for key in keys:
                va = A[key].astype(np.float32)
                vb = B[key].astype(np.float32)
                if float(np.dot(va, vb)) > 0.995:
                    same += 1
            if same >= 0.9 * min(len(A), len(B)):
                ra, rb = root(a['path']), root(b['path'])
                if ra != rb:
                    parent[rb] = ra
    return {s['path']: root(s['path']) for s in sources}


def info(d):
    a, b = find(d, 12, len(d), b'LIST', b'INFO')
    out = {}
    if a is None:
        return out
    for cid, f, x, y in chunks(d, a, b):
        out[cid.decode('latin1', 'replace')] = d[x:y].split(b'\0')[0].decode(
            'latin1', 'replace')
    return out


def wave_metrics(d, entry):
    """One sample, measured rather than taken on trust.

    bw      99% of the spectral energy lies below this frequency. The
            header's rate is not the sound: a 44.1 kHz file can hold a
            recording that stops at 8 kHz, and an upsampled one has
            nothing above its original Nyquist.
    air     the same reading at 99.9%: how far the top of the spectrum
            actually reaches. A pan flute whose energy is 99% under
            4 kHz still has its 0.1% up at 9 kHz, and that top is the
            air the instrument is heard for; the 99% figure is blind
            to it.
    noise   the worse of two readings: the quietest tenth of the 20 ms
            frames as a fraction of the whole RMS - audible hiss where
            the instrument is not playing - and the spectral flatness
            of the loop, which is noise-like sustain. One number,
            because both are the same defect heard in the decay and in
            the sustain.
    loopstep   amplitude step across the loop point, divided by the
               loop's first-difference RMS so the reading does not
               depend on the note's pitch.
    loopslope  change of slope across the same point - the last
               difference inside the loop against the first - divided
               by the loop's second-difference RMS. A seam can jump
               without a step: equal levels, opposed slopes.
    clip    fraction of frames at digital full scale.
    dc      |mean| / peak: offset that wastes headroom and can thump.
    rumble  energy between 3 and 20 Hz as a share of the whole. Below DC
            and above the notes: transport noise, not pitch.
    startclick  |first frame| / peak: a note that begins with a jump.
    bits    log2 of the distinct levels in the sample, capped by the
            recording: an 8-bit source stored in 16 bits measures 7-8.
    periods loop length in fundamental periods at the unity note. Below
            one is a repeating fragment, not a recorded sustain.
    """
    wa, wb = entry
    fa, _ = find(d, wa, wb, b'fmt ')
    if fa is None:
        return None
    tag, ch, rate, avg, align, bits = struct.unpack('<HHIIHH', d[fa:fa + 16])
    if tag != 1 or bits != 16 or rate < 1000:
        return None
    da, db = find(d, wa, wb, b'data')
    if da is None:
        return None
    n = (db - da) // (2 * max(1, ch))
    if n < 64:
        return None
    x = np.frombuffer(d[da:da + n * 2 * ch], dtype='<i2').astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)

    # THE WINDOW IS THE STEADY STATE. Prefer the loop; the middle of a
    # one-shot can be the decaying tail.
    wsm = find(d, wa, wb, b'wsmp')
    ls0 = ll0 = 0
    if wsm[0] is not None and wsm[1] - wsm[0] >= 20:
        lc0 = struct.unpack('<i', d[wsm[0] + 16:wsm[0] + 20])[0]
        if lc0 > 0:
            cb0, typ0, ls0, ll0 = struct.unpack(
                '<IIII', d[wsm[0] + 20:wsm[0] + 36])
    if ll0 > 512 and ls0 + ll0 <= len(x):
        m = min(ll0, 65536)
        a = ls0 + max(0, ll0 - m)
    else:
        m = min(len(x), 65536)
        a = (len(x) - m) // 2
    y = x[a:a + m]
    seg = y * np.hanning(m)
    sp = np.abs(np.fft.rfft(seg)) ** 2
    fr = np.fft.rfftfreq(m, 1.0 / rate)
    cum = np.cumsum(sp)
    bw = float(fr[np.searchsorted(cum, 0.99 * cum[-1])]) if cum[-1] > 0 else 0.0
    # AIR IS THE TOP OF THE SIGNAL, NOT OF THE NOISE FLOOR. A fixed
    # 99.9% of the energy lands on the noise in anything with a hiss
    # floor, so a noisier source read as airier; the top is the highest
    # frequency whose smoothed level stays 20 dB above the median bin.
    if cum[-1] > 0 and m >= 64:
        sdb = 10.0 * np.log10(sp + 1e-12)
        sm = np.convolve(sdb, np.ones(5) / 5.0, 'same')
        # THE FLOOR IS NOT THE SIGNAL. A pure tone has a numerical
        # floor far below its peak, so "median + 20 dB" passes the
        # whole band and a 1 kHz sine read 22050 Hz - measured on a
        # synthetic tone. Content more than 60 dB under the peak is
        # not air, whatever the floor does.
        thr = max(float(np.median(sdb)) + 20.0, float(sdb.max()) - 60.0)
        idx = np.nonzero(sm > thr)[0]
        air = float(fr[idx[-1]]) if len(idx) else 0.0
    else:
        air = 0.0
    # THE CENTRE OF GRAVITY OF THE SPECTRUM, the one figure the timbre
    # literature agrees on (Grey & Gordon): brightness as a single
    # number, so two samples can be compared by how far apart they sit.
    cen = float((fr * sp).sum() / cum[-1]) if cum[-1] > 0 else 0.0

    peak = max(1.0, float(np.abs(x).max()))
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0

    fl = max(16, int(rate * 0.02))
    nf = len(x) // fl
    noise = 0.0
    if nf >= 4:
        block = x[:nf * fl].reshape(nf, fl)
        rms_b = np.sqrt((block * block).mean(axis=1))
        # DIGITAL SILENCE IS NOT THE NOISIEST THING: frames at or below
        # four LSB are dropped before the quiet tenth is chosen.
        live = block[rms_b >= 4.0]
        if len(live) < 4:
            live = block
        lrms = np.sqrt((live * live).mean(axis=1))
        quiet = live[np.argsort(lrms)[:max(1, len(live) // 10)]]
        # AUDIBLE NOISE, NOT SPECTRAL FLATNESS. Flatness is scale
        # invariant - a -90 dBFS dither floor scored the same as -25
        # dBFS hiss, and an all-zero frame scored 1.0. What is heard is
        # how far the quiet tenth sits below the signal.
        noise = float(np.sqrt((quiet * quiet).mean()) / (rms + 1e-9))

    # A HUNDREDTH OF A PERCENT IS NOT DISTORTION. The utility is
    # min-max normalised, so a source clipping 22 frames in a hundred
    # thousand (SGM's pan flute) scored 0 against everyone else's 1.0
    # and lost the whole criterion - as if it were heavily clipped.
    clip = float(np.mean(np.abs(x) >= 32700))
    if clip < 1e-3:
        clip = 0.0
    dc = abs(float(x.mean())) / peak
    # THE CUT IS THE FIRST AUDIBLE FRAME, not frame zero: a sample with
    # a little digital silence in front cuts later, and |x[0]| sees
    # nothing.
    head = x[:max(1, int(rate * 0.01))]
    hk = np.nonzero(np.abs(head) > 0.01 * peak)[0]
    startclick = (abs(float(head[hk[0]])) / peak) if len(hk) else 0.0
    # WHERE THE SOUND BEGINS, in ms. Two layers of one note that start a
    # millisecond apart comb-filter against each other (correlated
    # copies, offset); this is the figure that says how far apart.
    onset = (float(hk[0]) / rate * 1000.0) if len(hk) else 0.0
    # THE STAIRCASE, NOT THE COUNT. A count of distinct levels asks how
    # many values the recording uses, and that is the wrong question
    # twice: dither or any linear resample fills every level of an
    # 8-bit master, and a square wave has two levels by construction.
    # What is audible is the STEP - the spacing the values sit on. An
    # 8-bit source stored in 16 bits shows it as a gap between adjacent
    # levels, and the median gap reads the step whatever the offsets
    # are (256, 257, or any scaled lattice). The window is the same
    # steady one the loop metrics use, so a quiet decaying tail cannot
    # make a clean sample look coarse.
    q = min(len(y), 65536)
    uv = np.unique(y[:q])
    if len(uv) < 3:
        bits_eff = 1.0
    else:
        step = float(np.median(np.diff(uv)))
        bits_eff = float(min(16.0, max(1.0, 16.0 - math.log2(max(1.0, step)))))
    # 3 Hz rather than 0: the offset itself is `dc`, and counting it here
    # as well would make the two the same measurement. The window is
    # de-meaned and de-trended first: a few LSB of offset puts its
    # skirt right across this band.
    yt = y - y.mean()
    if m > 2:
        tt = np.arange(m, dtype=np.float64)
        tt = tt - tt.mean()
        slope = float((tt * yt).sum() / max(1e-9, (tt * tt).sum()))
        yt = yt - slope * tt
    sp2 = np.abs(np.fft.rfft(yt * np.hanning(m))) ** 2
    cum2 = np.cumsum(sp2)
    rumble = float(sp2[(fr > 3.0) & (fr < 20.0)].sum() / (cum2[-1] + 1e-12))

    loopstep = 0.0
    ls = 0
    ll = 0
    loopslope = 0.0
    seamrate = 0.0
    loopflat = 0.0
    periods = 0.0
    looppow = 0.0
    loopfreq = 0.0
    flutter = 0.0
    unity = 60
    f, _ = find(d, wa, wb, b'wsmp')
    if f is not None:
        unity = struct.unpack('<h', d[f + 4:f + 6])[0]
        lc = struct.unpack('<i', d[f + 16:f + 20])[0]
        if lc > 0:
            cb, typ, ls, ll = struct.unpack('<IIII', d[f + 20:f + 36])
            if ll > 0 and ls + ll <= len(x):
                loop = x[ls:ls + ll]
                rms = float(np.sqrt(np.mean(loop * loop))) + 1e-9
                # A SEAM IS MEASURED AGAINST THE WAVE'S OWN STEP. The
                # endpoint difference by itself is the waveform's normal
                # advance: a 5 kHz sine at 44.1 kHz moves 0.71 of its
                # amplitude in one sample, so the old ratio read about
                # 1.0 for a seamless high note and 0.014 for a 100 Hz
                # one - it measured pitch, not seams, and short loops
                # were then punished twice through seamrate's repeat
                # factor. Dividing by the loop's own first-difference
                # RMS makes a seamless loop read the same whatever its
                # pitch.
                if ll >= 3:
                    dl = np.diff(loop)
                    drms = float(np.sqrt(np.mean(dl * dl))) + 1e-9
                    # THE JUMP IS AGAINST THE CONTINUATION, not the
                    # sample before the seam: x[ls+ll-1] to x[ls] is
                    # one normal step of the waveform, so a perfect
                    # loop read ~1.0 (and 0.7-1.4 with phase) instead
                    # of zero. What the ear hears is where playback
                    # lands against where it would have gone - the
                    # sample after the loop end, or its linear
                    # continuation when the loop ends the sample.
                    if ls + ll < len(x):
                        nxt = float(x[ls + ll])
                    else:
                        nxt = (float(x[ls + ll - 1])
                               + float(x[ls + ll - 1] - x[ls + ll - 2]))
                    loopstep = abs(float(x[ls]) - nxt) / drms
                    # THE SLOPE ACROSS THE SEAM, not outside it. The old
                    # pair took x[ls]-x[ls-1] and x[ls+ll]-x[ls+ll-1]:
                    # two frames OUTSIDE the loop, which are continuous
                    # with the recording and measured almost nothing.
                    d_beg = float(x[ls + 1] - x[ls])
                    d_end = float(x[ls + ll - 1] - x[ls + ll - 2])
                    d2l = np.diff(loop, 2)
                    d2rms = float(np.sqrt(np.mean(d2l * d2l))) + 1e-9
                    loopslope = abs(d_beg - d_end) / d2rms
                else:
                    loopstep = loopslope = 0.0
                q = min(ll, 65536)
                qs = loop[len(loop) - q:] * np.hanning(q)
                qp = np.abs(np.fft.rfft(qs)) ** 2 + 1e-12
                loopflat = float(np.exp(np.log(qp).mean()) / qp.mean())
                # A SEAM IS AS BAD AS IT IS REPEATED. The same step at
                # 0.7 repeats a second (SGM's pan flute loop) is one
                # click; at 4 repeats a second (Fury's) it is a buzz.
                # step + slope, scaled by how often they recur, with the
                # factor capped at one past four repeats.
                reps = rate / float(ll)
                seamrate = (loopstep + loopslope) * min(1.0, reps / 4.0)
                noise = max(noise, loopflat)
                freq = 440.0 * 2.0 ** ((unity - 69) / 12.0)
                periods = ll / max(1e-9, rate / freq)
                # POWER AND TIMBRE ACROSS THE SEAM: the other two
                # continuity axes the looping literature names. Phase
                # continuity is loopstep/loopslope above; these compare
                # the loop's last N frames with its first N, which is
                # the join the player makes. A power step is a level
                # flutter once per repeat; a centroid step is a colour
                # change on every pass.
                nw = min(max(16, ll // 4), 2048)
                if ll >= 2 * nw:
                    head = x[ls:ls + nw]
                    tail = x[ls + ll - nw:ls + ll]
                    ar = float(np.sqrt(np.mean(head * head)))
                    br = float(np.sqrt(np.mean(tail * tail)))
                    looppow = abs(ar - br) / (ar + br + 1e-9)
                    hs = np.abs(np.fft.rfft(head * np.hanning(nw))) ** 2
                    ts = np.abs(np.fft.rfft(tail * np.hanning(nw))) ** 2
                    hfr = np.fft.rfftfreq(nw, 1.0 / rate)
                    ha = float(hs.sum())
                    ta = float(ts.sum())
                    if ha > 0.0 and ta > 0.0:
                        hc = float((hfr * hs).sum() / ha)
                        tc = float((hfr * ts).sum() / ta)
                        loopfreq = abs(math.log2((tc + 1.0) / (hc + 1.0)))
                # FLUTTER: a loop is stationary, so an envelope that
                # still moves is beating - two partials, usually two
                # layers, a few hertz apart. The spread of 20 ms frame
                # RMS over the loop, relative to its mean.
                fw = max(16, int(rate * 0.02))
                fn = ll // fw
                if fn >= 4:
                    q2 = loop[:fn * fw].reshape(fn, fw)
                    ev = np.sqrt((q2 * q2).mean(axis=1)) + 1e-9
                    flutter = float(ev.std() / (ev.mean() + 1e-9))
    # A SYNTHESISED WAVEFORM: a loop shorter than one period of the note
    # it is recorded at, silent between the harmonics, and a spectrum that
    # still reaches the top. A natural recording's loop spans many
    # periods; a loop of a fraction of one is a repeating fragment whose
    # "bandwidth" is its own harmonics, not recorded content. Measured
    # here - 99 of MT32.dls's 526 waves, 0 of gm.dls's 495 - rather than
    # inferred from the bank's name.
    synth = 1.0 if (0.0 < periods < 0.75 and noise < 1e-3 and bw > 8000) else 0.0
    # THE PITCH THE SAMPLE ACTUALLY SOUNDS. The tuning criterion needs
    # this rather than what the wsmp declares: measured through the
    # finished bank, selected instruments rendered up to 20 cents off
    # the key. A loop that repeats at the note's own rate is measured
    # from the loop - rate/ll is then the fundamental and is exact,
    # which is how a 58-frame synth loop is read - otherwise from the
    # harmonic product spectrum of the loop, or of the middle of the
    # sample when there is none.
    f0 = 0.0
    f_unity = 440.0 * 2.0 ** ((unity - 69) / 12.0)
    looprate = rate / float(ll) if ll >= 8 else 0.0
    if looprate > 0.0 and abs(1200.0 * math.log2(looprate / f_unity)) < 150.0:
        f0 = looprate
    else:
        if ll > 512 and ls + ll <= len(x):
            seg2 = x[ls:ls + min(ll, 32768)]
        elif len(x) >= 8192:
            seg2 = x[len(x) // 4:len(x) // 4 + min(len(x) // 2, 32768)]
        else:
            seg2 = x
        if len(seg2) >= 2048:
            seg2 = seg2 - seg2.mean()
            sp2 = np.abs(np.fft.rfft(seg2 * np.hanning(len(seg2))))
            fr2 = np.fft.rfftfreq(len(seg2), 1.0 / rate)
            hps = sp2.copy()
            for h in (2, 3, 4, 5):
                part = sp2[::h]
                hps[:len(part)] *= part
            lo2 = int(np.searchsorted(fr2, 30.0))
            hi2 = int(np.searchsorted(fr2, 2000.0))
            if hi2 > lo2 + 2:
                band = hps[lo2:hi2]
                if float(band.max()) >= 20.0 * float(np.median(band) + 1e-12):
                    k2 = lo2 + int(np.argmax(band))
                    best2 = k2
                    for j in range(max(1, k2 - 4), min(len(sp2) - 1, k2 + 5)):
                        if sp2[j] > sp2[best2]:
                            best2 = j
                    y0, y1, y2 = (float(sp2[best2 - 1]), float(sp2[best2]),
                                  float(sp2[best2 + 1]))
                    den = y0 - 2.0 * y1 + y2
                    dd = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
                    f0 = (best2 + dd) * rate / float(len(seg2))
                    # A SECOND OPINION. Distorted, layered or gated
                    # material has no single period and the harmonic
                    # product spectrum can land on a different line
                    # than the waveform repeats at: onestop's
                    # distortion guitar measured 18 cents off by HPS
                    # while its period was exact. When the two name
                    # different pitches the reading is not evidence,
                    # and the program is left with the other regions.
                    n2 = 1
                    while n2 < 2 * len(seg2):
                        n2 <<= 1
                    ac2 = np.fft.irfft(np.abs(np.fft.rfft(seg2, n2)) ** 2,
                                       n2)[:len(seg2)]
                    ac2 /= ac2[0] + 1e-12
                    alo2 = max(2, int(rate / 3000.0))
                    ahi2 = min(len(ac2) - 2, int(rate / 30.0))
                    if ahi2 > alo2 + 2 and f0 > 0.0:
                        ka2 = alo2 + int(np.argmax(ac2[alo2:ahi2]))
                        if float(ac2[ka2]) >= 0.5 and ka2 > 0:
                            d2 = (1200.0 * math.log2(
                                f0 / (rate / float(ka2))))
                            d2 -= 1200.0 * round(d2 / 1200.0)
                            if abs(d2) > 25.0:
                                f0 = 0.0
                    if f0 <= 0.0:
                        f0 = 0.0
    # HARMONICS-TO-NOISE RATIO of the steady window: the share of the
    # power that sits on the harmonic series against everything else.
    # The standard voice-quality reading (Yumoto; Praat's HNR), used
    # here to separate a cleanly recorded instrument from one whose
    # sustain is half noise, which `noise` reads only as its quietest
    # tenth.
    hnr = 0.0
    oddeven = 0.0
    if f0 > 0.0 and sp.sum() > 0.0:
        # THE VALLEYS ARE THE REFERENCE, not the whole spectrum: a
        # fixed band around every harmonic covers most of the band at
        # low f0, so against the total even white noise reads +9 dB.
        # Equal-width windows between the harmonics make noise 0 dB by
        # construction.
        har = res = odd = even = 0.0
        k = 1
        while (k + 0.5) * f0 < fr[-1]:
            hb = (fr > k * f0 * 0.97) & (fr < k * f0 * 1.03)
            vb = (fr > (k + 0.5) * f0 * 0.97) & (fr < (k + 0.5) * f0 * 1.03)
            e = float(sp[hb].sum())
            har += e
            res += float(sp[vb].sum())
            if k % 2:
                odd += e
            else:
                even += e
            k += 1
        if har > 0.0:
            hnr = 10.0 * math.log10(max(1e-12, har) / max(1e-12, res))
            # ODD-TO-EVEN HARMONIC RATIO, the third timbre dimension
            # McAdams and Wu find after attack time and centroid.
            if odd > 0.0 and even > 0.0:
                oddeven = 10.0 * math.log10(odd / even)
    return {'bw': bw, 'air': air, 'noise': noise, 'rms': rms, 'loopstep': loopstep,
            'loopslope': loopslope, 'seamrate': seamrate, 'clip': clip, 'dc': dc,
            'rumble': rumble, 'startclick': startclick, 'bits': bits_eff,
            'unity': unity, 'synth': synth,
            'f0': f0, 'rate': rate, 'll': ll,
            'centroid': cen, 'looppow': looppow, 'loopfreq': loopfreq,
            'flutter': flutter, 'onset': onset, 'hnr': hnr,
            'oddeven': oddeven}


# The criteria, and which way is better. No weight: each is ranked
# among the candidates for the same program and the ranks are averaged,
# so a program's score says how well a source does overall, not how far
# it pushes one measurement. This is the moderate reading of a bank
# comparison - a source that is good on every axis beats one that is
# extreme on one and poor on the rest, which is what a raw sum in
# bandwidth units did.
# EVERY SIGN POINTS AT THE BETTER SAMPLE. `size` too, since it became a
# cost per recording rather than a distance from a band. Nothing here
# rewards a lower reading of a defect or a higher reading of a good
# property.
# SAMPLING DEPTH IS ONE CRITERION, not two. keycover and vellayers measure
# the same property and rank-correlate at +0.4..+0.6 with each other and
# with bits; giving each 2.0 counted depth three times. `depth` is their
# mean rank, and carries what the two used to.
PROFILE = [
    ('bw', +1, 0.4),         # usable response; cut so raw brightness
                             # is not paid for twice (air/rair below and
                             # mtone carry it against the reference).
    ('air', +1, 1.0),        # how far the top of the spectrum reaches.
                             # CUT FROM 2.0: brightness is a preference,
                             # and mtone scores it against FF8 instead of
                             # paying more for more of it.
                             # a pan flute with 99% of its energy under
                             # 4 kHz reads 6.6 kHz here and 11.8 kHz
                             # from SGM, and that top is the clarity the
                             # instrument is heard for
    ('rair', +1, 1.0),       # air MEASURED ON THE RENDER; cut with air.
                             # reading is taken at the stored rate; a
                             # program played up from a 19 kHz sample
                             # loses its top octave and sounds dull.
                             # p100 read 7908 at the sample and 2972
                             # rendered - mid-field versus last.
    ('airfit', +1, 1.5),     # THE RENDERED AIR, AGAINST gm. The
                             # two brightness readings answer different
                             # questions - where a source sits among its
                             # peers, and whether it reaches the
                             # reference at all. air and rair are field
                             # relative: a program whose whole field sits
                             # at half gm's top octave still ranks a
                             # winner. This one is a FLOOR, and it only
                             # ever subtracts - reaching gm costs
                             # nothing, falling short costs in
                             # proportion. Scoring it as a second
                             # brightness value averaged against rair
                             # rewarded dullness: a source at half the
                             # reference would gain on that average
                             # while still being duller than everyone.
    ('noise', -1, 1.0),      # hiss, in the decay and in the sustain
    ('loopstep', -1, 0.8),   # a jump on every repeat
    ('seamrate', -1, 1.2),   # the same seam and slope weighted by how
                             # often they recur
    ('loopslope', -1, 0.5),  # equal levels, opposed slopes: still a click
    ('clip', -1, 1.5),       # distortion; lowered from 2.0, zero for most
    ('dc', -1, 0.5),         # offset that wastes headroom
    ('rumble', -1, 1.5),     # sub-audio energy, not pitch: FluidR3's
                             # FX 4 carried 21% of its energy below 5 Hz,
                             # and at 0.5 that source still won
    ('rsub', -1, 1.5),       # THE SAME DEFECT, MEASURED ON THE OUTPUT.
                             # The sample-level reading above is band
                             # limited to one window; a rendered program
                             # sums every region the notes reach, and
                             # the two disagree - p87's SC-55 read clean
                             # at the sample and 19% below 5 Hz when
                             # played, which is a speaker movement the
                             # score never saw.
    ('rdc', -1, 0.5),        # rendered offset, as a fraction of the peak
    ('bits', +1, 1.2),       # EFFECTIVE QUANTISATION DEPTH, as the
                             # spacing the sample values sit on: an
                             # 8-bit master stored in 16 bits shows a
                             # 256-wide gap and reads 8. A count of
                             # distinct levels was the other estimator
                             # and missed every dithered or resampled
                             # copy while calling a square wave 4 bits.
    ('gain', +1, 0.8),       # region attenuation; calibration cannot boost
    ('gmfit', +1, 1.0),      # can the engine reach gm.dls's level at all
    ('keyfit', -1, 1.2),     # measured spread of the key profile vs gm
    ('keylow', +1, 1.0),     # the quietest measured key: a soft floor
    ('keyhigh', -1, 0.6),    # the loudest measured key: a soft ceiling
    ('levelfit', +1, 1.0),   # overall program level vs gm (drums)
    ('density', +1, 0.6),    # samples per octave: a stretched key is uneven
    ('startrisk', -1, 0.9),  # a note that begins with a jump, only
                             # where the envelope opens in under 10 ms
                             # to expose it. The ungated startclick read
                             # the same defect and ranked alike (rho
                             # 0.78 in every program), so one criterion
                             # carries both.
    ('tunefit', +1, 2.5),    # the same reading on a fixed 30-cent
                             # scale, so one outlier cannot flatten it.
                             # WEIGHTED FOR BEATING, NOT ONLY FOR PITCH:
                             # a region a quarter-tone off does not just
                             # sound out of tune, it beats against the
                             # notes already ringing - on any instrument
                             # whose notes overlap, which is every one of
                             # them. Measured on Chorium's harp: regions
                             # past the 30-cent scale (fine +50/-45/-18),
                             # the field's only candidate scoring 0, heard
                             # as overtones that are not the instrument's.
    ('sim', +1, 2.0),        # WHERE THE SOURCE STANDS AMONG ITS PEERS
                             # by spectral similarity to gm's own render
                             # of the same note: the middle of the field
                             # scores highest, both ends zero. A source
                             # that renders as gm cannot improve on gm,
                             # and one that renders as nothing else has
                             # left the family. The map holds the raw
                             # correlation; the rank is taken here.
    ('ff8sim', +1, 3.5),     # LIKE FF8, WHERE FF8 HAS THE PROGRAM. THE
                             # TARGET. The standard the bank is judged
                             # by is "as much like FF8 as it can be, and
                             # not too far off gm" - so this is the one
                             # the weight is spent on, and `sim` below
                             # is the ceiling that keeps the likeness
                             # from running away from the reference.
                             # FF8 itself carries no reading: the
                             # reference cannot win by being itself.
                             # FF8 is a character reference, not a
                             # tuning standard: a source that renders
                             # like it is preferred, higher is better.
                             # FF8 carries no reading of its own - the
                             # reference cannot win by being itself.
    ('rtune', -1, 2.0),      # WORST rendered note, cents: measured by
                             # playing the program ON ITS OWN REGION
                             # BOUNDARIES, one note per render. The
                             # old low weight was for a reading that
                             # missed boundaries and mixed tails; with
                             # those two faults gone it is a verdict
                             # again, not a hint
    ('rseam', -1, 0.8),      # rendered second-difference peak: a loop
                             # that clicks every period, weighted too
    ('size', -1, 1.2),       # COST OF ONE RECORDING, log2 KB, computed
                             # in programme() where bytes and samples are
                             # both in hand. See the block above SOURCE_
                             # PENALTY: this is a value, not a size, and
                             # it needs no band and no kit special case.
    ('stretch', -1, 0.5),    # pulled from the recorded pitch
    ('keycover', +1, 0.5),   # FRACTION OF THE KEYBOARD WITH A ZONE.
                             # choose() already gates at 90%, so this is
                             # not about silence: it is the difference
                             # between a source that stops at key 97 and
                             # one that reaches 108, which shows up as
                             # the extremes of a run. Kept low because
                             # the gate already carries the defect.
    ('depth', +1, 1.2),      # velocity layers as a rank. Key coverage
                             # is scored separately at 0.5: the two
                             # used to be one averaged number, and what
                             # was lost was that they measure different
                             # things - how many keys, and how many
                             # dynamics within them.
    ('snr', +1, 1.5),        # THE RENDERED SIGNAL-TO-NOISE, peak against
                             # what is left of the note after it stops.
                             # This is the noise a listener hears on a
                             # sustained instrument, and nothing else in
                             # this table measures it: `noise` is hiss in
                             # the raw sample, a different quantity. On
                             # the shipped field the harp went to the
                             # source scoring 41 dB here against gm's 55,
                             # while Creative scored 72 - and the harp's
                             # `noise` reading was the highest in the
                             # pool, so the defect was visible and still
                             # lost. Higher is better.
                             #
                             # WEIGHT 1.5, WITH noise TAKING THE REST OF
                             # THE NOISE VOTE. This was 3.0 and that
                             # over-ruled the character axis: noise ended
                             # up 12.9% of the effective weight against
                             # 4.8% for `sim` and `ff8sim` together, and
                             # the bank drifted to sources that are clean
                             # and plain. The trade is real - SC-55 tunes
                             # at 9 cents and balances at 6.6 dB, Creative
                             # at 18 cents and 10.7 - so this stays above
                             # what it can move at, but not above what the
                             # instruments themselves are worth.
    ('balance', -1, 1.5),    # THE INTERNAL LEVEL SPREAD, in dB, of a
                             # program's keys against gm's on the same
                             # keys - measured on the render, per note.
                             # A program can sit at gm's median level and
                             # still be loud at one end and quiet at the
                             # other: a kit whose hi-hat plays 18 dB
                             # under gm while its crashes match, a
                             # stretched piano whose top keys run hot.
                             # The level pass moves one number per
                             # program and cannot see it. Wider is worse;
                             # it is the SPREAD, not the offset, because
                             # the offset is what the level pass
                             # corrects. Melodic programs measure 5 to
                             # 10 dB and a drum kit 25 to 44 - a kit is
                             # sixty-one instruments at their own
                             # levels - so the readings are normalised
                             # per program and no cap is applied.
    ('mtone', -1, 2.0),      # RENDERED BRIGHTNESS AGAINST FF8, on the
                             # notes the corpus actually plays: the
                             # floor is |log2(tone / tone of FF8)|, so
                             # matching FF8's brightness costs nothing
                             # and drifting from it costs in proportion.
                             # The standard is "like FF8, not
                             # especially like gm, not especially unlike
                             # it"; gm's side is airfit's one-sided
                             # floor, and bw/air/rair do NOT predict
                             # any of it - their log-ratios against
                             # the corpus correlate with the rendered
                             # brightness at +0.11, -0.08 and +0.26.
    ('mlevel', -1, 1.0),      # LEVEL AGAINST GM ON REAL NOTES, in dB.
                             # The floor is the deficit only: being
                             # louder than gm costs nothing (the mixer
                             # attenuates), being under it costs in
                             # proportion, and more than 12 dB under is
                             # unreachable and gated.
                             # as a floor. The calibration matches a
                             # program on notes it chooses; a source can
                             # match there and still sit 24 dB under gm
                             # at the keys and velocities the pieces
                             # play - p115's woodblock measured -1.3 dB
                             # for FF8 and -23.9 for Creative on the
                             # same 24 notes.
    ('release', -1, 0.0),    # RELEASE TIME, ms. Zero in the base: how
                             # long a note rings after note-off is a
                             # character for most instruments and a
                             # defect for a pad, whose notes are held
                             # and retriggered - a seven-second tail
                             # under a chord is a hum, not breathing.
                             # Turned on for 88-96 below.
    ('looppow', -1, 0.4),    # POWER ACROSS THE SEAM: RMS of the loop's
                             # last frames against its first. A step is a
                             # level flutter on every repeat, and phase
                             # continuity (loopstep) does not see it.
    ('loopfreq', -1, 0.6),   # TIMBRE ACROSS THE SAME SEAM: spectral
                             # centroid of those windows. A colour change
                             # per pass is the "buzz" a long loop can
                             # carry even with a continuous waveform.
    ('flutter', -1, 0.5),    # ENVELOPE MODULATION INSIDE THE LOOP: a
                             # stationary loop whose level still moves is
                             # two partials beating - usually two layers.
    ('joindist', -1, 0.5),   # JOIN COST, after unit selection: the
                             # spectral distance (octaves of centroid)
                             # between the two samples a boundary puts
                             # either side of the same key - a key-zone
                             # change or a velocity-layer switch. One
                             # measurement, because the listener hears
                             # one discontinuity.
    ('joinpow', -1, 0.3),    # ...and the level step there, dB.
    ('layerslip', -1, 0.4),  # ONSET OFFSET BETWEEN OVERLAPPING LAYERS,
                             # ms. Correlated layers a fraction of a
                             # millisecond apart comb-filter: at 1 ms
                             # the first null is 500 Hz.
]
# THE BASE PROFILE IS THE ONLY TABLE, AND EVERY FAMILY IS A PATCH ON IT.
# A family entry below lists ONLY the keys it changes; every other
# criterion keeps the weight here. That is what keeps the families
# comparable: a piano and a pan flute are scored by the same
# measurements, and a family edit is a visible, reviewable delta of
# one or two numbers rather than a second, parallel table.
CRITERIA = PROFILE

# THE SCORE IS A WEIGHTED UTILITY, NOT A RANK. Each criterion is
# min-max normalised across the candidates for this program (1 is the
# best available, 0 the worst), then weighted. A rank-only score let a
# candidate win while 7 kHz narrower or a bit lower, because only order
# entered it; with magnitudes a large loss costs proportionally.
# The rank blend that used to sit beside the magnitude was pinned to
# 1.0 - the ranks were computed and thrown away - so it is gone.

# A SCORE PENALTY on gm, separate from the tie-break. gm.dls is the
# reference the loudness target comes from and it wins many programs by
# being the safe middle; subtracting a little keeps it from crowding
# the bank without pretending its measurements are worse.
#
# THE NUMBER IS SET BY A SWEEP, AND THE SWEEP IS RE-RUN WHEN THE FIELD
# MOVES. The figures that used to be quoted here - 0.9999 leaves 30,
# 0.97 leaves 25, 0.95 leaves 22 - were measured on an older criteria
# set and were already wrong by three programs when re-measured. The
# debug print carries each candidate's balanced score BEFORE this
# multiplier, so one run answers every value (build/tmp/
# gm_penalty_sweep.py replays it):
#
#     penalty   1.00  0.98  0.97  0.96  0.95  0.94  0.92  0.88  0.85  0.80
#     gm wins    27    23    22    19    19    19    15    13    10     7
#
# 0.94..0.96 is a PLATEAU: no program's margin falls inside it, so any
# value there changes nothing and 0.95 was sitting in a dead zone rather
# than at a measured operating point. 0.92 is the first real step.
#
# WHY 0.92 AND NOT LOWER. `size` became cost-per-recording, and gm's
# recordings are the cheapest in the field (4.78 KB each against the
# bank's median 13.8) because they are short and 8-12 bit - so the value
# criterion handed it p15, p38, p80 and p99 on economy alone, and the
# 5% that used to hold it back no longer did. At 0.92 gm gives up
# exactly those four (margins 0.036-0.040) and keeps every program it
# leads by more than 0.065: p13 Xylophone 0.622 against 0.557, p95 Pad
# sweep 0.628, p7 Clavi, p120 Fret Noise 0.745. Below 0.92 the sweep
# starts taking those, which is gm being the better recording and not
# gm crowding the bank.
#
# A KNIFE-EDGE TO KNOW ABOUT: p99 does not go to a better measurement
# but to the FF8 preference. Once gm falls out of the 0.10 band, FF8's
# 1.5 factor wins it at 0.3630 against gm's 0.4607 - a 0.098 giveaway.
# That is the preference working as designed, and it is listed here so
# the cost of a lower penalty is visible rather than discovered.
SOURCE_PENALTY = {'gm.dls': 0.92}

# `size` IS THE COST OF ONE RECORDING, not of one program.
#
# WHAT IT REPLACED, IN TWO STEPS, AND WHY BOTH WERE WRONG.
#
# First it carried os.path.getsize() - the WHOLE SOURCE DLS, identical
# for every program that bank offers. SGM's 127.5 MiB was charged to
# every SGM program although the five it won cost 2.4 MiB together, and
# gm's 2.8 MiB made every gm program look free. It also pointed one way
# only: smaller always won, which crowned a one-shot stretched across
# nine octaves as the cheapest possible answer.
#
# Then it was an absolute band around a fixed centre, two-sided, in
# octaves - and a kit had to be given a SECOND band, because program 128
# is sixty-one instruments rather than one and the melodic band's 1 MiB
# ceiling put every kit in the field at utility 0, tie and tie. Two
# bands, five constants and a special case in the scorer, for a
# criterion that was still only saying "cheap is good".
#
# IT IS NOW A VALUE: cost per distinct recording, in octaves.
#
#     size = log2(bytes / samples / 1024)
#
# sign -1, so a higher cost scores worse, and no band: the reading is
# compared against the other candidates by the ordinary min-max path
# like everything else. That deletes the bands, the kit special case
# and the bespoke utility branch - a kit needs no separate treatment
# because it is measured on the same axis as a flute.
#
# WHY THE LOG. Cost per recording spans 0.39 .. 188.6 KB across the
# shipped bank, 8.9 octaves; min-max on the raw ratio would leave every
# candidate between the median and the maximum scored as one smear at
# the top. In octaves the spread is linear and every candidate lands
# somewhere distinct.
#
# WHY THIS DOES NOT DOUBLE-COUNT. The expensive readings are exactly the
# high-resolution ones (p21 188.6 KB/recording at bits 16.0, density
# 1.32; p19 105.2 at bits 16.0, density 1.86) and the cheap ones exactly
# the thin ones (p78 0.39 at bits 8.4, density 0.11). Cost per recording
# correlates +0.823 with `bits` because that IS the trade: resolution is
# what the bytes buy. The two carry OPPOSITE signs, so they state a
# contest with stated weights rather than counting one fact twice - and
# the "too thin" risk is carried where it is measured, by `density`
# (0.09-0.47 on the thin ones), `stretch` (4800-8300 cents) and `bits`,
# not by a size floor that only re-said the same thing.
#
# The log keeps `size` at sign -1, which matters: the GM_EDGE cap on
# beating the reference applies only to positive criteria, so a program
# can be judged far better value than gm's without being clamped to
# gm + 0.25.

# PROGRAMS WITH A SOURCE THAT MUST NOT BE USED, for a CORROBORATED
# FAULT in that source's samples for that instrument. The grand piano is
# the one instrument every arrangement exposes for minutes at a time and
# gm's is the reference rather than the best recording; Airfont's flutes
# carry a grafted percussive attack layer. Everything else competes on
# the measured criteria.
PROGRAM_BAN = {
    0: ('gm.dls',),
    # AIRFONT 340'S ONE CORROBORATED FAULT. The released library stacks
    # a percussive attack layer on the flutes - the NoteWorthy Composer
    # forum calls it "a 'percussive' flute" and its users repair it by
    # hand. The guitar clicks and the grafted drum samples are reported
    # only by a single reviewer each and are left to the weights: a
    # measured rseam, a low bits reading or a clip count is a better
    # reason to refuse a program than one person's video.
    73: ('airfont_340.dls',), 75: ('airfont_340.dls',),
}

# THE DRUM KIT'S OWN BLACKLIST, held apart from PROGRAM_BAN because the
# two answer different questions. PROGRAM_BAN records a fault IN THE
# SAMPLES; this records a BUDGET decision - and merging them would make
# a size preference read as a claim that a source's kit is defective.
#
# WHY THE KIT NEEDS ITS OWN AT ALL. Program 128 is the single largest
# item in the bank: FluidR3's kit alone is 6042 KB, a fifth of the whole
# thing. The byte budget cannot reach it, and that is not a bug in the
# budget - no other kit can take the job for less than MAX_MOVE_LOSS,
# because FluidR3's measures best by a wide margin (bal 0.74 against
# Creative's 0.64, the runner-up, at a quarter of the size). A budget
# that could move it would have to be allowed to override a 0.10 quality
# gap, which is the same as having no quality guard at all.
#
# So the last lever on the kit is a decision, not a measurement, and it
# belongs in a table a reader can see. FluidR3's kit goes: the bank pays
# 4597 KB less - 11% of it - and takes Creative's 1445 KB kit at bal
# 0.64. That is the whole trade, and it is one line to reverse.
PROGRAM_BAN_KIT = frozenset({'fluidr3_gm.dls'})


def program_ban(prog):
    """The sources barred from this program, as lowercase basenames.

    Returns None when nothing is barred, which is what the callers test
    for; an empty set would be a second way to say the same thing."""
    banned = set(PROGRAM_BAN.get(prog, ()))
    if prog == 128:
        banned |= PROGRAM_BAN_KIT
    return banned or None

# PROGRAMS WHOSE SOURCE IS A POLICY CHOICE, NOT A SLOT: the grand
# piano. They take no preference band and no exposure move.
POLICY_PROGRAMS = (0,)

# A SOURCE-LEVEL PREFERENCE, USED ONLY AS A TIE-BREAK. The weights
# above are about what a sample set measures; this is about which set
# the caller wants when two are close. IT DOES NOT OVERRIDE A
# MEASUREMENT: a candidate competes only if its balanced score is within
# PREFER_MARGIN of the best score among the same-penalty candidates, and
# inside that band the higher factor wins. FF8.dls is the bank this
# project is built around, so its factor is 1.5. --prefer NAME=FACTOR
# overrides one, and --margin sets the band; 1.0 everywhere is no
# preference.
#
# The criterion SIGNS are untouched by any of this: noise, clipping,
# size and the rest keep the direction that makes a worse sample score
# worse. A preference may pick the nearer of two equals, never a worse
# one.
# ONLY FF8 IS PREFERRED. Reputation factors for the other strong
# sources were tried at 1.15-1.25 and they did what a boost does: they
# pulled every near-tie to one group and starved the rest. The measured
# utility already reflects what reputation is a proxy for; the only
# source-level correction left is FF8 (the target) and gm's small
# penalty. A floor (MT32/Airfont forced to N programs) was also tried
# and rejected: it hides a weight problem instead of fixing it.
# A SEAM IS AS BAD AS IT IS REPEATED, and past a point no average of
# other virtues makes it a usable sample: seamrate rises with both the
# step and the slope across the loop and with how often the loop
# recurs, so 0.6 is already a buzz on a held note, not one click. A
# candidate above this is passed over when any candidate sits below it;
# a program whose whole field is seamy still gets its best one.
SEAM_MAX = 0.0

# THE QUOTA HAS A PRICE CEILING. Exposure balance moves a program off
# the source that measured best, and the loss is the score it gives up.
# A move worth 0.08 or more is the quota overriding a real quality
# difference - the census found sixteen programs there, the worst
# losing a third of the scale - so those programs stay where they
# measured best and the over-full source keeps them. The bank is then
# slightly less even and no instrument is chosen against its own
# measurement by a wide margin.
MAX_MOVE_LOSS = 0.05

# A MANUAL TUNING CORRECTION, EMPTY BY DESIGN. Pitch is now handled at
# selection: `tunefit` scores a candidate by how far its regions render
# from equal temperament, and the worst of the field is dropped before
# scoring (see choose). This table stays for the case selection cannot
# reach - a program whose whole field is off, as the choir's still is
# at -10 cents, where its layers glide through the note - and as a
# standing form for tools/retune.py. Entries are cents added to every
# region's fine-tune of that program.
RETUNE = {}

# LEVEL CORRECTIONS MEASURED ON MUSIC. One entry per program whose
# rendered level over the corpus (22 Windows/xqwizard pieces plus 84
# Lakh files, 128/128 programs) missed gm's by more than 2 dB: tenths
# of a decibel to add to every region gain of that program after the
# calibration passes. The passes measure notes they choose; the corpus
# is what the pieces actually play, and the two disagree where a
# source's velocity response differs from the reference's.
LEVEL_TRIM = {}

PREFER = {'ff8.dls': 1.5}
PREFER_MARGIN = 0.10
# THE BAND IS A LAST TIE-BREAK, NOT A RE-WEIGHTING. 0.05 was a tie and
# 0.10 is the widest band that still leaves FF8 around a sixth of the
# programs; beyond it the preference starts outranking measurements.
# The cap below is a second ceiling on how many flips it may cause.
PREFER_CAP = 0.35
# THE EQUAL-SHARE FALLBACK, used when EXPOSURE_SLOPE is zero. With a
# slope the targets come from the measured mean scores instead; this
# fraction is the flat quota those targets replace. --max-exposure
# NAME=N overrides either path for one source.
MAX_EXPOSURE_FRAC = 0.13
# THE BYTE BUDGET, SAME SHAPE AS THE COUNT ONE. A count quota balances
# how many programs a source holds and says nothing about how much
# space they cost, so one source can hold 9% of the programs and 57% of
# the bytes. The target is the same sloped weights expressed in bytes,
# scaled by this: 1.0 is strict proportion, 0 disables the pass. Kept
# above strict proportion because a proportional budget would strip
# every large multi-sampled instrument in favour of the thin ones, and
# how much room the big ones keep is a preference with no measurement
# behind it - iron law nine.
EXPOSURE_BYTE_SHARE = 2.0
# THE QUOTA HAS A SLOPE. Equal shares assume the banks are equally good
# on average, which they are not: measured mean balanced scores differ
# by about 0.11 (FluidR3 0.737, FF8 0.629). The target for a source is
# 1 + SLOPE*SPREAD*(2z-1), z its mean score normalised across sources,
# times the source preference. At SLOPE 0.5 and SPREAD 0.6 the best and
# worst differ by a factor of about 1.9 before preference.
EXPOSURE_SLOPE = 0.5
EXPOSURE_SPREAD = 0.6
MAX_EXPOSURE = {}
# A FLOOR, not a preference: a named source gets at least this many
# programs, taken from the ones where its score deficit is smallest.
# Empty by default; the sloped quota already gives weak banks a share.
MIN_EXPOSURE = {}
# TWO FILES, ONE SET OF RECORDINGS. A source that repackages another's
# wave pool key for key would compete with itself, and a penalty on one
# would simply hand the program to the twin. Sources whose wave pools
# match for 90% or more are one candidate per program.
DEDUP_CONTENT = True

# A GLOBAL BAD-SIDE VETO ON THE CONTINUOUS MEASUREMENTS. A candidate
# whose reading is in the worst VETO_PCT of the whole candidate pool on
# any continuous criterion is not allowed to compete - unless that
# would leave its program with nothing, in which case the veto is
# dropped for that program rather than silencing it.
#
# The two discrete criteria are not percentile material. vellayers is
# tied at 1 for most of the pool, so a 5% cut lands on the tie and
# bans nothing; synth is 0 for all but a handful, so a 5% cut is wider
# than the whole flagged population. They get absolute rules instead:
# a sub-period repeating fragment (synth > 0) is vetoed, and a single
# velocity layer is a weight, not a defect.
#
# SPECIAL CASE FOR BANKS WHOSE LOW READING IS INTRINSIC: `bits` counts
# distinct levels, and a synthesised waveform with few levels is not a
# low-resolution recording. A candidate already flagged synth is
# exempt from the bits veto (it is handled by the synth rule), and a
# source whose samples measure that way is not penalised twice.
VETO_PCT = 5.0
VETO_DISCRETE = ('vellayers', 'synth', 'release', 'looppow', 'loopfreq',
                 'flutter', 'joindist', 'joinpow', 'layerslip', 'mtone',
                 'mlevel')
VETO_STATS = {'removed': 0, 'fallback': 0, 'total': 0}
GATE_STATS = {'removed': 0, 'fallback': 0, 'total': 0, 'tone': 0, 'level': 0}
# THE LOUDNESS BOUND, as a ratio to gm.dls: 25% to 200%, which is
# -12.04 to +6.02 dB. This one is a limit rather than a score: outside
# it the note is either inaudible under the mix or sticks out of it, no
# matter what the weighted score says. gm itself is always inside.
# THE PEAK CEILING, tighter than the RMS one. The mix sums programs;
# a note whose peak sits at 200% of gm's is 6 dB of headroom spent on
# one instrument, and with several coincident that is where the
# clipping comes from. 125% is +1.9 dB: room for the instrument to be
# a little hotter than the reference, not a lot.
PEAK_MAX_DB = 20.0 * math.log10(1.25)
# A FLOOR THAT CANNOT BE REACHED, so a pass can bound the top only.
# enforce_bound moves a program when EITHER side leaves its window; for
# the peak that is wrong - see the call site.
NO_FLOOR_DB = -200.0
LOUD_MIN_DB = 20.0 * math.log10(0.25)
LOUD_MAX_DB = 20.0 * math.log10(2.0)

# HOW MUCH OF AN OUT-OF-BAND ERROR THE LEVEL PASS REMOVES, AS A
# FUNCTION OF THE EXCESS. A flat fraction either flattens the keyboard
# or leaves the worst keys untouched; the excess decides instead - just
# outside the band 30% is removed, twelve decibels outside 60%, linear
# in between. A key that is barely out keeps its character, and a key
# 15 dB hot still gives up most of the error in one pass.
FIX_MIN = 0.30
FIX_MAX = 0.60
FIX_SPAN = 12.0


def _loud_fix(excess):
    return FIX_MIN + (FIX_MAX - FIX_MIN) * min(1.0, excess / FIX_SPAN)

# THE KEY PROFILE IS SCORED, NOT GATED. `keyfit`, `keylow` and
# `keyhigh` carry the measured spread and tails as ordinary weighted
# criteria, and `gmfit` is a soft reach penalty that is zero while the
# engine can lift the quietest key and negative beyond that. Crossing a
# line no longer disqualifies a source; it costs score in proportion.


def veto_thresholds(sources):
    """Bad-side cut values for the continuous criteria, one per key."""
    out = {}
    if VETO_PCT <= 0:
        return out
    for key, sign, _ in CRITERIA:
        if key in VETO_DISCRETE:
            continue
        vals = []
        for prog in range(129):
            if key not in [c[0] for c in criteria_for(prog)]:
                continue
            for s in sources:
                m = s['metrics'].get(prog)
                if m and key in m:
                    vals.append(float(m[key]))
        if len(vals) < 20:
            continue
        vals.sort()
        n = len(vals)
        k = max(1, int(n * VETO_PCT / 100.0))
        out[key] = (vals[k - 1] if sign > 0 else vals[n - k], sign)
    return out


def vetoed(prog, m, thresholds):
    """True when this candidate is in the bad tail of any continuous
    criterion, or is a repeating fragment. A synthesised sample is not
    judged on `bits`: its few levels are its waveform, not a recording
    resolution."""
    synth = m.get('synth', 0.0)
    bad = 0
    for key, sign, _ in criteria_for(prog):
        if key in VETO_DISCRETE or key not in thresholds:
            continue
        if key == 'bits' and synth > 0:
            continue
        v = m.get(key)
        if v is None:
            continue
        thr, sg = thresholds[key]
        if (sg > 0 and v < thr) or (sg < 0 and v > thr):
            bad += 1
    # TWO AXES, not one. A single bottom-5% reading removed 358 of 1019
    # candidates, 60 of them scoring above the eventual winner.
    return synth > 0 or bad >= 2


def prefer(name):
    return PREFER.get(name.lower(), 1.0)


# Programs whose sound IS noise: GM 120-127 are the effects (fret and
# breath noise, seashore, birds, telephone, helicopter, applause,
# gunshot) and 128 is the drum kit. Spectral flatness is what these
# samples should have, so scoring a lower reading as better would reward
# a gated or dull version; the sub-period loop test is equally
# meaningless for them. Everything else still applies.
NOISE_LIKE = set(range(120, 129))


# INSTRUMENT FAMILIES GET THEIR OWN WEIGHTS. The same measurements
# mean different things to different instruments: a wind expresses
# itself with air and a seamless sustain, an organ with a quiet steady
# loop, a guitar with its attack. These are overrides on CRITERIA, not
# different criteria - directions and the rest of the measurement stay
# identical. GM ranges; 128 is the drum kit.
FAMILIES = (
    # TUNING DECIDES THE PIANO, AND THE CURVE IS ONLY A TIEBREAK
    # THERE. The family carries rtune because a grand piano is judged
    # by its worst key. At 3.0 the score still preferred a source whose
    # worst measured key was 23.6 cents out over one at 6.8, on a
    # margin of 0.011 - brightness and the similarity term together
    # outweighed three times the tuning weight. At 4.0 the measured
    # difference decides, and the similarity term stays as the
    # character tiebreak at half weight.
    (0, 8, {'rtune': 4.0, 'sim': 0.8}),               # piano
    (8, 16, {'depth': 0.8}),
    (16, 24, {'noise': 1.4, 'seamrate': 2.0, 'depth': 0.6}),  # organ
    (24, 32, {'startrisk': 1.0, 'seamrate': 1.5, 'air': 2.5,
               'loopslope': 2.5}),                   # guitar
    (32, 40, {'seamrate': 1.5}),                      # bass
    (40, 48, {'depth': 1.4, 'seamrate': 1.5}),       # strings
    (48, 56, {'depth': 1.4, 'seamrate': 1.5}),        # ensemble
    (56, 64, {'seamrate': 1.5}),                     # brass
    (64, 72, {'seamrate': 1.5}),                      # reed
    # THE SEAM HEARD OUTRANKS THE SEAM COUNTED. seamrate discounts a
    # step by how often it repeats, which flatters a long loop with a
    # big step - SGM's pan flute reads 0.443 there, better than
    # SC-55's 2.976, while the rendered seam is 0.487 against 0.159.
    # A wind is played legato; the click is the click. 10 and 12
    # select identically; 8 moves four programs (p78's whistle from
    # arachno's 47-cent worst key to gm's 6, but p110's fiddle from
    # 11 to 26), so the smaller number that costs nothing is kept.
    # Under the group cap this is a fifth of a wind's score.
    (72, 80, {'depth': 0.4, 'air': 2.5, 'seamrate': 2.0,
              'rseam': 10.0, 'rtune': 3.5}),          # pipe
    (88, 96, {'seamrate': 2.0, 'noise': 1.3, 'release': 5.0}),  # synth pad
    (96, 104, {'seamrate': 1.8, 'rtune': 1.2}),      # synth effects
    # DYNAMICS COME FIRST FOR A KIT. bRIGHTNESS AND BIT DEPTH WON THE
    # KIT FOR A ONE-LAYER BANK at a 0.010 margin: a kit with velocity
    # layers plays a ghost note and an accent as different sounds, and
    # that is the difference a drummer hears first. depth now carries
    # more than the spectrum.
    (128, 129, {'bw': 0.3, 'bits': 1.4, 'density': 1.5,
                'startrisk': 1.0, 'levelfit': 1.5, 'depth': 2.0,
                'size': 0.8}),                        # drum kit
                             # bw 0.3, not 1.0: a kit's bandwidth is the
                             # cymbals, not its fidelity. At 1.0 the
                             # brightness term carried Creative_28MB's
                             # kit (11.7 kHz against FluidR3's 4.7) into
                             # the bank, and on the piece it rendered a
                             # 0.44 seam against 0.15 with 5.7% of its
                             # energy above 8 kHz against 1.8 - the kit
                             # heard as odd before.
)


# ONE MEASUREMENT, ONE SCORE. Several criteria above look at the same
# property, and weighting each on its own pays for it twice. The pairs
# fall into two kinds:
#
#   DERIVED  seamrate IS (loopstep + loopslope) x a repeat rate, and
#            startrisk IS startclick gated by the attack. Measured
#            across the eight sources, startclick and startrisk rank
#            alike within a program at a median +0.81, and
#            loopslope/seamrate at +0.64 - they are the same number
#            counted twice.
#
#   TWO LAYERS  rumble/rsub, dc/rdc, rtune/tunefit and the rendered
#            seam compare a sample-level reading with a rendered one.
#            The rank correlation is weak (rumble/rsub +0.04,
#            rtune/tunefit -0.14) because the render sums regions and
#            shifts registers, so the two are NOT interchangeable -
#            but they answer the same question, and a source with the
#            defect shows it in both.
#
# THE GROUP'S TOTAL IS ITS LARGEST MEMBER, NOT THE SUM. A family that
# raises seamrate to 4.0 says seams matter that much; it does not say
# they matter 4.0 + 0.8 + 0.5 + 0.8. Splitting the cap in proportion
# to the listed weights keeps the family's intent and stops the count
# from depending on how many ways the same defect is measured.
#
# NOT GROUPED, because the measurement says they are different
# questions: noise and bits (median +0.02), density and stretch
# (+0.05), bw and air (+0.33), air and rair (-0.10). The last is
# worth stating: the sample-level top and the rendered top rank
# differently, so the render carries information the sample does not.
GROUP_OF = {
    'loopstep': 'loop', 'loopslope': 'loop', 'seamrate': 'loop',
    'rseam': 'loop',
    # Power and timbre across the seam are the other two continuity
    # axes; kept apart from the phase group so that adding them does
    # not dilute loopstep by splitting one cap further.
    'looppow': 'loopc', 'loopfreq': 'loopc',
    'flutter': 'beat',
    # THE TARGET AND THE CEILING ARE TWO ROLES, NOT ONE MEASUREMENT.
    # The standard the bank is judged by is "as much like FF8 as it can
    # be, and not too far off gm". `ff8sim` and `mtone` both measure
    # likeness to FF8 - the target. `sim` and `mlevel` both measure
    # distance from gm - the ceiling that keeps the likeness from
    # running away. They were all under `spectral` and `bright` by
    # instrument rather than by role, which let the target and the
    # ceiling dilute each other: ff8sim came out at 2.2 of 32.6 and sim
    # at 1.3, an FF8 axis of 10.9% where the standard asks for it to
    # lead.
    'ff8sim': 'ff8like', 'mtone': 'ff8like',
    'sim': 'gmfar', 'mlevel': 'gmfar',
    'joindist': 'join', 'joinpow': 'join', 'layerslip': 'join',
    'startrisk': 'start',
    'rumble': 'infra', 'rsub': 'infra',
    'dc': 'offset', 'rdc': 'offset',
    'rtune': 'tune', 'tunefit': 'tune',
    'keyfit': 'keyprofile', 'keylow': 'keyprofile',
    'keyhigh': 'keyprofile',
    # air and rair stay outside every group. They were checked against
    # the others (bw/air +0.33, air/rair -0.10): the sample reading and
    # the rendered one rank the field differently, so they are not one
    # measurement counted twice. airfit is grouped with nothing either -
    # see the PROFILE entry: it is a one-sided floor, not a second
    # brightness score.
}


# BEATING THE REFERENCE IS WORTH SOMETHING, AND ONLY SOMETHING.
# Every criterion already lives on 0..1 utilities (best in the field is
# 1, worst is 0, the sign folded in). A candidate may sit at most
# GM_EDGE above gm's own utility on any one of them; below that line
# the field ranks as it always did, and gm keeps its own place (it is
# NOT moved to the top, which is what a hard ceiling at the reference
# did - that handed gm 71 of 128 programs). The rule is the same for
# defects, because the utility already has the sign folded in: a source
# far cleaner than gm earns at most GM_EDGE over it, not the whole
# range. The cap applies to the "more is better" criteria; a defect
# reading is left uncapped because there is no such thing as too clean:
# a source 6 dB quieter than gm is better, not suspicious, and capping
# that also squeezed every other criterion toward the reference - it
# handed gm 73 of 128 programs at a tighter edge, a sweep in the wrong
# direction. Criteria without a gm reading (airfit, sim, ff8sim) are
# left alone: airfit is already one-sided and the similarity curves
# exclude the reference by construction.
GM_EDGE = 0.25


# WHAT A GROUP OF MEASUREMENTS IS WORTH, stated once here.
#
# It used to be derived: a group's total was set to its LARGEST MEMBER's
# weight, and every member scaled to fit. That made a group's share
# accidental - whatever its biggest criterion happened to be declared as
# - and it made membership a silent penalty: adding a criterion cut
# every other member of the group. Measured on the old rule, `loop` held
# four criteria at 3.6 raw and 1.5 effective (42% of their own weights)
# while `start`, with one, kept 100%.
#
# Now the group's total is a decision and the members split it in
# proportion to their own weights. Adding a member redistributes inside
# the group, which is what grouping is for: they measure one fact and
# the fact is worth what it is worth regardless of how many angles it
# is looked at from.
GROUP_CAP = {
    # THE TARGET: like FF8. ff8sim is the likeness, mtone is the
    # brightness half of the same question.
    #
    # 7.0, SET SO SIMILARITY LEADS THE TIMBRE AXIS. The standard is
    # "the timbre weight is led by likeness to FF8 and gm"; against the
    # raw brightness criteria (bw, air, rair, airfit = 3.9) and the
    # sampling quality ones (bits, density, depth, stretch = 3.5), 5.0
    # left likeness at 7.0 to their 7.4 - level, not lead. 7.0 puts it
    # at 8.8 against their 7.4.
    'ff8like': 7.0,
    # THE CEILING: not too far from gm. sim is the spectral distance,
    # mlevel the level one. Kept below the target so the bank leans
    # toward FF8 rather than toward the middle of the two.
    'gmfar': 2.5,
    'tune': 3.0,
    'loop': 2.0,
    'infra': 1.8,
    'keyprofile': 1.4,
    'loopc': 0.8,
    'offset': 0.6,
    'join': 0.6,
}


def share_group_weight(out, active):
    """Split each group's declared weight among its active members.

    `active` is the set of criteria the field actually carries for this
    program: a program FF8 does not have makes ff8sim inactive, and its
    share is then not split with it - the rest of the group gets more,
    because the group's total is a decision and not a sum of its parts."""
    gsum = {}
    for k, _sign, w in out:
        g = GROUP_OF.get(k)
        if g is None or not active.get(k, True):
            continue
        gsum[g] = gsum.get(g, 0.0) + w
    res = []
    for k, sign, w in out:
        g = GROUP_OF.get(k)
        cap = GROUP_CAP.get(g)
        if cap is not None and active.get(k, True) and gsum.get(g, 0.0) > 0.0:
            w = cap * w / gsum[g]
        res.append((k, sign, w))
    return res


def criteria_for(prog):
    weights = {}
    for lo, hi, over in FAMILIES:
        if lo <= prog < hi:
            weights = over
            break
    out = [(a, b, weights.get(a, w)) for a, b, w in CRITERIA]
    if os.environ.get('KSB_NO_SIM'):
        out = [c for c in out if c[0] != 'sim']
    if prog in NOISE_LIKE:
        out = [c for c in out if c[0] != 'noise']
    return out


def ranks(values):
    """Rank values 0..1, larger always better, ties sharing a rank."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Average position of the tie group, as a percentile.
        r = (i + j) / 2.0 / (n - 1)
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def _magnitude(vals, sign):
    n = len(vals)
    if n == 0:
        return []
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [1.0] * n
    if sign > 0:
        return [(v - lo) / (hi - lo) for v in vals]
    return [(hi - v) / (hi - lo) for v in vals]


def _depth_vector(cands):
    # keycover IS COVERAGE, NOT DEPTH, and choose() already gates on it
    # at 90%: scoring it again counted one fact twice. Depth here is
    # the number of velocity layers.
    vl = [float(m.get('vellayers', 0.0)) for _, m in cands]
    return ranks(vl)


def fill_missing(total, used, us, idx, weight, n):
    """Give the candidates without a reading the field's median utility.

    THEY USED TO BE LEFT OUT OF THE DENOMINATOR, which made the score an
    average over different criteria for different candidates: a source
    missing keyfit or gmfit was compared against one averaged over more
    terms. A missing reading carries no information, and the median is
    what 'no information' is worth in the rest of the score."""
    if not us:
        return
    med = sorted(us)[len(us) // 2]
    have = set(idx)
    for i in range(n):
        if i not in have:
            total[i] += weight * med
            used[i] += weight


def balanced_score(cands, prog):
    """cands: [(source, metrics)]. -> one score in 0..1 per candidate.

    Each criterion is ranked across the sources offering this program and
    the ranks are averaged with the criterion's weight. A criterion a
    source does not carry - a one-shot has no loop seam, FF8 has no
    similarity to itself - is left out of THAT source's average rather
    than scored as its worst. The criteria themselves depend on the
    program: a noise instrument is not judged on how little noise it has
    (see NOISE_LIKE)."""
    n = len(cands)
    total = [0.0] * n
    used = [0.0] * n
    # PER-CRITERION UTILITY, kept so a caller can ask WHICH criterion
    # decided a program. The weighted mean alone cannot answer that: a
    # candidate may win by 0.02 with one criterion carrying the whole
    # gap and thirty-nine flat, which is a decision made by a single
    # instrument rather than by the field. See the debug print at the
    # call site.
    per = [dict() for _ in range(n)]
    crit = criteria_for(prog)
    # THE REFERENCE, for the capped criteria. gm is in the candidate
    # list for every program except the piano, where it is barred; with
    # no reference the capped criteria fall back to ranking alone.
    gidx = [i for i, (s, m) in enumerate(cands)
            if os.path.basename(s['path']).lower() == 'gm.dls']
    gref = cands[gidx[0]][1] if gidx else None
    # WHICH CRITERIA THIS FIELD CAN ACTUALLY SCORE. Two candidates are
    # the minimum for a comparison; `depth` is computed across the whole
    # field. A criterion the field does not carry stays out of the
    # group split as well as out of the sum.
    active = {}
    for k, _sign, _w in crit:
        if k == 'depth':
            active[k] = True
        else:
            active[k] = sum(1 for _, m in cands if k in m) >= 2
    crit = share_group_weight(crit, active)
    for key, sign, weight in crit:
        if not active.get(key, True):
            continue
        if key == 'airfit':
            # A FLOOR, NOT A READING. What is stored is rair/gm; the
            # utility is min(1, ratio) - 1, so reaching the reference is
            # 0 and every hertz it falls short costs in proportion. No
            # ranks, no min-max: either would make the floor a second
            # brightness score and pay for it twice.
            idx = [i for i, (_, m) in enumerate(cands) if key in m]
            us = [min(1.0, float(cands[i][1][key])) - 1.0 for i in idx]
            for i, u in zip(idx, us):
                total[i] += weight * u
                used[i] += weight
                per[i][key] = u
            fill_missing(total, used, us, idx, weight, n)
            continue
        if key == 'mlevel':
            # A FLOOR AGAINST GM'S LEVEL, dB measured on the corpus's
            # own notes, saturated past 12 dB so one pathological read
            # cannot decide a program on its own.
            #
            # /12.0 IS THE POINT OF THIS BRANCH. The floor is stated in
            # decibels and the min-max criteria next door are stated in
            # 0..1, and a weighted sum that mixes the two has weights
            # that do not mean what they say: this criterion's 1.0 could
            # move the score 12 units while `ff8sim`'s 3.5 moved it 3.5,
            # which put the gm ceiling at 25% of the influence against
            # the FF8 target's 12% - the standard asks for the target to
            # lead. Dividing by the span puts every criterion's utility
            # in [-1, 0] or [0, 1], so the weights are comparable and
            # GROUP_CAP's totals describe influence rather than rank.
            idx = [i for i, (_, m) in enumerate(cands) if key in m]
            us = [-min(12.0, max(0.0, -float(cands[i][1][key]))) / 12.0
                  for i in idx]
            for i, u in zip(idx, us):
                total[i] += weight * u
                used[i] += weight
                per[i][key] = u
            fill_missing(total, used, us, idx, weight, n)
            continue
        if key == 'mtone':
            # A FLOOR: zero when the candidate's brightness matches
            # FF8's on the same notes, and the log distance beyond.
            idx = [i for i, (_, m) in enumerate(cands) if key in m]
            us = [-min(1.5, abs(float(cands[i][1][key]))) / 1.5 for i in idx]
            for i, u in zip(idx, us):
                total[i] += weight * u
                used[i] += weight
                per[i][key] = u
            fill_missing(total, used, us, idx, weight, n)
            continue
        if key == 'depth':
            vals = _depth_vector(cands)
            idx = list(range(n))
        else:
            vals = []
            idx = []
            for i, (_, m) in enumerate(cands):
                if key in m:
                    idx.append(i)
                    vals.append(float(m[key]))
        if len(idx) < 2:
            continue
        mag = _magnitude(vals, sign)
        # THE DRUM KIT IS NOT A BETTER RECORDING OF THE SAME THING.
        # A kit's key map and its velocity layers are the instrument's
        # dynamics, not a quality margin over gm's; capping them at the
        # reference's single layer is what put a one-layer FF8 kit in
        # the bank, and one layer plays every ghost note as an accent.
        # Program 128 is exempt from the edge, as it is from the
        # exposure quota.
        if prog != 128 and sign > 0 and gref is not None and len(vals) > 1:
            # THE EDGE OVER THE REFERENCE IS CAPPED, IN UTILITY SPACE.
            # `mag` is already "better is higher" for both signs, so
            # one rule covers a wider band and a quieter floor alike.
            gpos = None
            for pos, i in enumerate(idx):
                if i in gidx:
                    gpos = pos
                    break
            if gpos is not None:
                ug = mag[gpos]
                for pos in range(len(mag)):
                    if idx[pos] in gidx:
                        continue
                    mag[pos] = min(mag[pos], ug + GM_EDGE)
        for pos, i in enumerate(idx):
            total[i] += weight * mag[pos]
            used[i] += weight
            per[i][key] = mag[pos]
        fill_missing(total, used, mag, idx, weight, n)
    # A PURE WEIGHTED SUM. What used to follow was a second scoring
    # rule: seamrate > 0.6 cost 0.10, > 0.9 cost 0.20, a looping
    # slope 0.10, an exposed start click 0.15 and broadband hiss 0.20.
    # Those defects already have weights above; subtracting them again
    # made a candidate's score depend on which list a number appeared
    # in, and it is the same over-correction the gates were.
    return ([total[i] / used[i] if used[i] > 0.0 else 0.5 for i in range(n)],
            per)


def programme(d, lins):
    """-> ({program: features}, regions, rates) for one DLS."""
    feats = collections.defaultdict(collections.Counter)
    regions = collections.Counter()
    metrics = {}
    used = collections.defaultdict(set)
    stretch = {}
    gain_by_prog = collections.defaultdict(list)
    region_level = collections.defaultdict(list)
    cache = {}
    keycover = collections.defaultdict(set)
    release_by_prog = collections.defaultdict(list)
    region_info = collections.defaultdict(list)
    attack_by_prog = collections.defaultdict(list)
    detune_by_prog = collections.defaultdict(list)
    wave_bytes = {}
    wv = find(d, 12, len(d), b'LIST', b'wvpl')
    pool = []
    if wv[0] is not None:
        for c3, f3, d3, e3 in chunks(d, wv[0], wv[1]):
            if c3 == b'LIST' and f3 == b'wave':
                pool.append((d3, e3))
    for cid, f, a, b in chunks(d, lins[0], lins[1]):
        if cid != b'LIST' or f != b'ins ':
            continue
        ih, _ = find(d, a, b, b'insh')
        if ih is None:
            continue
        _, bank, prog = struct.unpack('<III', d[ih:ih + 12])
        # THE DRUM CHANNEL'S KIT IS ONE INSTRUMENT: bank 0x80000000,
        # program 0. A DLS may carry further percussion banks under that
        # flag; the builder ignores them, so measuring their regions as
        # part of program 128 would credit a kit with keys the built kit
        # does not have (Airfont states 27..87 across several drum-bank
        # instruments, while its kit proper starts at 34).
        if bank == 0x80000000:
            if (prog & 0x7F) != 0:
                continue
            key = 128
        elif bank != 0:
            continue
        else:
            key = prog & 0x7F
        regs = []

        def art(x, y):
            la, lb = find(d, x, y, b'LIST', b'lart')
            if la is None:
                la, lb = find(d, x, y, b'LIST', b'lar2')
            if la is None:
                return
            aa, _ = find(d, la, lb, b'art1')
            if aa is None:
                aa, _ = find(d, la, lb, b'art2')
            if aa is None:
                return
            cb, n = struct.unpack('<II', d[aa:aa + 8])
            for i in range(n):
                o = aa + cb + i * 12
                if o + 12 > len(d):
                    break
                src, ctl, dst, tr, sc = struct.unpack('<HHHHi', d[o:o + 12])
                s = src & 0xFF
                if dst in (0x500, 0x501):
                    feats[key]['filter'] += 1
                # EG1 attack, in DLS time cents scaled by 65536. Kept
                # because startclick is only a click when the envelope
                # does not cover it: a sample that opens at half scale
                # under a 20 ms attack is a natural onset, and the same
                # sample at 0 ms is a tick.
                if dst == 0x206:
                    attack_by_prog[key].append(sc)
                if dst in (0x20B, 0x20C, 0x30F, 0x310):
                    feats[key]['delayhold'] += 1
                if dst in (0x114, 0x115) or s == 9:
                    feats[key]['viblfo'] += 1
                if s == 3 and dst == 3:
                    feats[key]['keypitch'] += 1
                if dst in (0x80, 0x81):
                    feats[key]['sends'] += 1

        art(a, b)
        lr, lrb = find(d, a, b, b'LIST', b'lrgn')
        if lr is not None:
            for c2, f2, ra, rb in chunks(d, lr, lrb):
                if c2 != b'LIST' or f2 not in (b'rgn ', b'rgn2'):
                    continue
                regs.append((ra, rb))
                art(ra, rb)
        regions[key] += len(regs)
    # Waves each program reaches, through its regions' wave links.
    for cid, f, a, b in chunks(d, lins[0], lins[1]):
        if cid != b'LIST' or f != b'ins ':
            continue
        ih, _ = find(d, a, b, b'insh')
        if ih is None:
            continue
        _, bank, prog = struct.unpack('<III', d[ih:ih + 12])
        # THE DRUM CHANNEL'S KIT IS ONE INSTRUMENT: bank 0x80000000,
        # program 0. A DLS may carry further percussion banks under that
        # flag; the builder ignores them, so measuring their regions as
        # part of program 128 would credit a kit with keys the built kit
        # does not have (Airfont states 27..87 across several drum-bank
        # instruments, while its kit proper starts at 34).
        if bank == 0x80000000:
            if (prog & 0x7F) != 0:
                continue
            key = 128
        elif bank != 0:
            continue
        else:
            key = prog & 0x7F
        lr, lrb = find(d, a, b, b'LIST', b'lrgn')
        if lr is None:
            continue
        # The instrument's own articulators, which a region that
        # declares none inherits - the same precedence the builder uses.
        eg_inst = _ksb_build.envelope(d, a, b)
        for c2, f2, ra, rb in chunks(d, lr, lrb):
            if c2 != b'LIST' or f2 not in (b'rgn ', b'rgn2'):
                continue
            wa, wb = find(d, ra, rb, b'wlnk')
            if wa is None or wb - wa < 12:
                continue
            idx = struct.unpack('<I', d[wa + 8:wa + 12])[0]
            if idx >= len(pool):
                continue
            used[key].add(idx)
            kh2, _ = find(d, ra, rb, b'rgnh')
            if kh2 is not None and kh2 + 8 <= len(d):
                region_info[key].append(
                    struct.unpack('<HHHH', d[kh2:kh2 + 8]) + (idx,))
            ws_off, ws_end = find(d, ra, rb, b'wsmp')
            att = None
            if ws_off is not None and ws_end - ws_off >= 12:
                att = struct.unpack_from('<i', d, ws_off + 8)[0]
            if att is None:
                ws_off, ws_end = find(d, pool[idx][0], pool[idx][1], b'wsmp')
                if ws_off is not None and ws_end - ws_off >= 12:
                    att = struct.unpack_from('<i', d, ws_off + 8)[0]
            kh, _ = find(d, ra, rb, b'rgnh')
            klo, khi = struct.unpack('<HH', d[kh:kh + 4])
            # THE TAIL, from the same articulation reader the bank
            # builder uses - one authority for the envelope. A region
            # that declares nothing inherits the instrument's list.
            eg = _ksb_build.envelope(d, ra, rb) or eg_inst
            if eg:
                release_by_prog[key].append(eg.get('release', 0))
            if att is not None:
                tenths = att / 65536.0
                gain_by_prog[key].append(tenths)   # tenths of a dB
                rm = cache.get(idx)
                if rm is None:
                    cache[idx] = wave_metrics(d, pool[idx])
                    rm = cache[idx]
                if rm and rm.get('rms', 0.0) > 0.0:
                    lvl = rm['rms'] * 10.0 ** (tenths / 200.0)
                    region_level[key].append((klo, khi, lvl))
            if idx not in wave_bytes:
                da, db = find(d, pool[idx][0], pool[idx][1], b'data')
                wave_bytes[idx] = (db - da) if da is not None else 0
            # How far this region pulls its sample from the recorded
            # pitch: the worst key it covers, in cents.
            if idx not in cache:
                cache[idx] = wave_metrics(d, pool[idx])
            unity = cache[idx]['unity'] if cache[idx] else 60
            # THE TUNING THIS REGION WILL RENDER AT: the sample's own
            # offset from its unity note, plus the file's fine-tune, is
            # what a keyboard note comes out as. One bad layer spoils a
            # chord, so the program's figure is the worst tenth of its
            # regions, not the average.
            rm = cache.get(idx) or {}
            ru, rfine, rll = rm.get('unity', unity), 0, rm.get('ll', 0)
            if ws_off is not None and ws_end - ws_off >= 12:
                cb, u2, f2v, _g, _o, nlp = struct.unpack(
                    '<IHhiII', d[ws_off:ws_off + 20])
                ru, rfine = int(u2), int(f2v)
                if nlp and ws_end - ws_off >= cb + 16:
                    _, _, _rls, rll = struct.unpack(
                        '<IIII', d[ws_off + cb:ws_off + cb + 16])
            rate = rm.get('rate', 0)
            f_unity = 440.0 * 2.0 ** ((ru - 69) / 12.0)
            det = None
            # A FOUR-FIGURE FINE IS A RANK, NOT A DETUNE. Reed Organ
            # stacks one key twice, at fine -17 and 1179, so the layers
            # sit an octave apart on purpose and the second one's sample
            # is an octave low by design. Judging it by that offset
            # rejects a bank that renders in tune.
            if abs(rfine) > 100:
                det = None
            elif rate and rll and rll >= 8:
                looprate = rate / float(rll)
                off = 1200.0 * math.log2(looprate / f_unity)
                # AN OCTAVE IS NOT A DETUNE. A sample recorded a whole
                # number of octaves from its note - which is a unity
                # note written one octave out, not a voice out of tune -
                # reads 1200 or 2400 cents away, and folding it back
                # before judging keeps that separate from the 15 or 20
                # cents that is a voice off. Both lose the candidate;
                # the number says which it was.
                fold = off - 1200.0 * round(off / 1200.0)
                # NEAR THE NOTE, OR A WHOLE OCTAVE ON A SHORT LOOP. A
                # long loop repeats many periods, so rate/ll is the loop
                # rate and not the pitch: it lands somewhere random, and
                # where it happens to sit an octave from the note is
                # coincidence, not a unity note written out. A short
                # loop is one period - then rate/ll IS the pitch, and a
                # whole octave away means the unity note is wrong by
                # that octave, which is what the number reports.
                if abs(off) <= 150.0:
                    det = abs(off)
                elif abs(fold) <= 30.0 and rll <= 1024:
                    # A SHORT LOOP MAY HOLD TWO PERIODS. rate/ll is
                    # then the loop rate, an octave above the note,
                    # while the rendered pitch is right: measured
                    # against the finished bank, every such candidate
                    # rendered within 4 cents. Ask the fundamental
                    # itself before calling it a unity note an octave
                    # out; only where that agrees is it an octave.
                    f_loop = rm.get('f0', 0.0)
                    det = (abs(1200.0 * math.log2(f_loop / f_unity))
                           if f_loop > 0.0 else None)
                else:
                    det = None
            elif rm.get('f0', 0.0) > 0.0 and rate:
                # ONE-SHOT: no loop gives a period, so the middle of
                # the sample is measured directly. Piano, percussion
                # and plucked notes are all one-shot; without this
                # their tuning was never looked at, and `tunefit` sat
                # at 1.0 for every one of them.
                off = 1200.0 * math.log2(rm['f0'] / f_unity)
                if abs(off) <= 150.0:
                    det = abs(off)
            # ONLY A LOOP THAT IS THE PERIOD. rate/ll is exact when
            # the loop repeats at the note's own rate (a synthesised
            # one-period loop), and that is the case where the file's
            # declared pitch and the rendered pitch disagree by a
            # fixed, audible amount. A spectral estimate on a
            # distorted or layered sample was measured wrong both
            # ways on onestop's guitar, so it is not evidence for a
            # selection.
            if det is not None:
                # TWO READINGS, and they fail differently. detune is the
                # sample against its own note: a bank whose recording is
                # flat is the one to pass over. tunedev is what a key
                # then renders as, fine-tune included, and a program
                # still scored on that can be penalised without being
                # rejected for a field that is otherwise fine.
                detune_by_prog[key].append(abs(det))
            kh, _ = find(d, ra, rb, b'rgnh')
            if kh is not None:
                klo, khi, vlo, vhi = struct.unpack('<HHHH', d[kh:kh + 8])
                st = max(abs(int(klo) - unity), abs(int(khi) - unity)) * 100
                stretch[key] = max(stretch.get(key, 0), st)
                keycover[key].update(range(int(klo), int(khi) + 1))
    for key, idxs in used.items():
        vals = [cache[i] for i in sorted(idxs) if cache.get(i)]
        if not vals:
            continue
        med = {}
        for k in ('bw', 'air', 'noise', 'loopstep', 'loopslope', 'seamrate', 'clip', 'dc',
                  'rumble', 'bits', 'synth', 'centroid', 'looppow', 'loopfreq',
                  'flutter', 'hnr', 'oddeven'):
            col = sorted(v[k] for v in vals)
            med[k] = col[len(col) // 2]
        # A CUT IS EITHER AUDIBLE OR NOT: one region's tick is not
        # averaged away by the other regions' clean starts.
        med['startclick'] = max(v['startclick'] for v in vals)
        # A SYNTHESISED WAVEFORM'S FEW LEVELS ARE ITS SHAPE, not a
        # recording resolution; the criterion says nothing there.
        if med.get('synth', 0.0) > 0.0:
            med.pop('bits', None)
        med['stretch'] = stretch.get(key, 0)
        rv = sorted(release_by_prog.get(key, []))
        if rv:
            med['release'] = rv[len(rv) // 2]
        # JOIN COST, after the unit-selection literature (Hunt & Black):
        # what a listener hears at a boundary is the spectral distance
        # and the level step between the two units the keyboard puts on
        # either side of it. Two kinds of boundary: the key where the
        # zone changes sample, and the velocity where the layer does.
        ri = region_info.get(key) or []
        jd, jp, sl = [], [], []

        def join_at(i1, i2, kb):
            """Spectral and level step between two regions at key kb."""
            m1, m2 = cache.get(i1), cache.get(i2)
            if not m1 or not m2:
                return
            if m1.get('centroid', 0.0) <= 0 or m2.get('centroid', 0.0) <= 0:
                return
            if m1.get('rms', 0.0) <= 0 or m2.get('rms', 0.0) <= 0:
                return
            c1 = m1['centroid'] * 2.0 ** ((kb - m1.get('unity', 60)) / 12.0)
            c2 = m2['centroid'] * 2.0 ** ((kb - m2.get('unity', 60)) / 12.0)
            jd.append(abs(math.log2(c2 / c1)))
            jp.append(abs(20.0 * math.log10(m2['rms'] / m1['rms'])))

        for k1, h1, l1, u1, i1 in ri:
            for k2, h2, l2, u2, i2 in ri:
                if i1 == i2 and (k1, h1, l1, u1) == (k2, h2, l2, u2):
                    continue
                overlap = not (u1 < l2 or u2 < l1)
                if h1 + 1 == k2 and overlap:
                    join_at(i1, i2, k2)
                if k1 == k2 and h1 == h2 and overlap:
                    m1, m2 = cache.get(i1), cache.get(i2)
                    if m1 and m2:
                        sl.append(abs(m2.get('onset', 0.0) - m1.get('onset', 0.0)))
                    if u1 + 1 == l2:
                        join_at(i1, i2, k1)
        if jd:
            med['joindist'] = max(jd)
            med['joinpow'] = max(jp)
        if sl:
            med['layerslip'] = max(sl)
        med['samples'] = len(idxs)
        med['bytes'] = sum(wave_bytes.get(i, 0) for i in idxs)
        # COST OF ONE RECORDING, in octaves. See the `size` block at
        # SOURCE_PENALTY for why this is the quantity and why the log.
        med['size'] = (math.log(med['bytes'] / 1024.0 / med['samples'], 2.0)
                       if med['bytes'] > 0 and med['samples'] else None)
        gcol = sorted(gain_by_prog.get(key, [0.0]))
        med['gain'] = gcol[len(gcol) // 2]
        # SAMPLING DENSITY: distinct samples per octave of covered keys.
        ks = keycover.get(key, set())
        octaves = max(0.5, (max(ks) - min(ks) + 1) / 12.0) if ks else 1.0
        med['density'] = len(idxs) / octaves
        # PROGRAM LEVEL, in dB of the estimated region levels.
        lv = sorted(20.0 * math.log10(v) for _, _, v in
                    region_level.get(key, []) if v > 0)
        # PER-KEY LEVELS, for the same reason the calibration is per
        # region: a program can average to gm's level while one end of
        # the keyboard is silent. Each region's estimated level applies
        # to every key it covers; overlapping regions are averaged in
        # the log domain.
        bykey = collections.defaultdict(list)
        for klo, khi, lvl in region_level.get(key, []):
            for kk in range(klo, khi + 1):
                bykey[kk].append(20.0 * math.log10(lvl))
        med['keylevels'] = {kk: sum(v) / len(v) for kk, v in bykey.items()}
        # PROGRAM LEVEL, in dB of the estimated region levels. This is
        # the scale gm.dls is measured on: a source whose program is far
        # below gm's cannot be lifted by the engine's +12 dB boost, and
        # the deficit is invisible to gain, which is relative
        # to the source's own samples.
        med['level'] = lv[len(lv) // 2] if lv else None
        # HOW MANY SAMPLES PLAY ONE KEY, median across the keys it
        # covers. This used to be the number of DISTINCT (vlo, vhi)
        # tuples in the program, which measures something else: a kit
        # that gives each of its 61 drums its own velocity ceiling
        # scored 68 while every key held ONE sample (GeneralUser), and
        # a kit with a real two-layer split across the board scored 8
        # (FluidR3). The metric crowned the per-key ceiling as depth.
        # Only regions that resolve to a wave are counted - a region
        # with a key range and no sample makes no sound.
        per_key = {}
        for klo2, khi2, _vlo2, _vhi2, idx2 in region_info.get(key, []):
            for k in range(int(klo2), int(khi2) + 1):
                per_key.setdefault(k, set()).add(idx2)
        nl = sorted(len(v) for v in per_key.values())
        med['vellayers'] = nl[len(nl) // 2] if nl else 1
        # A drum kit is judged over the GM percussion range, not the
        # whole keyboard: keys 27..87 are what a GM file asks a kit for,
        # and a kit that stops at 34 leaves a note silent (onestop's
        # low drum is key 28). A melodic program is judged over 0..127.
        if key == 128:
            med['keycover'] = len(keycover.get(key, set())
                                  & set(range(27, 88))) / 61.0
        else:
            # COVERAGE OVER THE MUSICAL RANGE, not 0..127. Missing keys at
            # the extremes are the norm - gm itself is the exception - and
            # scoring them made a whole bank look broken for keys no
            # arrangement uses. What matters is the 88-key piano range.
            med['keycover'] = len(keycover.get(key, set())
                                  & set(range(21, 109))) / 88.0
        # EG1 ATTACK, through the builder's own conversion, and the
        # click it may cover. A sample whose first frame is a jump is a
        # tick only when the envelope opens fast enough to expose it;
        # under a 10 ms attack the same frame is an ordinary onset.
        # THE FASTEST REGION DECIDES. A program's regions may restate
        # the attack, and the instrument's own 1 s attack does not cover
        # a region that opens in 0 ms: key 21 would tick while the rest
        # of the keyboard swells. Measured on the bank: p18's regions
        # hold 0 ms while the instrument says 1000.
        at = sorted(attack_by_prog.get(key, [0]))
        tc = at[0] if at else 0
        med['attack'] = int(max(0.0, min(65.0,
                                        2.0 ** (tc / 65536.0 / 1200.0)))
                            * 1000.0)
        med['startrisk'] = med['startclick'] * (0.15 if med['attack'] >= 10
                                                else 1.0)
        dv = sorted(detune_by_prog.get(key, []))
        # THE WORST REGION, not the average: one detuned layer spoils
        # the chords that reach it while the rest of the program is fine.
        med['detune'] = dv[-1] if dv else 0.0
        # A FIXED SCALE, not min-max over the field. One bank with a
        # 486-cent outlier would stretch the range and make a 2 cent
        # source and a 25 cent one nearly equal; past 30 cents the
        # reading is useless anyway, so that is the floor. The reading
        # is the sample against its own note, NOT that plus the file's
        # fine-tune: a fine of 1179 is a rank stacked an octave apart,
        # and a bank whose recording is true is not out of tune because
        # a field in the file says otherwise.
        med['tunefit'] = max(0.0, 30.0 - med['detune']) / 30.0
        metrics[key] = med
    return feats, regions, metrics


def penalty(feats):
    return sum(PENALTY[k] * v for k, v in feats.items() if k in PENALTY)


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else 0


def _shed_over_budget(picked, scores, elig, by_name, progs, have, cap, amt):
    """Move programs out of any source holding more than its budget.

    ONE ALGORITHM, TWO BUDGETS. It is called once with a program COUNT
    and once with SAMPLE BYTES; the two used to be one loop written out
    twice, and the byte half did not exist, which is how a source came
    to hold 9% of the programs and 57% of the bytes.

    `have` maps a source to what it currently holds, `cap` to what it
    may hold, both in the same unit; `amt(prog, source)` is what one
    program occupies in a given source, so a move that also changes the
    amount - a program costing 6 MB in one bank and 0.6 MB in another -
    is accounted for on both sides.

    A destination is only offered if it stays inside its own budget
    AFTER the move. A move is only made when it costs at most
    MAX_MOVE_LOSS, and among the affordable ones the cheapest is taken.
    When nothing is affordable the source keeps what it has: the budget
    is a preference and the measurement is not overridden by force."""
    for name in [n for n in list(have) if have[n] > cap.get(n, 1 << 30)]:
        while have[name] > cap.get(name, 1 << 30):
            options = []
            for prog in [q for q in progs
                         if os.path.basename(picked[q]['path']).lower()
                         == name]:
                alt = None
                for n2, s2 in elig.get(prog, []):
                    if n2 == name or n2 not in by_name:
                        continue
                    if have[n2] + amt(prog, n2) > cap.get(n2, 1 << 30):
                        continue
                    if alt is None or s2 > alt[0]:
                        alt = (s2, n2)
                if alt and scores[prog] - alt[0] <= MAX_MOVE_LOSS:
                    options.append((scores[prog] - alt[0], prog, alt[1],
                                    alt[0]))
            if not options:
                break
            _, prog, n2, s2 = min(options)
            old = os.path.basename(picked[prog]['path']).lower()
            a_from, a_to = amt(prog, old), amt(prog, n2)
            picked[prog] = by_name[n2]
            scores[prog] = s2
            have[old] -= a_from
            have[n2] += a_to


def choose(sources):
    """sources: list of dicts with name, feats, regions, rates.

    THE COMMAND LINE IS THE QUALITY ORDER. A smaller or older bank can
    sound better than a larger one, and no measurable figure here knows
    that; the caller's order does. Penalty comes first because an
    instrument that needs a feature this synth lacks is not a matter of
    taste, then the order decides."""
    picked = {}
    scores = {}
    elig = {}
    top_src = {}
    gaps = {}
    thresholds = veto_thresholds(sources)
    VETO_STATS['removed'] = VETO_STATS['fallback'] = VETO_STATS['total'] = 0
    GATE_STATS['removed'] = GATE_STATS['fallback'] = GATE_STATS['total'] = 0
    GATE_STATS['tone'] = GATE_STATS['level'] = 0
    aliases = content_aliases(sources) if DEDUP_CONTENT else None
    for prog in range(129):
        cands = []
        for s in sources:
            if prog not in s['regions']:
                continue
            m = s['metrics'].get(prog)
            if m is None:
                continue
            cands.append((s, m))
        if not cands:
            continue
        # A DRUM KIT MUST COVER THE GM PERCUSSION RANGE. This is not a
        # weight: a kit without key 28 does not play key 28, and no
        # average of other virtues fixes a silent note. Candidates with
        # the full 27..87 are preferred; if none has it, the rest still
        # compete.
        if prog == 128:
            full = [c for c in cands if c[1].get('keycover', 0.0) >= 0.999]
            if full:
                cands = full
        else:
            # A MELODIC PROGRAM MUST COVER THE KEYBOARD. A zone table that
            # stops at key 108 or starts at 31 leaves notes silent - the
            # reader does not fall back to another region - and no average
            # of other virtues fixes that. Candidates covering 90% or more
            # are preferred; if none has it, the rest still compete.
            full = [c for c in cands if c[1].get('keycover', 0.0) >= 0.9]
            if full:
                cands = full
            # A SEAM IS SCORED (seamrate, rseam), not filtered: the
            # rejected-candidate rule made a slightly buzzy source
            # unavailable even where every alternative was duller.
            # TUNING IS A WEIGHT (tunefit, rtune), not a gate. gm.dls
            # is the pitch reference; excluding it or any other source
            # for a few cents is the over-correction this replaced.
        if thresholds:
            VETO_STATS['total'] += len(cands)
            # GM IS NEVER VETOED. It is the level reference and the only
            # instrument whose key profile is the target itself; when a
            # program's other candidates all sit 20 dB off gm's profile,
            # vetoing gm is what pushed the bank onto the one source it
            # had just measured as unusable.
            keep = [c for c in cands
                    if os.path.basename(c[0]['path']).lower() == 'gm.dls'
                    or not vetoed(prog, c[1], thresholds)]
            if keep:
                VETO_STATS['removed'] += len(cands) - len(keep)
                cands = keep
            else:
                VETO_STATS['fallback'] += 1
        if aliases:
            # SAME RECORDINGS, ONE CANDIDATE. Keep the best-scoring
            # member of each alias group; otherwise gm.dls and its
            # repackaged twin would both compete for the same slot.
            # THE FIRST SOURCE ON THE COMMAND LINE REPRESENTS ITS
            # GROUP. Choosing by score made a penalised gm hand every
            # slot to its identical twin; the caller's order is the
            # provenance preference, so FreeFont listed after gm never
            # competes.
            order = {t['path']: i for i, t in enumerate(sources)}
            best = {}
            for k, (s, m) in enumerate(cands):
                g = aliases.get(s['path'], s['path'])
                rank = (-order.get(s['path'], 0), s['path'])
                if g not in best or rank > best[g][0]:
                    best[g] = (rank, k)
            keep = sorted(v[1] for v in best.values())
            cands = [cands[k] for k in keep]
        banned = program_ban(prog)
        if banned:
            keep = [c for c in cands
                    if os.path.basename(c[0]['path']).lower() not in banned]
            if keep:
                cands = keep
        # ABSOLUTE DEFECT GATES. These are values no instrument should
        # have, and the pool says where the cliff is: clip is zero in
        # 71% of programs and p99 is 3%, rumble p99 0.2% against a
        # 47% maximum, startrisk's p99 is 0.8. A candidate past one of
        # these is not "a bit worse on one axis", and averaging it with
        # the rest was how a clipped or thumping sample stayed in the
        # race. Same fallback as keycover: if every candidate is gated,
        # none is.
        keep = []
        dropped = 0
        # A DRUM HIT STARTS WITH A JUMP AND HAS NO SUSTAIN TO LOOP.
        # startrisk, loopstep, looppow, loopfreq and flutter describe a
        # held note; applied to one-shots they removed 8 of 11 kits
        # (FluidR3 on looppow 0.996, gm on startrisk 0.707) and handed
        # p128 to whichever kit happened to survive. Percussive
        # programs keep only the gates that hold for any sample.
        perc = prog == 128 or 112 <= prog <= 119
        gates = (('clip', 0.02), ('rumble', 0.05), ('dc', 0.10))
        if not perc:
            gates += (('startrisk', 0.6), ('loopstep', 6.0),
                      ('looppow', 0.9), ('loopfreq', 2.0),
                      ('flutter', 1.2))
        for s, m in cands:
            bad = False
            for k, thr in gates:
                v = m.get(k)
                if v is not None and v > thr:
                    bad = True
                    break
            if bad:
                dropped += 1
            else:
                keep.append((s, m))
        GATE_STATS['total'] += len(cands)
        if keep:
            GATE_STATS['removed'] += dropped
            cands = keep
        else:
            GATE_STATS['fallback'] += 1
        # LEVEL: THE ONLY REAL RISK IS UNREACHABLE. The mixer boosts
        # at most +12 dB, so a program more than 12 dB under gm at the
        # notes the pieces play cannot be brought to the reference at
        # any gain; that is a silent instrument, not a matter of
        # degree. Six decibels is a preference, and preferences are
        # scored, not gated.
        if any(m.get('mlevel') is not None for _s, m in cands):
            near = [c for c in cands
                    if c[1].get('mlevel') is not None and c[1]['mlevel'] >= -6.0]
            if near:
                keep = [c for c in cands
                        if c[1].get('mlevel') is None or c[1]['mlevel'] >= -12.0]
                if keep and len(keep) < len(cands):
                    GATE_STATS['level'] += len(cands) - len(keep)
                    cands = keep
        # BRIGHTNESS AGAINST FF8, where the corpus map has the program:
        # the standard is "like FF8, not especially like gm, not
        # especially unlike it", so a candidate more than 1.5x away
        # from FF8's brightness is out when one within 1.27x exists.
        # The same fallback: a field that is entirely far keeps its
        # candidates and the score decides.
        # BRIGHTNESS IS NOT A GATE. Being brighter than the reference
        # is a matter of degree, and banning it put GeneralUser's
        # overdriven guitar in place of Creative's on p29 - 4.8 dB
        # quieter and 700 Hz darker on the piece. The criterion below
        # pays the middle of the field and nothing at the extremes;
        # only a real risk (distortion, a thump, an unreachable level)
        # removes a candidate.
        # GM IS THE SCALE, AND THE ENGINE HAS A CEILING. Calibration
        # targets gm.dls's level per program, but the mixer boosts at
        # most +12 dB (see db_boost_whole): a candidate whose estimated
        # program level is more than 12 dB under gm's cannot be brought
        # to it at any gain, and sounds quiet however well it measures.
        # Candidates within the ceiling all score the same, so this
        # criterion only removes the unusable tail.
        #
        # THE TEST IS PER KEY. A program-level average hid p4's low end:
        # GeneralUser's electric piano averaged fine and played key 24
        # 49 dB under gm, which is a silent note in an arrangement. A
        # key the candidate has no region for at all is silent too, and
        # counts.
        gref = None
        ggrid = None
        for s in sources:
            if os.path.basename(s['path']).lower() == 'gm.dls':
                gref = s['metrics'].get(prog, {})
                ggrid = (s.get('grid') or {}).get(prog) or {}
                break
        if gref is not None:
            rrow = gref.get('keylevels') or {}
            lo, hi = (27, 87) if prog == 128 else (21, 108)
            for s, m in cands:
                meas = (s.get('grid') or {}).get(prog) or {}
                if ggrid and meas:
                    devs = []
                    for v, gv in ggrid.items():
                        mv = meas.get(v)
                        if not mv:
                            continue
                        for k, r in gv.items():
                            c = mv.get(k)
                            if r > 0 and c and c > 0:
                                devs.append(20.0 * math.log10(c / r))
                elif prog == 128:
                    # A KIT IS NOT A KEYBOARD PROFILE. Kick, snare and
                    # hi-hat have different natural levels, so a per-key
                    # difference against gm is not a defect and its span
                    # rejected every non-gm kit. What matters is the
                    # kit's overall level, so the median against gm's
                    # median is the reading, and the per-key gates stand
                    # aside for this program.
                    crow = m.get('keylevels') or {}
                    gv = [rrow[k] for k in rrow]
                    cv = [crow[k] for k in crow]
                    if gv and cv:
                        dl = (sorted(gv)[len(gv) // 2] -
                              sorted(cv)[len(cv) // 2])
                        m['levelfit'] = -abs(dl)
                    m['keyfit'] = 0.0
                    m['keymin'] = 0.0
                    m['keymax'] = 0.0
                    m['gmfit'] = 0.0
                    devs = []
                    if m.get('levelfit') is not None:
                        devs = [m['levelfit']]
                else:
                    devs = []
                    crow = m.get('keylevels') or {}
                    for k in range(lo, hi + 1):
                        r = rrow.get(k)
                        c = crow.get(k)
                        if r is not None and c is not None:
                            # CANDIDATE MINUS GM, the same direction the
                            # grid branch above uses. It was gm minus
                            # candidate here, which inverted keymin,
                            # keymax and gmfit for exactly the
                            # candidates without a grid: the quietest
                            # key read as the loudest, and the
                            # reachability rule passed a source the
                            # engine cannot boost.
                            devs.append(c - r)
                    if (not devs and gref.get('level') is not None
                            and m.get('level') is not None):
                        devs = [m['level'] - gref['level']]
                if len(devs) >= 4:
                    devs.sort()
                    # THREE CONTINUOUS READINGS, NO THRESHOLDS. The
                    # spread, the quietest key and the loudest key enter
                    # the weighted score like every other measurement;
                    # a source is not switched off for crossing a line,
                    # it simply scores worse in proportion.
                    m['keyfit'] = (devs[int(len(devs) * 0.9)] -
                                   devs[int(len(devs) * 0.1)])
                    m['keymin'] = devs[0]
                    m['keymax'] = devs[-1]
                    # SOFT REACH: how far the lowest key lies beyond the
                    # mixer's +12 dB. Zero when it can be reached.
                    m['gmfit'] = min(0.0, devs[0] + 12.0)
        # GM IS THE REFERENCE, SO ITS OWN PROFILE IS PERFECT BY
        # DEFINITION. Leaving that zero in would hand gm first place on
        # `keyfit` for every program, which is not a measurement; it
        # takes the median of the field instead - no information.
        others = sorted(m['keyfit'] for s, m in cands
                        if 'keyfit' in m
                        and os.path.basename(s['path']).lower() != 'gm.dls')
        if others:
            mid = others[len(others) // 2]
            for s, m in cands:
                if (os.path.basename(s['path']).lower() == 'gm.dls'
                        and 'keyfit' in m):
                    m['keyfit'] = mid
        # THE SAME FOR `gmfit`. Reaching its own level is not a
        # measurement of gm, and leaving the free pass in made the
        # reference the automatic winner wherever another candidate had
        # one unreachable key. It takes the median of the field too.
        gf = sorted(m['gmfit'] for s, m in cands
                    if 'gmfit' in m
                    and os.path.basename(s['path']).lower() != 'gm.dls')
        if gf:
            gmid = gf[len(gf) // 2]
            for s, m in cands:
                if (os.path.basename(s['path']).lower() == 'gm.dls'
                        and 'gmfit' in m):
                    m['gmfit'] = gmid
        # THE QUIET SIDE IS THE ONLY SIDE THAT DISQUALIFIES. The engine
        # boosts at most +12 dB, so a source below 25% of gm cannot be
        # brought into range; the loud side can always be attenuated, and
        # the final pass enforces 200% after calibration. Gating the loud
        # side at selection excluded FluidR3 from 63 programs whose loud
        # keys were perfectly correctable - it is mastered about 3 dB hot,
        # not unusable. gm is the reference and is always allowed.
        # A PROFILE WIDER THAN THE BOUND ITSELF CANNOT BE MET BY ANY
        # GAIN: p113 spanned 19.7 dB and one number cannot put both ends
        # inside an 18.1 dB window. Such a candidate is out for that
        # program; gm, whose profile is the reference, is always in.
        span = LOUD_MAX_DB - LOUD_MIN_DB
        fit = [c for c in cands
               if os.path.basename(c[0]['path']).lower() == 'gm.dls'
               or 'keymin' not in c[1]
               or (c[1]['keymin'] >= LOUD_MIN_DB and
                   c[1]['keymax'] - c[1]['keymin'] <= span)]
        if fit:
            cands = fit
        bal, per = balanced_score(cands, prog)
        _dbg = os.environ.get('KSB_DEBUG_PROGS')
        if _dbg and prog in set(int(x) for x in _dbg.split(',')):
            print('DEBUG p%d %s' % (prog, ' '.join(
                '%s=%.4f%s%s' % (os.path.basename(t['path'])[:-4], bal[k],
                                 '/sim%.2f' % m['sim'] if 'sim' in m else '',
                                 '/air%.2f' % m['airfit'] if 'airfit' in m else '')
                for k, (t, m) in enumerate(cands))))
            # WHICH CRITERIA MADE THE DECISION, winner against the
            # runner-up. The headline score is a weighted mean and says
            # nothing about how it was reached: "most criteria good" and
            # "one criterion decided it" can print the same number. The
            # gap is split per criterion and the ones that carry it are
            # listed, so a single instrument running away with a program
            # is visible rather than assumed away.
            # HOW MANY CRITERIA THE WINNER ACTUALLY WINS. A high mean
            # can come from three criteria being brilliant and thirty-six
            # being merely fine; "most metrics excellent" is a count, not
            # an average, so it is printed as one.
            order_all = sorted(range(len(cands)), key=lambda k: -bal[k])
            _w = order_all[0]
            _best = 0
            _good = 0
            _tot = 0
            for k in set().union(*[set(per[x]) for x in range(len(cands))]):
                _tot += 1
                _vals = [per[x].get(k, -1e9) for x in range(len(cands))]
                if per[_w].get(k, -1e9) >= max(_vals) - 1e-12:
                    _best += 1
                if per[_w].get(k, -1e9) >= 0.5:
                    _good += 1
            print('DEBUG p%d winner %s : best on %d/%d criteria, %d at or above 0.5'
                  % (prog, os.path.basename(cands[_w][0]['path'])[:-4],
                     _best, _tot, _good))
            order = sorted(range(len(cands)), key=lambda k: -bal[k])
            if len(order) > 1:
                w, l = order[0], order[1]
                ww = {k: wv for k, _s, wv in
                      [(c[0], c[1], c[2]) for c in criteria_for(prog)]}
                gaprows = []
                for k in set(per[w]) | set(per[l]):
                    uw = per[w].get(k, 0.0)
                    ul = per[l].get(k, 0.0)
                    gaprows.append((ww.get(k, 0.0) * (uw - ul), k, uw, ul))
                gaprows.sort(reverse=True)
                tot = sum(abs(g) for g, _, _, _ in gaprows) or 1.0
                print('DEBUG p%d gap %s vs %s = %+.4f ; %s' % (
                    prog, os.path.basename(cands[w][0]['path'])[:-4],
                    os.path.basename(cands[l][0]['path'])[:-4],
                    bal[w] - bal[l],
                    ' '.join('%s %+.3f (%.0f%%)' % (k, g, 100.0 * abs(g) / tot)
                             for g, k, _, _ in gaprows[:6])))
        for k in range(len(cands)):
            name = os.path.basename(cands[k][0]['path']).lower()
            bal[k] *= SOURCE_PENALTY.get(name, 1.0)
        pens = [penalty(s['feats'].get(prog, {})) for s, m in cands]
        low = min(pens)
        group = [k for k in range(len(cands)) if pens[k] == low]
        top = max(bal[k] for k in group)
        # Within the margin the preference decides; outside it the
        # measurement does. This is the tie-break, not a reweighting.
        near = [k for k in group if bal[k] >= top - PREFER_MARGIN]
        # THE PIANO HAS NO PROVENANCE PREFERENCE. Its source is already
        # constrained by name (gm is barred there), so the preference
        # that places FF8 within a margin is the one thing that turns
        # the choice back into an identity decision; outside that band
        # the measurement decides.
        pref = ((lambda n: 1.0) if prog in POLICY_PROGRAMS else prefer)
        best_k = max(near, key=lambda k: (pref(os.path.basename(cands[k][0]['path'])),
                                          bal[k], cands[k][1]['bw']))
        top_k = max(group, key=lambda k: bal[k])
        picked[prog] = cands[best_k][0]
        scores[prog] = bal[best_k]
        elig[prog] = [(os.path.basename(t['path']).lower(), bal[k])
                      for k, (t, m) in enumerate(cands)]
        top_src[prog] = cands[top_k][0]
        gaps[prog] = bal[top_k] - bal[best_k]
    flips = [p for p in picked
             if p != 128
             and picked[p]['path'] != top_src[p]['path']
             and prefer(os.path.basename(picked[p]['path'])) > 1.0]
    cap = int(PREFER_CAP * len(picked))
    if len(flips) > cap:
        for p in sorted(flips, key=lambda q: -gaps[q])[:len(flips) - cap]:
            picked[p] = top_src[p]
            scores[p] = scores[p] + gaps[p]
    # EXPOSURE TARGETS. Move the cheapest programs of an over-full bank
    # to the best alternative that is itself under its target. The score
    # loss of each move is known from elig, so the moves made are the
    # least damaging ones.
    by_name = {os.path.basename(t['path']).lower(): t for t in sources}
    # THE DRUM KIT IS CHOSEN ON ITS OWN. Program 128 is one kit for the
    # whole channel: it is not interchangeable with a melodic program,
    # and counting it in a bank's exposure would move it for a quota
    # reason that has nothing to do with how the kit sounds.
    # A PROGRAM WITH A NAMED BAN IS ALSO A POLICY CHOICE. The grand
    # piano is the one program whose source is constrained by name;
    # moving it for a quota undoes that, and the move made here was at
    # the edge of the loss budget (0.044 of 0.05) from a sampled piano
    # to one stretched nine octaves from a single sample.
    quota = [p for p in picked if p != 128 and p not in POLICY_PROGRAMS]
    if EXPOSURE_SLOPE > 0.0:
        sums, nums = {}, {}
        for p in quota:
            for n2, s2 in elig.get(p, []):
                sums[n2] = sums.get(n2, 0.0) + s2
                nums[n2] = nums.get(n2, 0) + 1
        avg = {n2: sums[n2] / nums[n2] for n2 in sums if nums[n2]}
        lo = min(avg.values()) if avg else 0.0
        hi = max(avg.values()) if avg else 0.0
        w = {}
        for n2 in by_name:
            z = ((avg[n2] - lo) / (hi - lo)) if (n2 in avg and hi > lo) else 0.5
            w[n2] = 1.0 + EXPOSURE_SLOPE * EXPOSURE_SPREAD * (2.0 * z - 1.0)
            if prefer(n2) > 1.0:
                w[n2] *= prefer(n2)
        tw = sum(w.values()) or 1.0
        maxn = {n2: max(1, int(round(len(quota) * w[n2] / tw)))
                for n2 in by_name}
        delta = len(quota) - sum(maxn.values())
        order = sorted(maxn, key=lambda q: -maxn[q])
        i = 0
        while delta != 0 and order:
            n2 = order[i % len(order)]
            if delta > 0:
                maxn[n2] += 1; delta -= 1
            elif maxn[n2] > 1:
                maxn[n2] -= 1; delta += 1
            i += 1
    else:
        w = {n: 1.0 for n in by_name}
        tw = float(len(by_name)) or 1.0
        maxn = {n: max(1, int(MAX_EXPOSURE_FRAC * len(quota)))
                for n in by_name}
    maxn.update({n.lower(): v for n, v in MAX_EXPOSURE.items()})
    count = collections.Counter(
        os.path.basename(picked[p]['path']).lower() for p in quota)
    _shed_over_budget(picked, scores, elig, by_name, quota, count, maxn,
                      lambda _p, _n: 1)
    if EXPOSURE_BYTE_SHARE > 0.0:
        # THE SAME RULE ON BYTES, because a count quota is blind to
        # size. Twelve FluidR3 programs are 9% of the bank's programs
        # and 57% of its sample bytes - 23.8 MB against Airfont's
        # eighteen programs at 710 KB - and the count pass not only
        # failed to see it, it was PUSHING programs in: FluidR3's mean
        # score is the highest in the field, so its count target was
        # not full while its byte share was already six times an equal
        # share. Measured on the shipped field, p1 went to FluidR3 at
        # 0.5245 when Chorium scored 0.5964 for it, p4 at 0.4537 against
        # Creative's 0.5146, p18 at 0.5442 against 0.5951 - so every
        # move this pass makes on those is a gain in score AND a saving
        # in bytes, not a trade.
        #
        # THE TARGET IS THE SAME WEIGHTS, expressed in bytes: a source
        # that deserves a tenth of the programs deserves a tenth of the
        # space. EXPOSURE_BYTE_SHARE scales it - 1.0 is strict
        # proportion, 0 disables the pass - because proportion alone
        # would strip every large multi-sampled instrument down to the
        # thin sources, and how much room the big ones may keep is a
        # preference, not a measurement.
        prog_bytes = {}
        for s in sources:
            nm = os.path.basename(s['path']).lower()
            for p, m in s['metrics'].items():
                if m.get('bytes'):
                    prog_bytes[(p, nm)] = m['bytes']
        # THE KIT IS IN THIS PASS, although the count pass excludes it.
        # A kit is not a PROGRAM SLOT - one channel, one sound set - so
        # counting it among the quotas would move it for a reason that
        # has nothing to do with how it sounds. But it is 5.2 MB of
        # SAMPLES, the single largest item in the bank, and a space
        # budget that exempts the biggest item is not a space budget:
        # excluding it here is what handed the kit to a source and left
        # that source at a quarter of the bank.
        quota_bytes = [p for p in picked if p not in POLICY_PROGRAMS]
        total_b = sum(prog_bytes.get((p, os.path.basename(
            picked[p]['path']).lower()), 0) for p in quota_bytes)
        maxb = {n2: total_b * w[n2] / tw * EXPOSURE_BYTE_SHARE
                for n2 in by_name}
        have = collections.Counter()
        for p in quota_bytes:
            have[os.path.basename(picked[p]['path']).lower()] += \
                prog_bytes.get((p, os.path.basename(
                    picked[p]['path']).lower()), 0)
        _shed_over_budget(picked, scores, elig, by_name, quota_bytes, have,
                          maxb, lambda p, n: prog_bytes.get((p, n), 0))
    if MIN_EXPOSURE:
        by_name = {os.path.basename(t['path']).lower(): t for t in sources}
        count = collections.Counter(os.path.basename(s['path']).lower()
                                    for s in picked.values())
        for name, need in MIN_EXPOSURE.items():
            if name not in by_name:
                continue
            options = []
            for prog, lst in elig.items():
                cur = os.path.basename(picked[prog]['path']).lower()
                if cur == name:
                    continue
                d = dict(lst)
                if name not in d:
                    continue
                top = max(v for _, v in lst)
                options.append((top - d[name], prog, d[name], cur))
            for gap, prog, sc, cur in sorted(options):
                if count[name] >= need:
                    break
                if cur in MIN_EXPOSURE and count[cur] <= MIN_EXPOSURE[cur]:
                    continue
                picked[prog] = by_name[name]
                scores[prog] = sc
                count[cur] -= 1
                count[name] += 1
    return picked, scores


# THE CALIBRATION GRID IS THE BANK'S OWN ZONES. A fixed ladder every
# eight semitones cannot see a zone narrower than the ladder: FluidR3's
# piano is split at 75-78, no ladder key lands inside, and the region
# was corrected from its neighbours - keys 75-78 came out 15.6 dB under
# gm while the ladder keys either side read within 1 dB. The test keys
# are therefore the boundaries of the bank's narrow regions (with the
# midpoints and the ladder as a floor), capped so the render stays
# affordable: 32 keys against the old 11 is about two and a half times
# the calibration time and closes exactly this class of gap.
# A BOUND, NOT A SAMPLING. cal_keys now selects the keys that can be
# wrong - the ends of each region and the one farthest from its unity
# note - which is 1234 keys here against 16445 for every semitone and
# 12349 for the old stride. The cap stays at the full MIDI range so it
# can never bite and turn the proxy back into a subsample.
CAL_KEY_CAP = 128
# THE VELOCITY GRID IS A HITTING SET, NOT A SAMPLE. cal_vels picks the
# velocities that cover every layer in the bank: 11 to 22 points here,
# against the four that were drawn evenly from the union of all layers
# and covered almost none of them. 24 leaves room for a bank with more
# layers than this field's 22, and the cost is time - one render of the
# matrix per velocity, into the same tmpwav - not disk.
CAL_VEL_CAP = 24


def cal_keys(bankpath, cap=CAL_KEY_CAP):
    """The keys that can be wrong, not the whole keyboard.

    THE GRID USED TO BE EVERY SEMITONE, and before that the boundaries
    of narrow regions. Boundaries alone missed p3's 100-104 zone - one
    sample stretched over five keys, grid point 100 read fine and key
    104 measured 15.5 dB over gm - because the level drifts WITH the
    stretch and the boundary is not where it lands. Every semitone
    catches that and costs the full keyboard, 16445 keys across this
    field.

    THE DRIFT IS MONOTONIC IN PITCH DISTANCE from the recorded note, so
    the worst key of a region is the one farthest from its unity. Three
    points per region - the two ends and that worst key - cover the
    shape the failure actually has: 1234 keys here, 8% of the full
    sweep, and p3's 104 is one of the three.
    """
    b = open(bankpath, 'rb').read()
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', b, 4)
    io, ro = 16, 16 + n * 7
    out = {}
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from(
            '<BBBHH', b, io + i * 7)
        keys = set()
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi = struct.unpack_from('<HH', b, o)
            if khi < klo or klo > 127:
                continue
            khi = min(khi, 127)
            unity = struct.unpack_from('<H', b, o + 16)[0]
            keys.add(klo)
            keys.add(khi)
            keys.add(klo if abs(klo - unity) >= abs(khi - unity) else khi)
        ks = sorted(keys)
        if len(ks) > cap:
            step = len(ks) / float(cap)
            ks = [ks[int(i * step)] for i in range(cap)]
        out[prog] = ks or [60]
    return out


def cal_vels(bankpath, cap=CAL_VEL_CAP):
    """The velocities that cover every layer, chosen not sampled.

    WHAT IT USED TO DO was collect one midpoint per region across the
    whole bank and keep an evenly spaced four. That has no relation to
    the layers it is supposed to measure: GeneralUser carries 101
    distinct velocity zones and the four kept are drawn from the union
    of every program's, so a program's own layers are not guaranteed to
    be among them - a layer never played is a layer the level pass
    cannot see.

    A LAYER IS AN INTERVAL, so this is a hitting set: greedily take the
    velocity that falls inside the most layers not yet covered, until
    every layer is covered or `cap` runs out. On this field that is 11
    to 22 points for complete coverage of every layer in the bank,
    against the four that covered almost none - and the greedy order
    means an early cap drops the least-populated layers first.

    The cost is time, not space: levels_grid renders one velocity of
    the matrix per iteration into the same tmpwav, so the disk holds a
    single 1.2 GB WAV however many velocities are measured.
    """
    b = open(bankpath, 'rb').read()
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', b, 4)
    io, ro = 16, 16 + n * 7
    layers = []
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from(
            '<BBBHH', b, io + i * 7)
        for j in range(cnt):
            o = ro + (first + j) * 104
            vlo, vhi = struct.unpack_from('<HH', b, o + 4)
            if vhi >= vlo:
                layers.append((int(vlo), int(vhi)))
    out = []
    left = set(range(len(layers)))
    while left and len(out) < cap:
        hits = {}
        for i in left:
            for v in range(layers[i][0], layers[i][1] + 1):
                hits.setdefault(v, 0)
                hits[v] += 1
        if not hits:
            break
        best = max(hits, key=lambda v: (hits[v], -abs(v - 63)))
        out.append(best)
        left = {i for i in left if not (layers[i][0] <= best <= layers[i][1])}
    return sorted(out) or list(GRID_VELS)


def _cal_starts(keys):
    starts, t = {}, 0.0
    # 0..128, NOT 0..127. The drum kit is program 128 and the grid used
    # to stop one short of it: write_midi never emitted a note for the
    # kit, so the calibration grid had no row for it, and calib,
    # split_regions, enforce_bound and the balance pass all found
    # nothing to read. The kit's region gains were left exactly as the
    # source bank shipped them - measured on the built bank, 59 of 62
    # sat at 0.0 dB and three at -15, -20 and -20, untouched.
    for p in range(129):
        starts[p] = t
        t += len((keys or {}).get(p) or NOTE_KEYS) * note_dt(p) + 0.2
    return starts


def write_kit_midi(path, vel, notes):
    """The drum kit alone: channel 9, nothing before it.

    THE KIT CANNOT SHARE A MIDI WITH THE PROGRAMS BEFORE IT. It is last
    in the calibration grid and Applause, Gunshot and Seashore are
    three programs earlier; those are sustained sounds whose tails span
    the whole kit section. Measured on the shipped kit, the window
    before each drum carried more energy than the drum itself - 62.4
    against 21.0 for key 36 - and the quiet drums read 7 to 20 dB hot
    because someone else's applause was in the sample. With the kit in
    its own file the preceding window reads exactly 0. Isolated is what
    tune_pick has always done, one note per file, which is why its
    sim and balance readings never had this.
    """
    def vlq(v):
        o = [v & 0x7F]
        v >>= 7
        while v:
            o.append(0x80 | (v & 0x7F))
            v >>= 7
        return bytes(reversed(o))
    evs = []
    for k, note in enumerate(notes):
        on = k * int(round(note_dt(128) * 192))
        evs.append((on, bytes([0x99, note, vel])))
        evs.append((on + int(round(max(0.8 * note_dt(128), 0.6) * 192)), bytes([0x89, note, 0])))
    evs.sort(key=lambda x: x[0])
    out = []
    last = 0
    for t, b in evs:
        out.append(vlq(t - last) + b)
        last = t
    body = b''.join(out) + vlq(0) + bytes([0xFF, 0x2F, 0])
    tempo = bytes([0, 0xFF, 0x51, 3, 7, 0xA1, 0x20, 0, 0xFF, 0x2F, 0])
    open(path, 'wb').write(b'MThd' + struct.pack('>IHHH', 6, 1, 2, 96) +
                           b'MTrk' + struct.pack('>I', len(tempo)) + tempo +
                           b'MTrk' + struct.pack('>I', len(body)) + body)


def write_midi(path, vel=VEL, keys=None):
    def vlq(v):
        o = [v & 0x7F]
        v >>= 7
        while v:
            o.append(0x80 | (v & 0x7F))
            v >>= 7
        return bytes(reversed(o))

    keys = keys or {}
    starts = _cal_starts(keys)
    evs = []
    for p in range(128):
        t = int(round(starts[p] * 192))
        evs.append((t, bytes([0xC0, p])))
        for k, note in enumerate(keys.get(p) or NOTE_KEYS):
            on = t + k * int(round(note_dt(p) * 192))
            evs.append((on, bytes([0x90, note, vel])))
            evs.append((on + int(round(max(0.8 * note_dt(p), 0.6) * 192)), bytes([0x80, note, 0])))
    evs.sort(key=lambda x: x[0])
    out = []
    last = 0
    for t, b in evs:
        out.append(vlq(t - last) + b)
        last = t
    tempo = b'' + vlq(0) + bytes([0xFF, 0x51, 3, 7, 0xA1, 0x20]) + \
        vlq(0) + bytes([0xFF, 0x2F, 0])
    trk = b''.join(out) + vlq(0) + bytes([0xFF, 0x2F, 0])
    open(path, 'wb').write(
        b'MThd' + struct.pack('>IHHH', 6, 1, 2, 96) +
        b'MTrk' + struct.pack('>I', len(tempo)) + tempo +
        b'MTrk' + struct.pack('>I', len(trk)) + trk)


def _note_rms(w, a0, a_end, fl):
    """RMS over one note's ACTIVE REGION, read straight out of the file.

    THE RENDER IS NOT LOADED WHOLE. A calibration grid is 7400 s long:
    readframes, the strided copy onto the left channel and the float64
    cast peaked near 3 GiB, and the second of the two passes that do it
    raised MemoryError. Every caller passes the note's own slot and only
    that is read.

    THE MONO MIX, NOT THE LEFT CHANNEL. This used to take channel 0
    alone, which is a different quantity from what anyone hears, and
    the kit showed it: gm's Bell Tree is panned hard right - 5.4 left
    against 49.7 right - so the left channel agreed with the other
    bank's to within 0.5 dB while the mix the listener gets was 10 dB
    apart. Calibration was then correcting a pan, not a level. Every
    other measurement in this project - corpus_report, tone_scan and
    the hand checks - already mixes down; this was the only one that
    did not.

    THE WINDOW IS FIXED, AND DELIBERATELY SO. 0.02-0.55 s is the same
    span for every bank of every program, and that is what makes the
    ratio a ratio. It was briefly replaced by each note's own active
    region and that was a mistake: a window derived from the bank being
    measured means a slow-decaying bank gets a longer window and reads
    higher, so the calibration pushed the quiet ones up and pulled the
    loud ones down. Measured on the shipped bank, piano, harp and crash
    read 2.6 to 3.2 dB low that way and were then boosted; the mix
    reached 83 near-clipped frames against gm's 22.

    A woodblock's 0.06 s is diluted by the silence after it, and both
    banks are diluted identically. That is fine. What is NOT fine is a
    window that moves with the thing it is measuring.

    Returns None where the old code `continue`d."""
    ch = w.getnchannels()
    if a_end <= a0:
        return None
    w.setpos(max(0, a0))
    x = np.frombuffer(w.readframes(a_end - a0), dtype='<i2')
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    nf = len(x) // fl
    if nf < 4:
        return None
    blk = x[:nf * fl].reshape(nf, fl).astype(np.float64)
    e = (blk * blk).mean(axis=1)
    if e.max() <= 0:
        return None
    # -25 dB under the loudest frame is the main duration.
    keep = e >= e.max() * 10 ** (-25 / 10.0)
    return float(np.sqrt(e[keep].mean()))


def _note_peak(w, a0, a1, fl):
    """The loudest instant of one note's window.

    THE SAME WINDOW AND THE SAME MIX AS _note_rms. The RMS is what the
    level pass centres; the peak is what the MIXOVERLOADS ON. A
    transient-heavy instrument can sit at gm's median level and still
    drive the sum past full scale - measured on the shipped bank, Tinkle
    Bell ran +10.8 dB over gm at the peak while the level pass read it
    in band, and the mix clipped on the quietest of the three pieces.
    """
    ch = w.getnchannels()
    if a1 <= a0:
        return None
    w.setpos(max(0, a0))
    x = np.frombuffer(w.readframes(a1 - a0), dtype='<i2')
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if len(x) < 8:
        return None
    return float(np.abs(x).max())


def levels(bank, player, mid, tmpwav, keys=None, peaks=False):
    """One loudness figure per program: the energy over the note's main
    duration, not its loudest instant.

    The window runs from note-on for 1.2 s or to the end of the program's
    segment. 20 ms frames below 25 dB under the loudest frame are left
    out, so leading silence and a tail long after the sound has gone do
    not dilute the figure; what remains is averaged as ENERGY and turned
    into an RMS. A peak reading made a pluck and a pad of equal loudness
    compare by their transients; this compares what they carry while they
    sound."""
    subprocess.run([player, '--bank', bank, '--multi', '--wav', tmpwav, mid],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)
    import wave
    w = wave.open(tmpwav, 'rb')
    n = w.getnframes()
    r = w.getframerate()
    fl = max(1, int(0.02 * r))
    starts = _cal_starts(keys)
    out = {}
    for p in range(128):
        vs = []
        ks = (keys or {}).get(p) or NOTE_KEYS
        end = starts[p] + len(ks) * note_dt(p) + 0.2
        for k in range(len(ks)):
            t0 = starts[p] + k * note_dt(p)
            a0 = int(t0 * r)
            a1 = min(n, int((t0 + 0.55) * r), int(end * r))
            v = (_note_peak if peaks else _note_rms)(w, a0, a1, fl)
            if v is not None:
                vs.append(v)
        if vs:
            out[p] = median(vs)
    w.close()
    # THE KIT IS MEASURED IN ITS OWN FILE, for the reason write_kit_midi
    # gives: the tail of Applause and Gunshot covers the whole kit
    # section of the shared grid.
    ks = (keys or {}).get(128) or NOTE_KEYS
    km = mid + '.kit'
    write_kit_midi(km, VEL, ks)
    subprocess.run([player, '--bank', bank, '--multi', '--wav', tmpwav, km],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)
    w = wave.open(tmpwav, 'rb')
    n = w.getnframes()
    r = w.getframerate()
    fl = max(1, int(0.02 * r))
    vs = []
    for k in range(len(ks)):
        t0 = k * note_dt(128)
        a0 = int(t0 * r)
        a1 = min(n, int((t0 + 0.55) * r))
        v = (_note_peak if peaks else _note_rms)(w, a0, a1, fl)
        if v is not None:
            vs.append(v)
    if vs:
        out[128] = median(vs)
    w.close()
    return out


def levels_grid(bank, player, mid, tmpwav, vels=None, keys=None, peaks=False):
    """The same measurement per NOTE and per VELOCITY rather than per
    program.

    One figure per program averages the keyboard, and an instrument can
    average out to gm's level while being 10 dB loud down low and 10 dB
    quiet up high: FluidR3's trumpet measured within the band as a whole
    and landed -9.5 dB at key 77 against gm in a real arrangement. The
    region gain is per region, so the correction has to be too. The
    velocity dimension matters for the same reason - velocity layers are
    regions, and the layer a soft note reaches is not the one velocity
    100 measured. -> {prog: {velocity: {midi note: rms}}}"""
    if vels is None:
        vels = cal_vels(bank)
    out = {}
    starts = _cal_starts(keys)
    import wave
    for vel in vels:
        m = '%s.v%d' % (mid, vel)
        write_midi(m, vel, keys)
        subprocess.run([player, '--bank', bank, '--multi', '--wav', tmpwav, m],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        w = wave.open(tmpwav, 'rb')
        n = w.getnframes()
        r = w.getframerate()
        fl = max(1, int(0.02 * r))
        for p in range(128):
            row = out.setdefault(p, {}).setdefault(vel, {})
            ks = (keys or {}).get(p) or NOTE_KEYS
            end = starts[p] + len(ks) * note_dt(p) + 0.2
            for k, note in enumerate(ks):
                t0 = starts[p] + k * note_dt(p)
                a0 = int(t0 * r)
                a1 = min(n, int((t0 + 0.55) * r), int(end * r))
                v = (_note_peak if peaks else _note_rms)(w, a0, a1, fl)
                if v is not None:
                    row[note] = v
        w.close()
        # THE KIT IS MEASURED IN ITS OWN FILE. See write_kit_midi: the
        # programs before it in the grid are Applause, Gunshot and
        # Seashore, and their tails cover the whole kit section. Its
        # notes now start at tick 0 with nothing ahead of them.
        ks = (keys or {}).get(128) or NOTE_KEYS
        km = '%s.k%d' % (mid, vel)
        write_kit_midi(km, vel, ks)
        subprocess.run([player, '--bank', bank, '--multi', '--wav', tmpwav, km],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        w = wave.open(tmpwav, 'rb')
        n = w.getnframes()
        r = w.getframerate()
        fl = max(1, int(0.02 * r))
        row = out.setdefault(128, {}).setdefault(vel, {})
        for k, note in enumerate(ks):
            t0 = k * note_dt(128)
            a0 = int(t0 * r)
            a1 = min(n, int((t0 + 0.55) * r))
            v = (_note_peak if peaks else _note_rms)(w, a0, a1, fl)
            if v is not None:
                row[note] = v
        w.close()
    return out


def gm_bank(srcs, kac, cap, tmp):
    """A full bank built from the gm.dls among the sources, for reference.
    Built here rather than taken from disk so the reference is the same
    file the selection scored, at the same rate cap."""
    for path in srcs:
        if os.path.basename(path).lower() == 'gm.dls':
            out = os.path.join(tmp, 'ref.ksb')
            subprocess.run([sys.executable,
                            os.path.join(os.path.dirname(__file__), 'ksb_build.py'),
                            '--codec', 'kac', '--kac', kac, '-c', str(cap),
                            '--rate', 'keep', path, out], check=True,
                           stdout=subprocess.DEVNULL)
            return out
    return None


# WHAT A SOURCE ACTUALLY SOUNDS LIKE, KEY BY KEY. The DLS metrics read
# the samples; they cannot see that the converted filter or envelope
# silences half the keyboard. GeneralUser's violin sits 28 dB under gm
# at key 104 because its region carries a 124 Hz cutoff and an envelope
# that never opens it, and every estimate said the sample was fine.
# Measured once per source into a bank+grid cache beside the processed
# DLS, since the cost is a full build per source.
MEASURE_VERSION = 8


def measure_source(path, player, kac, cap, rate_mode, tmp):
    """Build the source's full bank and measure one level per program per
    note. Cached by content hash and parameters, so a selection rerun
    after a criteria change does not rebuild the nine banks."""
    import hashlib
    import json
    raw = open(path, 'rb').read()
    sig = {'sha': hashlib.sha256(raw).hexdigest(), 'cap': cap,
           'rate': rate_mode, 'tool': MEASURE_VERSION}
    cache_path = os.path.join(os.path.dirname(path), 'measure.json')
    try:
        cache = json.load(open(cache_path))
    except (OSError, ValueError):
        cache = {}
    key = os.path.basename(path)
    rec = cache.get(key)
    if rec and rec.get('sig') == sig:
        return {int(p): {int(v): {int(k): x for k, x in row.items()}
                         for v, row in byvel.items()}
                for p, byvel in rec['grid'].items()}
    bank = os.path.join(tmp, 'measure.ksb')
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), 'ksb_build.py'),
                    '--codec', 'kac', '--kac', kac, '-c', str(cap),
                    '--rate', rate_mode, path, bank], check=True,
                   stdout=subprocess.DEVNULL)
    mid = os.path.join(tmp, 'measure.mid')
    write_midi(mid)
    grid = levels_grid(bank, player, mid, os.path.join(tmp, 'measure.wav'))
    cache[key] = {'sig': sig, 'grid': grid}
    json.dump(cache, open(cache_path, 'w'))
    return grid


def ref_levels(refpath, srcs, player, kac, cap, tmp, keys=None, vels=None, peaks=False):
    """One level per program for the reference, and a fallback for programs
    it does not carry."""
    bank = refpath or gm_bank(srcs, kac, cap, tmp)
    if not bank:
        raise SystemExit('--calibrate needs a gm.dls among the sources or --ref')
    mid = os.path.join(tmp, 'ref.mid')
    write_midi(mid, keys=keys)
    lv = levels(bank, player, mid, os.path.join(tmp, 'r.wav'), keys, peaks=peaks)
    if not lv:
        raise SystemExit('%s rendered no levels' % bank)
    grid = levels_grid(bank, player, mid, os.path.join(tmp, 'rg.wav'),
                       vels=vels, keys=keys, peaks=peaks)
    xs = sorted(lv.values())
    return (lv, xs[len(xs) // 2], grid)


def _calib_dev(dev, keep):
    """The correction for one deviation: keep the character inside the
    band, remove the error outside it."""
    kept = max(-KEEP_BAND_DB, min(KEEP_BAND_DB, dev))
    return (1.0 - keep) * kept + (dev - kept)


def _dev_at(brow, rrow, k):
    b = brow.get(k)
    r = rrow.get(k)
    if b and r and b > 0 and r > 0:
        return 20 * math.log10(b / r)
    return None


def calib(path, before, before_grid, ref, keep, done=None):
    """Move every region toward the reference level without erasing the
    difference between instruments.

    GM.DLS IS THE YARDSTICK: `dev` is the deviation from the reference
    in dB, positive when louder. Inside KEEP_BAND_DB the deviation is
    character and `keep` of it survives (0.3 by default); outside the
    band it is an error and the excess is corrected in full. A flat
    `keep` across the whole range was tried and left a program 29 dB
    under gm at 17 dB under it, because 30% of a large error is still a
    large error.

    THE CORRECTION IS PER REGION AND PER VELOCITY WINDOW, AND IT TAKES
    THE MEDIAN OF THE PAIRS.

    IT USED TO TAKE THE LOWER QUARTILE, and that is a bias: the target
    became the region's quietest 25%, so a region whose keys span a few
    dB was lifted until its quietest matched gm and its loudest
    overshot. Measured on the shipped bank, the per-key deviation sat at
    +2.4 dB on the piano, +2.4 on the bass and +2.5 on the pad while the
    energy matched gm to +0.1 - the level was right and the PEAKS were
    hot, which is what clipped. The median centres the region instead.

    The lower quartile was chosen before balance_regions existed, when
    "the median left whichever key was below it still below it, by 13 dB"
    had no other answer. There is one now: balance_regions flattens the
    spread within a program afterwards, so the quiet side is taken care
    of there and the level here only has to be centred.

    A region covering velocities 0-74 is the sound a soft note reaches
    and is calibrated from the soft measurement, not from velocity 100.

    REGIONS ARE NOT SUBDIVIDED. Splitting them at measured keys tracks
    gm's per-key profile more closely, but it writes a step into the
    gain wherever the profile is steep: the same instrument then plays
    louder on one side of the split than the other, which is heard as
    the volume moving with the melody. A source whose profile is that
    far from gm's is rejected by `keyfit` at selection instead, where
    the fix costs nothing audible.

    Both directions are written into the region gain; the engine's
    positive side is bounded at +12 dB, which is also the clamp here.
    That ceiling is why the selector scores a candidate's program level
    against gm's (`gmfit`): a source needing more than the mixer can
    give is not made loud by calibration."""
    lv, fallback, rgrid = ref
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', d, 4)
    io = 16
    ro = io + n * 7
    for idx in range(n):
        prog, msb, lsb, count, first = struct.unpack_from(
            '<BBBHH', d, io + idx * 7)
        brow = before_grid.get(prog)
        if brow is None:
            continue
        rrow = rgrid.get(prog, {})
        target = lv.get(prog, fallback)
        pdev = None
        if prog in before and target > 0:
            pdev = 20 * math.log10(before[prog] / target)
        for j in range(count):
            o = ro + (first + j) * 104
            klo, khi, vlo, vhi = struct.unpack_from('<HHHH', d, o)
            # ONE SCALING PER REGION. A region corrected on an earlier
            # pass is not corrected again: applying the same 30-60% fix
            # every pass walks the residual all the way to the band
            # edge, which is the flattening the fraction exists to
            # avoid. Only pieces a later split created are new.
            tag = (prog, klo, khi, vlo, vhi)
            if done is not None and tag in done:
                continue
            if done is not None:
                done.add(tag)
            pairs = []
            for v, bv in brow.items():
                if not (vlo <= v <= vhi):
                    continue
                rv = rrow.get(v, {})
                for k in range(klo, min(khi, 127) + 1):
                    dev = _dev_at(bv, rv, k)
                    if dev is not None:
                        pairs.append(dev)
            if not pairs:
                # A NARROW VELOCITY BAND CAN FALL BETWEEN MEASURED
                # POINTS (a layer of 80-87 contains none of the seven).
                # It takes the nearest measured velocity instead of the
                # average of all of them, which belongs to other layers.
                best = None
                for v, bv in brow.items():
                    dist = 0.0 if vlo <= v <= vhi else min(abs(v - vlo),
                                                          abs(v - vhi))
                    if best is None or dist < best[0]:
                        best = (dist, v, bv)
                if best is not None:
                    rv = rrow.get(best[1], {})
                    for k in range(klo, min(khi, 127) + 1):
                        dev = _dev_at(best[2], rv, k)
                        if dev is not None:
                            pairs.append(dev)
            dev = (sorted(pairs)[len(pairs) // 2] if pairs else pdev)
            if dev is None:
                continue
            # THE BOUND IS ENFORCED, NOT AVERAGED. The lower-quartile
            # target leaves the loud edge wherever the region's spread
            # puts it (p116 measured +12.4 dB against a +6.02 limit).
            # When a measured pair is outside 25%-200% of gm, the gain
            # is computed to put that exact pair on the bound; the
            # in-band path keeps the quartile behaviour.
            lo = min(pairs) if pairs else dev
            hi = max(pairs) if pairs else dev
            under = LOUD_MIN_DB - lo
            over = hi - LOUD_MAX_DB
            if under > 0.0:
                # ONE GAIN PER REGION, AND THE QUIET SIDE IS THE ONE
                # THAT CANNOT WAIT. Attenuating for the loud end spends
                # room the quiet end does not have - measured three ways:
                # quiet-wins left 439 pairs over +6.02 dB, splitting
                # the excess left 489 over and 47 under, and correcting
                # the larger error left 447 over and 78 under. A region
                # whose keys span the whole band needs its own gains
                # split, which is the trade the region table has to
                # take; the level pass takes the side that is missing.
                add = -int(round(under * 10 * _loud_fix(under)))
            elif over > 0.0:
                add = int(round(over * 10 * _loud_fix(over)))
            else:
                add = int(round(_calib_dev(dev, keep) * 10))
            if add == 0:
                continue
            g = struct.unpack_from('<i', d, o + 20)[0]
            struct.pack_into('<i', d, o + 20,
                             max(GAIN_MIN_TENTHS, min(GAIN_MAX_TENTHS, g - add)))
    open(path, 'wb').write(bytes(d))


def balance_regions(path, grid, rgrid):
    """Close a program's internal level spread without moving its level.

    WHAT calib LEAVES BEHIND. It moves each region toward gm's level and
    in principle each drum lands where gm's does; on the shipped kit it
    did not. 59 of 62 regions sat at 0.0 dB while the three hi-hats sat
    at -15, -20 and -20, and the corpus read the open hi-hat 18 dB under
    gm after all of it. Whatever the cause - the grid's velocities and
    the note's own are not the same layers - the result is the same: a
    27 dB spread against a loudness bound that is an 18 dB window, so no
    shift of the whole program can fit it.

    ONE GAIN PER REGION, SET AGAINST THE PROGRAM'S OWN MEAN. For each
    region take the mean deviation of the keys it covers and move the
    gain by that mean minus the program's mean deviation. The spread
    closes and the program's level does not move, because the two cancel
    in the average - and that is why the DIFFERENCE is applied and not
    the deviation itself: the program's level is the level pass's job
    and this must not undo it.

    A REGION WITH NO MEASUREMENT IS LEFT ALONE. Silence here is not
    evidence that the region is right, but moving it on no evidence is
    worse than leaving it: the gain would be whatever the program mean
    happens to be.
    """
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', d, 4)
    io = 16
    ro = io + n * 7
    moved = 0
    for idx in range(n):
        prog, msb, lsb, count, first = struct.unpack_from(
            '<BBBHH', d, io + idx * 7)
        meas = grid.get(prog)
        ref = rgrid.get(prog)
        if not meas or not ref:
            continue
        # ONE DEVIATION PER KEY, averaged over the measured velocities.
        devs = {}
        for v, bv in meas.items():
            rv = ref.get(v)
            if not rv:
                continue
            for k, b in bv.items():
                dev = _dev_at(bv, rv, k)
                if dev is not None:
                    devs.setdefault(k, []).append(dev)
        devs = {k: sum(v) / len(v) for k, v in devs.items() if v}
        if len(devs) < 2:
            continue
        pmean = sum(devs.values()) / len(devs)
        for j in range(count):
            o = ro + (first + j) * 104
            klo, khi, vlo, vhi = struct.unpack_from('<HHHH', d, o)
            # THE PAIRS COME FROM THIS REGION'S OWN VELOCITY WINDOW.
            # Averaging one key over every velocity mixes the layers
            # that are the thing being balanced: p27's key 45 read
            # -14 dB at velocity 30 and +0.2 at 94, the average was
            # -7, and both the soft region and the hard one were moved
            # by the same step - so the 14 dB between them survived.
            # A region is a (key range, velocity range) pair, and only
            # the notes inside that pair describe it.
            inr = []
            for v, bv in meas.items():
                if not (vlo <= v <= vhi):
                    continue
                rv = ref.get(v)
                if not rv:
                    continue
                for k in range(klo, min(khi, 127) + 1):
                    dkv = _dev_at(bv, rv, k)
                    if dkv is not None:
                        inr.append((k, dkv))
            if not inr:
                continue
            # THE READINGS HAVE TO AGREE. The same key heard at several
            # loudnesses must describe one region; when they do not,
            # something outside the model is moving - an overlapping
            # zone, a layer the sweep cannot separate - and a step taken
            # from them is a guess with a gain attached. This is the
            # guard that stands in for "two keys", now that a one-key
            # region is allowed to move.
            byk = {}
            for k, dv in inr:
                byk.setdefault(k, []).append(dv)
            if any(max(v) - min(v) > TRUST_SPREAD_DB
                   for v in byk.values()):
                continue
            # ONE READING IS WEAK EVIDENCE, AND LEAVING A REGION 9 dB
            # OFF IS NOT A BETTER ANSWER. Two keys used to be required,
            # which was right while the velocity bands moved on their
            # own; now that a zone is held to one gain, a single-key
            # region has nothing else to stand on and p21's key 64 was
            # left 9.3 dB under gm because it could never raise two
            # keys. The step cap below is what keeps a noisy reading
            # from running away.
            step = (sum(d for _, d in inr) / len(inr)) - pmean
            # AND THE STEP IS BOUNDED. The measurement carries its own
            # error; a full correction of a reading that is off by that
            # error walks the wrong way by the same amount, and the next
            # pass corrects that. Six decibels a pass converges without
            # letting one bad reading run away.
            step = max(-STEP_CAP_DB, min(STEP_CAP_DB, step))
            if abs(step) < 0.25:
                continue
            g = struct.unpack_from('<i', d, o + 20)[0]
            ng = max(GAIN_MIN_TENTHS, min(GAIN_MAX_TENTHS, g - int(round(step * 10))))
            if ng != g:
                struct.pack_into('<i', d, o + 20, ng)
                moved += 1
    open(path, 'wb').write(d)
    return moved


def unify_velocity_gains(path, grid, rgrid):
    """One gain per key zone, whatever velocity layers the source has.

    THE REFERENCE DOES NOT SPLIT VELOCITY HERE. gm.dls gives p27, p64
    and p66 a single 0-127 region per key zone; the shipped bank had
    12, 7 and 5 bands for the same programs, inherited from the source
    DLS, each with its own gain - twelve free parameters where the
    reference has one, and two notes of evidence behind each. On p27
    the bands for keys 45-46 stood at -18.5 dB and -4.1 dB, and the
    difference between them (14.4 dB) was exactly the difference the
    measurement found between the soft and hard layers: the layer error
    was the calibration's own, not the samples'.

    ONE GAIN FOR THE WHOLE ZONE. Every band of a key zone is SET to the
    same gain: each band's own gain is first carried to the level its
    measurements ask for, and the zone takes the median of those. The
    bands then carry whatever level differences their samples have and
    none of the ones the calibration wrote. Evidence per zone is every
    note played in it instead of a handful per band.

    OVERLAPPING BANDS ARE LEFT ALONE. Two regions sounding at once are
    a stacked instrument, and forcing one gain on them would change
    their mix rather than level it.

    Returns how many regions moved.
    """
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, doff = struct.unpack_from('<HHHHI', d, 4)
    io, ro = 16, 16 + n * 7
    moved = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from(
            '<BBBHH', d, io + i * 7)
        byzone = {}
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi, vlo, vhi = struct.unpack_from('<HHHH', d, o)
            byzone.setdefault((klo, min(khi, 127)), []).append(
                (j, vlo, vhi))
        meas = grid.get(prog) or {}
        ref = rgrid.get(prog) or {}
        for (klo, khi), regs in byzone.items():
            if len(regs) < 2:
                continue
            bands = sorted(set((vlo, vhi) for _, vlo, vhi in regs))
            if len(bands) < 2:
                continue
            if not all(bands[t][1] < bands[t + 1][0]
                       for t in range(len(bands) - 1)):
                continue
            # ONE GAIN, NOT ONE DELTA. A shared delta leaves the bands
            # as far apart as they were - p27's soft band sat 10.9 dB
            # under its neighbours and twelve identical shifts moved
            # all of them together. Each band is measured against gm,
            # its own gain is carried to the level that measurement asks
            # for, and the zone takes the median of those; every band is
            # then SET to that gain. The median of the deltas is not
            # used, because it is zero whenever most bands are right -
            # which is exactly the case one wrong band is in.
            want = []
            for j, vlo, vhi in regs:
                o = ro + (first + j) * 104
                g = struct.unpack_from('<i', d, o + 20)[0]
                bdevs = []
                for v, bv in meas.items():
                    if not (vlo <= v <= vhi):
                        continue
                    rv = ref.get(v)
                    if not rv:
                        continue
                    for k in range(klo, khi + 1):
                        dv = _dev_at(bv, rv, k)
                        if dv is not None:
                            bdevs.append(dv)
                if not bdevs:
                    continue
                bdevs.sort()
                want.append(g - int(round(bdevs[len(bdevs) // 2] * 10)))
            if len(want) < 2:
                continue
            want.sort()
            target = want[len(want) // 2]
            for j, vlo, vhi in regs:
                o = ro + (first + j) * 104
                g = struct.unpack_from('<i', d, o + 20)[0]
                ng = max(GAIN_MIN_TENTHS, min(GAIN_MAX_TENTHS, target))
                if ng != g:
                    struct.pack_into('<i', d, o + 20, ng)
                    moved += 1
    open(path, 'wb').write(bytes(d))
    return moved


def balance_from_audit(path, rows, min_step=0.5):
    """Level each region against the readings the audit actually took.

    THE CORRECTION MUST USE THE AUDIT'S OWN POINTS. balance_regions
    reads a grid measured at region velocity midpoints; the audit reads
    the (key, velocity) pairs the corpus plays. Correcting the first and
    reporting the second never converged - five attempts moved the
    outlier from one key to another instead of removing it, and p64
    stayed 13.7 dB out. Correcting the second landed in one pass.

    THE TRUST CHECK IS PER KEY, NOT PER REGION. A region's keys may
    legitimately differ by more than the tolerance - p11's 91-97 holds
    keys 91 and 93 at -9.3 dB and key 97 at +0.7 - and rejecting the
    whole region over that difference left exactly the keys the audit
    complains about untouched. What has to agree is one key's readings
    at the velocities it was played at. A key that fails that is DROPPED
    from the correction; it does not take the region's other keys down
    with it.

    ZONES OF SEVERAL VELOCITY BANDS GO TO collapse_velocity_bands. One
    gain per key zone is the reference's structure, and this pass cannot
    enforce it - it moves each region on its own. Worse, running both on
    the same region applies the correction twice: each derives it from
    the same readings and each writes the same field. The domains are
    disjoint by construction - this levels zones of one band, that
    unifies and levels zones of many.

    Returns how many regions moved.
    """
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, doff = struct.unpack_from('<HHHHI', d, 4)
    io, ro = 16, 16 + n * 7
    moved = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from(
            '<BBBHH', d, io + i * 7)
        r = rows.get(str(prog))
        if not r or not r.get('key_devs'):
            continue
        kd = [(int(k), int(v), float(dv)) for k, v, dv in r['key_devs']]
        pmean = sum(dv for _, _, dv in kd) / len(kd)
        # A ZONE WITH SEVERAL VELOCITY BANDS IS collapse_velocity_bands'
        # JOB, NOT THIS ONE'S. Both passes derive their correction from
        # the same readings and both write the region's gain, so running
        # one after the other doubles it: the second sees a gain the
        # first already moved and a deviation that no longer describes
        # it. The domains are disjoint by construction - this levels
        # zones of one band, that unifies and levels zones of many.
        multi = set()
        zone = {}
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi, vlo, vhi = struct.unpack_from('<HHHH', d, o)
            zone.setdefault((klo, min(khi, 127)), []).append((vlo, vhi))
        for z, bands in zone.items():
            b = sorted(set(bands))
            if len(b) > 1 and all(b[t][1] < b[t + 1][0]
                                  for t in range(len(b) - 1)):
                multi.add(z)
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi, vlo, vhi = struct.unpack_from('<HHHH', d, o)
            khi = min(khi, 127)
            if (klo, khi) in multi:
                continue
            inr = [(k, dv) for k, v, dv in kd
                   if klo <= k <= khi and vlo <= v <= vhi]
            if not inr:
                continue
            byk = {}
            for k, dv in inr:
                byk.setdefault(k, []).append(dv)
            # A KEY WHOSE OWN VELOCITIES DISAGREE IS NOT EVIDENCE, AND
            # IT IS NOT A VETO EITHER. The check used to `continue` the
            # whole region when any one key failed it, so a single bad
            # reading silenced every good reading beside it: p49's
            # region 44-49 holds keys 46, 47 and 48 all at -23 dB and
            # key 45 at -23.7 on two velocities and +1.3 on a third,
            # and that one number left the region 23 dB under gm through
            # four rounds. Dropping the key costs nothing here - the
            # correction is per region, and the key shares that gain
            # with the keys that did agree.
            #
            # How often this fires says whether it is noise: over 3491
            # keys measured at two or more velocities the median
            # velocity spread is 0.00 dB and only 65 exceed the
            # tolerance, so the velocity response is sound and these
            # are bad readings, not a property of the bank.
            keep = set(k for k, v in byk.items()
                       if max(v) - min(v) <= TRUST_SPREAD_DB)
            inr = [(k, dv) for k, dv in inr if k in keep]
            if not inr:
                continue
            step = (sum(dv for _, dv in inr) / len(inr)) - pmean
            # THE CAP BOUNDS A NOISY READING, NOT A CONVERGED ONE.
            # p97's keys 115-127 sit 18 to 24 dB under gm on ten
            # readings that agree with each other to 6 dB, and twelve
            # decibels a pass meant that tail needed two full audits to
            # drain while everything around it was already flat. When
            # several readings agree, the measurement is not what the
            # bound is protecting and the whole correction is the right
            # one; a lone or quarrelsome reading keeps the cap it has
            # always had.
            if (len(inr) < 2 or max(d for _, d in inr)
                    - min(d for _, d in inr) > TRUST_SPREAD_DB):
                step = max(-STEP_CAP_DB, min(STEP_CAP_DB, step))
            if abs(step) < min_step:
                continue
            g = struct.unpack_from('<i', d, o + 20)[0]
            ng = max(GAIN_MIN_TENTHS, min(GAIN_MAX_TENTHS, g - int(round(step * 10))))
            if ng != g:
                struct.pack_into('<i', d, o + 20, ng)
                moved += 1
    open(path, 'wb').write(bytes(d))
    return moved


def _trusted_readings(kd, klo, khi, vlo, vhi, trust=TRUST_SPREAD_DB):
    """The (key, deviation) pairs inside one region that can be trusted.

    A key whose own velocities disagree is not evidence: the reading is
    dropped, and the region's other keys still speak. That is the whole
    rule, and it is the same rule balance_from_audit applies.
    """
    inr = [(k, dv) for k, v, dv in kd
           if klo <= k <= khi and vlo <= v <= vhi]
    byk = {}
    for k, dv in inr:
        byk.setdefault(k, []).append(dv)
    keep = set(k for k, v in byk.items() if max(v) - min(v) <= trust)
    return [(k, dv) for k, dv in inr if k in keep]


def collapse_velocity_bands(path, rows):
    """One gain per key zone, whatever velocity layers the source gave it.

    THE REFERENCE DOES NOT SPLIT VELOCITY AT ALL. Every one of gm.dls's
    129 programs carries a single 0-127 band per key zone, so the
    velocity response is the engine's curve and nothing else. This bank
    inherits up to 22 bands from the source DLS and the calibration has
    been writing each one its own gain ever since - on p75's keys 92-124
    the nine bands run -287, +99, -181, -14, -77, -73, -72, -25, +107
    in tenths, a 40 dB sawtooth where the reference has one number. The
    measured consequence is exactly that sawtooth: keys 118-127 read
    +14.5 dB at velocity 32 and -5 dB at velocity 110.

    ONE GAIN, SET WHERE THE MEASUREMENTS ASK FOR IT. For each band the
    gain it wants is its own gain minus the level its readings are off
    by; the zone takes the median of those and every band is set to it.
    The median is well defined because the bands' gains and their
    readings move together - p75's two ends want -3.8 and -2.3 dB, 1.5
    dB apart, while the gains stood 18 dB apart. Bands with no reading
    are set to the zone's number too: the whole point is that they were
    never supposed to differ.

    OVERLAPPING BANDS ARE LEFT ALONE. Two regions sounding at once are a
    stacked instrument, and forcing one gain on them would change their
    mix rather than level it.

    ONCE PER MEASUREMENT, like every other pass here. The target is
    "the gain that would make these readings sit at pmean", which sounds
    absolute and is: but the readings describe the gain that was in
    force when they were taken. Run this twice on the same rows and the
    second run sees a gain the first already moved and a deviation that
    no longer describes it - 9, 9 became 91, 91 in the unit test. One
    pass per audit, and the audit between them is what makes the next
    one correct.

    Returns how many regions moved.
    """
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, doff = struct.unpack_from('<HHHHI', d, 4)
    io, ro = 16, 16 + n * 7
    moved = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from(
            '<BBBHH', d, io + i * 7)
        r = rows.get(str(prog))
        if not r or not r.get('key_devs'):
            continue
        kd = [(int(k), int(v), float(dv)) for k, v, dv in r['key_devs']]
        pmean = sum(dv for _, _, dv in kd) / len(kd)
        byzone = {}
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi, vlo, vhi = struct.unpack_from('<HHHH', d, o)
            byzone.setdefault((klo, min(khi, 127)), []).append(
                (o, vlo, vhi))
        for (klo, khi), regs in byzone.items():
            if len(regs) < 2:
                continue
            bands = sorted(set((vlo, vhi) for _, vlo, vhi in regs))
            # AT LEAST TWO *DISTINCT* BANDS. Four regions stacked on the
            # same band give a one-element list, and the non-overlap test
            # below is then vacuously true - so a plain zone looked like a
            # layered one and BOTH this pass and balance_from_audit
            # corrected it, doubling the move. p92's top keys swung from
            # -10.7 to +6.8 dB in one round because of it, and the swing
            # read as an unstable measurement rather than as two writers.
            if len(bands) < 2:
                continue
            if not all(bands[t][1] < bands[t + 1][0]
                       for t in range(len(bands) - 1)):
                continue
            want = []
            for o, vlo, vhi in regs:
                inr = _trusted_readings(kd, klo, khi, vlo, vhi)
                if not inr:
                    continue
                step = (sum(dv for _, dv in inr) / len(inr)) - pmean
                step = max(-STEP_CAP_DB, min(STEP_CAP_DB, step))
                g = struct.unpack_from('<i', d, o + 20)[0]
                want.append(g - int(round(step * 10)))
            if not want:
                # UNIFICATION IS STRUCTURAL, LEVELLING NEEDS EVIDENCE.
                # A zone whose bands nobody played can still be told
                # that the reference has one gain, not nine: p64's keys
                # 86-89 carry bands at 69-81 and 109-120 and the sweep's
                # single velocity lands in neither, so `want` is empty
                # and the 14.3 dB sawtooth survived a pass that had
                # flattened everything it had readings for. Setting
                # them to their own median keeps the zone's level where
                # it is and removes the layer the reference never had.
                pivot = sorted(struct.unpack_from('<i', d, o + 20)[0]
                               for o, _, _ in regs)[len(regs) // 2]
                for o, _, _ in regs:
                    g = struct.unpack_from('<i', d, o + 20)[0]
                    if g != pivot:
                        struct.pack_into('<i', d, o + 20, pivot)
                        moved += 1
                continue
            pivot = sorted(struct.unpack_from('<i', d, o + 20)[0]
                           for o, _, _ in regs)[len(regs) // 2]
            target = sorted(want)[len(want) // 2]
            # THE STEP CAP APPLIES TO THE ZONE, NOT TO EACH BAND. The
            # bands have to end up EQUAL, so they cannot be moved by
            # different amounts; what can be bounded is how far the
            # zone as a whole travels in one pass.
            target = pivot + int(max(-STEP_CAP_DB, min(STEP_CAP_DB,
                                 (target - pivot) / 10.0)) * 10)
            target = max(GAIN_MIN_TENTHS, min(GAIN_MAX_TENTHS, target))
            for o, _, _ in regs:
                g = struct.unpack_from('<i', d, o + 20)[0]
                if g != target:
                    struct.pack_into('<i', d, o + 20, target)
                    moved += 1
    open(path, 'wb').write(bytes(d))
    return moved


def fill_key_gaps(path):
    """Make every key sound, not only the ones the source had zones for.

    59 of 129 programs left keys gm plays completely silent. p126 covers
    85-108 and gm covers 0-127, so key 109 measured -55.7 dB with a note
    a tenth as long as the reference's; p77 and p63 are each missing 40
    keys. Nothing in the corpus could have shown this - it plays 24 to 78
    keys per program - and it was the audit's key sweep that found it.

    THE GAP GOES TO THE NEAREST ZONE, which is what the reference does:
    gm's p126 is one region over 0-127 carrying one sample. A zone
    widened this far plays its sample at pitches the source never
    recorded, and that is the point - a stretched sample is an
    approximation, silence is not, and the next calibration pass sets
    whatever gain the stretched one needs. Widening only ever reaches
    into a gap, so two zones cannot end up overlapping and the
    one-gain-per-band invariant stays true.

    Returns how many key slots were filled.
    """
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, doff = struct.unpack_from('<HHHHI', d, 4)
    io, ro = 16, 16 + n * 7
    filled = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from(
            '<BBBHH', d, io + i * 7)
        bands = {}
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi, vlo, vhi = struct.unpack_from('<HHHH', d, o)
            bands.setdefault((vlo, vhi), []).append([o, klo, min(khi, 127)])
        for _, regs in bands.items():
            cover = [False] * 128
            for _, klo, khi in regs:
                for k in range(klo, khi + 1):
                    cover[k] = True
            gaps = []
            k = 0
            while k < 128:
                if cover[k]:
                    k += 1
                    continue
                a = k
                while k < 128 and not cover[k]:
                    k += 1
                gaps.append((a, k - 1))
            for a, b in gaps:
                # NEAREST ZONE BY KEY DISTANCE, lower one on a tie.
                best = None
                for reg in regs:
                    _, klo, khi = reg
                    dist = (a - 1 - khi) if khi < a else ((klo - b - 1)
                                                          if klo > b else -1)
                    if dist < 0:
                        continue
                    if best is None or dist < best[0]:
                        best = (dist, reg)
                if best is None:
                    continue
                reg = best[1]
                o, klo, khi = reg
                nklo, nkhi = min(klo, a), max(khi, b)
                if (nklo, nkhi) != (klo, khi):
                    struct.pack_into('<HH', d, o, nklo, nkhi)
                    reg[1], reg[2] = nklo, nkhi
                    filled += b - a + 1
    open(path, 'wb').write(bytes(d))
    return filled


def _cluster_pieces(vals, lo, hi, edge_db, max_dev=3):
    """Where to cut [lo, hi] so that one gain can serve each piece.

    THE CUT LANDS ON A KEY SOMEBODY READ, never in the middle of the
    unmeasured gap between two of them. The old rule cut at the
    MIDPOINT of the widest gap between two measured keys, which is a
    guess about every key in between - and p123 paid for it: key 95
    read -7.0 dB against gm while its unmeasured neighbours 83..94 and
    96..107 read 0.0, and the midpoint cut at 88 and 101 raised a whole
    span whose neighbours were fine. The outlier then walked from key
    95 to keys 89 and 101 instead of being removed.

    So a cluster of measured keys that disagrees with the rest gets
    EXACTLY ITS OWN KEYS: cuts at its first and last measured key. The
    unmeasured keys between clusters keep the gain they already have,
    because nothing was read there. That is the whole rule - correct
    the keys there is evidence about, leave the rest alone.

    Clustering is by adjacency, so a lone hot key becomes its own
    cluster while a run of keys that agree stays together however far
    apart they are. A single cluster that is itself too wide is a DRIFT
    rather than a spike - there is no outlier to isolate and the keys
    in the gap are a guess either way - and is cut at its widest
    internal gap, on the lower measured key's neighbour, so the piece
    taking the unmeasured keys is the one that continues past them.
    """
    if len(vals) < 2:
        return [(lo, hi)]
    # THE SPAN IS MAX MINUS MIN, NOT LAST MINUS FIRST. `vals` is
    # ordered by KEY, so the values in it are not sorted - p3's keys
    # 49, 50 and 51 read +1.2, -4.0 and -4.1, and "last minus first"
    # came out -5.3, which passed as "under the tolerance" and left
    # the one hot key glued to its neighbours.
    span = max(v for _, v in vals) - min(v for _, v in vals)
    if span <= edge_db:
        return [(lo, hi)]
    cl = [[vals[0]]]
    for kv in vals[1:]:
        if abs(kv[1] - cl[-1][-1][1]) > edge_db:
            cl.append([kv])
        else:
            cl[-1].append(kv)
    cval = [sum(v for _, v in c) / len(c) for c in cl]
    if len(cl) > 1:
        base = max(range(len(cl)), key=lambda i: len(cl[i]))
        dev = sorted(((abs(cval[i] - cval[base]), i)
                      for i in range(len(cl))
                      if i != base and abs(cval[i] - cval[base]) > edge_db),
                     reverse=True)
        cuts = set()
        for _, i in dev[:max_dev]:
            cuts.add(cl[i][0][0])
            cuts.add(cl[i][-1][0] + 1)
        cuts = sorted(k for k in cuts if lo < k <= hi)
        if cuts:
            b = [lo] + cuts + [hi + 1]
            return [(b[t], b[t + 1] - 1) for t in range(len(b) - 1)
                    if b[t] <= b[t + 1] - 1]
        return [(lo, hi)]
    gaps = [(abs(vals[t + 1][1] - vals[t][1]), t)
            for t in range(len(vals) - 1)]
    _, t = max(gaps)
    cut = vals[t][0] + 1
    if not (lo < cut <= hi):
        return [(lo, hi)]
    left = [(k, v) for k, v in vals if k < cut]
    right = [(k, v) for k, v in vals if k >= cut]
    if max(len(left), len(right)) < 2:
        return [(lo, hi)]
    return (_cluster_pieces(left, lo, cut - 1, edge_db, max_dev) +
            _cluster_pieces(right, cut, hi, edge_db, max_dev))


def split_from_audit(path, rows, edge_db=KEY_EDGE_DB):
    """Cut a region where the audit's own keys say it turns.

    split_regions cuts against a grid measured at region velocity
    midpoints; the audit measures the keys the corpus plays. p11's
    region 91-97 holds keys 91 and 93 at -9.3 dB and key 97 at +0.7 -
    one gain cannot hold both, and no grid-based cut ever landed
    between them. The cuts are taken from the audit's readings.

    WHERE each cut lands is decided by _cluster_pieces: on the keys
    that were measured, not on a guess about the keys that were not.

    Returns how many cuts were made.
    """
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, doff = struct.unpack_from('<HHHHI', d, 4)
    io, ro = 16, 16 + n * 7
    out = bytearray()
    counts = []
    cuts = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from(
            '<BBBHH', d, io + i * 7)
        base = len(out) // 104
        kept = 0
        r = rows.get(str(prog)) or {}
        kb = {}
        for k, v, dv in (r.get('key_devs') or []):
            kb.setdefault(int(k), []).append(float(dv))
        med = {k: sorted(v)[len(v) // 2] for k, v in kb.items()}
        for j in range(cnt):
            o = ro + (first + j) * 104
            rec = bytearray(d[o:o + 104])
            klo, khi = struct.unpack_from('<HH', rec, 0)
            khi = min(khi, 127)
            vals = sorted((k, med[k]) for k in range(klo, khi + 1)
                          if k in med)
            pieces = _cluster_pieces(vals, klo, khi, edge_db)
            cuts += max(0, len(pieces) - 1)
            for lo, hi in pieces:
                piece = bytearray(rec)
                struct.pack_into('<HH', piece, 0, lo, hi)
                out += piece
                kept += 1
        counts.append((prog, msb, lsb, kept, base))
    nreg = len(out) // 104
    so_old = 16 + n * 7 + nr * 104
    buf = bytearray()
    buf += d[:16]
    struct.pack_into('<HHHHI', buf, 4, n, nreg, ns, ver, doff)
    for prog, msb, lsb, cnt, first in counts:
        buf += struct.pack('<BBBHH', prog, msb, lsb, cnt, first)
    buf += out
    buf += d[so_old:]
    open(path, 'wb').write(bytes(buf))
    return cuts


def enforce_bound(path, grid, ref, keep, lo_db=LOUD_MIN_DB, hi_db=LOUD_MAX_DB):
    """Move a whole program inside 25%-200% of gm when one gain per
    region could not. The per-region pass corrects what each region
    measures, but layered regions sum: two layers each corrected to
    +6 dB still measure +12 together. This pass reads the program's
    worst measured pair and shifts every region of it by the excess,
    bounded by the room left on the other side so a loud correction
    does not create a quiet note."""
    lv, fallback, rgrid = ref
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', d, 4)
    io = 16
    ro = io + n * 7
    for idx in range(n):
        prog, msb, lsb, count, first = struct.unpack_from(
            '<BBBHH', d, io + idx * 7)
        meas = grid.get(prog)
        rrow = rgrid.get(prog)
        if not meas or not rrow:
            continue
        devs = []
        for v, gv in rrow.items():
            mv = meas.get(v, {})
            for k, r in gv.items():
                c = mv.get(k)
                if r > 0 and c and c > 0:
                    devs.append(20 * math.log10(c / r))
        if not devs:
            continue
        hi, lo = max(devs), min(devs)
        add = 0.0
        if hi > hi_db:
            over = min(hi - hi_db, max(0.0, lo - lo_db))
            add = over * _loud_fix(over)
        elif lo < lo_db:
            under = min(lo_db - lo, max(0.0, hi_db - hi))
            add = -under * _loud_fix(under)
        step = int(round(add * 10))
        if step == 0:
            continue
        for j in range(count):
            o = ro + (first + j) * 104
            g = struct.unpack_from('<i', d, o + 20)[0]
            struct.pack_into('<i', d, o + 20,
                             max(GAIN_MIN_TENTHS, min(GAIN_MAX_TENTHS, g - step)))
    open(path, 'wb').write(bytes(d))


def split_regions(path, grid, rgrid):
    """Split a region at the key where its measured level turns.

    ONE GAIN CANNOT PLEASE A REGION THAT SPANS THE WHOLE BAND. FluidR3's
    piano carries top zones 101-127 and 107-127: inside them one key
    measures 16 dB over gm and another 17 dB under, and every rule for a
    single gain failed the same way - quiet first left 454 pairs over
    the +6.02 dB bound, splitting the excess left 489 over and 47 under,
    correcting the larger left 447 over and 78 under. A step of gain at
    the key where the level turns is the only thing that fixes both, and
    that step is written here as two regions sharing the sample and the
    velocity window.

    THE CUT IS WHERE THE MEASUREMENT CROSSES THE BOUND. The grid is the
    bank's own zone boundaries, so each side of the cut is measured; a
    region whose keys all sit on one side of the bound is left alone.
    """
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, doff = struct.unpack_from('<HHHHI', d, 4)
    io, ro = 16, 16 + n * 7
    # A DEVIATION PER PROGRAM AND KEY, the median over the measured
    # velocities - a velocity layer should not move the cut point.
    dev = {}
    for prog, meas in grid.items():
        rrow = rgrid.get(prog, {})
        for v, gv in meas.items():
            rv = rrow.get(v, {})
            for k, c in gv.items():
                r = rv.get(k)
                if r and c and r > 0 and c > 0:
                    dev.setdefault((prog, k), []).append(
                        20.0 * math.log10(c / r))
    med = dict((k, sorted(v)[len(v) // 2]) for k, v in dev.items())

    def cuts_for(prog, klo, khi):
        """Cut at both edges of every run that needs a gain of its own.

        A run can be hot (over 200%), cold (under 25%), or simply off
        its neighbours by KEY_EDGE_DB; the first two leave the loudness
        window and the third does not, and all three are keys that need
        a gain of their own. A run inside the region is separated from
        its neighbours only by cuts at both ends. Six cuts per region
        bounds the table.
        """
        vals = [(k, med[(prog, k)]) for k in range(klo, khi + 1)
                if (prog, k) in med]
        if len(vals) < 3:
            return []
        out = []

        def runs(before):
            prev = False
            for k, v in vals:
                here = before(v)
                if here != prev and klo < k - 1 < khi:
                    out.append(k - 1)
                prev = here
        runs(lambda v: v > LOUD_MAX_DB)
        runs(lambda v: v < LOUD_MIN_DB)
        # AND AT THE EDGES OF A RUN THAT SITS OFF THE REGION'S OWN
        # MEDIAN. The loudness bounds only find keys that leave the
        # band, and most of the keys that are wrong do not: p3's key 51
        # measured 9.7 dB under gm with every neighbouring key on it,
        # p11's key 91 9.0 and p78's key 100 7.6 - all inside the
        # window, so no cut was ever made and the region's median
        # carried the key along with the keys that were right.
        if len(vals) >= 5:
            mid = sorted(v for _, v in vals)[len(vals) // 2]
            prev = False
            for k, v in vals:
                here = abs(v - mid) > KEY_EDGE_DB
                # klo <= k-1, NOT klo < k-1: an outlier at a region's
                # own first key could never be cut out, because the cut
                # it needs lands on klo and the strict test rejected it.
                # p11's key 91 sat 9.6 dB under gm in a region that
                # STARTS at 91 and was never separated from its
                # neighbours.
                if here != prev and klo <= k - 1 < khi:
                    out.append(k - 1)
                prev = here
        return sorted(set(out))[:6]

    out = bytearray()
    counts = []
    cuts = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from(
            '<BBBHH', d, io + i * 7)
        base = len(out) // 104
        kept = 0
        for j in range(cnt):
            o = ro + (first + j) * 104
            rec = bytearray(d[o:o + 104])
            klo, khi = struct.unpack_from('<HH', rec, 0)
            cs = cuts_for(prog, klo, khi)
            if not cs:
                out += rec
                kept += 1
                continue
            lo = klo
            for c in cs:
                piece = bytearray(rec)
                struct.pack_into('<HH', piece, 0, lo, c)
                out += piece
                kept += 1
                lo = c + 1
            piece = bytearray(rec)
            struct.pack_into('<HH', piece, 0, lo, khi)
            out += piece
            kept += 1
            cuts += len(cs)
        counts.append((prog, msb, lsb, kept, base))

    nreg = len(out) // 104
    so_old = 16 + n * 7 + nr * 104
    buf = bytearray()
    buf += d[:16]
    # THE FIELD AT +12 IS THE DATA BLOCK'S SIZE, NOT ITS OFFSET. ksb_open
    # computes the offset from the tables and adds this field to get the
    # last byte it needs. Adding the region growth to it made every bank
    # with a split fail to open ("truncated"); it must be copied as is.
    struct.pack_into('<HHHHI', buf, 4, n, nreg, ns, ver, doff)
    for prog, msb, lsb, cnt, first in counts:
        # BBBHH IS SEVEN BYTES (3 + 2*2); the record's own size, and
        # the stride the reader walks. No padding was ever written here
        # - adding one shifted the whole region block by 129 bytes.
        buf += struct.pack('<BBBHH', prog, msb, lsb, cnt, first)
    buf += out
    buf += d[so_old:]
    open(path, 'wb').write(bytes(buf))
    print('split %d region(s) at the measured bound' % cuts)
    return cuts


def apply_retune(path, table):
    """Add each program's measured detune to every region's fine-tune."""
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', d, 4)
    io = 16
    ro = io + n * 7
    hit = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from('<BBBHH',
                                                       d, io + i * 7)
        if prog not in table:
            continue
        delta = table[prog]
        for j in range(cnt):
            o = ro + (first + j) * 104
            fine = struct.unpack_from('<h', d, o + 18)[0]
            struct.pack_into('<h', d, o + 18,
                             max(-32767, min(32767, fine + delta)))
            hit += 1
    open(path, 'wb').write(bytes(d))
    print('retuned %d region(s) for %s' % (
        hit, ', '.join('p%d %+d' % kv for kv in sorted(table.items()))))


# THE TARGET IS A BAND, NOT A POINT. A four is meant to land between
# 440.0 and 440.5 Hz - the top of that band is +1.97 cents against the
# 440.0 the measurement assumes - so anything already inside it is left
# alone and the half correction only walks a region back to the nearer
# edge. Equal temperament (0) is the floor, not the goal.
TUNE_BAND = (0.0, 1.97)


def apply_level_trim(path, trim):
    """Shift every region gain of the listed programs by tenths of a
    decibel. The value is a measurement, not a taste: it is the level
    the corpus played that program at against gm's, negated."""
    b = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', b, 4)
    io, ro = 16, 16 + n * 7
    hit = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from('<BBBHH', b, io + i * 7)
        d = trim.get(prog)
        if d is None:
            continue
        for j in range(cnt):
            o = ro + (first + j) * 104
            g = struct.unpack_from('<i', b, o + 20)[0]
            struct.pack_into('<i', b, o + 20, max(GAIN_MIN_TENTHS, min(GAIN_MAX_TENTHS, g + int(d))))
            hit += 1
    open(path, 'wb').write(b)
    print('level trim: %d region(s) in %d program(s)' % (hit, len(trim)))


def level_trim_from_rows(rows, peak_ceil=PEAK_MAX_DB, dead=1.0,
                         peak_min_notes=10, peak_cost_db=6.0):
    """Tenths of a decibel to add to each program's gain, from a corpus
    report - the table apply_level_trim consumes.

    ONE SIGN, ONE PLACE. The residual is the program's measured
    deviation against gm, and the gain moves by its NEGATION: a program
    playing 2 dB under gm gets +20 tenths. This was written out by hand
    three times and got the sign wrong twice - once applying the
    inverse of a whole table, which doubled every error and left
    programs 31 dB under gm with the gain nowhere near its clamp.

    THE PEAK CEILING IS A CONSTRAINT, NOT A SECOND OBJECTIVE. The
    correction is the larger of the level error and the overshoot of
    the peak past `peak_ceil`, because one that fixes the level and
    lifts the peak over the ceiling trades a level problem for a
    clipping one. Where the instrument's peak already sits at gm's the
    overshoot term is negative and the level is free to match gm; where
    it runs hot the peak wins and the level stays down. That cost is
    real and it is the only one a constant gain can pay: the trim moves
    the attack and the body together, so a source whose crest factor
    exceeds gm's cannot have both.

    `peak_ceil` is dB against gm's own peak on the same note, and the
    default is the selection side's PEAK_MAX_DB. Measured on the
    shipped bank that holds the corpus at or under gm's clipping on all
    three test songs. Measured again on 2026-10-30 with
    tools/clip_report.py: 253 clip frames against gm's 248, the crest
    factor within 1.8 dB and the RMS within 0.9 dB.

    `peak_cost_db` IS THE LEVER, AND 2.5 SITS AT ITS KNEE. Dropping it
    to 1.0 - which buys 1.5 dB of level on the 15 programs the guard
    holds down - moves the clipping to 357 frames against gm's 248, and
    town alone goes from 186 to 288. So the 2.5 dB those programs sit
    under gm is not a gap left unfixed: it is what keeping the mix at
    the reference's clipping costs, and the piece-level numbers are the
    gate that says so. Tightening the ceiling instead answers the peak
    harder and takes the level further down.

    TWO GUARDS KEEP THE CEILING FROM BECOMING THE OLD MAX RULE.
    `peak_min_notes` refuses a percentile computed from a handful of
    notes, where the 90th percentile IS the maximum that failed before.
    `peak_cost_db` bounds how much level the peak may buy: past it the
    peak runs over the ceiling and the piece-level clipping gate
    decides, because a transient-heavy instrument answered by lowering
    its body is an instrument that arrives quiet, not one that is fixed.

    `dead` is how far from gm a program has to sit before it is worth
    moving at all - the gain is tenths of a decibel and a program within
    half a tenth is already where the measurement can see.
    """
    trim = {}
    for p, r in rows.items():
        lv = r.get('level_db')
        if lv is None:
            continue
        dev = lv
        pk = r.get('peak_p90_db')
        # A PERCENTILE NEEDS A DISTRIBUTION. p125 was read from four
        # notes, one 19 dB over the rest; its 90th percentile was that
        # note, and the ceiling priced the whole program 15 dB down.
        # Under this many notes the peak says nothing a median would
        # not, so the level is left to the level rule alone.
        if pk is not None and (r.get('peak_notes', r.get('notes',
                                                         peak_min_notes))
                               < peak_min_notes):
            pk = None
        if pk is not None:
            dev = max(dev, pk - peak_ceil)
            # THE PEAK MAY NOT COST MORE THAN THIS MUCH LEVEL. A
            # transient instrument - a slap bass whose attack runs 8 dB
            # over gm's while its body matches - would otherwise be
            # pulled down until the attack agreed, and the instrument
            # would arrive quiet with its character intact and nothing
            # else. Past this the peak is allowed to run over the
            # ceiling instead; the mix's clipping is measured directly
            # on the test pieces and is the gate that decides.
            dev = min(dev, lv + peak_cost_db)
        if abs(dev) > dead:
            # The negation is the whole point of this function. A
            # negative deviation means the program is too quiet, so the
            # gain has to go UP.
            trim[int(p)] = int(round(-dev * 10))
    return trim


def apply_retune_map(path, rmap, chosen, frac=0.5, cap=50,
                     band=TUNE_BAND):
    """Move each region half way to the pitch it was measured at.

    THE MEASUREMENT IS THE RENDER, so the correction is in the same
    cents the engine applies: a region read +23 cents gets -11 cents of
    fine. Half, not all - the reading has its own error, a region often
    has a single test note, and overshooting past equal temperament is
    worse than leaving half the error in place. The half is taken
    against TUNE_BAND: a region already inside 440.0-440.5 Hz is left
    as it is.

    THE MAP MUST BE A READING OF THE UNCORRECTED BANK. tune_pick renders
    the source banks, so the numbers are the source's own detune; a map
    measured on a bank this function has already corrected would be a
    second half-correction on top of the first.
    """
    d = bytearray(open(path, 'rb').read())
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', d, 4)
    io, ro = 16, 16 + n * 7
    hit = 0
    for i in range(n):
        prog, msb, lsb, cnt, first = struct.unpack_from('<BBBHH',
                                                        d, io + i * 7)
        # A DRUM KIT HAS NO PITCH TO CORRECT. Its regions are playback
        # rates on noise-like samples; a "detune" reading there is a
        # measurement of the estimator, not of the instrument.
        if prog == 128 or prog not in chosen:
            continue
        src = chosen[prog].lower()
        row = None
        for name, v in rmap.get(str(prog), {}).items():
            if name.lower() == src:
                row = v
                break
        if not isinstance(row, dict):
            continue
        notes = [(int(k), float(c)) for k, c in row.get('notes', [])]
        # A CLAMPED READING IS NOT A MEASUREMENT. tune_pick pins
        # anything beyond the search window at +-69 so a semitone-out
        # source cannot win on the notes it has left; those pins would
        # here become a 35-cent correction based on nothing.
        notes = [(k, c) for k, c in notes if abs(c) < 40.0]
        if len(notes) < 3:
            continue
        allc = sorted(c for _, c in notes)
        fallback = allc[len(allc) // 2]
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi = struct.unpack_from('<HH', d, o)
            sub = sorted(c for k, c in notes if klo <= k <= khi)
            # A SINGLE TRUSTED NOTE IS ENOUGH. The reading has already
            # been checked against the strongest spectral line of the
            # same render (tune_pick drops it if the two disagree by
            # more than 20 cents), which is what the earlier two-note
            # rule was standing in for; requiring a second note now
            # only leaves real single-region detune - GeneralUser's
            # choir at key 48, 38 cents - uncorrected.
            med = sub[len(sub) // 2] if sub else fallback
            # WALK BACK TO THE NEAREST EDGE OF THE BAND. Below the band
            # the note is flatter than 440.0 and is pulled up; above it
            # the note is sharper than 440.5 and is pulled down; inside
            # it nothing happens, which is the point of naming a band.
            if med > band[1]:
                target = band[1]
            elif med < band[0]:
                target = band[0]
            else:
                continue
            delta = max(-cap, min(cap,
                                  int(round(-frac * (med - target)))))
            if delta == 0:
                continue
            fine = struct.unpack_from('<h', d, o + 18)[0]
            struct.pack_into('<h', d, o + 18,
                             max(-32767, min(32767, fine + delta)))
            hit += 1
    open(path, 'wb').write(bytes(d))
    print('retune %.0f%%: %d region(s) moved' % (frac * 100.0, hit))


def stats(tag, lv, ref):
    lvr, fallback = ref[0], ref[1]
    dev = []
    for prog, v in lv.items():
        target = lvr.get(prog, fallback)
        if v > 0 and target > 0:
            dev.append(20 * math.log10(v / target))
    if not dev:
        return
    dev.sort()
    ad = sorted(abs(x) for x in dev)
    print('level vs gm %s: median %+.1f dB, deviation median %.1f p90 %.1f dB'
          % (tag, dev[len(dev) // 2], ad[len(ad) // 2],
             ad[int(len(ad) * 0.9)]))


def _main(argv):
    global PREFER_MARGIN, PREFER_CAP, SOURCE_PENALTY, VETO_PCT
    global DEDUP_CONTENT, MIN_EXPOSURE
    out = None
    cap = 37800
    kac = os.environ.get('KAC') or ('D:/kac/src/build/kac.exe' if os.path.exists('D:/kac/src/build/kac.exe') else 'kac')
    cal = False
    rate_mode = 'keep'
    keep = 0.3
    refpath = None
    mapping = None
    rendermap = None
    airmap = None
    tonemap = None
    dirtmap = None
    retune_frac = 0.0
    dump_profiles = False
    player = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'kunkun98-play.exe')
    if not os.path.exists(player):
        player = 'kunkun98-play.exe'
    args = list(argv[1:])
    srcs = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--out':
            out = args[i + 1]; i += 2
        elif a == '--cap':
            cap = int(args[i + 1]); i += 2
        elif a == '--kac':
            kac = args[i + 1]; i += 2
        elif a == '--player':
            player = args[i + 1]; i += 2
        elif a == '--calibrate':
            cal = True; i += 1
        elif a == '--keep':
            keep = float(args[i + 1]); i += 2
        elif a == '--ref':
            refpath = args[i + 1]; i += 2
        elif a == '--mapping':
            mapping = args[i + 1]; i += 2
        elif a == '--render-map':
            rendermap = args[i + 1]; i += 2
        elif a == '--tone-map':
            tonemap = args[i + 1]; i += 2
        elif a == '--air-map':
            airmap = args[i + 1]; i += 2
        elif a == '--dirt-map':
            dirtmap = args[i + 1]; i += 2
        elif a == '--retune-frac':
            retune_frac = float(args[i + 1]); i += 2
        elif a == '--profiles':
            dump_profiles = True; i += 1
        elif a == '--prefer':
            name, fac = args[i + 1].rsplit('=', 1)
            PREFER[name.lower()] = float(fac)
            i += 2
        elif a == '--margin':
            PREFER_MARGIN = float(args[i + 1])
            i += 2
        elif a == '--prefer-cap':
            PREFER_CAP = float(args[i + 1])
            i += 2
        elif a == '--crit-weight':
            # ONE CRITERION, NOT A GROUP. --group-cap scales every
            # member together and their ratio stays fixed, so raising the
            # FF8 axis boosts `ff8sim` (a ranked similarity) and `mtone`
            # (a brightness floor) by the same factor - and the two can
            # disagree, which is how a 3x sweep moved p19 from gm
            # (mtone 0.384) to Fury (mtone 0.941), i.e. worse tone from a
            # higher tone weight. When the question is specifically
            # "be closer to FF8's brightness", the brightness criterion
            # is what has to move.
            name, val = args[i + 1].rsplit('=', 1)
            for j, (ck, cs, cw) in enumerate(PROFILE):
                if ck == name.lower():
                    PROFILE[j] = (ck, cs, float(val))
                    break
            i += 2
        elif a == '--group-cap':
            # WHAT A GROUP OF MEASUREMENTS IS WORTH is a decision, so
            # make it an argument instead of a recompile. The FF8 axis
            # is `ff8like` and it is the one worth sweeping: the bank
            # already leans on it (7.0 against the timbre field's 7.4)
            # and still leaves 14 programs where gm and FF8 agree a
            # quarter to a double off both, so "raise it and see" is a
            # question the tool should be able to answer.
            name, val = args[i + 1].rsplit('=', 1)
            GROUP_CAP[name.lower()] = float(val)
            i += 2
        elif a == '--penalty':
            name, fac = args[i + 1].rsplit('=', 1)
            SOURCE_PENALTY[name.lower()] = float(fac)
            i += 2
        elif a == '--no-dedup-content':
            DEDUP_CONTENT = False
            i += 1
        elif a == '--min-exposure':
            name, num = args[i + 1].rsplit('=', 1)
            MIN_EXPOSURE[name.lower()] = int(num)
            i += 2
        elif a == '--trim-map':
            LEVEL_TRIM.update({int(k): int(v) for k, v in
                               json.load(open(args[i + 1])).items()})
            i += 2
        elif a == '--veto':
            VETO_PCT = float(args[i + 1])
            i += 2
        elif a == '--rate':
            rate_mode = args[i + 1]
            i += 2
        else:
            srcs.append(a); i += 1
    if dump_profiles:
        # THE BASE PLUS EACH PATCH, as one line per family. A family
        # that restates a base value is a no-op entry and shows as
        # x.xx->x.xx here, which is how a dead override is found.
        base = {k: w for k, sign, w in PROFILE}
        for lo, hi, patch in FAMILIES:
            parts = ['%s %.2f->%.2f' % (k, base.get(k, 0.0), patch[k])
                     for k in patch]
            print('family %3d-%3d  %s' % (lo, hi - 1, ', '.join(parts)))
        return 0
    if out is None or not srcs:
        raise SystemExit(__doc__)
    global _WORKDIR
    _WORKDIR = tmp = tempfile.mkdtemp(prefix='ksbpick')
    sources = []
    for path in srcs:
        d, lins, wvpl, cues = load(path)
        feats, regions, metrics = programme(d, lins)
        sources.append({'path': path, 'name': info(d).get('INAM', os.path.basename(path)),
                        'feats': feats, 'regions': regions, 'metrics': metrics,
                        'fp': content_signatures(path) if DEDUP_CONTENT else None})
        print('%-24s %-28s programs %3d  %5.1f MiB' % (
            os.path.basename(path), sources[-1]['name'][:28],
            len(regions), os.path.getsize(path) / 1048576.0))
        sources[-1]['grid'] = measure_source(path, player, kac, cap,
                                             rate_mode, tmp)
    # EXPOSURE IS A QUOTA WITH THE LEAST QUALITY LOSS, not a price
    # iterated against the scores: the measured utilities differ by
    # hundredths, so a multiplier large enough to move a count moves
    # every count and the passes oscillate. choose() reassigns only the
    # cheapest programs of an over-full bank instead.
    # RENDERED TUNING AND SEAM, AS CRITERIA. The map comes from
    # tune_pick: each candidate played alone on seven notes and on the
    # whole program for its second-difference peak. Both enter the
    # weighted score; neither excludes a candidate, because gm.dls is
    # the pitch reference and a few cents must not disqualify it.
    if rendermap:
        rmap = json.load(open(rendermap))
        hit = 0
        weak = []
        by_prog = {}
        # WHERE THE SOURCE STANDS AMONG ITS PEERS, ON AN ASYMMETRIC
        # CURVE. Ranks, not raw correlations, because the field moves
        # with the program and a raw number means nothing without its
        # neighbours.
        #
        # THE SHAPE IS NOT A BELL AROUND THE MIDDLE. It used to be
        # symmetric - the middle of the field scored highest, both ends
        # zero - which treats "renders exactly as gm" exactly as bad as
        # "renders as nothing else". The standard is not that:
        #
        #     not so like gm that it is a clone, and not so unlike gm
        #     that it has left the family - and of the two, the like
        #     side is preferred.
        #
        # So the peak sits toward the like side (three quarters of the
        # way up the ranking) and the two tails fall at different
        # rates: leaving the family is the worse failure and is
        # punished twice as steeply. Measured on a full field of ten,
        # the most-gm-like source keeps 0.78 where the least-gm-like
        # keeps 0.00 - at equal distance from the target, like beats
        # unlike by that margin.
        simrank = {}
        SIM_PEAK = 0.75     # where the curve tops out: fairly like gm
        SIM_S_UP = 0.50     # how gently the like side falls away
        SIM_S_DOWN = 0.25   # how steeply the unlike side does
        for k, row in rmap.items():
            # gm IS THE REFERENCE, NOT A CANDIDATE IN ITS OWN
            # COMPARISON. Its correlation with itself is 1.0 by
            # construction, not a measurement. It is left out of the
            # ranking AND out of the scoring: the criterion is absent
            # for gm, so the per-candidate denominator does not carry
            # its weight and the reference is neither rewarded for
            # being itself nor punished for it.
            vals = [(name, float(v['sim'])) for name, v in row.items()
                    if isinstance(v, dict) and v.get('sim') is not None
                    and name.lower() != 'gm']
            if len(vals) < 3:
                continue
            vals.sort(key=lambda t: t[1])
            mid = (len(vals) - 1) / 2.0
            for r, (name, _v) in enumerate(vals):
                x = r / mid if mid > 0 else 0.5
                d = x - SIM_PEAK
                sc = (SIM_S_UP if d > 0 else SIM_S_DOWN)
                simrank[(int(k), name.lower())] = math.exp(-(d / sc) ** 2)
        for s in sources:
            short = os.path.basename(s['path']).rsplit('.', 1)[0].lower()
            for k, row in rmap.items():
                val = None
                for name, v in row.items():
                    if name.lower() == short:
                        val = v; break
                if val is None:
                    continue
                m = s['metrics'].get(int(k))
                if m is None:
                    continue
                notes = val['notes'] if isinstance(val, dict) else val
                g = simrank.get((int(k), short))
                if g is not None:
                    m['sim'] = g
                # FEWER THAN FOUR NOTES IS NOT EVIDENCE. A short decay
                # loses its tail to the next onset and a long one
                # bleeds into it; ranking those as if all seven agreed
                # is how a 0.4-cent instrument read 68. Such a
                # candidate takes the field's median for this program.
                if len(notes) >= 4:
                    m['rtune'] = max(abs(v[1]) for v in notes)
                    by_prog.setdefault(int(k), []).append(m['rtune'])
                else:
                    weak.append((int(k), m))
                if isinstance(val, dict):
                    # ONLY WHERE THERE IS A LOOP. A 99.9th percentile
                    # of the second difference is the brightness of any
                    # attack; on a one-shot it ranks "is this
                    # percussion", not "does this loop click". The
                    # normalized seam value is relative to the peak,
                    # so calibration cannot change the ordering.
                    if m.get('loopstep', 0.0) > 0.0:
                        m['rseam'] = val.get('seam', 0.0)
                    # NO CAP ON THE SPREAD. min-max normalises across
                    # one program's candidates, so a pathological
                    # reading flattens that program's field and no
                    # other - and a cap at 20 dB would have flattened
                    # the drum kit, whose whole field measures 25 to
                    # 44 and would have read 20 across the board.
                    #
                    # GM IS NOT GIVEN A READING. Its spread is measured
                    # against itself and is 0 by construction, which is
                    # a tautology and not evidence that gm's keys are
                    # evenly balanced. Scored, it would hand the
                    # reference a perfect 1.0 on all 129 programs. Left
                    # unset, gm takes the field median like any other
                    # candidate without a reading - the same treatment
                    # `sim` gives it.
                    if short != 'gm':
                        b = val.get('balance')
                        if b is not None:
                            m['balance'] = float(b)
                    # THE RENDERED SIGNAL-TO-NOISE IS NOT A RATIO, so gm
                    # keeps its own reading and competes on it like every
                    # other source. What is measured is peak against the
                    # note's own tail; see the `snr` entry in PROFILE.
                    n = val.get('snr')
                    if n is not None:
                        m['snr'] = float(n)
                hit += 1
        for prog, m in weak:
            vals = by_prog.get(prog)
            if vals:
                m['rtune'] = sorted(vals)[len(vals) // 2]
            else:
                m.pop('rtune', None)
        # LIKE FF8, WHERE FF8 HAS THE PROGRAM. A program FF8 does not
        # carry has no ff8sim reading at all, and the criterion then
        # simply does not apply. FF8 itself is skipped: its correlation
        # with its own render is 1.0 by construction, not a measurement.
        fhit = 0
        for s in sources:
            short = os.path.basename(s['path']).rsplit('.', 1)[0].lower()
            if short == 'ff8':
                continue
            for k, row in rmap.items():
                val = None
                for name, v in row.items():
                    if name.lower() == short:
                        val = v; break
                if not isinstance(val, dict) or val.get('ff8sim') is None:
                    continue
                m = s['metrics'].get(int(k))
                if m is not None:
                    m['ff8sim'] = float(val['ff8sim'])
                    fhit += 1
        if fhit:
            print('ff8 reference: %d candidate reading(s)' % fhit)
        print('render map: %d candidate reading(s) from %s' % (hit, rendermap))
    if tonemap:
        # RENDERED BRIGHTNESS FROM THE CORPUS. Each entry is what that
        # source measured against gm on the notes the corpus plays for
        # this program; gm carries none, so it is excluded the way sim
        # is.
        tmap = json.load(open(tonemap))
        hit = 0
        for s in sources:
            short = os.path.basename(s['path']).rsplit('.', 1)[0].lower()
            for k, row in tmap.items():
                if short in ('gm', 'ff8'):
                    continue
                ent = row.get(short)
                ref = row.get('ff8')
                if not isinstance(ent, dict) or not isinstance(ref, dict):
                    continue
                v = ent.get('tone')
                f8 = ref.get('tone')
                if v is None or f8 is None or v <= 0 or f8 <= 0:
                    continue
                m = s['metrics'].get(int(k))
                if m is None:
                    continue
                m['mtone'] = abs(math.log2(float(v) / float(f8)))
                lv = ent.get('level_db')
                if lv is not None:
                    m['mlevel'] = float(lv)
                hit += 1
        print('tone map: %d candidate reading(s) from %s' % (hit, tonemap))
    if airmap:
        amap = json.load(open(airmap))
        hit = 0
        # THE REFERENCE'S OWN BRIGHTNESS. airfit is the rendered top of
        # the spectrum divided by gm's, and gm is not scored on it: its
        # ratio with itself is 1 by construction, not a measurement,
        # and with per-candidate denominators leaving it out costs it
        # nothing either.
        gram = {}
        gmed = {}
        for k, row in amap.items():
            for name, v in row.items():
                if name.lower() == 'gm' and v:
                    gram[int(k)] = float(v)
            g = gram.get(int(k))
            # GM TAKES THE FIELD'S MEDIAN, AS keyfit AND gmfit DO. A
            # floor the reference does not pay is not a floor, it is a
            # free pass: nine programs moved to gm on this alone while
            # every other candidate sat under it. Its own ratio with
            # itself is 1 by construction, which is not a measurement,
            # so it takes what the field's middle earns.
            if g and g > 0.0:
                rs = [float(v) / g for name, v in row.items()
                      if name.lower() != 'gm' and v]
                if rs:
                    rs.sort()
                    gmed[int(k)] = rs[len(rs) // 2]
        for s in sources:
            short = os.path.basename(s['path']).rsplit('.', 1)[0].lower()
            for k, row in amap.items():
                val = None
                for name, v in row.items():
                    if name.lower() == short:
                        val = v; break
                if val is None:
                    continue
                m = s['metrics'].get(int(k))
                if m is not None:
                    m['rair'] = float(val)
                    # A FLOOR, CAPPED AT ONE. Beyond gm's brightness there
                    # is nothing to gain: rewarding more would put the
                    # brightest candidate back on top, which is what
                    # rair already does field relatively.
                    if short == 'gm':
                        if int(k) in gmed:
                            m['airfit'] = gmed[int(k)]
                    elif gram.get(int(k), 0.0) > 0.0:
                        m['airfit'] = min(1.0, float(val) / gram[int(k)])
                    hit += 1
        print('air map: %d candidate reading(s) from %s' % (hit, airmap))
    if dirtmap:
        # SUB-AUDIO ENERGY AND OFFSET, MEASURED ON THE RENDERED NOTE.
        # These are the same two defects as `rumble` and `dc`, taken at
        # the point the listener hears them; the sample-level readings
        # are band limited to one window and miss a whole program's sum.
        dmap = json.load(open(dirtmap))
        hit = 0
        for s in sources:
            short = os.path.basename(s['path']).rsplit('.', 1)[0].lower()
            for k, row in dmap.items():
                val = None
                for name, v in row.items():
                    if name.lower() == short:
                        val = v; break
                if val is None:
                    continue
                m = s['metrics'].get(int(k))
                if m is not None:
                    m['rsub'] = float(val['sub'])
                    m['rdc'] = float(val['dc'])
                    hit += 1
        print('dirt map: %d candidate reading(s) from %s' % (hit, dirtmap))
    picked, balanced = choose(sources)
    # A MAPPING OVERRIDES THE SELECTION, and it is the render-level
    # measurement that produces it: each program's source is the one
    # whose rendered A4 came nearest equal temperament. Scoring cannot
    # see that - on the finished bank the sample metrics explained
    # none of p58's -30 cents - so where the two disagree, the
    # measured pitch wins.
    if mapping:
        mp = json.load(open(mapping))
        by_name = {}
        for s in sources:
            base = os.path.basename(s['path']).lower()
            by_name[base] = s
            if base.endswith('.dls'):
                by_name[base[:-4]] = s
        hit = 0
        gsrc = next((s2 for s2 in sources
                     if os.path.basename(s2['path']).lower() == 'gm.dls'), None)
        for k, v in mp.items():
            prog = int(k)
            src = by_name.get(v.lower())
            if src is None:
                continue
            # A RENDERED A4 IS NOT A KEYBOARD. The tuning picker tests
            # one note; a source that has that note but stops at key 60
            # leaves the rest of the program silent, and the mapping
            # must not trade a tuned A4 for half an instrument. The
            # same coverage floor the selection uses applies here.
            m = src['metrics'].get(prog)
            if m is None:
                continue
            # THE CALLER'S BAN STILL APPLIES. The mapping is measured
            # pitch, not policy: p0 is not to come from gm however well
            # that one renders. The kit's budget blacklist goes in the
            # same place - if a source is barred from a program it is
            # not worth measuring that program on it, and reversing the
            # bar needs a rebuilt map either way.
            banned = program_ban(prog)
            if banned and os.path.basename(src['path']).lower() in banned:
                continue
            if m.get('keycover', 0.0) < 0.75:
                continue
            # AND A TUNED A4 IS NOT A LEVEL. The selection's loudness
            # gate is the same one here: a source whose quietest key
            # sits below 25% of gm cannot be lifted by the mixer's
            # +12 dB, and its loudest must leave the program inside the
            # 200% band after calibration. Measured on the map alone,
            # one source had a key 78 dB down - silent, and tuned.
            devs = []
            meas = (src.get('grid') or {}).get(prog) or {}
            gref_grid = ((gsrc.get('grid') or {}).get(prog)
                         if gsrc is not None else None) or {}
            if gref_grid:
                for v, gv in gref_grid.items():
                    mv = meas.get(v)
                    if not mv:
                        continue
                    for k, r in gv.items():
                        c = mv.get(k)
                        if r > 0 and c and c > 0:
                            devs.append(20.0 * math.log10(c / r))
            if devs and (min(devs) < LOUD_MIN_DB
                         or max(devs) - min(devs) > 20.0):
                continue
            picked[prog] = src
            hit += 1
        print('mapping: %d program(s) assigned from %s' % (hit, mapping))
    print('gates: %d of %d candidate(s) removed, %d program(s) kept ' 'all (no survivor), %d removed on brightness'
          % (GATE_STATS['removed'], GATE_STATS['total'],
             GATE_STATS['fallback'], GATE_STATS['tone']))
    print('  level gate: %d candidate(s) removed' % GATE_STATS['level'])
    if VETO_PCT > 0:
        print('veto %.1f%%: %d of %d candidates removed, %d program(s) '
              'kept all because nothing would have been left'
              % (VETO_PCT, VETO_STATS['removed'], VETO_STATS['total'],
                 VETO_STATS['fallback']))
    print('\nselection (penalty, balanced 0..1 over the ranked criteria):')
    by_src = collections.defaultdict(list)
    for prog, s in sorted(picked.items()):
        pen = penalty(s['feats'].get(prog, {}))
        by_src[s['path']].append(prog)
        m = s['metrics'].get(prog, {})
        print('  %3d -> %-24s pen %2d  bal %.2f  bw %5d  noise %.3f  '
              'loop %.2f/%.2f  bits %.1f  cover %.2f  %4dK  x%d  '
              'synth %.2f  stretch %d' % (
                  prog, os.path.basename(s['path']), pen,
                  balanced.get(prog, 0.0), int(m.get('bw', 0)),
                  m.get('noise', 0.0), m.get('loopstep', 0.0),
                  m.get('loopslope', 0.0), m.get('bits', 0.0),
                  m.get('keycover', 0.0),
                  int(m.get('bytes', 0) // 1024), int(m.get('samples', 0)),
                  m.get('synth', 0.0), int(m.get('stretch', 0))))
    # Build one bank per source for the programs it won, then merge.
    parts = []
    for n, (path, progs) in enumerate(by_src.items()):
        part = os.path.join(tmp, 'p%d.ksb' % n)
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'ksb_build.py'),
               '--codec', 'kac', '--kac', kac, '-c', str(cap), '--rate', rate_mode,
               path, part]
        cmd += [str(p) for p in sorted(progs)]
        subprocess.run(cmd, check=True)
        parts.append(part)
    merged = parts[0]
    # MERGING IS A CONCATENATION, NOT A RE-ENCODE. ksb_merge.py copies
    # every sample blob and the whole data block byte for byte and only
    # renumbers tables, so no KAC stream is decoded and recompressed no
    # matter how many parts are joined. The output name used to be
    # derived from the input (`p0.ksb.m.ksb.m.ksb...`), which read like
    # a chain of lossy passes; the step number says what actually
    # happened.
    for n, part in enumerate(parts[1:], 1):
        m = os.path.join(tmp, 'merged%d.ksb' % n)
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'ksb_merge.py'),
                        merged, part, m], check=True)
        merged = m
    open(out, 'wb').write(open(merged, 'rb').read())
    print('\nwrote', out)
    if RETUNE:
        apply_retune(out, RETUNE)
    if retune_frac > 0.0 and rendermap:
        apply_retune_map(out, rmap,
                         {p: os.path.basename(v['path']).rsplit('.', 1)[0]
                          for p, v in picked.items()}, retune_frac)
    if cal:
        mid = os.path.join(tmp, 'cal.mid')
        ckeys = cal_keys(out)
        cvels = cal_vels(out)
        write_midi(mid, keys=ckeys)
        ref = ref_levels(refpath, srcs, player, kac, cap, tmp, ckeys,
                         cvels)
        before = levels(out, player, mid, os.path.join(tmp, 'b.wav'), ckeys)
        before_grid = levels_grid(out, player, mid,
                                  os.path.join(tmp, 'bg.wav'),
                                  vels=cvels, keys=ckeys)
        # THE SPLIT COMES FIRST, so the pass below measures the pieces
        # it is about to correct. Everything after it - the passes, the
        # bound, the retune - reads the same table, so a region split
        # here is simply two regions further down.
        split_regions(out, before_grid, ref[2])
        before = levels(out, player, mid, os.path.join(tmp, 'b.wav'), ckeys)
        before_grid = levels_grid(out, player, mid,
                                  os.path.join(tmp, 'bg0.wav'),
                                  vels=cvels, keys=ckeys)
        done = set()
        calib(out, before, before_grid, ref, keep, done)
        # A RESIDUAL IS MEASURED AGAIN. One pass corrects the bank that
        # was measured; the bank it wrote is a different measurement -
        # a region narrow in velocity took its neighbour's deviation and
        # lands several dB off, which showed up as p114 14 dB quiet at
        # velocity 24 and p71 12 dB quiet at key 24. Two more passes
        # measure what is actually there and keep moving it toward gm;
        # the band means each pass keeps 30% of what is left, so they
        # converge rather than oscillate.
        curg = before_grid
        for it in range(3):
            # THE BOOSTS MAKE NEW HOT KEYS. A layer lifted to reach the
            # quiet floor takes its neighbouring keys over the loud
            # bound, and no split from before the pass can see that -
            # p71's bassoon was 20 dB hot at key 90 only after the
            # boost. The table is split again from the fresh
            # measurement, so the next pass corrects each piece.
            split_regions(out, curg, ref[2])
            cur = levels(out, player, mid,
                         os.path.join(tmp, 'c%d.wav' % it), ckeys)
            curg = levels_grid(out, player, mid,
                               os.path.join(tmp, 'cg%d.wav' % it),
                               vels=cvels, keys=ckeys)
            calib(out, cur, curg, ref, keep, done)
        curg = levels_grid(out, player, mid, os.path.join(tmp, 'cg2.wav'),
                           vels=cvels, keys=ckeys)
        # THE SPREAD, AFTER THE LEVEL IS SET. calib has by now moved
        # each region toward gm and left the program at the right level;
        # what remains is the width across its own keys, which is what
        # the level pass cannot see and the loudness bound cannot fit
        # when it is wider than the bound's window. Closing it here, in
        # one pass against the program's own mean, leaves the level
        # where calib put it.
        balance_regions(out, curg, ref[2])
        curg = levels_grid(out, player, mid, os.path.join(tmp, 'cg3.wav'),
                           vels=cvels, keys=ckeys)
        # ONE GAIN PER KEY ZONE, AFTER EVERY PER-BAND PASS HAS RUN. A
        # source with twelve velocity layers had twelve gains where the
        # reference has one, and two notes of evidence behind each; on
        # p27 they ended 14.4 dB apart and that difference WAS the
        # measured layer error. This is the last word on the bands
        # because calib and balance_regions both act per band and would
        # undo it.
        unify_velocity_gains(out, curg, ref[2])
        curg = levels_grid(out, player, mid, os.path.join(tmp, 'cg3b.wav'),
                           vels=cvels, keys=ckeys)
        enforce_bound(out, curg, ref, keep)
        # THE PEAK, NOT ONLY THE ENERGY. enforce_bound keeps a program
        # inside 25%-200% of gm on the RMS, and a transient-heavy
        # instrument can be inside that and still drive the mix past
        # full scale: Tinkle Bell measured +10.8 dB over gm at its peak
        # while its RMS was in band, and the three audition pieces
        # clipped 17/14/51 frames against gm's 3/22/37. The peak gets
        # its own bound, tighter on top - the mix has no headroom for
        # a drum over the reference - and loose below, because being
        # quiet is not what clips.
        pref = ref_levels(refpath, srcs, player, kac, cap, tmp, keys=ckeys,
                          vels=cvels, peaks=True)
        curp = levels_grid(out, player, mid, os.path.join(tmp, 'cg4.wav'),
                           vels=cvels, keys=ckeys, peaks=True)
        # NO FLOOR ON THE PEAK SIDE. The first attempt used the RMS
        # floor (25% of gm) and that BOOSTED most of the bank: the peak
        # deviation runs wide across a program's keys, so nearly every
        # program had one below the floor and was lifted toward it.
        # onestop went from 17 clipped frames to 61. Being quiet is not
        # what overloads the mix; only the top is.
        enforce_bound(out, curp, pref, keep,
                      lo_db=NO_FLOOR_DB, hi_db=PEAK_MAX_DB)
        after = levels(out, player, mid, os.path.join(tmp, 'a.wav'), ckeys)
        stats('before', before, ref)
        stats('after ', after, ref)
    if LEVEL_TRIM:
        apply_level_trim(out, LEVEL_TRIM)
    return 0

def main(argv):
    """Run a pick and remove the work directory it made.

    THE WORK DIRECTORY WAS NEVER REMOVED. tempfile.mkdtemp left one per
    run, about 15 GB once calibration had rendered its velocity grid,
    and nothing deleted it - not on success, not on failure. Ten
    iterations filled the system drive and aborted a run mid-calibration
    with "No space left on device"; the grid alone is four WAVs of 1.2
    GB each.

    Removed on success. KEPT on failure, with its path printed: those
    WAVs are the evidence for a failed calibration and deleting them
    along with the run would throw away the only copy."""
    global _WORKDIR
    _WORKDIR = None
    rc = None
    try:
        rc = _main(argv)
    finally:
        if _WORKDIR:
            if rc == 0:
                shutil.rmtree(_WORKDIR, ignore_errors=True)
            else:
                print('work directory kept: %s' % _WORKDIR, file=sys.stderr)
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv))
