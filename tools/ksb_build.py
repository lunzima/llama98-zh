"""Build a KSB sample bank from a DLS.

    python ksb_build.py <in.dls> <out.ksb> [-k K] [-p P] [-f HZ] <gm program>...

KSB carries the same thing a DLS does - instruments, key zones, loop
points, samples - in a flat little-endian layout with no RIFF, no GUIDs
and no articulation chunks, because the reader is a few hundred bytes of
C on a machine that may not have a filesystem cache, let alone COM.

TWO CODINGS, and choosing between them is the whole size/quality trade
of this bank:

    kac      one codec image per sample, decoded by the loader. The
             record's length field is the image's byte count; its frame
             count lives inside the image, where it cannot go stale.
    pcm      signed 16-bit, read in place. Twice the size, and the
             ceiling kac is judged against - "is the noise the codec or
             the resampler" is answerable by building the same rate
             both ways and measuring, not by reasoning about either.

WHY IT IS SMALLER THAN A DLS, measured on this bank rather than assumed:
per-wave rate. The waves are stored at 15 to 44 kHz and half of them
keep 99.9% of their energy below 3 kHz. Each is resampled to its own
content instead of to a common rate, which is where most of the saving
is.

THE LOOP DECIDES THE RATE, not the other way round. A looped sample
sounds at rate/loop_length, so resampling to a chosen rate and then
rounding the loop length to whole frames retunes it - on this bank's
single cycle waveforms, whose loops are 17 frames, by 44 cents. So the
loop is asked for exactly round(len*ratio) output frames spanning
exactly len input frames, and the rate stored is whatever makes that
loop sound at its original pitch. What is left is rounding the rate to
a whole hertz, which is under a fifth of a cent.

Wants numpy and scipy; it is a build tool and does not ship. What ships
is the .ksb and the C reader, which have no dependencies.

THE TARGET IS MICROSOFT GS, measured rather than remembered. Its own
bank - gm.dls, from the system32 drivers directory - spends 547 KB on
these ten programs and stores every one of them at 22050 Hz, 16-bit, so
it band limits the lot to 11 kHz. That fixes both ends of the knob:
content below each wave's own limit must survive, and content above
11 kHz need not, because the reference bank does not keep it either.

    -k K    sample rate as a multiple of the content bandwidth (2.2)
    -p P    the fraction of energy that defines that bandwidth (0.9999)
    -f HZ   never store below this rate (4000)
    -c HZ   never store above this rate (24000, i.e. Microsoft GS + 10%)
    --kac PATH, --kac-v N
            what --codec kac runs, and the rung of its own quality
            ladder it runs at. Default $KAC, or `kac` on PATH, and -V 6.
    --codec-for PROG:KLO:KHI=NAME
            code one instrument's key range as NAME instead of --codec,
            repeatable. The rest of the bank is unaffected.

The ten programs are the player's timbre menu, not a selection made
here. The source is a DLS: gm.dls is the size and bandwidth this bank
is measured AGAINST, not built from. Its 342 instruments hold one bank
per piece of music, so `bank != 0` below is what stops a program match
from returning seven different square leads.

A BUILD AT ITS SHIPPING SHAPE:

    python3 ksb_build.py --codec kac -c 12000 <FF8.dls> banks/kk98.ksb \
        0 4 6 7 12 16 25 46 80 81
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

from dls_survey import GM_NAMES, chunks, find
from dls_play import wsmp_of

MAGIC = b'KSB1'
BIAS = 0x84
CLIP = 32635


CODECS = {'pcm': 0, 'kac': 1}

# What --codec kac runs, and at what rung. Set from --kac / --kac-v.
KAC = {'bin': os.environ.get('KAC', 'kac'), 'v': 6}
KSB_FORMAT_VERSION = 1
def _wav_mono16(pcm, rate):
    """One mono 16-bit PCM WAV, the layout the encoder's reader takes.

    Written out rather than through the `wave` module: this file already
    has a function of that name, and the header is nine fields.
    """
    n = len(pcm)
    return (b'RIFF' + struct.pack('<I', 36 + 2 * n) + b'WAVE'
            + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, int(round(rate)),
                                    int(round(rate)) * 2, 2, 16)
            + b'data' + struct.pack('<I', 2 * n) + pcm.tobytes())


def kac_encode(x, rate):
    """Frames -> a KAC image, through that codec's own binary.

    The encoder owns its ladder, so this runs it rather than carrying a
    second copy of it here; KAC['bin'] is where it is and KAC['v'] the
    rung it is asked for.

    A whole file image, header and all, and its TotalSamples is the same
    frame count this bank's own sample record carries - so a reader that
    stops where the record says and one that stops where the image says
    agree, and nothing has to be trimmed. The image is a stream's worth
    longer than that only in its stream form (the last chunk pads up to
    its slice); the file form states the count and does not.

    `rate` arrives already rounded to whole hertz by the caller, so the
    WAV written here and the frames handed in are the same rate.
    """
    q = np.clip(np.rint(x), -32768, 32767).astype('<i2')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'in.wav')
        dst = os.path.join(d, 'out.kac')
        with open(src, 'wb') as f:
            f.write(_wav_mono16(q, rate))
        r = subprocess.run([KAC['bin'], '-V', str(KAC['v']), src, dst],
                           capture_output=True)
        if r.returncode != 0:
            # Reported, not swallowed: a bank that silently dropped this
            # wave would differ from the one the run says it built.
            raise RuntimeError('kac failed on %d frames at %d Hz: %s'
                               % (len(q), int(round(rate)),
                                  r.stderr.decode('utf-8', 'replace').strip()))
        with open(dst, 'rb') as f:
            return f.read()


def encode(x, codec, rate=22050):
    """Frames -> bytes.

    THE TWO EXIST TO BE COMPARED. "Is the noise the codec or the
    resampler" is not answerable by reasoning about either; it is
    answerable by building the same rate both ways and measuring what
    comes out of the player. `pcm` doubles the bank and is the ceiling
    `kac` is judged against.
    """
    if codec == CODECS['kac']:
        # Full range: KAC takes sixteen bits, so nothing is scaled down
        # and back on the way in.
        return kac_encode(x, rate)
    # ROUND, NOT SHIFT. `q >> 8` truncates towards minus infinity, which
    # would throw away about six decibels and leave a DC offset of half a
    # step. Rounding first and storing at full width is what the
    # compressed coding is measured against.
    return np.clip(np.rint(x), -32768, 32767).astype('<i2').tobytes()


def resample(x, pos, ratio, taps=32, beta=9.0, wrap=None):
    """x read at arbitrary positions, band-limited to the new rate.

    A KAISER-WINDOWED SINC, 64 taps wide. This runs once, on a desktop,
    on about a million frames, so there is no reason to be cheap about
    it - and the sixteen tap Blackman that was here cost 1.7 dB at
    10 kHz when downsampling 44100 to 22050, measured on a chirp, where
    this is flat. The player's interpolator is where cost matters and
    that one is eight taps; see fir_table.h.

    `taps` and `beta` are parameters rather than buried constants so
    that whatever built a shipped .ksb can be read back off
    `resample.__defaults__` and compared against the values recorded
    for that artifact - see resample_floor_gate.py.

    WITH `wrap` = (start, length) THE FILTER SEES THE LOOP AS ENDLESS.
    Every loop in this bank runs to the last frame of its wave, so a
    filter that clamps at the array's end has real samples on the loop's
    near side and a repeated last sample on its far side; folding the
    taps back into the loop makes both ends see the same thing.

    It is right by derivation and small in effect: measured against the
    clamping version it moves no sample by more than 3.6% of the loop's
    RMS and most by under 0.1%. An earlier note here claimed it was
    worth a factor of two at the join, which was wrong - that pair of
    numbers was taken before the loop points themselves were fixed, so
    it compared two different loops."""
    cut = min(1.0, ratio)
    n = len(x)
    out = np.zeros(len(pos))
    base = np.floor(pos).astype(np.int64)
    i0b = np.i0(beta)
    for k in range(-taps + 1, taps + 1):
        t = pos - (base + k)
        i = base + k
        if wrap is not None:
            ls, ll = wrap
            i = ls + np.mod(i - ls, ll)
        a = np.clip(t / taps, -1.0, 1.0)
        w = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - a * a))) / i0b
        out += x[np.clip(i, 0, n - 1)] * np.where(np.abs(t) <= taps,
                                                  np.sinc(t * cut) * w, 0.0)
    return out * cut


# DLS articulation destinations that a STATIC connection (src == control
# == none) can carry. The four EG1 ones were all this file ever read;
# modEnv, the mod LFO and pan are static on gm.dls too and were dropped
# silently, so a bank built from it lacked every slow modulation the
# sound has.
#
# The unit of `scale` for each, from SpessaSynth's conversion of the
# same chunk (its `shortScale` is `scale >> 16`, which is the same
# number this file gets by dividing by 65536):
#
#   volEnv/modEnv times  time cents, 2^(v/1200) seconds
#   both sustains        0.1 percent of the level
#   pan                  0.1 percent, -500..500 is the full span
#   mod LFO freq/delay   time cents, the same as an envelope time
DST_EG1 = {0x0206: 'attack', 0x0207: 'decay', 0x0209: 'release',
           0x020A: 'sustain'}
DST_STATIC = {
    0x0206: 'attack', 0x0207: 'decay', 0x0209: 'release',
    0x020A: 'sustain',
    0x030A: 'mattack', 0x030B: 'mdecay', 0x030D: 'mrelease',
    0x030E: 'msustain',
    0x0004: 'pan', 0x0104: 'lfofreq', 0x0105: 'lfodelay',
}

# What a zone with no articulation at all plays as. Every field is
# present so the merge below is a plain dict update: a list that
# declares nothing leaves every one of these, and a list that declares
# one thing overwrites that one key and nothing else.
ARTIC_DEFAULT = {
    'attack': 0, 'decay': 65000, 'release': 200, 'sustain': 0,
    'mattack': 0, 'mdecay': 65000, 'mrelease': 200, 'msustain': 0,
    'pan': 0, 'lfofreq': 0, 'lfodelay': -32768,
    'lpitch': 0, 'lgain': 0, 'mpitch': 0, 'kdecay': 0, 'vatk': 0,
}


def envelope(d, a, b):
    """-> dict of every STATIC articulator, or None.

    WITHOUT THIS EVERY NOTE IS AN ORGAN NOTE. Both banks' pianos loop
    for ever and declare EG1_SUSTAIN = 0% with a decay of twenty to
    forty seconds: the sample never stops, and the whole of the piano's
    decay is the volume envelope. A player that ignores the envelope
    holds every note flat until note-off, and three voices of that is a
    wash rather than a piano - which is exactly what it sounded like.

    Times are DLS time cents, 2^(v/65536/1200) seconds, and the sustain
    is in tenths of a percent as 16.16 - both confirmed against
    OpenMPT's CDLSBank, which has been rendering this exact gm.dls for
    twenty years. Its DLS32BitTimeCentsToMilliseconds is the same
    expression, and its DLSSustainLevelToLinear divides by 1000*512,
    which agrees with the calibration here: the square and saw leads,
    the two that obviously sustain, both read 63438848 for 96.8%.

    WHAT IS RETURNED IS THE SUSTAIN'S ATTENUATION, not its level, and
    the times mean something specific that had to be read rather than
    guessed. OpenMPT computes `decayTime = attenuation * decay / 960`,
    where 960 is 96 dB in tenths - so the DLS decay time is the time to
    fall 96 dB and the envelope is linear in DECIBELS. Guessing 60 dB,
    which was the plan, would have made every decay 1.6 times too fast.
    Handing the attenuation over in tenths of a decibel also keeps the
    player from needing a logarithm.
    """
    la, lb = find(d, a, b, b'LIST', b'lart')
    if la is None:
        la, lb = find(d, a, b, b'LIST', b'lar2')
    if la is None:
        return None
    aa, _ = find(d, la, lb, b'art1')
    if aa is None:
        aa, _ = find(d, la, lb, b'art2')
    if aa is None:
        return None
    cb, n = struct.unpack('<II', d[aa:aa + 8])
    got = {}
    mod = {}
    for i in range(n):
        src, ctl, dst, tr, sc = struct.unpack(
            '<HHHHi', d[aa + cb + i * 12:aa + cb + i * 12 + 12])
        if src == 0 and ctl == 0:
            # Only the unmodulated connection for the static values. The
            # others are the same destination scaled by key number or
            # velocity, and taking one of those as the value gives a
            # decay that depends on nothing.
            if dst in DST_STATIC:
                got[DST_STATIC[dst]] = sc
        else:
            # A modulator, kept by its own pair: gm.dls's largest is
            # modLfo -> pitch, 890 of them, which is the vibrato every
            # instrument in this bank has and none of them was read.
            key = (src & 0xFF, dst)
            mod[key] = mod.get(key, 0) + sc

    def sh(key, default=0):
        """A modulator's amount, as it stands: cents for pitch, dB for
        gain, and time cents per full-scale source for a time."""
        if key not in mod:
            return default
        v = int(round(mod[key] / 65536.0))
        return max(-32768, min(32767, v))

    def ms(key, default):
        if key not in got:
            return default
        return int(max(0.0, min(65.0, 2.0 ** (got[key] / 65536.0 / 1200.0)))
                   * 1000.0)

    def tc(key, default):
        """A time-cents value kept as time cents, not as milliseconds."""
        if key not in got:
            return default
        v = int(round(got[key] / 65536.0))
        return max(-32768, min(32767, v))

    def sus(key, default=0):
        """Sustain the way the reference player converts it: `1000 -
        value`, where the value is the destination's own 0.1 percent
        units - so the answer is centibels of attenuation and 1000 is
        silence.

        This used to be read as a percentage LEVEL and a logarithm taken
        of that. It is a different curve and not a rounding of this one:
        the value 968 comes out 2 cB the old way and 32 cB this way, and
        3 dB of held level is not a quantisation. It is also the form
        that inverts cleanly, which matters for the modulation envelope
        where the same number is turned back into a linear level."""
        if key not in got:
            return default
        v = int(round(got[key] / 65536.0))
        if v <= 0:
            return 960
        if v >= 1000:
            return 0
        return min(960, 1000 - v)

    # ONLY WHAT THIS LIST DECLARES. The merge with the instrument's own
    # articulation happens per destination downstream, so a region that
    # restates its attack does not take the instrument's pan, LFO or
    # sustain away with it. Replacing wholesale - which is what this did
    # - means a region carrying one connection discards everything the
    # instrument said that it does not itself mention.
    out = {}
    for name in ('attack', 'decay', 'release', 'mattack', 'mdecay',
                 'mrelease'):
        if name in got:
            out[name] = ms(name, 0)
    for name in ('sustain', 'msustain'):
        if name in got:
            out[name] = sus(name)
    if 'pan' in got:
        out['pan'] = max(-32768, min(32767,
                                     int(round(got['pan'] / 65536.0))))
    if 'lfofreq' in got:
        out['lfofreq'] = tc('lfofreq', 0)
    if 'lfodelay' in got:
        out['lfodelay'] = tc('lfodelay', -32768)
    # A modulator amount, by its own (source, destination) pair. DLS
    # source ids: modLfo 1, velocity 2, keyNum 3, modEnv 5.
    for name, key in (('lpitch', (0x01, 0x0003)),   # vibrato, cents
                      ('lgain', (0x01, 0x0001)),    # tremolo, dB
                      ('mpitch', (0x05, 0x0003)),   # pitch envelope, cents
                      ('kdecay', (0x03, 0x0207)),   # decay per key
                      ('vatk', (0x02, 0x0206))):    # attack per velocity
        if key in mod:
            out[name] = sh(key)
    return out or None


def bandwidth(x, rate, frac):
    """The frequency below which `frac` of the signal's energy lies."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), 1.0 / rate)
    cum = np.cumsum(spec)
    if cum[-1] <= 0:
        return rate / 2.0
    return freq[np.searchsorted(cum / cum[-1], frac)]


def read_dls(path, want):
    d = open(path, 'rb').read()
    top = 8 + struct.unpack('<I', d[4:8])[0]
    lins = find(d, 12, top, b'LIST', b'lins')
    wvpl_a, wvpl_b = find(d, 12, top, b'LIST', b'wvpl')
    pt_a, _ = find(d, 12, top, b'ptbl')
    cb, ncue = struct.unpack('<II', d[pt_a:pt_a + 8])
    cues = struct.unpack('<%dI' % ncue, d[pt_a + cb:pt_a + cb + 4 * ncue])

    insts = []
    for cid, f, a, b in chunks(d, lins[0], lins[1]):
        if cid != b'LIST' or f != b'ins ':
            continue
        ih_a, _ = find(d, a, b, b'insh')
        if ih_a is None:
            continue
        _, bank, prog = struct.unpack('<III', d[ih_a:ih_a + 12])
        if prog & 0x80000000 or bank != 0 or (prog & 0x7F) not in want:
            continue
        regs = []
        lr_a, lr_b = find(d, a, b, b'LIST', b'lrgn')
        for c2, f2, r_a, r_b in chunks(d, lr_a, lr_b):
            if c2 != b'LIST' or f2 not in (b'rgn ', b'rgn2'):
                continue
            rh_a, _ = find(d, r_a, r_b, b'rgnh')
            klo, khi = struct.unpack('<HH', d[rh_a:rh_a + 4])
            wl_a, _ = find(d, r_a, r_b, b'wlnk')
            idx = struct.unpack('<I', d[wl_a + 8:wl_a + 12])[0]
            regs.append((klo, khi, idx, wsmp_of(d, r_a, r_b),
                         envelope(d, r_a, r_b)))
        insts.append((prog & 0x7F, envelope(d, a, b), regs))
    insts.sort()
    return d, wvpl_a, cues, insts


def wave(d, wvpl_a, cues, idx):
    a = wvpl_a + cues[idx]
    b = a + 8 + struct.unpack('<I', d[a + 4:a + 8])[0]
    fa, _ = find(d, a + 12, b, b'fmt ')
    tag, ch, rate, _, _, bits = struct.unpack('<HHIIHH', d[fa:fa + 16])
    if tag != 1 or bits != 16:
        raise SystemExit('wave %d is not 16-bit PCM' % idx)
    da, db = find(d, a + 12, b, b'data')
    n = (db - da) // (2 * ch)
    x = np.frombuffer(d[da:da + n * ch * 2], dtype='<i2').astype(np.float64)
    return x[::ch] if ch > 1 else x, rate, wsmp_of(d, a + 12, b)
def main(argv):
    k, frac, floor, cap = 2.2, 0.9999, 4000.0, 24000.0
    codec = CODECS['pcm']
    # PER-REGION, OFF BY DEFAULT. The sample table's codec field is
    # already per sample; this is the knob for a region whose own
    # measured cost - a lossy transform not closing its loop, for
    # instance - argues for a different coding than the rest of the
    # bank, without a second whole-bank build. Empty unless asked for,
    # so nothing already shipping moves.
    codec_for = {}
    src = dst = None
    want = set()
    it = iter(argv[1:])
    for a in it:
        if a == '-k':
            k = float(next(it))
        elif a == '-p':
            frac = float(next(it))
        elif a == '-f':
            floor = float(next(it))
        elif a == '-c':
            cap = float(next(it))
        elif a == '--codec':
            codec = CODECS[next(it)]
        elif a == '--codec-for':
            loc, name = next(it).split('=')
            pr, klo_s, khi_s = loc.split(':')
            codec_for[(int(pr), int(klo_s), int(khi_s))] = CODECS[name]
        elif a == '--kac':
            KAC['bin'] = next(it)
        elif a == '--kac-v':
            KAC['v'] = int(next(it))
        elif src is None:
            src = a
        elif dst is None:
            dst = a
        else:
            want.add(int(a))
    if not want:
        print(__doc__.strip().splitlines()[2].strip())
        return 2

    # --match's WHOLE VALUE IS BEING READ FRESH, not carried from
    # whatever built the reference bank: two invocations of this file
    # agreeing on (ls2, ll2, rate) is only worth anything checked from
    # the artifact each one actually wrote, the same reason
    # `preecho_verify.py` reads its own driver's stdout rather than
    # trusting a value computed earlier in the same run.

    d, wvpl_a, cues, insts = read_dls(src, want)

    # One entry per DLS wave actually referenced, encoded once even when
    # several regions share it.
    smap, samples, blobs, off = {}, [], [], 0
    regions, inst_rows = [], []
    for prog, ienv, regs in insts:
        inst_rows.append((prog, len(regions), len(regs)))
        for klo, khi, idx, rws, renv in regs:
            # THE REGION'S wsmp WINS, loop points included. A region may
            # override the wave's, and in this bank 37 of the 86 do -
            # taking the pitch from the region and the loop from the
            # wave gave the harp a 137-frame loop in one reader and a
            # 6284-frame one in the other. Ten waves serve more than one
            # region, so the key is the loop as well as the wave: two
            # regions sharing a wave with different loops are two
            # different samples.
            unity, fine, gain, ls, ll = (
                rws if rws is not None else wave(d, wvpl_a, cues, idx)[2])
            # A sample the dedup key (idx, ls, ll) also matches from a
            # DIFFERENT program keeps whichever codec encoded it first -
            # the override is keyed by the region asking for it, and
            # this bank's ten programs do not share waves across
            # instruments, so that case does not arise here.
            this_codec = codec_for.get((prog, klo, khi), codec)
            key = (idx, ls, ll)
            if key not in smap:
                x, rate, _ = wave(d, wvpl_a, cues, idx)
                tgt = max(floor, min(rate, cap,
                                     k * bandwidth(x, rate, frac)))
                r = tgt / float(rate)
                if ll > 1 and ls + ll <= len(x):
                    ls2 = int(round(ls * r))
                    ll2 = max(8, int(round(ll * r)))
                    # SHORT LOOPS ARE UNROLLED FIRST. The rate below is
                    # derived from the rounded loop length, so a loop of
                    # 42 frames carries the whole half-frame of rounding
                    # error - 20 cents, and a rate 130 Hz off the one
                    # the bank is supposed to store uniformly. Storing
                    # several turns of the loop instead divides that
                    # error by the number of turns; 512 frames puts it
                    # under two cents and costs at most half a kilobyte
                    # a sample.
                    turns = 1
                    while ll2 * turns < 512:
                        turns += 1
                    # The span is what gets stored; `ll` stays the
                    # real loop, because that is what the filter
                    # folds its taps into and folding over the
                    # unrolled length would reach past the end of
                    # the source.
                    span = ll * turns
                    ll2 = max(8, int(round(span * r)))
                    # THE STORED RATE FOLLOWS THE ROUNDED LOOP, not
                    # the other way round. A looped sample's pitch
                    # is rate/loop_length, so rounding the length
                    # and keeping the rate detunes it - and for the
                    # single cycle waveforms in this bank, where the
                    # loop is 17 frames, by 44 cents. Deriving the
                    # rate from the length instead makes rate/ll2
                    # exactly the original rate/ll, and the only
                    # error left is rounding the rate itself to a
                    # whole hertz.
                    tgt = rate * ll2 / float(span)
                    r = tgt / float(rate)
                    head = (np.arange(ls2) * (ls / float(ls2))
                            if ls2 else np.zeros(0))
                    body = ls + np.arange(ll2) * (span / float(ll2))
                    # The attack is filtered against the wave, the
                    # loop against itself, and joined; see
                    # resample().
                    y = np.concatenate(
                        [resample(x, head, r),
                         resample(x, body, r, wrap=(ls, ll))])
                    blob = encode(y, this_codec, int(round(tgt)))
                else:
                    ls2 = ll2 = 0
                    m = max(8, int(round(len(x) * r)))
                    y = resample(x, np.arange(m) * (len(x) / float(m)), r)
                    blob = encode(y, this_codec, int(round(tgt)))
                smap[key] = len(samples)
                # The record's length field is the frame count, except for a
                # coding whose frames cannot be multiplied out from it. KAC is
                # that: a reader hands the decoder a pointer and a length and
                # nothing else, so its count lives inside the image where it
                # cannot go stale, and the record carries the byte count.
                nfield = len(blob) if this_codec == CODECS['kac'] else len(y)
                samples.append((off, nfield, ls2, ll2, int(round(tgt)),
                                this_codec, 0))
                blobs.append(blob)
                off += len(blob)
            # PER DESTINATION, not wholesale. The region's list covers
            # only the destinations it declares and the instrument's
            # covers the rest, so a region that says nothing about the
            # LFO keeps the instrument's LFO - which is what the
            # reference player does by hoisting the shared values and
            # merging zone by zone. Replacing wholesale meant a region
            # carrying one connection threw away everything the
            # instrument had said that the region did not restate.
            eg = dict(ARTIC_DEFAULT)
            if ienv:
                eg.update(ienv)
            if renv:
                eg.update(renv)
            # THE wsmp's GAIN IS NOT IN DECIBELS. It is in 32-bit relative
            # gain units of 1/655360 dB, which is a tenth of a decibel as
            # 16.16 - so the number in the file is a DECIBEL TIMES 655360,
            # and reading it as a decibel attenuated every sample in this
            # bank by whatever the low three bytes happened to hold. On
            # gm.dls that is -4 to -12, i.e. four to twelve decibels of
            # attenuation that the bank never asked for, on every wave.
            # The reference reader divides; so does this now.
            # THE wsmp's GAIN ARRIVES IN DECIBELS ALREADY. The file
            # carries it in relative-gain units of 1/655360 dB, and
            # wsmp_of() divides by that on the way out - that division
            # is the one that matters, and it is the one that used to be
            # wrong in the other direction: this line divided AGAIN, so
            # the region's gain was written as zero for every region of
            # every bank and no sample ever got the trim its own wave
            # asked for.
            #
            # WHAT IS LEFT is to keep tenths of a decibel rather than
            # whole ones. FF8's values are -6.50, -4.80, -4.30, -2.50,
            # -2.20, -2.10, and rounding those to whole decibels moved
            # five waves by half a decibel - half a decibel across a
            # bank is a balance change. A signed byte holds -12.7 to
            # +12.7 dB in tenths, which covers both banks.
            gain_cb = int(round(gain * 10.0))
            regions.append((klo, khi, smap[key], unity,
                            max(-128, min(127, int(round(fine)))),
                            max(-127, min(127, gain_cb)), eg))

    out = bytearray()
    out += MAGIC
    # KSB_FORMAT_VERSION (ksb.h): the header's third H. Written as 1, and
    # the reader checks for that exact value, so a bank this does not
    # recognise is refused rather than read as something else.
    out += struct.pack('<HHHHI', len(inst_rows), len(regions), len(samples),
                       KSB_FORMAT_VERSION, off)
    for prog, first, n in inst_rows:
        out += struct.pack('<BBH', prog, n, first)
    for klo, khi, si, unity, fine, gain, eg in regions:
        # REGION RECORD, 30 BYTES. The first eight are what they always
        # were; the next eight are the volume envelope, as before. What
        # is new is a modulation envelope, a pan and a mod LFO - all of
        # which gm.dls carries on every instrument and which a reader
        # that only kept EG1 dropped without saying so. Every field is
        # 16 bit because the source values fit: measured over gm.dls,
        # pan spans -433..433 in 0.1%, the LFO times span -32768..1419
        # time cents, and no envelope time exceeds 40 s.
        out += struct.pack('<BBHBbbB', klo, khi, si, unity, fine, gain, 0)
        out += struct.pack('<HHHH', eg['attack'], eg['decay'],
                           eg['release'], eg['sustain'])
        out += struct.pack('<HHHH', eg['mattack'], eg['mdecay'],
                           eg['mrelease'], eg['msustain'])
        out += struct.pack('<hhh', eg['pan'], eg['lfofreq'], eg['lfodelay'])
        out += struct.pack('<hhhhh', eg['lpitch'], eg['lgain'],
                           eg['mpitch'], eg['kdecay'], eg['vatk'])
    for o, n, ls, ll, rate, cod, spare in samples:
        out += struct.pack('<IIIIHBB', o, n, ls, ll, rate, cod, spare)
    head = len(out)
    for b in blobs:
        out += b
    open(dst, 'wb').write(bytes(out))

    print('%s -> %s' % (src, dst))
    print('  %d instruments, %d regions, %d samples'
          % (len(inst_rows), len(regions), len(samples)))
    rates = [s[4] for s in samples]
    print('  rates %d .. %d Hz (median %d)'
          % (min(rates), max(rates), int(np.median(rates))))
    print('  %.1f KB header + %.1f KB samples = %.1f KB'
          % (head / 1024.0, off / 1024.0, len(out) / 1024.0))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
