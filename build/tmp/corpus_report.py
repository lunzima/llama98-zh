"""Calibrate and audit all 128 GM programs against a corpus of songs.

For each program, take the notes the corpus actually plays (key,
velocity, the channel's CC7), lay them out as a test phrase, and render
it twice: once with the bank under test and once with the gm reference.
Level, brightness, seam and high-frequency share are compared per
program, and anything far from the reference is printed.
"""
import sys, os, glob, json, struct, subprocess, math, hashlib, collections
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from note_window import note_window  # noqa: E402  (needs HERE first)
PLAYER = os.path.join(HERE, 'kunkun98-play.exe')
CORPUS = os.path.join(HERE, 'build', 'tmp', 'corpus')
TMP = os.path.join(HERE, 'build', 'tmp')
BANK = os.environ.get('CORPUS_REPORT_BANK') or os.path.join(HERE, 'banks', 'picked.ksb')
REF = os.path.join(HERE, 'banks', 'gm.ksb')
# THE TARGET IS FF8, NOT gm. The standing standard for this bank is
# "like ff8, and not too unlike gm" - gm is the bound, not the aim - so
# the tone is read against ff8 as well. A program can sit 2x gm's
# brightness and be exactly right if that is where ff8 sits, and the
# audit could not tell the two apart before this existed.
FF8 = os.path.join(HERE, 'build', 'tmp', 'full_FF8.ksb')

# WHERE THIS RUN'S OUTPUT GOES. Defaults to build/tmp, which is what the
# calibration reads (corpus_rows.json). A demo run points it at its own
# directory so it cannot overwrite the shipped table.
OUTDIR = os.environ.get('CORPUS_REPORT_OUT') or TMP

# OFF BY DEFAULT, AND THE MEASUREMENT IS WHY. Compressing a phrase to the
# ring it is measured at needs the ring of EVERY key the sweep plays, and
# probing three of them is not enough: the kit's corpus keys do not
# include the crash cymbal, so p128 measured a 0.94 s tail, compressed to
# 1.18 s, and its stlmax spread went from 10.9 dB to 48.7 dB - the
# cymbal's tail in the next note's window, which is the exact failure the
# family rule was written for. Probing all 128 keys costs a phrase as
# long as the one being compressed, so there is no version of this that
# pays. CORPUS_REPORT_COMPRESS=1 turns it back on for inspection.
NO_COMPRESS = os.environ.get('CORPUS_REPORT_COMPRESS') != '1'

# STLmax, THE SHORT-TERM-LOUDNESS CEILING. tools/stlmax.py's header states
# what it is NOT (A-weighting standing in for the spectral stage, no 24-band
# filter bank, no 0.23 compression, a level-domain output rather than sone).
# It is used here for the one thing it was built for: the same program at the
# same pitch, read by TIME POOLING instead of by a window average.
import stlmax as STLMAX  # noqa: E402

# THE DURATION AXIS, STATED IN SECONDS OF GATE.
#
# `level_db` is one reading at one note length, and a program's error grows
# with how long the note is held: a pad measured on 1 s notes can sit 1 dB
# under gm and 3 dB under it at 5 s. That much was already covered by
# stretching the phrase x4 and x16.
#
# WHAT WAS NOT COVERED WAS THE OTHER END, and it is the end the corpus
# lives at: 638302 notes have a median length of 0.182 s and 54% of them
# are under 0.2 s, while every reading above was taken on a 0.8 s gate.
# A program can be dead on at 0.8 s and 3.7 dB under at 0.05 s - p110 is,
# and town's fiddle plays it at 62 ms.
#
# So the axis is the GATE, in seconds, over a range that reaches both ends:
# the staccato the corpus actually plays and the sustain it never does.
# `gate_s` turns a gate into the spacing that keeps one note out of the
# next one's window, which reproduces every previously shipped number
# exactly - 0.8/3.2/12.8 s are the old x1/x4/x16 for a melodic program -
# and adds the two short points on top. Nothing measured before is
# redefined; two more points are measured.
DUR_HOLDS_S = (0.05, 0.19, 0.8, 3.2, 12.8)
DUR_CAP_S = 96.0
DUR_MIN_NOTES = 6
# A single note spaced further than this is one shot's worth of audio with
# nothing to average; that gate is skipped and reported as null.
DUR_MAX_DT_S = 24.0

# THE FLOOR BELOW WHICH AN HF SHARE IS SILENCE. Both hf figures are
# fractions of the note's own energy, so 0.001 is one part in a thousand
# above 8 kHz - under any listening level at which the rest of the note
# is audible. The ratio is only reported when the REFERENCE clears it;
# a source with no highs against a reference that has them still reads
# near zero and still flags, which is the direction worth catching.
HF_FLOOR = 1e-3

# HOW LOUD THE REFERENCE HAS TO BE FOR A LEVEL RATIO TO MEAN ANYTHING,
# as a fraction of the program's own median reference level. Below it
# the reference is not playing that key and the ratio is a division by
# a near-zero - gm's shanai reads 14 RMS where it reads 1500, and the
# "deviation" that came out was +34 dB on four ordinary keys. HF has
# had this guard all along (HF_FLOOR); the level never did.
REF_NOTE_FLOOR = 0.05

# THE WIDTH OF A PROGRAM'S INTERNAL LEVEL IMBALANCE, in dB between its
# loudest and quietest key relative to gm on the same keys. A program-
# level trim moves one number and cannot see this. 6 dB is a factor of
# two - loud enough at one end to change what a listener hears, and the
# floor below which the spread is the level knob's own granularity.
# REPORTED, NOT CORRECTED: the gain to fix it is per key and per zone,
# and that is a decision about an instrument's balance rather than a
# measurement error.
BALANCE_SPREAD = 6.0

# THE LEVEL OF A SINGLE-SAMPLE JUMP, as how far the 99.9th percentile
# of the second difference sits above the body of it. REPORTED, NOT A
# GATE: it cannot tell a splice from an attack or from a decaying
# waveform, so nothing is flagged on it. The sample-level reading for
# real splices is `seamrate` at selection. See `spike` in measure.
SEAM_SPIKE = 5.0

# How many of a key's played velocities are measured. Three points span
# the soft, middle and hard readings the corpus uses for one key; more
# than that lengthens every test phrase for a reading the median of
# three already settles.
MAX_VELS = 3

# The one velocity the full key sweep uses. The velocity DIRECTION is
# not what this sweep is for - collapse_velocity_bands removes the
# band structure, and one reading per key zone is all that takes - so a
# single fixed value keeps the note count at 128 per program instead of
# 384. 100 is high enough that a zone whose only band is a soft one
# still sounds, and it is a velocity the corpus also uses.
SWEEP_VEL = 100



def parse(path):
    d = open(path, 'rb').read()
    if d[:4] != b'MThd':
        return None
    fmt, ntr, div = struct.unpack('>HHH', d[8:14])
    if div <= 0:
        return None
    i = 14
    tracks = []
    while i + 8 <= len(d):
        ln = struct.unpack('>I', d[i + 4:i + 8])[0]
        tracks.append(d[i + 8:i + 8 + ln])
        i += 8 + ln

    def rv(b, i):
        v = 0
        while True:
            x = b[i]
            i += 1
            v = (v << 7) | (x & 0x7f)
            if not x & 0x80:
                return v, i

    progs = {}
    cc7 = {}
    notes = []
    for tr in tracks:
        i = 0
        t = 0
        run = None
        while i < len(tr):
            dt, i = rv(tr, i)
            t += dt
            st = tr[i]
            if st == 0xff:
                # META: FF <type> <vlq-length> <data>. The length is a VLQ,
                # so taking one byte of it reads the wrong span as soon as a
                # meta event carries 128 bytes or more.
                ln, j = rv(tr, i + 2)
                i = j + ln
                continue
            if st == 0xf0 or st == 0xf7:
                # SYSEX: F0/F7 <vlq-length> <data>. Stepping one byte walks
                # the parser into the payload, where manufacturer bytes are
                # read as events.
                ln, j = rv(tr, i + 1)
                i = j + ln
                continue
            if st & 0x80:
                run = st
                i += 1
            else:
                st = run
            hi = st & 0xF0
            ch = st & 0x0F
            if hi in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                a = tr[i]
                b = tr[i + 1]
                i += 2
                if hi == 0xB0 and a == 7:
                    cc7[ch] = b
                if hi == 0x90 and b > 0:
                    # CHANNEL 9 IS THE DRUM KIT, and its program byte is
                    # not what the notes mean - a GM file leaves it at 0
                    # and plays percussion anyway. Recording progs.get(ch)
                    # sent every drum hit to program 0, the grand piano,
                    # so the kit was never measured at all and p0's
                    # reading was a piano with a drum kit mixed into it.
                    notes.append((ch, a, b,
                                  128 if ch == 9 else progs.get(ch, 0)))
            elif hi in (0xC0, 0xD0):
                a = tr[i]
                i += 1
                if hi == 0xC0:
                    progs[ch] = a
            else:
                i += 1
    return notes, cc7


def vlq(v):
    o = [v & 0x7f]
    v >>= 7
    while v:
        o.append((v & 0x7f) | 0x80)
        v >>= 7
    return bytes(reversed(o))


def note_dt(prog):
    """How far apart one program's notes are written.

    A DRUM KIT'S NOTES DO NOT FIT IN A SECOND. A cymbal rings for 1.2
    to 1.8 s and is still 13 to 25 dB up when the next note's window
    opens, so at 1.0 s spacing Crash 1's tail filled 188% of the next
    window and the reading described the previous drum. Measured on the
    shipped kit, the open hi-hat was 15 dB off under that spacing and
    3 dB off once the notes were 2.5 s apart. Melodic programs keep
    1.0 s: their release is the character being measured and the notes
    are meant to overlap the way the pieces play them.

    THE SOUND EFFECTS NEED THE SAME ROOM, and did not have it. p125
    rings for seconds; at 1.0 s its key 69 read +18 dB against gm while
    playing it on its own measured -0.1 to -1.3 dB at every velocity -
    the figure was the previous note's tail. Anything from the effects
    family is spaced the way a kit is.

    THIS IS THE CEILING, NOT THE SPACING. Four seconds is what a kit and
    an effects bank are given because neither has been measured; a program
    whose tail dies in half a second renders 3.5 s of silence per note for
    nothing. `spacing()` measures the tail and takes whichever is shorter,
    so this function stays the safe upper bound it always was.
    """
    return 4.0 if (prog == 128 or prog >= 118) else 1.0


# THE RING, AS A NUMBER RATHER THAN A FAMILY GUESS. A note is probed on
# its own and the tail is the time of the LAST 10 ms frame still within
# TAIL_DB of the peak - which is why a brief dip and re-rise cannot end it
# early. -30 dB is one part in thirty of the peak: confined to one window
# it moves that window by 0.1 dB, under every threshold this audit uses.
TAIL_DB = -30.0
TAIL_HOLD_NOTE_S = 0.50


def _tail_of(wav, dt):
    """Seconds from a note's onset until it is TAIL_DB under its own peak.

    MEASURED PER NOTE, NOT PER FILE. The probe renders several keys into
    one phrase, and a tail read from the file's start would be the last
    note's onset plus its ring - a number that grows with the key count.
    Each note is measured inside its own spacing.
    """
    a, _ch, r = load_wav(wav)
    if len(a) == 0 or dt <= 0:
        return 0.0
    step = max(1, int(0.01 * r))
    k = (len(a) // step) * step
    if k == 0:
        return 0.0
    lv = np.sqrt((a[:k].reshape(-1, step) ** 2).mean(axis=1))
    per = max(1, int(round(dt / 0.01)))
    worst = 0.0
    for s in range(0, len(lv), per):
        seg = lv[s:s + per]
        peak = float(seg.max()) if len(seg) else 0.0
        if peak <= 0.0:
            continue
        thr = peak * 10.0 ** (TAIL_DB / 20.0)
        last = 0
        for i in range(len(seg) - 1, -1, -1):
            if seg[i] >= thr:
                last = i
                break
        # A NOTE THAT NEVER RISES HAS NO TAIL worth compressing for;
        # zero leaves the caller on the family's own spacing.
        if last > 0:
            worst = max(worst, (last + 1) * 0.01)
    return worst


def tail_s(prog, keys):
    """The longest ring either bank gives this program over `keys`.

    MORE THAN ONE KEY, AND NOT THE MEDIAN ONE. A kit's median corpus key
    is a kick; the crash cymbal that forced the 4 s spacing in the first
    place sits at another key entirely, and probing the median measured
    the kit at 1.0 s - which would have cut every cymbal in the sweep
    short. The keys probed here are the ends of the range as well as the
    middle, so whatever key rings longest sets the spacing.

    Cached on the two banks' bytes and the key list, so a re-run costs
    nothing and a bank edit only re-probes what it changed.
    """
    keys = sorted(set(k for k in keys if k is not None))
    if not keys:
        return None
    k = 'tail|%s|%s|%d|%s' % (digest(BANK), digest(REF), prog,
                              ','.join(str(x) for x in keys))
    if k in _CACHE:
        return _CACHE[k]
    tag = 'tl%d' % prog
    midi = os.path.join(OUTDIR, 'cr_%s.mid' % tag)
    write_test(midi, prog, [(x, SWEEP_VEL) for x in keys], None,
               dt=note_dt(prog), gate=TAIL_HOLD_NOTE_S)
    got = []
    for bank in (BANK, REF):
        wav = os.path.join(OUTDIR, 'cr_%s_%d.wav' % (tag, len(got)))
        if render(bank, midi, wav):
            got.append(_tail_of(wav, note_dt(prog)))
        try:
            os.remove(wav)
        except OSError:
            pass
    try:
        os.remove(midi)
    except OSError:
        pass
    if not got:
        return None
    _CACHE[k] = max(got)
    return _CACHE[k]


def spacing(prog, notes):
    """The spacing this program's phrase is actually written at.

    NEVER MORE THAN note_dt() AND NEVER LESS THAN ONE SECOND. The first
    bound keeps a program from being given more room than the family rule
    already gives it; the second keeps the notes of a phrase from running
    together, which is what a listener hears as a different instrument.

    What it buys: p125's phrase is 628 s of which 311 s is sound - nearly
    three seconds of silence after every note, for a tail nobody measured.
    """
    base = note_dt(prog)
    if NO_COMPRESS or not notes:
        return base
    ks = sorted(set(n for n, _v in notes))
    probes = sorted({ks[0], ks[len(ks) // 2], ks[-1]})
    t = tail_s(prog, probes)
    if t is None:
        return base
    # THE GATE IS 0.8 OF THE SPACING, so the spacing cannot go below the
    # tail over 0.8 without releasing the note while it is still ringing.
    # p125's tail is 2.32 s; at the naive tail + 0.2 the gate came out
    # 2.0 s, the note was cut before it died, and the reading moved 0.15
    # dB. Tying the spacing to the gate instead leaves the release where
    # the sound actually ends.
    return min(base, max(1.0, t / 0.8))


def gate_s(dt, gate=None):
    """How long a note is held, given how far apart the notes are written.

    ONE RULE, IN ONE PLACE. This expression was spelled out three times -
    in write_test's note-off, in note_windows and in note_windows_long -
    and the duration axis could not be moved without moving all three.

    The default is the stretch rule: hold 80% of the spacing, floored at
    0.6 s so a window is long enough to average. A caller that is after a
    SPECIFIC note length - the corpus plays 0.182 s, not 0.8 s - passes
    the gate it wants and the floor gets out of the way.

    `dt = max(gate / 0.8, note_dt(prog))` is the inverse, and it is what
    the duration pass uses: it reproduces every gate this file used to
    reach through a stretch factor, and reaches the short ones the floor
    used to clamp away.
    """
    return max(0.8 * dt, 0.6) if gate is None else gate


def write_test(path, prog, notes, cc7, dt=None, gate=None):
    div = 96
    # A KIT PLAYS ON THE DRUM CHANNEL. Channel 0 with a program of 128
    # is not a thing - the byte does not fit and no zone resolves. The
    # kit keeps channel 9 and needs no program change at all.
    drum = (prog == 128)
    ch = 9 if drum else 0
    # `dt` is the note spacing and `gate` how long a note is held; gate_s
    # ties them together when the caller does not name a gate, and the
    # duration pass names one because it is after a note length.
    if dt is None:
        dt = note_dt(prog)
    gate = gate_s(dt, gate)
    ev = [] if drum else [(0, bytes([0xC0, prog]))]
    if cc7 is not None:
        ev.append((0, bytes([0xB0 | ch, 7, cc7])))
    for k, (key, vel) in enumerate(notes):
        t = 72 + int(round(k * dt * 192))   # 192 ticks = 1.0 s at 120 bpm
        ev.append((t, bytes([0x90 | ch, key, vel])))
        ev.append((t + int(round(gate * 192)), bytes([0x80 | ch, key, 0])))
    ev.sort(key=lambda x: x[0])
    body = b''
    last = 0
    for tt, data in ev:
        body += vlq(tt - last) + data
        last = tt
    body += vlq(0) + bytes([0xFF, 0x2F, 0])
    tempo = bytes([0, 0xFF, 0x51, 3, 7, 0xA1, 0x20, 0, 0xFF, 0x2F, 0])
    open(path, 'wb').write(b'MThd' + struct.pack('>IHHH', 6, 1, 2, div) +
                           b'MTrk' + struct.pack('>I', len(tempo)) + tempo +
                           b'MTrk' + struct.pack('>I', len(body)) + body)


def render(bank, midi, wav):
    r = subprocess.run([PLAYER, '--bank', bank, '--multi', '--wav', wav, midi],
                       capture_output=True)
    return r.returncode == 0 and os.path.exists(wav)


# A RENDER AND ITS READING ARE A FUNCTION OF TWO FILES' BYTES.
#
# The reference banks never change, so on every re-run after a bank edit
# the gm and ff8 halves of this audit were re-rendered, re-read and
# re-measured to the same answer - six of the eleven renders and six of
# the eleven measurements per program, including the two whole-file STLmax
# passes. Nothing about them is ours to recompute.
#
# The key is the CONTENT digest of the bank and the phrase, not a path or
# a timestamp: a file that was touched and not written keeps its digest,
# and a bank that changed cannot hit an old entry. THE KNOBS THAT CHANGE A
# READING WITHOUT CHANGING EITHER FILE RIDE IN THE VERSION - a content key
# cannot see them, and a cache built at one crop margin answering for
# another is the stale-artifact failure with no symptom.
CACHE_VERSION = (2, STLMAX.CROP_MARGIN_S, STLMAX.FUSE_TRANSFORMS,
                 REF_NOTE_FLOOR, HF_FLOOR, MAX_VELS)
_CACHE = {}
_DIGEST = {}


def digest(path):
    """sha1 of a file's bytes, memoised on (path, mtime_ns, size)."""
    try:
        st = os.stat(path)
    except OSError:
        return 'missing'
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    got = _DIGEST.get(key)
    if got is not None:
        return got
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    _DIGEST[key] = h.hexdigest()
    return _DIGEST[key]


def cache_load(path):
    global _CACHE
    try:
        raw = json.load(open(path, encoding='utf-8'))
    except (OSError, ValueError):
        return 0
    if raw.get('v') != list(CACHE_VERSION):
        return 0
    _CACHE = raw.get('e') or {}
    return len(_CACHE)


def cache_save(path):
    # THROUGH A TEMP FILE. A run that dies mid-write would otherwise
    # leave a truncated cache behind, and a cache that cannot be parsed
    # is indistinguishable from an empty one - the next run would redo
    # everything and nobody would know why.
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'v': CACHE_VERSION, 'e': _CACHE}, f)
    os.replace(tmp, path)


# A RENDER IS READ TWICE AND THE READ IS THE SLOW PART. note_windows()
# and measure() both take a path and both load the whole thing; on a kit's
# key sweep that is 221 MB decoded twice. Two entries is exactly what the
# access pattern needs - reference, ours, reference again, ff8 - and keeps
# at most two mono float64 arrays alive.
_WAV_MEMO = collections.OrderedDict()
_WAV_MEMO_MAX = 2


def load_wav(p):
    try:
        st = os.stat(p)
        key = (os.path.abspath(p), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _WAV_MEMO:
        _WAV_MEMO.move_to_end(key)
        return _WAV_MEMO[key]
    got = _load_wav(p)
    if key is not None:
        _WAV_MEMO[key] = got
        while len(_WAV_MEMO) > _WAV_MEMO_MAX:
            _WAV_MEMO.popitem(last=False)
    return got


def _load_wav(p):
    """The mono mixdown, already done, in float64, read in chunks.

    IT USED TO READ THE WHOLE FILE AT ONCE and let each caller mix it
    down. A full key sweep puts 384 notes of a kit's 4 s spacing in one
    file - 271 MB of interleaved stereo on disk, 1.08 GB as float64,
    and another copy for the mixdown - which is past what this machine
    will hand out. Mixing as the frames arrive keeps one mono float64
    array: 542 MB for the same signal.

    FLOAT64 IS KEPT DELIBERATELY. Going to float32 was tried and it
    moves the numbers - not the ratios (the two loaders return
    identical arrays bit for bit) but note_window's threshold
    crossings, which land one 10 ms grid step apart and shift the
    measurement window by that much. A memory fix that moves the
    reading is not a memory fix.

    `ch` comes back as 1 so the callers' `if ch > 1` mixdown is skipped
    rather than made wrong - the contract is "a is the mono signal", not
    "a is interleaved".
    """
    import wave as W
    w = W.open(p, 'rb')
    n, ch, r = w.getnframes(), w.getnchannels(), w.getframerate()
    a = np.empty(n, dtype=np.float64)
    pos = 0
    step = max(1, (1 << 22) // max(1, ch * 2))     # ~4 MiB of frames
    while pos < n:
        k = min(step, n - pos)
        raw = np.frombuffer(w.readframes(k), dtype='<i2')
        if ch > 1:
            a[pos:pos + k] = raw.reshape(-1, ch).mean(axis=1)
        else:
            a[pos:pos + k] = raw
        pos += k
    w.close()
    return a, 1, r


def note_windows(wav, nnotes, dt=1.0, gate=None):
    """Where each note actually sounds, read off the REFERENCE render.

    A FIXED SLICE IS NOT A COMMON WINDOW. 0.02-0.45 s is the same
    number of seconds for both banks but not the same part of the note.
    A source played 29 semitones below the pitch it was recorded at runs
    its attack 5.3 times slower, so at 0.45 s it is still rising while
    gm is already in its body; the ratio then compares two moments and
    reports the difference as a level error. On the shipped bank p64's
    keys 12 to 50 read 7 to 21 dB under gm that way, and the region
    covering them got one gain set at the key where the slice happened
    to agree.

    THE SPAN COMES FROM gm, NOT FROM THE BANK UNDER TEST. A window
    derived from the measured bank is self-referential - a slow-decaying
    bank reads longer and higher, which is why _note_rms in ksb_pick
    uses a fixed slice and was right to. gm is the reference for every
    other reading here, so its span is the one both banks are measured
    over.

    The span is capped at the note's own hold so a long tail cannot
    reach into the next note of the phrase.
    """
    a, ch, r = load_wav(wav)
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    hold = gate_s(dt, gate)
    out = []
    for k in range(nnotes):
        t0 = k * dt
        x = a[int(t0 * r):int((t0 + min(hold, 2.5)) * r)]
        w = note_window(x, r)
        if w is None:
            out.append(None)
            continue
        lo = w[0] / float(r)
        hi = min(w[1] / float(r), hold)
        if hi - lo < 0.03:
            hi = min(lo + 0.03, hold)
        out.append((lo, hi) if hi > lo else None)
    return out


def measure(wav, nnotes, dt=1.0, wins=None, with_stlmax=True):
    a, ch, r = load_wav(wav)
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    # THE STLmax PASS RUNS ON THE WHOLE WAVEFORM, ONCE. The smoother is a
    # recursion from the file's start, so it must see everything before the
    # window; handing analyse_note() the `seg` slice would restart it at the
    # window edge and read a different number (silently, because lo is 0 on
    # every first note). analyse_windows() is the same series over the whole
    # array with one FFT pass instead of one per note.
    abs_wins = []
    for k in range(nnotes):
        t0 = k * dt
        if wins is not None and k < len(wins) and wins[k]:
            lo, hi = wins[k]
        else:
            lo, hi = 0.02, 0.45
        abs_wins.append((t0 + lo, t0 + hi))
    stl = (STLMAX.analyse_windows(a, r, abs_wins)
           if with_stlmax and abs_wins else None)
    out = []
    for k in range(nnotes):
        # THE PLAYER TRIMS THE LEAD-IN, so the first note is at 0 s
        # in the render, not at its file tick. A 0.375 s offset put the
        # window between the notes for anything that decays fast
        # (Tinkle Bell measured as silence).
        t0 = k * dt
        # THE SAME WINDOW FOR BOTH BANKS - and it is the reference's
        # own span, so the two readings cover the same musical part
        # instead of the same number of seconds. See note_windows.
        if wins is not None and k < len(wins) and wins[k]:
            lo, hi = wins[k]
        else:
            lo, hi = 0.02, 0.45
        seg = a[int((t0 + lo) * r):int((t0 + hi) * r)]
        if len(seg) < 256 or np.abs(seg).max() < 20:
            out.append(None)
            continue
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2 + 1e-9
        fr = np.fft.rfftfreq(len(seg), 1.0 / r)
        d2 = np.abs(seg[2:] - 2 * seg[1:-1] + seg[:-2])
        peak = float(np.abs(seg).max())
        # HOW LONG THE NOTE RINGS, at -20 dB from its own peak. It is
        # REPORTED, not used to choose the figure any more: with the
        # window running over the note's own span the energy is already
        # the level of the part that sounds, and judging a short note by
        # its peak on top of that reads a transient difference as a
        # level error.
        fl = max(8, int(0.005 * r))
        nf = max(1, len(seg) // fl)
        env = np.sqrt((seg[:nf * fl].reshape(nf, fl) ** 2).mean(axis=1)) + 1e-9
        thr = 0.1 * peak
        dur = 0.005 * (1 + int(np.nonzero(env > thr)[0][-1])) if (env > thr).any() else 0.0
        # A SPLICE IS ONE SAMPLE OUT OF LINE AMONG SMOOTH NEIGHBOURS,
        # NOT A SHARP WAVEFORM. The old reading was the 99.9th
        # percentile of the second difference over the peak, which
        # measures how edgy the note is: p101 scored 0.402 with its only
        # spike at the onset, p111 0.321 sustained across the whole note
        # because a shannai is buzzy, and neither of those is a splice.
        # The tail-to-body ratio is what a discontinuity looks like - it
        # is high when one sample breaks a smooth run and low when the
        # whole waveform is consistently steep. The first 30 ms is left
        # out, because an attack is a genuine slope change and every
        # note has one.
        body = d2[int(0.03 * r):] if len(d2) > int(0.03 * r) else d2
        body_med = float(np.percentile(body, 50)) if len(body) else 0.0
        spike = (float(np.percentile(body, 99.9)) / body_med
                 if body_med > 1e-9 else 0.0)
        out.append({'rms': float(np.sqrt((seg * seg).mean())),
                    # THE SHORT-TERM-LOUDNESS PEAK over the same window, in
                    # the same dB scale as `rms` (both are level differences
                    # against the reference, so the dBFS zero cancels).
                    'stlmax': (stl[k]['stlmax_db'] if stl is not None
                               else None),
                    'peak': peak,
                    'dur': dur,
                    'cen': float((fr * sp).sum() / sp.sum()),
                    'hf': float(sp[fr > 8000].sum() / sp.sum()),
                    'seam': float(np.percentile(d2, 99.9) /
                                  max(1, np.abs(seg).max())),
                    'spike': spike})
    return out


def med(vals, key):
    v = sorted(x[key] for x in vals if x)
    return v[len(v) // 2] if v else None


def note_windows_long(wav, nnotes, dt, gate=None):
    """Where each note sounds, with BOTH 2 s caps off.

    note_windows() slices the reference at min(hold, 2.5) and note_window
    itself caps its span at CAP_S = 2.0. Both exist so a long tail cannot
    reach into the next note of a phrase. The duration pass has the opposite
    job: its gates run from 50 ms to 12.8 s and the point of the pass is
    that gate, which a 2 s cap would flatten at one end and inflate at the
    other. The spacing is set from the gate so one note sounds at a time
    and there is nothing to reach into.
    """
    a, ch, r = load_wav(wav)
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    out = []
    for k in range(nnotes):
        t0 = k * dt
        hold = gate_s(dt, gate)
        x = a[int(t0 * r):int((t0 + hold) * r)]
        w = note_window(x, r, cap_s=None)
        if w is None:
            out.append(None)
            continue
        lo = w[0] / float(r)
        hi = min(w[1] / float(r), hold)
        if hi - lo < 0.03:
            hi = min(lo + 0.03, hold)
        out.append((lo, hi) if hi > lo else None)
    return out


def duration_pass(prog, notes, gate):
    """Render the same notes held `gate` seconds and read the level.

    Returns (median dev, worst key, worst dev, notes used) or None. The note
    list is the same one the shipped reading uses; when the phrase at this
    gate would run past DUR_CAP_S the list is subsampled with an even stride,
    so the keys keep their coverage instead of the phrase keeping its head.

    The gate is the whole point and the spacing is derived from it: far
    enough apart that one note is out of the next one's window, and never
    closer than the program's own spacing (a kit rings for 1.8 s and its
    notes need the room regardless of how short they are held).
    """
    dt = max(gate / 0.8, spacing(prog, notes))
    if dt > DUR_MAX_DT_S:
        return None
    maxn = max(1, int(DUR_CAP_S // dt))
    if maxn >= len(notes):
        sel = list(notes)
    else:
        step = len(notes) / float(maxn)
        sel = [notes[int(i * step)] for i in range(maxn)]
    tag = 'd%d_%g' % (prog, gate)
    midi = os.path.join(OUTDIR, 'cr_%s.mid' % tag)
    wb = os.path.join(OUTDIR, 'cr_%s_cur.wav' % tag)
    wr = os.path.join(OUTDIR, 'cr_%s_ref.wav' % tag)
    write_test(midi, prog, sel, None, dt=dt, gate=gate)
    # THE PHRASE IS WRITTEN FIRST BECAUSE ITS BYTES ARE THE KEY. A gate whose
    # bank and phrase have not changed cannot produce a different reading,
    # and the reference half of every duration pass is one of the six renders
    # per program that never changes.
    key = 'dl|%s|%s|%s|%g' % (digest(BANK), digest(REF), digest(midi), gate)
    if key in _CACHE:
        try:
            os.remove(midi)
        except OSError:
            pass
        v = _CACHE[key]
        return tuple(v) if v else None
    ok = render(BANK, midi, wb) and render(REF, midi, wr)
    if not ok:
        return None
    wins = note_windows_long(wr, len(sel), dt, gate)
    mb = measure(wb, len(sel), dt, wins, with_stlmax=False)
    mr = measure(wr, len(sel), dt, wins, with_stlmax=False)
    for f in (wb, wr, midi):
        try:
            os.remove(f)
        except OSError:
            pass
    rlev = [y['rms'] for y in mr if y]
    rmed = sorted(rlev)[len(rlev) // 2] if rlev else None
    floor = REF_NOTE_FLOOR * rmed if rmed else None
    devs = []
    kb = {}
    for i, (x, y) in enumerate(zip(mb, mr)):
        if not x or not y:
            continue
        a, b = x['rms'], y['rms']
        if a > 0 and b > 0 and (floor is None or b >= floor):
            d = 20.0 * math.log10(a / b)
            devs.append(d)
            kb.setdefault(sel[i][0], []).append(d)
    if not devs:
        return None
    md = sorted(devs)[len(devs) // 2]
    kv = sorted((k, sorted(v)[len(v) // 2]) for k, v in kb.items())
    wk, wd = max(kv, key=lambda t: abs(t[1])) if kv else (None, None)
    _CACHE[key] = [md, wk, wd, len(devs)]
    return (md, wk, wd, len(devs))


def main():
    ncache = cache_load(os.path.join(OUTDIR, 'measure_cache.json'))
    if ncache:
        print('measurement cache: %d entr%s reused where the bytes match'
              % (ncache, 'y' if ncache == 1 else 'ies'), flush=True)
    pieces = sorted(set(glob.glob(os.path.join(CORPUS, '**', '*.mid'), recursive=True) +
                       glob.glob(os.path.join(CORPUS, '**', '*.MID'), recursive=True)))
    # A DEMO RUN MAY TAKE A SUBSET. Per-program rows are independent (each
    # one is rendered and measured on its own), so a subset's `level_db` is
    # the same number the full run would write for that program. Unset runs
    # every program, as before.
    only = os.environ.get('CORPUS_REPORT_ONLY')
    only = set(int(x) for x in only.split(',')) if only else None
    usage = {p: [] for p in range(129)}
    cc_by_prog = {}
    for path in pieces:
        r = parse(path)
        if not r:
            continue
        notes, cc7 = r
        per = {}
        for ch, key, vel, prog in notes:
            if prog > 128:
                continue
            per.setdefault(prog, []).append((key, vel, ch))
        for prog, evs in per.items():
            # THE SAME SAMPLE AS tone_scan: every note the corpus plays
            # for this program, then one spread of 24 by key. Taking
            # three per piece made the two tools disagree by up to 0.75
            # octaves on the same rendering of the same source.
            usage[prog].extend(evs)
    used = sum(1 for p in usage if usage[p])
    print('corpus: %d pieces, %d/128 programs used' % (len(pieces), used))
    rows = {}
    for prog in range(129):
        if only is not None and prog not in only:
            continue
        evs = sorted(usage[prog], key=lambda e: e[0])
        if not evs:
            continue
        # EVERY KEY THE CORPUS PLAYS, and no key the banks choose.
        #
        # THE GRID MAY NOT COME FROM THE BANK UNDER TEST. It used to be
        # "the ends of the played range plus every boundary of either
        # bank inside it" - and the second half made the reading
        # self-referential in the same way the note WINDOW was before it
        # was moved onto gm. Splitting a region adds its new boundaries
        # to the grid, so the run after a split measures DIFFERENT KEYS
        # and reports a different worst key. p123's outlier sat at key
        # 95; one split pass later the readings had moved to keys 89 and
        # 101 - the cuts that pass had just made - and the correction
        # raised a whole span whose unmeasured neighbours turned out to
        # be fine. A correction cannot be checked against a grid the
        # correction itself reshapes, which is why this loop never
        # converged before. The corpus is the only thing here that
        # neither bank can move.
        #
        # PLUS EVERY KEY THE CORPUS DOES NOT PLAY. The corpus reaches
        # 24 to 78 of a program's keys, and the gaps held two defects no
        # corpus-driven number could ever have shown: p92's keys 120-127
        # reading 2.7 to 12.6 dB under gm because the gain ceiling
        # refused the correction, and p75's keys 118-127 reading +14.5
        # dB at velocity 32 and -5 dB at velocity 110 because its
        # velocity bands carried gains 40 dB apart. Both are audible on
        # ordinary music that simply is not these three pieces. The
        # sweep is keys 0-127 at one fixed velocity - the velocity
        # direction is handled by collapse_velocity_bands, and one
        # reading per key zone is what that needs.
        #
        # EVERY KEY AT THE LOUDNESSES IT IS PLAYED. Keeping the first
        # event per key made the reading depend on the order the corpus
        # files were scanned in: a key played twenty times at different
        # velocities was measured once, at whichever velocity the scan
        # happened to reach first. Velocity is half of what a region is,
        # and one sample of it is not a measurement - it is also why a
        # correction that levelled one reading moved the error to
        # another key instead of removing it.
        bykey = {}
        for k, v, ch in evs:
            bykey.setdefault(k, set()).add(v)
        notes = []
        for k in sorted(bykey):
            vs = sorted(bykey[k])
            if len(vs) > MAX_VELS:
                pick = [int(round(i * (len(vs) - 1) / float(MAX_VELS - 1)))
                        for i in range(MAX_VELS)]
                vs = sorted({vs[i] for i in pick})
            for v in vs:
                notes.append((k, v))
        corpus_pairs = set(notes)
        for k in range(128):
            p = (k, SWEEP_VEL)
            if p not in corpus_pairs:
                notes.append(p)
        notes.sort()
        synth = [p for p in notes if p not in corpus_pairs]
        # NO CHANNEL VOLUME. CC7 is a mix parameter and both banks
        # receive the same curve, so it cancels; worse, some corpus
        # channels carry CC7=0 and the test phrase fell silent.
        cc = None
        midi = os.path.join(OUTDIR, 'cr_%d.mid' % prog)
        wb = os.path.join(OUTDIR, 'cr_%d_cur.wav' % prog)
        wr = os.path.join(OUTDIR, 'cr_%d_ref.wav' % prog)
        wf = os.path.join(OUTDIR, 'cr_%d_ff8.wav' % prog)
        dt = spacing(prog, notes)
        if dt < note_dt(prog) - 1e-9:
            print('  p%d: notes %.2f s apart, the family rule says %.2f'
                  % (prog, dt, note_dt(prog)), flush=True)
        write_test(midi, prog, notes, cc, dt=dt)
        mdig = digest(midi)
        # THE WINDOWS COME FROM THE REFERENCE, so they are a function of the
        # reference bank and the phrase and of nothing else - which is why
        # this key does not mention our bank.
        wkey = 'w|%s|%s' % (digest(REF), mdig)
        wins = _CACHE.get(wkey)
        if wins is None:
            if not render(REF, midi, wr):
                rows[prog] = {'error': 'render'}
                continue
            wins = note_windows(wr, len(notes), dt)
            _CACHE[wkey] = wins

        def get(bank, wav, with_stlmax):
            """Measure one bank against this phrase, from the cache when the
            bytes are the ones the cached reading came from."""
            k = 'm|%s|%s|%d' % (digest(bank), mdig, 1 if with_stlmax else 0)
            if k in _CACHE:
                return _CACHE[k]
            if not os.path.exists(wav) and not render(bank, midi, wav):
                return None
            v = measure(wav, len(notes), dt, wins, with_stlmax=with_stlmax)
            _CACHE[k] = v
            return v

        mb = get(BANK, wb, True)
        mr = get(REF, wr, True)
        if mb is None or mr is None:
            rows[prog] = {'error': 'render'}
            continue
        # THE FF8 RENDER IS READ FOR TONE AND HF ONLY, so it does not need the
        # STLmax pass: stlmax_db is calibrated against gm, and the ff8 pass is
        # a third set of whole-file FFTs per program for a field nothing reads.
        mf = get(FF8, wf, False) or []
        for f in (wb, wr, wf, midi):
            try:
                os.remove(f)
            except OSError:
                pass
        # THE REFERENCE DECIDES THE FIGURE. A note short enough to be
        # judged by its peak is judged by its peak for both banks; the
        # other way round the two banks could be measured by different
        # quantities on the same note, which is not a comparison.
        vb, vr = [], []
        devs = []
        # STLmax'S RATIOS, ON THE SAME NOTES AND THE SAME GATE. The gate
        # stays on the RMS reference level on purpose: if the STLmax median
        # were taken over a different note set, a difference between
        # stlmax_db and level_db could come from the note set instead of the
        # pooling. See the wiring plan in tools/stlmax.py.
        sdevs = []
        # THE SAME READINGS WITH THE VELOCITY KEPT. `devs` is per key
        # because the spread and the worst key are per key; a correction
        # also has to know WHICH REGION sounded, and that needs the
        # velocity the note was played at - two layers over one key
        # answer to different regions at different velocities.
        kdev = []
        pkd = []
        # THE REFERENCE HAS TO ACTUALLY PLAY THE NOTE. A ratio against a
        # near-silent denominator is not a level error - it is a
        # statement about the reference. gm's shanai reads 14 to 37 RMS
        # at keys 95 to 107 where it reads 1500 at key 90, and our notes
        # there came out "+34 dB", which made four ordinary keys the
        # worst defects in the bank and p11's internal spread 34 dB.
        # Same guard HF has had all along (HF_FLOOR): the reading is
        # withheld, not reported as an error. The floor is a twentieth
        # of the program's own median reference level - far below any
        # note the instrument really plays, far above the ones it does
        # not.
        _rlev = [y['rms'] for y in mr if y]
        _rmed = sorted(_rlev)[len(_rlev) // 2] if _rlev else None
        ref_floor = REF_NOTE_FLOOR * _rmed if _rmed else None
        for i, (x, y) in enumerate(zip(mb, mr)):
            if not x or not y:
                continue
            # THE WINDOW IS THE NOTE'S OWN SPAN, SO THE ENERGY IS THE
            # LEVEL. This used to judge a short note by its peak, and
            # it had to: the window was a fixed 0.43 s slice, a 40 ms
            # attack inside it was mostly silence, and only the peak
            # survived that. Now that note_windows hands both banks the
            # reference's own span, the RMS is taken over the part that
            # sounds and nothing dilutes it - so the peak switch became
            # a way to report a transient difference as a level error.
            # p1's top key read -13.3 dB that way against an RMS 3.0 dB
            # under gm, and 13 other programs were flagged the same.
            # The peak is still measured, and still bounds the ceiling;
            # it is no longer how the level is read.
            a, b = x['rms'], y['rms']
            vb.append(a)
            vr.append(b)
            if a > 0 and b > 0 and (ref_floor is None or b >= ref_floor):
                dev = 20.0 * math.log10(a / b)
                devs.append((notes[i][0], dev))
                kdev.append([notes[i][0], notes[i][1], dev])
            # THE PEAK AGAINST GM'S PEAK, on every note whatever the
            # duration convention says. The level trim brings a
            # program's median to gm's, and for a transient-heavy
            # instrument that lifts the peak past full scale: Tinkle
            # Bell measured +8.5 dB at the peak while its median sat in
            # band. The trim has to answer to the peak as well - see
            # the peak fields below.
            # THE SAME REF_FLOOR THE LEVEL AND STLMAX HAVE. p11 reads
            # gm's peak at 92 to 195 counts on keys 96 to 124 where it
            # reads 2400 to 4074 on keys 0 to 68, so the ratio there is
            # a statement about gm's silence and not about our peak. It
            # was then summarised at the 90th percentile as +21.4 dB and
            # the ceiling priced the whole program 2.5 dB down.
            if x.get('peak') and y.get('peak') and \
                    (ref_floor is None or b >= ref_floor):
                pkd.append(20.0 * math.log10(x['peak'] / y['peak']))
            sa, sb = x.get('stlmax'), y.get('stlmax')
            if (sa is not None and sb is not None and sa > -300.0
                    and sb > -300.0
                    and (ref_floor is None or b >= ref_floor)):
                sdevs.append((notes[i][0], sa - sb))
        # ONE CALIBER FOR THE LEVEL. This used to be the ratio of the
        # two banks' own medians, which is not the median of the
        # per-note ratios and disagrees with it whenever the two level
        # distributions differ in shape: p21 read -4.72 one way and
        # -8.68 the other, and the level pass was correcting the first
        # while balance_regions was answering the second - two passes
        # fighting over the same gain. The per-note ratio is the
        # comparison (same note, same window), so the level is the
        # median of those.
        dv = sorted(d for _, d in devs)
        lb = sorted(vb)[len(vb) // 2] if vb else None
        lr = sorted(vr)[len(vr) // 2] if vr else None
        # THE STLmax MEDIAN AND ITS PER-KEY SPREAD, mirroring level_db and
        # level_spread. Both are differences of two dB values, so a plain
        # difference - no ratio.
        sd = sorted(d for _, d in sdevs)
        stlmax_db = sd[len(sd) // 2] if sd else None
        stlmax_spread = None
        if len(sdevs) >= 2:
            skb = {}
            for _k, _d in sdevs:
                skb.setdefault(_k, []).append(_d)
            skv = sorted((k, sorted(v)[len(v) // 2]) for k, v in skb.items())
            if len(skv) >= 2:
                stlmax_spread = (max(d for _, d in skv)
                                 - min(d for _, d in skv))
        # THE INTERNAL BALANCE, one number per key instead of one per
        # program. A program can sit at gm's median level and still be
        # loud at one end and quiet at the other - a kit whose hi-hats
        # play at a fifth of its crashes, a stretched piano whose top
        # keys run 15 dB hot. A program-level trim cannot see it
        # because it has one number to move. This REPORTS the spread;
        # it does not correct it.
        spread = None
        spread_iqr = None
        worst_key = None
        worst_dev = None
        # ONE READING PER KEY, taken as the median of the loudnesses it
        # was played at. The spread is a question about keys, so a key
        # measured three times contributes one number - otherwise the
        # widest difference inside one key would be reported as a
        # difference between keys.
        kb = {}
        for _k, _v, _d in kdev:
            kb.setdefault(_k, []).append(_d)
        kv = sorted((k, sorted(v)[len(v) // 2]) for k, v in kb.items())
        if len(kv) >= 2:
            spread = max(d for _, d in kv) - min(d for _, d in kv)
            worst_key, worst_dev = max(kv, key=lambda t: abs(t[1]))
            # MAX MINUS MIN IS ONE KEY'S OPINION ON A SHORT LIST. p125
            # played four notes; three sat within 1.2 dB and the fourth
            # was 18.7 dB hot, so the spread read 19.9 and called the
            # instrument unstable. The interquartile range says what the
            # bulk of the keys do, which is the question the number is
            # asked; both are kept because a lone rogue key is a defect
            # too, just a different one.
            _v = sorted(d for _, d in kv)
            spread_iqr = _v[(3 * len(_v)) // 4] - _v[len(_v) // 4]
            # ONE KEY AGAINST ITS OWN PROGRAM. The full spread is
            # max minus min, and on a flat instrument that is one key's
            # opinion: p11 sits within 0.3 dB of gm on twenty keys and
            # 3.5 dB off on one, so the spread read 6.2 and the program
            # was called uneven. Two questions were under one label -
            # "are its keys at odds with each other" (the bulk, which is
            # the interquartile range) and "is one key out of line" (its
            # distance from the program's own median). Now apart.
            _m = _v[len(_v) // 2]
            key_rogue = max(abs(d - _m) for _, d in kv)
        # HOW LONG THE NOTE RINGS, as the median of the PER-NOTE ratios
        # against gm - the level's caliber, not the ratio of two medians.
        # p75's notes come out at 0.40-0.52 of gm's, and the reason is
        # the same shape as its gain sawtooth: the release is written per
        # velocity band (80, 100, ... 240) where gm carries one value
        # (20) for the whole key zone. Gains got collapsed to one per
        # zone; the envelope did not.
        _dr = sorted(x['dur'] / y['dur'] for x, y in zip(mb, mr)
                     if x and y and y['dur'] > 0)
        dur_ratio = _dr[len(_dr) // 2] if _dr else None
        cb, cr = med(mb, 'cen'), med(mr, 'cen')
        cf = med(mf, 'cen')
        sb, sr = med(mb, 'seam'), med(mr, 'seam')
        hb, hr = med(mb, 'hf'), med(mr, 'hf')
        hf8 = med(mf, 'hf')
        # A SHARE IS ALREADY A FRACTION, so 0 means nothing above 8 kHz,
        # not a small amount. Dividing two of those gave ratios of 3129x
        # and 1060x on programs whose gm reference carried no energy up
        # there at all - 64 of 128 flagged, p90 at 34.9x. That is the
        # denominator failing, not the bank. A share under one part in
        # a thousand is inaudible, and a ratio of two inaudible shares
        # is not a comparison: the reading is withheld instead of being
        # invented. Both raw shares are kept so the withheld case can
        # still be read.
        hf = None
        if hb is not None and hr is not None and hr >= HF_FLOOR:
            hf = hb / hr
        rows[prog] = {
            'notes': len(notes),
            'keys': len(kb),
            'level_db': (dv[len(dv) // 2] if dv else None),
            'level_spread': spread,
            'level_iqr': spread_iqr, 'key_rogue': key_rogue,
            # THE PEAK AGAINST GM'S PEAK, summarised twice because the
            # two answer different questions. The median is where a
            # typical note sits; the 90th percentile is what a hot one
            # does to a sum of voices. THE MAX WAS TRIED AND WAS WRONG
            # FOR THE CEILING: one note at +8 dB dragged its whole
            # program down by 8, and the bank's median level fell to
            # -2.1 dB against gm. The ceiling reads the 90th, which caps
            # what a typical note can do without letting one of them
            # price the rest of the instrument out.
            'peak_db': (sorted(pkd)[len(pkd) // 2] if pkd else None),
            'peak_p90_db': (sorted(pkd)[int(len(pkd) * 0.9)] if pkd
                            else None),
            # HOW MANY NOTES THE TWO PEAK FIGURES REST ON. A percentile
            # of four samples is the largest of them: p125 was measured
            # from four notes, one of which ran 19 dB over the others,
            # and the 90th percentile priced the whole program 15 dB
            # down - the max-based ceiling's failure in a new coat. The
            # count travels with the reading so a consumer can refuse to
            # act on a distribution that is not one.
            'peak_notes': len(pkd),
            'worst_key': worst_key,
            'worst_dev': worst_dev,
            # [key, velocity, dB against gm] for every note that made
            # it into the spread above. This is the reading a balance
            # pass has to consume, so that the thing being corrected is
            # the thing being reported.
            'key_devs': kdev,
            # WHICH KEYS THE CORPUS ACTUALLY PLAYS. The sweep covers
            # every key so the calibration has no blind spot, and that
            # changes what the summary numbers mean: "how spread out is
            # this instrument" over the whole keyboard rather than over
            # the keys these three pieces touch. Both are reported, and
            # this is how the second one is cut back to the first.
            'corpus_keys': sorted(bykey),
            'cen_ratio': cb / cr if cb and cr else None,
            # AGAINST FF8, WHICH IS WHAT THE BANK IS AIMING AT. gm is
            # the bound - "not too unlike" - and ff8 is the target, so
            # a program can sit far from gm's brightness and be right.
            'cen': cb, 'cen_ref': cr,
            'cen_ratio_ff8': cb / cf if cb and cf else None,
            'cen_ff8': cf,
            'hf_ratio_ff8': (hb / hf8 if hb is not None and hf8
                             and hf8 >= HF_FLOOR else None),
            'hf_ff8': hf8,
            'hf_ratio': hf, 'hf': hb, 'hf_ref': hr,
            'hf_withheld': (hb is not None and hr is not None
                            and hr < HF_FLOOR),
            'seam': sb, 'seam_ref': sr,
            'dur_ratio': dur_ratio,
            'spike': med(mb, 'spike'), 'spike_ref': med(mr, 'spike'),
            # --- ADDITIVE, since the duration pass and STLmax were added.
            # `level_db` above keeps its exact meaning and value: it is what
            # ksb_pick's level trim reads (tools/ksb_pick.py:4247 and :4692).
            'stlmax_db': stlmax_db,
            'stlmax_notes': len(sdevs),
            'stlmax_spread_db': stlmax_spread}
        # --- DURATION COVERAGE. The same notes, at every gate on the axis.
        # The first entry is the SHIPPED reading, at the gate it was taken
        # with - 0.8 s for a melodic program, 3.2 s for a kit - so a consumer
        # sees the whole trajectory in one field and nothing measured above
        # is redefined. Each other gate is a pass of its own. The keys are
        # seconds of gate, which is the unit the corpus's note lengths are
        # counted in; a stretch factor is a property of the phrase and hides
        # the number the reader is asking about.
        shipped_gate = gate_s(note_dt(prog))
        by_dur = [[shipped_gate, rows[prog]['level_db']]]
        wd_dev = rows[prog]['level_db']
        wd_dur = shipped_gate if wd_dev is not None else None
        wd_key = rows[prog].get('worst_key')
        for g in sorted(set(DUR_HOLDS_S) | {shipped_gate}):
            if g == shipped_gate:
                continue
            got = duration_pass(prog, notes, g)
            by_dur.append([g, got[0] if got else None])
            if got and (wd_dev is None or abs(got[0]) > abs(wd_dev)):
                wd_dev = got[0]
                wd_dur = g
                wd_key = got[1]
        rows[prog]['level_by_dur'] = by_dur
        rows[prog]['level_worst_db'] = wd_dev
        rows[prog]['level_worst_dur'] = wd_dur
        rows[prog]['level_worst_key'] = wd_key
        if prog % 16 == 0:
            print('  ... p%d' % prog, flush=True)
            cache_save(os.path.join(OUTDIR, 'measure_cache.json'))
    json.dump(rows, open(os.path.join(OUTDIR, 'corpus_rows.json'), 'w'), indent=0)
    cache_save(os.path.join(OUTDIR, 'measure_cache.json'))
    nhold = sum(1 for r in rows.values() if r.get('hf_withheld'))
    print('\nflagged programs (level >2 dB, balance >6 dB, brightness '
          'TONE off ff8, GMFAR off gm, HF, seam):')
    bad = []
    for p in sorted(rows):
        r = rows[p]
        if 'error' in r:
            continue
        lv, cr, hf = r['level_db'], r['cen_ratio'], r['hf_ratio']
        flag = ''
        if lv is not None and abs(lv) > 2.0:
            flag += ' LEVEL'
        # BALANCE IS THE BULK, ROGUE IS ONE KEY. Both used to be
        # ' BALANCE', driven by max minus min.
        if r.get('level_iqr') is not None and r['level_iqr'] > BALANCE_SPREAD:
            flag += ' BALANCE'
        if r.get('key_rogue') is not None and r['key_rogue'] > 3.0:
            flag += ' ROGUE'
        # TONE IS JUDGED AGAINST THE TARGET, NOT THE BOUND. The
        # standard for this bank is "like ff8, not especially like gm,
        # and not especially unlike gm" - ff8 is what is aimed at and gm
        # is how far it may stray. Judged against gm alone, 56 programs
        # were called fine while being 2-4x off the target they are
        # supposed to sound like (p9 read 1.00 against gm and 4.44
        # against ff8). TONE is a miss of the target; GMFAR is a miss of
        # the bound.
        # THE FF8 REFERENCE IS ONLY MEANINGFUL WHERE FF8 HAS AN
        # INSTRUMENT OF ITS OWN. full_FF8.ksb was built with the GM
        # family fallback, so for a program FF8 lacks it renders some
        # other instrument - comparing to that is comparing to nothing.
        # FF8's own render differs from gm's wherever it really has the
        # program, so that is the test: 81 of 128 clear 15%.
        ff8ok = (r.get('cen') and r.get('cen_ff8') and r.get('cen_ref')
                 and abs(r['cen_ff8'] / r['cen_ref'] - 1.0) > 0.15)
        if ff8ok:
            if not (0.74 <= r['cen_ratio_ff8'] <= 1.35):
                flag += ' TONE'
        elif cr is not None and (cr > 1.35 or cr < 0.74):
            flag += ' TONE'
        if cr is not None and not (0.5 <= cr <= 2.0):
            flag += ' GMFAR'
        # HF READS AGAINST THE TARGET TOO, for the reason TONE does.
        # Against gm alone it reports how unlike gm the top end is,
        # which is not what this bank is aiming for.
        h8 = r.get('hf_ratio_ff8')
        if ff8ok:
            if h8 is not None and (h8 > 2.5 or h8 < 0.3):
                flag += ' HF'
        elif hf is not None and (hf > 2.5 or hf < 0.3):
            flag += ' HF'
        if hf is not None and (hf > 5.0 or hf < 0.2):
            flag += ' HFGM'
        # NO SEAM FLAG FROM THE RENDER. Both readings are kept as
        # descriptors and neither is called a defect, because neither
        # can tell a splice from an attack. The 99.9th percentile of
        # the second difference over the peak scored p101 at 0.402 with
        # its only spike at the onset; the tail-to-body ratio of the
        # same difference scored p117 at 6.0x gm and then found zero
        # single-sample jumps outside the first 150 ms of the note. A
        # decaying waveform drives the body of that difference toward
        # zero, so any sustained tone reads as hundreds of "jumps" -
        # p105 reported 10295 of them. A splice is a property of the
        # SAMPLE, and the sample-level reading for it is `seamrate`,
        # which the selection scores already.
        if flag:
            bad.append(p)
            print('  p%-3s %+7.2f dB  bal %5.1f dB@key%-3s  cen %5.2fx  '
                  'hf %5.2fx  seam %5.3f/%5.3f %s' % (
                      p, lv if lv is not None else 0.0,
                      r['level_spread'] if r['level_spread'] is not None else 0.0,
                      r['worst_key'] if r['worst_key'] is not None else -1,
                      cr if cr is not None else 0.0,
                      hf if hf is not None else 0.0,
                      r['seam'] or 0.0, r['seam_ref'] or 0.0, flag))
    print('\nflagged: %d of %d used programs' % (len(bad), len(rows)))
    sp = sorted((r['level_spread'], p) for p, r in rows.items()
                if r.get('level_spread') is not None)
    if sp:
        top = rows[sp[-1][1]]
        print('internal balance: median %.1f dB spread, p90 %.1f, worst %.1f dB '
              '(p%d key %d)' % (
                  sp[len(sp) // 2][0],
                  sp[int(len(sp) * 0.9)][0], sp[-1][0],
                  sp[-1][1], top['worst_key']))
    iq = sorted((r['level_iqr'], p) for p, r in rows.items()
                if r.get('level_iqr') is not None)
    rg = sorted((r.get('key_rogue') or 0, p) for p, r in rows.items())
    if rg:
        print('  rogue key: median %.1f dB off its own program, worst %.1f (p%s)'
              % (rg[len(rg) // 2][0], rg[-1][0], rg[-1][1]))
    if iq:
        print('  interquartile: median %.1f dB, p90 %.1f, worst %.1f (p%d) - '
              'what the bulk of the keys do, without one rogue note' % (
                  iq[len(iq) // 2][0], iq[int(len(iq) * 0.9)][0],
                  iq[-1][0], iq[-1][1]))
    # THE SAME THREE NUMBERS OVER THE CORPUS'S OWN KEYS ONLY. The sweep
    # above covers the whole keyboard; this is the criterion earlier
    # rounds were judged on, kept so the two are comparable and so
    # "where music plays" does not get lost under keys nobody has
    # played yet.
    cs = []
    ci = []
    cr_ = []
    for p, r in rows.items():
        ck = set(r.get('corpus_keys') or [])
        byk = {}
        for k, v, d in (r.get('key_devs') or []):
            if k in ck:
                byk.setdefault(k, []).append(d)
        kv = sorted((k, sorted(v)[len(v) // 2]) for k, v in byk.items())
        if len(kv) < 2:
            continue
        vals = [d for _, d in kv]
        # NOT CALLED `med`: that is this module's median-of-a-column
        # helper, and a local of the same name shadows it for the whole
        # of main() - the tone block above reads med(mb, 'cen') and
        # died with UnboundLocalError the first time this ran.
        mid = sorted(vals)[len(vals) // 2]
        cs.append(max(vals) - min(vals))
        cr_.append(max(abs(d - mid) for d in vals))
        s = sorted(vals)
        q1 = s[len(s) // 4]
        q3 = s[(3 * len(s)) // 4]
        ci.append(q3 - q1)
    if cs:
        cs.sort()
        ci.sort()
        cr_.sort()
        print('  on the corpus keys alone: spread median %.1f p90 %.1f worst %.1f; '
              'iqr median %.1f worst %.1f; rogue worst %.1f'
              % (cs[len(cs) // 2], cs[int(len(cs) * 0.9)], cs[-1],
                 ci[len(ci) // 2], ci[-1], cr_[-1]))
    dr = sorted((r['dur_ratio'], p) for p, r in rows.items()
                if r.get('dur_ratio'))
    if dr:
        print('  note length against gm: median %.2fx, p10 %.2f, worst %.2fx (p%s)'
              % (dr[len(dr) // 2][0], dr[len(dr) // 10][0], dr[0][0], dr[0][1]))
    if nhold:
        print('HF reading withheld on %d program(s): gm carries under '
              '%.2g of its energy above 8 kHz there' % (nhold, HF_FLOOR))
    # --- THE TWO NEW READINGS, AS SUMMARY LINES. Additive: the flags and
    # the table above are unchanged, and level_db above is what the level
    # trim still reads.
    dur = sorted((abs(r['level_worst_db'] - r['level_db']), p)
                 for p, r in rows.items()
                 if r.get('level_worst_db') is not None
                 and r.get('level_db') is not None)
    if dur:
        lp = rows[dur[-1][1]]
        print('duration coverage: %d program(s) at gate %s s; '
              '|worst - level_db| median %.2f dB, p90 %.2f, worst %.2f '
              '(p%d at %g s)'
              % (len(dur), '/'.join('%g' % g for g in DUR_HOLDS_S),
                 dur[len(dur) // 2][0],
                 dur[int(len(dur) * 0.9)][0], dur[-1][0], dur[-1][1],
                 lp.get('level_worst_dur')))
        # THE TRAJECTORY, GATE BY GATE. One number spanning the whole axis
        # answers neither end: the corpus plays the short one and the audit
        # was calibrated on the long one. This is the shape that was missing.
        print('   %-9s %5s %12s %7s %7s  %s'
              % ('gate s', 'n', '|dev| med', 'p90', 'max', 'program'))
        for g in sorted(set(DUR_HOLDS_S)):
            v = []
            for p, r in rows.items():
                if r.get('level_db') is None:
                    continue
                for e in (r.get('level_by_dur') or []):
                    if e[0] == g and e[1] is not None:
                        v.append((abs(e[1] - r['level_db']), p, e[1]))
            if not v:
                continue
            v.sort()
            ab = [t[0] for t in v]
            print('   %-9g %5d %12.2f %7.2f %7.2f  p%d (%+.2f)'
                  % (g, len(v), ab[len(ab) // 2], ab[int(len(ab) * 0.9)],
                     ab[-1], v[-1][1], v[-1][2]))
        # AND WHERE EACH PROGRAM'S WORST READING LIVES. A program whose
        # worst gate is a short one is the case the 0.8 s calibration
        # cannot see and a gain cannot fix without moving the 0.8 s reading.
        for lo_, hi_, nm in ((0.0, 0.25, 'short (<= 0.25 s)'),
                             (0.25, 99.0, 'long  (>  0.25 s)')):
            w = [t for t in dur
                 if rows[t[1]].get('level_worst_dur') is not None
                 and lo_ <= rows[t[1]]['level_worst_dur'] < hi_]
            if w:
                print('   worst gate is %-16s on %d program(s); |dev| '
                      'median %.2f, p90 %.2f, max %.2f (p%d at %g s)'
                      % (nm, len(w), w[len(w) // 2][0],
                         w[int(len(w) * 0.9)][0], w[-1][0], w[-1][1],
                         rows[w[-1][1]].get('level_worst_dur')))
    sl = sorted((abs(r['stlmax_db'] - r['level_db']), p)
                for p, r in rows.items()
                if r.get('stlmax_db') is not None
                and r.get('level_db') is not None)
    if sl:
        print('STLmax vs window average: %d program(s); |stlmax_db - level_db| '
              'median %.2f dB, p90 %.2f, worst %.2f (p%d)'
              % (len(sl), sl[len(sl) // 2][0],
                 sl[int(len(sl) * 0.9)][0], sl[-1][0], sl[-1][1]))


# ---------------------------------------------------------------------------
# BALANCE_FROM_AUDIT AND THE LEVEL TRIM, AFTER THE DURATION PASS -- A PLAN,
# NOT IMPLEMENTED HERE.
#
# THE TWO CONSUMERS ARE NOT THE SAME FUNCTION:
#   * tools/ksb_pick.py:3506 balance_from_audit() moves PER-REGION gains from
#     `key_devs` (the per key/velocity deviations), not from level_db;
#   * tools/ksb_pick.py:4194 level_trim_from_rows() is the one that reads
#     `level_db` and sets a PER-PROGRAM gain (with the peak ceiling).
#
# WHAT CHANGES IF level_trim_from_rows MOVES TO level_worst_db. A region gain
# is one number, so it shifts every duration equally. Calibrating it to the
# worst duration moves every other duration by (level_worst_db - level_db).
# Measured on the 2026-10-30 demo (the ten programs in build/tmp/w4_env/):
# delta median -0.05 dB, min -1.56 (p89, x16), max +0.84 (p28, x16). p89's
# short notes - the ones the corpus actually plays - would come out 1.56 dB
# hot while its sustained notes land on gm; p28's would run 0.84 dB quiet.
# The delta is not noise: it is the duration slope, and a slope is what one
# gain cannot fix.
#
# WHAT TO DO INSTEAD (recommended):
#   1. Keep both gain passes on the readings that match the music: level_db
#      for the program trim, key_devs for the region balance. They describe
#      the notes the corpus plays, which is what the bank is judged on.
#   2. Use level_worst_db / level_worst_dur as a FLAG, not a target. A
#      program whose worst reading is a real sustained error belongs to the
#      envelope work (decay, key->decay modulation), not to a gain trim.
#   3. If a sustained trim is ever wanted, gate it: only where level_by_dur
#      is FLAT (the factors agree, so the error really is an offset) may a
#      gain move it; where it slopes, a gain trades one duration for
#      another and the gain is the wrong knob.
#   4. A weighted target - each factor weighted by the share of corpus
#      note-time actually spent near that duration - is defensible, but the
#      audit does not record note lengths, so that weight is a change to the
#      corpus side, not to this table.
#
# STLmax HAS NO CONSUMER YET. When it gets one, compare it only on the same
# notes (stlmax_notes) and treat stlmax_spread_db with care: at low keys the
# A-weighting acts as the spectral stage, so two banks with the same
# unweighted level but different harmonic content read tens of dB apart
# (p89, key 4: +28 dB under 'A', +0.3 dB under 'none' - the pooling is not
# what moved it).
#
# WITNESSES FOR THE DURATION COVERAGE, chosen without looking at any level
# reading (2026-10-30, build/tmp/w4_env/dur_select.py): ranking the 129
# programs by their corpus notes' MEDIAN DURATION from the MIDI alone gives
# the ten shortest (median 0.031-0.125 s, group median 0.103 s) and the ten
# longest (0.842-2.000 s, group median 1.112 s). The degradation
# |level_worst_db| - |level_db| is median 0.03 dB (max 2.87, p62) on the
# short group and median 1.07 dB (max 11.11, p103) on the long one: the
# coverage bites where the notes are long, and it was not told which
# programs those were.
#
# CROSS-INSTRUMENT CHECK, against chan_level.py on flourish (a different
# measurement path: real songs, per-channel isolation, no synthetic phrase):
#     prog  chan_level  level_db  level_worst_db
#     p12      -4.04      -0.19      -0.22   (not aligned; explained by the
#                                             storage-rate line, source
#                                             sample level in 0-10 ms)
#     p89      -3.16      -1.16      -2.72   (gap 0.44 dB - aligned)
#     p11      -2.44      -2.30      -2.56   (aligned already)
#     p108     -1.66      -1.43      -1.49   (aligned already)
# The point is that the second instrument shares no note set and no window
# with this one: level_worst_db moving p89 1.56 dB toward chan_level is not
# this table agreeing with itself. Only p89 needed explaining; p11 and p108
# were already aligned, and p12's gap is not a duration effect.
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    main()
