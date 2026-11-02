"""How loud one note is, read as STLmax instead of as a window average.

THE QUESTION. `corpus_report` compares a program's level against gm by
taking the energy average over a note's window. That figure depends on the
window: a 120 ms marimba note and a 5 s pad put their energy in different
places, and the same program read at different note lengths moves by dB.
For sounds shorter than ~400 ms the average is the wrong summary in the
first place - a short sound's overall loudness is predicted by the MAXIMUM
short-term loudness (Glasberg & Moore 2002; Chalupper & Fastl 2002;
ISO 532-2 / ANSI S3.4), and ITU-R BS.1770's integrated loudness is not even
defined below 400 ms. This instrument answers: how loud is THIS note, read
the way the literature reads a short sound.

WHAT IT DOES NOT ANSWER. It is not a full loudness model and it is not a
perceptual model of the bank. It has no auditory filter bank, no threshold
in quiet, no compressive specific loudness, and its outer/middle-ear stage
is A-weighting, not the Glasberg-Moore transfer function. Its output is a
LEVEL-domain STLmax (dBFS), not a sone loudness - see the deviation note
below for why that is deliberate. It says nothing about timbre, stereo,
seam or clipping. Same-program, same-pitch comparisons are what it was
built for; a cross-program loudness ranking is outside its approximations.

HOW. Instantaneous loudness is the A-weighted analytic envelope squared,
halved, averaged over 1 ms frames. Short-term loudness is that sequence
through a first-order attack/release smoother (22/50 ms, TVL 2002), and
STLmax is the maximum over the window. The energy average reported
alongside uses the SAME weighting and the SAME frames, so the only thing
that differs between the two figures is the time pooling.

    import stlmax
    d = stlmax.analyse_note(x, sr, 0.0, 0.25)
    d['stlmax_db'], d['energy_db'], d['inst_max_db']

    python tools/stlmax_selftest.py      # 7 checks, 6 mutations that must go red

The self-test is the reason this file is trusted: it pins the published
alpha values for both TVL revisions, and every mutation of a constant or
of the pooling turns at least one check red.

---
STLmax as a note-loudness criterion, pure numpy.

WHY NOT A WINDOW AVERAGE. For a note shorter than a few hundred ms the
energy average is dominated by where the window happens to end: a fast
decay read over its own 0.12 s and a steady pad over 5 s are averaging
different parts of the sound, and the same program measured at 0.5 and 8
beats drifts by dB. The literature's answer is the maximum of the
short-term loudness (TVL, Glasberg & Moore 2002; DLM, Chalupper & Fastl
2002), and that is what `stlmax_db` is.

INTERFACE
    analyse_note(x, sr, t0, t1)      -> dict   one window
    analyse_windows(x, sr, windows)  -> list   many windows (one FFT pass)
    note_stlmax(x, sr, t0, t1)       -> float
    energy_average_db(x, sr, t0, t1) -> float
    short_term_curve(x, sr)          -> (t, frame level, short-term level)
    alphas(...)                      -> the published alpha set, for checking
    load_wav_mono(path)              -> (float64 mono, sr) for 16-bit PCM

Returned dict: 'stlmax_db', 'energy_db', 'inst_max_db', 't0', 't1',
'n_frames', 'weighting' (+ 'ltmax_db' with long_term=True).
STLmax never exceeds inst_max_db.

dB ZERO is full-scale digital amplitude 1.0 (dBFS). Both banks are read
by the same pipeline, so the zero does not enter a difference. Unweighted,
energy_db matches a plain window RMS to within 0.01 dB.

TIME STAGE: THE PINNED CONSTANTS, 1 ms STEPS
    S(n) = aa*L(n) + (1-aa)*S(n-1)   if L(n) > S(n-1)   (attack)
    S(n) = ar*L(n) + (1-ar)*S(n-1)   otherwise          (release)
    alpha = 1 - exp(-dt/tau), dt = 1 ms

    TVL_2002  (Glasberg & Moore 2002, JAES 50(5):331-342; now the
               Moore-Glasberg-Schlittenlacher method of ISO 532-3)
        short: ta = 22 ms -> aa = 0.0444 (published as 0.045)
               tr = 50 ms -> ar = 0.0198 (0.020)
        long:  ta = 99 ms -> 0.0101 (0.010)
               tr = 2000 ms -> 0.00050 (0.0005)
    TVL_2018  (Moore, Glasberg & Schlittenlacher 2018) changes ONLY release:
        short tr = 30 ms -> ar = 0.0328 (0.033)
        long  tr = 751 ms -> 0.00133
        attack unchanged (22 / 99 ms). BOTH VERSIONS ARE VALID; do not mix
        the two sets.

    THIS IMPLEMENTATION USES THE 2002 SET BY DEFAULT. Pass **TVL_2018 to
    any entry point to run the 2018 set. alphas() turns either set's tau
    into the published alphas so the numbers can be checked, not trusted.

STLmax = max over the window of S(n); the long-term stage (99/2000 ms,
optional) smooths S again. Both start from zero because the render starts
at the first note-on, before which there is silence.

THE THREE DEVIATIONS FROM THE MODEL, SPELLED OUT
 1. LEVEL DOMAIN, NOT SONE. TVL's compressive stage N' ~ E^0.23 is felt
    loudness, and it is not gain-invariant: +6 dB of level buys about
    +2.8 dB of loudness. This instrument compares two banks of the SAME
    program at the SAME pitch, and its acceptance test is "+-6 dB of gain
    -> +-6 dB of STLmax within 0.01 dB", which a compressive stage cannot
    pass by construction. So the output is the short-term LEVEL's peak.
 2. SPECTRAL STAGE NOT IMPLEMENTED. Outer/middle ear is A-weighting (the
    IEC 61672 closed form), not Glasberg-Moore's 29-point transfer
    function. There is no 24-band roex filter bank, no through component,
    no threshold in quiet, and no (E/E_TQ)^0.23 compression. A-weighting
    is a 40-phon equal-loudness approximation. In a same-program,
    same-pitch difference most of this cancels; the quantity actually
    being swapped is the TIME POOLING (window max vs window mean).
 3. ENVELOPE, NOT A 20 ms RUNNING SPECTRUM. Instantaneous loudness is
    |analytic signal|^2/2 averaged over 1 ms frames. The halving makes a
    steady tone's instantaneous power equal its mean square, so the
    unweighted energy figure sits on the window-RMS scale. 1 ms frames
    replace TVL's 20 ms spectrum because 20 ms smears exactly the short
    transients this criterion is about (it cuts the short-vs-sustained
    gap from the 7.4 dB bound to about 4 dB). Cost: for noise-like
    signals (cymbals, breath) the analytic envelope fluctuates, so
    STLmax can land on one fluctuation.

    In one line: TVL's short-term TIME structure + A-weighting standing in
    for the spectral stage + a level-domain output. NOT "the TVL model".

THE ENERGY AVERAGE, FOR COMPARISON. `energy_db` is 10*log10 of the mean
1 ms frame power over the window - the same weighting, the same frames.
The only difference from `stlmax_db` is the pooling. (The shipped
calibration's RMS is unweighted; for the same program and pitch the
A-weighted and unweighted differences usually sit within a ~0.2 dB
constant offset, and realdata.py reports both columns.)

WINDOWS MAY RUN PAST THE END OF THE FILE: the missing part is silence.
The player stops writing frames once the voices die, so an 8-beat
marimba's WAV is often 1.4 s long and the last 2.6 s of its window are
silence. That is a real and important case here, not an edge bug.
"""
import os as _os
import wave as _wave

import numpy as np

# --- model constants (seconds). The self-test mutates these; a wrong value
# --- there must turn an assertion red.
FRAME_S = 0.001         # the instantaneous-loudness step; the literature's dt
ATTACK_S = 0.022        # TVL short-term attack
RELEASE_S = 0.050       # TVL short-term release (2002)
LT_ATTACK_S = 0.099     # long-term attack
LT_RELEASE_S = 2.000    # long-term release (2002)

# HOW MUCH SIGNAL EITHER SIDE OF A WINDOW THE SERIES IS COMPUTED OVER.
# See analyse_windows: the transforms are O(n) in what they are handed, and
# the callers hand them whole pieces to read windows out of. 1 s covers the
# smoother's 50 ms release twenty times over and both filters' kernels;
# infinity restores the whole-file pass, which is the gate's control arm.
CROP_MARGIN_S = 1.0

# THE MOST ONE TRANSFORM PASS MAY COVER. 0 - the default - means no cap.
# Splitting a merged run is 1.4x faster on an effects program but it is
# NOT transparent: a window's level legitimately includes the previous
# notes' decay, so cutting the run gives that window a shorter history.
# Measured on p4's dense key sweep, stlmax_spread_db (a worst-key minus
# the rest, and outlier-driven by construction) went 5.44 -> 12.86 dB at
# a 4 s cap and 11.10 at 8 s, non-monotonically, while the median
# stlmax_db held at 1.411. A performance knob that moves a reported
# figure by 2.4x is not a performance knob.
CROP_SPAN_MAX_S = float(_os.environ.get('STLMAX_SPAN_MAX') or 0.0)

# WHETHER THE WEIGHTING AND THE ANALYTIC ENVELOPE GO THROUGH THE SAME
# TRANSFORM. See _power_fused: they commute, and the composition costs
# four whole-array passes where one does. False restores the composition
# for the gate to compare against.
FUSE_TRANSFORMS = True

# Both published sets; the default is 2002. Do not take half from each.
TVL_2002 = {'attack_s': 0.022, 'release_s': 0.050,
            'lt_attack_s': 0.099, 'lt_release_s': 2.000}
TVL_2018 = {'attack_s': 0.022, 'release_s': 0.030,
            'lt_attack_s': 0.099, 'lt_release_s': 0.751}

_EPS = 1e-30            # -300 dBFS; the digital-silence floor


# --------------------------------------------------------------------------
# basic quantities
# --------------------------------------------------------------------------
def power_to_db(p):
    """Power -> dBFS. Silence reads -300 dB rather than -inf, so tables work."""
    return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=np.float64), _EPS))


def alphas(attack_s=None, release_s=None, dt=FRAME_S,
           lt_attack_s=None, lt_release_s=None):
    """tau -> the published alpha = 1 - exp(-dt/tau), for checking constants."""
    a = ATTACK_S if attack_s is None else float(attack_s)
    r = RELEASE_S if release_s is None else float(release_s)
    la = LT_ATTACK_S if lt_attack_s is None else float(lt_attack_s)
    lr = LT_RELEASE_S if lt_release_s is None else float(lt_release_s)
    return {'attack': 1.0 - np.exp(-dt / a),
            'release': 1.0 - np.exp(-dt / r),
            'lt_attack': 1.0 - np.exp(-dt / la),
            'lt_release': 1.0 - np.exp(-dt / lr)}


def a_weighting_db(freq):
    """IEC 61672 A-weighting in dB (exactly 0 dB at 1 kHz).

        R_A(f) = (12194^2 f^4) / ((f^2+20.6^2) * sqrt((f^2+107.7^2)(f^2+737.9^2))
                                  * (f^2+12194^2))
        A(f) = 20 log10(R_A(f)) + 2.00
    """
    f = np.asarray(freq, dtype=np.float64)
    f2 = f * f
    num = (12194.0 ** 2) * (f2 * f2)
    den = ((f2 + 20.6 ** 2)
           * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
           * (f2 + 12194.0 ** 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        ra = num / den
    out = 20.0 * np.log10(np.maximum(ra, 1e-300)) + 2.0
    out = np.where(f > 0.0, out, -300.0)   # DC carries no loudness
    return out


def apply_weighting(x, sr, weighting='A'):
    """Zero-phase frequency-domain weighting. 'A' or 'none'."""
    x = np.asarray(x, dtype=np.float64)
    if weighting == 'none':
        return x.copy()
    if weighting != 'A':
        raise ValueError("weighting must be 'A' or 'none'")
    n = len(x)
    if n == 0:
        return x.copy()
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / float(sr))
    spec *= 10.0 ** (a_weighting_db(f) / 20.0)
    return np.fft.irfft(spec, n)


def analytic_envelope(x):
    """|x_a(t)|, the analytic-signal envelope (Hilbert, frequency domain)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x.copy()
    spec = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[1:n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spec * h))


def instantaneous_power(x, sr, weighting='A'):
    """Per-sample instantaneous loudness: half the weighted envelope squared.

    The half makes a steady tone's instantaneous power equal its mean square
    (an analytic signal carries twice the energy of the real one), so the
    unweighted energy figure sits on the same scale as a window RMS.
    """
    if FUSE_TRANSFORMS:
        return _power_fused(x, sr, weighting)
    xw = apply_weighting(x, sr, weighting)
    return 0.5 * analytic_envelope(xw) ** 2


def _power_fused(x, sr, weighting):
    """The four transforms folded into two.

    A-WEIGHTING COMMUTES WITH THE ANALYTIC MASK. Both are real, even
    multipliers on the frequency axis - one is the weighting response, the
    other takes the positive half and doubles it - so applying them in
    either order gives the same signal. Written as the composition it ran

        rfft -> irfft -> fft -> ifft

    and every one of those is a whole-array pass. Written as one pass:

        rfft -> ifft

    which is where the measurement spends nearly all its time; the audit
    profile put 40 of 70 seconds in these transforms, and a benchmark of
    1700x1700-point work on this machine shows it is memory bandwidth, not
    arithmetic - eight concurrent processes each took 5.7x as long, so
    there is no core count left to buy. Halving the passes is the only
    lever that is left.

    FUSE_TRANSFORMS=False restores the composition, which is what the gate
    at build/tmp/stlmax_fuse_gate.py compares against.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x.copy()
    if weighting not in ('A', 'none'):
        raise ValueError("weighting must be 'A' or 'none'")
    spec = np.fft.rfft(x)
    if weighting == 'A':
        f = np.fft.rfftfreq(n, 1.0 / float(sr))
        spec *= 10.0 ** (a_weighting_db(f) / 20.0)
    m = spec.shape[0]                      # n // 2 + 1 bins, DC first
    z = np.zeros(n, dtype=np.complex128)   # the negative half stays zero
    z[:m] = spec
    if n % 2 == 0:
        z[1:m - 1] *= 2.0                  # DC and Nyquist are not doubled
    else:
        z[1:m] *= 2.0
    return 0.5 * np.abs(np.fft.ifft(z)) ** 2


def frame_levels(p, sr, frame_s=FRAME_S):
    """Fold per-sample power into 1 ms frame means. Returns (centres, level).

    Frame edges are round(k*frame_s*sr), so these are true 1 ms frames
    (44/45 samples alternating at 44.1 kHz), not 44-sample blocks. A
    trailing fraction of a frame is dropped.
    """
    p = np.asarray(p, dtype=np.float64)
    if len(p) == 0:
        return np.zeros(0), np.zeros(0)
    step = float(frame_s) * sr
    nfr = int(np.floor(len(p) / step))
    if nfr < 1:
        return np.array([0.5 * len(p) / sr]), np.array([p.mean()])
    idx = np.rint(np.arange(nfr + 1) * step).astype(np.int64)
    idx = np.clip(idx, 0, len(p))
    cs = np.concatenate(([0.0], np.cumsum(p)))
    counts = np.maximum(idx[1:] - idx[:-1], 1)
    lev = (cs[idx[1:]] - cs[idx[:-1]]) / counts
    centres = (idx[:-1] + idx[1:]) * 0.5 / sr
    return centres, lev


def _resolve_taus(attack_s, release_s, lt_attack_s, lt_release_s):
    """Resolve tau AT CALL TIME from module globals, not at def time.

    The self-test's mutations change stlmax.ATTACK_S and friends. A default
    argument binds at definition time, so mutating the module constant
    would do nothing and the mutation would come out falsely green.
    """
    return (float(globals()['ATTACK_S'] if attack_s is None else attack_s),
            float(globals()['RELEASE_S'] if release_s is None else release_s),
            float(globals()['LT_ATTACK_S'] if lt_attack_s is None else lt_attack_s),
            float(globals()['LT_RELEASE_S'] if lt_release_s is None else lt_release_s))


def smooth_levels(levels, dt=FRAME_S, attack_s=None, release_s=None):
    """First-order attack/release smoother - the two recursion lines above.

    S = alpha*L + (1-alpha)*S, alpha = 1 - exp(-dt/tau), alpha chosen by
    L > S. Applied in the power domain, once per 1 ms frame.
    """
    lev = np.asarray(levels, dtype=np.float64)
    if len(lev) == 0:
        return lev.copy()
    a_tau, r_tau, _, _ = _resolve_taus(attack_s, release_s, None, None)
    aa = 1.0 - np.exp(-float(dt) / a_tau)
    ar = 1.0 - np.exp(-float(dt) / r_tau)
    out = np.empty_like(lev)
    prev = 0.0                     # the render starts from silence
    for i in range(len(lev)):
        v = lev[i]
        a = aa if v > prev else ar
        prev = prev + a * (v - prev)
        out[i] = prev
    return out


# --------------------------------------------------------------------------
# one waveform -> frames, short-term level, [long-term level]
# --------------------------------------------------------------------------
def _series(x, sr, weighting='A', long_term=False,
            attack_s=None, release_s=None, lt_attack_s=None, lt_release_s=None):
    """The whole waveform's 1 ms frame levels and short-term level."""
    p = instantaneous_power(x, sr, weighting)
    t, lev = frame_levels(p, sr)
    a_tau, r_tau, la_tau, lr_tau = _resolve_taus(attack_s, release_s,
                                                 lt_attack_s, lt_release_s)
    y = smooth_levels(lev, FRAME_S, a_tau, r_tau)
    if not long_term:
        return t, lev, y
    yl = smooth_levels(y, FRAME_S, la_tau, lr_tau)
    return t, lev, y, yl


def _select(t, t0, t1):
    """Frames whose centre lies in [t0, t1). A sub-frame window gets one frame."""
    sel = np.nonzero((t >= t0) & (t < t1))[0]
    if len(sel) == 0:
        k = int(np.clip(np.searchsorted(t, t0), 0, len(t) - 1))
        sel = np.array([k])
    return sel


def _pad_to(x, sr, t1):
    b = int(round(t1 * sr))
    if b > len(x):
        return np.pad(x, (0, b - len(x)))
    return x


def analyse_note(x, sr, t0=0.0, t1=None, weighting='A', long_term=False,
                 attack_s=None, release_s=None,
                 lt_attack_s=None, lt_release_s=None):
    """One note window: STLmax, energy average, instantaneous ceiling.

    x       mono float waveform at full scale 1.0
    sr      sample rate
    t0, t1  window in seconds; t1=None means to the end of the file
    """
    x = np.asarray(x, dtype=np.float64)
    if t1 is None:
        t1 = len(x) / float(sr)
    if t1 <= t0:
        raise ValueError('window is empty: t0=%r t1=%r' % (t0, t1))
    x = _pad_to(x, sr, t1)
    got = _series(x, sr, weighting, long_term,
                  attack_s, release_s, lt_attack_s, lt_release_s)
    t, lev, y = got[0], got[1], got[2]
    sel = _select(t, float(t0), float(t1))
    out = {
        't0': float(t0), 't1': float(t1),
        'stlmax_db': float(power_to_db(y[sel].max())),
        'energy_db': float(power_to_db(lev[sel].mean())),
        'inst_max_db': float(power_to_db(lev[sel].max())),
        'n_frames': int(len(sel)),
        'weighting': weighting,
    }
    if long_term:
        out['ltmax_db'] = float(power_to_db(got[3][sel].max()))
    return out


def analyse_windows(x, sr, windows, weighting='A', long_term=False,
                    attack_s=None, release_s=None,
                    lt_attack_s=None, lt_release_s=None):
    """Batch entry: evaluate a list of (t0, t1) windows.

    THE SERIES IS COMPUTED OVER THE WINDOWS' NEIGHBOURHOOD, NOT THE FILE.
    The four transforms in _series are the whole cost and they are O(n) in
    the length handed to them; the caller hands them a key sweep, which for
    an effects program is 384 notes four seconds apart - 1536 s, 68 M
    samples - to read 384 windows of a few hundred milliseconds each. On
    the audit that was 67 of 89 seconds.

    Each window is taken with CROP_MARGIN_S either side and the spans are
    merged, so overlapping windows still share one pass. The margin covers
    the smoother (attack 22 ms, release 50 ms) and the two filters' kernels.

    THIS IS NOT AN IDENTITY. The weighting and the analytic envelope are
    zero-phase transforms whose frequency grid is sampled at n points, so
    they depend on the array length; a crop of a different length answers
    by a slightly different filter. Measured at build/tmp/stlmax_crop_gate.py
    on real renders: <= 0.05 dB, against a |stlmax_db - level_db| that runs
    a median 0.71 dB and a p90 2.66 dB. Set CROP_MARGIN_S to infinity and
    the old whole-file pass comes back unchanged, which is what the gate
    uses as its control arm.
    """
    x = np.asarray(x, dtype=np.float64)
    if not windows:
        return []
    dur = len(x) / float(sr)

    iv = []
    for t0, t1 in windows:
        iv.append((max(0.0, float(t0) - CROP_MARGIN_S),
                   min(dur, float(t1) + CROP_MARGIN_S)))
    iv.sort()
    # A CAP ON HOW MUCH ONE PASS MAY COVER. Windows that touch merge into
    # one span, and a heavily populated key sweep merges into the whole
    # file - exactly the case the crop was introduced to avoid. Measured:
    # shortening a phrase from 628 s to 393 s made the pass SLOWER (19.1 s
    # against 18.1 s) because the shorter file merged into one big
    # transform while the longer one stayed in pieces. Many small
    # transforms beat one large one here: a 17 M-point complex array is
    # 272 MB and does not fit anywhere.
    #
    # A span is only ever extended by a whole window, so the cap can make
    # a span shorter than the cap and never cuts one in half.
    spans = []
    for a, b in iv:
        if (spans and a <= spans[-1][1]
                and (CROP_SPAN_MAX_S <= 0
                     or max(spans[-1][1], b) - spans[-1][0] <= CROP_SPAN_MAX_S)):
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))

    out = [None] * len(windows)
    # THE CROP HAS TO START ON A GLOBAL 1 ms FRAME EDGE. frame_levels()
    # counts frames from the array's first sample, so a crop that begins
    # mid-frame shifts every boundary after it and the reading moves -
    # measured on a real render: sliding the crop start by 22 samples
    # moved stlmax_db 0.08 dB, and it does not shrink with the margin
    # because the offset is fixed. Aligning to rint(j*step) makes crop
    # frame m the same frame the whole-file pass called j+m.
    step = FRAME_S * float(sr)
    for (a, b) in spans:
        i0 = int(round(np.floor(a * sr / step) * step)) if step > 0 else 0
        if i0 < 0:
            i0 = 0
        i1 = int(round(b * sr))
        a = i0 / float(sr)
        seg = x[i0:i1]
        # THE SAME PAD THE WHOLE-FILE PASS APPLIES: a window running past
        # the signal is read against zeroes, not against the last sample.
        need = len(x) if b >= dur else int(round(b * sr)) - i0
        if need > len(seg):
            seg = np.pad(seg, (0, need - len(seg)))
        got = _series(seg, sr, weighting, long_term,
                      attack_s, release_s, lt_attack_s, lt_release_s)
        t, lev, y = got[0], got[1], got[2]
        for k, (t0, t1) in enumerate(windows):
            if out[k] is not None:
                continue
            if t0 < a - 1e-9 or t1 > b + 1e-9:
                continue
            sel = _select(t, float(t0) - a, float(t1) - a)
            d = {
                't0': float(t0), 't1': float(t1),
                'stlmax_db': float(power_to_db(y[sel].max())),
                'energy_db': float(power_to_db(lev[sel].mean())),
                'inst_max_db': float(power_to_db(lev[sel].max())),
                'n_frames': int(len(sel)),
                'weighting': weighting,
            }
            if long_term:
                d['ltmax_db'] = float(power_to_db(got[3][sel].max()))
            out[k] = d
    if any(d is None for d in out):
        raise ValueError('a window fell outside every span it was given')
    return out


def short_term_curve(x, sr, weighting='A', long_term=False,
                     attack_s=None, release_s=None,
                     lt_attack_s=None, lt_release_s=None):
    """(frame centres, frame level, short-term level[, long-term level])."""
    x = np.asarray(x, dtype=np.float64)
    return _series(x, sr, weighting, long_term,
                   attack_s, release_s, lt_attack_s, lt_release_s)


def note_stlmax(x, sr, t0=0.0, t1=None, **kw):
    return analyse_note(x, sr, t0, t1, **kw)['stlmax_db']


def energy_average_db(x, sr, t0=0.0, t1=None, **kw):
    return analyse_note(x, sr, t0, t1, **kw)['energy_db']


# --------------------------------------------------------------------------
# read 16-bit PCM WAV (kunkun98-play's output: stereo, 44100 Hz)
# --------------------------------------------------------------------------
def load_wav_mono(path):
    """Read a 16-bit PCM WAV; average the channels down. Returns (float64, sr)."""
    w = _wave.open(path, 'rb')
    try:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    finally:
        w.close()
    if sw != 2:
        raise ValueError('%s: only 16-bit PCM is handled (sampwidth %d)'
                         % (path, sw))
    a = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


# ---------------------------------------------------------------------------
# WIRING PLAN FOR corpus_report -- NOT IMPLEMENTED HERE.
#
# Nothing in this file changes `corpus_report.py`; this block is the plan a
# later pass applies. Keep `level_db` EXACTLY as it is: corpus_rows.json's
# `level_db` is what ksb_pick's calibration reads (ksb_pick.py:4247 and
# :4692), so changing its meaning would silently move the gain trims.
#
# NEW FIELDS (additive; every existing key keeps its meaning and value) in
# the per-program row dict of corpus_report.py (built near :607, dumped at
# :663):
#
#   'stlmax_db'     median over the program's notes of (STLmax_ours -
#                   STLmax_gm) in dB. The STLmax counterpart of level_db,
#                   same statistic, same gate, same note set.
#   'stlmax_notes'  how many notes produced that median. Needed because the
#                   gating is on the reference's level and an operator must
#                   be able to see an empty or thin reading instead of a
#                   number.
#   'stlmax_spread_db'  the per-key spread (max-min over keys) recomputed
#                   from the STLmax ratios, mirroring 'level_spread'. A
#                   display figure; the calibration does not read it.
#
# Nothing else. In particular do NOT add a per-note array: the JSON holds
# per-program aggregates only, and a per-note list multiplies its size.
#
# SITE 1 -- corpus_report.py:343, inside measure():
#
#   before:
#       out.append({'rms': float(np.sqrt((seg * seg).mean())),
#                   'peak': peak, ...})
#   after:
#       st = _STLMAX.analyse_note(a, r, t0 + lo, t0 + hi)   # WHOLE array!
#       out.append({'rms': float(np.sqrt((seg * seg).mean())),
#                   'stlmax': st['stlmax_db'],
#                   'peak': peak, ...})
#
#   `a` is measure()'s already-loaded mono mix; `seg` is a slice of it and
#   MUST NOT be passed: the smoother would start from zero at the window's
#   start and lose everything before it (harmless for lo=0 windows, which
#   is why a quick test on this bank would not notice).
#
# SITE 2 -- corpus_report.py:508-512, next to the existing ratio:
#
#   before:
#       a, b = x['rms'], y['rms']
#       vb.append(a); vr.append(b)
#       if a > 0 and b > 0 and (ref_floor is None or b >= ref_floor):
#           dev = 20.0 * math.log10(a / b)
#           devs.append((notes[i][0], dev))
#   after (add, do not replace):
#       sa, sb = x.get('stlmax'), y.get('stlmax')
#       if (sa is not None and sb is not None
#               and sa > -300.0 and sb > -300.0
#               and (ref_floor is None or b >= ref_floor)):
#           sdevs.append((notes[i][0], sa - sb))   # both are dB, so a diff
#
#   The gate stays on the OLD figure's reference level on purpose: the two
#   medians must be taken over the SAME notes, or a difference between
#   level_db and stlmax_db could come from a different note set.
#   Then, next to 'level_db': ('stlmax_db' from sorted(sdevs) median,
#   'stlmax_notes': len(sdevs)). `lb`/`lr` (computed at :534-535 and never
#   read) can be left alone.
#
# ACCEPTANCE, on the same render batch (deterministic inputs, so the WAVs
# are identical):
#   1. corpus_rows.json 'level_db' is BIT-IDENTICAL before and after, for
#      every program (compare the JSONs, ignoring the new keys).
#   2. 'stlmax_db' is present and non-null wherever level_db is; on the
#      decaying instruments it should disagree with level_db by several dB
#      at short windows - if it comes out equal to level_db, the pooling
#      mutation was applied somewhere.
#   3. corpus_report's exit status and printed table are unchanged apart
#      from any new column.
#   4. Re-run tools/stlmax_selftest.py: 7/7 and all six mutations red.
# ---------------------------------------------------------------------------
