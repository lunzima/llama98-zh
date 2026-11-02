"""Where one note is actually sounding.

ONE RULE, ONE COPY. This was written out four times - in tune_pick's
render_note, ksb_pick's _note_rms, corpus_report's measure and
render_census2's census - and the four did not agree. Two used a
RMS envelope with an energy threshold (10^(-25/10)) and two used the
matching RMS one (10^(-25/20)), so the same -25 dB rule was -50 dB in
half the project and the measurements were not the same measurement.

WHY THE WINDOW CANNOT BE A FIXED SLICE. A fixed 0.02-0.55 s reads a
pad's attack and a woodblock's entire decay - two quantities under one
name - and two banks are then compared on whichever the program happens
to be. What a note delivers while it sounds runs from its onset to
where it has fallen 25 dB, and that is what level and timbre mean here.
"""
import numpy as np

# The note is "up" while its frame RMS is within this many dB of its
# own loudest frame. The original, in corpus_report's duration
# measurement, was the same 25 dB.
FLOOR_DB = -25.0

# A note whose window runs longer than this is measured over the first
# part of it. A pad with a twenty-second tail would otherwise spend the
# whole tail in the window; both banks meet the same cap, so a ratio
# taken over it is still a ratio.
CAP_S = 2.0

# One frame is 10 ms - fine enough to find an onset, coarse enough that
# a hi-hat's 60 ms is a handful of frames and not one.
FRAME_S = 0.01


def note_window(x, rate, floor_db=FLOOR_DB, cap_s=CAP_S):
    """(lo, hi) of the samples of one note that carry its sound.

    `x` is one channel of the rendered note, already mixed down and
    already starting at the onset the caller measured from. Returns None
    when there is nothing there to measure.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 8:
        return None
    fl = max(8, int(FRAME_S * rate))
    nf = len(x) // fl
    if nf < 2:
        return None
    env = np.sqrt((x[:nf * fl].reshape(nf, fl) ** 2).mean(axis=1))
    pk = float(env.max())
    if pk <= 0.0:
        return None
    # A LEVEL THRESHOLD, and the envelope is a level: -25 dB is
    # 10^(-25/20) here. The copies this replaced used the energy
    # factor with an RMS envelope and read -50 dB instead.
    live = np.nonzero(env >= pk * 10.0 ** (floor_db / 20.0))[0]
    if len(live) == 0:
        return None
    lo = int(live[0]) * fl
    hi = min(len(x), (int(live[-1]) + 1) * fl)
    if cap_s:
        hi = min(hi, lo + int(cap_s * rate))
    if hi - lo < 8:
        return None
    return lo, hi
