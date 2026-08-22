"""Where src/ops_quant.c's LZ_SP_LOG1P coefficients and B1's cap come from.

Three things, in order:
  1. Remez minimax coefficients for P(u) = log1p(u)/u on [0,1], rounded
     to float32.  The result is log1p(u) = u * P(u), which is what makes
     the tail relatively accurate for nothing: as u -> 0 the answer is u
     times a P that rounds to exactly 1.
  2. The DERIVED cap .prof/verify_softplus.c's B1 row asserts - the fit
     error with float32 coefficients, plus a forward running bound on
     the Horner roundings, plus the final multiply.  Not fitted to any
     measurement.
  3. An exhaustive float32 sweep per degree, so "degree 8 is where
     float32 stops paying" is a measurement and not a claim.

Read/measure only.  Run from the engine tree root:
    python3 .prof/sp_fit.py
"""
import struct

import mpmath as mp
import numpy as np

mp.mp.dps = 40
EPS = mp.mpf(2) ** -24


def P(u):
    u = mp.mpf(u)
    return mp.mpf(1) if u == 0 else mp.log1p(u) / u


def f32(x):
    return struct.unpack('f', struct.pack('f', float(x)))[0]


def remez(deg, lo=mp.mpf(0), hi=mp.mpf(1), iters=60):
    """Exchange algorithm on deg+2 alternating extrema."""
    n = deg + 2
    xs = sorted((lo + hi) / 2 + (hi - lo) / 2 * mp.cos(mp.pi * k / (n - 1))
                for k in range(n))
    c = None
    for _ in range(iters):
        A = mp.matrix(n, n)
        b = mp.matrix(n, 1)
        for i, x in enumerate(xs):
            for j in range(deg + 1):
                A[i, j] = x ** j
            A[i, deg + 1] = (-1) ** i
            b[i] = P(x)
        sol = mp.lu_solve(A, b)
        c = [sol[j] for j in range(deg + 1)]

        def err(x):
            p = mp.mpf(0)
            for j in reversed(range(deg + 1)):
                p = p * x + c[j]
            return p - P(x)

        m = 20001
        grid = [lo + (hi - lo) * mp.mpf(k) / (m - 1) for k in range(m)]
        e = [err(x) for x in grid]
        cand = ([grid[0]]
                + [grid[k] for k in range(1, m - 1)
                   if (e[k] - e[k - 1]) * (e[k + 1] - e[k]) <= 0]
                + [grid[-1]])
        sel = []
        for x in cand:
            if sel and mp.sign(err(x)) == mp.sign(err(sel[-1])):
                if abs(err(x)) > abs(err(sel[-1])):
                    sel[-1] = x
            else:
                sel.append(x)
        while len(sel) > n:
            sel = sel[1:] if abs(err(sel[0])) < abs(err(sel[-1])) else sel[:-1]
        if len(sel) < n:
            break
        xs = sel
    return [f32(v) for v in c]


def cap_of(c):
    """fit error (float32 coefficients, exact arithmetic)
       + forward running bound on the float32 Horner
       + the final u*p rounding."""
    deg = len(c) - 1
    cm = [mp.mpf(v) for v in c]
    worst_fit = mp.mpf(0)
    worst_run = mp.mpf(0)
    for k in range(200001):
        u = mp.mpf(k) / 200000
        p = mp.mpf(0)
        for cc in reversed(cm):
            p = p * u + cc
        ref = P(u)
        worst_fit = max(worst_fit, abs(p - ref) / ref)
        e = mp.mpf(0)
        pk = cm[deg]
        for j in range(deg - 1, -1, -1):
            prod = pk * u
            e = e * u + EPS * abs(prod)      # multiply: propagate, round
            pk = prod + cm[j]
            e = e + EPS * abs(pk)            # add: round
        if pk != 0:
            worst_run = max(worst_run, e / abs(pk))
    return worst_fit, worst_run, worst_fit + worst_run + EPS


def sweep(c):
    """Exhaustive: every float32 u in [2^-24, 1], float32 Horner against
    a float64 log1p (which is 9 decades tighter than what is measured)."""
    deg = len(c) - 1
    lo = np.float32(2.0 ** -24).view(np.int32)
    hi = np.float32(1.0).view(np.int32)
    worst = 0.0
    worst_u = 0.0
    step = 1 << 22
    for base in range(int(lo), int(hi) + 1, step):
        n = min(step, int(hi) + 1 - base)
        u = np.arange(base, base + n, dtype=np.int32).view(np.float32)
        p = np.full(n, np.float32(c[deg]))
        for j in reversed(range(deg)):
            p = (p * u + np.float32(c[j])).astype(np.float32)
        got = (u * p).astype(np.float32)
        ref = np.log1p(u.astype(np.float64))
        rel = np.abs(got.astype(np.float64) - ref) / ref
        i = int(rel.argmax())
        if rel[i] > worst:
            worst, worst_u = float(rel[i]), float(u[i])
    return worst, worst_u


for deg in range(6, 11):
    c = remez(deg)
    w, wu = sweep(c)
    fit, run, cap = cap_of(c)
    print("deg %2d  exhaustive rel %.4e @u=%.9g   cap %.4e "
          "(fit %.4e + horner %.4e + mul %.4e)"
          % (deg, w, wu, float(cap), float(fit), float(run), float(EPS)))
    print("   " + ", ".join("%.9ff" % v for v in c))
