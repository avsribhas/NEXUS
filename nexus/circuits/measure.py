import math
import numpy as np

def window_mean(v, dt, t0, t1):
    """Mean of v over [t0, t1) along the last axis."""
    v = np.asarray(v, dtype=float)
    i0 = int(round(t0 / dt))
    i1 = int(round(t1 / dt))
    return v[..., i0:i1].mean(axis=-1)

def pulse_response(v, dt, base, test):
    """Mean over the test window minus mean over the base window."""
    return window_mean(v, dt, test[0], test[1]) - window_mean(v, dt, base[0], base[1])

def crossing(xs, ys, level) -> float:
    """First x where ys reaches level, by linear interpolation; nan if never."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    d = ys - level
    if d[0] == 0: return float(xs[0])
    for i in range(len(d) - 1):
        if d[i + 1] == 0: return float(xs[i + 1])
        if d[i] * d[i + 1] < 0:
            return float(xs[i] + (xs[i + 1] - xs[i]) * d[i] / (d[i] - d[i + 1]))
    return math.nan

def time_to_fraction(v, dt, t_step, t_end, frac=0.632) -> float:
    """Time after t_step at which v first gets frac of the way from its level before t_step to its level before t_end; nan if flat or never."""
    v = np.asarray(v, dtype=float)
    v0 = float(window_mean(v, dt, t_step - 50.0, t_step))
    v1 = float(window_mean(v, dt, t_end - 50.0, t_end))
    if abs(v1 - v0) < 1e-6: return math.nan
    target = v0 + frac * (v1 - v0)
    i0 = int(round(t_step / dt))
    i1 = int(round(t_end / dt))
    seg = v[i0:i1] - target
    if v1 < v0: seg = -seg
    hits = np.nonzero(seg >= 0)[0]
    if len(hits) == 0: return math.nan
    j = int(hits[0])
    if j == 0: return 0.0
    a, b = seg[j - 1], seg[j]
    return float((j - 1 + (-a) / (b - a)) * dt)

def gate_ratio(r_off, r_on) -> float:
    """abs(r_on) / abs(r_off); inf when r_off is zero."""
    if abs(r_off) < 1e-9: return math.inf
    return abs(r_on) / abs(r_off)

def classify_ratio(ratio, switch_min=2.0, none_max=1.5) -> str:
    """"switch" for a change of at least switch_min fold either way, "none" for at most none_max fold, else "ambiguous"."""
    if ratio >= switch_min or ratio <= 1.0 / switch_min: return "switch"
    if 1.0 / none_max <= ratio <= none_max: return "none"
    return "ambiguous"

def priority_ranks(effects) -> list:
    """Rank 1 for the largest effect; ties keep input order."""
    order = sorted(range(len(effects)), key=lambda i: (-effects[i], i))
    ranks = [0] * len(effects)
    for r, i in enumerate(order): ranks[i] = r + 1
    return ranks
