import math
import numpy as np
from nexus.circuits.measure import crossing, time_to_fraction
from nexus.models.data import BASE_MS, RESP_MS, SAMPLE_MS, poke_responses
from nexus.perturb.protocol import KEEP, PAIRED, PAIRED_STEP

FEATURES = ("log_gain_30", "log_gain_20", "log_gain_10", "log_r_off", "threshold_mV", "tau_on_ms", "tau_off_ms", "tau_cell_ms", "log_tau_on_ratio", "log_tau_off_ratio", "reach", "log_state_ratio")
N_FEATURES = 2 * len(FEATURES)
EPS = 1e-3

def _log_ratio(a, b) -> float:
    """log10 of |a| / |b|, with a small floor on both."""
    return math.log10((abs(a) + EPS) / (abs(b) + EPS))

def mean_traces(animal) -> np.ndarray:
    """Seed-averaged voltage traces, shape (pokes, states, neurons, time)."""
    v = animal["voltages"]
    out = np.zeros((len(animal["pokes"]), int(animal["state_index"].max()) + 1, v.shape[1], v.shape[2]))
    count = np.zeros(out.shape[:2])
    for t in range(v.shape[0]):
        p = animal["poke_index"][t]; s = animal["state_index"][t]
        out[p, s] += v[t]; count[p, s] += 1
    return out / np.maximum(count, 1)[:, :, None, None]

def _index(pokes, **match) -> list:
    """Indices of pokes whose fields equal every given value."""
    return [i for i, p in enumerate(pokes) if all(p.get(k) == val for k, val in match.items())]

def tau_cell(animal, traces) -> float:
    """Median 63% rise time, in ms, of neurons poked directly by the 300 ms poke, at state 0."""
    ids = animal["ids"]; times = []
    for i in _index(animal["pokes"], type="long"):
        k = ids.index(animal["pokes"][i]["target"])
        t = time_to_fraction(traces[i, 0, k], SAMPLE_MS, RESP_MS[0] - KEEP[0], RESP_MS[0] - KEEP[0] + 300.0)
        if math.isfinite(t) and t > 0: times.append(t)
    return float(np.median(times)) if times else math.nan

def candidate_features(animal, trigger, traces=None, responses=None) -> np.ndarray:
    """Feature vector for one candidate trigger: values with nan replaced by 0, followed by one is-missing flag per value."""
    traces = mean_traces(animal) if traces is None else traces
    responses = poke_responses(animal) if responses is None else responses
    R = np.nanmean(responses, axis=2)   # shape (pokes, states, neurons)
    ids = animal["ids"]; pokes = animal["pokes"]; roles = animal["roles"]
    outputs = [ids.index(o) for o in roles["outputs"]]
    others = [i for i, nid in enumerate(ids) if nid not in set(roles["inputs"]) | {roles["state"]}]
    values = dict.fromkeys(FEATURES, math.nan)
    best = None   # will hold (abs log gain, k_name, o_index, r_off, {level: r_on})
    for k_name in roles["inputs"]:
        if k_name == trigger: continue
        off = _index(pokes, type="dep_large", target=k_name)
        if not off: continue
        for o in outputs:
            r_off = float(R[off[0], 0, o])
            r_on = {}
            for level in PAIRED["hold_levels_mV"]:
                on = _index(pokes, type=PAIRED["type"], target=k_name, hold=trigger, hold_mV=float(level))
                if on: r_on[level] = float(np.mean([R[i, 0, o] for i in on]))
            if 30.0 in r_on:
                score = abs(_log_ratio(r_on[30.0], r_off))
                if best is None or score > best[0]: best = (score, k_name, o, r_off, r_on)
    if best is not None:
        _, k_name, o, r_off, r_on = best
        for level in (30.0, 20.0, 10.0):
            if level in r_on: values[f"log_gain_{int(level)}"] = _log_ratio(r_on[level], r_off)
        values["log_r_off"] = math.log10(abs(r_off) + EPS)
        levels = [0.0] + [lv for lv in (10.0, 20.0, 30.0) if lv in r_on]
        mags = [abs(r_off)] + [abs(r_on[lv]) for lv in levels[1:]]
        if len(levels) >= 2: values["threshold_mV"] = crossing(levels, mags, 0.5 * (mags[0] + mags[-1]))
        on30 = _index(pokes, type=PAIRED["type"], target=k_name, hold=trigger, hold_mV=30.0)
        off15 = _index(pokes, type="dep_large", target=k_name)
        if on30 and off15:
            diff = np.abs(np.mean([R[i, 0, :] for i in on30], axis=0) - R[off15[0], 0, :])
            values["reach"] = float(np.mean([diff[i] > 1.0 for i in others])) if others else math.nan
        values["log_state_ratio"] = _log_ratio(R[off15[0], -1, o], R[off15[0], 0, o]) if off15 else math.nan
        steps = _index(pokes, type=PAIRED_STEP["type"], target=trigger, hold=k_name)
        if steps:
            tr = np.mean([traces[i, 0, o] for i in steps], axis=0)
            t_on = RESP_MS[0] - KEEP[0]; t_off = t_on + PAIRED_STEP["duration_ms"]; t_end = KEEP[1] - KEEP[0]
            values["tau_on_ms"] = time_to_fraction(tr, SAMPLE_MS, t_on, t_off)
            values["tau_off_ms"] = time_to_fraction(tr, SAMPLE_MS, t_off, t_end)
    tc = tau_cell(animal, traces); values["tau_cell_ms"] = tc
    for name, src in (("log_tau_on_ratio", "tau_on_ms"), ("log_tau_off_ratio", "tau_off_ms")):
        a = values[src]
        values[name] = math.log10(a / tc) if math.isfinite(a) and math.isfinite(tc) and a > 0 and tc > 0 else math.nan
    x = np.array([values[f] for f in FEATURES], dtype=float)
    missing = ~np.isfinite(x)
    return np.concatenate([np.where(missing, 0.0, x), missing.astype(float)])
