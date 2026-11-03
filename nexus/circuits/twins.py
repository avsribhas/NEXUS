import copy
import math
import numpy as np
from nexus.circuits.probe import measure_circuit
from nexus.score.recovery import CONTINUOUS_FIELDS

def perturb(spec, rng, circuit_id, sd=0.25) -> dict:
    """Copy a circuit with every capacitance, leak and conductance multiplied by its own random log-normal factor."""
    t = copy.deepcopy(spec)
    t["circuit_id"] = circuit_id
    for n in t["neurons"]:
        n["C_nF"] *= float(np.exp(rng.normal(0.0, sd)))
        n["g_leak_nS"] *= float(np.exp(rng.normal(0.0, sd)))
    for g in t["graded"]:
        g["g_nS"] *= float(np.exp(rng.normal(0.0, sd)))
    for g in t["gap"]:
        g["g_nS"] *= float(np.exp(rng.normal(0.0, sd)))
    return t

def _scaled(value, field, config) -> float:
    """A field value on its scoring scale."""
    if config["fields"][field]["scale"] == "log10":
        return math.log10(value) if value > 0 else -math.inf
    return float(value)

def datasheet_distance(a, b, config) -> float:
    """Largest difference over gates and continuous fields, as a fraction of the prior range; infinite if the gates, triggers or priorities differ."""
    if len(a) != len(b):
        return math.inf
    by_trigger = {d["trigger_channel"]: d for d in b}
    worst = 0.0
    for d in a:
        e = by_trigger.get(d["trigger_channel"])
        if e is None or e["priority"] != d["priority"]:
            return math.inf
        for f in CONTINUOUS_FIELDS:
            low, high = config["fields"][f]["range"]
            x = _scaled(d[f], f, config)
            y = _scaled(e[f], f, config)
            if not (math.isfinite(x) and math.isfinite(y)):
                return math.inf
            worst = max(worst, abs(x - y) / (high - low))
    return worst

def find_twin(spec, roles, datasheets, config, rng, workdir, circuit_id, tries=20, tol=0.1, sd=0.25) -> dict:
    """Perturb the circuit until its measured datasheets match within tol, keeping the closest accepted candidate if none does."""
    best = None
    for k in range(tries):
        cand = perturb(spec, rng, circuit_id, sd)
        m = measure_circuit(cand, roles, workdir / f"twin_{k}")
        if not m["accepted"]:
            continue
        dist = datasheet_distance(datasheets, m["datasheets"], config)
        if best is None or dist < best["distance"]:
            best = {"spec": cand, "measurement": m, "distance": float(dist), "tries": k + 1}
        if dist <= tol:
            return dict(best, found=True)
    if best is None:
        return {"spec": None, "measurement": None, "distance": math.inf, "tries": tries, "found": False}
    return dict(best, found=False, tries=tries)
