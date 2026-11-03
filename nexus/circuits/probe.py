import math
from pathlib import Path
import numpy as np
from nexus.circuits.batch import BatchRunner
from nexus.circuits.build import drive_to_current
from nexus.circuits.graded import write_network
from nexus.circuits.measure import classify_ratio, crossing, gate_ratio, priority_ranks, time_to_fraction, window_mean

DT = 0.2
T_MAX = 5000.0
D_TEST = 15.0
U_MAX = 30.0
U_GRID = np.linspace(0.0, 30.0, 13)
PULSE = (2400.0, 2500.0)
TEST = (2450.0, 2500.0)
TIMING_ON = 1500.0
TIMING_OFF = 3200.0
REACH_MV = 1.0
V_LOW = -150.0
V_HIGH = 80.0

def make_runner(spec, workdir) -> tuple:
    """Write the circuit to NeuroML and return a compiled runner and the neuron ids."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"{spec['circuit_id']}.net.nml"; write_network(spec, path)
    ids = [n["id"] for n in spec["neurons"]]
    return BatchRunner(path, ids, DT, T_MAX), ids

def _currents(spec, ids, drives) -> np.ndarray:
    """Current in nA for each neuron and time step; drives are (neuron_id, drive_mV, t0, t1) tuples."""
    n_steps = int(round(T_MAX / DT)) + 1
    out = np.zeros((len(ids), n_steps), dtype=np.float32)
    for nid, mv, t0, t1 in drives:
        i0 = int(round(t0 / DT)); i1 = int(round(t1 / DT))
        out[ids.index(nid), i0:i1] += drive_to_current(spec, nid, mv)
    return out

def _response(v_pulse, v_rest) -> np.ndarray:
    """Mean test-window voltage with the pulse minus without it, per neuron."""
    return window_mean(v_pulse, DT, TEST[0], TEST[1]) - window_mean(v_rest, DT, TEST[0], TEST[1])

def _pathway_trials(spec, ids, trigger_drives, test_input) -> list:
    """A pulse trial and a matching no-pulse trial with the given drives held on."""
    held = [(nid, mv, 0.0, T_MAX) for nid, mv in trigger_drives]
    return [_currents(spec, ids, held + [(test_input, D_TEST, PULSE[0], PULSE[1])]), _currents(spec, ids, held)]

def measure_circuit(spec, roles, workdir) -> dict:
    """Run the probe protocol on one circuit; return its datasheets, ratios and acceptance."""
    runner, ids = make_runner(spec, workdir)
    gates = roles["gates"]; candidates = roles["candidates"]; all_triggers = [g["trigger"] for g in gates]
    trials = []
    for g in gates:
        for u in U_GRID:
            trials.extend(_pathway_trials(spec, ids, [(g["trigger"], u)], g["test_input"]))
        trials.append(_currents(spec, ids, [(g["test_input"], D_TEST, 0.0, T_MAX), (g["trigger"], U_MAX, TIMING_ON, TIMING_OFF)]))
        trials.extend(_pathway_trials(spec, ids, [(t, U_MAX) for t in all_triggers], g["test_input"]))
    for c in candidates:
        trials.extend(_pathway_trials(spec, ids, [(c["trigger"], 0.0)], c["test_input"]))
        trials.extend(_pathway_trials(spec, ids, [(c["trigger"], U_MAX)], c["test_input"]))
    v = runner.run(np.stack(trials))
    if not np.all(np.isfinite(v)) or v.min() < V_LOW or v.max() > V_HIGH:
        return {"datasheets": [], "gate_ratios": [], "cross_ratios": {}, "accepted": False, "reason": "voltage out of range"}
    skip = set(roles["inputs"]) | {roles["state"]}; others = [i for i, nid in enumerate(ids) if nid not in skip]
    datasheets = []; gate_ratios = []; effects = []
    for k, g in enumerate(gates):
        base = 29 * k
        o = ids.index(g["output"])
        resp = [_response(v[base + 2 * j], v[base + 2 * j + 1]) for j in range(13)]
        r = np.array([x[o] for x in resp])
        ratio = gate_ratio(r[0], r[-1])
        threshold = crossing(U_GRID, r, 0.5 * (r[0] + r[-1]))
        changed = [i for i in others if abs(resp[-1][i] - resp[0][i]) > REACH_MV]
        reach = len(changed) / len(others)
        tau_set = changed if changed else [o]
        tau_ref = float(np.median([1000.0 * spec["neurons"][i]["C_nF"] / spec["neurons"][i]["g_leak_nS"] for i in tau_set]))
        vt = v[base + 26, o]
        tau_on = time_to_fraction(vt, DT, TIMING_ON, TIMING_OFF)
        tau_off = time_to_fraction(vt, DT, TIMING_OFF, T_MAX)
        r_all = _response(v[base + 27], v[base + 28])[o]
        effect = abs(math.log(max(gate_ratio(r[0], r_all), 1e-12)))
        datasheets.append({
            "gate_id": g["gate_id"], "trigger_channel": g["trigger"], "trigger_gain": float(ratio), "threshold": float(threshold), "threshold_unit": "mV",
            "tau_on_over_tau_ref": float(tau_on / tau_ref), "tau_off_over_tau_ref": float(tau_off / tau_ref), "tau_ref_definition": "median_membrane_time_constant",
            "reach": float(reach), "reach_criterion": "test-pulse response changes by more than 1.0 mV", "priority": 0, "cost_atp_per_flip": None
        })
        gate_ratios.append(ratio); effects.append(effect)
    ranks = priority_ranks(effects)
    for k, ds in enumerate(datasheets):
        ds["priority"] = ranks[k]
    cross_ratios = {}
    for k, c in enumerate(candidates):
        base = 29 * len(gates) + 4 * k
        o = ids.index(c["output"])
        r_off = _response(v[base], v[base + 1])[o]; r_on = _response(v[base + 2], v[base + 3])[o]
        cross_ratios[f"{c['trigger']}>{c['test_input']}>{c['output']}"] = float(gate_ratio(r_off, r_on))
    if gates:
        bad = [g["gate_id"] for g, x in zip(gates, gate_ratios) if classify_ratio(x) != "switch"]
        for d in datasheets:
            if any(math.isnan(d[key]) for key in ("threshold", "tau_on_over_tau_ref", "tau_off_over_tau_ref")) and d["gate_id"] not in bad:
                bad.append(d["gate_id"])
        accepted = not bad
        reason = "ok" if accepted else "gate not a clear switch: " + ", ".join(bad)
    else:
        bad = [key for key, x in cross_ratios.items() if classify_ratio(x) != "none"]
        accepted = not bad
        reason = "ok" if accepted else "candidate changes the response: " + ", ".join(bad)
    return {"datasheets": datasheets, "gate_ratios": [float(x) for x in gate_ratios], "cross_ratios": cross_ratios, "accepted": bool(accepted), "reason": reason}
