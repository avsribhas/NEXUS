import hashlib
import zlib
import numpy as np
from nexus.circuits.build import drive_to_current

DT = 0.2
T_TRIAL = 2000.0
N_STEPS = int(round(T_TRIAL / DT)) + 1
STATES_MV = (0.0, 6.0, 12.0, 18.0, 24.0, 30.0)
N_SEEDS = 3
POKE_START = 1500.0
KEEP = (1400.0, 2000.0)
KEEP_EVERY = 5
N_KEEP = int(round((KEEP[1] - KEEP[0]) / DT)) // KEEP_EVERY
NOISE_MV = 10.0
POKE_TYPES = {"dep_small": (5.0, 100.0), "dep_large": (15.0, 100.0), "hyp": (-15.0, 100.0), "brief": (30.0, 20.0), "long": (10.0, 300.0)}
TARGETS_PER_TYPE = 10
PAIRED = {"type": "paired", "amplitude_mV": 15.0, "duration_ms": 100.0, "hold_levels_mV": (10.0, 20.0, 30.0)}
PAIRED_STEP = {"type": "paired_step", "amplitude_mV": 30.0, "duration_ms": 300.0, "hold_mV": 15.0}

def make_pokes(spec, roles, rng):
    """Fifty single pokes (five types on ten neurons that always include the inputs), a paired poke for every ordered pair of inputs at every hold level, and a paired step for every ordered pair."""
    eligible = [n["id"] for n in spec["neurons"] if n["id"] != roles["state"]]
    inputs_first = [str(x) for x in roles["inputs"] if x in eligible][:TARGETS_PER_TYPE]
    rest = [x for x in eligible if x not in inputs_first]
    need = TARGETS_PER_TYPE - len(inputs_first)
    if len(rest) >= need:
        fill = [str(x) for x in rng.choice(rest, size=need, replace=False)]
    else:
        fill = [str(x) for x in rng.choice(eligible, size=need, replace=True)]
    targets = inputs_first + fill
    pokes = []
    for t in POKE_TYPES:
        for j, target in enumerate(targets):
            pokes.append({"poke_id": f"{t}_{j}", "type": t, "target": target, "amplitude_mV": float(POKE_TYPES[t][0]), "start_ms": float(POKE_START), "stop_ms": float(POKE_START + POKE_TYPES[t][1]), "hold": "", "hold_mV": 0.0})
    inputs = [str(x) for x in roles["inputs"]]
    pairs = [(a, b) for a in inputs for b in inputs if a != b]
    for j, (hold, target, level) in enumerate((a, b, level) for a, b in pairs for level in PAIRED["hold_levels_mV"]):
        pokes.append({"poke_id": f"paired_{j}", "type": PAIRED["type"], "target": target, "amplitude_mV": float(PAIRED["amplitude_mV"]), "start_ms": float(POKE_START), "stop_ms": float(POKE_START + PAIRED["duration_ms"]), "hold": hold, "hold_mV": float(level)})
    for j, (hold, target) in enumerate(pairs):
        pokes.append({"poke_id": f"paired_step_{j}", "type": PAIRED_STEP["type"], "target": target, "amplitude_mV": float(PAIRED_STEP["amplitude_mV"]), "start_ms": float(POKE_START), "stop_ms": float(POKE_START + PAIRED_STEP["duration_ms"]), "hold": hold, "hold_mV": float(PAIRED_STEP["hold_mV"])})
    return pokes

def trial_table(pokes):
    """Every poke under every brain state with every noise seed."""
    trials = []
    for poke_index in range(len(pokes)):
        for state_index in range(len(STATES_MV)):
            for seed_index in range(N_SEEDS):
                trials.append({"trial_id": len(trials), "poke_index": poke_index, "state_index": state_index, "seed_index": seed_index})
    return trials

def trial_currents(spec, ids, roles, poke, state_index, seed_index, circuit_seed, noise=True):
    """Injected currents in nA for one trial: brain-state drive, the poke, any held input, and seeded noise."""
    out = np.zeros((len(ids), N_STEPS), dtype=np.float64)
    s = ids.index(roles["state"])
    out[s, :] += drive_to_current(spec, roles["state"], STATES_MV[state_index])
    k = ids.index(poke["target"])
    i0 = int(round(poke["start_ms"] / DT))
    i1 = int(round(poke["stop_ms"] / DT))
    out[k, i0:i1] += drive_to_current(spec, poke["target"], poke["amplitude_mV"])
    if poke.get("hold"):
        out[ids.index(poke["hold"]), :] += drive_to_current(spec, poke["hold"], poke["hold_mV"])
    if noise:
        rng = np.random.default_rng([int(circuit_seed), int(state_index), int(seed_index), zlib.crc32(poke["poke_id"].encode())])
        sd = np.array([NOISE_MV * drive_to_current(spec, nid, 1.0) for nid in ids])
        out += rng.standard_normal((len(ids), N_STEPS)) * sd[:, None]
    return out.astype(np.float32)

def currents_sha256(currents) -> str:
    """SHA-256 of one trial's current array, as float32 bytes in C order."""
    return hashlib.sha256(np.ascontiguousarray(currents, dtype=np.float32).tobytes()).hexdigest()

def run_trials(runner, spec, ids, roles, pokes, trials, circuit_seed, batch_size=100, hashes=None):
    """Run trials in equal-size batches and keep the voltages from KEEP[0] to KEEP[1] every KEEP_EVERY steps."""
    i0 = int(round(KEEP[0] / DT))
    i1 = int(round(KEEP[1] / DT))
    results = []
    for start in range(0, len(trials), batch_size):
        chunk = trials[start:start + batch_size]
        batch = [trial_currents(spec, ids, roles, pokes[t["poke_index"]], t["state_index"], t["seed_index"], circuit_seed) for t in chunk]
        if hashes is not None:
            hashes.extend(currents_sha256(c) for c in batch)
        while len(batch) < batch_size:
            batch.append(np.zeros((len(ids), N_STEPS), dtype=np.float32))
        v = runner.run(np.stack(batch))
        results.append(v[: len(chunk), :, i0:i1:KEEP_EVERY])
    return np.concatenate(results, axis=0).astype(np.float32)
