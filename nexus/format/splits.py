import json
import numpy as np

SCHEMA_VERSION = "0.1.0"
KINDS = ("circuits", "neurons", "nudge_types", "states")
TRIAL_KEYS = {
    "circuits": "circuit_id",
    "neurons": "neuron_id",
    "nudge_types": "nudge_type",
    "states": "state"
}

def _split_units(units, rng, test_fraction, val_fraction):
    ordered = sorted(units)
    n = len(ordered)
    if len(set(ordered)) != n:
        raise ValueError(f"Duplicate unit IDs found: {ordered}")
    if n < 3:
        raise ValueError("At least 3 units are needed")
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_test + n_val >= n:
        raise ValueError("No training units would remain")
    perm = rng.permutation(n)
    test = sorted(ordered[i] for i in perm[:n_test])
    val = sorted(ordered[i] for i in perm[n_test:n_test + n_val])
    train = sorted(ordered[i] for i in perm[n_test + n_val:])
    return {"train": train, "val": val, "test": test}

def make_split_manifest(circuit_ids, neurons_by_circuit, nudge_types, states, seed, test_fraction=0.2, val_fraction=0.2):
    rng = np.random.default_rng(seed)
    circuits = _split_units(circuit_ids, rng, test_fraction, val_fraction)
    neurons = {c: _split_units(neurons_by_circuit[c], rng, test_fraction, val_fraction) for c in circuits["train"]}
    nudges = _split_units(nudge_types, rng, test_fraction, val_fraction)
    state_split = _split_units(states, rng, test_fraction, val_fraction)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "val_fraction": float(val_fraction),
        "circuits": circuits,
        "neurons": neurons,
        "nudge_types": nudges,
        "states": state_split
    }

def _side(manifest, kind, trial):
    unit = trial[TRIAL_KEYS[kind]]
    if kind == "neurons":
        circuit = trial["circuit_id"]
        if circuit not in manifest["neurons"]:
            return None
        parts = manifest["neurons"][circuit]
    else:
        parts = manifest[kind]
    for side in ("train", "val", "test"):
        if unit in parts[side]:
            return side
    raise ValueError(f"Unit {unit} not found in {kind} split")

def trial_split(manifest, trial):
    sides = [_side(manifest, kind, trial) for kind in KINDS]
    return "test" if "test" in sides else "val" if "val" in sides else "train"

def held_out_kinds(manifest, trial):
    return [kind for kind in KINDS if _side(manifest, kind, trial) == "test"]

def write_manifest(manifest, path):
    with open(path, "w") as f:
        f.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

def read_manifest(path):
    with open(path, "r") as f:
        return json.load(f)
