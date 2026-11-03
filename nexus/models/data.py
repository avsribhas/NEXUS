import json
import numpy as np
from pynwb import NWBHDF5IO
from nexus.format.ground_truth import read_ground_truth
from nexus.format.inputs import read_inputs
from nexus.perturb.protocol import KEEP, N_KEEP, N_SEEDS, POKE_START, STATES_MV

SAMPLE_MS = (KEEP[1] - KEEP[0]) / N_KEEP
BASE_MS = (POKE_START - 100.0, POKE_START)
RESP_MS = (POKE_START, POKE_START + 100.0)

def _text(x) -> str:
    """Decode bytes from HDF5 into str."""
    return x.decode() if isinstance(x, bytes) else str(x)

def load_animal(path) -> dict:
    """Read one animal: circuit, roles, corrupted diagram, pokes, trial table, voltages and planted answers."""
    with NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        inputs = read_inputs(f)
        gt = read_ground_truth(f)
        ids = [_text(x) for x in inputs["neuron_ids"]]
        data = np.asarray(f.acquisition["membrane_potential"].data[:], dtype=np.float32)
        spec = json.loads(_text(f.get_scratch("circuit_spec_json")))
        roles = json.loads(_text(f.get_scratch("roles_json")))
        diagram = json.loads(_text(f.get_scratch("wiring_diagram_json")))
        pokes = json.loads(_text(f.get_scratch("pokes_json")))
        poke_index = np.asarray(inputs["trial_poke_index"], dtype=int)
        state_index = np.asarray(inputs["trial_state_index"], dtype=int)
        seed_index = np.asarray(inputs["trial_seed_index"], dtype=int)
    n = len(poke_index)
    voltages = data.reshape(n, N_KEEP, len(ids)).transpose(0, 2, 1)
    return {"ids": ids, "spec": spec, "roles": roles, "diagram": diagram, "pokes": pokes, "poke_index": poke_index, "state_index": state_index, "seed_index": seed_index, "voltages": voltages, "switch_type": gt["switch_type"], "is_decoy": gt["is_decoy"], "datasheets": gt["datasheets"], "circuit_id": gt["base_circuit_id"], "animal_id": gt["animal_id"]}

def window_mean(voltages, t0, t1) -> np.ndarray:
    """Mean voltage over [t0, t1) ms of trial time, along the last axis."""
    i0 = int(round((t0 - KEEP[0]) / SAMPLE_MS)); i1 = int(round((t1 - KEEP[0]) / SAMPLE_MS)); return voltages[..., i0:i1].mean(axis=-1)

def poke_responses(animal) -> np.ndarray:
    """Response of every neuron to every poke, as an array (pokes, states, seeds, neurons); nan where a trial is missing."""
    v = animal["voltages"]
    r = window_mean(v, RESP_MS[0], RESP_MS[1]) - window_mean(v, BASE_MS[0], BASE_MS[1])   # shape (trials, neurons)
    out = np.full((len(animal["pokes"]), len(STATES_MV), N_SEEDS, len(animal["ids"])), np.nan, dtype=np.float32)
    out[animal["poke_index"], animal["state_index"], animal["seed_index"]] = r
    return out
