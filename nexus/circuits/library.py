from pathlib import Path
import numpy as np
from nexus.circuits.batch import BatchRunner
from nexus.circuits.build import build_circuit
from nexus.circuits.corrupt import corrupt_wiring
from nexus.circuits.graded import write_network
from nexus.circuits.probe import V_HIGH, V_LOW, measure_circuit
from nexus.circuits.write_run import write_run
from nexus.perturb.protocol import DT, T_TRIAL, make_pokes, run_trials, trial_table

def sample_accepted(motif_name, common, motifs, seed, workdir, max_tries=10) -> dict:
    """Draw circuits from a motif until one passes the probe; give up after max_tries."""
    workdir = Path(workdir); reasons = []
    circuit_id = f"{motif_name}_{seed:05d}"
    for i in range(max_tries):
        spec, roles = build_circuit(motif_name, common, motifs, np.random.default_rng([seed, i]), circuit_id)
        m = measure_circuit(spec, roles, workdir / f"{circuit_id}_try{i}")
        reasons.append(m["reason"])
        if m["accepted"]:
            return {"spec": spec, "roles": roles, "measurement": m, "tries": i + 1, "reasons": reasons}
    return {"spec": None, "roles": None, "measurement": None, "tries": max_tries, "reasons": reasons}

def generate_animal(motif_name, common, motifs, rates, seed, workdir, out_path, max_tries=10, batch_size=100) -> dict:
    """Make one accepted circuit, simulate all its trials for one wiring-corruption draw, and write the NWB file."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    s = sample_accepted(motif_name, common, motifs, seed, workdir, max_tries)
    if s["spec"] is None: return {"accepted": False, "tries": s["tries"], "reasons": s["reasons"]}
    spec, roles, m = s["spec"], s["roles"], s["measurement"]
    stored, rows = corrupt_wiring(spec, rates, np.random.default_rng([seed, 1000]))
    pokes = make_pokes(spec, roles, np.random.default_rng([seed, 2000]))
    trials = trial_table(pokes)
    ids = [n["id"] for n in spec["neurons"]]
    nml = workdir / f"{spec['circuit_id']}.trials.net.nml"; write_network(spec, nml)
    runner = BatchRunner(nml, ids, DT, T_TRIAL)
    hashes = []
    v = run_trials(runner, spec, ids, roles, pokes, trials, seed, batch_size, hashes)
    bad = int(np.sum(~np.all(np.isfinite(v) & (v >= V_LOW) & (v <= V_HIGH), axis=(1, 2))))
    animal_id = f"{spec['circuit_id']}_a0"
    write_run(out_path, spec, roles, m, stored, rows, pokes, trials, v, hashes, seed, "", spec["circuit_id"], animal_id)
    return {"accepted": True, "tries": s["tries"], "reasons": s["reasons"], "circuit_id": spec["circuit_id"], "animal_id": animal_id, "n_neurons": len(ids), "n_trials": len(trials), "n_bad_trials": bad, "datasheets": m["datasheets"], "gate_ratios": m["gate_ratios"], "cross_ratios": m["cross_ratios"]}
