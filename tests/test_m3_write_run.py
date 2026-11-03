from pathlib import Path
import numpy as np
import pytest
from pynwb import NWBHDF5IO
from nexus.circuits.batch import BatchRunner
from nexus.circuits.build import build_circuit, load_motifs
from nexus.circuits.corrupt import corrupt_wiring
from nexus.circuits.graded import write_network
from nexus.circuits.probe import measure_circuit
from nexus.circuits.write_run import write_run
from nexus.format.checker import check_file
from nexus.format.ground_truth import read_ground_truth
from nexus.format.inputs import read_inputs
from nexus.format.pokes import read_pokes
from nexus.perturb.protocol import DT, N_KEEP, T_TRIAL, currents_sha256, make_pokes, run_trials, trial_currents, trial_table

ROOT = Path(__file__).resolve().parents[1]
COMMON, MOTIFS = load_motifs(ROOT / "configs" / "motifs")
RATES = {"missing_edge": 0.1, "false_edge": 0.05, "sign_flip": 0.05, "weight_sd_log": 0.3}

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    work = tmp_path_factory.mktemp("run")
    spec, roles = build_circuit("release", COMMON, MOTIFS, np.random.default_rng(0), "wr")
    measurement = measure_circuit(spec, roles, work / "probe")
    stored, rows = corrupt_wiring(spec, RATES, np.random.default_rng(1))
    pokes = make_pokes(spec, roles, np.random.default_rng(2))
    trials = trial_table(pokes)[:4]
    ids = [n["id"] for n in spec["neurons"]]
    write_network(spec, work / "wr.net.nml")
    runner = BatchRunner(work / "wr.net.nml", ids, DT, T_TRIAL)
    hashes = []
    v = run_trials(runner, spec, ids, roles, pokes, trials, 7, batch_size=4, hashes=hashes)
    path = work / "wr.nwb"
    write_run(path, spec, roles, measurement, stored, rows, pokes, trials, v, hashes, 7, "", "wr", "wr_a0")
    return {"path": path, "spec": spec, "roles": roles, "measurement": measurement, "rows": rows, "pokes": pokes, "trials": trials, "ids": ids, "v": v, "hashes": hashes}

def test_checker_accepts_the_file(run):
    assert check_file(run["path"], allow_null_cost=True) == []

def test_voltages_round_trip(run):
    with NWBHDF5IO(str(run["path"]), "r") as io:
        f = io.read()
        data = f.acquisition["membrane_potential"].data[:]
        assert data.shape == (4 * N_KEEP, len(run["ids"]))
        back = data.reshape(4, N_KEEP, len(run["ids"])).transpose(0, 2, 1)
        assert np.array_equal(back, run["v"])

def test_ground_truth_round_trip(run):
    with NWBHDF5IO(str(run["path"]), "r") as io:
        gt = read_ground_truth(io.read())
        assert gt["switch_type"] == "release"
        assert gt["animal_id"] == "wr_a0"
        assert [d["gate_id"] for d in gt["datasheets"]] == [d["gate_id"] for d in run["measurement"]["datasheets"]]
        assert abs(gt["datasheets"][0]["trigger_gain"] - run["measurement"]["datasheets"][0]["trigger_gain"]) < 1e-9
        assert len(gt["wiring_corruption"]) == len(run["rows"])

def test_pokes_and_trials(run):
    with NWBHDF5IO(str(run["path"]), "r") as io:
        f = io.read()
        pokes = read_pokes(f)
        n_trials = len(f.trials)
        assert n_trials == 4
        assert len(pokes) == 4
        for p, t in zip(pokes, run["trials"]):
            assert p["target_neuron"] == run["pokes"][t["poke_index"]]["target"]

def test_stored_inputs_match_what_the_simulator_got(run):
    with NWBHDF5IO(str(run["path"]), "r") as io:
        stored = read_inputs(io.read())
        assert [h.decode() if isinstance(h, bytes) else str(h) for h in stored["currents_sha256"]] == run["hashes"]
        regenerated = [currents_sha256(trial_currents(run["spec"], run["ids"], run["roles"], run["pokes"][t["poke_index"]], t["state_index"], t["seed_index"], int(stored["circuit_seed"]))) for t in run["trials"]]
        assert regenerated == run["hashes"]
