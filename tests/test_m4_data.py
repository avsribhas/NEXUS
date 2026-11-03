from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.batch import BatchRunner
from nexus.circuits.build import build_circuit, load_motifs
from nexus.circuits.corrupt import corrupt_wiring
from nexus.circuits.graded import write_network
from nexus.circuits.probe import measure_circuit
from nexus.circuits.write_run import write_run
from nexus.models.data import BASE_MS, RESP_MS, SAMPLE_MS, load_animal, poke_responses, window_mean
from nexus.perturb.protocol import DT, KEEP, N_KEEP, T_TRIAL, make_pokes, run_trials, trial_table

ROOT = Path(__file__).resolve().parents[1]
COMMON, MOTIFS = load_motifs(ROOT / "configs" / "motifs")
RATES = {"missing_edge": 0.1, "false_edge": 0.05, "sign_flip": 0.05, "weight_sd_log": 0.3}

@pytest.fixture(scope="module")
def made(tmp_path_factory):
    work = tmp_path_factory.mktemp("data")
    spec, roles = build_circuit("release", COMMON, MOTIFS, np.random.default_rng(0), "dt")
    measurement = measure_circuit(spec, roles, work / "probe")
    stored, rows = corrupt_wiring(spec, RATES, np.random.default_rng(1))
    pokes = make_pokes(spec, roles, np.random.default_rng(2))
    trials = [t for t in trial_table(pokes) if t["poke_index"] in (0, 10)]
    ids = [n["id"] for n in spec["neurons"]]
    write_network(spec, work / "dt.net.nml")
    runner = BatchRunner(work / "dt.net.nml", ids, DT, T_TRIAL)
    hashes = []
    v = run_trials(runner, spec, ids, roles, pokes, trials, 7, batch_size=36, hashes=hashes)
    path = work / "dt.nwb"
    write_run(path, spec, roles, measurement, stored, rows, pokes, trials, v, hashes, 7, "", "dt", "dt_a0")
    return {"path": path, "spec": spec, "roles": roles, "stored": stored, "pokes": pokes, "trials": trials, "ids": ids, "v": v}

def test_window_mean():
    v = np.zeros((1, 2, N_KEEP), dtype=np.float32)
    i0 = int(round((RESP_MS[0] - KEEP[0]) / SAMPLE_MS))
    v[..., i0:] = 3.0
    assert np.allclose(window_mean(v, RESP_MS[0], RESP_MS[1]), 3.0)
    assert np.allclose(window_mean(v, BASE_MS[0], BASE_MS[1]), 0.0)

def test_load_animal(made):
    a = load_animal(made["path"])
    assert a["ids"] == made["ids"]
    assert a["spec"] == made["spec"]
    assert a["diagram"] == made["stored"]
    assert a["pokes"] == made["pokes"]
    assert a["switch_type"] == "release" and a["animal_id"] == "dt_a0" and a["circuit_id"] == "dt"
    assert np.array_equal(a["voltages"], made["v"])
    assert list(a["poke_index"]) == [t["poke_index"] for t in made["trials"]]

def test_poke_responses(made):
    a = load_animal(made["path"])
    r = poke_responses(a)
    assert r.shape == (len(made["pokes"]), 6, 3, len(made["ids"]))
    assert np.all(np.isfinite(r[0])) and np.all(np.isfinite(r[10]))
    assert np.all(np.isnan(r[1]))
    k = made["ids"].index(made["pokes"][10]["target"])
    assert np.all(r[10, :, :, k] > 5.0)
