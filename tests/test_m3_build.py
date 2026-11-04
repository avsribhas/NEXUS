from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.batch import BatchRunner
from nexus.circuits.build import build_circuit, drive_to_current, load_motifs, make_twin
from nexus.circuits.graded import validate_spec, write_network

MOTIF_DIR = Path(__file__).resolve().parents[1] / "configs" / "motifs"
COMMON, MOTIFS = load_motifs(MOTIF_DIR)
NAMES = ["release", "gain", "winner_take_all", "global_shutoff", "decoy_adaptation", "decoy_shunt", "plain", "decoy_bistable"]

def _build(name, seed):
    return build_circuit(name, COMMON, MOTIFS, np.random.default_rng(seed), f"c_{name}_{seed}")

def _neuron(spec, nid):
    return next(n for n in spec["neurons"] if n["id"] == nid)

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_circuits_are_valid(name, seed):
    spec, roles = _build(name, seed)
    assert validate_spec(spec) == []
    ids = {n["id"] for n in spec["neurons"]}
    assert set(MOTIFS[name]["neurons"]) <= ids
    assert roles["state"] == "S" and "S" in ids
    assert set(roles["inputs"]) <= ids and set(roles["outputs"]) <= ids
    assert set(roles["background"]) <= ids
    assert COMMON["background_n"][0] <= len(roles["background"]) <= COMMON["background_n"][1]

def test_all_motifs_load():
    assert sorted(MOTIFS) == sorted(NAMES) and COMMON["n_states"] == 6

def test_same_seed_same_circuit():
    assert _build("release", 7)[0] == _build("release", 7)[0] and _build("release", 7)[0] != _build("release", 8)[0]

@pytest.mark.parametrize("name", NAMES)
def test_background_never_feeds_the_motif(name):
    spec, roles = _build(name, 3)
    bg = set(roles["background"])
    for s in spec["graded"]:
        if s["pre"] in bg:
            assert s["post"] in bg
    for g in spec["gap"]:
        assert g["a"] in bg and g["b"] in bg

def test_weights_are_relative_to_the_receiving_cell():
    spec, roles = _build("release", 5)
    s = next(s for s in spec["graded"] if s["pre"] == "in" and s["post"] == "R")
    w = s["g_nS"] / _neuron(spec, "R")["g_leak_nS"]
    assert 0.3 <= w <= 0.6 and s["erev_mV"] == 0.0

def test_shunt_reverses_at_rest():
    spec, roles = _build("decoy_shunt", 2)
    s = next(s for s in spec["graded"] if s["pre"] == "trig")
    assert s["erev_mV"] == _neuron(spec, "O")["e_leak_mV"]

def test_drive_to_current():
    spec = {"circuit_id": "x", "neurons": [{"id": "a", "C_nF": 0.2, "g_leak_nS": 10.0, "e_leak_mV": -70.0}], "graded": [], "gap": []}
    assert abs(drive_to_current(spec, "a", 15.0) - 0.15) < 1e-12

def test_twin_scales_parameters():
    spec, roles = _build("gain", 4)
    twin = make_twin(spec, 2.0, "twin")
    assert twin["circuit_id"] == "twin"
    for n, m in zip(spec["neurons"], twin["neurons"]):
        assert abs(m["C_nF"] - 2.0 * n["C_nF"]) < 1e-12 and abs(m["g_leak_nS"] - 2.0 * n["g_leak_nS"]) < 1e-12 and m["e_leak_mV"] == n["e_leak_mV"]
    for s, t in zip(spec["graded"], twin["graded"]):
        assert abs(t["g_nS"] - 2.0 * s["g_nS"]) < 1e-12 and t["vth_mV"] == s["vth_mV"]

def test_twin_has_identical_voltages(tmp_path):
    spec, roles = _build("release", 6)
    twin = make_twin(spec, 2.0, "twin")
    ids = [n["id"] for n in spec["neurons"]]
    write_network(spec, tmp_path / "a" / "c.net.nml")
    write_network(twin, tmp_path / "b" / "c.net.nml")
    ra = BatchRunner(tmp_path / "a" / "c.net.nml", ids, delta_t=0.1, t_max=200.0)
    rb = BatchRunner(tmp_path / "b" / "c.net.nml", ids, delta_t=0.1, t_max=200.0)
    n = ra.n_steps
    ca = np.zeros((1, len(ids), n)); cb = np.zeros((1, len(ids), n))
    for nid, drive in (("in", 15.0), ("trig", 20.0)):
        i = ids.index(nid)
        ca[0, i, 500:1500] = drive_to_current(spec, nid, drive)
        cb[0, i, 500:1500] = drive_to_current(twin, nid, drive)
    assert np.max(np.abs(ra.run(ca) - rb.run(cb))) < 1e-3
