import copy
import json
import math
from pathlib import Path
import numpy as np
from nexus.circuits.build import build_circuit, load_motifs
from nexus.circuits.probe import measure_circuit
from nexus.circuits.twins import datasheet_distance, find_twin, perturb
from nexus.score.recovery import CONTINUOUS_FIELDS

ROOT = Path(__file__).resolve().parents[1]
COMMON, MOTIFS = load_motifs(ROOT / "configs" / "motifs")
CONFIG = json.loads((ROOT / "configs" / "scoring_priors.json").read_text())
SHEET = {"trigger_channel": "trig", "priority": 1, "trigger_gain": 10.0, "threshold": 15.0, "tau_on_over_tau_ref": 1.0, "tau_off_over_tau_ref": 2.0, "reach": 0.5}

def test_perturb_changes_each_parameter_differently():
    spec, roles = build_circuit("release", COMMON, MOTIFS, np.random.default_rng([0, 0]), "release_00000")
    t = perturb(spec, np.random.default_rng(1), "release_00000_t")
    assert t["circuit_id"] == "release_00000_t"
    assert spec["circuit_id"] == "release_00000"
    assert [n["id"] for n in t["neurons"]] == [n["id"] for n in spec["neurons"]]
    ratios = [a["g_leak_nS"] / b["g_leak_nS"] for a, b in zip(t["neurons"], spec["neurons"])]
    assert len(set(round(r, 6) for r in ratios)) == len(ratios)
    assert all(r != 1.0 for r in ratios)

def test_distance():
    other = copy.deepcopy(SHEET)
    other["threshold"] = 18.0
    assert datasheet_distance([SHEET], [SHEET], CONFIG) == 0.0
    span = CONFIG["fields"]["threshold"]["range"][1] - CONFIG["fields"]["threshold"]["range"][0]
    assert math.isclose(datasheet_distance([SHEET], [other], CONFIG), 3.0 / span)
    swapped = copy.deepcopy(SHEET)
    swapped["priority"] = 2
    assert datasheet_distance([SHEET], [swapped], CONFIG) == math.inf
    assert datasheet_distance([SHEET], [], CONFIG) == math.inf

def test_find_twin(tmp_path):
    spec, roles = build_circuit("release", COMMON, MOTIFS, np.random.default_rng([0, 0]), "release_00000")
    m = measure_circuit(spec, roles, tmp_path / "orig")
    assert m["accepted"]
    r = find_twin(spec, roles, m["datasheets"], CONFIG, np.random.default_rng(2), tmp_path, "release_00000_t", tries=3, tol=1.0)
    assert r["found"]
    assert r["tries"] == 1
    assert r["distance"] <= 1.0
    assert r["spec"]["circuit_id"] == "release_00000_t"
