import math
from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.build import build_circuit, load_motifs, make_twin
from nexus.circuits.probe import measure_circuit
from nexus.format.ground_truth import DATASHEET_COLUMNS, TAU_REF_DEFINITIONS

COMMON, MOTIFS = load_motifs(Path(__file__).resolve().parents[1] / "configs" / "motifs")

def _measure(name, seed, tmp_path):
    spec, roles = build_circuit(name, COMMON, MOTIFS, np.random.default_rng(seed), f"c_{name}_{seed}")
    return spec, roles, measure_circuit(spec, roles, tmp_path / f"{name}_{seed}")

@pytest.mark.parametrize("name, opens", [("release", True), ("gain", True), ("global_shutoff", False)])
@pytest.mark.parametrize("seed", [0, 1])
def test_switch_directions(name, opens, seed, tmp_path):
    spec, roles, m = _measure(name, seed, tmp_path)
    r = m["gate_ratios"][0]
    if opens:
        assert r > 1.0
    else:
        assert r < 1.0

def test_release_result_shape(tmp_path):
    spec, roles, m = _measure("release", 0, tmp_path)
    assert len(m["datasheets"]) == 1
    d = m["datasheets"][0]
    assert set(d) == set(DATASHEET_COLUMNS)
    assert d["gate_id"] == "g0" and d["trigger_channel"] == "trig"
    assert d["priority"] == 1 and d["cost_atp_per_flip"] is None
    assert d["tau_ref_definition"] in TAU_REF_DEFINITIONS
    assert 0.0 < d["reach"] <= 1.0
    assert isinstance(m["accepted"], bool) and isinstance(m["reason"], str)

def test_shunt_never_raises_the_response(tmp_path):
    spec, roles, m = _measure("decoy_shunt", 0, tmp_path)
    assert m["datasheets"] == []
    assert len(m["cross_ratios"]) > 0
    assert all(r <= 1.0 + 1e-6 for r in m["cross_ratios"].values())

def test_winner_take_all_has_two_ranked_gates(tmp_path):
    spec, roles, m = _measure("winner_take_all", 0, tmp_path)
    assert [d["gate_id"] for d in m["datasheets"]] == ["gA", "gB"]
    assert sorted(d["priority"] for d in m["datasheets"]) == [1, 2]

def test_twin_gets_the_same_datasheet(tmp_path):
    spec, roles = build_circuit("release", COMMON, MOTIFS, np.random.default_rng(3), "orig")
    twin = make_twin(spec, 2.0, "twin")
    a = measure_circuit(spec, roles, tmp_path / "a")["datasheets"][0]
    b = measure_circuit(twin, roles, tmp_path / "b")["datasheets"][0]
    for key in ("trigger_gain", "threshold", "tau_on_over_tau_ref", "tau_off_over_tau_ref", "reach"):
        if math.isnan(a[key]):
            assert math.isnan(b[key])
        else:
            assert abs(a[key] - b[key]) <= 1e-3 * max(1.0, abs(a[key]))
