import copy
from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.graded import validate_spec, write_lems, write_network
from nexus.circuits.run import run_jaxley, run_neuron, validate_circuit

REF_LEMS = Path(__file__).resolve().parents[1] / "configs" / "circuits" / "LEMS_ref_graded4.xml"
SPEC = {
    "circuit_id": "w4",
    "neurons": [{"id": f"n{i}", "C_nF": 0.2, "g_leak_nS": 10.0, "e_leak_mV": -70.0} for i in range(4)],
    "graded": [{"pre": "n0", "post": "n2", "g_nS": 5.0, "erev_mV": 0.0, "vth_mV": -60.0, "delta_mV": 5.0, "k_per_ms": 0.025}],
    "gap": [{"a": "n0", "b": "n1", "g_nS": 2.0}],
}
PULSES = [{"target": "n0", "delay_ms": 10.0, "duration_ms": 200.0, "amplitude_nA": 0.2}]

@pytest.fixture(scope="module")
def written(tmp_path_factory):
    d = tmp_path_factory.mktemp("w4")
    write_network(SPEC, d / "w4.net.nml", pulses=PULSES)
    write_lems(SPEC, "w4.net.nml", d / "LEMS_w4.xml", 300.0, 0.01)
    return d

def test_spec_is_valid():
    assert validate_spec(SPEC) == []

def test_unknown_neuron_is_a_problem():
    bad = copy.deepcopy(SPEC)
    bad["graded"][0]["post"] = "n9"
    assert len(validate_spec(bad)) > 0

def test_duplicate_neuron_is_a_problem():
    bad = copy.deepcopy(SPEC)
    bad["neurons"].append(dict(bad["neurons"][0]))
    assert len(validate_spec(bad)) > 0

def test_written_network_is_valid(written):
    validate_circuit(written / "w4.net.nml")

def test_neuron_matches_reference(written):
    t_w, v_w = run_neuron(written / "LEMS_w4.xml")
    t_r, v_r = run_neuron(REF_LEMS)
    assert len(t_w) == len(t_r)
    for i in range(4):
        assert np.max(np.abs(v_w[f"n{i}"] - v_r[f"v{i}"])) < 1e-6

def test_jaxley_matches_neuron(written):
    t_j, v_j, faithful = run_jaxley(written / "LEMS_w4.xml")
    t_n, v_n = run_neuron(written / "LEMS_w4.xml")
    assert faithful is True
    assert len(t_j) == len(t_n)
    for i in range(4):
        assert np.max(np.abs(v_j[f"n{i}"] - v_n[f"n{i}"])) < 0.02
