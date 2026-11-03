from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.run import run_jaxley, run_neuron, validate_circuit

CIRCUITS = Path(__file__).resolve().parents[1] / "configs" / "circuits"
LEMS = CIRCUITS / "LEMS_ref_graded4.xml"
NML = CIRCUITS / "ref_graded4.net.nml"
G = 10.0
GJ = 2.0
I = 0.2
V0_SS = -70.0 + I * (G + GJ) / (G * (G + 2 * GJ)) * 1000.0
V1_SS = -70.0 + GJ / (G + GJ) * (V0_SS + 70.0)
AT_209 = 20900

@pytest.fixture(scope="module")
def jax():
    t, v, faithful = run_jaxley(LEMS)
    assert faithful
    assert set(v) == {"v0", "v1", "v2", "v3"}
    assert len(t) == 30001
    return t, v, faithful

@pytest.fixture(scope="module")
def nrn():
    t, v = run_neuron(LEMS)
    assert len(t) == 30001
    return t, v

def test_graded_circuit_is_valid():
    validate_circuit(NML)

def test_conversion_is_faithful(jax):
    _, _, faithful = jax
    assert faithful

def test_gap_junction_steady_state_neuron(nrn):
    _, v_n = nrn
    assert abs(v_n["v0"][AT_209] - V0_SS) < 0.005
    assert abs(v_n["v1"][AT_209] - V1_SS) < 0.005

def test_gap_junction_steady_state_jaxley(jax):
    _, v_j, _ = jax
    assert abs(v_j["v0"][AT_209] - V0_SS) < 0.02
    assert abs(v_j["v1"][AT_209] - V1_SS) < 0.02

def test_graded_synapse_drives_cell2(jax, nrn):
    v_j, v_n = jax[1], nrn[1]
    assert v_j["v2"].max() > -55.0
    assert v_n["v2"].max() > -55.0

def test_control_cell_at_rest(jax, nrn):
    v_j, v_n = jax[1], nrn[1]
    assert np.max(np.abs(v_j["v3"] + 70.0)) < 1e-4
    assert np.max(np.abs(v_n["v3"] + 70.0)) < 1e-4

def test_jaxley_matches_neuron(jax, nrn):
    t_j, v_j, _ = jax
    t_n, v_n = nrn
    assert len(t_j) == len(t_n)
    for key in ("v0", "v1", "v2", "v3"):
        assert np.max(np.abs(v_j[key] - v_n[key])) < 0.02
