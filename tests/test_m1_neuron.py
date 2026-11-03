from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.run import analytic_iaf_isi, reset_times, run_jaxley, run_neuron

CIRCUITS = Path(__file__).resolve().parents[1] / "configs" / "circuits"
LEMS = CIRCUITS / "LEMS_ref_iaf3.xml"
ISI = analytic_iaf_isi(0.2, 10.0, -70.0, -50.0, -70.0, 0.3)
STEP = 0.01

@pytest.fixture(scope="module")
def nrn():
    return run_neuron(LEMS)

@pytest.fixture(scope="module")
def jax():
    return run_jaxley(LEMS)

def test_neuron_runs(nrn):
    t_n, v_n = nrn
    assert set(v_n) == {"v0", "v1", "v2"}
    assert len(t_n) == 25001

def test_neuron_matches_analytic(nrn):
    t_n, v_n = nrn
    resets = reset_times(t_n, v_n["v0"])
    assert len(resets) >= 8
    assert abs(resets[0] - (10.0 + ISI)) < 2 * STEP
    assert np.all(np.abs(np.diff(resets) - ISI) < 2 * STEP)

def test_neuron_synapse_drives_cell1(nrn):
    t_n, v_n = nrn
    assert v_n["v1"].max() > -69.5

def test_neuron_control_cell_at_rest(nrn):
    t_n, v_n = nrn
    assert np.max(np.abs(v_n["v2"] + 70.0)) < 1e-4

def test_spike_times_match_jaxley(nrn, jax):
    t_n, v_n = nrn
    t_j, v_j, _ = jax
    r_n = reset_times(t_n, v_n["v0"])
    r_j = reset_times(t_j, v_j["v0"])
    assert len(r_n) == len(r_j)
    assert np.max(np.abs(r_n - r_j)) <= STEP + 1e-9

@pytest.mark.xfail(strict=True, reason="tarjuman 0.5.1 delivers no synaptic events from iafCell presynaptic cells; NEURON and jNeuroML do; see logs/phase1-v2-log.md, 2026-11-02")
def test_cell1_matches_neuron(nrn, jax):
    t_n, v_n = nrn
    t_j, v_j, _ = jax
    assert np.max(np.abs(v_j["v1"] - v_n["v1"])) < 0.5

def test_run_leaves_configs_clean(nrn):
    assert sorted(p.name for p in CIRCUITS.iterdir()) == ["LEMS_ref_iaf3.xml", "ref_iaf3.net.nml"]
