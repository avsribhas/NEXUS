import math
from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.run import analytic_iaf_isi, reset_times, run_jaxley, validate_circuit

CIRCUITS = Path(__file__).resolve().parents[1] / "configs" / "circuits"
LEMS = CIRCUITS / "LEMS_ref_iaf3.xml"
NML = CIRCUITS / "ref_iaf3.net.nml"
ISI = analytic_iaf_isi(0.2, 10.0, -70.0, -50.0, -70.0, 0.3)

@pytest.fixture(scope="module")
def sim():
    return run_jaxley(LEMS)

def test_reference_circuit_is_valid():
    validate_circuit(NML)

def test_analytic_isi():
    assert abs(ISI - 20.0 * math.log(3.0)) < 1e-9
    assert analytic_iaf_isi(0.2, 10.0, -70.0, -50.0, -70.0, 0.1) == math.inf

def test_conversion_is_faithful(sim):
    time, traces, faithful = sim
    assert faithful is True
    assert set(traces) == {"v0", "v1", "v2"}
    assert len(time) == 25001

def test_driven_cell_matches_analytic(sim):
    time, traces, _ = sim
    resets = reset_times(time, traces["v0"])
    assert len(resets) >= 8
    assert abs(resets[0] - (10.0 + ISI)) < 0.02
    assert np.all(np.abs(np.diff(resets) - ISI) < 0.02)

def test_control_cell_stays_at_rest(sim):
    time, traces, _ = sim
    assert np.max(np.abs(traces["v2"] + 70.0)) < 1e-4

def test_rerun_is_bitwise_identical(sim):
    time, traces, _ = sim
    time2, traces2, _ = run_jaxley(LEMS)
    assert np.array_equal(time, time2)
    for key in traces:
        assert np.array_equal(traces[key], traces2[key])

@pytest.mark.xfail(strict=True, reason="tarjuman 0.5.1 delivers no synaptic events from iafCell presynaptic cells; see logs/phase1-v2-log.md, 2026-11-02")
def test_synapse_drives_cell1(sim):
    time, traces, _ = sim
    assert traces["v1"].max() > -69.5
