import jaxley as jx
import numpy as np
import pytest
from nexus.circuits.batch import BatchRunner
from nexus.circuits.graded import write_network, write_lems
from nexus.circuits.run import run_jaxley

IDS = ["n0", "n1", "n2", "n3"]
SPEC = {
    "circuit_id": "w4",
    "neurons": [{"id": n, "C_nF": 0.2, "g_leak_nS": 10.0, "e_leak_mV": -70.0} for n in IDS],
    "graded": [{"pre": "n0", "post": "n2", "g_nS": 5.0, "erev_mV": 0.0, "vth_mV": -60.0, "delta_mV": 5.0, "k_per_ms": 0.025}],
    "gap": [{"a": "n0", "b": "n1", "g_nS": 2.0}],
}
PULSES = [{"target": "n0", "delay_ms": 10.0, "duration_ms": 200.0, "amplitude_nA": 0.2}]

S_INF = 1.0 / (1.0 + np.exp((-60.0 + 70.0) / 5.0))
V2_REST = (10.0 * -70.0 + 5.0 * S_INF * 0.0) / (10.0 + 5.0 * S_INF)

@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("batch")
    write_network(SPEC, d / "plain" / "w4.net.nml")
    write_network(SPEC, d / "pulsed" / "w4.net.nml", pulses=PULSES)
    write_lems(SPEC, "w4.net.nml", d / "pulsed" / "LEMS_w4.xml", 300.0, 0.01)
    return d

@pytest.fixture(scope="module")
def runner(paths):
    return BatchRunner(paths / "plain" / "w4.net.nml", IDS, delta_t=0.01, t_max=300.0)

def test_n_steps(runner):
    assert runner.n_steps == 30001

def test_shape_and_rest(runner):
    out = runner.run(np.zeros((3, 4, runner.n_steps)))
    assert out.shape == (3, 4, runner.n_steps)
    assert np.max(np.abs(out[:, [0, 1, 3]] + 70.0)) < 1e-3
    assert np.max(np.abs(out[:, 2, -1] - V2_REST)) < 0.01

def test_matches_run_jaxley(runner, paths):
    n = runner.n_steps
    pulse = np.asarray(jx.step_current(i_delay=10.0, i_dur=200.0, i_amp=0.2, delta_t=0.01, t_max=300.0))[:n]
    currents = np.zeros((2, 4, n)); currents[0, 0] = pulse
    out = runner.run(currents)
    t, v, faithful = run_jaxley(paths / "pulsed" / "LEMS_w4.xml")
    for i, nid in enumerate(IDS): assert np.max(np.abs(out[0, i] - v[nid])) < 1e-3
    assert np.max(np.abs(out[1, [0, 1, 3]] + 70.0)) < 1e-3

def test_trials_are_independent(runner):
    rng = np.random.default_rng(0)
    currents = rng.normal(0.0, 0.05, size=(4, 4, runner.n_steps))
    all_out = runner.run(currents)
    one_out = runner.run(currents[2:3])
    assert np.max(np.abs(all_out[2] - one_out[0])) < 1e-4

def test_wrong_shape_raises(runner):
    with pytest.raises(ValueError): runner.run(np.zeros((2, 3, runner.n_steps)))
