import jax
from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.build import build_circuit, drive_to_current, load_motifs
from nexus.circuits.graded import write_lems, write_network
from nexus.circuits.run import run_jaxley, run_neuron

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64, reason="run with JAX_ENABLE_X64=1; float32 drifts at a 0.0025 ms step")

COMMON, MOTIFS = load_motifs(Path(__file__).resolve().parents[1] / "configs" / "motifs")

def pulses(spec, roles):
    out = [{"target": roles["state"], "delay_ms": 0.0, "duration_ms": 600.0, "amplitude_nA": drive_to_current(spec, roles["state"], 20.0)}]
    for k, nid in enumerate(roles["inputs"]):
        out.append({"target": nid, "delay_ms": 50.0 + 100.0 * k, "duration_ms": 300.0, "amplitude_nA": drive_to_current(spec, nid, 25.0)})
    return out

@pytest.mark.parametrize("name", sorted(MOTIFS))
def test_jaxley_matches_neuron(name, tmp_path):
    spec, roles = build_circuit(name, COMMON, MOTIFS, np.random.default_rng([0, 0]), f"{name}_check")
    write_network(spec, tmp_path / "c.net.nml", pulses=pulses(spec, roles))
    write_lems(spec, "c.net.nml", tmp_path / "LEMS_c.xml", 600.0, 0.0025)
    t_j, v_j, faithful = run_jaxley(tmp_path / "LEMS_c.xml")
    t_n, v_n = run_neuron(tmp_path / "LEMS_c.xml")
    assert faithful
    assert len(t_j) == len(t_n)
    for n in spec["neurons"]:
        assert np.max(np.abs(v_j[n["id"]] - v_n[n["id"]])) < 0.02
