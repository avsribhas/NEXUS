from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.build import build_circuit, load_motifs
from nexus.circuits.probe import DT, _currents, make_runner

COMMON, MOTIFS = load_motifs(Path(__file__).resolve().parents[1] / "configs" / "motifs")

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pulse_latches(seed, tmp_path):
    spec, roles = build_circuit("decoy_bistable", COMMON, MOTIFS, np.random.default_rng([seed, 0]), f"bistable_{seed}")
    runner, ids = make_runner(spec, tmp_path)
    pulse = _currents(spec, ids, [("trig", 30.0, 1000.0, 1100.0)])
    rest = _currents(spec, ids, [])
    v = runner.run(np.stack([pulse, rest]))
    a = ids.index("A")
    late = slice(int(round(3000.0 / DT)), int(round(4000.0 / DT)))
    before = slice(int(round(500.0 / DT)), int(round(900.0 / DT)))
    assert v[0, a, late].mean() - v[0, a, before].mean() > 20.0
    assert abs(v[1, a, late].mean() - v[1, a, before].mean()) < 2.0
