from pathlib import Path
import numpy as np
from nexus.circuits.build import build_circuit, drive_to_current, load_motifs
from nexus.perturb.protocol import DT, PAIRED, PAIRED_BOTH, POKE_START, make_pokes, trial_currents

COMMON, MOTIFS = load_motifs(Path(__file__).resolve().parents[1] / "configs" / "motifs")

def _build(name):
    spec, roles = build_circuit(name, COMMON, MOTIFS, np.random.default_rng(0), "both")
    return spec, roles, [n["id"] for n in spec["neurons"]]

def test_two_inputs_get_no_both_held_pokes():
    spec, roles, ids = _build("release")
    pokes = make_pokes(spec, roles, np.random.default_rng(1))
    assert not any(poke["type"] == PAIRED_BOTH["type"] for poke in pokes)

def test_four_inputs_get_every_pair_and_target():
    spec, roles, ids = _build("winner_take_all")
    pokes = make_pokes(spec, roles, np.random.default_rng(1))
    both = [p for p in pokes if p["type"] == PAIRED_BOTH["type"]]
    inputs = list(roles["inputs"])
    assert len(inputs) == 4
    assert len(both) == 12
    for p in both:
        holds = p["hold"].split("+")
        assert len(holds) == 2
        assert p["target"] in inputs
        assert p["target"] not in holds
        assert all(held_name in inputs for held_name in holds)
        assert p["hold_mV"] == PAIRED_BOTH["hold_mV"]
        assert p["amplitude_mV"] == PAIRED_BOTH["amplitude_mV"]
    assert len(set((p["hold"], p["target"]) for p in both)) == 12

def test_poke_ids_are_unique():
    spec, roles, ids = _build("winner_take_all")
    pokes = make_pokes(spec, roles, np.random.default_rng(1))
    assert len(set(poke["poke_id"] for poke in pokes)) == len(pokes)

def test_both_neurons_are_held():
    spec, roles, ids = _build("winner_take_all")
    pokes = make_pokes(spec, roles, np.random.default_rng(1))
    p = next(poke for poke in pokes if poke["type"] == PAIRED_BOTH["type"])
    c = trial_currents(spec, ids, roles, p, 0, 0, 7, noise=False)
    before = int(round(POKE_START / DT)) - 10
    for name in p["hold"].split("+"):
        assert np.isclose(c[ids.index(name), before], drive_to_current(spec, name, p["hold_mV"]), atol=1e-6)
    assert np.isclose(c[ids.index(p["target"]), before], 0.0, atol=1e-6)
    assert np.isclose(c[ids.index(p["target"]), int(round(POKE_START / DT))], drive_to_current(spec, p["target"], p["amplitude_mV"]), atol=1e-6)
