import math
import numpy as np
from nexus.models.summary import FEATURES, path_evidence

def _animal():
    ids = ["a", "t", "n1", "n2", "O"]
    pokes = [
        {"type": "dep_large", "target": "a"},
        {"type": "paired", "target": "a", "hold": "t", "hold_mV": 30.0},
        {"type": "dep_large", "target": "b"},
        {"type": "paired", "target": "b", "hold": "t", "hold_mV": 30.0},
    ]
    roles = {"inputs": ["a", "b", "t"], "state": "s", "outputs": ["O"]}
    responses = np.zeros((4, 2, 3, 5))
    responses[1, :, :, 2] = 5.0
    responses[1, :, :, 4] = np.array([2.0, 3.0, 4.0])
    responses[3, :, :, 2] = 5.0
    responses[3, :, :, 3] = 5.0
    return {"ids": ids, "pokes": pokes, "roles": roles}, responses

def test_new_names_are_features():
    animal, responses = _animal()
    assert set(path_evidence(animal, "t", "a", 4, responses)) <= set(FEATURES)

def test_reach_on_every_path_largest_first():
    animal, responses = _animal()
    out = path_evidence(animal, "t", "a", 4, responses)
    assert abs(out["reach_path_1"] - 2 / 3) < 1e-9
    assert abs(out["reach_path_2"] - 2 / 3) < 1e-9
    assert math.isnan(out["reach_path_3"])

def test_largest_change_among_other_neurons():
    animal, responses = _animal()
    assert abs(path_evidence(animal, "t", "a", 4, responses)["max_others_diff"] - 5.0) < 1e-6

def test_output_z_uses_seed_spread():
    animal, responses = _animal()
    expected = 3.0 / (1.0 / math.sqrt(3.0) + 0.05)
    out = path_evidence(animal, "t", "a", 4, responses)
    assert abs(out["output_z"] - expected) < 1e-6
    assert abs(out["output_z_all_states"] - expected) < 1e-6

def test_missing_pokes_give_nan():
    animal, responses = _animal()
    animal["pokes"] = animal["pokes"][2:] + [{"type": "x", "target": "a"}, {"type": "x", "target": "a"}]
    out = path_evidence(animal, "t", "a", 4, responses)
    assert math.isnan(out["log_gain_all_states"])
    assert math.isnan(out["output_z"])
