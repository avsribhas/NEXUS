import math
import numpy as np
from nexus.models.summary import both_held_effect, select_path

IDS = ["a", "b", "c", "x", "y", "S"]
ROLES = {"inputs": ["a", "b", "c"], "outputs": ["x", "y"], "state": "S"}
POKES = [
    {"type": "dep_large", "target": "b", "hold": "", "hold_mV": 0.0},
    {"type": "dep_large", "target": "c", "hold": "", "hold_mV": 0.0},
    {"type": "paired", "target": "b", "hold": "a", "hold_mV": 30.0},
    {"type": "paired", "target": "c", "hold": "a", "hold_mV": 30.0},
    {"type": "paired_both", "target": "c", "hold": "a+b", "hold_mV": 30.0},
]

def make_animal(rows):
    """rows maps (poke index, neuron name) to a response in mV; state 0, one seed."""
    r = np.zeros((len(POKES), 1, 1, len(IDS)))
    for (p, name), value in rows.items():
        r[p, 0, 0, IDS.index(name)] = value
    return {"ids": IDS, "roles": ROLES, "pokes": POKES, "responses": r}

def test_floor_ignores_tiny_responses():
    animal = make_animal({(0, "x"): 0.001, (2, "x"): 0.1, (1, "y"): 5.0, (3, "y"): 0.5})
    R = np.nanmean(animal["responses"], axis=2)
    path = select_path(animal, "a", R)
    assert path[0] == "c" and path[1] == IDS.index("y")

def test_exclude_skips_the_partner():
    animal = make_animal({(0, "x"): 5.0, (2, "x"): 0.5, (1, "y"): 5.0, (3, "y"): 2.5})
    R = np.nanmean(animal["responses"], axis=2)
    assert select_path(animal, "a", R)[0] == "b"
    assert select_path(animal, "a", R, exclude=("b",))[0] == "c"

def test_both_held_effect_value():
    animal = make_animal({(1, "y"): 5.0, (3, "y"): 2.5, (4, "y"): 0.5})
    value = both_held_effect(animal, "a", "b")
    assert math.isclose(value, abs(math.log10((0.5 + 1e-3) / (5.0 + 1e-3))), abs_tol=1e-9)
    assert math.isnan(both_held_effect(animal, "b", "c"))
