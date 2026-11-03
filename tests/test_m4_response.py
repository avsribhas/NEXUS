import numpy as np
from nexus.models.response import DuskResponse, adjacency, node_features
from nexus.score.prediction import pearson

IDS = ["a", "b", "S"]
DIAGRAM = [{"pre": "a", "post": "b", "type": "graded", "sign": "exc", "weight": 0.5}, {"pre": "a", "post": "S", "type": "gap", "sign": "gap", "weight": 0.2}]
LEVELS = (0.0, 6.0, 12.0, 18.0, 24.0, 30.0)

def make_animal(gain):
    pokes = [{"target": "a", "amplitude_mV": amp, "start_ms": 1500.0, "stop_ms": 1600.0, "hold": "", "hold_mV": 0.0} for amp in (5.0, 10.0, 15.0, 20.0)]
    responses = np.zeros((4, 6, 3, 3))
    for p, amp in enumerate((5.0, 10.0, 15.0, 20.0)):
        for s, level in enumerate(LEVELS):
            responses[p, s, :, 0] = amp
            responses[p, s, :, 1] = gain * amp * (1.0 + level / 30.0)
    return {"ids": IDS, "roles": {"state": "S", "inputs": ["a"]}, "diagram": DIAGRAM, "pokes": pokes, "responses": responses}

def test_adjacency():
    A = adjacency(DIAGRAM, IDS)
    assert A.shape == (4, 3, 3)
    assert A[0, 1, 0] > 0.0
    assert A[3, 0, 2] == A[3, 2, 0] > 0.0
    assert A[1].sum() == 0.0 and A[2].sum() == 0.0

def test_node_features():
    x = node_features(make_animal(0.5))
    assert x.shape == (4, 6, 3, 8)
    assert np.allclose(x[:, :, 0, 2], 1.0)
    assert np.allclose(x[:, :, 1, 2], 0.0)
    assert np.allclose(x[:, :, 2, 5], 1.0)
    assert np.allclose(x[0, :, 0, 6], np.array(LEVELS) / 30.0)
    assert np.allclose(x[:, :, 0, 7], 1.0)

def test_learns_state_dependent_gain():
    animals = [make_animal(g) for g in (0.3, 0.5, 0.7)]
    model = DuskResponse(hidden=32, layers=2, epochs=300, seed=0).fit(animals)
    pred = model.predict(animals[1])
    assert pred.shape == (4, 6, 3)
    sd_pred = pred[:, :, 1] - pred[:, :, 1].mean(axis=1, keepdims=True)
    y = animals[1]["responses"].mean(axis=2)[:, :, 1]
    sd_y = y - y.mean(axis=1, keepdims=True)
    assert pearson(sd_pred, sd_y) > 0.9

def test_same_seed_same_model():
    animals = [make_animal(0.5)]
    a = DuskResponse(hidden=16, layers=1, epochs=5, seed=1).fit(animals).predict(animals[0])
    b = DuskResponse(hidden=16, layers=1, epochs=5, seed=1).fit(animals).predict(animals[0])
    assert np.array_equal(a, b)
