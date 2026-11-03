import numpy as np
import pytest
from nexus.baselines.wiring import KINDS, diagram_matrix, kind_matrix, normalized, propagator, response_columns

IDS = ["a", "b", "c"]
DIAGRAM = [{"pre": "a", "post": "b", "type": "graded", "sign": "exc", "weight": 0.5}, {"pre": "b", "post": "c", "type": "graded", "sign": "inh", "weight": 2.0}, {"pre": "a", "post": "c", "type": "gap", "sign": "gap", "weight": 0.1}, {"pre": "c", "post": "a", "type": "graded", "sign": "shunt", "weight": 0.3}]

def test_diagram_matrix():
    W = diagram_matrix(DIAGRAM, IDS)
    expected = np.array([[0.0, 0.0, 0.1], [0.5, 0.0, 0.0], [0.1, -2.0, 0.0]])
    assert np.allclose(W, expected)

def test_normalized_rows():
    Wn = normalized(diagram_matrix(DIAGRAM, IDS))
    assert np.all(np.abs(Wn).sum(axis=1) < 1.0)
    assert np.isclose(Wn[2, 1], -2.0 / 3.1)

def test_propagator():
    Wn = normalized(diagram_matrix(DIAGRAM, IDS))
    P = propagator(Wn, 0.5)
    assert np.allclose(P @ (np.eye(3) - 0.5 * Wn), np.eye(3))
    assert np.allclose(propagator(Wn, 0.0), np.eye(3))

def test_kinds():
    for kind in KINDS:
        C = response_columns(kind, DIAGRAM, IDS, 0.5)
        assert C.shape == (3, 3) and np.all(np.isfinite(C))
        assert np.allclose(response_columns("blind", DIAGRAM, IDS, 0.5), np.eye(3))
    full = kind_matrix("full", DIAGRAM, IDS)
    assert np.allclose(np.diag(full), 0.0) and np.all(full[~np.eye(3, dtype=bool)] > 0)
    a = kind_matrix("shuffled", DIAGRAM, IDS, np.random.default_rng(3))
    b = kind_matrix("shuffled", DIAGRAM, IDS, np.random.default_rng(3))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        kind_matrix("nope", DIAGRAM, IDS)

def test_one_step_is_identity_plus_matrix():
    C = response_columns("one_step", DIAGRAM, IDS, 0.5)
    assert np.allclose(C, np.eye(3) + kind_matrix("wiring", DIAGRAM, IDS))
