import numpy as np
import pytest
from nexus.baselines.linear import LinearBaseline
from nexus.baselines.wiring import response_columns
from nexus.perturb.protocol import POKE_TYPES, STATES_MV

IDS = ["a", "b", "c", "d"]
DIAGRAM = [{"pre": "a", "post": "b", "type": "graded", "sign": "exc", "weight": 1.0}, {"pre": "b", "post": "c", "type": "graded", "sign": "inh", "weight": 1.5}, {"pre": "c", "post": "d", "type": "graded", "sign": "exc", "weight": 0.8}]
POKES = [{"type": t, "target": n} for t in POKE_TYPES for n in IDS]
GAINS = {t: float(k + 1) for k, t in enumerate(POKE_TYPES)}

def _circuit(cid):
    return {"circuit_id": cid, "ids": IDS, "diagram": DIAGRAM, "pokes": POKES}

def _truth(alpha, state_slope=0.0):
    C = response_columns("wiring", DIAGRAM, IDS, alpha)
    Y = np.zeros((len(POKES), len(STATES_MV), len(IDS)))
    for p, poke in enumerate(POKES):
        for s, mv in enumerate(STATES_MV):
            Y[p, s] = GAINS[poke["type"]] * (1.0 + state_slope * mv / 30.0) * C[:, IDS.index(poke["target"])]
    return Y

def test_recovers_alpha_and_gains():
    Y = _truth(0.5)
    m = LinearBaseline(alphas=[0.0, 0.25, 0.5, 0.75]).fit([_circuit("x")], [Y], [np.ones(Y.shape, dtype=bool)])
    assert m.alpha == 0.5
    assert np.allclose(m.predict(_circuit("x")), Y, atol=1e-9)

def test_pooled_model_has_no_state_dependence():
    Y = _truth(0.5, state_slope=1.0)
    m = LinearBaseline(alphas=[0.5]).fit([_circuit("x")], [Y], [np.ones(Y.shape, dtype=bool)])
    P = m.predict(_circuit("x"))
    assert np.allclose(P - P.mean(axis=1, keepdims=True), 0.0)

def test_state_term_fits_state_dependence():
    Y = _truth(0.5, state_slope=1.0)
    m = LinearBaseline(alphas=[0.5], state_term=True).fit([_circuit("x")], [Y], [np.ones(Y.shape, dtype=bool)])
    assert np.allclose(m.predict(_circuit("x")), Y, atol=1e-9)

def test_per_state_uses_nearest_fitted_state():
    Y = _truth(0.5, state_slope=1.0)
    M = np.ones(Y.shape, dtype=bool)
    M[:, 5, :] = False
    m = LinearBaseline(alphas=[0.5], per_state=True).fit([_circuit("x")], [Y], [M])
    P = m.predict(_circuit("x"))
    assert np.allclose(P[:, 4], Y[:, 4], atol=1e-9)
    assert np.allclose(P[:, 5], Y[:, 4], atol=1e-9)

def test_masked_items_are_not_used():
    Y = _truth(0.5)
    bad = Y.copy()
    bad[:, :, 0] = 1000.0
    M = np.ones(Y.shape, dtype=bool)
    M[:, :, 0] = False
    m = LinearBaseline(alphas=[0.5]).fit([_circuit("x")], [bad], [M])
    assert np.allclose(m.predict(_circuit("x"))[:, :, 1:], Y[:, :, 1:], atol=1e-9)

def test_every_kind_runs():
    Y = _truth(0.5)
    for kind in ("wiring", "one_step", "blind", "full", "shuffled"):
        m = LinearBaseline(kind=kind, alphas=[0.0, 0.5]).fit([_circuit("x")], [Y], [np.ones(Y.shape, dtype=bool)])
        P = m.predict(_circuit("x"))
        assert P.shape == Y.shape and np.all(np.isfinite(P))
