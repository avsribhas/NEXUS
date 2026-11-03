import pytest
import numpy as np
import nexus.models.rows as rows

ANIMAL = {"ids": ["a", "S", "b"], "roles": {"state": "S"}, "animal_id": "x_a0"}
RNG = np.random.default_rng(0)
SIGNAL = RNG.standard_normal((4, 6, 3))
MEASURED = SIGNAL[:, :, None, :] + 0.01 * RNG.standard_normal((4, 6, 3, 3))

@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(rows, "poke_responses", lambda animal: MEASURED)

def test_measured_shape(fake):
    m = rows.measured_sd(ANIMAL)
    assert m.shape == (3, 4, 2, 6)
    assert np.allclose(m.mean(axis=-1), 0.0)

def test_predicted_shape(fake):
    p = rows.predicted_sd(ANIMAL, SIGNAL)
    assert p.shape == (4, 2, 6)
    assert np.allclose(p.mean(axis=-1), 0.0)

def test_perfect_prediction(fake):
    row = rows.prediction_row(ANIMAL, {"perfect": SIGNAL})
    assert row["r"]["perfect"] > 0.95
    assert row["ceiling"] > 0.9

def test_pooled_model_scores_zero(fake):
    pooled = np.repeat(SIGNAL.mean(axis=1, keepdims=True), 6, axis=1)
    row = rows.prediction_row(ANIMAL, {"pooled": pooled})
    assert row["r"]["pooled"] == 0.0

def test_state_neuron_is_ignored(fake):
    changed = SIGNAL.copy()
    changed[:, :, 1] = 100.0 * RNG.standard_normal((4, 6))
    row = rows.prediction_row(ANIMAL, {"perfect": SIGNAL, "changed": changed})
    assert row["r"]["changed"] == row["r"]["perfect"]

def test_row_keys(fake):
    row = rows.prediction_row(ANIMAL, {"perfect": SIGNAL})
    assert row["circuit_id"] == "x_a0"
    assert len(row["ceiling_samples"]) == 100
    assert set(row) == {"circuit_id", "ceiling", "ceiling_samples", "r"}
