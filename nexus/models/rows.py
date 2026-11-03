import math
import numpy as np
from nexus.models.data import poke_responses
from nexus.score.prediction import noise_ceiling, pearson
from nexus.score.report import state_dependent_part

def response_neurons(animal) -> list:
    """Indices of every neuron except the brain-state neuron."""
    return [i for i in range(len(animal["ids"])) if animal["ids"][i] != animal["roles"]["state"]]

def measured_sd(animal) -> np.ndarray:
    """Measured state-dependent responses, shape (seeds, pokes, neurons, states); a cached animal carries its poke responses under "responses"."""
    y = animal["responses"] if "responses" in animal else poke_responses(animal)
    y = y[..., response_neurons(animal)]
    y = np.transpose(y, (2, 0, 3, 1))   # now (seeds, pokes, neurons, states)
    return state_dependent_part(y)

def predicted_sd(animal, pred) -> np.ndarray:
    """Predicted state-dependent responses, shape (pokes, neurons, states)."""
    pred = np.asarray(pred, dtype=float)[..., response_neurons(animal)]
    pred = np.transpose(pred, (0, 2, 1))   # now (pokes, neurons, states)
    return state_dependent_part(pred)

def prediction_row(animal, preds, n_splits=100, seed=0) -> dict:
    """One scoring row: noise ceiling and each model's correlation with the measured state-dependent responses; a model whose prediction does not change across states has no state-dependent part and scores 0."""
    m = measured_sd(animal)
    flat = m.reshape(m.shape[0], -1)
    ceiling, samples = noise_ceiling(flat, n_splits, seed)
    target = flat.mean(axis=0)
    r = {}
    for name, pred in preds.items():
        p = predicted_sd(animal, pred).ravel()
        v = math.nan if np.max(np.abs(p)) < 1e-9 else pearson(p, target)
        r[name] = 0.0 if math.isnan(v) else float(v)
    return {"circuit_id": animal["animal_id"], "ceiling": float(ceiling), "ceiling_samples": [float(x) for x in samples], "r": r}
