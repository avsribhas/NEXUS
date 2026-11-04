import json
from pathlib import Path
import numpy as np
from nexus.models.dusk import DuskRecovery
from nexus.models.summary import N_FEATURES
from nexus.score.recovery import CONTINUOUS_FIELDS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "scoring_priors.json").read_text())

def fake_animal(rng):
    x = rng.normal(size=(2, N_FEATURES))
    labels = np.array([1.0, 0.0])
    values = np.zeros((2, len(CONTINUOUS_FIELDS))); values[0, :] = x[0, 0] + rng.normal(scale=1.0, size=len(CONTINUOUS_FIELDS))
    masks = np.zeros((2, len(CONTINUOUS_FIELDS))); masks[0, :] = 1.0
    return {"rows": (x, labels, values, masks, ["a", "b"])}

def coverage(model, animals):
    total_covered = 0
    for a in animals:
        _, _, quant = model.predict_rows(a)
        _, labels, values, masks, _ = a["rows"]
        for j in range(len(CONTINUOUS_FIELDS)):
            if quant[0, j, 0] <= values[0, j] <= quant[0, j, 1]:
                total_covered += 1
    return total_covered / (len(animals) * len(CONTINUOUS_FIELDS))

def test_default_does_not_widen():
    rng = np.random.default_rng(0)
    animals = [fake_animal(rng) for _ in range(20)]
    model = DuskRecovery(CONFIG, epochs=5).fit(animals)
    assert np.all(model.widen == 0.0)

def test_calibration_reaches_coverage():
    rng = np.random.default_rng(1)
    train = [fake_animal(rng) for _ in range(200)]
    test = [fake_animal(rng) for _ in range(200)]
    model = DuskRecovery(CONFIG, epochs=300, calibrate=0.3).fit(train)
    assert coverage(model, test) >= 0.8

def test_intervals_stay_ordered():
    rng = np.random.default_rng(2)
    animals = [fake_animal(rng) for _ in range(20)]
    model = DuskRecovery(CONFIG, epochs=5, calibrate=0.5).fit(animals)
    _, _, quant = model.predict_rows(animals[0])
    assert np.all(quant[:, :, 0] <= quant[:, :, 1])
