import numpy as np
from nexus.models.identifiability import identifiable_fields, posterior_widths, type_rows
from nexus.score.recovery import CONTINUOUS_FIELDS

CONFIG = {"fields": {f: {"range": [0.0, 1.0]} for f in CONTINUOUS_FIELDS}}
THRESHOLDS = {"identifiability": {"posterior_level": 0.9, "max_width_fraction_of_prior": 0.5, "min_fraction_of_training_circuits": 0.8}}

def make_animals(n, seed):
    rng = np.random.default_rng(seed)
    animals = []
    for _ in range(n):
        theta = rng.uniform(0.05, 0.95, size=2)
        X = np.array([[theta[0] + 0.01 * rng.standard_normal(), rng.uniform(), rng.uniform()], [rng.uniform(), rng.uniform(), rng.uniform()]])
        L = np.array([1.0, 0.0])
        Y = np.zeros((2, 5))
        Y[0, 0] = theta[0]
        Y[0, 1] = theta[1]
        M = np.zeros((2, 5))
        M[0, 0] = 1.0
        M[0, 1] = 1.0
        animals.append({"switch_type": "release", "rows": (X, L, Y, M, ["a", "b"])})
    animals.append({"switch_type": "none", "rows": (np.zeros((1, 3)), np.array([0.0]), np.zeros((1, 5)), np.zeros((1, 5)), ["a"])})
    return animals

def test_type_rows():
    X, Y, fields = type_rows(make_animals(10, 0), "release")
    assert X.shape == (10, 3)
    assert Y.shape == (10, 5)
    assert fields == [0, 1]

def test_informative_feature_gives_narrow_interval():
    X, Y, fields = type_rows(make_animals(200, 1), "release")
    w = posterior_widths(X, Y[:, fields], [0.0, 0.0], [1.0, 1.0], n_folds=2, n_samples=500)
    assert w.shape == (200, 2)
    assert np.median(w[:, 0]) < 0.3
    assert np.median(w[:, 1]) > 0.6

def test_identifiable_fields():
    out = identifiable_fields(make_animals(200, 2), CONFIG, THRESHOLDS, n_folds=2, n_samples=500)
    assert set(out) == {"release"}
    assert out["release"]["fields"] == [CONTINUOUS_FIELDS[0]]
    assert out["release"]["undefined"] == list(CONTINUOUS_FIELDS[2:])
    assert out["release"]["n_rows"] == 200
