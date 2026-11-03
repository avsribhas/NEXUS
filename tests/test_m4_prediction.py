import math
import numpy as np
import pytest
from nexus.score.prediction import eligible, is_tie, noise_ceiling, normalized_score, paired_bootstrap, pearson, prediction_passes

def test_pearson():
    a = np.arange(10.0)
    assert abs(pearson(a, 2 * a + 1) - 1.0) < 1e-12
    assert abs(pearson(a, -a) + 1.0) < 1e-12
    assert math.isnan(pearson(a, np.ones(10)))

def test_ceiling_without_noise():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=500)
    responses = np.tile(signal, (10, 1))
    c, samples = noise_ceiling(responses)
    assert abs(c - 1.0) < 1e-9
    assert samples.shape == (100,)

def test_ceiling_of_pure_noise():
    rng = np.random.default_rng(1)
    responses = rng.normal(size=(10, 500))
    c, _ = noise_ceiling(responses)
    assert c < 0.3

def test_ceiling_matches_theory():
    rng = np.random.default_rng(2)
    signal = rng.normal(size=4000)
    responses = signal + rng.normal(size=(10, 4000))
    c, _ = noise_ceiling(responses)
    assert abs(c - math.sqrt(1.0 / 1.1)) < 0.02
    assert abs(normalized_score(signal, responses, c) - 1.0) < 0.03

def test_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(3)
    a = 0.6 + 0.05 * rng.normal(size=200)
    b = 0.4 + 0.05 * rng.normal(size=200)
    ceil = 0.9 + 0.01 * rng.normal(size=(200, 50))
    boot = paired_bootstrap(a, b, ceil, n_boot=2000)
    assert boot["low"] > 0 and prediction_passes(boot)
    assert boot["low"] <= boot["mean"] <= boot["high"]

def test_bootstrap_no_difference():
    rng = np.random.default_rng(4)
    a = 0.5 + 0.05 * rng.normal(size=200)
    b = a + 0.05 * rng.normal(size=200)
    b = b - (b.mean() - a.mean())
    ceil = 0.9 + 0.01 * rng.normal(size=(200, 50))
    boot = paired_bootstrap(a, b, ceil, n_boot=2000)
    assert boot["low"] < 0 < boot["high"]
    assert not prediction_passes(boot)

def test_eligible_and_tie():
    assert list(eligible(np.array([0.05, 0.1, 0.5]))) == [False, True, True]
    assert is_tie(0.50, 0.515)
    assert not is_tie(0.50, 0.53)
