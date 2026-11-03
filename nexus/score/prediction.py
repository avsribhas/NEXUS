import math
import numpy as np

def pearson(a, b) -> float:
    """Pearson correlation of two 1-D arrays; nan if either is constant."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    return math.nan if d == 0.0 else float(np.sum(a * b) / d)

def noise_ceiling(responses, n_splits=100, seed=0) -> tuple:
    """Split-half noise ceiling with Spearman-Brown correction, as the best correlation a perfect model could reach."""
    responses = np.asarray(responses, dtype=float)
    n = responses.shape[0]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a = responses[perm[: n // 2]].mean(axis=0)
        b = responses[perm[n // 2 :]].mean(axis=0)
        r = pearson(a, b)
        if math.isnan(r):
            r = 0.0
        rel = 2.0 * r / (1.0 + r) if r > -1.0 else 0.0
        samples.append(math.sqrt(max(rel, 0.0)))
    samples = np.array(samples)
    return float(samples.mean()), samples

def normalized_score(pred, responses, ceiling) -> float:
    """Correlation with the seed-averaged response, as a fraction of the noise ceiling."""
    return pearson(pred, np.asarray(responses, dtype=float).mean(axis=0)) / ceiling

def paired_bootstrap(score_a, score_b, ceiling_samples, n_boot=10000, seed=0, ci=0.95) -> dict:
    """Bootstrap over circuits of the mean ceiling-normalized difference between two models, carrying each circuit's ceiling uncertainty."""
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    c = np.asarray(ceiling_samples, dtype=float)
    n, k = c.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    pick = rng.integers(0, k, size=(n_boot, n))
    stats = ((a[idx] - b[idx]) / c[idx, pick]).mean(axis=1)
    mean = float(((a - b) / c.mean(axis=1)).mean())
    tail = 100.0 * (1.0 - ci) / 2.0
    return {"mean": mean, "low": float(np.percentile(stats, tail)), "high": float(np.percentile(stats, 100.0 - tail))}

def eligible(ceilings, floor=0.1) -> np.ndarray:
    """Circuits whose noise ceiling is at least the floor."""
    return np.asarray(ceilings, dtype=float) >= floor

def prediction_passes(boot) -> bool:
    """The pre-registered pass: the interval of the paired difference lies above zero."""
    return bool(boot["low"] > 0.0)

def is_tie(mean_a, mean_b, margin=0.02) -> bool:
    """Two normalized scores within the margin count as a tie."""
    return bool(abs(mean_a - mean_b) <= margin)
