import numpy as np
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform
from nexus.score.recovery import CONTINUOUS_FIELDS

MIN_ROWS = 20

def type_rows(animals, switch_type) -> tuple:
    """Features, planted values and the always-defined field indices for every planted trigger of one switch type."""
    Xs = []
    Ys = []
    Ms = []
    for a in animals:
        if a["switch_type"] != switch_type:
            continue
        X, L, Y, M, _ = a["rows"]
        sel = L == 1.0
        Xs.append(X[sel])
        Ys.append(Y[sel])
        Ms.append(M[sel])
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    M = np.concatenate(Ms)
    fields = [j for j in range(len(CONTINUOUS_FIELDS)) if np.all(M[:, j] == 1.0)]
    return X, Y, fields

def posterior_widths(X, theta, low, high, level=0.9, n_folds=5, n_samples=1000, seed=0) -> np.ndarray:
    """Width of each row's posterior interval as a fraction of the prior range, from k-fold NPE so no row is scored by a model trained on it."""
    keep = X.std(axis=0) > 1e-8
    X = X[:, keep]
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    span = high - low
    theta = np.clip(theta, low + 1e-6 * span, high - 1e-6 * span)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(X)) % n_folds
    widths = np.zeros(theta.shape)
    prior = BoxUniform(low=torch.tensor(low, dtype=torch.float32), high=torch.tensor(high, dtype=torch.float32))
    tail = 100.0 * (1.0 - level) / 2.0
    for k in range(n_folds):
        torch.manual_seed(seed + k)
        train = fold != k
        test = np.where(fold == k)[0]
        inference = NPE(prior=prior, density_estimator="maf", show_progress_bars=False)
        inference.append_simulations(torch.tensor(theta[train], dtype=torch.float32), torch.tensor(X[train], dtype=torch.float32)).train(show_train_summary=False)
        posterior = inference.build_posterior()
        for i in test:
            s = posterior.sample((n_samples,), x=torch.tensor(X[i], dtype=torch.float32), show_progress_bars=False).numpy()
            q = np.percentile(s, [tail, 100.0 - tail], axis=0)
            widths[i] = (q[1] - q[0]) / span
    return widths

def identifiable_fields(animals, config, thresholds, n_folds=5, n_samples=1000, seed=0) -> dict:
    """For each switch type, the fields whose posterior interval is narrow enough in enough training circuits, the fields never defined for it, and the share of narrow intervals per field; a type with fewer than MIN_ROWS planted triggers is left undecided, with no fields."""
    t = thresholds["identifiability"]
    out = {}
    for st in sorted(set(a["switch_type"] for a in animals) - {"none"}):
        X, Y, fields = type_rows(animals, st)
        names = [CONTINUOUS_FIELDS[j] for j in fields]
        undefined = [f for f in CONTINUOUS_FIELDS if f not in names]
        if not fields or len(X) < MIN_ROWS:
            out[st] = {"fields": [], "undefined": undefined, "fraction": {}, "n_rows": int(len(X)), "decided": False}
            continue
        low = [config["fields"][f]["range"][0] for f in names]
        high = [config["fields"][f]["range"][1] for f in names]
        w = posterior_widths(X, Y[:, fields], low, high, t["posterior_level"], n_folds, n_samples, seed)
        frac = (w <= t["max_width_fraction_of_prior"]).mean(axis=0)
        out[st] = {"fields": [f for f, fr in zip(names, frac) if fr >= t["min_fraction_of_training_circuits"]], "undefined": undefined, "fraction": {f: float(fr) for f, fr in zip(names, frac)}, "n_rows": int(len(X)), "decided": True}
    return out
