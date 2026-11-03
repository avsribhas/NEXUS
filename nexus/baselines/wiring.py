import numpy as np

SIGN = {"exc": 1.0, "inh": -1.0, "shunt": 0.0}
KINDS = ("wiring", "one_step", "blind", "full", "shuffled")

def diagram_matrix(diagram, ids) -> np.ndarray:
    """Signed weight matrix W[post, pre] from a diagram; gap junctions count in both directions."""
    index = {nid: i for i, nid in enumerate(ids)}
    W = np.zeros((len(ids), len(ids)))
    for e in diagram:
        i = index[e["post"]]
        j = index[e["pre"]]
        if e["type"] == "gap":
            W[i, j] += e["weight"]
            W[j, i] += e["weight"]
        else:
            W[i, j] += SIGN[e["sign"]] * e["weight"]
    return W

def normalized(W) -> np.ndarray:
    """Divide each row by one plus its total absolute weight, so every row's absolute sum is below one."""
    W = np.asarray(W, dtype=float)
    return W / (1.0 + np.abs(W).sum(axis=1, keepdims=True))

def propagator(Wn, alpha) -> np.ndarray:
    """Steady-state linear response matrix inv(I - alpha * Wn)."""
    return np.linalg.inv(np.eye(Wn.shape[0]) - alpha * Wn)

def kind_matrix(kind, diagram, ids, rng=None) -> np.ndarray:
    """The normalized matrix a baseline uses: the diagram, all-to-all, none, or a shuffled diagram."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}")
    n = len(ids)
    if kind == "blind":
        return np.zeros((n, n))
    W = diagram_matrix(diagram, ids)
    if kind == "full":
        off = ~np.eye(n, dtype=bool)
        m = float(np.abs(W[off]).mean()) if n > 1 else 0.0
        W = np.where(off, m, 0.0)
    if kind == "shuffled":
        rng = np.random.default_rng(0) if rng is None else rng
        perm = rng.permutation(n)
        W = W[np.ix_(perm, perm)]
    return normalized(W)

def response_columns(kind, diagram, ids, alpha, rng=None) -> np.ndarray:
    """Matrix whose column j is the predicted relative response of every neuron to a poke on neuron j."""
    Wn = kind_matrix(kind, diagram, ids, rng)
    if kind == "one_step":
        return np.eye(len(ids)) + Wn
    if kind == "blind":
        return np.eye(len(ids))
    return propagator(Wn, alpha)
