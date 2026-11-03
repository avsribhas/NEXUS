import zlib
import numpy as np
from nexus.baselines.wiring import response_columns
from nexus.perturb.protocol import PAIRED, POKE_TYPES, STATES_MV

class LinearBaseline:
    """Gain-times-response-column baseline; kind picks the response matrix."""

    def __init__(self, kind="wiring", per_state=False, state_term=False, alphas=None, seed=0):
        self.kind = kind
        self.per_state = per_state
        self.state_term = state_term
        self.alphas = list(np.linspace(0.0, 0.95, 20)) if alphas is None else list(alphas)
        self.seed = seed
        self.alpha = None
        self.params = None

    def features(self, circuit, alpha) -> np.ndarray:
        """Response column of the poked neuron for every poke, shape (pokes, neurons)."""
        rng = np.random.default_rng([self.seed, zlib.crc32(circuit["circuit_id"].encode())])
        C = response_columns(self.kind, circuit["diagram"], circuit["ids"], alpha, rng)
        return np.stack([C[:, circuit["ids"].index(p["target"])] for p in circuit["pokes"]])

    def _solve(self, circuits, feats, Y, M) -> tuple:
        """Least-squares gains for one alpha; returns (params, total squared error)."""
        f = []
        y = []
        t = []
        s = []
        for c, feats_c in enumerate(feats):
            F = np.broadcast_to(feats_c[:, None, :], Y[c].shape)
            T = np.broadcast_to(np.array([p["type"] for p in circuits[c]["pokes"]])[:, None, None], Y[c].shape)
            S = np.broadcast_to(np.arange(len(STATES_MV))[None, :, None], Y[c].shape)
            keep = M[c] & np.isfinite(Y[c])
            f.extend(F[keep].flatten())
            y.extend(Y[c][keep].flatten())
            t.extend(T[keep].flatten())
            s.extend(S[keep].flatten())
        f = np.array(f)
        y = np.array(y)
        t = np.array(t)
        s = np.array(s)
        params = {}
        sse = 0.0
        for tn in list(POKE_TYPES) + [PAIRED["type"]]:
            for si in range(len(STATES_MV)) if self.per_state else [None]:
                sel = (t == tn) if si is None else ((t == tn) & (s == si))
                if not np.any(sel):
                    continue
                ff = f[sel]
                yy = y[sel]
                if self.state_term:
                    X = np.stack([ff, ff * np.asarray(STATES_MV)[s[sel]] / 30.0], axis=1)
                    coef = np.linalg.lstsq(X, yy, rcond=None)[0]
                    pred = X @ coef
                else:
                    denom = float(np.sum(ff * ff))
                    coef = np.array([float(np.sum(ff * yy)) / denom if denom > 0 else 0.0])
                    pred = coef[0] * ff
                params[(tn, si)] = coef
                sse += float(np.sum((yy - pred) ** 2))
        return params, sse

    def fit(self, circuits, Y, M) -> "LinearBaseline":
        """Pick alpha from the grid by total squared error and keep its gains."""
        best = None
        for alpha in self.alphas:
            feats = [self.features(c, alpha) for c in circuits]
            params, sse = self._solve(circuits, feats, Y, M)
            if best is None or sse < best[0]:
                best = (sse, alpha, params)
        self.alpha = best[1]
        self.params = best[2]
        return self

    def _coef(self, type_name, state_index) -> np.ndarray:
        """Gains for a poke type and state; a state never seen in fitting uses the nearest fitted state."""
        if not self.per_state:
            return self.params.get((type_name, None), np.zeros(2 if self.state_term else 1))
        fitted = [si for (tn, si) in self.params if tn == type_name]
        if not fitted:
            return np.zeros(2 if self.state_term else 1)
        if state_index in fitted:
            return self.params[(type_name, state_index)]
        nearest = min(fitted, key=lambda si: (abs(STATES_MV[si] - STATES_MV[state_index]), si))
        return self.params[(type_name, nearest)]

    def predict(self, circuit) -> np.ndarray:
        """Predicted responses, shape (pokes, states, neurons)."""
        F = self.features(circuit, self.alpha)
        out = np.zeros((len(circuit["pokes"]), len(STATES_MV), len(circuit["ids"])))
        for p, poke in enumerate(circuit["pokes"]):
            for si in range(len(STATES_MV)):
                coef = self._coef(poke["type"], si)
                out[p, si] = coef[0] * F[p]
                if self.state_term:
                    out[p, si] += coef[1] * F[p] * STATES_MV[si] / 30.0
        return out
