import math
import numpy as np
import torch
from nexus.models.summary import N_FEATURES, candidate_features, mean_traces
from nexus.models.data import poke_responses
from nexus.score.recovery import CONTINUOUS_FIELDS

QUANTILES = (0.05, 0.95)

def to_scale(value, scale) -> float:
    """Natural units to scoring scale; nan when a log scale gets a non-positive value."""
    if scale == "log10":
        return math.log10(value) if value is not None and value > 0 else math.nan
    return float(value) if value is not None else math.nan

def from_scale(value, scale) -> float:
    """Scoring scale back to natural units."""
    return float(10.0 ** value) if scale == "log10" else float(value)

def animal_rows(animal, config) -> tuple:
    """Features, switch labels, field targets and target masks for every candidate trigger of one animal."""
    traces = mean_traces(animal)
    responses = poke_responses(animal)
    planted = {d["trigger_channel"]: d for d in animal["datasheets"]}
    X = []
    L = []
    Y = []
    triggers = []
    for trig in animal["roles"]["inputs"]:
        X.append(candidate_features(animal, trig, traces, responses))
        L.append(1.0 if trig in planted else 0.0)
        d = planted.get(trig)
        Y.append([to_scale(d[f], config["fields"][f]["scale"]) if d is not None else math.nan for f in CONTINUOUS_FIELDS])
        triggers.append(trig)
    Y = np.array(Y, dtype=float)
    return np.array(X, dtype=float), np.array(L, dtype=float), np.nan_to_num(Y, nan=0.0), np.isfinite(Y).astype(float), triggers

class DuskRecovery:
    """Switch probability and quantile intervals per candidate trigger, from measured summaries."""
    def __init__(self, config, hidden=128, epochs=400, lr=1e-3, seed=0):
        self.config = config
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.net = None
        self.mu = None
        self.sd = None

    def _make(self) -> torch.nn.Module:
        """The network: two hidden layers; output 1 logit plus a low and a high quantile per field."""
        n_out = 1 + 2 * len(CONTINUOUS_FIELDS)
        return torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, n_out)
        )

    def fit(self, animals) -> "DuskRecovery":
        """Train on every candidate trigger of every training animal."""
        rows = [animal_rows(a, self.config) for a in animals]
        X = np.concatenate([r[0] for r in rows])
        L = np.concatenate([r[1] for r in rows])
        Y = np.concatenate([r[2] for r in rows])
        M = np.concatenate([r[3] for r in rows])
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        torch.manual_seed(self.seed)
        self.net = self._make()
        x = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        l = torch.tensor(L, dtype=torch.float32)
        y = torch.tensor(Y, dtype=torch.float32)
        m = torch.tensor(M, dtype=torch.float32) * l[:, None]
        q = torch.tensor(QUANTILES, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self.net(x)
            logit = out[:, 0]
            quant = out[:, 1:].reshape(-1, len(CONTINUOUS_FIELDS), 2)
            err = y[:, :, None] - quant
            pinball = torch.maximum(q * err, (q - 1.0) * err)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, l) + (pinball * m[:, :, None]).sum() / (m.sum() * 2.0 + 1e-6)
            loss.backward()
            opt.step()
        return self

    def predict_rows(self, animal) -> tuple:
        """Switch probability and (low, high) per field on the scoring scale, for every candidate trigger."""
        X, _, _, _, triggers = animal_rows(animal, self.config)
        with torch.no_grad():
            out = self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)).numpy()
        prob = 1.0 / (1.0 + np.exp(-out[:, 0]))
        quant = np.sort(out[:, 1:].reshape(-1, len(CONTINUOUS_FIELDS), 2), axis=2)
        return triggers, prob, quant

    def predict(self, animal, threshold=0.5) -> dict:
        """Dusk's answer for one animal: no switch, or one datasheet per trigger it believes in, with intervals in natural units."""
        triggers, prob, quant = self.predict_rows(animal)
        sheets = []
        gain = list(CONTINUOUS_FIELDS).index("trigger_gain")
        for i, trig in enumerate(triggers):
            if prob[i] < threshold:
                continue
            s = {"trigger_channel": trig, "probability": float(prob[i]), "priority": 0}
            for j, f in enumerate(CONTINUOUS_FIELDS):
                scale = self.config["fields"][f]["scale"]
                s[f] = [from_scale(quant[i, j, 0], scale), from_scale(quant[i, j, 1], scale)]
            s["_size"] = abs(float(quant[i, gain].mean()))
            sheets.append(s)
        order = sorted(range(len(sheets)), key=lambda k: -sheets[k]["_size"])
        for rank, k in enumerate(order):
            sheets[k]["priority"] = rank + 1
        for s in sheets:
            del s["_size"]
        return {"switch": bool(sheets), "datasheets": sheets}
