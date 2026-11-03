import numpy as np
import torch
from nexus.perturb.protocol import STATES_MV

EDGE_KINDS = ("exc", "inh", "shunt", "gap")
N_NODE = 8
SCALE_MV = 10.0

def adjacency(diagram, ids) -> np.ndarray:
    """Edge weights by kind, shape (4, post, pre), gap junctions both ways, each row divided by one plus its total weight."""
    index = {nid: i for i, nid in enumerate(ids)}
    A = np.zeros((len(EDGE_KINDS), len(ids), len(ids)))
    for e in diagram:
        k = EDGE_KINDS.index(e["sign"])
        i = index[e["post"]]
        j = index[e["pre"]]
        A[k, i, j] += e["weight"]
        if e["sign"] == "gap":
            A[k, j, i] += e["weight"]
    total = A.sum(axis=(0, 2))
    return A / (1.0 + total)[None, :, None]

def node_features(animal, states=STATES_MV) -> np.ndarray:
    """Per-neuron inputs for every poke and state: poke size and length on the target, hold level on the held neuron, the state neuron, the state level and the input neurons."""
    ids = animal["ids"]
    n = len(ids)
    x = np.zeros((len(animal["pokes"]), len(states), n, N_NODE))
    for p, poke in enumerate(animal["pokes"]):
        t = ids.index(poke["target"])
        x[p, :, t, 0] = poke["amplitude_mV"] / 30.0
        x[p, :, t, 1] = (poke["stop_ms"] - poke["start_ms"]) / 300.0
        x[p, :, t, 2] = 1.0
        if poke["hold"]:
            h = ids.index(poke["hold"])
            x[p, :, h, 3] = poke["hold_mV"] / 30.0
            x[p, :, h, 4] = 1.0
    x[:, :, ids.index(animal["roles"]["state"]), 5] = 1.0
    for s, v in enumerate(states):
        x[:, s, :, 6] = v / 30.0
    for nid in animal["roles"]["inputs"]:
        x[:, :, ids.index(nid), 7] = 1.0
    return x

class MessageLayer(torch.nn.Module):
    """One round of message passing: sums from each edge kind, and their products with the neuron's own state, through an MLP with a residual and LayerNorm."""
    def __init__(self, hidden):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(9 * hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden)
        )
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, h, A):
        agg = [torch.einsum("ij,bjh->bih", A[k], h) for k in range(A.shape[0])]
        m = torch.cat([h] + agg + [h * a for a in agg], dim=-1)
        return self.norm(h + self.mlp(m))

class ResponseNet(torch.nn.Module):
    """Encoder, message-passing layers and a decoder that gives one response per neuron."""
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.enc = torch.nn.Linear(N_NODE, hidden)
        self.layers = torch.nn.ModuleList([MessageLayer(hidden) for _ in range(layers)])
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x, A):
        h = self.enc(x)
        for layer in self.layers:
            h = layer(h, A)
        return self.dec(h).squeeze(-1)

class DuskResponse:
    """Message-passing response model trained across circuits; predicts responses in mV, shape (pokes, states, neurons)."""
    def __init__(self, hidden=64, layers=4, epochs=200, lr=1e-3, seed=0, device="cpu"):
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.device = device
        self.net = None

    def _tensors(self, animal, states=STATES_MV) -> tuple:
        """Node features and adjacency for one animal as tensors."""
        x = torch.tensor(node_features(animal, states), dtype=torch.float32, device=self.device)
        A = torch.tensor(adjacency(animal["diagram"], animal["ids"]), dtype=torch.float32, device=self.device)
        return x, A

    def _forward(self, x, A) -> torch.Tensor:
        """Run the network on every poke and state at once."""
        P, S, n, F = x.shape
        return self.net(x.reshape(P * S, n, F), A).reshape(P, S, n)

    def fit(self, animals) -> "DuskResponse":
        """Train on the seed-averaged responses of every training animal, weighting the state-dependent part twice."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = ResponseNet(self.hidden, self.layers).to(self.device)
        data = []
        for a in animals:
            x, A = self._tensors(a)
            y = np.nan_to_num(np.nanmean(a["responses"], axis=2), nan=0.0) / SCALE_MV
            data.append((x, A, torch.tensor(y, dtype=torch.float32, device=self.device)))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for i in rng.permutation(len(data)):
                x, A, y = data[i]
                pred = self._forward(x, A)
                sd_pred = pred - pred.mean(dim=1, keepdim=True)
                sd_y = y - y.mean(dim=1, keepdim=True)
                loss = torch.mean((pred - y) ** 2) + torch.mean((sd_pred - sd_y) ** 2)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    def predict(self, animal, states=STATES_MV) -> np.ndarray:
        """Predicted responses in mV, shape (pokes, states, neurons)."""
        x, A = self._tensors(animal, states)
        with torch.no_grad():
            return self._forward(x, A).cpu().numpy() * SCALE_MV
