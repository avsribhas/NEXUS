import os
import random
import numpy as np
import torch

def set_deterministic(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def train_tiny(seed: int, device: str) -> torch.Tensor:
    set_deterministic(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(64, 8, generator=g).to(device)
    y = torch.randn(64, 1, generator=g).to(device)
    index = torch.randint(0, 16, (64,), generator=g).to(device)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(20):
        optimizer.zero_grad()
        h = model(x)
        agg = torch.zeros(16, 1, device=device).index_add(0, index, h)
        loss = ((agg[index] - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
