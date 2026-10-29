import os

import pytest
import torch

from nexus.models.repro import set_deterministic, train_tiny

def test_set_deterministic_configures_torch():
    set_deterministic(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()

def test_cpu_bitwise_same_seed():
    a = train_tiny(seed=0, device="cpu")
    b = train_tiny(seed=0, device="cpu")
    assert torch.equal(a, b)

def test_cpu_different_seed_differs():
    assert not torch.equal(train_tiny(seed=0, device="cpu"), train_tiny(seed=1, device="cpu"))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_bitwise_same_seed():
    a = train_tiny(seed=0, device="cuda")
    b = train_tiny(seed=0, device="cuda")
    assert torch.equal(a, b)
