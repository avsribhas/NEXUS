import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m4_joint_check import joint_margins

def test_small_set_covers_the_largest_score():
    per_circuit = [(1, [(0, 1.0)]), (1, [(0, 2.0)]), (1, [(0, 3.0)]), (1, [(0, 4.0)])]
    out = joint_margins(per_circuit, [1.0])
    assert abs(out[1][0] - 4.0) < 1e-6

def test_never_narrower_than_the_per_field_widening():
    per_circuit = [(1, [(0, 1.0)]), (1, [(0, 2.0)]), (1, [(0, 3.0)]), (1, [(0, 4.0)])]
    out = joint_margins(per_circuit, [10.0])
    assert abs(out[1][0] - 10.0) < 1e-9

def test_gate_counts_get_separate_margins():
    per_circuit = [(1, [(0, 1.0)]), (1, [(0, 1.0)]), (2, [(0, 1.0), (0, 5.0)])]
    out = joint_margins(per_circuit, [0.0])
    assert set(out) == {1, 2}
    assert out[2][0] > out[1][0]
    assert abs(out[2][0] - 5.0) < 1e-6

def test_a_circuit_counts_its_worst_field():
    per_circuit = [(1, [(0, 2.0), (1, 0.0)]), (1, [(0, 0.0), (1, 2.0)])]
    out = joint_margins(per_circuit, [0.0, 0.0])
    assert abs(out[1][0] - 2.0) < 1e-6
    assert abs(out[1][1] - 2.0) < 1e-6
