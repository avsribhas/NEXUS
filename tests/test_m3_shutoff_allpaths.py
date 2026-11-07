import importlib.util
import math
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("m3_shutoff_allpaths", Path(__file__).resolve().parents[1] / "scripts" / "m3_shutoff_allpaths.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def sheet(**kw):
    out = {"trigger_channel": "trig", "trigger_gain": 0.1, "threshold": 10.0, "tau_on_over_tau_ref": 1.0, "tau_off_over_tau_ref": 1.0, "reach": 0.4, "gate_id": "g0"}
    out.update(kw)
    return out

CONFIG = {"fields": {"trigger_gain": {"scale": "log10"}, "threshold": {"scale": "linear"}, "tau_on_over_tau_ref": {"scale": "log10"}, "tau_off_over_tau_ref": {"scale": "log10"}, "reach": {"scale": "linear"}}}

def test_mean_is_geometric_for_log_fields():
    result = mod.mean_datasheet([sheet(trigger_gain=0.01), sheet(trigger_gain=1.0)])
    assert abs(result["trigger_gain"] - 0.1) < 1e-9, f"Expected 0.1, got {result['trigger_gain']}"

def test_mean_is_plain_for_linear_fields():
    result = mod.mean_datasheet([sheet(threshold=10.0, reach=0.2), sheet(threshold=20.0, reach=0.6)])
    assert abs(result["threshold"] - 15.0) < 1e-9, f"Expected 15.0, got {result['threshold']}"
    assert abs(result["reach"] - 0.4) < 1e-9, f"Expected 0.4, got {result['reach']}"

def test_mean_is_nan_when_a_log_value_is_zero():
    result = mod.mean_datasheet([sheet(trigger_gain=0.0), sheet(trigger_gain=1.0)])
    assert math.isnan(result["trigger_gain"]), f"Expected NaN, got {result['trigger_gain']}"

def test_mean_keeps_other_keys():
    result = mod.mean_datasheet([sheet(), sheet()])
    assert result["trigger_channel"] == "trig", f"Expected 'trig', got {result['trigger_channel']}"

def test_replaced_rows_changes_only_the_trigger():
    X = np.zeros((2, 3))
    L = np.array([1.0, 0.0])
    Y = np.zeros((2, 5))
    M = np.array([[1.0]*5, [0.0]*5])
    triggers = ["trig", "in1"]
    new_trigger = sheet(trigger_gain=0.01, threshold=12.0)
    X2, L2, Y2, M2, T2 = mod.replaced_rows((X, L, Y, M, triggers), new_trigger, CONFIG)
    assert abs(Y2[0, 0] + 2.0) < 1e-9, f"Expected -2.0, got {Y2[0, 0]}"
    assert abs(Y2[0, 1] - 12.0) < 1e-9, f"Expected 12.0, got {Y2[0, 1]}"
    assert np.all(Y2[1] == 0.0), f"Expected all zeros, got {Y2[1]}"
    assert np.all(M2[1] == 0.0), f"Expected all zeros, got {M2[1]}"
    assert np.all(Y == 0.0), f"Expected all zeros, got {Y}"

import numpy as np
from unittest.mock import patch

def test_replaced_rows_masks_nan():
    X = np.zeros((2, 3))
    L = np.array([1.0, 0.0])
    Y = np.zeros((2, 5))
    M = np.array([[1.0]*5, [0.0]*5])
    triggers = ["trig", "in1"]
    X2, L2, Y2, M2, T2 = mod.replaced_rows((X, L, Y, M, triggers), sheet(trigger_gain=0.0), CONFIG)
    assert M2[0, 0] == 0.0 and Y2[0, 0] == 0.0
