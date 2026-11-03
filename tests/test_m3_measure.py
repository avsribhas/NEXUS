import math
import numpy as np
import pytest
from nexus.circuits.measure import classify_ratio, crossing, gate_ratio, priority_ranks, pulse_response, time_to_fraction, window_mean

def test_window_mean():
    v = np.arange(10.0)
    assert window_mean(v, 1.0, 2.0, 5.0) == 3.0
    v2 = np.stack([v, 2 * v])
    assert np.allclose(window_mean(v2, 1.0, 2.0, 5.0), [3.0, 6.0])

def test_pulse_response():
    v = np.zeros(1000)
    v[500:] = 4.0
    assert abs(pulse_response(v, 0.5, (100.0, 200.0), (300.0, 400.0)) - 4.0) < 1e-12

def test_crossing_interpolates():
    assert abs(crossing([0.0, 10.0, 20.0], [0.0, 1.0, 3.0], 2.0) - 15.0) < 1e-12

def test_crossing_falling():
    assert abs(crossing([0.0, 10.0], [4.0, 2.0], 3.0) - 5.0) < 1e-12

def test_crossing_never():
    assert math.isnan(crossing([0.0, 10.0], [0.0, 1.0], 5.0))

def test_time_to_fraction_exponential():
    dt = 0.1
    t = np.arange(0, 3000.0, dt)
    tau = 40.0
    v = np.where(t < 1000.0, -70.0, -70.0 + 10.0 * (1.0 - np.exp(-(t - 1000.0) / tau)))
    assert abs(time_to_fraction(v, dt, 1000.0, 2000.0) - tau) < 0.5

def test_time_to_fraction_falling():
    dt = 0.1
    t = np.arange(0, 3000.0, dt)
    tau = 25.0
    v = np.where(t < 1000.0, -60.0, -60.0 - 5.0 * (1.0 - np.exp(-(t - 1000.0) / tau)))
    assert abs(time_to_fraction(v, dt, 1000.0, 2000.0) - tau) < 0.5

def test_time_to_fraction_flat():
    assert math.isnan(time_to_fraction(np.full(30000, -70.0), 0.1, 1000.0, 2000.0))

def test_gate_ratio():
    assert gate_ratio(2.0, 6.0) == 3.0 and gate_ratio(-2.0, -1.0) == 0.5 and gate_ratio(0.0, 1.0) == math.inf

def test_classify_ratio():
    assert classify_ratio(3.0) == "switch" and classify_ratio(0.4) == "switch" and classify_ratio(1.2) == "none" and classify_ratio(0.8) == "none" and classify_ratio(1.7) == "ambiguous" and classify_ratio(0.6) == "ambiguous"

def test_priority_ranks():
    assert priority_ranks([0.5, 2.0, 1.0]) == [3, 1, 2] and priority_ranks([1.0, 1.0]) == [1, 2] and priority_ranks([0.3]) == [1]
