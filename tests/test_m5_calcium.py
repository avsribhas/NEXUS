import numpy as np
from nexus.degrade.calcium import bleach, calcium_drive, fluorescence, indicator_state, saturate, to_frames

def test_drive_ignores_hyperpolarization():
    assert np.allclose(calcium_drive(np.array([-70.0, -60.0, -80.0]), -70.0, 10.0), [0.0, 1.0, 0.0])

def test_rise_and_decay_time_constants():
    dt = 1.0
    drive = np.concatenate([np.zeros(1), np.ones(2000), np.zeros(2000)])
    x = indicator_state(drive, dt, 50.0, 400.0)
    assert abs(x[50] - (1 - np.exp(-1.0))) < 0.02
    assert x[2000] > 0.99
    assert abs(x[2400] / x[2000] - np.exp(-1.0)) < 0.02

def test_works_on_many_traces():
    drive = np.zeros((3, 4, 100))
    assert indicator_state(drive, 1.0, 10.0, 100.0).shape == (3, 4, 100)

def test_saturation():
    assert abs(saturate(np.array([2.0]), 2.0, 2.0)[0] - 0.5) < 1e-9
    assert saturate(np.array([1000.0]), 2.0, 2.0)[0] > 0.99
    assert saturate(np.array([-1.0]), 2.0, 2.0)[0] == 0.0

def test_bleach():
    f = np.ones(1001)
    out = bleach(f, 10.0, 10.0)
    assert out[0] == 1.0
    assert abs(out[1000] - np.exp(-1.0)) < 1e-9

def test_frames():
    f = np.arange(10.0)
    out = to_frames(f, 100.0, 5.0)
    assert out.tolist() == [0.5, 2.5, 4.5, 6.5, 8.5]

def test_fluorescence_shape_and_noise():
    params = dict(rest_mV=-70.0, scale_mV=10.0, tau_rise_ms=50.0, tau_decay_ms=400.0, half_level=2.0, hill=2.0, tau_bleach_s=100.0, frame_rate_hz=2.0, noise_sd=0.1)
    v = np.full((5, 10000), -60.0)
    out = fluorescence(v, 1.0, params, np.random.default_rng(0))
    assert out.shape == (5, 20)
    
    params_no_noise = dict(params, noise_sd=0.0)
    out1 = fluorescence(v, 1.0, params_no_noise, np.random.default_rng(0))
    out2 = fluorescence(v, 1.0, params_no_noise, np.random.default_rng(1))
    assert np.array_equal(out1, out2)
