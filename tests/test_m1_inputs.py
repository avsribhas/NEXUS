from datetime import datetime, timezone

import numpy as np
from pynwb import NWBFile, NWBHDF5IO

from nexus.format.inputs import InputRecorder, check_inputs_match, read_inputs, write_inputs

def toy_simulator(params: dict) -> np.ndarray:
    return params["g_leak"] * params["drive"]

def requested_params() -> dict:
    return {"g_leak": np.array([1.0, 2.0, 3.0]), "drive": np.array([0.5, 0.5, 0.5])}

def honest_generator(recorder):
    params = requested_params()
    out = recorder(params)
    return params, out

def realizing_generator(recorder):
    requested = requested_params()
    realized = {"g_leak": np.full(3, requested["g_leak"].mean()), "drive": requested["drive"].copy()}
    out = recorder(realized)
    return requested, out

def test_received_is_none_before_call():
    assert InputRecorder(toy_simulator).received is None

def test_recorder_returns_simulator_output():
    rec = InputRecorder(toy_simulator)
    out = rec(requested_params())
    assert np.array_equal(out, np.array([0.5, 1.0, 1.5]))

def test_recorder_stores_a_copy():
    rec = InputRecorder(toy_simulator)
    p = requested_params()
    rec(p)
    p["g_leak"][0] = 99.0
    assert rec.received["g_leak"][0] == 1.0

def test_inputs_round_trip(tmp_path):
    f = NWBFile(session_description="test", identifier="inputs-id", session_start_time=datetime(2026, 10, 29, tzinfo=timezone.utc))
    params = requested_params()
    write_inputs(f, params)
    path = tmp_path / "inputs.nwb"
    with NWBHDF5IO(str(path), "w") as io: io.write(f)
    with NWBHDF5IO(str(path), "r") as io:
        got = read_inputs(io.read())
        assert sorted(got) == sorted(params)
        for key in params: assert np.array_equal(got[key], params[key])

def test_honest_generator_passes():
    rec = InputRecorder(toy_simulator)
    stored, _ = honest_generator(rec)
    assert check_inputs_match(stored, rec.received) == []

def test_realizing_generator_is_caught():
    rec = InputRecorder(toy_simulator)
    stored, _ = realizing_generator(rec)
    assert check_inputs_match(stored, rec.received) == ["g_leak"]

def test_missing_and_extra_keys_reported():
    assert check_inputs_match({"a": np.zeros(1)}, {"b": np.zeros(1)}) == ["a", "b"]
