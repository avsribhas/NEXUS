import re
from datetime import datetime, timezone

import pynwb
import pytest
from pynwb import NWBFile, NWBHDF5IO

from nexus.format.metadata import collect_run_metadata, make_run_metadata, read_run_metadata
from nexus.format.pokes import add_pokes, read_pokes

METADATA_KEYS = {"schema_version", "simulated", "simulator", "simulator_version", "pynwb_version", "seed", "thread_count", "gpu_model", "gpu_driver", "git_commit", "git_dirty"}
POKES = [
    {"start_time": 1.0, "stop_time": 1.5, "target_neuron": "AVAL", "amplitude": 0.2, "amplitude_unit": "nA", "modality": "current"},
    {"start_time": 3.0, "stop_time": 3.25, "target_neuron": "AVAR", "amplitude": -0.1, "amplitude_unit": "nA", "modality": "current"},
]

def new_file() -> NWBFile:
    return NWBFile(session_description="test", identifier="test-id", session_start_time=datetime(2026, 10, 29, tzinfo=timezone.utc))

def cpu_metadata() -> dict:
    return collect_run_metadata(seed=7, simulated=True, simulator="jaxley", simulator_version="0.14.0", thread_count=1, use_gpu=False)

def write_and_read(nwbfile, path):
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    io = NWBHDF5IO(str(path), "r")
    return io, io.read()

def test_collect_run_metadata_keys():
    assert set(cpu_metadata()) == METADATA_KEYS

def test_collect_run_metadata_values():
    md = cpu_metadata()
    assert md["seed"] == 7
    assert md["simulated"] is True
    assert md["simulator"] == "jaxley"
    assert md["simulator_version"] == "0.14.0"
    assert md["pynwb_version"] == pynwb.__version__
    assert md["thread_count"] == 1
    assert md["schema_version"] == "0.1.0"
    assert re.fullmatch(r"[0-9a-f]{40}", md["git_commit"])
    assert isinstance(md["git_dirty"], bool)

def test_gpu_fields_none_on_cpu():
    md = cpu_metadata()
    assert md["gpu_model"] is None
    assert md["gpu_driver"] is None

def test_run_metadata_round_trip(tmp_path):
    md = cpu_metadata()
    f = new_file()
    f.add_lab_meta_data(make_run_metadata(md))
    io, g = write_and_read(f, tmp_path / "run.nwb")
    try:
        assert read_run_metadata(g) == md
    finally:
        io.close()

def test_pokes_round_trip(tmp_path):
    f = new_file(); add_pokes(f, POKES)
    io, g = write_and_read(f, tmp_path / "pokes.nwb")
    try:
        assert read_pokes(g) == POKES
    finally:
        io.close()

def test_poke_effector_name(tmp_path):
    f = new_file(); add_pokes(f, POKES)
    io, g = write_and_read(f, tmp_path / "eff.nwb")
    try:
        meta = g.lab_meta_data["optogenetic_experiment_metadata"]
        assert "simulated_current_injection" in meta.optogenetic_effectors.effectors
    finally:
        io.close()

def test_poke_tables_are_ndx_optogenetics(tmp_path):
    f = new_file(); add_pokes(f, POKES)
    io, g = write_and_read(f, tmp_path / "tables.nwb")
    try:
        assert type(g.intervals["optogenetic_pulses"]).__name__ == "OptogeneticPulsesTable"
        assert type(g.intervals["optogenetic_epochs"]).__name__ == "OptogeneticEpochsTable"
        assert len(g.intervals["optogenetic_epochs"]) == len(POKES)
    finally:
        io.close()

def test_add_pokes_rejects_empty_list():
    with pytest.raises(ValueError): add_pokes(new_file(), [])
