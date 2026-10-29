import math
from datetime import datetime, timezone

from nwbinspector import Importance, inspect_nwbfile
from pynwb import NWBHDF5IO

from nexus.format.checker import check_file, check_ground_truth
from nexus.format.ground_truth import make_ground_truth, read_ground_truth
from nexus.format.inputs import write_inputs
from nexus.format.metadata import collect_run_metadata, make_run_metadata
from nexus.format.pokes import add_pokes
from nexus.format.runfile import new_run_file

import numpy as np

GATE = {"gate_id": "g1", "trigger_channel": "state_input", "trigger_gain": 1.7, "threshold": 0.42, "threshold_unit": "nA", "tau_on_over_tau_ref": 2.5, "tau_off_over_tau_ref": 4.0, "tau_ref_definition": "trigger_time_constant", "reach": 0.3, "reach_criterion": "0.1 of baseline activity", "priority": 1, "cost_atp_per_flip": None}
CORRUPTION = [{"pre_neuron": "AVAL", "post_neuron": "AVBL", "kind": "sign_flip", "true_weight": 0.8, "stored_weight": -0.8}]
POKE = {"start_time": 1.0, "stop_time": 1.5, "target_neuron": "AVAL", "amplitude": 0.2, "amplitude_unit": "nA", "modality": "current"}

def build(switch_type="gain", is_decoy=False, datasheets=None, animal_id="animal-001", subject_id="animal-001"):
    if datasheets is None:
        datasheets = [dict(GATE)]
    f = new_run_file(identifier="gt-1", session_description="ground truth test", session_start_time=datetime(2026, 10, 29, tzinfo=timezone.utc), subject_id=subject_id, species="Caenorhabditis elegans")
    f.add_lab_meta_data(make_run_metadata(collect_run_metadata(seed=1, simulated=True, simulator="jaxley", simulator_version="0.14.0", thread_count=1)))
    f.add_lab_meta_data(make_ground_truth(switch_type=switch_type, is_decoy=is_decoy, twin_id="", base_circuit_id="base-1", animal_id=animal_id, datasheets=datasheets, wiring_corruption=CORRUPTION))
    return f

def write(f, path):
    with NWBHDF5IO(str(path), "w") as io:
        io.write(f)
    return path

def test_ground_truth_round_trip(tmp_path):
    path = write(build(), tmp_path / "gt.nwb")
    with NWBHDF5IO(str(path), "r") as io:
        gt = read_ground_truth(io.read())
    assert gt["switch_type"] == "gain"
    assert gt["is_decoy"] is False
    assert gt["animal_id"] == "animal-001"
    assert gt["datasheets"] == [GATE]
    assert gt["wiring_corruption"] == CORRUPTION

def test_no_switch_round_trip_has_no_datasheets(tmp_path):
    path = write(build(switch_type="none", datasheets=[]), tmp_path / "none.nwb")
    with NWBHDF5IO(str(path), "r") as io:
        gt = read_ground_truth(io.read())
    assert gt["datasheets"] == []

def test_cost_none_survives_round_trip(tmp_path):
    path = write(build(), tmp_path / "cost.nwb")
    with NWBHDF5IO(str(path), "r") as io:
        gt = read_ground_truth(io.read())
    assert gt["datasheets"][0]["cost_atp_per_flip"] is None

def test_check_passes_valid_switch_circuit():
    assert check_ground_truth(build()) == []

def test_check_passes_valid_no_switch_circuit():
    assert check_ground_truth(build(switch_type="none", datasheets=[])) == []

def test_check_rejects_switch_without_datasheets():
    assert check_ground_truth(build(switch_type="gain", datasheets=[])) != []

def test_check_rejects_no_switch_with_datasheets():
    assert check_ground_truth(build(switch_type="none")) != []

def test_check_rejects_decoy_with_a_switch_type():
    assert check_ground_truth(build(switch_type="gain", is_decoy=True)) != []

def test_check_rejects_bad_fields():
    for change in [{"reach": 1.5}, {"priority": 0}, {"threshold": math.nan}, {"tau_ref_definition": "other"}]:
        sheet = dict(GATE); sheet.update(change)
        assert check_ground_truth(build(datasheets=[sheet])) != [], change

def test_check_cost_rules():
    assert check_ground_truth(build(), allow_null_cost=True) == []
    assert check_ground_truth(build(), allow_null_cost=False) != []

def test_full_file_all_extensions(tmp_path):
    f = build()
    add_pokes(f, [POKE])
    write_inputs(f, {"g_leak": np.array([1.0, 2.0])})
    path = write(f, tmp_path / "full.nwb")
    assert check_file(path) == []
    critical = [m for m in inspect_nwbfile(nwbfile_path=str(path)) if m.importance == Importance.CRITICAL]
    assert critical == [], [m.message for m in critical]

def test_check_rejects_subject_not_animal():
    assert check_ground_truth(build(subject_id="animal-999")) != []

def test_check_rejects_simulated_file_without_ground_truth():
    f = new_run_file(identifier="gt-2", session_description="no ground truth", session_start_time=datetime(2026, 10, 29, tzinfo=timezone.utc), subject_id="animal-001", species="Caenorhabditis elegans")
    f.add_lab_meta_data(make_run_metadata(collect_run_metadata(seed=1, simulated=True, simulator="jaxley", simulator_version="0.14.0", thread_count=1)))
    assert check_ground_truth(f) != []

def test_check_rejects_unknown_corruption_kind():
    bad = [dict(CORRUPTION[0], kind="other")]
    f = new_run_file(identifier="gt-3", session_description="bad kind", session_start_time=datetime(2026, 10, 29, tzinfo=timezone.utc), subject_id="animal-001", species="Caenorhabditis elegans")
    f.add_lab_meta_data(make_run_metadata(collect_run_metadata(seed=1, simulated=True, simulator="jaxley", simulator_version="0.14.0", thread_count=1)))
    f.add_lab_meta_data(make_ground_truth(switch_type="gain", is_decoy=False, twin_id="", base_circuit_id="base-1", animal_id="animal-001", datasheets=[dict(GATE)], wiring_corruption=bad))
    assert check_ground_truth(f) != []

def test_check_rejects_duplicate_priority():
    second = dict(GATE, gate_id="g2")
    assert check_ground_truth(build(datasheets=[dict(GATE), second])) != []

def test_check_rejects_priority_gap():
    second = dict(GATE, gate_id="g2", priority=3)
    assert check_ground_truth(build(datasheets=[dict(GATE), second])) != []
