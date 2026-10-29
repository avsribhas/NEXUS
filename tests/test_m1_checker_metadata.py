from datetime import datetime, timezone

from pynwb import NWBFile, NWBHDF5IO

from nexus.format.checker import check_file, check_metadata
from nexus.format.metadata import collect_run_metadata, make_run_metadata
from nexus.format.runfile import new_run_file

def build(with_metadata=True, subject_id="animal-001"):
    f = new_run_file(identifier="chk-1", session_description="checker test", session_start_time=datetime(2026, 10, 29, tzinfo=timezone.utc), subject_id=subject_id, species="Caenorhabditis elegans")
    if with_metadata:
        f.add_lab_meta_data(make_run_metadata(collect_run_metadata(seed=1, simulated=False, simulator="none", simulator_version="none", thread_count=1)))
    return f

def test_new_run_file_sets_subject():
    f = build()
    assert f.subject is not None
    assert f.subject.subject_id == "animal-001"
    assert f.subject.species == "Caenorhabditis elegans"

def test_new_run_file_sets_general_metadata():
    f = build()
    assert len(f.experimenter) >= 1
    assert f.institution
    assert f.experiment_description
    assert f.keywords is not None and len(f.keywords) >= 1

def test_check_metadata_passes_good_file():
    assert check_metadata(build()) == []

def test_check_metadata_flags_missing_run_metadata():
    problems = check_metadata(build(with_metadata=False))
    assert any("nexus_run_metadata" in p for p in problems)

def test_check_metadata_flags_missing_subject():
    f = NWBFile(session_description="no subject", identifier="chk-2", session_start_time=datetime(2026, 10, 29, tzinfo=timezone.utc))
    f.add_lab_meta_data(make_run_metadata(collect_run_metadata(seed=1, simulated=False, simulator="none", simulator_version="none", thread_count=1)))
    problems = check_metadata(f)
    assert any("subject" in p.lower() for p in problems)

def test_check_metadata_flags_empty_subject_id():
    problems = check_metadata(build(subject_id=""))
    assert any("subject_id" in p for p in problems)

def test_check_file_reads_from_disk(tmp_path):
    path = tmp_path / "real.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(build())
    assert check_file(path) == []
