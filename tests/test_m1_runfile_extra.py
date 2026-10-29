from datetime import datetime, timezone

from nexus.format import runfile
from nexus.format.checker import check_metadata
from nexus.format.metadata import collect_run_metadata, make_run_metadata
from nexus.format.runfile import new_run_file

def make(**kwargs):
    base = {"identifier": "rf-1", "session_description": "runfile test", "session_start_time": datetime(2026, 10, 29, tzinfo=timezone.utc), "subject_id": "animal-001", "species": "Caenorhabditis elegans"}
    base.update(kwargs)
    return new_run_file(**base)

def test_worm_sex_defaults_to_hermaphrodite():
    assert make().subject.sex == "XX"

def test_explicit_sex_is_used():
    assert make(species="Drosophila melanogaster", sex="F").subject.sex == "F"

def test_age_recorded_when_given():
    s = make(age="PT50H").subject
    assert s.age == "PT50H"
    assert s.age__reference == "birth"

def test_age_absent_when_not_given():
    assert make().subject.age is None

def test_module_lists_not_shared():
    f = make()
    assert f.keywords is not runfile.KEYWORDS
    assert list(runfile.KEYWORDS) == ["NEXUS", "state-dependent gating", "simulation"]

def test_git_commit_with_trailing_text_rejected():
    md = collect_run_metadata(seed=1, simulated=False, simulator="none", simulator_version="none", thread_count=1)
    md["git_commit"] = md["git_commit"] + "junk"
    f = make()
    f.add_lab_meta_data(make_run_metadata(md))
    assert any("git_commit" in p for p in check_metadata(f))
