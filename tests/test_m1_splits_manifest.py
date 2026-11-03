import json
import random
import pytest
from nexus.format.splits import make_split_manifest, read_manifest, write_manifest

CIRCUITS = [f"c{i:02d}" for i in range(10)]
NEURONS = {c: ["n0", "n1", "n2", "n3", "n4"] for c in CIRCUITS}
NUDGES = ["graded_single", "competing_pair", "state_variant"]
STATES = ["s0", "s1", "s2", "s3"]

def manifest(seed=0):
    return make_split_manifest(CIRCUITS, NEURONS, NUDGES, STATES, seed=seed)

def test_same_seed_same_manifest():
    assert manifest(0) == manifest(0)

def test_input_order_does_not_matter():
    shuffled_circuits = list(CIRCUITS)
    random.Random(5).shuffle(shuffled_circuits)
    shuffled_nudges = list(NUDGES)
    random.Random(5).shuffle(shuffled_nudges)
    shuffled_states = list(STATES)
    random.Random(5).shuffle(shuffled_states)
    assert make_split_manifest(shuffled_circuits, NEURONS, shuffled_nudges, shuffled_states, 0) == manifest(0)

def test_different_seed_changes_circuit_split():
    circuits = [f"c{i:02d}" for i in range(20)]
    neurons_by_circuit = {c: NEURONS["c00"] for c in circuits}
    m0 = make_split_manifest(circuits, neurons_by_circuit, NUDGES, STATES, seed=0)
    m1 = make_split_manifest(circuits, neurons_by_circuit, NUDGES, STATES, seed=1)
    assert m0["circuits"] != m1["circuits"]

def test_each_kind_is_a_partition():
    m = manifest()
    EXPECTED = {
        "circuits": set(CIRCUITS),
        "nudge_types": set(NUDGES),
        "states": set(STATES)
    }
    for kind in EXPECTED:
        tr, va, te = set(m[kind]["train"]), set(m[kind]["val"]), set(m[kind]["test"])
        assert tr and va and te
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        assert tr | va | te == EXPECTED[kind]
    for circuit in m["neurons"]:
        tr, va, te = set(m["neurons"][circuit]["train"]), set(m["neurons"][circuit]["val"]), set(m["neurons"][circuit]["test"])
        assert tr and va and te
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        assert tr | va | te == set(NEURONS[circuit])

def test_split_sizes():
    m = manifest()
    assert len(m["circuits"]["train"]) == 6
    assert len(m["circuits"]["val"]) == 2
    assert len(m["circuits"]["test"]) == 2
    assert len(m["states"]["train"]) == 2
    assert len(m["states"]["val"]) == 1
    assert len(m["states"]["test"]) == 1
    assert len(m["nudge_types"]["train"]) == 1
    assert len(m["nudge_types"]["val"]) == 1
    assert len(m["nudge_types"]["test"]) == 1
    for circuit in m["neurons"]:
        assert len(m["neurons"][circuit]["train"]) == 3
        assert len(m["neurons"][circuit]["val"]) == 1
        assert len(m["neurons"][circuit]["test"]) == 1

def test_neurons_only_for_training_circuits():
    m = manifest()
    assert set(m["neurons"]) == set(m["circuits"]["train"])

def test_too_few_units_raises():
    with pytest.raises(ValueError):
        make_split_manifest(CIRCUITS, NEURONS, NUDGES, ["s0", "s1"], seed=0)

def test_write_read_round_trip(tmp_path):
    m = manifest()
    path = tmp_path / "manifest.json"
    write_manifest(m, path)
    assert read_manifest(path) == m
    assert path.read_text().endswith("\n")
    assert path.read_text() == json.dumps(m, sort_keys=True, indent=2) + "\n"
