import pytest
from nexus.format.splits import KINDS, held_out_kinds, make_split_manifest, trial_split

CIRCUITS = [f"c{i:02d}" for i in range(10)]
NEURONS = {c: ["n0", "n1", "n2", "n3", "n4"] for c in CIRCUITS}
NUDGES = ["graded_single", "competing_pair", "state_variant"]
STATES = ["s0", "s1", "s2", "s3"]
M = make_split_manifest(CIRCUITS, NEURONS, NUDGES, STATES, seed=0)

TC = M["circuits"]["test"][0]
VC = M["circuits"]["val"][0]
TRC = M["circuits"]["train"][0]
TN = M["neurons"][TRC]["test"][0]
TRN = M["neurons"][TRC]["train"][0]
TNU = M["nudge_types"]["test"][0]
VNU = M["nudge_types"]["val"][0]
TRNU = M["nudge_types"]["train"][0]
TS = M["states"]["test"][0]
TRS = M["states"]["train"][0]

def t(circuit, neuron, nudge, state, seed=0):
    return {"circuit_id": circuit, "neuron_id": neuron, "nudge_type": nudge, "state": state, "seed": seed}

def test_trial_split_rules():
    assert trial_split(M, t(TRC, TRN, TRNU, TRS)) == "train"
    assert trial_split(M, t(TC, "n0", TRNU, TRS)) == "test"
    assert trial_split(M, t(TRC, TRN, TRNU, TS)) == "test"
    assert trial_split(M, t(TRC, TRN, VNU, TRS)) == "val"
    assert trial_split(M, t(TRC, TN, TRNU, TRS)) == "test"
    assert trial_split(M, t(VC, "n0", TRNU, TS)) == "test"

def test_seed_does_not_change_split():
    for seed in [0, 1, 2, 99]:
        assert trial_split(M, t(TRC, TRN, TRNU, TRS, seed)) == "train"
        assert trial_split(M, t(TRC, TRN, TRNU, TS, seed)) == "test"

def test_held_out_kinds():
    assert held_out_kinds(M, t(TRC, TRN, TRNU, TRS)) == []
    assert held_out_kinds(M, t(TRC, TRN, TRNU, TS)) == ["states"]
    assert held_out_kinds(M, t(TC, "n0", TNU, TS)) == ["circuits", "nudge_types", "states"]
    assert held_out_kinds(M, t(TRC, TN, TRNU, TRS)) == ["neurons"]

def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        trial_split(M, t("c99", TRN, TRNU, TRS))
    with pytest.raises(ValueError):
        trial_split(M, t(TRC, TRN, TRNU, "s9"))
