from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.batch import BatchRunner
from nexus.circuits.build import build_circuit, drive_to_current, load_motifs
from nexus.circuits.graded import write_network
from nexus.perturb.protocol import (DT, KEEP, N_KEEP, N_SEEDS, N_STEPS, NOISE_MV, POKE_START, POKE_TYPES, STATES_MV, T_TRIAL, TARGETS_PER_TYPE, make_pokes, run_trials, trial_currents, trial_table, PAIRED, PAIRS_PER_ANIMAL)

COMMON, MOTIFS = load_motifs(Path(__file__).resolve().parents[1] / "configs" / "motifs")
SPEC, ROLES = build_circuit("release", COMMON, MOTIFS, np.random.default_rng(0), "proto")
IDS = [n["id"] for n in SPEC["neurons"]]

def test_constants_agree():
    assert N_STEPS == int(round(T_TRIAL / DT)) + 1
    assert N_KEEP == int(round((KEEP[1] - KEEP[0]) / DT)) // 5
    assert len(STATES_MV) == 6
    assert N_SEEDS == 3
    assert len(POKE_TYPES) * TARGETS_PER_TYPE == 50

def test_make_pokes():
    pokes = make_pokes(SPEC, ROLES, np.random.default_rng(1))
    assert len(pokes) == 50 + PAIRS_PER_ANIMAL
    assert all(p["target"] != "S" for p in pokes)
    assert all(p["target"] in IDS for p in pokes)
    assert len(set(p["poke_id"] for p in pokes)) == 50 + PAIRS_PER_ANIMAL
    for t in POKE_TYPES:
        assert sum(1 for p in pokes if p["type"] == t) == TARGETS_PER_TYPE
        assert all(p["amplitude_mV"] == POKE_TYPES[t][0] for p in pokes if p["type"] == t)
        assert all(p["start_ms"] == POKE_START and p["stop_ms"] == POKE_START + POKE_TYPES[t][1] for p in pokes if p["type"] == t)
    assert len(set(p["poke_id"] for p in pokes)) == 50 + PAIRS_PER_ANIMAL
    pokes_again = make_pokes(SPEC, ROLES, np.random.default_rng(1))
    assert pokes == pokes_again

def test_trial_table():
    pokes = make_pokes(SPEC, ROLES, np.random.default_rng(1))
    trials = trial_table(pokes)
    assert len(trials) == (50 + PAIRS_PER_ANIMAL) * 6 * 3
    assert len(set((t["poke_index"], t["state_index"], t["seed_index"]) for t in trials)) == len(trials)
    assert [t["trial_id"] for t in trials] == list(range(len(trials)))

def test_currents_without_noise():
    poke = make_pokes(SPEC, ROLES, np.random.default_rng(1))[0]
    c = trial_currents(SPEC, IDS, ROLES, poke, 4, 0, 7, noise=False)
    assert c.shape == (len(IDS), N_STEPS) and c.dtype == np.float32
    s = IDS.index("S")
    assert np.allclose(c[s], drive_to_current(SPEC, "S", STATES_MV[4]), atol=1e-7)
    k = IDS.index(poke["target"])
    i0 = int(round(poke["start_ms"] / DT))
    i1 = int(round(poke["stop_ms"] / DT))
    expected = np.zeros(N_STEPS)
    expected[i0:i1] = drive_to_current(SPEC, poke["target"], poke["amplitude_mV"])
    if k == s:
        expected = expected + drive_to_current(SPEC, "S", STATES_MV[4])
    assert np.allclose(c[k], expected, atol=1e-7)
    for i in range(len(IDS)):
        if i not in (s, k):
            assert np.all(c[i] == 0.0)

def test_noise_is_seeded():
    poke = make_pokes(SPEC, ROLES, np.random.default_rng(1))[0]
    a = trial_currents(SPEC, IDS, ROLES, poke, 2, 0, 7)
    b = trial_currents(SPEC, IDS, ROLES, poke, 2, 0, 7)
    c = trial_currents(SPEC, IDS, ROLES, poke, 2, 1, 7)
    d = trial_currents(SPEC, IDS, ROLES, poke, 2, 0, 8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)
    clean = trial_currents(SPEC, IDS, ROLES, poke, 2, 0, 7, noise=False)
    for i in range(len(IDS)):
        g = SPEC["neurons"][i]["g_leak_nS"]
        expected_sd = NOISE_MV * g / 1000.0
        assert abs(np.std(a[i] - clean[i]) - expected_sd) < 0.05 * expected_sd

def test_run_trials_shape_and_batching(tmp_path):
    path = tmp_path / "proto.net.nml"
    write_network(SPEC, path)
    runner = BatchRunner(path, IDS, DT, T_TRIAL)
    pokes = make_pokes(SPEC, ROLES, np.random.default_rng(1))
    trials = trial_table(pokes)[:3]
    v = run_trials(runner, SPEC, IDS, ROLES, pokes, trials, 7, batch_size=2)
    assert v.shape == (3, len(IDS), N_KEEP) and v.dtype == np.float32
    assert np.all(np.isfinite(v))
    w = run_trials(runner, SPEC, IDS, ROLES, pokes, trials[2:3], 7, batch_size=2)
    assert np.allclose(v[2], w[0], atol=1e-4)

def test_paired_pokes():
    pokes = make_pokes(SPEC, ROLES, np.random.default_rng(1))
    paired = [p for p in pokes if p["type"] == PAIRED["type"]]
    assert len(paired) == PAIRS_PER_ANIMAL
    for p in paired:
        assert p["hold"] in ROLES["inputs"]
        assert p["target"] in ROLES["inputs"]
        assert p["hold"] != p["target"]
    p = paired[0]
    c = trial_currents(SPEC, IDS, ROLES, p, 0, 0, 7, noise=False)
    h = IDS.index(p["hold"])
    assert np.allclose(c[h], drive_to_current(SPEC, p["hold"], PAIRED["hold_mV"]), atol=1e-7)
    k = IDS.index(p["target"])
    i0 = int(round(p["start_ms"] / DT))
    assert np.isclose(c[k, i0], drive_to_current(SPEC, p["target"], PAIRED["amplitude_mV"]), atol=1e-7)
    assert c[k, 0] == 0.0

def test_inputs_are_always_targets():
    for seed in range(5):
        pokes = make_pokes(SPEC, ROLES, np.random.default_rng(seed))
        for t in POKE_TYPES:
            targets = [p["target"] for p in pokes if p["type"] == t]
            for nid in ROLES["inputs"]:
                assert nid in targets
