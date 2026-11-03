import json
from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.build import build_circuit, load_motifs, make_twin
from nexus.circuits.corrupt import corrupt_wiring, true_edges
from nexus.format.ground_truth import CORRUPTION_KINDS

ROOT = Path(__file__).resolve().parents[1]
COMMON, MOTIFS = load_motifs(ROOT / "configs" / "motifs")
SPEC, ROLES = build_circuit("release", COMMON, MOTIFS, np.random.default_rng(0), "corrupt")
ZERO = {"missing_edge": 0.0, "false_edge": 0.0, "sign_flip": 0.0, "weight_sd_log": 0.0}

def test_true_edges_cover_the_circuit():
    edges = true_edges(SPEC)
    assert len(edges) == len(SPEC["graded"]) + len(SPEC["gap"])
    assert all(e["weight"] > 0 for e in edges)
    assert any(e["sign"] == "inh" for e in edges) and any(e["sign"] == "exc" for e in edges)

def test_zero_rates_change_nothing():
    stored, rows = corrupt_wiring(SPEC, ZERO, np.random.default_rng(1))
    assert stored == true_edges(SPEC)
    assert rows == []

def test_all_missing():
    rates = dict(ZERO, missing_edge=1.0)
    stored, rows = corrupt_wiring(SPEC, rates, np.random.default_rng(1))
    assert stored == []
    assert len(rows) == len(true_edges(SPEC))
    for r in rows:
        assert r["kind"] == "missing_edge" and r["stored_weight"] == 0.0

def test_all_signs_flipped():
    rates = dict(ZERO, sign_flip=1.0)
    stored, rows = corrupt_wiring(SPEC, rates, np.random.default_rng(1))
    true_pairs = set((e["pre"], e["post"]) for e in true_edges(SPEC))
    for e, s in zip(true_edges(SPEC), stored):
        if e["sign"] == "exc":
            assert s["sign"] == "inh"
        elif e["sign"] == "inh":
            assert s["sign"] == "exc"
        else:
            assert s["sign"] == e["sign"]

def test_false_edges_are_new():
    rates = dict(ZERO, false_edge=1.0)
    stored, rows = corrupt_wiring(SPEC, rates, np.random.default_rng(1))
    true_pairs = set((e["pre"], e["post"]) for e in true_edges(SPEC))
    false = [r for r in rows if r["kind"] == "false_edge"]
    assert len(false) > 0
    for r in false:
        assert (r["pre_neuron"], r["post_neuron"]) not in true_pairs and r["pre_neuron"] != r["post_neuron"] and r["true_weight"] == 0.0

def test_rows_use_known_kinds_and_repeat():
    rates = json.loads((ROOT / "configs" / "corruption.json").read_text())["rates"]
    a = corrupt_wiring(SPEC, rates, np.random.default_rng(5))
    b = corrupt_wiring(SPEC, rates, np.random.default_rng(5))
    assert a == b
    for r in a[1]:
        assert r["kind"] in CORRUPTION_KINDS

def test_twin_gets_the_same_diagram():
    rates = json.loads((ROOT / "configs" / "corruption.json").read_text())["rates"]
    twin = make_twin(SPEC, 3.0, "twin")
    a, _ = corrupt_wiring(SPEC, rates, np.random.default_rng(5))
    b, _ = corrupt_wiring(twin, rates, np.random.default_rng(5))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x["pre"], x["post"], x["type"], x["sign"]) == (y["pre"], y["post"], y["type"], y["sign"]) and abs(x["weight"] - y["weight"]) < 1e-9 * max(1.0, x["weight"])
