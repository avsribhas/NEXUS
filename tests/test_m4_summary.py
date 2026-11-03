import json
import math
from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.build import load_motifs
from nexus.circuits.library import generate_animal
from nexus.models.data import load_animal
from nexus.models.summary import FEATURES, N_FEATURES, candidate_features, mean_traces, tau_cell

ROOT = Path(__file__).resolve().parents[1]
COMMON, MOTIFS = load_motifs(ROOT / "configs" / "motifs")
RATES = {"missing_edge": 0.0, "false_edge": 0.0, "sign_flip": 0.0, "weight_sd_log": 0.0}
GAIN = FEATURES.index("log_gain_30")

@pytest.fixture(scope="module")
def animals(tmp_path_factory):
    work = tmp_path_factory.mktemp("summary")
    out = {}
    for name in ("release", "decoy_shunt"):
        r = generate_animal(name, COMMON, MOTIFS, RATES, 0, work / name, work / f"{name}.nwb")
        assert r["accepted"]
        out[name] = load_animal(work / f"{name}.nwb")
    return out

def test_shapes_and_flags(animals):
    for a in animals.values():
        for trig in a["roles"]["inputs"]:
            x = candidate_features(a, trig)
            assert x.shape == (N_FEATURES,)
            assert np.all(np.isfinite(x))
            assert set(np.unique(x[len(FEATURES):])) <= {0.0, 1.0}

def test_release_trigger_shows_gain(animals):
    a = animals["release"]
    x = candidate_features(a, "trig")
    assert x[len(FEATURES) + GAIN] == 0.0
    assert x[GAIN] > math.log10(2.0)

def test_shunt_trigger_shows_no_gain(animals):
    a = animals["decoy_shunt"]
    x = candidate_features(a, "trig")
    assert x[len(FEATURES) + GAIN] == 0.0
    assert abs(x[GAIN]) < math.log10(1.6)

def test_tau_cell_is_positive(animals):
    for a in animals.values():
        t = tau_cell(a, mean_traces(a))
        assert math.isfinite(t) and t > 0
