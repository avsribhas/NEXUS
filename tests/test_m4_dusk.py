import json
import math
from pathlib import Path
import numpy as np
import pytest
from nexus.circuits.build import load_motifs
from nexus.circuits.library import generate_animal
from nexus.models.data import load_animal
from nexus.models.dusk import DuskRecovery, from_scale, to_scale
from nexus.score.recovery import CONTINUOUS_FIELDS

ROOT = Path(__file__).resolve().parents[1]
COMMON, MOTIFS = load_motifs(ROOT / "configs" / "motifs")
CONFIG = json.loads((ROOT / "configs" / "scoring_priors.json").read_text())
RATES = {"missing_edge": 0.0, "false_edge": 0.0, "sign_flip": 0.0, "weight_sd_log": 0.0}

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    work = tmp_path_factory.mktemp("dusk")
    animals = {}
    for name in ("release", "decoy_shunt"):
        r = generate_animal(name, COMMON, MOTIFS, RATES, 0, work / name, work / f"{name}.nwb")
        assert r["accepted"]
        animals[name] = load_animal(work / f"{name}.nwb")
    model = DuskRecovery(CONFIG, epochs=600).fit(list(animals.values()))
    return model, animals

def test_scales():
    assert to_scale(100.0, "log10") == 2.0
    assert math.isnan(to_scale(0.0, "log10"))
    assert to_scale(3.0, "linear") == 3.0
    assert from_scale(2.0, "log10") == 100.0
    assert from_scale(3.0, "linear") == 3.0

def test_learns_its_training_animals(trained):
    model, animals = trained
    p = model.predict(animals["release"])
    assert p["switch"]
    assert [d["trigger_channel"] for d in p["datasheets"]] == ["trig"]
    assert model.predict(animals["decoy_shunt"])["switch"] is False

def test_output_format(trained):
    model, animals = trained
    p = model.predict(animals["release"], threshold=0.0)
    for d in p["datasheets"]:
        assert 0.0 <= d["probability"] <= 1.0
        for f in CONTINUOUS_FIELDS:
            assert len(d[f]) == 2 and d[f][0] <= d[f][1]
    assert sorted(d["priority"] for d in p["datasheets"]) == list(range(1, len(p["datasheets"]) + 1))

def test_predict_rows_shapes(trained):
    model, animals = trained
    triggers, prob, quant = model.predict_rows(animals["release"])
    assert triggers == animals["release"]["roles"]["inputs"]
    assert prob.shape == (len(triggers),)
    assert quant.shape == (len(triggers), len(CONTINUOUS_FIELDS), 2)
    assert np.all(quant[:, :, 0] <= quant[:, :, 1])
