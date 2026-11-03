import json
from pathlib import Path
import numpy as np
import pytest
from nexus.score.recovery import CONTINUOUS_FIELDS
from nexus.score.report import prior_prediction, score_all, state_dependent_part

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = json.loads((ROOT / "prereg" / "thresholds.json").read_text())
PRIORS = {"trigger_gain": [0.0, 100.0], "threshold": [0.0, 30.0], "tau_on_over_tau_ref": [0.0, 10.0], "tau_off_over_tau_ref": [0.0, 10.0], "reach": [0.0, 1.0]}
IDENT = {"release": list(CONTINUOUS_FIELDS), "gain": list(CONTINUOUS_FIELDS)}

def _sheet(trigger, k):
    return {"trigger_channel": trigger, "trigger_gain": 5.0 + k, "threshold": 10.0 + 0.1 * k, "tau_on_over_tau_ref": 1.0 + 0.01 * k, "tau_off_over_tau_ref": 2.0, "reach": 0.4, "priority": 1}

def _truth():
    truth = []
    for k in range(10):
        truth.append({"circuit_id": f"r{k}", "switch_type": "release", "is_decoy": False, "decoy_kind": "", "twin_of": "", "datasheets": [_sheet("trig", k)]})
        truth.append({"circuit_id": f"g{k}", "switch_type": "gain", "is_decoy": False, "decoy_kind": "", "twin_of": f"r{k}" if k < 5 else "", "datasheets": [_sheet("trig", k)]})
        truth.append({"circuit_id": f"p{k}", "switch_type": "none", "is_decoy": False, "decoy_kind": "", "twin_of": "", "datasheets": []})
        truth.append({"circuit_id": f"d{k}", "switch_type": "none", "is_decoy": True, "decoy_kind": "shunting", "twin_of": "", "datasheets": []})
    return truth

def _perfect(truth):
    preds = {}
    for c in truth:
        sheets = []
        for d in c["datasheets"]:
            s = {"trigger_channel": d["trigger_channel"], "priority": d["priority"]}
            for f in CONTINUOUS_FIELDS:
                w = 0.05 * (PRIORS[f][1] - PRIORS[f][0])
                s[f] = [d[f] - w, d[f] + w]
            sheets.append(s)
        preds[c["circuit_id"]] = {"switch": bool(sheets), "datasheets": sheets}
    return preds

def _rows(model_r, b4_r, n=50, low=0):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        ceil = 0.05 if i < low else 0.9
        rows.append({"circuit_id": f"c{i}", "ceiling_samples": list(ceil + 0.001 * rng.normal(size=20)), "r": {"dusk": model_r + 0.02 * rng.normal(), "baseline1_pooled": 0.4 + 0.02 * rng.normal(), "baseline1_per_state": 0.45 + 0.02 * rng.normal(), "baseline4": b4_r + 0.02 * rng.normal()}})
    return rows

def test_state_dependent_part():
    x = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    y = state_dependent_part(x)
    assert np.allclose(y, [[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

def test_perfect_predictions_pass_everything():
    truth = _truth()
    res = score_all(truth, _perfect(truth), PRIORS, IDENT, _rows(0.7, 0.3), "dusk", THRESHOLDS, n_boot=1000)
    for st in ("release", "gain"):
        assert res["recovery"][st]["rate"] == 1.0 and res["recovery"][st]["passes"] and res["recovery"][st]["n"] == 10
        assert res["no_false_switch"]["rate"] == 1.0 and res["no_false_switch"]["passes"]
        assert res["no_false_switch"]["decoys"] == {"shunting": 1.0}
        assert res["twins"]["n_pairs"] == 5 and res["twins"]["passes"]
        for f in CONTINUOUS_FIELDS:
            assert res["calibration"][f]["coverage"] == 1.0
            assert res["prediction"]["passes"] and not res["prediction"]["tie_with_baseline4"]

def test_null_row_fails():
    truth = _truth()
    res = score_all(truth, _perfect(truth), PRIORS, IDENT, _rows(0.7, 0.3), "dusk", THRESHOLDS, n_boot=1000)
    for st in ("release", "gain"):
        assert res["recovery"][st]["null_rate"] == 0.0 and not res["recovery"][st]["null_passes"]
    p = prior_prediction(truth[0], PRIORS)
    assert p["switch"] and len(p["datasheets"]) == 1

def test_inventing_switches_fails():
    truth = _truth()
    preds = _perfect(truth)
    for c in truth:
        if c["is_decoy"]:
            preds[c["circuit_id"]] = {"switch": True, "datasheets": []}
    res = score_all(truth, preds, PRIORS, IDENT, _rows(0.7, 0.3), "dusk", THRESHOLDS, n_boot=1000)
    assert res["no_false_switch"]["rate"] == 0.5 and not res["no_false_switch"]["passes"]
    assert res["no_false_switch"]["plain"] == 1.0 and res["no_false_switch"]["decoys"]["shunting"] == 0.0

def test_tie_overrides_a_win():
    truth = _truth()
    res = score_all(truth, _perfect(truth), PRIORS, IDENT, _rows(0.7, 0.7), "dusk", THRESHOLDS, n_boot=1000)
    assert res["prediction"]["comparisons"]["baseline1_pooled"]["passes"]
    assert res["prediction"]["tie_with_baseline4"]
    assert not res["prediction"]["passes"]

def test_low_ceiling_circuits_are_left_out():
    truth = _truth()
    res = score_all(truth, _perfect(truth), PRIORS, IDENT, _rows(0.7, 0.3, n=50, low=7), "dusk", THRESHOLDS, n_boot=1000)
    assert res["prediction"]["n_circuits"] == 50 and res["prediction"]["n_excluded_low_ceiling"] == 7
