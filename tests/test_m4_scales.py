import copy
import json
import math
from pathlib import Path
from nexus.score.report import apply_scales

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "scoring_priors.json").read_text())
TRUTH = [{"circuit_id": "c0", "datasheets": [{"trigger_channel": "t", "trigger_gain": 100.0, "threshold": 12.0, "tau_on_over_tau_ref": 10.0, "tau_off_over_tau_ref": 1.0, "reach": 0.5, "priority": 1}]}]
PREDS = {"c0": {"switch": True, "datasheets": [{"trigger_channel": "t", "priority": 1, "trigger_gain": [10.0, 1000.0], "threshold": [11.0, 13.0], "tau_on_over_tau_ref": [1.0, 100.0], "tau_off_over_tau_ref": [0.0, 10.0], "reach": [0.4, 0.6]}]}}

def test_log_and_linear_fields():
    t, p, priors = apply_scales(TRUTH, PREDS, CONFIG)
    d = t[0]["datasheets"][0]
    s = p["c0"]["datasheets"][0]
    assert abs(d["trigger_gain"] - 2.0) < 1e-12
    assert abs(d["tau_on_over_tau_ref"] - 1.0) < 1e-12
    assert d["threshold"] == 12.0 and d["reach"] == 0.5
    assert s["trigger_gain"] == [1.0, 3.0]
    assert s["threshold"] == [11.0, 13.0]
    assert s["tau_off_over_tau_ref"][0] == -math.inf and s["tau_off_over_tau_ref"][1] == 1.0

def test_priors_come_from_the_config():
    _, _, priors = apply_scales(TRUTH, PREDS, CONFIG)
    assert priors == {f: spec["range"] for f, spec in CONFIG["fields"].items()}

def test_inputs_are_not_changed():
    t0 = copy.deepcopy(TRUTH)
    p0 = copy.deepcopy(PREDS)
    apply_scales(TRUTH, PREDS, CONFIG)
    assert TRUTH == t0 and PREDS == p0
