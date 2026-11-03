import math
import pytest
from nexus.score.recovery import CONTINUOUS_FIELDS, circuit_recovered, field_recovered, interval_score, no_switch_correct, order_agreement, pass_rate, twins_match

PRIORS = {"trigger_gain": (0.0, 100.0), "threshold": (0.0, 30.0), "tau_on_over_tau_ref": (0.0, 10.0), "tau_off_over_tau_ref": (0.0, 10.0), "reach": (0.0, 1.0)}
ALL = set(CONTINUOUS_FIELDS)
TRUTH = [{"trigger_channel": "trigA", "trigger_gain": 10.0, "threshold": 12.0, "tau_on_over_tau_ref": 2.0, "tau_off_over_tau_ref": 3.0, "reach": 0.5, "priority": 1}, {"trigger_channel": "trigB", "trigger_gain": 0.2, "threshold": 8.0, "tau_on_over_tau_ref": 1.0, "tau_off_over_tau_ref": 4.0, "reach": 0.3, "priority": 2}]

def _good(truth, half=0.1):
    sheets = []
    for d in truth:
        s = {"trigger_channel": d["trigger_channel"], "priority": d["priority"]}
        for f in CONTINUOUS_FIELDS:
            w = half * (PRIORS[f][1] - PRIORS[f][0])
            s[f] = [d[f] - w, d[f] + w]
        sheets.append(s)
    return {"switch": True, "datasheets": sheets}

def test_field_recovered():
    assert field_recovered([1.0, 3.0], 2.0, (0.0, 10.0))
    assert not field_recovered([1.0, 3.0], 4.0, (0.0, 10.0))
    assert not field_recovered([0.0, 6.0], 2.0, (0.0, 10.0))
    assert field_recovered([0.0, 5.0], 5.0, (0.0, 10.0))

def test_order_agreement():
    assert order_agreement({"a": 1, "b": 2}, {"a": 1, "b": 2}) == 1.0
    assert order_agreement({"a": 2, "b": 1}, {"a": 1, "b": 2}) == 0.0
    assert order_agreement({"a": 1}, {"a": 1}) == 1.0
    assert abs(order_agreement({"a": 1, "b": 3, "c": 2}, {"a": 1, "b": 2, "c": 3}) - 2.0 / 3.0) < 1e-12

def test_good_prediction_recovers():
    assert circuit_recovered(_good(TRUTH), TRUTH, PRIORS, ALL)

def test_wide_interval_fails():
    pred = _good(TRUTH, half=0.3)
    assert not circuit_recovered(pred, TRUTH, PRIORS, ALL)
    assert circuit_recovered(pred, TRUTH, PRIORS, set())

def test_missing_gate_or_no_switch_fails():
    pred = _good(TRUTH)
    pred["datasheets"] = pred["datasheets"][:1]
    assert not circuit_recovered(pred, TRUTH, PRIORS, ALL)
    assert not circuit_recovered({"switch": False, "datasheets": []}, TRUTH, PRIORS, ALL)

def test_wrong_priority_fails():
    pred = _good(TRUTH)
    pred["datasheets"][0]["priority"] = 2
    pred["datasheets"][1]["priority"] = 1
    assert not circuit_recovered(pred, TRUTH, PRIORS, ALL)

def test_no_switch_correct():
    assert no_switch_correct({"switch": False, "datasheets": []})
    assert not no_switch_correct(_good(TRUTH))

def test_twins_match():
    a = _good(TRUTH)
    b = _good(TRUTH)
    assert twins_match(a, b, TRUTH, PRIORS, ALL)
    for f in CONTINUOUS_FIELDS:
        lo, hi = b["datasheets"][0][f]
        w = hi - lo
        b["datasheets"][0][f] = [lo + 0.6 * w, hi + 0.6 * w]
    assert not twins_match(a, b, TRUTH, PRIORS, ALL)

def test_interval_score():
    assert interval_score([1.0, 3.0], 2.0) == 2.0
    assert abs(interval_score([1.0, 3.0], 0.0) - (2.0 + 20.0)) < 1e-12
    assert abs(interval_score([1.0, 3.0], 4.0) - (2.0 + 20.0)) < 1e-12

def test_pass_rate():
    assert pass_rate([True, False, True, True]) == 0.75
    assert math.isnan(pass_rate([]))
