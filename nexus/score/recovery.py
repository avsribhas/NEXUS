import math
from itertools import combinations

CONTINUOUS_FIELDS = ("trigger_gain", "threshold", "tau_on_over_tau_ref", "tau_off_over_tau_ref", "reach")

def field_recovered(interval, truth, prior_range, max_width=0.5) -> bool:
    """True when the interval contains the planted value and is no wider than max_width of the prior range."""
    low, high = float(interval[0]), float(interval[1])
    return low <= truth <= high and (high - low) <= max_width * (prior_range[1] - prior_range[0])

def order_agreement(pred_priority, true_priority) -> float:
    """Share of gate pairs ranked in the same order as the planted priorities; 1.0 with fewer than two gates."""
    keys = sorted(true_priority)
    pairs = list(combinations(keys, 2))
    if len(pairs) == 0:
        return 1.0
    same = 0
    for a, b in pairs:
        if (true_priority[a] < true_priority[b]) == (pred_priority[a] < pred_priority[b]):
            same += 1
    return same / len(pairs)

def _match(pred, truth) -> dict:
    """Pair each planted gate with the first predicted datasheet that has the same trigger."""
    by_trigger = {}
    for s in pred["datasheets"]:
        by_trigger.setdefault(s["trigger_channel"], s)
    return {d["trigger_channel"]: by_trigger[d["trigger_channel"]] for d in truth if d["trigger_channel"] in by_trigger}

def circuit_recovered(pred, truth, priors, identifiable, max_width=0.5) -> bool:
    """True when a switch is called, every planted gate is found, every identifiable field is recovered and priorities are in the right order."""
    if not pred["switch"]:
        return False
    matched = _match(pred, truth)
    if len(matched) < len(truth):
        return False
    for d in truth:
        s = matched[d["trigger_channel"]]
        for f in CONTINUOUS_FIELDS:
            if f in identifiable and not field_recovered(s[f], d[f], priors[f], max_width):
                return False
    true_priority = {d["trigger_channel"]: d["priority"] for d in truth}
    pred_priority = {k: s["priority"] for k, s in matched.items()}
    return order_agreement(pred_priority, true_priority) >= 1.0

def no_switch_correct(pred) -> bool:
    """True when the model calls no switch."""
    return not pred["switch"]

def twins_match(pred_a, pred_b, truth, priors, identifiable, max_width=0.5) -> bool:
    """True when both twins are recovered and their intervals overlap on every identifiable field."""
    if not (circuit_recovered(pred_a, truth, priors, identifiable, max_width) and circuit_recovered(pred_b, truth, priors, identifiable, max_width)):
        return False
    ma = _match(pred_a, truth)
    mb = _match(pred_b, truth)
    for d in truth:
        for f in CONTINUOUS_FIELDS:
            if f in identifiable:
                a = ma[d["trigger_channel"]][f]
                b = mb[d["trigger_channel"]][f]
                if max(a[0], b[0]) > min(a[1], b[1]):
                    return False
    return True

def interval_score(interval, truth, alpha=0.1) -> float:
    """Interval score at level 1 - alpha: width plus a penalty for missing the planted value."""
    low, high = float(interval[0]), float(interval[1])
    score = high - low
    if truth < low:
        score += (2.0 / alpha) * (low - truth)
    if truth > high:
        score += (2.0 / alpha) * (truth - high)
    return score

def pass_rate(flags) -> float:
    """Share of True values; nan for an empty list."""
    flags = list(flags)
    if not flags:
        return math.nan
    return sum(bool(x) for x in flags) / len(flags)
