import copy
import math
import numpy as np
from nexus.score.prediction import eligible, is_tie, paired_bootstrap, prediction_passes
from nexus.score.recovery import CONTINUOUS_FIELDS, circuit_recovered, interval_score, no_switch_correct, pass_rate, twins_match

def state_dependent_part(x) -> np.ndarray:
    """Remove the part of a response that is the same in every brain state (the last axis is the state)."""
    x = np.asarray(x, dtype=float); return x - x.mean(axis=-1, keepdims=True)

def prior_prediction(circuit, priors) -> dict:
    """The null estimator: ignores the data and returns the central 90% of each prior range."""
    sheets = []
    for d in circuit["datasheets"]:
        s = {"trigger_channel": d["trigger_channel"], "priority": 1}
        for f in CONTINUOUS_FIELDS:
            lo, hi = priors[f]
            s[f] = [lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)]
        sheets.append(s)
    return {"switch": True, "datasheets": sheets}

def score_recovery(truth, preds, priors, identifiable, thresholds) -> dict:
    """Recovery per switch type, with the prior-only null row."""
    mw = thresholds["recovery"]["max_width_fraction_of_prior"]; need = thresholds["recovery"]["pass_fraction_of_switch_circuits"]
    out = {}
    for st in sorted(set(c["switch_type"] for c in truth if c["switch_type"] != "none")):
        cs = [c for c in truth if c["switch_type"] == st]
        fields = set(identifiable.get(st, []))
        flags = [circuit_recovered(preds[c["circuit_id"]], c["datasheets"], priors, fields, mw) for c in cs]
        null = [circuit_recovered(prior_prediction(c, priors), c["datasheets"], priors, fields, mw) for c in cs]
        rate = pass_rate(flags); null_rate = pass_rate(null)
        out[st] = {"n": len(cs), "rate": rate, "passes": bool(rate >= need), "null_rate": null_rate, "null_passes": bool(null_rate >= need)}
    return out

def score_no_false_switch(truth, preds, thresholds) -> dict:
    """Share of no-switch circuits called no switch: overall, plain circuits, and each decoy kind."""
    need = thresholds["no_false_switch"]["pass_fraction"]
    cs = [c for c in truth if c["switch_type"] == "none"]
    overall = pass_rate([no_switch_correct(preds[c["circuit_id"]]) for c in cs])
    plain = pass_rate([no_switch_correct(preds[c["circuit_id"]]) for c in cs if not c["is_decoy"]])
    kinds = {}
    for k in sorted(set(c["decoy_kind"] for c in cs if c["is_decoy"])):
        kinds[k] = pass_rate([no_switch_correct(preds[c["circuit_id"]]) for c in cs if c["is_decoy"] and c["decoy_kind"] == k])
    return {"n": len(cs), "rate": overall, "passes": bool(overall >= need), "plain": plain, "decoys": kinds}

def score_twins(truth, preds, priors, identifiable, thresholds) -> dict:
    """Share of twin pairs whose datasheets both recover and overlap."""
    mw = thresholds["recovery"]["max_width_fraction_of_prior"]; need = thresholds["degenerate_twins"]["pass_fraction"]
    flags = []
    for c in truth:
        if c.get("twin_of"):
            flags.append(twins_match(preds[c["circuit_id"]], preds[c["twin_of"]], c["datasheets"], priors, set(identifiable.get(c["switch_type"], [])), mw))
    rate = pass_rate(flags)
    return {"n_pairs": len(flags), "rate": rate, "passes": bool(rate >= need)}

def score_calibration(truth, preds) -> dict:
    """For each continuous field: share of predicted intervals that contain the planted value, and the mean interval score."""
    out = {}
    for f in CONTINUOUS_FIELDS:
        hits = []; scores = []
        for c in truth:
            p = preds[c["circuit_id"]]
            by_trigger = {s["trigger_channel"]: s for s in p["datasheets"]}
            for d in c["datasheets"]:
                if d["trigger_channel"] in by_trigger:
                    lo, hi = by_trigger[d["trigger_channel"]][f]
                    hits.append(lo <= d[f] <= hi); scores.append(interval_score([lo, hi], d[f]))
        out[f] = {"n": len(hits), "coverage": pass_rate(hits), "interval_score": float(np.mean(scores)) if scores else float("nan")}
    return out

def score_prediction(rows, model, thresholds, n_boot=10000) -> dict:
    """Prediction against baseline 1 (pooled and per state) with the noise-ceiling floor and the tie rule against baseline 4."""
    floor = thresholds["prediction"]["noise_ceiling_floor"]; ci = thresholds["prediction"]["bootstrap_ci"]; margin = thresholds["tie"]["margin_fraction_of_ceiling"]
    samples = np.array([r["ceiling_samples"] for r in rows], dtype=float)
    keep = eligible(samples.mean(axis=1), floor)
    kept = [r for r, k in zip(rows, keep) if k]; s = samples[keep]
    score = lambda name: np.array([r["r"][name] for r in kept], dtype=float)
    comparisons = {}
    for b in ("baseline1_pooled", "baseline1_per_state"):
        boot = paired_bootstrap(score(model), score(b), s, n_boot=n_boot, ci=ci)
        comparisons[b] = dict(boot, passes=prediction_passes(boot))
    norm = lambda name: float(np.mean(score(name) / s.mean(axis=1)))
    tie = is_tie(norm(model), norm("baseline4"), margin)
    passes = all(v["passes"] for v in comparisons.values()) and not tie
    return {"n_circuits": len(rows), "n_excluded_low_ceiling": int(len(rows) - int(keep.sum())), "normalized": {name: norm(name) for name in sorted(kept[0]["r"])}, "comparisons": comparisons, "tie_with_baseline4": bool(tie), "passes": bool(passes)}

def score_all(truth, preds, priors, identifiable, prediction_rows, model, thresholds, n_boot=10000) -> dict:
    """The full Phase 1 result table for one model."""
    return {"model": model, "recovery": score_recovery(truth, preds, priors, identifiable, thresholds), "no_false_switch": score_no_false_switch(truth, preds, thresholds), "twins": score_twins(truth, preds, priors, identifiable, thresholds), "calibration": score_calibration(truth, preds), "prediction": score_prediction(prediction_rows, model, thresholds, n_boot)}

def _to_scale(value, scale):
    """Move one value onto its scoring scale; non-positive values on a log scale become -inf."""
    if scale == "log10":
        return math.log10(value) if value > 0 else -math.inf
    return float(value)

def apply_scales(truth, preds, config):
    """Put planted values and predicted intervals on each field's scoring scale; return them with the prior ranges."""
    fields = config["fields"]
    truth = copy.deepcopy(truth)
    preds = copy.deepcopy(preds)
    for c in truth:
        for d in c["datasheets"]:
            for f, spec in fields.items():
                d[f] = _to_scale(d[f], spec["scale"])
    for p in preds.values():
        for s in p["datasheets"]:
            for f, spec in fields.items():
                s[f] = [_to_scale(s[f][0], spec["scale"]), _to_scale(s[f][1], spec["scale"])]
    priors = {f: list(spec["range"]) for f, spec in fields.items()}
    return truth, preds, priors

def score_from_config(truth, preds, identifiable, prediction_rows, model, thresholds, config, n_boot=10000) -> dict:
    """score_all after moving every field onto its scoring scale."""
    truth, preds, priors = apply_scales(truth, preds, config)
    return score_all(truth, preds, priors, identifiable, prediction_rows, model, thresholds, n_boot)
