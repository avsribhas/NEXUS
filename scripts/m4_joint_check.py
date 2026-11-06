import copy
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from m4_pilot_eval import CONFIG, truth_row
from nexus.models.dusk import DuskRecovery, animal_rows
from nexus.score.recovery import CONTINUOUS_FIELDS, _match, circuit_recovered
from nexus.score.report import apply_scales

def load_side(cache, ids) -> list:
    """Load the cached animals for these circuit ids only."""
    out = []
    for circuit_id in sorted(ids):
        with open(f"{cache}/{circuit_id}_a0.pkl", "rb") as f:
            out.append(pickle.load(f))
    return out

def identifiable_lists(found) -> dict:
    """Return {st: v["fields"] if v["decided"] else [f for f in CONTINUOUS_FIELDS if f not in v["undefined"]] for st, v in found.items()}"""
    return {st: v["fields"] if v["decided"] else [f for f in CONTINUOUS_FIELDS if f not in v["undefined"]] for st, v in found.items()}

def raw_scores(model, animals, identifiable) -> list:
    """Interval miss scores for each switch circuit, with no widening: a list of (n_gates, [(field_index, score), ...])."""
    model.widen = np.zeros(len(CONTINUOUS_FIELDS))
    out = []
    for a in animals:
        if a["switch_type"] not in identifiable: continue
        _, labels, values, masks, _ = animal_rows(a, CONFIG)
        _, _, quant = model.predict_rows(a)
        fields = identifiable[a["switch_type"]]
        scores = []
        for i in range(len(labels)):
            if labels[i] > 0:
                for j, f in enumerate(CONTINUOUS_FIELDS):
                    if masks[i, j] > 0 and f in fields:
                        scores.append((j, float(max(quant[i, j, 0] - values[i, j], values[i, j] - quant[i, j, 1]))))
        if scores: out.append((int(labels.sum()), scores))
    return out

def joint_margins(per_circuit, base) -> dict:
    """Per-field widening for each gate count, so all of a circuit's intervals hold together in 90% of calibration circuits."""
    base = np.asarray(base, dtype=float)
    b = np.ones(len(base))
    for j in range(len(base)):
        values = [score for _, scores in per_circuit for k, score in scores if k == j]
        if values: b[j] = float(np.std(values)) + 1e-6
    result = {}
    for g in sorted(set(c[0] for c in per_circuit)):
        t = np.array([max((score - base[j]) / b[j] for j, score in scores) for n, scores in per_circuit if n == g])
        level = min(1.0, 0.9 * (1.0 + 1.0 / len(t)))
        q = max(0.0, float(np.quantile(t, level)))
        result[g] = base + q * b
    return result

def predict_all(model, animals, base, margins) -> dict:
    """Predict with the per-field widening, or, when margins is given, with the widening for the number of gates called."""
    preds = {}
    for a in animals:
        model.widen = base
        p = model.predict(a)
        if margins is not None and p["datasheets"]:
            g = len(p["datasheets"])
            model.widen = margins[g] if g in margins else margins[max(margins)]
            p = model.predict(a)
        preds[a["animal_id"]] = p
    return preds

def score_rule(truth, preds, identifiable) -> dict:
    """Per type: circuits recovered, gates all found, intervals all containing the truth, intervals over the width limit, and median width as a share of the prior."""
    truth_s, preds_s, priors = apply_scales(copy.deepcopy(truth), copy.deepcopy(preds), CONFIG)
    out = {}
    for st in identifiable:
        sel = [t for t in truth_s if t["switch_type"] == st]
        recovered = 0; found = 0; contained = 0; too_wide = 0; widths = {f: [] for f in identifiable[st]}
        for t in sel:
            p = preds_s[t["circuit_id"]]
            recovered += int(circuit_recovered(p, t["datasheets"], priors, identifiable[st]))
            m = _match(p, t["datasheets"])
            if not p["switch"] or len(m) < len(t["datasheets"]):
                continue
            found += 1
            holds = True; wide = False
            for d in t["datasheets"]:
                s = m[d["trigger_channel"]]
                for f in identifiable[st]:
                    lo, hi = float(s[f][0]), float(s[f][1])
                    share = (hi - lo) / (priors[f][1] - priors[f][0])
                    widths[f].append(share)
                    if not lo <= d[f] <= hi: holds = False
                    if share > 0.5: wide = True
            contained += int(holds)
            too_wide += int(wide)
        out[st] = {"n": len(sel), "recovered": recovered, "all_gates_found": found, "all_intervals_contain": contained, "any_too_wide": too_wide, "median_width": {f: float(np.median(v)) if v else float("nan") for f, v in widths.items()}}
    return out

def run_seed(seed, train, val, identifiable, truth) -> dict:
    """Train the recovery model once and score the validation circuits under both rules."""
    model = DuskRecovery(CONFIG, epochs=400, calibrate=0.2, seed=seed).fit(train)
    base = model.widen.copy()
    order = np.random.default_rng(seed).permutation(len(train))
    n_cal = max(1, int(round(0.2 * len(train))))
    cal = [train[int(i)] for i in order[:n_cal]]
    margins = joint_margins(raw_scores(model, cal, identifiable), base)
    return {"seed": seed, "per_field_widen": base.tolist(), "joint_widen": {str(g): w.tolist() for g, w in margins.items()}, "per_field": score_rule(truth, predict_all(model, val, base, None), identifiable), "joint": score_rule(truth, predict_all(model, val, base, margins), identifiable)}

def main(cache, split_path, found_path, out_path, seeds) -> None:
    circuits = json.loads(split_path.read_text())["circuits"]
    train = load_side(cache, circuits["train"])
    val = load_side(cache, circuits["val"])
    identifiable = identifiable_lists(json.loads(found_path.read_text()))
    truth = [truth_row(a) for a in val]
    results = []
    for seed in seeds:
        results.append(run_seed(seed, train, val, identifiable, truth))
        out_path.write_text(json.dumps(results, indent=1, default=float))
        print("seed", seed, "done", flush=True)

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), [int(x) for x in sys.argv[5].split(",")])
