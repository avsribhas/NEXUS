import copy
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from nexus.baselines.linear import LinearBaseline
from nexus.format.splits import make_split_manifest, write_manifest
from nexus.models.dusk import DuskRecovery
from nexus.models.rows import prediction_row
from nexus.score.recovery import CONTINUOUS_FIELDS
from nexus.score.report import apply_scales, score_calibration, score_no_false_switch, score_recovery

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "scoring_priors.json").read_text())
THRESHOLDS = json.loads((ROOT / "prereg" / "thresholds.json").read_text())
BASELINES = {
    "baseline1_pooled": dict(kind="wiring"),
    "baseline1_per_state": dict(kind="wiring", per_state=True),
    "baseline4": dict(kind="wiring", state_term=True),
    "one_step": dict(kind="one_step", state_term=True),
    "blind": dict(kind="blind", state_term=True),
    "full": dict(kind="full", state_term=True),
    "shuffled": dict(kind="shuffled", state_term=True),
}

def load_cache(cache) -> list:
    result = []
    for path in sorted(cache.glob("*.pkl")):
        with open(path, "rb") as f:
            result.append(pickle.load(f))
    return result

def split(animals, out) -> dict:
    ids = sorted(a["circuit_id"] for a in animals)
    neurons = {a["circuit_id"]: list(a["ids"]) for a in animals}
    types = sorted(set(p["type"] for a in animals for p in a["pokes"]))
    manifest = make_split_manifest(ids, neurons, types, [0, 1, 2, 3, 4, 5], seed=0)
    write_manifest(manifest, out / "split.json")
    return {side: [a for a in animals if a["circuit_id"] in manifest["circuits"][side]] for side in ("train", "val")}

def circuit(a) -> dict:
    return {"circuit_id": a["circuit_id"], "ids": a["ids"], "diagram": a["diagram"], "pokes": a["pokes"]}

def seed_mean(a) -> np.ndarray:
    return np.nanmean(a["responses"], axis=2)

def fit_baselines(train) -> dict:
    cs = [circuit(a) for a in train]
    Y = [seed_mean(a) for a in train]
    M = [np.isfinite(y) for y in Y]
    return {name: LinearBaseline(**kw).fit(cs, Y, M) for name, kw in BASELINES.items()}

def truth_row(a) -> dict:
    kind = a["circuit_id"].rsplit("_", 1)[0].replace("decoy_", "") if a["is_decoy"] else ""
    return {"circuit_id": a["animal_id"], "switch_type": a["switch_type"], "is_decoy": bool(a["is_decoy"]), "decoy_kind": kind, "datasheets": copy.deepcopy(a["datasheets"])}

def prediction_summary(rows) -> dict:
    floor = THRESHOLDS["prediction"]["noise_ceiling_floor"]
    kept = [r for r in rows if r["ceiling"] >= floor]
    names = sorted(rows[0]["r"])
    return {"n_circuits": len(rows), "n_excluded_low_ceiling": len(rows) - len(kept), "median_ceiling": float(np.median([r["ceiling"] for r in rows])), "normalized": {n: float(np.mean([r["r"][n] / r["ceiling"] for r in kept])) for n in names}}

def switch_calls(truth, preds) -> dict:
    out = {}
    for st in sorted(set(t["switch_type"] for t in truth)):
        sel = [t for t in truth if t["switch_type"] == st]
        said = [preds[t["circuit_id"]]["switch"] for t in sel]
        right = [sorted(d["trigger_channel"] for d in preds[t["circuit_id"]]["datasheets"]) == sorted(d["trigger_channel"] for d in t["datasheets"]) for t in sel]
        out[st] = {"n": len(sel), "called_switch": float(np.mean(said)), "right_triggers": float(np.mean(right))}
    return out

def main(cache, out, epochs) -> None:
    out.mkdir(parents=True, exist_ok=True)
    parts = split(load_cache(cache), out)
    train = parts["train"]
    val = parts["val"]
    baselines = fit_baselines(train)
    dusk = DuskRecovery(CONFIG, epochs=epochs).fit(train)
    rows = [prediction_row(a, {name: b.predict(circuit(a)) for name, b in baselines.items()}) for a in val]
    preds = {a["animal_id"]: dusk.predict(a) for a in val}
    truth = [truth_row(a) for a in val]
    calls = switch_calls(truth, preds)
    identifiable = {t["switch_type"]: list(CONTINUOUS_FIELDS) for t in truth}
    truth_s, preds_s, priors = apply_scales(copy.deepcopy(truth), copy.deepcopy(preds), CONFIG)
    report = {"n_train": len(train), "n_val": len(val), "epochs": epochs, "identifiable_fields": "all, provisional until the identifiability step", "switch_calls": calls, "recovery": score_recovery(truth_s, preds_s, priors, identifiable, THRESHOLDS), "no_false_switch": score_no_false_switch(truth_s, preds_s, THRESHOLDS), "calibration": score_calibration(truth_s, preds_s), "prediction": prediction_summary(rows)}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    with open(out / "val_predictions.pkl", "wb") as f:
        pickle.dump({"rows": rows, "preds": preds}, f)
    print(json.dumps(report, indent=1, default=float))

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 400)
