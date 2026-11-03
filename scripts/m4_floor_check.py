import copy
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from nexus.score.report import score_prediction

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = json.loads((ROOT / "prereg" / "thresholds.json").read_text())
FLOOR = 0.1

def floored(rows, floor):
    """Drop circuits whose mean ceiling is below the floor, then floor each remaining ceiling sample."""
    out = [copy.deepcopy(r) for r in rows if float(np.mean(np.asarray(r["ceiling_samples"], dtype=float))) >= floor]
    for r in out:
        r["ceiling_samples"] = np.maximum(np.asarray(r["ceiling_samples"], dtype=float), floor)
    return out

def sample_stats(rows):
    """Returns statistics about the 'ceiling_samples' in 'rows'."""
    s = np.array([r["ceiling_samples"] for r in rows], dtype=float)
    return {
        "n_circuits": len(rows),
        "n_samples": int(s.size),
        "n_zero": int((s <= 0.0).sum()),
        "n_below_floor": int((s < FLOOR).sum()),
        "circuits_with_zero": int((s <= 0.0).any(axis=1).sum())
    }

def main(src: Path) -> None:
    """Re-scores the prediction test with each noise-ceiling sample floored at 0.1."""
    with open(src / "val_predictions.pkl", "rb") as f:
        data = pickle.load(f)
    rows = data["rows"]
    out = {
        "floor": FLOOR,
        "samples": sample_stats(rows),
        "locked": score_prediction(rows, "dusk", THRESHOLDS),
        "floored": score_prediction(floored(rows, FLOOR), "dusk", THRESHOLDS)
    }
    (src / "floor_check.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))

if __name__ == "__main__":
    main(Path(sys.argv[1]))
