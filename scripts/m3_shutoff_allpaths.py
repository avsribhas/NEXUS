import copy
import json
import math
import pickle
import sys
from pathlib import Path
import numpy as np
from nexus.circuits.build import build_circuit, load_motifs
from nexus.circuits.probe import measure_circuit
from nexus.models.dusk import to_scale
from nexus.score.recovery import CONTINUOUS_FIELDS

MOTIF = "global_shutoff"
PATHS = (("in1", "O1"), ("in2", "O2"))
GEOMETRIC = ("trigger_gain", "tau_on_over_tau_ref", "tau_off_over_tau_ref")

def mean_datasheet(sheets) -> dict:
    out = dict(sheets[0])
    for f in CONTINUOUS_FIELDS:
        values = [s[f] for s in sheets]
        if f in GEOMETRIC:
            if all(x > 0 for x in values):
                out[f] = math.exp(sum(math.log(x) for x in values) / len(values))
            else:
                out[f] = math.nan
        else:
            out[f] = sum(values) / len(values)
    return out

def measure_paths(spec, roles, workdir) -> list:
    results = []
    for test_input, output in PATHS:
        r = copy.deepcopy(roles)
        r["gates"] = [{"gate_id": "g0", "trigger": "trig", "test_input": test_input, "output": output}]
        result = measure_circuit(spec, r, Path(workdir) / test_input)
        results.append(result)
    return results

def replaced_rows(rows, sheet, config) -> tuple:
    X, L, Y, M, triggers = rows
    Y = Y.copy()
    M = M.copy()
    i = triggers.index(sheet["trigger_channel"])
    for j, f in enumerate(CONTINUOUS_FIELDS):
        v = to_scale(sheet[f], config["fields"][f]["scale"])
        Y[i, j] = 0.0 if math.isnan(v) else v
        M[i, j] = 0.0 if math.isnan(v) else 1.0
    return (X, L, Y, M, triggers)

def main(results_path, cache_in, split_in, cache_out, split_out, report_path) -> None:
    """
    Measure every path of each global shutoff circuit, replace its planted answer with the mean, and write a separate cache and split; resumes from an existing report.
    """
    ROOT = Path(__file__).resolve().parents[1]
    common, motifs = load_motifs(ROOT / "configs" / "motifs")
    config = json.loads((ROOT / "configs" / "scoring_priors.json").read_text())
    rows = json.loads(results_path.read_text())
    cache_out.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    dropped = [c for c, v in report.items() if not v["accepted"]]

    for r in rows:
        if r["motif"] == MOTIF and r["accepted"]:
            seed = int(r["seed"])
            circuit_id = f"{MOTIF}_{seed:05d}"
            if circuit_id in report: continue
            spec, roles = build_circuit(MOTIF, common, motifs, np.random.default_rng([seed, int(r["tries"]) - 1]), circuit_id)
            ms = measure_paths(spec, roles, cache_out / "work" / circuit_id)
            ok = all(m["accepted"] for m in ms)
            stored = r["datasheets"][0]
            repro = max(abs(ms[0]["datasheets"][0][f] - stored[f]) for f in CONTINUOUS_FIELDS) if ms[0]["datasheets"] else math.nan
            report[circuit_id] = {"accepted": ok, "reasons": [m["reason"] for m in ms], "repro": repro, "sheets": [m["datasheets"] for m in ms]}
            if not ok:
                dropped.append(circuit_id)
            else:
                sheet = mean_datasheet([m["datasheets"][0] for m in ms])
                with open(cache_in / f"{circuit_id}_a0.pkl", "rb") as f:
                    light = pickle.load(f)
                light["datasheets"] = [sheet]
                light["rows"] = replaced_rows(light["rows"], sheet, config)
                with open(cache_out / f"{circuit_id}_a0.pkl", "wb") as f:
                    pickle.dump(light, f)
            report_path.write_text(json.dumps(report, indent=1, default=float))
            print(circuit_id, ok, repro, flush=True)

    for p in sorted(cache_in.glob("*.pkl")):
        target = cache_out / p.name
        if not target.exists() and not p.name.startswith(MOTIF):
            target.symlink_to(p.resolve())

    split = json.loads(split_in.read_text())
    for side in split["circuits"]:
        split["circuits"][side] = [c for c in split["circuits"][side] if c not in dropped]
    split_out.write_text(json.dumps(split, indent=1))
    print(len(report), "measured,", len(dropped), "dropped", flush=True)

if __name__ == "__main__":
    main(*[Path(x) for x in sys.argv[1:7]])
