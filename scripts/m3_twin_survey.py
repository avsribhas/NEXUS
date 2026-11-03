import json
import sys
import time
from pathlib import Path
import numpy as np
from nexus.circuits.build import build_circuit, load_motifs
from nexus.circuits.probe import measure_circuit
from nexus.circuits.twins import find_twin

def main():
    ROOT = Path(__file__).resolve().parents[1]
    n = int(sys.argv[1])
    tries = int(sys.argv[2])
    tol = float(sys.argv[3])
    only = sys.argv[4:]
    common, motifs = load_motifs(ROOT / "configs" / "motifs")
    config = json.loads((ROOT / "configs" / "scoring_priors.json").read_text())
    work = ROOT / "outputs" / "m3_twins"
    work.mkdir(parents=True, exist_ok=True)
    names = [m for m in sorted(motifs) if motifs[m]["switch_type"] != "none" and (not only or m in only)]
    rows = []
    for name in names:
        for seed in range(n):
            t0 = time.time()
            spec, roles = build_circuit(name, common, motifs, np.random.default_rng([seed, 0]), f"{name}_{seed:05d}")
            m = measure_circuit(spec, roles, work / f"{name}_{seed:05d}")
            if not m["accepted"]:
                continue
            r = find_twin(spec, roles, m["datasheets"], config, np.random.default_rng([seed, 3000]), work / f"{name}_{seed:05d}_twin", f"{name}_{seed:05d}_t", tries=tries, tol=tol)
            rows.append({"motif": name, "seed": seed, "found": r["found"], "tries": r["tries"], "distance": r["distance"], "seconds": round(time.time() - t0, 2)})
            print(f"{name:18s} seed {seed:3d} found={r['found']!s:5s} tries={r['tries']} distance={r['distance']:.3f} {rows[-1]['seconds']}s", flush=True)
    for name in names:
        sel = [r for r in rows if r["motif"] == name]
        print(f"{name:18s} found {sum(r['found'] for r in sel)}/{len(sel)}")
    (work / "results.json").write_text(json.dumps(rows, indent=1, default=str))

if __name__ == "__main__":
    main()
