import json
import sys
import time
from pathlib import Path
import numpy as np
from nexus.circuits.build import build_circuit, load_motifs
from nexus.circuits.probe import measure_circuit

def main():
    ROOT = Path(__file__).resolve().parents[1]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    common, motifs = load_motifs(ROOT / "configs" / "motifs")
    work = ROOT / "outputs" / "m3_acceptance"; work.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in sorted(motifs):
        for seed in range(n):
            t0 = time.time()
            spec, roles = build_circuit(name, common, motifs, np.random.default_rng([seed, 0]), f"{name}_{seed:05d}")
            m = measure_circuit(spec, roles, work / f"{name}_{seed:05d}")
            rows.append({"motif": name, "seed": seed, "n_neurons": len(spec["neurons"]), "accepted": m["accepted"], "reason": m["reason"], "gate_ratios": m["gate_ratios"], "cross_ratios": m["cross_ratios"], "datasheets": m["datasheets"], "seconds": round(time.time() - t0, 2)})
            print(f"{name:18s} seed {seed:3d} accepted={m['accepted']!s:5s} ratios={m['gate_ratios'] or m['cross_ratios']} {rows[-1]['seconds']}s", flush=True)
    for name in sorted(motifs):
        a = sum(1 for row in rows if row["motif"] == name and row["accepted"])
        print(f"{name:18s} accepted {a}/{n}")
    with open(work / "results.json", "w") as f:
        f.write(json.dumps(rows, indent=1, default=str))

if __name__ == "__main__":
    main()
