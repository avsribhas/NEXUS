import json
import sys
import time
from pathlib import Path
from nexus.circuits.build import load_motifs
from nexus.circuits.library import generate_animal

def main(motif, src, out) -> None:
    ROOT = Path(__file__).resolve().parents[1]
    common, motifs = load_motifs(ROOT / "configs" / "motifs")
    rates = json.loads((ROOT / "configs" / "corruption.json").read_text())["rates"]
    work = out / "work"; out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    rows = json.loads(results_path.read_text()) if results_path.exists() else []
    done = set(r["seed"] for r in rows)
    seeds = sorted(r["seed"] for r in json.loads((src / "results.json").read_text()) if r["motif"] == motif)
    for seed in seeds:
        if seed in done:
            continue
        path = out / f"{motif}_{seed:05d}_a0.nwb"
        t0 = time.time()
        r = generate_animal(motif, common, motifs, rates, seed, work, path)
        r["motif"] = motif
        r["seed"] = seed
        r["seconds"] = round(time.time() - t0, 2)
        r["bytes"] = path.stat().st_size if r["accepted"] else 0
        rows.append(r)
        results_path.write_text(json.dumps(rows, indent=1, default=str))
        print(f"{motif} {seed} accepted={r['accepted']} tries={r['tries']} {r['seconds']}s {r['bytes'] / 1e6:.1f} MB", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
