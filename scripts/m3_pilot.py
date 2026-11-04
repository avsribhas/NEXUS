import json
import sys
import time
from pathlib import Path
from nexus.circuits.build import load_motifs
from nexus.circuits.library import generate_animal

def main():
    ROOT = Path(__file__).resolve().parents[1]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    common, motifs = load_motifs(ROOT / "configs" / "motifs")
    rates = json.loads((ROOT / "configs" / "corruption.json").read_text())["rates"]
    out = ROOT / "outputs" / "m3_pilot"; work = out / "work"; out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    rows = json.loads(results_path.read_text()) if results_path.exists() else []
    done = set(r["seed"] for r in rows)
    names = sorted(motifs)
    for i in range(n):
        seed = 10000 + i
        if seed in done:
            continue
        first_round = [m for m in names if m != "decoy_bistable"]
        name = first_round[i % len(first_round)] if i < 200 else names[i % len(names)]
        path = out / f"{name}_{seed:05d}_a0.nwb"
        t0 = time.time()
        r = generate_animal(name, common, motifs, rates, seed, work, path)
        r["motif"] = name
        r["seed"] = seed
        r["seconds"] = round(time.time() - t0, 2)
        r["bytes"] = path.stat().st_size if r["accepted"] else 0
        rows.append(r)
        results_path.write_text(json.dumps(rows, indent=1, default=str))
        print(f"{i:4d} {name:18s} accepted={r['accepted']!s:5s} tries={r['tries']} {r['seconds']}s {r['bytes'] / 1e6:.1f} MB", flush=True)
    for name in names:
        sel = [r for r in rows if r["motif"] == name]
        acc = [r for r in sel if r["accepted"]]
        print(f"{name:18s} accepted {len(acc)}/{len(sel)}  mean tries {sum(r['tries'] for r in sel) / max(len(sel), 1):.2f}  mean seconds {sum(r['seconds'] for r in sel) / max(len(sel), 1):.1f}  mean MB {sum(r['bytes'] for r in acc) / max(len(acc), 1) / 1e6:.1f}")
    print(f"total {len(rows)} circuits, {sum(r['seconds'] for r in rows) / 3600:.2f} h, {sum(r['bytes'] for r in rows) / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
