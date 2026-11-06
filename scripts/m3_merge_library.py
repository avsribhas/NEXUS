import json
import sys
from pathlib import Path

def main(motif, old, new, out) -> None:
    out.mkdir(parents=True, exist_ok=True)
    old_rows = json.loads((old / "results.json").read_text())
    new_rows = json.loads((new / "results.json").read_text())
    old_seeds = sorted(r["seed"] for r in old_rows if r["motif"] == motif)
    new_seeds = sorted(r["seed"] for r in new_rows if r["motif"] == motif)
    if old_seeds != new_seeds: raise SystemExit(f"seed mismatch for {motif}")
    rows = [r for r in old_rows if r["motif"] != motif] + new_rows
    for r in rows:
        if r["accepted"]:
            name = f"{r['motif']}_{int(r['seed']):05d}_a0.nwb"
            source = (new if r["motif"] == motif else old) / name
            link = out / name
            if not link.exists(): link.symlink_to(source.resolve())
    (out / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    print(len(rows), "rows,", sum(1 for r in rows if r["motif"] == motif), motif)

if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
