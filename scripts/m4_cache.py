import json
import pickle
import sys
from pathlib import Path
from nexus.models.data import load_animal, poke_responses
from nexus.models.dusk import animal_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "scoring_priors.json").read_text())
LIGHT_KEYS = ("ids", "roles", "diagram", "pokes", "switch_type", "is_decoy", "datasheets", "circuit_id", "animal_id")

def light_animal(animal) -> dict:
    out = {k: animal[k] for k in LIGHT_KEYS}
    out["rows"] = animal_rows(animal, CONFIG)
    out["responses"] = poke_responses(animal)
    return out

def accepted_paths(src) -> list:
    rows = json.loads((src / "results.json").read_text())
    return [src / f"{r['motif']}_{int(r['seed']):05d}_a0.nwb" for r in rows if r["accepted"]]

def main(src, out) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for path in accepted_paths(src):
        target = out / (path.stem + ".pkl")
        if target.exists(): continue
        light = light_animal(load_animal(path))
        tmp = out / (path.stem + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(light, f)
        tmp.rename(target)
        print(path.stem, flush=True)

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
