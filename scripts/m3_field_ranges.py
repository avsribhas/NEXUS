import json
import math
import sys

def main():
    rows = []
    for p in sys.argv[1:]:
        rows.extend(json.load(open(p)))
    fields = ("trigger_gain", "threshold", "tau_on_over_tau_ref", "tau_off_over_tau_ref", "reach")
    for f in fields:
        values = [d[f] for r in rows if r.get("accepted") for d in r.get("datasheets", [])]
        values = [v for v in values if v is not None and math.isfinite(v)]
        if values:
            print(f"{f:22s} n={len(values):4d} min={min(values):.4g} max={max(values):.4g}" + (f" log10 min={math.log10(min(values)):.3f} log10 max={math.log10(max(values)):.3f}" if min(values) > 0 else ""))

if __name__ == "__main__":
    main()
