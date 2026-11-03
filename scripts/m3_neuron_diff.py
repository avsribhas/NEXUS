import sys
import tempfile
from pathlib import Path
import numpy as np
from nexus.circuits.build import build_circuit, drive_to_current, load_motifs
from nexus.circuits.graded import write_lems, write_network
from nexus.circuits.run import run_jaxley, run_neuron

def pulses(spec, roles):
    out = [{"target": roles["state"], "delay_ms": 0.0, "duration_ms": 600.0, "amplitude_nA": drive_to_current(spec, roles["state"], 20.0)}]
    for k, nid in enumerate(roles["inputs"]):
        out.append({"target": nid, "delay_ms": 50.0 + 100.0 * k, "duration_ms": 300.0, "amplitude_nA": drive_to_current(spec, nid, 25.0)})
    return out

def first_nan(t, v):
    bad = np.where(~np.isfinite(v))[0]
    return float(t[bad[0]]) if len(bad) else None

def main():
    ROOT = Path(__file__).resolve().parents[1]
    step = float(sys.argv[1])
    coarse = float(sys.argv[2])
    common, motifs = load_motifs(ROOT / "configs" / "motifs")
    for name in sys.argv[3:]:
        spec, roles = build_circuit(name, common, motifs, np.random.default_rng([0, 0]), f"{name}_check")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_network(spec, d / "c.net.nml", pulses=pulses(spec, roles))
            write_lems(spec, "c.net.nml", d / "LEMS_c.xml", 600.0, step)
            write_lems(spec, "c.net.nml", d / "LEMS_coarse.xml", 600.0, coarse)
            t_j, v_j, faithful = run_jaxley(d / "LEMS_c.xml")
            t_n, v_n = run_neuron(d / "LEMS_c.xml")
            t_c, v_c, _ = run_jaxley(d / "LEMS_coarse.xml")
        print(f"{name}: step={step} coarse={coarse} faithful={faithful} steps jaxley={len(t_j)} neuron={len(t_n)} coarse={len(t_c)}")
        for n in spec["neurons"]:
            nid = n["id"]
            diff = np.abs(v_j[nid] - v_n[nid])
            k = int(np.nanargmax(diff)) if np.any(np.isfinite(diff)) else 0
            fine_at_coarse = np.interp(t_c, t_j, v_j[nid])
            coarse_err = float(np.max(np.abs(v_c[nid] - fine_at_coarse)))
            print(f"  {nid:6s} max_diff={np.nanmax(diff) if np.any(np.isfinite(diff)) else float('nan'):.4f} at t={t_j[k]:.2f} v_jaxley={v_j[nid][k]:.3f} v_neuron={v_n[nid][k]:.3f} nan_jaxley_at={first_nan(t_j, v_j[nid])} nan_neuron_at={first_nan(t_n, v_n[nid])} range_jaxley=({np.nanmin(v_j[nid]):.2f}, {np.nanmax(v_j[nid]):.2f}) coarse_vs_fine={coarse_err:.4f}")

if __name__ == "__main__":
    main()
