import copy
import json
from pathlib import Path

def load_motifs(folder) -> tuple:
    """Load common.json and every other motif table in a folder; return (common, motifs)."""
    folder = Path(folder)
    common = json.loads((folder / "common.json").read_text())
    motifs = {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json")) if p.stem != "common"}
    return common, motifs

def _neuron(nid, overrides, common, rng) -> dict:
    out = {"id": nid}
    for key in ("C_nF", "g_leak_nS", "e_leak_mV"):
        low, high = overrides.get(key, common[key])
        out[key] = float(rng.uniform(low, high))
    return out

def _synapse(pre, post, sign, w_range, post_neuron, common, rng, vth_range=None, delta_range=None) -> dict:
    w = rng.uniform(*w_range)
    erev = common["erev_exc_mV"] if sign == "exc" else common["erev_inh_mV"] if sign == "inh" else post_neuron["e_leak_mV"]
    vth = rng.uniform(*(vth_range if vth_range is not None else common["vth_mV"]))
    delta = rng.uniform(*(delta_range if delta_range is not None else common["delta_mV"]))
    k = rng.uniform(*common["k_per_ms"])
    return {
        "pre": pre,
        "post": post,
        "g_nS": float(w * post_neuron["g_leak_nS"]),
        "erev_mV": float(erev),
        "vth_mV": float(vth),
        "delta_mV": float(delta),
        "k_per_ms": float(k)
    }

def build_circuit(motif_name, common, motifs, rng, circuit_id) -> tuple:
    """Sample one circuit from a motif table, add the brain-state neuron S and downstream background neurons; return (spec, roles)."""
    m = motifs[motif_name]
    neurons = {name: _neuron(name, m["neurons"][name], common, rng) for name in m["neurons"]}
    neurons["S"] = _neuron("S", {}, common, rng)
    graded = [_synapse(s["pre"], s["post"], s["sign"], s["w"], neurons[s["post"]], common, rng, s.get("vth_mV"), s.get("delta_mV")) for s in m["synapses"]]
    st = m["state"]
    graded.append(_synapse("S", st["target"], st["sign"], st["w"], neurons[st["target"]], common, rng))
    n_bg = int(rng.integers(common["background_n"][0], common["background_n"][1] + 1))
    pool = [name for name in m["neurons"] if name not in m["inputs"]]
    background = []
    gap = []
    for j in range(n_bg):
        bid = f"b{j}"
        neurons[bid] = _neuron(bid, {}, common, rng)
        n_in = min(int(rng.integers(1, 4)), len(pool))
        pres = rng.choice(len(pool), size=n_in, replace=False)
        for p in sorted(int(x) for x in pres):
            sign = "exc" if rng.random() < common["background_exc_fraction"] else "inh"
            graded.append(_synapse(pool[p], bid, sign, common["background_w"], neurons[bid], common, rng))
        if background and rng.random() < common["background_gap_fraction"]:
            other = background[int(rng.integers(0, len(background)))]
            gap.append({"a": other, "b": bid, "g_nS": float(rng.uniform(*common["background_gap_w"]) * neurons[bid]["g_leak_nS"])})
        background.append(bid)
        pool.append(bid)
    spec = {
        "circuit_id": circuit_id,
        "neurons": list(neurons.values()),
        "graded": graded,
        "gap": gap
    }
    roles = {
        "motif": motif_name,
        "switch_type": m["switch_type"],
        "is_decoy": m["is_decoy"],
        "decoy_kind": m["decoy_kind"],
        "inputs": list(m["inputs"]),
        "outputs": list(m["outputs"]),
        "gates": copy.deepcopy(m["gates"]),
        "candidates": copy.deepcopy(m["candidates"]),
        "state": "S",
        "background": background
    }
    return spec, roles

def make_twin(spec, scale, circuit_id) -> dict:
    """Copy a circuit with every capacitance and conductance multiplied by scale; its voltages under the same mV drives are unchanged."""
    t = copy.deepcopy(spec)
    t["circuit_id"] = circuit_id
    for n in t["neurons"]:
        n["C_nF"] *= scale
        n["g_leak_nS"] *= scale
    for g in t["graded"]:
        g["g_nS"] *= scale
    for g in t["gap"]:
        g["g_nS"] *= scale
    return t

def drive_to_current(spec, neuron_id, drive_mV) -> float:
    """Current in nA that would raise this neuron, on its own, by drive_mV at steady state."""
    n = next(n for n in spec["neurons"] if n["id"] == neuron_id)
    return float(drive_mV) * n["g_leak_nS"] / 1000.0
