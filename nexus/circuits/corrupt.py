import numpy as np

def true_edges(spec) -> list:
    """Generate the true edges of the circuit based on the given specification."""
    leak = {n["id"]: n for n in spec["neurons"]}
    edges = []
    for s in spec["graded"]:
        post = leak[s["post"]]
        sign = "exc" if s["erev_mV"] > post["e_leak_mV"] + 5.0 else "inh" if s["erev_mV"] < post["e_leak_mV"] - 5.0 else "shunt"
        edges.append({"pre": s["pre"], "post": s["post"], "type": "graded", "sign": sign, "weight": float(s["g_nS"] / post["g_leak_nS"])})
    for j in spec["gap"]:
        edges.append({"pre": j["a"], "post": j["b"], "type": "gap", "sign": "gap", "weight": float(j["g_nS"] / leak[j["b"]]["g_leak_nS"])})
    return edges

def corrupt_wiring(spec, rates, rng) -> tuple:
    """Corrupt the wiring diagram based on the given rates and random number generator."""
    edges = true_edges(spec)
    stored = []
    rows = []
    for e in edges:
        if rng.random() < rates["missing_edge"]:
            rows.append({"pre_neuron": e["pre"], "post_neuron": e["post"], "kind": "missing_edge", "true_weight": e["weight"], "stored_weight": 0.0})
            continue
        s = dict(e)
        if e["sign"] in ("exc", "inh") and rng.random() < rates["sign_flip"]:
            s["sign"] = "inh" if e["sign"] == "exc" else "exc"
            rows.append({"pre_neuron": e["pre"], "post_neuron": e["post"], "kind": "sign_flip", "true_weight": e["weight"], "stored_weight": e["weight"]})
        if rates["weight_sd_log"] > 0:
            s["weight"] = float(e["weight"] * np.exp(rng.normal(0.0, rates["weight_sd_log"])))
            rows.append({"pre_neuron": e["pre"], "post_neuron": e["post"], "kind": "weight_error", "true_weight": e["weight"], "stored_weight": s["weight"]})
        stored.append(s)
    ids = [n["id"] for n in spec["neurons"]]
    existing = set((e["pre"], e["post"]) for e in edges)
    n_false = int(rng.binomial(len(edges), rates["false_edge"]))
    for _ in range(n_false):
        for _ in range(100):
            pre, post = [str(x) for x in rng.choice(ids, size=2, replace=False)]
            if (pre, post) not in existing:
                break
        else:
            continue
        existing.add((pre, post))
        w = float(rng.uniform(0.1, 0.5))
        sign = "exc" if rng.random() < 0.7 else "inh"
        stored.append({"pre": pre, "post": post, "type": "graded", "sign": sign, "weight": w})
        rows.append({"pre_neuron": pre, "post_neuron": post, "kind": "false_edge", "true_weight": 0.0, "stored_weight": w})
    return stored, rows
