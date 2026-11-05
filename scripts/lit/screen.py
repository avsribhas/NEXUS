import json
import math
import os
import pathlib
import re
import sys
import urllib.request

URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"
NUM_GPU = int(os.environ.get("LIT_NUM_GPU", "0"))
THEMES = "connectome, state, motif, degeneracy, sbi, inference, calibration, ceiling, rigor, sim, errors, calcium, energy, other"
PROMPT = """Project: we test a tool that reads a neural circuit's activity, the perturbations applied to it and an imperfect wiring diagram, and reports the rules that decide which parts of the circuit switch on in a given brain state. We test it on simulated graded circuits with planted switches and decoys, using simulation-based inference and a pre-registered held-out set.

Paper title: {title}
Abstract: {abstract}

Reply with one JSON object with these keys:
relevance: 0 unrelated, 1 background, 2 useful, 3 directly competes with or is required by the project
theme: one of THEMES_HERE
finding: one plain sentence saying what the paper shows
method: a few words
organism: a few words, or none
quote: one sentence copied exactly from the abstract that supports the finding"""

PROMPT = PROMPT.replace("THEMES_HERE", THEMES)

def norm(s: str) -> str:
    return re.sub(r'\W+', ' ', s.lower()).strip()

TERMS = ("connectome", "wiring diagram", "neuromodul", "brain state", "behavioral state", "internal state", "state-dependent", "switch", "gating", "disinhibit", "gain modulation", "winner-take-all", "mutual inhibition", "flip-flop", "bistab", "perturb", "optogenetic", "c. elegans", "caenorhabditis", "simulation-based", "likelihood-free", "posterior", "identifiab", "degenera", "ground truth", "ground-truth", "benchmark", "calibrat", "conformal", "noise ceiling", "explainable variance", "preregist", "pre-regist", "leakage", "differentiable", "neuroml", "graded", "non-spiking", "gcamp", "calcium indicator", "synapse detection", "proofreading", "neurotransmitter prediction")

def priority(r: dict) -> tuple:
    text = (r["title"] + " " + r["abstract"]).lower()
    score = sum(term in text for term in TERMS)
    return (0 if "seed" in r["found_by"] else 1, -score, -math.log1p(r["cited_by"] or 0))

def ask(title: str, abstract: str) -> dict:
    data = json.dumps({"model": MODEL, "prompt": PROMPT.replace("{title}", title).replace("{abstract}", abstract), "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 220, "num_ctx": 2048, "num_gpu": NUM_GPU, "num_thread": 10}}).encode('utf-8')
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as f:
        response = json.loads(f.read().decode('utf-8'))
    return json.loads(response["response"])

def load_records(in_jsonl: str, out_jsonl: str, limit: int) -> list[dict]:
    records = []
    with open(in_jsonl, 'r') as f:
        for line in f:
            record = json.loads(line)
            if len(record["abstract"]) >= 200:
                records.append(record)
    records.sort(key=lambda r: priority(r))
    records = records[:limit]
    if pathlib.Path(out_jsonl).exists():
        done = {json.loads(line)["openalex"] for line in open(out_jsonl, 'r')}
        records = [r for r in records if r["openalex"] not in done]
    return records

def main():
    in_jsonl = sys.argv[1]
    out_jsonl = sys.argv[2]
    limit = int(sys.argv[3])

    records = load_records(in_jsonl, out_jsonl, limit)

    for i, record in enumerate(records):
        try:
            reply = ask(record["title"], record["abstract"][:1800])
            output = {
                "openalex": record["openalex"],
                "doi": record.get("doi", ""),
                "title": record["title"],
                "year": record.get("year", ""),
                "venue": record.get("venue", ""),
                "authors": record.get("authors", []),
                "n_authors": record.get("n_authors", 0),
                "cited_by": record.get("cited_by", 0),
                "found_by": record.get("found_by", ""),
                "themes": record.get("themes", []),
                "relevance": reply.get("relevance", ""),
                "theme": reply.get("theme", ""),
                "finding": reply.get("finding", ""),
                "method": reply.get("method", ""),
                "organism": reply.get("organism", ""),
                "quote": reply.get("quote", ""),
                "quote_ok": bool(norm(reply.get("quote", ""))) and norm(reply.get("quote", "")) in norm(record["abstract"])
            }
            with open(out_jsonl, 'a') as f:
                f.write(json.dumps(output) + '\n')
            print(f"{i+1}/{len(records)} {output['relevance']} {output['quote_ok']} {record['title'][:70]}", flush=True)
        except Exception as e:
            print(f"{record['title']}: {e}", flush=True)

if __name__ == "__main__":
    main()
