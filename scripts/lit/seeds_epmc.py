import difflib
import json
import re
import sys
import time
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from pathlib import Path

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

def get_json(url: str) -> dict:
    for _ in range(4):
        time.sleep(0.2)
        try:
            return json.load(urlopen(Request(url, headers={"User-Agent": "dusk-lit/0.1"}), timeout=60))
        except HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(5)
            else:
                return {}
        except URLError:
            time.sleep(5)
    return {}

def to_record(p: dict) -> dict:
    return {
        "openalex": f"epmc:{p.get('source','')}:{p.get('id','')}",
        "doi": f"https://doi.org/{p['doi']}" if p.get("doi") else "",
        "title": (p.get("title") or "").rstrip("."),
        "year": int(p["pubYear"]) if p.get("pubYear") else None,
        "venue": ((p.get("journalInfo") or {}).get("journal",{}).get("title") or p.get("bookOrReportDetails",{}).get("publisher") or ""),
        "authors": [a.get("fullName", "") for a in (p.get("authorList") or {}).get("author",[])[:8]],
        "n_authors": len(p.get("authorList",{}).get("author",[])),
        "cited_by": p.get("citedByCount") or 0,
        "type": p.get("pubType") or "",
        "abstract": re.sub(r"<[^>]+>", " ", p.get("abstractText") or "").strip(),
        "pmid": p.get("pmid") or "",
        "pmcid": p.get("pmcid") or "",
        "oa_url": "",
        "refs": [],
        "themes": [],
        "found_by": []
    }

def search_core(query: str, n: int) -> list[dict]:
    return get_json(BASE + "/search?query=" + quote(query) + "&resultType=core&format=json&pageSize=" + str(n)).get("resultList", {}).get("result", [])

def key(title: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', "", title.lower())

def find_seed(title: str) -> dict | None:
    clean = re.sub(r'["]', " ", title)
    results = search_core('TITLE:"' + clean + '"', 5) + search_core(clean, 5)
    for r in results:
        if difflib.SequenceMatcher(None, key(title), key(r.get("title") or "")).ratio() >= 0.8:
            return r
    return None

def linked(p: dict, kind: str, n: int) -> list[dict]:
    data = get_json(f"{BASE}/{p['source']}/{p['id']}/{kind}?format=json&page=1&pageSize={n}")
    items = data.get("referenceList", {}).get("reference", []) + data.get("citationList", {}).get("citation", [])
    ids = [item.get("id") for item in items if item.get("source") == "MED" and item.get("id")]
    results = []
    for j in range(0, len(ids), 20):
        batch = ids[j:j + 20]
        results.extend(search_core(" OR ".join("EXT_ID:" + x for x in batch), 20))
    return results

def add(store: dict, result: dict, theme: str, how: str) -> None:
    rec = to_record(result)
    if not rec["title"]:
        return
    k = key(rec["title"])
    if k not in store:
        store[k] = rec
    if theme not in store[k]["themes"]:
        store[k]["themes"].append(theme)
    if how not in store[k]["found_by"]:
        store[k]["found_by"].append(how)
    if not store[k]["abstract"]:
        store[k]["abstract"] = rec["abstract"]

def save(store: dict, path) -> None:
    with open(path, "w") as f:
        for record in store.values():
            f.write(json.dumps(record) + "\n")

def process_seed(store: dict, theme: str, title: str) -> None:
    seed = find_seed(title)
    if seed is None:
        print("NOT FOUND: " + title, flush=True)
        return
    print(title, " -> ", seed.get("title"), seed.get("pubYear"), flush=True)
    add(store, seed, theme, "seed")
    for r in linked(seed, "citations", 100):
        add(store, r, theme, "cited_by_seed")
    for r in linked(seed, "references", 100):
        add(store, r, theme, "ref_of_seed")

def main():
    store = {}
    seeds_tsv = Path(sys.argv[1])
    corpus_jsonl = Path(sys.argv[2])
    with corpus_jsonl.open() as f:
        for line in f:
            record = json.loads(line)
            store[key(record["title"])] = record
    with seeds_tsv.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            theme, title = line.split("\t", 1)
            try:
                process_seed(store, theme, title)
            except Exception as e:
                print("ERROR " + title + ": " + repr(e), flush=True)
    save(store, corpus_jsonl)
    print(len(store))
    print(sum(1 for record in store.values() if record.get("abstract")))

if __name__ == "__main__":
    main()
