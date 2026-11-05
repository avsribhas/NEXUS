import json
import sys
import time
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import HTTPError, URLError

def get_json(url: str) -> dict:
  for _ in range(5):
    try:
      time.sleep(0.15)
      response = urlopen(Request(url, headers={"User-Agent": "dusk-lit/0.1"}), timeout=60)
      return json.load(response)
    except HTTPError as e:
      if e.code == 429 or e.code >= 500:
        time.sleep(5)
      else:
        return {}
    except URLError:
      time.sleep(5)
  return {}

def abstract_text(inv: dict | None) -> str:
    if not inv:
        return ""
    pairs = [(pos, word) for word, positions in inv.items() for pos in positions]
    pairs.sort()
    return " ".join(word for _, word in pairs)

def to_record(w: dict) -> dict:
    return {
        "openalex": w["id"],
        "doi": w.get("doi") or "",
        "title": w.get("title") or "",
        "year": w.get("publication_year"),
        "venue": (((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""),
        "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:8]],
        "n_authors": len(w.get("authorships", [])),
        "cited_by": w.get("cited_by_count", 0),
        "type": w.get("type", ""),
        "abstract": abstract_text(w.get("abstract_inverted_index")),
        "pmid": ((w.get("ids") or {}).get("pmid") or ""),
        "pmcid": ((w.get("ids") or {}).get("pmcid") or ""),
        "oa_url": ((w.get("open_access") or {}).get("oa_url") or ""),
        "refs": w.get("referenced_works", []),
        "themes": [],
        "found_by": []
    }

def add(store: dict, rec: dict, theme: str, how: str) -> None:
    if not rec["title"]:
        return
    if rec["openalex"] not in store:
        store[rec["openalex"]] = rec
    if theme not in store[rec["openalex"]]["themes"]:
        store[rec["openalex"]]["themes"].append(theme)
    if how not in store[rec["openalex"]]["found_by"]:
        store[rec["openalex"]]["found_by"].append(how)

def search(query: str, n: int) -> list[dict]:
    url = f"https://api.openalex.org/works?search={quote(query)}&per-page={n}&filter=type:article|preprint|review"
    res = get_json(url)
    return res.get("results", [])

def seed_work(title: str) -> dict | None:
    url = f"https://api.openalex.org/works?filter=title.search:{quote(title.replace(',', ''))}&per-page=5&sort=cited_by_count:desc"
    res = get_json(url)
    return res.get("results", [])[0] if res.get("results") else None

def citers(openalex_id: str, n: int) -> list[dict]:
    short_id = openalex_id.split("/")[-1]
    url = f"https://api.openalex.org/works?filter=cites:{short_id}&per-page={n}&sort=cited_by_count:desc"
    res = get_json(url)
    return res.get("results", [])

def by_ids(ids: list[str]) -> list[dict]:
    short = [i.rsplit("/", 1)[-1] for i in ids]
    results = []
    for i in range(0, len(short), 50):
        batch = short[i:i + 50]
        data = get_json("https://api.openalex.org/works?filter=openalex:" + "|".join(batch) + "&per-page=50")
        results.extend(data.get("results", []))
    return results

def add_seed(store: dict, theme: str, title: str) -> None:
  seed = seed_work(title)
  if seed is None:
    print("NOT FOUND: " + title, flush=True)
    return
  print(title, " -> ", seed.get("title"), seed.get("publication_year"), flush=True)
  add(store, to_record(seed), theme, "seed")
  for w in citers(seed["id"], 40):
    add(store, to_record(w), theme, "cited_by_seed")
  for w in by_ids(seed.get("referenced_works") or []):
    add(store, to_record(w), theme, "ref_of_seed")

def main():
    TOPICS_TSV, SEEDS_TSV, OUT_JSONL = sys.argv[1:]
    store = {}

    with open(TOPICS_TSV) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                theme, query = line.strip().split("\t")
                for rec in search(query, 60):
                    add(store, to_record(rec), theme, "search")

    with open(SEEDS_TSV) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                theme, title = line.strip().split("\t")
                add_seed(store, theme, title)

    with open(OUT_JSONL, "w") as f:
        for rec in store.values():
            f.write(json.dumps(rec) + "\n")

    print(f"Total count: {len(store)}")
    print(f"Non-empty abstracts: {sum(1 for rec in store.values() if rec['abstract'])}")

if __name__ == "__main__":
    main()
