import json
import sys
import time
import urllib.request
import urllib.parse

def first(v):
    if isinstance(v, list) and v:
        return v[0]
    elif v is None:
        return ""
    else:
        return v

def fetch_metadata(doi):
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="/()")
    headers = {
        "Accept": "application/vnd.citationstyles.csl+json",
        "User-Agent": "dusk-lit/0.1"
    }
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return None

def process_line(line, out_file):
    parts = line.strip().split('\t')
    if len(parts) != 2 or not parts[1]:
        return None
    key, doi = parts
    if key.startswith('#'):
        return None

    metadata = fetch_metadata(doi)
    if metadata:
        m = metadata
        first_author = first(m.get("author", [{}])[0].get("family", m.get("author", [{}])[0].get("literal", "")))
        n_authors = len(m.get("author", []))
        year = None
        try:
            year = m["issued"]["date-parts"][0][0]
        except (KeyError, IndexError, TypeError):
            pass
        title = first(m.get("title", [""]))
        container = first(m.get("container-title", [""]))
        publisher = m.get("publisher", "")
        doc_type = m.get("type", "")
        out_file.write(json.dumps({
            "key": key,
            "doi": doi,
            "ok": True,
            "first_author": first_author,
            "n_authors": n_authors,
            "year": year,
            "title": title,
            "container": container,
            "publisher": publisher,
            "type": doc_type
        }) + '\n')
        out_file.flush()
        return True
    else:
        out_file.write(json.dumps({
            "key": key,
            "doi": doi,
            "ok": False,
            "error": "Failed to fetch metadata"
        }) + '\n')
        out_file.flush()
        print(f"FAIL {key} {doi} Failed to fetch metadata")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: verify_refs.py IN_TSV OUT_JSONL")
        sys.exit(1)

    in_tsv, out_jsonl = sys.argv[1], sys.argv[2]
    with open(in_tsv, 'r') as in_file, open(out_jsonl, 'w') as out_file:
        ok_count = 0
        fail_count = 0
        for line in in_file:
            result = process_line(line, out_file)
            if result is True:
                ok_count += 1
            elif result is False:
                fail_count += 1
            time.sleep(0.2)
        print(f"OK: {ok_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
