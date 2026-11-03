import json
from pathlib import Path
from nexus.format.locking import decrypt_file, sha256_file, stamp_file

SCHEMA_VERSION = "0.1.0"

def lock_files(paths, record_path, root, stamp=False) -> dict:
    root = Path(root).resolve()
    files = {}
    for p in paths:
        full = Path(p).resolve()
        if root not in full.parents and root != full:
            raise ValueError(f"Path {p} is outside root {root}")
        rel = full.relative_to(root).as_posix()
        files[rel] = sha256_file(full)
    record = {"schema_version": SCHEMA_VERSION, "files": files}
    with open(record_path, "w") as f:
        f.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
    if stamp:
        stamp_file(record_path)
    return record

def verify_lock(record_path, root) -> list:
    root = Path(root)
    with open(record_path, "r") as f:
        record = json.load(f)
    changed = []
    for rel, digest in record["files"].items():
        full = root / rel
        if not full.is_file():
            changed.append(rel)
        elif sha256_file(full) != digest:
            changed.append(rel)
    return sorted(changed)

def open_held_out(encrypted_path, out_path, secret, access_log, who, reason) -> None:
    access_log.record(who=who, what=str(encrypted_path), reason=reason)
    decrypt_file(encrypted_path, out_path, secret)
