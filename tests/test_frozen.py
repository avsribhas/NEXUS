import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "frozen.sha256"

def read_manifest() -> list[tuple[str, str]]:
    with MANIFEST.open("r") as f:
        lines = f.readlines()
    return [line.strip().split("  ", 1) for line in lines if line.strip()]

def test_manifest_not_empty():
    assert len(read_manifest()) >= 1

def test_frozen_files_unchanged():
    for digest, rel in read_manifest():
        path = ROOT / rel
        assert path.is_file(), f"frozen file missing: {rel}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, f"frozen file changed: {rel}"
