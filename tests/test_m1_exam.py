import json
import os
import pytest
from nexus.format.exam import lock_files, open_held_out, verify_lock
from nexus.format.locking import AccessLimitExceeded, AccessLog, encrypt_file, generate_placeholder_identity, sha256_file

def make_files(root):
    os.makedirs(root / "scoring", exist_ok=True)
    (root / "manifest.json").write_text("{}\n")
    (root / "scoring" / "score.py").write_text("x = 1\n")
    (root / "thresholds.json").write_text('{"recovery": 0.8}\n')
    return [root / "manifest.json", root / "scoring" / "score.py", root / "thresholds.json"]

def test_lock_record_contents(tmp_path):
    paths = make_files(tmp_path)
    record = lock_files(paths, tmp_path / "lock.json", tmp_path)
    assert record["schema_version"] == "0.1.0"
    assert set(record["files"]) == {"manifest.json", "scoring/score.py", "thresholds.json"}
    assert record["files"]["scoring/score.py"] == sha256_file(tmp_path / "scoring" / "score.py")

def test_lock_file_is_canonical_json(tmp_path):
    paths = make_files(tmp_path)
    record = lock_files(paths, tmp_path / "lock.json", tmp_path)
    text = (tmp_path / "lock.json").read_text()
    assert text == json.dumps(record, sort_keys=True, indent=2) + "\n"

def test_verify_clean(tmp_path):
    paths = make_files(tmp_path)
    record = lock_files(paths, tmp_path / "lock.json", tmp_path)
    assert verify_lock(tmp_path / "lock.json", tmp_path) == []

def test_verify_detects_change(tmp_path):
    paths = make_files(tmp_path)
    record = lock_files(paths, tmp_path / "lock.json", tmp_path)
    (tmp_path / "scoring" / "score.py").write_text("x = 2\n")
    assert verify_lock(tmp_path / "lock.json", tmp_path) == ["scoring/score.py"]

def test_verify_detects_missing_file(tmp_path):
    paths = make_files(tmp_path)
    record = lock_files(paths, tmp_path / "lock.json", tmp_path)
    os.remove(tmp_path / "thresholds.json")
    assert verify_lock(tmp_path / "lock.json", tmp_path) == ["thresholds.json"]

def test_path_outside_root_rejected(tmp_path):
    inner = tmp_path / "inner"
    inner.mkdir()
    (tmp_path / "outside.txt").write_text("x")
    with pytest.raises(ValueError):
        lock_files([tmp_path / "outside.txt"], inner / "lock.json", inner)

def test_open_held_out_logs_and_decrypts(tmp_path):
    secret, recipient = generate_placeholder_identity()
    (tmp_path / "labels.bin").write_bytes(b"labels")
    encrypt_file(tmp_path / "labels.bin", tmp_path / "labels.age", recipient)
    log = AccessLog(tmp_path / "access.jsonl", limit=1)
    open_held_out(tmp_path / "labels.age", tmp_path / "labels.out", secret, log, "claude", "scoring")
    assert (tmp_path / "labels.out").read_bytes() == b"labels" and log.count() == 1

def test_open_held_out_enforces_limit(tmp_path):
    secret, recipient = generate_placeholder_identity()
    (tmp_path / "labels.bin").write_bytes(b"labels")
    encrypt_file(tmp_path / "labels.bin", tmp_path / "labels.age", recipient)
    log = AccessLog(tmp_path / "access.jsonl", limit=1)
    open_held_out(tmp_path / "labels.age", tmp_path / "first.out", secret, log, "claude", "scoring")
    with pytest.raises(AccessLimitExceeded):
        open_held_out(tmp_path / "labels.age", tmp_path / "second.out", secret, log, "claude", "scoring")
    assert not (tmp_path / "second.out").exists() and log.count() == 1

def test_failed_open_still_counts(tmp_path):
    secret, recipient = generate_placeholder_identity()
    other_secret, _ = generate_placeholder_identity()
    (tmp_path / "labels.bin").write_bytes(b"labels")
    encrypt_file(tmp_path / "labels.bin", tmp_path / "labels.age", recipient)
    log = AccessLog(tmp_path / "access.jsonl", limit=5)
    with pytest.raises(Exception):
        open_held_out(tmp_path / "labels.age", tmp_path / "labels.out", other_secret, log, "claude", "scoring")
    assert log.count() == 1

@pytest.mark.skipif(os.environ.get("NEXUS_NETWORK_TESTS") != "1", reason="needs network")
def test_lock_with_stamp(tmp_path):
    paths = make_files(tmp_path)
    lock_files(paths, tmp_path / "lock.json", tmp_path, stamp=True)
    assert not (tmp_path / "lock.ots").exists() and (tmp_path / "lock.json.ots").is_file()
