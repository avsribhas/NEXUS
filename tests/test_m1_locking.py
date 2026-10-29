import json
import os

import pytest

from nexus.format.locking import (
    AccessLimitExceeded,
    AccessLog,
    decrypt_file,
    encrypt_file,
    generate_placeholder_identity,
    sha256_file,
    stamp_file,
)

def test_sha256_file(tmp_path):
    p = tmp_path / "abc.txt"; p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

def test_placeholder_identity_format():
    secret, recipient = generate_placeholder_identity()
    assert secret.startswith("AGE-SECRET-KEY-1")
    assert recipient.startswith("age1")

def test_encrypt_decrypt_round_trip(tmp_path):
    secret, recipient = generate_placeholder_identity()
    src = tmp_path / "labels.bin"; src.write_bytes(b"held-out labels")
    enc = tmp_path / "labels.age"; out = tmp_path / "labels.out"
    encrypt_file(src, enc, recipient)
    assert enc.read_bytes() != src.read_bytes()
    decrypt_file(enc, out, secret)
    assert out.read_bytes() == b"held-out labels"

def test_wrong_key_cannot_decrypt(tmp_path):
    _, recipient = generate_placeholder_identity()
    other_secret, _ = generate_placeholder_identity()
    src = tmp_path / "a.bin"; src.write_bytes(b"x")
    enc = tmp_path / "a.age"
    encrypt_file(src, enc, recipient)
    with pytest.raises(Exception): decrypt_file(enc, tmp_path / "a.out", other_secret)

def test_access_log_enforces_limit(tmp_path):
    path = tmp_path / "access.jsonl"
    log = AccessLog(path, limit=2)
    log.record(who="claude", what="heldout/circuits", reason="test")
    log.record(who="claude", what="heldout/circuits", reason="test")
    with pytest.raises(AccessLimitExceeded): log.record(who="claude", what="heldout/circuits", reason="test")
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    for line in lines: assert set(json.loads(line)) == {"time", "who", "what", "reason"}

def test_access_log_counts_existing_entries(tmp_path):
    path = tmp_path / "access.jsonl"
    first = AccessLog(path, limit=2)
    first.record(who="a", what="b", reason="c")
    first.record(who="a", what="b", reason="c")
    second = AccessLog(path, limit=2)
    with pytest.raises(AccessLimitExceeded): second.record(who="a", what="b", reason="c")

@pytest.mark.skipif(os.environ.get("NEXUS_NETWORK_TESTS") != "1", reason="needs network")
def test_stamp_file(tmp_path):
    p = tmp_path / "manifest.txt"; p.write_text("manifest")
    proof = stamp_file(p)
    assert proof == tmp_path / "manifest.txt.ots"
    assert proof.is_file()
