import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pyrage

def sha256_file(path) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_placeholder_identity() -> tuple:
    identity = pyrage.x25519.Identity.generate()
    return str(identity), str(identity.to_public())

def encrypt_file(src, dst, recipient: str) -> None:
    with open(src, 'rb') as f:
        data_bytes = f.read()
    encrypted_bytes = pyrage.encrypt(data_bytes, [pyrage.x25519.Recipient.from_str(recipient)])
    with open(dst, 'wb') as f:
        f.write(encrypted_bytes)

def decrypt_file(src, dst, secret: str) -> None:
    with open(src, 'rb') as f:
        encrypted_bytes = f.read()
    identity = pyrage.x25519.Identity.from_str(secret)
    plain_bytes = pyrage.decrypt(encrypted_bytes, [identity])
    with open(dst, 'wb') as f:
        f.write(plain_bytes)

class AccessLimitExceeded(RuntimeError):
    """Raised when the held-out access limit is reached."""

class AccessLog:
    def __init__(self, path, limit: int):
        self.path = Path(path)
        self.limit = limit

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, 'r') as f:
            return sum(1 for line in f if line.strip())

    def record(self, who: str, what: str, reason: str) -> None:
        if self.count() >= self.limit:
            raise AccessLimitExceeded(f"Access limit {self.limit} reached for {self.path}")
        with open(self.path, 'a') as f:
            f.write(json.dumps({
                "time": datetime.now(timezone.utc).isoformat(),
                "who": who,
                "what": what,
                "reason": reason
            }) + "\n")

def stamp_file(path) -> Path:
    ots_path = Path(sys.executable).parent / "ots"
    subprocess.run([str(ots_path), "stamp", str(path)], check=True, capture_output=True)
    return Path(str(path) + ".ots")
