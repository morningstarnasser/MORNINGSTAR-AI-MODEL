from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import hmac
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

KEY_PREFIX = "msh_live_"
KEY_RE = re.compile(r"^msh_live_([A-Za-z0-9_-]{43,})$")
STORE_VERSION = 1

ROLES = frozenset({"admin", "user"})
DEFAULT_ROLE = "user"


def validate_role(role: str) -> str:
    """Strenge Pruefung fuer alle, die eine Rolle *vergeben*."""
    if not isinstance(role, str) or role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    return role


def normalize_role(value: Any) -> str:
    """Nachsichtige Pruefung fuer alles, was von der Platte zurueckkommt.

    Eine fehlende, beschaedigte oder unbekannte Rolle darf niemals Rechte
    ausweiten. Schluessel aus der Zeit vor den Rollen lesen sich deshalb als
    ``user``, nicht als ``admin``.
    """
    return value if isinstance(value, str) and value in ROLES else DEFAULT_ROLE


def generate_secret() -> str:
    raw = os.urandom(32)
    return KEY_PREFIX + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def hash_api_key(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def parse_key_id(secret: str) -> str:
    if not KEY_RE.fullmatch(secret):
        raise ValueError("invalid API key format")
    return "key_" + hash_api_key(secret)[:16]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ApiKeyStore:
    """Atomic JSON key store.

    The store keeps only SHA-256 hashes plus non-secret metadata. This is appropriate
    here because generated keys contain at least 256 bits of random entropy, making
    offline preimage attacks impractical when the store file is disclosed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _reject_symlink_components(self) -> None:
        absolute = Path(os.path.abspath(self.path))
        cursor = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            cursor /= component
            if cursor.is_symlink():
                raise ValueError("API key store path components must not be symlinks")

    def _ensure_safe_path(self) -> None:
        self._reject_symlink_components()
        parent = self.path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._reject_symlink_components()
        os.chmod(parent, 0o700)
        if self.path.exists():
            if not os.path.isfile(self.path):
                raise ValueError("API key store path must be a regular file")
            os.chmod(self.path, 0o600)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self._ensure_safe_path()
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            os.chmod(lock_path, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._ensure_safe_path()
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": STORE_VERSION, "keys": []}
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError("API key store contains invalid JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
            raise ValueError("API key store has invalid structure")
        return data

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        self._ensure_safe_path()
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=".keys.", suffix=".tmp", dir=str(self.path.parent))
        try:
            os.fchmod(tmp_fd, 0o600)
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def _sanitize(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": record["id"],
            "name": record["name"],
            "role": normalize_role(record.get("role")),
            "prefix": record["prefix"],
            "created_at": record["created_at"],
            "revoked_at": record.get("revoked_at"),
        }

    def create(self, name: str, role: str = DEFAULT_ROLE) -> tuple[dict[str, Any], str]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("key name is required")
        clean_role = validate_role(role)
        secret = generate_secret()
        digest = hash_api_key(secret)
        record = {
            "id": "key_" + digest[:16],
            "name": clean_name,
            "role": clean_role,
            "prefix": secret[:18],
            "hash": digest,
            "created_at": _now(),
            "revoked_at": None,
        }
        with self._locked():
            data = self._read_unlocked()
            data.setdefault("version", STORE_VERSION)
            data.setdefault("keys", []).append(record)
            self._write_unlocked(data)
        return self._sanitize(record), secret

    def list(self, include_revoked: bool = True) -> list[dict[str, Any]]:
        with self._locked():
            data = self._read_unlocked()
        records = data.get("keys", [])
        if not include_revoked:
            records = [item for item in records if not item.get("revoked_at")]
        return [self._sanitize(item) for item in records]

    def active_count(self) -> int:
        return len(self.list(include_revoked=False))

    def revoke(self, key_id: str) -> bool:
        changed = False
        found = False
        with self._locked():
            data = self._read_unlocked()
            for record in data.get("keys", []):
                if hmac.compare_digest(str(record.get("id", "")), key_id):
                    found = True
                    if not record.get("revoked_at"):
                        record["revoked_at"] = _now()
                        changed = True
                    break
            if not found:
                return False
            if changed:
                self._write_unlocked(data)
        return True

    def set_role(self, key_id: str, role: str) -> bool:
        wanted = validate_role(role)
        with self._locked():
            data = self._read_unlocked()
            for record in data.get("keys", []):
                if hmac.compare_digest(str(record.get("id", "")), key_id):
                    if record.get("role") != wanted:
                        record["role"] = wanted
                        self._write_unlocked(data)
                    return True
        return False

    def authenticate(self, secret: str) -> dict[str, Any] | None:
        """Wer fragt? Gibt den Datensatz ohne Hash zurueck, sonst ``None``."""
        if not isinstance(secret, str) or not KEY_RE.fullmatch(secret):
            return None
        digest = hash_api_key(secret)
        with self._locked():
            data = self._read_unlocked()
        # Kein vorzeitiges Verlassen der Schleife: die Laufzeit soll nicht
        # verraten, an welcher Stelle im Speicher ein Schluessel steht.
        found: dict[str, Any] | None = None
        for record in data.get("keys", []):
            stored = str(record.get("hash", ""))
            active = not record.get("revoked_at")
            matched = hmac.compare_digest(stored, digest)
            if active and matched:
                found = record
        return None if found is None else self._sanitize(found)

    def verify(self, secret: str) -> bool:
        return self.authenticate(secret) is not None
