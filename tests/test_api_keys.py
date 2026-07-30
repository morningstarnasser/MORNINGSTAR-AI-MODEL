import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from morningstar_hydra.api_keys import (
    DEFAULT_ROLE,
    ROLES,
    ApiKeyStore,
    hash_api_key,
    parse_key_id,
)

from tests.support import temporary_directory


class ApiKeyStoreTests(unittest.TestCase):
    def test_create_verify_list_revoke_without_plaintext_storage(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "secure" / "keys.json"
            store = ApiKeyStore(path)
            record, secret = store.create("prod")

            self.assertTrue(secret.startswith("msh_live_"))
            self.assertEqual(parse_key_id(secret), record["id"])
            self.assertTrue(store.verify(secret))
            self.assertFalse(store.verify(secret + "x"))

            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self.assertNotIn(secret, raw)
            self.assertEqual(data["keys"][0]["hash"], hash_api_key(secret))

            listed = store.list()
            self.assertEqual(listed[0]["id"], record["id"])
            self.assertEqual(listed[0]["name"], "prod")
            self.assertNotIn("hash", listed[0])
            self.assertNotIn("secret", listed[0])

            self.assertTrue(store.revoke(record["id"]))
            self.assertFalse(store.verify(secret))
            self.assertTrue(store.revoke(record["id"]))

    def test_file_permissions_and_parent_mode(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "secure" / "keys.json"
            store = ApiKeyStore(path)
            store.create("prod")
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_rejects_symlink_store_path(self):
        with temporary_directory() as tmp:
            target = Path(tmp) / "target.json"
            link = Path(tmp) / "keys.json"
            target.write_text("{}", encoding="utf-8")
            os.symlink(target, link)
            with self.assertRaises(ValueError):
                ApiKeyStore(link).list()

    def test_rejects_symlinked_parent_component(self):
        with temporary_directory() as tmp:
            trusted = Path(tmp) / "trusted"
            trusted.mkdir()
            parent_link = Path(tmp) / "linked-parent"
            os.symlink(trusted, parent_link)
            with self.assertRaises(ValueError):
                ApiKeyStore(parent_link / "keys.json").create("prod")

    def test_malformed_keys_do_not_verify(self):
        with temporary_directory() as tmp:
            store = ApiKeyStore(Path(tmp) / "keys.json")
            store.create("prod")
            self.assertFalse(store.verify(""))
            self.assertFalse(store.verify("not-a-key"))
            self.assertFalse(store.verify("msh_live_short"))


class ApiKeyRoleTests(unittest.TestCase):
    """Rollen entscheiden ueber Rechte, darum faellt jeder Zweifelsfall auf ``user``."""

    def _store(self, tmp: str) -> ApiKeyStore:
        return ApiKeyStore(Path(tmp) / "secure" / "keys.json")

    def _write_raw(self, path: Path, record: dict) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 1, "keys": [record]}, indent=2), encoding="utf-8"
        )

    def test_default_role_is_user(self):
        with temporary_directory() as tmp:
            store = self._store(tmp)
            record, secret = store.create("laptop")
            self.assertEqual(record["role"], DEFAULT_ROLE)
            self.assertEqual(DEFAULT_ROLE, "user")
            self.assertEqual(store.authenticate(secret)["role"], "user")

    def test_admin_role_round_trips(self):
        with temporary_directory() as tmp:
            store = self._store(tmp)
            record, secret = store.create("ali", role="admin")
            self.assertEqual(record["role"], "admin")
            self.assertEqual(store.authenticate(secret)["role"], "admin")
            self.assertEqual(store.list()[0]["role"], "admin")

    def test_create_rejects_unknown_role(self):
        with temporary_directory() as tmp:
            store = self._store(tmp)
            with self.assertRaises(ValueError):
                store.create("ali", role="superadmin")
            with self.assertRaises(ValueError):
                store.create("ali", role="")

    def test_legacy_record_without_role_authenticates_as_user(self):
        """Eine Migration darf niemandem Rechte schenken."""
        with temporary_directory() as tmp:
            path = Path(tmp) / "secure" / "keys.json"
            store = ApiKeyStore(path)
            _, secret = store.create("alt")
            data = json.loads(path.read_text(encoding="utf-8"))
            del data["keys"][0]["role"]
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            self.assertTrue(store.verify(secret))
            self.assertEqual(store.authenticate(secret)["role"], "user")
            self.assertEqual(store.list()[0]["role"], "user")

    def test_unknown_stored_role_degrades_to_user(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "secure" / "keys.json"
            store = ApiKeyStore(path)
            _, secret = store.create("ali", role="admin")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["keys"][0]["role"] = "root"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            self.assertEqual(store.authenticate(secret)["role"], "user")

    def test_non_string_stored_role_degrades_to_user(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "secure" / "keys.json"
            store = ApiKeyStore(path)
            _, secret = store.create("ali", role="admin")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["keys"][0]["role"] = {"role": "admin"}
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            self.assertEqual(store.authenticate(secret)["role"], "user")

    def test_authenticate_rejects_revoked_and_invalid_keys(self):
        with temporary_directory() as tmp:
            store = self._store(tmp)
            record, secret = store.create("ali", role="admin")
            self.assertIsNotNone(store.authenticate(secret))

            store.revoke(record["id"])
            self.assertIsNone(store.authenticate(secret))
            self.assertIsNone(store.authenticate("not-a-key"))
            self.assertIsNone(store.authenticate(""))

    def test_authenticate_never_returns_the_hash(self):
        with temporary_directory() as tmp:
            store = self._store(tmp)
            _, secret = store.create("ali", role="admin")
            identity = store.authenticate(secret)
            self.assertNotIn("hash", identity)
            self.assertEqual(
                set(identity), {"id", "name", "role", "prefix", "created_at", "revoked_at"}
            )

    def test_set_role_promotes_and_demotes(self):
        with temporary_directory() as tmp:
            store = self._store(tmp)
            record, secret = store.create("ali")
            self.assertEqual(store.authenticate(secret)["role"], "user")

            self.assertTrue(store.set_role(record["id"], "admin"))
            self.assertEqual(store.authenticate(secret)["role"], "admin")

            self.assertTrue(store.set_role(record["id"], "user"))
            self.assertEqual(store.authenticate(secret)["role"], "user")

    def test_set_role_rejects_unknown_role_and_unknown_id(self):
        with temporary_directory() as tmp:
            store = self._store(tmp)
            record, _ = store.create("ali")
            with self.assertRaises(ValueError):
                store.set_role(record["id"], "root")
            self.assertFalse(store.set_role("key_doesnotexist", "admin"))

    def test_known_roles_are_exactly_admin_and_user(self):
        self.assertEqual(ROLES, frozenset({"admin", "user"}))


if __name__ == "__main__":
    unittest.main()
