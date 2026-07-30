#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morningstar_hydra.api_keys import DEFAULT_ROLE, ROLES, ApiKeyStore


def _store_from_env_or_args(path: str | None) -> ApiKeyStore:
    selected = path or os.environ.get("HYDRA_API_KEY_STORE")
    if not selected:
        raise SystemExit("--store or HYDRA_API_KEY_STORE is required")
    return ApiKeyStore(Path(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Morningstar Hydra API keys")
    parser.add_argument("--store", default=None, help="JSON key store path. Defaults to HYDRA_API_KEY_STORE.")
    sub = parser.add_subparsers(dest="command", required=True)

    roles = sorted(ROLES)

    create = sub.add_parser("create", help="Create a new API key")
    create.add_argument("--name", required=True)
    create.add_argument(
        "--role",
        choices=roles,
        default=DEFAULT_ROLE,
        help=f"Rolle des Schluessels (Standard: {DEFAULT_ROLE}). "
             "Nur 'admin' erreicht rollengeschuetzte Modelle.",
    )

    list_parser = sub.add_parser("list", help="List API key metadata")
    list_parser.add_argument("--active-only", action="store_true", help="Hide revoked keys")

    revoke = sub.add_parser("revoke", help="Revoke an API key by id")
    revoke.add_argument("key_id")

    set_role = sub.add_parser("set-role", help="Change the role of an existing key")
    set_role.add_argument("key_id")
    set_role.add_argument("--role", choices=roles, required=True)

    args = parser.parse_args()
    store = _store_from_env_or_args(args.store)
    if args.command == "create":
        record, secret = store.create(args.name, role=args.role)
        print(secret)
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
    elif args.command == "list":
        print(json.dumps(store.list(include_revoked=not args.active_only), indent=2, sort_keys=True))
    elif args.command == "revoke":
        if store.revoke(args.key_id):
            print(json.dumps({"id": args.key_id, "revoked": True}, sort_keys=True))
        else:
            raise SystemExit("key id not found")
    elif args.command == "set-role":
        if store.set_role(args.key_id, args.role):
            print(json.dumps({"id": args.key_id, "role": args.role}, sort_keys=True))
        else:
            raise SystemExit("key id not found")


if __name__ == "__main__":
    main()
