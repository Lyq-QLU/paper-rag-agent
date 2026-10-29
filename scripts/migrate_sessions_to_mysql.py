"""Import existing JSON session catalogues into an empty MySQL catalogue.

Run with PAPER_MYSQL_URL set. Existing MySQL rows are never overwritten.
"""

import argparse
import json
from pathlib import Path

from src.mysql_sessions import MySQLSessionStore
from src.session_manager import sanitize_identifier


def migrate(data_root: Path, store: MySQLSessionStore) -> tuple[int, int]:
    imported_users = imported_sessions = 0
    for source in sorted(data_root.glob("*/sessions.json")):
        user_id = sanitize_identifier(source.parent.name)
        sessions = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(sessions, list) or any(
            not isinstance(item, dict) or not item.get("session_id") for item in sessions
        ):
            raise ValueError(f"Invalid session catalogue: {source}")
        if store.load(user_id):
            raise RuntimeError(f"MySQL catalogue already contains sessions for {user_id}; no rows overwritten")
        store.replace(user_id, sessions)
        imported_users += 1
        imported_sessions += len(sessions)
    return imported_users, imported_sessions


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/users"))
    args = parser.parse_args()
    url = os.getenv("PAPER_MYSQL_URL", "").strip()
    if not url:
        parser.error("Set PAPER_MYSQL_URL before running migration")
    users, sessions = migrate(args.data_root, MySQLSessionStore(url))
    print(f"Imported {sessions} sessions for {users} users")
