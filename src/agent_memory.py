"""SQLite-backed long-term preferences and resumable approvals."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentMemoryStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1,
                ttl_days INTEGER NOT NULL DEFAULT 90,
                preferences_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pending_action (
                action_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pending_user
                ON pending_action(user_id, status, expires_at);
            """
        )
        self.connection.commit()

    def get_profile(self, user_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM user_profile WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return {"user_id": user_id, "enabled": True, "ttl_days": 90, "preferences": {}}
        preferences = json.loads(row["preferences_json"] or "{}")
        updated_at = datetime.fromisoformat(row["updated_at"])
        ttl_days = int(row["ttl_days"])
        if preferences and updated_at + timedelta(days=ttl_days) < utc_now():
            preferences = {}
            self.connection.execute(
                "UPDATE user_profile SET preferences_json='{}', updated_at=? WHERE user_id=?",
                (utc_now().isoformat(), user_id),
            )
            self.connection.commit()
        return {
            "user_id": user_id,
            "enabled": bool(row["enabled"]),
            "ttl_days": ttl_days,
            "preferences": preferences,
            "updated_at": row["updated_at"],
        }

    def update_settings(
        self, user_id: str, *, enabled: bool | None = None, ttl_days: int | None = None
    ) -> dict[str, Any]:
        profile = self.get_profile(user_id)
        if enabled is not None:
            profile["enabled"] = bool(enabled)
        if ttl_days is not None:
            profile["ttl_days"] = max(1, min(int(ttl_days), 3650))
        self._write_profile(profile)
        return self.get_profile(user_id)

    def remember(self, user_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
        profile = self.get_profile(user_id)
        if not profile["enabled"]:
            return profile
        merged = dict(profile["preferences"])
        for key, value in preferences.items():
            if value not in (None, "", [], {}):
                merged[str(key)] = value
        profile["preferences"] = merged
        self._write_profile(profile)
        return self.get_profile(user_id)

    def clear_profile(self, user_id: str) -> None:
        self.connection.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
        self.connection.commit()

    def _write_profile(self, profile: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO user_profile(user_id, enabled, ttl_days, preferences_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                enabled=excluded.enabled,
                ttl_days=excluded.ttl_days,
                preferences_json=excluded.preferences_json,
                updated_at=excluded.updated_at
            """,
            (
                profile["user_id"], int(profile["enabled"]), int(profile["ttl_days"]),
                json.dumps(profile["preferences"], ensure_ascii=False), utc_now().isoformat(),
            ),
        )
        self.connection.commit()

    def create_pending(
        self, *, action_id: str, user_id: str, thread_id: str,
        action_type: str, payload: dict[str, Any], ttl_minutes: int = 30,
    ) -> dict[str, Any]:
        created = utc_now()
        expires = created + timedelta(minutes=max(1, ttl_minutes))
        self.connection.execute(
            """
            INSERT OR REPLACE INTO pending_action
            (action_id, user_id, thread_id, action_type, payload_json, status, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (action_id, user_id, thread_id, action_type,
             json.dumps(payload, ensure_ascii=False), created.isoformat(), expires.isoformat()),
        )
        self.connection.commit()
        return self.get_pending(action_id) or {}

    def get_pending(self, action_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM pending_action WHERE action_id = ?", (action_id,)
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        if item["status"] == "pending" and datetime.fromisoformat(item["expires_at"]) < utc_now():
            self.connection.execute(
                "UPDATE pending_action SET status='expired' WHERE action_id=?", (action_id,)
            )
            self.connection.commit()
            item["status"] = "expired"
        return item

    def list_pending(self, user_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT action_id FROM pending_action WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
        return [item for row in rows if (item := self.get_pending(row["action_id"])) is not None]

    def resolve_pending(self, action_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        item = self.get_pending(action_id)
        if item is None:
            raise KeyError(action_id)
        if item["status"] != "pending":
            return item
        payload = dict(item["payload"])
        payload["decision"] = decision
        self.connection.execute(
            """UPDATE pending_action SET status='resolved', payload_json=?, resolved_at=?
               WHERE action_id=?""",
            (json.dumps(payload, ensure_ascii=False), utc_now().isoformat(), action_id),
        )
        self.connection.commit()
        return self.get_pending(action_id) or {}

    def close(self) -> None:
        self.connection.close()
