"""Optional MySQL backend for session metadata.

PDFs, indexes, chat histories, approvals, and task snapshots remain in their
existing stores. This module owns only the per-user session catalogue.
"""

from __future__ import annotations

import json
import threading
from urllib.parse import unquote, urlsplit


def connection_options(url: str) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme != "mysql" or not parsed.hostname or not parsed.username:
        raise ValueError("PAPER_MYSQL_URL must be mysql://user:password@host:3306/database")
    database = unquote(parsed.path.lstrip("/"))
    if not database or "/" in database or parsed.query or parsed.fragment:
        raise ValueError("PAPER_MYSQL_URL must name one database without query parameters")
    try:
        port = parsed.port or 3306
    except ValueError as exc:
        raise ValueError("PAPER_MYSQL_URL has an invalid port") from exc
    return {
        "host": parsed.hostname,
        "port": port,
        "user": unquote(parsed.username),
        "password": unquote(parsed.password or ""),
        "database": database,
        "charset": "utf8mb4",
        "connect_timeout": 5,
        "read_timeout": 10,
        "write_timeout": 10,
        "autocommit": False,
    }


class MySQLSessionStore:
    def __init__(self, url: str) -> None:
        self.options = connection_options(url)
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def _connect(self):
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError("MySQL 会话存储需要安装 PyMySQL：pip install -r requirements.txt") from exc
        options = {**self.options, "cursorclass": pymysql.cursors.DictCursor}
        connection = pymysql.connect(**options)
        try:
            self._ensure_schema(connection)
        except Exception:
            connection.close()
            raise
        return connection

    def _ensure_schema(self, connection) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            with connection.cursor() as cursor:
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS paper_sessions (
                        user_id VARCHAR(48) NOT NULL,
                        session_id VARCHAR(48) NOT NULL,
                        payload_json JSON NOT NULL,
                        updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        PRIMARY KEY (user_id, session_id),
                        INDEX idx_paper_sessions_user_updated (user_id, updated_at)
                    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"""
                )
            connection.commit()
            self._schema_ready = True

    @staticmethod
    def _decode(value) -> dict:
        return value if isinstance(value, dict) else json.loads(value)

    def load(self, user_id: str) -> list[dict]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT payload_json FROM paper_sessions WHERE user_id=%s "
                    "ORDER BY updated_at DESC, session_id DESC", (user_id,)
                )
                return [self._decode(row["payload_json"]) for row in cursor.fetchall()]
        finally:
            connection.close()

    def upsert(self, user_id: str, session_id: str, updates: dict, default: dict) -> dict:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT IGNORE INTO paper_sessions (user_id, session_id, payload_json) "
                    "VALUES (%s, %s, %s)",
                    (user_id, session_id, json.dumps(default, ensure_ascii=False)),
                )
                cursor.execute(
                    "SELECT payload_json FROM paper_sessions WHERE user_id=%s AND session_id=%s FOR UPDATE",
                    (user_id, session_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError("会话元数据写入失败")
                item = self._decode(row["payload_json"])
                item.update(updates)
                from src.session_manager import now_text
                item["updated_at"] = now_text()
                cursor.execute(
                    "UPDATE paper_sessions SET payload_json=%s, updated_at=NOW(6) "
                    "WHERE user_id=%s AND session_id=%s",
                    (json.dumps(item, ensure_ascii=False), user_id, session_id),
                )
            connection.commit()
            return item
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def delete(self, user_id: str, session_id: str) -> str | None:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM paper_sessions WHERE user_id=%s AND session_id=%s",
                    (user_id, session_id),
                )
                cursor.execute(
                    "SELECT session_id FROM paper_sessions WHERE user_id=%s "
                    "ORDER BY updated_at DESC, session_id DESC LIMIT 1", (user_id,)
                )
                row = cursor.fetchone()
            connection.commit()
            return row["session_id"] if row else None
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def replace(self, user_id: str, sessions: list[dict]) -> None:
        """Bulk import for an explicit migration; ordinary writes use upsert."""
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM paper_sessions WHERE user_id=%s", (user_id,))
                for item in sessions:
                    cursor.execute(
                        "INSERT INTO paper_sessions (user_id, session_id, payload_json, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (user_id, item["session_id"], json.dumps(item, ensure_ascii=False),
                         item.get("updated_at") or item.get("created_at") or "1970-01-01 00:00:00"),
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
