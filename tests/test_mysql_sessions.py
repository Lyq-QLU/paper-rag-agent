import json

import pytest

from src import session_manager as sm
from src.mysql_sessions import connection_options
from scripts.migrate_sessions_to_mysql import migrate


def test_mysql_url_parses_encoded_credentials():
    options = connection_options("mysql://paper:p%40ss%2Fword@localhost:3307/paper_rag")
    assert options["password"] == "p@ss/word"
    assert options["port"] == 3307
    assert options["database"] == "paper_rag"


@pytest.mark.parametrize("url", ["postgres://u:p@host/db", "mysql://host/db", "mysql://u:p@host/", "mysql://u:p@host:bad/db"])
def test_mysql_url_rejects_invalid_configuration(url):
    with pytest.raises(ValueError):
        connection_options(url)


def test_mysql_backend_routes_session_operations_without_json_fallback(monkeypatch, tmp_path):
    class Store:
        def __init__(self):
            self.rows = {}

        def load(self, user):
            return list(self.rows.get(user, {}).values())

        def upsert(self, user, session, updates, default):
            item = self.rows.setdefault(user, {}).setdefault(session, default)
            item.update(updates)
            return item

        def delete(self, user, session):
            self.rows.get(user, {}).pop(session, None)
            return next(iter(self.rows.get(user, {})), None)

    store = Store()
    monkeypatch.setattr(sm, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(sm, "_mysql_store", lambda url: store)
    monkeypatch.setenv("PAPER_SESSION_STORE", "mysql")
    monkeypatch.setenv("PAPER_MYSQL_URL", "mysql://u:p@localhost/db")
    meta = sm.upsert_session_meta("u", "s", title="论文")
    assert meta["title"] == "论文"
    assert sm.load_user_sessions("u") == [meta]
    assert sm.delete_session("u", "s") is None
    assert sm.load_user_sessions("u") == []
    assert not sm.get_sessions_file("u").exists()


def test_mysql_configuration_missing_url_fails(monkeypatch):
    monkeypatch.setenv("PAPER_SESSION_STORE", "mysql")
    monkeypatch.delenv("PAPER_MYSQL_URL", raising=False)
    with pytest.raises(ValueError, match="PAPER_MYSQL_URL"):
        sm.load_user_sessions("u")


def test_migration_refuses_existing_mysql_rows(tmp_path):
    source = tmp_path / "u" / "sessions.json"
    source.parent.mkdir()
    source.write_text(json.dumps([{"session_id": "s"}]), encoding="utf-8")

    class Store:
        def load(self, user):
            return [{"session_id": "old"}]

        def replace(self, user, sessions):
            raise AssertionError("must not overwrite")

    with pytest.raises(RuntimeError, match="already contains"):
        migrate(tmp_path, Store())
