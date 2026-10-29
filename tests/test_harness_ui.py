from pathlib import Path

from streamlit.testing.v1 import AppTest

import src.session_manager as sessions
from src.agent_memory import AgentMemoryStore


def test_approval_form_survives_page_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "DATA_ROOT", tmp_path / "users")
    sessions.upsert_session_meta("synthetic-ui", "s", title="test")
    store = AgentMemoryStore(sessions.get_user_dir("synthetic-ui") / "agent_memory.sqlite3")
    try:
        store.create_pending(action_id="ui-test", user_id="synthetic-ui", thread_id="s",
                             action_type="ingest_search_results",
                             payload={"query": "synthetic task", "papers": [{"title": "Synthetic paper"}]})
    finally:
        store.close()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    app.text_input[0].set_value("synthetic-ui")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any(button.label == "确认选择并继续任务" for button in app.button)
    assert any(toggle.label == "科研任务模式" for toggle in app.toggle)
    app.run(timeout=30)
    assert not app.exception
    assert any(button.label == "确认选择并继续任务" for button in app.button)
