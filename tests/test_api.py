from fastapi.testclient import TestClient

from api import app
import src.session_manager as session_manager


def test_health_and_profile_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(session_manager, "DATA_ROOT", tmp_path / "users")
    # api imported helpers still resolve DATA_ROOT through the module globals.
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    profile = client.get("/api/user-profile", params={"user_id": "api-user"}).json()
    assert profile["enabled"] is True
    updated = client.patch(
        "/api/user-profile/settings", params={"user_id": "api-user"},
        json={"enabled": False, "ttl_days": 30},
    ).json()
    assert updated["enabled"] is False
    assert updated["ttl_days"] == 30


def test_upload_openapi_exposes_binary_file_picker():
    schema = app.openapi()
    body = schema["components"]["schemas"]["Body_upload_pdf_api_upload_pdf_post"]
    assert body["properties"]["files"]["items"]["format"] == "binary"


def test_api_chat_loads_and_persists_server_history(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(session_manager, "DATA_ROOT", tmp_path / "users")
    histories = []
    class Workflow:
        def invoke(self, query, **kwargs):
            histories.append(kwargs["history"])
            return {"answer": "answer", "sources": []}
        def close(self): pass
    monkeypatch.setattr(api, "workflow_for", lambda *args: Workflow())
    client = TestClient(app)
    for message in ("first", "second"):
        response = client.post("/api/chat", json={"message": message, "user_id": "u", "thread_id": "s"})
        assert response.status_code == 200
    assert histories[0] == []
    assert histories[1][0]["question"] == "first"
    assert len(session_manager.load_session_history("u", "s")) == 2
    assert client.get("/api/threads", params={"user_id": "u"}).json()[0]["question_count"] == 2


def test_stream_delivers_progress_before_workflow_finishes(tmp_path, monkeypatch):
    import anyio
    import json
    import api
    monkeypatch.setattr(session_manager, "DATA_ROOT", tmp_path / "users")
    progress_seen = []
    closed = []
    class Workflow:
        def invoke(self, query, **kwargs):
            assert kwargs["stream"] is True
            yield {"events": [{"node": "resolve", "message": "resolved"}]}
            assert progress_seen, "progress must reach consumer before next node executes"
            yield {"events": [{"node": "resolve", "message": "resolved"},
                              {"node": "rag", "message": "retrieved"}], "answer": "done"}
        def close(self): closed.append(True)
    monkeypatch.setattr(api, "workflow_for", lambda *args: Workflow())
    response = api.chat_stream(api.ChatRequest(message="question", user_id="u", thread_id="s"))
    async def consume():
        results = []
        async for raw in response.body_iterator:
            event = json.loads(raw.removeprefix("data: ").strip())
            if event["type"] == "progress": progress_seen.append(event)
            results.append(event)
        return results
    results = anyio.run(consume)
    assert [event["node"] for event in progress_seen] == ["resolve", "rag"]
    assert results[-1]["answer"] == "done"
    assert closed == [True]
    assert len(session_manager.load_session_history("u", "s")) == 1


def test_api_uploads_append_and_are_visible_in_paper_list(tmp_path, monkeypatch):
    import src.agent_workflow as workflows
    monkeypatch.setattr(session_manager, "DATA_ROOT", tmp_path / "users")
    seen = []
    class Index:
        chunks = [1]
        def __init__(self, **kwargs): pass
        def build_index(self, documents): seen.append(documents)
        def save(self, directory): pass
    monkeypatch.setattr(workflows, "PaperRAG", Index)
    monkeypatch.setattr(workflows, "load_pdf_documents", lambda paths: paths)
    client = TestClient(app)
    for data in (b"%PDF A", b"%PDF B"):
        response = client.post("/api/upload-pdf", data={"user_id": "u", "thread_id": "s"},
                               files=[("files", ("same.pdf", data, "application/pdf"))])
        assert response.status_code == 200
        assert response.json()["error"] == ""
    assert [len(paths) for paths in seen] == [1, 2]
    assert len(client.get("/api/papers", params={"user_id": "u", "thread_id": "s"}).json()) == 2
