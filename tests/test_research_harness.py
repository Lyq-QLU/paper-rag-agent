import json
from types import SimpleNamespace

import pytest

import src.agent_workflow as workflows
import src.research_harness as harness
import src.session_manager as sessions
from src.rag_pipeline import Chunk
from src.evaluation import chunk_id


CHUNK = Chunk("The method combines dense retrieval and BM25.", {"source": "a.pdf", "page": 2})
ANSWER = f"该方法采用混合检索（来源：a.pdf，第 2 页，证据ID：{chunk_id(CHUNK)}）。"


def decision(action="tool", tool="retrieve_evidence", **args):
    return {"action": action, "tool": tool if action == "tool" else None,
            "arguments": args, "requirements": ["解释方法"]}


def verdict(done=True):
    return {"grounded": done, "goal_satisfied": done,
            "requirements": [{"requirement": "解释方法", "done": done, "reason": "evidence"}]}


class RAG:
    def __init__(self): self.calls = []
    def retrieve_by_mode(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [CHUNK]


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "DATA_ROOT", tmp_path / "users")
    monkeypatch.setattr(workflows, "get_config", lambda **kwargs: SimpleNamespace(openai_api_key="fake"))
    return tmp_path


def script(monkeypatch, outputs):
    remaining = iter(outputs)
    calls = []
    def model(prompt, **kwargs):
        calls.append((prompt, kwargs))
        value = next(remaining)
        if isinstance(value, Exception): raise value
        return {"content": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                "input_tokens": 10, "output_tokens": 5}
    monkeypatch.setattr(harness, "call_agent_model", model)
    return calls


def run(workflow, **kwargs):
    return workflow.invoke("解释论文的方法", user_id="u", session_id="s",
                           execution_mode="harness", index_dir="unused", **kwargs)


def test_standalone_harness_has_no_langgraph_runner(environment):
    workflow = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    try:
        assert not hasattr(workflow, "graph")
        assert not hasattr(workflow, "_checkpointer")
    finally:
        workflow.close()


def test_skill_tool_boundary_rejects_mismatched_decision(environment, monkeypatch):
    invalid = decision(query="method", sources=["a.pdf"])
    invalid["skill"] = "paper_discovery"
    calls = script(monkeypatch, [invalid])
    workflow = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    try:
        run_state = harness.new_run("解释方法")
        state = {"harness": run_state, "user_id": "u", "session_id": "s", "events": [],
                 "history": [], "sources": [], "verification": {}, "pending_action": {}}
        result = workflow.harness_runtime.decide(state)
        assert result["harness"]["next"] == "decide"
        assert "selected skill" in result["harness"]["observations"][-1]["error"]
        assert len(calls) == 1
    finally:
        workflow.close()


def test_discovery_skill_routes_crossref_with_filters(environment, monkeypatch):
    class Gateway:
        def __init__(self): self.calls = []
        def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return {"papers": [{"title": "Published Study", "published": "2025-01-01",
                                "doi": "https://doi.org/10.1234/test", "venue": "Test Journal",
                                "venue_kind": "journal", "source": "crossref"}]}
    gateway = Gateway()
    outputs = [
        {**decision(tool="search_papers", query="retrieval", provider="crossref",
                    year_from=2024, year_to=2026, venue_kind="journal"), "skill": "paper_discovery"},
        decision("finish"),
        "找到一篇 DOI 登记的期刊论文。",
        verdict(),
    ]
    script(monkeypatch, outputs)
    workflow = workflows.PaperAgentWorkflow(None, environment / "checkpoint.sqlite", tool_gateway=gateway)
    try:
        result = run(workflow)
        assert result["harness"]["status"] == "completed"
        assert gateway.calls[0][0] == "search_crossref"
        assert gateway.calls[0][1]["venue_kind"] == "journal"
        assert result["harness"]["active_skill"] == "research_synthesis"
        assert result["harness"]["skills_used"] == ["paper_discovery", "research_synthesis"]
    finally:
        workflow.close()


def test_dynamic_retrieval_and_independent_completion(environment, monkeypatch):
    calls = script(monkeypatch, [decision(query="method", sources=["a.pdf"]),
                               decision("finish"), ANSWER, verdict()])
    rag = RAG()
    workflow = workflows.PaperAgentWorkflow(rag, environment / "checkpoint.sqlite")
    try:
        result = run(workflow)
        agent = result["harness"]
        assert agent["status"] == "completed"
        assert agent["usage"]["model_calls"] == 4
        assert agent["usage"]["tool_calls"] == 1
        assert agent["usage"]["input_tokens"] == 40
        assert rag.calls[0][1]["source_filter"] == ["a.pdf"]
        snapshot = json.loads(harness.task_path("u", "s", agent["run_id"]).read_text())
        assert snapshot["harness"]["status"] == "completed"
        assert workflow.resume_task(agent["run_id"], user_id="u", session_id="s", index_dir="unused")["answer"] == ANSWER
        assert len(calls) == 4
    finally:
        workflow.close()


def test_tool_failure_is_observed_then_replanned(environment, monkeypatch):
    calls = script(monkeypatch, [decision(query="first"), decision(query="second"),
                               decision("finish"), ANSWER, verdict()])
    class FlakyRAG(RAG):
        def retrieve_by_mode(self, query, **kwargs):
            if query == "first": raise RuntimeError("temporary retrieval error")
            return super().retrieve_by_mode(query, **kwargs)
    workflow = workflows.PaperAgentWorkflow(FlakyRAG(), environment / "checkpoint.sqlite")
    try:
        result = run(workflow)
        assert result["harness"]["status"] == "completed"
        assert "temporary retrieval error" in calls[1][0]
        assert result["harness"]["usage"]["tool_calls"] == 2
    finally:
        workflow.close()


def test_invalid_tool_and_args_never_execute(environment, monkeypatch):
    script(monkeypatch, [decision(tool="shell", command="delete"),
                        decision(query="method", output_dir="/etc")])
    rag = RAG()
    workflow = workflows.PaperAgentWorkflow(rag, environment / "checkpoint.sqlite")
    try:
        result = run(workflow, budget={"max_steps": 2})
        assert result["harness"]["status"] == "budget_exhausted"
        assert result["harness"]["usage"]["tool_calls"] == 0
        assert not rag.calls
    finally:
        workflow.close()


def test_model_errors_have_bounded_retries(environment, monkeypatch):
    calls = script(monkeypatch, [RuntimeError("temporary"), "invalid JSON"])
    workflow = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    try:
        result = run(workflow, budget={"max_steps": 2})
        assert result["harness"]["status"] == "budget_exhausted"
        assert len(calls) == 2
    finally:
        workflow.close()


def test_completion_failure_returns_to_tools(environment, monkeypatch):
    script(monkeypatch, [decision(query="method"), decision("finish"), ANSWER, verdict(False),
                        decision(query="more evidence"), decision("finish"), ANSWER, verdict()])
    workflow = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    try:
        result = run(workflow)
        assert result["harness"]["status"] == "completed"
        assert result["harness"]["usage"]["tool_calls"] == 2
        assert len([entry for entry in result["harness"]["trace"] if entry["node"] == "harness_check"]) == 2
    finally:
        workflow.close()


def test_wrong_citations_cannot_complete(environment, monkeypatch):
    script(monkeypatch, [decision(query="method"), decision("finish"),
                        "结论（来源：a.pdf，第 99 页）", verdict()])
    workflow = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    try:
        result = run(workflow, budget={"max_steps": 2})
        assert result["harness"]["status"] == "budget_exhausted"
        assert "99" not in result["answer"]
        assert result["verification"]["status"] == "warning"
    finally:
        workflow.close()


def test_clarification_resumes_same_goal_and_budget(environment, monkeypatch):
    script(monkeypatch, [{"action": "clarify", "question": "哪篇论文？", "requirements": ["解释方法"]}])
    workflow = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    try:
        first = run(workflow)
        run_id = first["harness"]["run_id"]
        assert first["harness"]["status"] == "needs_input"
    finally:
        workflow.close()
    calls = script(monkeypatch, [decision(query="method"), decision("finish"), ANSWER, verdict()])
    restored = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    try:
        result = restored.resume_task(run_id, user_id="u", session_id="s", index_dir="unused", message="a.pdf")
        assert result["harness"]["goal"] == "解释论文的方法"
        assert result["harness"]["run_id"] == run_id
        assert result["harness"]["usage"]["model_calls"] == 5
        assert "a.pdf" in calls[0][0]
        assert result["harness"]["status"] == "completed"
    finally:
        restored.close()


def test_expired_budget_dispatches_no_model(environment, monkeypatch):
    calls = script(monkeypatch, [])
    snapshot = harness.new_run("解释方法")
    snapshot["usage"]["elapsed_seconds"] = 180
    workflow = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    try:
        result = run(workflow, harness_resume=snapshot)
        assert result["harness"]["status"] == "budget_exhausted"
        assert calls == []
    finally:
        workflow.close()


def test_approval_resumes_original_task_and_replays_result(environment, monkeypatch):
    class Gateway:
        downloads = 0
        def call_tool(self, name, arguments):
            if name == "search_arxiv":
                return {"papers": [{"title": "A paper", "pdf_url": "https://arxiv.org/pdf/123",
                                     "authors": ["A"], "url": "https://arxiv.org/abs/123"}]}
            self.downloads += 1
            target = environment / "download.pdf"
            target.write_bytes(b"%PDF test")
            return {"path": str(target)}
    gateway = Gateway()
    calls = []
    outputs = iter([decision(tool="search_papers", query="method"),
                    decision(tool="request_ingestion", candidate_ids=["placeholder"])])
    def initial_model(prompt, **kwargs):
        value = next(outputs)
        if value.get("tool") == "request_ingestion":
            context = json.loads(prompt.split("\n", 1)[1])["context"]
            result = json.loads(context["observations"][-1]["result"])
            value["arguments"]["candidate_ids"] = [result["papers"][0]["candidate_id"]]
        calls.append(value)
        return {"content": json.dumps(value)}
    monkeypatch.setattr(harness, "call_agent_model", initial_model)
    workflow = workflows.PaperAgentWorkflow(None, environment / "checkpoint.sqlite", tool_gateway=gateway)
    try:
        first = run(workflow)
        assert first["harness"]["status"] == "awaiting_approval"
        action_id = first["pending_action"]["action_id"]
        with pytest.raises(ValueError, match="审批"):
            workflow.resume_task(first["harness"]["run_id"], user_id="u", session_id="s", index_dir="unused")
        model_calls = script(monkeypatch, [decision(query="method"), decision("finish"), ANSWER, verdict()])
        original_invoke = workflow.invoke
        def invoke(query, **kwargs):
            if kwargs.get("pdf_paths"):
                workflow.rag = RAG()
                return {"answer": "ingested", "error": ""}
            return original_invoke(query, **kwargs)
        monkeypatch.setattr(workflow, "invoke", invoke)
        result = workflow.resume_approval(action_id, [1], user_id="u", session_id="s", index_dir=str(environment / "index"))
        assert result["harness"]["run_id"] == first["harness"]["run_id"]
        assert result["harness"]["status"] == "completed"
        assert result["harness"]["usage"]["model_calls"] == 6
        assert result["harness"]["usage"]["tool_calls"] == 4
        assert gateway.downloads == 1
        assert workflow.resume_approval(action_id, [1], index_dir=str(environment / "index"))["answer"] == ANSWER
        assert len(model_calls) == 4
    finally:
        workflow.close()


def test_cancelled_approval_saves_resumable_task(environment, monkeypatch):
    workflow = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    agent = harness.new_run("解释方法")
    agent["status"] = "awaiting_approval"
    workflow.memory_store.create_pending(action_id="cancel", user_id="u", thread_id="s", action_type="ingest",
        payload={"papers": [], "harness_continuation": agent})
    try:
        result = workflow.resume_approval("cancel", [], user_id="u", session_id="s", index_dir="unused")
        assert result["harness"]["status"] == "needs_input"
        snapshot = json.loads(harness.task_path("u", "s", agent["run_id"]).read_text())
        assert snapshot["harness"]["status"] == "needs_input"
    finally:
        workflow.close()


def test_tool_call_budget_stops_before_dispatch(environment, monkeypatch):
    script(monkeypatch, [decision(query="first"), decision(query="second")])
    rag = RAG()
    workflow = workflows.PaperAgentWorkflow(rag, environment / "checkpoint.sqlite")
    try:
        result = run(workflow, budget={"max_tool_calls": 1})
        assert result["harness"]["status"] == "budget_exhausted"
        assert len(rag.calls) == 1
    finally:
        workflow.close()


def test_task_paths_are_scoped_and_reject_traversal(environment, monkeypatch):
    with pytest.raises(ValueError): harness.task_path("u", "s", "../../bad")
    script(monkeypatch, [{"action": "clarify", "question": "哪篇？"}])
    workflow = workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite")
    try:
        first = run(workflow)
        with pytest.raises(FileNotFoundError):
            workflow.resume_task(first["harness"]["run_id"], user_id="other", session_id="s", index_dir="unused")
    finally:
        workflow.close()


def test_api_exposes_task_state_and_budget_validation(environment, monkeypatch):
    from fastapi.testclient import TestClient
    import api
    monkeypatch.setattr(api, "workflow_for", lambda *args:
        workflows.PaperAgentWorkflow(RAG(), environment / "checkpoint.sqlite"))
    calls = script(monkeypatch, [decision(query="method"), decision("finish"), ANSWER, verdict()])
    client = TestClient(api.app)
    response = client.post('/api/chat', json={"message": "解释方法", "user_id": "u", "thread_id": "s",
                                             "execution_mode": "harness"})
    assert response.status_code == 200
    result = response.json()
    run_id = result["harness"]["run_id"]
    snapshot = client.get(f'/api/tasks/{run_id}', params={"user_id": "u", "thread_id": "s"})
    assert snapshot.json()["harness"]["status"] == "completed"
    assert client.get(f'/api/tasks/{run_id}/report', params={"user_id": "u", "thread_id": "s"}).text == ANSWER
    assert client.get(f'/api/tasks/{run_id}', params={"user_id": "other", "thread_id": "s"}).status_code == 404
    assert client.post(f'/api/tasks/{run_id}/resume', params={"user_id": "u", "thread_id": "s"}, json={}).json()["answer"] == ANSWER
    assert len(calls) == 4
    assert client.post('/api/chat', json={"message": "test", "budget": {"max_steps": 999}}).status_code == 422


def test_source_filter_applies_before_dense_top_k(monkeypatch):
    import numpy as np
    from src.rag_pipeline import PaperRAG
    rag = PaperRAG()
    monkeypatch.setattr(rag, "_embed", lambda texts: np.array([[1.0, 0.0]] * len(texts), dtype='float32'))
    rag.build_index_from_chunks([Chunk("method A", {"source": "a.pdf", "page": 1}),
                                 Chunk("method B", {"source": "b.pdf", "page": 2})])
    hits = rag.retrieve_by_mode("method", top_k=1, mode="dense", source_filter=["a.pdf"])
    assert [hit.metadata["source"] for hit in hits] == ["a.pdf"]
    with pytest.raises(ValueError, match="unknown"):
        rag.retrieve_by_mode("method", source_filter=["missing.pdf"])


def test_metadata_only_answer_cannot_invent_pdf_citations(environment, monkeypatch):
    script(monkeypatch, [decision(tool="list_papers"), decision("finish"),
                        "没有论文（来源：unknown，第 1 页）", verdict()])
    workflow = workflows.PaperAgentWorkflow(None, environment / "checkpoint.sqlite")
    try:
        result = run(workflow, budget={"max_steps": 2})
        assert result["harness"]["status"] == "budget_exhausted"
        assert result["verification"]["citation_match_rate"] == 0
    finally:
        workflow.close()


def test_source_free_inventory_answer_can_complete(environment, monkeypatch):
    calls = script(monkeypatch, [decision(tool="list_papers"), decision("finish"),
                               "list_papers显示当前会话没有已登记论文。", verdict()])
    workflow = workflows.PaperAgentWorkflow(None, environment / "checkpoint.sqlite")
    try:
        result = run(workflow)
        assert result["harness"]["status"] == "completed"
        assert "不要生成论文来源、页码或证据ID引用" in calls[2][0]
    finally:
        workflow.close()
