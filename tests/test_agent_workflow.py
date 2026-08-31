from pathlib import Path

from src.agent_workflow import (
    PaperAgentWorkflow, citation_report, extract_preferences, parse_json_object, select_route,
)
from src.rag_pipeline import Chunk


class FakeRAG:
    def answer(self, question, top_k=4, conversation_history=None):
        source = Chunk(
            text="The proposed method uses hybrid retrieval.",
            metadata={"source": "paper.pdf", "page": 3, "section": "method"},
        )
        return "该方法采用混合检索（来源：paper.pdf，第 3 页）。", [source]


def test_supervisor_routes_analysis_and_rag():
    assert select_route("请总结这篇论文的创新点")[0] == "analysis"
    assert select_route("作者使用了什么模型？")[0] == "rag"
    assert select_route("帮我搜索论文 agentic rag")[0] == "search"
    assert select_route("", ["paper.pdf"])[0] == "ingest"
    assert select_route("生成一份文献综述")[0] == "report"


def test_extract_explicit_preferences_only():
    assert extract_preferences("我的研究方向是多目标路径规划。普通问题") == {
        "research_focus": "多目标路径规划"
    }
    assert extract_preferences("请解释方法") == {}
    assert parse_json_object('```json\n{"grounded": true}\n```') == {"grounded": True}


def test_missing_reference_routes_to_clarify(tmp_path: Path):
    workflow = PaperAgentWorkflow(FakeRAG(), tmp_path / "checkpoints.sqlite")
    result = workflow.invoke("这篇论文有什么创新？", user_id="u", session_id="s")
    workflow.close()
    assert "提供论文标题" in result["answer"]


def test_cancelled_approval_returns_resolved_snapshot(tmp_path: Path):
    workflow = PaperAgentWorkflow(FakeRAG(), tmp_path / "checkpoints.sqlite")
    workflow.memory_store.create_pending(
        action_id="cancel-1", user_id="u", thread_id="s",
        action_type="ingest_search_results", payload={"papers": [{"title": "p"}]},
    )
    result = workflow.resume_approval("cancel-1", [], index_dir=str(tmp_path / "index"))
    workflow.close()
    assert result["pending_action"]["status"] == "resolved"
    assert result["pending_action"]["payload"]["decision"]["cancelled"] is True


def test_citation_report_matches_retrieved_source():
    report = citation_report(
        "结论（来源：paper.pdf，第 3 页）。",
        [{"text": "evidence", "metadata": {"source": "paper.pdf"}}],
    )
    assert report["citation_coverage"] == 1.0
    assert report["matched_sources"] == ["paper.pdf"]


def test_graph_runs_rag_and_verify(tmp_path: Path):
    workflow = PaperAgentWorkflow(FakeRAG(), tmp_path / "checkpoints.sqlite")
    try:
        result = workflow.invoke(
            "论文使用了什么方法？",
            user_id="tester",
            session_id="session-1",
            top_k=4,
        )
    finally:
        workflow.close()

    assert result["route"] == "rag"
    assert result["verification"]["status"] == "passed"
    assert [event["node"] for event in result["events"]] == [
        "resolve", "supervisor", "rag", "verify"
    ]
