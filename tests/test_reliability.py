from pathlib import Path

import pytest

import src.session_manager as sessions
import src.agent_workflow as workflows
from src.agent_workflow import PaperAgentWorkflow, citation_report
from src.rag_pipeline import Chunk, PaperRAG
from src.agent_memory import AgentMemoryStore


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "DATA_ROOT", tmp_path / "users")
    monkeypatch.setattr(workflows, "call_llm", lambda *args, **kwargs: None)


def test_document_store_deduplicates_and_preserves_same_name():
    a = sessions.register_pdf(b"%PDF A", "same.pdf", "u", "s")
    assert sessions.register_pdf(b"%PDF A", "renamed.pdf", "u", "s") == a
    b = sessions.register_pdf(b"%PDF B", "same.pdf", "u", "s")
    assert a != b
    assert len(sessions.list_uploaded_pdfs("u", "s")) == 2
    assert sessions.list_uploaded_pdfs("other", "s") == []


def test_legacy_documents_imported_once():
    root = sessions.get_session_dir("u", "s")
    for folder in ("api_uploads", "search_downloads/action"):
        path = root / folder / "old.pdf"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"%PDF legacy")
    assert len(sessions.get_uploaded_pdf_paths("u", "s")) == 1
    assert len(sessions.get_uploaded_pdf_paths("u", "s")) == 1


def test_ingest_append_keeps_previous_documents(tmp_path, monkeypatch):
    seen = []
    class Index:
        chunks = [1]
        def __init__(self, **kwargs): pass
        def build_index(self, documents): seen.append(documents)
        def save(self, directory): pass
    monkeypatch.setattr(workflows, "PaperRAG", Index)
    monkeypatch.setattr(workflows, "load_pdf_documents", lambda paths: paths)
    workflow = PaperAgentWorkflow(None, tmp_path / "checkpoint.sqlite")
    try:
        for name in ("a", "b"):
            path = tmp_path / f"{name}.pdf"
            path.write_bytes(f"%PDF {name}".encode())
            result = workflow.invoke("入库", user_id="u", session_id="s", pdf_paths=[str(path)],
                                     index_dir=str(tmp_path / "index"))
            assert not result["error"]
        assert [len(paths) for paths in seen] == [1, 2]
    finally:
        workflow.close()


def test_disabled_memory_not_injected(tmp_path):
    class RAG:
        def answer(self, question, **kwargs):
            assert "secret-focus" not in question
            return "当前资料不足", []
    workflow = PaperAgentWorkflow(RAG(), tmp_path / "checkpoint.sqlite")
    try:
        workflow.memory_store.remember("u", {"research_focus": "secret-focus"})
        workflow.memory_store.update_settings("u", enabled=False)
        workflow.invoke("方法是什么", user_id="u", session_id="s")
    finally:
        workflow.close()


def test_citations_require_exact_name_and_supported_pages():
    sources = [{"metadata": {"source": "long-paper.pdf", "page_start": 3, "page_end": 4}}]
    for answer in ("（来源：paper.pdf，第 3 页）", "（来源：long-paper.pdf，第 5 页）",
                   "（来源：long-paper.pdf）"):
        assert citation_report(answer, sources)["citation_match_rate"] == 0
    assert citation_report("（来源：long-paper.pdf，第 3-4 页）", sources)["citation_match_rate"] == 1


def test_approval_retry_and_replay(tmp_path, monkeypatch):
    class Gateway:
        calls = 0
        def call_tool(self, name, arguments):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            path = tmp_path / "paper.pdf"
            path.write_bytes(b"%PDF test")
            return {"path": str(path)}
    gateway = Gateway()
    workflow = PaperAgentWorkflow(object(), tmp_path / "checkpoint.sqlite", tool_gateway=gateway)
    workflow.memory_store.create_pending(action_id="a", user_id="u", thread_id="s",
        action_type="ingest", payload={"papers": [{"title": "p", "pdf_url": "https://arxiv.org/pdf/123"}]})
    monkeypatch.setattr(workflow, "invoke", lambda *args, **kwargs: {"answer": "done", "error": ""})
    try:
        with pytest.raises(PermissionError):
            workflow.resume_approval("a", [1], index_dir=str(tmp_path), user_id="u", session_id="wrong")
        with pytest.raises(RuntimeError):
            workflow.resume_approval("a", [1], index_dir=str(tmp_path), user_id="u", session_id="s")
        assert workflow.memory_store.get_pending("a")["status"] == "failed"
        first = workflow.resume_approval("a", [1], index_dir=str(tmp_path), user_id="u", session_id="s")
        second = workflow.resume_approval("a", [1], index_dir=str(tmp_path), user_id="u", session_id="s")
        assert first["answer"] == second["answer"] == "done"
        assert gateway.calls == 2
    finally:
        workflow.close()


def test_pending_claim_has_single_owner(tmp_path):
    first = AgentMemoryStore(tmp_path / "memory.sqlite")
    second = AgentMemoryStore(tmp_path / "memory.sqlite")
    try:
        first.create_pending(action_id="a", user_id="u", thread_id="s", action_type="ingest", payload={})
        assert first.claim_pending("a")
        assert second.claim_pending("a") is None
    finally:
        first.close()
        second.close()


def test_index_generation_does_not_publish_failed_save(tmp_path, monkeypatch):
    import numpy as np
    rag = PaperRAG()
    monkeypatch.setattr(rag, "_embed", lambda texts: np.ones((len(texts), 2), dtype="float32"))
    rag.build_index_from_chunks([Chunk("old", {"source": "a.pdf", "page": 1})])
    rag.save(tmp_path)
    old = (tmp_path / "CURRENT").read_text()
    def fail(*args): raise OSError("disk full")
    monkeypatch.setattr("src.rag_pipeline.write_faiss_index", fail)
    with pytest.raises(OSError): rag.save(tmp_path)
    assert (tmp_path / "CURRENT").read_text() == old
    loaded = PaperRAG()
    assert loaded.load(tmp_path)
    assert loaded.chunks[0].text == "old"


def test_revision_is_bounded(tmp_path, monkeypatch):
    class RAG:
        calls = 0
        def answer(self, question, **kwargs):
            self.calls += 1
            return "结论（来源：p.pdf，第 1 页）", [Chunk("evidence", {"source": "p.pdf", "page": 1})]
    monkeypatch.setattr(workflows, "call_llm", lambda *args, **kwargs: '{"grounded":false,"sufficient_context":false}')
    from src.config import RAGConfig
    monkeypatch.setattr("src.config.get_config", lambda **kwargs: RAGConfig(openai_api_key="fake"))
    rag = RAG()
    workflow = PaperAgentWorkflow(rag, tmp_path / "checkpoint.sqlite")
    try:
        result = workflow.invoke("方法是什么", user_id="u", session_id="s")
        assert rag.calls == 2
        assert result["revision_count"] == 1
        assert "资料不足" in result["answer"]
    finally:
        workflow.close()


def test_failed_ingest_keeps_approval_retryable(tmp_path, monkeypatch):
    class Gateway:
        def call_tool(self, *args):
            path = tmp_path / "p.pdf"
            path.write_bytes(b"%PDF test")
            return {"path": str(path)}
    workflow = PaperAgentWorkflow(object(), tmp_path / "checkpoint.sqlite", tool_gateway=Gateway())
    workflow.memory_store.create_pending(action_id="a", user_id="u", thread_id="s",
        action_type="ingest", payload={"papers": [{"title": "p"}]})
    monkeypatch.setattr(workflow, "invoke", lambda *args, **kwargs: {"error": "index failed"})
    try:
        with pytest.raises(RuntimeError, match="index failed"):
            workflow.resume_approval("a", [1], index_dir=str(tmp_path / "index"))
        assert workflow.memory_store.get_pending("a")["status"] == "failed"
    finally:
        workflow.close()


def test_crashed_approval_can_recover_after_lock_release(tmp_path):
    workflow = PaperAgentWorkflow(object(), tmp_path / "checkpoint.sqlite")
    try:
        workflow.memory_store.create_pending(action_id="a", user_id="u", thread_id="s",
            action_type="ingest", payload={"papers": []})
        assert workflow.memory_store.claim_pending("a")
        result = workflow.resume_approval("a", [], index_dir=str(tmp_path))
        assert result["pending_action"]["status"] == "resolved"
    finally:
        workflow.close()


def test_live_approval_cannot_be_recovered_by_second_request(tmp_path):
    import hashlib
    from filelock import FileLock
    workflow = PaperAgentWorkflow(object(), tmp_path / "checkpoint.sqlite")
    try:
        workflow.memory_store.create_pending(action_id="a", user_id="u", thread_id="s",
            action_type="ingest", payload={"papers": []})
        workflow.memory_store.claim_pending("a")
        lock_dir = tmp_path / "approval_locks"
        lock_dir.mkdir()
        with FileLock(str(lock_dir / (hashlib.sha256(b"a").hexdigest() + ".lock"))):
            with pytest.raises(ValueError, match="already running"):
                workflow.resume_approval("a", [], index_dir=str(tmp_path))
        assert workflow.memory_store.get_pending("a")["status"] == "running"
    finally:
        workflow.close()


def test_clarification_does_not_trap_next_turn(tmp_path):
    class RAG:
        def answer(self, *args, **kwargs): return "当前资料不足", []
    workflow = PaperAgentWorkflow(RAG(), tmp_path / "checkpoint.sqlite")
    try:
        assert workflow.invoke("这篇论文有什么创新？", user_id="u", session_id="s")["route"] == "clarify"
        result = workflow.invoke("论文使用了什么方法？", user_id="u", session_id="s")
        assert result["route"] == "rag"
        assert result["clarification_required"] is False
    finally:
        workflow.close()


def test_no_evidence_skips_answer_model(monkeypatch):
    rag = PaperRAG()
    monkeypatch.setattr(rag, "retrieve", lambda *args, **kwargs: [])
    def unexpected(*args, **kwargs): raise AssertionError("must not generate without evidence")
    monkeypatch.setattr("src.rag_pipeline.call_llm", unexpected)
    answer, sources = rag.answer("question")
    assert "资料不足" in answer
    assert sources == []


def test_evidence_id_binds_citation_to_specific_chunk():
    sources = [{"metadata": {"source": "p.pdf", "page": 3, "chunk_id": "abc123"}},
               {"metadata": {"source": "p.pdf", "page": 7, "chunk_id": "def456"}}]
    assert citation_report("（来源：p.pdf，第 3 页，证据ID：abc123）", sources)["citation_match_rate"] == 1
    for answer in ("（来源：p.pdf，第 3 页，证据ID：def456）", "（来源：p.pdf，第 3 页）",
                   "（来源：p.pdf，第 3 页，证据ID：invented）"):
        assert citation_report(answer, sources)["citation_match_rate"] == 0


def test_real_pdf_append_round_trip_without_model_download(tmp_path, monkeypatch):
    import fitz
    import numpy as np
    monkeypatch.setattr(PaperRAG, "_embed", lambda self, texts: np.ones((len(texts), 3), dtype="float32"))
    workflow = PaperAgentWorkflow(None, tmp_path / "checkpoint.sqlite")
    index_dir = sessions.get_index_dir("u", "s")
    try:
        for number in (1, 2):
            path = tmp_path / f"paper{number}.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), f"2 Methodology\nThe proposed algorithm number {number} uses hybrid retrieval.")
            pdf.save(path)
            pdf.close()
            result = workflow.invoke("入库", user_id="u", session_id="s", pdf_paths=[str(path)],
                                     index_dir=str(index_dir))
            assert result["error"] == ""
        restored = PaperRAG()
        assert restored.load(index_dir)
        assert len({chunk.metadata["source"] for chunk in restored.chunks}) == 2
        assert len({chunk.metadata["document_id"] for chunk in restored.chunks}) == 2
        for chunk in restored.chunks:
            assert chunk.metadata["chunk_id"]
            answer = f"（来源：{chunk.metadata['source']}，第 1 页，证据ID：{chunk.metadata['chunk_id']}）"
            assert citation_report(answer, [workflows.serialize_chunk(chunk)])["citation_match_rate"] == 1
    finally:
        workflow.close()
