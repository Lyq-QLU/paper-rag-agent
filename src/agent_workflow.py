"""LangGraph orchestration for the scientific-paper RAG application.

The graph intentionally keeps retrieval and PDF parsing in the existing modules.
Agent nodes coordinate those capabilities, record decisions, and verify outputs.
"""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Any, Literal, TypedDict
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import httpx

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.paper_loader import load_pdf_documents
from src.rag_pipeline import Chunk, PaperRAG


Route = Literal["search", "ingest", "analysis", "rag", "finish"]


class AgentState(TypedDict, total=False):
    query: str
    user_id: str
    session_id: str
    history: list[dict]
    top_k: int
    pdf_paths: list[str]
    index_dir: str
    route: Route
    route_reason: str
    answer: str
    sources: list[dict]
    verification: dict
    events: list[dict]
    error: str


ANALYSIS_SIGNALS = (
    "结构化总结", "核心方法", "创新点", "实验设置", "对比算法", "复现建议",
    "多论文对比", "局限", "消融实验", "评价指标", "数据集",
)
SEARCH_SIGNALS = ("搜索论文", "找论文", "检索论文", "查论文", "arxiv", "相关文献")


def serialize_chunk(chunk: Chunk) -> dict:
    return {"text": chunk.text, "metadata": dict(chunk.metadata)}


def deserialize_chunk(data: dict) -> Chunk:
    return Chunk(text=str(data.get("text", "")), metadata=dict(data.get("metadata", {})))


def append_event(state: AgentState, node: str, message: str) -> list[dict]:
    return [*state.get("events", []), {"node": node, "message": message}]


def select_route(query: str, pdf_paths: list[str] | None = None) -> tuple[Route, str]:
    """Deterministic first-line routing keeps the workflow stable without an API key."""
    if pdf_paths:
        return "ingest", "检测到待处理PDF，进入论文解析与知识库构建。"
    normalized = " ".join((query or "").split())
    if not normalized:
        return "finish", "问题为空，结束本轮。"
    if any(signal in normalized.lower() for signal in SEARCH_SIGNALS):
        return "search", "检测到外部论文发现意图，调用Search Agent。"
    if any(signal in normalized for signal in ANALYSIS_SIGNALS):
        return "analysis", "问题属于科研分析任务，调用Analysis Agent。"
    return "rag", "问题需要检索论文证据，调用RAG Agent。"


def citation_report(answer: str, sources: list[dict]) -> dict:
    available = {
        str(item.get("metadata", {}).get("source", "")).strip()
        for item in sources
        if item.get("metadata", {}).get("source")
    }
    cited = {
        match.strip()
        for match in re.findall(r"来源[:：]\s*([^，,）)\n]+)", answer or "")
    }
    matched = {
        citation
        for citation in cited
        if any(citation in source or source in citation for source in available)
    }
    return {
        "available_sources": sorted(available),
        "cited_sources": sorted(cited),
        "matched_sources": sorted(matched),
        "citation_coverage": round(len(matched) / len(cited), 3) if cited else 0.0,
    }


class PaperAgentWorkflow:
    """Stateful LangGraph facade used by Streamlit and tests."""

    def __init__(
        self,
        rag: PaperRAG | None,
        checkpoint_path: Path,
    ) -> None:
        self.rag = rag
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        self._checkpointer = SqliteSaver(self._connection)
        self.graph = self._build_graph().compile(checkpointer=self._checkpointer)

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("search", self._search_node)
        builder.add_node("ingest", self._ingest_node)
        builder.add_node("analysis", self._analysis_node)
        builder.add_node("rag", self._rag_node)
        builder.add_node("verify", self._verify_node)
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state.get("route", "finish"),
            {"search": "search", "ingest": "ingest", "analysis": "analysis", "rag": "rag", "finish": END},
        )
        builder.add_edge("search", END)
        builder.add_edge("ingest", END)
        builder.add_edge("analysis", "verify")
        builder.add_edge("rag", "verify")
        builder.add_edge("verify", END)
        return builder

    def _supervisor_node(self, state: AgentState) -> dict:
        route, reason = select_route(state.get("query", ""), state.get("pdf_paths"))
        return {
            "route": route,
            "route_reason": reason,
            "events": append_event(state, "supervisor", f"{route}: {reason}"),
        }

    def _ingest_node(self, state: AgentState) -> dict:
        paths = [Path(path) for path in state.get("pdf_paths", [])]
        try:
            documents = load_pdf_documents(paths)
            rag = self.rag or PaperRAG(user_id=state.get("user_id"))
            rag.build_index(documents)
            index_dir = Path(state["index_dir"])
            rag.save(index_dir)
            self.rag = rag
            answer = f"已完成 {len(paths)} 篇论文的解析与知识库构建，共生成 {len(rag.chunks)} 个检索片段。"
            return {
                "answer": answer,
                "error": "",
                "events": append_event(state, "ingest", answer),
            }
        except Exception as exc:
            message = f"知识库构建失败：{exc}"
            return {
                "answer": message,
                "error": str(exc),
                "events": append_event(state, "ingest", message),
            }

    def _search_node(self, state: AgentState) -> dict:
        query = state.get("query", "").strip()
        for signal in SEARCH_SIGNALS:
            query = query.replace(signal, " ")
        query = " ".join(query.split()) or state.get("query", "").strip()
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{quote_plus(query)}&start=0&max_results=5"
            "&sortBy=relevance&sortOrder=descending"
        )
        try:
            response = httpx.get(url, timeout=15.0, follow_redirects=True)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", namespace):
                title = " ".join((entry.findtext("atom:title", default="", namespaces=namespace)).split())
                summary = " ".join((entry.findtext("atom:summary", default="", namespaces=namespace)).split())
                published = entry.findtext("atom:published", default="", namespaces=namespace)[:10]
                paper_url = entry.findtext("atom:id", default="", namespaces=namespace)
                authors = [
                    author.findtext("atom:name", default="", namespaces=namespace)
                    for author in entry.findall("atom:author", namespace)
                ]
                if title:
                    papers.append({
                        "title": title,
                        "authors": authors,
                        "published": published,
                        "url": paper_url,
                        "summary": summary[:360],
                    })
            if not papers:
                answer = f"arXiv未找到与“{query}”匹配的论文，请尝试更具体的英文关键词。"
            else:
                lines = [f"找到 {len(papers)} 篇相关论文："]
                for index, paper in enumerate(papers, start=1):
                    author_text = ", ".join(paper["authors"][:3])
                    lines.append(
                        f"{index}. {paper['title']}\n"
                        f"   {paper['published']} | {author_text} | {paper['url']}"
                    )
                answer = "\n\n".join(lines)
            return {
                "answer": answer,
                "sources": [],
                "error": "",
                "events": append_event(state, "search", f"arXiv返回 {len(papers)} 条结果。"),
            }
        except Exception as exc:
            message = f"外部论文搜索暂不可用：{exc}"
            return {
                "answer": message,
                "sources": [],
                "error": str(exc),
                "events": append_event(state, "search", message),
            }
    def _answer_with_rag(self, state: AgentState, node: str) -> dict:
        if self.rag is None:
            message = "当前会话尚未构建论文知识库，请先上传PDF。"
            return {
                "answer": message,
                "sources": [],
                "error": "knowledge_base_missing",
                "events": append_event(state, node, message),
            }
        answer, chunks = self.rag.answer(
            state.get("query", ""),
            top_k=int(state.get("top_k", 4)),
            conversation_history=state.get("history", []),
        )
        return {
            "answer": answer,
            "sources": [serialize_chunk(chunk) for chunk in chunks],
            "error": "",
            "events": append_event(state, node, f"检索到 {len(chunks)} 个证据片段。"),
        }

    def _rag_node(self, state: AgentState) -> dict:
        return self._answer_with_rag(state, "rag")

    def _analysis_node(self, state: AgentState) -> dict:
        return self._answer_with_rag(state, "analysis")

    def _verify_node(self, state: AgentState) -> dict:
        sources = state.get("sources", [])
        answer = state.get("answer", "")
        report = citation_report(answer, sources)
        if state.get("error"):
            status = "failed"
            message = "上游节点失败，跳过证据核验。"
        elif not sources:
            status = "insufficient_context"
            message = "没有检索到可核验的论文证据。"
        elif report["cited_sources"] and report["citation_coverage"] < 1:
            status = "warning"
            message = "部分引用未能映射到本轮检索来源。"
        elif not report["cited_sources"]:
            status = "warning"
            message = "回答已基于检索片段生成，但未包含可机器核对的来源标注。"
        else:
            status = "passed"
            message = "回答引用均可映射到本轮检索来源。"
        verification = {"status": status, "message": message, **report}
        return {
            "verification": verification,
            "events": append_event(state, "verify", f"{status}: {message}"),
        }

    def invoke(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str,
        history: list[dict] | None = None,
        top_k: int = 4,
        pdf_paths: list[str] | None = None,
        index_dir: str = "",
    ) -> AgentState:
        initial: AgentState = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "history": history or [],
            "top_k": top_k,
            "pdf_paths": pdf_paths or [],
            "index_dir": index_dir,
            "events": [],
            "sources": [],
            "verification": {},
            "error": "",
        }
        config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
        return self.graph.invoke(initial, config=config)

    def close(self) -> None:
        self._connection.close()


def result_sources(result: AgentState) -> list[Chunk]:
    return [deserialize_chunk(item) for item in result.get("sources", [])]
