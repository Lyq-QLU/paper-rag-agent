"""LangGraph orchestration for the scientific-paper RAG application.

The graph intentionally keeps retrieval and PDF parsing in the existing modules.
Agent nodes coordinate those capabilities, record decisions, and verify outputs.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import sqlite3
import uuid
from typing import Any, Literal, TypedDict
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import httpx

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.paper_loader import load_pdf_documents
from src.rag_pipeline import Chunk, PaperRAG
from src.agent_memory import AgentMemoryStore
from src.llm import call_llm


Route = Literal["search", "ingest", "analysis", "rag", "report", "clarify", "finish"]


class AgentState(TypedDict, total=False):
    query: str
    resolved_query: str
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
    memory: dict
    pending_action: dict
    events: list[dict]
    error: str


ANALYSIS_SIGNALS = (
    "结构化总结", "核心方法", "创新点", "实验设置", "对比算法", "复现建议",
    "多论文对比", "局限", "消融实验", "评价指标", "数据集",
)
SEARCH_SIGNALS = ("搜索论文", "找论文", "检索论文", "查论文", "arxiv", "相关文献")
REPORT_SIGNALS = ("综述", "研究报告", "调研报告", "文献报告", "生成报告")
REFERENCE_SIGNALS = ("这篇", "那篇", "上述论文", "前一篇", "第二篇", "它的", "这个方法")


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
    if any(signal in normalized for signal in REPORT_SIGNALS):
        return "report", "检测到综述或报告生成意图，调用Report Agent。"
    if any(signal in normalized for signal in ANALYSIS_SIGNALS):
        return "analysis", "问题属于科研分析任务，调用Analysis Agent。"
    return "rag", "问题需要检索论文证据，调用RAG Agent。"


def extract_preferences(query: str) -> dict[str, str]:
    """Only persist explicit user preferences; never store the full conversation."""
    preferences: dict[str, str] = {}
    focus = re.search(r"(?:我的研究方向是|我主要研究|我关注)\s*([^。；;\n]{2,80})", query)
    style = re.search(r"(?:以后|后续)?(?:回答)?(?:请)?(?:尽量|偏好)\s*([^。；;\n]{2,60})", query)
    if focus:
        preferences["research_focus"] = focus.group(1).strip()
    if style:
        preferences["answer_preference"] = style.group(1).strip()
    return preferences


def parse_json_object(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def citation_report(answer: str, sources: list[dict]) -> dict:
    available = {
        str(item.get("metadata", {}).get("source", "")).strip()
        for item in sources
        if item.get("metadata", {}).get("source")
    }
    cited: set[str] = set()
    last_explicit = ""
    for raw in re.findall(r"来源[:：]\s*([^，,）)\n]+)", answer or ""):
        citation = raw.strip().strip("`*_ ")
        if citation.lower() in {"同上", "ibid", "ibid."}:
            citation = last_explicit
        elif citation:
            last_explicit = citation
        if citation:
            cited.add(citation)
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
        memory_path: Path | None = None,
    ) -> None:
        self.rag = rag
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        self._checkpointer = SqliteSaver(self._connection)
        self.memory_store = AgentMemoryStore(
            memory_path or checkpoint_path.with_name("agent_memory.sqlite3")
        )
        self.graph = self._build_graph().compile(checkpointer=self._checkpointer)

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("resolve", self._resolve_node)
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("clarify", self._clarify_node)
        builder.add_node("search", self._search_node)
        builder.add_node("ingest", self._ingest_node)
        builder.add_node("analysis", self._analysis_node)
        builder.add_node("rag", self._rag_node)
        builder.add_node("report", self._report_node)
        builder.add_node("verify", self._verify_node)
        builder.add_edge(START, "resolve")
        builder.add_conditional_edges(
            "resolve",
            lambda state: "clarify" if state.get("route") == "clarify" else "supervisor",
            {"clarify": "clarify", "supervisor": "supervisor"},
        )
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state.get("route", "finish"),
            {"search": "search", "ingest": "ingest", "analysis": "analysis", "rag": "rag", "report": "report", "finish": END},
        )
        builder.add_edge("clarify", END)
        builder.add_edge("search", END)
        builder.add_edge("ingest", END)
        builder.add_edge("analysis", "verify")
        builder.add_edge("rag", "verify")
        builder.add_edge("report", "verify")
        builder.add_edge("verify", END)
        return builder

    def _resolve_node(self, state: AgentState) -> dict:
        query = " ".join(state.get("query", "").split())
        user_id = state.get("user_id", "anonymous")
        preferences = extract_preferences(query)
        profile = self.memory_store.remember(user_id, preferences) if preferences else self.memory_store.get_profile(user_id)
        history = state.get("history", [])
        if any(signal in query for signal in REFERENCE_SIGNALS) and not history:
            return {
                "resolved_query": query,
                "route": "clarify",
                "memory": profile,
                "events": append_event(state, "resolve", "检测到缺少上下文的论文指代，转入澄清。"),
            }
        resolved = query
        if history and any(signal in query for signal in REFERENCE_SIGNALS):
            last_question = str(history[-1].get("question", ""))
            resolved = f"上下文：上一轮问题为“{last_question}”。当前问题：{query}"
        return {
            "resolved_query": resolved,
            "memory": profile,
            "events": append_event(state, "resolve", "已完成指代与用户偏好解析。"),
        }

    def _clarify_node(self, state: AgentState) -> dict:
        message = "你提到了“这篇/那篇论文”，但当前线程没有可定位的论文。请提供论文标题、arXiv ID，或先上传PDF。"
        return {"answer": message, "events": append_event(state, "clarify", message)}

    def _supervisor_node(self, state: AgentState) -> dict:
        route, reason = select_route(state.get("resolved_query") or state.get("query", ""), state.get("pdf_paths"))
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
        query = (state.get("resolved_query") or state.get("query", "")).strip()
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
                pdf_url = ""
                for link in entry.findall("atom:link", namespace):
                    if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                        pdf_url = link.attrib.get("href", "")
                        break
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
                        "pdf_url": pdf_url,
                        "summary": summary[:360],
                    })
            if not papers:
                answer = f"arXiv未找到与“{query}”匹配的论文，请尝试更具体的英文关键词。"
            else:
                lines = [f"找到 {len(papers)} 篇相关论文。回复待入库的序号（如 1,3）或选择取消："]
                for index, paper in enumerate(papers, start=1):
                    author_text = ", ".join(paper["authors"][:3])
                    lines.append(
                        f"{index}. {paper['title']}\n"
                        f"   {paper['published']} | {author_text} | {paper['url']}"
                    )
                answer = "\n\n".join(lines)
            pending = {}
            if papers:
                action_id = uuid.uuid4().hex
                pending = self.memory_store.create_pending(
                    action_id=action_id,
                    user_id=state.get("user_id", "anonymous"),
                    thread_id=state.get("session_id", "default"),
                    action_type="ingest_search_results",
                    payload={"query": query, "papers": papers},
                )
                answer += f"\n\n确认操作ID：`{action_id}`（30分钟内有效）"
            return {
                "answer": answer,
                "pending_action": pending,
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
        query = state.get("resolved_query") or state.get("query", "")
        preferences = state.get("memory", {}).get("preferences", {})
        if preferences:
            query = f"用户长期偏好：{preferences}\n{query}"
        answer, chunks = self.rag.answer(
            query,
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

    def _report_node(self, state: AgentState) -> dict:
        original = state.get("resolved_query") or state.get("query", "")
        report_query = (
            "请生成结构化科研综述报告，至少包含：研究问题、方法分类、关键论文或证据、"
            "实验结论、局限性、可复现建议和未来方向。每个关键结论标注来源与页码。\n"
            f"用户需求：{original}"
        )
        enriched = dict(state)
        enriched["resolved_query"] = report_query
        return self._answer_with_rag(enriched, "report")

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
        semantic = None
        if sources and answer and not state.get("error"):
            evidence = "\n\n".join(
                f"[{index}] {item.get('text', '')[:1200]}"
                for index, item in enumerate(sources[:6], start=1)
            )
            prompt = (
                "你是科研回答事实核验器。依据证据判断回答中的主要事实是否得到支持。"
                "只输出JSON，字段为 grounded(bool), sufficient_context(bool), confidence(0到1), unsupported_claims(list)。\n"
                f"回答：\n{answer}\n\n证据：\n{evidence}"
            )
            try:
                raw = call_llm(prompt, user_id=state.get("user_id"))
                if raw:
                    verdict = parse_json_object(raw)
                    semantic = {"status": "available", "verdict": verdict, "raw": raw}
                    if verdict and verdict.get("grounded") is False:
                        status = "warning"
                        message = "引用格式可映射，但语义核验发现可能缺少证据支持的主张。"
                else:
                    semantic = {"status": "unavailable", "reason": "llm_not_configured"}
            except Exception as exc:
                semantic = {"status": "unavailable", "reason": str(exc)}
        verification = {"status": status, "message": message, **report, "semantic_check": semantic}
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
            "pending_action": {},
            "error": "",
        }
        config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
        return self.graph.invoke(initial, config=config)

    def resume_approval(
        self, action_id: str, selections: list[int], *, index_dir: str
    ) -> AgentState:
        pending = self.memory_store.get_pending(action_id)
        if pending is None:
            raise KeyError(f"approval not found: {action_id}")
        if pending["status"] != "pending":
            raise ValueError(f"approval is {pending['status']}")
        papers = pending["payload"].get("papers", [])
        selected = sorted({int(value) for value in selections if 1 <= int(value) <= len(papers)})
        if not selected:
            resolved = self.memory_store.resolve_pending(
                action_id, {"selected": [], "cancelled": True}
            )
            return {
                "answer": "已取消本次论文入库。", "events": [{"node": "ingest", "message": "用户取消入库。"}],
                "sources": [], "verification": {}, "pending_action": resolved, "error": "",
            }
        download_dir = Path(index_dir).parent / "search_downloads" / action_id
        download_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for number in selected:
            paper = papers[number - 1]
            url = str(paper.get("pdf_url") or "")
            if not url.startswith("https://"):
                continue
            response = httpx.get(url, timeout=45.0, follow_redirects=True)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                continue
            safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", paper.get("title", "paper"))[:80]
            path = download_dir / f"{number:02d}_{safe_name}.pdf"
            path.write_bytes(response.content)
            paths.append(str(path))
        resolved = self.memory_store.resolve_pending(
            action_id, {"selected": selected, "downloaded": len(paths)}
        )
        if not paths:
            return {
                "answer": "已确认选择，但PDF下载失败或来源不符合安全校验。", "events": [],
                "sources": [], "verification": {}, "pending_action": resolved,
                "error": "download_failed",
            }
        result = self.invoke(
            "确认搜索结果并入库", user_id=pending["user_id"], session_id=pending["thread_id"],
            pdf_paths=paths, index_dir=index_dir,
        )
        result["pending_action"] = resolved
        return result

    def close(self) -> None:
        self.memory_store.close()
        self._connection.close()


def result_sources(result: AgentState) -> list[Chunk]:
    return [deserialize_chunk(item) for item in result.get("sources", [])]
