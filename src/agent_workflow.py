"""Deterministic application workflow and standalone research harness facade."""

from __future__ import annotations

from pathlib import Path
import json
import hashlib
from filelock import FileLock, Timeout
import re
import uuid
import copy
import time
from typing import Any, Iterator, Literal, TypedDict


from src.session_manager import get_uploaded_pdf_paths, register_pdf, sanitize_identifier, upsert_session_meta
from src.paper_loader import load_pdf_documents
from src.rag_pipeline import Chunk, PaperRAG
from src.agent_memory import AgentMemoryStore
from src.llm import call_llm
from src.mcp_client import MCPPaperToolGateway
from src.research_harness import ResearchHarness, HarnessBudget, new_run, task_path
from src.config import get_config


Route = Literal["search", "ingest", "analysis", "rag", "report", "clarify", "finish", "harness"]


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
    clarification_required: bool
    clarification_type: str
    missing_fields: list[str]
    clarification_question: str
    clarification_context: dict
    events: list[dict]
    error: str
    revision_count: int
    revision_feedback: str
    execution_mode: str
    harness: dict


ANALYSIS_SIGNALS = (
    "结构化总结", "核心方法", "创新点", "实验设置", "对比算法", "复现建议",
    "多论文对比", "局限", "消融实验", "评价指标", "数据集",
)
SEARCH_SIGNALS = ("搜索论文", "找论文", "检索论文", "查论文", "arxiv", "相关文献")
REPORT_SIGNALS = ("综述", "研究报告", "调研报告", "文献报告", "生成报告")
REFERENCE_SIGNALS = (
    "这篇", "那篇", "上述论文", "前一篇", "第二篇", "这两篇", "它的", "这个方法",
)
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
METRIC_SIGNALS = (
    "成本", "距离", "运行时间", "耗时", "gap", "最优性", "满意度", "均衡",
    "准确率", "召回率", "precision", "recall", "f1", "目标值", "指标",
)
TASK_GOAL_SIGNALS = ("处理一下", "分析一下这些资料", "看看这些资料", "看一下这些资料")
CLARIFICATION_MESSAGES = {
    "paper_identity": "请提供论文标题、作者、DOI、ArXiv ID，或先上传PDF。",
    "task_goal": "请说明希望执行的任务，例如概括论文、提取模型与参数、分析创新点、整理实验结果、给出复现步骤或进行论文比较。",
    "comparison_target": "请说明需要比较的论文或算法，例如给出两篇论文标题，或指定PPO、ALNS、NSGA-II等比较对象。",
    "data_scope": "请说明数据范围，例如起止年份、论文数量，以及使用当前知识库、ArXiv还是OpenAlex。",
    "evaluation_metric": "请说明判断优劣的指标，例如总成本、运行时间、最优性Gap、服务满意度或工作量均衡。",
    "figure_or_table_identity": "请提供图表编号或页码，例如Figure 3、第8页Table 2，也可以直接上传对应截图。",
}


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


def detect_clarification_needs(query: str, history: list[dict] | None = None) -> list[str]:
    """Detect only missing information that can materially change the result."""
    normalized = " ".join((query or "").split())
    lower = normalized.lower()
    history = history or []
    missing: list[str] = []

    if any(signal in normalized for signal in REFERENCE_SIGNALS) and not history:
        missing.append("paper_identity")

    if any(signal in normalized for signal in TASK_GOAL_SIGNALS):
        has_specific_goal = any(
            signal in normalized
            for signal in (*ANALYSIS_SIGNALS, *SEARCH_SIGNALS, *REPORT_SIGNALS)
        ) or any(keyword in lower for keyword in ["模型", "算法", "参数", "总结", "复现", "实验"])
        if not has_specific_goal:
            missing.append("task_goal")

    ambiguous_comparison = any(
        signal in normalized
        for signal in ["比较一下", "这两篇", "两种算法", "哪个更好", "谁更好", "相比怎么样"]
    )
    explicit_comparison = bool(re.search(r"(?:与|和|跟|对比)\s*[^，。；;]{2,60}", normalized))
    if ambiguous_comparison and not explicit_comparison:
        missing.append("comparison_target")

    if any(signal in normalized for signal in ["近几年", "近年来", "近期", "最新"]):
        if not YEAR_PATTERN.search(normalized) and not re.search(r"最近\s*\d+\s*年", normalized):
            missing.append("data_scope")

    if any(signal in normalized for signal in ["哪个更好", "谁更好", "效果最好", "最优方法"]):
        if not any(signal in lower for signal in METRIC_SIGNALS):
            missing.append("evaluation_metric")

    visual_reference = any(
        signal in normalized
        for signal in ["图中", "图里", "这个图", "这张图", "表中", "表里", "这个表", "这张表"]
    )
    has_visual_identity = bool(
        re.search(r"(?:figure|fig\.?|table|图|表)\s*[A-Za-z]?\d+", normalized, re.I)
        or re.search(r"第\s*\d+\s*页", normalized)
    )
    if visual_reference and not has_visual_identity:
        missing.append("figure_or_table_identity")

    return list(dict.fromkeys(missing))


def build_clarification_question(missing_fields: list[str]) -> str:
    messages = [CLARIFICATION_MESSAGES[field] for field in missing_fields if field in CLARIFICATION_MESSAGES]
    if not messages:
        return "请补充完成当前任务所需的信息。"
    if len(messages) == 1:
        return messages[0]
    lines = ["为了避免检索或比较范围错误，请补充以下信息："]
    lines.extend(f"{index}. {message}" for index, message in enumerate(messages, start=1))
    return "\n".join(lines)


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
    available = {str(item.get("metadata", {}).get("source", "")).strip()
                 for item in sources if item.get("metadata", {}).get("source")}
    citations = []
    last_explicit = ""
    for match in re.finditer(r"来源[:：]\s*([^，,）)\n]+)([^）)\n]*)", answer or ""):
        name = match[1].strip().strip("`*_ ")
        if name.lower() in {"同上", "ibid", "ibid."}:
            name = last_explicit
        else:
            last_explicit = name
        pages = re.search(r"第\s*(\d+)(?:\s*[-–—至]\s*(\d+))?\s*页", match[2])
        start = int(pages[1]) if pages else None
        end = int(pages[2] or pages[1]) if pages else None
        evidence_match = re.search(r"证据ID[:：]\s*([a-zA-Z0-9_-]+)", match[2])
        evidence_id = evidence_match[1] if evidence_match else None
        requires_id = any(item.get("metadata", {}).get("chunk_id") for item in sources)
        valid = False
        if (not requires_id or evidence_id) and name in available and start is not None and end >= start:
            supported_pages = set()
            for item in sources:
                metadata = item.get("metadata", {})
                if metadata.get("source") != name or (evidence_id and metadata.get("chunk_id") != evidence_id):
                    continue
                try:
                    first = int(metadata.get("page_start", metadata.get("page")))
                    last = int(metadata.get("page_end", first))
                    supported_pages.update(range(first, last + 1))
                except (ValueError, TypeError):
                    continue
            valid = end - start < 10000 and all(page in supported_pages for page in range(start, end + 1))
        citations.append({"source": name, "page_start": start, "page_end": end,
                          "chunk_id": evidence_id, "matched": valid})
    matched = [item for item in citations if item["matched"]]
    rate = round(len(matched) / len(citations), 3) if citations else 0.0
    return {"available_sources": sorted(available),
            "cited_sources": sorted({item["source"] for item in citations}),
            "matched_sources": sorted({item["source"] for item in matched}),
            "citations": citations, "citation_match_rate": rate,
            "citation_coverage": rate}


class PaperAgentWorkflow:
    """Application facade backed by a direct workflow and standalone harness."""

    def __init__(
        self,
        rag: PaperRAG | None,
        workspace_path: Path,
        memory_path: Path | None = None,
        tool_gateway: MCPPaperToolGateway | None = None,
    ) -> None:
        self.rag = rag
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_store = AgentMemoryStore(
            memory_path or workspace_path.with_name("agent_memory.sqlite3")
        )
        if tool_gateway is None:
            from src.config import get_openalex_api_key
            user_id = workspace_path.parent.parent.parent.name
            tool_gateway = MCPPaperToolGateway(openalex_api_key=get_openalex_api_key(user_id))
        self.tool_gateway = tool_gateway
        self.harness_runtime = ResearchHarness(self)

    @staticmethod
    def _step(state: AgentState, node) -> AgentState:
        """Apply one bounded node update and expose a complete stream snapshot."""
        return {**state, **node(state)}

    def _run(self, initial: AgentState) -> Iterator[AgentState]:
        state = self._step(initial, self._resolve_node)
        yield state
        if state.get("route") == "clarify":
            yield self._step(state, self._clarify_node)
            return
        state = self._step(state, self._supervisor_node)
        yield state
        route = state.get("route", "finish")
        if route == "harness":
            while state["harness"]["status"] == "running":
                state = self._step(state, self.harness_runtime.decide)
                yield state
                if state["harness"]["status"] != "running":
                    break
                if state["harness"].get("next") == "tool":
                    state = self._step(state, self.harness_runtime.execute)
                    yield state
                    continue
                if state["harness"].get("next") == "finish":
                    state = self._step(state, self.harness_runtime.finish)
                    yield state
                    if state["harness"]["status"] != "running":
                        break
                    state = self._step(state, self.harness_runtime.check)
                    yield state
                    continue
            return
        handlers = {"search": self._search_node, "ingest": self._ingest_node,
                    "analysis": self._analysis_node, "rag": self._rag_node,
                    "report": self._report_node}
        if route in handlers:
            state = self._step(state, handlers[route])
            yield state
        if route in {"analysis", "rag", "report"}:
            state = self._step(state, self._verify_node)
            yield state
            if self._after_verify(state) == "revise":
                state = self._step(state, self._revise_node)
                yield state
                yield self._step(state, self._verify_node)

    def _resolve_node(self, state: AgentState) -> dict:
        query = " ".join(state.get("query", "").split())
        user_id = state.get("user_id", "anonymous")
        preferences = extract_preferences(query)
        profile = self.memory_store.remember(user_id, preferences) if preferences else self.memory_store.get_profile(user_id)
        history = state.get("history", [])
        missing_fields = [] if state.get("execution_mode") == "harness" else detect_clarification_needs(query, history)
        if missing_fields:
            question = build_clarification_question(missing_fields)
            return {
                "resolved_query": query,
                "route": "clarify",
                "memory": profile,
                "clarification_required": True,
                "clarification_type": missing_fields[0],
                "missing_fields": missing_fields,
                "clarification_question": question,
                "clarification_context": {"original_query": query},
                "events": append_event(
                    state,
                    "resolve",
                    f"检测到缺失信息：{', '.join(missing_fields)}，转入澄清。",
                ),
            }
        resolved = query
        if history and any(signal in query for signal in REFERENCE_SIGNALS):
            last_question = str(history[-1].get("question", ""))
            resolved = f"上下文：上一轮问题为“{last_question}”。当前问题：{query}"
        return {
            "resolved_query": resolved,
            "memory": profile,
            "clarification_required": False,
            "missing_fields": [],
            "events": append_event(state, "resolve", "已完成指代与用户偏好解析。"),
        }

    def _clarify_node(self, state: AgentState) -> dict:
        message = state.get("clarification_question") or "请补充完成当前任务所需的信息。"
        return {"answer": message, "events": append_event(state, "clarify", message)}

    def _supervisor_node(self, state: AgentState) -> dict:
        if state.get("execution_mode") == "harness" and not state.get("pdf_paths"):
            return {"route": "harness", "route_reason": "受预算约束的科研任务循环。",
                    "events": append_event(state, "supervisor", "进入科研任务执行循环。")}
        route, reason = select_route(state.get("resolved_query") or state.get("query", ""), state.get("pdf_paths"))
        return {
            "route": route,
            "route_reason": reason,
            "events": append_event(state, "supervisor", f"{route}: {reason}"),
        }

    def _ingest_node(self, state: AgentState) -> dict:
        paths = [Path(path) for path in state.get("pdf_paths", [])]
        try:
            for path in paths:
                register_pdf(path.read_bytes(), path.name, state["user_id"], state["session_id"])
            paths = get_uploaded_pdf_paths(state["user_id"], state["session_id"])
            documents = load_pdf_documents(paths)
            rag = PaperRAG(user_id=state.get("user_id"))
            rag.build_index(documents)
            index_dir = Path(state["index_dir"])
            rag.save(index_dir)
            self.rag = rag
            upsert_session_meta(state["user_id"], state["session_id"],
                                paper_count=len(paths), chunk_count=len(rag.chunks))
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
        try:
            tool_result = self.tool_gateway.call_tool(
                "search_arxiv", {"query": query, "max_results": 5}
            )
            papers = list(tool_result.get("papers", []))
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
                "events": append_event(
                    state, "search", f"MCP工具 search_arxiv 返回 {len(papers)} 条结果。"
                ),
            }
        except Exception as exc:
            message = f"MCP论文搜索暂不可用：{exc}"
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
        if state.get("memory", {}).get("enabled", True) and preferences:
            query = f"用户长期偏好：{preferences}\n{query}"
        revision_options = {}
        if isinstance(self.rag, PaperRAG) and state.get("revision_feedback"):
            revision_options["revision_feedback"] = state["revision_feedback"]
        answer, chunks = self.rag.answer(
            query,
            top_k=int(state.get("top_k", 4)),
            conversation_history=state.get("history", []),
            **revision_options,
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
                f"[{index}] 来源：{item.get('metadata', {}).get('source', '')} "
                f"页码：{item.get('metadata', {}).get('page_start', item.get('metadata', {}).get('page', ''))} "
                f"{item.get('text', '')[:1200]}"
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
                    if not verdict or not isinstance(verdict.get("grounded"), bool) or not isinstance(verdict.get("sufficient_context"), bool):
                        semantic = {"status": "unavailable", "reason": "invalid_verdict"}
                    else:
                        semantic = {"status": "available", "verdict": verdict}
                    if verdict and verdict.get("grounded") is False:
                        status = "warning"
                        message = "引用格式可映射，但语义核验发现可能缺少证据支持的主张。"
                else:
                    semantic = {"status": "unavailable", "reason": "llm_not_configured"}
            except Exception as exc:
                semantic = {"status": "unavailable", "reason": str(exc)}
        if semantic and (semantic.get("verdict") or {}).get("sufficient_context") is False:
            status, message = "warning", "语义核验发现当前证据不足。"
        if state.get("revision_count", 0) >= 1 and status == "warning":
            answer = "当前资料不足，本次回答在补充检索后仍未通过证据核验，请补充相关论文或缩小问题范围。"
        verification = {"status": status, "message": message, **report, "semantic_check": semantic}
        return {
            "verification": verification,
            "answer": answer,
            "events": append_event(state, "verify", f"{status}: {message}"),
        }

    def _after_verify(self, state: AgentState) -> str:
        verification = state.get("verification", {})
        verdict = (verification.get("semantic_check") or {}).get("verdict") or {}
        from src.config import get_config
        needs_revision = bool(get_config(user_id=state.get("user_id")).openai_api_key) and (
            verification.get("status") == "warning" or verdict.get("sufficient_context") is False)
        return "revise" if needs_revision and state.get("revision_count", 0) < 1 and not state.get("error") else "finish"

    def _revise_node(self, state: AgentState) -> dict:
        enriched = dict(state)
        enriched["revision_feedback"] = (
            "上一版回答未通过证据核验，请保留原任务要求并修订。只保留证据直接支持的结论，"
            "逐字引用真实来源和对应页码；证据不足时明确说明当前资料不足。\n"
            f"核验反馈：{json.dumps(state.get('verification', {}), ensure_ascii=False)}"
        )
        enriched["top_k"] = min(20, int(state.get("top_k", 4)) * 2)
        result = self._answer_with_rag(enriched, "revise")
        result["revision_count"] = state.get("revision_count", 0) + 1
        return result

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
        stream: bool = False,
        execution_mode: str = "workflow",
        budget: dict | None = None,
        harness_resume: dict | None = None,
    ) -> AgentState | Iterator[AgentState]:
        if execution_mode not in {"workflow", "harness", "auto"}:
            raise ValueError("invalid execution_mode")
        configured = bool(get_config(user_id=user_id).openai_api_key)
        use_harness = not pdf_paths and (execution_mode == "harness" or (execution_mode == "auto" and configured))
        if use_harness and not configured:
            raise ValueError("科研任务模式需要先配置模型接口；无模型时可使用workflow模式。")
        run = copy.deepcopy(harness_resume) if harness_resume else (new_run(query, budget) if use_harness else {})
        if run and budget:
            run["budget"] = HarnessBudget(**budget).model_dump()
        if run:
            run.setdefault("history", [{"question": str(item.get("question", ""))[:220],
                                        "answer": str(item.get("answer", ""))[:420]}
                                       for item in (history or [])[-3:]])
            run["status"] = "running"
            run["next"] = "decide"
        initial: AgentState = {
            "execution_mode": "harness" if use_harness else "workflow",
            "harness": run,
            "query": query,
            "answer": "",
            "route": "finish",
            "resolved_query": "",
            "clarification_required": False,
            "clarification_question": "",
            "clarification_context": {},
            "missing_fields": [],
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
            "revision_count": 0,
            "revision_feedback": "",
        }
        def execute():
            lock_path = task_path(user_id, session_id, run["run_id"]).with_suffix(".lock") if use_harness else None
            if lock_path:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                lock = FileLock(str(lock_path), timeout=0) if lock_path else None
                if lock:
                    lock.acquire()
                try:
                    yield from self._run(initial)
                finally:
                    if lock:
                        lock.release()
            except Timeout as exc:
                raise ValueError("task is already running") from exc
        if stream:
            return execute()
        result = initial
        for result in execute():
            pass
        return result

    def resume_task(self, run_id: str, *, user_id: str, session_id: str,
                    index_dir: str, message: str = "", budget: dict | None = None) -> AgentState:
        path = task_path(user_id, session_id, run_id)
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot["user_id"] != user_id or snapshot["session_id"] != session_id:
            raise PermissionError("task ownership mismatch")
        run = snapshot["harness"]
        if run["status"] == "completed":
            return snapshot
        # A crash after creating an approval but before publishing its snapshot must
        # not bypass the approval or create an unrelated second task.
        for pending in self.memory_store.list_pending(user_id):
            continuation = pending["payload"].get("harness_continuation", {})
            if (pending["thread_id"] == session_id and continuation.get("run_id") == run_id
                    and pending["status"] in {"pending", "running", "failed"}):
                raise ValueError("该任务存在待处理审批，请先确认或取消。")
        if run["status"] == "awaiting_approval":
            run["observations"].append({"kind": "approval_expired", "message": "先前审批已过期，可重新请求确认。"})
        if message:
            run["observations"].append({"kind": "user_clarification", "message": message[:4000]})
        return self.invoke(run["goal"], user_id=user_id, session_id=session_id, index_dir=index_dir,
                           execution_mode="harness", budget=budget, harness_resume=run)

    def resume_approval(
        self, action_id: str, selections: list[int], *, index_dir: str,
        user_id: str | None = None, session_id: str | None = None,
        budget: dict | None = None,
    ) -> AgentState:
        lock_dir = self.memory_store.path.parent / "approval_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / (hashlib.sha256(action_id.encode()).hexdigest() + ".lock")
        try:
            with FileLock(str(lock_path), timeout=0):
                pending = self.memory_store.get_pending(action_id)
                if pending and ((user_id is not None and pending["user_id"] != user_id) or
                                (session_id is not None and pending["thread_id"] != session_id)):
                    raise PermissionError("approval does not belong to this user/session")
                # A running row without a live lock belongs to a crashed process.
                if pending and pending["status"] == "running":
                    self.memory_store.connection.execute(
                        "UPDATE pending_action SET status='failed' WHERE action_id=? AND status='running'",
                        (action_id,),
                    )
                    self.memory_store.connection.commit()
                return self._resume_approval(action_id, selections, index_dir=index_dir,
                                             user_id=user_id, session_id=session_id, budget=budget)
        except Timeout as exc:
            raise ValueError("approval is already running") from exc

    def _resume_approval(
        self, action_id: str, selections: list[int], *, index_dir: str,
        user_id: str | None = None, session_id: str | None = None,
        budget: dict | None = None,
    ) -> AgentState:
        pending = self.memory_store.get_pending(action_id)
        if pending is None:
            raise KeyError(f"approval not found: {action_id}")
        if ((user_id is not None and pending["user_id"] != user_id) or
                (session_id is not None and pending["thread_id"] != session_id)):
            raise PermissionError("approval does not belong to this user/session")
        if pending["status"] == "resolved":
            result = dict(pending["payload"].get("result") or {"answer": "该操作已完成。", "error": ""})
            if not result.get("pending_action"):
                result["pending_action"] = pending
            if self.rag is None:
                rag = PaperRAG(user_id=pending["user_id"])
                if rag.load(Path(index_dir)):
                    self.rag = rag
            return result
        token = self.memory_store.claim_pending(action_id)
        if token is None:
            raise ValueError(f"approval is unavailable: {self.memory_store.get_pending(action_id)['status']}")
        continuation = copy.deepcopy(pending["payload"].get("harness_continuation"))
        if continuation and budget:
            continuation["budget"] = HarnessBudget(**budget).model_dump()
        def save_progress(**updates):
            if continuation:
                updates["harness_continuation"] = continuation
            self.memory_store.checkpoint_pending(action_id, token, updates)
            pending["payload"].update(updates)
        papers = pending["payload"].get("papers", [])
        selected = sorted({int(value) for value in selections if 1 <= int(value) <= len(papers)})
        try:
            if not selected:
                result = {"answer": "已取消本次论文入库。", "events": [], "sources": [],
                          "verification": {}, "error": ""}
                decision = {"selected": [], "cancelled": True}
                continuation = pending["payload"].get("harness_continuation")
                if continuation:
                    continuation["observations"].append({"kind": "user_cancelled_ingestion", "selected": []})
                    result = self.harness_runtime._stop({"user_id": pending["user_id"],
                        "session_id": pending["thread_id"], "events": []}, continuation,
                        "needs_input", "已取消下载。原任务已保存，可补充资料或调整要求后继续。")
            else:
                download_dir = Path(index_dir).parent / "approval_staging" / action_id
                download_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                downloaded_paths = dict(pending["payload"].get("downloaded_paths", {}))
                # A partial download is retryable; never silently mark it successful.
                for number in selected:
                    paper = papers[number - 1]
                    cached = Path(downloaded_paths[str(number)]) if str(number) in downloaded_paths else None
                    if cached and cached.is_file() and cached.read_bytes().startswith(b"%PDF"):
                        paths.append(str(cached))
                        continue
                    if continuation:
                        if not self.harness_runtime._available(continuation, tool=True):
                            raise ValueError("下载达到工具或时间预算，请提高审批预算后重试。")
                        continuation["usage"]["tool_calls"] += 1
                        save_progress()
                    started = time.monotonic()
                    try:
                        download = self.tool_gateway.call_tool("download_arxiv_pdf", {
                            "pdf_url": str(paper.get("pdf_url") or ""),
                            "title": f"{number:02d}_{paper.get('title', 'paper')}",
                            "output_dir": str(download_dir),
                        })
                    finally:
                        if continuation:
                            continuation["usage"]["elapsed_seconds"] += time.monotonic() - started
                            save_progress()
                    path = Path(str(download.get("path", "")))
                    if not path.is_file() or not path.read_bytes().startswith(b"%PDF"):
                        raise ValueError("downloaded content is not a PDF")
                    paths.append(str(path))
                    downloaded_paths[str(number)] = str(path)
                    save_progress(downloaded_paths=downloaded_paths)
                if continuation and not self.harness_runtime._available(continuation):
                    raise ValueError("时间预算已用尽，请提高审批预算后重试。")
                started = time.monotonic()
                result = self.invoke(
                    "确认搜索结果并入库", user_id=pending["user_id"], session_id=pending["thread_id"],
                    pdf_paths=paths, index_dir=index_dir,
                )
                if continuation:
                    continuation["usage"]["elapsed_seconds"] += time.monotonic() - started
                    save_progress()
                if result.get("error"):
                    raise RuntimeError(result["error"])
                decision = {"selected": selected, "downloaded": len(paths)}
                if continuation:
                    saved_path = task_path(pending["user_id"], pending["thread_id"], continuation["run_id"])
                    if saved_path.exists():
                        saved = json.loads(saved_path.read_text(encoding="utf-8"))
                        if saved["harness"]["status"] == "completed":
                            resolved = self.memory_store.finish_pending(action_id, token, decision, saved)
                            saved["pending_action"] = resolved
                            return saved
                        if saved["harness"]["status"] not in {"awaiting_approval"}:
                            saved["harness"]["usage"]["tool_calls"] = max(saved["harness"]["usage"]["tool_calls"], continuation["usage"]["tool_calls"])
                            continuation = saved["harness"]
                    continuation["observations"].append({"kind": "approved_ingestion",
                        "selected": selected, "downloaded": len(paths)})
                    result = self.invoke(continuation["goal"], user_id=pending["user_id"],
                        session_id=pending["thread_id"], index_dir=index_dir,
                        execution_mode="harness", harness_resume=continuation)
            resolved = self.memory_store.finish_pending(action_id, token, decision, result)
            if not result.get("pending_action"):
                result["pending_action"] = resolved
            return result
        except Exception as exc:
            self.memory_store.finish_pending(action_id, token, {}, {"error": str(exc)}, failed=True)
            raise

    def close(self) -> None:
        self.memory_store.close()


def result_sources(result: AgentState) -> list[Chunk]:
    return [deserialize_chunk(item) for item in result.get("sources", [])]
