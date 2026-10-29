"""FastAPI and SSE surface for the paper agent workflow."""

from __future__ import annotations

import json
from pathlib import Path
from filelock import FileLock
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from src.agent_workflow import PaperAgentWorkflow
from src.agent_memory import AgentMemoryStore
from src.research_harness import HarnessBudget, task_path
from src.rag_pipeline import PaperRAG
from src.venue_rankings import import_rankings, journal_rankings
from src.paper_library import (
    search_papers, load_library, set_saved, download_selected,
    make_reading_card, comparison_markdown,
)
from src.session_manager import (
    get_index_dir, get_session_dir, get_user_dir, list_uploaded_pdfs,
    load_user_sessions, sanitize_identifier, register_pdf, load_session_history,
    save_session_history, upsert_session_meta,
)


app = FastAPI(title="Paper RAG Agent API", version="2.0.0")


def compatible_openapi() -> dict:
    """Add Swagger-compatible binary hints for OpenAPI 3.1 file arrays."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    upload_schema = schema.get("components", {}).get("schemas", {}).get(
        "Body_upload_pdf_api_upload_pdf_post", {}
    )
    files = upload_schema.get("properties", {}).get("files", {})
    items = files.get("items", {})
    if items.get("type") == "string":
        items["format"] = "binary"
    app.openapi_schema = schema
    return schema


app.openapi = compatible_openapi


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = "anonymous"
    thread_id: str = "default"
    top_k: int = Field(default=4, ge=1, le=20)
    history: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: Literal["auto", "workflow", "harness"] = "auto"
    budget: HarnessBudget = Field(default_factory=HarnessBudget)


class TaskResumeRequest(BaseModel):
    message: str = Field(default="", max_length=4000)
    budget: HarnessBudget | None = None


class ResumeRequest(BaseModel):
    action_id: str
    selections: list[int] = Field(default_factory=list)
    budget: HarnessBudget | None = None


class ProfileSettings(BaseModel):
    enabled: bool | None = None
    ttl_days: int | None = Field(default=None, ge=1, le=3650)


class PaperSearchRequest(BaseModel):
    user_id: str = "anonymous"
    thread_id: str = "default"
    keywords: str = Field(min_length=1, max_length=300)
    year_from: int = Field(ge=1900)
    year_to: int = Field(ge=1900)
    paper_type: Literal["all", "review", "article", "preprint"] = "all"
    sources: list[Literal["arxiv", "openalex"]] = Field(default_factory=lambda: ["arxiv", "openalex"])
    venue_kind: Literal["all", "journal", "conference", "preprint"] = "all"
    venue_query: str = Field(default="", max_length=300)
    ccf_levels: list[Literal["A", "B", "C"]] = Field(default_factory=list)
    jcr_quartiles: list[Literal["Q1", "Q2", "Q3", "Q4"]] = Field(default_factory=list)
    cas_zones: list[Literal["1", "2", "3", "4"]] = Field(default_factory=list)
    ranking_year: int | None = Field(default=None, ge=2000, le=2100)


class PaperSelectionRequest(BaseModel):
    user_id: str = "anonymous"
    thread_id: str = "default"
    ids: list[str] = Field(min_length=1)
    saved: bool = True


class ReadingCardRequest(BaseModel):
    user_id: str = "anonymous"
    thread_id: str = "default"
    source: str


def workflow_for(user_id: str, thread_id: str) -> PaperAgentWorkflow:
    user_id = sanitize_identifier(user_id)
    thread_id = sanitize_identifier(thread_id)
    index_dir = get_index_dir(user_id, thread_id)
    rag = PaperRAG(user_id=user_id)
    loaded = rag.load(index_dir)
    return PaperAgentWorkflow(
        rag if loaded else None,
        get_session_dir(user_id, thread_id) / "harness_workspace",
        get_user_dir(user_id) / "agent_memory.sqlite3",
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/threads")
def threads(user_id: str = "anonymous") -> list[dict]:
    return load_user_sessions(user_id)


@app.get("/api/papers")
def papers(user_id: str = "anonymous", thread_id: str = "default") -> list[dict]:
    return list_uploaded_pdfs(user_id, thread_id)


@app.post("/api/library/search")
def library_search(request: PaperSearchRequest) -> dict:
    try:
        return search_papers(request.user_id, request.thread_id, keywords=request.keywords,
                             year_from=request.year_from, year_to=request.year_to,
                             paper_type=request.paper_type, sources=tuple(request.sources),
                             venue_kind=request.venue_kind, venue_query=request.venue_query,
                             ccf_levels=tuple(request.ccf_levels), jcr_quartiles=tuple(request.jcr_quartiles),
                             cas_zones=tuple(request.cas_zones), ranking_year=request.ranking_year)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/library")
def library_get(user_id: str = "anonymous", thread_id: str = "default") -> dict:
    return load_library(user_id, thread_id)


@app.get("/api/library/rankings")
def rankings_get(user_id: str = "anonymous") -> list[dict]:
    return journal_rankings(user_id)


@app.post("/api/library/rankings/import")
async def rankings_import(user_id: str = Form("anonymous"), file: UploadFile = File(...)) -> dict:
    try:
        count = import_rankings(user_id, await file.read(5 * 1024 * 1024 + 1))
        return {"imported": count}
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/library/save")
def library_save(request: PaperSelectionRequest) -> list[dict]:
    try:
        return set_saved(request.user_id, request.thread_id, request.ids, request.saved)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/library/download")
def library_download(request: PaperSelectionRequest) -> list[dict]:
    try:
        return download_selected(request.user_id, request.thread_id, request.ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/library/card")
def library_card(request: ReadingCardRequest) -> dict:
    rag = PaperRAG(user_id=request.user_id)
    if not rag.load(get_index_dir(request.user_id, request.thread_id)):
        raise HTTPException(409, "请先下载或上传论文并建立索引。")
    try:
        return make_reading_card(request.user_id, request.thread_id, request.source, rag)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/library/comparison")
def library_comparison(user_id: str = "anonymous", thread_id: str = "default",
                       sources: str = "") -> dict:
    cards = load_library(user_id, thread_id)["cards"]
    names = [name.strip() for name in sources.split(",") if name.strip()]
    try:
        if any(name not in cards for name in names):
            raise ValueError("所选论文尚未生成精读卡片。")
        return {"markdown": comparison_markdown([cards[name] for name in names])}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def persist_chat(request: ChatRequest, result: dict, history: list[dict]) -> None:
    session_dir = get_session_dir(request.user_id, request.thread_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(session_dir / "history.lock")):
        existing = load_session_history(request.user_id, request.thread_id) or history
        updated = existing + [{"question": request.message, "answer": result.get("answer", ""),
                               "source_count": len(result.get("sources", [])),
                               "harness": result.get("harness", {}), "sources": result.get("sources", []),
                               "pending_action": result.get("pending_action", {}),
                               "verification": result.get("verification", {})}]
        save_session_history(request.user_id, request.thread_id, updated)
        upsert_session_meta(request.user_id, request.thread_id, question_count=len(updated))


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    workflow = workflow_for(request.user_id, request.thread_id)
    try:
        history = load_session_history(request.user_id, request.thread_id) or request.history
        result = workflow.invoke(
            request.message, user_id=request.user_id, session_id=request.thread_id,
            history=history, top_k=request.top_k,
            execution_mode=request.execution_mode, budget=request.budget.model_dump(),
            index_dir=str(get_index_dir(request.user_id, request.thread_id)),
        )
        persist_chat(request, result, history)
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        workflow.close()


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def generate():
        workflow = None
        def event(payload):
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield event({"type": "status", "message": "workflow_started"})
        try:
            workflow = workflow_for(request.user_id, request.thread_id)
            history = load_session_history(request.user_id, request.thread_id) or request.history
            emitted = 0
            result = {}
            for result in workflow.invoke(
                request.message, user_id=request.user_id, session_id=request.thread_id,
                history=history, top_k=request.top_k,
                execution_mode=request.execution_mode, budget=request.budget.model_dump(),
                index_dir=str(get_index_dir(request.user_id, request.thread_id)), stream=True,
            ):
                events = result.get("events", [])
                for progress in events[emitted:]:
                    yield event({"type": "progress", **progress})
                emitted = len(events)
            persist_chat(request, result, history)
            yield event({"type": "answer", "answer": result.get("answer", ""),
                         "sources": result.get("sources", []),
                         "verification": result.get("verification", {}),
                         "pending_action": result.get("pending_action", {}),
                         "harness": result.get("harness", {})})
        except Exception as exc:
            yield event({"type": "error", "message": str(exc)})
        finally:
            if workflow is not None:
                workflow.close()
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/upload-pdf")
async def upload_pdf(
    user_id: str = Form("anonymous"), thread_id: str = Form("default"),
    files: list[UploadFile] = File(...),
) -> dict:
    paths: list[str] = []
    for number, upload in enumerate(files, start=1):
        data = await upload.read()
        if len(data) > 50 * 1024 * 1024 or not data.startswith(b"%PDF"):
            raise HTTPException(400, f"invalid PDF: {upload.filename}")
        path = register_pdf(data, upload.filename or f"paper_{number}.pdf", user_id, thread_id)
        paths.append(str(path))
    workflow = workflow_for(user_id, thread_id)
    try:
        return workflow.invoke(
            "上传PDF并构建知识库", user_id=user_id, session_id=thread_id,
            pdf_paths=paths, index_dir=str(get_index_dir(user_id, thread_id)),
        )
    finally:
        workflow.close()


@app.post("/api/chat/resume")
def resume(request: ResumeRequest, user_id: str = "anonymous", thread_id: str = "default") -> dict:
    workflow = workflow_for(user_id, thread_id)
    try:
        return workflow.resume_approval(
            request.action_id, request.selections,
            index_dir=str(get_index_dir(user_id, thread_id)),
            user_id=user_id, session_id=thread_id,
            budget=request.budget.model_dump() if request.budget else None,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        workflow.close()


def memory_store(user_id: str) -> AgentMemoryStore:
    return AgentMemoryStore(get_user_dir(user_id) / "agent_memory.sqlite3")


@app.get("/api/approvals")
def approvals(user_id: str = "anonymous") -> list[dict]:
    store = memory_store(user_id)
    try:
        return store.list_pending(user_id)
    finally:
        store.close()


@app.get("/api/user-profile")
def get_profile(user_id: str = "anonymous") -> dict:
    store = memory_store(user_id)
    try:
        return store.get_profile(user_id)
    finally:
        store.close()


@app.patch("/api/user-profile/settings")
def update_profile(settings: ProfileSettings, user_id: str = "anonymous") -> dict:
    store = memory_store(user_id)
    try:
        return store.update_settings(user_id, enabled=settings.enabled, ttl_days=settings.ttl_days)
    finally:
        store.close()


@app.delete("/api/user-profile")
def delete_profile(user_id: str = "anonymous") -> dict[str, bool]:
    store = memory_store(user_id)
    try:
        store.clear_profile(user_id)
        return {"deleted": True}
    finally:
        store.close()


@app.get("/api/tasks/{run_id}")
def get_task(run_id: str, user_id: str = "anonymous", thread_id: str = "default") -> dict:
    try:
        snapshot = json.loads(task_path(user_id, thread_id, run_id).read_text(encoding="utf-8"))
        if snapshot["user_id"] != user_id or snapshot["session_id"] != thread_id:
            raise HTTPException(403, "task ownership mismatch")
        return snapshot
    except FileNotFoundError as exc:
        raise HTTPException(404, "task not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/tasks/{run_id}/resume")
def resume_task(run_id: str, request: TaskResumeRequest,
                user_id: str = "anonymous", thread_id: str = "default") -> dict:
    workflow = workflow_for(user_id, thread_id)
    try:
        result = workflow.resume_task(run_id, user_id=user_id, session_id=thread_id,
            index_dir=str(get_index_dir(user_id, thread_id)), message=request.message,
            budget=request.budget.model_dump() if request.budget else None)
        persist_chat(ChatRequest(message=request.message or "继续科研任务", user_id=user_id, thread_id=thread_id),
                     result, load_session_history(user_id, thread_id))
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, "task not found") from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        workflow.close()


@app.get("/api/tasks/{run_id}/report")
def task_report(run_id: str, user_id: str = "anonymous", thread_id: str = "default"):
    snapshot = get_task(run_id, user_id, thread_id)
    path = task_path(user_id, thread_id, run_id).with_suffix(".md")
    if snapshot["harness"]["status"] != "completed" or not path.is_file():
        raise HTTPException(404, "task report not available")
    return FileResponse(path, media_type="text/markdown", filename=f"research_{run_id}.md")
