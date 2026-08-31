"""FastAPI and SSE surface for the paper agent workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent_workflow import PaperAgentWorkflow
from src.agent_memory import AgentMemoryStore
from src.rag_pipeline import PaperRAG
from src.session_manager import (
    get_index_dir, get_session_dir, get_user_dir, list_uploaded_pdfs,
    load_user_sessions, sanitize_identifier,
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


class ResumeRequest(BaseModel):
    action_id: str
    selections: list[int] = Field(default_factory=list)


class ProfileSettings(BaseModel):
    enabled: bool | None = None
    ttl_days: int | None = Field(default=None, ge=1, le=3650)


def workflow_for(user_id: str, thread_id: str) -> PaperAgentWorkflow:
    user_id = sanitize_identifier(user_id)
    thread_id = sanitize_identifier(thread_id)
    index_dir = get_index_dir(user_id, thread_id)
    rag = PaperRAG(user_id=user_id)
    loaded = rag.load(index_dir)
    return PaperAgentWorkflow(
        rag if loaded else None,
        get_session_dir(user_id, thread_id) / "agent_checkpoints.sqlite",
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


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    workflow = workflow_for(request.user_id, request.thread_id)
    try:
        return workflow.invoke(
            request.message, user_id=request.user_id, session_id=request.thread_id,
            history=request.history, top_k=request.top_k,
            index_dir=str(get_index_dir(request.user_id, request.thread_id)),
        )
    finally:
        workflow.close()


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def generate():
        yield f"data: {json.dumps({'type': 'status', 'message': 'workflow_started'}, ensure_ascii=False)}\n\n"
        try:
            result = chat(request)
            for event in result.get("events", []):
                yield f"data: {json.dumps({'type': 'progress', **event}, ensure_ascii=False)}\n\n"
            payload = {"type": "answer", "answer": result.get("answer", ""),
                       "verification": result.get("verification", {}),
                       "pending_action": result.get("pending_action", {})}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/upload-pdf")
async def upload_pdf(
    user_id: str = Form("anonymous"), thread_id: str = Form("default"),
    files: list[UploadFile] = File(...),
) -> dict:
    upload_dir = get_session_dir(user_id, thread_id) / "api_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for number, upload in enumerate(files, start=1):
        data = await upload.read()
        if len(data) > 50 * 1024 * 1024 or not data.startswith(b"%PDF"):
            raise HTTPException(400, f"invalid PDF: {upload.filename}")
        name = sanitize_identifier(Path(upload.filename or f"paper_{number}.pdf").stem)
        path = upload_dir / f"{number:02d}_{name}.pdf"
        path.write_bytes(data)
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
        )
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
