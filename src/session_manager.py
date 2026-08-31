from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import uuid


DATA_ROOT = Path("data/users")


def sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned[:48] or "user"


def create_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_user_dir(user_id: str) -> Path:
    return DATA_ROOT / sanitize_identifier(user_id)


def get_session_dir(user_id: str, session_id: str) -> Path:
    return get_user_dir(user_id) / "sessions" / sanitize_identifier(session_id)


def get_index_dir(user_id: str, session_id: str) -> Path:
    return get_session_dir(user_id, session_id) / "index"


def get_sessions_file(user_id: str) -> Path:
    return get_user_dir(user_id) / "sessions.json"


def load_user_sessions(user_id: str) -> list[dict]:
    sessions_file = get_sessions_file(user_id)
    if not sessions_file.exists():
        return []
    return json.loads(sessions_file.read_text(encoding="utf-8"))


def save_user_sessions(user_id: str, sessions: list[dict]) -> None:
    sessions_file = get_sessions_file(user_id)
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    sessions_file.write_text(
        json.dumps(sessions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_session_meta(user_id: str, session_id: str, **updates) -> dict:
    sessions = load_user_sessions(user_id)
    existing = next((item for item in sessions if item["session_id"] == session_id), None)
    if existing is None:
        existing = {
            "session_id": session_id,
            "title": updates.get("title") or f"会话 {session_id}",
            "created_at": now_text(),
            "updated_at": now_text(),
            "chunk_count": 0,
            "paper_count": 0,
            "question_count": 0,
        }
        sessions.insert(0, existing)

    existing.update(updates)
    existing["updated_at"] = now_text()
    save_user_sessions(user_id, sessions)
    return existing


def get_session_label(meta: dict) -> str:
    title = meta.get("title") or meta["session_id"]
    updated_at = meta.get("updated_at", "")
    return f"{title} | {updated_at}"


def get_session_meta(user_id: str, session_id: str) -> dict | None:
    return next(
        (item for item in load_user_sessions(user_id) if item["session_id"] == session_id),
        None,
    )


def _preview_text(text: str, max_length: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length - 3]}..."


def list_session_summaries(user_id: str) -> list[dict]:
    rows: list[dict] = []
    for meta in load_user_sessions(user_id):
        rows.append(
            {
                "会话名称": meta.get("title") or meta["session_id"],
                "创建时间": meta.get("created_at", ""),
                "更新时间": meta.get("updated_at", ""),
                "论文数": meta.get("paper_count", 0),
                "问题数": meta.get("question_count", 0),
                "文本块": meta.get("chunk_count", 0),
                "会话ID": meta["session_id"],
            }
        )
    return rows


def rename_session(user_id: str, session_id: str, title: str) -> None:
    if not title.strip():
        return
    upsert_session_meta(user_id, session_id, title=title.strip())


def delete_session(user_id: str, session_id: str) -> str | None:
    sessions = [
        item for item in load_user_sessions(user_id) if item["session_id"] != session_id
    ]
    save_user_sessions(user_id, sessions)

    session_dir = get_session_dir(user_id, session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)

    if sessions:
        return sessions[0]["session_id"]
    return None


def get_history_file(user_id: str, session_id: str) -> Path:
    return get_session_dir(user_id, session_id) / "history.json"


def get_reports_dir(user_id: str, session_id: str) -> Path:
    return get_session_dir(user_id, session_id) / "reports"


def load_session_history(user_id: str, session_id: str) -> list[dict]:
    history_file = get_history_file(user_id, session_id)
    if not history_file.exists():
        return []
    return json.loads(history_file.read_text(encoding="utf-8"))


def search_user_history(user_id: str, keyword: str) -> list[dict]:
    query = keyword.strip().lower()
    if not query:
        return []

    results: list[dict] = []
    for meta in load_user_sessions(user_id):
        session_id = meta["session_id"]
        history = load_session_history(user_id, session_id)
        for item in reversed(history):
            question = item.get("question", "")
            answer = item.get("answer", "")
            haystack = f"{question}\n{answer}".lower()
            if query not in haystack:
                continue

            results.append(
                {
                    "会话名称": meta.get("title") or session_id,
                    "更新时间": meta.get("updated_at", ""),
                    "问题": _preview_text(question, 96),
                    "回答摘要": _preview_text(answer, 160),
                    "检索片段": item.get("source_count", 0),
                    "报告文件": item.get("report_path", ""),
                    "会话ID": session_id,
                }
            )

    return results


def save_session_history(user_id: str, session_id: str, history: list[dict]) -> None:
    history_file = get_history_file(user_id, session_id)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_markdown_report(
    user_id: str,
    session_id: str,
    question: str,
    answer: str,
    sources: list,
) -> Path:
    reports_dir = get_reports_dir(user_id, session_id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = reports_dir / report_name

    source_lines = []
    for index, source in enumerate(sources, start=1):
        metadata = source.metadata
        page_start = metadata.get("page_start", metadata.get("page", "unknown"))
        page_end = metadata.get("page_end", page_start)
        page_label = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        source_lines.append(
            f"{index}. {metadata.get('source', 'unknown')} | "
            f"第 {page_label} 页 | "
            f"{metadata.get('section', 'body')} | "
            f"{metadata.get('section_title', 'unknown')} | "
            f"{metadata.get('content_type', 'text')}"
        )

    content = "\n".join(
        [
            "# 科研论文问答报告",
            "",
            f"生成时间：{now_text()}",
            "",
            "## 问题",
            "",
            question,
            "",
            "## 回答",
            "",
            answer,
            "",
            "## 检索来源",
            "",
            "\n".join(source_lines) if source_lines else "无",
            "",
        ]
    )
    report_path.write_text(content, encoding="utf-8")
    return report_path


def save_uploaded_pdfs(uploaded_files, user_id: str, session_id: str) -> list[Path]:
    upload_dir = get_session_dir(user_id, session_id) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        original_name = Path(uploaded_file.name)
        stem = sanitize_identifier(original_name.stem)
        suffix = original_name.suffix or ".pdf"
        saved_path = upload_dir / f"{index:02d}_{stem}{suffix}"
        saved_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(saved_path)

    return saved_paths


def list_uploaded_pdfs(user_id: str, session_id: str) -> list[dict]:
    upload_dir = get_session_dir(user_id, session_id) / "uploads"
    if not upload_dir.exists():
        return []

    files = []
    for path in sorted(upload_dir.glob("*.pdf")):
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "path": str(path),
            }
        )
    return files


def get_uploaded_pdf_paths(user_id: str, session_id: str) -> list[Path]:
    upload_dir = get_session_dir(user_id, session_id) / "uploads"
    if not upload_dir.exists():
        return []
    return sorted(upload_dir.glob("*.pdf"))


def ensure_session_store(session_state) -> None:
    if "user_sessions" not in session_state:
        session_state.user_sessions = {}


def get_or_create_session_data(session_state, session_id: str) -> dict:
    ensure_session_store(session_state)
    if session_id not in session_state.user_sessions:
        session_state.user_sessions[session_id] = {
            "rag": None,
            "history": [],
        }
    return session_state.user_sessions[session_id]
