from pathlib import Path

import streamlit as st

from src.analysis_tasks import get_task_names, get_task_prompt
from src.config import save_api_config
from src.evaluation import evaluate_retrieval, parse_evaluation_cases
from src.agent_workflow import PaperAgentWorkflow, result_sources, select_route
from src.agent_memory import AgentMemoryStore
from src.llm import get_llm_status, test_llm_connection
from src.paper_loader import load_pdf_documents
from src.providers import build_secrets_example, get_preset_by_name, get_provider_options
from src.rag_pipeline import PaperRAG
from src.session_manager import (
    create_session_id,
    delete_session,
    get_index_dir,
    get_session_dir,
    get_user_dir,
    get_or_create_session_data,
    get_uploaded_pdf_paths,
    get_session_label,
    list_uploaded_pdfs,
    load_session_history,
    load_user_sessions,
    rename_session,
    sanitize_identifier,
    save_session_history,
    save_markdown_report,
    save_uploaded_pdfs,
    search_user_history,
    upsert_session_meta,
)


st.set_page_config(
    page_title="RAG 科研论文问答系统",
    page_icon="RAG",
    layout="wide",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rag-bg: #f7f7f5;
            --rag-panel: #ffffff;
            --rag-soft: #f1f2f4;
            --rag-border: #d8dadd;
            --rag-text: #1f2328;
            --rag-muted: #69707a;
            --rag-blue: #1f5eff;
            --rag-green: #0f766e;
            --rag-amber: #a16207;
        }

        .stApp {
            background: var(--rag-bg);
            color: var(--rag-text);
        }

        section[data-testid="stSidebar"] {
            background: #f2f3f5;
            border-right: 1px solid var(--rag-border);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stMarkdown {
            color: var(--rag-text);
        }

        .main .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 1040px;
        }

        .rag-title {
            padding: 0.35rem 0 0.95rem 0;
            border-bottom: 1px solid var(--rag-border);
            margin-bottom: 0.9rem;
        }

        .rag-title h1 {
            font-size: 1.55rem;
            line-height: 1.2;
            margin: 0;
            letter-spacing: 0;
        }

        .rag-title p {
            margin: 0.45rem 0 0 0;
            color: var(--rag-muted);
            font-size: 0.98rem;
        }

        .rag-workspace-note {
            color: var(--rag-muted);
            font-size: 0.9rem;
            margin: -0.35rem 0 0.8rem 0;
        }

        div[data-testid="stMetric"] {
            background: var(--rag-panel);
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }

        div[data-testid="stMetric"] label {
            color: var(--rag-muted);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            background: var(--rag-panel);
        }

        div[data-testid="stForm"] {
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            padding: 0.8rem;
            background: rgba(255, 255, 255, 0.72);
        }

        .stButton > button,
        .stDownloadButton > button,
        .stLinkButton > a {
            border-radius: 8px;
            border: 1px solid var(--rag-border);
            font-weight: 600;
            min-height: 2.55rem;
        }

        .stButton > button[kind="primary"] {
            background: var(--rag-blue);
            border-color: var(--rag-blue);
        }

        textarea,
        input,
        div[data-baseweb="select"] > div {
            border-radius: 8px !important;
        }

        div[data-testid="stChatMessage"] {
            background: transparent;
            padding: 0.35rem 0;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            line-height: 1.65;
        }

        div[data-testid="stTabs"] button {
            font-weight: 600;
        }

        .rag-section-note {
            color: var(--rag-muted);
            font-size: 0.92rem;
            margin-top: -0.25rem;
            margin-bottom: 0.75rem;
        }

        .rag-answer-block {
            background: var(--rag-panel);
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin-top: 0.75rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }

        .rag-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.25rem 0 0.8rem 0;
        }

        .rag-pill {
            display: inline-flex;
            align-items: center;
            min-height: 1.85rem;
            padding: 0.18rem 0.65rem;
            border-radius: 999px;
            border: 1px solid var(--rag-border);
            background: #ffffff;
            color: var(--rag-muted);
            font-size: 0.86rem;
            font-weight: 600;
        }

        .rag-pill strong {
            color: var(--rag-text);
            margin-left: 0.3rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_title() -> None:
    st.markdown(
        """
        <div class="rag-title">
            <h1>基于 RAG 的科研论文智能问答系统</h1>
            <p>面向算法论文阅读、方法分析、实验对比和复现辅助的科研知识库。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def truncate_text(text: str, max_length: int = 72) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 3]}..."


def serialize_sources(sources: list) -> list[dict]:
    return [
        {
            "text": source.text,
            "metadata": source.metadata,
        }
        for source in sources
    ]


def render_retrieval_sources(
    sources: list,
    key_prefix: str,
    use_expanders: bool = True,
) -> None:
    for index, source in enumerate(sources, start=1):
        if isinstance(source, dict):
            source_text = source.get("text", "")
            metadata = source.get("metadata", {})
        else:
            source_text = source.text
            metadata = source.metadata

        source_name = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        page_start = metadata.get("page_start", page)
        page_end = metadata.get("page_end", page_start)
        page_label = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        section = metadata.get("section", "body")
        section_title = metadata.get("section_title", "")
        content_type = metadata.get("content_type", "text")
        caption = metadata.get("caption", "")
        image_path = metadata.get("image_path", "")
        retrieval = metadata.get("_retrieval", {})
        vector_score = retrieval.get("vector_rank_score", 0)
        keyword_score = retrieval.get("keyword_rank_score", 0)
        fused_score = retrieval.get("fused_score", 0)
        final_score = retrieval.get("final_score", 0)

        title = f"片段 {index} | {source_name} | 第 {page_label} 页 | {section} | {content_type}"
        source_container = st.expander(title) if use_expanders else st.container(border=True)
        with source_container:
            if not use_expanders:
                st.caption(title)
            st.markdown(
                f"""
                <div class="rag-pill-row">
                    <span class="rag-pill">文件<strong>{source_name}</strong></span>
                    <span class="rag-pill">页码<strong>{page_label}</strong></span>
                    <span class="rag-pill">章节<strong>{section}</strong></span>
                    <span class="rag-pill">章节标题<strong>{section_title or 'unknown'}</strong></span>
                    <span class="rag-pill">类型<strong>{content_type}</strong></span>
                    <span class="rag-pill">向量<strong>{vector_score}</strong></span>
                    <span class="rag-pill">BM25<strong>{keyword_score}</strong></span>
                    <span class="rag-pill">融合<strong>{fused_score}</strong></span>
                    <span class="rag-pill">重排<strong>{final_score}</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(
                "片段内容",
                value=source_text,
                height=180,
                disabled=True,
                key=f"{key_prefix}_{index}_{page}",
            )
            if caption:
                st.caption(f"图表题：{caption}")
            if content_type == "figure" and image_path and Path(image_path).exists():
                st.image(image_path, caption=caption or source_name, use_column_width=True)


def get_content_type_counts(rag: PaperRAG | None) -> dict[str, int]:
    counts = {"text": 0, "table": 0, "figure": 0}
    if rag is None:
        return counts
    for chunk in rag.chunks:
        content_type = chunk.metadata.get("content_type", "text")
        counts[content_type] = counts.get(content_type, 0) + 1
    return counts


def render_knowledge_overview(rag: PaperRAG) -> None:
    counts = get_content_type_counts(rag)
    overview_tab, table_tab, figure_tab = st.tabs(["概览", "表格", "图片"])

    with overview_tab:
        columns = st.columns(4)
        columns[0].metric("总 Chunk", len(rag.chunks))
        columns[1].metric("正文", counts.get("text", 0))
        columns[2].metric("表格", counts.get("table", 0))
        columns[3].metric("图片", counts.get("figure", 0))
        st.caption("表格以 Markdown 独立入库；图片以图题、附近正文和原图路径入库。")

    table_chunks = [
        chunk for chunk in rag.chunks if chunk.metadata.get("content_type") == "table"
    ]
    with table_tab:
        if not table_chunks:
            st.info("当前知识库未检测到可提取的表格。")
        else:
            table_options = {
                build_asset_label(chunk, index): chunk
                for index, chunk in enumerate(table_chunks, start=1)
            }
            selected_table = table_options[
                st.selectbox("选择表格", list(table_options), key=f"table_preview_{id(rag)}")
            ]
            render_asset_metadata(selected_table.metadata)
            st.markdown(selected_table.text)

    figure_chunks = [
        chunk for chunk in rag.chunks if chunk.metadata.get("content_type") == "figure"
    ]
    with figure_tab:
        if not figure_chunks:
            st.info("当前知识库未检测到带 Caption 的图片。")
        else:
            figure_options = {
                build_asset_label(chunk, index): chunk
                for index, chunk in enumerate(figure_chunks, start=1)
            }
            selected_figure = figure_options[
                st.selectbox("选择图片", list(figure_options), key=f"figure_preview_{id(rag)}")
            ]
            render_asset_metadata(selected_figure.metadata)
            image_path = selected_figure.metadata.get("image_path", "")
            if image_path and Path(image_path).exists():
                st.image(
                    image_path,
                    caption=selected_figure.metadata.get("caption", ""),
                    use_column_width=True,
                )
            else:
                st.warning("已识别图题，但 PDF 中没有可直接提取的嵌入位图。")
            st.markdown(selected_figure.text)


def build_asset_label(chunk, index: int) -> str:
    metadata = chunk.metadata
    caption = metadata.get("caption") or f"图表 {index}"
    source = metadata.get("source", "unknown")
    page = metadata.get("page", "unknown")
    return f"{index}. {caption} | {source} | 第 {page} 页"


def render_asset_metadata(metadata: dict) -> None:
    st.caption(
        f"论文：{metadata.get('source', 'unknown')} · "
        f"页码：{metadata.get('page', 'unknown')} · "
        f"章节：{metadata.get('section_title', 'unknown')} · "
        f"类型：{metadata.get('content_type', 'text')}"
    )


inject_theme()

if "user_id" not in st.session_state:
    render_app_title()
    with st.form("login_form"):
        st.subheader("进入系统")
        username = st.text_input("用户名", placeholder="例如：marc 或 student01")
        login = st.form_submit_button("进入", use_container_width=True)

    if login:
        if not username.strip():
            st.error("请先填写用户名。")
        else:
            st.session_state.user_id = sanitize_identifier(username)
            st.session_state.user_display_name = username.strip()
            st.session_state.user_sessions = {}
            saved_sessions = load_user_sessions(st.session_state.user_id)
            if saved_sessions:
                st.session_state.active_session_id = saved_sessions[0]["session_id"]
            else:
                st.session_state.active_session_id = create_session_id()
                upsert_session_meta(
                    st.session_state.user_id,
                    st.session_state.active_session_id,
                    title="新会话",
                )
            st.rerun()

    st.stop()

user_id = st.session_state.user_id
active_session_id = st.session_state.active_session_id
session_data = get_or_create_session_data(st.session_state, active_session_id)
active_uploaded_pdfs = list_uploaded_pdfs(user_id, active_session_id)
if not session_data["history"]:
    session_data["history"] = load_session_history(user_id, active_session_id)

if session_data["rag"] is None:
    index_dir = get_index_dir(user_id, active_session_id)
    restored_rag = PaperRAG(user_id=user_id)
    if restored_rag.load(index_dir):
        session_data["rag"] = restored_rag

current_meta = next(
    (
        item
        for item in load_user_sessions(user_id)
        if item["session_id"] == active_session_id
    ),
    {},
)
llm_status = get_llm_status(user_id=user_id)

with st.sidebar:
    st.subheader("工作区")
    st.caption("科研论文 RAG")
    workspace_line = f"论文 {len(active_uploaded_pdfs)} · 问答 {len(session_data['history'])}"
    if session_data["rag"] is not None:
        content_counts = get_content_type_counts(session_data["rag"])
        workspace_line += (
            f" · Chunk {len(session_data['rag'].chunks)}"
            f" · 表 {content_counts.get('table', 0)}"
            f" · 图 {content_counts.get('figure', 0)}"
        )
    st.caption(workspace_line)
    if session_data["rag"] is not None:
        st.success("知识库已就绪")
    else:
        st.warning("请先构建知识库")

    with st.expander("论文与索引", expanded=session_data["rag"] is None):
        uploaded_files = st.file_uploader(
            "上传 PDF 论文",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_uploader_{active_session_id}",
        )

        top_k = st.slider("检索片段数量", min_value=3, max_value=12, value=8)
        build_index = st.button("构建知识库", type="primary", use_container_width=True)
        rebuild_saved_index = st.button("重建当前索引", use_container_width=True)

        if active_uploaded_pdfs:
            for file_info in active_uploaded_pdfs:
                st.caption(f"{file_info['name']} | {file_info['size_kb']} KB")

    with st.expander("模型接口", expanded=not llm_status["configured"]):
        st.caption(f"模型：{llm_status['model']}")
        st.caption(f"接口：{llm_status['base_url']}")
        if llm_status["configured"]:
            st.success("API Key 已配置")
        else:
            st.warning("API Key 未配置")

        if st.button("测试 API 连接", use_container_width=True):
            with st.spinner("正在测试 API..."):
                try:
                    st.info(test_llm_connection(user_id=user_id))
                except Exception as exc:
                    st.error(f"API 测试失败：{exc}")

        provider_name = st.selectbox("模型服务商", get_provider_options())
        preset = get_preset_by_name(provider_name)

        with st.form("api_config_form"):
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="粘贴你的 API Key",
            )
            base_url = st.text_input("Base URL", value=preset.base_url)
            model = st.text_input("模型名", value=preset.model)
            saved = st.form_submit_button("保存 API 配置", use_container_width=True)

        if saved:
            if not api_key.strip():
                st.error("请填写 API Key。")
            elif not base_url.strip() or not model.strip():
                st.error("请填写 Base URL 和模型名。")
            elif not base_url.startswith("http"):
                st.error("Base URL 应该是以 http 或 https 开头的接口地址。")
            else:
                save_api_config(api_key, base_url, model, user_id=user_id)
                st.success("当前用户的 API 配置已保存。")
                st.rerun()

        st.caption("文档与示例")
        st.caption(preset.note)
        if preset.base_url.startswith("http"):
            st.markdown(f"API 地址：[{preset.base_url}]({preset.base_url})")
        else:
            st.caption(f"API 地址：{preset.base_url}")
        if preset.docs_url:
            st.link_button("打开官方文档", preset.docs_url, use_container_width=True)
        if preset.console_url:
            st.link_button("打开 API Key 页面", preset.console_url, use_container_width=True)
        st.code(build_secrets_example(preset), language="toml")

    with st.expander("长期记忆"):
        memory_store = AgentMemoryStore(get_user_dir(user_id) / "agent_memory.sqlite3")
        try:
            profile = memory_store.get_profile(user_id)
            memory_enabled = st.toggle("启用用户偏好记忆", value=profile["enabled"])
            memory_ttl = st.number_input(
                "记忆 TTL（天）", min_value=1, max_value=3650,
                value=int(profile["ttl_days"]), step=1,
            )
            if st.button("保存记忆设置", use_container_width=True):
                memory_store.update_settings(
                    user_id, enabled=memory_enabled, ttl_days=int(memory_ttl)
                )
                st.success("长期记忆设置已保存。")
            preferences = profile.get("preferences", {})
            if preferences:
                st.json(preferences)
            else:
                st.caption("尚未提取到显式研究偏好。")
            if st.button("清除长期记忆", use_container_width=True):
                memory_store.clear_profile(user_id)
                st.success("长期记忆已清除。")
                st.rerun()
        finally:
            memory_store.close()

    with st.expander("RAG 评估"):
        st.caption("格式：问题 | 期望关键词1, 期望关键词2")
        default_eval_cases = "\n".join(
            [
                "这篇论文用了什么算法？ | algorithm, method, model",
                "实验对比了哪些方法？ | baseline, experiment, comparison",
                "这篇论文的创新点是什么？ | propose, contribution, novel",
            ]
        )
        eval_text = st.text_area("评估用例", value=default_eval_cases, height=120)
        eval_top_k = st.slider("评估 Top-K", min_value=3, max_value=12, value=5)
        run_eval = st.button("运行检索评估", use_container_width=True)

        if run_eval:
            if session_data["rag"] is None:
                st.warning("请先构建论文知识库。")
            else:
                cases = parse_evaluation_cases(eval_text)
                if not cases:
                    st.warning("请按格式填写至少一个评估用例。")
                else:
                    rows, summary = evaluate_retrieval(
                        session_data["rag"],
                        cases,
                        top_k=eval_top_k,
                    )
                    st.metric("命中率", summary["hit_rate"])
                    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("会话")

    saved_sessions = load_user_sessions(user_id)
    if saved_sessions:
        session_labels = [get_session_label(item) for item in saved_sessions]
        current_index = next(
            (
                index
                for index, item in enumerate(saved_sessions)
                if item["session_id"] == active_session_id
            ),
            0,
        )
        selected_label = st.radio(
            "会话列表",
            session_labels,
            index=current_index,
            label_visibility="collapsed",
        )
        selected_index = session_labels.index(selected_label)
        selected_session_id = saved_sessions[selected_index]["session_id"]
        if selected_session_id != active_session_id:
            st.session_state.active_session_id = selected_session_id
            st.rerun()

    if st.button("新建会话", use_container_width=True):
        st.session_state.active_session_id = create_session_id()
        upsert_session_meta(user_id, st.session_state.active_session_id, title="新会话")
        st.rerun()

    with st.expander("管理当前会话"):
        new_title = st.text_input(
            "会话名称",
            value=current_meta.get("title", "新会话"),
            key=f"session_title_{active_session_id}",
        )
        if st.button("保存会话名称", use_container_width=True):
            rename_session(user_id, active_session_id, new_title)
            st.success("会话名称已保存。")
            st.rerun()

        confirm_delete = st.checkbox(
            "确认删除当前会话",
            key=f"confirm_delete_{active_session_id}",
        )
        if st.button("删除当前会话", use_container_width=True, disabled=not confirm_delete):
            next_session_id = delete_session(user_id, active_session_id)
            st.session_state.user_sessions.pop(active_session_id, None)
            if next_session_id is None:
                next_session_id = create_session_id()
                upsert_session_meta(user_id, next_session_id, title="新会话")
            st.session_state.active_session_id = next_session_id
            st.warning("当前会话已删除。")
            st.rerun()

    with st.expander("搜索历史问答"):
        history_keyword = st.text_input(
            "关键词",
            placeholder="创新点、ALNS、复现、对比算法",
        )
        if history_keyword.strip():
            search_rows = search_user_history(user_id, history_keyword)
            if search_rows:
                matched_options = [
                    (f"{index}. {row['会话名称']} | {row['问题']}", row["会话ID"])
                    for index, row in enumerate(search_rows, start=1)
                ]
                selected_match_label = st.selectbox(
                    "搜索结果",
                    [label for label, _ in matched_options],
                    key="history_search_session_select",
                )
                matched_session_id = next(
                    session_id
                    for label, session_id in matched_options
                    if label == selected_match_label
                )
                if st.button(
                    "打开搜索结果",
                    use_container_width=True,
                    disabled=matched_session_id == active_session_id,
                ):
                    st.session_state.active_session_id = matched_session_id
                    st.rerun()
            else:
                st.info("没有找到匹配记录。")

    st.divider()
    st.subheader("用户")
    st.caption(st.session_state.user_display_name)
    if st.button("退出登录", use_container_width=True):
        for key in [
            "user_id",
            "user_display_name",
            "active_session_id",
            "user_sessions",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

if build_index:
    if not uploaded_files:
        st.warning("请先上传至少一篇 PDF 论文。")
    else:
        with st.spinner("正在解析论文并构建向量索引..."):
            pdf_paths = save_uploaded_pdfs(uploaded_files, user_id, active_session_id)
            documents = load_pdf_documents(pdf_paths)
            rag = PaperRAG(user_id=user_id)
            rag.build_index(documents)
            rag.save(get_index_dir(user_id, active_session_id))
            session_data["rag"] = rag
            session_data["history"] = []
            save_session_history(user_id, active_session_id, session_data["history"])
            upsert_session_meta(
                user_id,
                active_session_id,
                title=uploaded_files[0].name if uploaded_files else "论文问答会话",
                chunk_count=len(rag.chunks),
                paper_count=len(uploaded_files),
                question_count=0,
            )
        counts = get_content_type_counts(session_data["rag"])
        st.success(
            f"知识库构建完成：正文 {counts.get('text', 0)} 块，"
            f"表格 {counts.get('table', 0)} 块，图片 {counts.get('figure', 0)} 块。"
        )

if rebuild_saved_index:
    saved_pdf_paths = get_uploaded_pdf_paths(user_id, active_session_id)
    if not saved_pdf_paths:
        st.warning("当前会话还没有已保存的 PDF，请先上传论文并构建知识库。")
    else:
        with st.spinner("正在使用已保存论文重建索引..."):
            documents = load_pdf_documents(saved_pdf_paths)
            rag = PaperRAG(user_id=user_id)
            rag.build_index(documents)
            rag.save(get_index_dir(user_id, active_session_id))
            session_data["rag"] = rag
            session_data["history"] = []
            save_session_history(user_id, active_session_id, session_data["history"])
            upsert_session_meta(
                user_id,
                active_session_id,
                chunk_count=len(rag.chunks),
                paper_count=len(saved_pdf_paths),
                question_count=0,
            )
        counts = get_content_type_counts(session_data["rag"])
        st.success(
            f"索引已重建：正文 {counts.get('text', 0)} 块，"
            f"表格 {counts.get('table', 0)} 块，图片 {counts.get('figure', 0)} 块。"
        )

active_uploaded_pdfs = list_uploaded_pdfs(user_id, active_session_id)

st.markdown(f"### {current_meta.get('title', '新会话')}")
chat_status = "知识库已就绪" if session_data["rag"] is not None else "未构建知识库"
st.caption(f"{chat_status} · {len(session_data['history'])} 条问答")

if session_data["rag"] is not None:
    with st.expander("知识库内容与图表预览", expanded=False):
        render_knowledge_overview(session_data["rag"])

if session_data["history"]:
    for item in session_data["history"]:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            st.caption(f"检索片段数：{item.get('source_count', 0)}")
else:
    with st.chat_message("assistant"):
        if session_data["rag"] is None:
            st.write("先在左侧工作区上传论文并构建知识库，然后就可以开始问答。")
        else:
            st.write("知识库已准备好。可以直接问论文的方法、创新点、实验设置或复现问题。")

with st.expander("快捷问题", expanded=False):
    question_templates = [
        "这篇论文的创新点是什么？",
        "这篇论文用了什么算法？",
        "实验对比了哪些方法？",
        "这个算法能不能作为我的对比算法？",
        "这篇论文如何复现？",
        "继续分析它的实验设置。",
        "和上面提到的方法相比，它的优势是什么？",
    ]
    quick_question = st.selectbox("选择问题", ["不使用"] + question_templates)
    if quick_question != "不使用" and st.button("发送快捷问题", type="primary"):
        st.session_state.pending_quick_question = quick_question

with st.expander("科研分析任务", expanded=False):
    selected_task = st.selectbox("选择分析任务", get_task_names())
    task_prompt = get_task_prompt(selected_task)
    st.text_area("任务提示", value=task_prompt, height=140, disabled=True)
    custom_focus = st.text_input(
        "补充关注点",
        placeholder="例如：重点判断是否适合作为家庭医疗路径规划问题的对比算法",
    )
    task_question = task_prompt if not custom_focus.strip() else f"{task_prompt}\n补充关注点：{custom_focus.strip()}"
    if st.button("发送分析任务", type="primary"):
        st.session_state.pending_quick_question = task_question

chat_question = st.chat_input("继续追问当前论文知识库...")
pending_question = (chat_question or "").strip()
if not pending_question:
    pending_question = st.session_state.pop("pending_quick_question", "").strip()

if pending_question:
    pending_route, _ = select_route(pending_question)
    if session_data["rag"] is None and pending_route != "search":
        st.warning("请先上传 PDF 并构建论文知识库。")
    else:
        with st.spinner("正在结合会话上下文检索论文片段并生成回答..."):
            workflow = PaperAgentWorkflow(
                session_data["rag"],
                get_session_dir(user_id, active_session_id) / "agent_checkpoints.sqlite",
                get_user_dir(user_id) / "agent_memory.sqlite3",
            )
            try:
                agent_result = workflow.invoke(
                    pending_question,
                    user_id=user_id,
                    session_id=active_session_id,
                    history=session_data["history"],
                    top_k=top_k,
                )
            finally:
                workflow.close()
            answer = agent_result.get("answer", "")
            sources = result_sources(agent_result)
            report_path = save_markdown_report(
                user_id,
                active_session_id,
                pending_question,
                answer,
                sources,
            )
            source_details = serialize_sources(sources)
            session_data["history"].append(
                {
                    "question": pending_question,
                    "answer": answer,
                    "source_count": len(sources),
                    "report_path": str(report_path),
                    "sources": source_details,
                    "agent_route": agent_result.get("route", ""),
                    "agent_events": agent_result.get("events", []),
                    "verification": agent_result.get("verification", {}),
                    "pending_action": agent_result.get("pending_action", {}),
                }
            )
            session_data["last_sources"] = source_details
            save_session_history(user_id, active_session_id, session_data["history"])
            upsert_session_meta(
                user_id,
                active_session_id,
                question_count=len(session_data["history"]),
            )

        st.subheader("最新回答")
        with st.chat_message("user"):
            st.write(pending_question)
        with st.chat_message("assistant"):
            st.write(answer)
            verification = agent_result.get("verification", {})
            if verification:
                st.caption(
                    f"Agent路由：{agent_result.get('route', 'unknown')} · "
                    f"核验：{verification.get('status', 'unknown')}"
                )
            st.download_button(
                "下载本轮 Markdown 报告",
                data=report_path.read_text(encoding="utf-8"),
                file_name=report_path.name,
                mime="text/markdown",
            )
            pending_action = agent_result.get("pending_action", {})
            if pending_action and pending_action.get("status") == "pending":
                with st.form(f"approval_{pending_action['action_id']}"):
                    selected_text = st.text_input("输入待下载入库的论文序号", placeholder="例如：1,3")
                    approved = st.form_submit_button("确认下载并入库", type="primary")
                if approved:
                    selections = [
                        int(value) for value in selected_text.replace("，", ",").split(",")
                        if value.strip().isdigit()
                    ]
                    approval_workflow = PaperAgentWorkflow(
                        session_data["rag"],
                        get_session_dir(user_id, active_session_id) / "agent_checkpoints.sqlite",
                        get_user_dir(user_id) / "agent_memory.sqlite3",
                    )
                    try:
                        approval_result = approval_workflow.resume_approval(
                            pending_action["action_id"], selections,
                            index_dir=str(get_index_dir(user_id, active_session_id)),
                        )
                        session_data["rag"] = approval_workflow.rag
                    finally:
                        approval_workflow.close()
                    st.success(approval_result.get("answer", "入库操作完成。"))
                    st.rerun()

        with st.expander("本轮检索来源", expanded=False):
            render_retrieval_sources(
                sources,
                f"latest_source_{active_session_id}_{len(session_data['history'])}",
                use_expanders=False,
            )
