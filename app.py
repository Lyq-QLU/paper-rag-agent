from pathlib import Path
from html import escape
from datetime import date

import streamlit as st

from src.analysis_tasks import get_task_names, get_task_prompt
from src.config import get_openalex_api_key, save_api_config, save_openalex_api_key
from src.evaluation import evaluate_retrieval, parse_evaluation_cases
from src.agent_workflow import PaperAgentWorkflow, result_sources, select_route
from src.agent_memory import AgentMemoryStore
from src.llm import get_llm_status, test_llm_connection
from src.paper_loader import load_pdf_documents
from src.paper_library import (
    search_papers, load_library, set_saved, download_selected,
    make_reading_card, comparison_markdown, source_error_message,
)
from src.providers import build_secrets_example, get_preset_by_name, get_provider_options
from src.venue_rankings import classify_venue, journal_rankings, import_rankings, RANKING_TEMPLATE
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
    page_title="Paper Atlas · 论文研究工作台",
    page_icon="📖",
    layout="wide",
)


def inject_theme() -> None:
    st.markdown("""
    <style>
    :root { --ink:#18352e; --muted:#76817b; --line:#e3e7e0; --paper:#fafbf7; --accent:#22634d; }
    .stApp { background:var(--paper); color:var(--ink); }
    header[data-testid="stHeader"] { background:rgba(250,251,247,.92); }
    [data-testid="stToolbar"] { display:none; }
    [data-testid="stMainBlockContainer"], .main .block-container {
        max-width:1220px; padding:4rem 3rem 5rem;
    }
    section[data-testid="stSidebar"] { background:#f0f3ed; border-right:1px solid var(--line); }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top:1.5rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
    h3 { font-size:1.2rem !important; }
    p,li { line-height:1.7; }
    .atlas-brand { display:flex; align-items:center; gap:10px; margin-bottom:26px; color:var(--ink); }
    .atlas-mark { display:grid; place-items:center; width:36px; height:36px; background:var(--ink);
        color:#edf3d9; border-radius:11px; font-size:19px; font-family:Georgia,serif; }
    .atlas-brand b { letter-spacing:-.03em; font-size:19px; }
    .atlas-brand span { color:var(--muted); font-size:12px; margin-left:4px; }
    .atlas-hero { padding:28px 32px; border-radius:22px; background:#193e32; color:#f4f7ec;
        position:relative; overflow:hidden; margin-bottom:24px; }
    .atlas-hero:after { content:""; position:absolute; width:230px; height:230px; border:1px solid #547561;
        border-radius:50%; right:-40px; top:-100px; box-shadow:0 0 0 35px #ffffff04,0 0 0 70px #ffffff04; pointer-events:none; }
    .atlas-eyebrow { font-size:11px; letter-spacing:.16em; color:#bed1ac; font-weight:600; }
    .atlas-hero h1 { color:#f4f7ec; font-size:clamp(25px,3.1vw,36px); line-height:1.35; margin:9px 0 8px; padding:0; }
    .atlas-hero p { margin:0; color:#c6d5c8; font-size:14px; }
    .atlas-session { font-size:12px; color:#9ebaa5; margin-top:24px; }
    .atlas-empty { background:white; border:1px dashed #cfd8ca; border-radius:18px; padding:34px 28px;
        text-align:center; margin:18px 0; }
    .atlas-empty b { display:block; font-size:19px; color:var(--ink); margin-bottom:8px; }
    .atlas-empty p { color:var(--muted); font-size:14px; margin:0; }
    .atlas-section { margin:22px 0 14px; }
    .atlas-section h2 { font-size:22px; margin:0 0 5px; padding:0; }
    .atlas-section p { color:var(--muted); font-size:13px; margin:0; }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:15px; padding:16px 20px; }
    div[data-testid="stMetricValue"] { font-size:27px; color:var(--ink); }
    div[data-testid="stMetricLabel"] { color:var(--muted); font-size:12px; }
    div[data-testid="stTabs"] [role="tablist"] { gap:22px; border-bottom:1px solid var(--line); margin:10px 0 8px; }
    div[data-testid="stTabs"] button[role="tab"] { background:transparent; padding:14px 4px; font-size:15px; font-weight:600; color:#7b857d; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--accent); }
    div[data-baseweb="tab-highlight"] { background:var(--accent); height:3px; }
    div[data-testid="stForm"] { border:1px solid var(--line); border-radius:18px; padding:24px; background:white; }
    div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:#ffffffb3; }
    div[data-testid="stExpander"] summary { padding:13px 16px; }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] { background:transparent; border-color:#dce3d7; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:16px !important; }
    .stButton > button,.stDownloadButton > button,.stLinkButton > a, [data-testid="stFormSubmitButton"] button {
        border-radius:10px; border:1px solid #d8e0d4; min-height:42px; font-weight:500; }
    button[kind="primary"],button[data-testid="stBaseButton-primaryFormSubmit"] { background:var(--accent) !important; border-color:var(--accent) !important; color:white !important; }
    button:hover { border-color:#719a7b !important; }
    input,textarea { color:var(--ink) !important; }
    [data-baseweb="input"],[data-baseweb="select"] > div,[data-baseweb="textarea"] { border-radius:10px !important; background:#f6f8f3 !important; }
    [data-baseweb="tag"] { background:#e4eddf !important; color:#2c573c !important; border-radius:6px; }
    div[data-testid="stChatMessage"] { border:1px solid var(--line); background:white; border-radius:16px; padding:20px; margin:10px 0; }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { line-height:1.8; }
    [data-testid="stChatInput"] { border-radius:16px; border:1px solid #cbd7c6; }
    [data-testid="stBottom"] { background:var(--paper); }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    .rag-pill-row { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
    .rag-pill { background:#edf2e7; color:#47634b; padding:5px 12px; border-radius:20px; font-size:12px; }
    .rag-section-note,.rag-workspace-note { color:var(--muted); font-size:13px; }
    @media(max-width:760px) {
        [data-testid="stMainBlockContainer"],.main .block-container { padding:4rem 1rem 4rem; }
        [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) { flex-wrap:wrap; gap:10px; }
        [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
            width:calc(50% - 5px) !important; min-width:0 !important; flex:1 1 calc(50% - 5px) !important;
        }
        div[data-testid="stMetric"] { padding:12px 16px; }
        .atlas-hero { padding:24px; border-radius:17px; }
        .atlas-brand { margin-bottom:18px; }
        .atlas-brand span { display:none; }
        div[data-testid="stForm"] { padding:16px; }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:12px; }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            width:auto !important; min-width:140px !important; flex:1 1 140px !important;
        }
        div[data-testid="stTabs"] [role="tablist"] { gap:16px; }
    }
    </style>
    """, unsafe_allow_html=True)


def render_app_title(session_title: str = "") -> None:
    st.markdown('<div class="atlas-brand"><div class="atlas-mark">P</div><b>Paper Atlas</b><span>论文研究工作台</span></div>', unsafe_allow_html=True)
    session = f'<div class="atlas-session">当前研究 · {escape(session_title)}</div>' if session_title else ""
    st.markdown(f'<div class="atlas-hero"><div class="atlas-eyebrow">YOUR RESEARCH, WITH EVIDENCE</div><h1>让每一次阅读，都有据可循。</h1><p>发现相关论文，整理研究证据，从单篇精读走向多篇对比。</p>{session}</div>', unsafe_allow_html=True)


def render_section(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="atlas-section"><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


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
    st.markdown("### 研究资料")
    st.caption("上传论文，建立你的研究上下文")
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
        st.caption("还没有研究资料，可先搜索论文或上传 PDF")

    with st.expander("论文与索引", expanded=session_data["rag"] is None):
        uploaded_files = st.file_uploader(
            "上传 PDF 论文",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_uploader_{active_session_id}",
        )

        build_index = st.button("构建知识库", type="primary", use_container_width=True)
        rebuild_saved_index = st.button("重建当前索引", use_container_width=True)

        if active_uploaded_pdfs:
            for file_info in active_uploaded_pdfs:
                st.caption(f"{file_info['name']} | {file_info['size_kb']} KB")

    with st.expander("任务运行设置", expanded=False):
        top_k = st.slider("检索片段数量", min_value=3, max_value=12, value=8)
        research_mode = st.toggle("科研任务模式", value=bool(llm_status["configured"]),
                                  help="模型连续选择工具，支持多步骤任务、证据检查与中断恢复。")
        task_max_steps = st.number_input("任务步数上限", min_value=1, max_value=30, value=10)
        task_max_seconds = st.number_input("任务时间预算（秒）", min_value=10, max_value=900, value=180)
        task_max_tools = st.number_input("工具调用上限", min_value=1, max_value=25, value=8)
        task_max_models = st.number_input("模型调用上限", min_value=3, max_value=40, value=14)

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
            pdf_paths = get_uploaded_pdf_paths(user_id, active_session_id)
            documents = load_pdf_documents(pdf_paths)
            rag = PaperRAG(user_id=user_id)
            rag.build_index(documents)
            rag.save(get_index_dir(user_id, active_session_id))
            session_data["rag"] = rag
            save_session_history(user_id, active_session_id, session_data["history"])
            upsert_session_meta(
                user_id,
                active_session_id,
                title=uploaded_files[0].name if uploaded_files else "论文问答会话",
                chunk_count=len(rag.chunks),
                paper_count=len(pdf_paths),
                question_count=len(session_data["history"]),
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
            save_session_history(user_id, active_session_id, session_data["history"])
            upsert_session_meta(
                user_id,
                active_session_id,
                chunk_count=len(rag.chunks),
                paper_count=len(saved_pdf_paths),
                question_count=len(session_data["history"]),
            )
        counts = get_content_type_counts(session_data["rag"])
        st.success(
            f"索引已重建：正文 {counts.get('text', 0)} 块，"
            f"表格 {counts.get('table', 0)} 块，图片 {counts.get('figure', 0)} 块。"
        )

active_uploaded_pdfs = list_uploaded_pdfs(user_id, active_session_id)

render_app_title(current_meta.get("title", "新会话"))
library_stats = load_library(user_id, active_session_id)
stats = st.columns(4)
stats[0].metric("研究论文", len(active_uploaded_pdfs))
stats[1].metric("已收藏", sum(bool(p.get("saved")) for p in library_stats["papers"].values()))
stats[2].metric("精读卡片", len(library_stats["cards"]))
stats[3].metric("研究问答", len(session_data["history"]))
discovery_tab, reading_tab, chat_tab = st.tabs(["论文发现", "精读与对比", "论文问答"])

with discovery_tab:
    render_section("找到下一篇值得读的论文", "设置研究关键词和时间范围，从真实学术来源建立候选清单。")
    library = load_library(user_id, active_session_id)
    previous_search = library.get("last_search", {})
    with st.form(f"discovery_{active_session_id}"):
        search_terms = st.text_input("研究关键词", value=previous_search.get("keywords", ""),
                                     placeholder="例如 multimodal retrieval augmented generation")
        year_cols = st.columns(2)
        with year_cols[0]:
            year_from = st.number_input("起始年份", min_value=1900, max_value=2100,
                                        value=int(previous_search.get("year_from", date.today().year - 3)))
        with year_cols[1]:
            year_to = st.number_input("截止年份", min_value=1900, max_value=2100,
                                      value=int(previous_search.get("year_to", date.today().year)))
        kind_names = {"不限": "all", "综述（仅来源确认）": "review",
                      "研究论文（仅来源确认）": "article", "arXiv 预印本": "preprint"}
        filter_cols = st.columns(2)
        with filter_cols[0]:
            kind_label = st.selectbox("论文类型", list(kind_names))
        with filter_cols[1]:
            selected_sources = st.multiselect("查询来源", ["arxiv", "openalex", "crossref"],
                                              default=list(dict.fromkeys(previous_search.get("sources", ["arxiv", "openalex"]) + ["crossref"])))
        venue_cols = st.columns(2)
        with venue_cols[0]:
            venue_kind_names = {"全部发表类型": "all", "期刊论文": "journal", "会议论文": "conference", "预印本": "preprint"}
            venue_kind_label = st.selectbox("发表类型", list(venue_kind_names))
        with venue_cols[1]:
            venue_query = st.text_input("期刊 / 会议名称", value=previous_search.get("venue_query", ""),
                                        placeholder="例如 IEEE Transactions… 或 ICML")
        rank_cols = st.columns(3)
        with rank_cols[0]:
            ccf_levels = st.multiselect("CCF 等级 · 2026 版", ["A", "B", "C"], default=previous_search.get("ccf_levels", []))
        with rank_cols[1]:
            jcr_quartiles = st.multiselect("JCR 分区", ["Q1", "Q2", "Q3", "Q4"], default=previous_search.get("jcr_quartiles", []))
        with rank_cols[2]:
            cas_zones = st.multiselect("中科院分区", ["1", "2", "3", "4"], default=previous_search.get("cas_zones", []))
        available_rank_years = sorted({r["year"] for r in journal_rankings(user_id)}, reverse=True)
        ranking_year_label = st.selectbox("期刊分区年份", ["各体系最新已导入年份"] + [str(y) for y in available_rank_years])
        st.caption("Crossref 提供 DOI 登记的期刊/会议元数据；登记信息不等同于同行评审证明。CCF 匹配刊会目录，会议长文/短文类型需另行核实。JCR 与中科院分区按学科、年份独立记录；无数据时显示未核实。")
        search_clicked = st.form_submit_button("查询真实论文", type="primary")
    if search_clicked:
        try:
            with st.spinner("正在查询论文来源..."):
                search_papers(user_id, active_session_id, keywords=search_terms,
                              year_from=int(year_from), year_to=int(year_to),
                              paper_type=kind_names[kind_label], sources=tuple(selected_sources),
                              venue_kind=venue_kind_names[venue_kind_label], venue_query=venue_query,
                              ccf_levels=tuple(ccf_levels), jcr_quartiles=tuple(jcr_quartiles), cas_zones=tuple(cas_zones),
                              ranking_year=int(ranking_year_label) if ranking_year_label.isdigit() else None)
            st.rerun()
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    with st.expander("论文来源设置"):
        st.caption("OpenAlex API Key 已配置" if get_openalex_api_key(user_id) else "OpenAlex 当前使用匿名额度；遇到限流可配置免费 API Key。")
        st.markdown("[获取 OpenAlex API Key](https://openalex.org/settings/api)")
        openalex_key = st.text_input("OpenAlex API Key", type="password", key="openalex_api_key_input")
        if st.button("保存 OpenAlex API Key"):
            try:
                save_openalex_api_key(openalex_key, user_id)
                st.success("OpenAlex API Key 已保存，请重新查询。")
            except ValueError as exc:
                st.error(str(exc))
    with st.expander("分区数据与来源"):
        st.caption("内置 CCF 2026 年正式目录 681 条记录。期刊分区使用你导入的授权数据，保留学科、年份与来源，不由模型猜测。")
        st.markdown("[查看 CCF 正式目录原文](https://lib.zjgsu.edu.cn/_upload/article/files/f5/b1/f4f7201343b88c0f10564f590ebe/854a3c68-36a6-4698-a359-5184d5e30a0c.pdf)")
        ranking_upload = st.file_uploader("导入期刊分区 CSV", type=["csv"], key="rankings_csv")
        st.caption("scheme 填 JCR 或 CAS；level 分别填 Q1–Q4 或 1–4；category 中注明学科及中科院大类/小类口径。导入记录标记为用户提供。")
        st.download_button("下载 CSV 表头模板", RANKING_TEMPLATE, "journal_rankings_template.csv", mime="text/csv")
        if st.button("导入分区表", disabled=ranking_upload is None):
            try:
                count = import_rankings(user_id, ranking_upload.getvalue())
                st.success(f"已导入 {count} 条分区记录，可重新查询应用筛选。")
            except (ValueError, UnicodeError) as exc:
                st.error(str(exc))
    papers = library.get("papers", {})
    last = library.get("last_search", {})
    if last:
        for source_name, error in last.get("errors", {}).items():
            st.warning(error)
        shown = [papers[key] for key in last.get("ids", []) if key in papers]
        st.caption(f"查询结果 {len(shown)} 篇，符合已核实筛选条件 {sum(bool(p.get('included')) for p in shown)} 篇。主题相关性请结合摘要确认。")
        if sum(bool(p.get("included")) for p in shown) < 3:
            st.info("结果较少：可放宽年份，或尝试同义关键词、review、survey。")
        chosen_ids = []
        for paper in shown:
            title = paper.get("title", "未知标题")
            label = f"{title} · {paper.get('published', '年份未知')} · {', '.join(paper.get('versions', []))}"
            with st.container(border=True):
                chosen = st.checkbox(label, key=f"paper_select_{active_session_id}_{paper['id']}")
                if chosen:
                    chosen_ids.append(paper["id"])
                st.caption(paper.get("filter_reason", ""))
                st.caption("作者：" + (", ".join(paper.get("authors", [])) or "未提供"))
                publication = classify_venue(paper, journal_rankings(user_id))
                kind_display = {"journal": "期刊", "conference": "会议", "preprint": "预印本", "unknown": "待核实"}.get(publication["kind"], "待核实")
                st.write(f"**发表刊会**：{paper.get('venue') or '尚未核实正式发表刊会'} · {kind_display}")
                if paper.get("journal_ref"):
                    st.caption("arXiv 作者填写的发表信息（待核实）：" + paper["journal_ref"])
                if publication["ccf"]:
                    rank = publication["ccf"]
                    st.markdown(f"**CCF {rank['level']} · {rank['year']}版** · [目录第 {rank['page']} 页]({rank['source_url']})")
                else:
                    st.caption("CCF：未匹配到目录记录")
                for scheme, label in (("jcr", "JCR"), ("cas", "中科院")):
                    if publication[scheme]:
                        for rank in publication[scheme]:
                            st.markdown(f"{label} **{rank['level']}** · {rank['year']} · {rank['category']} · [用户导入来源]({rank['source_url']})")
                    else:
                        st.caption(f"{label}分区：未核实")
                if paper.get("summary"):
                    st.write(paper["summary"])
                if paper.get("publication_note"):
                    st.warning(paper["publication_note"])
                if paper.get("url"):
                    st.markdown(f"[原始页面]({paper['url']})")
                if paper.get("pdf_url"):
                    st.markdown(f"[PDF 链接]({paper['pdf_url']})")
                st.caption("已收藏" if paper.get("saved") else "未收藏")
                if paper.get("downloaded_path"):
                    st.caption("已下载：" + Path(paper["downloaded_path"]).name)
                if paper.get("download_error"):
                    st.error("上次下载失败：" + paper["download_error"])
        actions = st.columns(3)
        if actions[0].button("收藏选中论文", disabled=not chosen_ids):
            set_saved(user_id, active_session_id, chosen_ids)
            st.rerun()
        if actions[1].button("取消收藏", disabled=not chosen_ids):
            set_saved(user_id, active_session_id, chosen_ids, saved=False)
            st.rerun()
        if actions[2].button("下载选中论文", disabled=not chosen_ids):
            with st.spinner("正在下载所选 arXiv PDF..."):
                results = download_selected(user_id, active_session_id, chosen_ids)
            for result in results:
                if result["status"] in {"downloaded", "already_downloaded"}:
                    st.success(f"{result['id']}：{result['status']}")
                else:
                    st.warning(f"{result['id']}：{result.get('message', result['status'])}")
            st.info("下载后请点击左侧‘重建当前索引’，再生成精读卡片。")
    saved_papers = [p for p in papers.values() if p.get("saved")]
    if saved_papers:
        render_section("我的收藏", "保存待阅读论文，逐步构建你的研究清单。")
        st.dataframe([{"标题": p["title"], "年份": p.get("published"),
                       "来源": ", ".join(p.get("versions", [])),
                       "状态": "已下载" if p.get("downloaded_path") else "待下载"}
                      for p in saved_papers], use_container_width=True, hide_index=True)

with reading_tab:
    render_section("从阅读到研究洞察", "为每篇论文建立证据卡片，再按方法、实验和局限逐项对比。")
    if session_data["rag"] is None:
        st.markdown('<div class="atlas-empty"><b>从一篇论文开始</b><p>在左侧上传 PDF 并构建知识库，即可生成精读卡片与对比表。</p></div>', unsafe_allow_html=True)
    if session_data["rag"] is not None:
        indexed_sources = sorted({str(c.metadata.get("source", "")) for c in session_data["rag"].chunks})
        selected_paper = st.selectbox("生成精读卡片：选择已索引论文", indexed_sources)
        if st.button("生成精读卡片", disabled=not selected_paper):
            try:
                with st.spinner("正在逐项提取证据..."):
                    make_reading_card(user_id, active_session_id, selected_paper, session_data["rag"])
                st.rerun()
            except Exception as exc:
                st.error(f"精读卡片生成失败：{exc}")
    cards = library.get("cards", {})
    if cards:
        render_section("已生成的精读卡片", "每条信息关联原文证据；未找到的信息会明确标记。")
        card_name = st.selectbox("查看论文", list(cards))
        card = cards[card_name]
        for field, label in zip(("research_question", "method", "datasets", "metrics", "results", "limitations", "reproducibility"),
                                ("研究问题", "核心方法", "数据集", "评价指标", "主要结果", "局限性", "复现条件")):
            entry = card["fields"][field]
            refs = ", ".join(f"第{e['page']}页" for e in entry.get("evidence", []))
            st.write(f"**{label}**：{entry['text']}" + (f"（{refs}）" if refs else ""))
        comparison_names = st.multiselect("选择卡片进行对比", list(cards),
                                          default=list(cards)[:2] if len(cards) >= 2 else [])
        if len(comparison_names) >= 2:
            comparison = comparison_markdown([cards[name] for name in comparison_names])
            st.markdown(comparison)
            st.download_button("下载对比表 Markdown", comparison, file_name="paper_comparison.md",
                               mime="text/markdown")

with chat_tab:

    st.markdown(f"### {current_meta.get('title', '新会话')}")
    chat_status = "知识库已就绪" if session_data["rag"] is not None else "未构建知识库"
    st.caption(f"{chat_status} · {len(session_data['history'])} 条问答")

    if session_data["rag"] is not None:
        with st.expander("知识库内容与图表预览", expanded=False):
            render_knowledge_overview(session_data["rag"])

    if session_data["history"]:
        for history_index, item in enumerate(session_data["history"]):
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                st.caption(f"检索片段数：{item.get('source_count', 0)}")
                if item.get("harness"):
                    artifact_path = item["harness"].get("artifact_path")
                    if artifact_path and Path(artifact_path).is_file():
                        st.download_button("下载任务结果", Path(artifact_path).read_text(encoding="utf-8"),
                            file_name=f"research_{item['harness']['run_id']}.md", mime="text/markdown",
                            key=f"task_artifact_{history_index}_{item['harness']['run_id']}")
                    with st.expander(f"任务轨迹 · {item['harness']['run_id']}"):
                        st.json({key: item["harness"].get(key) for key in
                                 ("status", "active_skill", "skills_used", "requirements", "usage", "completion", "trace")})
    else:
        with st.chat_message("assistant"):
            if session_data["rag"] is None:
                st.write("可以先到「论文发现」寻找资料，或上传 PDF 构建知识库。科研任务模式也支持直接描述你的研究目标。")
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
        if session_data["rag"] is None and pending_route != "search" and not research_mode:
            st.warning("请先上传 PDF 并构建论文知识库。")
        else:
            with st.spinner("正在结合会话上下文检索论文片段并生成回答..."):
                workflow = PaperAgentWorkflow(
                    session_data["rag"],
                    get_session_dir(user_id, active_session_id) / "harness_workspace",
                    get_user_dir(user_id) / "agent_memory.sqlite3",
                )
                try:
                    agent_result = workflow.invoke(
                        pending_question,
                        user_id=user_id,
                        session_id=active_session_id,
                        history=session_data["history"],
                        top_k=top_k,
                        index_dir=str(get_index_dir(user_id, active_session_id)),
                        execution_mode="harness" if research_mode else "workflow",
                        budget={"max_steps": int(task_max_steps), "max_seconds": int(task_max_seconds),
                                "max_tool_calls": int(task_max_tools), "max_model_calls": int(task_max_models)},
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                finally:
                    workflow.close()
                session_data["rag"] = workflow.rag
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
                        "harness": agent_result.get("harness", {}),
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
                run = agent_result.get("harness", {})
                if run:
                    st.caption(f"任务 {run['run_id']} · {run['status']} · 工具 {run['usage']['tool_calls']} 次 · 模型 {run['usage']['model_calls']} 次")
                    with st.expander("科研任务执行轨迹"):
                        st.json({"active_skill": run.get("active_skill"),
                                 "skills_used": run.get("skills_used", []),
                                 "requirements": run["requirements"], "usage": run["usage"],
                                 "completion": run.get("completion", {}), "trace": run["trace"]})
            with st.expander("本轮检索来源", expanded=False):
                render_retrieval_sources(
                    sources,
                    f"latest_source_{active_session_id}_{len(session_data['history'])}",
                    use_expanders=False,
                )


    # Persistent approvals must render on every rerun, not only while submitting chat.
    approval_store = AgentMemoryStore(get_user_dir(user_id) / "agent_memory.sqlite3")
    try:
        pending_approvals = [item for item in approval_store.list_pending(user_id)
                             if item["thread_id"] == active_session_id and item["status"] in {"pending", "failed", "running"}]
    finally:
        approval_store.close()
    if pending_approvals:
        with st.expander("待确认论文入库", expanded=True):
            for pending in pending_approvals:
                st.write("原任务：" + str(pending["payload"].get("query", "论文入库")))
                for index, paper in enumerate(pending["payload"].get("papers", []), 1):
                    st.write(f"{index}. {paper['title']}")
                with st.form(f"persistent_approval_{pending['action_id']}"):
                    selection_text = st.text_input("论文序号（例如1,3；留空取消）")
                    submit = st.form_submit_button("确认选择并继续任务")
                if submit:
                    selections = [int(value.strip()) for value in selection_text.replace("，", ",").split(",")
                                  if value.strip().isdigit()]
                    approval_workflow = PaperAgentWorkflow(session_data["rag"],
                        get_session_dir(user_id, active_session_id) / "harness_workspace",
                        get_user_dir(user_id) / "agent_memory.sqlite3")
                    try:
                        result = approval_workflow.resume_approval(pending["action_id"], selections,
                            user_id=user_id, session_id=active_session_id,
                            index_dir=str(get_index_dir(user_id, active_session_id)),
                            budget={"max_steps": int(task_max_steps), "max_seconds": int(task_max_seconds),
                                    "max_tool_calls": int(task_max_tools), "max_model_calls": int(task_max_models)})
                        session_data["rag"] = approval_workflow.rag
                        session_data["history"].append({"question": "确认论文选择并继续任务",
                            "answer": result.get("answer", ""), "source_count": len(result.get("sources", [])),
                            "sources": serialize_sources(result_sources(result)), "harness": result.get("harness", {}),
                            "pending_action": result.get("pending_action", {}), "verification": result.get("verification", {})})
                        save_session_history(user_id, active_session_id, session_data["history"])
                        st.rerun()
                    except Exception as exc:
                        st.error(f"执行失败，可重试：{exc}")
                    finally:
                        approval_workflow.close()


    with st.expander("继续已保存的科研任务"):
        saved_tasks = sorted((get_session_dir(user_id, active_session_id) / "tasks").glob("*.json"),
                             key=lambda path: path.stat().st_mtime, reverse=True)
        if not saved_tasks:
            st.caption("当前会话尚无科研任务。")
        else:
            selected_task = st.selectbox("任务ID", [path.stem for path in saved_tasks])
            task_message = st.text_input("补充说明", key="task_resume_message")
            st.caption("恢复时沿用已有调用次数；新的步数和时间上限取自侧栏。待审批任务请先确认论文选择。")
            if st.button("继续科研任务"):
                resumed_workflow = PaperAgentWorkflow(session_data["rag"],
                    get_session_dir(user_id, active_session_id) / "harness_workspace",
                    get_user_dir(user_id) / "agent_memory.sqlite3")
                try:
                    resumed = resumed_workflow.resume_task(selected_task, user_id=user_id,
                        session_id=active_session_id, index_dir=str(get_index_dir(user_id, active_session_id)),
                        message=task_message,
                        budget={"max_steps": int(task_max_steps), "max_seconds": int(task_max_seconds),
                                "max_tool_calls": int(task_max_tools), "max_model_calls": int(task_max_models)})
                    session_data["rag"] = resumed_workflow.rag
                    session_data["history"].append({"question": task_message or "继续科研任务",
                        "answer": resumed.get("answer", ""), "source_count": len(resumed.get("sources", [])),
                        "sources": serialize_sources(result_sources(resumed)),
                        "harness": resumed.get("harness", {}), "verification": resumed.get("verification", {}),
                        "pending_action": resumed.get("pending_action", {})})
                    save_session_history(user_id, active_session_id, session_data["history"])
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
                finally:
                    resumed_workflow.close()
