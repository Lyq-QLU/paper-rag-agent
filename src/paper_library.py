"""Session-scoped scholarly discovery, reading cards, and comparisons."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock

from src.llm import call_agent_model
from src.config import get_openalex_api_key
from src.venue_rankings import classify_venue, journal_rankings
from src.session_manager import atomic_write_text_or_bytes, get_session_dir, register_pdf


CARD_FIELDS = ("research_question", "method", "datasets", "metrics", "results", "limitations", "reproducibility")
CARD_LABELS = ("研究问题", "核心方法", "数据集", "评价指标", "主要结果", "局限性", "复现条件")

CARD_SEARCH = {
    "research_question": ("research problem objective home health care routing scheduling optimization", ("this paper addresses", "the problem", "objective", "research question"), ("introduction", "front_matter")),
    "method": ("MOMBO migrating birds optimization algorithm external archive procedure", ("mombo", "migrating birds", "external archive", "proposed algorithm", "procedure"), ("method", "experiment")),
    "datasets": ("benchmark test instances dataset experimental setting", ("benchmark", "test instances", "instances", "dataset", "data set", "solomon"), ("experiment",)),
    "metrics": ("evaluation metrics C-metric IGD hypervolume statistical test", ("metric", "igd", "hypervolume", "evaluation", "t-test"), ("experiment",)),
    "results": ("experimental results comparison outperform conclusion", ("results show", "outperform", "better than", "compared with", "experimental results", "comparison experiments", "superior"), ("discussion", "conclusion", "experiment")),
    "limitations": ("limitations future work threats to validity", ("limitation", "future work", "future research", "further study", "drawback"), ("discussion", "conclusion")),
    "reproducibility": ("code data availability parameters implementation settings", ("code", "data availability", "parameter", "implementation", "available on request"), ("experiment", "back_matter")),
}


def _card_evidence(rag, source: str, field: str) -> list[dict]:
    query, terms, preferred_sections = CARD_SEARCH[field]
    pool = [chunk for chunk in rag.chunks if chunk.metadata.get("source") == source]
    ranked = []
    for chunk in pool:
        body = chunk.text.lower()
        section = str(chunk.metadata.get("section") or "").lower()
        if section == "references" or body.startswith("[章节：references]"):
            continue
        matches = [term for term in terms if term in body]
        if not matches:
            continue
        score = len(matches) * 3 + min(sum(body.count(term) for term in matches), 4)
        if any(name in section for name in preferred_sections):
            score += 8
        if field == "results" and ("discussion" in section or "conclusion" in section):
            score += 10
        if field == "results" and ("fig. a" in body or "appendix" in body):
            score -= 12
        if len(chunk.text) < 180:
            score -= 8
        ranked.append((score, chunk, matches[0]))
    ranked.sort(key=lambda row: row[0], reverse=True)
    selected = [(chunk, term) for _, chunk, term in ranked[:3 if field == "results" else 2]]
    for chunk in rag.retrieve_by_mode(query, top_k=2, mode="hybrid", max_per_source=12, source_filter=[source]):
        if str(chunk.metadata.get("section") or "").lower() == "references":
            continue
        if re.search(r"\bReferences\b", chunk.text[:220], re.I):
            continue
        if all(chunk.metadata.get("chunk_id") != existing.metadata.get("chunk_id") for existing, _ in selected):
            selected.append((chunk, ""))
            break
    result = []
    for chunk, term in selected:
        text = chunk.text
        position = text.lower().find(term) if term else 0
        start = max(0, position - 150)
        result.append({"chunk_id": chunk.metadata["chunk_id"],
                       "page": chunk.metadata.get("page_start", chunk.metadata.get("page")),
                       "text": text[start:start + 1000]})
    return result


def source_error_message(source: str, detail: str) -> str:
    message = str(detail).lower()
    label = {"arxiv": "arXiv", "openalex": "OpenAlex", "crossref": "Crossref"}.get(source, source)
    if source == "openalex" and ("匿名额度已用完" in message or "当前额度已用完" in message):
        return str(detail)
    if "timed out" in message or "timeout" in message or "超时" in message:
        return f"{label} 暂时响应超时，可稍后重试；其他来源成功的结果仍会显示。"
    if source == "openalex" and ("额度" in message or "credits" in message):
        return "OpenAlex 今日查询额度已用完；请等待额度重置或更换有余额的 API Key。"
    if source == "openalex" and ("api key" in message or "请求次数受限" in message or "429" in message):
        return "OpenAlex 请求次数受限；可在论文来源设置中配置免费的 OpenAlex API Key，或稍后重试。"
    if "429" in message or "rate limit" in message:
        return f"{label} 请求次数受限，请稍后重试。"
    return f"{label} 查询暂不可用，请稍后重试。"


def _path(user_id: str, session_id: str) -> Path:
    return get_session_dir(user_id, session_id) / "paper_library.json"


def load_library(user_id: str, session_id: str) -> dict:
    path = _path(user_id, session_id)
    if not path.exists():
        return {"papers": {}, "last_search": {}, "cards": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {"papers": value.get("papers", {}), "last_search": value.get("last_search", {}),
            "cards": value.get("cards", {})}


def _update(user_id: str, session_id: str, transform):
    path = _path(user_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        library = load_library(user_id, session_id)
        result = transform(library)
        atomic_write_text_or_bytes(path, json.dumps(library, ensure_ascii=False, indent=2))
        return result


def _arxiv_id(paper: dict) -> str:
    for value in (paper.get("arxiv_id"), paper.get("url"), paper.get("pdf_url")):
        match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)(\d{4}\.\d{4,5})(?:v\d+)?", str(value or ""), re.I)
        if match:
            return match.group(1)
    return ""


def paper_id(paper: dict) -> str:
    arxiv = _arxiv_id(paper)
    if arxiv:
        return "arxiv:" + arxiv
    doi = str(paper.get("doi") or "").lower().strip().removeprefix("https://doi.org/")
    if doi:
        return "doi:" + doi
    title = re.sub(r"\W+", "", str(paper.get("title") or "").casefold())
    return "title:" + hashlib.sha256(title.encode()).hexdigest()[:16]


def _normalize(paper: dict) -> dict:
    item = {key: paper.get(key) for key in ("title", "authors", "published", "url", "pdf_url", "summary", "source", "doi", "openalex_id", "work_type", "venue", "venue_kind", "venue_issns", "venue_id", "journal_ref")}
    item["title"] = " ".join(str(item["title"] or "").split())
    item["authors"] = [str(name) for name in (item["authors"] or [])]
    for key in ("published", "url", "pdf_url", "summary", "source", "doi", "openalex_id", "work_type", "venue"):
        item[key] = str(item[key] or "")
    item["id"] = paper_id(item)
    item["arxiv_id"] = _arxiv_id(item)
    item["versions"] = [item["source"]] if item["source"] else []
    item["saved"] = False
    item["downloaded_path"] = ""
    item["download_error"] = ""
    item["venue_kind"] = item.get("venue_kind") or "unknown"
    item["venue_issns"] = item.get("venue_issns") or []
    return item


def _merge(old: dict, new: dict) -> dict:
    merged = dict(old)
    for key in ("title", "published", "url", "summary", "doi", "openalex_id", "work_type", "venue"):
        if not merged.get(key) and new.get(key):
            merged[key] = new[key]
    if merged.get("work_type") == "preprint" and new.get("work_type") not in {"", "preprint"}:
        merged["work_type"] = new["work_type"]
    if new.get("venue") and (not old.get("venue") or old.get("venue_kind") not in {"journal", "conference"}):
        for field in ("venue", "venue_kind", "venue_issns", "venue_id"):
            merged[field] = new.get(field)
    if new.get("pdf_url") and _arxiv_id(new) and not _arxiv_id(merged):
        merged["pdf_url"] = new["pdf_url"]
    elif not merged.get("pdf_url") and new.get("pdf_url"):
        merged["pdf_url"] = new["pdf_url"]
    if not merged.get("arxiv_id") and new.get("arxiv_id"):
        merged["arxiv_id"] = new["arxiv_id"]
    merged["versions"] = sorted(set(old.get("versions", []) + new.get("versions", [])))
    if "arxiv" in merged["versions"] and ({"openalex", "crossref"} & set(merged["versions"])):
        merged["publication_note"] = "可能同时有预印本与正式发表记录，请核对版本"
    return merged


def search_papers(user_id: str, session_id: str, *, keywords: str, year_from: int,
                  year_to: int, paper_type: str = "all", sources: tuple[str, ...] = ("arxiv", "openalex"),
                  per_source: int = 10, gateway=None, venue_kind: str = "all", venue_query: str = "",
                  ccf_levels: tuple[str, ...] = (), jcr_quartiles: tuple[str, ...] = (),
                  cas_zones: tuple[str, ...] = (), ranking_year: int | None = None) -> dict:
    keywords = " ".join(keywords.split())
    if not keywords or len(keywords) > 300:
        raise ValueError("请填写 1–300 字的关键词。")
    if year_from < 1900 or year_to > date.today().year or year_from > year_to:
        raise ValueError("年份范围无效。")
    if paper_type not in {"all", "review", "article", "preprint"}:
        raise ValueError("论文类型无效。")
    if not sources or not set(sources).issubset({"arxiv", "openalex", "crossref"}):
        raise ValueError("请至少选择一个有效来源。")
    if not 1 <= per_source <= 10:
        raise ValueError("每个来源最多查询 10 篇。")
    if venue_kind not in {"all", "journal", "conference", "preprint"}:
        raise ValueError("发表类型无效。")
    if not set(ccf_levels).issubset({"A", "B", "C"}) or not set(jcr_quartiles).issubset({"Q1", "Q2", "Q3", "Q4"}) or not set(cas_zones).issubset({"1", "2", "3", "4"}):
        raise ValueError("刊会等级筛选值无效。")
    if gateway is None:
        from src.mcp_client import MCPPaperToolGateway
        gateway = MCPPaperToolGateway(openalex_api_key=get_openalex_api_key(user_id))
    found, errors = [], {}
    for source in sources:
        try:
            arguments = {"query": keywords, "max_results": per_source,
                         "year_from": year_from, "year_to": year_to}
            if source == "crossref":
                arguments["venue_kind"] = venue_kind if venue_kind in {"journal", "conference"} else "all"
            result = gateway.call_tool(f"search_{source}", arguments)
            found.extend(_normalize({**item, "source": source}) for item in result.get("papers", []))
        except Exception as exc:
            errors[source] = source_error_message(source, str(exc))
    if len(errors) == len(sources):
        raise RuntimeError("所有论文来源均查询失败：" + "; ".join(errors.values()))
    unique = {}
    for item in found:
        key = item["id"]
        unique[key] = _merge(unique[key], item) if key in unique else item
    # Cross-source duplicates with different IDs are joined by normalized title.
    title_map = {}
    for key, item in list(unique.items()):
        title_key = re.sub(r"\W+", "", item["title"].casefold())
        if title_key in title_map:
            primary = title_map[title_key]
            unique[primary] = _merge(unique[primary], item)
            del unique[key]
        else:
            title_map[title_key] = key
    rows = []
    ranking_records = journal_rankings(user_id)
    for item in unique.values():
        reasons = []
        publication = classify_venue(item, ranking_records)
        item["publication"] = publication
        if venue_kind != "all" and publication["kind"] != venue_kind:
            reasons.append("发表类型不符合或尚未核实")
        if venue_query.strip() and venue_query.casefold().strip() not in str(item.get("venue", "")).casefold():
            reasons.append("刊会名称不匹配")
        if ccf_levels and (not publication["ccf"] or publication["ccf"]["level"] not in ccf_levels):
            reasons.append("CCF 等级不符合或尚未匹配")
        for scheme, selected in (("jcr", jcr_quartiles), ("cas", cas_zones)):
            records = publication[scheme]
            if ranking_year is not None:
                records = [record for record in records if record["year"] == ranking_year]
            elif records:
                latest = max(record["year"] for record in records)
                records = [record for record in records if record["year"] == latest]
            if selected and not any(record["level"] in selected for record in records):
                reasons.append(f"{scheme.upper()} 分区不符合或缺少对应年份数据")
        year = int(item["published"][:4]) if re.match(r"^\d{4}", item["published"]) else None
        if year is None:
            reasons.append("缺少可核验的发表年份")
        elif not year_from <= year <= year_to:
            reasons.append(f"年份 {year} 不在 {year_from}–{year_to} 范围内")
        work_type = item.get("work_type", "").lower()
        if paper_type == "preprint" and "arxiv" not in item["versions"]:
            reasons.append("没有 arXiv 预印本记录")
        elif paper_type == "review" and work_type not in {"review", "review-article"}:
            reasons.append("来源未确认其为综述论文")
        elif paper_type == "article" and work_type not in {"article", "journal-article", "proceedings-article"}:
            reasons.append("来源未确认其为研究论文")
        item["included"] = not reasons
        item["filter_reason"] = "符合已设置筛选条件；主题相关性需阅读摘要核实" if not reasons else "；".join(reasons)
        rows.append(item)
    rows.sort(key=lambda item: (not item["included"], item["published"]), reverse=False)
    def save(library):
        for item in rows:
            previous = library["papers"].get(item["id"], {})
            merged = _merge(previous, item) if previous else item
            merged["saved"] = bool(previous.get("saved"))
            merged["downloaded_path"] = previous.get("downloaded_path", "")
            merged["download_error"] = previous.get("download_error", "")
            for field in ("included", "filter_reason", "publication"):
                merged[field] = item[field]
            library["papers"][item["id"]] = merged
        library["last_search"] = {"keywords": keywords, "year_from": year_from, "year_to": year_to,
                                  "paper_type": paper_type, "sources": list(sources), "errors": errors,
                                  "venue_kind": venue_kind, "venue_query": venue_query,
                                  "ccf_levels": list(ccf_levels), "jcr_quartiles": list(jcr_quartiles),
                                  "cas_zones": list(cas_zones), "ranking_year": ranking_year,
                                  "ids": [item["id"] for item in rows],
                                  "searched_at": datetime.now(timezone.utc).isoformat()}
        return {"papers": [library["papers"][item["id"]] for item in rows], "errors": errors,
                "suggestions": [keywords + " review", keywords + " survey", "尝试放宽年份或使用同义关键词"] if len([x for x in rows if x["included"]]) < 3 else []}
    return _update(user_id, session_id, save)


def set_saved(user_id: str, session_id: str, ids: list[str], saved: bool = True) -> list[dict]:
    def apply(library):
        missing = set(ids) - set(library["papers"])
        if missing:
            raise ValueError("论文不在当前用户的候选清单中。")
        for key in ids:
            library["papers"][key]["saved"] = saved
        return [library["papers"][key] for key in ids]
    return _update(user_id, session_id, apply)


def download_selected(user_id: str, session_id: str, ids: list[str], gateway=None) -> list[dict]:
    if gateway is None:
        from src.mcp_client import MCPPaperToolGateway
        gateway = MCPPaperToolGateway()
    library = load_library(user_id, session_id)
    if set(ids) - set(library["papers"]):
        raise ValueError("论文不在当前用户的候选清单中。")
    results = []
    for key in dict.fromkeys(ids):
        paper = library["papers"][key]
        pdf_url = paper.get("pdf_url", "")
        parsed = urlparse(pdf_url)
        if parsed.scheme != "https" or parsed.hostname not in {"arxiv.org", "export.arxiv.org"}:
            results.append({"id": key, "status": "unavailable", "message": "仅支持自动下载 arXiv HTTPS PDF；请使用原始链接手动下载并上传。"})
            continue
        old_path = paper.get("downloaded_path", "")
        if old_path and Path(old_path).is_file():
            results.append({"id": key, "status": "already_downloaded", "path": old_path})
            continue
        try:
            result = gateway.call_tool("download_arxiv_pdf", {"pdf_url": pdf_url, "title": paper["title"],
                                                               "output_dir": str(get_session_dir(user_id, session_id) / "search_downloads")})
            path = Path(result["path"]).resolve()
            download_dir = (get_session_dir(user_id, session_id) / "search_downloads").resolve()
            if download_dir not in path.parents or not path.is_file():
                raise ValueError("下载工具返回了不属于当前会话的文件。")
            data = path.read_bytes()
            registered = register_pdf(data, paper["title"] + ".pdf", user_id, session_id)
            def mark(lib):
                lib["papers"][key].update({"downloaded_path": str(registered), "download_error": "", "saved": True})
            _update(user_id, session_id, mark)
            results.append({"id": key, "status": "downloaded", "path": str(registered)})
        except Exception as exc:
            def mark_error(lib):
                lib["papers"][key]["download_error"] = str(exc)
            _update(user_id, session_id, mark_error)
            results.append({"id": key, "status": "failed", "message": str(exc)})
    return results


def make_reading_card(user_id: str, session_id: str, source: str, rag) -> dict:
    names = {str(chunk.metadata.get("source", "")) for chunk in rag.chunks}
    if source not in names:
        raise ValueError("该论文尚未建立索引。")
    evidence = {field: _card_evidence(rag, source, field) for field in CARD_FIELDS}
    prompt = ("从以下论文证据生成精读卡片。只输出 JSON 对象，键为 " + ", ".join(CARD_FIELDS) +
              "。每个键的值必须是 {\"text\":字符串,\"chunk_ids\":字符串数组}。"+
              "只能引用对应字段下给出的 chunk_id；证据不足写‘未报告’，并使用空数组。"+
              "不要根据常识填补数值或结论。\n" + json.dumps(evidence, ensure_ascii=False))
    if len(prompt) > 30000:
        raise ValueError("证据上下文过长，请缩小论文范围。")
    response = call_agent_model(prompt, user_id=user_id, system_prompt="你是谨慎的论文信息提取器。只依据提供的证据作答。", timeout=120, max_tokens=2200)
    raw = response["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    value = json.loads(raw)
    fields = {}
    for field in CARD_FIELDS:
        entry = value.get(field, {})
        allowed = {item["chunk_id"] for item in evidence[field]}
        cited = entry.get("chunk_ids", []) if isinstance(entry, dict) else []
        if not isinstance(cited, list) or any(key not in allowed for key in cited):
            raise ValueError("精读卡片引用了无效证据，已拒绝保存。")
        text = str(entry.get("text", "未报告")) if isinstance(entry, dict) else "未报告"
        if text != "未报告" and not cited:
            text = "未报告"
        fields[field] = {"text": text, "evidence": [item for item in evidence[field] if item["chunk_id"] in cited]}
    card = {"source": source, "fields": fields, "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_usage": {k: response.get(k, 0) for k in ("input_tokens", "output_tokens")}}
    def save(lib):
        lib["cards"][source] = card
        return card
    return _update(user_id, session_id, save)


def comparison_markdown(cards: list[dict]) -> str:
    if len(cards) < 2:
        raise ValueError("至少选择两篇已生成精读卡片的论文。")
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# 多论文对比", "", "| 维度 | " + " | ".join(cell(card["source"]) for card in cards) + " |",
             "| --- | " + " | ".join("---" for _ in cards) + " |"]
    for field, label in zip(CARD_FIELDS, CARD_LABELS):
        cells = []
        for card in cards:
            item = card["fields"][field]
            refs = ", ".join(f"第{e['page']}页" for e in item.get("evidence", []))
            cells.append(cell(item["text"] + (f"（{refs}）" if refs else "")))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += ["", "不同论文的数据集、指标或实验条件可能不同；表中结果不代表可直接比较的性能排名。"]
    return "\n".join(lines)
