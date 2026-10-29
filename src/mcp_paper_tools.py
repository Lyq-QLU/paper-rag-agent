"""Pure implementations behind the paper MCP server tools."""

from __future__ import annotations

from pathlib import Path
import os
import re
import time
import html
from urllib.parse import quote_plus, urlparse
import xml.etree.ElementTree as ET

import httpx


ARXIV_HOSTS = {"arxiv.org", "export.arxiv.org"}
MAX_PDF_BYTES = 50 * 1024 * 1024


def _arxiv_get(url: str):
    """Retry transient API timeouts inside one MCP call, within its 60s deadline."""
    timeout = httpx.Timeout(24.0, connect=5.0)
    for attempt in range(2):
        try:
            response = httpx.get(url, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
            return response
        except httpx.TimeoutException as exc:
            if attempt:
                raise TimeoutError("arXiv 接口连续两次响应超时，请稍后重试。") from exc
            time.sleep(0.5)


def _openalex_get(params: dict, api_key: str):
    headers = {"User-Agent": "paper-rag-agent/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for attempt in range(3):
        try:
            response = httpx.get("https://api.openalex.org/works", params=params,
                                 timeout=httpx.Timeout(20.0, connect=5.0),
                                 follow_redirects=True, headers=headers)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429:
                raise
            remaining = exc.response.headers.get("x-ratelimit-remaining")
            reset = exc.response.headers.get("x-ratelimit-reset")
            try:
                reset_seconds = int(reset or "0")
            except ValueError:
                reset_seconds = 0
            if remaining == "0" and reset_seconds > 60:
                hours = (reset_seconds + 3599) // 3600
                if api_key:
                    raise RuntimeError(f"OpenAlex API Key 的当前额度已用完，约 {hours} 小时后重置。") from exc
                raise RuntimeError(
                    f"OpenAlex 匿名额度已用完，约 {hours} 小时后重置；可在论文来源设置中配置免费的 OpenAlex API Key。"
                ) from exc
            try:
                error_code = str(exc.response.json().get("error", ""))
            except (ValueError, AttributeError):
                error_code = ""
            if error_code == "not_enough_credits":
                raise RuntimeError("OpenAlex 今日查询额度已用完；请等待额度重置或使用有余额的 API Key。") from exc
            if attempt == 2:
                suggestion = "请配置免费的 OpenAlex API Key。" if not api_key else "请稍后重试。"
                raise RuntimeError("OpenAlex 请求次数受限；" + suggestion) from exc
            time.sleep(2 ** attempt)
        except httpx.TimeoutException as exc:
            if attempt >= 1:
                raise TimeoutError("OpenAlex 接口连续两次响应超时，请稍后重试。") from exc
            time.sleep(0.5)


def search_arxiv_impl(query: str, max_results: int = 5,
                      year_from: int | None = None, year_to: int | None = None) -> dict:
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    limit = max(1, min(int(max_results), 10))
    search_query = f"all:{normalized}"
    if year_from is not None and year_to is not None:
        search_query += f" AND submittedDate:[{int(year_from)}01010000 TO {int(year_to)}12312359]"
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query={quote_plus(search_query)}&start=0&max_results={limit}"
        "&sortBy=relevance&sortOrder=descending"
    )
    response = _arxiv_get(url)
    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", namespace):
        title = " ".join(entry.findtext("atom:title", default="", namespaces=namespace).split())
        if not title:
            continue
        pdf_url = ""
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        papers.append({
            "title": title,
            "authors": [
                author.findtext("atom:name", default="", namespaces=namespace)
                for author in entry.findall("atom:author", namespace)
            ],
            "published": entry.findtext("atom:published", default="", namespaces=namespace)[:10],
            "url": entry.findtext("atom:id", default="", namespaces=namespace),
            "pdf_url": pdf_url,
            "summary": " ".join(
                entry.findtext("atom:summary", default="", namespaces=namespace).split()
            )[:360],
            "source": "arxiv",
            "work_type": "preprint",
            "venue_kind": "preprint",
            "venue": "",
            "journal_ref": entry.findtext("{http://arxiv.org/schemas/atom}journal_ref", default=""),
        })
    return {"query": normalized, "count": len(papers), "papers": papers}


def search_openalex_impl(query: str, max_results: int = 5,
                         year_from: int | None = None, year_to: int | None = None) -> dict:
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    limit = max(1, min(int(max_results), 10))
    params = {"search": normalized, "per-page": limit}
    if year_from is not None and year_to is not None:
        params["filter"] = f"from_publication_date:{int(year_from)}-01-01,to_publication_date:{int(year_to)}-12-31"
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    response = _openalex_get(params, api_key)
    payload = response.json()
    papers = []
    for work in payload.get("results", []):
        title = " ".join(str(work.get("display_name") or "").split())
        if not title:
            continue
        primary = work.get("primary_location") or {}
        locations = [primary] + (work.get("locations") or [])
        publication = next((loc for loc in locations if (loc.get("source") or {}).get("type") in {"journal", "conference"}), primary)
        venue_source = publication.get("source") or {}
        best_oa = work.get("best_oa_location") or {}
        pdf_url = str(primary.get("pdf_url") or best_oa.get("pdf_url") or "")
        landing_url = str(
            primary.get("landing_page_url") or work.get("doi") or work.get("id") or ""
        )
        papers.append({
            "title": title,
            "authors": [
                str(item.get("author", {}).get("display_name", ""))
                for item in work.get("authorships", [])[:10]
                if item.get("author", {}).get("display_name")
            ],
            "published": str(work.get("publication_date") or work.get("publication_year") or ""),
            "url": landing_url,
            "pdf_url": pdf_url,
            "doi": str(work.get("doi") or ""),
            "openalex_id": str(work.get("id") or ""),
            "cited_by_count": int(work.get("cited_by_count") or 0),
            "source": "openalex",
            "work_type": str(work.get("type") or ""),
            "venue": str(venue_source.get("display_name") or "") if venue_source.get("type") != "repository" else "",
            "venue_kind": str(venue_source.get("type") or "unknown") if venue_source.get("type") in {"journal", "conference"} else ("preprint" if work.get("type") == "preprint" else "unknown"),
            "venue_issns": venue_source.get("issn") or [],
            "venue_id": str(venue_source.get("id") or ""),
            "summary": " ".join(
                word for _, word in sorted(
                    (position, word)
                    for word, positions in (work.get("abstract_inverted_index") or {}).items()
                    for position in positions
                )
            )[:500],
        })
    return {"query": normalized, "count": len(papers), "papers": papers}


def search_crossref_impl(query: str, max_results: int = 5,
                         year_from: int | None = None, year_to: int | None = None,
                         venue_kind: str = "all") -> dict:
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    if venue_kind not in {"all", "journal", "conference"}:
        raise ValueError("invalid venue kind")
    limit = max(1, min(int(max_results), 10))
    kinds = (["journal-article", "proceedings-article"] if venue_kind == "all"
             else ["journal-article" if venue_kind == "journal" else "proceedings-article"])
    papers = []
    for kind in kinds:
        filters = [f"type:{kind}"]
        if year_from is not None:
            filters.append(f"from-pub-date:{int(year_from)}-01-01")
        if year_to is not None:
            filters.append(f"until-pub-date:{int(year_to)}-12-31")
        response = httpx.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": normalized, "filter": ",".join(filters),
                    "rows": limit if len(kinds) == 1 else (limit + 1) // 2},
            timeout=httpx.Timeout(20.0, connect=5.0), follow_redirects=True,
            headers={"User-Agent": "paper-rag-agent/1.0 (scholarly metadata search)"},
        )
        response.raise_for_status()
        for work in response.json().get("message", {}).get("items", []):
            title = html.unescape(re.sub(r"<[^>]+>", "", str((work.get("title") or [""])[0]))).strip()
            venue = html.unescape(str((work.get("container-title") or [""])[0])).strip()
            doi = str(work.get("DOI") or "").strip()
            if not title or not venue or not doi or work.get("type") != kind:
                continue
            date_parts = (work.get("published") or work.get("issued") or {}).get("date-parts") or []
            published = "-".join(str(part).zfill(2) if index else str(part)
                                 for index, part in enumerate(date_parts[0])) if date_parts else ""
            papers.append({
                "title": title, "authors": [" ".join(filter(None, [a.get("given"), a.get("family")]))
                                              for a in work.get("author", [])[:10]],
                "published": published, "url": str(work.get("URL") or f"https://doi.org/{doi}"),
                "pdf_url": "", "doi": f"https://doi.org/{doi}", "summary": "",
                "source": "crossref", "work_type": kind, "venue": venue,
                "venue_kind": "journal" if kind == "journal-article" else "conference",
                "venue_issns": work.get("ISSN") or [],
            })
    return {"query": normalized, "count": len(papers), "papers": papers[:limit]}


def download_arxiv_pdf_impl(pdf_url: str, title: str, output_dir: str) -> dict:
    parsed = urlparse(pdf_url)
    if parsed.scheme != "https" or parsed.hostname not in ARXIV_HOSTS:
        raise ValueError("only HTTPS arXiv PDF URLs are allowed")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    response = httpx.get(pdf_url, timeout=45.0, follow_redirects=True)
    response.raise_for_status()
    content = response.content
    if len(content) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 50 MB safety limit")
    if not content.startswith(b"%PDF"):
        raise ValueError("downloaded content is not a PDF")
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", title or "paper").strip("._")[:80]
    path = (directory / f"{safe_name or 'paper'}.pdf").resolve()
    if directory not in path.parents:
        raise ValueError("invalid output path")
    path.write_bytes(content)
    return {"path": str(path), "bytes": len(content), "source_url": pdf_url}
