"""Pure implementations behind the paper MCP server tools."""

from __future__ import annotations

from pathlib import Path
import os
import re
from urllib.parse import quote_plus, urlparse
import xml.etree.ElementTree as ET

import httpx


ARXIV_HOSTS = {"arxiv.org", "export.arxiv.org"}
MAX_PDF_BYTES = 50 * 1024 * 1024


def search_arxiv_impl(query: str, max_results: int = 5) -> dict:
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    limit = max(1, min(int(max_results), 10))
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{quote_plus(normalized)}&start=0&max_results={limit}"
        "&sortBy=relevance&sortOrder=descending"
    )
    response = httpx.get(url, timeout=15.0, follow_redirects=True)
    response.raise_for_status()
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
        })
    return {"query": normalized, "count": len(papers), "papers": papers}


def search_openalex_impl(query: str, max_results: int = 5) -> dict:
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    limit = max(1, min(int(max_results), 10))
    params = {"search": normalized, "per-page": limit}
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    response = httpx.get(
        "https://api.openalex.org/works", params=params, timeout=15.0,
        follow_redirects=True, headers={"User-Agent": "paper-rag-agent/1.0"},
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if response.status_code == 429:
            raise RuntimeError(
                "OpenAlex rate limit or API budget exhausted; configure OPENALEX_API_KEY"
            ) from exc
        raise
    payload = response.json()
    papers = []
    for work in payload.get("results", []):
        title = " ".join(str(work.get("display_name") or "").split())
        if not title:
            continue
        primary = work.get("primary_location") or {}
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
        })
    return {"query": normalized, "count": len(papers), "papers": papers}


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
