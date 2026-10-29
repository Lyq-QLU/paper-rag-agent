from pathlib import Path
import time

import pytest
import httpx

from src.agent_workflow import PaperAgentWorkflow
from src.mcp_client import MCPToolError, MCPPaperToolGateway
from src.mcp_paper_tools import (
    download_arxiv_pdf_impl, search_arxiv_impl, search_openalex_impl, search_crossref_impl,
)
import src.mcp_paper_tools as paper_tools


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>  A Test Paper  </title>
    <summary>Evidence grounded retrieval.</summary>
    <published>2026-01-02T00:00:00Z</published>
    <id>https://arxiv.org/abs/2601.00001</id>
    <author><name>Alice</name></author>
    <link title="pdf" type="application/pdf" href="https://arxiv.org/pdf/2601.00001" />
  </entry>
</feed>"""


class FakeResponse:
    def __init__(self, text="", content=b""):
        self.text = text
        self.content = content

    def raise_for_status(self):
        return None


class FakeGateway:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "search_arxiv":
            return {
                "query": arguments["query"], "count": 1,
                "papers": [{
                    "title": "MCP Paper", "authors": ["Alice"], "published": "2026-01-02",
                    "url": "https://arxiv.org/abs/2601.00001",
                    "pdf_url": "https://arxiv.org/pdf/2601.00001", "summary": "test",
                }],
            }
        raise AssertionError(name)


def test_search_arxiv_tool_normalizes_atom_response(monkeypatch):
    monkeypatch.setattr(paper_tools.httpx, "get", lambda *args, **kwargs: FakeResponse(text=ATOM))
    result = search_arxiv_impl(" agentic   rag ", max_results=5)
    assert result["query"] == "agentic rag"
    assert result["count"] == 1
    assert result["papers"][0]["title"] == "A Test Paper"


def test_search_sources_apply_year_filters(monkeypatch):
    seen = []
    response = FakeResponse(text=ATOM)
    response.json = lambda: {"results": []}
    def fake_get(url, **kwargs):
        seen.append((url, kwargs))
        return response
    monkeypatch.setattr(paper_tools.httpx, "get", fake_get)
    search_arxiv_impl("agent", year_from=2024, year_to=2026)
    search_openalex_impl("agent", year_from=2024, year_to=2026)
    assert "submittedDate" in seen[0][0]
    assert seen[1][1]["params"]["filter"] == "from_publication_date:2024-01-01,to_publication_date:2026-12-31"


def test_arxiv_retries_read_timeout_inside_one_mcp_call(monkeypatch):
    attempts = []
    def fake_get(*_args, **kwargs):
        attempts.append(kwargs["timeout"])
        if len(attempts) == 1:
            raise httpx.ReadTimeout("slow response")
        return FakeResponse(text=ATOM)
    monkeypatch.setattr(paper_tools.httpx, "get", fake_get)
    monkeypatch.setattr(paper_tools.time, "sleep", lambda *_: None)
    assert search_arxiv_impl("quantum")["count"] == 1
    assert len(attempts) == 2


def test_gateway_does_not_reconnect_after_upstream_tool_error(monkeypatch):
    gateway = MCPPaperToolGateway()
    gateway._schemas = {"search_arxiv": {"required": ["query"]}}
    gateway._schema_expires_at = time.monotonic() + 60
    attempts = []
    def fake_run(awaitable):
        awaitable.close()
        attempts.append(1)
        raise ExceptionGroup("MCP nursery", [MCPToolError("arXiv read timeout")])
    monkeypatch.setattr(gateway, "_run", fake_run)
    with pytest.raises(MCPToolError, match="read timeout"):
        gateway.call_tool("search_arxiv", {"query": "quantum"})
    assert len(attempts) == 1


def test_download_tool_rejects_non_arxiv_url(tmp_path):
    with pytest.raises(ValueError, match="arXiv"):
        download_arxiv_pdf_impl("https://example.com/paper.pdf", "paper", str(tmp_path))


def test_search_openalex_tool_normalizes_metadata(monkeypatch):
    response = FakeResponse()
    response.json = lambda: {"results": [{
        "display_name": "OpenAlex Paper", "publication_date": "2026-02-03",
        "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/test",
        "cited_by_count": 12,
        "authorships": [{"author": {"display_name": "Bob"}}],
        "primary_location": {"landing_page_url": "https://doi.org/10.1/test", "pdf_url": None},
        "best_oa_location": {"pdf_url": "https://repository.example/paper.pdf"},
    }]}
    monkeypatch.setattr(paper_tools.httpx, "get", lambda *args, **kwargs: response)
    result = search_openalex_impl("home healthcare", max_results=3)
    assert result["count"] == 1
    assert result["papers"][0]["source"] == "openalex"
    assert result["papers"][0]["authors"] == ["Bob"]


def test_openalex_retries_rate_limit_and_sends_key_in_header(monkeypatch):
    attempts = []
    monkeypatch.setenv("OPENALEX_API_KEY", "synthetic-key")
    def fake_get(url, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return httpx.Response(429, request=httpx.Request("GET", url), json={"error": "rate_limit_exceeded"})
        return httpx.Response(200, request=httpx.Request("GET", url), json={"results": []})
    monkeypatch.setattr(paper_tools.httpx, "get", fake_get)
    monkeypatch.setattr(paper_tools.time, "sleep", lambda *_: None)
    assert search_openalex_impl("quantum")["count"] == 0
    assert len(attempts) == 2
    assert attempts[0]["headers"]["Authorization"] == "Bearer synthetic-key"
    assert "api_key" not in attempts[0]["params"]


def test_openalex_daily_budget_does_not_retry(monkeypatch):
    attempts = []
    def fake_get(url, **kwargs):
        attempts.append(1)
        return httpx.Response(429, request=httpx.Request("GET", url), json={"error": "not_enough_credits"})
    monkeypatch.setattr(paper_tools.httpx, "get", fake_get)
    with pytest.raises(RuntimeError, match="今日查询额度"):
        search_openalex_impl("quantum")
    assert len(attempts) == 1


def test_openalex_empty_anonymous_budget_reports_reset_without_retry(monkeypatch):
    attempts = []
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    def fake_get(url, **kwargs):
        attempts.append(1)
        return httpx.Response(
            429, request=httpx.Request("GET", url), json={"error": "Rate limit exceeded"},
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "60939"},
        )
    monkeypatch.setattr(paper_tools.httpx, "get", fake_get)
    with pytest.raises(RuntimeError, match="约 17 小时后重置"):
        search_openalex_impl("quantum")
    assert len(attempts) == 1


def test_crossref_returns_verified_journal_and_conference_types(monkeypatch):
    seen = []
    def fake_get(url, **kwargs):
        seen.append(kwargs["params"]["filter"])
        kind = kwargs["params"]["filter"].split(",")[0].split(":")[1]
        item = {"type": kind, "title": ["A Published Paper"],
                "container-title": ["Test Venue"], "DOI": f"10.1234/{kind}",
                "published": {"date-parts": [[2025, 3, 1]]},
                "author": [{"given": "Ada", "family": "Lovelace"}]}
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"message": {"items": [item]}})
    monkeypatch.setattr(paper_tools.httpx, "get", fake_get)
    result = search_crossref_impl("retrieval", year_from=2024, year_to=2026)
    assert {p["venue_kind"] for p in result["papers"]} == {"journal", "conference"}
    assert all(p["doi"].startswith("https://doi.org/") for p in result["papers"])
    assert all("from-pub-date:2024-01-01" in query for query in seen)


def test_download_tool_writes_valid_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(
        paper_tools.httpx, "get",
        lambda *args, **kwargs: FakeResponse(content=b"%PDF-1.7\nmock"),
    )
    result = download_arxiv_pdf_impl(
        "https://arxiv.org/pdf/2601.00001", "A Test Paper", str(tmp_path)
    )
    assert Path(result["path"]).read_bytes().startswith(b"%PDF")


def test_mcp_server_exposes_paper_tools():
    schemas = MCPPaperToolGateway().list_tools()
    assert set(schemas) == {"search_arxiv", "search_openalex", "search_crossref", "download_arxiv_pdf"}
    assert schemas["search_arxiv"]["required"] == ["query"]


def test_gateway_validates_required_schema_before_call():
    gateway = MCPPaperToolGateway()
    gateway._schemas = {"search_arxiv": {"required": ["query"]}}
    gateway._schema_expires_at = time.monotonic() + 60
    with pytest.raises(MCPToolError, match="query"):
        gateway.call_tool("search_arxiv", {})


def test_search_agent_calls_mcp_gateway(tmp_path: Path):
    gateway = FakeGateway()
    workflow = PaperAgentWorkflow(
        None, tmp_path / "checkpoints.sqlite", tool_gateway=gateway
    )
    try:
        result = workflow.invoke(
            "搜索论文 agentic rag", user_id="u", session_id="s"
        )
    finally:
        workflow.close()
    assert gateway.calls[0][0] == "search_arxiv"
    assert result["pending_action"]["status"] == "pending"
    assert "MCP工具 search_arxiv" in result["events"][-1]["message"]
