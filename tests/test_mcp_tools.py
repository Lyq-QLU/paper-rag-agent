from pathlib import Path
import time

import pytest

from src.agent_workflow import PaperAgentWorkflow
from src.mcp_client import MCPToolError, MCPPaperToolGateway
from src.mcp_paper_tools import (
    download_arxiv_pdf_impl, search_arxiv_impl, search_openalex_impl,
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
    assert set(schemas) == {"search_arxiv", "search_openalex", "download_arxiv_pdf"}
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
