"""MCP server exposing academic search and controlled PDF download tools."""

from mcp.server.fastmcp import FastMCP

from src.mcp_paper_tools import (
    download_arxiv_pdf_impl, search_arxiv_impl, search_openalex_impl,
)


mcp = FastMCP(
    "paper-tools",
    instructions="Search arXiv and download user-approved arXiv PDFs for the Paper RAG Agent.",
)


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> dict:
    """Search arXiv and return normalized paper metadata."""
    return search_arxiv_impl(query, max_results=max_results)


@mcp.tool()
def search_openalex(query: str, max_results: int = 5) -> dict:
    """Search OpenAlex and return normalized scholarly-work metadata."""
    return search_openalex_impl(query, max_results=max_results)


@mcp.tool()
def download_arxiv_pdf(pdf_url: str, title: str, output_dir: str) -> dict:
    """Download one user-approved arXiv PDF after URL, size and content validation."""
    return download_arxiv_pdf_impl(pdf_url, title, output_dir)


if __name__ == "__main__":
    mcp.run(transport="stdio")
