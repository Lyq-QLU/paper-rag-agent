from pathlib import Path

import pytest

import src.paper_library as library


class SearchGateway:
    def call_tool(self, name, arguments):
        if name == "search_arxiv":
            return {"papers": [
                {"title": "A Shared Paper", "published": "2025-02-01", "url": "https://arxiv.org/abs/2502.00001",
                 "pdf_url": "https://arxiv.org/pdf/2502.00001", "summary": "method", "authors": ["A"]},
                {"title": "Too Old", "published": "2020-01-01", "url": "https://arxiv.org/abs/2001.00001",
                 "pdf_url": "https://arxiv.org/pdf/2001.00001"},
            ]}
        if name == "search_openalex":
            return {"papers": [
                {"title": "A Shared Paper", "published": "2025-02-02", "doi": "https://doi.org/10.1/shared",
                 "pdf_url": "https://repository.example/shared.pdf", "work_type": "article"},
                {"title": "A Review", "published": "2026-01-01", "doi": "https://doi.org/10.1/review",
                 "work_type": "review"},
            ]}
        raise AssertionError(name)


def test_discovery_filters_deduplicates_and_persists(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "get_session_dir", lambda *_: tmp_path)
    result = library.search_papers("u", "s", keywords="  retrieval   agent ", year_from=2024,
                                   year_to=2026, sources=("arxiv", "openalex"), gateway=SearchGateway())
    assert len(result["papers"]) == 3
    shared = next(item for item in result["papers"] if item["title"] == "A Shared Paper")
    assert set(shared["versions"]) == {"arxiv", "openalex"}
    assert shared["pdf_url"].startswith("https://arxiv.org/")
    old = next(item for item in result["papers"] if item["title"] == "Too Old")
    assert not old["included"] and "年份" in old["filter_reason"]
    library.set_saved("u", "s", [shared["id"]])
    assert library.load_library("u", "s")["papers"][shared["id"]]["saved"]
    again = library.search_papers("u", "s", keywords="retrieval agent", year_from=2024,
                                  year_to=2026, sources=("arxiv", "openalex"), gateway=SearchGateway())
    assert next(item for item in again["papers"] if item["id"] == shared["id"])["saved"]


def test_review_requires_source_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "get_session_dir", lambda *_: tmp_path)
    result = library.search_papers("u", "s", keywords="retrieval", year_from=2024, year_to=2026,
                                   paper_type="review", gateway=SearchGateway())
    assert [paper["title"] for paper in result["papers"] if paper["included"]] == ["A Review"]


def test_one_source_timeout_keeps_other_source_results(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "get_session_dir", lambda *_: tmp_path)
    class PartialGateway(SearchGateway):
        def call_tool(self, name, arguments):
            if name == "search_arxiv":
                raise RuntimeError("The read operation timed out")
            return super().call_tool(name, arguments)
    result = library.search_papers("u", "s", keywords="retrieval", year_from=2024, year_to=2026,
                                   gateway=PartialGateway())
    assert result["papers"]
    assert "稍后重试" in result["errors"]["arxiv"]


def test_crossref_published_results_survive_openalex_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "get_session_dir", lambda *_: tmp_path)
    class Gateway:
        def call_tool(self, name, arguments):
            if name == "search_openalex":
                raise RuntimeError("HTTP 429")
            if name == "search_crossref":
                assert arguments["venue_kind"] == "journal"
                return {"papers": [{"title": "Published Study", "published": "2025-03-01",
                                    "doi": "https://doi.org/10.1234/study", "venue": "Journal of Test",
                                    "venue_kind": "journal", "work_type": "journal-article"}]}
            raise AssertionError(name)
    result = library.search_papers("u", "s", keywords="retrieval", year_from=2024, year_to=2026,
                                   sources=("openalex", "crossref"), venue_kind="journal", gateway=Gateway())
    assert result["papers"][0]["included"]
    assert result["papers"][0]["publication"]["kind"] == "journal"
    assert "openalex" in result["errors"]


def test_download_failure_can_retry_and_is_registered(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "get_session_dir", lambda *_: tmp_path)
    monkeypatch.setattr(library, "register_pdf", lambda data, name, *_: tmp_path / "registered.pdf")
    found = library.search_papers("u", "s", keywords="retrieval", year_from=2024, year_to=2026,
                                  sources=("arxiv",), gateway=SearchGateway())
    key = next(p["id"] for p in found["papers"] if p["title"] == "A Shared Paper")
    class DownloadGateway:
        calls = 0
        def call_tool(self, *_):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            path = tmp_path / "search_downloads" / "source.pdf"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"%PDF-test")
            return {"path": str(path)}
    gateway = DownloadGateway()
    assert library.download_selected("u", "s", [key], gateway)[0]["status"] == "failed"
    assert library.download_selected("u", "s", [key], gateway)[0]["status"] == "downloaded"
    assert library.load_library("u", "s")["papers"][key]["saved"]


def test_comparison_marks_missing_and_keeps_evidence():
    def card(source, method):
        fields = {name: {"text": "未报告", "evidence": []} for name in library.CARD_FIELDS}
        fields["method"] = {"text": method, "evidence": [{"page": 3}]}
        return {"source": source, "fields": fields}
    markdown = library.comparison_markdown([card("A.pdf", "Method A"), card("B.pdf", "Method B")])
    assert "Method A（第3页）" in markdown
    assert "未报告" in markdown
    with pytest.raises(ValueError):
        library.comparison_markdown([card("A.pdf", "Method A")])


def test_reading_card_rejects_unknown_citation(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "get_session_dir", lambda *_: tmp_path)
    class Chunk:
        text = "The method uses retrieval."
        metadata = {"source": "A.pdf", "chunk_id": "c1", "page": 2}
    class Rag:
        chunks = [Chunk()]
        def retrieve_by_mode(self, *_args, **_kwargs):
            return self.chunks
    response = {field: {"text": "Claim", "chunk_ids": ["invented"]} for field in library.CARD_FIELDS}
    def fake_model(*_args, **kwargs):
        assert kwargs["timeout"] == 120
        return {"content": __import__("json").dumps(response)}
    monkeypatch.setattr(library, "call_agent_model", fake_model)
    with pytest.raises(ValueError, match="无效证据"):
        library.make_reading_card("u", "s", "A.pdf", Rag())
    assert not library.load_library("u", "s")["cards"]


def test_card_evidence_finds_english_benchmark_and_results():
    class Chunk:
        def __init__(self, chunk_id, text, section, page):
            self.text = text
            self.metadata = {"chunk_id": chunk_id, "source": "paper.pdf", "section": section, "page_start": page}
    class Rag:
        chunks = [
            Chunk("a", "Algorithm steps and parameter definitions. " * 8, "method", 3),
            Chunk("b", "We extend Solomon's benchmark to create 30 test instances. " * 5, "experiment", 13),
            Chunk("c", "The results show our method outperforms the baselines on 28 of 30 instances. " * 4, "discussion", 15),
        ]
        def retrieve_by_mode(self, *_args, **_kwargs):
            return [self.chunks[0]]
    assert any(item["page"] == 13 for item in library._card_evidence(Rag(), "paper.pdf", "datasets"))
    assert any(item["page"] == 15 for item in library._card_evidence(Rag(), "paper.pdf", "results"))
