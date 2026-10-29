import pytest

import src.session_manager as sessions
from src.venue_rankings import classify_venue, import_rankings, journal_rankings, ccf_catalogue
from src.paper_library import search_papers


def test_ccf_matches_2026_catalogue_and_ignores_workshops():
    assert len(ccf_catalogue()["entries"]) == 681
    iclr = classify_venue({"venue": "ICLR", "venue_kind": "conference"})
    assert iclr["ccf"]["level"] == "A" and iclr["ccf"]["year"] == 2026
    assert classify_venue({"venue": "IJCAI"})["ccf"]["level"] == "B"
    assert classify_venue({"venue": "ICLR Workshop"})["ccf"] is None
    assert classify_venue({"venue": "AI for Health Conference"})["ccf"] is None
    assert classify_venue({"venue": "ICLR", "work_type": "preprint"})["ccf"] is None


def test_partition_import_preserves_year_subject_and_user_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(sessions, "DATA_ROOT", tmp_path)
    data = ("journal,issn,scheme,year,category,level,source_url\n"
            "Test Journal,1234-5678,JCR,2025,Computer Science,Q1,https://example.org/jcr\n"
            "Test Journal,1234-5678,JCR,2024,Computer Science,Q2,https://example.org/jcr\n"
            "Test Journal,1234-5678,CAS,2025,大类计算机科学,2,https://example.org/cas\n").encode()
    assert import_rankings("user1", data) == 3
    assert not journal_rankings("user2")
    result = classify_venue({"venue": "Alternate Title", "venue_kind": "journal", "venue_issns": ["1234-5678"]}, journal_rankings("user1"))
    assert len(result["jcr"]) == 2 and result["cas"][0]["level"] == "2"
    assert all(record["provenance"] == "user_import" for record in result["jcr"])
    with pytest.raises(ValueError):
        import_rankings("user1", data.replace(b"Q1", b"Q9"))
    assert len(journal_rankings("user1")) == 3


def test_ranking_filter_requires_evidence_and_correct_year(monkeypatch, tmp_path):
    monkeypatch.setattr(sessions, "DATA_ROOT", tmp_path)
    import_rankings("u", b"journal,scheme,year,category,level,source_url\nTest Journal,JCR,2024,CS,Q1,https://example.org/2024\nTest Journal,JCR,2025,CS,Q2,https://example.org/2025\n")
    class Gateway:
        def call_tool(self, *_):
            return {"papers": [{"title": "An Article", "published": "2025", "venue": "Test Journal", "venue_kind": "journal"},
                               {"title": "Unknown Venue", "published": "2025", "venue": "Other Journal", "venue_kind": "journal"}]}
    args = dict(keywords="test", year_from=2024, year_to=2026, sources=("openalex",), jcr_quartiles=("Q1",), gateway=Gateway())
    latest = search_papers("u", "s", **args)
    assert not any(p["included"] for p in latest["papers"])
    historical = search_papers("u", "s", ranking_year=2024, **args)
    assert [p["title"] for p in historical["papers"] if p["included"]] == ["An Article"]
