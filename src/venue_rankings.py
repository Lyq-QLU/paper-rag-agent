"""Venue identity and versioned ranking records; never infer journal quartiles."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
from functools import lru_cache
from urllib.parse import urlparse

from filelock import FileLock
from src.session_manager import get_user_dir, atomic_write_text_or_bytes


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", str(value).casefold())


@lru_cache(maxsize=1)
def ccf_catalogue() -> dict:
    return json.loads((Path(__file__).resolve().parents[1] / "resources" / "ccf_2026.json").read_text())


def journal_rankings(user_id: str) -> list[dict]:
    path = get_user_dir(user_id) / "journal_rankings.json"
    return json.loads(path.read_text()) if path.exists() else []


def import_rankings(user_id: str, data: bytes) -> int:
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("分区表不能超过 5 MB。")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    required = {"journal", "scheme", "year", "category", "level", "source_url"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("分区表需要 journal,scheme,year,category,level,source_url 列，可选 issn 列。")
    rows = []
    for number, row in enumerate(reader, 2):
        scheme = str(row["scheme"]).strip().upper()
        level = str(row["level"]).strip().upper()
        year = str(row["year"]).strip()
        url = str(row["source_url"]).strip()
        if scheme not in {"JCR", "CAS"} or level not in ({"Q1", "Q2", "Q3", "Q4"} if scheme == "JCR" else {"1", "2", "3", "4"}):
            raise ValueError(f"第 {number} 行：体系或分区无效；CAS 使用 1–4，JCR 使用 Q1–Q4。")
        if not re.fullmatch(r"20\d{2}", year) or not str(row["journal"]).strip() or not str(row["category"]).strip():
            raise ValueError(f"第 {number} 行：缺少有效期刊名、年份或学科。")
        if urlparse(url).scheme != "https" or not urlparse(url).hostname:
            raise ValueError(f"第 {number} 行：需要 HTTPS 数据来源链接。")
        rows.append({"journal": str(row["journal"]).strip(), "scheme": scheme, "year": int(year),
                     "category": str(row["category"]).strip(), "level": level,
                     "issn": str(row.get("issn") or "").strip().upper(), "source_url": url,
                     "provenance": "user_import"})
    if not rows:
        raise ValueError("分区表没有数据。")
    path = get_user_dir(user_id) / "journal_rankings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        records = journal_rankings(user_id)
        key = lambda r: (normalized_name(r["journal"]), r["scheme"], r["year"], r["category"])
        merged = {key(r): r for r in records}
        merged.update({key(r): r for r in rows})
        atomic_write_text_or_bytes(path, json.dumps(list(merged.values()), ensure_ascii=False, indent=2))
    return len(rows)


def classify_venue(paper: dict, records: list[dict] | None = None) -> dict:
    venue = str(paper.get("venue") or "")
    kind = str(paper.get("venue_kind") or "unknown")
    if paper.get("work_type") == "preprint" and kind not in {"journal", "conference"}:
        kind = "preprint"
    result = {"kind": kind, "ccf": None, "jcr": [], "cas": []}
    if not venue or kind == "preprint" or re.search(r"\b(workshops?|findings|companion|demos?)\b", venue, re.I):
        return result
    # Whole names/acronyms only: a substring such as 'AI' must not grant a grade.
    name = normalized_name(venue)
    stripped = re.sub(r"^(proceedings of (?:the )?)?\d{4}\s+", "", venue, flags=re.I)
    names = {name, normalized_name(stripped)}
    catalogue = ccf_catalogue()
    matches = []
    for entry in catalogue["entries"]:
        full = re.split(r"（原|\(formerly", entry["name"])[0]
        short = re.split(r"（原|\(formerly", entry["abbreviation"])[0]
        if names & {normalized_name(full), normalized_name(short)}:
            matches.append(entry)
    if len(matches) == 1:
        entry = matches[0]
        result["kind"] = entry["kind"]
        result["ccf"] = {"level": entry["level"], "year": catalogue["year"],
                         "source_url": catalogue["source_url"], "page": entry["page"],
                         "venue": entry["name"], "scope": "刊会目录等级；会议论文类型仍需核实"}
    issns = {str(v).upper() for v in paper.get("venue_issns", [])}
    if result["kind"] == "journal":
        for record in records or []:
            if normalized_name(record["journal"]) == name or (record.get("issn") and record["issn"] in issns):
                result[record["scheme"].lower()].append(record)
    return result


RANKING_TEMPLATE = "journal,issn,scheme,year,category,level,source_url\n"
