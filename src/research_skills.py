"""Trusted, repository-owned research skills available to the standalone harness."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "skills"


@dataclass(frozen=True)
class ResearchSkill:
    name: str
    description: str
    tools: frozenset[str]
    instructions: str


_MANIFEST = {
    "paper_discovery": ("检索真实论文元数据，按年份、发表类型整理候选。", frozenset({"search_papers", "request_ingestion"})),
    "evidence_reading": ("读取当前会话的论文正文证据，核实方法、实验与结论。", frozenset({"list_papers", "retrieve_evidence"})),
    "research_synthesis": ("基于已核实证据写作和逐项检查研究答案。", frozenset()),
}


@lru_cache(maxsize=1)
def load_research_skills() -> dict[str, ResearchSkill]:
    skills = {}
    for name, (description, tools) in _MANIFEST.items():
        instructions = (ROOT / name / "SKILL.md").read_text(encoding="utf-8").strip()
        if not instructions:
            raise ValueError(f"empty research skill: {name}")
        skills[name] = ResearchSkill(name, description, tools, instructions)
    return skills


def skill_for_tool(tool: str) -> str:
    matches = [skill.name for skill in load_research_skills().values() if tool in skill.tools]
    if len(matches) != 1:
        raise ValueError(f"tool has no unique skill: {tool}")
    return matches[0]
