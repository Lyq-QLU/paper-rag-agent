from src.research_harness import TOOLS
from src.research_skills import load_research_skills, skill_for_tool


def test_every_harness_tool_belongs_to_one_skill():
    skills = load_research_skills()
    assert set(skills) == {"paper_discovery", "evidence_reading", "research_synthesis"}
    for tool in TOOLS:
        skill = skill_for_tool(tool)
        assert tool in skills[skill].tools
        assert skills[skill].instructions
