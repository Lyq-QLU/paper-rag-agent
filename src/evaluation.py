from dataclasses import dataclass


@dataclass
class EvaluationCase:
    question: str
    expected_keywords: list[str]


def parse_evaluation_cases(raw_text: str) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        question, keywords_text = line.split("|", 1)
        keywords = [
            keyword.strip()
            for keyword in keywords_text.replace("，", ",").split(",")
            if keyword.strip()
        ]
        if question.strip() and keywords:
            cases.append(EvaluationCase(question=question.strip(), expected_keywords=keywords))
    return cases


def evaluate_retrieval(rag, cases: list[EvaluationCase], top_k: int = 5) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    hit_count = 0

    for case in cases:
        sources = rag.retrieve(case.question, top_k=top_k)
        joined_text = "\n".join(source.text.lower() for source in sources)
        matched_keywords = [
            keyword for keyword in case.expected_keywords if keyword.lower() in joined_text
        ]
        hit = bool(matched_keywords)
        if hit:
            hit_count += 1

        rows.append(
            {
                "问题": case.question,
                "期望关键词": ", ".join(case.expected_keywords),
                "命中关键词": ", ".join(matched_keywords) if matched_keywords else "未命中",
                "是否命中": "是" if hit else "否",
                "Top-K 来源": format_sources(sources),
            }
        )

    summary = {
        "case_count": len(cases),
        "hit_count": hit_count,
        "hit_rate": round(hit_count / len(cases), 3) if cases else 0,
    }
    return rows, summary


def format_sources(sources) -> str:
    parts = []
    for source in sources:
        metadata = source.metadata
        page_start = metadata.get("page_start", metadata.get("page", "unknown"))
        page_end = metadata.get("page_end", page_start)
        page_label = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        parts.append(
            f"{metadata.get('source', 'unknown')} p.{page_label} "
            f"({metadata.get('section', 'body')}/{metadata.get('content_type', 'text')})"
        )
    return " | ".join(parts)
