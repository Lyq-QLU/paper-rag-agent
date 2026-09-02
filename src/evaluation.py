from dataclasses import dataclass, field
import hashlib
import time


@dataclass
class EvaluationCase:
    question: str
    expected_keywords: list[str]


@dataclass
class GroundTruthCase:
    """人工确认的问题及其相关证据块。"""

    case_id: str
    question: str
    relevant_chunk_ids: list[str]
    split: str = "test"
    metadata: dict = field(default_factory=dict)


def chunk_id(chunk) -> str:
    """基于来源、页码、类型与文本生成稳定证据ID。"""
    metadata = chunk.metadata
    identity = "\n".join(
        [
            str(metadata.get("source", "unknown")),
            str(metadata.get("page_start", metadata.get("page", "unknown"))),
            str(metadata.get("page_end", metadata.get("page", "unknown"))),
            str(metadata.get("section", "body")),
            str(metadata.get("content_type", "text")),
            chunk.text.strip(),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def parse_ground_truth_cases(data: list[dict]) -> list[GroundTruthCase]:
    cases: list[GroundTruthCase] = []
    for index, item in enumerate(data, start=1):
        question = str(item.get("question", "")).strip()
        relevant = [str(value).strip() for value in item.get("relevant_chunk_ids", []) if str(value).strip()]
        if not question or not relevant:
            continue
        cases.append(
            GroundTruthCase(
                case_id=str(item.get("case_id") or f"case_{index:04d}"),
                question=question,
                relevant_chunk_ids=relevant,
                split=str(item.get("split") or "test"),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return cases


def evaluate_ground_truth(
    rag,
    cases: list[GroundTruthCase],
    top_k: int = 5,
    mode: str = "hybrid",
    dense_weight: float = 0.65,
) -> tuple[list[dict], dict]:
    """计算严格的 Recall@K、Hit@K、MRR 与平均检索耗时。"""
    rows: list[dict] = []
    recalls: list[float] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    document_hits: list[float] = []
    document_reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for case in cases:
        started = time.perf_counter()
        sources = rag.retrieve_by_mode(
            case.question,
            top_k=top_k,
            mode=mode,
            dense_weight=dense_weight,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)

        retrieved_ids = [chunk_id(source) for source in sources]
        relevant = set(case.relevant_chunk_ids)
        matched = [value for value in retrieved_ids if value in relevant]
        recall = len(set(matched)) / len(relevant)
        first_rank = next(
            (rank for rank, value in enumerate(retrieved_ids, start=1) if value in relevant),
            None,
        )
        reciprocal_rank = 1 / first_rank if first_rank else 0.0
        expected_source = str(case.metadata.get("source", ""))
        retrieved_sources = [str(source.metadata.get("source", "")) for source in sources]
        document_first_rank = next(
            (rank for rank, source in enumerate(retrieved_sources, start=1) if source == expected_source),
            None,
        ) if expected_source else None
        document_hit = 1.0 if document_first_rank else 0.0
        document_reciprocal_rank = 1 / document_first_rank if document_first_rank else 0.0
        hit = 1.0 if matched else 0.0
        recalls.append(recall)
        hits.append(hit)
        reciprocal_ranks.append(reciprocal_rank)
        document_hits.append(document_hit)
        document_reciprocal_ranks.append(document_reciprocal_rank)
        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "mode": mode,
                "recall_at_k": round(recall, 4),
                "hit_at_k": int(hit),
                "reciprocal_rank": round(reciprocal_rank, 4),
                "first_relevant_rank": first_rank,
                "document_hit_at_k": int(document_hit),
                "document_first_rank": document_first_rank,
                "relevant_chunk_ids": case.relevant_chunk_ids,
                "retrieved_chunk_ids": retrieved_ids,
            }
        )

    count = len(cases)
    summary = {
        "mode": mode,
        "top_k": top_k,
        "dense_weight": dense_weight if mode == "hybrid" else None,
        "case_count": count,
        "recall_at_k": round(sum(recalls) / count, 4) if count else 0.0,
        "hit_at_k": round(sum(hits) / count, 4) if count else 0.0,
        "mrr": round(sum(reciprocal_ranks) / count, 4) if count else 0.0,
        "document_hit_at_k": round(sum(document_hits) / count, 4) if count else 0.0,
        "document_mrr": round(sum(document_reciprocal_ranks) / count, 4) if count else 0.0,
        "average_latency_ms": round(sum(latencies_ms) / count, 2) if count else 0.0,
    }
    return rows, summary


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
