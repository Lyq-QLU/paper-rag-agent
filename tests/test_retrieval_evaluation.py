from src.evaluation import (
    GroundTruthCase,
    chunk_id,
    evaluate_ground_truth,
    parse_ground_truth_cases,
)
import faiss
import numpy as np

from src.rag_pipeline import (
    Chunk,
    read_faiss_index,
    rerank_candidates,
    parse_reranker_ranking,
    select_with_source_cap,
    write_faiss_index,
)


class FakeRAG:
    def __init__(self, results):
        self.results = results

    def retrieve_by_mode(
        self, question, top_k=5, mode="hybrid", dense_weight=0.65, max_per_source=2
    ):
        return self.results[mode][:top_k]


def make_chunk(text: str, page: int) -> Chunk:
    return Chunk(text=text, metadata={"source": "paper.pdf", "page": page, "section": "method"})


def test_chunk_id_is_stable_and_content_sensitive():
    first = make_chunk("evidence", 2)
    assert chunk_id(first) == chunk_id(make_chunk("evidence", 2))
    assert chunk_id(first) != chunk_id(make_chunk("other evidence", 2))


def test_parse_ground_truth_skips_unlabelled_cases():
    cases = parse_ground_truth_cases(
        [
            {"case_id": "q1", "question": "What method?", "relevant_chunk_ids": ["abc"]},
            {"case_id": "q2", "question": "Missing label", "relevant_chunk_ids": []},
        ]
    )
    assert len(cases) == 1
    assert cases[0].case_id == "q1"


def test_evaluate_ground_truth_computes_recall_hit_and_mrr():
    relevant = make_chunk("target", 3)
    distractor = make_chunk("noise", 1)
    second_relevant = make_chunk("second target", 4)
    rag = FakeRAG({"hybrid": [distractor, relevant], "dense": [], "bm25": []})
    case = GroundTruthCase(
        case_id="q1",
        question="target?",
        relevant_chunk_ids=[chunk_id(relevant), chunk_id(second_relevant)],
        metadata={"source": "paper.pdf"},
    )

    rows, summary = evaluate_ground_truth(rag, [case], top_k=2, mode="hybrid")

    assert rows[0]["first_relevant_rank"] == 2
    assert summary["recall_at_k"] == 0.5
    assert summary["hit_at_k"] == 1.0
    assert summary["mrr"] == 0.5
    assert summary["document_hit_at_k"] == 1.0
    assert summary["document_mrr"] == 1.0


def test_faiss_index_round_trip_supports_unicode_path(tmp_path):
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0]], dtype="float32"))
    target = tmp_path / "中文目录" / "index.faiss"

    write_faiss_index(index, target)
    loaded = read_faiss_index(target)

    assert target.exists()
    assert loaded.ntotal == 1


def test_rerank_keeps_fusion_score_primary():
    strong = Chunk(
        text="general evidence",
        metadata={"section": "body", "_retrieval": {"fused_score": 0.8}},
    )
    weak_method = Chunk(
        text="proposed algorithm model framework",
        metadata={"section": "method", "_retrieval": {"fused_score": 0.2}},
    )

    ranked = rerank_candidates("使用了什么算法？", [weak_method, strong])

    assert ranked[0] is strong


def test_source_cap_prevents_one_paper_from_filling_top_k():
    chunks = [
        Chunk(text=str(index), metadata={"source": "a.pdf" if index < 4 else "b.pdf"})
        for index in range(6)
    ]

    selected = select_with_source_cap(chunks, top_k=4, max_per_source=2)

    assert [chunk.metadata["source"] for chunk in selected] == ["a.pdf", "a.pdf", "b.pdf", "b.pdf"]


def test_parse_reranker_ranking_requires_all_unique_candidates():
    assert parse_reranker_ranking('{"ranking":["c2","c0","c1"]}', 3) == [2, 0, 1]
    assert parse_reranker_ranking('{"ranking":["c0","c0"]}', 2) == []
