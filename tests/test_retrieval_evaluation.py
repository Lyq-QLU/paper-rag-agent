from src.evaluation import (
    GroundTruthCase,
    chunk_id,
    evaluate_ground_truth,
    parse_ground_truth_cases,
)
import faiss
import numpy as np

from src.rag_pipeline import Chunk, read_faiss_index, write_faiss_index


class FakeRAG:
    def __init__(self, results):
        self.results = results

    def retrieve_by_mode(self, question, top_k=5, mode="hybrid"):
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
    )

    rows, summary = evaluate_ground_truth(rag, [case], top_k=2, mode="hybrid")

    assert rows[0]["first_relevant_rank"] == 2
    assert summary["recall_at_k"] == 0.5
    assert summary["hit_at_k"] == 1.0
    assert summary["mrr"] == 0.5


def test_faiss_index_round_trip_supports_unicode_path(tmp_path):
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0]], dtype="float32"))
    target = tmp_path / "中文目录" / "index.faiss"

    write_faiss_index(index, target)
    loaded = read_faiss_index(target)

    assert target.exists()
    assert loaded.ntotal == 1
