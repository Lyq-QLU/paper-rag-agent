"""构建并运行 PaperGuide 的可复现检索评测。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.evaluation import chunk_id, evaluate_ground_truth, parse_ground_truth_cases
from src.paper_loader import load_pdf_documents
from src.rag_pipeline import Chunk, PaperRAG, split_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "corpus_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "local"


QUESTION_TEMPLATES = {
    "abstract": "论文《{title}》研究的核心问题、研究目标和主要贡献是什么？",
    "method": "论文《{title}》采用了什么模型或算法，关键步骤是什么？",
    "experiment": "论文《{title}》如何设置实验、使用哪些评价指标，主要结果是什么？",
    "conclusion": "论文《{title}》得到哪些主要结论，并指出了哪些局限或未来工作？",
    "table": "论文《{title}》的表格报告了哪些关键实验结果？",
    "figure": "论文《{title}》的图中展示了什么方法流程或实验趋势？",
    "body": "论文《{title}》在该证据片段中说明了什么关键内容？",
}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_corpus(manifest: dict, source_root: Path) -> list[tuple[Path, dict]]:
    resolved: list[tuple[Path, dict]] = []
    seen_hashes: set[str] = set()
    errors: list[str] = []
    for item in manifest.get("documents", []):
        path = source_root / Path(item["relative_path"])
        if not path.is_file():
            errors.append(f"文件不存在：{item['relative_path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = str(item.get("sha256_prefix", ""))
        if expected and not digest.startswith(expected):
            errors.append(f"哈希不匹配：{item['relative_path']}")
            continue
        if digest in seen_hashes:
            errors.append(f"清单包含重复PDF：{item['relative_path']}")
            continue
        seen_hashes.add(digest)
        resolved.append((path, item))
    if errors:
        raise ValueError("\n".join(errors))
    return resolved


def document_title(path: Path) -> str:
    return path.stem.replace("_", " ").strip()


def choose_evidence_chunks(source_chunks: list) -> list:
    """每篇论文选择四类候选证据；这些标签必须经过人工审核。"""
    chosen = []
    used: set[str] = set()
    priorities = ["abstract", "method", "experiment", "conclusion"]
    for section in priorities:
        candidate = next(
            (
                chunk
                for chunk in source_chunks
                if chunk.metadata.get("section") == section and len(chunk.text) >= 120
            ),
            None,
        )
        if candidate is not None:
            chosen.append(candidate)
            used.add(chunk_id(candidate))

    fallbacks = sorted(
        source_chunks,
        key=lambda chunk: (
            chunk.metadata.get("section") in {"references", "back_matter"},
            chunk.metadata.get("content_type") == "figure",
            -len(chunk.text),
        ),
    )
    for candidate in fallbacks:
        candidate_id = chunk_id(candidate)
        if candidate_id in used or len(candidate.text) < 120:
            continue
        chosen.append(candidate)
        used.add(candidate_id)
        if len(chosen) == 4:
            break
    return chosen[:4]


def build_candidate_cases(chunks: list, corpus: list[tuple[Path, dict]]) -> list[dict]:
    cases: list[dict] = []
    for paper_index, (path, manifest_item) in enumerate(corpus, start=1):
        source_chunks = [chunk for chunk in chunks if chunk.metadata.get("source") == path.name]
        for question_index, evidence in enumerate(choose_evidence_chunks(source_chunks), start=1):
            section = str(evidence.metadata.get("section", "body"))
            content_type = str(evidence.metadata.get("content_type", "text"))
            template_key = content_type if content_type in {"table", "figure"} else section
            template = QUESTION_TEMPLATES.get(template_key, QUESTION_TEMPLATES["body"])
            cases.append(
                {
                    "case_id": f"p{paper_index:02d}_q{question_index}",
                    "question": template.format(title=document_title(path)),
                    "relevant_chunk_ids": [chunk_id(evidence)],
                    "split": "dev" if question_index == 1 else "test",
                    "status": "needs_review",
                    "metadata": {
                        "source": path.name,
                        "relative_path": manifest_item["relative_path"],
                        "topic": manifest_item.get("topic", "unknown"),
                        "page_start": evidence.metadata.get("page_start", evidence.metadata.get("page")),
                        "page_end": evidence.metadata.get("page_end", evidence.metadata.get("page")),
                        "section": section,
                        "content_type": content_type,
                        "evidence_preview": evidence.text[:600],
                    },
                }
            )
    return cases


def prepare(args) -> None:
    manifest = load_manifest(args.manifest)
    corpus = resolve_corpus(manifest, args.source_root)
    args.output.mkdir(parents=True, exist_ok=True)
    documents = []
    for path, _ in corpus:
        paper_key = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
        documents.extend(load_pdf_documents([path], asset_root=args.output / "assets" / paper_key))
    chunks = split_documents(documents)
    chunk_records = [{"chunk_id": chunk_id(chunk), **chunk.to_dict()} for chunk in chunks]
    candidates = build_candidate_cases(chunks, corpus)

    (args.output / "chunks.json").write_text(
        json.dumps(chunk_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "candidate_cases.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_review_csv(candidates, args.output / "candidate_cases.csv")
    (args.output / "resolved_corpus.json").write_text(
        json.dumps(
            [{**item, "absolute_path": str(path)} for path, item in corpus],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"papers": len(corpus), "chunks": len(chunks), "candidate_cases": len(candidates)}, ensure_ascii=False))

    if args.build_index:
        rag = PaperRAG()
        rag.build_index(documents)
        rag.save(args.output / "index")
        print(f"索引已保存：{args.output / 'index'}")


def write_review_csv(cases: list[dict], path: Path) -> None:
    columns = [
        "status", "case_id", "split", "question", "relevant_chunk_ids",
        "source", "page_start", "page_end", "section", "content_type", "evidence_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            metadata = case.get("metadata", {})
            writer.writerow(
                {
                    "status": case.get("status", "needs_review"),
                    "case_id": case.get("case_id", ""),
                    "split": case.get("split", "test"),
                    "question": case.get("question", ""),
                    "relevant_chunk_ids": ";".join(case.get("relevant_chunk_ids", [])),
                    "source": metadata.get("source", ""),
                    "page_start": metadata.get("page_start", ""),
                    "page_end": metadata.get("page_end", ""),
                    "section": metadata.get("section", ""),
                    "content_type": metadata.get("content_type", ""),
                    "evidence_preview": metadata.get("evidence_preview", ""),
                }
            )


def load_case_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() != ".csv":
        raise ValueError("评测案例必须是JSON或CSV文件。")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        records = []
        for row in csv.DictReader(stream):
            records.append(
                {
                    "status": row.get("status", "needs_review").strip(),
                    "case_id": row.get("case_id", "").strip(),
                    "split": row.get("split", "test").strip(),
                    "question": row.get("question", "").strip(),
                    "relevant_chunk_ids": [
                        value.strip()
                        for value in row.get("relevant_chunk_ids", "").split(";")
                        if value.strip()
                    ],
                    "metadata": {
                        "source": row.get("source", ""),
                        "page_start": row.get("page_start", ""),
                        "page_end": row.get("page_end", ""),
                        "section": row.get("section", ""),
                        "content_type": row.get("content_type", ""),
                    },
                }
            )
    return records


def run_evaluation(args) -> None:
    raw_cases = load_case_records(args.cases)
    approved = [item for item in raw_cases if item.get("status") == "approved"]
    cases = parse_ground_truth_cases(approved)
    if not cases:
        raise ValueError("没有已审核案例。请把确认后的案例 status 改为 approved。")
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]

    rag = PaperRAG()
    if not rag.load(args.index):
        raise ValueError(f"无法加载索引：{args.index}")

    report = {"summaries": [], "rows": {}}
    for mode in ("bm25", "dense", "hybrid"):
        rows, summary = evaluate_ground_truth(rag, cases, top_k=args.top_k, mode=mode)
        report["summaries"].append(summary)
        report["rows"][mode] = rows
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2))


def build_index(args) -> None:
    records = json.loads(args.chunks.read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(record) for record in records]
    rag = PaperRAG()
    rag.build_index_from_chunks(chunks)
    rag.save(args.index)
    print(json.dumps({"chunks": len(chunks), "index": str(args.index)}, ensure_ascii=False))


def export_review(args) -> None:
    cases = json.loads(args.input.read_text(encoding="utf-8"))
    write_review_csv(cases, args.output)
    print(json.dumps({"cases": len(cases), "output": str(args.output)}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="解析语料并生成待审核问题")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--build-index", action="store_true")
    prepare_parser.set_defaults(handler=prepare)

    index_parser = subparsers.add_parser("build-index", help="从已解析Chunk构建FAISS索引")
    index_parser.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT / "chunks.json")
    index_parser.add_argument("--index", type=Path, default=DEFAULT_OUTPUT / "index")
    index_parser.set_defaults(handler=build_index)

    export_parser = subparsers.add_parser("export-review", help="把候选JSON导出为Excel可读CSV")
    export_parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT / "candidate_cases.json")
    export_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "candidate_cases.csv")
    export_parser.set_defaults(handler=export_review)

    evaluate_parser = subparsers.add_parser("evaluate", help="对已审核问题运行三路检索评测")
    evaluate_parser.add_argument("--cases", type=Path, default=DEFAULT_OUTPUT / "candidate_cases.csv")
    evaluate_parser.add_argument("--index", type=Path, default=DEFAULT_OUTPUT / "index")
    evaluate_parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT / "report.json")
    evaluate_parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    evaluate_parser.add_argument("--top-k", type=int, default=5)
    evaluate_parser.set_defaults(handler=run_evaluation)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.handler(parsed)
