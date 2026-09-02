from dataclasses import dataclass
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_config
from src.llm import LLMServiceError, call_llm
from src.paper_loader import Document
from src.prompts import build_qa_prompt


@dataclass
class Chunk:
    text: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class SectionBlock:
    """一篇论文中连续的结构化章节。"""

    source: str
    title: str
    section: str
    paragraphs: list[tuple[str, int]]


class PaperRAG:
    def __init__(self, user_id: str | None = None) -> None:
        self.user_id = user_id
        self.embedding_model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []
        self.bm25: BM25Index | None = None

    def build_index(self, documents: list[Document]) -> None:
        self.build_index_from_chunks(split_documents(documents))

    def build_index_from_chunks(self, chunks: list[Chunk]) -> None:
        """从已解析Chunk构建索引，便于离线评测复用解析结果。"""
        self.chunks = chunks
        if not self.chunks:
            raise ValueError("没有从 PDF 中解析到可用文本。")

        embeddings = self._embed([chunk.text for chunk in self.chunks])
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.bm25 = BM25Index.from_chunks(self.chunks)

    def save(self, directory: Path) -> None:
        if self.index is None:
            raise ValueError("没有可保存的向量索引。")

        directory.mkdir(parents=True, exist_ok=True)
        write_faiss_index(self.index, directory / "index.faiss")
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        (directory / "chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> bool:
        index_path = directory / "index.faiss"
        chunks_path = directory / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            return False

        self.index = read_faiss_index(index_path)
        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(item) for item in chunks_data]
        self.bm25 = BM25Index.from_chunks(self.chunks)
        return True

    def answer(
        self,
        question: str,
        top_k: int = 4,
        conversation_history: list[dict] | None = None,
    ) -> tuple[str, list[Chunk]]:
        retrieval_query = build_contextual_retrieval_query(
            question,
            conversation_history or [],
        )
        sources = self.retrieve(retrieval_query, top_k=top_k)
        prompt = build_qa_prompt(
            question,
            [
                {
                    "text": source.text,
                    "source": source.metadata.get("source", "unknown"),
                    "page": source.metadata.get("page", "unknown"),
                    "page_start": source.metadata.get("page_start", source.metadata.get("page", "unknown")),
                    "page_end": source.metadata.get("page_end", source.metadata.get("page", "unknown")),
                    "section": source.metadata.get("section", "body"),
                    "section_title": source.metadata.get("section_title", ""),
                    "content_type": source.metadata.get("content_type", "text"),
                    "caption": source.metadata.get("caption", ""),
                }
                for source in sources
            ],
            conversation_history=conversation_history,
        )
        try:
            answer = call_llm(
                prompt,
                user_id=self.user_id,
                image_paths=[
                    str(source.metadata.get("image_path", ""))
                    for source in sources
                    if source.metadata.get("content_type") == "figure"
                ] if should_attach_images(question) else [],
            )
        except LLMServiceError as exc:
            return (
                f"⚠️ {exc}\n\n"
                "RAG 检索已正常完成，你仍可以在下方查看本轮检索来源。",
                sources,
            )

        if answer:
            return answer, sources

        fallback = (
            "当前没有配置大模型 API Key，下面是系统检索到的相关论文片段。\n\n"
            "配置 OPENAI_API_KEY 后，系统会基于这些片段生成完整回答。"
        )
        return fallback, sources

    def retrieve(self, question: str, top_k: int = 4) -> list[Chunk]:
        return self.retrieve_by_mode(question, top_k=top_k, mode="hybrid")

    def retrieve_by_mode(
        self,
        question: str,
        top_k: int = 4,
        mode: str = "hybrid",
    ) -> list[Chunk]:
        """分别运行 Dense、BM25 或完整 Hybrid 检索，供消融评测使用。"""
        if self.index is None:
            raise ValueError("请先构建向量索引。")

        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("mode 必须是 dense、bm25 或 hybrid。")

        limit = min(max(top_k * 10, 30), len(self.chunks))
        if normalized_mode == "bm25":
            keyword_hits = self.bm25.search(question, limit) if self.bm25 else []
            return [hit.chunk for hit in keyword_hits[:top_k]]

        query_embedding = self._embed([question])
        scores, indices = self.index.search(query_embedding, limit)
        vector_hits = [
            SearchHit(chunk=self.chunks[index], score=float(score), source="vector")
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]
        if normalized_mode == "dense":
            return [hit.chunk for hit in vector_hits[:top_k]]

        keyword_hits = self.bm25.search(question, limit) if self.bm25 else []
        candidates = merge_search_hits(vector_hits, keyword_hits)
        candidates = filter_retrieval_candidates(question, candidates)
        if needs_source_diversity(question):
            return retrieve_diverse_method_chunks(question, candidates, top_k)
        return rerank_candidates(question, candidates)[:top_k]

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._get_embedding_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def _get_embedding_model(self) -> SentenceTransformer:
        if self.embedding_model is not None:
            return self.embedding_model

        config = get_config(user_id=self.user_id)
        try:
            self.embedding_model = SentenceTransformer(config.embedding_model)
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(
                "Embedding 模型下载或缓存不完整。"
                "请检查网络和 Hugging Face 缓存后重试。"
            ) from exc
        return self.embedding_model


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    source: str


def write_faiss_index(index, path: Path) -> None:
    """兼容Windows版FAISS无法直接写入Unicode路径的问题。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt" or str(path).isascii():
        faiss.write_index(index, str(path))
        return
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        faiss.write_index(index, str(temporary_path))
        shutil.copyfile(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_faiss_index(path: Path):
    """兼容Windows版FAISS无法直接读取Unicode路径的问题。"""
    if os.name != "nt" or str(path).isascii():
        return faiss.read_index(str(path))
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        shutil.copyfile(path, temporary_path)
        return faiss.read_index(str(temporary_path))
    finally:
        temporary_path.unlink(missing_ok=True)


class BM25Index:
    def __init__(
        self,
        chunks: list[Chunk],
        tokenized_documents: list[list[str]],
        document_frequencies: Counter,
        average_document_length: float,
    ) -> None:
        self.chunks = chunks
        self.tokenized_documents = tokenized_documents
        self.document_frequencies = document_frequencies
        self.average_document_length = average_document_length
        self.document_count = len(tokenized_documents)

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "BM25Index":
        tokenized_documents = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        document_frequencies: Counter = Counter()
        for tokens in tokenized_documents:
            document_frequencies.update(set(tokens))

        total_length = sum(len(tokens) for tokens in tokenized_documents)
        average_length = total_length / max(len(tokenized_documents), 1)
        return cls(chunks, tokenized_documents, document_frequencies, average_length)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        hits: list[SearchHit] = []
        for index, document_tokens in enumerate(self.tokenized_documents):
            score = self._score(query_tokens, document_tokens)
            if score > 0:
                hits.append(SearchHit(chunk=self.chunks[index], score=score, source="keyword"))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _score(self, query_tokens: list[str], document_tokens: list[str]) -> float:
        k1 = 1.5
        b = 0.75
        score = 0.0
        term_frequencies = Counter(document_tokens)
        document_length = len(document_tokens)

        for token in query_tokens:
            if token not in term_frequencies:
                continue
            document_frequency = self.document_frequencies.get(token, 0)
            idf = math.log(1 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            frequency = term_frequencies[token]
            denominator = frequency + k1 * (
                1 - b + b * document_length / max(self.average_document_length, 1)
            )
            score += idf * frequency * (k1 + 1) / denominator

        return score


def split_documents(documents: list[Document]) -> list[Chunk]:
    config = get_config()
    chunks: list[Chunk] = []

    attach_section_context_to_special_documents(documents)

    special_documents = [
        document
        for document in documents
        if document.metadata.get("content_type", "text") != "text"
    ]
    for document in special_documents:
        chunks.extend(split_special_document(document, config.chunk_size))

    # 先跨页恢复论文的章节边界，再在章节内切块。这样可避免
    # Method/Experiment 等语义单元被页码或固定字符窗口任意截断。
    for block in build_section_blocks(documents):
        block_chunks = split_section_block(
            block,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        for chunk_index, (chunk_text, page_start, page_end) in enumerate(block_chunks):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "source": block.source,
                        # 保留 page 以兼容现有 UI/Prompt，同时增加跨页范围。
                        "page": page_start,
                        "page_start": page_start,
                        "page_end": page_end,
                        "chunk": chunk_index,
                        "section": block.section,
                        "section_title": block.title,
                        "content_type": "text",
                    },
                )
            )

    return chunks


def split_special_document(document: Document, chunk_size: int) -> list[Chunk]:
    """表格和图片说明独立切块，不与普通正文或其他章节合并。"""
    content_type = document.metadata.get("content_type", "text")
    page = int(document.metadata.get("page", 0))
    section = document.metadata.get("section") or infer_section(document.text)

    if content_type == "table":
        parts = split_markdown_table(document.text, chunk_size)
    else:
        parts = split_oversized_text(document.text, chunk_size)

    chunks: list[Chunk] = []
    for index, text in enumerate(parts):
        chunks.append(
            Chunk(
                text=text,
                metadata={
                    **document.metadata,
                    "page": page,
                    "page_start": page,
                    "page_end": page,
                    "chunk": index,
                    "section": section,
                    "section_title": document.metadata.get("section_title", "Unknown Section"),
                    "content_type": content_type,
                },
            )
        )
    return chunks


def attach_section_context_to_special_documents(documents: list[Document]) -> None:
    """
    用页码和纵坐标将图表绑定到它上方最近的章节/小节标题。

    同页没有标题时，自动继承前面页面的最近章节。
    """
    anchors_by_source: dict[str, list[dict]] = {}
    for document in documents:
        if document.metadata.get("content_type", "text") != "text":
            continue
        source = str(document.metadata.get("source", "unknown"))
        page = int(document.metadata.get("page", 0))
        for candidate in document.metadata.get("heading_candidates", []):
            heading = parse_section_heading(str(candidate.get("text", "")))
            if not heading:
                continue
            title, section = heading
            anchors_by_source.setdefault(source, []).append(
                {
                    "page": page,
                    "y": float(candidate.get("y", 0)),
                    "title": title,
                    "section": section,
                }
            )

    for anchors in anchors_by_source.values():
        anchors.sort(key=lambda item: (item["page"], item["y"]))
        inherited_section = "body"
        for anchor in anchors:
            if anchor["section"] == "body":
                anchor["section"] = inherited_section
            elif anchor["section"] not in {"front_matter", "body"}:
                inherited_section = anchor["section"]

    for document in documents:
        if document.metadata.get("content_type", "text") == "text":
            continue
        source = str(document.metadata.get("source", "unknown"))
        page = int(document.metadata.get("page", 0))
        y = float(document.metadata.get("bbox_top", float("inf")))
        eligible = [
            anchor
            for anchor in anchors_by_source.get(source, [])
            if anchor["page"] < page or (anchor["page"] == page and anchor["y"] <= y)
        ]
        if not eligible:
            continue
        anchor = eligible[-1]
        document.metadata["section"] = anchor["section"]
        document.metadata["section_title"] = anchor["title"]


def split_markdown_table(text: str, chunk_size: int) -> list[str]:
    """大表按行分块，每块重复表题和表头，保留列语义。"""
    lines = text.splitlines()
    if len(text) <= chunk_size or len(lines) <= 3:
        return [text]

    title = lines[0]
    header = lines[1:3]
    prefix = "\n".join([title, *header])
    parts: list[str] = []
    current = prefix
    for row in lines[3:]:
        if len(current) + len(row) + 1 > chunk_size and current != prefix:
            parts.append(current)
            current = prefix
        current = f"{current}\n{row}"
    if current != prefix:
        parts.append(current)
    return parts or [text]


SECTION_ALIASES = {
    "abstract": "abstract",
    "摘要": "abstract",
    "introduction": "introduction",
    "引言": "introduction",
    "绪论": "introduction",
    "related work": "related_work",
    "literature review": "related_work",
    "相关工作": "related_work",
    "文献综述": "related_work",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "model": "method",
    "algorithm": "method",
    "方法": "method",
    "模型": "method",
    "算法": "method",
    "experiment": "experiment",
    "experiments": "experiment",
    "experimental": "experiment",
    "experimental results": "experiment",
    "results": "experiment",
    "evaluation": "experiment",
    "实验": "experiment",
    "实验结果": "experiment",
    "结果": "experiment",
    "discussion": "discussion",
    "讨论": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "结论": "conclusion",
    "references": "references",
    "bibliography": "references",
    "参考文献": "references",
    "acknowledgements": "back_matter",
    "acknowledgments": "back_matter",
    "致谢": "back_matter",
}


def build_section_blocks(documents: list[Document]) -> list[SectionBlock]:
    """根据标题编号和常见论文章节名，在同一篇论文内跨页构建章节。"""
    grouped: dict[str, list[Document]] = {}
    for document in documents:
        if document.metadata.get("content_type", "text") != "text":
            continue
        source = str(document.metadata.get("source", "unknown"))
        grouped.setdefault(source, []).append(document)

    blocks: list[SectionBlock] = []
    for source, source_documents in grouped.items():
        source_documents.sort(key=lambda item: int(item.metadata.get("page", 0)))
        current = SectionBlock(source, "Front Matter", "front_matter", [])

        for document in source_documents:
            page = int(document.metadata.get("page", 0))
            for raw_line in document.text.splitlines():
                line = " ".join(raw_line.split()).strip()
                if not line:
                    continue

                heading = parse_section_heading(line)
                if heading:
                    if current.paragraphs:
                        blocks.append(current)
                    title, section = heading
                    if section == "body" and current.section not in {"front_matter", "body"}:
                        section = current.section
                    current = SectionBlock(source, title, section, [])
                    continue

                current.paragraphs.append((line, page))

        if current.paragraphs:
            blocks.append(current)

    return blocks


def parse_section_heading(text: str) -> tuple[str, str] | None:
    """识别 `3 Methodology`、`3.2 Network Architecture`及中英文无编号标题。"""
    compact = " ".join(text.split()).strip()
    if not compact or len(compact) > 120 or compact.endswith((".", ",", ";", "。", "，", "；")):
        return None

    numbered = re.match(r"^(?:section\s+)?(?:[ivxlcdm]+|\d+(?:\.\d+)*)[\s\.\u3001:：-]+(.+)$", compact, re.I)
    candidate = numbered.group(1).strip() if numbered else compact
    normalized = re.sub(r"\s+", " ", candidate.lower()).strip(" :-")

    section = canonical_section(normalized)
    if section:
        return compact, section

    # 带编号的短文本通常是小节标题；小节继承其语义标签，
    # 无法判定时标为 body，避免误将普通正文当成标题。
    if numbered and len(candidate.split()) <= 14 and len(candidate) <= 80:
        return compact, infer_section(candidate)
    return None


def canonical_section(title: str) -> str | None:
    normalized = re.sub(r"[^a-z\u4e00-\u9fff ]", " ", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for alias, section in sorted(SECTION_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized == alias or normalized.startswith(f"{alias} "):
            return section
    return None


def split_section_block(
    block: SectionBlock,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """在单个章节内按段落/句子递归切分，最后才使用字符长度兜底。"""
    header = f"[章节：{block.title}]"
    available_size = max(chunk_size - len(header) - 1, 100)
    units: list[tuple[str, int]] = []
    for paragraph, page in block.paragraphs:
        units.extend((part, page) for part in split_oversized_text(paragraph, available_size))

    results: list[tuple[str, int, int]] = []
    current: list[tuple[str, int]] = []
    current_length = 0

    for unit in units:
        unit_length = len(unit[0]) + (1 if current else 0)
        if current and current_length + unit_length > available_size:
            results.append(format_section_chunk(header, current))
            current = overlap_units(current, chunk_overlap)
            current_length = sum(len(text) for text, _ in current) + max(len(current) - 1, 0)
            while current and current_length + unit_length > available_size:
                current.pop(0)
                current_length = sum(len(text) for text, _ in current) + max(len(current) - 1, 0)

        current.append(unit)
        current_length += len(unit[0]) + (1 if len(current) > 1 else 0)

    if current:
        results.append(format_section_chunk(header, current))
    return results


def split_oversized_text(text: str, max_length: int) -> list[str]:
    if len(text) <= max_length:
        return [text]

    sentences = [item.strip() for item in re.split(r"(?<=[.!?。！？])\s+", text) if item.strip()]
    if len(sentences) == 1:
        return [text[index:index + max_length] for index in range(0, len(text), max_length)]

    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_length:
            parts.append(current)
            current = ""
        if len(sentence) > max_length:
            parts.extend(sentence[index:index + max_length] for index in range(0, len(sentence), max_length))
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def overlap_units(units: list[tuple[str, int]], overlap: int) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    length = 0
    for unit in reversed(units):
        if not selected and len(unit[0]) > overlap:
            return [(unit[0][-overlap:], unit[1])] if overlap > 0 else []
        if selected and length + len(unit[0]) > overlap:
            break
        selected.insert(0, unit)
        length += len(unit[0])
    return selected


def format_section_chunk(header: str, units: list[tuple[str, int]]) -> tuple[str, int, int]:
    pages = [page for _, page in units]
    return f"{header}\n" + "\n".join(text for text, _ in units), min(pages), max(pages)


def tokenize_for_bm25(text: str) -> list[str]:
    lower_text = text.lower()
    latin_tokens = re.findall(r"[a-z0-9][a-z0-9_\-\.]*", lower_text)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    symbol_tokens = re.findall(r"\b(?:alns|nsga-ii|moea/d|cvrp|vrp|tsp|2-opt|transformer|drl|rl)\b", lower_text)
    return latin_tokens + chinese_tokens + symbol_tokens


def should_attach_images(question: str) -> bool:
    """只在用户明确请求视觉解读时附加图片，避免表格/正文问题被偶然召回的 Figure 干扰。"""
    normalized = question.lower()
    visual_keywords = [
        "图片",
        "图像",
        "原图",
        "图中",
        "图里",
        "曲线图",
        "折线图",
        "散点图",
        "柱状图",
        "流程图",
        "架构图",
        "示意图",
        "视觉",
        "figure",
        "fig.",
        "image",
        "diagram",
        "plot",
        "chart",
    ]
    return any(keyword in normalized for keyword in visual_keywords)


def build_contextual_retrieval_query(question: str, conversation_history: list[dict]) -> str:
    if not conversation_history:
        return question

    recent_parts: list[str] = []
    for item in conversation_history[-3:]:
        recent_parts.append(str(item.get("question", "")))
        recent_parts.append(str(item.get("answer", ""))[:500])

    recent_context = "\n".join(part for part in recent_parts if part.strip())
    return f"{recent_context}\n当前追问：{question}"


def merge_search_hits(vector_hits: list[SearchHit], keyword_hits: list[SearchHit]) -> list[Chunk]:
    merged: dict[int, dict] = {}

    for rank, hit in enumerate(vector_hits):
        key = id(hit.chunk)
        item = merged.setdefault(
            key,
            {
                "chunk": hit.chunk,
                "vector_rank_score": 0.0,
                "keyword_rank_score": 0.0,
            },
        )
        item["vector_rank_score"] = max(item["vector_rank_score"], 1 / (rank + 1))

    for rank, hit in enumerate(keyword_hits):
        key = id(hit.chunk)
        item = merged.setdefault(
            key,
            {
                "chunk": hit.chunk,
                "vector_rank_score": 0.0,
                "keyword_rank_score": 0.0,
            },
        )
        item["keyword_rank_score"] = max(item["keyword_rank_score"], 1 / (rank + 1))

    for item in merged.values():
        item["fused_score"] = item["vector_rank_score"] * 0.65 + item["keyword_rank_score"] * 0.35
        item["chunk"].metadata["_retrieval"] = {
            "vector_rank_score": round(item["vector_rank_score"], 4),
            "keyword_rank_score": round(item["keyword_rank_score"], 4),
            "fused_score": round(item["fused_score"], 4),
            "final_score": None,
        }

    ranked = sorted(merged.values(), key=lambda item: item["fused_score"], reverse=True)
    return [item["chunk"] for item in ranked]


def rerank_candidates(question: str, candidates: list[Chunk]) -> list[Chunk]:
    for chunk in candidates:
        final_score = hybrid_relevance_score(question, chunk)
        retrieval = chunk.metadata.setdefault("_retrieval", {})
        retrieval["final_score"] = final_score
    return sorted(candidates, key=lambda chunk: chunk.metadata.get("_retrieval", {}).get("final_score", 0), reverse=True)


def hybrid_relevance_score(question: str, chunk: Chunk) -> float:
    """以融合排名为主，只施加小幅、可解释的意图奖励。"""
    retrieval = chunk.metadata.get("_retrieval", {})
    score = float(retrieval.get("fused_score", 0.0))
    question_tokens = set(tokenize_for_bm25(question))
    chunk_tokens = set(tokenize_for_bm25(chunk.text))
    score += min(len(question_tokens & chunk_tokens), 5) * 0.01

    if asks_method_question(question):
        if chunk.metadata.get("section") == "method":
            score += 0.08
        elif chunk.metadata.get("section") == "abstract":
            score += 0.03
    if asks_experiment_question(question):
        if chunk.metadata.get("section") == "experiment":
            score += 0.08
        if chunk.metadata.get("content_type") == "table":
            score += 0.05
    return round(score, 6)


def split_text_by_paragraphs(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{1,}", text) if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        if len(current) + len(paragraph) + 1 <= chunk_size:
            current = f"{current}\n{paragraph}"
        else:
            chunks.append(current)
            overlap_text = build_overlap_text(current, chunk_overlap)
            current = f"{overlap_text}\n{paragraph}".strip()

    if current:
        chunks.append(current)

    return chunks


def build_overlap_text(text: str, chunk_overlap: int) -> str:
    if chunk_overlap <= 0:
        return ""

    words = text.split()
    selected: list[str] = []
    length = 0
    for word in reversed(words):
        next_length = length + len(word) + (1 if selected else 0)
        if next_length > chunk_overlap:
            break
        selected.append(word)
        length = next_length
    return " ".join(reversed(selected))


def infer_section(text: str) -> str:
    lower_text = text.lower()
    if is_reference_like(text):
        return "references"
    if is_author_or_back_matter(text):
        return "back_matter"
    if re.search(r"\b(abstract|摘要)\b", lower_text):
        return "abstract"
    if re.search(r"\b(introduction|引言)\b", lower_text):
        return "introduction"
    if re.search(r"\b(related work|literature review|preliminaries|background|相关工作|文献综述|背景)\b", lower_text):
        return "related_work"
    if re.search(r"\b(method|methodology|approach|算法|方法)\b", lower_text):
        return "method"
    if re.search(r"\b(experiment|evaluation|result|discussion|case study|实验|结果|讨论|案例)\b", lower_text):
        return "experiment"
    if re.search(r"\b(conclusion|结论)\b", lower_text):
        return "conclusion"
    return "body"


def is_reference_like(text: str) -> bool:
    lower_text = text.lower()
    if re.search(r"(^|\n)\s*(references|bibliography|参考文献)\s*($|\n)", lower_text):
        return True

    citation_count = len(re.findall(r"\[\d+\]|\(\d{4}\)|\b(19|20)\d{2}\b", text))
    doi_or_journal_count = len(re.findall(r"\bdoi\b|arxiv|journal|proceedings|conference", lower_text))
    return citation_count >= 8 and doi_or_journal_count >= 2


def is_author_or_back_matter(text: str) -> bool:
    lower_text = text.lower()
    if re.search(r"\b(author biography|biography|about the author|acknowledg(e)?ments?)\b", lower_text):
        return True
    if any(keyword in text for keyword in ["作者简介", "作者介绍", "致谢"]):
        return True
    bio_words = ["professor", "received the", "ph.d.", "ieee", "editorial board", "associate editor"]
    return sum(1 for word in bio_words if word in lower_text) >= 3


def filter_retrieval_candidates(question: str, candidates: list[Chunk]) -> list[Chunk]:
    if asks_about_references(question):
        return candidates

    filtered = [
        chunk
        for chunk in candidates
        if chunk.metadata.get("section") not in {"references", "back_matter"}
        and not is_reference_like(chunk.text)
        and not is_author_or_back_matter(chunk.text)
    ]
    return filtered or candidates


def asks_about_references(question: str) -> bool:
    return any(
        keyword in question.lower()
        for keyword in ["参考文献", "引用", "related work", "reference", "references", "文献综述"]
    )


def needs_source_diversity(question: str) -> bool:
    return any(keyword in question for keyword in ["几篇", "三篇", "两篇", "每篇", "各自", "分别", "对比"])


def asks_method_question(question: str) -> bool:
    lower_question = question.lower()
    return any(
        keyword in lower_question
        for keyword in [
            "方法",
            "算法",
            "模型",
            "框架",
            "method",
            "algorithm",
            "approach",
            "model",
            "framework",
        ]
    )


def asks_experiment_question(question: str) -> bool:
    lower_question = question.lower()
    return any(
        keyword in lower_question
        for keyword in ["实验", "对比", "指标", "数据集", "experiment", "evaluation", "dataset", "metric", "baseline"]
    )


def retrieve_diverse_method_chunks(question: str, candidates: list[Chunk], top_k: int) -> list[Chunk]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in candidates:
        source = chunk.metadata.get("source", "unknown")
        grouped.setdefault(source, []).append(chunk)

    selected: list[Chunk] = []
    per_source = max(2, min(3, top_k // max(len(grouped), 1) + 1))

    for source in sorted(grouped):
        ranked = sorted(grouped[source], key=method_relevance_score, reverse=True)
        selected.extend(ranked[:per_source])

    if len(selected) < top_k:
        for chunk in candidates:
            if chunk not in selected:
                selected.append(chunk)
            if len(selected) >= top_k:
                break

    return rerank_candidates(question, selected)[:top_k]


def method_relevance_score(chunk: Chunk) -> int:
    text = chunk.text.lower()
    section = chunk.metadata.get("section", "body")
    score = 0

    section_scores = {
        "method": 8,
        "abstract": 6,
        "introduction": 5,
        "related_work": 1,
        "experiment": 4,
        "body": 2,
        "conclusion": 1,
        "references": -20,
        "back_matter": -20,
    }
    score += section_scores.get(section, 0)

    positive_keywords = [
        "we propose",
        "we present",
        "proposed",
        "method",
        "methodology",
        "approach",
        "framework",
        "algorithm",
        "model",
        "architecture",
        "neural",
        "heuristic",
        "reinforcement learning",
        "optimization",
        "我们提出",
        "本文提出",
        "方法",
        "算法",
        "模型",
        "框架",
        "启发式",
        "强化学习",
        "优化",
    ]
    score += sum(2 for keyword in positive_keywords if keyword in text)

    if is_reference_like(chunk.text) or is_author_or_back_matter(chunk.text):
        score -= 40

    return score
