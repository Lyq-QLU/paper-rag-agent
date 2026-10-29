import sys
import tempfile
import types
import unittest
from pathlib import Path


# 切块测试不需要加载体积较大的向量模型与 FAISS。
fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatIP = object
sys.modules.setdefault("faiss", fake_faiss)

fake_numpy = types.ModuleType("numpy")
fake_numpy.ndarray = object
sys.modules.setdefault("numpy", fake_numpy)

fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", fake_sentence_transformers)

fake_fitz = types.ModuleType("fitz")
fake_fitz.open = object
sys.modules.setdefault("fitz", fake_fitz)

fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = object
fake_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
fake_openai.APIStatusError = type("APIStatusError", (Exception,), {})
fake_openai.RateLimitError = type("RateLimitError", (fake_openai.APIStatusError,), {})
sys.modules.setdefault("openai", fake_openai)

from src.paper_loader import CAPTION_PATTERN, Document, bbox_overlap_ratio, table_to_markdown
from src.llm import build_user_content, is_known_text_only_provider
from src.rag_pipeline import (
    build_section_blocks,
    parse_section_heading,
    should_attach_images,
    split_documents,
)


class StructuredChunkingTests(unittest.TestCase):
    def test_caption_pattern_rejects_body_reference_sentence(self):
        self.assertIsNone(CAPTION_PATTERN.match("Fig. 13 displays the interaction graphs"))
        self.assertIsNotNone(CAPTION_PATTERN.match("Fig. 13. Interaction graphs between approaches."))

    def test_multimodal_content_includes_existing_image(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "figure.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfigure-data")
            content = build_user_content("解读该图", [str(image_path)])

        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "解读该图"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_multimodal_content_falls_back_to_text_without_image(self):
        self.assertEqual(build_user_content("问题", ["missing.png"]), "问题")

    def test_deepseek_api_is_detected_as_text_only(self):
        self.assertTrue(is_known_text_only_provider("https://api.deepseek.com", "deepseek-chat"))
        self.assertFalse(is_known_text_only_provider("https://api.example.com/v1", "vision-model"))

    def test_images_are_only_attached_for_visual_questions(self):
        self.assertFalse(should_attach_images("分析实验表格数据"))
        self.assertTrue(should_attach_images("请分析 Fig. 13 中的曲线趋势"))
        self.assertTrue(should_attach_images("解读这张架构图"))

    def test_table_rows_become_markdown(self):
        markdown = table_to_markdown([["Method", "Score"], ["Ours", "95"]])
        self.assertIn("| Method | Score |", markdown)
        self.assertIn("| Ours | 95 |", markdown)

    def test_bbox_overlap_detects_table_text(self):
        self.assertEqual(bbox_overlap_ratio((10, 10, 20, 20), (0, 0, 30, 30)), 1.0)

    def test_recognizes_common_numbered_headings(self):
        self.assertEqual(parse_section_heading("2 Methodology"), ("2 Methodology", "method"))
        self.assertEqual(parse_section_heading("4.1 Experimental Setup"), ("4.1 Experimental Setup", "experiment"))
        self.assertEqual(parse_section_heading("参考文献"), ("参考文献", "references"))

    def test_builds_sections_across_page_boundaries(self):
        documents = [
            Document("1 Introduction\n背景内容。\n2 Methodology\n方法的第一部分。", {"source": "paper.pdf", "page": 1}),
            Document("方法的第二部分。\n3 Experiments\n实验设置。", {"source": "paper.pdf", "page": 2}),
        ]
        blocks = build_section_blocks(documents)

        self.assertEqual([block.section for block in blocks], ["introduction", "method", "experiment"])
        self.assertEqual([page for _, page in blocks[1].paragraphs], [1, 2])

    def test_chunks_keep_section_and_page_range_metadata(self):
        documents = [
            Document("2 Methodology\n方法的第一部分。", {"source": "paper.pdf", "page": 3}),
            Document("方法的第二部分。", {"source": "paper.pdf", "page": 4}),
        ]
        chunks = split_documents(documents)

        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].text.startswith("[章节：2 Methodology]"))
        self.assertEqual(chunks[0].metadata["section"], "method")
        self.assertEqual(chunks[0].metadata["page_start"], 3)
        self.assertEqual(chunks[0].metadata["page_end"], 4)

    def test_chunks_do_not_cross_section_boundaries(self):
        documents = [
            Document(
                "2 Methodology\n方法内容。\n3 Experiments\n实验内容。\nReferences\n[1] Other method.",
                {"source": "paper.pdf", "page": 5},
            )
        ]
        chunks = split_documents(documents)

        self.assertEqual([chunk.metadata["section"] for chunk in chunks], ["method", "experiment", "references"])
        self.assertNotIn("实验内容", chunks[0].text)
        self.assertNotIn("Other method", chunks[1].text)

    def test_table_and_figure_are_independent_chunks(self):
        documents = [
            Document("正文内容。", {"source": "paper.pdf", "page": 6, "content_type": "text"}),
            Document(
                "[表格：Table 1 Results]\n| Method | Score |\n| --- | --- |\n| Ours | 95 |",
                {
                    "source": "paper.pdf",
                    "page": 6,
                    "content_type": "table",
                    "caption": "Table 1 Results",
                },
            ),
            Document(
                "[图片：Figure 2 Architecture]\nFigure 2 Architecture",
                {
                    "source": "paper.pdf",
                    "page": 7,
                    "content_type": "figure",
                    "caption": "Figure 2 Architecture",
                    "image_path": "assets/figure.png",
                },
            ),
        ]
        chunks = split_documents(documents)
        by_type = {chunk.metadata["content_type"]: chunk for chunk in chunks}

        self.assertEqual(set(by_type), {"text", "table", "figure"})
        self.assertIn("| Ours | 95 |", by_type["table"].text)
        self.assertEqual(by_type["figure"].metadata["image_path"], "assets/figure.png")

    def test_table_and_figure_inherit_nearest_section_title(self):
        documents = [
            Document(
                "4 Experiments\n4.1 Comparison Results\n实验正文。",
                {
                    "source": "paper.pdf",
                    "page": 8,
                    "content_type": "text",
                    "heading_candidates": [
                        {"text": "4 Experiments", "y": 80},
                        {"text": "4.1 Comparison Results", "y": 220},
                    ],
                },
            ),
            Document(
                "[表格：Table 2 Results]\n| Method | Score |\n| --- | --- |\n| Ours | 95 |",
                {
                    "source": "paper.pdf",
                    "page": 8,
                    "bbox_top": 400,
                    "content_type": "table",
                    "caption": "Table 2 Results",
                },
            ),
            Document(
                "[图片：Figure 3 Curves]\nFigure 3 Curves",
                {
                    "source": "paper.pdf",
                    "page": 9,
                    "bbox_top": 100,
                    "content_type": "figure",
                    "caption": "Figure 3 Curves",
                },
            ),
        ]
        chunks = split_documents(documents)
        special = [chunk for chunk in chunks if chunk.metadata["content_type"] != "text"]

        self.assertTrue(all(chunk.metadata["section"] == "experiment" for chunk in special))
        self.assertTrue(all(chunk.metadata["section_title"] == "4.1 Comparison Results" for chunk in special))
        self.assertEqual(special[0].metadata["caption"], "Table 2 Results")


if __name__ == "__main__":
    unittest.main()
