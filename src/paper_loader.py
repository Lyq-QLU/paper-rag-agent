from dataclasses import dataclass
from pathlib import Path
import re

import fitz


@dataclass
class Document:
    text: str
    metadata: dict


CAPTION_PATTERN = re.compile(
    r"^\s*("
    r"(?:(?:figure|fig\.?)\s*[A-Za-z]?\d+(?:[.\-]\d+)*\s*[.:])"
    r"|(?:table\s*[A-Za-z]?\d+(?:[.\-]\d+)*\s*[.:])"
    r"|(?:[图表]\s*[A-Za-z]?\d+(?:[.\-]\d+)*\s*[.:：、]?))"
    r"\s*(.*)$",
    re.IGNORECASE,
)


def load_pdf_documents(pdf_paths: list[Path]) -> list[Document]:
    """
    从 PDF 中生成三类可检索文档：正文、Markdown 表格和图片说明。

    表格和图片会作为独立 Document，避免被普通正文切块破坏。
    嵌入的位图会保存在 PDF 旁边的 assets/<pdf-name>/ 目录中。
    """
    documents: list[Document] = []

    for pdf_path in pdf_paths:
        pdf = fitz.open(str(pdf_path))
        asset_dir = pdf_path.parent / "assets" / sanitize_asset_name(pdf_path.stem)

        try:
            for page_index, page in enumerate(pdf, start=1):
                captions = extract_captions(page)
                text, heading_candidates = extract_body_content(page, captions)

                if text:
                    documents.append(
                        Document(
                            text=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_index,
                                "content_type": "text",
                                "heading_candidates": heading_candidates,
                            },
                        )
                    )

                documents.extend(extract_table_documents(page, pdf_path.name, page_index, captions))
                documents.extend(
                    extract_figure_documents(
                        pdf,
                        page,
                        pdf_path.name,
                        page_index,
                        captions,
                        asset_dir,
                    )
                )
        finally:
            pdf.close()

    return documents


def extract_body_content(page, captions: list[dict]) -> tuple[str, list[dict]]:
    """从正文中排除已单独入库的表格区域和图表题，减少重复召回。"""
    try:
        table_bboxes = [table.bbox for table in page.find_tables().tables]
    except Exception:
        table_bboxes = []
    caption_bboxes = [item["bbox"] for item in captions]

    lines: list[str] = []
    heading_candidates: list[dict] = []
    for block in page.get_text("blocks", sort=True):
        bbox = block[:4]
        if any(bbox_overlap_ratio(bbox, table_bbox) >= 0.5 for table_bbox in table_bboxes):
            continue
        if any(bbox_overlap_ratio(bbox, caption_bbox) >= 0.8 for caption_bbox in caption_bboxes):
            continue
        text = normalize_text(block[4])
        if text:
            lines.append(text)
            first_line = text.splitlines()[0]
            if is_heading_candidate(first_line):
                heading_candidates.append({"text": first_line, "y": float(block[1])})
    return "\n".join(lines), heading_candidates


def extract_body_text(page, captions: list[dict]) -> str:
    """保留原有文本提取接口，便于独立调用。"""
    return extract_body_content(page, captions)[0]


def is_heading_candidate(text: str) -> bool:
    compact = " ".join(text.split()).strip()
    if not compact or len(compact) > 120 or compact.endswith((".", ",", ";", "。", "，", "；")):
        return False
    if re.match(r"^(?:section\s+)?(?:[ivxlcdm]+|\d+(?:\.\d+)*)[\s\.\u3001:：-]+.+$", compact, re.I):
        return True
    normalized = compact.lower().strip(" :-")
    return normalized in {
        "abstract", "introduction", "related work", "literature review",
        "method", "methods", "methodology", "experiments", "experimental results",
        "results", "discussion", "conclusion", "conclusions", "references",
        "摘要", "引言", "绪论", "相关工作", "文献综述", "方法", "模型", "算法",
        "实验", "实验结果", "结果", "讨论", "结论", "参考文献",
    }


def bbox_overlap_ratio(first, second) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    return intersection / first_area


def extract_table_documents(page, source: str, page_number: int, captions: list[dict]) -> list[Document]:
    documents: list[Document] = []
    try:
        finder = page.find_tables()
    except Exception:
        return documents

    for table_index, table in enumerate(finder.tables, start=1):
        rows = clean_table_rows(table.extract())
        if not rows or not any(any(cell for cell in row) for row in rows):
            continue

        caption = nearest_caption(captions, table.bbox, kind="table")
        title = caption["text"] if caption else f"Table {table_index}"
        markdown = table_to_markdown(rows)
        documents.append(
            Document(
                text=f"[表格：{title}]\n{markdown}",
                metadata={
                    "source": source,
                    "page": page_number,
                    "content_type": "table",
                    "caption": title,
                    "table_index": table_index,
                    "bbox_top": float(table.bbox[1]),
                },
            )
        )
    return documents


def extract_figure_documents(
    pdf,
    page,
    source: str,
    page_number: int,
    captions: list[dict],
    asset_dir: Path,
) -> list[Document]:
    documents: list[Document] = []
    figure_captions = [item for item in captions if item["kind"] == "figure"]
    image_infos = [item for item in page.get_image_info(xrefs=True) if item.get("xref", 0)]
    saved_paths: set[str] = set()

    for figure_index, caption in enumerate(figure_captions, start=1):
        image_info = nearest_image_above(image_infos, caption["bbox"])
        image_path = ""
        if image_info:
            image_path = save_embedded_image(
                pdf,
                image_info["xref"],
                asset_dir,
                page_number,
                figure_index,
            )
            if image_path:
                saved_paths.add(image_path)
        if not image_path:
            image_path = render_figure_region(
                page,
                caption["bbox"],
                asset_dir,
                page_number,
                figure_index,
            )
            if image_path:
                saved_paths.add(image_path)

        context = nearby_figure_context(page, caption["bbox"])
        description = caption["text"]
        if context and context not in description:
            description = f"{description}\n图片附近正文：{context}"

        documents.append(
            Document(
                text=f"[图片：{caption['text']}]\n{description}",
                metadata={
                    "source": source,
                    "page": page_number,
                    "content_type": "figure",
                    "caption": caption["text"],
                    "figure_index": figure_index,
                    "image_path": image_path,
                    "bbox_top": float(caption["bbox"][1]),
                },
            )
        )

    # 没有可识别图题的嵌入图片仍保存，但不用无语义的占位文本污染索引。
    for image_index, image_info in enumerate(image_infos, start=1):
        if image_info.get("width", 0) < 120 or image_info.get("height", 0) < 120:
            continue
        path = save_embedded_image(pdf, image_info["xref"], asset_dir, page_number, image_index)
        if path:
            saved_paths.add(path)

    return documents


def render_figure_region(
    page,
    caption_bbox,
    asset_dir: Path,
    page_number: int,
    figure_index: int,
) -> str:
    """
    对矢量图或组合图进行页面区域渲染。

    论文 Caption 通常位于图下方，因此截取 Caption 上方一段区域。
    对双栏 Caption 使用所在栏，跨栏 Caption 则使用整页宽度。
    """
    try:
        page_rect = page.rect
        caption_width = caption_bbox[2] - caption_bbox[0]
        full_width = float(page_rect.width)
        if caption_width < full_width * 0.62:
            center = (caption_bbox[0] + caption_bbox[2]) / 2
            if center < full_width / 2:
                left, right = float(page_rect.x0), full_width / 2
            else:
                left, right = full_width / 2, float(page_rect.x1)
        else:
            left, right = float(page_rect.x0), float(page_rect.x1)

        bottom = max(float(page_rect.y0), float(caption_bbox[1]) - 2)
        region_height = min(float(page_rect.height) * 0.58, 470.0)
        top = max(float(page_rect.y0), bottom - region_height)
        if bottom - top < 40:
            return ""

        clip = fitz.Rect(left, top, right, bottom)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
        asset_dir.mkdir(parents=True, exist_ok=True)
        path = asset_dir / f"page_{page_number:04d}_figure_{figure_index:02d}_rendered.png"
        pixmap.save(str(path))
        return str(path)
    except Exception:
        return ""


def extract_captions(page) -> list[dict]:
    captions: list[dict] = []
    for block in page.get_text("blocks", sort=True):
        text = normalize_text(block[4])
        if not text:
            continue
        first_line = text.splitlines()[0]
        match = CAPTION_PATTERN.match(first_line)
        if not match:
            continue
        label = match.group(1).lower()
        kind = "table" if label.startswith(("table", "表")) else "figure"
        captions.append({"text": first_line, "kind": kind, "bbox": block[:4]})
    return captions


def nearest_caption(captions: list[dict], bbox, kind: str) -> dict | None:
    matching = [item for item in captions if item["kind"] == kind]
    if not matching:
        return None
    top, bottom = bbox[1], bbox[3]
    return min(
        matching,
        key=lambda item: min(abs(item["bbox"][1] - bottom), abs(top - item["bbox"][3])),
    )


def nearest_image_above(image_infos: list[dict], caption_bbox) -> dict | None:
    candidates = []
    for image in image_infos:
        bbox = image.get("bbox")
        if not bbox or image.get("width", 0) < 120 or image.get("height", 0) < 120:
            continue
        # 论文图题通常在图片下方，优先匹配水平相交的最近图片。
        horizontal_overlap = max(0, min(bbox[2], caption_bbox[2]) - max(bbox[0], caption_bbox[0]))
        if bbox[3] <= caption_bbox[1] + 20 and horizontal_overlap > 0:
            candidates.append((caption_bbox[1] - bbox[3], image))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def nearby_figure_context(page, caption_bbox, max_length: int = 500) -> str:
    candidates: list[tuple[float, str]] = []
    for block in page.get_text("blocks", sort=True):
        text = normalize_text(block[4])
        if not text or CAPTION_PATTERN.match(text.splitlines()[0]):
            continue
        distance = min(abs(block[1] - caption_bbox[3]), abs(caption_bbox[1] - block[3]))
        candidates.append((distance, text))
    candidates.sort(key=lambda item: item[0])
    return " ".join(text for _, text in candidates[:2])[:max_length]


def save_embedded_image(pdf, xref: int, asset_dir: Path, page: int, index: int) -> str:
    try:
        image = pdf.extract_image(xref)
        if not image or len(image.get("image", b"")) < 1024:
            return ""
        asset_dir.mkdir(parents=True, exist_ok=True)
        extension = image.get("ext", "png")
        path = asset_dir / f"page_{page:04d}_figure_{index:02d}.{extension}"
        if not path.exists():
            path.write_bytes(image["image"])
        return str(path)
    except Exception:
        return ""


def clean_table_rows(rows) -> list[list[str]]:
    cleaned: list[list[str]] = []
    max_columns = max((len(row or []) for row in rows or []), default=0)
    for row in rows or []:
        cells = [normalize_cell(cell) for cell in (row or [])]
        cells.extend([""] * (max_columns - len(cells)))
        if any(cells):
            cleaned.append(cells)
    return cleaned


def normalize_cell(value) -> str:
    return " ".join(str(value or "").replace("|", "\\|").split())


def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    column_count = max(len(row) for row in rows)
    header = rows[0] + [""] * (column_count - len(rows[0]))
    body = [row + [""] * (column_count - len(row)) for row in rows[1:]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * column_count) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def sanitize_asset_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", value).strip("_")
    return cleaned[:80] or "paper"


def normalize_text(text: str) -> str:
    lines = []
    for line in text.replace("\x00", " ").splitlines():
        cleaned = " ".join(line.split())
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)
