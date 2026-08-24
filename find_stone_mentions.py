#!/usr/bin/env python3
"""Find natural-stone text mentions in a PDF and report their page numbers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from statistics import median
from typing import Any

from pipeline_logic import (
    DEFAULT_DPI,
    normalize_text,
    line_positions,
    preprocess_image,
    require_module,
    setup_logging,
    table_line_masks,
    words_in_bbox,
)
from pipestone_ocr import (
    DEFAULT_TESSERACT_LANGUAGE,
    collect_ocr_words,
    render_pdf_pages,
    words_to_lines,
)
from pipestone_semantic import STONE_KEYWORD_RE

TABLE_TITLE_REFERENCES = ("условные обозначения", "спецификация")
DEFAULT_TITLE_THRESHOLD = 0.55
DEFAULT_SEMANTIC_MODEL = "cointegrated/rubert-tiny2"
QUANTITY_HEADER_RE = re.compile(r"^кол(?:во|ич|ичество)?$", re.IGNORECASE)
FIRST_NUMBER_RE = re.compile(r"(?<!\d)\d+(?:[.,]\d+)?")
PERCENT_FORMULA_RE = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+)?)\s*\+\s*\d+(?:[.,]\d+)?\s*%\s*=\s*\d+(?:[.,]\d+)?",
    re.IGNORECASE,
)


class SemanticTitleMatcher:
    """Compare table titles to reference phrases using embedding cosine similarity."""

    method = "cosine"

    def __init__(self, references: tuple[str, ...], model_name: str) -> None:
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError(
                "sentence-transformers is required for semantic table-title search. "
                "Install it with: pip install sentence-transformers"
            )

        from sentence_transformers import SentenceTransformer

        self.references = references
        self.model = SentenceTransformer(model_name)
        self.reference_embeddings = self.model.encode(
            list(references),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def best_match(self, text: str) -> tuple[str, float]:
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        # The vectors are normalized, so their dot product is cosine similarity.
        similarities = self.reference_embeddings @ embedding
        best_index = int(similarities.argmax())
        return self.references[best_index], float(similarities[best_index])


class RegexTitleMatcher:
    """Match known Russian table titles without loading an embedding model."""

    method = "regex"

    def best_match(self, text: str) -> tuple[str, float]:
        normalized = normalize_text(text)
        if "спецификац" in normalized:
            return "спецификация", 1.0
        if "услов" in normalized and ("обознач" in normalized or "обозн" in normalized):
            return "условные обозначения", 1.0
        return "", 0.0


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = float((ix1 - ix0) * (iy1 - iy0))
    area_a = float((a[2] - a[0]) * (a[3] - a[1]))
    area_b = float((b[2] - b[0]) * (b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1.0)


def find_table_bboxes(
    image: Any,
    max_tables: int = 20,
    masks: tuple[Any, Any, Any] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Detect ruled tables on a rendered PDF page."""
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    height, width = image.shape[:2]
    if masks is None:
        binary = preprocess_image(image)["binary"]
        horizontal, vertical, table_mask = table_line_masks(binary)
    else:
        horizontal, vertical, table_mask = masks
    contours, _ = cv2.findContours(table_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(height * width)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if w < width * 0.05 or h < height * 0.03:
            continue
        if area < page_area * 0.002 or area > page_area * 0.92:
            continue

        crop = table_mask[y : y + h, x : x + w]
        horizontal_crop = horizontal[y : y + h, x : x + w]
        vertical_crop = vertical[y : y + h, x : x + w]
        line_density = float(np.count_nonzero(crop)) / max(area, 1.0)
        if line_density < 0.006:
            continue
        if not np.any(horizontal_crop) or not np.any(vertical_crop):
            continue
        candidates.append((area * (1.0 + line_density), (x, y, x + w, y + h)))

    selected: list[tuple[int, int, int, int]] = []
    for _, bbox in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(_bbox_iou(bbox, existing) >= 0.85 for existing in selected):
            continue
        selected.append(bbox)
        if len(selected) >= max_tables:
            break
    return selected


def find_semantic_tables(
    image: Any,
    words: list[Any],
    matcher: Any,
    threshold: float,
) -> list[dict[str, Any]]:
    """Return tables whose nearby title is semantically similar to a reference."""
    word_heights = [max(1.0, word.bbox[3] - word.bbox[1]) for word in words]
    typical_word_height = float(median(word_heights)) if word_heights else 20.0
    matches: list[dict[str, Any]] = []
    binary = preprocess_image(image)["binary"]
    masks = table_line_masks(binary)
    horizontal, vertical, _ = masks

    for table_bbox in find_table_bboxes(image, masks=masks):
        x0, y0, x1, y1 = table_bbox
        table_height = max(1, y1 - y0)
        title_band_above = max(typical_word_height * 8.0, min(table_height * 0.25, 300.0))
        title_band_inside = max(typical_word_height * 5.0, min(table_height * 0.25, 220.0))
        title_bbox = (
            float(x0),
            max(0.0, float(y0) - title_band_above),
            float(x1),
            min(float(image.shape[0]), float(y0) + title_band_inside),
        )
        title_lines = words_to_lines(words_in_bbox(words, title_bbox))
        if not title_lines:
            continue

        best_title = ""
        best_reference = ""
        best_similarity = -1.0
        for line in title_lines:
            reference, similarity = matcher.best_match(normalize_text(line.text))
            if similarity > best_similarity:
                best_title = line.text
                best_reference = reference
                best_similarity = similarity

        if best_similarity < threshold:
            continue
        matches.append(
            {
                "bbox": table_bbox,
                "title": best_title,
                "reference": best_reference,
                "similarity": round(best_similarity, 4) if matcher.method == "cosine" else None,
                "match_method": matcher.method,
                "horizontal_lines": line_positions(horizontal, table_bbox, "horizontal"),
                "vertical_lines": line_positions(vertical, table_bbox, "vertical"),
            }
        )

    matches.sort(
        key=lambda item: float(item["similarity"]) if item["similarity"] is not None else 1.0,
        reverse=True,
    )
    return matches


def _compact_quantity_header(text: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", normalize_text(text))


def find_quantity_header(words: list[Any]) -> tuple[str, tuple[float, float, float, float]] | None:
    """Find a quantity-column header, including OCR-split forms such as `Кол - во`."""
    candidates: list[tuple[int, str, tuple[float, float, float, float]]] = []
    sorted_words = sorted(words, key=lambda word: ((word.bbox[1] + word.bbox[3]) / 2.0, word.bbox[0]))

    for index, word in enumerate(sorted_words):
        parts = [word]
        for next_index in range(index, min(index + 3, len(sorted_words))):
            if next_index > index:
                previous = sorted_words[next_index - 1]
                current = sorted_words[next_index]
                previous_cy = (previous.bbox[1] + previous.bbox[3]) / 2.0
                current_cy = (current.bbox[1] + current.bbox[3]) / 2.0
                typical_height = max(previous.bbox[3] - previous.bbox[1], current.bbox[3] - current.bbox[1], 1.0)
                horizontal_gap = current.bbox[0] - previous.bbox[2]
                if abs(current_cy - previous_cy) > typical_height or horizontal_gap > typical_height * 2.5:
                    break
                parts.append(current)

            text = " ".join(part.text for part in parts)
            compact = _compact_quantity_header(text)
            if QUANTITY_HEADER_RE.fullmatch(compact):
                bbox = (
                    min(part.bbox[0] for part in parts),
                    min(part.bbox[1] for part in parts),
                    max(part.bbox[2] for part in parts),
                    max(part.bbox[3] for part in parts),
                )
                candidates.append((len(compact), text, bbox))

    if not candidates:
        return None
    _, text, bbox = max(candidates, key=lambda item: item[0])
    return text.strip(" :;"), bbox


def _grid_interval(lines: list[int], value: float, lower: int, upper: int) -> tuple[int, int] | None:
    positions = sorted({lower, upper, *(line for line in lines if lower <= line <= upper)})
    for start, end in zip(positions, positions[1:]):
        if start <= value <= end and end > start:
            return start, end
    return None


def extract_quantity_column(words: list[Any], table: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the first numeric value from the quantity cell of every table row."""
    table_bbox = tuple(int(value) for value in table["bbox"])
    x0, y0, x1, y1 = table_bbox
    table_words = words_in_bbox(words, tuple(float(value) for value in table_bbox))
    header = find_quantity_header(table_words)
    if header is None:
        return None

    header_text, header_bbox = header
    header_cx = (header_bbox[0] + header_bbox[2]) / 2.0
    header_cy = (header_bbox[1] + header_bbox[3]) / 2.0
    column_interval = _grid_interval(list(table["vertical_lines"]), header_cx, x0, x1)
    header_row = _grid_interval(list(table["horizontal_lines"]), header_cy, y0, y1)
    if column_interval is None or header_row is None:
        return None

    column_x0, column_x1 = column_interval
    horizontal_lines = sorted({y0, y1, *(line for line in table["horizontal_lines"] if y0 <= line <= y1)})
    rows: list[dict[str, Any]] = []
    for row_y0, row_y1 in zip(horizontal_lines, horizontal_lines[1:]):
        if row_y1 <= header_row[1] or row_y1 - row_y0 < 3:
            continue
        cell_words = words_in_bbox(
            table_words,
            (float(column_x0), float(row_y0), float(column_x1), float(row_y1)),
        )
        raw_text = " ".join(word.text for word in sorted(cell_words, key=lambda word: word.bbox[0])).strip()
        number_match = FIRST_NUMBER_RE.search(raw_text)
        rows.append(
            {
                "row_bbox": [float(x0), float(row_y0), float(x1), float(row_y1)],
                "cell_bbox": [float(column_x0), float(row_y0), float(column_x1), float(row_y1)],
                "raw": raw_text,
                "value": number_match.group(0) if number_match else None,
            }
        )

    return {
        "header": header_text,
        "column_bbox": [float(column_x0), float(y0), float(column_x1), float(y1)],
        "rows": rows,
    }


def quantity_for_line(line_bbox: tuple[float, float, float, float], quantity_column: dict[str, Any] | None) -> dict[str, Any] | None:
    if quantity_column is None:
        return None
    line_cy = (line_bbox[1] + line_bbox[3]) / 2.0
    for row in quantity_column["rows"]:
        row_y0, row_y1 = row["row_bbox"][1], row["row_bbox"][3]
        if row_y0 <= line_cy <= row_y1:
            return row
    return None


def find_stone_mentions(
    pdf_path: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    ocr_backend: str = "tesseract",
    ocr_language: str = DEFAULT_TESSERACT_LANGUAGE,
    force_ocr: bool = False,
    semantic_search: bool = False,
    title_threshold: float = DEFAULT_TITLE_THRESHOLD,
    semantic_model: str = DEFAULT_SEMANTIC_MODEL,
) -> list[dict[str, Any]]:
    """Return stone mentions found only inside semantically selected tables."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    matcher = (
        SemanticTitleMatcher(TABLE_TITLE_REFERENCES, semantic_model)
        if semantic_search
        else RegexTitleMatcher()
    )
    rendered_pages = render_pdf_pages(pdf_path, dpi=dpi)
    words_by_page = collect_ocr_words(
        pdf_path,
        rendered_pages,
        backend=ocr_backend,
        force_ocr=force_ocr,
        tesseract_language=ocr_language,
    )
    images_by_page = {int(page["page"]): page["image"] for page in rendered_pages}

    mentions: list[dict[str, Any]] = []
    seen: set[tuple[int, str, tuple[int, int, int, int]]] = set()
    for page, words in sorted(words_by_page.items()):
        image = images_by_page.get(page)
        if image is None:
            continue
        semantic_tables = find_semantic_tables(image, words, matcher, title_threshold)
        for table in semantic_tables:
            table_words = words_in_bbox(words, tuple(float(value) for value in table["bbox"]))
            quantity_column = extract_quantity_column(words, table)
            for line in words_to_lines(table_words):
                normalized_line = normalize_text(line.text)
                match = STONE_KEYWORD_RE.search(normalized_line)
                if match is None:
                    continue

                rounded_bbox = tuple(int(round(value)) for value in line.bbox)
                key = (page, normalized_line, rounded_bbox)
                if key in seen:
                    continue
                seen.add(key)
                quantity_row = quantity_for_line(line.bbox, quantity_column)
                formula_match = PERCENT_FORMULA_RE.search(line.text)
                quantity_value = quantity_row["value"] if quantity_row else None
                quantity_raw = quantity_row["raw"] if quantity_row else None
                if formula_match is not None:
                    quantity_value = formula_match.group(1)
                    quantity_raw = formula_match.group(0)
                mentions.append(
                    {
                        "page": page,
                        "match": match.group(0),
                        "line": line.text,
                        "bbox": list(line.bbox),
                        "confidence": line.confidence,
                        "source": line.source,
                        "table_bbox": list(table["bbox"]),
                        "table_title": table["title"],
                        "title_reference": table["reference"],
                        "title_similarity": table["similarity"],
                        "title_match_method": table["match_method"],
                        "quantity_header": quantity_column["header"] if quantity_column else None,
                        "quantity": quantity_value,
                        "quantity_raw": quantity_raw,
                        "quantity_cell_bbox": quantity_row["cell_bbox"] if quantity_row else None,
                    }
                )

    return [mention for mention in mentions if mention["quantity"] is not None]


def _markdown_cell(value: Any) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find natural-stone phrases in a PDF and print their page numbers."
    )
    parser.add_argument("--pdf", required=True, help="PDF file to search.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PDF rendering DPI used for OCR.")
    parser.add_argument("--ocr-backend", default="tesseract", help="OCR backend: tesseract, auto, or none.")
    parser.add_argument("--ocr-language", default=DEFAULT_TESSERACT_LANGUAGE, help="Tesseract languages.")
    parser.add_argument(
        "--semantic-search",
        action="store_true",
        help="Enable embedding-based table-title matching; disabled by default.",
    )
    parser.add_argument(
        "--title-threshold",
        type=float,
        default=DEFAULT_TITLE_THRESHOLD,
        help="Minimum cosine similarity for a table title (default: 0.55).",
    )
    parser.add_argument(
        "--semantic-model",
        default=DEFAULT_SEMANTIC_MODEL,
        help="Sentence Transformer model used for title embeddings.",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Run OCR even when the PDF already contains a text layer.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    if not 0.0 <= args.title_threshold <= 1.0:
        raise SystemExit("--title-threshold must be between 0 and 1")
    setup_logging()
    try:
        mentions = find_stone_mentions(
            args.pdf,
            dpi=args.dpi,
            ocr_backend=args.ocr_backend,
            ocr_language=args.ocr_language,
            force_ocr=args.force_ocr,
            semantic_search=args.semantic_search,
            title_threshold=args.title_threshold,
            semantic_model=args.semantic_model,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(mentions, ensure_ascii=False, indent=2))
        return 0

    if not mentions:
        print("Совпадения не найдены.")
        return 0

    mentions_by_page: dict[int, list[dict[str, Any]]] = {}
    for mention in mentions:
        mentions_by_page.setdefault(int(mention["page"]), []).append(mention)

    print(f"Найдено совпадений: {len(mentions)} на страницах: {len(mentions_by_page)}")
    for page, page_mentions in sorted(mentions_by_page.items()):
        print(f"\nСтраница {page} — совпадений: {len(page_mentions)}")
        for index, mention in enumerate(page_mentions, start=1):
            print(
                f"  {index}. {mention['match']}\n"
                f"     Таблица: {mention['table_title']} "
                + (
                    f"(эталон: {mention['title_reference']}, cosine={mention['title_similarity']:.4f})\n"
                    if mention["title_match_method"] == "cosine"
                    else f"(эталон: {mention['title_reference']}, regex)\n"
                )
                + f"     Строка: {mention['line']}\n"
                f"     {mention['quantity_header'] or 'Количество'}: "
                f"{mention['quantity'] if mention['quantity'] is not None else 'не найдено'}"
                + (
                    f" (ячейка: {mention['quantity_raw']})"
                    if mention["quantity_raw"] and mention["quantity_raw"] != mention["quantity"]
                    else ""
                )
            )

    print("\nИтоговая таблица")
    print("| Страница | Таблица | Строка | Кол-во |")
    print("|---:|---|---|---:|")
    for mention in mentions:
        print(
            f"| {mention['page']} "
            f"| {_markdown_cell(mention['table_title'])} "
            f"| {_markdown_cell(mention['line'])} "
            f"| {_markdown_cell(mention['quantity'])} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
