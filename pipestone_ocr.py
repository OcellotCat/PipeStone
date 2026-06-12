#!/usr/bin/env python3
"""OCR and text extraction utilities for PipeStone."""

from __future__ import annotations

import importlib.util
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

logger = logging.getLogger("pipestone.ocr")

DIMENSION_RE = re.compile(
    r"(?<![\d.,])("
    r"\d{1,3}(?:[\s\u00a0]\d{3})+|"
    r"\d{3,6}(?:[.,]\d+)?"
    r")\s*(мм|mm|м|m)?(?![\d.,])",
    re.IGNORECASE,
)
BAD_DIMENSION_CONTEXT_RE = re.compile(
    r"(лист|дата|стадия|масштаб|гост|проверил|разраб|инв\.?|подп\.?|формат|"
    r"экспликац|ведомост|примечан|толщин|кам(?:е|ё)?н|натурал)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OcrWord:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None
    source: str


@dataclass(frozen=True)
class OcrLine:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None
    source: str


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def require_module(module_name: str, install_hint: str) -> Any:
    if not has_module(module_name):
        raise RuntimeError(f"Не найден модуль {module_name}. Установите: {install_hint}")
    return __import__(module_name)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("ё", "е").replace("Ё", "Е")).strip().lower()


def bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def extract_pdf_text_words(pdf_path: Path, rendered_pages: list[dict]) -> dict[int, list[OcrWord]]:
    if not has_module("fitz"):
        return {}

    fitz = require_module("fitz", "pip install pymupdf")
    page_by_number = {page["page"]: page for page in rendered_pages}
    words_by_page: dict[int, list[OcrWord]] = defaultdict(list)

    doc = fitz.open(str(pdf_path))
    for index, page in enumerate(doc, start=1):
        rendered = page_by_number[index]
        scale_x = rendered["width_px"] / max(float(page.rect.width), 1.0)
        scale_y = rendered["height_px"] / max(float(page.rect.height), 1.0)
        for item in page.get_text("words"):
            x0, y0, x1, y1, text = item[:5]
            text = str(text).strip()
            if not text:
                continue
            words_by_page[index].append(
                OcrWord(
                    page=index,
                    text=text,
                    bbox=(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                    confidence=1.0,
                    source="pdf_text",
                )
            )
    doc.close()
    return words_by_page


def words_to_lines(words: list[OcrWord]) -> list[OcrLine]:
    if not words:
        return []

    heights = [max(1.0, word.bbox[3] - word.bbox[1]) for word in words]
    y_threshold = max(8.0, median(heights) * 0.85)
    sorted_words = sorted(words, key=lambda word: (bbox_center(word.bbox)[1], word.bbox[0]))
    raw_lines: list[list[OcrWord]] = []

    for word in sorted_words:
        _, cy = bbox_center(word.bbox)
        if raw_lines:
            last_line = raw_lines[-1]
            last_y = median([bbox_center(item.bbox)[1] for item in last_line])
            if abs(cy - last_y) <= y_threshold:
                last_line.append(word)
                continue
        raw_lines.append([word])

    lines: list[OcrLine] = []
    for raw_line in raw_lines:
        raw_line.sort(key=lambda word: word.bbox[0])
        text = " ".join(word.text for word in raw_line).strip()
        if not text:
            continue
        confidences = [word.confidence for word in raw_line if word.confidence is not None]
        source = "+".join(sorted({word.source for word in raw_line}))
        lines.append(
            OcrLine(
                page=raw_line[0].page,
                text=text,
                bbox=bbox_union([word.bbox for word in raw_line]),
                confidence=sum(confidences) / len(confidences) if confidences else None,
                source=source,
            )
        )
    return lines


def collect_ocr_words(
    pdf_path: Path,
    rendered_pages: list[dict],
    backend: str,
    force_ocr: bool,
    tesseract_psm: int = 11,
) -> dict[int, list[OcrWord]]:
    text_words = extract_pdf_text_words(pdf_path, rendered_pages)
    words_by_page: dict[int, list[OcrWord]] = defaultdict(list)
    warnings: list[str] = []

    for page in rendered_pages:
        pdf_words = text_words.get(page["page"], [])
        if pdf_words and not force_ocr:
            words_by_page[page["page"]].extend(pdf_words)
            logger.info("Page %s OCR: using PDF text layer (%s words)", page["page"], len(pdf_words))
            continue

        if pdf_words:
            words_by_page[page["page"]].extend(pdf_words)

        image_words, warning = run_image_ocr(page["image"], page["page"], backend, tesseract_psm=tesseract_psm)
        if warning:
            warnings.append(f"page {page['page']}: {warning}")
            logger.warning("Page %s OCR warning: %s", page["page"], warning)
        words_by_page[page["page"]].extend(image_words)
        logger.info(
            "Page %s OCR: %s image words, %s PDF words",
            page["page"],
            len(image_words),
            len(pdf_words),
        )

    if warnings:
        logger.warning("OCR warnings: %s", "; ".join(warnings))
    return dict(words_by_page)


def run_image_ocr(image: Any, page_number: int, backend: str, tesseract_psm: int = 11) -> tuple[list[OcrWord], str | None]:
    backend = backend.lower()
    if backend == "none":
        return [], "image OCR disabled"

    if backend == "auto":
        for candidate in ("tesseract", "paddleocr", "easyocr"):
            words, warning = run_image_ocr(image, page_number, candidate, tesseract_psm=tesseract_psm)
            if words:
                return words, None
            if warning:
                logger.debug("OCR backend %s skipped: %s", candidate, warning)
        return [], "no OCR backend produced text; install pytesseract/easyocr/paddleocr"

    if backend == "tesseract":
        return run_tesseract_ocr(image, page_number, tesseract_psm=tesseract_psm)
    if backend == "easyocr":
        return run_easyocr(image, page_number)
    if backend == "paddleocr":
        return run_paddleocr(image, page_number)
    return [], f"unknown OCR backend: {backend}"


def run_tesseract_ocr(
    image: Any, page_number: int, tesseract_psm: int = 11
) -> tuple[list[OcrWord], str | None]:
    if not has_module("pytesseract"):
        return [], "pytesseract is not installed"
    if not has_module("PIL"):
        return [], "Pillow is not installed"

    import pytesseract
    from PIL import Image

    try:
        pil_image = Image.fromarray(image)
        try:
            data = pytesseract.image_to_data(
                pil_image,
                lang="rus+eng",
                config=f"--oem 3 --psm {tesseract_psm}",
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            data = pytesseract.image_to_data(
                pil_image,
                lang="eng",
                config=f"--oem 3 --psm {tesseract_psm}",
                output_type=pytesseract.Output.DICT,
            )
    except Exception as exc:
        return [], f"tesseract failed: {exc}"

    words: list[OcrWord] = []
    for i, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text:
            continue
        conf_raw = data.get("conf", ["-1"])[i]
        try:
            confidence = float(conf_raw) / 100.0
        except ValueError:
            confidence = None
        if confidence is not None and confidence < 0.2:
            continue
        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        word = OcrWord(
            page=page_number,
            text=text,
            bbox=(x, y, x + w, y + h),
            confidence=confidence,
            source="tesseract",
        )
        words.append(word)
        logger.debug("OCR word: page=%s text=%r conf=%s bbox=(%.1f,%.1f,%.1f,%.1f)", page_number, text, confidence, x, y, x + w, y + h)
    return words, None


_EASYOCR_READER: Any = None


def run_easyocr(image: Any, page_number: int) -> tuple[list[OcrWord], str | None]:
    if not has_module("easyocr"):
        return [], "easyocr is not installed"

    global _EASYOCR_READER
    try:
        import easyocr

        if _EASYOCR_READER is None:
            _EASYOCR_READER = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
        result = _EASYOCR_READER.readtext(image)
    except Exception as exc:
        return [], f"easyocr failed: {exc}"

    words: list[OcrWord] = []
    for box, text, confidence in result:
        text = str(text).strip()
        if not text:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        word = OcrWord(
            page=page_number,
            text=text,
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            confidence=float(confidence),
            source="easyocr",
        )
        words.append(word)
        logger.debug(
            "OCR word: page=%s text=%r conf=%.4f bbox=(%.1f,%.1f,%.1f,%.1f)",
            page_number,
            text,
            float(confidence),
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )
    return words, None


_PADDLE_OCR: Any = None


def run_paddleocr(image: Any, page_number: int) -> tuple[list[OcrWord], str | None]:
    if not has_module("paddleocr"):
        return [], "paddleocr is not installed"

    global _PADDLE_OCR
    try:
        from paddleocr import PaddleOCR

        if _PADDLE_OCR is None:
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="ru", show_log=False)
        result = _PADDLE_OCR.ocr(image, cls=True)
    except Exception as exc:
        return [], f"paddleocr failed: {exc}"

    words: list[OcrWord] = []
    for page_result in result or []:
        for item in page_result or []:
            if not item or len(item) < 2:
                continue
            box = item[0]
            text_info = item[1]
            text = str(text_info[0]).strip()
            confidence = float(text_info[1]) if len(text_info) > 1 else None
            if not text:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            word = OcrWord(
                page=page_number,
                text=text,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                confidence=confidence,
                source="paddleocr",
            )
            words.append(word)
            logger.debug(
                "OCR word: page=%s text=%r conf=%s bbox=(%.1f,%.1f,%.1f,%.1f)",
                page_number,
                text,
                confidence,
                min(xs),
                min(ys),
                max(xs),
                max(ys),
            )
    return words, None


def parse_dimension_value(raw_value: str, raw_unit: str | None) -> float | None:
    value_text = raw_value.replace("\u00a0", " ").replace(" ", "").replace(",", ".")
    try:
        value = float(value_text)
    except ValueError:
        return None

    unit = (raw_unit or "").lower()
    if unit in {"м", "m"}:
        value *= 1000.0

    if value < 100.0 or value > 500000.0:
        return None
    return value


def find_dimension_candidates(words: list[OcrWord]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    for word in words:
        for match in DIMENSION_RE.finditer(word.text):
            value = parse_dimension_value(match.group(1), match.group(2))
            if value is None:
                continue
            key = (int(word.bbox[0] / 8), int(word.bbox[1] / 8), int(value))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "page": word.page,
                    "text": word.text,
                    "value_mm": value,
                    "bbox": word.bbox,
                    "source": word.source,
                }
            )

    for line in words_to_lines(words):
        if BAD_DIMENSION_CONTEXT_RE.search(line.text):
            continue
        for match in DIMENSION_RE.finditer(line.text):
            value = parse_dimension_value(match.group(1), match.group(2))
            if value is None:
                continue
            key = (int(line.bbox[0] / 8), int(line.bbox[1] / 8), int(value))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "page": line.page,
                    "text": match.group(0),
                    "value_mm": value,
                    "bbox": line.bbox,
                    "source": line.source,
                }
            )

    return candidates


@dataclass(frozen=True)
class RenderedPage:
    page: int
    image: Any
    width_px: int
    height_px: int
    width_pt: float
    height_pt: float


def render_pdf_pages(pdf_path: Path, dpi: int = 220) -> list[RenderedPage | dict]:
    if has_module("fitz"):
        return render_pdf_pages_pymupdf(pdf_path, dpi)
    if has_module("pdf2image"):
        return render_pdf_pages_pdf2image(pdf_path, dpi)
    raise RuntimeError(
        "Нет backend для рендера PDF. Установите pymupdf или pdf2image: "
        "pip install pymupdf"
    )


def render_pdf_pages_pymupdf(pdf_path: Path, dpi: int) -> list[dict]:
    fitz = require_module("fitz", "pip install pymupdf")
    np = require_module("numpy", "pip install numpy")

    pages: list[dict] = []
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = image[:, :, :3]
        pages.append({
            "page": index,
            "image": image.copy(),
            "width_px": pix.width,
            "height_px": pix.height,
            "width_pt": float(page.rect.width),
            "height_pt": float(page.rect.height),
        })
    doc.close()
    logger.info("PDF rendered: %s pages at %s DPI", len(pages), dpi)
    return pages


def render_pdf_pages_pdf2image(pdf_path: Path, dpi: int) -> list[dict]:
    pdf2image = require_module("pdf2image", "pip install pdf2image")
    np = require_module("numpy", "pip install numpy")
    images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)
    pages: list[dict] = []
    for index, pil_image in enumerate(images, start=1):
        rgb = pil_image.convert("RGB")
        image = np.array(rgb)
        pages.append({
            "page": index,
            "image": image,
            "width_px": rgb.width,
            "height_px": rgb.height,
            "width_pt": rgb.width / (dpi / 72.0),
            "height_pt": rgb.height / (dpi / 72.0),
        })
    logger.info("PDF rendered with pdf2image: %s pages at %s DPI", len(pages), dpi)
    return pages


def estimate_scale_for_page(image: Any, words: list[OcrWord], page_number: int) -> dict[str, Any]:
    dimension_candidates = find_dimension_candidates(words)
    if not dimension_candidates:
        return {"page": page_number, "mm_per_px": None, "source": "missing", "candidates": []}

    from pipestone_cv import detect_line_segments
    segments = detect_line_segments(image)
    if not segments:
        return {
            "page": page_number,
            "mm_per_px": None,
            "source": "missing_lines",
            "candidates": [],
        }

    height_px, width_px = image.shape[:2]
    search_distance = max(45.0, min(width_px, height_px) * 0.035)
    scale_candidates: list[dict[str, Any]] = []

    for dim in dimension_candidates:
        box = dim["bbox"]
        cx, cy = bbox_center(box)
        best: tuple[float, dict[str, Any]] | None = None
        for segment in segments:
            if segment["length_px"] > max(width_px, height_px) * 0.92:
                continue
            if segment["orientation"] == "horizontal":
                axis_distance = abs(segment["cy"] - cy)
                axis_inside = segment["x1"] - search_distance * 2 <= cx <= segment["x2"] + search_distance * 2
                off_axis_distance = max(0.0, segment["x1"] - cx, cx - segment["x2"])
            else:
                axis_distance = abs(segment["cx"] - cx)
                axis_inside = segment["y1"] - search_distance * 2 <= cy <= segment["y2"] + search_distance * 2
                off_axis_distance = max(0.0, segment["y1"] - cy, cy - segment["y2"])

            if axis_distance > search_distance or not axis_inside:
                continue
            score = axis_distance + off_axis_distance * 0.25
            if best is None or score < best[0]:
                best = (score, segment)

        if best is None:
            continue
        _, segment = best
        mm_per_px = float(dim["value_mm"]) / max(float(segment["length_px"]), 1.0)
        if mm_per_px < 0.02 or mm_per_px > 500.0:
            continue
        scale_candidates.append(
            {
                "page": page_number,
                "dimension_text": dim["text"],
                "dimension_mm": round(float(dim["value_mm"]), 3),
                "line_length_px": round(float(segment["length_px"]), 3),
                "mm_per_px": mm_per_px,
                "orientation": segment["orientation"],
                "ocr_source": dim["source"],
            }
        )

    if not scale_candidates:
        return {"page": page_number, "mm_per_px": None, "source": "missing_match", "candidates": []}

    from statistics import median as med
    values = [item["mm_per_px"] for item in scale_candidates]
    base = med(values)
    consistent = [value for value in values if abs(value - base) / max(base, 0.0001) <= 0.35]
    chosen = med(consistent or values)
    source = "dimension_line"
    confidence = min(1.0, 0.45 + 0.1 * len(consistent or values))

    return {
        "page": page_number,
        "mm_per_px": chosen,
        "source": source,
        "confidence": round(confidence, 3),
        "candidates": [
            {
                **item,
                "mm_per_px": round(float(item["mm_per_px"]), 6),
            }
            for item in scale_candidates[:20]
        ],
    }


def add_metric_fields(zone: dict[str, Any], mm_per_px: float | None, scale_source: str) -> dict[str, Any]:
    result = dict(zone)
    result["scale_mm_per_px"] = round(mm_per_px, 6) if mm_per_px is not None else None
    result["scale_source"] = scale_source
    if mm_per_px is None:
        result.update(
            {
                "width_mm": None,
                "height_mm": None,
                "area_m2": None,
                "bbox_area_m2": None,
            }
        )
        return result

    width_mm = result["width_px"] * mm_per_px
    height_mm = result["height_px"] * mm_per_px
    area_m2 = result["area_px"] * mm_per_px * mm_per_px / 1_000_000.0
    bbox_area_m2 = result["bbox_area_px"] * mm_per_px * mm_per_px / 1_000_000.0
    result.update(
        {
            "width_mm": round(width_mm, 1),
            "height_mm": round(height_mm, 1),
            "area_m2": round(area_m2, 4),
            "bbox_area_m2": round(bbox_area_m2, 4),
        }
    )
    return result


def summarize_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for panel in panels:
        grouped[panel["material_name"]].append(panel)

    summary: list[dict[str, Any]] = []
    for material_name, rows in sorted(grouped.items(), key=lambda item: item[0].lower()):
        areas = [row["area_m2"] for row in rows if row.get("area_m2") is not None]
        bbox_areas = [row["bbox_area_m2"] for row in rows if row.get("bbox_area_m2") is not None]
        summary.append(
            {
                "material_name": material_name,
                "panel_count": len(rows),
                "area_m2": round(sum(areas), 4) if areas else None,
                "bbox_area_m2": round(sum(bbox_areas), 4) if bbox_areas else None,
                "pages": sorted({row["page"] for row in rows}),
            }
        )
    return summary