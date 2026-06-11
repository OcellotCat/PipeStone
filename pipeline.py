#!/usr/bin/env python3
"""
PipeStone MVP: FastAPI endpoint for facade layout PDFs.

Run API:
    python pipeline.py serve --host 0.0.0.0 --port 8000

Analyze one PDF from CLI:
    python pipeline.py analyze --pdf input/drawing.pdf --fallback-mm-per-px 2.5

Install runtime dependencies:
    pip install fastapi uvicorn python-multipart pymupdf numpy opencv-python-headless

Install at least one OCR backend for scanned PDFs:
    pip install pytesseract
    # plus system package: tesseract-ocr and rus/eng language packs

Optional OCR backends:
    pip install easyocr paddleocr
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
except Exception:  # FastAPI is optional until the API server is started.
    FastAPI = None
    File = None
    HTTPException = None
    Query = None
    UploadFile = None


APP_NAME = "PipeStone PDF Stone Area MVP"
DEFAULT_DPI = 220
DEFAULT_OUTPUT_DIR = "output"
STONE_KEYWORDS = (
    "камень натуральный",
    "натуральный камень",
    "облицовка натуральным камнем",
    "камень 30 мм",
    "изделия из натурального камня",
)
STONE_KEYWORD_RE = re.compile(
    r"("
    r"кам(?:е|ё)?н[ьяеиюом]*\s+натурал\w*|"
    r"натурал\w*\s+кам(?:е|ё)?н[ьяеиюом]*|"
    r"облицовк\w*\s+натурал\w*\s+кам(?:е|ё)?н[ьяеиюом]*|"
    r"кам(?:е|ё)?н[ьяеиюом]*\s*30\s*мм|"
    r"издел\w*\s+из\s+натурал\w*\s+кам(?:е|ё)?н[ьяеиюом]*"
    r")",
    re.IGNORECASE,
)
KNOWN_STONE_TYPES = (
    "гранит",
    "мрамор",
    "травертин",
    "известняк",
    "лабрадорит",
    "габбро",
    "базальт",
    "кварцит",
    "сланец",
    "оникс",
    "доломит",
    "песчаник",
    "ракушечник",
    "серпентинит",
)
BAD_DIMENSION_CONTEXT_RE = re.compile(
    r"(лист|дата|стадия|масштаб|гост|проверил|разраб|инв\.?|подп\.?|формат|"
    r"экспликац|ведомост|примечан|толщин|кам(?:е|ё)?н|натурал)",
    re.IGNORECASE,
)
DIMENSION_RE = re.compile(
    r"(?<![\d.,])("
    r"\d{1,3}(?:[\s\u00a0]\d{3})+|"
    r"\d{3,6}(?:[.,]\d+)?"
    r")\s*(мм|mm|м|m)?(?![\d.,])",
    re.IGNORECASE,
)

logger = logging.getLogger("pipestone")


@dataclass(frozen=True)
class RenderedPage:
    page: int
    image: Any
    width_px: int
    height_px: int
    width_pt: float
    height_pt: float


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


@dataclass(frozen=True)
class MaterialMention:
    page: int
    material_name: str
    line_text: str
    keyword: str
    bbox: tuple[float, float, float, float]
    confidence: float | None
    source: str


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def require_module(module_name: str, install_hint: str) -> Any:
    if not has_module(module_name):
        raise RuntimeError(f"Не найден модуль {module_name}. Установите: {install_hint}")
    return __import__(module_name)


def dependency_report() -> dict[str, bool]:
    modules = [
        "fastapi",
        "uvicorn",
        "multipart",
        "fitz",
        "pdf2image",
        "numpy",
        "cv2",
        "pytesseract",
        "easyocr",
        "paddleocr",
        "PIL",
    ]
    return {name: has_module(name) for name in modules}


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


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def render_pdf_pages(pdf_path: Path, dpi: int) -> list[RenderedPage]:
    if has_module("fitz"):
        return render_pdf_pages_pymupdf(pdf_path, dpi)
    if has_module("pdf2image"):
        return render_pdf_pages_pdf2image(pdf_path, dpi)
    raise RuntimeError(
        "Нет backend для рендера PDF. Установите pymupdf или pdf2image: "
        "pip install pymupdf"
    )


def render_pdf_pages_pymupdf(pdf_path: Path, dpi: int) -> list[RenderedPage]:
    fitz = require_module("fitz", "pip install pymupdf")
    np = require_module("numpy", "pip install numpy")

    pages: list[RenderedPage] = []
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = image[:, :, :3]
        pages.append(
            RenderedPage(
                page=index,
                image=image.copy(),
                width_px=pix.width,
                height_px=pix.height,
                width_pt=float(page.rect.width),
                height_pt=float(page.rect.height),
            )
        )
    doc.close()
    logger.info("PDF rendered: %s pages at %s DPI", len(pages), dpi)
    return pages


def render_pdf_pages_pdf2image(pdf_path: Path, dpi: int) -> list[RenderedPage]:
    pdf2image = require_module("pdf2image", "pip install pdf2image")
    np = require_module("numpy", "pip install numpy")
    images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)
    pages: list[RenderedPage] = []
    for index, pil_image in enumerate(images, start=1):
        rgb = pil_image.convert("RGB")
        image = np.array(rgb)
        pages.append(
            RenderedPage(
                page=index,
                image=image,
                width_px=rgb.width,
                height_px=rgb.height,
                width_pt=rgb.width / (dpi / 72.0),
                height_pt=rgb.height / (dpi / 72.0),
            )
        )
    logger.info("PDF rendered with pdf2image: %s pages at %s DPI", len(pages), dpi)
    return pages


def extract_pdf_text_words(pdf_path: Path, rendered_pages: list[RenderedPage]) -> dict[int, list[OcrWord]]:
    if not has_module("fitz"):
        return {}

    fitz = require_module("fitz", "pip install pymupdf")
    page_by_number = {page.page: page for page in rendered_pages}
    words_by_page: dict[int, list[OcrWord]] = defaultdict(list)

    doc = fitz.open(str(pdf_path))
    for index, page in enumerate(doc, start=1):
        rendered = page_by_number[index]
        scale_x = rendered.width_px / max(float(page.rect.width), 1.0)
        scale_y = rendered.height_px / max(float(page.rect.height), 1.0)
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


def collect_ocr_words(
    pdf_path: Path,
    rendered_pages: list[RenderedPage],
    backend: str,
    force_ocr: bool,
) -> dict[int, list[OcrWord]]:
    text_words = extract_pdf_text_words(pdf_path, rendered_pages)
    words_by_page: dict[int, list[OcrWord]] = defaultdict(list)
    warnings: list[str] = []

    for page in rendered_pages:
        pdf_words = text_words.get(page.page, [])
        if pdf_words and not force_ocr:
            words_by_page[page.page].extend(pdf_words)
            logger.info("Page %s OCR: using PDF text layer (%s words)", page.page, len(pdf_words))
            continue

        if pdf_words:
            words_by_page[page.page].extend(pdf_words)

        image_words, warning = run_image_ocr(page.image, page.page, backend)
        if warning:
            warnings.append(f"page {page.page}: {warning}")
            logger.warning("Page %s OCR warning: %s", page.page, warning)
        words_by_page[page.page].extend(image_words)
        logger.info(
            "Page %s OCR: %s image words, %s PDF words",
            page.page,
            len(image_words),
            len(pdf_words),
        )

    if warnings:
        logger.warning("OCR warnings: %s", "; ".join(warnings))
    return dict(words_by_page)


def run_image_ocr(image: Any, page_number: int, backend: str) -> tuple[list[OcrWord], str | None]:
    backend = backend.lower()
    if backend == "none":
        return [], "image OCR disabled"

    if backend == "auto":
        for candidate in ("tesseract", "paddleocr", "easyocr"):
            words, warning = run_image_ocr(image, page_number, candidate)
            if words:
                return words, None
            if warning:
                logger.debug("OCR backend %s skipped: %s", candidate, warning)
        return [], "no OCR backend produced text; install pytesseract/easyocr/paddleocr"

    if backend == "tesseract":
        return run_tesseract_ocr(image, page_number)
    if backend == "easyocr":
        return run_easyocr(image, page_number)
    if backend == "paddleocr":
        return run_paddleocr(image, page_number)
    return [], f"unknown OCR backend: {backend}"


def run_tesseract_ocr(image: Any, page_number: int) -> tuple[list[OcrWord], str | None]:
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
                config="--oem 3 --psm 6",
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            data = pytesseract.image_to_data(
                pil_image,
                lang="eng",
                config="--oem 3 --psm 6",
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
        words.append(
            OcrWord(
                page=page_number,
                text=text,
                bbox=(x, y, x + w, y + h),
                confidence=confidence,
                source="tesseract",
            )
        )
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
        words.append(
            OcrWord(
                page=page_number,
                text=text,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                confidence=float(confidence),
                source="easyocr",
            )
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
            words.append(
                OcrWord(
                    page=page_number,
                    text=text,
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=confidence,
                    source="paddleocr",
                )
            )
    return words, None


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


def guess_material_name(line_text: str) -> str:
    normalized = normalize_text(line_text)
    cleaned = line_text.strip(" :-\t")

    for stone_type in KNOWN_STONE_TYPES:
        match = re.search(rf"\b{stone_type}\b.*", normalized, re.IGNORECASE)
        if match:
            start = match.start()
            return cleanup_material_label(cleaned[start:])

    after_colon = re.split(r"[:;-]", cleaned, maxsplit=1)
    if len(after_colon) == 2:
        candidate = cleanup_material_label(after_colon[1])
        if candidate and normalize_text(candidate) not in {"30 мм", "мм"}:
            return candidate

    candidate = cleanup_material_label(cleaned)
    if len(candidate) >= 8 and len(candidate.split()) <= 12:
        return candidate
    return "Натуральный камень"


def cleanup_material_label(label: str) -> str:
    label = re.sub(
        r"\b(облицовка|изделия|из|камень|камня|камнем|натуральный|натурального|"
        r"натуральным|толщина|толщ\.?)\b",
        " ",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:мм|mm|м|m)\b", " ", label, flags=re.IGNORECASE)
    label = re.sub(r"[^0-9A-Za-zА-Яа-яЁё.\- ]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip(" .:-")
    if not label:
        return "Натуральный камень"
    return label[:120]


def extract_material_mentions(words_by_page: dict[int, list[OcrWord]]) -> list[MaterialMention]:
    mentions: list[MaterialMention] = []
    seen: set[tuple[int, str, tuple[int, int, int, int]]] = set()

    for page, words in words_by_page.items():
        for line in words_to_lines(words):
            normalized = normalize_text(line.text)
            match = STONE_KEYWORD_RE.search(normalized)
            if not match:
                continue
            material_name = guess_material_name(line.text)
            key = (
                page,
                normalize_text(material_name),
                tuple(int(round(value / 10.0)) for value in line.bbox),
            )
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                MaterialMention(
                    page=page,
                    material_name=material_name,
                    line_text=line.text,
                    keyword=match.group(0),
                    bbox=line.bbox,
                    confidence=line.confidence,
                    source=line.source,
                )
            )
    return mentions


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


def detect_line_segments(image: Any) -> list[dict[str, Any]]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_line_length = max(40, int(min(image.shape[:2]) * 0.025))
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=min_line_length,
        maxLineGap=12,
    )
    segments: list[dict[str, Any]] = []
    if raw_lines is None:
        return segments

    for raw in raw_lines[:, 0, :]:
        x1, y1, x2, y2 = [float(value) for value in raw]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_line_length:
            continue
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        orientation: str | None = None
        if angle <= 8 or angle >= 172:
            orientation = "horizontal"
        elif 82 <= angle <= 98:
            orientation = "vertical"
        if orientation is None:
            continue
        segments.append(
            {
                "x1": min(x1, x2),
                "y1": min(y1, y2),
                "x2": max(x1, x2),
                "y2": max(y1, y2),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "length_px": length,
                "orientation": orientation,
            }
        )
    return segments


def estimate_scale_for_page(image: Any, words: list[OcrWord], page_number: int) -> dict[str, Any]:
    dimension_candidates = find_dimension_candidates(words)
    if not dimension_candidates:
        return {"page": page_number, "mm_per_px": None, "source": "missing", "candidates": []}

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

    values = [item["mm_per_px"] for item in scale_candidates]
    base = median(values)
    consistent = [value for value in values if abs(value - base) / max(base, 0.0001) <= 0.35]
    chosen = median(consistent or values)
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


def detect_material_zones(
    image: Any,
    page_number: int,
    min_zone_area_px: int | None,
    ignore_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[dict[str, Any]]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    height_px, width_px = image.shape[:2]
    page_area = float(width_px * height_px)
    if min_zone_area_px is None:
        min_zone_area_px = int(max(900, page_area * 0.00018))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if ignore_bboxes:
        gray = gray.copy()
        mask_pad = max(4, int(min(width_px, height_px) * 0.003))
        for box in ignore_bboxes:
            x0 = max(0, int(box[0]) - mask_pad)
            y0 = max(0, int(box[1]) - mask_pad)
            x1 = min(width_px - 1, int(box[2]) + mask_pad)
            y1 = min(height_px - 1, int(box[3]) + mask_pad)
            cv2.rectangle(gray, (x0, y0), (x1, y1), 255, thickness=-1)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        12,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    zones: list[dict[str, Any]] = []
    for contour in contours:
        area_px = float(abs(cv2.contourArea(contour)))
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(w * h)
        if w < 24 or h < 24:
            continue
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        if aspect_ratio > 20 and min(w, h) < 80:
            continue
        if bbox_area < min_zone_area_px:
            continue
        if bbox_area > page_area * 0.55:
            continue
        fill_ratio = area_px / max(bbox_area, 1.0)
        if fill_ratio < 0.025:
            continue

        epsilon = 0.012 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 4:
            continue

        zones.append(
            {
                "page": page_number,
                "bbox_px": [int(x), int(y), int(x + w), int(y + h)],
                "width_px": int(w),
                "height_px": int(h),
                "area_px": round(area_px, 3),
                "bbox_area_px": round(bbox_area, 3),
                "fill_ratio": round(fill_ratio, 4),
                "vertices": int(len(approx)),
                "source": "opencv_contour",
            }
        )

    zones.sort(key=lambda item: item["bbox_area_px"], reverse=True)
    deduped: list[dict[str, Any]] = []
    for zone in zones:
        box = tuple(float(value) for value in zone["bbox_px"])
        if any(bbox_iou(box, tuple(float(value) for value in existing["bbox_px"])) > 0.82 for existing in deduped):
            continue
        deduped.append(zone)

    deduped.sort(key=lambda item: (item["bbox_px"][1], item["bbox_px"][0]))
    for index, zone in enumerate(deduped, start=1):
        zone["zone_id"] = f"p{page_number:03d}-z{index:04d}"
    return deduped


def assign_material(
    zone: dict[str, Any],
    page_mentions: list[MaterialMention],
    all_mentions: list[MaterialMention],
) -> MaterialMention | None:
    if page_mentions:
        if len(page_mentions) == 1:
            return page_mentions[0]
        zx, zy = bbox_center(tuple(float(value) for value in zone["bbox_px"]))
        return min(
            page_mentions,
            key=lambda mention: math.hypot(zx - bbox_center(mention.bbox)[0], zy - bbox_center(mention.bbox)[1]),
        )

    unique = dedupe_material_names(all_mentions)
    if len(unique) == 1:
        return all_mentions[0]
    return None


def dedupe_material_names(mentions: list[MaterialMention]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for mention in mentions:
        key = normalize_text(mention.material_name)
        if key in seen:
            continue
        seen.add(key)
        names.append(mention.material_name)
    return names


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


def build_scale_map(
    rendered_pages: list[RenderedPage],
    words_by_page: dict[int, list[OcrWord]],
    fallback_mm_per_px: float | None,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    scale_by_page: dict[int, dict[str, Any]] = {}

    for page in rendered_pages:
        scale = estimate_scale_for_page(page.image, words_by_page.get(page.page, []), page.page)
        scale_by_page[page.page] = scale

    detected_values = [
        scale["mm_per_px"]
        for scale in scale_by_page.values()
        if scale.get("mm_per_px") is not None and scale.get("source") == "dimension_line"
    ]
    global_scale = median(detected_values) if detected_values else None

    for page in rendered_pages:
        scale = scale_by_page[page.page]
        if scale.get("mm_per_px") is not None:
            continue
        if global_scale is not None:
            scale_by_page[page.page] = {
                **scale,
                "mm_per_px": global_scale,
                "source": "global_dimension_line",
                "confidence": 0.45,
            }
        elif fallback_mm_per_px is not None:
            scale_by_page[page.page] = {
                **scale,
                "mm_per_px": float(fallback_mm_per_px),
                "source": "fallback_query_param",
                "confidence": 0.2,
            }
            warnings.append(
                f"page {page.page}: scale not detected, using fallback {fallback_mm_per_px} mm/px"
            )
        else:
            warnings.append(f"page {page.page}: scale not detected; metric sizes are null")

    return scale_by_page, warnings


def analyze_pdf_file(
    pdf_path: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ocr_backend: str = "auto",
    force_ocr: bool = False,
    fallback_mm_per_px: float | None = None,
    min_zone_area_px: int | None = None,
    save_csv: bool = True,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported in this MVP: {pdf_path.name}")

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting PDF analysis: file=%s run=%s", pdf_path, run_id)
    rendered_pages = render_pdf_pages(pdf_path, dpi=dpi)
    words_by_page = collect_ocr_words(pdf_path, rendered_pages, backend=ocr_backend, force_ocr=force_ocr)
    mentions = extract_material_mentions(words_by_page)

    warnings: list[str] = []
    if mentions:
        logger.info("Natural stone material mentions found: %s", len(mentions))
        for mention in mentions:
            logger.info(
                "Material mention: page=%s material=%r line=%r source=%s",
                mention.page,
                mention.material_name,
                mention.line_text,
                mention.source,
            )
    else:
        warning = "No natural-stone material keywords found in OCR/PDF text"
        warnings.append(warning)
        logger.warning(warning)

    scale_by_page, scale_warnings = build_scale_map(rendered_pages, words_by_page, fallback_mm_per_px)
    warnings.extend(scale_warnings)

    mentions_by_page: dict[int, list[MaterialMention]] = defaultdict(list)
    for mention in mentions:
        mentions_by_page[mention.page].append(mention)

    single_global_material = len(dedupe_material_names(mentions)) == 1
    panels: list[dict[str, Any]] = []

    for page in rendered_pages:
        page_mentions = mentions_by_page.get(page.page, [])
        if mentions and not page_mentions and not single_global_material:
            logger.info(
                "Page %s skipped: no material mention on page and multiple materials were detected",
                page.page,
            )
            continue
        if not mentions:
            continue

        scale_info = scale_by_page[page.page]
        zones = detect_material_zones(
            page.image,
            page.page,
            min_zone_area_px=min_zone_area_px,
            ignore_bboxes=[word.bbox for word in words_by_page.get(page.page, [])],
        )
        logger.info("Page %s OpenCV zones: %s", page.page, len(zones))

        for zone in zones:
            mention = assign_material(zone, page_mentions, mentions)
            material_name = mention.material_name if mention else "Натуральный камень (не привязан к легенде)"
            panel = add_metric_fields(
                zone,
                mm_per_px=scale_info.get("mm_per_px"),
                scale_source=str(scale_info.get("source", "missing")),
            )
            panel.update(
                {
                    "material_name": material_name,
                    "material_keyword": mention.keyword if mention else None,
                    "material_line": mention.line_text if mention else None,
                    "material_source": mention.source if mention else None,
                }
            )
            panels.append(panel)

    summary = summarize_panels(panels)
    csv_path = write_csv(run_dir, panels, summary) if save_csv else None
    result = {
        "run_id": run_id,
        "file_name": pdf_path.name,
        "pages": len(rendered_pages),
        "dpi": dpi,
        "ocr_backend": ocr_backend,
        "warnings": warnings,
        "materials_found": [
            {
                "page": mention.page,
                "material_name": mention.material_name,
                "line_text": mention.line_text,
                "keyword": mention.keyword,
                "source": mention.source,
                "confidence": mention.confidence,
            }
            for mention in mentions
        ],
        "scale_by_page": {
            str(page): {
                **{key: value for key, value in scale.items() if key != "mm_per_px"},
                "mm_per_px": round(float(scale["mm_per_px"]), 6)
                if scale.get("mm_per_px") is not None
                else None,
            }
            for page, scale in scale_by_page.items()
        },
        "summary": summary,
        "panels": panels,
        "csv_path": str(csv_path) if csv_path else None,
    }
    log_analysis_result(result)
    return result


def write_csv(run_dir: Path, panels: list[dict[str, Any]], summary: list[dict[str, Any]]) -> Path:
    csv_path = run_dir / "stone_panels.csv"
    fields = [
        "material_name",
        "page",
        "zone_id",
        "width_mm",
        "height_mm",
        "area_m2",
        "bbox_area_m2",
        "scale_mm_per_px",
        "scale_source",
        "area_px",
        "width_px",
        "height_px",
        "bbox_px",
        "material_line",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for panel in panels:
            row = dict(panel)
            row["bbox_px"] = json.dumps(row.get("bbox_px", []), ensure_ascii=False)
            writer.writerow(row)

    summary_path = run_dir / "stone_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["material_name", "panel_count", "area_m2", "bbox_area_m2", "pages"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in summary:
            out = dict(row)
            out["pages"] = ",".join(str(page) for page in out.get("pages", []))
            writer.writerow(out)

    return csv_path


def log_analysis_result(result: dict[str, Any]) -> None:
    logger.info("========== NATURAL STONE RESULT ==========")
    if result["warnings"]:
        for warning in result["warnings"]:
            logger.warning("Warning: %s", warning)

    if not result["summary"]:
        logger.info("No natural stone panels were calculated")
        return

    for item in result["summary"]:
        logger.info(
            "Material: %s | panels=%s | contour_area_m2=%s | bbox_area_m2=%s | pages=%s",
            item["material_name"],
            item["panel_count"],
            item["area_m2"],
            item["bbox_area_m2"],
            item["pages"],
        )
        for panel in [row for row in result["panels"] if row["material_name"] == item["material_name"]]:
            logger.info(
                "  %s page=%s size=%sx%s mm area=%s m2 bbox_area=%s m2 scale=%s (%s)",
                panel["zone_id"],
                panel["page"],
                panel["width_mm"],
                panel["height_mm"],
                panel["area_m2"],
                panel["bbox_area_m2"],
                panel["scale_mm_per_px"],
                panel["scale_source"],
            )


async def save_upload_to_temp(upload: Any, suffix: str = ".pdf") -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="pipestone_upload_"))
    temp_path = temp_dir / f"upload{suffix}"
    try:
        with temp_path.open("wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return temp_path


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI не установлен. Установите: "
            "pip install fastapi uvicorn python-multipart"
        )

    api = FastAPI(title=APP_NAME, version="0.1.0")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "dependencies": dependency_report()}

    @api.post("/analyze-pdf")
    async def analyze_pdf_endpoint(
        file: UploadFile = File(...),
        dpi: int = Query(DEFAULT_DPI, ge=120, le=500),
        ocr_backend: str = Query("auto", pattern="^(auto|none|tesseract|easyocr|paddleocr)$"),
        force_ocr: bool = Query(False),
        fallback_mm_per_px: float | None = Query(None, gt=0),
        min_zone_area_px: int | None = Query(None, ge=100),
        save_csv: bool = Query(True),
    ) -> dict[str, Any]:
        filename = file.filename or "upload.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Upload a PDF file")

        temp_path = await save_upload_to_temp(file, suffix=".pdf")
        try:
            return analyze_pdf_file(
                temp_path,
                dpi=dpi,
                output_dir=DEFAULT_OUTPUT_DIR,
                ocr_backend=ocr_backend,
                force_ocr=force_ocr,
                fallback_mm_per_px=fallback_mm_per_px,
                min_zone_area_px=min_zone_area_px,
                save_csv=save_csv,
            )
        except Exception as exc:
            logger.exception("PDF analysis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            shutil.rmtree(temp_path.parent, ignore_errors=True)

    return api


APP_CREATION_ERROR: Exception | None = None
try:
    app = create_app() if FastAPI is not None else None
except Exception as exc:
    APP_CREATION_ERROR = exc
    app = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--verbose", action="store_true", help="verbose logs")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="start FastAPI server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    analyze = subparsers.add_parser("analyze", help="analyze one local PDF")
    analyze.add_argument("--pdf", required=True)
    analyze.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    analyze.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    analyze.add_argument(
        "--ocr-backend",
        choices=["auto", "none", "tesseract", "easyocr", "paddleocr"],
        default="auto",
    )
    analyze.add_argument("--force-ocr", action="store_true")
    analyze.add_argument("--fallback-mm-per-px", type=float, default=None)
    analyze.add_argument("--min-zone-area-px", type=int, default=None)
    analyze.add_argument("--no-csv", action="store_true")
    analyze.add_argument("--json", action="store_true", help="print full JSON result")

    subparsers.add_parser("doctor", help="show dependency status")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["serve"]
    elif argv[0].startswith("--pdf"):
        argv = ["analyze", *argv]

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    if args.command == "doctor":
        print(json.dumps(dependency_report(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "serve":
        if app is None:
            if APP_CREATION_ERROR:
                raise SystemExit(f"Cannot create FastAPI app: {APP_CREATION_ERROR}")
            raise SystemExit(
                "FastAPI app is unavailable. Install: "
                "pip install fastapi uvicorn python-multipart"
            )
        if not has_module("uvicorn"):
            raise SystemExit("uvicorn is not installed. Install: pip install uvicorn")
        import uvicorn

        uvicorn.run("pipeline:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "analyze":
        result = analyze_pdf_file(
            args.pdf,
            dpi=args.dpi,
            output_dir=args.output_dir,
            ocr_backend=args.ocr_backend,
            force_ocr=args.force_ocr,
            fallback_mm_per_px=args.fallback_mm_per_px,
            min_zone_area_px=args.min_zone_area_px,
            save_csv=not args.no_csv,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
