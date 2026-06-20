#!/usr/bin/env python3
"""OCR and text extraction utilities for PipeStone."""

from __future__ import annotations

import importlib.util
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

logger = logging.getLogger("pipestone.ocr")

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
    if backend in {"auto", "tesseract"}:
        return run_tesseract_ocr(image, page_number, tesseract_psm=tesseract_psm)
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


def render_pdf_pages(pdf_path: Path, dpi: int = 220) -> list[dict]:
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
