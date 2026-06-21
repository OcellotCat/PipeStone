#!/usr/bin/env python3
"""Main PipeStone logic: find and outline legend hatch samples."""

from __future__ import annotations

import datetime as dt
import importlib.util
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from pipestone_ocr import OcrWord, collect_ocr_words, render_pdf_pages, run_image_ocr, words_to_lines
from pipestone_semantic import STONE_KEYWORD_RE

logger = logging.getLogger("pipestone")

APP_NAME = "PipeStone legend hatch finder"
DEFAULT_DPI = 400
DEFAULT_OUTPUT_DIR = "output"


@dataclass(frozen=True)
class MaterialLine:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None
    source: str


@dataclass(frozen=True)
class LegendPatternMatch:
    page: int
    line_text: str
    table_bbox: tuple[float, float, float, float]
    row_bbox: tuple[float, float, float, float]
    pattern_bbox: tuple[float, float, float, float]
    score: float
    annotated_image: str
    pattern_image: str = ""


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def require_module(module_name: str, install_hint: str) -> Any:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing module {module_name}. Install it with: {install_hint}")
    return __import__(module_name)


def load_image_rgb(image_path: str | Path) -> Any:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_text(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е").lower()
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return {token for token in re.split(r"[^0-9a-zа-яе]+", normalize_text(text)) if len(token) > 2}


def extract_material_lines(words_by_page: dict[int, list[OcrWord]]) -> list[MaterialLine]:
    material_lines: list[MaterialLine] = []
    seen: set[tuple[int, str, tuple[int, int, int, int]]] = set()

    for page, words in sorted(words_by_page.items()):
        for line in words_to_lines(words):
            if not STONE_KEYWORD_RE.search(normalize_text(line.text)):
                continue
            key = (
                page,
                normalize_text(line.text),
                tuple(int(round(value / 5.0)) for value in line.bbox),
            )
            if key in seen:
                continue
            seen.add(key)
            material_lines.append(
                MaterialLine(
                    page=page,
                    text=line.text,
                    bbox=line.bbox,
                    confidence=line.confidence,
                    source=line.source,
                )
            )

    logger.info("Material lines found: %s", len(material_lines))
    return material_lines


def preprocess_image(image: Any) -> dict[str, Any]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        9,
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return {"gray": gray, "contrast": contrast, "binary": binary}


def cluster_positions(values: list[int], max_gap: int = 4) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value <= groups[-1][-1] + max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(median(group))) for group in groups]


def table_line_masks(binary: Any) -> tuple[Any, Any, Any]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")

    height, width = binary.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 60), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, height // 60)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    table_mask = cv2.add(horizontal, vertical)
    table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return horizontal, vertical, table_mask


def find_legend_table_bboxes(binary: Any) -> list[tuple[int, int, int, int]]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    _, _, table_mask = table_line_masks(binary)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape[:2]
    page_area = float(height * width)
    candidates: list[tuple[int, int, int, int, float]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if w < width * 0.04 or h < height * 0.035:
            continue
        if area < page_area * 0.004 or area > page_area * 0.45:
            continue
        crop = table_mask[y : y + h, x : x + w]
        line_density = float(np.count_nonzero(crop)) / max(area, 1.0)
        if line_density < 0.01:
            continue
        edge_bias = x / max(width, 1) + y / max(height, 1)
        candidates.append((x, y, x + w, y + h, line_density + edge_bias * 0.05))

    candidates.sort(key=lambda item: item[4], reverse=True)
    return [(x0, y0, x1, y1) for x0, y0, x1, y1, _ in candidates[:5]]


def bbox_overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    return overlap / max(1.0, min(a1 - a0, b1 - b0))


def bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def word_height(words: list[OcrWord]) -> float:
    heights = [max(1.0, word.bbox[3] - word.bbox[1]) for word in words]
    return float(median(heights)) if heights else 20.0


def legend_title_word_kind(text: str) -> str | None:
    normalized = normalize_text(text)
    compact = re.sub(r"[^a-zа-яе]+", "", normalized)
    if not compact:
        return None
    if compact.startswith("услов") or compact.startswith("усл"):
        return "condition"
    if compact.startswith("обознач") or compact.startswith("обозн"):
        return "designation"
    return None


def is_legend_title(text: str) -> bool:
    kinds = {kind for token in normalize_text(text).split() if (kind := legend_title_word_kind(token))}
    if {"condition", "designation"} <= kinds:
        return True
    normalized = normalize_text(text)
    return "услов" in normalized and ("обознач" in normalized or "обозн" in normalized)


def find_legend_title_bboxes(words: list[OcrWord]) -> list[tuple[float, float, float, float]]:
    title_bboxes = [line.bbox for line in words_to_lines(words) if is_legend_title(line.text)]
    if title_bboxes:
        return title_bboxes

    condition_words = [word for word in words if legend_title_word_kind(word.text) == "condition"]
    designation_words = [word for word in words if legend_title_word_kind(word.text) == "designation"]
    if not condition_words or not designation_words:
        return []

    max_gap_y = max(20.0, word_height(words) * 4.0)
    paired_bboxes: list[tuple[float, float, float, float]] = []
    for condition_word in condition_words:
        cx0, cy0, cx1, cy1 = condition_word.bbox
        condition_cy = (cy0 + cy1) / 2.0
        for designation_word in designation_words:
            dx0, dy0, dx1, dy1 = designation_word.bbox
            designation_cy = (dy0 + dy1) / 2.0
            x_overlap = bbox_overlap_1d(cx0, cx1, dx0, dx1)
            center_gap_x = abs(((cx0 + cx1) / 2.0) - ((dx0 + dx1) / 2.0))
            if abs(condition_cy - designation_cy) <= max_gap_y and (x_overlap > 0.0 or center_gap_x <= max(cx1 - cx0, dx1 - dx0) * 1.8):
                paired_bboxes.append(bbox_union([condition_word.bbox, designation_word.bbox]))

    return paired_bboxes


def title_table_score(
    title_bbox: tuple[float, float, float, float],
    table_bbox: tuple[int, int, int, int],
) -> float:
    tx0, ty0, tx1, ty1 = title_bbox
    x0, y0, x1, y1 = table_bbox
    title_cx = (tx0 + tx1) / 2.0
    title_cy = (ty0 + ty1) / 2.0
    table_height = max(1.0, float(y1 - y0))

    if x0 <= title_cx <= x1 and y0 <= title_cy <= y1:
        y_position = (title_cy - y0) / table_height
        return 3.0 - min(y_position, 1.0)

    x_overlap = bbox_overlap_1d(tx0, tx1, float(x0), float(x1))
    vertical_gap = float(y0) - ty1
    if x_overlap >= 0.35 and -table_height * 0.15 <= vertical_gap <= table_height * 0.35:
        return 2.0 + x_overlap - max(0.0, vertical_gap) / table_height

    return 0.0


def table_legend_word_score(
    words: list[OcrWord],
    table_bbox: tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = table_bbox
    table_height = max(1.0, float(y1 - y0))
    search_bbox = (
        float(x0),
        max(0.0, float(y0) - table_height * 0.25),
        float(x1),
        float(y0) + table_height * 0.35,
    )
    kinds = {legend_title_word_kind(word.text) for word in words_in_bbox(words, search_bbox)}
    if {"condition", "designation"} <= kinds:
        return 1.5
    if "condition" in kinds or "designation" in kinds:
        return 0.5
    return 0.0


def find_named_legend_table_bboxes(
    binary: Any,
    words: list[OcrWord],
) -> list[tuple[int, int, int, int]]:
    table_bboxes = find_legend_table_bboxes(binary)
    title_bboxes = find_legend_title_bboxes(words)
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for table_bbox in table_bboxes:
        title_score = max((title_table_score(title_bbox, table_bbox) for title_bbox in title_bboxes), default=0.0)
        score = max(title_score, table_legend_word_score(words, table_bbox))
        if score > 0.0:
            scored.append((score, table_bbox))

    if not title_bboxes:
        logger.info("Exact legend title line was not found; using split-word table title fallback")
    scored.sort(key=lambda item: item[0], reverse=True)
    logger.info("Named legend tables found: %s", len(scored))
    return [table_bbox for _, table_bbox in scored]


def line_positions(mask: Any, bbox: tuple[int, int, int, int], orientation: str) -> list[int]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    x0, y0, x1, y1 = bbox
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    if orientation == "horizontal":
        projection = cv2.reduce(crop, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).ravel()
        threshold = max(float(np.percentile(projection, 85)) * 0.6, (x1 - x0) * 255 * 0.08)
        positions = [y0 + int(idx) for idx in np.flatnonzero(projection >= threshold)]
    else:
        projection = cv2.reduce(crop, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).ravel()
        threshold = max(float(np.percentile(projection, 85)) * 0.6, (y1 - y0) * 255 * 0.08)
        positions = [x0 + int(idx) for idx in np.flatnonzero(projection >= threshold)]
    return cluster_positions(positions)


def words_in_bbox(words: list[OcrWord], bbox: tuple[float, float, float, float]) -> list[OcrWord]:
    x0, y0, x1, y1 = bbox
    selected: list[OcrWord] = []
    for word in words:
        cx = (word.bbox[0] + word.bbox[2]) / 2.0
        cy = (word.bbox[1] + word.bbox[3]) / 2.0
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            selected.append(word)
    return selected


def bbox_center_y(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def row_for_y(horizontal_lines: list[int], y: float) -> tuple[int, int] | None:
    for y0, y1 in zip(horizontal_lines, horizontal_lines[1:]):
        if y1 - y0 >= 12 and float(y0) <= y <= float(y1):
            return int(y0), int(y1)
    return None


def label_match_score(line_text: str, target_text: str) -> float:
    line_tokens = token_set(line_text)
    target_tokens = token_set(target_text)
    if not line_tokens or not target_tokens:
        return 0.0
    if normalize_text(line_text) in normalize_text(target_text):
        return 1.0
    return len(line_tokens & target_tokens) / max(len(target_tokens), 1)


def first_left_cell_columns(
    table_bbox: tuple[int, int, int, int],
    vertical_lines: list[int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = table_bbox
    if len(vertical_lines) >= 2:
        return vertical_lines[0], vertical_lines[1], vertical_lines[1], x1

    split = x0 + int((x1 - x0) * 0.28)
    return x0, split, split, x1


def find_legend_pattern_match(
    image: Any,
    words: list[OcrWord],
    target_line: MaterialLine,
    *,
    page: int,
) -> LegendPatternMatch | None:
    processed = preprocess_image(image)
    horizontal, vertical, _ = table_line_masks(processed["binary"])
    best: LegendPatternMatch | None = None
    table_bboxes = find_named_legend_table_bboxes(processed["binary"], words)
    if not table_bboxes:
        logger.info("No named legend table found; falling back to row search in all detected tables")
        table_bboxes = find_legend_table_bboxes(processed["binary"])

    for table_bbox in table_bboxes:
        x0, y0, x1, y1 = table_bbox
        horizontal_lines = line_positions(horizontal, table_bbox, "horizontal")
        vertical_lines = line_positions(vertical, table_bbox, "vertical")
        if len(horizontal_lines) < 2:
            continue

        sample_x0, sample_x1, text_x0, text_x1 = first_left_cell_columns(table_bbox, vertical_lines)
        table_words = words_in_bbox(words, (float(text_x0), float(y0), float(text_x1), float(y1)))
        text_lines = words_to_lines(table_words)

        row_candidates: list[tuple[float, int, int, str]] = []
        for line in text_lines:
            row = row_for_y(horizontal_lines, bbox_center_y(line.bbox))
            if row is None:
                continue
            row_y0, row_y1 = row
            score = label_match_score(line.text, target_line.text)
            if score > 0.0:
                row_candidates.append((score, row_y0, row_y1, line.text))

        if not row_candidates:
            continue

        score, row_y0, row_y1, line_text = max(row_candidates, key=lambda item: item[0])
        if score < 0.25:
            continue

        margin_x = max(2, int((sample_x1 - sample_x0) * 0.06))
        margin_y = max(2, int((row_y1 - row_y0) * 0.12))
        pattern_bbox = (
            float(sample_x0 + margin_x),
            float(row_y0 + margin_y),
            float(sample_x1 - margin_x),
            float(row_y1 - margin_y),
        )
        if pattern_bbox[2] <= pattern_bbox[0] or pattern_bbox[3] <= pattern_bbox[1]:
            continue

        candidate = LegendPatternMatch(
            page=page,
            line_text=line_text,
            table_bbox=(float(x0), float(y0), float(x1), float(y1)),
            row_bbox=(float(x0), float(row_y0), float(x1), float(row_y1)),
            pattern_bbox=pattern_bbox,
            score=round(float(score), 3),
            annotated_image="",
        )
        if best is None or candidate.score > best.score:
            best = candidate

    return best


def clip_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = [int(round(value)) for value in bbox]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def trim_white_margins(image: Any, *, white_threshold: int = 248, padding: int = 3) -> Any:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    if image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    content = gray < int(white_threshold)
    if not np.any(content):
        return image

    rows, cols = np.where(content)
    height, width = gray.shape[:2]
    x0 = max(0, int(cols.min()) - padding)
    y0 = max(0, int(rows.min()) - padding)
    x1 = min(width, int(cols.max()) + padding + 1)
    y1 = min(height, int(rows.max()) + padding + 1)
    if x1 <= x0 or y1 <= y0:
        return image
    return image[y0:y1, x0:x1]


def save_annotated_pattern_image(
    image: Any,
    match: LegendPatternMatch,
    run_dir: Path,
    *,
    page: int,
) -> LegendPatternMatch:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    height, width = image.shape[:2]
    annotated = np.array(image, copy=True)
    thickness = max(2, min(width, height) // 700)

    row_box = clip_bbox(match.row_bbox, width, height)
    pattern_box = clip_bbox(match.pattern_bbox, width, height)
    if row_box is not None:
        x0, y0, x1, y1 = row_box
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 190, 0), max(1, thickness - 1), cv2.LINE_AA)
    if pattern_box is not None:
        x0, y0, x1, y1 = pattern_box
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 180, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"PATTERN {match.score:.2f}",
            (x0, max(18, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 140, 0),
            max(1, thickness),
            cv2.LINE_AA,
        )

    image_dir = run_dir / "pattern_results"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_path = image_dir / f"page_{page:03d}_legend_pattern.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    logger.info("Saved pattern annotation: %s", output_path)

    pattern_image_path = ""
    if pattern_box is not None:
        x0, y0, x1, y1 = pattern_box
        pattern_crop = trim_white_margins(image[y0:y1, x0:x1], padding=max(2, thickness))
        pattern_path = image_dir / f"page_{page:03d}_legend_pattern_trimmed.png"
        cv2.imwrite(str(pattern_path), cv2.cvtColor(pattern_crop, cv2.COLOR_RGB2BGR))
        pattern_image_path = str(pattern_path)
        logger.info("Saved trimmed legend pattern image: %s", pattern_path)

    return LegendPatternMatch(
        page=match.page,
        line_text=match.line_text,
        table_bbox=match.table_bbox,
        row_bbox=match.row_bbox,
        pattern_bbox=match.pattern_bbox,
        score=match.score,
        annotated_image=str(output_path),
        pattern_image=pattern_image_path,
    )


def match_to_dict(match: LegendPatternMatch) -> dict[str, Any]:
    return {
        "page": match.page,
        "line_text": match.line_text,
        "table_bbox": list(match.table_bbox),
        "row_bbox": list(match.row_bbox),
        "pattern_bbox": list(match.pattern_bbox),
        "score": match.score,
        "annotated_image": match.annotated_image,
        "pattern_image": match.pattern_image,
    }


def material_line_to_dict(line: MaterialLine) -> dict[str, Any]:
    return {
        "page": line.page,
        "text": line.text,
        "bbox": list(line.bbox),
        "confidence": line.confidence,
        "source": line.source,
    }


def make_run_dir(output_dir: str | Path) -> Path:
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def analyze_page_image(
    image: Any,
    words: list[OcrWord],
    run_dir: Path,
    *,
    page: int,
) -> tuple[list[MaterialLine], list[LegendPatternMatch]]:
    material_lines = extract_material_lines({page: words})
    matches: list[LegendPatternMatch] = []

    for material_line in material_lines:
        match = find_legend_pattern_match(image, words, material_line, page=page)
        if match is None:
            logger.info("No legend pattern match for page %s line %r", page, material_line.text)
            continue
        matches.append(save_annotated_pattern_image(image, match, run_dir, page=page))
        break

    return material_lines, matches


def analyze_image_file(
    image_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ocr_backend: str = "tesseract",
    tesseract_psm: int = 11,
) -> dict[str, Any]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    run_dir = make_run_dir(output_dir)
    image = load_image_rgb(image_path)
    words, warning = run_image_ocr(image, 1, ocr_backend, tesseract_psm=tesseract_psm)
    if warning:
        logger.warning("Image OCR warning: %s", warning)

    material_lines, matches = analyze_page_image(image, words, run_dir, page=1)
    return {
        "image_path": str(image_path),
        "run_dir": str(run_dir),
        "ocr_warning": warning,
        "material_lines": [material_line_to_dict(line) for line in material_lines],
        "pattern_matches": [match_to_dict(match) for match in matches],
        "annotated_images": [match.annotated_image for match in matches],
        "pattern_images": [match.pattern_image for match in matches if match.pattern_image],
    }


def analyze_pdf_file(
    pdf_path: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ocr_backend: str = "tesseract",
    force_ocr: bool = False,
    tesseract_psm: int = 11,
    save_rendered_pages: bool = False,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cv2 = require_module("cv2", "pip install opencv-python-headless")
    run_dir = make_run_dir(output_dir)
    rendered_pages = render_pdf_pages(pdf_path, dpi=dpi)
    words_by_page = collect_ocr_words(
        pdf_path,
        rendered_pages,
        backend=ocr_backend,
        force_ocr=force_ocr,
        tesseract_psm=tesseract_psm,
    )

    if save_rendered_pages:
        pages_dir = run_dir / "rendered_pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        for rendered_page in rendered_pages:
            page_path = pages_dir / f"page_{rendered_page['page']:03d}.png"
            cv2.imwrite(str(page_path), cv2.cvtColor(rendered_page["image"], cv2.COLOR_RGB2BGR))

    all_material_lines: list[MaterialLine] = []
    all_matches: list[LegendPatternMatch] = []
    for rendered_page in rendered_pages:
        page = int(rendered_page["page"])
        material_lines, matches = analyze_page_image(
            rendered_page["image"],
            words_by_page.get(page, []),
            run_dir,
            page=page,
        )
        all_material_lines.extend(material_lines)
        all_matches.extend(matches)

    return {
        "pdf_path": str(pdf_path),
        "run_dir": str(run_dir),
        "material_lines": [material_line_to_dict(line) for line in all_material_lines],
        "pattern_matches": [match_to_dict(match) for match in all_matches],
        "annotated_images": [match.annotated_image for match in all_matches],
        "pattern_images": [match.pattern_image for match in all_matches if match.pattern_image],
    }
