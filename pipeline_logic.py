#!/usr/bin/env python3
"""Main PipeStone logic: find and outline legend hatch samples."""

from __future__ import annotations

import datetime as dt
import importlib.util
import logging
import math
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
    correlation_mask_image: str = ""
    correlation_regions_image: str = ""
    hatch_matches_image: str = ""
    hatch_matches: tuple[dict[str, Any], ...] = ()


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


def trim_white_margins(image: Any, *, white_threshold: int = 248, padding: int = 0) -> Any:
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


def extract_color_template_64(pattern_rgb: Any) -> Any:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    if pattern_rgb.size == 0:
        return np.full((64, 64, 3), 255, dtype=np.uint8)

    rgb = pattern_rgb if pattern_rgb.ndim == 3 else cv2.cvtColor(pattern_rgb, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    content = gray < 248
    if np.any(content):
        rows, cols = np.where(content)
        rgb = rgb[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]
        gray = gray[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]

    height, width = rgb.shape[:2]
    if height >= 64 and width >= 64:
        score_image = (255.0 - gray.astype(np.float32)) + cv2.absdiff(gray, cv2.blur(gray, (9, 9))).astype(np.float32)
        integral = cv2.integral(score_image)
        best_score = -1.0
        best_xy = (0, 0)
        step = max(1, min(height, width) // 24)
        for y in range(0, height - 63, step):
            for x in range(0, width - 63, step):
                score = float(integral[y + 64, x + 64] - integral[y, x + 64] - integral[y + 64, x] + integral[y, x])
                if score > best_score:
                    best_score = score
                    best_xy = (x, y)
        x, y = best_xy
        return rgb[y : y + 64, x : x + 64]

    scale = min(1.0, 64.0 / max(height, width))
    resized = cv2.resize(rgb, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    canvas = np.full((64, 64, 3), 255, dtype=np.uint8)
    y0 = (64 - resized.shape[0]) // 2
    x0 = (64 - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def color_correlation_mask(image_rgb: Any, template_rgb: Any) -> dict[str, Any]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    if image_rgb.shape[0] < 64 or image_rgb.shape[1] < 64 or float(np.std(template_rgb)) < 1.0:
        empty = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        return {"mask": empty, "regions": empty, "score_map": empty, "threshold": 1.0, "max_score": 0.0}

    result = cv2.matchTemplate(image_rgb, template_rgb, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, _ = cv2.minMaxLoc(result)
    threshold = max(0.18, min(0.5, float(max_score) * 0.55))
    points = (result >= threshold).astype(np.uint8)

    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    contours, _ = cv2.findContours(points, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        mask[y : y + h + 63, x : x + w + 63] = 255

    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33)), iterations=1)
    regions = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (65, 65)), iterations=1)
    score_map = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return {
        "mask": mask,
        "regions": regions,
        "score_map": score_map,
        "threshold": round(float(threshold), 4),
        "max_score": round(float(max_score), 4),
    }


def trim_binary_foreground(mask: Any, *, padding: int = 0) -> Any:
    np = require_module("numpy", "pip install numpy")

    if mask.size == 0:
        return mask
    rows, cols = np.where(mask > 0)
    if rows.size == 0 or cols.size == 0:
        return mask
    height, width = mask.shape[:2]
    x0 = max(0, int(cols.min()) - padding)
    y0 = max(0, int(rows.min()) - padding)
    x1 = min(width, int(cols.max()) + padding + 1)
    y1 = min(height, int(rows.max()) + padding + 1)
    if x1 <= x0 or y1 <= y0:
        return mask
    return mask[y0:y1, x0:x1]


def clean_pattern_recognition_image(image: Any) -> dict[str, Any]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    denoised = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    denoised = cv2.medianBlur(denoised, 3)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    binary = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros(binary.shape, dtype=np.uint8)
    min_area = max(2.0, float(binary.shape[0] * binary.shape[1]) * 0.000002)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)

    visible = cv2.cvtColor(255 - cleaned, cv2.COLOR_GRAY2RGB)
    return {"gray": gray, "binary": cleaned, "visible": visible}


def normalize_hatch_angle(angle: float) -> float:
    angle = float(angle) % 180.0
    if angle > 90.0:
        angle -= 180.0
    return angle


def angle_distance(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def grouped_centers(indices: Any, max_gap: int = 2) -> list[float]:
    values = [int(value) for value in indices]
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value <= groups[-1][-1] + max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(sum(group)) / len(group) for group in groups]


def hatch_descriptor(line_mask: Any, *, expected_angle: float | None = None) -> dict[str, Any]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    if line_mask.size == 0 or int(np.count_nonzero(line_mask)) < 8:
        return {"angle": None, "spacing": None, "density": 0.0, "line_count": 0}

    height, width = line_mask.shape[:2]
    min_len = max(8, int(min(height, width) * 0.35))
    raw_lines = cv2.HoughLinesP(
        line_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=max(8, min_len // 2),
        minLineLength=min_len,
        maxLineGap=max(3, min_len // 3),
    )

    angles: list[float] = []
    lengths: list[float] = []
    if raw_lines is not None:
        for raw in raw_lines[:, 0, :]:
            x0, y0, x1, y1 = [float(value) for value in raw]
            length = math.hypot(x1 - x0, y1 - y0)
            if length < min_len:
                continue
            angles.append(normalize_hatch_angle(math.degrees(math.atan2(y1 - y0, x1 - x0))))
            lengths.append(length)

    selected_angles = angles
    selected_lengths = lengths
    if expected_angle is not None and angles:
        filtered = [
            (value, length)
            for value, length in zip(angles, lengths)
            if (distance := angle_distance(value, expected_angle)) is not None and distance <= 18.0
        ]
        if filtered:
            selected_angles = [value for value, _ in filtered]
            selected_lengths = [length for _, length in filtered]

    angle = None
    if selected_angles:
        bins: dict[float, float] = {}
        for value, length in zip(selected_angles, selected_lengths):
            key = round(value / 5.0) * 5.0
            bins[key] = bins.get(key, 0.0) + length
        best_bin = max(bins.items(), key=lambda item: item[1])[0]
        near = [value for value in selected_angles if (distance := angle_distance(value, best_bin)) is not None and distance <= 7.5]
        angle = float(median(near or selected_angles))

    spacing = None
    if angle is not None:
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(line_mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        projection = cv2.reduce(rotated, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).ravel()
        if projection.size:
            threshold = max(float(np.percentile(projection, 75)), float(width * 255 * 0.04))
            centers = grouped_centers(np.flatnonzero(projection >= threshold), max_gap=2)
            if len(centers) >= 2:
                spacing = float(median(np.diff(centers)))

    density = float(np.count_nonzero(line_mask)) / max(float(height * width), 1.0)
    return {
        "angle": round(angle, 2) if angle is not None else None,
        "spacing": round(spacing, 2) if spacing is not None else None,
        "density": round(density, 4),
        "line_count": len(selected_angles),
    }


def gabor_hatch_response(gray: Any, angle: float | None, spacing: float | None) -> float:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    if gray.size == 0 or angle is None:
        return 0.0
    wavelength = float(spacing) if spacing and spacing > 2.0 else max(6.0, min(gray.shape[:2]) / 6.0)
    theta = math.radians(float(angle) + 90.0)
    kernel = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta, lambd=wavelength, gamma=0.45, psi=0, ktype=cv2.CV_32F)
    response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
    return round(float(np.mean(np.abs(response))) / 255.0, 4)


def bbox_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def template_match_candidates(
    image_binary: Any,
    template_binary: Any,
    *,
    exclude_bbox: tuple[float, float, float, float] | None = None,
    max_candidates: int = 80,
) -> list[dict[str, Any]]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    th, tw = template_binary.shape[:2]
    ih, iw = image_binary.shape[:2]
    if th < 6 or tw < 6 or th >= ih or tw >= iw:
        return []

    result = cv2.matchTemplate(image_binary, template_binary, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, _ = cv2.minMaxLoc(result)
    threshold = max(0.38, min(0.72, float(max_value) * 0.72))
    dilated = cv2.dilate(result, np.ones((max(3, tw // 3), max(3, th // 3)), dtype=np.uint8))
    ys, xs = np.where((result >= threshold) & (result == dilated))

    candidates: list[dict[str, Any]] = []
    for x, y in sorted(zip(xs, ys), key=lambda item: float(result[item[1], item[0]]), reverse=True):
        bbox = (float(x), float(y), float(x + tw), float(y + th))
        if exclude_bbox is not None and bbox_overlap_ratio(bbox, exclude_bbox) > 0.2:
            continue
        if any(bbox_overlap_ratio(bbox, tuple(candidate["bbox"])) > 0.45 for candidate in candidates):
            continue
        candidates.append({"bbox": bbox, "template_score": round(float(result[y, x]), 4)})
        if len(candidates) >= max_candidates:
            break
    return candidates


def gabor_response_map(gray: Any, angle: float | None, spacing: float | None) -> Any:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    if gray.size == 0 or angle is None:
        return np.zeros(gray.shape[:2], dtype=np.uint8)
    wavelength = float(spacing) if spacing and spacing > 2.0 else max(6.0, min(gray.shape[:2]) / 80.0)
    theta = math.radians(float(angle) + 90.0)
    kernel_size = max(15, int(round(wavelength * 4.0)) | 1)
    kernel_size = min(kernel_size, 61)
    kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size),
        sigma=max(3.0, wavelength * 0.6),
        theta=theta,
        lambd=wavelength,
        gamma=0.45,
        psi=0,
        ktype=cv2.CV_32F,
    )
    response = np.abs(cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel))
    return cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def hatch_region_candidates(
    page_binary: Any,
    page_gray: Any,
    reference: dict[str, Any],
    template_shape: tuple[int, int],
    *,
    exclude_bbox: tuple[float, float, float, float] | None = None,
    max_candidates: int = 120,
) -> list[dict[str, Any]]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    ref_angle = reference.get("angle")
    if ref_angle is None:
        return []

    th, tw = template_shape
    response = gabor_response_map(page_gray, ref_angle, reference.get("spacing"))
    threshold = max(35, int(np.percentile(response, 88)))
    response_mask = (response >= threshold).astype(np.uint8) * 255
    line_mask = cv2.bitwise_and(response_mask, page_binary)

    spacing = float(reference.get("spacing") or max(6.0, min(th, tw) / 3.0))
    kernel_size = max(3, min(31, int(round(spacing * 1.5)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    region_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    region_mask = cv2.dilate(region_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(page_binary.shape[0] * page_binary.shape[1])
    template_area = max(1.0, float(th * tw))
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if w < max(8, tw // 2) or h < max(8, th // 2):
            continue
        if area < template_area * 0.35 or area > image_area * 0.35:
            continue
        bbox = (float(x), float(y), float(x + w), float(y + h))
        if exclude_bbox is not None and bbox_overlap_ratio(bbox, exclude_bbox) > 0.2:
            continue
        fill = float(np.count_nonzero(page_binary[y : y + h, x : x + w])) / max(area, 1.0)
        if fill < 0.003:
            continue
        score = float(np.mean(response[y : y + h, x : x + w])) / 255.0
        candidates.append({"bbox": bbox, "template_score": round(score, 4), "source": "gabor_region"})

    candidates.sort(key=lambda item: (item["template_score"], (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])), reverse=True)
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(bbox_overlap_ratio(tuple(candidate["bbox"]), tuple(existing["bbox"])) > 0.6 for existing in deduped):
            continue
        deduped.append(candidate)
        if len(deduped) >= max_candidates:
            break
    return deduped


def merge_hatch_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.get("template_score", 0.0), reverse=True):
        bbox = tuple(candidate["bbox"])
        current = next((item for item in merged if bbox_overlap_ratio(bbox, tuple(item["bbox"])) > 0.55), None)
        if current is None:
            merged.append(candidate)
            continue
        if candidate.get("template_score", 0.0) > current.get("template_score", 0.0):
            current.update(candidate)
    return merged


def recognize_hatch_pattern(
    image: Any,
    pattern_crop: Any,
    match: LegendPatternMatch,
    output_dir: Path,
    *,
    page: int,
) -> dict[str, Any]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    output_dir.mkdir(parents=True, exist_ok=True)
    template_64 = extract_color_template_64(pattern_crop)
    correlation = color_correlation_mask(image, template_64)

    correlation_mask_path = output_dir / f"page_{page:03d}_correlation_mask_64.png"
    cv2.imwrite(str(correlation_mask_path), correlation["mask"])
    correlation_regions_path = output_dir / f"page_{page:03d}_correlation_regions.png"
    cv2.imwrite(str(correlation_regions_path), correlation["regions"])

    annotated = np.full_like(image, 255)
    annotated[correlation["regions"] > 0] = image[correlation["regions"] > 0]

    verified: list[dict[str, Any]] = []
    contours, _ = cv2.findContours(correlation["regions"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < 64 * 64 * 0.25:
            continue
        verified.append(
            {
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "template_score": correlation["max_score"],
                "source": "color_correlation_64",
            }
        )

    matches_path = output_dir / f"page_{page:03d}_hatch_matches.png"
    cv2.imwrite(str(matches_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    logger.info(
        "Hatch color correlation: page=%s regions=%s threshold=%.4f max=%.4f",
        page,
        len(verified),
        correlation["threshold"],
        correlation["max_score"],
    )
    return {
        "correlation_mask_image": str(correlation_mask_path),
        "correlation_regions_image": str(correlation_regions_path),
        "matches_image": str(matches_path),
        "matches": verified,
        "reference": {
            "correlation_threshold": correlation["threshold"],
            "correlation_max_score": correlation["max_score"],
        },
    }


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
    correlation_mask_image_path = ""
    correlation_regions_image_path = ""
    hatch_matches_image_path = ""
    hatch_matches: tuple[dict[str, Any], ...] = ()
    if pattern_box is not None:
        x0, y0, x1, y1 = pattern_box
        pattern_crop = trim_white_margins(image[y0:y1, x0:x1], padding=0)
        pattern_path = image_dir / f"page_{page:03d}_legend_pattern_trimmed.png"
        cv2.imwrite(str(pattern_path), cv2.cvtColor(pattern_crop, cv2.COLOR_RGB2BGR))
        pattern_image_path = str(pattern_path)
        logger.info("Saved trimmed legend pattern image: %s", pattern_path)
        recognition = recognize_hatch_pattern(image, pattern_crop, match, image_dir, page=page)
        correlation_mask_image_path = recognition.get("correlation_mask_image", "")
        correlation_regions_image_path = recognition.get("correlation_regions_image", "")
        hatch_matches_image_path = recognition.get("matches_image", "")
        hatch_matches = tuple(recognition.get("matches", []))

    return LegendPatternMatch(
        page=match.page,
        line_text=match.line_text,
        table_bbox=match.table_bbox,
        row_bbox=match.row_bbox,
        pattern_bbox=match.pattern_bbox,
        score=match.score,
        annotated_image=str(output_path),
        pattern_image=pattern_image_path,
        correlation_mask_image=correlation_mask_image_path,
        correlation_regions_image=correlation_regions_image_path,
        hatch_matches_image=hatch_matches_image_path,
        hatch_matches=hatch_matches,
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
        "correlation_mask_image": match.correlation_mask_image,
        "correlation_regions_image": match.correlation_regions_image,
        "hatch_matches_image": match.hatch_matches_image,
        "hatch_matches": list(match.hatch_matches),
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
        "correlation_mask_images": [match.correlation_mask_image for match in matches if match.correlation_mask_image],
        "correlation_regions_images": [match.correlation_regions_image for match in matches if match.correlation_regions_image],
        "hatch_matches_images": [match.hatch_matches_image for match in matches if match.hatch_matches_image],
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
        "correlation_mask_images": [match.correlation_mask_image for match in all_matches if match.correlation_mask_image],
        "correlation_regions_images": [match.correlation_regions_image for match in all_matches if match.correlation_regions_image],
        "hatch_matches_images": [match.hatch_matches_image for match in all_matches if match.hatch_matches_image],
    }
