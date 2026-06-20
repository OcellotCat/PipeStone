#!/usr/bin/env python3
"""Material text search utilities for PipeStone."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from statistics import median
from typing import Any

from pipestone_ocr import OcrWord, words_to_lines, bbox_union
from pipestone_cv import _angle_distance_deg, extract_pattern_descriptor, require_module
from pipestone_semantic import (
    KNOWN_STONE_TYPES,
    STONE_KEYWORD_RE,
    STONE_SEMANTIC_THRESHOLD,
    semantic_best_stone_type,
)

logger = logging.getLogger("pipestone.material_search")

@dataclass(frozen=True)
class MaterialMention:
    page: int
    material_name: str
    line_text: str
    keyword: str
    bbox: tuple[float, float, float, float]
    confidence: float | None
    source: str


@dataclass(frozen=True)
class MaterialLegendSample:
    page: int
    material_name: str
    table_bbox: tuple[float, float, float, float]
    row_bbox: tuple[float, float, float, float]
    sample_bbox: tuple[float, float, float, float]
    descriptor: dict[str, Any]
    texture_type: str
    confidence: float


@dataclass(frozen=True)
class MaterialRegionMatch:
    page: int
    material_name: str
    bbox: tuple[float, float, float, float]
    confidence: float
    descriptor: dict[str, Any]
    reference_sample_bbox: tuple[float, float, float, float]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("ё", "е").replace("Ё", "Е")).strip().lower()


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


def guess_material_name_by_regexp(line_text: str) -> str:
    cleaned = line_text.strip(" :-\t")

    for stone_type in KNOWN_STONE_TYPES:
        match = re.search(rf"\b{stone_type}\b", normalize_text(line_text), re.IGNORECASE)
        if match:
            start = match.start()
            name = cleanup_material_label(cleaned[start:])
            logger.info("Material guess via regex: type=%s name=%r", stone_type, name)
            return name

    logger.info("Material guess default")
    return "Натуральный камень"


def extract_material_mentions(words_by_page: dict[int, list[OcrWord]]) -> list[MaterialMention]:
    mentions: list[MaterialMention] = []
    seen: set[tuple[int, str, tuple[int, int, int, int]]] = set()

    for page, words in words_by_page.items():
        page_lines = words_to_lines(words)
        for line in page_lines:
            normalized = normalize_text(line.text)
            match = STONE_KEYWORD_RE.search(normalized)
            keyword = match.group(0) if match else None
            if not match:
                stone_type, score = semantic_best_stone_type(normalized)
                if stone_type is None or score < STONE_SEMANTIC_THRESHOLD:
                    continue
                match = re.search(rf"\b{stone_type}\b", normalized, re.IGNORECASE)
                if not match:
                    continue
                keyword = match.group(0)

            material_name = guess_material_name_by_regexp(line.text)

            keyword_bbox = line.bbox
            if keyword:
                normalized_line = normalize_text(line.text)
                keyword_match = re.search(re.escape(keyword), normalized_line, re.IGNORECASE)
                if keyword_match:
                    keyword_start_idx = keyword_match.start()
                    normalized_prefix = normalized_line[:keyword_start_idx]
                    n_words_before = len(normalized_prefix.split()) if normalized_prefix.strip() else 0

                    line_words = [w for w in words if w.page == page]
                    line_words_sorted = sorted(line_words, key=lambda w: w.bbox[0])
                    if len(line_words_sorted) > n_words_before:
                        keyword_words = line_words_sorted[n_words_before:]
                        if keyword_words:
                            keyword_bbox = bbox_union([w.bbox for w in keyword_words])

            key = (
                page,
                normalize_text(material_name),
                tuple(int(round(value / 10.0)) for value in keyword_bbox),
            )
            if key in seen:
                continue
            seen.add(key)
            mention = MaterialMention(
                page=page,
                material_name=material_name,
                line_text=line.text,
                keyword=keyword,
                bbox=keyword_bbox,
                confidence=line.confidence,
                source=line.source,
            )
            mentions.append(mention)
            logger.info(
                "Material mention: page=%s material=%r line_text=%r keyword=%r",
                page,
                material_name,
                line.text,
                keyword,
            )

    if mentions:
        logger.info("Material extraction complete: found %s mentions", len(mentions))
    return mentions


def load_drawing_image(image_path: str) -> Any:
    """Load a drawing image from disk as RGB for the CV helpers."""
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_drawing_image(image: Any) -> dict[str, Any]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
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


def _cluster_positions(values: list[int], max_gap: int = 3) -> list[int]:
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


def extract_table_line_masks(binary: Any) -> tuple[Any, Any, Any]:
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

    _, _, table_mask = extract_table_line_masks(binary)
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
        # Legends on construction drawings are often near the bottom or right edge.
        edge_bias = 0.0
        edge_bias += x / max(width, 1)
        edge_bias += y / max(height, 1)
        candidates.append((x, y, x + w, y + h, line_density + edge_bias * 0.05))

    candidates.sort(key=lambda item: item[4], reverse=True)
    return [(x0, y0, x1, y1) for x0, y0, x1, y1, _ in candidates[:3]]


def _line_positions(mask: Any, bbox: tuple[int, int, int, int], orientation: str) -> list[int]:
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
    return _cluster_positions(positions, max_gap=4)


def _words_in_bbox(words: list[OcrWord], bbox: tuple[float, float, float, float]) -> list[OcrWord]:
    x0, y0, x1, y1 = bbox
    selected: list[OcrWord] = []
    for word in words:
        cx = (word.bbox[0] + word.bbox[2]) / 2.0
        cy = (word.bbox[1] + word.bbox[3]) / 2.0
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            selected.append(word)
    return selected


def classify_texture_type(descriptor: dict[str, Any]) -> str:
    angle = descriptor.get("hatch_angle")
    spacing = descriptor.get("hatch_spacing_px")
    texture = descriptor.get("texture") or {}
    fill_ratio = texture.get("fill_ratio")
    edge_density = texture.get("edge_density")

    if angle is not None and spacing is not None:
        if fill_ratio is not None and fill_ratio > 0.18 and edge_density is not None and edge_density > 0.08:
            return "dense_hatch"
        return "single_hatch"
    if edge_density is not None and edge_density > 0.12:
        return "texture"
    if fill_ratio is not None and fill_ratio > 0.2:
        return "solid_or_dense"
    return "unknown"


def _descriptor_has_pattern(descriptor: dict[str, Any]) -> bool:
    texture = descriptor.get("texture") or {}
    fill_ratio = texture.get("fill_ratio") or 0.0
    edge_density = texture.get("edge_density") or 0.0
    contrast = texture.get("contrast") or 0.0
    return bool(
        descriptor.get("hatch_angle") is not None
        or fill_ratio >= 0.015
        or edge_density >= 0.015
        or contrast >= 8.0
    )


def extract_material_legend_samples(
    image: Any,
    words: list[OcrWord],
    *,
    page: int = 1,
) -> list[MaterialLegendSample]:
    processed = preprocess_drawing_image(image)
    horizontal, vertical, _ = extract_table_line_masks(processed["binary"])
    samples_by_table: list[list[MaterialLegendSample]] = []

    for table_bbox in find_legend_table_bboxes(processed["binary"]):
        table_samples: list[MaterialLegendSample] = []
        x0, y0, x1, y1 = table_bbox
        horizontal_lines = _line_positions(horizontal, table_bbox, "horizontal")
        vertical_lines = _line_positions(vertical, table_bbox, "vertical")
        if len(horizontal_lines) < 2:
            continue

        if len(vertical_lines) >= 2:
            sample_x0, sample_x1 = vertical_lines[0], vertical_lines[1]
            text_x0, text_x1 = sample_x1, x1
        else:
            sample_x0, sample_x1 = x0, x0 + int((x1 - x0) * 0.28)
            text_x0, text_x1 = sample_x1, x1

        for row_index, (row_y0, row_y1) in enumerate(zip(horizontal_lines, horizontal_lines[1:]), start=1):
            if row_y1 - row_y0 < 12:
                continue
            row_bbox = (float(x0), float(row_y0), float(x1), float(row_y1))
            margin_x = max(2, int((sample_x1 - sample_x0) * 0.06))
            margin_y = max(2, int((row_y1 - row_y0) * 0.12))
            sample_bbox = (
                float(sample_x0 + margin_x),
                float(row_y0 + margin_y),
                float(sample_x1 - margin_x),
                float(row_y1 - margin_y),
            )
            if sample_bbox[2] <= sample_bbox[0] or sample_bbox[3] <= sample_bbox[1]:
                continue

            text_words = _words_in_bbox(words, (float(text_x0), float(row_y0), float(text_x1), float(row_y1)))
            line_text = " ".join(line.text for line in words_to_lines(text_words)).strip()
            material_name = guess_material_name_by_regexp(line_text) if line_text else f"Материал {row_index}"
            descriptor = extract_pattern_descriptor(image, sample_bbox)
            if not _descriptor_has_pattern(descriptor):
                continue
            texture_type = classify_texture_type(descriptor)
            confidence = 0.85 if line_text else 0.55
            if descriptor.get("hatch_angle") is not None:
                confidence += 0.1
            table_samples.append(
                MaterialLegendSample(
                    page=page,
                    material_name=material_name,
                    table_bbox=(float(x0), float(y0), float(x1), float(y1)),
                    row_bbox=row_bbox,
                    sample_bbox=sample_bbox,
                    descriptor=descriptor,
                    texture_type=texture_type,
                    confidence=min(confidence, 0.95),
                )
            )
        if table_samples:
            samples_by_table.append(table_samples)

    if samples_by_table:
        max_rows = max(len(table_samples) for table_samples in samples_by_table)
        selected_tables = [
            table_samples
            for table_samples in samples_by_table
            if len(table_samples) == max_rows or (max_rows >= 4 and len(table_samples) >= 4)
        ]
        samples = [sample for table_samples in selected_tables for sample in table_samples]
    else:
        samples = []
    logger.info("Legend material samples: page=%s count=%s", page, len(samples))
    return samples


def _descriptor_similarity(candidate: dict[str, Any], reference: dict[str, Any]) -> float:
    score = 0.0
    weight = 0.0

    angle_distance = _angle_distance_deg(candidate.get("hatch_angle"), reference.get("hatch_angle"))
    if angle_distance is not None:
        score += max(0.0, 1.0 - angle_distance / 35.0) * 0.45
        weight += 0.45

    candidate_spacing = candidate.get("hatch_spacing_px")
    reference_spacing = reference.get("hatch_spacing_px")
    if candidate_spacing and reference_spacing:
        rel = abs(float(candidate_spacing) - float(reference_spacing)) / max(float(reference_spacing), 1.0)
        score += max(0.0, 1.0 - rel / 0.75) * 0.25
        weight += 0.25

    candidate_texture = candidate.get("texture") or {}
    reference_texture = reference.get("texture") or {}
    for key, part_weight, scale in (("fill_ratio", 0.15, 0.18), ("edge_density", 0.15, 0.14)):
        cand_value = candidate_texture.get(key)
        ref_value = reference_texture.get(key)
        if cand_value is None or ref_value is None:
            continue
        diff = abs(float(cand_value) - float(ref_value))
        score += max(0.0, 1.0 - diff / scale) * part_weight
        weight += part_weight

    if weight == 0.0:
        return 0.0
    return score / weight


def _contour_bbox(contour: Any) -> tuple[float, float, float, float]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    x, y, w, h = cv2.boundingRect(contour)
    return float(x), float(y), float(x + w), float(y + h)


def find_material_regions(
    image: Any,
    legend_samples: list[MaterialLegendSample],
    *,
    page: int = 1,
    min_confidence: float = 0.45,
) -> list[MaterialRegionMatch]:
    if not legend_samples:
        return []

    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    processed = preprocess_drawing_image(image)
    binary = processed["binary"]
    height, width = binary.shape[:2]
    page_area = float(height * width)
    legend_boxes = list({sample.table_bbox for sample in legend_samples})

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    region_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(region_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    matches: list[MaterialRegionMatch] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < page_area * 0.0005 or area > page_area * 0.6:
            continue
        bbox = _contour_bbox(contour)
        if any(_bbox_overlap_ratio(bbox, legend_box) > 0.25 for legend_box in legend_boxes):
            continue
        x0, y0, x1, y1 = [int(round(value)) for value in bbox]
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue

        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        descriptor = extract_pattern_descriptor(image, bbox, mask=mask)
        best_sample = max(legend_samples, key=lambda sample: _descriptor_similarity(descriptor, sample.descriptor))
        confidence = _descriptor_similarity(descriptor, best_sample.descriptor)
        if confidence < min_confidence:
            continue
        matches.append(
            MaterialRegionMatch(
                page=page,
                material_name=best_sample.material_name,
                bbox=bbox,
                confidence=round(float(confidence), 3),
                descriptor=descriptor,
                reference_sample_bbox=best_sample.sample_bbox,
            )
        )

    matches.sort(key=lambda match: (match.page, match.bbox[1], match.bbox[0]))
    logger.info("Material regions: page=%s count=%s", page, len(matches))
    return matches


def _bbox_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def analyze_image_materials(
    image_path: str,
    *,
    words: list[OcrWord] | None = None,
    page: int = 1,
    image: Any | None = None,
) -> dict[str, Any]:
    image = image if image is not None else load_drawing_image(image_path)
    page_words = words or []
    legend_samples = extract_material_legend_samples(image, page_words, page=page)
    regions = find_material_regions(image, legend_samples, page=page)
    return {
        "image_path": image_path,
        "legend_samples": [_legend_sample_to_dict(sample) for sample in legend_samples],
        "material_regions": [_region_match_to_dict(region) for region in regions],
    }


def _legend_sample_to_dict(sample: MaterialLegendSample) -> dict[str, Any]:
    return {
        "page": sample.page,
        "material_name": sample.material_name,
        "table_bbox": list(sample.table_bbox),
        "row_bbox": list(sample.row_bbox),
        "sample_bbox": list(sample.sample_bbox),
        "descriptor": sample.descriptor,
        "texture_type": sample.texture_type,
        "confidence": sample.confidence,
    }


def _region_match_to_dict(match: MaterialRegionMatch) -> dict[str, Any]:
    return {
        "page": match.page,
        "material_name": match.material_name,
        "bbox": list(match.bbox),
        "confidence": match.confidence,
        "descriptor": match.descriptor,
        "reference_sample_bbox": list(match.reference_sample_bbox),
    }
