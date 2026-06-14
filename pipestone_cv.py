#!/usr/bin/env python3
"""Computer vision utilities for zone detection and counting."""

from __future__ import annotations

import importlib.util
import logging
import math
from statistics import median
from typing import Any

import numpy as np

logger = logging.getLogger("pipestone.cv")


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def require_module(module_name: str, install_hint: str) -> Any:
    if not has_module(module_name):
        raise RuntimeError(f"Не найден модуль {module_name}. Установите: {install_hint}")
    return __import__(module_name)


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


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


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


def _require_cv2_np() -> tuple[Any, Any]:
    return require_module("cv2", "pip install opencv-python-headless"), require_module("numpy", "pip install numpy")


def _rgb_array(image: Any) -> Any:
    cv2, np = _require_cv2_np()
    array = np.array(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return array[:, :, :3].astype(np.uint8)


def _normalize_angle_deg(angle: float) -> float:
    angle = float(angle) % 180.0
    if angle > 90.0:
        angle -= 180.0
    return angle


def _angle_distance_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def _detect_hatch_angles(line_mask: Any, min_line_length: int | None = None) -> tuple[list[float], list[float]]:
    cv2, np = _require_cv2_np()
    height, width = line_mask.shape[:2]
    min_len = min_line_length or max(12, int(min(height, width) * 0.08))
    raw_lines = cv2.HoughLinesP(
        line_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=max(12, int(min_len * 0.45)),
        minLineLength=max(8, int(min_len * 0.65)),
        maxLineGap=max(3, int(min_len * 0.15)),
    )
    if raw_lines is None:
        return [], []

    angles: list[float] = []
    lengths: list[float] = []
    for raw in raw_lines[:, 0, :]:
        x1, y1, x2, y2 = [float(value) for value in raw]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < max(8, min_len * 0.5):
            continue
        angles.append(_normalize_angle_deg(math.degrees(math.atan2(y2 - y1, x2 - x1))))
        lengths.append(length)
    return angles, lengths


def _estimate_hatch_spacing(line_mask: Any, mask: Any, angle_deg: float | None) -> tuple[float | None, float | None]:
    cv2, np = _require_cv2_np()
    height, width = line_mask.shape[:2]
    if np.count_nonzero(mask) < 10 or np.count_nonzero(line_mask) < 10:
        return None, None

    angle = 0.0 if angle_deg is None else float(angle_deg)
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_lines = cv2.warpAffine(line_mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rotated_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    projection = cv2.reduce(rotated_lines, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).ravel()
    weights = cv2.reduce(rotated_mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).ravel()
    valid = weights > max(1.0, width * 0.15)
    if int(np.count_nonzero(valid)) < 2:
        return None, None

    density = projection / np.maximum(weights, 1.0)
    threshold = max(float(np.percentile(density[valid], 55)), 0.025)
    rows = np.flatnonzero(valid & (density >= threshold))
    if rows.size < 2:
        return None, None

    groups: list[tuple[int, int]] = []
    start = prev = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row <= prev + 2:
            prev = row
            continue
        groups.append((start, prev))
        start = prev = row
    groups.append((start, prev))

    centers = [(start + end) / 2.0 for start, end in groups if end - start >= 1]
    thicknesses = [end - start + 1 for start, end in groups if end - start >= 1]
    if len(centers) < 2 or not thicknesses:
        return None, None

    spacing = float(median(np.diff(centers)))
    thickness = float(median(thicknesses))
    if spacing <= 1.0 or thickness <= 0.0:
        return None, None
    return spacing, thickness


def extract_pattern_descriptor(
    image: Any,
    bbox: tuple[float, float, float, float] | list[float] | None = None,
    mask: Any | None = None,
) -> dict[str, Any]:
    cv2, np = _require_cv2_np()
    rgb = _rgb_array(image)

    if bbox is not None:
        x0, y0, x1, y1 = [int(round(value)) for value in bbox]
        height, width = rgb.shape[:2]
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        crop = rgb[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1] if mask is not None else None
    else:
        crop = rgb
        crop_mask = mask

    if crop.size == 0:
        return {
            "mean_color": [None, None, None],
            "color": [None, None, None],
            "hatch_angle": None,
            "hatch_spacing_px": None,
            "hatch_line_thickness_px": None,
            "texture": {"fill_ratio": None, "edge_density": None, "contrast": None, "color_std": [None, None, None]},
        }

    if bbox is not None and crop.shape[0] > 8 and crop.shape[1] > 8:
        margin = max(2, int(min(crop.shape[:2]) * 0.04))
        crop = crop[margin:-margin, margin:-margin]
        if crop_mask is not None:
            crop_mask = crop_mask[margin:-margin, margin:-margin]

    if crop.size == 0:
        return {
            "mean_color": [None, None, None],
            "color": [None, None, None],
            "hatch_angle": None,
            "hatch_spacing_px": None,
            "hatch_line_thickness_px": None,
            "texture": {"fill_ratio": None, "edge_density": None, "contrast": None, "color_std": [None, None, None]},
        }

    if crop_mask is None:
        crop_mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
    else:
        crop_mask = (np.asarray(crop_mask) > 0).astype(np.uint8) * 255

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mean_color = [round(float(value), 1) for value in cv2.mean(crop, mask=crop_mask)[:3]]
    color_std = [round(float(value), 1) for value in cv2.meanStdDev(crop, mask=crop_mask)[1].ravel()]
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 150)
    line_mask = cv2.bitwise_or(otsu, edges)
    line_mask = cv2.bitwise_and(line_mask, line_mask, mask=crop_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    mask_area = max(float(np.count_nonzero(crop_mask)), 1.0)
    line_pixels = float(np.count_nonzero(line_mask))
    edge_pixels = float(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=crop_mask)))
    line_density = line_pixels / mask_area
    edge_density = edge_pixels / mask_area
    contrast_values = gray[crop_mask > 0]
    contrast = float(np.std(contrast_values)) if contrast_values.size else 0.0

    angles, angle_lengths = _detect_hatch_angles(line_mask)
    angle = None
    if angles:
        bins: dict[float, float] = {}
        for angle_value, length in zip(angles, angle_lengths):
            key = round(angle_value / 5.0) * 5.0
            bins[key] = bins.get(key, 0.0) + length
        best_bin = max(bins.items(), key=lambda item: item[1])[0]
        near = [angle_value for angle_value in angles if abs(angle_value - best_bin) <= 7.5]
        angle = float(median(near or angles))

    spacing, thickness = _estimate_hatch_spacing(line_mask, crop_mask, angle)

    return {
        "mean_color": mean_color,
        "color": mean_color,
        "hatch_angle": round(angle, 2) if angle is not None else None,
        "hatch_spacing_px": round(float(spacing), 2) if spacing is not None else None,
        "hatch_line_thickness_px": round(float(thickness), 2) if thickness is not None else None,
        "texture": {
            "fill_ratio": round(line_density, 4),
            "edge_density": round(edge_density, 4),
            "contrast": round(contrast, 2),
            "color_std": color_std,
        },
    }


def pattern_distance(zone_descriptor: dict[str, Any], pattern_descriptor: dict[str, Any]) -> float:
    if not zone_descriptor or not pattern_descriptor:
        return float("inf")

    def relative_score(a: float | None, b: float | None, tolerance: float) -> float | None:
        if a is None or b is None:
            return None
        return min(abs(float(a) - float(b)) / max(abs(float(b)), tolerance, 1.0e-6), 2.0)

    zone_color = zone_descriptor.get("mean_color") or zone_descriptor.get("color") or []
    pattern_color = pattern_descriptor.get("mean_color") or pattern_descriptor.get("color") or []
    color_score = None
    if len(zone_color) >= 3 and len(pattern_color) >= 3 and all(value is not None for value in zone_color[:3] + pattern_color[:3]):
        diff = math.sqrt(sum((float(zone_color[i]) - float(pattern_color[i])) ** 2 for i in range(3)))
        color_score = min(diff / 255.0, 2.0)

    angle_score = None
    zone_angle = zone_descriptor.get("hatch_angle")
    pattern_angle = pattern_descriptor.get("hatch_angle")
    angle_diff = _angle_distance_deg(zone_angle, pattern_angle)
    if angle_diff is not None:
        angle_score = min(angle_diff / 45.0, 2.0)

    spacing_score = relative_score(zone_descriptor.get("hatch_spacing_px"), pattern_descriptor.get("hatch_spacing_px"), 4.0)
    thickness_score = relative_score(zone_descriptor.get("hatch_line_thickness_px"), pattern_descriptor.get("hatch_line_thickness_px"), 1.0)

    zone_texture = zone_descriptor.get("texture") or {}
    pattern_texture = pattern_descriptor.get("texture") or {}
    texture_scores = [
        relative_score(zone_texture.get("fill_ratio"), pattern_texture.get("fill_ratio"), 0.03),
        relative_score(zone_texture.get("edge_density"), pattern_texture.get("edge_density"), 0.01),
        relative_score(zone_texture.get("contrast"), pattern_texture.get("contrast"), 8.0),
    ]
    texture_score_values = [score for score in texture_scores if score is not None]
    texture_score = sum(texture_score_values) / len(texture_score_values) if texture_score_values else None

    weighted_scores = [
        (0.25, color_score),
        (0.25, angle_score),
        (0.20, spacing_score),
        (0.10, thickness_score),
        (0.20, texture_score),
    ]
    selected = [(weight, score) for weight, score in weighted_scores if score is not None]
    if not selected:
        return float("inf")

    total_weight = sum(weight for weight, _ in selected)
    return sum(weight * float(score) for weight, score in selected) / total_weight


def match_zones_to_patterns(
    zones: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    image: Any,
    max_score: float = 1.35,
) -> list[dict[str, Any]]:
    if not patterns:
        return []

    source_zone_ids = {str(pattern.get("zone_id")) for pattern in patterns if pattern.get("zone_id") is not None}
    matched: list[dict[str, Any]] = []

    for zone in zones:
        if str(zone.get("zone_id")) in source_zone_ids:
            continue
        descriptor = extract_pattern_descriptor(image, bbox=zone["bbox_px"])
        best_pattern = None
        best_score = float("inf")
        for pattern in patterns:
            score = pattern_distance(descriptor, pattern.get("pattern") or {})
            if score < best_score:
                best_score = score
                best_pattern = pattern
        if best_pattern is None or best_score > max_score:
            continue
        matched.append(
            {
                **zone,
                "pattern_id": best_pattern.get("id"),
                "pattern_name": best_pattern.get("name"),
                "pattern_score": round(float(best_score), 4),
                "pattern_descriptor": descriptor,
                "matched_pattern": best_pattern,
            }
        )
    return matched


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
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_intensity = float(cv2.mean(gray, mask=mask)[0])

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
                "mean_intensity": round(mean_intensity, 1),
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


def filter_zones_by_hatch_pattern(
    zones: list[dict[str, Any]],
    pattern_zones: list[dict[str, Any]] | None = None,
    intensity_tolerance: float = 30.0,
) -> list[dict[str, Any]]:
    if pattern_zones is None:
        return zones

    pattern_intensities = [z.get("mean_intensity", 128) for z in pattern_zones if "mean_intensity" in z]
    if not pattern_intensities:
        return zones

    pattern_min = min(pattern_intensities)
    pattern_max = max(pattern_intensities)

    filtered = [
        zone
        for zone in zones
        if pattern_min <= zone.get("mean_intensity", 128) <= pattern_max
    ]
    return filtered if filtered else zones


def estimate_zone_scale(
    zone: dict[str, Any],
    words: list[Any],
    segments: list[dict[str, Any]],
    page_width_px: int,
    page_height_px: int,
    fallback_mm_per_px: float | None = None,
) -> tuple[float | None, list[dict[str, Any]]]:
    zone_bbox = tuple(float(v) for v in zone["bbox_px"])
    zx0, zy0, zx1, zy1 = zone_bbox
    search_dist = max(40.0, min(page_width_px, page_height_px) * 0.025)

    zone_words = [
        word
        for word in words
        if word.bbox[0] >= zx0 - 100 and word.bbox[2] <= zx1 + 100
        and word.bbox[1] >= zy0 - 100 and word.bbox[3] <= zy1 + 100
    ]

    from pipestone_ocr import find_dimension_candidates
    dim_candidates = find_dimension_candidates(zone_words)

    if not dim_candidates:
        return fallback_mm_per_px, []

    matched_dims: list[dict[str, Any]] = []
    for dim in dim_candidates:
        box = dim["bbox"]
        cx, cy = bbox_center(box)
        best_seg = None
        best_score = float("inf")
        for seg in segments:
            if seg["orientation"] == "horizontal":
                axis_dist = abs(seg["cy"] - cy)
                axis_inside = seg["x1"] - search_dist * 2 <= cx <= seg["x2"] + search_dist * 2
                off_axis = max(0.0, seg["x1"] - cx, cx - seg["x2"])
            else:
                axis_dist = abs(seg["cx"] - cx)
                axis_inside = seg["y1"] - search_dist * 2 <= cy <= seg["y2"] + search_dist * 2
                off_axis = max(0.0, seg["y1"] - cy, cy - seg["y2"])

            if axis_dist > search_dist or not axis_inside:
                continue
            score = axis_dist + off_axis * 0.3
            if score < best_score:
                best_score = score
                best_seg = seg

        if best_seg:
            mm_per_px = dim["value_mm"] / max(best_seg["length_px"], 1.0)
            if 0.02 <= mm_per_px <= 500.0:
                matched_dims.append({"text": dim["text"], "value_mm": dim["value_mm"], "orientation": best_seg["orientation"]})
                return round(mm_per_px, 6), matched_dims

    return fallback_mm_per_px, matched_dims