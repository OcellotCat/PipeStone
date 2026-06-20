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


def trim_white_margins(
    image: Any,
    mask: Any | None = None,
    *,
    white_threshold: int = 248,
    padding: int = 2,
) -> tuple[Any, Any | None, tuple[int, int, int, int]]:
    """Crop white margins around a hatch sample, preserving a small padding."""
    cv2, np = _require_cv2_np()
    rgb = _rgb_array(image)
    if rgb.size == 0:
        return rgb, mask, (0, 0, 0, 0)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    content = gray < int(white_threshold)

    if mask is not None:
        mask_array = (np.asarray(mask) > 0)
        if mask_array.shape == content.shape:
            content &= mask_array

    if not np.any(content):
        return rgb, mask, (0, 0, rgb.shape[1], rgb.shape[0])

    rows, cols = np.where(content)
    height, width = rgb.shape[:2]
    x0 = max(0, int(cols.min()) - padding)
    y0 = max(0, int(rows.min()) - padding)
    x1 = min(width, int(cols.max()) + padding + 1)
    y1 = min(height, int(rows.max()) + padding + 1)
    if x1 <= x0 or y1 <= y0:
        return rgb, mask, (0, 0, width, height)

    trimmed_mask = None
    if mask is not None:
        trimmed_mask = np.asarray(mask)[y0:y1, x0:x1]
    return rgb[y0:y1, x0:x1], trimmed_mask, (x0, y0, x1, y1)


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
