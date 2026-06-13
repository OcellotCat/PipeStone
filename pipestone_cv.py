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