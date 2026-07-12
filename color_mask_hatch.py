#!/usr/bin/env python3
"""Mask an image by colors learned from a hatch sample."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Could not write output image: {path}")


def rgb_to_hex(color: np.ndarray) -> str:
    red, green, blue = [int(value) for value in color]
    return f"#{red:02x}{green:02x}{blue:02x}"


def normalize_angle_180(angle_degrees: float) -> float:
    while angle_degrees < -90.0:
        angle_degrees += 180.0
    while angle_degrees >= 90.0:
        angle_degrees -= 180.0
    return angle_degrees


def hatch_foreground_mask(image_rgb: np.ndarray, min_saturation: int, max_value: int) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    return (saturation >= min_saturation) & (value <= max_value)


def build_palette(
    sample_rgb: np.ndarray,
    max_colors: int,
    min_saturation: int,
    max_value: int,
) -> np.ndarray:
    foreground = sample_rgb[hatch_foreground_mask(sample_rgb, min_saturation, max_value)]
    if foreground.size == 0:
        raise ValueError("No foreground hatch colors found in sample. Lower --min-saturation or raise --max-value.")

    # Quantize lightly so anti-aliased hatch pixels become a compact, stable palette.
    quantized = (foreground.astype(np.uint16) // 8) * 8
    colors, counts = np.unique(quantized.astype(np.uint8), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    return colors[order[:max_colors]].astype(np.uint8)


def dominant_palette_summary(palette_rgb: np.ndarray, limit: int) -> list[dict[str, object]]:
    return [
        {"rgb": [int(channel) for channel in color], "hex": rgb_to_hex(color)}
        for color in palette_rgb[:limit]
    ]


def weighted_orientation_mean(angles_degrees: list[float], weights: list[float]) -> float:
    if not angles_degrees:
        return -45.0
    radians = np.deg2rad(np.asarray(angles_degrees, dtype=np.float64) * 2.0)
    weights_array = np.asarray(weights, dtype=np.float64)
    x = float(np.sum(np.cos(radians) * weights_array))
    y = float(np.sum(np.sin(radians) * weights_array))
    return normalize_angle_180(math.degrees(math.atan2(y, x)) / 2.0)


def estimate_hatch_angle(mask: np.ndarray) -> float:
    mask_u8 = (mask.astype(np.uint8) * 255)
    edges = cv2.Canny(mask_u8, 50, 150)
    min_line_length = max(12, min(mask.shape[:2]) // 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 12, minLineLength=min_line_length, maxLineGap=5)
    if lines is None:
        return -45.0

    angles: list[float] = []
    weights: list[float] = []
    for line in lines[:, 0, :]:
        x0, y0, x1, y1 = [int(value) for value in line]
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < min_line_length:
            continue
        angles.append(normalize_angle_180(math.degrees(math.atan2(dy, dx))))
        weights.append(length)
    return weighted_orientation_mean(angles, weights)


def local_maxima(values: np.ndarray, min_distance: int) -> list[int]:
    if values.size < 3:
        return []
    threshold = float(np.max(values)) * 0.25
    candidates: list[tuple[float, int]] = []
    for index in range(1, values.size - 1):
        if values[index] >= threshold and values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            candidates.append((float(values[index]), index))

    selected: list[int] = []
    for _, index in sorted(candidates, reverse=True):
        if all(abs(index - existing) >= min_distance for existing in selected):
            selected.append(index)
    return sorted(selected)


def estimate_hatch_spacing(mask: np.ndarray, angle_degrees: float) -> float:
    y_coords, x_coords = np.nonzero(mask)
    if len(x_coords) < 2:
        return 10.0

    theta = math.radians(angle_degrees)
    normal_x = -math.sin(theta)
    normal_y = math.cos(theta)
    projection = x_coords * normal_x + y_coords * normal_y
    projection -= float(np.min(projection))

    bins = np.bincount(np.rint(projection).astype(np.int32))
    if bins.size < 3:
        return 10.0
    smooth = cv2.GaussianBlur(bins.astype(np.float32).reshape(1, -1), (1, 9), 0).ravel()
    peaks = local_maxima(smooth, min_distance=4)
    if len(peaks) < 2:
        return 10.0

    gaps = np.diff(peaks)
    gaps = gaps[gaps >= 3]
    if gaps.size == 0:
        return 10.0
    return float(np.median(gaps))


def estimate_line_thickness(mask: np.ndarray) -> float:
    mask_u8 = mask.astype(np.uint8)
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    values = distance[mask]
    if values.size == 0:
        return 1.0
    return float(max(1.0, np.percentile(values, 90) * 2.0))


def build_hatch_definition(sample_rgb: np.ndarray, palette_rgb: np.ndarray, min_saturation: int, max_value: int) -> dict[str, object]:
    sample_mask = hatch_foreground_mask(sample_rgb, min_saturation, max_value)
    angle = estimate_hatch_angle(sample_mask)
    spacing = estimate_hatch_spacing(sample_mask, angle)
    thickness = estimate_line_thickness(sample_mask)
    line_type = "parallel_diagonal" if abs(angle) > 10.0 else "parallel_horizontal"
    if abs(abs(angle) - 90.0) < 10.0:
        line_type = "parallel_vertical"

    primary_line = {
        "type": line_type,
        "angle_degrees_image_xy": round(angle, 2),
        "spacing_px": round(spacing, 2),
        "line_thickness_px": round(thickness, 2),
    }

    return {
        "line_type": line_type,
        "line_types": [primary_line],
        "angle_degrees_image_xy": round(angle, 2),
        "spacing_px": round(spacing, 2),
        "line_thickness_px": round(thickness, 2),
        "foreground_ratio": round(float(np.count_nonzero(sample_mask)) / float(sample_mask.size), 4),
        "colors": dominant_palette_summary(palette_rgb, 8),
    }


def mask_by_palette(
    target_rgb: np.ndarray,
    palette_rgb: np.ndarray,
    distance_threshold: float,
    preserve_source_colors: bool,
    target_min_saturation: int,
    target_max_value: int,
    hue_threshold: int,
    chunk_size: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    palette_lab = cv2.cvtColor(palette_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    target_hue = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)[:, 0].astype(np.int16)
    palette_hue = cv2.cvtColor(palette_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)[:, 0].astype(np.int16)
    flat_target = target_rgb.reshape(-1, 3)
    color_gate = hatch_foreground_mask(target_rgb, target_min_saturation, target_max_value).reshape(-1)

    output = np.full_like(flat_target, 255)
    mask = np.zeros(flat_target.shape[0], dtype=bool)
    threshold_sq = distance_threshold * distance_threshold

    for start in range(0, target_lab.shape[0], chunk_size):
        end = min(start + chunk_size, target_lab.shape[0])
        diff = target_lab[start:end, None, :] - palette_lab[None, :, :]
        distances_sq = np.sum(diff * diff, axis=2)
        nearest = np.argmin(distances_sq, axis=1)
        hue_diff = np.abs(target_hue[start:end] - palette_hue[nearest])
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        matched = (
            (distances_sq[np.arange(end - start), nearest] <= threshold_sq)
            & (hue_diff <= hue_threshold)
            & color_gate[start:end]
        )

        mask[start:end] = matched
        if preserve_source_colors:
            output[start:end][matched] = flat_target[start:end][matched]
        else:
            output[start:end][matched] = palette_rgb[nearest[matched]]

    height, width = target_rgb.shape[:2]
    return output.reshape(height, width, 3), mask.reshape(height, width)


def gabor_hatch_response(
    image_rgb: np.ndarray,
    color_mask: np.ndarray,
    hatch_definition: dict[str, object],
    tile_size: int = 1024,
) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    inverted = 255 - gray

    angle_degrees = float(hatch_definition["angle_degrees_image_xy"])
    spacing = max(4.0, float(hatch_definition["spacing_px"]))
    theta = math.radians(angle_degrees + 90.0)
    kernel_size = int(max(21, round(spacing * 4.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size),
        sigma=max(2.0, spacing * 0.55),
        theta=theta,
        lambd=spacing,
        gamma=0.35,
        psi=0.0,
        ktype=cv2.CV_32F,
    )
    kernel -= float(kernel.mean())
    norm = float(np.linalg.norm(kernel))
    if norm > 0.0:
        kernel /= norm

    if tile_size <= 0 or (gray.shape[0] <= tile_size and gray.shape[1] <= tile_size):
        response = cv2.filter2D(inverted, cv2.CV_32F, kernel)
        np.abs(response, out=response)
        np.multiply(response, color_mask, out=response)
        cv2.GaussianBlur(response, (5, 5), 0, dst=response)
        return cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Disk-backed float response keeps RAM nearly independent of drawing size.
    # The extra two-pixel halo is required by the final 5x5 Gaussian blur.
    image_height, image_width = gray.shape
    halo = kernel_size // 2 + 2
    with tempfile.NamedTemporaryFile(prefix="gabor_response_", suffix=".dat") as backing_file:
        response_map = np.memmap(backing_file.name, dtype=np.float32, mode="w+", shape=(image_height, image_width))
        response_min = math.inf
        response_max = -math.inf
        for top in range(0, image_height, tile_size):
            bottom = min(image_height, top + tile_size)
            source_top = max(0, top - halo)
            source_bottom = min(image_height, bottom + halo)
            for left in range(0, image_width, tile_size):
                right = min(image_width, left + tile_size)
                source_left = max(0, left - halo)
                source_right = min(image_width, right + halo)
                tile = inverted[source_top:source_bottom, source_left:source_right]
                tile_response = cv2.filter2D(tile, cv2.CV_32F, kernel)
                np.abs(tile_response, out=tile_response)
                np.multiply(
                    tile_response,
                    color_mask[source_top:source_bottom, source_left:source_right],
                    out=tile_response,
                )
                cv2.GaussianBlur(tile_response, (5, 5), 0, dst=tile_response)
                crop_top = top - source_top
                crop_left = left - source_left
                core = tile_response[
                    crop_top : crop_top + (bottom - top),
                    crop_left : crop_left + (right - left),
                ]
                response_map[top:bottom, left:right] = core
                response_min = min(response_min, float(np.min(core)))
                response_max = max(response_max, float(np.max(core)))

        normalized = np.zeros((image_height, image_width), dtype=np.uint8)
        if response_max > response_min:
            scale = 255.0 / (response_max - response_min)
            for top in range(0, image_height, tile_size):
                bottom = min(image_height, top + tile_size)
                for left in range(0, image_width, tile_size):
                    right = min(image_width, left + tile_size)
                    values = response_map[top:bottom, left:right]
                    normalized[top:bottom, left:right] = np.clip(
                        (values - response_min) * scale,
                        0,
                        255,
                    ).astype(np.uint8)
        del response_map
        return normalized


def build_gabor_match_mask(
    response: np.ndarray,
    color_mask: np.ndarray,
    threshold: int,
    close_size: int,
    dilate_size: int,
) -> np.ndarray:
    _, response_mask = cv2.threshold(response, threshold, 255, cv2.THRESH_BINARY)
    response_mask = cv2.bitwise_and(response_mask, (color_mask.astype(np.uint8) * 255))

    if close_size > 0:
        close_size = close_size if close_size % 2 == 1 else close_size + 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        response_mask = cv2.morphologyEx(response_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    if dilate_size > 0:
        dilate_size = dilate_size if dilate_size % 2 == 1 else dilate_size + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
        response_mask = cv2.dilate(response_mask, dilate_kernel, iterations=1)

    return response_mask


def find_match_bounds(match_mask: np.ndarray, response: np.ndarray, min_area: int) -> list[dict[str, object]]:
    contours, _ = cv2.findContours(match_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds: list[dict[str, object]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        crop_mask = match_mask[y : y + height, x : x + width] > 0
        crop_response = response[y : y + height, x : x + width]
        mean_response = float(np.mean(crop_response[crop_mask])) if np.any(crop_mask) else 0.0
        bounds.append(
            {
                "x": int(x),
                "y": int(y),
                "x1": int(x + width),
                "y1": int(y + height),
                "width": int(width),
                "height": int(height),
                "area_px": round(area, 2),
                "mean_gabor_response": round(mean_response, 2),
            }
        )

    bounds.sort(key=lambda item: float(item["area_px"]), reverse=True)
    return bounds


def draw_bounds(image_rgb: np.ndarray, bounds: list[dict[str, object]]) -> np.ndarray:
    annotated = image_rgb.copy()
    for index, bound in enumerate(bounds, start=1):
        x = int(bound["x"])
        y = int(bound["y"])
        width = int(bound["width"])
        height = int(bound["height"])
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 180, 0), 3)
        cv2.putText(
            annotated,
            str(index),
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 120, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _cluster_projection_peaks(projection: np.ndarray, threshold: float) -> list[int]:
    indices = np.flatnonzero(projection >= threshold)
    if indices.size == 0:
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [int(group[np.argmax(projection[group])]) for group in groups if group.size]


def find_inner_grid_bounds(
    image_rgb: np.ndarray,
    outer_bound: dict[str, object],
    min_cell_width: int,
    min_cell_height: int,
) -> list[dict[str, int]]:
    """Find panel-like rectangles drawn with the coloured hatch/grid pen."""
    x0, y0 = int(outer_bound["x"]), int(outer_bound["y"])
    width, height = int(outer_bound["width"]), int(outer_bound["height"])
    crop = image_rgb[y0 : y0 + height, x0 : x0 + width]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # Construction drawings normally use a moderately saturated coloured pen
    # for panel borders.  It cleanly separates the grid from black dimensions.
    # Warm construction/hatch pen. Red revision clouds and blue adjacent
    # materials are deliberately excluded because they often cross a bound.
    saturated = (
        (hsv[:, :, 0] >= 3)
        & (hsv[:, :, 0] <= 40)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] <= 245)
    ).astype(np.uint8) * 255
    vertical = cv2.morphologyEx(
        saturated,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, height // 5))),
    )
    horizontal = cv2.morphologyEx(
        saturated,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 12), 1)),
    )
    # A lower threshold preserves interrupted panel borders. Long morphology
    # kernels still suppress the diagonal hatch strokes themselves.
    xs = _cluster_projection_peaks(np.count_nonzero(vertical, axis=0), height * 0.08)
    ys = _cluster_projection_peaks(np.count_nonzero(horizontal, axis=1), width * 0.08)

    def merge_close(values: list[int], distance: int) -> list[int]:
        merged: list[int] = []
        for value in sorted(values):
            if not merged or value - merged[-1] > distance:
                merged.append(value)
            else:
                merged[-1] = (merged[-1] + value) // 2
        return merged

    # The external contour may obscure the coloured line, so the outer edges
    # are always valid grid candidates too.
    xs = merge_close([0, *xs, width - 1], max(3, min_cell_width // 8))
    ys = merge_close([0, *ys, height - 1], max(3, min_cell_height // 8))

    def restore_missing_lines(values: list[int], minimum_gap: int) -> list[int]:
        if len(values) < 4:
            return values
        gaps = np.diff(values).astype(np.float64)
        eligible_gaps = gaps[gaps >= minimum_gap]
        typical_gap = float(np.median(eligible_gaps)) if eligible_gaps.size else 0.0
        recovered = list(values)
        for left, right in zip(values, values[1:]):
            gap = right - left
            # A conservative factor avoids splitting intentionally taller rows.
            if typical_gap > 0 and gap >= typical_gap * 1.75:
                parts = max(2, int(round(gap / typical_gap)))
                recovered.extend(int(round(left + gap * part / parts)) for part in range(1, parts))
        return merge_close(recovered, max(3, minimum_gap // 8))

    # Recover interrupted separators from the dominant pitch. Both directions
    # are supported, but the conservative gap factor preserves unequal rows.
    xs = restore_missing_lines(xs, min_cell_width)
    ys = restore_missing_lines(ys, min_cell_height)
    cells: list[dict[str, int]] = []
    for top, bottom in zip(ys, ys[1:]):
        for left, right in zip(xs, xs[1:]):
            cell_width, cell_height = right - left, bottom - top
            if cell_width < min_cell_width or cell_height < min_cell_height:
                continue
            cells.append(
                {
                    "x": x0 + left,
                    "y": y0 + top,
                    "x1": x0 + right,
                    "y1": y0 + bottom,
                    "width": cell_width,
                    "height": cell_height,
                }
            )
    return cells


def bound_is_inside(inner: dict[str, object], outer: dict[str, object]) -> bool:
    """Return True only when the complete inner rectangle is inside outer."""
    return (
        int(outer["x"]) <= int(inner["x"])
        and int(outer["y"]) <= int(inner["y"])
        and int(inner["x1"]) <= int(outer["x1"])
        and int(inner["y1"]) <= int(outer["y1"])
    )


def validate_cell_hatch(
    cell: dict[str, int],
    hatch_mask: np.ndarray,
    min_ratio: float,
    min_pixels: int,
) -> tuple[bool, int, float]:
    """Confirm that a proposed inner cell actually contains learned hatch."""
    crop = hatch_mask[cell["y"] : cell["y1"], cell["x"] : cell["x1"]]
    if crop.size == 0:
        return False, 0, 0.0
    pixels = int(np.count_nonzero(crop))
    ratio = pixels / float(crop.size)
    return pixels >= min_pixels and ratio >= min_ratio, pixels, ratio


def _tesseract_text(image: np.ndarray, language: str, psm: int, whitelist: str = "") -> str:
    if shutil.which("tesseract") is None:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".png") as temporary:
        cv2.imwrite(temporary.name, image)
        command = ["tesseract", temporary.name, "stdout", "-l", language, "--psm", str(psm)]
        if whitelist:
            command.extend(["-c", f"tessedit_char_whitelist={whitelist}"])
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    return " ".join(result.stdout.split())


def _prepare_ocr_crop(crop_rgb: np.ndarray, threshold: int, scale: int = 4) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    return cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def recognize_element_name(cell_rgb: np.ndarray, language: str) -> str:
    height, width = cell_rgb.shape[:2]
    # Names sit in the upper-left part of a panel; excluding the rest greatly
    # reduces interference from diagonal hatch strokes.
    crop = cell_rgb[max(2, height // 30) : max(3, min(height - 2, height * 45 // 100)), 3 : max(4, width * 60 // 100)]
    candidates: list[str] = []
    for threshold in (120, 160):
        text = _tesseract_text(_prepare_ocr_crop(crop, threshold, 5), language, 7)
        text = re.sub(r"[^0-9A-Za-zА-Яа-яЁё.\-]+", "", text)
        if re.search(r"[A-Za-zА-Яа-яЁё]", text) and re.search(r"\d", text):
            candidates.append(text)
    if not candidates:
        # Interrupted borders, a nearby dimension line, or text shifted away
        # from the corner can make the narrow PSM-7 pass fail. Retry only failed
        # cells with a wider crop and sparse-text segmentation.
        fallback_crop = cell_rgb[
            2 : max(3, min(height - 2, height * 65 // 100)),
            2 : max(4, width * 85 // 100),
        ]
        for threshold in (110, 145):
            text = _tesseract_text(_prepare_ocr_crop(fallback_crop, threshold, 4), language, 11)
            for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё.\-]+", text):
                if re.search(r"[A-Za-zА-Яа-яЁё]", token) and re.search(r"\d", token):
                    candidates.append(token)
    raw = max(candidates, key=len, default="")
    if not raw:
        return ""
    match = re.match(r"(.+?)-?(\d[\d.]*)", raw)
    if not match:
        return raw
    prefix, number = match.groups()
    prefix = prefix.replace("K", "К").replace("k", "К")
    if prefix in {"Кp", "Кр", "КP"}:
        prefix = "Кр"
    elif prefix.startswith(("Кст", "Ксп", "Кem", "Кем", "Ксm")):
        prefix = "Ксп"
    digits = re.sub(r"\D", "", number)
    if len(digits) == 3:
        number = f"{digits[0]}.{digits[1:]}"
    else:
        number = number.strip(".")
    return f"{prefix}-{number}"


def _pick_dimension(candidates: list[str]) -> str:
    if not candidates:
        return ""
    return max(set(candidates), key=lambda value: (candidates.count(value), len(value), value))


def _read_dimension_candidates(crop_rgb: np.ndarray, rotate: bool, psm: int = 11) -> list[str]:
    if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 5:
        return []
    if rotate:
        crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_90_CLOCKWISE)
    candidates: list[str] = []
    for threshold in (110, 140, 170):
        text = _tesseract_text(
            _prepare_ocr_crop(crop_rgb, threshold, 4),
            "eng",
            psm,
            whitelist="0123456789.,",
        )
        for token in re.findall(r"\d{2,5}(?:[.,]\d+)?", text):
            normalized = token.strip()
            # Element labels such as 1.01 and small anchor callouts are not
            # cell dimensions. Dimensions must be whole numbers: candidates
            # containing a decimal point or comma are rejected in full.
            if re.fullmatch(r"\d{2,5}", normalized) and int(normalized) >= 100:
                candidates.append(normalized)
    return candidates


def recognize_vertical_dimension(
    cell_rgb: np.ndarray,
    image_rgb: np.ndarray | None = None,
    cell: dict[str, int] | None = None,
    outer_bound: dict[str, object] | None = None,
    allow_external: bool = True,
) -> str:
    height, width = cell_rgb.shape[:2]
    strip_width = max(24, min(40, width // 8))
    candidates: list[str] = []
    # Vertical size callouts are normally near the middle of a panel.  The
    # slightly off-centre samples accommodate the dimension line beside text.
    centers = (width * 42 // 100, width * 44 // 100, width * 46 // 100)
    left_edges = sorted({max(0, min(width - strip_width, center - strip_width // 2)) for center in centers})
    for left in left_edges:
        strip = cell_rgb[3 : height - 3, left : left + strip_width]
        rotated = cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE)
        for threshold in (110, 120, 130):
            text = _tesseract_text(
                _prepare_ocr_crop(rotated, threshold, 5),
                "eng",
                7,
                whitelist="0123456789.,",
            )
            normalized = text.strip()
            if re.fullmatch(r"\d{2,5}", normalized):
                candidates.append(normalized)
    inside_dimension = _pick_dimension(candidates)
    if inside_dimension:
        return inside_dimension

    # Fallback for a vertical value placed elsewhere inside the cell.
    whole_cell = _pick_dimension(_read_dimension_candidates(cell_rgb, rotate=True))
    if whole_cell:
        return whole_cell

    if not allow_external or image_rgb is None or cell is None or outer_bound is None:
        return ""

    # If no vertical number is inside the cell, inspect dimension zones to the
    # left and right. They may lie outside both the cell and the hatch-bound.
    image_height, image_width = image_rgb.shape[:2]
    search_width = max(60, int(outer_bound["width"]) // 3)
    pad_y = max(2, int(cell["height"]) // 12)
    top = max(0, int(cell["y"]) + pad_y)
    bottom = min(image_height, int(cell["y1"]) - pad_y)
    zones = [
        image_rgb[top:bottom, max(0, int(cell["x"]) - search_width) : int(cell["x"])],
        image_rgb[top:bottom, int(cell["x1"]) : min(image_width, int(cell["x1"]) + search_width)],
        image_rgb[
            max(0, int(outer_bound["y"]) - max(60, int(outer_bound["height"]) // 3)) : int(outer_bound["y"]),
            int(cell["x"]) : int(cell["x1"]),
        ],
        image_rgb[
            int(outer_bound["y1"]) : min(image_height, int(outer_bound["y1"]) + max(60, int(outer_bound["height"]) // 3)),
            int(cell["x"]) : int(cell["x1"]),
        ],
    ]
    external: list[str] = []
    for zone in zones:
        external.extend(_read_dimension_candidates(zone, rotate=True))
    return _pick_dimension(external)


def recognize_horizontal_dimension(
    image_rgb: np.ndarray,
    cell: dict[str, int],
    outer_bound: dict[str, object],
    element: str = "",
) -> str:
    """Read the dimension line below an outer bound, aligned with a cell."""
    image_height, image_width = image_rgb.shape[:2]
    outer_top = int(outer_bound["y"])
    outer_bottom = int(outer_bound["y1"])
    search_height = max(80, int(outer_bound["height"]) * 2 // 3)
    left = max(0, int(cell["x"]) + int(cell["width"]) // 12)
    right = min(image_width, int(cell["x1"]) - int(cell["width"]) // 12)
    if right <= left:
        return ""

    # First inspect narrow bands immediately inside the upper/lower cell edge.
    # This supports dimensions printed within the panel without scanning its
    # central label area.
    band_height = max(20, int(cell["height"]) // 3)
    inside_zones = [
        image_rgb[int(cell["y"]) : min(int(cell["y1"]), int(cell["y"]) + band_height), left:right],
        image_rgb[max(int(cell["y"]), int(cell["y1"]) - band_height) : int(cell["y1"]), left:right],
    ]
    inside: list[str] = []
    for zone in inside_zones:
        inside.extend(_read_dimension_candidates(zone, rotate=False))
    element_number = re.sub(r"\D", "", element.rsplit("-", 1)[-1]) if element else ""
    if element_number:
        inside = [value for value in inside if re.sub(r"\D", "", value) != element_number]
    inside_dimension = _pick_dimension(inside)
    if inside_dimension:
        return inside_dimension

    whole_cell = _read_dimension_candidates(
        image_rgb[int(cell["y"]) : int(cell["y1"]), int(cell["x"]) : int(cell["x1"])],
        rotate=False,
    )
    if element_number:
        whole_cell = [value for value in whole_cell if re.sub(r"\D", "", value) != element_number]
    whole_cell_dimension = _pick_dimension(whole_cell)
    if whole_cell_dimension:
        return whole_cell_dimension

    # Otherwise search the conventional dimension zones below and above the
    # whole hatch-bound, retaining X alignment with the current cell.
    outside_zones = [
        image_rgb[min(image_height, outer_bottom + 2) : min(image_height, outer_bottom + search_height), left:right],
        image_rgb[max(0, outer_top - search_height) : max(0, outer_top - 2), left:right],
        image_rgb[
            int(cell["y"]) : int(cell["y1"]),
            max(0, int(outer_bound["x"]) - max(60, int(outer_bound["width"]) // 3)) : int(outer_bound["x"]),
        ],
        image_rgb[
            int(cell["y"]) : int(cell["y1"]),
            int(outer_bound["x1"]) : min(image_width, int(outer_bound["x1"]) + max(60, int(outer_bound["width"]) // 3)),
        ],
    ]
    outside: list[str] = []
    for zone in outside_zones:
        outside.extend(_read_dimension_candidates(zone, rotate=False))
    if element_number:
        outside = [value for value in outside if re.sub(r"\D", "", value) != element_number]
    return _pick_dimension(outside)


def dimensions_match_cell(horizontal: str, vertical: str, cell: dict[str, int], tolerance: float = 0.20) -> bool:
    """Reject façade-wide dimensions by comparing X/Y drawing scales."""
    if not re.fullmatch(r"\d+", horizontal) or not re.fullmatch(r"\d+", vertical):
        return False
    try:
        horizontal_scale = float(horizontal) / max(1, int(cell["width"]))
        vertical_scale = float(vertical) / max(1, int(cell["height"]))
    except ValueError:
        return False
    difference = abs(horizontal_scale - vertical_scale) / max(horizontal_scale, vertical_scale)
    return difference <= tolerance


def analyze_labeled_inner_bounds(
    image_rgb: np.ndarray,
    hatch_mask: np.ndarray,
    outer_bounds: list[dict[str, object]],
    ocr_language: str,
    min_cell_width: int,
    min_cell_height: int,
    min_cell_hatch_ratio: float,
    min_cell_hatch_pixels: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, object]],
]:
    size_source_bounds: list[dict[str, object]] = []
    all_labeled_bounds: list[dict[str, object]] = []
    validated_inner_bounds: list[dict[str, object]] = []
    element_counts: Counter[str] = Counter()
    unique_elements: dict[str, dict[str, set[str]]] = {}
    element_cells: dict[str, list[tuple[np.ndarray, dict[str, int], dict[str, object]]]] = {}
    for outer_index, outer_bound in enumerate(outer_bounds, start=1):
        detected_cells = find_inner_grid_bounds(image_rgb, outer_bound, min_cell_width, min_cell_height)
        # This is a hard OCR boundary: names from a cell extending even one
        # pixel outside its hatch-bound must never enter the element registry.
        cells: list[dict[str, int]] = []
        for cell in detected_cells:
            if not bound_is_inside(cell, outer_bound):
                continue
            hatch_ok, hatch_pixels, hatch_ratio = validate_cell_hatch(
                cell,
                hatch_mask,
                min_cell_hatch_ratio,
                min_cell_hatch_pixels,
            )
            if not hatch_ok:
                continue
            cell["hatch_match_pixels"] = hatch_pixels
            cell["hatch_match_ratio"] = round(hatch_ratio, 5)
            cell["outer_bound"] = outer_index
            cells.append(cell)
        recognized_names: list[str] = []
        for cell in cells:
            crop = image_rgb[cell["y"] : cell["y1"], cell["x"] : cell["x1"]]
            recognized_names.append(recognize_element_name(crop, ocr_language))

        # Repeated panels in one horizontal band have the same designation.
        # Majority voting repairs a missed or partly obscured OCR result without
        # merging different rows or different outer bounds.
        row_indices: dict[tuple[int, int], list[int]] = {}
        for index, cell in enumerate(cells):
            row_indices.setdefault((cell["y"], cell["height"]), []).append(index)
        for indices in row_indices.values():
            votes = Counter(recognized_names[index] for index in indices if recognized_names[index])
            if votes:
                dominant, vote_count = votes.most_common(1)[0]
                if vote_count >= 2:
                    for index in indices:
                        recognized_names[index] = dominant

        for cell, element in zip(cells, recognized_names):
            validated: dict[str, object] = dict(cell)
            validated["element"] = element or None
            validated_inner_bounds.append(validated)
            if not element:
                continue

            labeled: dict[str, object] = dict(cell)
            labeled.update({"outer_bound": outer_index, "element": element})
            all_labeled_bounds.append(labeled)
            element_counts[element] += 1
            sizes = unique_elements.setdefault(element, {"vertical_dimensions": set(), "horizontal_dimensions": set()})
            crop = image_rgb[cell["y"] : cell["y1"], cell["x"] : cell["x1"]]
            element_cells.setdefault(element, []).append((crop, cell, outer_bound))

            # Dimensions are auxiliary metadata. Try to read them only until a
            # complete size has been obtained for this designation; all later
            # occurrences merely increment the counter above.
            size_is_known = bool(sizes["vertical_dimensions"] and sizes["horizontal_dimensions"])
            if size_is_known:
                continue
            vertical_dimension = recognize_vertical_dimension(
                crop,
                image_rgb,
                cell,
                outer_bound,
                allow_external=False,
            )
            horizontal_dimension = recognize_horizontal_dimension(image_rgb, cell, outer_bound, element)
            if not vertical_dimension or not horizontal_dimension:
                continue
            if not dimensions_match_cell(horizontal_dimension, vertical_dimension, cell):
                continue
            sizes["vertical_dimensions"].add(vertical_dimension)
            sizes["horizontal_dimensions"].add(horizontal_dimension)
            size_source: dict[str, object] = dict(cell)
            size_source.update(
                {
                    "outer_bound": outer_index,
                    "element": element,
                    "vertical_dimension": vertical_dimension,
                    "horizontal_dimension": horizontal_dimension,
                }
            )
            size_source_bounds.append(size_source)

    # External vertical dimensions have lower priority because a façade-wide
    # height beside a hatch-bound can overlap a cell row. Use them only when no
    # instance of the designation supplied an internal vertical size.
    for element, sizes in unique_elements.items():
        if sizes["vertical_dimensions"]:
            continue
        for crop, cell, outer_bound in element_cells.get(element, []):
            vertical_dimension = recognize_vertical_dimension(crop, image_rgb, cell, outer_bound, allow_external=True)
            horizontal_dimension = recognize_horizontal_dimension(image_rgb, cell, outer_bound, element)
            if not vertical_dimension or not horizontal_dimension:
                continue
            if not dimensions_match_cell(horizontal_dimension, vertical_dimension, cell):
                continue
            sizes["vertical_dimensions"].add(vertical_dimension)
            sizes["horizontal_dimensions"].add(horizontal_dimension)
            break
    serialized = {
        name: {
            "count": int(element_counts[name]),
            "vertical_dimensions": sorted(values["vertical_dimensions"]),
            "horizontal_dimensions": sorted(values["horizontal_dimensions"]),
        }
        for name, values in sorted(unique_elements.items())
    }
    # Include designations without a dimension callout as well.
    for name, count in sorted(element_counts.items()):
        serialized.setdefault(
            name,
            {"count": int(count), "vertical_dimensions": [], "horizontal_dimensions": []},
        )
    return size_source_bounds, all_labeled_bounds, validated_inner_bounds, serialized


def draw_labeled_inner_bounds(image_rgb: np.ndarray, labeled_bounds: list[dict[str, object]]) -> np.ndarray:
    annotated = image_rgb.copy()
    for index, bound in enumerate(labeled_bounds, start=1):
        x, y, x1, y1 = (int(bound[key]) for key in ("x", "y", "x1", "y1"))
        cv2.rectangle(annotated, (x, y), (x1, y1), (220, 0, 220), 3)
        # OpenCV's built-in font is ASCII-only; full names are stored in JSON.
        cv2.putText(annotated, str(index), (x + 5, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 0, 220), 2, cv2.LINE_AA)
    return annotated


def add_area_totals(unique_elements: dict[str, dict[str, object]]) -> float:
    """Add element areas in m² and return the total across all elements."""
    total_area_m2 = 0.0
    for values in unique_elements.values():
        horizontal = values.get("horizontal_dimensions", [])
        vertical = values.get("vertical_dimensions", [])
        count = int(values.get("count", 0))
        area_m2: float | None = None
        if len(horizontal) == 1 and len(vertical) == 1 and count > 0:
            try:
                width_text = str(horizontal[0])
                height_text = str(vertical[0])
                if not re.fullmatch(r"\d+", width_text) or not re.fullmatch(r"\d+", height_text):
                    raise ValueError("Cell dimensions must be integers")
                width_mm = int(width_text)
                height_mm = int(height_text)
                area_m2 = width_mm * height_mm * count / (1000.0 * 1000.0)
            except ValueError:
                area_m2 = None
        values["total_area_m2"] = round(area_m2, 3) if area_m2 is not None else None
        if area_m2 is not None:
            total_area_m2 += area_m2
    return round(total_area_m2, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Color-mask an image using the non-white colors from a hatch sample.")
    parser.add_argument("--sample", default="hatch.png", help="Reference hatch image to learn colors from.")
    parser.add_argument("--target", default="test_tiled.jpg", help="Image to process.")
    parser.add_argument("--output", default="test_tiled_color_masked.png", help="Masked output image path.")
    parser.add_argument("--mask-output", default="", help="Optional grayscale mask output path.")
    parser.add_argument("--hatch-definition-output", default="hatch_definition.json", help="Hatch definition JSON output path.")
    parser.add_argument("--bounds-output", default="test_tiled_hatch_bounds.json", help="Matched area bounds JSON output path.")
    parser.add_argument("--bounds-image-output", default="test_tiled_hatch_bounds.png", help="Annotated bounds image output path.")
    parser.add_argument(
        "--elements-output",
        default="test_tiled_elements.json",
        help="JSON with every labeled inner bound, unique element counts, and available sizes.",
    )
    parser.add_argument(
        "--elements-image-output",
        default="test_tiled_elements.png",
        help="Image highlighting every inner bound that contains an element name.",
    )
    parser.add_argument("--gabor-response-output", default="", help="Optional grayscale Gabor response output path.")
    parser.add_argument("--gabor-mask-output", default="", help="Optional grayscale Gabor match mask output path.")
    parser.add_argument("--threshold", type=float, default=18.0, help="LAB color distance threshold for matches.")
    parser.add_argument("--gabor-threshold", type=int, default=42, help="Gabor response threshold for hatch matches.")
    parser.add_argument(
        "--gabor-tile-size",
        type=int,
        default=1024,
        help="Gabor tile side in pixels; use 0 for legacy full-image filtering.",
    )
    parser.add_argument("--gabor-close-size", type=int, default=31, help="Morphological close size for grouping Gabor matches.")
    parser.add_argument("--gabor-dilate-size", type=int, default=11, help="Morphological dilation size for grouping Gabor matches.")
    parser.add_argument("--min-bound-area", type=int, default=1200, help="Minimum contour area for reported hatch bounds.")
    parser.add_argument("--min-cell-width", type=int, default=80, help="Minimum width of an inner panel bound.")
    parser.add_argument("--min-cell-height", type=int, default=55, help="Minimum height of an inner panel bound.")
    parser.add_argument(
        "--min-cell-hatch-ratio",
        type=float,
        default=0.005,
        help="Minimum fraction of learned hatch pixels required inside an inner bound.",
    )
    parser.add_argument(
        "--min-cell-hatch-pixels",
        type=int,
        default=25,
        help="Minimum absolute number of learned hatch pixels required inside an inner bound.",
    )
    parser.add_argument("--ocr-language", default="rus+eng", help="Tesseract languages used for element names.")
    parser.add_argument("--max-colors", type=int, default=48, help="Maximum number of learned palette colors.")
    parser.add_argument("--min-saturation", type=int, default=25, help="Minimum sample saturation treated as hatch color.")
    parser.add_argument("--max-value", type=int, default=250, help="Maximum sample value treated as hatch color.")
    parser.add_argument(
        "--target-min-saturation",
        type=int,
        default=12,
        help="Minimum target saturation considered for palette matching.",
    )
    parser.add_argument(
        "--target-max-value",
        type=int,
        default=252,
        help="Maximum target value considered for palette matching.",
    )
    parser.add_argument("--hue-threshold", type=int, default=5, help="Maximum OpenCV HSV hue distance from sample colors.")
    parser.add_argument(
        "--preserve-source-colors",
        action="store_true",
        help="Keep matched target pixels as-is instead of normalizing them to the sample palette.",
    )
    args = parser.parse_args()

    sample_path = Path(args.sample)
    target_path = Path(args.target)
    output_path = Path(args.output)

    sample_rgb = read_rgb(sample_path)
    target_rgb = read_rgb(target_path)
    palette_rgb = build_palette(sample_rgb, args.max_colors, args.min_saturation, args.max_value)
    hatch_definition = build_hatch_definition(sample_rgb, palette_rgb, args.min_saturation, args.max_value)
    masked_rgb, mask = mask_by_palette(
        target_rgb,
        palette_rgb,
        args.threshold,
        args.preserve_source_colors,
        args.target_min_saturation,
        args.target_max_value,
        args.hue_threshold,
    )

    write_rgb(output_path, masked_rgb)
    if args.mask_output:
        mask_path = Path(args.mask_output)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))

    gabor_response = gabor_hatch_response(target_rgb, mask, hatch_definition, tile_size=args.gabor_tile_size)
    gabor_mask = build_gabor_match_mask(
        gabor_response,
        mask,
        args.gabor_threshold,
        args.gabor_close_size,
        args.gabor_dilate_size,
    )
    bounds = find_match_bounds(gabor_mask, gabor_response, args.min_bound_area)
    annotated_rgb = draw_bounds(target_rgb, bounds)
    size_source_bounds, all_labeled_bounds, validated_inner_bounds, unique_elements = analyze_labeled_inner_bounds(
        target_rgb,
        mask,
        bounds,
        args.ocr_language,
        args.min_cell_width,
        args.min_cell_height,
        args.min_cell_hatch_ratio,
        args.min_cell_hatch_pixels,
    )
    total_area_m2 = add_area_totals(unique_elements)
    elements_annotated_rgb = draw_labeled_inner_bounds(target_rgb, validated_inner_bounds)

    hatch_definition_path = Path(args.hatch_definition_output)
    hatch_definition_path.parent.mkdir(parents=True, exist_ok=True)
    hatch_definition_path.write_text(json.dumps(hatch_definition, indent=2), encoding="utf-8")

    bounds_result = {
        "sample": str(sample_path),
        "target": str(target_path),
        "hatch_definition": hatch_definition,
        "gabor": {
            "threshold": int(args.gabor_threshold),
            "tile_size": int(args.gabor_tile_size),
            "close_size": int(args.gabor_close_size),
            "dilate_size": int(args.gabor_dilate_size),
            "min_bound_area": int(args.min_bound_area),
        },
        "bounds": bounds,
    }
    bounds_output_path = Path(args.bounds_output)
    bounds_output_path.parent.mkdir(parents=True, exist_ok=True)
    bounds_output_path.write_text(json.dumps(bounds_result, indent=2), encoding="utf-8")

    elements_result = {
        "target": str(target_path),
        "labels_restricted_to_hatch_bounds": True,
        "inner_bounds_require_hatch_match": True,
        "cell_hatch_validation": {
            "min_ratio": args.min_cell_hatch_ratio,
            "min_pixels": args.min_cell_hatch_pixels,
        },
        "containment_verified": all(
            bound_is_inside(bound, bounds[int(bound["outer_bound"]) - 1])
            for bound in all_labeled_bounds
        ),
        "unique_elements": unique_elements,
        "total_area_m2": total_area_m2,
        "validated_inner_bounds": validated_inner_bounds,
        "all_labeled_inner_bounds": all_labeled_bounds,
        "size_source_bounds": size_source_bounds,
    }
    elements_output_path = Path(args.elements_output)
    elements_output_path.parent.mkdir(parents=True, exist_ok=True)
    elements_output_path.write_text(json.dumps(elements_result, ensure_ascii=False, indent=2), encoding="utf-8")

    write_rgb(Path(args.bounds_image_output), annotated_rgb)
    write_rgb(Path(args.elements_image_output), elements_annotated_rgb)
    if args.gabor_response_output:
        response_path = Path(args.gabor_response_output)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(response_path), gabor_response)
    if args.gabor_mask_output:
        gabor_mask_path = Path(args.gabor_mask_output)
        gabor_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(gabor_mask_path), gabor_mask)

    matched_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    print(f"Palette colors: {len(palette_rgb)}")
    print(f"Hatch angle: {hatch_definition['angle_degrees_image_xy']} degrees")
    print(f"Hatch spacing: {hatch_definition['spacing_px']} px")
    print(f"Matched pixels: {matched_pixels} / {total_pixels} ({matched_pixels / total_pixels:.2%})")
    print(f"Bounds found: {len(bounds)}")
    print(f"Hatch-validated inner bounds found: {len(validated_inner_bounds)}")
    print(f"Labeled inner bounds found: {len(all_labeled_bounds)}")
    for element, dimensions in unique_elements.items():
        horizontal = ", ".join(dimensions["horizontal_dimensions"]) or "?"
        vertical = ", ".join(dimensions["vertical_dimensions"]) or "?"
        area = dimensions.get("total_area_m2")
        area_text = f"{area:.3f} m^2" if isinstance(area, (int, float)) else "?"
        print(f"  {element}: count={dimensions['count']}, size={horizontal} x {vertical}, area={area_text}")
    print(f"Total area: {total_area_m2:.3f} m^2")
    print(f"Wrote: {output_path}")
    print(f"Wrote hatch definition: {hatch_definition_path}")
    print(f"Wrote bounds: {bounds_output_path}")
    print(f"Wrote bounds image: {args.bounds_image_output}")
    print(f"Wrote elements: {elements_output_path}")
    print(f"Wrote elements image: {args.elements_image_output}")
    if args.mask_output:
        print(f"Wrote mask: {args.mask_output}")
    if args.gabor_response_output:
        print(f"Wrote Gabor response: {args.gabor_response_output}")
    if args.gabor_mask_output:
        print(f"Wrote Gabor mask: {args.gabor_mask_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
