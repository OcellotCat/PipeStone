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


def recognize_external_horizontal_dimension(
    image_rgb: np.ndarray,
    cell: dict[str, object],
    outer_bound: dict[str, object],
    element: str = "",
    vertical_dimension: str = "",
) -> str:
    """Read only external horizontal zones and prefer the cell-scale match."""
    image_height, image_width = image_rgb.shape[:2]
    search_height = max(80, int(outer_bound["height"]) * 2 // 3)
    left = max(0, int(cell["x"]) + int(cell["width"]) // 12)
    right = min(image_width, int(cell["x1"]) - int(cell["width"]) // 12)
    zones = [
        image_rgb[int(outer_bound["y1"]) : min(image_height, int(outer_bound["y1"]) + search_height), left:right],
        image_rgb[max(0, int(outer_bound["y"]) - search_height) : int(outer_bound["y"]), left:right],
        image_rgb[
            int(cell["y"]) : int(cell["y1"]),
            max(0, int(outer_bound["x"]) - max(60, int(outer_bound["width"]) // 3)) : int(outer_bound["x"]),
        ],
        image_rgb[
            int(cell["y"]) : int(cell["y1"]),
            int(outer_bound["x1"]) : min(image_width, int(outer_bound["x1"]) + max(60, int(outer_bound["width"]) // 3)),
        ],
    ]
    candidates: list[str] = []
    for zone in zones:
        candidates.extend(_read_dimension_candidates(zone, rotate=False))
    element_number = re.sub(r"\D", "", element.rsplit("-", 1)[-1]) if element else ""
    if element_number:
        candidates = [value for value in candidates if re.sub(r"\D", "", value) != element_number]
    if not candidates:
        return ""
    if vertical_dimension and re.fullmatch(r"\d+", vertical_dimension):
        expected = float(vertical_dimension) * int(cell["width"]) / max(1, int(cell["height"]))
        return min(candidates, key=lambda value: abs(float(value) - expected))
    return _pick_dimension(candidates)


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


def analyze_euclidean_bound_buckets(
    image_rgb: np.ndarray,
    hatch_mask: np.ndarray,
    outer_bounds: list[dict[str, object]],
    ocr_language: str,
    min_cell_width: int,
    min_cell_height: int,
    min_cell_hatch_ratio: float,
    min_cell_hatch_pixels: int,
    bucket_tolerance_px: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, object]], list[dict[str, object]]]:
    """Default pipeline: cluster validated bounds geometrically, then OCR each bucket."""
    print("EUCLIDEAN BUCKETING: collecting hatch-validated inner bounds")
    validated: list[dict[str, object]] = []
    for outer_index, outer_bound in enumerate(outer_bounds, start=1):
        for cell in find_inner_grid_bounds(image_rgb, outer_bound, min_cell_width, min_cell_height):
            if not bound_is_inside(cell, outer_bound):
                continue
            hatch_ok, hatch_pixels, hatch_ratio = validate_cell_hatch(
                cell, hatch_mask, min_cell_hatch_ratio, min_cell_hatch_pixels
            )
            if not hatch_ok:
                continue
            item: dict[str, object] = dict(cell)
            item.update(
                {
                    "outer_bound": outer_index,
                    "hatch_match_pixels": hatch_pixels,
                    "hatch_match_ratio": round(hatch_ratio, 5),
                }
            )
            validated.append(item)

    # Deterministic online clustering in (width_px, height_px). Each bound is
    # assigned only to its nearest centroid when the Euclidean distance passes.
    buckets: list[dict[str, object]] = []
    for bound in sorted(validated, key=lambda item: (int(item["width"]) * int(item["height"]), int(item["width"]), int(item["height"]))):
        nearest: dict[str, object] | None = None
        nearest_distance = math.inf
        for bucket in buckets:
            distance = math.hypot(
                int(bound["width"]) - float(bucket["centroid_width_px"]),
                int(bound["height"]) - float(bucket["centroid_height_px"]),
            )
            existing_bounds = bucket["bounds"]
            assert isinstance(existing_bounds, list)
            max_member_distance = max(
                (
                    math.hypot(
                        int(bound["width"]) - int(existing["width"]),
                        int(bound["height"]) - int(existing["height"]),
                    )
                    for existing in existing_bounds
                ),
                default=0.0,
            )
            # Complete-link validation prevents gradual chaining of different
            # cell sizes through intermediate bounds near the centroid.
            if distance <= bucket_tolerance_px and max_member_distance <= bucket_tolerance_px and distance < nearest_distance:
                nearest_distance = distance
                nearest = bucket
        if nearest is None or nearest_distance > bucket_tolerance_px:
            nearest = {
                "bucket_id": len(buckets) + 1,
                "centroid_width_px": float(bound["width"]),
                "centroid_height_px": float(bound["height"]),
                "bounds": [],
            }
            buckets.append(nearest)
            nearest_distance = 0.0
        bucket_bounds = nearest["bounds"]
        assert isinstance(bucket_bounds, list)
        bucket_bounds.append(bound)
        nearest["centroid_width_px"] = float(np.mean([int(item["width"]) for item in bucket_bounds]))
        nearest["centroid_height_px"] = float(np.mean([int(item["height"]) for item in bucket_bounds]))
        bound["bucket_id"] = int(nearest["bucket_id"])
        bound["bucket_distance_px"] = round(nearest_distance, 3)

    size_sources: list[dict[str, object]] = []
    all_labeled: list[dict[str, object]] = []
    unique: dict[str, dict[str, object]] = {}
    bucket_summary: list[dict[str, object]] = []
    # Larger geometry buckets claim the plain OCR label first. A later,
    # geometrically distinct bucket with the same OCR result stays separate.
    for bucket in sorted(buckets, key=lambda item: (-len(item["bounds"]), int(item["bucket_id"]))):
        bucket_bounds = bucket["bounds"]
        assert isinstance(bucket_bounds, list)
        recognized: list[str] = []
        for bound in bucket_bounds:
            crop = image_rgb[int(bound["y"]) : int(bound["y1"]), int(bound["x"]) : int(bound["x1"])]
            recognized.append(recognize_element_name(crop, ocr_language))
        votes = Counter(name for name in recognized if name)
        ocr_element = votes.most_common(1)[0][0] if votes else f"UNLABELED-BUCKET-{bucket['bucket_id']}"
        element = ocr_element
        if element in unique:
            element = f"{ocr_element} [bucket {bucket['bucket_id']}]"

        horizontal_dimension = ""
        vertical_dimension = ""
        source_bound: dict[str, object] | None = None
        # Search every bound in this geometry bucket until one coherent size is found.
        for bound, raw_name in zip(bucket_bounds, recognized):
            outer_bound = outer_bounds[int(bound["outer_bound"]) - 1]
            crop = image_rgb[int(bound["y"]) : int(bound["y1"]), int(bound["x"]) : int(bound["x1"])]
            vertical = recognize_vertical_dimension(crop, image_rgb, bound, outer_bound, allow_external=False)
            horizontal = recognize_horizontal_dimension(image_rgb, bound, outer_bound, raw_name or element)
            if vertical and (not horizontal or not dimensions_match_cell(horizontal, vertical, bound)):
                horizontal = recognize_external_horizontal_dimension(
                    image_rgb, bound, outer_bound, raw_name or element, vertical
                )
            if vertical and horizontal and dimensions_match_cell(horizontal, vertical, bound):
                horizontal_dimension, vertical_dimension = horizontal, vertical
                source_bound = bound
                break
        if not vertical_dimension:
            for bound in bucket_bounds:
                outer_bound = outer_bounds[int(bound["outer_bound"]) - 1]
                crop = image_rgb[int(bound["y"]) : int(bound["y1"]), int(bound["x"]) : int(bound["x1"])]
                vertical = recognize_vertical_dimension(crop, image_rgb, bound, outer_bound, allow_external=True)
                horizontal = recognize_horizontal_dimension(image_rgb, bound, outer_bound, element)
                if vertical and (not horizontal or not dimensions_match_cell(horizontal, vertical, bound)):
                    horizontal = recognize_external_horizontal_dimension(image_rgb, bound, outer_bound, element, vertical)
                if vertical and horizontal and dimensions_match_cell(horizontal, vertical, bound):
                    horizontal_dimension, vertical_dimension = horizontal, vertical
                    source_bound = bound
                    break

        for bound, raw_name in zip(bucket_bounds, recognized):
            bound["raw_ocr_element"] = raw_name or None
            bound["element"] = element
            bound["label_source"] = "euclidean_bucket_majority"
            all_labeled.append(dict(bound))
        if element not in unique:
            unique[element] = {
                "count": 0,
                "horizontal_dimensions": [],
                "vertical_dimensions": [],
            }
        unique[element]["count"] = int(unique[element]["count"]) + len(bucket_bounds)
        if horizontal_dimension and horizontal_dimension not in unique[element]["horizontal_dimensions"]:
            unique[element]["horizontal_dimensions"].append(horizontal_dimension)
        if vertical_dimension and vertical_dimension not in unique[element]["vertical_dimensions"]:
            unique[element]["vertical_dimensions"].append(vertical_dimension)
        if source_bound is not None:
            source = dict(source_bound)
            source.update(
                {
                    "element": element,
                    "horizontal_dimension": horizontal_dimension,
                    "vertical_dimension": vertical_dimension,
                }
            )
            size_sources.append(source)
        bucket_summary.append(
            {
                "bucket_id": int(bucket["bucket_id"]),
                "count": len(bucket_bounds),
                "centroid_bound_px": {
                    "width": round(float(bucket["centroid_width_px"]), 2),
                    "height": round(float(bucket["centroid_height_px"]), 2),
                },
                "element": element,
                "ocr_element": ocr_element,
                "label_votes": dict(votes),
                "horizontal_dimension": horizontal_dimension or None,
                "vertical_dimension": vertical_dimension or None,
            }
        )
        print(
            f"EUCLIDEAN BUCKETING bucket={bucket['bucket_id']}, count={len(bucket_bounds)}, "
            f"centroid={bucket['centroid_width_px']:.2f}x{bucket['centroid_height_px']:.2f}px, "
            f"element={element}, size={horizontal_dimension or '?'}x{vertical_dimension or '?'}"
        )
    return size_sources, all_labeled, validated, unique, bucket_summary


def draw_labeled_inner_bounds(image_rgb: np.ndarray, labeled_bounds: list[dict[str, object]]) -> np.ndarray:
    annotated = image_rgb.copy()
    for bound in labeled_bounds:
        x, y, x1, y1 = (int(bound[key]) for key in ("x", "y", "x1", "y1"))
        cv2.rectangle(annotated, (x, y), (x1, y1), (220, 0, 220), 3)
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


def add_average_bound_sizes(
    unique_elements: dict[str, dict[str, object]],
    all_labeled_bounds: list[dict[str, object]],
) -> None:
    """Attach average bound width, height, and area in pixels to every bucket."""
    for element, values in unique_elements.items():
        bounds = [bound for bound in all_labeled_bounds if bound.get("element") == element]
        if not bounds:
            values["average_bound_size_px"] = None
            values["average_bound_area_px"] = None
            continue
        average_width = float(np.mean([int(bound["width"]) for bound in bounds]))
        average_height = float(np.mean([int(bound["height"]) for bound in bounds]))
        average_area = float(np.mean([int(bound["width"]) * int(bound["height"]) for bound in bounds]))
        values["average_bound_size_px"] = {
            "width": round(average_width, 2),
            "height": round(average_height, 2),
        }
        values["average_bound_area_px"] = round(average_area, 2)


def merge_correction_step(
    unique_elements: dict[str, dict[str, object]],
    all_labeled_bounds: list[dict[str, object]],
    validated_inner_bounds: list[dict[str, object]],
    tolerance_mm: float,
) -> list[dict[str, object]]:
    """Merge singleton OCR failures into a geometrically matching known group."""
    print("MERGE CORRECTION: starting correction bounds analysis")
    bounds_by_element: dict[str, list[dict[str, object]]] = {}
    for bound in all_labeled_bounds:
        element = str(bound.get("element", ""))
        if element:
            bounds_by_element.setdefault(element, []).append(bound)

    reliable: list[dict[str, object]] = []
    for element, values in unique_elements.items():
        horizontal = values.get("horizontal_dimensions", [])
        vertical = values.get("vertical_dimensions", [])
        element_bounds = bounds_by_element.get(element, [])
        if int(values.get("count", 0)) == 1 or len(horizontal) != 1 or len(vertical) != 1 or not element_bounds:
            continue
        reliable.append({
            "element": element,
            "count": int(values.get("count", 0)),
            "width_mm": float(str(horizontal[0])),
            "height_mm": float(str(vertical[0])),
            "width_px": float(np.median([int(bound["width"]) for bound in element_bounds])),
            "height_px": float(np.median([int(bound["height"]) for bound in element_bounds])),
        })
    reliable.sort(key=lambda item: (-int(item["count"]), str(item["element"])))
    print(
        "MERGE CORRECTION baskets sorted by count: "
        + ", ".join(f"{item['element']}={item['count']}" for item in reliable)
    )

    suspicious = [
        element
        for element, values in unique_elements.items()
        if int(values.get("count", 0)) == 1
        and not values.get("horizontal_dimensions")
        and not values.get("vertical_dimensions")
    ]
    corrections: list[dict[str, object]] = []
    for suspicious_element in suspicious:
        element_bounds = bounds_by_element.get(suspicious_element, [])
        if len(element_bounds) != 1:
            continue
        bound = element_bounds[0]
        print(
            "MERGE CORRECTION suspicious: "
            f"element={suspicious_element}, outer_bound={bound.get('outer_bound')}, "
            f"bbox=({bound['x']}, {bound['y']}, {bound['width']}, {bound['height']}), sizes=not found"
        )
        match: dict[str, object] | None = None
        closest_distance = math.inf
        for basket_index, reference in enumerate(reliable, start=1):
            target_element = str(reference["element"])
            inferred_width_mm = int(bound["width"]) * reference["width_mm"] / reference["width_px"]
            inferred_height_mm = int(bound["height"]) * reference["height_mm"] / reference["height_px"]
            width_diff_mm = abs(inferred_width_mm - reference["width_mm"])
            height_diff_mm = abs(inferred_height_mm - reference["height_mm"])
            distance_mm = math.hypot(width_diff_mm, height_diff_mm)
            print(
                f"MERGE CORRECTION compare: suspicious={suspicious_element}, "
                f"basket={target_element}, basket_count={reference['count']}, order={basket_index}, "
                f"delta=({width_diff_mm:.2f} mm, {height_diff_mm:.2f} mm), distance={distance_mm:.2f} mm"
            )
            if distance_mm < closest_distance:
                closest_distance = distance_mm
                match = {
                    "target_element": target_element,
                    "target_count_before_merge": int(unique_elements[target_element]["count"]),
                    "basket_order": basket_index,
                    "width_diff_mm": width_diff_mm,
                    "height_diff_mm": height_diff_mm,
                    "euclidean_distance_mm": distance_mm,
                }
        if match is None or closest_distance > tolerance_mm:
            print(f"MERGE CORRECTION skipped: element={suspicious_element}, no geometry match within {tolerance_mm:g} mm")
            continue

        target_element = str(match["target_element"])
        unique_elements[target_element]["count"] = int(unique_elements[target_element]["count"]) + 1
        del unique_elements[suspicious_element]
        bound["original_element"] = suspicious_element
        bound["element"] = target_element
        for validated in validated_inner_bounds:
            if (
                validated.get("element") == suspicious_element
                and int(validated["x"]) == int(bound["x"])
                and int(validated["y"]) == int(bound["y"])
            ):
                validated["original_element"] = suspicious_element
                validated["element"] = target_element
        correction = {
            "from_element": suspicious_element,
            "to_element": target_element,
            "bound": {key: bound[key] for key in ("x", "y", "x1", "y1", "width", "height", "outer_bound")},
            "basket_order": int(match["basket_order"]),
            "target_count_before_merge": int(match["target_count_before_merge"]),
            "target_count_after_merge": int(unique_elements[target_element]["count"]),
            "width_diff_mm": round(float(match["width_diff_mm"]), 2),
            "height_diff_mm": round(float(match["height_diff_mm"]), 2),
            "euclidean_distance_mm": round(float(match["euclidean_distance_mm"]), 2),
        }
        corrections.append(correction)
        print(
            f"MERGE CORRECTION merged: {suspicious_element} -> {target_element}, "
            f"delta=({correction['width_diff_mm']} mm, {correction['height_diff_mm']} mm), "
            f"distance={correction['euclidean_distance_mm']} mm"
        )
    print(f"MERGE CORRECTION: completed, merged={len(corrections)}, suspicious={len(suspicious)}")
    return corrections


def merge_buckets_by_bound_size(
    unique_elements: dict[str, dict[str, object]],
    all_labeled_bounds: list[dict[str, object]],
    validated_inner_bounds: list[dict[str, object]],
    size_source_bounds: list[dict[str, object]],
    tolerance_mm: float,
) -> list[dict[str, object]]:
    """Merge whole OCR buckets into larger buckets with matching bound sizes."""
    print("BUCKETS MERGE: starting bounds-size bucket merge")

    def current_bounds(element: str) -> list[dict[str, object]]:
        return [bound for bound in all_labeled_bounds if bound.get("element") == element]

    ordered_elements = sorted(
        unique_elements,
        key=lambda element: (-int(unique_elements[element].get("count", 0)), element),
    )
    print(
        "BUCKETS MERGE buckets sorted by count: "
        + ", ".join(f"{element}={unique_elements[element].get('count', 0)}" for element in ordered_elements)
    )
    canonical: list[str] = []
    merges: list[dict[str, object]] = []
    for source_element in ordered_elements:
        if source_element not in unique_elements:
            continue
        source_bounds = current_bounds(source_element)
        if not source_bounds:
            canonical.append(source_element)
            continue
        source_width_px = float(np.median([int(bound["width"]) for bound in source_bounds]))
        source_height_px = float(np.median([int(bound["height"]) for bound in source_bounds]))
        matched_target: str | None = None
        match_info: dict[str, float] = {}
        closest_distance = math.inf
        for target_element in canonical:
            if target_element not in unique_elements:
                continue
            target_values = unique_elements[target_element]
            horizontal = target_values.get("horizontal_dimensions", [])
            vertical = target_values.get("vertical_dimensions", [])
            target_bounds = current_bounds(target_element)
            if len(horizontal) != 1 or len(vertical) != 1 or not target_bounds:
                continue
            target_width_px = float(np.median([int(bound["width"]) for bound in target_bounds]))
            target_height_px = float(np.median([int(bound["height"]) for bound in target_bounds]))
            target_width_mm = float(str(horizontal[0]))
            target_height_mm = float(str(vertical[0]))
            inferred_width_mm = source_width_px * target_width_mm / target_width_px
            inferred_height_mm = source_height_px * target_height_mm / target_height_px
            width_diff_mm = abs(inferred_width_mm - target_width_mm)
            height_diff_mm = abs(inferred_height_mm - target_height_mm)
            distance_mm = math.hypot(width_diff_mm, height_diff_mm)
            print(
                f"BUCKETS MERGE compare: source={source_element}, target={target_element}, "
                f"target_count={target_values.get('count', 0)}, "
                f"delta=({width_diff_mm:.2f} mm, {height_diff_mm:.2f} mm), distance={distance_mm:.2f} mm"
            )
            if distance_mm < closest_distance:
                closest_distance = distance_mm
                matched_target = target_element
                match_info = {
                    "source_width_px": source_width_px,
                    "source_height_px": source_height_px,
                    "target_width_px": target_width_px,
                    "target_height_px": target_height_px,
                    "width_diff_mm": width_diff_mm,
                    "height_diff_mm": height_diff_mm,
                    "euclidean_distance_mm": distance_mm,
                }
        if matched_target is None or closest_distance > tolerance_mm:
            canonical.append(source_element)
            continue

        source_count = int(unique_elements[source_element].get("count", 0))
        target_count_before = int(unique_elements[matched_target].get("count", 0))
        unique_elements[matched_target]["count"] = target_count_before + source_count
        del unique_elements[source_element]
        for collection in (all_labeled_bounds, validated_inner_bounds, size_source_bounds):
            for bound in collection:
                if bound.get("element") == source_element:
                    bound.setdefault("original_element", source_element)
                    bound["element"] = matched_target
        merge = {
            "from_bucket": source_element,
            "to_bucket": matched_target,
            "source_count": source_count,
            "target_count_before_merge": target_count_before,
            "target_count_after_merge": target_count_before + source_count,
            "source_median_bound_px": [round(match_info["source_width_px"], 2), round(match_info["source_height_px"], 2)],
            "target_median_bound_px": [round(match_info["target_width_px"], 2), round(match_info["target_height_px"], 2)],
            "width_diff_mm": round(match_info["width_diff_mm"], 2),
            "height_diff_mm": round(match_info["height_diff_mm"], 2),
            "euclidean_distance_mm": round(match_info["euclidean_distance_mm"], 2),
        }
        merges.append(merge)
        print(
            f"BUCKETS MERGE merged: {source_element} -> {matched_target}, "
            f"count={source_count}, delta=({merge['width_diff_mm']} mm, {merge['height_diff_mm']} mm), "
            f"distance={merge['euclidean_distance_mm']} mm"
        )
    print(f"BUCKETS MERGE: completed, merged_buckets={len(merges)}, remaining_buckets={len(unique_elements)}")
    return merges


def final_merge_buckets_by_sorted_bounds(
    unique_elements: dict[str, dict[str, object]],
    all_labeled_bounds: list[dict[str, object]],
    validated_inner_bounds: list[dict[str, object]],
    size_source_bounds: list[dict[str, object]],
    tolerance_px: float,
) -> list[dict[str, object]]:
    """Final pass: sort buckets by median bound size and merge close clusters."""
    print("FINAL BUCKETS MERGE: starting size-sorted merge")

    records: list[dict[str, object]] = []
    for element, values in unique_elements.items():
        bounds = [bound for bound in all_labeled_bounds if bound.get("element") == element]
        if not bounds:
            continue
        width_px = float(np.median([int(bound["width"]) for bound in bounds]))
        height_px = float(np.median([int(bound["height"]) for bound in bounds]))
        records.append(
            {
                "element": element,
                "count": int(values.get("count", 0)),
                "width_px": width_px,
                "height_px": height_px,
                "area_px": width_px * height_px,
                "has_size": bool(values.get("horizontal_dimensions") and values.get("vertical_dimensions")),
            }
        )
    records.sort(key=lambda item: (float(item["area_px"]), float(item["width_px"]), float(item["height_px"]), str(item["element"])))
    for index, record in enumerate(records, start=1):
        print(
            f"FINAL BUCKETS MERGE sorted[{index}]: element={record['element']}, count={record['count']}, "
            f"median_bound={record['width_px']:.2f}x{record['height_px']:.2f}px, area={record['area_px']:.2f}px2"
        )

    canonical: list[dict[str, object]] = []
    merges: list[dict[str, object]] = []
    for source in records:
        source_element = str(source["element"])
        if source_element not in unique_elements:
            continue
        target: dict[str, object] | None = None
        closest_distance = math.inf
        for candidate in canonical:
            width_diff = abs(float(source["width_px"]) - float(candidate["width_px"]))
            height_diff = abs(float(source["height_px"]) - float(candidate["height_px"]))
            distance = math.hypot(width_diff, height_diff)
            print(
                f"FINAL BUCKETS MERGE compare: source={source_element}, target={candidate['element']}, "
                f"delta=({width_diff:.2f}px, {height_diff:.2f}px), distance={distance:.2f}px"
            )
            if distance < closest_distance:
                closest_distance = distance
                target = candidate
        if target is None or closest_distance > tolerance_px:
            canonical.append(source)
            continue
        target_element = str(target["element"])
        width_diff = abs(float(source["width_px"]) - float(target["width_px"]))
        height_diff = abs(float(source["height_px"]) - float(target["height_px"]))
        source_count = int(unique_elements[source_element].get("count", 0))
        target_count_before = int(unique_elements[target_element].get("count", 0))
        unique_elements[target_element]["count"] = target_count_before + source_count
        del unique_elements[source_element]
        changed_bounds = 0
        for collection in (all_labeled_bounds, validated_inner_bounds, size_source_bounds):
            for bound in collection:
                if bound.get("element") == source_element:
                    bound.setdefault("original_element", source_element)
                    bound["element"] = target_element
                    if collection is all_labeled_bounds:
                        changed_bounds += 1
        merge = {
            "from_bucket": source_element,
            "to_bucket": target_element,
            "reason": f"closest Euclidean distance {closest_distance:.2f}px within {tolerance_px:g}px tolerance",
            "source_count": source_count,
            "target_count_before_merge": target_count_before,
            "target_count_after_merge": target_count_before + source_count,
            "changed_bounds": changed_bounds,
            "source_median_bound_px": [round(float(source["width_px"]), 2), round(float(source["height_px"]), 2)],
            "target_median_bound_px": [round(float(target["width_px"]), 2), round(float(target["height_px"]), 2)],
            "width_diff_px": round(width_diff, 2),
            "height_diff_px": round(height_diff, 2),
            "euclidean_distance_px": round(closest_distance, 2),
        }
        merges.append(merge)
        print(
            f"FINAL BUCKETS MERGE merged: {source_element} -> {target_element}; "
            f"distance={closest_distance:.2f}px; count {target_count_before}+{source_count}="
            f"{target_count_before + source_count}; renamed_bounds={changed_bounds}"
        )
    print(
        f"FINAL BUCKETS MERGE: completed, canonical_buckets={len(canonical)}, "
        f"merged_buckets={len(merges)}, remaining_buckets={len(unique_elements)}"
    )
    return merges


def merge_unresolved_bounds_with_buckets(
    unique_elements: dict[str, dict[str, object]],
    all_labeled_bounds: list[dict[str, object]],
    validated_inner_bounds: list[dict[str, object]],
    size_source_bounds: list[dict[str, object]],
    tolerance_px: float,
) -> dict[str, list[dict[str, object]]]:
    """Resolve buckets without sizes and cells without labels from bound geometry."""
    print("UNRESOLVED BOUNDS MERGE: starting no-size and no-label analysis")

    def bucket_record(element: str) -> dict[str, object] | None:
        bounds = [bound for bound in all_labeled_bounds if bound.get("element") == element]
        if not bounds or element not in unique_elements:
            return None
        values = unique_elements[element]
        return {
            "element": element,
            "count": int(values.get("count", 0)),
            "width_px": float(np.mean([int(bound["width"]) for bound in bounds])),
            "height_px": float(np.mean([int(bound["height"]) for bound in bounds])),
            "has_size": bool(values.get("horizontal_dimensions") and values.get("vertical_dimensions")),
        }

    records = [record for element in unique_elements if (record := bucket_record(element)) is not None]
    reliable = sorted(
        [record for record in records if record["has_size"]],
        key=lambda item: (-int(item["count"]), str(item["element"])),
    )
    no_size = sorted(
        [record for record in records if not record["has_size"]],
        key=lambda item: (-int(item["count"]), str(item["element"])),
    )
    print(
        "UNRESOLVED BOUNDS MERGE reliable buckets: "
        + ", ".join(f"{item['element']}={item['count']}" for item in reliable)
    )

    bucket_merges: list[dict[str, object]] = []
    for source in no_size:
        source_element = str(source["element"])
        if source_element not in unique_elements:
            continue
        print(
            f"UNRESOLVED BOUNDS MERGE no-size bucket: element={source_element}, count={source['count']}, "
            f"average_bound={source['width_px']:.2f}x{source['height_px']:.2f}px"
        )
        target: dict[str, object] | None = None
        closest_distance = math.inf
        for candidate in reliable:
            width_diff = abs(float(source["width_px"]) - float(candidate["width_px"]))
            height_diff = abs(float(source["height_px"]) - float(candidate["height_px"]))
            distance = math.hypot(width_diff, height_diff)
            print(
                f"UNRESOLVED BOUNDS MERGE compare bucket: source={source_element}, target={candidate['element']}, "
                f"target_count={candidate['count']}, delta=({width_diff:.2f}px, {height_diff:.2f}px), "
                f"distance={distance:.2f}px"
            )
            if distance < closest_distance:
                closest_distance = distance
                target = candidate
        if target is None:
            print(f"UNRESOLVED BOUNDS MERGE bucket skipped: {source_element}, no reliable buckets")
            continue
        target_element = str(target["element"])
        source_count = int(unique_elements[source_element]["count"])
        target_count_before = int(unique_elements[target_element]["count"])
        unique_elements[target_element]["count"] = target_count_before + source_count
        del unique_elements[source_element]
        changed_bounds = 0
        for collection in (all_labeled_bounds, validated_inner_bounds, size_source_bounds):
            for bound in collection:
                if bound.get("element") == source_element:
                    bound.setdefault("original_element", source_element)
                    bound["element"] = target_element
                    if collection is all_labeled_bounds:
                        changed_bounds += 1
        merge = {
            "from_bucket": source_element,
            "to_bucket": target_element,
            "source_count": source_count,
            "target_count_before_merge": target_count_before,
            "target_count_after_merge": target_count_before + source_count,
            "changed_bounds": changed_bounds,
            "source_average_bound_px": [round(float(source["width_px"]), 2), round(float(source["height_px"]), 2)],
            "target_average_bound_px": [round(float(target["width_px"]), 2), round(float(target["height_px"]), 2)],
            "width_diff_px": round(abs(float(source["width_px"]) - float(target["width_px"])), 2),
            "height_diff_px": round(abs(float(source["height_px"]) - float(target["height_px"])), 2),
            "euclidean_distance_px": round(closest_distance, 2),
            "selection_reason": "closest reliable bucket by average bound width/height in pixels",
        }
        bucket_merges.append(merge)
        print(
            f"UNRESOLVED BOUNDS MERGE bucket merged: {source_element} -> {target_element}, "
            f"distance={closest_distance:.2f}px, count {target_count_before}+{source_count}={target_count_before + source_count}"
        )

    # Rebuild references after whole-bucket merges because counts and labels changed.
    references = [record for element in unique_elements if (record := bucket_record(element)) is not None]
    references.sort(key=lambda item: (-bool(item["has_size"]), -int(item["count"]), str(item["element"])))
    unlabeled_assignments: list[dict[str, object]] = []
    for bound in validated_inner_bounds:
        if bound.get("element"):
            continue
        print(
            f"UNRESOLVED BOUNDS MERGE unlabeled cell: outer_bound={bound.get('outer_bound')}, "
            f"bbox=({bound['x']},{bound['y']},{bound['width']},{bound['height']})"
        )
        target: dict[str, object] | None = None
        closest_distance = math.inf
        for candidate in references:
            width_diff = abs(int(bound["width"]) - float(candidate["width_px"]))
            height_diff = abs(int(bound["height"]) - float(candidate["height_px"]))
            distance = math.hypot(width_diff, height_diff)
            print(
                f"UNRESOLVED BOUNDS MERGE compare cell: target={candidate['element']}, "
                f"target_count={candidate['count']}, delta=({width_diff:.2f}px, {height_diff:.2f}px), "
                f"distance={distance:.2f}px"
            )
            if distance < closest_distance:
                closest_distance = distance
                target = candidate
        if target is None or closest_distance > tolerance_px:
            print("UNRESOLVED BOUNDS MERGE unlabeled skipped: no matching bucket")
            continue
        target_element = str(target["element"])
        target_count_before = int(unique_elements[target_element]["count"])
        unique_elements[target_element]["count"] = target_count_before + 1
        bound["element"] = target_element
        bound["label_source"] = "bounds_size_merge"
        labeled = dict(bound)
        all_labeled_bounds.append(labeled)
        assignment = {
            "to_bucket": target_element,
            "bound": {key: bound[key] for key in ("x", "y", "x1", "y1", "width", "height", "outer_bound")},
            "target_count_before_merge": target_count_before,
            "target_count_after_merge": target_count_before + 1,
            "target_median_bound_px": [round(float(target["width_px"]), 2), round(float(target["height_px"]), 2)],
            "width_diff_px": round(abs(int(bound["width"]) - float(target["width_px"])), 2),
            "height_diff_px": round(abs(int(bound["height"]) - float(target["height_px"])), 2),
            "euclidean_distance_px": round(closest_distance, 2),
        }
        unlabeled_assignments.append(assignment)
        print(
            f"UNRESOLVED BOUNDS MERGE unlabeled assigned: bbox=({bound['x']},{bound['y']}) -> {target_element}, "
            f"distance={closest_distance:.2f}px, count {target_count_before}+1={target_count_before + 1}"
        )
    print(
        f"UNRESOLVED BOUNDS MERGE: completed, bucket_merges={len(bucket_merges)}, "
        f"unlabeled_assignments={len(unlabeled_assignments)}"
    )
    return {"bucket_merges": bucket_merges, "unlabeled_assignments": unlabeled_assignments}


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
    parser.add_argument(
        "--element-bucketing-algorithm",
        choices=("euclidean", "legacy"),
        default="euclidean",
        help="Element grouping branch: geometry-first euclidean (default) or legacy OCR/merge pipeline.",
    )
    parser.add_argument(
        "--euclidean-bucket-tolerance-px",
        type=float,
        default=6.0,
        help="Maximum Euclidean distance in (bound width, bound height) for the default bucket algorithm.",
    )
    parser.add_argument(
        "--merge-correction-tolerance-mm",
        type=float,
        default=10.0,
        help="Maximum width/height difference for singleton OCR merge correction.",
    )
    parser.add_argument(
        "--bucket-merge-tolerance-mm",
        type=float,
        default=10.0,
        help="Maximum width/height difference when merging complete buckets by bound size.",
    )
    parser.add_argument(
        "--final-bucket-merge-tolerance-px",
        type=float,
        default=3.0,
        help="Final median-bound width/height tolerance for size-sorted bucket merging.",
    )
    parser.add_argument(
        "--unresolved-bound-merge-tolerance-px",
        type=float,
        default=3.0,
        help="Tolerance for no-size buckets and unlabeled cells matched to known buckets.",
    )
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
    merge_corrections: list[dict[str, object]] = []
    bucket_merges: list[dict[str, object]] = []
    final_bucket_merges: list[dict[str, object]] = []
    unresolved_merges: dict[str, list[dict[str, object]]] = {"bucket_merges": [], "unlabeled_assignments": []}
    euclidean_buckets: list[dict[str, object]] = []
    if args.element_bucketing_algorithm == "euclidean":
        size_source_bounds, all_labeled_bounds, validated_inner_bounds, unique_elements, euclidean_buckets = (
            analyze_euclidean_bound_buckets(
                target_rgb,
                mask,
                bounds,
                args.ocr_language,
                args.min_cell_width,
                args.min_cell_height,
                args.min_cell_hatch_ratio,
                args.min_cell_hatch_pixels,
                args.euclidean_bucket_tolerance_px,
            )
        )
    else:
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
        merge_corrections = merge_correction_step(
            unique_elements, all_labeled_bounds, validated_inner_bounds, args.merge_correction_tolerance_mm
        )
        bucket_merges = merge_buckets_by_bound_size(
            unique_elements, all_labeled_bounds, validated_inner_bounds, size_source_bounds, args.bucket_merge_tolerance_mm
        )
        final_bucket_merges = final_merge_buckets_by_sorted_bounds(
            unique_elements,
            all_labeled_bounds,
            validated_inner_bounds,
            size_source_bounds,
            args.final_bucket_merge_tolerance_px,
        )
        unresolved_merges = merge_unresolved_bounds_with_buckets(
            unique_elements,
            all_labeled_bounds,
            validated_inner_bounds,
            size_source_bounds,
            args.unresolved_bound_merge_tolerance_px,
        )
    add_average_bound_sizes(unique_elements, all_labeled_bounds)
    total_area_m2 = add_area_totals(unique_elements)
    elements_annotated_rgb = draw_labeled_inner_bounds(target_rgb, validated_inner_bounds)
    unlabeled_bucket_bounds = [
        {
            key: bound.get(key)
            for key in ("element", "bucket_id", "outer_bound", "x", "y", "x1", "y1", "width", "height")
        }
        for bound in all_labeled_bounds
        if str(bound.get("element", "")).startswith("UNLABELED-BUCKET-")
    ]

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
        "element_bucketing_algorithm": args.element_bucketing_algorithm,
        "euclidean_bucketing": {
            "enabled": args.element_bucketing_algorithm == "euclidean",
            "tolerance_px": args.euclidean_bucket_tolerance_px,
            "buckets": euclidean_buckets,
        },
        "labels_restricted_to_hatch_bounds": True,
        "inner_bounds_require_hatch_match": True,
        "cell_hatch_validation": {
            "min_ratio": args.min_cell_hatch_ratio,
            "min_pixels": args.min_cell_hatch_pixels,
        },
        "merge_correction": {
            "tolerance_mm": args.merge_correction_tolerance_mm,
            "corrections": merge_corrections,
        },
        "buckets_merge": {
            "tolerance_mm": args.bucket_merge_tolerance_mm,
            "merges": bucket_merges,
        },
        "final_buckets_merge": {
            "tolerance_px": args.final_bucket_merge_tolerance_px,
            "merges": final_bucket_merges,
        },
        "unresolved_bounds_merge": {
            "tolerance_px": args.unresolved_bound_merge_tolerance_px,
            **unresolved_merges,
        },
        "containment_verified": all(
            bound_is_inside(bound, bounds[int(bound["outer_bound"]) - 1])
            for bound in all_labeled_bounds
        ),
        "unique_elements": unique_elements,
        "unlabeled_bucket_bounds": unlabeled_bucket_bounds,
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
        if element.startswith("UNLABELED-BUCKET-"):
            element_bounds = [bound for bound in unlabeled_bucket_bounds if bound.get("element") == element]
            for bound in element_bounds:
                print(
                    "    UNLABELED BOUND: "
                    f"outer_bound={bound.get('outer_bound')}, bucket_id={bound.get('bucket_id')}, "
                    f"bbox=({bound.get('x')}, {bound.get('y')}, {bound.get('x1')}, {bound.get('y1')}), "
                    f"size_px={bound.get('width')}x{bound.get('height')}"
                )
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
