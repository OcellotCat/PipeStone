#!/usr/bin/env python3
"""Mask an image by colors learned from a hatch sample."""

from __future__ import annotations

import argparse
import json
import math
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


def gabor_hatch_response(image_rgb: np.ndarray, color_mask: np.ndarray, hatch_definition: dict[str, object]) -> np.ndarray:
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

    response = cv2.filter2D(inverted.astype(np.float32), cv2.CV_32F, kernel)
    response = np.abs(response)
    response *= color_mask.astype(np.float32)
    response = cv2.GaussianBlur(response, (5, 5), 0)
    return cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Color-mask an image using the non-white colors from a hatch sample.")
    parser.add_argument("--sample", default="hatch.png", help="Reference hatch image to learn colors from.")
    parser.add_argument("--target", default="test_tiled.jpg", help="Image to process.")
    parser.add_argument("--output", default="test_tiled_color_masked.png", help="Masked output image path.")
    parser.add_argument("--mask-output", default="", help="Optional grayscale mask output path.")
    parser.add_argument("--hatch-definition-output", default="hatch_definition.json", help="Hatch definition JSON output path.")
    parser.add_argument("--bounds-output", default="test_tiled_hatch_bounds.json", help="Matched area bounds JSON output path.")
    parser.add_argument("--bounds-image-output", default="test_tiled_hatch_bounds.png", help="Annotated bounds image output path.")
    parser.add_argument("--gabor-response-output", default="", help="Optional grayscale Gabor response output path.")
    parser.add_argument("--gabor-mask-output", default="", help="Optional grayscale Gabor match mask output path.")
    parser.add_argument("--threshold", type=float, default=18.0, help="LAB color distance threshold for matches.")
    parser.add_argument("--gabor-threshold", type=int, default=42, help="Gabor response threshold for hatch matches.")
    parser.add_argument("--gabor-close-size", type=int, default=31, help="Morphological close size for grouping Gabor matches.")
    parser.add_argument("--gabor-dilate-size", type=int, default=11, help="Morphological dilation size for grouping Gabor matches.")
    parser.add_argument("--min-bound-area", type=int, default=1200, help="Minimum contour area for reported hatch bounds.")
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

    gabor_response = gabor_hatch_response(target_rgb, mask, hatch_definition)
    gabor_mask = build_gabor_match_mask(
        gabor_response,
        mask,
        args.gabor_threshold,
        args.gabor_close_size,
        args.gabor_dilate_size,
    )
    bounds = find_match_bounds(gabor_mask, gabor_response, args.min_bound_area)
    annotated_rgb = draw_bounds(target_rgb, bounds)

    hatch_definition_path = Path(args.hatch_definition_output)
    hatch_definition_path.parent.mkdir(parents=True, exist_ok=True)
    hatch_definition_path.write_text(json.dumps(hatch_definition, indent=2), encoding="utf-8")

    bounds_result = {
        "sample": str(sample_path),
        "target": str(target_path),
        "hatch_definition": hatch_definition,
        "gabor": {
            "threshold": int(args.gabor_threshold),
            "close_size": int(args.gabor_close_size),
            "dilate_size": int(args.gabor_dilate_size),
            "min_bound_area": int(args.min_bound_area),
        },
        "bounds": bounds,
    }
    bounds_output_path = Path(args.bounds_output)
    bounds_output_path.parent.mkdir(parents=True, exist_ok=True)
    bounds_output_path.write_text(json.dumps(bounds_result, indent=2), encoding="utf-8")

    write_rgb(Path(args.bounds_image_output), annotated_rgb)
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
    print(f"Wrote: {output_path}")
    print(f"Wrote hatch definition: {hatch_definition_path}")
    print(f"Wrote bounds: {bounds_output_path}")
    print(f"Wrote bounds image: {args.bounds_image_output}")
    if args.mask_output:
        print(f"Wrote mask: {args.mask_output}")
    if args.gabor_response_output:
        print(f"Wrote Gabor response: {args.gabor_response_output}")
    if args.gabor_mask_output:
        print(f"Wrote Gabor mask: {args.gabor_mask_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
