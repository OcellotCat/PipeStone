#!/usr/bin/env python3
"""Mask an image by colors learned from a hatch sample."""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Color-mask an image using the non-white colors from a hatch sample.")
    parser.add_argument("--sample", default="hatch.png", help="Reference hatch image to learn colors from.")
    parser.add_argument("--target", default="test_tiled.jpg", help="Image to process.")
    parser.add_argument("--output", default="test_tiled_color_masked.png", help="Masked output image path.")
    parser.add_argument("--mask-output", default="", help="Optional grayscale mask output path.")
    parser.add_argument("--threshold", type=float, default=18.0, help="LAB color distance threshold for matches.")
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

    matched_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    print(f"Palette colors: {len(palette_rgb)}")
    print(f"Matched pixels: {matched_pixels} / {total_pixels} ({matched_pixels / total_pixels:.2%})")
    print(f"Wrote: {output_path}")
    if args.mask_output:
        print(f"Wrote mask: {args.mask_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
