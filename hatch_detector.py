from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class MaterialRegion:
    bbox: tuple[int, int, int, int]
    area_px: int
    score: float


def build_gabor_bank():
    kernels = []

    for theta in (
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
    ):
        kernel = cv2.getGaborKernel(
            (31, 31),
            sigma=4.0,
            theta=theta,
            lambd=8.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F,
        )

        kernels.append(kernel)

    return kernels


def extract_pattern_patch(
    image: np.ndarray,
    material_bbox: tuple[int, int, int, int],
    patch_width: int = 150,
) -> np.ndarray:
    x1, y1, x2, y2 = material_bbox

    px1 = max(0, x1 - patch_width)
    px2 = x1

    return image[y1:y2, px1:px2]


def compute_gabor_descriptor(
    patch: np.ndarray,
    kernels,
) -> np.ndarray:

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    features = []

    for kernel in kernels:
        resp = cv2.filter2D(gray, cv2.CV_32F, kernel)

        features.append(resp.mean())
        features.append(resp.std())

    return np.array(features, dtype=np.float32)


def sliding_similarity_map(
    image: np.ndarray,
    template_descriptor: np.ndarray,
    kernels,
    tile_size: int = 128,
    stride: int = 32,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    heatmap = np.zeros(gray.shape, dtype=np.float32)

    h, w = gray.shape

    for y in range(0, h - tile_size, stride):
        for x in range(0, w - tile_size, stride):

            tile = image[y:y + tile_size, x:x + tile_size]

            desc = compute_gabor_descriptor(tile, kernels)

            score = cv2.compareHist(
                template_descriptor.reshape(-1, 1),
                desc.reshape(-1, 1),
                cv2.HISTCMP_CORREL,
            )

            heatmap[
                y:y + tile_size,
                x:x + tile_size,
            ] += score

    return heatmap


def heatmap_to_regions(
    heatmap: np.ndarray,
    threshold: float = 0.75,
):
    mask = (heatmap > threshold).astype(np.uint8) * 255

    kernel = np.ones((15, 15), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    regions = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 5000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        score = float(
            heatmap[y:y+h, x:x+w].mean()
        )

        regions.append(
            MaterialRegion(
                bbox=(x, y, x + w, y + h),
                area_px=int(area),
                score=score,
            )
        )

    return regions


def find_material_regions(
    image: np.ndarray,
    material_bbox: tuple[int, int, int, int],
):
    kernels = build_gabor_bank()

    pattern_patch = extract_pattern_patch(
        image,
        material_bbox,
    )

    template_descriptor = compute_gabor_descriptor(
        pattern_patch,
        kernels,
    )

    heatmap = sliding_similarity_map(
        image,
        template_descriptor,
        kernels,
    )

    regions = heatmap_to_regions(
        heatmap,
    )

    return {
        "count": len(regions),
        "regions": regions,
        "heatmap": heatmap,
        "pattern_patch": pattern_patch,
    }