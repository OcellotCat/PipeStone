#!/usr/bin/env python3
"""Main analysis pipeline for material mention search."""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from pathlib import Path
from typing import Any

from pipestone_cv import require_module, trim_white_margins
from pipestone_ocr import collect_ocr_words, render_pdf_pages, run_image_ocr
from pipeline_material_search import (
    MaterialLegendSample,
    MaterialMention,
    MaterialRegionMatch,
    analyze_image_materials,
    extract_material_legend_samples,
    extract_material_mentions,
    find_material_regions,
    load_drawing_image,
)

logger = logging.getLogger("pipestone")

APP_NAME = "PipeStone PDF Material Mention Search"
DEFAULT_DPI = 400
DEFAULT_OUTPUT_DIR = "output"

DEBUG_REF_DIR_NAME = "material_reference_debug"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def log_material_mentions(mentions: list[MaterialMention]) -> None:
    logger.info("========== MATERIAL MENTIONS ==========")
    if not mentions:
        logger.info("No material mentions found")
        return

    mentions_by_page: dict[int, list[MaterialMention]] = {}
    for mention in mentions:
        mentions_by_page.setdefault(mention.page, []).append(mention)

    for page in sorted(mentions_by_page):
        logger.info("Page %s:", page)
        for index, mention in enumerate(mentions_by_page[page], start=1):
            logger.info(
                "  M%s: material=%r keyword=%r line_text=%r bbox=(%.1f,%.1f,%.1f,%.1f) source=%s confidence=%s",
                index,
                mention.material_name,
                mention.keyword,
                mention.line_text,
                mention.bbox[0],
                mention.bbox[1],
                mention.bbox[2],
                mention.bbox[3],
                mention.source,
                mention.confidence,
            )


def log_material_regions(regions: list[MaterialRegionMatch]) -> None:
    logger.info("========== MATERIAL REGIONS ==========")
    if not regions:
        logger.info("No material regions found")
        return

    regions_by_page: dict[int, list[MaterialRegionMatch]] = {}
    for region in regions:
        regions_by_page.setdefault(region.page, []).append(region)

    for page in sorted(regions_by_page):
        logger.info("Page %s:", page)
        for index, region in enumerate(regions_by_page[page], start=1):
            logger.info(
                "  R%s: material=%r confidence=%.3f bbox=(%.1f,%.1f,%.1f,%.1f)",
                index,
                region.material_name,
                region.confidence,
                region.bbox[0],
                region.bbox[1],
                region.bbox[2],
                region.bbox[3],
            )


def _mention_to_dict(mention: MaterialMention) -> dict[str, Any]:
    return {
        "page": mention.page,
        "material_name": mention.material_name,
        "line_text": mention.line_text,
        "keyword": mention.keyword,
        "bbox": list(mention.bbox),
        "confidence": mention.confidence,
        "source": mention.source,
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


def _region_to_dict(region: MaterialRegionMatch) -> dict[str, Any]:
    return {
        "page": region.page,
        "material_name": region.material_name,
        "bbox": list(region.bbox),
        "confidence": region.confidence,
        "descriptor": region.descriptor,
        "reference_sample_bbox": list(region.reference_sample_bbox),
    }


def _clip_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = [int(round(value)) for value in bbox]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def save_reference_sample_debug_images(
    image: Any,
    regions: list[MaterialRegionMatch],
    run_dir: Path,
    *,
    page: int,
) -> list[str]:
    if not regions:
        return []

    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    debug_dir = run_dir / DEBUG_REF_DIR_NAME
    debug_dir.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]
    annotated = np.array(image, copy=True)
    thickness = max(2, min(width, height) // 850)
    saved_paths: list[str] = []
    seen_reference_boxes: set[tuple[int, int, int, int]] = set()

    for index, region in enumerate(regions, start=1):
        region_box = _clip_bbox(region.bbox, width, height)
        reference_box = _clip_bbox(region.reference_sample_bbox, width, height)

        if region_box is not None:
            x0, y0, x1, y1 = region_box
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 160, 255), thickness, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"R{index} {region.confidence:.2f}",
                (x0, max(18, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 120, 255),
                max(1, thickness),
                cv2.LINE_AA,
            )

        if reference_box is None:
            continue

        rx0, ry0, rx1, ry1 = reference_box
        cv2.rectangle(annotated, (rx0, ry0), (rx1, ry1), (255, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"REF R{index}",
            (rx0, max(18, ry0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            max(1, thickness),
            cv2.LINE_AA,
        )

        if reference_box in seen_reference_boxes:
            continue
        seen_reference_boxes.add(reference_box)
        crop = image[ry0:ry1, rx0:rx1]
        if crop.size == 0:
            continue
        crop, _, _ = trim_white_margins(crop, padding=max(1, min(crop.shape[:2]) // 50))
        crop_path = debug_dir / f"page_{page:03d}_reference_sample_{len(seen_reference_boxes):03d}.png"
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        saved_paths.append(str(crop_path))

    annotated_path = debug_dir / f"page_{page:03d}_reference_sample_bboxes.png"
    cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    saved_paths.insert(0, str(annotated_path))
    logger.info("Saved reference sample debug image: %s", annotated_path)
    return saved_paths


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

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting material mention search: file=%s", pdf_path)
    rendered_pages = render_pdf_pages(pdf_path, dpi=dpi)
    words_by_page = collect_ocr_words(
        pdf_path,
        rendered_pages,
        backend=ocr_backend,
        force_ocr=force_ocr,
        tesseract_psm=tesseract_psm,
    )

    if save_rendered_pages:
        logger.warning("save_rendered_pages is ignored in mention-only mode")

    mentions = extract_material_mentions(words_by_page)
    log_material_mentions(mentions)

    legend_samples: list[MaterialLegendSample] = []
    material_regions: list[MaterialRegionMatch] = []
    reference_debug_images: list[str] = []
    for rendered_page in rendered_pages:
        page = rendered_page["page"]
        page_words = words_by_page.get(page, [])
        page_samples = extract_material_legend_samples(rendered_page["image"], page_words, page=page)
        legend_samples.extend(page_samples)
        page_regions = find_material_regions(rendered_page["image"], page_samples, page=page)
        material_regions.extend(page_regions)
        reference_debug_images.extend(
            save_reference_sample_debug_images(rendered_page["image"], page_regions, run_dir, page=page)
        )
    log_material_regions(material_regions)

    return {
        "pdf_path": str(pdf_path),
        "run_dir": str(run_dir),
        "mentions": [_mention_to_dict(mention) for mention in mentions],
        "legend_samples": [_legend_sample_to_dict(sample) for sample in legend_samples],
        "material_regions": [_region_to_dict(region) for region in material_regions],
        "reference_debug_images": reference_debug_images,
    }


def analyze_image_file(
    image_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ocr_backend: str = "tesseract",
    tesseract_psm: int = 11,
) -> dict[str, Any]:
    image_path = Path(image_path)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    image = load_drawing_image(str(image_path))
    words, warning = run_image_ocr(image, 1, ocr_backend, tesseract_psm=tesseract_psm)
    if warning:
        logger.warning("Image OCR warning: %s", warning)
    result = analyze_image_materials(str(image_path), words=words, page=1, image=image)
    material_regions = [
        MaterialRegionMatch(
            page=item["page"],
            material_name=item["material_name"],
            bbox=tuple(item["bbox"]),
            confidence=item["confidence"],
            descriptor=item["descriptor"],
            reference_sample_bbox=tuple(item["reference_sample_bbox"]),
        )
        for item in result["material_regions"]
    ]
    reference_debug_images = save_reference_sample_debug_images(image, material_regions, run_dir, page=1)
    log_material_regions(material_regions)
    result["run_dir"] = str(run_dir)
    result["reference_debug_images"] = reference_debug_images
    return result
