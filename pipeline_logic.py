#!/usr/bin/env python3
"""Main analysis pipeline for PipeStone - facade layout PDF processing."""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from pipestone_cv import (
    bbox_center,
    detect_line_segments,
    detect_material_zones,
    estimate_zone_scale,
    extract_pattern_descriptor,
    filter_zones_by_hatch_pattern,
    match_zones_to_patterns,
    require_module,
)
from pipestone_ocr import (
    OcrWord,
    collect_ocr_words,
    render_pdf_pages,
    summarize_panels,
    words_to_lines,
)
from pipeline_material_search import (
    MaterialMention,
    extract_material_mentions,
    guess_material_name,
)

logger = logging.getLogger("pipestone")

APP_NAME = "PipeStone PDF Stone Area MVP"
DEFAULT_DPI = 220
DEFAULT_OUTPUT_DIR = "output"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def estimate_scale_for_page(image: Any, words: list[OcrWord], page_number: int) -> dict[str, Any]:
    from pipestone_ocr import find_dimension_candidates
    dimension_candidates = find_dimension_candidates(words)
    if not dimension_candidates:
        return {"page": page_number, "mm_per_px": None, "source": "missing", "candidates": []}

    segments = detect_line_segments(image)
    if not segments:
        return {"page": page_number, "mm_per_px": None, "source": "missing_lines", "candidates": []}

    height_px, width_px = image.shape[:2]
    search_distance = max(45.0, min(width_px, height_px) * 0.035)
    scale_candidates: list[dict[str, Any]] = []

    for dim in dimension_candidates:
        box = dim["bbox"]
        cx, cy = bbox_center(box)
        best = None
        best_score = float("inf")
        for segment in segments:
            if segment["length_px"] > max(width_px, height_px) * 0.92:
                continue
            if segment["orientation"] == "horizontal":
                axis_distance = abs(segment["cy"] - cy)
                axis_inside = segment["x1"] - search_distance * 2 <= cx <= segment["x2"] + search_distance * 2
                off_axis_distance = max(0.0, segment["x1"] - cx, cx - segment["x2"])
            else:
                axis_distance = abs(segment["cx"] - cx)
                axis_inside = segment["y1"] - search_distance * 2 <= cy <= segment["y2"] + search_distance * 2
                off_axis_distance = max(0.0, segment["y1"] - cy, cy - segment["y2"])

            if axis_distance > search_distance or not axis_inside:
                continue
            score = axis_distance + off_axis_distance * 0.25
            if best is None or score < best_score:
                best_score = score
                best = segment

        if best is None:
            continue
        mm_per_px = float(dim["value_mm"]) / max(float(best["length_px"]), 1.0)
        if mm_per_px < 0.02 or mm_per_px > 500.0:
            continue
        scale_candidates.append({
            "page": page_number,
            "dimension_text": dim["text"],
            "dimension_mm": round(float(dim["value_mm"]), 3),
            "line_length_px": round(float(best["length_px"]), 3),
            "mm_per_px": mm_per_px,
            "orientation": best["orientation"],
            "ocr_source": dim["source"],
        })

    if not scale_candidates:
        return {"page": page_number, "mm_per_px": None, "source": "missing_match", "candidates": []}

    values = [item["mm_per_px"] for item in scale_candidates]
    base = median(values)
    consistent = [v for v in values if abs(v - base) / max(base, 0.0001) <= 0.35]
    chosen = median(consistent or values)

    return {
        "page": page_number,
        "mm_per_px": chosen,
        "source": "dimension_line",
        "confidence": min(1.0, 0.45 + 0.1 * len(consistent or values)),
        "candidates": [{**item, "mm_per_px": round(float(item["mm_per_px"]), 6)} for item in scale_candidates[:20]],
    }


def build_scale_map(rendered_pages: list[dict], words_by_page: dict[int, list[OcrWord]], fallback_mm_per_px: float | None) -> tuple[dict[int, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    scale_by_page: dict[int, dict[str, Any]] = {}

    for page in rendered_pages:
        scale = estimate_scale_for_page(page["image"], words_by_page.get(page["page"], []), page["page"])
        scale_by_page[page["page"]] = scale

    detected_values = [
        scale["mm_per_px"]
        for scale in scale_by_page.values()
        if scale.get("mm_per_px") is not None and scale.get("source") == "dimension_line"
    ]
    global_scale = median(detected_values) if detected_values else None

    for page in rendered_pages:
        scale = scale_by_page[page["page"]]
        if scale.get("mm_per_px") is not None:
            continue
        if global_scale is not None:
            scale_by_page[page["page"]] = {**scale, "mm_per_px": global_scale, "source": "global_dimension_line", "confidence": 0.45}
        elif fallback_mm_per_px is not None:
            scale_by_page[page["page"]] = {**scale, "mm_per_px": float(fallback_mm_per_px), "source": "fallback_query_param", "confidence": 0.2}
            warnings.append(f"page {page['page']}: scale not detected, using fallback {fallback_mm_per_px} mm/px")
        else:
            warnings.append(f"page {page['page']}: scale not detected; metric sizes are null")

    return scale_by_page, warnings


def add_metric_fields(zone: dict[str, Any], mm_per_px: float | None, scale_source: str) -> dict[str, Any]:
    result = dict(zone)
    result["scale_mm_per_px"] = round(mm_per_px, 6) if mm_per_px is not None else None
    result["scale_source"] = scale_source
    if mm_per_px is None:
        result.update({"width_mm": None, "height_mm": None, "area_m2": None, "bbox_area_m2": None})
        return result

    width_mm = result["width_px"] * mm_per_px
    height_mm = result["height_px"] * mm_per_px
    area_m2 = result["area_px"] * mm_per_px * mm_per_px / 1_000_000.0
    bbox_area_m2 = result["bbox_area_px"] * mm_per_px * mm_per_px / 1_000_000.0
    result.update({
        "width_mm": round(width_mm, 1),
        "height_mm": round(height_mm, 1),
        "area_m2": round(area_m2, 4),
        "bbox_area_m2": round(bbox_area_m2, 4),
    })
    return result


def write_csv(run_dir: Path, panels: list[dict[str, Any]], summary: list[dict[str, Any]]) -> Path:
    csv_path = run_dir / "stone_panels.csv"
    fields = [
        "material_name", "page", "zone_id", "width_mm", "height_mm", "area_m2",
        "bbox_area_m2", "scale_mm_per_px", "scale_source", "area_px", "width_px",
        "height_px", "bbox_px", "material_line", "mean_intensity", "zone_dimensions",
        "pattern_id", "pattern_name", "pattern_score",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for panel in panels:
            row = dict(panel)
            row["bbox_px"] = json.dumps(row.get("bbox_px", []), ensure_ascii=False)
            row["zone_dimensions"] = json.dumps(row.get("zone_dimensions", []), ensure_ascii=False)
            writer.writerow(row)

    summary_path = run_dir / "stone_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["material_name", "panel_count", "area_m2", "bbox_area_m2", "pages"], extrasaction="ignore")
        writer.writeheader()
        for row in summary:
            out = dict(row)
            out["pages"] = ",".join(str(page) for page in out.get("pages", []))
            writer.writerow(out)

    return csv_path


def save_mention_debug_images(
    rendered_pages: list[dict[str, Any]],
    mentions: list[MaterialMention],
    run_dir: Path,
) -> dict[str, Any]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    mentions_by_page: dict[int, list[MaterialMention]] = defaultdict(list)
    for mention in mentions:
        mentions_by_page[mention.page].append(mention)

    debug_dir = run_dir / "mentions_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    saved_images: list[str] = []
    saved_mentions: list[dict[str, Any]] = []

    for page in rendered_pages:
        page_mentions = mentions_by_page.get(page["page"], [])
        if not page_mentions:
            continue

        height_px, width_px = page["image"].shape[:2]
        debug = np.array(page["image"], copy=True)
        thickness = max(2, min(width_px, height_px) // 500)

        for index, mention in enumerate(page_mentions, start=1):
            x0, y0, x1, y1 = [int(round(value)) for value in mention.bbox]
            x0 = max(0, min(x0, width_px - 1))
            y0 = max(0, min(y0, height_px - 1))
            x1 = max(0, min(x1, width_px - 1))
            y1 = max(0, min(y1, height_px - 1))

            color = (255, 0, 0)
            cv2.rectangle(debug, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
            cv2.putText(debug, f"M{index}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness + 1, cv2.LINE_AA)

        image_path = debug_dir / f"page_{page['page']:03d}_mentions.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
        saved_images.append(str(image_path))

        for index, mention in enumerate(page_mentions, start=1):
            saved_mentions.append(
                {
                    "image": str(image_path),
                    "mention_index": f"M{index}",
                    "page": mention.page,
                    "material_name": mention.material_name,
                    "line_text": mention.line_text,
                    "keyword": mention.keyword,
                    "bbox": [int(round(value)) for value in mention.bbox],
                    "source": mention.source,
                }
            )

    meta_path = debug_dir / "mentions.json"
    if saved_mentions:
        meta_path.write_text(
            json.dumps({"images": saved_images, "mentions": saved_mentions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved mention bbox debug images: %s", len(saved_images))
        return {"images": saved_images, "mentions": saved_mentions, "meta": str(meta_path)}

    return {"images": [], "mentions": [], "meta": None}


def extract_pattern_legends_for_page(
    image: Any,
    page_number: int,
    page_mentions: list[MaterialMention],
    all_zones: list[dict[str, Any]],
    page_width_px: int,
    page_height_px: int,
    words: list[OcrWord],
) -> list[dict[str, Any]]:
    page_area = float(page_width_px * page_height_px)
    max_pattern_area = page_area * 0.04
    legend_distance = min(page_width_px, page_height_px) * 0.15

    candidates: list[dict[str, Any]] = []
    for mention in page_mentions:
        candidates.append(
            {
                "name": mention.material_name,
                "line_text": mention.line_text,
                "bbox": mention.bbox,
                "source": "material_mention",
            }
        )

    if not candidates:
        for line in words_to_lines(words):
            name = guess_material_name(line.text) or line.text
            candidates.append(
                {
                    "name": name,
                    "line_text": line.text,
                    "bbox": line.bbox,
                    "source": "ocr_line",
                }
            )

    patterns: list[dict[str, Any]] = []
    used_zone_ids: set[str] = set()

    for candidate in candidates:
        candidate_center = bbox_center(candidate["bbox"])
        nearest_zone = None
        nearest_distance = float("inf")
        for zone in all_zones:
            if str(zone.get("zone_id")) in used_zone_ids:
                continue
            if zone.get("bbox_area_px", 0) > max_pattern_area:
                continue
            zone_center = bbox_center(tuple(float(value) for value in zone["bbox_px"]))
            distance = ((candidate_center[0] - zone_center[0]) ** 2 + (candidate_center[1] - zone_center[1]) ** 2) ** 0.5
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_zone = zone
        if nearest_zone is None or nearest_distance > legend_distance:
            continue

        descriptor = extract_pattern_descriptor(image, bbox=nearest_zone["bbox_px"])
        name = candidate["name"] or candidate["line_text"] or "Pattern"
        pattern_id = f"p{page_number:03d}-pat{len(patterns) + 1:03d}"
        patterns.append(
            {
                "id": pattern_id,
                "page": page_number,
                "name": name,
                "pattern": descriptor,
                "color": descriptor.get("mean_color"),
                "bbox": nearest_zone["bbox_px"],
                "zone_id": nearest_zone.get("zone_id"),
                "line_text": candidate["line_text"],
                "source": candidate["source"],
            }
        )
        used_zone_ids.add(str(nearest_zone.get("zone_id")))

    return patterns


def summarize_pattern_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for panel in panels:
        key = str(panel.get("pattern_id") or panel.get("pattern_name") or panel.get("material_name"))
        grouped[key].append(panel)

    summary: list[dict[str, Any]] = []
    for pattern_id, rows in sorted(grouped.items(), key=lambda item: (str(item[1][0].get("pattern_name") or "").lower(), item[0])):
        areas = [row["area_m2"] for row in rows if row.get("area_m2") is not None]
        bbox_areas = [row["bbox_area_m2"] for row in rows if row.get("bbox_area_m2") is not None]
        summary.append(
            {
                "pattern_id": pattern_id,
                "material_name": rows[0].get("pattern_name") or rows[0].get("material_name"),
                "panel_count": len(rows),
                "area_m2": round(sum(areas), 4) if areas else None,
                "bbox_area_m2": round(sum(bbox_areas), 4) if bbox_areas else None,
                "pages": sorted({row["page"] for row in rows}),
            }
        )
    return summary


def write_pattern_summary_csv(run_dir: Path, pattern_summary: list[dict[str, Any]]) -> Path:
    csv_path = run_dir / "pattern_summary.csv"
    fields = ["pattern_id", "material_name", "panel_count", "area_m2", "bbox_area_m2", "pages"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in pattern_summary:
            out = dict(row)
            out["pages"] = ",".join(str(page) for page in out.get("pages", []))
            writer.writerow(out)
    return csv_path


def save_pattern_debug_images(
    rendered_pages: list[dict[str, Any]],
    patterns_by_page: dict[int, list[dict[str, Any]]],
    matched_panels_by_page: dict[int, list[dict[str, Any]]],
    run_dir: Path,
    mentions: list[MaterialMention] | None = None,
) -> dict[str, Any]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    debug_dir = run_dir / "pattern_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    matched_area_dir = run_dir / "matched_areas"
    matched_area_dir.mkdir(parents=True, exist_ok=True)
    palette = [(255, 0, 0), (0, 180, 0), (0, 120, 255), (255, 165, 0), (128, 0, 128), (0, 180, 180)]

    saved_images: list[str] = []
    saved_areas: list[dict[str, Any]] = []
    saved_panels: list[dict[str, Any]] = []
    saved_patterns: list[dict[str, Any]] = []
    saved_mentions: list[dict[str, Any]] = []

    mentions_by_page: dict[int, list[MaterialMention]] = defaultdict(list)
    if mentions:
        for mention in mentions:
            mentions_by_page[mention.page].append(mention)

    for page in rendered_pages:
        page_number = page["page"]
        page_patterns = patterns_by_page.get(page_number, [])
        page_panels = matched_panels_by_page.get(page_number, [])
        if not page_patterns and not page_panels:
            continue

        height_px, width_px = page["image"].shape[:2]
        debug = np.array(page["image"], copy=True)
        thickness = max(2, min(width_px, height_px) // 500)
        pattern_colors = {pattern["id"]: palette[index % len(palette)] for index, pattern in enumerate(page_patterns)}

        for panel in page_panels:
            color = pattern_colors.get(panel.get("pattern_id"), (255, 0, 0))
            x0, y0, x1, y1 = panel["bbox_px"]
            cv2.rectangle(debug, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
            label = str(panel.get("pattern_id") or "").replace("p", "P").replace("-", " ")
            cv2.putText(debug, label[-4:], (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness + 1, cv2.LINE_AA)

            crop = debug[y0:y1, x0:x1].copy()
            cv2.rectangle(crop, (0, 0), (max(0, x1 - x0 - 1), max(0, y1 - y0 - 1)), color, max(2, thickness), cv2.LINE_AA)
            area_path = matched_area_dir / f"page_{page_number:03d}_{panel.get('zone_id') or 'area'}_{panel.get('pattern_id') or 'pattern'}.png"
            cv2.imwrite(str(area_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            saved_areas.append(
                {
                    "image": str(area_path),
                    "page": panel["page"],
                    "zone_id": panel.get("zone_id"),
                    "pattern_id": panel.get("pattern_id"),
                    "pattern_name": panel.get("pattern_name"),
                    "pattern_score": panel.get("pattern_score"),
                    "bbox": panel["bbox_px"],
                }
            )

        for index, pattern in enumerate(page_patterns, start=1):
            x0, y0, x1, y1 = pattern["bbox"]
            cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(debug, f"P{index}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), thickness + 1, cv2.LINE_AA)

        for mention_index, mention in enumerate(mentions_by_page.get(page_number, []), start=1):
            x0, y0, x1, y1 = [int(round(value)) for value in mention.bbox]
            x0 = max(0, min(x0, width_px - 1))
            y0 = max(0, min(y0, height_px - 1))
            x1 = max(0, min(x1, width_px - 1))
            y1 = max(0, min(y1, height_px - 1))
            color = (0, 0, 255)
            cv2.rectangle(debug, (x0, y0), (x1, y1), color, max(1, thickness - 1), cv2.LINE_AA)
            cv2.putText(debug, f"M{mention_index}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, max(1, thickness), cv2.LINE_AA)
            saved_mentions.append(
                {
                    "image": "",
                    "mention_index": f"M{mention_index}",
                    "page": page_number,
                    "material_name": mention.material_name,
                    "line_text": mention.line_text,
                    "bbox": [x0, y0, x1, y1],
                }
            )

        image_path = debug_dir / f"page_{page_number:03d}_patterns.png"
        for mention in saved_mentions:
            if mention.get("page") == page_number and not mention.get("image"):
                mention["image"] = str(image_path)
        cv2.imwrite(str(image_path), cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
        saved_images.append(str(image_path))

        for panel in page_panels:
            saved_panels.append(
                {
                    "image": str(image_path),
                    "page": panel["page"],
                    "zone_id": panel.get("zone_id"),
                    "pattern_id": panel.get("pattern_id"),
                    "pattern_name": panel.get("pattern_name"),
                    "pattern_score": panel.get("pattern_score"),
                    "bbox": panel["bbox_px"],
                }
            )

        for index, pattern in enumerate(page_patterns, start=1):
            saved_patterns.append(
                {
                    "image": str(image_path),
                    "pattern_index": f"P{index}",
                    "page": pattern["page"],
                    "pattern_id": pattern["id"],
                    "name": pattern["name"],
                    "bbox": pattern["bbox"],
                    "zone_id": pattern.get("zone_id"),
                    "pattern": pattern["pattern"],
                }
            )

    meta_path = debug_dir / "patterns.json"
    if saved_panels or saved_patterns:
        meta_path.write_text(
            json.dumps({"images": saved_images, "matched_areas": saved_areas, "patterns": saved_patterns, "matched_panels": saved_panels, "mention_bboxes": saved_mentions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved pattern debug images: %s", len(saved_images))
        return {"images": saved_images, "pattern_images": saved_images, "matched_areas": saved_areas, "patterns": saved_patterns, "matched_panels": saved_panels, "mention_bboxes": saved_mentions, "meta": str(meta_path)}

    return {"images": [], "pattern_images": [], "matched_areas": [], "patterns": [], "matched_panels": [], "mention_bboxes": [], "meta": None}



def log_analysis_result(result: dict[str, Any]) -> None:
    logger.info("========== NATURAL STONE RESULT ==========")
    for warning in result["warnings"]:
        logger.warning("Warning: %s", warning)
    if not result["summary"]:
        logger.info("No natural stone panels were calculated")
        return
    for item in result["summary"]:
        logger.info("Material: %s | panels=%s | area_m2=%s", item["material_name"], item["panel_count"], item["area_m2"])


def analyze_pdf_file(
    pdf_path: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ocr_backend: str = "auto",
    force_ocr: bool = False,
    fallback_mm_per_px: float | None = None,
    min_zone_area_px: int | None = None,
    save_csv: bool = True,
    tesseract_psm: int = 11,
) -> dict[str, Any]:
    import re
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting PDF analysis: file=%s", pdf_path)
    rendered_pages = render_pdf_pages(pdf_path, dpi=dpi)
    words_by_page = collect_ocr_words(pdf_path, rendered_pages, backend=ocr_backend, force_ocr=force_ocr, tesseract_psm=tesseract_psm)
    mentions = extract_material_mentions(words_by_page)
    mentions_debug = save_mention_debug_images(rendered_pages, mentions, run_dir)

    # warnings: list[str] = []
    # if not mentions:
    #     warnings.append("No natural-stone material keywords found in OCR/PDF text")

    # scale_by_page, scale_warnings = build_scale_map(rendered_pages, words_by_page, fallback_mm_per_px)
    # warnings.extend(scale_warnings)

    # mentions_by_page: dict[int, list[MaterialMention]] = defaultdict(list)
    # for mention in mentions:
    #     mentions_by_page[mention.page].append(mention)

    # panels: list[dict[str, Any]] = []
    # patterns_by_page: dict[int, list[dict[str, Any]]] = {}
    # matched_panels_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    # for page in rendered_pages:
    #     page_number = page["page"]
    #     page_mentions = mentions_by_page.get(page_number, [])
    #     page_words = words_by_page.get(page_number, [])

    #     scale_info = scale_by_page[page_number]
    #     all_zones = detect_material_zones(page["image"], page_number, min_zone_area_px=min_zone_area_px, ignore_bboxes=[word.bbox for word in page_words])
    #     logger.info("Page %s OpenCV zones: %s", page_number, len(all_zones))

    #     page_patterns = extract_pattern_legends_for_page(page["image"], page_number, page_mentions, all_zones, page["width_px"], page["height_px"], page_words)
    #     patterns_by_page[page_number] = page_patterns
    #     if not page_patterns:
    #         continue

    #     matched_zones = match_zones_to_patterns(all_zones, page_patterns, page["image"])
    #     matched_panels_by_page[page_number] = []
    #     page_segments = detect_line_segments(page["image"])
    #     global_mm_per_px = scale_info.get("mm_per_px")

    #     for zone in matched_zones:
    #         pattern = next((item for item in page_patterns if item.get("id") == zone.get("pattern_id")), None)
    #         if pattern is None:
    #             continue
    #         zone_mm_per_px, zone_dims = estimate_zone_scale(zone, page_words, page_segments, page["width_px"], page["height_px"], fallback_mm_per_px=global_mm_per_px)
    #         zone_scale_source = "zone_dimension_line" if zone_mm_per_px != global_mm_per_px and zone_mm_per_px else scale_info.get("source", "missing")
    #         panel = add_metric_fields(zone, mm_per_px=zone_mm_per_px, scale_source=str(zone_scale_source))
    #         panel.update(
    #             {
    #                 "material_name": pattern["name"],
    #                 "material_keyword": pattern["name"],
    #                 "material_line": pattern.get("line_text"),
    #                 "material_source": "pattern_legend",
    #                 "pattern_id": pattern["id"],
    #                 "pattern_name": pattern["name"],
    #                 "pattern_score": zone.get("pattern_score"),
    #                 "zone_dimensions": zone_dims,
    #             }
    #         )
    #         panels.append(panel)
    #         matched_panels_by_page[page_number].append(panel)

    # pattern_summary = summarize_pattern_panels(panels)
    # summary = summarize_panels(panels)
    # write_csv(run_dir, panels, summary)
    # write_pattern_summary_csv(run_dir, pattern_summary)
    # pattern_debug = save_pattern_debug_images(rendered_pages, patterns_by_page, matched_panels_by_page, run_dir, mentions=mentions)

    # return {
    #     "run_id": run_id,
    #     "file_name": pdf_path.name,
    #     "pages": len(rendered_pages),
    #     "warnings": warnings,
    #     "mentions_debug": mentions_debug,
    #     "pattern_debug": pattern_debug,
    #     "combined_debug": pattern_debug,
    #     "pattern_images": pattern_debug.get("pattern_images", []),
    #     "matched_areas": pattern_debug.get("matched_areas", []),
    #     "pattern_summary": pattern_summary,
    #     "patterns": [{"page": page, "patterns": patterns} for page, patterns in sorted(patterns_by_page.items())],
    #     "summary": summary,
    #     "panels": panels,
    # }


# FastAPI app (optional)
def create_app():
    from fastapi import FastAPI, File, UploadFile, HTTPException, Query

    api = FastAPI(title=APP_NAME)

    @api.get("/health")
    def health():
        from pipestone_ocr import has_module
        return {"status": "ok", "dependencies": {
            "pymupdf": has_module("fitz"),
            "numpy": has_module("numpy"),
            "cv2": has_module("cv2"),
            "pytesseract": has_module("pytesseract"),
        }}

    @api.post("/analyze-pdf")
    async def analyze(pdf: UploadFile = File(...), ocr_backend: str = "auto", fallback_mm_per_px: float = None):
        import tempfile, shutil
        temp_dir = Path(tempfile.mkdtemp(prefix="pipestone_"))
        try:
            temp_path = temp_dir / "upload.pdf"
            content = await pdf.read()
            temp_path.write_bytes(content)
            return analyze_pdf_file(str(temp_path), ocr_backend=ocr_backend, fallback_mm_per_px=fallback_mm_per_px)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return api


app = None
try:
    app = create_app()
except Exception:
    pass