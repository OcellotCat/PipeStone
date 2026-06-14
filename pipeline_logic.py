#!/usr/bin/env python3
"""Main analysis pipeline for PipeStone - facade layout PDF processing."""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import re
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from pipestone_cv import (
    bbox_center,
    detect_line_segments,
    extract_pattern_descriptor,
    require_module,
)
from pipestone_ocr import (
    OcrWord,
    bbox_union,
    collect_ocr_words,
    render_pdf_pages,
    words_to_lines,
)
from pipeline_material_search import (
    MaterialMention,
    extract_material_mentions,
    guess_material_name_by_regexp,
    normalize_text,
)

logger = logging.getLogger("pipestone")

APP_NAME = "PipeStone PDF Stone Area MVP"
DEFAULT_DPI = 400
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


def normalize_match_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^0-9a-zа-яё]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_tokens(text: str) -> set[str]:
    return {token for token in normalize_match_text(text).split() if len(token) > 1}


def material_text_match_score(candidate_text: str, material_name: str) -> float:
    candidate = normalize_match_text(candidate_text)
    material = normalize_match_text(material_name)
    if not material:
        return 1.0
    if material in candidate or candidate in material:
        return 0.0
    tokens = text_tokens(material)
    if not tokens:
        return 1.0
    matched = sum(1 for token in tokens if token in candidate)
    return max(0.0, 1.0 - matched / len(tokens))


def legend_title_score(text: str) -> float:
    normalized = normalize_match_text(text)
    if not normalized:
        return 1.0
    if "условн" in normalized and "обознач" in normalized:
        return 0.0
    if "экспликац" in normalized:
        return 0.0
    if "обознач" in normalized and any(token in normalized for token in ("материал", "отдел", "облицов")):
        return 0.15
    if "легенд" in normalized:
        return 0.15
    return 1.0


def bbox_contains_center(bbox: tuple[float, float, float, float], center: tuple[float, float]) -> bool:
    return bbox[0] <= center[0] <= bbox[2] and bbox[1] <= center[1] <= bbox[3]


def center_distance_to_bbox(center: tuple[float, float], bbox: tuple[float, float, float, float]) -> float:
    dx = max(bbox[0] - center[0], 0.0, center[0] - bbox[2])
    dy = max(bbox[1] - center[1], 0.0, center[1] - bbox[3])
    return (dx * dx + dy * dy) ** 0.5


def find_legend_block(lines: list[Any], page_width_px: int, page_height_px: int) -> tuple[float, float, float, float] | None:
    title_lines = [line for line in lines if legend_title_score(line.text) < 0.5]
    if not title_lines:
        return None

    title = sorted(title_lines, key=lambda item: bbox_center(item.bbox)[1])[0]
    title_center = bbox_center(title.bbox)
    page_scale = min(page_width_px, page_height_px)
    expanded_bbox = (
        max(0.0, title.bbox[0] - page_scale * 0.18),
        max(0.0, title.bbox[1] - page_scale * 0.35),
        min(float(page_width_px), title.bbox[2] + page_scale * 0.18),
        min(float(page_height_px), title.bbox[3] + page_scale * 0.35),
    )
    block_lines = [line for line in lines if bbox_contains_center(expanded_bbox, bbox_center(line.bbox))]
    if len(block_lines) < 2:
        return None
    return bbox_union([line.bbox for line in block_lines])


def legend_zones(
    all_zones: list[dict[str, Any]],
    max_pattern_area: float,
    legend_bbox: tuple[float, float, float, float] | None,
    page_width_px: int,
    page_height_px: int,
) -> list[dict[str, Any]]:
    zones = [zone for zone in all_zones if float(zone.get("bbox_area_px", 0)) <= max_pattern_area]
    if legend_bbox is None:
        return sorted(zones, key=lambda item: (item["bbox_px"][1], item["bbox_px"][0]))

    page_scale = min(page_width_px, page_height_px)
    pad = max(50.0, page_scale * 0.04)
    expanded_bbox = (
        max(0.0, legend_bbox[0] - pad),
        max(0.0, legend_bbox[1] - pad),
        min(float(page_width_px), legend_bbox[2] + pad),
        min(float(page_height_px), legend_bbox[3] + pad),
    )
    return [
        zone
        for zone in zones
        if bbox_contains_center(expanded_bbox, bbox_center(tuple(float(value) for value in zone["bbox_px"])))
    ]


def candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(str(candidate.get(key) or "") for key in ("name", "line_text", "keyword"))


def match_candidate_to_zone(
    candidate: dict[str, Any],
    zone: dict[str, Any],
    max_pattern_area: float,
    page_width_px: int,
    page_height_px: int,
) -> float:
    text = candidate_text(candidate)
    material_name = str(candidate.get("name") or candidate.get("line_text") or "")
    text_score = material_text_match_score(text, material_name)
    keyword = candidate.get("keyword")
    if keyword:
        text_score = min(text_score, material_text_match_score(text, str(keyword)))

    candidate_center = bbox_center(candidate["bbox"])
    zone_bbox = tuple(float(value) for value in zone["bbox_px"])
    distance_score = center_distance_to_bbox(candidate_center, zone_bbox) / max(min(page_width_px, page_height_px), 1.0)
    area_score = float(zone.get("bbox_area_px", 0)) / max(max_pattern_area, 1.0)

    if text_score <= 0.55:
        return text_score * 0.72 + distance_score * 0.20 + min(area_score, 1.0) * 0.08
    return 0.72 + text_score * 0.18 + distance_score * 0.07 + min(area_score, 1.0) * 0.03


def match_patterns_to_mentions(
    patterns: list[dict[str, Any]],
    mentions: list[MaterialMention],
) -> tuple[dict[int, dict[str, Any]], set[str]]:
    matches_by_mention_index: dict[int, dict[str, Any]] = {}
    matched_pattern_ids: set[str] = set()
    for mention_index, mention in enumerate(mentions, start=1):
        best_pattern = None
        best_score = float("inf")
        for pattern in patterns:
            pattern_text = " ".join(str(pattern.get(key) or "") for key in ("name", "line_text"))
            score = min(
                material_text_match_score(pattern_text, mention.material_name),
                material_text_match_score(pattern_text, mention.line_text),
                material_text_match_score(pattern_text, mention.keyword or "") if mention.keyword else 1.0,
            )
            if score < best_score:
                best_score = score
                best_pattern = pattern
        if best_pattern is not None and best_score <= 0.55:
            matches_by_mention_index[mention_index] = {"pattern": best_pattern, "score": best_score}
            matched_pattern_ids.add(best_pattern["id"])
    return matches_by_mention_index, matched_pattern_ids

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
    lines = words_to_lines(words)
    legend_bbox = find_legend_block(lines, page_width_px, page_height_px)
    page_legend_zones = legend_zones(all_zones, max_pattern_area, legend_bbox, page_width_px, page_height_px)

    candidates: list[dict[str, Any]] = []
    for mention in page_mentions:
        candidates.append(
            {
                "name": mention.material_name,
                "line_text": mention.line_text,
                "keyword": mention.keyword,
                "bbox": mention.bbox,
                "source": "material_mention",
            }
        )

    if not candidates:
        for line in lines:
            name = guess_material_name_by_regexp(line.text) or line.text
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
        best_zone = None
        best_score = float("inf")
        for zone in page_legend_zones:
            if str(zone.get("zone_id")) in used_zone_ids:
                continue
            score = match_candidate_to_zone(candidate, zone, max_pattern_area, page_width_px, page_height_px)
            if score < best_score:
                best_score = score
                best_zone = zone

        max_match_score = 0.72 if legend_bbox is not None else 1.0
        if best_zone is None or best_score > max_match_score:
            continue

        descriptor = extract_pattern_descriptor(image, bbox=best_zone["bbox_px"])
        name = candidate["name"] or candidate["line_text"] or "Pattern"
        pattern_id = f"p{page_number:03d}-pat{len(patterns) + 1:03d}"
        patterns.append(
            {
                "id": pattern_id,
                "page": page_number,
                "name": name,
                "pattern": descriptor,
                "color": descriptor.get("mean_color"),
                "bbox": best_zone["bbox_px"],
                "zone_id": best_zone.get("zone_id"),
                "line_text": candidate["line_text"],
                "source": candidate["source"],
                "legend_bbox": list(legend_bbox) if legend_bbox else None,
                "match_score": round(best_score, 4),
            }
        )
        used_zone_ids.add(str(best_zone.get("zone_id")))

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



def save_rendered_pages(rendered_pages: list[dict[str, Any]], repo_root: Path, dpi: int) -> list[str]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    saved: list[str] = []
    for page in rendered_pages:
        image = page["image"]
        page_number = page["page"]
        image_path = repo_root / f"page_{page_number:03d}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        saved.append(str(image_path))
    if saved:
        logger.info("Saved rendered pages to repo root: %s images at %s DPI", len(saved), dpi)
    return saved


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
    tesseract_psm: int = 11,
    save_rendered_pages: bool = False,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting PDF analysis: file=%s", pdf_path)
    rendered_pages = render_pdf_pages(pdf_path, dpi=dpi)
    words_by_page = collect_ocr_words(pdf_path, rendered_pages, backend=ocr_backend, force_ocr=force_ocr, tesseract_psm=tesseract_psm)

    rendered_page_paths: list[str] = []
    if save_rendered_pages:
        rendered_page_paths = save_rendered_pages(rendered_pages, Path.cwd(), dpi)
    mentions = extract_material_mentions(words_by_page)