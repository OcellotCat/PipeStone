#!/usr/bin/env python3
"""Main analysis pipeline for PipeStone - facade layout PDF processing."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from pipestone_cv import (
    detect_line_segments,
    detect_material_zones,
    estimate_zone_scale,
    filter_zones_by_hatch_pattern,
)
from pipestone_ocr import (
    OcrWord,
    collect_ocr_words,
    render_pdf_pages,
    summarize_panels,
    words_to_lines,
)
from pipestone_semantic import (
    KNOWN_STONE_TYPES,
    STONE_KEYWORD_RE,
    STONE_SEMANTIC_THRESHOLD,
    semantic_best_stone_type,
)

logger = logging.getLogger("pipestone")

APP_NAME = "PipeStone PDF Stone Area MVP"
DEFAULT_DPI = 220
DEFAULT_OUTPUT_DIR = "output"


@dataclass(frozen=True)
class MaterialMention:
    page: int
    material_name: str
    line_text: str
    keyword: str
    bbox: tuple[float, float, float, float]
    confidence: float | None
    source: str


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_text(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.replace("ё", "е").replace("Ё", "Е")).strip().lower()


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def cleanup_material_label(label: str) -> str:
    import re
    label = re.sub(
        r"\b(облицовка|изделия|из|камень|камня|камнем|натуральный|натурального|"
        r"натуральным|толщина|толщ\.?)\b",
        " ",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:мм|mm|м|m)\b", " ", label, flags=re.IGNORECASE)
    label = re.sub(r"[^0-9A-Za-zА-Яа-яЁё.\- ]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip(" .:-")
    if not label:
        return "Натуральный камень"
    return label[:120]


def guess_material_name(line_text: str) -> str:
    import re
    cleaned = line_text.strip(" :-\t")

    for stone_type in KNOWN_STONE_TYPES:
        match = re.search(rf"\b{stone_type}\b", normalize_text(line_text), re.IGNORECASE)
        if match:
            start = match.start()
            name = cleanup_material_label(cleaned[start:])
            logger.info("Material guess via regex: type=%s name=%r", stone_type, name)
            return name

    name = _guess_with_semantic_fallback(cleaned)
    if name:
        logger.info("Material guess via semantic search: name=%r", name)
        return name

    after_colon = re.split(r"[:;-]", cleaned, maxsplit=1)
    if len(after_colon) == 2:
        candidate = cleanup_material_label(after_colon[1])
        if candidate and normalize_text(candidate) not in {"30 мм", "мм"}:
            logger.info("Material guess via colon fallback: candidate=%r", candidate)
            return candidate

    candidate = cleanup_material_label(cleaned)
    if 8 <= len(candidate) <= 12 * len(candidate.split()):
        logger.info("Material guess via cleaned fallback: candidate=%r", candidate)
        return candidate

    logger.info("Material guess default")
    return "Натуральный камень"


def _guess_with_semantic_fallback(text: str) -> str | None:
    stone_type, score = semantic_best_stone_type(text)
    if stone_type is None:
        return None
    match = re.search(rf"\b{stone_type}\b", text, re.IGNORECASE)
    if match:
        return cleanup_material_label(text[match.start():])
    return stone_type


def extract_material_mentions(words_by_page: dict[int, list[OcrWord]]) -> list[MaterialMention]:
    import re
    mentions: list[MaterialMention] = []
    seen: set[tuple[int, str, tuple[int, int, int, int]]] = set()

    for page, words in words_by_page.items():
        for line in words_to_lines(words):
            normalized = normalize_text(line.text)
            match = STONE_KEYWORD_RE.search(normalized)
            keyword = match.group(0) if match else None
            if not match:
                stone_type, score = semantic_best_stone_type(normalized)
                if stone_type is None or score < STONE_SEMANTIC_THRESHOLD:
                    continue
                match = re.search(rf"\b{stone_type}\b", normalized, re.IGNORECASE)
                if not match:
                    continue
                keyword = match.group(0)

            material_name = guess_material_name(line.text)
            key = (
                page,
                normalize_text(material_name),
                tuple(int(round(value / 10.0)) for value in line.bbox),
            )
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                MaterialMention(
                    page=page,
                    material_name=material_name,
                    line_text=line.text,
                    keyword=keyword,
                    bbox=line.bbox,
                    confidence=line.confidence,
                    source=line.source,
                )
            )

    if mentions:
        logger.info("Material extraction complete: found %s mentions", len(mentions))
    return mentions


def dedupe_material_names(mentions: list[MaterialMention]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for mention in mentions:
        key = normalize_text(mention.material_name)
        if key in seen:
            continue
        seen.add(key)
        names.append(mention.material_name)
    return names


def assign_material(zone: dict[str, Any], page_mentions: list[MaterialMention], all_mentions: list[MaterialMention]) -> MaterialMention | None:
    if page_mentions:
        if len(page_mentions) == 1:
            return page_mentions[0]
        zx, zy = bbox_center(tuple(float(value) for value in zone["bbox_px"]))
        return min(page_mentions, key=lambda m: ((zx - bbox_center(m.bbox)[0]) ** 2 + (zy - bbox_center(m.bbox)[1]) ** 2) ** 0.5)

    unique = dedupe_material_names(all_mentions)
    if len(unique) == 1:
        return all_mentions[0]
    return None


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

    warnings: list[str] = []
    if not mentions:
        warnings.append("No natural-stone material keywords found in OCR/PDF text")

    scale_by_page, scale_warnings = build_scale_map(rendered_pages, words_by_page, fallback_mm_per_px)
    warnings.extend(scale_warnings)

    mentions_by_page: dict[int, list[MaterialMention]] = defaultdict(list)
    for mention in mentions:
        mentions_by_page[mention.page].append(mention)

    single_global_material = len(dedupe_material_names(mentions)) == 1
    panels: list[dict[str, Any]] = []

    for page in rendered_pages:
        page_mentions = mentions_by_page.get(page["page"], [])
        if mentions and not page_mentions and not single_global_material:
            continue
        if not mentions:
            continue

        scale_info = scale_by_page[page["page"]]
        all_zones = detect_material_zones(page["image"], page["page"], min_zone_area_px=min_zone_area_px, ignore_bboxes=[word.bbox for word in words_by_page.get(page["page"], [])])
        logger.info("Page %s OpenCV zones: %s", page["page"], len(all_zones))

        pattern_zones = []
        for zone in all_zones:
            zone_box = tuple(float(v) for v in zone["bbox_px"])
            for mention in page_mentions:
                mid_x, mid_y = (zone_box[0] + zone_box[2]) / 2, (zone_box[1] + zone_box[3]) / 2
                m_cx, m_cy = (mention.bbox[0] + mention.bbox[2]) / 2, (mention.bbox[1] + mention.bbox[3]) / 2
                if ((mid_x - m_cx) ** 2 + (mid_y - m_cy) ** 2) ** 0.5 < min(page["width_px"], page["height_px"]) * 0.1:
                    pattern_zones.append(zone)
                    break

        zones = filter_zones_by_hatch_pattern(all_zones, pattern_zones)
        page_segments = detect_line_segments(page["image"])
        global_mm_per_px = scale_info.get("mm_per_px")

        for zone in zones:
            zone_mm_per_px, zone_dims = estimate_zone_scale(zone, words_by_page.get(page["page"], []), page_segments, page["width_px"], page["height_px"], fallback_mm_per_px=global_mm_per_px)
            zone_scale_source = "zone_dimension_line" if zone_mm_per_px != global_mm_per_px and zone_mm_per_px else scale_info.get("source", "missing")
            mention = assign_material(zone, page_mentions, mentions)
            material_name = mention.material_name if mention else "Натуральный камень (не привязан к легенде)"
            panel = add_metric_fields(zone, mm_per_px=zone_mm_per_px, scale_source=str(zone_scale_source))
            panel.update({
                "material_name": material_name,
                "material_keyword": mention.keyword if mention else None,
                "material_line": mention.line_text if mention else None,
                "material_source": mention.source if mention else None,
                "zone_dimensions": zone_dims,
            })
            panels.append(panel)

    summary = summarize_panels(panels)
    write_csv(run_dir, panels, summary)

    return {
        "run_id": run_id,
        "file_name": pdf_path.name,
        "pages": len(rendered_pages),
        "warnings": warnings,
        "summary": summary,
        "panels": panels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ocr-backend", default="auto")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--fallback-mm-per-px", type=float, default=None)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    setup_logging()
    result = analyze_pdf_file(args.pdf, ocr_backend=args.ocr_backend, fallback_mm_per_px=args.fallback_mm_per_px, force_ocr=args.force_ocr)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


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