#!/usr/bin/env python3
"""Material text search utilities for PipeStone."""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from pipestone_ocr import OcrWord, OcrLine, words_to_lines, bbox_union
from pipestone_semantic import (
    KNOWN_STONE_TYPES,
    STONE_KEYWORD_RE,
    STONE_SEMANTIC_THRESHOLD,
    semantic_best_stone_type,
)

logger = logging.getLogger("pipestone.material_search")

def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None

def require_module(module_name: str, install_hint: str) -> Any:
    if not has_module(module_name):
        raise RuntimeError(f"Не найден модуль {module_name}. Установите: {install_hint}")
    return __import__(module_name)

@dataclass(frozen=True)
class MaterialMention:
    page: int
    material_name: str
    line_text: str
    keyword: str
    bbox: tuple[float, float, float, float]
    confidence: float | None
    source: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("ё", "е").replace("Ё", "Е")).strip().lower()


def cleanup_material_label(label: str) -> str:
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


def guess_material_name_by_regexp(line_text: str) -> str:
    cleaned = line_text.strip(" :-\t")

    for stone_type in KNOWN_STONE_TYPES:
        match = re.search(rf"\b{stone_type}\b", normalize_text(line_text), re.IGNORECASE)
        if match:
            start = match.start()
            name = cleanup_material_label(cleaned[start:])
            logger.info("Material guess via regex: type=%s name=%r", stone_type, name)
            return name

    logger.info("Material guess default")
    return "Натуральный камень"


def extract_material_mentions(words_by_page: dict[int, list[OcrWord]]) -> list[MaterialMention]:
    mentions: list[MaterialMention] = []
    seen: set[tuple[int, str, tuple[int, int, int, int]]] = set()

    for page, words in words_by_page.items():
        page_lines = words_to_lines(words)
        for line in page_lines:
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

            material_name = guess_material_name_by_regexp(line.text)

            keyword_bbox = line.bbox
            if keyword:
                normalized_line = normalize_text(line.text)
                keyword_match = re.search(re.escape(keyword), normalized_line, re.IGNORECASE)
                if keyword_match:
                    keyword_start_idx = keyword_match.start()
                    normalized_prefix = normalized_line[:keyword_start_idx]
                    n_words_before = len(normalized_prefix.split()) if normalized_prefix.strip() else 0

                    line_words = [w for w in words if w.page == page]
                    line_words_sorted = sorted(line_words, key=lambda w: w.bbox[0])
                    if len(line_words_sorted) > n_words_before:
                        keyword_words = line_words_sorted[n_words_before:]
                        if keyword_words:
                            keyword_bbox = bbox_union([w.bbox for w in keyword_words])

            key = (
                page,
                normalize_text(material_name),
                tuple(int(round(value / 10.0)) for value in keyword_bbox),
            )
            if key in seen:
                continue
            seen.add(key)
            mention = MaterialMention(
                page=page,
                material_name=material_name,
                line_text=line.text,
                keyword=keyword,
                bbox=keyword_bbox,
                confidence=line.confidence,
                source=line.source,
            )
            mentions.append(mention)
            logger.info(
                "Material mention: page=%s material=%r line_text=%r keyword=%r",
                page,
                material_name,
                line.text,
                keyword,
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
        zx = (float(zone["bbox_px"][0]) + float(zone["bbox_px"][2])) / 2.0
        zy = (float(zone["bbox_px"][1]) + float(zone["bbox_px"][3])) / 2.0
        return min(
            page_mentions,
            key=lambda mention: (
                (zx - (mention.bbox[0] + mention.bbox[2]) / 2.0) ** 2
                + (zy - (mention.bbox[1] + mention.bbox[3]) / 2.0) ** 2
            ) ** 0.5,
        )

    unique = dedupe_material_names(all_mentions)
    if len(unique) == 1:
        return all_mentions[0]
    return None


def log_material_mentions(mentions: list[MaterialMention]) -> None:
    logger.info("========== MATERIAL MENTIONS ==========")
    if not mentions:
        logger.info("No material mentions found")
        return
    mentions_by_page: dict[int, list[MaterialMention]] = {}
    for mention in mentions:
        mentions_by_page.setdefault(mention.page, []).append(mention)
    for page in sorted(mentions_by_page.keys()):
        logger.info("Page %s:", page)
        for idx, mention in enumerate(mentions_by_page[page], start=1):
            logger.info(
                "  M%s: material=%r keyword=%r line_text=%r bbox=(%.1f,%.1f,%.1f,%.1f)",
                idx,
                mention.material_name,
                mention.keyword,
                mention.line_text,
                mention.bbox[0],
                mention.bbox[1],
                mention.bbox[2],
                mention.bbox[3],
            )


def save_material_mention_images(
    mentions: list[MaterialMention],
    rendered_pages: list[dict[str, Any]],
    run_dir: Path,
) -> list[str]:
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    np = require_module("numpy", "pip install numpy")

    image_dir = run_dir / "material_mentions"
    image_dir.mkdir(parents=True, exist_ok=True)

    saved_images: list[str] = []
    mentions_by_page: dict[int, list[MaterialMention]] = {}
    for mention in mentions:
        mentions_by_page.setdefault(mention.page, []).append(mention)

    for page in rendered_pages:
        page_number = page["page"]
        page_mentions = mentions_by_page.get(page_number, [])
        if not page_mentions:
            continue

        height_px, width_px = page["image"].shape[:2]
        debug = np.array(page["image"], copy=True)
        thickness = max(2, min(width_px, height_px) // 500)

        for idx, mention in enumerate(page_mentions, start=1):
            x0, y0, x1, y1 = [int(round(value)) for value in mention.bbox]
            x0 = max(0, min(x0, width_px - 1))
            y0 = max(0, min(y0, height_px - 1))
            x1 = max(0, min(x1, width_px - 1))
            y1 = max(0, min(y1, height_px - 1))
            color = (0, 0, 255)
            cv2.rectangle(debug, (x0, y0), (x1, y1), color, max(1, thickness - 1), cv2.LINE_AA)
            cv2.putText(debug, f"M{idx}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, max(1, thickness), cv2.LINE_AA)

        image_path = image_dir / f"page_{page_number:03d}_mentions.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
        saved_images.append(str(image_path))
        logger.info("Saved material mention image: %s", image_path)

    return saved_images
