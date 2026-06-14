#!/usr/bin/env python3
"""Material text search utilities for PipeStone."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from pipestone_ocr import OcrWord, words_to_lines
from pipestone_semantic import (
    KNOWN_STONE_TYPES,
    STONE_KEYWORD_RE,
    STONE_SEMANTIC_THRESHOLD,
    semantic_best_stone_type,
)

logger = logging.getLogger("pipestone.material_search")


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


def guess_material_name(line_text: str) -> str:
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
