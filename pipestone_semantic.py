#!/usr/bin/env python3
"""Semantic search for stone type detection."""

from __future__ import annotations

import importlib.util
import logging
import re
from typing import Any

logger = logging.getLogger("pipestone.semantic")

STONE_SEMANTIC_THRESHOLD = 0.35
_E5_MODEL: Any = None
_SEMANTIC_MODEL: Any = None

KNOWN_STONE_TYPES = (
    "гранит",
    "мрамор",
    "травертин",
    "известняк",
    "лабрадорит",
    "габбро",
    "базальт",
    "кварцит",
    "сланец",
    "оникс",
    "доломит",
    "песчаник",
    "ракушечник",
    "серпентинит",
    "натуральный камень",
)

STONE_KEYWORD_RE = re.compile(
    r"("
    r"кам(?:е|ё)?н[ьяеиюом]*\s+натурал\w*|"
    r"натурал\w*\s+кам(?:е|ё)?н[ьяеиюом]*|"
    r"облицовк\w*\s+натурал\w*\s+кам(?:е|ё)?н[ьяеиюом]*|"
    r"кам(?:е|ё)?н[ьяеиюом]*\s*30\s*мм|"
    r"издел\w*\s+из\s+натурал\w*\s+кам(?:е|ё)?н[ьяеиюом]*"
    r")",
    re.IGNORECASE,
)


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _load_semantic_model() -> Any | None:
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is not None:
        return _SEMANTIC_MODEL
    if not has_module("sentence_transformers"):
        logger.warning(
            "Semantic search unavailable: install sentence-transformers "
            "(pip install sentence-transformers) and the model will be downloaded from HF Hub automatically."
        )
        return None
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Downloading/loading semantic model: all-MiniLM-L6-v2 from HF Hub")
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Semantic model ready: all-MiniLM-L6-v2")
        return _SEMANTIC_MODEL
    except Exception as exc:
        logger.warning("Semantic model download failed: %s", exc)
        return None


def semantic_best_stone_type(text: str, threshold: float = STONE_SEMANTIC_THRESHOLD) -> tuple[str | None, float]:
    model = _load_semantic_model()
    if model is None or not has_module("numpy"):
        return None, 0.0
    try:
        import numpy as np

        corpus = list(KNOWN_STONE_TYPES)
        text_emb = model.encode(text, normalize_embeddings=True)
        corpus_emb = model.encode(corpus, normalize_embeddings=True)
        sims = (corpus_emb @ text_emb).tolist()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_type = corpus[best_idx]
        logger.debug(
            "Semantic search for %r: best=%s score=%.4f threshold=%.2f",
            text,
            best_type,
            best_score,
            threshold,
        )
        if best_score >= threshold:
            return best_type, best_score
        return None, best_score
    except Exception as exc:
        logger.debug("Semantic match failed for %r: %s", text, exc)
        return None, 0.0