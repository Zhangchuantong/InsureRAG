# -*- coding: utf-8 -*-
"""PDF text quality checks used before ingestion."""

from __future__ import annotations

import re
from typing import Any


VALID_CHAR_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9"
    r"，。！？；：、“”‘’（）《》【】—\-_,.!?;:'\"()\[\]{}<>/%‰+]"
)


def calc_pdf_text_quality(
    page_texts: list[str],
    sample_pages: int = 3,
    good_threshold: float = 0.8,
    reject_threshold: float = 0.6,
    min_text_density: int = 100,
) -> dict[str, Any]:
    """Calculate effective character ratio and text density for sampled pages."""
    sampled_pages = max(1, min(len(page_texts), sample_pages)) if page_texts else 1
    sample_text = "\n".join(page_texts[:sample_pages])
    non_space_chars = [ch for ch in sample_text if not ch.isspace()]
    total_chars = len(non_space_chars)
    valid_chars = [ch for ch in non_space_chars if VALID_CHAR_PATTERN.match(ch)]

    effective_rate = len(valid_chars) / total_chars if total_chars else 0.0
    text_density = len(valid_chars) / sampled_pages
    low_density = text_density < min_text_density

    if effective_rate < reject_threshold or low_density:
        level = "reject"
    elif effective_rate >= good_threshold:
        level = "good"
    else:
        level = "warning"

    return {
        "level": level,
        "effective_rate": effective_rate,
        "quality_score": effective_rate,
        "text_density": text_density,
        "low_density": low_density,
        "sampled_pages": sampled_pages,
        "valid_chars": len(valid_chars),
        "total_chars": total_chars,
    }


def should_use_ocr_fallback(
    quality: dict[str, Any],
    image_pdf: dict[str, Any] | None = None,
    ocr_enabled: bool = True,
) -> bool:
    """Return True when OCR fallback should be attempted."""
    if not ocr_enabled or quality.get("level") != "reject":
        return False
    if image_pdf is None:
        return True
    return bool(image_pdf.get("is_image_pdf"))
