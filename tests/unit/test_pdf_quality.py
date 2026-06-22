# -*- coding: utf-8 -*-

from parsers.quality_checker import calc_pdf_text_quality, should_use_ocr_fallback


def test_effective_character_rate_good_text():
    quality = calc_pdf_text_quality(
        ["第一条 保险责任。等待期为90天。"] * 3,
        sample_pages=3,
        min_text_density=5,
    )

    assert quality["level"] == "good"
    assert quality["effective_rate"] >= 0.8


def test_text_density_rejects_sparse_text():
    quality = calc_pdf_text_quality(
        ["险"] * 3,
        sample_pages=3,
        min_text_density=100,
    )

    assert quality["level"] == "reject"
    assert quality["low_density"] is True


def test_ocr_fallback_trigger_condition():
    quality = {"level": "reject"}
    image_pdf = {"is_image_pdf": True}

    assert should_use_ocr_fallback(quality, image_pdf=image_pdf, ocr_enabled=True) is True
    assert should_use_ocr_fallback(quality, image_pdf=image_pdf, ocr_enabled=False) is False
    assert should_use_ocr_fallback({"level": "good"}, image_pdf=image_pdf, ocr_enabled=True) is False
