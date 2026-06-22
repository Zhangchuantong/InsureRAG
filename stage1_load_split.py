# -*- coding: utf-8 -*-
"""Stage 1: load PDF files, run quality checks, and build parent-child chunks."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from chunking.parent_child_chunker import build_chunks, default_document_id
from config.settings import settings
from observability.error_codes import OCR_ERROR, PDF_PARSE_ERROR
from observability.ingestion_trace import IngestionTrace, save_ingestion_trace, update_ingestion_trace
from parsers.quality_checker import calc_pdf_text_quality, should_use_ocr_fallback

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = DATA_DIR / "chunks.json"

QUALITY_SAMPLE_PAGES = int(settings.ocr["sample_pages"])
QUALITY_GOOD_RATE = float(settings.ocr["good_threshold"])
QUALITY_REJECT_RATE = float(settings.ocr["fallback_threshold"])
QUALITY_MIN_DENSITY = int(settings.ocr["min_text_density"])
IMAGE_PDF_MIN_IMAGE_PAGE_RATIO = float(settings.ocr["image_page_ratio_threshold"])
OCR_ENABLED = bool(settings.ocr["enabled"])
COLLECTION = settings.vector_db["collection"]


def _quality(page_texts: list[str]) -> dict[str, Any]:
    return calc_pdf_text_quality(
        page_texts,
        sample_pages=QUALITY_SAMPLE_PAGES,
        good_threshold=QUALITY_GOOD_RATE,
        reject_threshold=QUALITY_REJECT_RATE,
        min_text_density=QUALITY_MIN_DENSITY,
    )


def print_pdf_quality_report(fname: str, quality: dict[str, Any]) -> None:
    rate = float(quality["effective_rate"])
    density = float(quality["text_density"])
    sampled_pages = int(quality["sampled_pages"])
    level = quality["level"]
    print(
        f"  PDF 质量预检：{fname} level={level}, "
        f"有效字符率={rate:.2%}, 文本密度={density:.1f} 字/页, 抽样页数={sampled_pages}"
    )


def detect_image_pdf(path: str | Path, sample_pages: int = QUALITY_SAMPLE_PAGES) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    total_pages = min(len(reader.pages), sample_pages)
    if total_pages == 0:
        return {
            "is_image_pdf": False,
            "image_page_ratio": 0.0,
            "image_pages": 0,
            "sampled_pages": 0,
            "page_count": len(reader.pages),
        }

    image_pages = 0
    for page in reader.pages[:total_pages]:
        has_image = False
        try:
            has_image = bool(page.images)
        except Exception:
            resources = page.get("/Resources") or {}
            xobjects = resources.get("/XObject") or {}
            for xobject in xobjects.values():
                obj = xobject.get_object()
                if obj.get("/Subtype") == "/Image":
                    has_image = True
                    break
        if has_image:
            image_pages += 1

    image_page_ratio = image_pages / total_pages
    return {
        "is_image_pdf": image_page_ratio >= IMAGE_PDF_MIN_IMAGE_PAGE_RATIO,
        "image_page_ratio": image_page_ratio,
        "image_pages": image_pages,
        "sampled_pages": total_pages,
        "page_count": len(reader.pages),
    }


def _extract_ocr_lines(result: Any) -> list[str]:
    txts = getattr(result, "txts", None)
    if txts:
        return [str(text) for text in txts if str(text).strip()]

    lines: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            text = node.get("text") or node.get("transcription") or node.get("rec_text")
            if text:
                lines.append(str(text))
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[1], (list, tuple)) and node[1]:
                if isinstance(node[1][0], str):
                    lines.append(node[1][0])
            for item in node:
                walk(item)

    walk(result)
    return lines


def ocr_pdf_with_rapidocr(path: str | Path) -> str:
    try:
        import fitz
        import tempfile
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("OCR fallback requires rapidocr, onnxruntime, and pymupdf.") from exc

    ocr = RapidOCR()
    doc = fitz.open(str(path))
    page_texts = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for page_index, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image_path = Path(tmpdir) / f"page_{page_index}.png"
                pix.save(str(image_path))
                result = ocr(str(image_path))
                lines = _extract_ocr_lines(result)
                page_text = "\n".join(line.strip() for line in lines if line.strip())
                page_texts.append(page_text)
                print(f"  OCR 完成：第 {page_index}/{len(doc)} 页，识别 {len(page_text)} 个字符")
    finally:
        # On Windows PyMuPDF keeps a handle on the file until close(); leaving it
        # open blocks deleting/overwriting the source PDF (WinError 32).
        doc.close()
    return "\f".join(page_texts)


SAMPLE_TEXT = """第一条 保险合同构成
本保险合同由保险条款、投保单、保险单、批注以及其他相关书面协议共同构成。
第二条 投保范围
出生满28天至65周岁、身体健康者，均可作为被保险人投保。
第三条 等待期
本合同等待期为90天。等待期内因疾病导致的，公司不承担给付保险金责任，但退还已交保险费。
第四条 责任免除
酒后驾驶属于责任免除情形，公司不承担给付保险金责任。
第五条 理赔申请
申请理赔需提供保险合同、身份证明、医疗诊断证明、医疗费用原始凭证等材料。公司收到完整材料后5日内作出核定。
第六条 既往症
既往症导致的相关费用，公司不承担责任。
"""


def _new_trace(fname: str) -> IngestionTrace:
    return IngestionTrace(
        document=fname,
        collection_name=COLLECTION,
        document_id=default_document_id(fname),
    )


def _doc_record(fname: str, text: str, page_count: int, parse_method: str, ocr_used: bool) -> dict[str, Any]:
    return {
        "source": fname,
        "text": text,
        "page_count": page_count,
        "parse_method": parse_method,
        "ocr_used": ocr_used,
        "document_id": default_document_id(fname),
        "document_type": "insurance_clause",
    }


def _load_with_ocr(path: Path, trace: IngestionTrace, reason: str) -> dict[str, Any] | None:
    if not OCR_ENABLED:
        trace.warnings.append(f"OCR fallback disabled: {reason}")
        trace.error = "PDF quality is too low and OCR fallback is disabled."
        trace.error_code = OCR_ERROR
        save_ingestion_trace(trace)
        return None

    trace.warnings.append(f"OCR fallback used: {reason}")
    try:
        full_text = ocr_pdf_with_rapidocr(path)
    except Exception as exc:
        trace.error = f"OCR fallback failed: {exc}"
        trace.error_code = OCR_ERROR
        save_ingestion_trace(trace)
        return None
    page_texts = full_text.split("\f")
    ocr_quality = _quality(page_texts)
    print_pdf_quality_report(f"{path.name}（OCR 后）", ocr_quality)

    trace.parse_method = "ocr"
    trace.ocr_used = True
    trace.quality_score = float(ocr_quality["quality_score"])
    trace.text_density = float(ocr_quality["text_density"])
    trace.page_count = len(page_texts)

    if ocr_quality["level"] == "reject":
        trace.warnings.append("OCR text quality is still below threshold.")
        trace.error = "OCR text quality is still below threshold."
        trace.error_code = OCR_ERROR
        save_ingestion_trace(trace)
        return None

    save_ingestion_trace(trace)
    return _doc_record(path.name, full_text, len(page_texts), "ocr", True)


def load_pdfs(data_dir: str | Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return docs

    pdf_files = sorted(path for path in data_path.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_files:
        return docs

    from langchain_community.document_loaders import PyPDFLoader

    for path in pdf_files:
        trace = _new_trace(path.name)
        try:
            loader = PyPDFLoader(str(path))
            pages = loader.load()
            page_texts = [page.page_content for page in pages]
            page_count = len(page_texts)
        except Exception as exc:
            logger.warning("PDF parse failed, trying OCR fallback: %s, reason=%s", path.name, exc)
            trace.warnings.append(f"PyPDF parse failed: {exc}")
            trace.error_code = PDF_PARSE_ERROR
            doc = _load_with_ocr(path, trace, "pypdf_parse_failed")
            if doc:
                docs.append(doc)
            continue

        quality = _quality(page_texts)
        print_pdf_quality_report(path.name, quality)
        trace.quality_score = float(quality["quality_score"])
        trace.text_density = float(quality["text_density"])
        trace.page_count = page_count

        if quality["level"] == "warning":
            trace.warnings.append("PDF text quality warning; manual spot check recommended.")

        if quality["level"] == "reject":
            image_pdf = detect_image_pdf(path)
            trace.warnings.append(
                f"PDF quality rejected; image_page_ratio={image_pdf['image_page_ratio']:.2%}."
            )
            if should_use_ocr_fallback(quality, image_pdf=image_pdf, ocr_enabled=OCR_ENABLED):
                doc = _load_with_ocr(path, trace, "low_quality_image_pdf")
                if doc:
                    docs.append(doc)
                continue

            trace.error = "PDF text quality rejected and OCR fallback was not used."
            trace.error_code = PDF_PARSE_ERROR
            save_ingestion_trace(trace)
            continue

        full_text = "\n".join(page_texts)
        trace.parse_method = "pypdf"
        trace.ocr_used = False
        save_ingestion_trace(trace)
        docs.append(_doc_record(path.name, full_text, page_count, "pypdf", False))
        print(f"  已加载 PDF：{path.name}（{page_count} 页）")

    if pdf_files and not docs:
        raise RuntimeError("All PDFs failed quality checks or OCR fallback; no chunks were built.")
    return docs


def _update_chunk_counts(docs: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    for doc in docs:
        doc_id = doc["document_id"]
        doc_chunks = [chunk for chunk in chunks if chunk.get("document_id") == doc_id]
        parent_ids = {chunk.get("parent_id") for chunk in doc_chunks}
        update_ingestion_trace(
            doc_id,
            document=doc["source"],
            collection_name=COLLECTION,
            parent_chunks=len(parent_ids),
            child_chunks=len(doc_chunks),
            embedding_status="pending",
            milvus_insert_status="pending",
        )


def main() -> None:
    print("【阶段1】加载文档...")
    docs = load_pdfs(DATA_DIR)

    if not docs:
        print("  data/ 下没有 PDF，使用内置示例文本。")
        docs = [_doc_record("sample_policy.txt", SAMPLE_TEXT, 1, "sample", False)]

    print("【阶段1】父子分块...")
    chunks = build_chunks(docs, collection_name=COLLECTION)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    _update_chunk_counts(docs, chunks)

    n_parents = len({(chunk["source"], chunk["parent_id"]) for chunk in chunks})
    print(f"  完成：{len(chunks)} 个子块，{n_parents} 个父块 -> 已写入 {OUTPUT_PATH}")
    if chunks:
        print("  示例子块：", chunks[0]["child_text"][:40], "...")


if __name__ == "__main__":
    main()
