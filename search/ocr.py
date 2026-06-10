from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import TextChunk
from .reporting import write_chunks_progress
from .utils import clean_page_text, clean_text, count_extra_text_chars, progress, safe_error, suppress_library_output


PADDLE_OCR_CACHE: dict[str, Any] = {}


def extract_tesseract_ocr_text(image_path: Path, lang: str) -> tuple[str | None, str | None]:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except ImportError as exc:
        return None, f"Tesseract OCR dependencies are not installed. Install with `pip install pytesseract Pillow`: {safe_error(exc)}"

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image, lang=lang)
    except Exception as exc:
        return None, safe_error(exc)
    text = clean_page_text(text, max_chars=20_000)
    if not text:
        return "", "OCR completed but returned no text."
    return text, None

def flatten_paddle_ocr_result(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        for key in ("rec_texts", "texts"):
            raw = value.get(key)
            if isinstance(raw, list):
                texts.extend(str(item) for item in raw if str(item).strip())
        if not texts:
            for item in value.values():
                texts.extend(flatten_paddle_ocr_result(item))
    elif isinstance(value, list):
        if len(value) >= 2 and isinstance(value[1], (list, tuple)) and value[1]:
            first = value[1][0]
            if isinstance(first, str):
                texts.append(first)
        for item in value:
            texts.extend(flatten_paddle_ocr_result(item))
    elif isinstance(value, tuple):
        texts.extend(flatten_paddle_ocr_result(list(value)))
    return texts

def extract_paddle_ocr_text(image_path: Path, lang: str) -> tuple[str | None, str | None]:
    try:
        with suppress_library_output():
            from paddleocr import PaddleOCR  # type: ignore
    except ImportError as exc:
        return None, f"PaddleOCR is not installed. Install with `pip install paddleocr paddlepaddle`: {safe_error(exc)}"

    try:
        with suppress_library_output():
            ocr = PADDLE_OCR_CACHE.get(lang)
            if ocr is None:
                init_errors: list[str] = []
                for kwargs in ({"lang": lang, "use_angle_cls": True}, {"lang": lang, "use_textline_orientation": True}, {"lang": lang}):
                    try:
                        ocr = PaddleOCR(**kwargs)
                        PADDLE_OCR_CACHE[lang] = ocr
                        break
                    except TypeError as exc:
                        init_errors.append(safe_error(exc))
                if ocr is None:
                    return None, "PaddleOCR init failed: " + "; ".join(init_errors)
            if hasattr(ocr, "ocr"):
                try:
                    result = ocr.ocr(str(image_path), cls=True)
                except TypeError:
                    result = ocr.ocr(str(image_path))
            elif hasattr(ocr, "predict"):
                result = ocr.predict(str(image_path))
            else:
                return None, "PaddleOCR object has neither ocr() nor predict()."
    except Exception as exc:
        return None, safe_error(exc)

    lines: list[str] = []
    seen: set[str] = set()
    for item in flatten_paddle_ocr_result(result):
        text = clean_text(item, max_chars=1000)
        if text and text not in seen:
            seen.add(text)
            lines.append(text)
    text = clean_page_text("\n".join(lines), max_chars=20_000)
    if not text:
        return "", "PaddleOCR completed but returned no text."
    return text, None

def extract_ocr_text(image_path: Path, engine: str, lang: str) -> tuple[str | None, str | None]:
    if engine == "paddle":
        return extract_paddle_ocr_text(image_path, lang)
    return extract_tesseract_ocr_text(image_path, lang)

def run_ocr_for_chunks(
    chunks: list[TextChunk],
    run_dir: Path,
    *,
    enabled: bool,
    engine: str,
    lang: str,
    min_extra_chars: int,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not enabled:
        return errors
    total = len(chunks)
    progress(f"OCR start: engine={engine}, lang={lang}, chunks={total}")
    for index, chunk in enumerate(chunks, start=1):
        if not chunk.chunk_shot_path:
            chunk.ocr_error = "OCR skipped: chunk has no screenshot."
            errors.append({"stage": "ocr", "chunk_id": chunk.chunk_id, "error": chunk.ocr_error})
            progress(f"OCR {index}/{total} {chunk.chunk_id}: skipped, no screenshot")
            write_chunks_progress(run_dir, chunks)
            continue
        image_path = run_dir / chunk.chunk_shot_path
        text, error = extract_ocr_text(image_path, engine, lang)
        chunk.ocr_text = text or None
        chunk.ocr_error = error
        if error and not text:
            errors.append(
                {
                    "stage": "ocr",
                    "engine": engine,
                    "chunk_id": chunk.chunk_id,
                    "path": chunk.chunk_shot_path,
                    "error": error,
                }
            )
            progress(f"OCR {index}/{total} {chunk.chunk_id}: failed - {error}")
            write_chunks_progress(run_dir, chunks)
            continue
        extra_chars = count_extra_text_chars(chunk.text, text or "")
        chunk.ocr_extra_chars = extra_chars
        if text and extra_chars >= min_extra_chars:
            sources = list(chunk.text_sources or ["dom"])
            if "ocr" not in sources:
                sources.append("ocr")
            chunk.text_sources = sources
            chunk.text = f"{chunk.text}\n\n[OCR补充]\n{text}".strip()
            chunk.end_char = max(chunk.end_char, chunk.start_char + len(chunk.text))
            progress(f"OCR {index}/{total} {chunk.chunk_id}: ok, extra_chars={extra_chars}, appended")
        else:
            progress(f"OCR {index}/{total} {chunk.chunk_id}: ok, extra_chars={extra_chars}, not appended")
        write_chunks_progress(run_dir, chunks)
    progress(f"OCR done: success={sum(1 for chunk in chunks if chunk.ocr_text)}, errors={len(errors)}")
    return errors
