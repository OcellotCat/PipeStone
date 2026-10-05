"""Public API for the PipeStone analysis package."""

from .ocr import DEFAULT_TESSERACT_LANGUAGE
from .pipeline_logic import (
    APP_NAME,
    DEFAULT_DPI,
    DEFAULT_OCR_WORKERS,
    DEFAULT_OUTPUT_DIR,
    analyze_image_file,
    analyze_pdf_file,
    analyze_pdf_legends,
    setup_logging,
)

__all__ = [
    "APP_NAME",
    "DEFAULT_DPI",
    "DEFAULT_OCR_WORKERS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TESSERACT_LANGUAGE",
    "analyze_image_file",
    "analyze_pdf_file",
    "analyze_pdf_legends",
    "setup_logging",
]
