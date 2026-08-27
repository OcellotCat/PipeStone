from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from pipestone_ocr import OcrWord, collect_ocr_words


class ParallelOcrTests(TestCase):
    def test_image_pages_are_processed_in_parallel_and_merged_in_page_order(self) -> None:
        pages = [
            {"page": page, "image": np.zeros((4, 4, 3), dtype=np.uint8)}
            for page in range(1, 5)
        ]
        lock = threading.Lock()
        active = 0
        max_active = 0

        def recognize(image, page, backend, **kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return [OcrWord(page, f"page-{page}", (0.0, 0.0, 1.0, 1.0), 1.0, "test")], None

        with (
            patch("pipestone_ocr.extract_pdf_text_words", return_value={}),
            patch("pipestone_ocr.run_image_ocr", side_effect=recognize),
        ):
            result = collect_ocr_words(
                Path("drawing.pdf"),
                pages,
                backend="tesseract",
                force_ocr=True,
                ocr_workers=4,
            )

        self.assertGreater(max_active, 1)
        self.assertEqual(list(result), [1, 2, 3, 4])
        self.assertEqual([result[page][0].text for page in result], [f"page-{page}" for page in range(1, 5)])

    def test_workers_are_capped_by_page_count(self) -> None:
        page = {"page": 1, "image": np.zeros((4, 4, 3), dtype=np.uint8)}
        with (
            patch("pipestone_ocr.extract_pdf_text_words", return_value={}),
            patch("pipestone_ocr.run_image_ocr", return_value=([], None)) as recognize,
        ):
            collect_ocr_words(
                Path("drawing.pdf"),
                [page],
                backend="tesseract",
                force_ocr=True,
                ocr_workers=4,
            )

        recognize.assert_called_once()
