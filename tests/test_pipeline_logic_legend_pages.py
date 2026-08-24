from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import pipeline_logic


class AnalyzePdfLegendPagesTests(TestCase):
    def tearDown(self) -> None:
        pipeline_logic.close_run_file_logging()

    def test_run_log_is_created_without_duplicate_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_dir = Path(temp_dir) / "first"
            second_dir = Path(temp_dir) / "second"
            first_dir.mkdir()
            second_dir.mkdir()

            first_log = pipeline_logic.setup_run_file_logging(first_dir)
            pipeline_logic.logger.info("first run message")
            pipeline_logic.logging.getLogger("pipestone.ocr").info("ocr child message")
            second_log = pipeline_logic.setup_run_file_logging(second_dir)
            pipeline_logic.logger.info("second run message")
            pipeline_logic.close_run_file_logging()

            first_text = first_log.read_text(encoding="utf-8")
            self.assertIn("first run message", first_text)
            self.assertIn("ocr child message", first_text)
            second_text = second_log.read_text(encoding="utf-8")
            self.assertIn("second run message", second_text)
            self.assertNotIn("second run message", first_log.read_text(encoding="utf-8"))

    def test_searches_all_pages_and_returns_hatch_and_legend_pages(self) -> None:
        image = np.full((20, 20, 3), 255, dtype=np.uint8)
        rendered_pages = [
            {"page": 1, "image": image.copy()},
            {"page": 2, "image": image.copy()},
            {"page": 3, "image": image.copy()},
        ]
        legend_match = pipeline_logic.LegendPatternMatch(
            page=2,
            line_text="granite",
            table_bbox=(0.0, 0.0, 10.0, 10.0),
            row_bbox=(0.0, 0.0, 10.0, 5.0),
            pattern_bbox=(0.0, 0.0, 5.0, 5.0),
            score=1.0,
            annotated_image="",
        )

        def find_legend(image, words, *, page):
            return ([], [legend_match]) if page == 2 else ([], [])

        def recognize(image, pattern_crop, match, output_dir, *, page):
            matches = []
            if page == 2:
                matches = [{"bbox": [1.0, 1.0, 5.0, 5.0]}]
            elif page == 3:
                matches = [{"bbox": [12.0, 12.0, 18.0, 18.0]}]
            return {"matches": matches, "matches_image": f"page-{page}.png"}

        with (
            patch.object(pipeline_logic, "require_module", return_value=object()),
            patch.object(pipeline_logic, "make_run_dir", return_value=Path("run")),
            patch.object(pipeline_logic, "render_pdf_pages", return_value=rendered_pages),
            patch.object(pipeline_logic, "collect_ocr_words", return_value={}),
            patch.object(pipeline_logic, "find_page_legend_matches", side_effect=find_legend),
            patch.object(
                pipeline_logic,
                "save_annotated_pattern_image",
                return_value=legend_match,
            ) as save,
            patch.object(pipeline_logic, "_legend_pattern_crop", return_value=image[:4, :4]),
            patch.object(pipeline_logic, "recognize_hatch_pattern", side_effect=recognize) as search,
        ):
            result = pipeline_logic.analyze_pdf_file("test.pdf")

        self.assertEqual(result["legend_pages"], [2])
        self.assertEqual(result["hatch_pages"], [3])
        self.assertEqual(result["hatch_page_matches"][0]["page"], 3)
        self.assertEqual(search.call_count, 3)
        save.assert_called_once_with(rendered_pages[1]["image"], legend_match, Path("run"), page=2)

    def test_returns_empty_hatch_pages_when_no_legend_was_found(self) -> None:
        rendered_pages = [{"page": 1, "image": np.zeros((5, 5, 3), dtype=np.uint8)}]
        with (
            patch.object(pipeline_logic, "require_module", return_value=object()),
            patch.object(pipeline_logic, "make_run_dir", return_value=Path("run")),
            patch.object(pipeline_logic, "render_pdf_pages", return_value=rendered_pages),
            patch.object(pipeline_logic, "collect_ocr_words", return_value={}),
            patch.object(pipeline_logic, "find_page_legend_matches", return_value=([], [])),
            patch.object(pipeline_logic, "recognize_hatch_pattern") as search,
        ):
            result = pipeline_logic.analyze_pdf_file("test.pdf")

        self.assertEqual(result["legend_pages"], [])
        self.assertEqual(result["hatch_pages"], [])
        search.assert_not_called()

    def test_area_calculation_uses_pattern_box_pages_and_saved_pattern(self) -> None:
        images = [
            {"page": 1, "image": np.full((4, 4, 3), 1, dtype=np.uint8)},
            {"page": 2, "image": np.full((4, 4, 3), 2, dtype=np.uint8)},
            {"page": 3, "image": np.full((4, 4, 3), 3, dtype=np.uint8)},
        ]
        pattern_image = np.full((2, 2, 3), 7, dtype=np.uint8)
        processed = [
            {"total_area_m2": 1.25, "unique_elements": {"A": {}}, "validated_inner_bounds": []},
            {"total_area_m2": 2.5, "unique_elements": {"B": {}}, "validated_inner_bounds": []},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(pipeline_logic, "load_image_rgb", return_value=pattern_image),
                patch.object(pipeline_logic, "require_module", return_value=object()),
                patch("color_mask_hatch.process_images", return_value=processed) as process,
            ):
                result = pipeline_logic.calculate_hatch_page_areas(
                    images,
                    [1, 3],
                    "run/pattern_results/page_001_legend_pattern_trimmed.png",
                    Path(temp_dir),
                    ocr_language="rus+eng",
                )

        self.assertEqual([item["page"] for item in result["pages"]], [1, 3])
        self.assertEqual(result["total_area_m2"], 3.75)
        selected_images, selected_pattern = process.call_args.args
        np.testing.assert_array_equal(selected_images[0], images[0]["image"])
        np.testing.assert_array_equal(selected_images[1], images[2]["image"])
        np.testing.assert_array_equal(selected_pattern, pattern_image)
        self.assertTrue(process.call_args.kwargs["calculate_area"])
