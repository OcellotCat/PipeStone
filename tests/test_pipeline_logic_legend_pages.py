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

    def test_color_correlation_threshold_ignores_legend_table_self_match(self) -> None:
        image = np.full((256, 256, 3), 255, dtype=np.uint8)
        template = np.zeros((64, 64, 3), dtype=np.uint8)
        template[::2, :, :] = 255
        score_map = np.zeros((193, 193), dtype=np.float32)
        score_map[10, 10] = 1.0
        score_map[120, 120] = 0.3

        with patch("cv2.matchTemplate", return_value=score_map):
            result = pipeline_logic.color_correlation_mask(
                image,
                template,
                exclude_bbox=(0.0, 0.0, 80.0, 80.0),
            )

        self.assertEqual(result["max_score"], 0.3)
        self.assertEqual(result["threshold"], 0.18)
        self.assertGreater(int(np.count_nonzero(result["regions"][120:184, 120:184])), 0)

    def test_source_pdf_is_saved_next_to_log_without_renaming(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "Фасад камень.pdf"
            run_dir = root / "results"
            source.write_bytes(b"%PDF-1.7\n%%EOF")
            run_dir.mkdir()

            saved = pipeline_logic.save_pdf_next_to_log(source, run_dir)

            destination = run_dir / source.name
            self.assertEqual(saved, str(destination))
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_legend_title_accepts_ocr_errors_in_designation_word(self) -> None:
        self.assertEqual(
            pipeline_logic.legend_title_word_kind("обозначения"),
            "designation",
        )
        self.assertEqual(
            pipeline_logic.legend_title_word_kind("обозначени"),
            "designation",
        )
        self.assertEqual(
            pipeline_logic.legend_title_word_kind("обозначния"),
            "designation",
        )
        self.assertIsNone(pipeline_logic.legend_title_word_kind("облицовка"))

    def test_unreadable_title_uses_structured_table_fallback(self) -> None:
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        target = pipeline_logic.MaterialLine(
            page=5,
            text="изделия из натурального камня",
            bbox=(1.0, 1.0, 20.0, 5.0),
            confidence=1.0,
            source="test",
        )

        with (
            patch.object(pipeline_logic, "preprocess_image", return_value={"binary": image[:, :, 0]}),
            patch.object(pipeline_logic, "table_line_masks", return_value=(image[:, :, 0], image[:, :, 0], image[:, :, 0])),
            patch.object(pipeline_logic, "find_named_legend_table_bboxes", return_value=[]),
            patch.object(pipeline_logic, "find_legend_table_bboxes", return_value=[]) as all_tables,
        ):
            match = pipeline_logic.find_legend_pattern_match(image, [], target, page=5)

        self.assertIsNone(match)
        all_tables.assert_called_once()

    def test_repeated_legend_tables_use_first_page_of_dominant_copy(self) -> None:
        def match(page: int, width: float, height: float) -> pipeline_logic.LegendPatternMatch:
            return pipeline_logic.LegendPatternMatch(
                page=page,
                line_text="Натуральный камень 30 мм",
                table_bbox=(0.0, 0.0, width, height),
                row_bbox=(0.0, 1.0, width, 2.0),
                pattern_bbox=(0.0, 1.0, 10.0, 2.0),
                score=1.0,
                annotated_image="",
            )

        repeated = [
            match(5, 740.0, 559.0),
            match(6, 742.0, 560.0),
            match(7, 742.0, 560.0),
            match(8, 741.0, 559.0),
            match(9, 742.0, 560.0),
        ]

        selected = pipeline_logic.select_unique_legend_matches(repeated)

        self.assertEqual([item.page for item in selected], [6])

    def test_keeps_each_page_with_a_readable_legend_title(self) -> None:
        def match(page: int) -> pipeline_logic.LegendPatternMatch:
            return pipeline_logic.LegendPatternMatch(
                page=page,
                line_text="Натуральный камень 30 мм",
                table_bbox=(0.0, 0.0, 740.0, 560.0),
                row_bbox=(0.0, 1.0, 740.0, 2.0),
                pattern_bbox=(0.0, 1.0, 10.0, 2.0),
                score=1.0,
                annotated_image="",
            )

        selected = pipeline_logic.select_unique_legend_matches(
            [match(1), match(2)],
            preferred_pages={1, 2},
        )

        self.assertEqual([item.page for item in selected], [1, 2])

    def test_collapses_fuzzy_title_matches_with_varying_geometry(self) -> None:
        def match(page: int, width: float, height: float) -> pipeline_logic.LegendPatternMatch:
            return pipeline_logic.LegendPatternMatch(
                page=page,
                line_text="Натуральный камень 30 мм",
                table_bbox=(0.0, 0.0, width, height),
                row_bbox=(0.0, 1.0, width, 2.0),
                pattern_bbox=(0.0, 1.0, 10.0, 2.0),
                score=1.0,
                annotated_image="",
            )

        selected = pipeline_logic.select_unique_legend_matches(
            [
                match(5, 740.0, 559.0),
                match(6, 742.0, 560.0),
                match(7, 742.0, 560.0),
                match(8, 741.0, 559.0),
                match(9, 742.0, 560.0),
            ],
            preferred_pages={5, 6, 7, 8, 9},
        )

        self.assertEqual([item.page for item in selected], [6])

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

    def test_page_sized_match_is_not_discarded_for_containing_small_legend_table(self) -> None:
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        legend_match = pipeline_logic.LegendPatternMatch(
            page=1,
            line_text="Натуральный камень 30 мм",
            table_bbox=(80.0, 0.0, 100.0, 20.0),
            row_bbox=(80.0, 5.0, 100.0, 10.0),
            pattern_bbox=(80.0, 5.0, 85.0, 10.0),
            score=1.0,
            annotated_image="",
        )

        with (
            patch.object(pipeline_logic, "_legend_pattern_crop", return_value=image[:5, :5]),
            patch.object(
                pipeline_logic,
                "recognize_hatch_pattern",
                return_value={"matches": [{"bbox": [0.0, 0.0, 100.0, 100.0]}]},
            ),
        ):
            pages, matches = pipeline_logic.find_hatch_pages(
                [{"page": 1, "image": image}],
                {1: [legend_match]},
                Path("run"),
            )

        self.assertEqual(pages, [1])
        self.assertEqual(matches[0]["page"], 1)

    def test_legend_only_analysis_does_not_run_hatch_or_area_processing(self) -> None:
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        rendered_pages = [{"page": 1, "image": image}]
        legend_match = pipeline_logic.LegendPatternMatch(
            page=1,
            line_text="Гранит",
            table_bbox=(0.0, 0.0, 10.0, 10.0),
            row_bbox=(0.0, 0.0, 10.0, 5.0),
            pattern_bbox=(0.0, 0.0, 5.0, 5.0),
            score=0.9,
            annotated_image="",
        )
        with (
            patch.object(pipeline_logic, "make_run_dir", return_value=Path("run")),
            patch.object(pipeline_logic, "render_pdf_pages", return_value=rendered_pages) as render_pages,
            patch.object(pipeline_logic, "collect_ocr_words", return_value={}),
            patch.object(pipeline_logic, "find_page_legend_matches", return_value=([], [legend_match])),
            patch.object(pipeline_logic, "find_hatch_pages") as hatch_search,
            patch.object(pipeline_logic, "calculate_hatch_page_areas") as area_search,
        ):
            result = pipeline_logic.analyze_pdf_legends("test.pdf")

        self.assertEqual(result["legends"][0]["name"], "Гранит")
        self.assertEqual(result["analysis_dpi"], 220)
        self.assertEqual(result["pattern_matches"][0]["line_text"], "Гранит")
        render_pages.assert_called_once_with(Path("test.pdf"), dpi=220)
        hatch_search.assert_not_called()
        area_search.assert_not_called()

    def test_reuses_precomputed_legend_without_ocr_and_renders_only_legend_pages(self) -> None:
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        rendered_pages = [{"page": 2, "image": image}]
        precomputed = {
            "analysis_dpi": 200,
            "legend_pages": [2],
            "material_lines": [
                {
                    "page": 2,
                    "text": "Гранит",
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "confidence": 0.9,
                    "source": "tesseract",
                }
            ],
            "pattern_matches": [
                {
                    "page": 2,
                    "line_text": "Гранит",
                    "table_bbox": [0.0, 0.0, 10.0, 10.0],
                    "row_bbox": [0.0, 0.0, 10.0, 5.0],
                    "pattern_bbox": [1.0, 1.0, 5.0, 5.0],
                    "score": 0.95,
                }
            ],
        }

        with (
            patch.object(pipeline_logic, "require_module", return_value=object()),
            patch.object(pipeline_logic, "make_run_dir", return_value=Path("run")),
            patch.object(pipeline_logic, "save_pdf_next_to_log", return_value="run/test.pdf"),
            patch.object(pipeline_logic, "render_pdf_pages", return_value=rendered_pages) as render,
            patch.object(pipeline_logic, "collect_ocr_words") as collect_ocr,
            patch.object(pipeline_logic, "find_page_legend_matches") as find_legend,
            patch.object(
                pipeline_logic,
                "save_annotated_pattern_image",
                side_effect=lambda page_image, match, run_dir, *, page: match,
            ) as save,
            patch.object(pipeline_logic, "find_hatch_pages", return_value=([], [])),
        ):
            result = pipeline_logic.analyze_pdf_file(
                "test.pdf",
                dpi=400,
                precomputed_legend_analysis=precomputed,
            )

        render.assert_called_once_with(Path("test.pdf"), dpi=400, page_numbers=[2])
        collect_ocr.assert_not_called()
        find_legend.assert_not_called()
        scaled_match = save.call_args.args[1]
        self.assertEqual(scaled_match.pattern_bbox, (2.0, 2.0, 10.0, 10.0))
        self.assertEqual(result["material_lines"][0]["bbox"], [2.0, 4.0, 6.0, 8.0])
        self.assertEqual(result["legend_pages"], [2])

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
