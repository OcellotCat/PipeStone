from pathlib import Path
import tempfile
from unittest import TestCase

import pipeline_logic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_ROOT = PROJECT_ROOT / "tests" / "pdfs"
REGRESSION_DPI = 400
AREA_RELATIVE_TOLERANCE = 0.10


class PdfAreaRegressionTests(TestCase):
    """End-to-end area baselines for the repository's reference PDFs."""

    def tearDown(self) -> None:
        pipeline_logic.close_run_file_logging()

    def assert_pdf_area(self, pdf_name: str, expected_area_m2: float) -> None:
        pdf_path = PDF_ROOT / pdf_name
        self.assertTrue(pdf_path.is_file(), f"Reference PDF is missing: {pdf_path}")

        with tempfile.TemporaryDirectory(prefix="pipestone_pdf_regression_") as temp_dir:
            try:
                output_root = Path(temp_dir)
                legend_analysis = pipeline_logic.analyze_pdf_legends(
                    pdf_path,
                    output_dir=output_root / "legend",
                    ocr_backend="tesseract",
                    ocr_workers=4,
                )
                self.assertTrue(
                    legend_analysis["pattern_matches"],
                    f"Legend pattern was not found in {pdf_name}",
                )

                result = pipeline_logic.analyze_pdf_file(
                    pdf_path,
                    dpi=REGRESSION_DPI,
                    output_dir=output_root / "calculation",
                    ocr_backend="tesseract",
                    ocr_workers=4,
                    calculate_area=True,
                    precomputed_legend_analysis=legend_analysis,
                )
                area_result = result["area_calculation"]
                actual_area_m2 = float(area_result["total_area_m2"])

                self.assertTrue(area_result["pages"], f"No pages were calculated for {pdf_name}")
                self.assertAlmostEqual(
                    actual_area_m2,
                    expected_area_m2,
                    delta=expected_area_m2 * AREA_RELATIVE_TOLERANCE,
                    msg=(
                        f"Unexpected area for {pdf_name}: {actual_area_m2:.3f} m²; "
                        f"expected about {expected_area_m2:.3f} m²"
                    ),
                )
            finally:
                # Windows cannot remove the temporary run directory while the
                # active FileHandler still owns pipeline.log.
                pipeline_logic.close_run_file_logging()

    def test_test_blue_pdf_area_is_about_10_5_m2(self) -> None:
        self.assert_pdf_area("test_blue.pdf", 10.5)

    def test_merged_test_pdf_area_is_about_288_m2(self) -> None:
        self.assert_pdf_area("merged_test.pdf", 288.0)

    def test_first_full_test_pdf_area_is_about_144_m2(self) -> None:
        self.assert_pdf_area("first_full_test.pdf", 144.0)
