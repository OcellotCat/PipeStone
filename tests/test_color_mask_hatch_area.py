from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "color_mask_hatch.py"
SAMPLE_PATH = PROJECT_ROOT / "tests" / "tests_data" / "hatch.png"
TOTAL_AREA_RE = re.compile(r"Total area:\s*([0-9]+(?:\.[0-9]+)?)\s*m\^2")

BASELINE_AREA_M2 = 144.0
BASELINE_RELATIVE_TOLERANCE = 0.10
SCALED_RELATIVE_TOLERANCE = 0.12


class ColorMaskHatchAreaTest(unittest.TestCase):
    def assert_cli_area(
        self,
        image_name: str,
        expected_area_m2: float,
        relative_tolerance: float = BASELINE_RELATIVE_TOLERANCE,
    ) -> None:
        target_path = PROJECT_ROOT / "tests" / "tests_data" / image_name
        self.assertTrue(CLI_PATH.is_file(), f"CLI not found: {CLI_PATH}")
        self.assertTrue(SAMPLE_PATH.is_file(), f"Hatch sample not found: {SAMPLE_PATH}")
        self.assertTrue(target_path.is_file(), f"Target image not found: {target_path}")

        with tempfile.TemporaryDirectory(prefix="pipestone_area_test_") as temporary_directory:
            output_dir = Path(temporary_directory)
            command = [
                sys.executable,
                str(CLI_PATH),
                "--sample",
                str(SAMPLE_PATH),
                "--target",
                str(target_path),
                "--ocr-language",
                "rus+eng",
                "--output",
                str(output_dir / "color_masked.png"),
                "--mask-output",
                str(output_dir / "mask.png"),
                "--hatch-definition-output",
                str(output_dir / "hatch_definition.json"),
                "--bounds-output",
                str(output_dir / "hatch_bounds.json"),
                "--bounds-image-output",
                str(output_dir / "hatch_bounds.png"),
                "--elements-output",
                str(output_dir / "elements.json"),
                "--elements-image-output",
                str(output_dir / "elements.png"),
            ]
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=900,
                check=False,
            )

        self.assertEqual(
            completed.returncode,
            0,
            f"CLI failed for {image_name}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        match = TOTAL_AREA_RE.search(completed.stdout)
        self.assertIsNotNone(
            match,
            f"Total area was not printed for {image_name}\nstdout:\n{completed.stdout}",
        )

        actual_area_m2 = float(match.group(1))
        allowed_delta = expected_area_m2 * relative_tolerance
        self.assertAlmostEqual(
            actual_area_m2,
            expected_area_m2,
            delta=allowed_delta,
            msg=(
                f"Unexpected total area for {image_name}: {actual_area_m2:.3f} m^2; "
                f"expected {expected_area_m2:.3f} ± {allowed_delta:.3f} m^2"
            ),
        )

    def test_baseline_area(self) -> None:
        self.assert_cli_area("test.jpg", BASELINE_AREA_M2)

    def test_tiled_up_area(self) -> None:
        self.assert_cli_area(
            "test_tiled_up.jpg",
            BASELINE_AREA_M2 / 2.0,
            relative_tolerance=SCALED_RELATIVE_TOLERANCE,
        )

    def test_tiled_down_area(self) -> None:
        self.assert_cli_area(
            "test_tiled_down.jpg",
            BASELINE_AREA_M2 / 2.0,
            relative_tolerance=SCALED_RELATIVE_TOLERANCE,
        )


if __name__ == "__main__":
    unittest.main()
