import unittest
from unittest.mock import patch

import numpy as np

from color_mask_hatch import (
    build_hatch_definition,
    build_palette,
    gabor_hatch_response,
    mask_by_palette,
    process_images,
)


class ColorMaskHatchApiTest(unittest.TestCase):
    @staticmethod
    def _patch() -> np.ndarray:
        patch = np.full((32, 32, 3), 255, dtype=np.uint8)
        for offset in range(-32, 32, 6):
            rows = np.arange(32)
            columns = rows + offset
            valid = (columns >= 0) & (columns < 32)
            patch[rows[valid], columns[valid]] = (160, 40, 30)
        return patch

    def test_processes_list_without_file_io_or_input_mutation(self) -> None:
        patch = self._patch()
        image = np.tile(patch, (2, 2, 1))
        original = image.copy()

        results = process_images([image, image.copy()], patch, min_bound_area=1)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["masked_image"].shape, image.shape)
        self.assertEqual(results[0]["color_mask"].shape, image.shape[:2])
        self.assertEqual(results[0]["color_mask"].dtype, np.bool_)
        self.assertIsInstance(results[0]["bounds"], list)
        np.testing.assert_array_equal(image, original)

    def test_accepts_stacked_array(self) -> None:
        patch = self._patch()
        images = np.stack([patch, patch])

        results = process_images(images, patch, min_bound_area=1)

        self.assertEqual(len(results), 2)

    def test_accepts_fully_bright_blue_hatch(self) -> None:
        hatch_patch = np.full((32, 32, 3), 255, dtype=np.uint8)
        for offset in range(-32, 32, 6):
            rows = np.arange(32)
            columns = rows + offset
            valid = (columns >= 0) & (columns < 32)
            hatch_patch[rows[valid], columns[valid]] = (0, 0, 255)
        image = np.tile(hatch_patch, (2, 2, 1))

        result = process_images([image], hatch_patch, min_bound_area=1)[0]

        self.assertGreater(int(np.count_nonzero(result["color_mask"])), 0)
        self.assertGreater(len(result["bounds"]), 0)
        self.assertEqual(result["hatch_definition"]["colors"][0]["hex"], "#0000f8")

    def test_rejects_non_rgb_image(self) -> None:
        patch = self._patch()

        with self.assertRaisesRegex(ValueError, "shape"):
            process_images([np.zeros((10, 10), dtype=np.uint8)], patch)

    def test_optionally_returns_area_results(self) -> None:
        hatch_patch = self._patch()
        image = np.tile(hatch_patch, (2, 2, 1))
        elements = {
            "E1": {
                "count": 2,
                "horizontal_dimensions": ["1000"],
                "vertical_dimensions": ["2000"],
            }
        }
        analysis = ([], [], [], elements, [], [], [], [], [])

        with (
            patch("color_mask_hatch.analyze_euclidean_bound_buckets", return_value=analysis),
            patch("color_mask_hatch.formal_merge_buckets_by_size", return_value={}),
        ):
            result = process_images(
                [image],
                hatch_patch,
                min_bound_area=1,
                calculate_area=True,
            )[0]

        self.assertEqual(result["total_area_m2"], 4.0)
        self.assertEqual(result["unique_elements"]["E1"]["total_area_m2"], 4.0)

    def test_parallel_palette_and_gabor_match_sequential_results(self) -> None:
        hatch_patch = self._patch()
        image = np.tile(hatch_patch, (4, 4, 1))
        palette = build_palette(hatch_patch, 48, 25, 255)
        hatch_definition = build_hatch_definition(hatch_patch, palette, 25, 255)

        sequential_rgb, sequential_mask = mask_by_palette(
            image,
            palette,
            18.0,
            False,
            12,
            255,
            5,
            chunk_size=500,
            workers=1,
        )
        parallel_rgb, parallel_mask = mask_by_palette(
            image,
            palette,
            18.0,
            False,
            12,
            255,
            5,
            chunk_size=500,
            workers=4,
        )
        np.testing.assert_array_equal(parallel_rgb, sequential_rgb)
        np.testing.assert_array_equal(parallel_mask, sequential_mask)

        sequential_gabor = gabor_hatch_response(
            image,
            sequential_mask,
            hatch_definition,
            tile_size=32,
            workers=1,
        )
        parallel_gabor = gabor_hatch_response(
            image,
            parallel_mask,
            hatch_definition,
            tile_size=32,
            workers=4,
        )
        np.testing.assert_array_equal(parallel_gabor, sequential_gabor)


if __name__ == "__main__":
    unittest.main()
