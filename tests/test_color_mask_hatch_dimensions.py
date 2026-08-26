from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from color_mask_hatch import recognize_external_horizontal_dimension, sum_dimension_chain


class DimensionChainTest(unittest.TestCase):
    def test_joint_dimensions_are_excluded_from_partial_chain(self) -> None:
        self.assertEqual(sum_dimension_chain(["2320", "10", "2650"]), "4970")

    def test_all_joint_dimensions_are_excluded_from_full_chain(self) -> None:
        self.assertEqual(
            sum_dimension_chain(["2320", "10", "2650", "10", "650"]),
            "5620",
        )

    def test_single_dimension_is_not_treated_as_chain(self) -> None:
        self.assertEqual(sum_dimension_chain(["2650", "10"]), "")

    def test_horizontal_dimension_closest_to_vertical_scale_is_selected(self) -> None:
        image = np.full((900, 500, 3), 255, dtype=np.uint8)
        bound = {"x": 100, "y": 100, "x1": 200, "y1": 600, "width": 100, "height": 500}
        with patch(
            "color_mask_hatch._read_dimension_candidates",
            return_value=["640", "995", "640"],
        ):
            result = recognize_external_horizontal_dimension(
                image,
                bound,
                bound,
                vertical_dimension="5000",
            )

        self.assertEqual(result, "995")


if __name__ == "__main__":
    unittest.main()
