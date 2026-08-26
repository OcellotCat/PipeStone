from __future__ import annotations

import unittest

from color_mask_hatch import sum_dimension_chain


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


if __name__ == "__main__":
    unittest.main()
