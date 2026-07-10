import numpy as np
import pandas as pd
import unittest

from spagraph.analysis.figure3e_statistics import (
    compare_lr_pair_groups,
    holm_adjust,
    make_disjoint_lr_pair_groups,
)


METRICS = {
    "edge_spatial_focality": "Edge spatial focality",
    "celltype_pair_count": "Cell-type-pair count",
}


class Figure3eStatisticsTests(unittest.TestCase):
    def test_overlap_is_removed_from_both_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "ranking_type": ["attention", "attention", "frequency", "frequency"],
                "lr_pair": ["A_B", "X_Y", "C_D", "X_Y"],
                "edge_spatial_focality": [0.9, 0.8, 0.4, 0.8],
                "celltype_pair_count": [2, 3, 9, 3],
            }
        )
        disjoint, overlap = make_disjoint_lr_pair_groups(frame)
        self.assertEqual(overlap, ["X_Y"])
        self.assertEqual(disjoint["lr_pair"].tolist(), ["A_B", "C_D"])

    def test_no_overlap_preserves_all_pairs(self) -> None:
        frame = pd.DataFrame(
            {
                "ranking_type": ["attention", "frequency"],
                "lr_pair": ["A_B", "C_D"],
            }
        )
        disjoint, overlap = make_disjoint_lr_pair_groups(frame)
        self.assertEqual(overlap, [])
        self.assertEqual(len(disjoint), 2)
        self.assertEqual(set(disjoint["overlap_excluded"]), {"none"})

    def test_duplicate_statistical_unit_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "ranking_type": ["attention", "attention"],
                "lr_pair": ["A_B", "A_B"],
            }
        )
        with self.assertRaisesRegex(ValueError, "Each LR pair must occur once"):
            make_disjoint_lr_pair_groups(frame)

    def test_holm_adjustment(self) -> None:
        adjusted = holm_adjust([0.00052230254117, 0.0000679921575])
        np.testing.assert_allclose(adjusted, [0.00052230254117, 0.000135984315])

    def test_known_figure3e_result(self) -> None:
        frame = pd.read_csv(
            "tests/fixtures/figure3e_selected_top_pairs.csv"
        )
        summary, disjoint, overlap = compare_lr_pair_groups(frame, METRICS)

        self.assertEqual(overlap, ["TNC_SDC1"])
        self.assertEqual(len(disjoint.loc[disjoint["ranking_type"] == "attention"]), 14)
        self.assertEqual(len(disjoint.loc[disjoint["ranking_type"] == "frequency"]), 14)

        focality = summary.set_index("metric").loc["edge_spatial_focality"]
        self.assertAlmostEqual(focality["attention_median"], 0.8508918513)
        self.assertAlmostEqual(focality["frequency_median"], 0.8040573625)
        self.assertAlmostEqual(focality["mannwhitney_u"], 174)
        self.assertAlmostEqual(focality["holm_p"], 5.2230254117e-4)

        count = summary.set_index("metric").loc["celltype_pair_count"]
        self.assertAlmostEqual(count["attention_median"], 3.0)
        self.assertAlmostEqual(count["frequency_median"], 28.5)
        self.assertAlmostEqual(count["mannwhitney_u"], 11)
        self.assertAlmostEqual(count["holm_p"], 1.35984315e-4)


if __name__ == "__main__":
    unittest.main()
