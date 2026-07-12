import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from spagraph.cellcom.relation_ranker import (
    DEFAULT_CALIBRATION_PROFILE,
    CalibrationWeights,
    LRRelationRanker,
    SYNTHETIC_V2_FROZEN_WEIGHTS,
    calibrate_lr_statistics,
    ensemble_lr_rankings,
    hard_negative_ranking_loss,
    within_context_ranking_loss,
)
from spagraph.training.cellcom import aggregate_cellcom_seed_outputs, run_cellcom
from spagraph.cellcom.cellcom import degree_scale_attention


class CalibrationTests(unittest.TestCase):
    def test_degree_scaled_attention_matches_edgewise_definition(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [2, 2, 4, 2]])
        attention = torch.tensor([[0.5, 1.0], [1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
        scaled = degree_scale_attention(attention, edge_index)
        expected = attention * torch.tensor([[3.0], [3.0], [1.0], [3.0]])
        self.assertTrue(torch.equal(scaled, expected))

    def test_low_support_candidate_is_penalized(self):
        frame = pd.DataFrame(
            {
                "lr_pair": ["A_R", "B_R"],
                "occurrence_count": [5, 100],
                "avg_attention_score": [1.0, 1.0],
                "std_attention_score": [0.2, 0.2],
            }
        )
        ranked = calibrate_lr_statistics(frame).set_index("lr_pair")
        self.assertGreater(ranked.loc["B_R", "calibrated_score"], ranked.loc["A_R", "calibrated_score"])
        self.assertEqual(int(ranked.loc["B_R", "rank"]), 1)

    def test_pair_names_do_not_receive_hardcoded_scores(self):
        frame = pd.DataFrame(
            {
                "lr_pair": ["TNC_SDC1", "UNRELATED_PAIR"],
                "occurrence_count": [50, 50],
                "avg_attention_score": [1.2, 1.2],
                "std_attention_score": [0.1, 0.1],
            }
        )
        ranked = calibrate_lr_statistics(frame).set_index("lr_pair")
        self.assertAlmostEqual(
            float(ranked.loc["TNC_SDC1", "calibrated_score"]),
            float(ranked.loc["UNRELATED_PAIR", "calibrated_score"]),
        )

    def test_ensemble_reports_seed_uncertainty(self):
        base = pd.DataFrame(
            {
                "lr_pair": ["A_R", "B_R"],
                "occurrence_count": [50, 20],
                "avg_attention_score": [1.0, 0.8],
                "std_attention_score": [0.1, 0.2],
            }
        )
        second = base.copy()
        second.loc[0, "avg_attention_score"] = 0.7
        result = ensemble_lr_rankings([base, second])
        self.assertTrue({"score_std", "rank_std", "n_seeds"}.issubset(result.columns))
        self.assertTrue((result["n_seeds"] == 2).all())

    def test_default_calibration_is_synthetic_frozen_and_keeps_raw_rank(self):
        source = pd.DataFrame(
            {
                "lr_pair": ["A_B", "C_D"],
                "associated_edge_attention_mean": [0.9, 0.8],
                "associated_edge_attention_std": [0.1, 0.1],
                "supporting_unique_edges": [10, 20],
                "n_source_spots": [5, 5],
                "n_target_spots": [5, 5],
            }
        )
        result = calibrate_lr_statistics(source)
        self.assertTrue((result["calibration_profile"] == DEFAULT_CALIBRATION_PROFILE).all())
        self.assertEqual(
            SYNTHETIC_V2_FROZEN_WEIGHTS,
            CalibrationWeights(0.60, 0.05, 0.30, 0.05, 0.25),
        )
        raw = result.set_index("lr_pair")["raw_attention_rank"]
        self.assertEqual(int(raw.loc["A_B"]), 1)

    def test_public_seed_aggregation_writes_manifest_and_ensemble(self):
        source = pd.DataFrame(
            {
                "lr_pair": ["A_B", "C_D"],
                "associated_edge_attention_mean": [0.9, 0.8],
                "associated_edge_attention_std": [0.1, 0.2],
                "supporting_unique_edges": [20, 10],
                "n_source_spots": [5, 5],
                "n_target_spots": [5, 5],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_dirs = []
            for seed in (11, 23):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir()
                source.to_csv(seed_dir / "lr_pair_associated_edge_statistics.csv", index=False)
                seed_dirs.append(seed_dir)
            result = aggregate_cellcom_seed_outputs(seed_dirs, root, [11, 23])
            self.assertTrue(Path(result["ensemble_path"]).exists())
            self.assertTrue(Path(result["manifest_path"]).exists())
            self.assertEqual(result["seeds"], [11, 23])
            self.assertTrue(
                (result["ensemble"]["calibration_profile"] == DEFAULT_CALIBRATION_PROFILE).all()
            )

    def test_public_api_runs_repeats_in_seed_subdirectories(self):
        source = pd.DataFrame(
            {
                "lr_pair": ["A_B", "C_D"],
                "associated_edge_attention_mean": [0.9, 0.8],
                "associated_edge_attention_std": [0.1, 0.2],
                "supporting_unique_edges": [20, 10],
                "n_source_spots": [5, 5],
                "n_target_spots": [5, 5],
            }
        )

        def fake_main(args):
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            calibrate_lr_statistics(source).to_csv(
                output / "lr_pair_associated_edge_statistics.csv", index=False
            )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "spagraph.training.cellcom.cellcom_main", side_effect=fake_main
        ) as mocked:
            result = run_cellcom(
                deconv_dir="unused", st_h5ad="unused", output_dir=tmp,
                seeds=[11, 23], n_repeats=2, device="cpu", epochs=1,
            )
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(result["seeds"], [11, 23])
            self.assertTrue((Path(tmp) / "seed_11").exists())
            self.assertTrue((Path(tmp) / "seed_23").exists())


class RelationModelTests(unittest.TestCase):
    def test_lr_identity_and_direction_change_scores(self):
        torch.manual_seed(7)
        model = LRRelationRanker(numeric_dim=3, n_lr_pairs=4, n_celltypes=3, dropout=0.0)
        features = torch.ones(3, 3)
        scores = model(
            features,
            lr_id=torch.tensor([0, 1, 0]),
            sender_id=torch.tensor([0, 0, 1]),
            receiver_id=torch.tensor([1, 1, 0]),
        )
        self.assertFalse(torch.isclose(scores[0], scores[1]))
        self.assertFalse(torch.isclose(scores[0], scores[2]))

    def test_hard_negative_loss_rewards_separation(self):
        bad = hard_negative_ranking_loss(torch.tensor([0.0]), torch.tensor([1.0]))
        good = hard_negative_ranking_loss(torch.tensor([2.0]), torch.tensor([0.0]))
        self.assertLess(float(good), float(bad))

    def test_within_context_loss_uses_competing_lr_edges(self):
        edge_index = torch.tensor([[1, 1, 2], [3, 3, 4]])
        target = torch.tensor([2.0, 0.5, 1.0])
        bad = within_context_ranking_loss(torch.tensor([0.0, 1.0, 0.0]), target, edge_index)
        good = within_context_ranking_loss(torch.tensor([2.0, 0.0, 0.0]), target, edge_index)
        self.assertLess(float(good), float(bad))


if __name__ == "__main__":
    unittest.main()
