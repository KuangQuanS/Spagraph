import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from spagraph.cellcom.relation_ranker import (
    DEFAULT_CALIBRATION_PROFILE,
    CalibrationWeights,
    LRCalibrationHead,
    SYNTHETIC_V2_FROZEN_WEIGHTS,
    calibrate_lr_statistics,
    ensemble_lr_rankings,
)
from spagraph.training.cellcom import aggregate_cellcom_seed_outputs, run_cellcom
from spagraph.cellcom.cellcom import degree_scale_attention
from spagraph.cellcom.cellcom_model import HeteroSTModel


class CalibrationTests(unittest.TestCase):
    def test_aggregate_gnn_excludes_representative_lr_identity(self):
        model = HeteroSTModel(
            n_genes=3,
            n_celltypes=2,
            gat_hidden_dims=[4],
            gat_heads=1,
            output_dim=4,
        )
        self.assertFalse(hasattr(model, "lr_id_embedding"))
        self.assertEqual(model.edge_attn_comm.layers[0].edge_dim, 1)

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
                "lr_pair": ["PAIR_A", "PAIR_B"],
                "occurrence_count": [50, 50],
                "avg_attention_score": [1.2, 1.2],
                "std_attention_score": [0.1, 0.1],
            }
        )
        ranked = calibrate_lr_statistics(frame).set_index("lr_pair")
        self.assertAlmostEqual(
            float(ranked.loc["PAIR_A", "calibrated_score"]),
            float(ranked.loc["PAIR_B", "calibrated_score"]),
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
        self.assertTrue(
            {
                "score_std",
                "rank_std",
                "n_seeds",
                "neural_attention_score",
                "raw_attention_rank",
            }.issubset(result.columns)
        )
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
        self.assertTrue(
            {
                "neural_attention_score",
                "attention_percentile",
                "support_percentile",
                "confidence_percentile",
                "uncertainty_percentile",
            }.issubset(result.columns)
        )
        self.assertEqual(
            SYNTHETIC_V2_FROZEN_WEIGHTS,
            CalibrationWeights(0.60, 0.05, 0.30, 0.05, 0.25),
        )
        raw = result.set_index("lr_pair")["raw_attention_rank"]
        self.assertEqual(int(raw.loc["A_B"]), 1)

    def test_calibration_head_is_frozen_and_auditable(self):
        head = LRCalibrationHead()
        self.assertEqual(list(head.parameters()), [])
        self.assertIn("coefficients", head.state_dict())
        features = torch.tensor(
            [[1.0, 0.5, 0.8, 0.4, 0.2]], dtype=torch.float64
        )
        expected = 0.60 + 0.05 * 0.5 + 0.30 * 0.8 + 0.05 * 0.4 - 0.25 * 0.2
        self.assertAlmostEqual(float(head(features)), expected)

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
            manifest = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(manifest["calibration_weights"]["attention"], 0.60)

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
            first_args = mocked.call_args_list[0].args[0]
            self.assertFalse(hasattr(first_args, "use_representative_lr_identity"))
            self.assertFalse(hasattr(first_args, "ablation_no_lr_identity"))

if __name__ == "__main__":
    unittest.main()
