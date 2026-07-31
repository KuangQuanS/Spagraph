import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import spagraph.training.deconv as deconv_module


class DeconvEnsembleTests(unittest.TestCase):
    def test_mean_aligned_frames_projects_rows_to_simplex(self):
        first = pd.DataFrame(
            [[0.8, 0.2], [0.1, 0.9]], index=["s1", "s2"], columns=["a", "b"]
        )
        second = pd.DataFrame(
            [[0.6, 0.4], [0.3, 0.7]], index=["s1", "s2"], columns=["a", "b"]
        )
        result = deconv_module._mean_aligned_frames(
            [first, second], normalize_rows=True
        )
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        np.testing.assert_allclose(result.loc["s1"], [0.7, 0.3])

    def test_deconv_ensemble_averages_repeats(self):
        calls = []

        def fake_run_deconv(*, output_dir, seed, **kwargs):
            calls.append(seed)
            value = 0.2 if seed == 11 else 0.6
            frame = pd.DataFrame(
                [[value, 1.0 - value]], index=["spot"], columns=["A", "B"]
            )
            path = Path(output_dir) / "Spatial_composition.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path)
            return {
                "deconv": frame,
                "deconv_path": str(path),
                "sample_name": "Spatial",
                "metrics": {"pearson": float(seed), "n_clusters": 2},
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(deconv_module, "run_deconv", fake_run_deconv):
                result = deconv_module.run_deconv_ensemble(
                    vae=object(),
                    output_dir=tmp_dir,
                    n_repeats=2,
                    seeds=[11, 23],
                )
            self.assertEqual(calls, [11, 23])
            np.testing.assert_allclose(
                result["deconv"].loc["spot"], [0.4, 0.6]
            )
            self.assertEqual(result["repeat_seeds"], [11, 23])
            self.assertTrue(Path(result["deconv_path"]).is_file())

    def test_union_alignment_treats_missing_spot_cell_rows_as_zero(self):
        first = pd.DataFrame([[2.0]], index=["spot_A"], columns=["g1"])
        second = pd.DataFrame([[4.0]], index=["spot_B"], columns=["g1"])
        result = deconv_module._mean_aligned_frames(
            [first, second], join="union"
        )
        self.assertEqual(list(result.index), ["spot_A", "spot_B"])
        np.testing.assert_allclose(result["g1"], [1.0, 2.0])

    def test_reconstructed_outputs_are_ensembled_for_stage3(self):
        def fake_run_deconv(*, output_dir, seed, **kwargs):
            repeat_dir = Path(output_dir)
            repeat_dir.mkdir(parents=True, exist_ok=True)
            composition = pd.DataFrame(
                [[0.5, 0.5]], index=["spot"], columns=["A", "B"]
            )
            composition_path = repeat_dir / "Spatial_composition.csv"
            composition.to_csv(composition_path)
            pd.DataFrame(
                [[float(seed)]], index=["spot"], columns=["g1"]
            ).to_csv(repeat_dir / "Spatial_reconstructed.csv")
            pd.DataFrame(
                [[2.0]], index=[f"spot_{seed}"], columns=["g1"]
            ).to_csv(repeat_dir / "Spatial_spot_cell_expr.csv")
            return {
                "deconv": composition,
                "deconv_path": str(composition_path),
                "sample_name": "Spatial",
                "metrics": {},
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(deconv_module, "run_deconv", fake_run_deconv):
                deconv_module.run_deconv_ensemble(
                    vae=object(),
                    output_dir=tmp_dir,
                    n_repeats=2,
                    seeds=[11, 23],
                    save_reconstructed_genes=True,
                )
            reconstructed = pd.read_csv(
                Path(tmp_dir) / "Spatial_reconstructed.csv", index_col=0
            )
            spot_cell = pd.read_csv(
                Path(tmp_dir) / "Spatial_spot_cell_expr.csv", index_col=0
            )
            np.testing.assert_allclose(reconstructed.loc["spot", "g1"], 17.0)
            np.testing.assert_allclose(
                spot_cell.loc[["spot_11", "spot_23"], "g1"], [1.0, 1.0]
            )

    def test_deconv_ensemble_rejects_seed_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, r"len\(seeds\)"):
            deconv_module.run_deconv_ensemble(
                vae=object(), n_repeats=3, seeds=[11, 23]
            )


if __name__ == "__main__":
    unittest.main()
