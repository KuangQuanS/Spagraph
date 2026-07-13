import unittest
import tempfile
from pathlib import Path

import numpy as np
import anndata as ad
from scipy import sparse
import torch

from spagraph.models.deconv_initialization import (
    aggregate_reference_by_labels,
    boundary_aware_graph_loss,
    compute_batched_simplex_initialization,
    compute_platform_calibrated_initialization,
    compute_signature_initialization,
    poisson_deviance_loss,
    power_calibrate_composition,
    select_celltype_specific_genes,
)
from spagraph.training.signature_deconv import run_signature_deconv


class FastSignatureDeconvTests(unittest.TestCase):
    def test_is_simplex_and_deterministic(self):
        sc_expression = sparse.csr_matrix(
            [
                [20, 1, 1, 2, 0, 1], [18, 1, 0, 2, 1, 1],
                [1, 20, 1, 1, 2, 0], [0, 18, 1, 1, 2, 1],
                [1, 1, 20, 0, 1, 2], [1, 0, 18, 1, 1, 2],
            ], dtype=np.float32
        )
        st_expression = sparse.csr_matrix(
            [[10, 10, 1, 1, 1, 1], [1, 2, 18, 1, 1, 2]], dtype=np.float32
        )
        genes = [f"g{i}" for i in range(sc_expression.shape[1])]
        sc_adata = ad.AnnData(sc_expression)
        sc_adata.var_names = genes
        sc_adata.obs["cell_type"] = ["A", "A", "B", "B", "C", "C"]
        st_adata = ad.AnnData(st_expression)
        st_adata.var_names = genes
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            sc_path, st_path = tmp_path / "sc.h5ad", tmp_path / "st.h5ad"
            sc_adata.write_h5ad(sc_path)
            st_adata.write_h5ad(st_path)
            first = run_signature_deconv(
                str(sc_path), str(st_path), genes_per_celltype=1,
                output_dir=str(tmp_path / "out")
            )
            second = run_signature_deconv(str(sc_path), str(st_path), genes_per_celltype=1)
        values = first["deconv"].to_numpy()
        np.testing.assert_allclose(values, second["deconv"].to_numpy(), atol=1e-7)
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(values >= 0))
        self.assertEqual(first["graph_source"], "not_used_signature_only")
        self.assertIsNotNone(first["deconv_path"])
from spagraph.models.deconv_model import HeterogeneousGATDeconvolution, SpatialDeconvolutionLoss
from spagraph.training.deconv import run_deconv_auto_k
from spagraph.models.stage2 import GATDeconvolution


class SignatureInitializationTests(unittest.TestCase):
    def test_power_calibration_preserves_simplex_and_sharpens(self):
        original = np.array([[0.6, 0.3, 0.1], [0.5, 0.5, 0.0]], dtype=float)
        calibrated = power_calibrate_composition(original, power=1.2)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-7)
        self.assertGreater(calibrated[0, 0], original[0, 0])
        self.assertTrue(np.all(calibrated >= 0))
        with self.assertRaises(ValueError):
            power_calibrate_composition(original, power=0)

    def test_recovers_simple_mixtures_on_the_simplex(self):
        signatures = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0]])
        truth = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
        spots = truth @ signatures
        estimate = compute_signature_initialization(spots, signatures, ridge=0.0)
        np.testing.assert_allclose(estimate.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(estimate, truth, atol=1e-5)

    def test_rejects_negative_expression(self):
        with self.assertRaises(ValueError):
            compute_signature_initialization(np.array([[1.0, -1.0]]), np.ones((2, 2)))

    def test_annotation_grouping_preserves_all_reference_types(self):
        grouped = aggregate_reference_by_labels(
            labels=["rare", "common", "common"],
            embeddings=np.array([[4, 0], [0, 2], [0, 4]], dtype=float),
            marker_expression=np.array([[8, 0], [0, 2], [0, 6]], dtype=float),
            raw_expression=np.array([[10, 0], [0, 3], [0, 7]], dtype=float),
        )
        self.assertEqual(list(grouped["encoder"].classes_), ["common", "rare"])
        self.assertEqual(grouped["prototypes"].shape, (2, 2))
        np.testing.assert_allclose(grouped["marker_signatures"][0], [0, 4])
        self.assertEqual(set(grouped["encoded_labels"]), {0, 1})
        np.testing.assert_allclose(
            grouped["cell_normalized_signatures"].sum(axis=1), 1.0, atol=1e-6
        )
        self.assertEqual(grouped["log_normalized_signatures"].shape, (2, 2))

    def test_celltype_specific_gene_selection_is_balanced(self):
        signatures = np.array(
            [
                [10.0, 0.0, 0.0, 5.0],
                [0.0, 10.0, 0.0, 5.0],
                [0.0, 0.0, 10.0, 5.0],
            ]
        )
        selected = select_celltype_specific_genes(signatures, top_per_celltype=1)
        self.assertEqual(set(selected.tolist()), {0, 1, 2})

    def test_platform_calibration_is_finite_and_simplex(self):
        signatures = np.array([[8.0, 1.0, 0.0], [0.0, 1.0, 8.0]])
        truth = np.array([[0.75, 0.25], [0.2, 0.8]])
        platform = np.array([2.0, 0.5, 1.5])
        spots = (truth @ signatures) * platform
        estimate, factors = compute_platform_calibrated_initialization(
            spots, signatures, ridge=0.0, iterations=3
        )
        self.assertTrue(np.isfinite(estimate).all())
        self.assertTrue(np.isfinite(factors).all())
        np.testing.assert_allclose(estimate.sum(axis=1), 1.0, atol=1e-6)

    def test_batched_simplex_solver_recovers_mixtures(self):
        signatures = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0]])
        truth = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
        estimate = compute_batched_simplex_initialization(
            truth @ signatures, signatures, ridge=0.0, max_iter=1000
        )
        np.testing.assert_allclose(estimate, truth, atol=2e-3)


class GraphAndPriorTests(unittest.TestCase):
    def _model(self, prior_strength=0.0):
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        model = HeterogeneousGATDeconvolution(
            embedding_dim=2,
            n_cell_types=2,
            gat_hidden_dim=4,
            gat_layers=1,
            gat_heads=1,
            dropout=0.0,
            k_spatial=1,
            k_celltype=2,
            celltype_prototypes=prototypes,
            signature_prior_strength=prior_strength,
        )
        return model

    def test_spatial_cache_is_invalidated_when_coordinates_change(self):
        model = self._model()
        emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        coords_a = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        coords_b = torch.tensor([[0.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        edges_a, _ = model._build_spatial_edges(coords_a, emb, False)
        edges_b, _ = model._build_spatial_edges(coords_b, emb, False)
        self.assertFalse(torch.equal(edges_a, edges_b))

    def test_signature_prior_changes_composition_without_ground_truth(self):
        model = self._model(prior_strength=2.0)
        for parameter in model.attention_mlp.parameters():
            torch.nn.init.zeros_(parameter)
        out = model(
            spot_embeddings=torch.tensor([[0.5, 0.5]]),
            spatial_coords=torch.tensor([[0.0, 0.0]]),
            celltype_prototypes=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            initial_weights=torch.tensor([[0.8, 0.2]]),
        )
        self.assertGreater(float(out["deconv_weights"][0, 0]), 0.8)
        self.assertAlmostEqual(float(out["deconv_weights"].sum()), 1.0, places=6)

    def test_robust_losses_are_finite(self):
        pred = torch.tensor([[0.0, 2.0], [1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        self.assertTrue(torch.isfinite(poisson_deviance_loss(pred, target)))
        edges = torch.tensor([[0, 1], [1, 0]])
        weights = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
        self.assertTrue(torch.isfinite(boundary_aware_graph_loss(weights, edges, target)))


class AutoKValidationTests(unittest.TestCase):
    def test_empty_candidates_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_deconv_auto_k(vae=None, k_celltype_range=[])


class OptimizedLossTests(unittest.TestCase):
    def test_heldout_poisson_and_spatial_losses_are_reported(self):
        loss = SpatialDeconvolutionLoss(
            celltype_expressions_full=np.array([[5, 0, 2, 0], [0, 5, 0, 2]], dtype=float),
            marker_gene_indices=[0, 1, 2, 3],
            scale_basis="none",
            lambda_poisson=0.1,
            lambda_spatial=0.1,
            heldout_gene_fraction=0.25,
            split_seed=42,
        )
        outputs = loss(
            attention_weights=torch.tensor([[0.8, 0.2], [0.7, 0.3]]),
            celltype_expression=torch.empty(0),
            true_spot_expression=torch.tensor([[4.0, 1.0, 1.5, 0.5], [3.5, 1.5, 1.2, 0.8]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        self.assertTrue({"poisson_loss", "spatial_loss", "heldout_loss"}.issubset(outputs))
        self.assertTrue(torch.isfinite(outputs["total_loss"]))
        self.assertGreater(float(outputs["heldout_loss"]), 0.0)

    def test_stage2_builder_wires_all_optimization_terms(self):
        trainer = GATDeconvolution(output_dir=None, device="cpu", seed=42)
        trainer.latent_dim = 2
        trainer.k_spatial = 1
        trainer.k_celltype = 2
        trainer.scale_basis = "none"
        trainer.celltype_prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        trainer.celltype_expressions_full = [np.array([5, 0, 2, 0]), np.array([0, 5, 0, 2])]
        trainer.all_genes = ["g0", "g1", "g2", "g3"]
        trainer.genes = list(trainer.all_genes)
        trainer.hvg_genes_union = None
        trainer.sc_clusters = np.array([0, 0, 1, 1])
        trainer.build_gat_model(
            n_cell_types=2,
            gat_hidden_dim=4,
            gat_layers=1,
            gat_heads=1,
            signature_prior_strength=1.0,
            lambda_poisson=0.1,
            lambda_spatial=0.1,
            heldout_gene_fraction=0.25,
        )
        self.assertEqual(trainer.loss_fn.heldout_marker_positions.numel(), 1)
        self.assertAlmostEqual(trainer.loss_fn.lambda_poisson, 0.1)
        self.assertAlmostEqual(trainer.loss_fn.lambda_spatial, 0.1)


if __name__ == "__main__":
    unittest.main()
