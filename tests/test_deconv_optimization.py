import unittest

import numpy as np
import anndata as ad
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
from spagraph.models.deconv_model import HeterogeneousGATDeconvolution, SpatialDeconvolutionLoss
from spagraph.models.stage1 import coEncoder
from spagraph.models.stage2 import GATDeconvolution
from spagraph.training.deconv import run_deconv_auto_k


class Stage1AnnotationTests(unittest.TestCase):
    def test_explicit_celltype_key_is_resolved_without_reserved_column_name(self):
        adata = ad.AnnData(np.ones((3, 2), dtype=np.float32))
        adata.obs["manual_annotation"] = ["A", "B", "B"]
        encoder = coEncoder(celltype_key="manual_annotation", save_to_disk=False)
        self.assertEqual(encoder._resolve_celltype_key(adata, required=True), "manual_annotation")

    def test_explicit_celltype_key_fails_loudly_when_missing(self):
        adata = ad.AnnData(np.ones((2, 2), dtype=np.float32))
        encoder = coEncoder(celltype_key="manual_annotation", save_to_disk=False)
        with self.assertRaisesRegex(ValueError, "manual_annotation"):
            encoder._resolve_celltype_key(adata, required=True)


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

    def test_zero_initialized_bounded_gat_starts_from_signature(self):
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
            signature_prior_strength=1.0,
            signature_prior_trainable=False,
            signature_residual_scale=0.05,
            signature_output_power=1.2,
            zero_init_signature_residual=True,
        )
        initial = torch.tensor([[0.8, 0.2]])
        out = model(
            spot_embeddings=torch.tensor([[0.5, 0.5]]),
            spatial_coords=torch.tensor([[0.0, 0.0]]),
            celltype_prototypes=prototypes,
            initial_weights=initial,
        )
        expected = initial.pow(1.2)
        expected = expected / expected.sum(dim=1, keepdim=True)
        torch.testing.assert_close(out["deconv_weights"], expected, atol=1e-6, rtol=1e-6)
        self.assertTrue(model.signature_prior_log_scale.requires_grad is False)

    def test_signature_residual_logits_are_strictly_bounded(self):
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
            signature_prior_strength=1.0,
            signature_prior_trainable=False,
            signature_residual_scale=0.05,
        )
        with torch.no_grad():
            model.attention_mlp[-1].weight.fill_(100.0)
            model.attention_mlp[-1].bias.fill_(-100.0)
        initial = torch.tensor([[0.8, 0.2]])
        out = model(
            spot_embeddings=torch.tensor([[0.5, 0.5]]),
            spatial_coords=torch.tensor([[0.0, 0.0]]),
            celltype_prototypes=prototypes,
            initial_weights=initial,
        )
        correction = out["attention_scores"] - initial.log()
        self.assertLessEqual(float(correction.abs().max()), 0.050001)

    def test_linear_signature_residual_is_not_tanh_bounded(self):
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
            signature_prior_strength=1.0,
            signature_prior_trainable=False,
            signature_residual_scale=1.0,
            signature_residual_mode="linear",
        )
        with torch.no_grad():
            model.attention_mlp[-1].weight.zero_()
            model.attention_mlp[-1].bias.fill_(5.0)
        out = model(
            spot_embeddings=torch.tensor([[0.5, 0.5]]),
            spatial_coords=torch.tensor([[0.0, 0.0]]),
            celltype_prototypes=prototypes,
            initial_weights=torch.tensor([[0.8, 0.2]]),
        )
        self.assertGreater(float(out["signature_residual_component"].abs().max()), 4.9)

    def test_signature_prior_multiplier_controls_and_checkpoints_warm_start(self):
        model = self._model(prior_strength=1.0)
        for parameter in model.attention_mlp.parameters():
            torch.nn.init.zeros_(parameter)
        inputs = dict(
            spot_embeddings=torch.tensor([[0.5, 0.5]]),
            spatial_coords=torch.tensor([[0.0, 0.0]]),
            celltype_prototypes=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            initial_weights=torch.tensor([[0.8, 0.2]]),
        )
        model.set_signature_prior_multiplier(0.0)
        out_without_prior = model(**inputs)
        torch.testing.assert_close(
            out_without_prior["deconv_weights"],
            torch.tensor([[0.5, 0.5]]),
            atol=1e-6,
            rtol=1e-6,
        )
        state = {key: value.clone() for key, value in model.state_dict().items()}
        model.set_signature_prior_multiplier(1.0)
        model.load_state_dict(state)
        self.assertAlmostEqual(float(model.signature_prior_multiplier), 0.0)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            model.set_signature_prior_multiplier(-0.1)

    def test_signature_affinity_selects_spot_celltype_edges(self):
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        model = HeterogeneousGATDeconvolution(
            embedding_dim=2,
            n_cell_types=2,
            gat_hidden_dim=4,
            gat_layers=1,
            gat_heads=1,
            dropout=0.0,
            k_spatial=0,
            k_celltype=1,
            celltype_prototypes=prototypes,
            signature_prior_strength=1.0,
            signature_affinity_graph=True,
        )
        out = model(
            spot_embeddings=torch.tensor([[1.0, 0.0]]),
            spatial_coords=torch.tensor([[0.0, 0.0]]),
            celltype_prototypes=prototypes,
            initial_weights=torch.tensor([[0.1, 0.9]]),
        )
        outgoing = out["edge_index"][:, out["edge_index"][0].eq(0)]
        self.assertEqual(outgoing.shape[1], 1)
        # Node 0 is the spot; cell-type nodes are offset by n_spots=1.
        self.assertEqual(int(outgoing[1, 0]), 2)

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
    def test_signature_consistency_is_soft_and_zero_at_reference(self):
        loss = SpatialDeconvolutionLoss(
            celltype_expressions_full=np.array([[5, 0], [0, 5]], dtype=float),
            marker_gene_indices=[0, 1],
            scale_basis="none",
            lambda_signature_consistency=1.0,
        )
        reference = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        matching = loss(
            attention_weights=reference,
            celltype_expression=torch.empty(0),
            true_spot_expression=torch.tensor([[4.0, 1.0], [1.5, 3.5]]),
            reference_weights=reference,
        )
        shifted = loss(
            attention_weights=torch.flip(reference, dims=[1]),
            celltype_expression=torch.empty(0),
            true_spot_expression=torch.tensor([[4.0, 1.0], [1.5, 3.5]]),
            reference_weights=reference,
        )
        self.assertAlmostEqual(float(matching["signature_consistency_loss"]), 0.0)
        self.assertGreater(
            float(shifted["signature_consistency_loss"]),
            float(matching["signature_consistency_loss"]),
        )

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
            signature_residual_mode="linear",
            signature_affinity_graph=True,
            signature_prior_final_multiplier=0.25,
            signature_prior_anneal_epochs=20,
            lambda_poisson=0.1,
            lambda_spatial=0.1,
            heldout_gene_fraction=0.25,
        )
        self.assertEqual(trainer.loss_fn.heldout_marker_positions.numel(), 1)
        self.assertAlmostEqual(trainer.loss_fn.lambda_poisson, 0.1)
        self.assertAlmostEqual(trainer.loss_fn.lambda_spatial, 0.1)
        self.assertAlmostEqual(trainer.loss_fn.lambda_signature_consistency, 3.0)
        self.assertEqual(trainer.gat_model.signature_residual_mode, "linear")
        self.assertTrue(trainer.gat_model.signature_affinity_graph)
        self.assertAlmostEqual(trainer.signature_prior_final_multiplier, 0.25)
        self.assertEqual(trainer.signature_prior_anneal_epochs, 20)


if __name__ == "__main__":
    unittest.main()
