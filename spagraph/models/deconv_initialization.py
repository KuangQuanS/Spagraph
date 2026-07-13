"""Reference-signature initialization and robust deconvolution losses.

These helpers never use simulated composition ground truth.  They derive an
initial simplex composition solely from the observed spot expression and the
single-cell reference signatures produced by Stage 1.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls
from sklearn.preprocessing import LabelEncoder


def _row_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    return values / np.maximum(totals, eps)


def compute_signature_initialization(
    spot_expression: np.ndarray,
    celltype_signatures: np.ndarray,
    gene_indices: Optional[Sequence[int]] = None,
    gene_weights: Optional[np.ndarray] = None,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Estimate non-negative spot compositions from reference signatures.

    Parameters
    ----------
    spot_expression
        Spots by genes, on any non-negative linear scale.
    celltype_signatures
        Cell types by genes on the same feature ordering.
    gene_indices
        Optional subset used for fitting (for held-out-gene validation).
    ridge
        Small L2 penalty used to stabilize collinear cell-type signatures.
    """
    spots = np.asarray(spot_expression, dtype=np.float64)
    signatures = np.asarray(celltype_signatures, dtype=np.float64)
    if spots.ndim != 2 or signatures.ndim != 2:
        raise ValueError("spot_expression and celltype_signatures must be 2-D")
    if spots.shape[1] != signatures.shape[1]:
        raise ValueError("spot and signature gene dimensions must match")
    if np.any(spots < 0) or np.any(signatures < 0):
        raise ValueError("signature initialization requires non-negative values")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be a finite non-negative value")

    if gene_indices is not None:
        idx = np.asarray(list(gene_indices), dtype=np.int64)
        if idx.size == 0:
            raise ValueError("gene_indices cannot be empty")
        spots = spots[:, idx]
        signatures = signatures[:, idx]
        if gene_weights is not None:
            gene_weights = np.asarray(gene_weights, dtype=np.float64)[idx]

    spots = _row_normalize(spots)
    signatures = _row_normalize(signatures)
    design = signatures.T
    if gene_weights is not None:
        gene_weights = np.asarray(gene_weights, dtype=np.float64)
        if gene_weights.shape != (spots.shape[1],):
            raise ValueError("gene_weights must have one value per fitted gene")
        if np.any(gene_weights < 0) or not np.isfinite(gene_weights).all():
            raise ValueError("gene_weights must be finite and non-negative")
        feature_scale = np.sqrt(gene_weights)
        design = design * feature_scale[:, None]
        spots = spots * feature_scale[None, :]
    n_celltypes = signatures.shape[0]
    if ridge > 0:
        design = np.vstack([design, np.sqrt(ridge) * np.eye(n_celltypes)])

    estimates = np.zeros((spots.shape[0], n_celltypes), dtype=np.float64)
    for i, target in enumerate(spots):
        if ridge > 0:
            target = np.concatenate([target, np.zeros(n_celltypes, dtype=np.float64)])
        weights, _ = nnls(design, target)
        total = weights.sum()
        estimates[i] = weights / total if total > 0 else 1.0 / n_celltypes
    return estimates.astype(np.float32)


def signature_specificity_weights(signatures: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Generic entropy weights that emphasize cell-type-discriminative genes."""
    signatures = np.asarray(signatures, dtype=np.float64)
    if signatures.ndim != 2 or signatures.shape[0] < 2:
        raise ValueError("signatures must contain at least two reference groups")
    gene_totals = signatures.sum(axis=0, keepdims=True)
    probabilities = signatures / np.maximum(gene_totals, 1e-12)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=0)
    specificity = 1.0 - entropy / np.log(signatures.shape[0])
    expressed = gene_totals.ravel() > 0
    weights = np.where(expressed, np.maximum(specificity, floor), 0.0)
    positive = weights > 0
    if positive.any():
        weights[positive] /= weights[positive].mean()
    return weights.astype(np.float32)


def select_celltype_specific_genes(
    signatures: np.ndarray,
    top_per_celltype: int = 100,
    min_total_expression: float = 0.0,
) -> np.ndarray:
    """Select a balanced union of annotation-specific genes without spot truth.

    The score combines relative enrichment against all other reference groups
    with within-group expression, preventing tiny but high-ratio values from
    dominating the selected set.
    """
    signatures = _row_normalize(np.maximum(np.asarray(signatures, dtype=np.float64), 0.0))
    if signatures.ndim != 2 or signatures.shape[0] < 2:
        raise ValueError("signatures must contain at least two cell types")
    if top_per_celltype < 1:
        raise ValueError("top_per_celltype must be at least 1")
    totals = signatures.sum(axis=0)
    other_mean = (totals[None, :] - signatures) / (signatures.shape[0] - 1)
    log_enrichment = np.log((signatures + 1e-12) / (other_mean + 1e-12))
    scores = signatures * np.maximum(log_enrichment, 0.0)
    scores[:, totals <= float(min_total_expression)] = -np.inf

    selected = set()
    take = min(int(top_per_celltype), signatures.shape[1])
    for row in scores:
        finite = np.flatnonzero(np.isfinite(row))
        if finite.size == 0:
            continue
        count = min(take, finite.size)
        candidates = finite[np.argpartition(row[finite], -count)[-count:]]
        selected.update(int(index) for index in candidates if row[index] > 0)
    if not selected:
        raise ValueError("no cell-type-specific genes passed the selection criteria")
    return np.asarray(sorted(selected), dtype=np.int64)


def _project_rows_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection of every row onto the probability simplex."""
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values, axis=1)[:, ::-1]
    cssv = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1, dtype=np.float64)
    positive = ordered - cssv / indices[None, :] > 0
    rho = positive.sum(axis=1) - 1
    theta = cssv[np.arange(len(values)), rho] / (rho + 1.0)
    return np.maximum(values - theta[:, None], 0.0)


def power_calibrate_composition(values: np.ndarray, power: float = 1.2) -> np.ndarray:
    """Sharpen or soften simplex compositions without using benchmark truth.

    A power above one reduces diffuse low-probability mass; a power below one
    softens the composition. Rows are projected back to a valid simplex.
    """
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    calibrated = np.maximum(np.asarray(values, dtype=np.float64), 0.0) ** float(power)
    if calibrated.ndim != 2 or calibrated.shape[1] == 0:
        raise ValueError("composition must be a non-empty 2-D matrix")
    totals = calibrated.sum(axis=1, keepdims=True)
    zero_rows = totals.ravel() <= 0
    if np.any(zero_rows):
        calibrated[zero_rows] = 1.0 / calibrated.shape[1]
        totals = calibrated.sum(axis=1, keepdims=True)
    return (calibrated / totals).astype(np.float32)


def compute_batched_simplex_initialization(
    spot_expression: np.ndarray,
    celltype_signatures: np.ndarray,
    gene_weights: Optional[np.ndarray] = None,
    ridge: float = 1e-4,
    max_iter: int = 300,
    tolerance: float = 1e-7,
    initial: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fast batched projected-gradient mixture regression on the simplex."""
    spots = _row_normalize(np.asarray(spot_expression, dtype=np.float64))
    signatures = _row_normalize(np.asarray(celltype_signatures, dtype=np.float64))
    if spots.ndim != 2 or signatures.ndim != 2 or spots.shape[1] != signatures.shape[1]:
        raise ValueError("spot and signature matrices must be aligned 2-D arrays")
    if gene_weights is not None:
        gene_weights = np.asarray(gene_weights, dtype=np.float64)
        if gene_weights.shape != (spots.shape[1],):
            raise ValueError("gene_weights must have one value per gene")
        feature_scale = np.sqrt(np.maximum(gene_weights, 0.0))
        spots = spots * feature_scale[None, :]
        signatures = signatures * feature_scale[None, :]

    gram = signatures @ signatures.T
    gram.flat[:: gram.shape[0] + 1] += ridge
    cross = spots @ signatures.T
    lipschitz = max(float(np.linalg.eigvalsh(gram).max()), 1e-8)
    if initial is None:
        composition = np.full(
            (spots.shape[0], signatures.shape[0]), 1.0 / signatures.shape[0], dtype=np.float64
        )
    else:
        composition = _project_rows_to_simplex(np.asarray(initial, dtype=np.float64))

    for _ in range(max_iter):
        updated = _project_rows_to_simplex(
            composition - (composition @ gram - cross) / lipschitz
        )
        if np.max(np.abs(updated - composition)) < tolerance:
            composition = updated
            break
        composition = updated
    return composition.astype(np.float32)


def compute_platform_calibrated_initialization(
    spot_expression: np.ndarray,
    celltype_signatures: np.ndarray,
    ridge: float = 1e-4,
    iterations: int = 5,
    damping: float = 0.5,
    min_factor: float = 0.25,
    max_factor: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating NNLS with unsupervised gene-wise SC-to-ST calibration.

    The calibration uses only aggregate observed ST expression and reference
    signatures. No spot composition labels or benchmark truth are accessed.
    """
    spots = np.asarray(spot_expression, dtype=np.float64)
    signatures = np.asarray(celltype_signatures, dtype=np.float64)
    if spots.ndim != 2 or signatures.ndim != 2 or spots.shape[1] != signatures.shape[1]:
        raise ValueError("spot and signature matrices must be aligned 2-D arrays")
    if iterations < 0 or not 0 < damping <= 1:
        raise ValueError("iterations must be non-negative and damping must be in (0, 1]")
    factors = np.ones(signatures.shape[1], dtype=np.float64)
    weights = signature_specificity_weights(signatures)
    observed = _row_normalize(spots)

    compositions = None
    for _ in range(iterations):
        adjusted = signatures * factors[None, :]
        compositions = compute_batched_simplex_initialization(
            spots, adjusted, gene_weights=weights, ridge=ridge, initial=compositions
        )
        adjusted_norm = _row_normalize(adjusted)
        predicted = compositions @ adjusted_norm
        ratio = observed.mean(axis=0) / np.maximum(predicted.mean(axis=0), 1e-8)
        ratio = np.clip(ratio, 0.5, 2.0)
        factors *= np.power(ratio, damping)
        factors = np.clip(factors, min_factor, max_factor)
        positive = factors > 0
        factors[positive] /= np.exp(np.mean(np.log(factors[positive])))

    adjusted = signatures * factors[None, :]
    compositions = compute_batched_simplex_initialization(
        spots, adjusted, gene_weights=weights, ridge=ridge, initial=compositions
    )
    return compositions, factors.astype(np.float32)


def aggregate_reference_by_labels(
    labels: Sequence[str],
    embeddings: np.ndarray,
    marker_expression: np.ndarray,
    raw_expression: np.ndarray,
) -> dict:
    """Aggregate aligned single cells into annotation-level references."""
    labels = np.asarray(labels).astype(str)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    marker_expression = np.asarray(marker_expression, dtype=np.float32)
    raw_expression = np.asarray(raw_expression, dtype=np.float32)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty 1-D array")
    matrices = (embeddings, marker_expression, raw_expression)
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("reference expression and embeddings must be 2-D")
    if any(len(matrix) != len(labels) for matrix in matrices):
        raise ValueError("aligned reference artifacts have inconsistent row counts")
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("reference artifacts must contain only finite values")

    encoder = LabelEncoder().fit(labels)
    encoded = encoder.transform(labels)
    prototypes = []
    marker_signatures = []
    raw_signatures = []
    cell_normalized_signatures = []
    log_normalized_signatures = []
    normalized_raw = _row_normalize(np.maximum(raw_expression, 0.0))
    for group_id in range(len(encoder.classes_)):
        mask = encoded == group_id
        prototypes.append(embeddings[mask].mean(axis=0))
        marker_signatures.append(marker_expression[mask].mean(axis=0))
        raw_signatures.append(raw_expression[mask].mean(axis=0))
        cell_normalized_signatures.append(normalized_raw[mask].mean(axis=0))
        log_normalized_signatures.append(np.log1p(normalized_raw[mask] * 1e4).mean(axis=0))
    return {
        "encoder": encoder,
        "encoded_labels": encoded,
        "prototypes": np.stack(prototypes).astype(np.float32),
        "marker_signatures": np.stack(marker_signatures).astype(np.float32),
        "raw_signatures": np.stack(raw_signatures).astype(np.float32),
        "cell_normalized_signatures": np.stack(cell_normalized_signatures).astype(np.float32),
        "log_normalized_signatures": np.stack(log_normalized_signatures).astype(np.float32),
    }


def poisson_deviance_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Poisson deviance, stable for zero counts and positive predictions."""
    prediction = prediction.clamp_min(1e-8)
    target = target.clamp_min(0)
    log_ratio = torch.where(target > 0, target * (torch.log(target + 1e-8) - torch.log(prediction)), 0.0)
    return 2.0 * (prediction - target + log_ratio).mean()


def boundary_aware_graph_loss(
    weights: torch.Tensor,
    edge_index: torch.Tensor,
    expression: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Smooth compositions across similar neighbors while preserving boundaries."""
    if edge_index.numel() == 0:
        return weights.new_zeros(())
    src, dst = edge_index
    expr_distance = 1.0 - F.cosine_similarity(expression[src], expression[dst], dim=-1)
    affinity = torch.exp(-expr_distance.clamp_min(0) / max(float(temperature), 1e-8))
    composition_distance = (weights[src] - weights[dst]).abs().mean(dim=-1)
    return (affinity * composition_distance).mean()
