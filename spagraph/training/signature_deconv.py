"""Fast annotation-guided reference-signature deconvolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from spagraph.models.deconv_initialization import (
    compute_platform_calibrated_initialization,
    compute_signature_initialization,
    select_celltype_specific_genes,
)


def _as_nonnegative_matrix(matrix):
    matrix = matrix.copy() if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float64)
    if sparse.issparse(matrix):
        if matrix.data.size and (not np.isfinite(matrix.data).all() or matrix.data.min() < 0):
            raise ValueError("expression must contain finite non-negative values")
        return matrix.astype(np.float64)
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("expression must contain finite non-negative values")
    return matrix


def _row_normalize(matrix):
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(totals, dtype=np.float64)
    positive = totals > 0
    inverse[positive] = 1.0 / totals[positive]
    if sparse.issparse(matrix):
        return sparse.diags(inverse) @ matrix
    return matrix * inverse[:, None]


def _group_means(matrix, encoded_labels: np.ndarray, n_groups: int) -> np.ndarray:
    rows = []
    for group_id in range(n_groups):
        group = matrix[encoded_labels == group_id]
        rows.append(np.asarray(group.mean(axis=0)).ravel())
    return np.vstack(rows)


def run_signature_deconv(
    sc_file: str,
    st_file: str,
    celltype_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    sample_name: Optional[str] = None,
    gene_selection: str = "celltype_specific",
    genes_per_celltype: int = 200,
    reference_scale: str = "log_normalized",
    platform_calibration: bool = True,
    calibration_iterations: int = 5,
    ridge: float = 1e-4,
    composition_power: float = 1.2,
) -> dict:
    """Deconvolve spots directly from annotated single-cell signatures.

    This deterministic route is intended for datasets with trusted reference
    annotations. It avoids VAE training and graph construction because neither
    is used by a signature-only model.
    """
    if gene_selection not in {"celltype_specific", "all_shared"}:
        raise ValueError("gene_selection must be 'celltype_specific' or 'all_shared'")
    if reference_scale not in {"log_normalized", "cell_normalized"}:
        raise ValueError("reference_scale must be 'log_normalized' or 'cell_normalized'")
    if genes_per_celltype < 1:
        raise ValueError("genes_per_celltype must be at least 1")
    if not np.isfinite(composition_power) or composition_power <= 0:
        raise ValueError("composition_power must be finite and positive")

    sc_adata = sc.read_h5ad(sc_file)
    st_adata = sc.read_h5ad(st_file)
    sc_adata.var_names_make_unique()
    st_adata.var_names_make_unique()
    if celltype_key is None:
        celltype_key = next(
            (key for key in ("cell_type", "celltype") if key in sc_adata.obs.columns), None
        )
    if celltype_key is None or celltype_key not in sc_adata.obs.columns:
        raise ValueError("celltype_key is required and must exist in scRNA obs")
    labels = sc_adata.obs[celltype_key]
    if labels.isna().any():
        raise ValueError("cell-type annotations cannot contain missing values")

    shared_genes = [gene for gene in sc_adata.var_names if gene in st_adata.var_names]
    if not shared_genes:
        raise ValueError("scRNA and ST contain no shared genes")
    # Match Stage 1 semantics: normalize each reference cell across its complete
    # measured transcriptome, then align the normalized matrix to shared genes.
    sc_expression = _as_nonnegative_matrix(sc_adata.X)
    cell_normalized_full = _row_normalize(sc_expression)
    sc_shared_indices = np.asarray(
        [sc_adata.var_names.get_loc(gene) for gene in shared_genes], dtype=np.int64
    )
    cell_normalized = cell_normalized_full[:, sc_shared_indices]
    encoder = LabelEncoder().fit(labels.astype(str).to_numpy())
    encoded = encoder.transform(labels.astype(str).to_numpy())
    n_celltypes = len(encoder.classes_)
    selection_signatures = _group_means(cell_normalized, encoded, n_celltypes)

    if gene_selection == "celltype_specific":
        selected = select_celltype_specific_genes(
            selection_signatures, top_per_celltype=genes_per_celltype
        )
    else:
        selected = np.arange(len(shared_genes), dtype=np.int64)
    selected_genes = [shared_genes[index] for index in selected]

    if reference_scale == "log_normalized":
        logged = cell_normalized_full.copy()
        if sparse.issparse(logged):
            logged = logged.tocsr()
            logged.data = np.log1p(logged.data * 1e4)
        else:
            logged = np.log1p(logged * 1e4)
        signatures = _group_means(
            logged[:, sc_shared_indices[selected]], encoded, n_celltypes
        )
        spot_expression = _as_nonnegative_matrix(st_adata[:, selected_genes].X)
        spot_expression = _row_normalize(spot_expression)
        if sparse.issparse(spot_expression):
            spot_expression = spot_expression.toarray()
        spot_expression = np.log1p(np.asarray(spot_expression) * 1e4)
    else:
        signatures = selection_signatures[:, selected]
        spot_expression = _as_nonnegative_matrix(st_adata[:, selected_genes].X)
        spot_expression = spot_expression.toarray() if sparse.issparse(spot_expression) else spot_expression

    if platform_calibration:
        weights, gene_factors = compute_platform_calibrated_initialization(
            spot_expression,
            signatures,
            ridge=ridge,
            iterations=calibration_iterations,
        )
    else:
        weights = compute_signature_initialization(
            spot_expression, signatures, ridge=ridge
        )
        gene_factors = None

    from spagraph.models.deconv_initialization import power_calibrate_composition

    weights = power_calibrate_composition(weights, power=composition_power)
    composition = pd.DataFrame(weights, index=st_adata.obs_names, columns=encoder.classes_)
    composition_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        name = sample_name or Path(st_file).stem
        composition_path = os.path.join(output_dir, f"{name}_composition.csv")
        composition.to_csv(composition_path)
    return {
        "deconv": composition,
        "deconv_path": composition_path,
        "sample_name": sample_name or Path(st_file).stem,
        "n_clusters": n_celltypes,
        "best_epoch": 0,
        "graph_source": "not_used_signature_only",
        "reference_grouping": "celltype",
        "reference_signature_mode": reference_scale,
        "signature_gene_selection": gene_selection,
        "signature_selected_genes": selected_genes,
        "signature_gene_factors": gene_factors,
        "signature_composition_power": composition_power,
        "deconv_weights_raw": weights,
    }
