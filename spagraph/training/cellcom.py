"""Stage 3 (cell communication) wrapper for Spagraph.

Provides a user-friendly API for cell-cell communication analysis.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd

from spagraph.cellcom.cellcom import main as cellcom_main, parse_args
from spagraph.cellcom.relation_ranker import (
    DEFAULT_CALIBRATION_PROFILE,
    ensemble_lr_rankings,
)


def aggregate_cellcom_seed_outputs(
    seed_dirs: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Aggregate calibrated LR rankings produced by independent Stage3 seeds."""
    if len(seed_dirs) != len(seeds) or not seed_dirs:
        raise ValueError("seed_dirs and seeds must be non-empty and have equal length")
    frames = []
    for seed_dir in seed_dirs:
        path = Path(seed_dir) / "lr_pair_associated_edge_statistics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-seed LR statistics: {path}")
        frames.append(pd.read_csv(path))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ensemble = ensemble_lr_rankings(frames)
    ensemble["calibration_profile"] = DEFAULT_CALIBRATION_PROFILE
    ensemble_path = output / "lr_pair_ensemble_statistics.csv"
    ensemble.to_csv(ensemble_path, index=False)
    manifest = {
        "seeds": [int(value) for value in seeds],
        "n_repeats": len(seeds),
        "calibration_profile": DEFAULT_CALIBRATION_PROFILE,
        "per_seed_directories": [str(Path(value)) for value in seed_dirs],
    }
    manifest_path = output / "cellcom_ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "ensemble": ensemble,
        "ensemble_path": str(ensemble_path),
        "manifest_path": str(manifest_path),
        "seeds": manifest["seeds"],
    }


def run_cellcom(
    deconv_dir: Optional[str] = None,
    st_h5ad: Optional[str] = None,
    output_dir: Optional[str] = None,
    composition_csv: Optional[str] = None,
    # MLP parameters
    mlp_latent_dim: int = 64,
    mlp_hidden_dims: str = '256,128',
    # Graph parameters
    n_spot_neighbors: int = 10,
    # LR communication parameters
    ligand_expr_threshold: float = 3.0,  # 配体表达阈值（CP10k）
    receptor_expr_threshold: float = 1.0,  # 受体表达阈值（CP10k，通常较低）
    lr_score_threshold: float = 1,  # LR得分阈值（log1p 空间）
    min_comm_edges: int = 1,
    spot_cell_expr_csv: Optional[str] = None,  # 可选，优先使用deconv_dir中的动态表达
    use_hvg_for_communication: bool = False,  # 只使用高变基因计算通讯（默认启用）、
    allow_same_celltype_comm: bool = True,
    # GAT parameters
    gat_hidden_dims: str = '512,256,128',
    gat_heads: int = 8,
    gat_dropout: float = 0.3,
    # Model parameters
    output_dim: int = 128,
    lambda_mask_recon: float = 1.0,
    lambda_node_recon: float = 0.5,
    attention_threshold: float = 1.0,
    edge_mask_ratio: float = 0.2,
    node_mask_ratio: float = 0.15,
    mask_seed: int = 1234,
    lr_id_emb_dim: int = 8,
    model_variant: str = 'legacy',
    lambda_relation_rank: float = 0.2,
    # Training parameters
    batch_size: int = 4,
    num_workers: int = 0,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    n_repeats: int = 1,
    seeds: Optional[Sequence[int]] = None,
    device: str = 'cuda',
    sample_rate: float = 1.0,
    val_split: float = 0.1,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 0.1,
    save_lr_scores_csv: bool = False,
    export_unified_csv: bool = False,
    export_filtered_csv: bool = True,
    # Ablation flags
    ablation_no_lr_identity: bool = True,
    # Legacy support
    args: Optional[Union[argparse.Namespace, Dict[str, Any]]] = None,
    **overrides: Any,
) -> Optional[Dict[str, Any]]:
    """Run Stage 3 (cell communication) analysis.
    
    Analyzes cell-cell communication based on ligand-receptor interactions
    using the deconvolution results from Stage 2.
    
    ✅ 极简依赖（只需 2 个文件）：
    - 必需: *_spot_cell_expr.csv (Stage 2 生成，包含动态表达)
    - 必需: *_cluster_composition.csv (deconv 比例矩阵)
    
    特征构建：自动从 spot_cell_expr.csv 选择 2000 个高变基因（使用 scanpy）
    
    Args:
        deconv_dir: Stage 2 output directory, must contain:
            - *_spot_cell_expr.csv (自动生成，需设置 save_reconstructed_genes=True)
            - *_cluster_composition.csv (deconv 结果)
        st_h5ad: Spatial transcriptomics h5ad file path
        output_dir: Output directory for results
        mlp_latent_dim: MLP latent dimension
        mlp_hidden_dims: MLP hidden dimensions (comma-separated)
        n_spot_neighbors: Number of spot neighbors
        mean_expr_threshold: Mean expression threshold for gene filtering
        min_comm_edges: Minimum communication edges threshold
        spot_cell_expr_csv: Pre-computed spot-cell expression CSV (optional)
        save_lr_scores_csv: Whether to save Stage 3.4 lr_scores.csv
        export_unified_csv: Whether to export full lr_communication.csv
        export_filtered_csv: Whether to export filtered lr_communication CSV
        gat_hidden_dims: GAT hidden dimensions (comma-separated)
        gat_heads: Number of attention heads
        gat_dropout: Dropout probability
        output_dim: Output dimension
        lambda_mask_recon: Mask reconstruction loss weight
        lambda_node_recon: Node reconstruction loss weight
        attention_threshold: Attention score threshold for edge filtering
        edge_mask_ratio: Edge mask ratio
        node_mask_ratio: Node mask ratio
        mask_seed: Random seed for masking
        lr_id_emb_dim: LR ID embedding dimension
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        seed: Random seed for a single run or deterministic repeat-seed generation
        n_repeats: Number of independent Stage3 runs to ensemble
        seeds: Explicit unique seeds; when supplied, overrides generated repeat seeds
        device: Computing device ('cuda' or 'cpu')
        args: Legacy argparse.Namespace or dict (for backward compatibility)
        **overrides: Additional overrides for arguments
    
    Returns:
        Single runs retain the legacy return value. Multi-seed runs return the
        ensemble table/path, manifest, seeds, and per-seed results.
    
    Example:
        >>> import spagraph
        >>> # After running deconvolution
        >>> spagraph.cellcom(
        ...     deconv_dir="output/deconv/",
        ...     st_h5ad="data/st.h5ad",
        ...     output_dir="output/cellcom/",
        ...     epochs=100,
        ...     batch_size=4
        ... )
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if seeds is not None:
        seed_values = [int(value) for value in seeds]
        if not seed_values:
            raise ValueError("seeds cannot be empty")
        if len(set(seed_values)) != len(seed_values):
            raise ValueError("seeds must be unique")
        if n_repeats not in {1, len(seed_values)}:
            raise ValueError("n_repeats must be 1 or match len(seeds)")
    elif n_repeats > 1:
        rng = random.Random(seed)
        seed_values = rng.sample(range(1, 2**31 - 1), n_repeats)
    else:
        seed_values = [int(seed)]

    if len(seed_values) > 1:
        if args is not None:
            raise ValueError("multi-seed execution is unavailable with legacy args; use keyword arguments")
        if output_dir is None:
            if deconv_dir is None:
                raise ValueError("deconv_dir is required")
            output_dir = str(Path(deconv_dir) / "cellcom")
        base_kwargs = dict(
            deconv_dir=deconv_dir, st_h5ad=st_h5ad, composition_csv=composition_csv,
            mlp_latent_dim=mlp_latent_dim, mlp_hidden_dims=mlp_hidden_dims,
            n_spot_neighbors=n_spot_neighbors, ligand_expr_threshold=ligand_expr_threshold,
            receptor_expr_threshold=receptor_expr_threshold, lr_score_threshold=lr_score_threshold,
            min_comm_edges=min_comm_edges, spot_cell_expr_csv=spot_cell_expr_csv,
            use_hvg_for_communication=use_hvg_for_communication,
            allow_same_celltype_comm=allow_same_celltype_comm,
            gat_hidden_dims=gat_hidden_dims, gat_heads=gat_heads, gat_dropout=gat_dropout,
            output_dim=output_dim, lambda_mask_recon=lambda_mask_recon,
            lambda_node_recon=lambda_node_recon, attention_threshold=attention_threshold,
            edge_mask_ratio=edge_mask_ratio, node_mask_ratio=node_mask_ratio,
            mask_seed=mask_seed, lr_id_emb_dim=lr_id_emb_dim, model_variant=model_variant,
            lambda_relation_rank=lambda_relation_rank, batch_size=batch_size,
            num_workers=num_workers, epochs=epochs, learning_rate=learning_rate,
            weight_decay=weight_decay, device=device, sample_rate=sample_rate,
            val_split=val_split, early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta, save_lr_scores_csv=save_lr_scores_csv,
            export_unified_csv=export_unified_csv, export_filtered_csv=export_filtered_csv,
            ablation_no_lr_identity=ablation_no_lr_identity,
        )
        base_kwargs.update(overrides)
        seed_dirs = []
        seed_results = []
        for repeat_seed in seed_values:
            seed_dir = Path(output_dir) / f"seed_{repeat_seed}"
            seed_dirs.append(seed_dir)
            seed_results.append(
                run_cellcom(
                    output_dir=str(seed_dir), seed=repeat_seed, n_repeats=1,
                    seeds=None, **base_kwargs
                )
            )
        result = aggregate_cellcom_seed_outputs(seed_dirs, output_dir, seed_values)
        result["seed_results"] = seed_results
        return result

    seed = seed_values[0]

    # Legacy support: if args is provided, use it directly
    if args is not None:
        if isinstance(args, dict):
            parsed_args = argparse.Namespace(**args)
        else:
            parsed_args = args
        # Apply overrides
        for key, value in overrides.items():
            setattr(parsed_args, key, value)
        return cellcom_main(parsed_args)
    
    # Build args from keyword arguments
    if deconv_dir is None:
        raise ValueError("deconv_dir is required (Stage1+Stage2 output directory)")
    if st_h5ad is None:
        raise ValueError("st_h5ad is required (spatial transcriptomics h5ad file)")
    if output_dir is None:
        output_dir = str(Path(deconv_dir) / "cellcom")
    if model_variant not in {'legacy', 'relation_ranker'}:
        raise ValueError("model_variant must be 'legacy' or 'relation_ranker'")
    
    os.makedirs(output_dir, exist_ok=True)
    
    parsed_args = argparse.Namespace(
        deconv_dir=deconv_dir,
        st_h5ad=st_h5ad,
        output_dir=output_dir,
        composition_csv=composition_csv,
        mlp_latent_dim=mlp_latent_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        n_spot_neighbors=n_spot_neighbors,
        ligand_expr_threshold=ligand_expr_threshold,
        receptor_expr_threshold=receptor_expr_threshold,
        lr_score_threshold=lr_score_threshold,
        min_comm_edges=min_comm_edges,
        spot_cell_expr_csv=spot_cell_expr_csv,
        save_lr_scores_csv=save_lr_scores_csv,
        export_unified_csv=export_unified_csv,
        export_filtered_csv=export_filtered_csv,
        use_hvg_for_communication=use_hvg_for_communication,
        allow_same_celltype_comm=allow_same_celltype_comm,
        gat_hidden_dims=gat_hidden_dims,
        gat_heads=gat_heads,
        gat_dropout=gat_dropout,
        output_dim=output_dim,
        lambda_mask_recon=lambda_mask_recon,
        lambda_node_recon=lambda_node_recon,
        attention_threshold=attention_threshold,
        edge_mask_ratio=edge_mask_ratio,
        node_mask_ratio=node_mask_ratio,
        mask_seed=mask_seed,
        lr_id_emb_dim=lr_id_emb_dim,
        model_variant=model_variant,
        lambda_relation_rank=lambda_relation_rank,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        sample_rate=sample_rate,
        val_split=val_split,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        ablation_no_lr_identity=ablation_no_lr_identity,
    )
    
    # Apply any additional overrides
    for key, value in overrides.items():
        setattr(parsed_args, key, value)
    
    return cellcom_main(parsed_args)


def run_cellcom_ensemble(
    *,
    seeds: Sequence[int] = (11, 23, 42, 67, 101),
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run public Stage3 repeatedly and return a calibrated seed ensemble."""
    return run_cellcom(seeds=seeds, n_repeats=len(seeds), **kwargs)


# Backward-compatible alias
analyze_cellchat = run_cellcom
