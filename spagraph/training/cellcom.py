"""Stage 3 (cell communication) wrapper for Spagraph.

Provides a user-friendly API for cell-cell communication analysis.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from spagraph.cellcom.cellcom import main as cellcom_main, parse_args


def run_cellcom(
    deconv_dir: Optional[str] = None,
    st_h5ad: Optional[str] = None,
    output_dir: Optional[str] = None,
    # MLP parameters
    mlp_latent_dim: int = 64,
    mlp_hidden_dims: str = '256,128',
    # Graph parameters
    n_spot_neighbors: int = 10,
    # LR communication parameters
    ligand_expr_threshold: float = 3.0,  # 配体表达阈值（CP10k）
    receptor_expr_threshold: float = 1.0,  # 受体表达阈值（CP10k，通常较低）
    lr_score_threshold: float = 0,  # LR得分阈值（log1p 空间）
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
    # Training parameters
    batch_size: int = 4,
    num_workers: int = 0,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device: str = 'cuda',
    sample_rate: float = 1.0,
    val_split: float = 0.1,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 0.1,
    # Legacy support
    args: Optional[Union[argparse.Namespace, Dict[str, Any]]] = None,
    **overrides: Any,
) -> None:
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
        seed: Random seed
        device: Computing device ('cuda' or 'cpu')
        args: Legacy argparse.Namespace or dict (for backward compatibility)
        **overrides: Additional overrides for arguments
    
    Returns:
        None (results are saved to output_dir)
    
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    parsed_args = argparse.Namespace(
        deconv_dir=deconv_dir,
        st_h5ad=st_h5ad,
        output_dir=output_dir,
        mlp_latent_dim=mlp_latent_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        n_spot_neighbors=n_spot_neighbors,
        ligand_expr_threshold=ligand_expr_threshold,
        receptor_expr_threshold=receptor_expr_threshold,
        lr_score_threshold=lr_score_threshold,
        min_comm_edges=min_comm_edges,
        spot_cell_expr_csv=spot_cell_expr_csv,
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
    )
    
    # Apply any additional overrides
    for key, value in overrides.items():
        setattr(parsed_args, key, value)
    
    return cellcom_main(parsed_args)


# Backward-compatible alias
analyze_cellchat = run_cellcom
