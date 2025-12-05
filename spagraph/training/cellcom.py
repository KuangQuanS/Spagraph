"""Stage 3 (cell communication) wrapper for Spagraph.

Provides a user-friendly API for cell-cell communication analysis.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

_current_dir = Path(__file__).parent.parent.parent
_sc_map_st_dir = _current_dir / "SC_MAP_ST"
if str(_sc_map_st_dir) not in sys.path:
    sys.path.insert(0, str(_sc_map_st_dir))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from spagraph.cellcom.cellcom import main as cellcom_main, parse_args


def run_cellcom(
    deconv_dir: Optional[str] = None,
    st_h5ad: Optional[str] = None,
    output_dir: Optional[str] = None,
    # MLP parameters
    mlp_latent_dim: int = 64,
    mlp_hidden_dims: str = '256,128',
    # Graph parameters
    n_spot_neighbors: int = 6,
    # LR communication parameters
    mean_expr_threshold: float = 3.0,
    min_comm_edges: int = 1,
    spot_cell_expr_csv: Optional[str] = None,
    # GAT parameters
    gat_hidden_dims: str = '512,256,128',
    gat_heads: int = 8,
    gat_dropout: float = 0.3,
    # Model parameters
    output_dim: int = 120,
    lambda_mask_recon: float = 1.0,
    lambda_node_recon: float = 0.5,
    attention_threshold: float = 1.0,
    edge_mask_ratio: float = 0.2,
    node_mask_ratio: float = 0.15,
    mask_seed: int = 1234,
    lr_id_emb_dim: int = 8,
    # Training parameters
    batch_size: int = 4,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device: str = 'cuda',
    # Legacy support
    args: Optional[Union[argparse.Namespace, Dict[str, Any]]] = None,
    **overrides: Any,
) -> None:
    """Run Stage 3 (cell communication) analysis.
    
    This function analyzes cell-cell communication based on ligand-receptor
    interactions using the deconvolution results from Stage 1 & 2.
    
    Args:
        deconv_dir: Stage1+Stage2 output directory (contains final_vae.pth, etc.)
        st_h5ad: Path to spatial transcriptomics h5ad file
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
        mean_expr_threshold=mean_expr_threshold,
        min_comm_edges=min_comm_edges,
        spot_cell_expr_csv=spot_cell_expr_csv,
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
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    
    # Apply any additional overrides
    for key, value in overrides.items():
        setattr(parsed_args, key, value)
    
    return cellcom_main(parsed_args)


# Backward-compatible alias
analyze_cellchat = run_cellcom
