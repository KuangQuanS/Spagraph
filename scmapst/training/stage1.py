"""Stage 1: VAE training for SC-ST integration

This module wraps the stage1.py functionality into a clean API.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directories to path
_current_dir = Path(__file__).parent.parent.parent
_sc_map_st_dir = _current_dir / "SC_MAP_ST"
if str(_sc_map_st_dir) not in sys.path:
    sys.path.insert(0, str(_sc_map_st_dir))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from SC_MAP_ST.stage1 import coEncoder


def train_integration(
    sc_file: str,
    st_file: str,
    output_dir: str = "./deconv_results",
    n_epochs: int = 150,
    resolution: float = 4.0,
    top_n_per_type: int = 100,
    hidden_dims: Optional[list] = None,
    latent_dim: int = 128,
    batch_size: int = 512,
    lr: float = 5e-4,
    beta: float = 0.1,
    loss_type: str = 'mse',
    lambda_mmd: float = 0.1,
    use_dual_decoder: bool = True,
    pretrained_path: Optional[str] = None,
    precomputed_marker_file: Optional[str] = None,
    aggregation_method: str = 'weighted',
    marker_selection_method: str = 'correlation',
    device: Optional[str] = None
) -> Dict[str, Any]:
    """Train VAE model for SC-ST integration (Stage 1)
    
    Args:
        sc_file: Path to single-cell h5ad file
        st_file: Path to spatial transcriptomics h5ad file
        output_dir: Output directory path
        n_epochs: Number of training epochs
        resolution: Leiden clustering resolution for auto-clustering
        top_n_per_type: Number of marker genes per cluster
        hidden_dims: VAE hidden layer dimensions (default: [512, 256])
        latent_dim: VAE latent space dimension
        batch_size: Training batch size
        lr: Learning rate
        beta: KL divergence weight (beta-VAE)
        loss_type: Reconstruction loss type ('mse' or 'zinb')
        lambda_mmd: MMD loss weight for modality alignment
        use_dual_decoder: Use DualDecoderVAE with separate SC/ST decoders
        pretrained_path: Path to pretrained VAE model checkpoint
        precomputed_marker_file: Path to precomputed marker genes file
        aggregation_method: Cluster aggregation method ('mean', 'median', 'weighted')
        marker_selection_method: Marker gene selection method ('l1', 'variance', 'correlation')
        device: Computing device ('cuda', 'cpu', or None for auto-select)
    
    Returns:
        Dictionary containing training results:
            - best_loss: Best validation loss
            - n_genes: Number of marker genes selected
            - n_clusters: Number of cell type clusters identified
            - model_path: Path to saved VAE model
            - cluster_data_path: Path to cluster data NPZ file
            - clusters: List of cluster names
    
    Example:
        >>> import scmapst
        >>> results = scmapst.train(
        ...     sc_file="data/sc.h5ad",
        ...     st_file="data/st.h5ad",
        ...     output_dir="output/stage1/",
        ...     n_epochs=150,
        ...     resolution=4.0
        ... )
        >>> print(f"Model saved to: {results['model_path']}")
    """
    if hidden_dims is None:
        hidden_dims = [512, 256]
    
    # Create encoder instance
    encoder = coEncoder(
        sc_file=sc_file,
        st_file=st_file,
        output_dir=output_dir,
        device=device
    )
    
    # Run training
    results = encoder.run_stage1_training(
        top_n_per_type=top_n_per_type,
        resolution=resolution,
        batch_size=batch_size,
        n_epochs=n_epochs,
        lr=lr,
        beta=beta,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        loss_type=loss_type,
        lambda_mmd=lambda_mmd,
        pretrained_path=pretrained_path,
        precomputed_marker_file=precomputed_marker_file,
        use_dual_decoder=use_dual_decoder,
        aggregation_method=aggregation_method,
        marker_selection_method=marker_selection_method
    )
    
    return results
