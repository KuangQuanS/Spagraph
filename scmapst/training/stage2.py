"""Unified spatial deconvolution API

This module provides a unified API that combines Stage 1 (VAE) and Stage 2 (GAT) into one function.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import scanpy as sc

# Add parent directories to path
_current_dir = Path(__file__).parent.parent.parent
_sc_map_st_dir = _current_dir / "SC_MAP_ST"
if str(_sc_map_st_dir) not in sys.path:
    sys.path.insert(0, str(_sc_map_st_dir))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from SC_MAP_ST.stage1 import coEncoder
from SC_MAP_ST.stage2 import GATDeconvolution


def deconvolve_spots(
    sc_file: str,
    st_file: str,
    output_dir: str = "./deconv_results",
    # Stage 1 parameters
    stage1_epochs: int = 150,
    resolution: float = 4,
    top_n_per_type: int = 100,
    # Stage 2 parameters
    stage2_epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 512,
    k_spatial: int = 20,
    k_celltype: int = 10,
    weight_threshold: float = 0.01,
    scale_basis: str = 'marker',
    # Optional
    stage1_model_path: Optional[str] = None,
    celltype_key: Optional[str] = None,
    device: Optional[str] = None,
    print_every: int = 50
) -> Dict[str, Any]:
    """One-step spatial deconvolution: Stage 1 (VAE) + Stage 2 (GAT)
    
    This function automatically runs both stages. If stage1_model_path is provided,
    it skips Stage 1 and uses the existing model.
    
    Args:
        sc_file: Path to single-cell h5ad file
        st_file: Path to spatial transcriptomics h5ad file
        output_dir: Output directory path
        stage1_epochs: Number of VAE training epochs (Stage 1)
        resolution: Leiden clustering resolution
        top_n_per_type: Number of marker genes per cluster
        stage2_epochs: Number of GAT training epochs (Stage 2)
        lr: Learning rate for GAT
        batch_size: Training batch size
        k_spatial: Number of spatial neighbors for GAT
        k_celltype: Number of cell type neighbors
        weight_threshold: Threshold for sparsifying weights
        scale_basis: Gene set for scaling ('all', 'marker', 'hvg', 'none')
        stage1_model_path: Path to existing Stage 1 model (skip Stage 1 if provided)
        celltype_key: Column name for cell type labels
        device: Computing device ('cuda', 'cpu', or None for auto)
        print_every: Print loss every N epochs (default: 50)
    
    Returns:
        Dictionary containing:
            - n_clusters: Number of cell type clusters
            - n_genes: Number of marker genes
            - best_pearson: Best Pearson loss
            - best_mse: Best MSE loss
            - best_cosine: Best cosine loss
            - model_path: Path to VAE model
            - deconv_weights_path: Path to deconvolution weights
    
    Example:
        >>> import scmapst
        >>> results = scmapst.deconvolve(
        ...     sc_file="data/sc.h5ad",
        ...     st_file="data/st.h5ad",
        ...     output_dir="output/"
        ... )
        >>> print(f"Found {results['n_clusters']} cell types")
        >>> print(f"Pearson: {results['best_pearson']:.4f}")
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_name = Path(st_file).stem
    
    # ========== Stage 1: VAE Training ==========
    if stage1_model_path is None:
        encoder = coEncoder(
            sc_file=sc_file,
            st_file=st_file,
            output_dir=output_dir,
            device=device
        )
        
        stage1_results = encoder.run_stage1_training(
            top_n_per_type=top_n_per_type,
            resolution=resolution,
            batch_size=batch_size,
            n_epochs=stage1_epochs,
            lr=5e-4,
            beta=0.1,
            hidden_dims=[512, 256],
            latent_dim=128,
            loss_type='mse',
            lambda_mmd=0.1,
            use_dual_decoder=True,
            aggregation_method='weighted',
            marker_selection_method='l1',
            print_every=print_every
        )
        
        stage1_model_path = stage1_results['model_path']
        n_clusters = stage1_results['n_clusters']
        n_genes = stage1_results['n_genes']
    else:
        n_clusters = None
        n_genes = None
    
    # ========== Stage 2: GAT Deconvolution ==========
    print(f"Stage 2 Deconvolution: {sample_name}")
    
    # Load ST data
    st_adata = sc.read_h5ad(st_file)
    st_adata.var_names_make_unique()
    
    # Initialize GAT
    gat_deconv = GATDeconvolution(
        stage1_model_path=stage1_model_path,
        output_dir=output_dir,
        device=device,
        weight_threshold=weight_threshold
    )
    
    gat_deconv.k_spatial = k_spatial
    gat_deconv.k_celltype = k_celltype
    
    # Load components
    gat_deconv.load_vae_encoder()
    gat_deconv.load_cluster_data(celltype_key=celltype_key)
    
    if n_clusters is None:
        n_clusters = len(gat_deconv.cluster_to_celltype)
    if n_genes is None:
        n_genes = len(gat_deconv.marker_genes)
    
    # Prepare data
    st_data_normalized, st_data_raw, spatial_coords = gat_deconv.prepare_st_data(
        st_adata=st_adata,
        marker_genes=gat_deconv.marker_genes
    )
    
    # Build GAT model
    gat_deconv.build_gat_model(
        st_data_raw=st_data_raw,
        scale_basis=scale_basis
    )
    
    # Train GAT
    stage2_results = gat_deconv.train_gat_deconvolution(
        st_data_normalized=st_data_normalized,
        st_data_raw=st_data_raw,
        spatial_coords=spatial_coords,
        sample_name=sample_name,
        st_adata=st_adata,
        n_epochs=stage2_epochs,
        lr=lr,
        batch_size=batch_size,
        print_every=print_every
    )
    
    # Return combined results
    return {
        'n_clusters': n_clusters,
        'n_genes': n_genes,
        'best_pearson': stage2_results['best_pearson'],
        'best_mse': stage2_results['best_mse'],
        'best_cosine': stage2_results['best_cosine'],
        'model_path': stage1_model_path,
        'deconv_weights_path': f"{output_dir}/{sample_name}_deconv_weights.csv",
        'reconstructed_expr_path': f"{output_dir}/{sample_name}_reconstructed_expression.csv",
        'sample_name': sample_name
    }
