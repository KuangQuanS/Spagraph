"""Example usage of scmapst package

This script demonstrates the basic workflow of SC-MAP-ST.
"""

import scmapst
from pathlib import Path

def main():
    """Run complete SC-MAP-ST pipeline"""
    
    # Define paths
    data_dir = Path("data")
    output_dir = Path("output")
    
    sc_file = data_dir / "sc_adata.h5ad"
    st_file = data_dir / "st_adata.h5ad"
    
    stage1_output = output_dir / "stage1"
    stage2_output = output_dir / "stage2"
    
    # Create output directories
    stage1_output.mkdir(parents=True, exist_ok=True)
    stage2_output.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SC-MAP-ST Pipeline Example")
    print("="*80)
    
    # ========================================================================
    # Stage 1: Train VAE for SC-ST integration
    # ========================================================================
    print("\n" + "="*80)
    print("Stage 1: VAE Training for SC-ST Integration")
    print("="*80)
    
    stage1_results = scmapst.train(
        sc_file=str(sc_file),
        st_file=str(st_file),
        output_dir=str(stage1_output),
        n_epochs=150,
        resolution=4.0,
        top_n_per_type=100,
        latent_dim=128,
        batch_size=512,
        lr=5e-4,
        beta=0.1,
        use_dual_decoder=True,
        aggregation_method='weighted',
        marker_selection_method='l1'
    )
    
    print("\nStage 1 Results:")
    print(f"  Model path: {stage1_results['model_path']}")
    print(f"  Clusters: {stage1_results['n_clusters']}")
    print(f"  Marker genes: {stage1_results['n_genes']}")
    print(f"  Best loss: {stage1_results['best_loss']:.4f}")
    
    # ========================================================================
    # Stage 2: GAT-based spatial deconvolution
    # ========================================================================
    print("\n" + "="*80)
    print("Stage 2: GAT-based Spatial Deconvolution")
    print("="*80)
    
    stage2_results = scmapst.deconvolve(
        stage1_model_path=stage1_results['model_path'],
        st_file=str(st_file),
        output_dir=str(stage2_output),
        n_epochs=200,
        lr=1e-3,
        batch_size=512,
        k_spatial=20,
        k_celltype=10,
        weight_threshold=0.01,
        scale_basis='marker'  # Options: 'all', 'marker', 'hvg', 'none'
    )
    
    print("\nStage 2 Results:")
    print(f"  Sample: {stage2_results['sample_name']}")
    print(f"  Best Pearson: {stage2_results['best_pearson']:.4f}")
    print(f"  Best MSE: {stage2_results['best_mse']:.4f}")
    print(f"  Best Cosine: {stage2_results['best_cosine']:.4f}")
    print(f"  Deconvolution weights: {stage2_results['deconv_weights_path']}")
    print(f"  Reconstructed expression: {stage2_results['reconstructed_expr_path']}")
    
    # ========================================================================
    # Load and inspect results
    # ========================================================================
    print("\n" + "="*80)
    print("Loading and Inspecting Results")
    print("="*80)
    
    import pandas as pd
    
    # Load deconvolution weights
    deconv_weights = pd.read_csv(stage2_results['deconv_weights_path'], index_col=0)
    print(f"\nDeconvolution weights shape: {deconv_weights.shape}")
    print(f"Spot barcodes (first 5): {list(deconv_weights.index[:5])}")
    print(f"Cell types: {list(deconv_weights.columns)}")
    
    # Show example predictions
    print("\nExample cell type proportions (first 3 spots):")
    print(deconv_weights.head(3))
    
    # Cell type abundance across all spots
    print("\nAverage cell type proportions across all spots:")
    print(deconv_weights.mean().sort_values(ascending=False))
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
