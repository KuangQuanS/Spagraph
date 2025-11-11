"""
Unified Deconvolution Pipeline - Stage 1 + Stage 2
Combines VAE training (Stage 1) and GAT deconvolution (Stage 2) into a single workflow
"""

import os
import argparse
import torch
from pathlib import Path

# Import Stage 1 components
from stage1 import coEncoder

# Import Stage 2 components
from stage2 import GATDeconvolution


def run_unified_pipeline(
    # Data files
    sc_file,
    st_file,
    
    # Output
    output_dir="./deconv_results",
    
    # Stage 1: VAE Training
    stage1_resolution=0.5,
    stage1_top_n_per_type=100,
    stage1_hidden_dims=None,
    stage1_latent_dim=128,
    stage1_batch_size=256,
    stage1_n_epochs=100,
    stage1_lr=1e-3,
    stage1_beta=1.0,
    stage1_loss_type='mse',
    stage1_lambda_mmd=0.0,
    stage1_use_dual_decoder=True,
    stage1_precomputed_marker_file=None,
    
    # Stage 2: GAT Deconvolution
    stage2_skip=False,
    stage2_gat_hidden_dim=64,
    stage2_gat_layers=3,
    stage2_gat_heads=4,
    stage2_dropout=0.1,
    stage2_k_spatial=6,
    stage2_k_celltype=10,
    stage2_n_epochs=50,
    stage2_lr=1e-3,
    stage2_batch_size=512,
    stage2_loss_lambda_pearson=1.0,
    stage2_loss_lambda_mse=1.0,
    stage2_loss_lambda_cosine=1.0,
    stage2_loss_lambda_reg=0.5,
    stage2_loss_lambda_sparse=0.01,
    stage2_loss_lambda_proportion=1.0,
    stage2_cells_per_spot=None,
    stage2_weight_threshold=0.01,
    
    # Device
    device=None,
):
    """
    Run unified deconvolution pipeline
    
    Args:
        sc_file: Path to single-cell h5ad file
        st_file: Path to spatial transcriptomics h5ad file
        output_dir: Output directory for all results
        
        stage1_*: Stage 1 (VAE training) hyperparameters
        stage2_*: Stage 2 (GAT deconvolution) hyperparameters
        
        device: Computing device (cuda/cpu, None for auto-select)
    """
    
    # Setup
    if stage1_hidden_dims is None:
        stage1_hidden_dims = [512, 256]
    
    output_dir = Path(output_dir)
    stage1_output_dir = output_dir / "stage1"
    stage2_output_dir = output_dir / "stage2"
    
    # Create directories
    stage1_output_dir.mkdir(parents=True, exist_ok=True)
    stage2_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("UNIFIED DECONVOLUTION PIPELINE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print()
    
    # ========== STAGE 1: VAE TRAINING ==========
    print("="*80)
    print("STAGE 1: VAE Training for SC-ST Integration")
    print("="*80)
    
    co_encoder = coEncoder(
        sc_file=sc_file,
        st_file=st_file,
        output_dir=str(stage1_output_dir),
        device=device
    )
    
    stage1_results = co_encoder.run_stage1_training(
        top_n_per_type=stage1_top_n_per_type,
        resolution=stage1_resolution,
        batch_size=stage1_batch_size,
        n_epochs=stage1_n_epochs,
        lr=stage1_lr,
        beta=stage1_beta,
        hidden_dims=stage1_hidden_dims,
        latent_dim=stage1_latent_dim,
        loss_type=stage1_loss_type,
        lambda_mmd=stage1_lambda_mmd,
        pretrained_path=None,
        precomputed_marker_file=stage1_precomputed_marker_file,
        use_dual_decoder=stage1_use_dual_decoder
    )
    
    print()
    print(f"Stage 1 Results:")
    print(f"  - Model: {stage1_results['model_path']}")
    print(f"  - Genes: {stage1_results['n_genes']}")
    print(f"  - Clusters: {stage1_results['n_clusters']}")
    print()
    
    # ========== STAGE 2: GAT DECONVOLUTION ==========
    if not stage2_skip:
        print("="*80)
        print("STAGE 2: GAT Deconvolution for Spatial Transcriptomics")
        print("="*80)
        
        # Auto-find Stage 1 model (no need to specify path)
        stage1_model_path = stage1_results['model_path']
        
        # Extract sample name from ST file
        sample_name = os.path.splitext(os.path.basename(st_file))[0]
        if sample_name.endswith('_ST'):
            sample_name = sample_name[:-3]
        
        print(f"Sample name: {sample_name}")
        print()
        
        # Initialize trainer
        trainer = GATDeconvolution(
            stage1_model_path=stage1_model_path,
            output_dir=str(stage2_output_dir),
            device=device,
            weight_threshold=stage2_weight_threshold
        )
        
        # Set graph construction parameters
        trainer.k_spatial = stage2_k_spatial
        trainer.k_celltype = stage2_k_celltype
        
        # Load VAE Encoder
        trainer.load_vae_encoder()
        
        # Load ST data
        print("Loading ST data...")
        import scanpy as sc
        st_adata = sc.read_h5ad(st_file)
        st_adata.var_names_make_unique()
        print(f"   ST shape: {st_adata.shape}")
        
        # Normalize ST data for VAE embedding
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        
        # Extract and normalize marker genes for VAE
        available_genes = [g for g in trainer.genes if g in st_adata.var.index]
        st_subset_vae = st_adata[:, available_genes].copy()
        st_X_vae = st_subset_vae.X.toarray() if hasattr(st_subset_vae.X, 'toarray') else st_subset_vae.X
        
        # Extract raw counts for reconstruction loss
        st_subset_raw = st_adata[:, available_genes].copy()
        st_X_raw = st_subset_raw.X.toarray() if hasattr(st_subset_raw.X, 'toarray') else st_subset_raw.X
        
        print(f"   ST data for VAE: {st_X_vae.shape}")
        print(f"   ST data (raw counts): {st_X_raw.shape}")
        
        # Extract spatial coordinates
        spatial_coords = st_adata.obsm['spatial'] if 'spatial' in st_adata.obsm else None
        if spatial_coords is None:
            raise ValueError("Spatial coordinates not found in st_adata.obsm['spatial']")
        
        print(f"   Spatial coordinates: {spatial_coords.shape}")
        print()
        
        # Train GAT deconvolution
        stage2_results = trainer.train_gat_deconvolution(
            st_data_normalized=st_X_vae,
            st_data_raw=st_X_raw,
            spatial_coords=spatial_coords,
            sample_name=sample_name,
            st_adata=st_adata,
            n_epochs=stage2_n_epochs,
            lr=stage2_lr,
            batch_size=stage2_batch_size,
            loss_lambda_pearson=stage2_loss_lambda_pearson,
            loss_lambda_mse=stage2_loss_lambda_mse,
            loss_lambda_cosine=stage2_loss_lambda_cosine,
            loss_lambda_reg=stage2_loss_lambda_reg,
            loss_lambda_sparse=stage2_loss_lambda_sparse,
            loss_lambda_proportion=stage2_loss_lambda_proportion,
            cells_per_spot=stage2_cells_per_spot,
            gat_hidden_dim=stage2_gat_hidden_dim,
            gat_layers=stage2_gat_layers,
            gat_heads=stage2_gat_heads,
            dropout=stage2_dropout
        )
        
        print()
        print(f"Stage 2 Results:")
        print(f"  - Sample: {stage2_results['sample_name']}")
        print(f"  - Output: {stage2_output_dir}")
        print()
    else:
        print("Stage 2 skipped (--skip_stage2)")
        stage2_results = None
    
    # Summary
    print("="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"  - Stage 1 results: {stage1_output_dir}")
    if not stage2_skip:
        print(f"  - Stage 2 results: {stage2_output_dir}")
    print()
    
    return {
        'stage1_results': stage1_results,
        'stage2_results': stage2_results,
        'output_dir': str(output_dir)
    }


def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description='Unified Deconvolution Pipeline (Stage 1 + Stage 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python deconv.py --sc_file data/sc.h5ad --st_file data/st.h5ad --output_dir ./results
  
  # Stage 1 only
  python deconv.py --sc_file data/sc.h5ad --st_file data/st.h5ad --output_dir ./results --skip_stage2
  
  # With custom hyperparameters
  python deconv.py --sc_file data/sc.h5ad --st_file data/st.h5ad \\
      --stage1_n_epochs 200 --stage2_n_epochs 100 --stage2_gat_hidden_dim 128
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--sc_file', type=str, required=True,
                       help='Path to single-cell h5ad file')
    parser.add_argument('--st_file', type=str, required=True,
                       help='Path to spatial transcriptomics h5ad file')
    parser.add_argument('--output_dir', type=str, default="./deconv_results",
                       help='Output directory for all results (default: ./deconv_results)')
    
    # Stage 1 arguments
    stage1_group = parser.add_argument_group('Stage 1 (VAE Training)', 'VAE training hyperparameters')
    stage1_group.add_argument('--stage1_resolution', type=float, default=2,
                              help='Leiden clustering resolution (default: 0.5)')
    stage1_group.add_argument('--stage1_top_n_per_type', type=int, default=100,
                              help='Marker genes per cluster (default: 100)')
    stage1_group.add_argument('--stage1_latent_dim', type=int, default=256,
                              help='VAE latent dimension (default: 256)')
    stage1_group.add_argument('--stage1_n_epochs', type=int, default=300,
                              help='Number of training epochs (default: 300)')
    stage1_group.add_argument('--stage1_lr', type=float, default=1e-3,
                              help='Learning rate (default: 1e-3)')
    stage1_group.add_argument('--stage1_beta', type=float, default=1.0,
                              help='KL divergence weight (default: 1.0)')
    stage1_group.add_argument('--stage1_loss_type', type=str, default='zinb', choices=['mse', 'zinb'],
                              help='Reconstruction loss type (default: zinb)')
    stage1_group.add_argument('--stage1_lambda_mmd', type=float, default=1.0,
                              help='MMD loss weight for modality alignment (default: 1.0)')
    stage1_group.add_argument('--stage1_batch_size', type=int, default=512,
                              help='Batch size (default: 512)')
    stage1_group.add_argument('--stage1_precomputed_marker_file', type=str, default=None,
                              help='Path to precomputed marker genes file (optional)')
    
    # Stage 2 arguments
    stage2_group = parser.add_argument_group('Stage 2 (GAT Deconvolution)', 'GAT deconvolution hyperparameters')
    stage2_group.add_argument('--skip_stage2', action='store_true',
                              help='Skip Stage 2 (VAE training only)')
    stage2_group.add_argument('--stage2_gat_hidden_dim', type=int, default=64,
                              help='GAT hidden dimension (default: 64)')
    stage2_group.add_argument('--stage2_gat_layers', type=int, default=3,
                              help='Number of GAT layers (default: 3)')
    stage2_group.add_argument('--stage2_k_spatial', type=int, default=6,
                              help='Number of spatial neighbors (default: 6)')
    stage2_group.add_argument('--stage2_k_celltype', type=int, default=10,
                              help='Number of celltype neighbors (default: 10)')
    stage2_group.add_argument('--stage2_n_epochs', type=int, default=50,
                              help='Number of training epochs (default: 50)')
    stage2_group.add_argument('--stage2_lr', type=float, default=1e-3,
                              help='Learning rate (default: 1e-3)')
    stage2_group.add_argument('--stage2_batch_size', type=int, default=512,
                              help='Batch size (default: 512)')
    stage2_group.add_argument('--stage2_loss_lambda_mse', type=float, default=1.0,
                              help='MSE loss weight (default: 1.0)')
    stage2_group.add_argument('--stage2_loss_lambda_cosine', type=float, default=1.0,
                              help='Cosine loss weight (default: 1.0)')
    stage2_group.add_argument('--stage2_weight_threshold', type=float, default=0.01,
                              help='Weight threshold for sparsification (default: 0.01)')
    
    # Device argument
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu, None for auto-select)')
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_unified_pipeline(
        # Data files
        sc_file=args.sc_file,
        st_file=args.st_file,
        output_dir=args.output_dir,
        
        # Stage 1
        stage1_resolution=args.stage1_resolution,
        stage1_top_n_per_type=args.stage1_top_n_per_type,
        stage1_latent_dim=args.stage1_latent_dim,
        stage1_batch_size=args.stage1_batch_size,
        stage1_n_epochs=args.stage1_n_epochs,
        stage1_lr=args.stage1_lr,
        stage1_beta=args.stage1_beta,
        stage1_loss_type=args.stage1_loss_type,
        stage1_lambda_mmd=args.stage1_lambda_mmd,
        stage1_precomputed_marker_file=args.stage1_precomputed_marker_file,
        
        # Stage 2
        stage2_skip=args.skip_stage2,
        stage2_gat_hidden_dim=args.stage2_gat_hidden_dim,
        stage2_gat_layers=args.stage2_gat_layers,
        stage2_k_spatial=args.stage2_k_spatial,
        stage2_k_celltype=args.stage2_k_celltype,
        stage2_n_epochs=args.stage2_n_epochs,
        stage2_lr=args.stage2_lr,
        stage2_batch_size=args.stage2_batch_size,
        stage2_loss_lambda_mse=args.stage2_loss_lambda_mse,
        stage2_loss_lambda_cosine=args.stage2_loss_lambda_cosine,
        stage2_weight_threshold=args.stage2_weight_threshold,
        
        # Device
        device=args.device
    )
    
    print("\nAll results saved to:", results['output_dir'])


if __name__ == "__main__":
    main()
