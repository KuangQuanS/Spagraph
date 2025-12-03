"""Command-line interface for scmapst

Provides CLI wrappers for training and deconvolution.
"""

import argparse
import sys
from pathlib import Path

from .training import train_integration, deconvolve_spots


def train_cli():
    """CLI for Stage 1 training"""
    parser = argparse.ArgumentParser(
        description="SC-MAP-ST Stage 1: Train VAE for SC-ST integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--sc_file", required=True, help="Path to single-cell h5ad file")
    parser.add_argument("--st_file", required=True, help="Path to spatial transcriptomics h5ad file")
    parser.add_argument("--output_dir", default="./stage1_results", help="Output directory")
    
    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="KL divergence weight")
    
    # Model architecture
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[512, 256], 
                       help="Hidden layer dimensions")
    parser.add_argument("--loss_type", choices=['mse', 'zinb'], default='mse',
                       help="Reconstruction loss type")
    parser.add_argument("--use_dual_decoder", action='store_true',
                       help="Use DualDecoderVAE with separate SC/ST decoders")
    
    # Data processing
    parser.add_argument("--resolution", type=float, default=4.0, 
                       help="Leiden clustering resolution")
    parser.add_argument("--top_n_per_type", type=int, default=100,
                       help="Number of marker genes per cluster")
    parser.add_argument("--aggregation_method", choices=['mean', 'median', 'weighted'],
                       default='weighted', help="Cluster aggregation method")
    parser.add_argument("--marker_selection_method", choices=['l1', 'variance', 'correlation'],
                       default='l1', help="Marker gene selection method")
    
    # Optional files
    parser.add_argument("--pretrained_path", help="Path to pretrained VAE model")
    parser.add_argument("--precomputed_marker_file", help="Path to precomputed marker genes")
    
    # Device
    parser.add_argument("--device", choices=['cuda', 'cpu'], help="Computing device")
    
    args = parser.parse_args()
    
    # Run training
    print("="*60)
    print("SC-MAP-ST Stage 1: VAE Training")
    print("="*60)
    
    results = train_integration(
        sc_file=args.sc_file,
        st_file=args.st_file,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        resolution=args.resolution,
        top_n_per_type=args.top_n_per_type,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        loss_type=args.loss_type,
        lambda_mmd=0.1 if args.use_dual_decoder else 0.0,
        use_dual_decoder=args.use_dual_decoder,
        pretrained_path=args.pretrained_path,
        precomputed_marker_file=args.precomputed_marker_file,
        aggregation_method=args.aggregation_method,
        marker_selection_method=args.marker_selection_method,
        device=args.device
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Number of clusters: {results['n_clusters']}")
    print(f"Number of marker genes: {results['n_genes']}")
    print("="*60)
    
    return 0


def deconvolve_cli():
    """CLI for Stage 2 deconvolution"""
    parser = argparse.ArgumentParser(
        description="SC-MAP-ST Stage 2: GAT-based spatial deconvolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--stage1_model", required=True, 
                       help="Path to Stage 1 VAE model checkpoint")
    parser.add_argument("--st_file", required=True,
                       help="Path to spatial transcriptomics h5ad file")
    parser.add_argument("--output_dir", default="./stage2_results",
                       help="Output directory")
    
    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    
    # Graph construction
    parser.add_argument("--k_spatial", type=int, default=20,
                       help="Number of spatial neighbors")
    parser.add_argument("--k_celltype", type=int, default=10,
                       help="Number of cell type neighbors")
    
    # Deconvolution parameters
    parser.add_argument("--weight_threshold", type=float, default=0.01,
                       help="Threshold for sparsifying weights")
    parser.add_argument("--scale_basis", choices=['all', 'marker', 'hvg', 'none'],
                       default='marker', help="Gene set for scaling reconstruction")
    parser.add_argument("--hvg_file", help="Path to HVG list (required if scale_basis=hvg)")
    parser.add_argument("--celltype_key", help="Cell type column name in cluster data")
    
    # Device
    parser.add_argument("--device", choices=['cuda', 'cpu'], help="Computing device")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scale_basis == 'hvg' and not args.hvg_file:
        parser.error("--hvg_file is required when --scale_basis=hvg")
    
    # Run deconvolution
    print("="*60)
    print("SC-MAP-ST Stage 2: GAT Deconvolution")
    print("="*60)
    
    results = deconvolve_spots(
        stage1_model_path=args.stage1_model,
        st_file=args.st_file,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        k_spatial=args.k_spatial,
        k_celltype=args.k_celltype,
        weight_threshold=args.weight_threshold,
        scale_basis=args.scale_basis,
        celltype_key=args.celltype_key,
        hvg_file=args.hvg_file,
        device=args.device
    )
    
    print("\n" + "="*60)
    print("Deconvolution completed successfully!")
    print(f"Sample: {results['sample_name']}")
    print(f"Best Pearson loss: {results['best_pearson']:.4f}")
    print(f"Best MSE loss: {results['best_mse']:.4f}")
    print(f"Deconvolution weights: {results['deconv_weights_path']}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(train_cli())
