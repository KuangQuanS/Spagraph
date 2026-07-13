"""Stage 1 (VAE) training for Spagraph.

Wraps the spagraph/models/stage1.py implementation behind a clean API.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure repository root is on path for local execution
_current_dir = Path(__file__).parent.parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from spagraph.models.stage1 import coEncoder
from .deconv import Stage1Artifacts


def train_vae(
    sc_file: str,
    st_file: str,
    output_dir: Optional[str] = None,
    as_artifacts: bool = True,
    n_epochs: int = 300,
    resolution: float = 4.0,
    top_n_per_type: int = 100,
    hidden_dims: Optional[list] = None,
    latent_dim: int = 256,
    batch_size: int = 512,
    lr: float = 5e-4,
    beta: float = 0.1,
    loss_type: str = 'mse',
    lambda_mmd: float = 0.03,
    use_dual_decoder: bool = True,
    pretrained_path: Optional[str] = None,
    precomputed_marker_file: Optional[str] = None,
    aggregation_method: str = 'mean',
    marker_selection_method: str = 'variance',
    celltype_key: Optional[str] = None,
    device: Optional[str] = None,
    print_every: int = 50,
    seed: int = 42
) -> Any:
    """Train VAE model for SC-ST integration (Stage 1)
    
    支持两种模式：
    1. 纯内存模式（output_dir=None）：不保存任何文件，只返回内存中的 Stage1Artifacts
    2. 文件模式（output_dir 指定）：保存模型和数据到磁盘
    
    Args:
        sc_file: Path to single-cell h5ad file
        st_file: Path to spatial transcriptomics h5ad file
        output_dir: Output directory path (None 表示纯内存模式，不保存任何文件)
        as_artifacts: 是否返回 Stage1Artifacts（默认 True），False 返回原始字典
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
        marker_selection_method: Marker gene selection method ('l1', 'variance',
            'correlation', or 'celltype_specific')
        celltype_key: Optional ``scRNA.obs`` annotation column used by
            ``celltype_specific`` selection and annotation-guided Stage 2.
        device: Computing device ('cuda', 'cpu', or None for auto-select)
        print_every: Print loss every N epochs
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Stage1Artifacts（默认）或原始字典（as_artifacts=False）:
            - 内存模式: 所有数据存储在 artifacts 内存字段中
            - 文件模式: model_path/cluster_data_path 指向保存的文件
    
    Example:
        >>> import spagraph
        >>> # 纯内存模式：不保存任何文件
        >>> art = spagraph.vae(
        ...     sc_file="data/sc.h5ad",
        ...     st_file="data/st.h5ad"
        ... )
        >>> # art 包含所有内存数据，可直接传给 deconv()
        >>>
        >>> # 文件模式：保存到磁盘
        >>> art = spagraph.vae(
        ...     sc_file="data/sc.h5ad",
        ...     st_file="data/st.h5ad",
        ...     output_dir="output/stage1/"
        ... )
        >>> print(f"Model saved to: {art.model_path}")
    """
    if hidden_dims is None:
        hidden_dims = [512, 256]
    
    # 判断是否保存到磁盘
    save_to_disk = output_dir is not None
    
    # Create encoder instance (seed 会在 __init__ 中设置)
    encoder = coEncoder(
        sc_file=sc_file,
        st_file=st_file,
        output_dir=output_dir or "./tmp_stage1",  # 临时目录（不会写入）
        celltype_key=celltype_key,
        device=device,
        save_to_disk=save_to_disk,
        seed=seed
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
        marker_selection_method=marker_selection_method,
        print_every=print_every
    )
    
    # Store best_loss in results for config saving
    if 'best_loss' not in results and hasattr(encoder, 'best_loss'):
        results['best_loss'] = encoder.best_loss
    
    artifacts = Stage1Artifacts.from_results(
        results=results,
        output_dir=output_dir,
        st_file=st_file,
        celltype_key=celltype_key
    )
    
    # 保存超参数到 txt 文件
    if save_to_disk:
        import datetime
        config_path = f"{output_dir}/config_vae.txt"
        with open(config_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Stage 1: VAE Training Configuration\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Input Files:\n")
            f.write(f"  SC File:       {sc_file}\n")
            f.write(f"  ST File:       {st_file}\n")
            f.write(f"  Output Dir:    {output_dir}\n\n")
            f.write(f"Training Hyperparameters:\n")
            f.write(f"  Epochs:        {n_epochs}\n")
            f.write(f"  Learning Rate: {lr}\n")
            f.write(f"  Batch Size:    {batch_size}\n")
            f.write(f"  Latent Dim:    {latent_dim}\n")
            f.write(f"  Hidden Dims:   {hidden_dims}\n")
            f.write(f"  Beta (KL):     {beta}\n")
            f.write(f"  Lambda MMD:    {lambda_mmd}\n")
            f.write(f"  Loss Type:     {loss_type}\n")
            f.write(f"  Seed:          {seed}\n\n")
            f.write(f"Clustering Hyperparameters:\n")
            f.write(f"  Resolution:    {resolution}\n")
            f.write(f"  Top N/Type:    {top_n_per_type}\n")
            f.write(f"  Marker Method: {marker_selection_method}\n")
            f.write(f"  Cell Type Key: {celltype_key or 'auto'}\n")
            f.write(f"  Aggregation:   {aggregation_method}\n\n")
            f.write(f"Model Architecture:\n")
            f.write(f"  Dual Decoder:  {use_dual_decoder}\n")
            f.write(f"  Device:        {device or 'auto'}\n\n")
            f.write(f"Results:\n")
            f.write(f"  N Clusters:    {results.get('n_clusters', 'N/A')}\n")
            f.write(f"  N Genes:       {results.get('n_genes', 'N/A')}\n\n")
            best_loss_val = results.get('best_loss', None)
            if best_loss_val is not None and best_loss_val != 'N/A':
                f.write(f"Final Loss:\n")
                f.write(f"  Best Loss:     {best_loss_val:.6f}\n")
            f.write("="*60 + "\n")
        print(f"Stage 1 config saved to: {config_path}")
    
    return artifacts if as_artifacts else results


# Backward-compatible alias (previously named train_integration)
train_integration = train_vae
