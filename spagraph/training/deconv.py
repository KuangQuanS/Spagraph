"""Unified spatial deconvolution API.

Combines Stage 1 (VAE) and Stage 2 (GAT) into a single call.
"""

import os
import sys
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import scanpy as sc
import torch

# Ensure repository root is on path for local execution
_current_dir = Path(__file__).parent.parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from spagraph.models.stage2 import GATDeconvolution


@dataclass
class Stage1Artifacts:
    """轻量封装：保存第一阶段的关键产物，便于直接跑第二阶段。
    
    支持两种模式：
    1. 纯内存模式：model_path=None，所有数据在 vae_encoder/vae_state_dict 等字段
    2. 文件模式：model_path 指向 .pth 文件
    """
    model_path: Optional[str] = None
    cluster_data_path: Optional[str] = None
    output_dir: Optional[str] = None
    st_file: Optional[str] = None
    
    # ===== 内存产物：纯内存模式下使用 =====
    # VAE 编码器（已加载的 PyTorch 模块，可直接用于推理）
    vae_encoder: Optional[Any] = None  # torch.nn.Module
    vae_state_dict: Optional[dict] = None
    input_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    output_type: Optional[str] = None
    
    # 聚类和标签信息
    label_encoder: Optional[Any] = None
    marker_genes: Optional[Any] = None
    genes: Optional[Any] = None
    sc_clusters: Optional[Any] = None
    resolution: Optional[float] = None
    celltype_key: Optional[str] = None
    
    # 聚类中心和表达（Stage 2 需要）
    avg_cell_counts: Optional[Any] = None
    all_genes: Optional[Any] = None
    cluster_to_celltype: Optional[Any] = None
    celltype_prototypes: Optional[Any] = None  # numpy array or tensor
    celltype_expressions: Optional[Any] = None  # numpy array or tensor
    celltype_expressions_full: Optional[Any] = None  # list of arrays
    hvg_genes_union: Optional[Any] = None
    
    # 元信息
    n_clusters: Optional[int] = None
    n_genes: Optional[int] = None
    raw_results: Optional[Dict[str, Any]] = None  # 保留原始结果，兼容旧逻辑
    
    def is_memory_mode(self) -> bool:
        """判断是否为纯内存模式（不依赖文件）"""
        return self.vae_encoder is not None or self.vae_state_dict is not None

    @classmethod
    def from_results(cls, results: Dict[str, Any], output_dir: Optional[str], st_file: Optional[str], celltype_key: Optional[str]):
        return cls(
            model_path=results.get('model_path'),
            cluster_data_path=results.get('cluster_data_path'),
            output_dir=output_dir,
            st_file=st_file,
            n_clusters=results.get('n_clusters'),
            n_genes=results.get('n_genes'),
            celltype_key=celltype_key or results.get('celltype_key'),
            raw_results=results,
            # 内存产物
            vae_encoder=results.get('vae_encoder'),
            vae_state_dict=results.get('vae_state_dict'),
            input_dim=results.get('input_dim'),
            latent_dim=results.get('latent_dim'),
            output_type=results.get('output_type'),
            label_encoder=results.get('label_encoder'),
            marker_genes=results.get('marker_genes'),
            genes=results.get('genes'),
            sc_clusters=results.get('sc_clusters'),
            resolution=results.get('resolution'),
            avg_cell_counts=results.get('avg_cell_counts'),
            all_genes=results.get('all_genes'),
            cluster_to_celltype=results.get('cluster_to_celltype'),
            celltype_prototypes=results.get('celltype_prototypes'),
            celltype_expressions=results.get('celltype_expressions'),
            celltype_expressions_full=results.get('celltype_expressions_full'),
            hvg_genes_union=results.get('hvg_genes_union')
        )


def run_deconv(
    vae: Stage1Artifacts,
    st_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    # Training parameters
    n_epochs: int = 250,
    lr: float = 5e-3,
    batch_size: int = 512,
    print_every: int = 25,
    # Graph parameters
    k_spatial: int = 5,
    k_celltype: int = 20,
    # GAT architecture
    gat_hidden_dim: int = 512,
    gat_layers: int = 4,
    gat_heads: int = 4,
    dropout: float = 0.1,
    # Loss weights
    lambda_pearson: float = 5.0,
    lambda_mse: float = 0.001,
    lambda_cosine: float = 5.0,
    lambda_gene_pearson: float = 1.0,
    lambda_gene_cosine: float = 1.0,
    lambda_reg: float = 0.1,
    lambda_sparse: float = 1,
    lambda_proportion: float = 0.1,
    # Other
    weight_threshold: float = 0.001,
    scale_basis: str = 'all',
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Stage 2: GAT Deconvolution
    
    Args:
        vae: Stage1Artifacts 对象（必须，由 spg.vae() 返回）
        st_file: ST h5ad 文件路径（可选，默认用 vae.st_file）
        output_dir: 输出目录（提供则保存，不提供则纯内存）
        
        # 训练参数
        n_epochs: 训练轮数
        lr: 学习率
        batch_size: 批大小
        print_every: 每 N 轮打印一次
        
        # 图构建参数
        k_spatial: 空间邻居数
        k_celltype: 细胞类型邻居数
        
        # GAT 架构
        gat_hidden_dim: GAT 隐藏层维度
        gat_layers: GAT 层数
        gat_heads: 注意力头数
        dropout: Dropout 比率
        
        # 损失权重
        lambda_pearson: Pearson 相关损失权重（spot level）
        lambda_mse: MSE 损失权重
        lambda_cosine: Cosine 相似度损失权重（spot level）
        lambda_gene_pearson: Gene-wise Pearson 相关损失权重
        lambda_gene_cosine: Gene-wise Cosine 相似度损失权重
        lambda_reg: 正则化损失权重
        lambda_sparse: 稀疏性损失权重
        lambda_proportion: 比例约束损失权重
        
        # 其他
        weight_threshold: 权重稀疏化阈值
        scale_basis: 缩放基准 ('all', 'marker', 'hvg', 'none')
        device: 计算设备
        seed: 随机种子
    
    Returns:
        dict: {deconv, deconv_path, sample_name, n_clusters, best_pearson, metrics}
    
    Example:
        >>> art = spg.vae(sc_file="sc.h5ad", st_file="st.h5ad")
        >>> res = spg.deconv(art, output_dir="output/")
        >>> print(res['deconv'].head())
    """
    # 参数检查
    if st_file is None:
        st_file = vae.st_file
    if st_file is None:
        raise ValueError("st_file is required")

    save_outputs = bool(output_dir)
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
    
    sample_name = Path(st_file).stem
    use_memory_mode = vae.is_memory_mode()
    
    # 打印配置
    print(f"\n{'='*60}")
    print(f"Stage 2: GAT Deconvolution")
    print(f"{'='*60}")
    print(f"  Sample:        {sample_name}")
    print(f"  Epochs:        {n_epochs}")
    print(f"  LR:            {lr}")
    print(f"  Batch Size:    {batch_size}")
    print(f"  K Spatial:     {k_spatial}")
    print(f"  K Celltype:    {k_celltype}")
    print(f"  GAT:           {gat_layers}L x {gat_hidden_dim}D x {gat_heads}H")
    print(f"  Loss Weights:  Pearson={lambda_pearson}, MSE={lambda_mse}, Cosine={lambda_cosine}")
    print(f"                 GenePearson={lambda_gene_pearson}, GeneCosine={lambda_gene_cosine}")
    print(f"  Regularization: Reg={lambda_reg}, Sparse={lambda_sparse}, Prop={lambda_proportion}")
    print(f"  Seed:          {seed}")
    print(f"  Memory Mode:   {use_memory_mode}")
    print(f"{'='*60}\n")
    
    # 初始化 trainer (seed 会在 __init__ 中设置)
    trainer = GATDeconvolution(
        stage1_model_path=vae.model_path,
        output_dir=output_dir or "./tmp_stage2",
        device=device,
        weight_threshold=weight_threshold,
        stage1_artifacts=vae if use_memory_mode else None,
        seed=seed
    )
    trainer.k_spatial = k_spatial
    
    trainer.scale_basis = scale_basis
    
    # 加载 VAE 编码器
    trainer.load_vae_encoder()
    
    if trainer.sc_clusters is None:
        raise ValueError("Stage 1 missing cluster info!")
    
    n_clusters = trainer.celltype_prototypes.shape[0]
    n_genes = len(trainer.marker_genes)
    
    # 自动调整 k_celltype：根据聚类数自适应调整邻居数
    if n_clusters < 30 and k_celltype == 20:  # 聚类数少，减少邻居数
        adjusted_k = 15
        print(f"⚠️  {n_clusters} clusters detected (< 40), auto-adjusting k_celltype: {k_celltype} → {adjusted_k}")
        trainer.k_celltype = adjusted_k
    elif n_clusters > 53 and k_celltype == 20:  # 聚类数多，增加邻居数
        adjusted_k = 30
        print(f"⚠️  {n_clusters} clusters detected (> 60), auto-adjusting k_celltype: {k_celltype} → {adjusted_k}")
        trainer.k_celltype = adjusted_k
    else:
        trainer.k_celltype = k_celltype
    
    print(f"Loaded {n_clusters} clusters, {n_genes} genes")

    # 加载 ST 数据
    st_adata = sc.read_h5ad(st_file)
    st_adata.var_names_make_unique()
    
    # 提取数据
    st_raw_all = st_adata.X.toarray() if hasattr(st_adata.X, "toarray") else st_adata.X
    st_proc = st_adata.copy()
    sc.pp.normalize_total(st_proc, target_sum=1e4)
    sc.pp.log1p(st_proc)
    
    # 空间坐标
    if 'spatial' in st_adata.obsm:
        spatial_coords = st_adata.obsm['spatial']
        trainer.use_embedding_knn = False
    else:
        print("⚠️ No spatial coords, using embedding-based KNN")
        spatial_coords = np.zeros((st_adata.n_obs, 2))
        trainer.use_embedding_knn = True
    
    # 提取 marker genes
    st_X_raw = st_adata[:, trainer.genes].X
    st_X_raw = st_X_raw.toarray() if hasattr(st_X_raw, 'toarray') else st_X_raw
    st_X_embed = st_proc[:, trainer.genes].X
    st_X_embed = st_X_embed.toarray() if hasattr(st_X_embed, 'toarray') else st_X_embed
    
    # 计算 spot counts
    if scale_basis == 'none':
        spot_total_counts = None
    elif scale_basis == 'all':
        spot_total_counts = np.asarray(st_raw_all.sum(axis=1)).ravel()
    elif scale_basis == 'hvg' and trainer.hvg_genes_union:
        hvg_in_st = [g for g in trainer.hvg_genes_union if g in st_adata.var_names]
        if hvg_in_st:
            st_hvg = st_adata[:, hvg_in_st].X
            st_hvg = st_hvg.toarray() if hasattr(st_hvg, 'toarray') else st_hvg
            spot_total_counts = np.asarray(st_hvg.sum(axis=1)).ravel()
        else:
            spot_total_counts = None
    else:  # marker
        spot_total_counts = np.asarray(st_X_raw.sum(axis=1)).ravel()
    
    # 构建 GAT 模型
    trainer.build_gat_model(
        n_cell_types=n_clusters,
        gat_hidden_dim=gat_hidden_dim,
        gat_layers=gat_layers,
        gat_heads=gat_heads,
        dropout=dropout,
        loss_lambda_pearson=lambda_pearson,
        loss_lambda_mse=lambda_mse,
        loss_lambda_cosine=lambda_cosine,
        loss_lambda_gene_pearson=lambda_gene_pearson,
        loss_lambda_gene_cosine=lambda_gene_cosine,
        loss_lambda_reg=lambda_reg,
        loss_lambda_sparse=lambda_sparse,
        loss_lambda_proportion=lambda_proportion,
        spot_total_counts=spot_total_counts
    )
    
    # 训练
    results = trainer.train_gat_deconvolution(
        st_data_normalized=st_X_embed,
        st_data_raw=st_X_raw,
        spatial_coords=spatial_coords,
        sample_name=sample_name,
        st_adata=st_adata,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        print_every=print_every
    )
    
    # 保存超参数到 txt 文件
    if save_outputs:
        import datetime
        config_path = f"{output_dir}/stage2_config.txt"
        with open(config_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Stage 2: GAT Deconvolution Configuration\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Input Files:\n")
            f.write(f"  ST File:       {st_file}\n")
            f.write(f"  Output Dir:    {output_dir}\n")
            f.write(f"  Sample Name:   {sample_name}\n\n")
            f.write(f"Training Hyperparameters:\n")
            f.write(f"  Epochs:        {n_epochs}\n")
            f.write(f"  Learning Rate: {lr}\n")
            f.write(f"  Batch Size:    {batch_size}\n")
            f.write(f"  Print Every:   {print_every}\n")
            f.write(f"  Seed:          {seed}\n\n")
            f.write(f"Graph Construction:\n")
            f.write(f"  K Spatial:     {k_spatial}\n")
            f.write(f"  K Celltype:    {trainer.k_celltype} (input: {k_celltype})\n")
            f.write(f"  Scale Basis:   {scale_basis}\n")
            f.write(f"  Weight Thresh: {weight_threshold}\n\n")
            f.write(f"GAT Architecture:\n")
            f.write(f"  Hidden Dim:    {gat_hidden_dim}\n")
            f.write(f"  Layers:        {gat_layers}\n")
            f.write(f"  Heads:         {gat_heads}\n")
            f.write(f"  Dropout:       {dropout}\n\n")
            f.write(f"Loss Weights:\n")
            f.write(f"  Pearson:       {lambda_pearson}\n")
            f.write(f"  MSE:           {lambda_mse}\n")
            f.write(f"  Cosine:        {lambda_cosine}\n")
            f.write(f"  Gene Pearson:  {lambda_gene_pearson}\n")
            f.write(f"  Gene Cosine:   {lambda_gene_cosine}\n")
            f.write(f"  Reg:           {lambda_reg}\n")
            f.write(f"  Sparse:        {lambda_sparse}\n")
            f.write(f"  Proportion:    {lambda_proportion}\n\n")
            f.write(f"Results:\n")
            f.write(f"  N Clusters:    {n_clusters}\n")
            f.write(f"  N Genes:       {n_genes}\n\n")
            f.write(f"Final Losses:\n")
            f.write(f"  Best Pearson:  {results.get('best_pearson', 'N/A'):.6f}\n")
            f.write(f"  Best MSE:      {results.get('best_mse', 'N/A'):.6f}\n")
            f.write(f"  Best Cosine:   {results.get('best_cosine', 'N/A'):.6f}\n\n")
            f.write(f"System:\n")
            f.write(f"  Memory Mode:   {use_memory_mode}\n")
            f.write(f"  Device:        {device or 'auto'}\n")
            f.write("="*60 + "\n")
        print(f"\u2705 Stage 2 config saved to: {config_path}")
    
    return {
        'deconv': results.get('composition_df'),
        'deconv_path': results.get('composition_path'),
        'sample_name': sample_name,
        'n_clusters': n_clusters,
        'best_pearson': results.get('best_pearson'),
        'metrics': {
            'pearson': results.get('best_pearson'),
            'mse': results.get('best_mse'),
            'cosine': results.get('best_cosine'),
            'n_clusters': n_clusters,
            'n_genes': n_genes
        }
    }


# Backward-compatible alias
deconvolve_spots = run_deconv
