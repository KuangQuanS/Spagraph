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
    
    # ST 数据相关（在 run_deconv 中计算并保存，用于重建基因表达时缩放）
    spot_total_counts: Optional[Any] = None  # [n_spots] array
    
    # ===== 动态cluster所需数据（可选）=====
    sc_cell_embeddings: Optional[Any] = None  # [n_sc_cells, latent_dim] SC细胞的VAE embeddings
    sc_cell_expressions_raw: Optional[Any] = None  # [n_sc_cells, n_all_genes] SC细胞所有基因原始count（后续按barcode和基因名截取）
    sc_cell_labels: Optional[Any] = None  # [n_sc_cells] SC细胞的cluster标签
    knn_cell_indices: Optional[Any] = None  # [n_spots, n_clusters, k] 预计算的k-nearest cell索引
    
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
            hvg_genes_union=results.get('hvg_genes_union'),
            # 动态cluster数据
            sc_cell_embeddings=results.get('sc_cell_embeddings'),
            sc_cell_expressions_raw=results.get('sc_cell_expressions_raw'),
            sc_cell_labels=results.get('sc_cell_labels')
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
    k_celltype_range: Optional[list] = [15, 20, 25, 30, 35],  # 自动网格搜索，默认 [15,20,25,30]，设为 [] 禁用
    # GAT architecture
    gat_hidden_dim: int = 512,
    gat_layers: int = 4,
    gat_heads: int = 4,
    dropout: float = 0.1,
    # Loss weights
    lambda_pearson: float = 5.0,
    lambda_mse: float = 0.01,
    lambda_cosine: float = 5.0,
    lambda_gene_pearson: float = 1.0,
    lambda_gene_cosine: float = 1.0,
    lambda_reg: float = 0.1,
    lambda_sparse: float = 1,
    lambda_proportion: float = 0.1,
    # Output options
    save_reconstructed_genes: bool = False,  # 是否保存重构的全基因表达
    # Dynamic cluster representation (optional)
    use_dynamic_cluster_repr: bool = True,  # 启用动态cluster表示（用k个最近细胞代替cluster平均）
    k_cells_per_cluster: int = 10,  # 每个cluster使用多少个最近细胞
    precompute_knn: bool = True,  # 是否在训练前预计算k-nearest cells（推荐True）
    use_learnable_weights: bool = False,  # 是否用MLP学习cell权重（False则用均匀平均）
    # Other
    weight_threshold: float = 0.001,
    scale_basis: str = 'all',
    device: Optional[str] = None,
    seed: int = 42,
    _silent_header: bool = False,  # 内部参数：禁止打印配置头部
) -> Dict[str, Any]:
    """Stage 2: GAT Deconvolution
    """
    # 参数检查
    if st_file is None:
        st_file = vae.st_file
    if st_file is None:
        raise ValueError("st_file is required")

    # 打印Stage 2配置信息（在网格搜索前）
    sample_name = Path(st_file).stem
    if not _silent_header:
        print(f"\n{'='*60}")
        print(f"Stage 2: GAT Deconvolution")
        print(f"{'='*60}")
        print(f"Sample:             {sample_name}")
        print(f"Epochs:             {n_epochs}")
        print(f"LR:                 {lr}")
        print(f"Batch Size:         {batch_size}")
        print(f"K Spatial:          {k_spatial}")
        if k_celltype_range is not None and len(k_celltype_range) > 1:
            print(f"K Celltype:         Grid Search {k_celltype_range}")
        else:
            print(f"K Celltype:         {k_celltype if k_celltype_range is None or len(k_celltype_range) == 0 else k_celltype_range[0]}")
        print(f"GAT Architecture:   {gat_layers}L × {gat_hidden_dim}D × {gat_heads}H")
        print(f"Dropout:            {dropout}")
        print(f"Loss Weights:")
        print(f"  Pearson:          {lambda_pearson}")
        print(f"  MSE:              {lambda_mse}")
        print(f"  Cosine:           {lambda_cosine}")
        print(f"  Gene Pearson:     {lambda_gene_pearson}")
        print(f"  Gene Cosine:      {lambda_gene_cosine}")
        print(f"  Reg:              {lambda_reg}")
        print(f"  Sparse:           {lambda_sparse}")
        print(f"  Proportion:       {lambda_proportion}")
        print(f"Dynamic Cluster:    {use_dynamic_cluster_repr}")
        if use_dynamic_cluster_repr:
            print(f"  K Cells/Cluster:  {k_cells_per_cluster}")
            print(f"  Learnable Weights: {use_learnable_weights}")
            print(f"  Precompute KNN:   {precompute_knn}")
        print(f"Scale Basis:        {scale_basis}")
        print(f"Weight Threshold:   {weight_threshold}")
        print(f"Seed:               {seed}")
        print(f"Save to Disk:       {bool(output_dir)}")
        print(f"{'='*60}\n")
    
    # 网格搜索启用条件：列表且长度 > 1
    if k_celltype_range is not None and len(k_celltype_range) > 1:
        # 启用网格搜索
        return run_deconv_auto_k(
            vae=vae,
            st_file=st_file,
            output_dir=output_dir,
            k_celltype_range=k_celltype_range,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            print_every=print_every,
            k_spatial=k_spatial,
            gat_hidden_dim=gat_hidden_dim,
            gat_layers=gat_layers,
            gat_heads=gat_heads,
            dropout=dropout,
            lambda_pearson=lambda_pearson,
            lambda_mse=lambda_mse,
            lambda_cosine=lambda_cosine,
            lambda_gene_pearson=lambda_gene_pearson,
            lambda_gene_cosine=lambda_gene_cosine,
            lambda_reg=lambda_reg,
            lambda_sparse=lambda_sparse,
            lambda_proportion=lambda_proportion,
            save_reconstructed_genes=save_reconstructed_genes,
            use_dynamic_cluster_repr=use_dynamic_cluster_repr,
            k_cells_per_cluster=k_cells_per_cluster,
            precompute_knn=precompute_knn,
            use_learnable_weights=use_learnable_weights,
            weight_threshold=weight_threshold,
            scale_basis=scale_basis,
            device=device,
            seed=seed
        )
    
    # 单次运行（禁用网格搜索）
    # 如果 k_celltype_range 是单元素列表，使用该值覆盖 k_celltype
    if k_celltype_range is not None and len(k_celltype_range) == 1:
        k_celltype = k_celltype_range[0]
    # 如果是空列表或 None，使用函数参数的 k_celltype 默认值

    save_outputs = bool(output_dir)
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
    
    sample_name = Path(st_file).stem
    use_memory_mode = vae.is_memory_mode()
    
    # 配置已经在函数开头打印过了，这里不再重复打印
    
    # 初始化 trainer (seed 会在 __init__ 中设置)
    # output_dir 可以是 None（纯内存模式），trainer 在实际保存文件时才创建目录
    trainer = GATDeconvolution(
        stage1_model_path=vae.model_path,
        output_dir=output_dir,  # 直接传递，可以是 None
        device=device,
        weight_threshold=weight_threshold,
        stage1_artifacts=vae if use_memory_mode else None,
        seed=seed
    )
    trainer.k_spatial = k_spatial
    trainer.scale_basis = scale_basis
    trainer.save_reconstructed_genes = save_reconstructed_genes  # 设置是否保存重构基因
    
    # 加载 VAE 编码器
    trainer.load_vae_encoder()
    
    if trainer.sc_clusters is None:
        raise ValueError("Stage 1 missing cluster info!")
    
    n_clusters = trainer.celltype_prototypes.shape[0]
    n_genes = len(trainer.marker_genes)
    
    trainer.k_celltype = k_celltype

    if not _silent_header:  # 网格搜索时不重复打印
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
        if not _silent_header:  # 网格搜索时不重复打印
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
    elif scale_basis == 'fixed_10':
        # 固定缩放因子10：比例×10 = 细胞数量
        spot_total_counts = np.full(st_X_raw.shape[0], 10.0)
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
    
    # 保存 spot_total_counts 到 vae 对象，便于后续使用（如重建基因表达）
    vae.spot_total_counts = spot_total_counts
    
    # ===== 动态cluster表示：预计算k-nearest cells =====
    knn_cell_indices = None
    if use_dynamic_cluster_repr:
        if vae.sc_cell_embeddings is None or vae.sc_cell_expressions_raw is None or vae.sc_cell_labels is None:
            raise ValueError(
                "Dynamic cluster representation requires sc_cell_embeddings, sc_cell_expressions_raw, "
                "and sc_cell_labels from stage1. Please ensure stage1 was run with recent version."
            )
        
        if precompute_knn:
            if not _silent_header:
                print(f"\n预计算k-nearest cells (k={k_cells_per_cluster})...")
            
            # 导入预计算工具
            from ..utils.knn_utils import precompute_knn_cells_torch
            import torch
            
            # 计算ST embeddings（用于找最近细胞）
            st_data_tensor = torch.FloatTensor(st_X_embed).to(trainer.device)
            with torch.no_grad():
                st_embeddings_mu, _ = trainer.vae_encoder(st_data_tensor)
                st_embeddings = st_embeddings_mu.cpu()  # [n_spots, latent_dim]
            
            # 预计算k-nearest cells
            sc_embeddings_torch = torch.FloatTensor(vae.sc_cell_embeddings).to(trainer.device)
            sc_labels_torch = torch.LongTensor(vae.sc_cell_labels).to(trainer.device)
            
            knn_cell_indices = precompute_knn_cells_torch(
                spot_embeddings=st_embeddings,
                sc_cell_embeddings=sc_embeddings_torch,
                sc_cell_labels=sc_labels_torch,
                k_cells_per_cluster=k_cells_per_cluster
            )  # [n_spots, n_clusters, k]
            
            # 保存到vae对象
            vae.knn_cell_indices = knn_cell_indices.cpu().numpy()
            
            if not _silent_header:
                print(f"✓ 预计算完成: {knn_cell_indices.shape}")
    
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
        spot_total_counts=spot_total_counts,
        # 动态cluster参数
        use_dynamic_cluster_repr=use_dynamic_cluster_repr,
        k_cells_per_cluster=k_cells_per_cluster,
        use_learnable_weights=use_learnable_weights,
        sc_cell_expressions=vae.sc_cell_expressions_raw if use_dynamic_cluster_repr else None
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
        print_every=print_every,
        # 动态cluster参数
        knn_cell_indices=vae.knn_cell_indices if use_dynamic_cluster_repr and precompute_knn else None,
        sc_cell_embeddings=vae.sc_cell_embeddings if use_dynamic_cluster_repr else None
    )
    
    # 保存超参数到 txt 文件（合并 Stage 1 配置，如果存在）
    if save_outputs:
        import datetime
        config_path = f"{output_dir}/config.txt"
        
        # 尝试读取 Stage 1 配置
        stage1_config_path = f"{output_dir}/stage1_config.txt"
        stage1_config_content = ""
        if os.path.exists(stage1_config_path):
            with open(stage1_config_path, 'r') as f:
                stage1_config_content = f.read()
            # 删除 Stage 1 的独立配置文件
            os.remove(stage1_config_path)
        
        with open(config_path, 'w') as f:
            # 如果有 Stage 1 配置，先写入
            if stage1_config_content:
                f.write(stage1_config_content)
                f.write("\n\n")
            
            # 写入 Stage 2 配置
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
            f.write(f"Dynamic Cluster Representation:\n")
            f.write(f"  Enabled:       {use_dynamic_cluster_repr}\n")
            if use_dynamic_cluster_repr:
                f.write(f"  K Cells/Cluster: {k_cells_per_cluster}\n")
                f.write(f"  Learnable Weights: {use_learnable_weights}\n")
                f.write(f"  Precompute KNN: {precompute_knn}\n\n")
            else:
                f.write("\n")
            f.write(f"Output Options:\n")
            f.write(f"  Save Reconstructed Genes: {save_reconstructed_genes}\n\n")
            f.write(f"Results:\n")
            f.write(f"  N Clusters:    {n_clusters}\n")
            f.write(f"  N Genes:       {n_genes}\n\n")
            f.write(f"Final Losses:\n")
            f.write(f"  Best Pearson:      {results.get('best_pearson', 'N/A'):.6f}\n")
            f.write(f"  Best MSE:          {results.get('best_mse', 'N/A'):.6f}\n")
            f.write(f"  Best Cosine:       {results.get('best_cosine', 'N/A'):.6f}\n")
            f.write(f"  Best Gene Pearson: {results.get('best_gene_pearson', 'N/A'):.6f}\n")
            f.write(f"  Best Gene Cosine:  {results.get('best_gene_cosine', 'N/A'):.6f}\n\n")
            f.write(f"System:\n")
            f.write(f"  Memory Mode:   {use_memory_mode}\n")
            f.write(f"  Device:        {device or 'auto'}\n")
            f.write("="*60 + "\n")
        print(f"Configuration saved to: {config_path}")
    
    return {
        'deconv': results.get('composition_df'),
        'deconv_path': results.get('composition_path'),
        'sample_name': sample_name,
        'n_clusters': n_clusters,
        'best_pearson': results.get('best_pearson'),
        'deconv_weights_raw': results.get('deconv_weights_raw'),  # 未合并的 cluster-level 权重
        'metrics': {
            'pearson': results.get('best_pearson'),
            'mse': results.get('best_mse'),
            'cosine': results.get('best_cosine'),
            'gene_pearson': results.get('best_gene_pearson'),
            'gene_cosine': results.get('best_gene_cosine'),
            'n_clusters': n_clusters,
            'n_genes': n_genes
        }
    }


def run_deconv_auto_k(
    vae: Stage1Artifacts,
    st_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    k_celltype_range: list = None,
    **kwargs
) -> Dict[str, Any]:
    """Stage 2 with automatic k_celltype grid search (纯内存版本)
    
    遍历 k_celltype 候选值，选择 spot_pearson + spot_cosine 最小的结果。
    所有训练都在内存中进行，只保存最优结果到磁盘（如果提供 output_dir）。
    
    Args:
        vae: Stage1Artifacts 对象
        st_file: ST h5ad 文件路径
        output_dir: 输出目录（仅保存最优结果的 deconv 矩阵和配置）
        k_celltype_range: k_celltype 候选值列表，默认 [15, 20, 25, 30]
        **kwargs: 传递给 run_deconv 的其他参数
        
    Returns:
        最优结果的字典，包含:
        - 'best_k': 最优的 k_celltype
        - 'best_score': 最优的 pearson + cosine
        - 'all_trials': 所有尝试的摘要列表
        - 其他 run_deconv 返回的字段（使用最优 k）
    """

    # 简化输出
    print(f"\nGrid search: k_celltype = {k_celltype_range}")
    
    best_k = None
    best_score = float('inf')
    best_result = None
    all_trials = []
    
    for i, k in enumerate(k_celltype_range, 1):
        print(f"  Trial {i}/{len(k_celltype_range)} (k={k})...", end=" ", flush=True)
        
        # 运行 deconv（纯内存模式，不保存文件，不打印训练过程）
        # 从 kwargs 中提取 print_every，避免重复传递
        deconv_kwargs = kwargs.copy()
        print_every_val = deconv_kwargs.pop('print_every', 9999)
        
        result = run_deconv(
            vae=vae,
            st_file=st_file,
            output_dir=None,  # 不保存文件，纯内存
            k_celltype=k,
            k_celltype_range=[],  # 禁用递归网格搜索（空列表表示禁用）
            print_every=print_every_val,  # 使用超参数的值，默认不打印
            _silent_header=True,  # 不打印配置头部
            **deconv_kwargs
        )
        
        # 计算评分：mse + cosine + gene_pearson（越小越好）
        pearson = result['metrics']['pearson']
        cosine = result['metrics']['cosine']
        mse = result['metrics']['mse']
        gene_pearson = result['metrics']['gene_pearson']
        score = mse + cosine + gene_pearson
        
        # 保存摘要（不保存完整的 deconv 矩阵，节省内存）
        trial_summary = {
            'k_celltype': k,
            'pearson': pearson,
            'cosine': cosine,
            'mse': mse,
            'gene_pearson': gene_pearson,
            'score': score
        }
        all_trials.append(trial_summary)
        
        # 更新最优结果（保存完整结果）
        is_best = score < best_score
        if is_best:
            best_score = score
            best_k = k
            best_result = result  # 保留最优的完整结果
        
        # 打印结果（紧凑格式）
        status = "*" if is_best else " "
        print(f"{status} Score={score:.4f} (M={mse:.4f}, C={cosine:.4f}, GP={gene_pearson:.4f})")
    
    # 打印最终结果（简化）
    print(f"\nBest: k={best_k}, Score={best_score:.4f}\n")
    
    # 如果提供了 output_dir，保存最优结果（不重新训练）
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving best result (k={best_k}) to {output_dir}...")
        
        # 保存 deconv 矩阵
        import pandas as pd
        import numpy as np
        deconv_df = best_result['deconv']
        deconv_path = os.path.join(output_dir, f"{best_result['sample_name']}_cell_composition.csv")
        deconv_df.to_csv(deconv_path)
        print(f"Deconvolution matrix saved to: {deconv_path}")
        
        # 如果启用了 save_reconstructed_genes，直接用 deconv 矩阵重建基因表达
        save_reconstructed_genes = kwargs.get('save_reconstructed_genes', False)
        if save_reconstructed_genes and vae.celltype_expressions_full is not None:
            print(f"Computing reconstructed gene expression...")
            
            # 使用原始的 cluster-level 权重（未合并）
            # best_result 中包含 'deconv_weights_raw'：[n_spots, n_clusters]
            deconv_weights = best_result.get('deconv_weights_raw')
            if deconv_weights is None:
                # 备用方案：从 deconv_df 提取（可能不匹配）
                print("⚠️  Warning: deconv_weights_raw not found, using deconv_df (may cause shape mismatch)")
                deconv_weights = deconv_df.values
            
            # 获取细胞类型的完整基因表达（来自 Stage 1）
            celltype_expr_full = vae.celltype_expressions_full  # [n_clusters, n_all_genes]
            
            # 重建：deconv @ celltype_expressions
            mixed_expr_full = np.dot(deconv_weights, celltype_expr_full)  # [n_spots, n_all_genes]
            
            # 缩放到原始 counts（如果有 scale_basis）
            scale_basis = kwargs.get('scale_basis', 'all')
            if scale_basis == 'none':
                reconstructed_full_expr = mixed_expr_full
            elif scale_basis == 'fixed_10':
                # 固定缩放因子10：比例×10 = 细胞数量
                reconstructed_full_expr = mixed_expr_full * 10.0
            elif vae.spot_total_counts is not None:
                spot_counts = vae.spot_total_counts[:len(deconv_weights)]
                
                if scale_basis == 'all':
                    mixed_totals = mixed_expr_full.sum(axis=1, keepdims=True)
                elif scale_basis == 'hvg' and vae.hvg_genes_union is not None:
                    hvg_indices = [i for i, g in enumerate(vae.all_genes) if g in vae.hvg_genes_union]
                    mixed_totals = mixed_expr_full[:, hvg_indices].sum(axis=1, keepdims=True)
                else:  # marker
                    marker_indices = [i for i, g in enumerate(vae.all_genes) if g in vae.marker_genes]
                    mixed_totals = mixed_expr_full[:, marker_indices].sum(axis=1, keepdims=True)
                
                scale = spot_counts[:, np.newaxis] / (mixed_totals + 1e-8)
                reconstructed_full_expr = mixed_expr_full * scale
            else:
                reconstructed_full_expr = mixed_expr_full
            
            # 保存为 CSV
            reconstructed_df = pd.DataFrame(
                reconstructed_full_expr,
                columns=vae.all_genes,
                index=deconv_df.index
            )
            reconstructed_path = os.path.join(output_dir, f"{best_result['sample_name']}_reconstructed_all_genes.csv")
            reconstructed_df.to_csv(reconstructed_path)
            print(f"Reconstructed genes saved to: {reconstructed_path}")
        
        # 保存配置文件（合并 Stage 1 配置，如果存在）
        import datetime
        config_path = os.path.join(output_dir, 'config.txt')
        
        # 尝试读取 Stage 1 配置
        stage1_config_path = os.path.join(output_dir, 'stage1_config.txt')
        stage1_config_content = ""
        if os.path.exists(stage1_config_path):
            with open(stage1_config_path, 'r') as f:
                stage1_config_content = f.read()
            # 删除 Stage 1 的独立配置文件
            os.remove(stage1_config_path)
        
        with open(config_path, 'w') as f:
            # 如果有 Stage 1 配置，先写入
            if stage1_config_content:
                f.write(stage1_config_content)
                f.write("\n\n")
            
            # 写入 Stage 2 配置
            f.write("="*60 + "\n")
            f.write("Stage 2: GAT Deconvolution (Auto K Grid Search)\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 从 kwargs 中提取所有超参数
            n_epochs = kwargs.get('n_epochs', 250)
            lr = kwargs.get('lr', 5e-3)
            batch_size = kwargs.get('batch_size', 512)
            k_spatial = kwargs.get('k_spatial', 5)
            gat_hidden_dim = kwargs.get('gat_hidden_dim', 512)
            gat_layers = kwargs.get('gat_layers', 4)
            gat_heads = kwargs.get('gat_heads', 4)
            dropout = kwargs.get('dropout', 0.1)
            lambda_pearson = kwargs.get('lambda_pearson', 5.0)
            lambda_mse = kwargs.get('lambda_mse', 0.001)
            lambda_cosine = kwargs.get('lambda_cosine', 5.0)
            lambda_gene_pearson = kwargs.get('lambda_gene_pearson', 1.0)
            lambda_gene_cosine = kwargs.get('lambda_gene_cosine', 1.0)
            lambda_reg = kwargs.get('lambda_reg', 0.1)
            lambda_sparse = kwargs.get('lambda_sparse', 1)
            lambda_proportion = kwargs.get('lambda_proportion', 0.1)
            weight_threshold = kwargs.get('weight_threshold', 0.001)
            scale_basis = kwargs.get('scale_basis', 'all')
            device = kwargs.get('device', None)
            seed = kwargs.get('seed', 42)
            save_reconstructed_genes = kwargs.get('save_reconstructed_genes', False)
            
            f.write(f"Training Hyperparameters:\n")
            f.write(f"  Epochs:        {n_epochs}\n")
            f.write(f"  Learning Rate: {lr}\n")
            f.write(f"  Batch Size:    {batch_size}\n")
            f.write(f"  Seed:          {seed}\n\n")
            f.write(f"Grid Search:\n")
            f.write(f"  Candidates:    {k_celltype_range}\n")
            f.write(f"  Best k:        {best_k}\n")
            f.write(f"  Best Score:    {best_score:.6f}\n\n")
            f.write(f"Graph Construction:\n")
            f.write(f"  K Spatial:     {k_spatial}\n")
            f.write(f"  K Celltype:    {best_k} (from grid search)\n")
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
            
            # 动态cluster参数
            use_dynamic_cluster_repr = kwargs.get('use_dynamic_cluster_repr', False)
            k_cells_per_cluster = kwargs.get('k_cells_per_cluster', 10)
            use_learnable_weights = kwargs.get('use_learnable_weights', True)
            precompute_knn = kwargs.get('precompute_knn', True)
            
            f.write(f"Dynamic Cluster Representation:\n")
            f.write(f"  Enabled:       {use_dynamic_cluster_repr}\n")
            if use_dynamic_cluster_repr:
                f.write(f"  K Cells/Cluster: {k_cells_per_cluster}\n")
                f.write(f"  Learnable Weights: {use_learnable_weights}\n")
                f.write(f"  Precompute KNN: {precompute_knn}\n\n")
            else:
                f.write("\n")
            
            f.write(f"Output Options:\n")
            f.write(f"  Save Reconstructed Genes: {save_reconstructed_genes}\n\n")
            f.write(f"All Trials (Score = MSE + Cosine + Gene_Pearson):\n")
            for trial in all_trials:
                marker = "*" if trial['k_celltype'] == best_k else " "
                f.write(f"  {marker} k={trial['k_celltype']:2d}: "
                       f"MSE={trial['mse']:.6f}, "
                       f"Cosine={trial['cosine']:.6f}, "
                       f"Gene_P={trial['gene_pearson']:.6f}, "
                       f"Score={trial['score']:.6f}\n")
            f.write(f"\nFinal Results:\n")
            f.write(f"  Best Pearson:      {best_result['metrics']['pearson']:.6f}\n")
            f.write(f"  Best MSE:          {best_result['metrics']['mse']:.6f}\n")
            f.write(f"  Best Cosine:       {best_result['metrics']['cosine']:.6f}\n")
            f.write(f"  Best Gene Pearson: {best_result['metrics']['gene_pearson']:.6f}\n")
            f.write(f"  Best Gene Cosine:  {best_result['metrics']['gene_cosine']:.6f}\n")
            f.write(f"  N Clusters:        {best_result['metrics']['n_clusters']}\n")
            f.write(f"  N Genes:           {best_result['metrics']['n_genes']}\n\n")
            f.write(f"System:\n")
            f.write(f"  Device:        {device or 'auto'}\n")
            f.write("="*60 + "\n")
        print(f"Configuration saved to: {config_path}")
        
        # 更新 best_result 的路径信息
        best_result['deconv_path'] = deconv_path
    
    # 添加网格搜索信息到结果
    best_result['best_k'] = best_k
    best_result['best_score'] = best_score
    best_result['all_trials'] = all_trials
    
    return best_result


# Backward-compatible alias
deconvolve_spots = run_deconv
