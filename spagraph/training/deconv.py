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
import pandas as pd
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
    auto_library_size: Optional[float] = None  # ✅ Stage1自动计算的library_size (ST_HVG_depth / SC_HVG_depth)
    
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
    
    sc_cell_marker_expressions: Optional[Any] = None
    sc_celltype_labels: Optional[Any] = None

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
            auto_library_size=results.get('auto_library_size', 1.0),  # ✅ 传递自动计算的library_size
            # 动态cluster数据
            sc_cell_embeddings=results.get('sc_cell_embeddings'),
            sc_cell_expressions_raw=results.get('sc_cell_expressions_raw'),
            sc_cell_labels=results.get('sc_cell_labels'),
            sc_cell_marker_expressions=results.get('sc_cell_marker_expressions'),
            sc_celltype_labels=results.get('sc_celltype_labels')
        )


def run_deconv(
    vae: Stage1Artifacts,
    st_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    # Training parameters
    n_epochs: int = 300,
    lr: float = 5e-3,
    batch_size: int = 128,
    print_every: int = 25,
    # Graph parameters
    k_spatial: int = 5,
    k_celltype: Any = 20,  # int: 单次运行, list: 网格搜索（如 [20, 25, 30]）
    # GAT architecture
    gat_hidden_dim: int = 512,
    gat_layers: int = 4,
    gat_heads: int = 4,
    dropout: float = 0.1,
    # Loss weights
    lambda_pearson: float = 1,
    lambda_mse: float = 0,
    lambda_cosine: float = 5.0,
    lambda_gene_pearson: float = 0,   # disabled: gene-level metrics are batch-dependent, used for monitoring only
    lambda_gene_cosine: float = 0,    # disabled: gene-level metrics are batch-dependent, used for monitoring only
    lambda_reg: float = 0.1,
    lambda_sparse: float = 0,
    lambda_proportion: float = 0.01,
    lambda_poisson: float = 0.0,
    lambda_spatial: float = 0.0,
    spatial_temperature: float = 1.0,
    heldout_gene_fraction: float = 0.0,
    # Output options
    save_reconstructed_genes: bool = False,  # 是否保存重构的全基因表达
    save_all_trials: bool = False,  # 网格搜索时是否保存所有k_celltype的deconv矩阵（命名为xxx_k15.csv等）
    # Dynamic cluster representation (optional)
    use_dynamic_cluster_repr: bool = True,  # 启用动态cluster表示（用k个最近细胞代替cluster平均）
    k_cells_per_cluster: int = 10,  # 每个cluster使用多少个最近细胞
    precompute_knn: bool = True,  # 是否在训练前预计算k-nearest cells（推荐True）
    # Other
    full_graph_training: bool = True,
    restore_best_state: bool = True,
    signature_init: bool = False,
    signature_only: bool = False,
    signature_ridge: float = 1e-4,
    signature_prior_strength: float = 1.0,
    signature_platform_calibration: bool = False,
    signature_calibration_iterations: int = 5,
    reference_grouping: str = 'leiden',
    reference_signature_mode: str = 'pseudobulk',
    signature_gene_selection: str = 'stage1_markers',
    signature_genes_per_celltype: int = 100,
    signature_composition_power: float = 1.2,
    weight_threshold: float = 0.001,
    scale_basis: str = 'all',
    use_ols_scaling: bool = False,  # 是否使用 OLS 最小二乘缩放（纯数学方法，非模型参数）
    library_size: float = 1.0,  # 手动文库因子，在 scale 基础上再乘此值（默认 1.0）
    device: Optional[str] = None,
    seed: int = 42,
    _silent_header: bool = False,  # 内部参数：禁止打印配置头部
) -> Dict[str, Any]:
    """Stage 2: GAT Deconvolution
    
    用于空间转录组的反卷积分析，基于 GAT 图神经网络预测每个 spot 的细胞类型组成。
    
    Args:
        vae: Stage1Artifacts，包含第一阶段训练的 VAE 模型和聚类信息
        st_file: 空间转录组 h5ad 文件路径（如果为 None，从 vae 中获取）
        output_dir: 输出目录（保存 deconv 矩阵、配置文件等）
        
        # 训练参数
        n_epochs: 训练轮数
        lr: 学习率
        batch_size: 批次大小
        print_every: 每隔多少轮打印一次训练信息
        
        # 图构建参数
        k_spatial: 空间邻居数量
        k_celltype: **动态 cluster 的 k 值**
            - 整数（如 20）: 单次运行，每个 cluster 使用 k 个最近细胞
            - 列表（如 [20, 25, 30]）: 网格搜索，自动选择最优 k
        
        # GAT 架构参数
        gat_hidden_dim: GAT 隐藏层维度
        gat_layers: GAT 层数
        gat_heads: 注意力头数
        dropout: Dropout 概率
        
        # 损失函数权重
        lambda_pearson, lambda_mse, lambda_cosine: spot 级别的损失权重
        lambda_gene_pearson, lambda_gene_cosine: 基因级别的损失权重
        lambda_reg, lambda_sparse, lambda_proportion: 正则化损失权重
        
        # 输出选项
        save_reconstructed_genes: 是否保存重构的全基因表达（包括 spot-cell 级别）
        save_all_trials: 网格搜索时是否保存所有 k 的 deconv 矩阵
        
        # 动态 cluster 表示
        use_dynamic_cluster_repr: 是否启用动态 cluster（用 k 个最近细胞代替平均）
        k_cells_per_cluster: 每个 cluster 使用多少个最近细胞（被 k_celltype 覆盖）
        precompute_knn: 是否预计算 k-nearest cells（推荐 True）
        
        # 其他
        weight_threshold: deconv 权重阈值（小于此值的权重置零）
        scale_basis: 缩放基准（'all', 'hvg', 'marker', 'none', 'fixed_10'）
        use_ols_scaling: 是否使用 OLS 最小二乘缩放（True=OLS，False=sum-based，纯数学方法）
        library_size: 手动文库因子，在 scale 基础上再乘此值（默认 1.0，>1 放大，<1 缩小）
        device: 计算设备（'cuda' 或 'cpu'）
        seed: 随机种子
        
    Returns:
        字典，包含:
        - 'deconv': deconv 矩阵（DataFrame, [n_spots, n_clusters]）
        - 'metrics': 评估指标（pearson, mse, cosine 等）
        - 'sample_name': 样本名称
        - 如果是网格搜索:
            - 'best_k': 最优 k_celltype
            - 'best_score': 最优评分
            - 'all_trials': 所有试验的摘要
    
    Examples:
        >>> # 单次运行
        >>> results = spg.deconv(
        ...     vae=vae,
        ...     st_h5ad="data/st.h5ad",
        ...     output_dir="output/",
        ...     k_celltype=20  # 单个值
        ... )
        
        >>> # 网格搜索
        >>> results = spg.deconv(
        ...     vae=vae,
        ...     st_h5ad="data/st.h5ad",
        ...     output_dir="output/",
        ...     k_celltype=[20, 25, 30, 35]  # 列表触发网格搜索
        ... )
        >>> print(f"Best k: {results['best_k']}")
    """
    # 参数检查
    if isinstance(k_celltype, (list, tuple)) and len(k_celltype) == 0:
        raise ValueError("k_celltype candidate list cannot be empty")
    if signature_ridge < 0:
        raise ValueError("signature_ridge must be non-negative")
    if signature_prior_strength < 0:
        raise ValueError("signature_prior_strength must be non-negative")
    if signature_calibration_iterations < 0:
        raise ValueError("signature_calibration_iterations must be non-negative")
    if signature_only and not signature_init:
        raise ValueError("signature_only requires signature_init=True")
    if reference_grouping not in {'leiden', 'celltype'}:
        raise ValueError("reference_grouping must be 'leiden' or 'celltype'")
    if reference_signature_mode not in {'pseudobulk', 'cell_normalized', 'log_normalized'}:
        raise ValueError(
            "reference_signature_mode must be pseudobulk, cell_normalized, or log_normalized"
        )
    if signature_gene_selection not in {'stage1_markers', 'celltype_specific', 'all_shared'}:
        raise ValueError(
            "signature_gene_selection must be stage1_markers, celltype_specific, or all_shared"
        )
    if signature_genes_per_celltype < 1:
        raise ValueError("signature_genes_per_celltype must be at least 1")
    if not np.isfinite(signature_composition_power) or signature_composition_power <= 0:
        raise ValueError("signature_composition_power must be finite and positive")
    if signature_gene_selection != 'stage1_markers' and reference_grouping != 'celltype':
        raise ValueError(
            "celltype-specific or all-shared signature genes require reference_grouping='celltype'"
        )
    if not 0.0 <= heldout_gene_fraction < 1.0:
        raise ValueError("heldout_gene_fraction must be in [0, 1)")

    if st_file is None:
        st_file = vae.st_file
    if st_file is None:
        raise ValueError("st_file is required")

    # ✅ 如果用户未手动指定 library_size，使用 Stage1 自动计算的值
    if library_size == 1.0 and hasattr(vae, 'auto_library_size') and vae.auto_library_size is not None:
        library_size = vae.auto_library_size
        if not _silent_header:
            print(f"✅ 使用 Stage1 自动计算的 library_size: {library_size:.4f}")

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
        # 判断 k_celltype 是列表还是整数
        if isinstance(k_celltype, (list, tuple)) and len(k_celltype) > 1:
            print(f"K Celltype:         Grid Search {k_celltype}")
        else:
            k_value = k_celltype[0] if isinstance(k_celltype, (list, tuple)) else k_celltype
            print(f"K Celltype:         {k_value}")
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
            print(f"  Precompute KNN:   {precompute_knn}")
        print(f"Scale Basis:        {scale_basis}")
        print(f"OLS Scaling:        {'✅ Enabled' if use_ols_scaling else 'Disabled (sum-based)'}")
        print(f"Library Size:       {library_size}")
        print(f"Weight Threshold:   {weight_threshold}")
        print(f"Seed:               {seed}")
        print(f"Save to Disk:       {bool(output_dir)}")
        print(f"{'='*60}\n")
    
    # 网格搜索启用条件：k_celltype 是列表且长度 > 1
    if isinstance(k_celltype, (list, tuple)) and len(k_celltype) > 1:
        # 启用网格搜索
        return run_deconv_auto_k(
            vae=vae,
            st_file=st_file,
            output_dir=output_dir,
            k_celltype_range=k_celltype,  # 传递列表
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
            lambda_poisson=lambda_poisson,
            lambda_spatial=lambda_spatial,
            spatial_temperature=spatial_temperature,
            heldout_gene_fraction=heldout_gene_fraction,
            save_reconstructed_genes=save_reconstructed_genes,
            save_all_trials=save_all_trials,
            use_dynamic_cluster_repr=use_dynamic_cluster_repr,
            k_cells_per_cluster=k_cells_per_cluster,
            precompute_knn=precompute_knn,
            full_graph_training=full_graph_training,
            restore_best_state=restore_best_state,
            signature_init=signature_init,
            signature_only=signature_only,
            signature_ridge=signature_ridge,
            signature_prior_strength=signature_prior_strength,
            signature_platform_calibration=signature_platform_calibration,
            signature_calibration_iterations=signature_calibration_iterations,
            reference_grouping=reference_grouping,
            reference_signature_mode=reference_signature_mode,
            signature_gene_selection=signature_gene_selection,
            signature_genes_per_celltype=signature_genes_per_celltype,
            signature_composition_power=signature_composition_power,
            weight_threshold=weight_threshold,
            scale_basis=scale_basis,
            use_ols_scaling=use_ols_scaling,  # ✅ 传递 OLS 参数到网格搜索
            library_size=library_size,  # ✅ 传递 library_size 到网格搜索
            device=device,
            seed=seed
        )
    
    # 单次运行（禁用网格搜索）
    # 如果 k_celltype 是单元素列表，提取为整数
    if isinstance(k_celltype, (list, tuple)):
        if len(k_celltype) == 1:
            k_celltype = k_celltype[0]
        else:
            # 空列表，使用默认值 20
            k_celltype = 20

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
        seed=seed,
        use_ols_scaling=use_ols_scaling,  # ✅ 传递 OLS 缩放参数
        library_size=library_size  # ✅ 传递 library_size
    )
    trainer.k_spatial = k_spatial
    trainer.scale_basis = scale_basis
    trainer.save_reconstructed_genes = save_reconstructed_genes  # 设置是否保存重构基因
    
    # 加载 VAE 编码器
    trainer.load_vae_encoder()

    if reference_grouping == 'celltype':
        from spagraph.models.deconv_initialization import aggregate_reference_by_labels

        labels = getattr(vae, 'sc_celltype_labels', None)
        marker_expr = getattr(vae, 'sc_cell_marker_expressions', None)
        embeddings = getattr(vae, 'sc_cell_embeddings', None)
        raw_expr = getattr(vae, 'sc_cell_expressions_raw', None)
        if labels is None or marker_expr is None or embeddings is None or raw_expr is None:
            raise ValueError(
                "reference_grouping='celltype' requires aligned SC annotation, "
                "marker expression, embeddings, and raw expression from Stage 1"
            )
        grouped = aggregate_reference_by_labels(labels, embeddings, marker_expr, raw_expr)
        annotation_encoder = grouped['encoder']
        encoded_labels = grouped['encoded_labels']

        trainer.label_encoder = annotation_encoder
        trainer.celltype_prototypes = torch.as_tensor(
            grouped['prototypes'], dtype=torch.float32, device=trainer.device
        )
        trainer.celltype_expressions = torch.as_tensor(
            grouped['marker_signatures'], dtype=torch.float32, device=trainer.device
        )
        trainer.celltype_expressions_full = list(grouped['raw_signatures'])
        trainer.signature_cell_normalized_full = grouped['cell_normalized_signatures']
        trainer.signature_log_normalized_full = grouped['log_normalized_signatures']
        trainer.cluster_to_celltype = {
            str(name): str(name) for name in annotation_encoder.classes_
        }
        # Dynamic nearest-cell lookup must use the same reference grouping.
        vae.sc_cell_labels = encoded_labels
    
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
        graph_source = 'spatial_coordinates'
    else:
        if not _silent_header:  # 网格搜索时不重复打印
            print("⚠️ No spatial coords, using embedding-based KNN")
        spatial_coords = np.zeros((st_adata.n_obs, 2))
        trainer.use_embedding_knn = True
        graph_source = 'vae_embedding_knn'
    
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
    # D2/D3 signature initialization uses no simulated composition truth.
    initial_weights = None
    signature_selected_genes = None
    if signature_init:
        from spagraph.models.deconv_initialization import (
            compute_platform_calibrated_initialization,
            compute_signature_initialization,
            select_celltype_specific_genes,
        )

        if signature_gene_selection == 'stage1_markers' and reference_signature_mode == 'log_normalized':
            signatures_marker = np.asarray(
                trainer.celltype_expressions.detach().cpu(), dtype=np.float64
            )
            signature_fit_spots = st_X_embed
            signature_selected_genes = list(trainer.genes)
        elif signature_gene_selection != 'stage1_markers':
            shared_genes = [gene for gene in trainer.all_genes if gene in st_adata.var_names]
            if not shared_genes:
                raise ValueError("no shared genes are available for signature initialization")
            sc_shared_indices = np.asarray(
                [trainer.all_genes.index(gene) for gene in shared_genes], dtype=np.int64
            )
            st_shared = st_adata[:, shared_genes].X
            st_shared = st_shared.toarray() if hasattr(st_shared, 'toarray') else np.asarray(st_shared)
            selection_signatures = np.asarray(
                trainer.signature_cell_normalized_full[:, sc_shared_indices], dtype=np.float64
            )
            if signature_gene_selection == 'celltype_specific':
                selected = select_celltype_specific_genes(
                    selection_signatures,
                    top_per_celltype=signature_genes_per_celltype,
                )
            else:
                selected = np.arange(len(shared_genes), dtype=np.int64)
            signature_selected_genes = [shared_genes[index] for index in selected]
            if reference_signature_mode == 'log_normalized':
                signatures_marker = np.asarray(
                    trainer.signature_log_normalized_full[:, sc_shared_indices[selected]],
                    dtype=np.float64,
                )
                spot_totals = np.maximum(st_shared.sum(axis=1, keepdims=True), 1e-12)
                signature_fit_spots = np.log1p(st_shared[:, selected] / spot_totals * 1e4)
            elif reference_signature_mode == 'cell_normalized':
                signatures_marker = selection_signatures[:, selected]
                signature_fit_spots = st_shared[:, selected]
            else:
                signatures_full = np.vstack([
                    np.asarray(expr, dtype=np.float64)
                    for expr in trainer.celltype_expressions_full
                ])
                signatures_marker = signatures_full[:, sc_shared_indices[selected]]
                signature_fit_spots = st_shared[:, selected]
        else:
            if reference_signature_mode == 'cell_normalized':
                signatures_full = np.asarray(
                    getattr(trainer, 'signature_cell_normalized_full'), dtype=np.float64
                )
            else:
                signatures_full = np.vstack([
                    np.asarray(expr, dtype=np.float64)
                    for expr in trainer.celltype_expressions_full
                ])
            marker_indices = [trainer.all_genes.index(g) for g in trainer.genes]
            signatures_marker = signatures_full[:, marker_indices]
            signature_fit_spots = st_X_raw
            signature_selected_genes = list(trainer.genes)
        signature_gene_factors = None
        if signature_platform_calibration:
            initial_weights, signature_gene_factors = compute_platform_calibrated_initialization(
                spot_expression=signature_fit_spots,
                celltype_signatures=signatures_marker,
                ridge=signature_ridge,
                iterations=signature_calibration_iterations,
            )
        else:
            initial_weights = compute_signature_initialization(
                spot_expression=signature_fit_spots,
                celltype_signatures=signatures_marker,
                ridge=signature_ridge,
            )
        if not _silent_header:
            print(
                f"Signature initialization: shape={initial_weights.shape}, "
                f"ridge={signature_ridge:g}, prior_strength={signature_prior_strength:g}"
            )

    if signature_only:
        if initial_weights is None:
            raise ValueError("signature_only requires signature_init=True")
        from spagraph.models.deconv_initialization import power_calibrate_composition

        initial_weights = power_calibrate_composition(
            initial_weights, power=signature_composition_power
        )
        cluster_ids = list(trainer.label_encoder.classes_)
        mapping = {str(k): str(v) for k, v in (trainer.cluster_to_celltype or {}).items()}
        celltype_columns = [mapping.get(str(cluster_id), f"Cluster_{cluster_id}") for cluster_id in cluster_ids]
        signature_df = pd.DataFrame(initial_weights, index=st_adata.obs_names, columns=celltype_columns)
        if signature_df.columns.duplicated().any():
            signature_df = signature_df.T.groupby(level=0).sum().T
        signature_path = None
        if output_dir:
            signature_path = os.path.join(output_dir, f"{sample_name}_composition.csv")
            signature_df.to_csv(signature_path)
        return {
            'deconv': signature_df,
            'deconv_path': signature_path,
            'sample_name': sample_name,
            'n_clusters': n_clusters,
            'best_pearson': None,
            'best_epoch': 0,
            'best_total_loss': None,
            'graph_source': graph_source,
            'reference_grouping': reference_grouping,
            'reference_signature_mode': reference_signature_mode,
            'signature_gene_selection': signature_gene_selection,
            'signature_selected_genes': signature_selected_genes,
            'signature_gene_factors': signature_gene_factors,
            'signature_composition_power': signature_composition_power,
            'deconv_weights_raw': initial_weights,
            'metrics': {
                'pearson': None,
                'mse': None,
                'cosine': None,
                'gene_pearson': None,
                'gene_cosine': None,
                'n_clusters': n_clusters,
                'n_genes': n_genes,
            },
        }

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
        sc_cell_expressions=vae.sc_cell_expressions_raw if use_dynamic_cluster_repr else None,
        signature_prior_strength=(signature_prior_strength if signature_init else 0.0),
        lambda_poisson=lambda_poisson,
        lambda_spatial=lambda_spatial,
        spatial_temperature=spatial_temperature,
        heldout_gene_fraction=heldout_gene_fraction,
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
        sc_cell_embeddings=vae.sc_cell_embeddings if use_dynamic_cluster_repr else None,
        initial_weights=initial_weights,
        full_graph_training=full_graph_training,
        restore_best_state=restore_best_state,
    )
    
    # 保存超参数到 txt 文件
    if save_outputs:
        import datetime
        config_path = f"{output_dir}/config_deconv.txt"
        
        with open(config_path, 'w') as f:
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
        'best_epoch': results.get('best_epoch'),
        'best_total_loss': results.get('best_total_loss'),
        'graph_source': graph_source,
        'reference_grouping': reference_grouping,
        'reference_signature_mode': reference_signature_mode,
        'signature_gene_selection': signature_gene_selection,
        'signature_selected_genes': signature_selected_genes,
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
    save_all_trials: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Stage 2 with automatic k_celltype grid search (内部函数，由 run_deconv 自动调用)
    
    注意：用户不应直接调用此函数，应使用 run_deconv(k_celltype=[20, 25, 30])
    
    遍历 k_celltype 候选值，选择评分最优的结果。
    所有训练都在内存中进行，只保存最优结果到磁盘（如果提供 output_dir）。
    
    Args:
        vae: Stage1Artifacts 对象
        st_file: ST h5ad 文件路径
        output_dir: 输出目录（仅保存最优结果的 deconv 矩阵和配置）
        k_celltype_range: k_celltype 候选值列表
        save_all_trials: 是否保存所有试验的 deconv 矩阵（命名为 xxx_k20.csv）
        **kwargs: 传递给 run_deconv 的其他参数
        
    Returns:
        最优结果的字典，包含:
        - 'best_k': 最优的 k_celltype
        - 'best_score': 最优评分
        - 'all_trials': 所有尝试的摘要列表
        - 其他 run_deconv 返回的字段（使用最优 k）
    """

    # 简化输出
    if not k_celltype_range:
        raise ValueError("k_celltype_range must contain at least one candidate")
    if any(not isinstance(k, (int, np.integer)) or int(k) <= 0 for k in k_celltype_range):
        raise ValueError("all k_celltype candidates must be positive integers")

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
            k_celltype=k,  # 传递整数，不会触发网格搜索
            print_every=print_every_val,  # 使用超参数的值，默认不打印
            _silent_header=True,  # 不打印配置头部
            **deconv_kwargs
        )
        
        # 计算评分：pearson + cosine（越小越好）
        # gene_pearson/gene_cosine 仅作监控指标，不参与选 k
        pearson = result['metrics']['pearson']
        cosine = result['metrics']['cosine']
        mse = result['metrics']['mse']
        gene_pearson = result['metrics']['gene_pearson']
        gene_cosine = result['metrics']['gene_cosine']
        score = pearson + cosine

        # 保存摘要（不保存完整的 deconv 矩阵，节省内存）
        trial_summary = {
            'k_celltype': k,
            'pearson': pearson,
            'cosine': cosine,
            'mse': mse,
            'gene_pearson': gene_pearson,
            'gene_cosine': gene_cosine,
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
        print(f"{status} Score={score:.4f} (P={pearson:.4f}, C={cosine:.4f})")
        
        # 如果启用 save_all_trials 且提供了 output_dir，保存当前trial的deconv矩阵
        if save_all_trials and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            import pandas as pd
            deconv_df = result['deconv']
            trial_path = os.path.join(output_dir, f"{result['sample_name']}_cell_composition_k{k}.csv")
            deconv_df.to_csv(trial_path)
            print(f"  → Saved: {result['sample_name']}_cell_composition_k{k}.csv")
    
    # 打印最终结果（简化）
    print(f"\nBest: k={best_k}, Score={best_score:.4f}\n")
    
    # 如果提供了 output_dir，需要重新运行最优配置以正确保存所有文件（包括重建基因表达）
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Re-running optimal k={best_k} with output_dir to save all files...")
        
        # 重新运行最优配置（这次会保存所有文件，包括 save_reconstructed_genes）
        deconv_kwargs = kwargs.copy()
        deconv_kwargs.pop('print_every', None)  # 移除，使用默认值
        
        best_result = run_deconv(
            vae=vae,
            st_file=st_file,
            output_dir=output_dir,  # 这次设置正确的 output_dir
            k_celltype=best_k,  # 传递整数，不会触发网格搜索
            print_every=kwargs.get('print_every', 9999),  # 不打印训练过程（已找到最优）
            _silent_header=True,
            **deconv_kwargs
        )
        
        # 保存 deconv 矩阵
        import pandas as pd
        import numpy as np
        deconv_df = best_result['deconv']
        deconv_path = os.path.join(output_dir, f"{best_result['sample_name']}_cell_composition.csv")
        deconv_df.to_csv(deconv_path)
        print(f"Deconvolution matrix saved to: {deconv_path}")
        
        # 保存配置文件
        import datetime
        config_path = os.path.join(output_dir, 'config_deconv.txt')
        
        with open(config_path, 'w') as f:
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
            precompute_knn = kwargs.get('precompute_knn', True)
            
            f.write(f"Dynamic Cluster Representation:\n")
            f.write(f"  Enabled:       {use_dynamic_cluster_repr}\n")
            if use_dynamic_cluster_repr:
                f.write(f"  K Cells/Cluster: {k_cells_per_cluster}\n")
                f.write(f"  Precompute KNN: {precompute_knn}\n\n")
            else:
                f.write("\n")
            
            f.write(f"Output Options:\n")
            f.write(f"  Save Reconstructed Genes: {save_reconstructed_genes}\n\n")
            f.write(f"All Trials (Score = Cosine + Gene_Cosine):\n")
            for trial in all_trials:
                marker = "*" if trial['k_celltype'] == best_k else " "
                f.write(f"  {marker} k={trial['k_celltype']:2d}: "
                       f"Cosine={trial['cosine']:.6f}, "
                       f"Gene_C={trial['gene_cosine']:.6f}, "
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
