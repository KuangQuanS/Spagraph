import argparse
import glob
import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

# VAE已移除：使用MLPEncoder作为编码器，不再引用DualDecoderVAE
# 已经将DGI集成到HeteroSTModel (compute_dgi_loss)，不再需要独立的 DGIPretrainModel
from .cellcom_model import HeteroSTModel
from .cellcom_graph_builder import (
    STHeteroSubgraphDataset,
    hetero_subgraph_collate_fn,
    hetero_subgraph_collate_fn_batched,
    set_seed,
)
from .cellcom_evaluate import evaluate_cell_communication, plot_training_loss
from .lr_scores import calculate_lr_scores

def _scalar(x):
    """Convert tensor/number to Python float for logging/accumulation."""
    if isinstance(x, (float, int)):
        return float(x)
    try:
        return x.item()
    except Exception:
        return float(x)

def _print_stage3_header(args, device: torch.device):
    sample_name = Path(args.st_h5ad).stem if getattr(args, "st_h5ad", None) else "Unknown"
    gat_hidden_dims = [x.strip() for x in str(args.gat_hidden_dims).split(",") if x.strip()]

    print(f"\n{'='*60}")
    print("Stage 3: Cell Communication")
    print(f"{'='*60}")
    print(f"Sample:             {sample_name}")
    print(f"Device:             {device}")
    print(f"Epochs:             {args.epochs}")
    print(f"LR:                 {args.learning_rate}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Num Workers:        {getattr(args, 'num_workers', 0)}")
    print(f"K Spot Neighbors:   {args.n_spot_neighbors}")
    print(f"Use HVG for Comm:   {args.use_hvg_for_communication}")
    print(f"Ligand Expr Thr:    {getattr(args, 'ligand_expr_threshold', 3.0)} (CP10k)")
    print(f"Receptor Expr Thr:  {getattr(args, 'receptor_expr_threshold', 1.0)} (CP10k)")
    print(f"LR Score Thr:       {args.lr_score_threshold}")
    print(f"Same-Type Comm:     {getattr(args, 'allow_same_celltype_comm', False)}")
    print(f"Min Comm Edges:     {args.min_comm_edges}")
    print(f"Attention Thr:      {args.attention_threshold}")
    print(f"Export Unified:     {getattr(args, 'export_unified_csv', False)}")
    print(f"Export Filtered:    {getattr(args, 'export_filtered_csv', True)}")
    print(f"MLP:               {args.mlp_hidden_dims} -> {args.mlp_latent_dim}")
    print(f"GAT Architecture:   {len(gat_hidden_dims)}L × [{', '.join(gat_hidden_dims)}]D × {args.gat_heads}H")
    print(f"Dropout:            {args.gat_dropout}")
    print("Loss Weights:")
    print(f"  Mask Recon:       {args.lambda_mask_recon} (ratio={args.edge_mask_ratio})")
    print(f"  Node Recon:       {args.lambda_node_recon} (ratio={args.node_mask_ratio})")
    print(f"Mask Seed:          {args.mask_seed}")
    print(
        'Representative LR:  '
        + ('ignored' if getattr(args, 'ablation_no_lr_identity', True) else 'used (legacy mode)')
    )
    if getattr(args, "early_stop_patience", 0) > 0:
        print(f"Early Stop:         patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}")
    else:
        print("Early Stop:         Disabled")
    print(f"Seed:               {args.seed}")
    print(f"Output Dir:         {args.output_dir}")
    print(f"{'='*60}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Heterogeneous ST Communication Model Training')
    parser.add_argument('--deconv_dir', type=str, required=True, 
                       help='Stage1+Stage2 输出目录（包含 final_vae.pth, final_vae_cluster_data.npz 等）')
    parser.add_argument('--st_h5ad', type=str, required=True, help='空间转录组h5ad文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    
    # MLP参数
    parser.add_argument('--mlp_latent_dim', type=int, default=64, help='MLP隐空间维度')
    parser.add_argument('--mlp_hidden_dims', type=str, default='256,128', help='MLP隐层维度')
    
    # 图参数
    parser.add_argument('--n_spot_neighbors', type=int, default=6, help='Spot邻近数')
    
    # LR通讯参数
    parser.add_argument('--mean_expr_threshold', type=float, default=3.0, 
                       help='（已弃用）通用阈值：保留以兼容旧脚本，不再作为默认回退使用')

    parser.add_argument('--ligand_expr_threshold', type=float, default=3.0,
                       help='配体活跃基因阈值（CP10k）。默认=3.0')
    parser.add_argument('--receptor_expr_threshold', type=float, default=1.0,
                       help='受体活跃基因阈值（CP10k，通常低于配体）。默认=1.0')
    parser.add_argument('--active_expr_threshold', type=float, default=3.0,
                       help='（已弃用）活跃基因阈值。请使用ligand_expr_threshold和receptor_expr_threshold')
    parser.add_argument('--lr_score_threshold', type=float, default=3.0,
                       help='LR 通讯得分阈值（log1p 空间）。默认=3.0')
    parser.add_argument('--min_comm_edges', type=int, default=1, 
                       help='最小通讯边数阈值，少于此值的spot将被过滤 (default: 1)')
    parser.add_argument('--spot_cell_expr_csv', type=str, default=None,
                       help='预计算的spot-cell全基因表达CSV文件路径，如果提供则跳过构建步骤')
    parser.add_argument(
        '--lr_database_csv',
        type=str,
        default=None,
        help='Optional LR database CSV with ligand and receptor columns',
    )
    parser.add_argument('--save_lr_scores_csv', type=lambda x: str(x).lower() == 'true', default=False,
                       help='Whether to save Stage 3.4 lr_scores.csv (default: False)')
    parser.add_argument('--use_hvg_for_communication', type=lambda x: str(x).lower() == 'true', default=True,
                       help='只使用高变基因计算LR通讯（而非全部基因），减少计算量并保持特征一致性 (default: True)')
    parser.add_argument(
        '--allow_same_celltype_comm',
        type=lambda x: str(x).lower() == 'true',
        default=False,
        help='是否允许相同细胞类型之间的通讯（同-type across spots）；默认关闭以保持与旧版本一致 (default: False)',
    )

    # GAT参数
    parser.add_argument('--gat_hidden_dims', type=str, default='512,256,128', help='GAT隐层维度')
    parser.add_argument('--gat_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--gat_dropout', type=float, default=0.3, help='Dropout概率')
    
    # 模型参数
    parser.add_argument('--output_dim', type=int, default=120, help='输出维度')
    
    parser.add_argument('--lambda_mask_recon', type=float, default=1.0, help='mask边重构损失的权重 (default: 1.0)')
    parser.add_argument('--lambda_node_recon', type=float, default=0.5, help='节点特征重构损失的权重 (default: 0.5)')
    parser.add_argument('--attention_threshold', type=float, default=1,
                       help='注意力得分阈值，用于过滤边 (default: 0)')
    parser.add_argument('--export_unified_csv', type=lambda x: str(x).lower() == 'true', default=False,
                       help='Whether to export full lr_communication.csv (default: False)')
    parser.add_argument('--export_filtered_csv', type=lambda x: str(x).lower() == 'true', default=True,
                       help='Whether to export filtered lr_communication CSV (default: True)')
    parser.add_argument('--edge_mask_ratio', type=float, default=0.2, help='mask通讯边的比例 (默认20%%)')
    parser.add_argument('--node_mask_ratio', type=float, default=0.15, help='mask节点特征比例 (默认15%%)')
    parser.add_argument('--mask_seed', type=int, default=1234, help='验证阶段mask的固定随机种子')
    parser.add_argument('--lr_id_emb_dim', type=int, default=8, help='LR id 嵌入维度，用于通讯边特征')
    parser.add_argument(
        '--ablation_no_lr_identity',
        type=lambda x: str(x).lower() == 'true',
        default=True,
        help='Ignore the representative LR ID embedding on aggregated communication edges',
    )
    

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16, larger is more efficient on CPU)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker数量 (default: 0)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='每个epoch采样比例 (default: 1.0, 即全部数据; 0.3表示采样30%%)')
    parser.add_argument('--val_split', type=float, default=0.1, 
                       help='验证集比例 (default: 0.1，即10%%作为验证集)')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='早停patience，0表示不使用早停 (default: 0)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.1, help='早停最小改善阈值 (default: 0.0001)')
    
    return parser.parse_args()

#---------------------------------主程序---------------------------------
def main(args=None):
    # 解析参数
    if args is None:
        args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # ========== CPU优化设置 ==========
    if device.type == 'cpu':
        # 设置PyTorch使用的CPU线程数（充分利用多核）
        import multiprocessing
        n_threads = min(multiprocessing.cpu_count(), 8)  # 最多8线程，避免过度竞争
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(2)  # 操作间并行
        
        # 启用优化的矩阵运算
        torch.backends.mkl.is_available() and None  # 确保MKL可用
        
        print(f"CPU Optimization:   threads={n_threads}, interop=2")
    
    _print_stage3_header(args, device)
    
    # ========== 阶段1：加载数据和构建激活基因特征 ==========
    print(f"{'='*60}\nStage 3.1: Load Data\n{'='*60}")
    
    # ========== 加载必要的数据 ==========
    deconv_dir = args.deconv_dir
    
    # 1. CellChat database (for LR communication computation)
    # Use absolute path based on the project root directory
    project_root = Path(__file__).parent.parent.parent  # Go up to project root from cellcom/cellcom.py
    custom_lr_database = getattr(args, "lr_database_csv", None)
    cellchat_file = Path(custom_lr_database) if custom_lr_database else project_root / 'cellchat_human.csv'
    if not cellchat_file.exists():
        raise FileNotFoundError(
            f"LR database file not found: {cellchat_file}\\n"
            f"Provide lr_database_csv or ensure {project_root}/cellchat_human.csv exists."
        )
    lr_db = pd.read_csv(cellchat_file)
    required_lr_columns = {"ligand", "receptor"}
    if not required_lr_columns.issubset(lr_db.columns):
        raise ValueError(
            f"LR database must contain columns {sorted(required_lr_columns)}: {cellchat_file}"
        )
    lr_pairs = []
    for _, row in lr_db.iterrows():
        lig = str(row['ligand']).strip()
        rec = str(row['receptor']).strip()
        lr_pairs.append((lig, rec))
    print(f"LR database:        {cellchat_file}")
    print(f"LR pairs:           {len(lr_pairs)}")

    # 2. 加载ST数据
    adata = sc.read_h5ad(args.st_h5ad)
    print(f"ST shape:           {adata.shape}")
    
    # 3. 加载spot坐标
    spot_coords = adata.obsm['spatial'] if 'spatial' in adata.obsm else None
    if spot_coords is not None:
        n_spots = len(spot_coords)
        print(f"Spots:              {n_spots}")
    else:
        raise ValueError("找不到spot坐标")

    # 4. 加载spot-cluster反卷积比例矩阵
    cluster_composition_file = os.path.join(deconv_dir, '*_composition.csv')
    cluster_composition_files = glob.glob(cluster_composition_file)
    if getattr(args, 'composition_csv', None) and os.path.exists(args.composition_csv):
        cluster_composition = pd.read_csv(args.composition_csv, index_col=0)
    else:
        if not cluster_composition_files:
            raise FileNotFoundError(
                f"No composition CSV found under {deconv_dir}. "
                f"Provide composition_csv explicitly or ensure '*_composition.csv' exists."
            )
        cluster_composition = pd.read_csv(cluster_composition_files[0], index_col=0)

    # 记录cluster数量
    n_clusters = cluster_composition.shape[1]
    print(f"Clusters:           {n_clusters}")
    
    # ========== 阶段2：加载 spot-cell 全基因表达 ==========
    print(f"\n{'='*60}\nStage 3.2: Load Spot-Cell Expression\n{'='*60}")
    
    spot_names = adata.obs_names.tolist()
    spot_total_counts = np.array(adata.X.sum(axis=1)).flatten()
    print(f"Avg spot counts:    {spot_total_counts.mean():.1f}")
    
    # ✅ 查找第二阶段生成的 spot-cell 动态表达文件
    stage2_spot_cell_file = None
    for pattern in ['*_spot_cell_expr.csv']:
        matches = glob.glob(os.path.join(deconv_dir, pattern))
        if matches:
            stage2_spot_cell_file = matches[0]
            break
    
    # 优先使用第二阶段生成的，否则使用用户手动指定的
    manual_spot_cell_file = args.spot_cell_expr_csv if (args.spot_cell_expr_csv and os.path.exists(args.spot_cell_expr_csv)) else None
    spot_cell_file_to_load = stage2_spot_cell_file or manual_spot_cell_file
    
    if not spot_cell_file_to_load:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ 找不到 spot-cell 表达文件\n"
            f"{'='*80}\n"
            f"需要的文件: *_spot_cell_expr.csv\n"
            f"搜索目录: {deconv_dir}\n\n"
            f"解决方案:\n"
            f"  重新运行 Stage 2 (deconv) 并设置 save_reconstructed_genes=True\n"
            f"  这会自动生成 spot-cell 动态表达文件\n\n"
            f"  示例:\n"
            f"    spg.deconv(\n"
            f"        vae=vae,\n"
            f"        st_h5ad='data/st.h5ad',\n"
            f"        output_dir='{deconv_dir}',\n"
            f"        save_reconstructed_genes=True  # ← 必须设置\n"
            f"    )\n"
            f"{'='*80}\n"
        )
        raise FileNotFoundError(error_msg)
    
    # 加载 spot-cell 表达
    if stage2_spot_cell_file:
        print(f"Spot-cell expr:     {spot_cell_file_to_load}")
    else:
        print(f"Spot-cell expr:     {spot_cell_file_to_load}")
    
    spot_cell_expr_df = pd.read_csv(spot_cell_file_to_load, index_col=0)
    print(f"Spot-cell shape:    {spot_cell_expr_df.shape}")
    
    # 验证数据格式
    if spot_cell_expr_df.index.name != 'spot_cell':
        print("WARNING: spot-cell CSV index name is not 'spot_cell' (format may be unexpected)")
    
    # 过滤全为0的行（如果有）
    row_sums = spot_cell_expr_df.sum(axis=1)
    if (row_sums == 0).any():
        n_zeros = (row_sums == 0).sum()
        print(f"WARNING: dropped {n_zeros} all-zero spot-cell rows")
        spot_cell_expr_df = spot_cell_expr_df[row_sums > 0]
    
    print(f"Spot-cells kept:    {len(spot_cell_expr_df)}")
    
    # ========== 基于 spot-cell 表达选择高变基因 ==========
    print(f"\n{'='*60}\nStage 3.3: Select HVGs\n{'='*60}")
    
    # ✅ 使用 scanpy 选择高变基因（替代基于阈值的激活基因选择）
    # 创建临时 AnnData 对象用于高变基因选择
    temp_adata = sc.AnnData(X=spot_cell_expr_df.values, obs=pd.DataFrame(index=spot_cell_expr_df.index))
    temp_adata.var_names = spot_cell_expr_df.columns
    
    # normalize_total + log1p + highly_variable_genes
    sc.pp.normalize_total(temp_adata, target_sum=1e4)
    sc.pp.log1p(temp_adata)
    
    n_top_genes = 2000
    if temp_adata.n_vars <= n_top_genes:
        activated_genes = temp_adata.var_names.tolist()
    else:
        sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_top_genes, flavor='seurat_v3')
        activated_genes = temp_adata.var_names[temp_adata.var['highly_variable']].tolist()
    print(f"HVG genes:          {len(activated_genes)} / {spot_cell_expr_df.shape[1]}")
    
    # 所有高变基因都可用（因为是从 spot_cell_expr_df 中选出来的）
    available_activated_genes = activated_genes
    
    # ✅ 提取实际存在的 celltype（从 spot_cell_expr_df 的 index 中解析）
    # 说明：cell 节点特征必须来自动态 spot-cell 表达谱（spot_cell_expr_df），不使用 celltype 均值。
    cell_names = sorted(set([idx.rsplit('_', 1)[1] for idx in spot_cell_expr_df.index]))
    print(f"Cell types:         {len(cell_names)}")

    # 占位：仅用于提供基因顺序/维度信息，cell 节点真实特征来自 spot_cell_expr_df
    activated_cluster_expr = pd.DataFrame(
        np.zeros((len(cell_names), len(available_activated_genes)), dtype=np.float32),
        index=cell_names,
        columns=available_activated_genes
    )
    cluster_expr = activated_cluster_expr  # Dataset 仅使用 columns
    cell_expr = activated_cluster_expr     # Dataset 仅使用 columns/index
    cell_full_expr = pd.DataFrame(index=cell_names)

    # 直接使用 cluster_composition 作为 composition（假设列名已经是 celltype）
    # 如果列名与 cell_names 不完全匹配，重新索引以对齐
    composition = cluster_composition.reindex(columns=cell_names, fill_value=0.0)
    print(f"Composition shape:  {composition.shape}")
    
    # ========== 阶段3.5：预计算KNN邻域和LR通讯得分 ==========
    print(f"\n{'='*60}\nStage 3.4: Precompute LR Scores\n{'='*60}")
    
    knn_mask, csv_path, graph_data, comm_by_spot_pair, lr_pair_to_id, lr_id_to_pair = calculate_lr_scores(
        spot_coords=spot_coords,
        composition=composition,
        args=args,
        lr_pairs=lr_pairs,
        adata=adata,
        cell_full_expr=spot_cell_expr_df,  
        output_dir=args.output_dir,
        n_neighbors=args.n_spot_neighbors,
        hvg_genes=available_activated_genes if args.use_hvg_for_communication else None,
        ligand_expr_threshold=getattr(args, 'ligand_expr_threshold', 3.0),
        receptor_expr_threshold=getattr(args, 'receptor_expr_threshold', 1.0),
        save_lr_scores_csv=getattr(args, 'save_lr_scores_csv', False),
    )
    print(f"\n{'='*60}\nStage 3.5: Build Dataset\n{'='*60}")
    
    dataset = STHeteroSubgraphDataset(
        st_h5ad_path=args.st_h5ad,
        cluster_expr=activated_cluster_expr,  # 使用激活基因特征
        cell_expr=cell_expr,
        cell_full_expr=cell_full_expr,
        graph_data=graph_data,
        lr_pairs=lr_pairs,
        k_neighbors=args.n_spot_neighbors,
        load_lr_scores_csv=csv_path,
        min_comm_edges=args.min_comm_edges,
        valid_cell_types=cell_names,
        device=device,
        spot_cell_expr=spot_cell_expr_df,
        adata=adata,
        spot_names=spot_names,
        comm_by_spot_pair=comm_by_spot_pair,
        lr_pair_to_id=lr_pair_to_id,
        lr_id_to_pair=lr_id_to_pair
    )
    
    # ========== 数据集划分：训练集和验证集 ==========
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    # 使用固定种子确保可重复性
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\nTrain/Val split:    {train_size} / {val_size} (val={args.val_split:.1%})")
    lr_mapping_path = os.path.join(args.output_dir, "lr_pair_mapping.txt")
    with open(lr_mapping_path, 'w') as f:
        f.write("lr_id\tligand\treceptor\n")  # 表头
        for lr_id, (ligand, receptor) in dataset.lr_id_to_pair.items():
            f.write(f"{lr_id}\t{ligand}\t{receptor}\n")
    print(f"LR mapping saved:   {lr_mapping_path}")
    
    # 创建训练和验证数据加载器
    num_workers = max(0, args.num_workers)
    pin_memory = device.type == 'cuda'
    if args.sample_rate < 1.0:
        num_samples = int(len(train_dataset) * args.sample_rate)
        sampler = RandomSampler(train_dataset, num_samples=num_samples, replacement=False)
        print(f"Sampling:           {args.sample_rate*100:.1f}% ({num_samples}/{len(train_dataset)} spots/epoch)")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            collate_fn=hetero_subgraph_collate_fn_batched
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            collate_fn=hetero_subgraph_collate_fn_batched
        )
    
    # 验证集数据加载器（不打乱顺序）
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=hetero_subgraph_collate_fn_batched
    )
    
    print(f"\n{'='*60}\nStage 3.6: Build Model\n{'='*60}")
    gat_hidden_dims = [int(x) for x in args.gat_hidden_dims.split(',')]
    
    n_genes = cluster_expr.shape[1]
    n_cells = cell_expr.shape[0]
    n_lr_pairs = len(dataset.lr_id_to_pair)
    
    model_kwargs = dict(
        n_genes=len(available_activated_genes),  # 使用激活基因数量
        mlp_latent_dim=args.mlp_latent_dim,
        mlp_hidden_dims=[int(x) for x in args.mlp_hidden_dims.split(',')],
        gat_hidden_dims=gat_hidden_dims,
        gat_heads=args.gat_heads,
        gat_dropout=args.gat_dropout,
        output_dim=args.output_dim,
        n_celltypes=n_cells,
        n_lr_pairs=n_lr_pairs,
        lr_id_emb_dim=args.lr_id_emb_dim,
    )
    if "ablation_no_lr_identity" in inspect.signature(HeteroSTModel.__init__).parameters:
        model_kwargs["ablation_no_lr_identity"] = getattr(args, "ablation_no_lr_identity", False)
    model = HeteroSTModel(**model_kwargs).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params:             {total_params:,} (trainable={trainable_params:,})")
    
    # 训练循环
    train_losses = []
    val_losses = []
    train_mask_losses = []
    val_mask_losses = []
    train_node_losses = []
    val_node_losses = []
    learning_rates = []
    has_self_supervised_objective = (
        (args.lambda_mask_recon > 0 and args.edge_mask_ratio > 0)
        or (args.lambda_node_recon > 0 and args.node_mask_ratio > 0)
    )
    effective_train_epochs = args.epochs if has_self_supervised_objective and args.epochs > 0 else 0
    
    # ========== 阶段4：训练循环 ==========
    print(f"\n{'='*60}\nStage 3.7: Train\n{'='*60}")

    # 使用外层tqdm跟踪epoch进度
    # 移除position参数避免Jupyter中重复显示，添加leave=True保持最终状态
    epoch_pbar = tqdm(range(effective_train_epochs), desc="Training", leave=True, dynamic_ncols=True)
    
    # 主训练早停参数
    best_val_metric = float('inf')
    patience_counter = 0
    early_stop_patience = args.early_stop_patience
    early_stop_min_delta = args.early_stop_min_delta
    
    if effective_train_epochs == 0:
        print("Training:           skipped (no self-supervised masking objective enabled)")
    elif early_stop_patience > 0:
        print(f"Early stop:         patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    else:
        print("Early stop:         Disabled")
    
    # 评估 dataloader（对整个数据集进行评估）
    eval_num_workers = 0
    if num_workers > 0:
        print("Eval workers:       0 (forced to avoid too many open files)")
    eval_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        collate_fn=hetero_subgraph_collate_fn
    )

    for epoch in epoch_pbar:
        # 固定验证mask种子，保证每个epoch验证使用相同的mask模式
        val_mask_gen = torch.Generator(device=device).manual_seed(args.mask_seed)
        # ========== 训练阶段 ==========
        model.train()
        total_train_loss = 0.0
        total_train_mask = 0.0
        total_train_node = 0.0
        processed_train_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader, 1):
            # 丢弃被collate过滤掉的空batch
            if batch is None or batch.get('batch_size', 0) == 0:
                continue
            batch_size = batch['batch_size']
            n_spots_sub_list = batch['n_spots_sub']  # ✅ 现在是列表
            n_cells_list = batch['n_cells']  # ✅ 现在是列表

            # === 打印图结构信息 (仅前1个batch) ===
            if batch_idx < 1 and epoch == 0:
                pass  # 移除调试信息

            if isinstance(batch['expr_raw'], list):
                # 兼容旧的 list-of-subgraphs 批次格式
                batch_loss = 0.0
                batch_mask = 0.0
                batch_node = 0.0

                for b in range(batch_size):
                    expr_raw = batch['expr_raw'][b].to(device, non_blocking=pin_memory)
                    cell_expr_raw = batch['cell_expr_raw'][b].to(device, non_blocking=pin_memory)
                    edge_index_like = batch['edge_index_like'][b].to(device, non_blocking=pin_memory)
                    edge_attr_like = batch['edge_attr_like'][b].to(device, non_blocking=pin_memory)
                    edge_index_cc = batch['edge_index_cc'][b].to(device, non_blocking=pin_memory)
                    edge_attr_cc = batch['edge_attr_cc'][b]
                    if edge_attr_cc.dim() == 1:
                        edge_attr_cc = edge_attr_cc.view(-1, 2) if edge_attr_cc.numel() > 0 else edge_attr_cc.new_zeros((0, 2))
                    edge_attr_cc = edge_attr_cc.to(device, non_blocking=pin_memory)
                    edge_attr_cc_input = edge_attr_cc[:, :2]

                    _, _, _, _, _, predicted_masked_edges, edge_mask, node_recon_pred, node_mask = model(
                        expr_raw=expr_raw,
                        cell_expr_raw=cell_expr_raw,
                        edge_index_like=edge_index_like,
                        edge_attr_like=edge_attr_like,
                        edge_index_cc=edge_index_cc,
                        edge_attr_cc=edge_attr_cc_input,
                        return_attention=True,
                        edge_mask_ratio=args.edge_mask_ratio,
                        node_mask_ratio=args.node_mask_ratio,
                        mask_generator=None
                    )

                    mask_recon_loss = edge_attr_cc.new_tensor(0.0)
                    if edge_mask is not None and edge_mask.any() and predicted_masked_edges is not None:
                        target_scores = edge_attr_cc[:, 0][edge_mask]
                        if target_scores.numel() > 0:
                            mask_recon_loss = torch.nn.functional.mse_loss(predicted_masked_edges, target_scores)

                    node_recon_loss = edge_attr_cc.new_tensor(0.0)
                    if node_recon_pred is not None and node_mask is not None and node_mask.any():
                        node_target = torch.cat([expr_raw, cell_expr_raw], dim=0)
                        node_recon_loss = torch.nn.functional.mse_loss(
                            node_recon_pred[node_mask], node_target[node_mask]
                        )

                    total_loss = (
                        args.lambda_mask_recon * mask_recon_loss
                        + args.lambda_node_recon * node_recon_loss
                    )

                    batch_loss += total_loss
                    batch_mask += mask_recon_loss
                    batch_node += node_recon_loss

                avg_batch_loss = batch_loss / batch_size
                avg_batch_mask = batch_mask / batch_size
                avg_batch_node = batch_node / batch_size
            else:
                # 真正 batch 化（disjoint-union 大图），只前向/反向一次
                expr_raw = batch['expr_raw'].to(device, non_blocking=pin_memory)
                cell_expr_raw = batch['cell_expr_raw'].to(device, non_blocking=pin_memory)
                edge_index_like = batch['edge_index_like'].to(device, non_blocking=pin_memory)
                edge_attr_like = batch['edge_attr_like'].to(device, non_blocking=pin_memory)
                edge_index_cc = batch['edge_index_cc'].to(device, non_blocking=pin_memory)
                edge_attr_cc = batch['edge_attr_cc']
                if edge_attr_cc.dim() == 1:
                    edge_attr_cc = edge_attr_cc.view(-1, 2) if edge_attr_cc.numel() > 0 else edge_attr_cc.new_zeros((0, 2))
                edge_attr_cc = edge_attr_cc.to(device, non_blocking=pin_memory)
                edge_attr_cc_input = edge_attr_cc[:, :2]

                _, _, _, _, _, predicted_masked_edges, edge_mask, node_recon_pred, node_mask = model(
                    expr_raw=expr_raw,
                    cell_expr_raw=cell_expr_raw,
                    edge_index_like=edge_index_like,
                    edge_attr_like=edge_attr_like,
                    edge_index_cc=edge_index_cc,
                    edge_attr_cc=edge_attr_cc_input,
                    return_attention=True,
                    edge_mask_ratio=args.edge_mask_ratio,
                    node_mask_ratio=args.node_mask_ratio,
                    mask_generator=None
                )

                mask_recon_loss = edge_attr_cc.new_tensor(0.0)
                if edge_mask is not None and edge_mask.any() and predicted_masked_edges is not None:
                    target_scores = edge_attr_cc[:, 0][edge_mask]
                    if target_scores.numel() > 0:
                        mask_recon_loss = torch.nn.functional.mse_loss(predicted_masked_edges, target_scores)

                node_recon_loss = edge_attr_cc.new_tensor(0.0)
                if node_recon_pred is not None and node_mask is not None and node_mask.any():
                    node_target = torch.cat([expr_raw, cell_expr_raw], dim=0)
                    node_recon_loss = torch.nn.functional.mse_loss(
                        node_recon_pred[node_mask], node_target[node_mask]
                    )

                avg_batch_mask = mask_recon_loss
                avg_batch_node = node_recon_loss
                avg_batch_loss = (
                    args.lambda_mask_recon * mask_recon_loss
                    + args.lambda_node_recon * node_recon_loss
                )

            # 反向传播
            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += _scalar(avg_batch_loss)
            total_train_mask += _scalar(avg_batch_mask)
            total_train_node += _scalar(avg_batch_node)
            processed_train_batches += 1
        
        # ========== 验证阶段 ==========
        model.eval()
        total_val_loss = 0.0
        total_val_mask = 0.0
        total_val_node = 0.0
        processed_val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader, 1):
                if batch is None or batch.get('batch_size', 0) == 0:
                    continue
                batch_size = batch['batch_size']
                if isinstance(batch['expr_raw'], list):
                    batch_loss = 0.0
                    batch_mask = 0.0
                    batch_node = 0.0

                    for b in range(batch_size):
                        expr_raw = batch['expr_raw'][b].to(device, non_blocking=pin_memory)
                        cell_expr_raw = batch['cell_expr_raw'][b].to(device, non_blocking=pin_memory)
                        edge_index_like = batch['edge_index_like'][b].to(device, non_blocking=pin_memory)
                        edge_attr_like = batch['edge_attr_like'][b].to(device, non_blocking=pin_memory)
                        edge_index_cc = batch['edge_index_cc'][b].to(device, non_blocking=pin_memory)
                        edge_attr_cc = batch['edge_attr_cc'][b]
                        if edge_attr_cc.dim() == 1:
                            edge_attr_cc = edge_attr_cc.view(-1, 2) if edge_attr_cc.numel() > 0 else edge_attr_cc.new_zeros((0, 2))
                        edge_attr_cc = edge_attr_cc.to(device, non_blocking=pin_memory)
                        edge_attr_cc_input = edge_attr_cc[:, :2]

                        _, _, _, _, _, predicted_masked_edges, edge_mask, node_recon_pred, node_mask = model(
                            expr_raw=expr_raw,
                            cell_expr_raw=cell_expr_raw,
                            edge_index_like=edge_index_like,
                            edge_attr_like=edge_attr_like,
                            edge_index_cc=edge_index_cc,
                            edge_attr_cc=edge_attr_cc_input,
                            return_attention=True,
                            edge_mask_ratio=args.edge_mask_ratio,
                            node_mask_ratio=args.node_mask_ratio,
                            mask_generator=val_mask_gen
                        )

                        mask_recon_loss = edge_attr_cc.new_tensor(0.0)
                        if edge_mask is not None and edge_mask.any() and predicted_masked_edges is not None:
                            target_scores = edge_attr_cc[:, 0][edge_mask]
                            if target_scores.numel() > 0:
                                mask_recon_loss = torch.nn.functional.mse_loss(predicted_masked_edges, target_scores)

                        node_recon_loss = edge_attr_cc.new_tensor(0.0)
                        if node_recon_pred is not None and node_mask is not None and node_mask.any():
                            node_target = torch.cat([expr_raw, cell_expr_raw], dim=0)
                            node_recon_loss = torch.nn.functional.mse_loss(
                                node_recon_pred[node_mask], node_target[node_mask]
                            )

                        val_loss = (
                            args.lambda_mask_recon * mask_recon_loss
                            + args.lambda_node_recon * node_recon_loss
                        )

                        batch_loss += val_loss
                        batch_mask += mask_recon_loss
                        batch_node += node_recon_loss

                    avg_batch_loss = batch_loss / batch_size
                    avg_batch_mask = batch_mask / batch_size
                    avg_batch_node = batch_node / batch_size
                else:
                    expr_raw = batch['expr_raw'].to(device, non_blocking=pin_memory)
                    cell_expr_raw = batch['cell_expr_raw'].to(device, non_blocking=pin_memory)
                    edge_index_like = batch['edge_index_like'].to(device, non_blocking=pin_memory)
                    edge_attr_like = batch['edge_attr_like'].to(device, non_blocking=pin_memory)
                    edge_index_cc = batch['edge_index_cc'].to(device, non_blocking=pin_memory)
                    edge_attr_cc = batch['edge_attr_cc']
                    if edge_attr_cc.dim() == 1:
                        edge_attr_cc = edge_attr_cc.view(-1, 2) if edge_attr_cc.numel() > 0 else edge_attr_cc.new_zeros((0, 2))
                    edge_attr_cc = edge_attr_cc.to(device, non_blocking=pin_memory)
                    edge_attr_cc_input = edge_attr_cc[:, :2]

                    _, _, _, _, _, predicted_masked_edges, edge_mask, node_recon_pred, node_mask = model(
                        expr_raw=expr_raw,
                        cell_expr_raw=cell_expr_raw,
                        edge_index_like=edge_index_like,
                        edge_attr_like=edge_attr_like,
                        edge_index_cc=edge_index_cc,
                        edge_attr_cc=edge_attr_cc_input,
                        return_attention=True,
                        edge_mask_ratio=args.edge_mask_ratio,
                        node_mask_ratio=args.node_mask_ratio,
                        mask_generator=val_mask_gen
                    )

                    avg_batch_mask = edge_attr_cc.new_tensor(0.0)
                    if edge_mask is not None and edge_mask.any() and predicted_masked_edges is not None:
                        target_scores = edge_attr_cc[:, 0][edge_mask]
                        if target_scores.numel() > 0:
                            avg_batch_mask = torch.nn.functional.mse_loss(predicted_masked_edges, target_scores)

                    avg_batch_node = edge_attr_cc.new_tensor(0.0)
                    if node_recon_pred is not None and node_mask is not None and node_mask.any():
                        node_target = torch.cat([expr_raw, cell_expr_raw], dim=0)
                        avg_batch_node = torch.nn.functional.mse_loss(
                            node_recon_pred[node_mask], node_target[node_mask]
                        )

                    avg_batch_loss = (
                        args.lambda_mask_recon * avg_batch_mask
                        + args.lambda_node_recon * avg_batch_node
                    )

                total_val_loss += _scalar(avg_batch_loss)
                total_val_mask += _scalar(avg_batch_mask)
                total_val_node += _scalar(avg_batch_node)
                processed_val_batches += 1
        
        # 计算epoch平均损失
        avg_train_loss = total_train_loss / processed_train_batches if processed_train_batches > 0 else 0
        avg_val_loss = total_val_loss / processed_val_batches if processed_val_batches > 0 else 0
        avg_train_mask = total_train_mask / processed_train_batches if processed_train_batches > 0 else 0
        avg_val_mask = total_val_mask / processed_val_batches if processed_val_batches > 0 else 0
        avg_train_node = total_train_node / processed_train_batches if processed_train_batches > 0 else 0
        avg_val_node = total_val_node / processed_val_batches if processed_val_batches > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_mask_losses.append(avg_train_mask)
        val_mask_losses.append(avg_val_mask)
        train_node_losses.append(avg_train_node)
        val_node_losses.append(avg_val_node)
        learning_rates.append(float(optimizer.param_groups[0]['lr']))
        
        # 每个epoch结束时更新学习率调度器
        scheduler.step()
        
        # ========== 早停检查 ==========
        if early_stop_patience > 0 and processed_val_batches > 0:
            # 监控指标：验证总loss
            val_metric = avg_val_loss
            if val_metric < best_val_metric - early_stop_min_delta:
                best_val_metric = val_metric
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(
                    f"Early stop triggered: val loss not improved for {early_stop_patience} epochs "
                    f"(best={best_val_metric:.6f}, current={val_metric:.6f})"
                )
                break
        
        # 更新epoch进度条（只在epoch结束时更新一次）
        epoch_pbar.set_postfix({
            'Train': f'{avg_train_loss:.4f}',
            'Val': f'{avg_val_loss:.4f}',
            'Train_Mask': f'{avg_train_mask:.4f}',
            'Train_Node': f'{avg_train_node:.4f}',
        })
    training_history_path = os.path.join(args.output_dir, "training_history.csv")
    training_history_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_mask_loss": train_mask_losses,
            "val_mask_loss": val_mask_losses,
            "train_node_loss": train_node_losses,
            "val_node_loss": val_node_losses,
            "learning_rate": learning_rates,
        }
    )
    training_history_df.to_csv(training_history_path, index=False)
    print(f"Training log saved: {training_history_path}")
    # 在训练结束后，用训练好的模型对完整数据集进行一次评估，收集注意力得分
    print(f"\n{'='*60}\nStage 3.8: Evaluate and Save\n{'='*60}")
    print(f"Evaluating:          spots={len(dataset)}, batches={len(eval_dataloader)}")
    
    model.eval()
    all_cc_attention_scores = []
    all_edge_index_cc = []
    all_edge_attr_cc = []
    all_spot_indices = []
    all_cell_node_mappings = []
    all_batch_indices = []
    all_n_spots_sub = []
    all_src_barcodes = []  # 新增：发送方barcode
    all_dst_barcodes = []  # 新增：接收方barcode
    
    # ✅ 定义细胞类型名称列表
    all_cell_names = cell_names
    
    processed_eval_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", leave=True)):
            if batch is None or batch.get('batch_size', 0) == 0:
                continue
            batch_size = batch['batch_size']
            
            for b in range(batch_size):
                expr_raw = batch['expr_raw'][b].to(device, non_blocking=pin_memory)
                cell_expr_raw = batch['cell_expr_raw'][b].to(device, non_blocking=pin_memory)
                edge_index_like = batch['edge_index_like'][b].to(device, non_blocking=pin_memory)
                edge_attr_like = batch['edge_attr_like'][b].to(device, non_blocking=pin_memory)
                edge_index_cc = batch['edge_index_cc'][b].to(device, non_blocking=pin_memory)
                edge_attr_cc = batch['edge_attr_cc'][b]
                if edge_attr_cc.dim() == 1:
                    edge_attr_cc = edge_attr_cc.view(-1, 2) if edge_attr_cc.numel() > 0 else edge_attr_cc.new_zeros((0, 2))
                edge_attr_cc = edge_attr_cc.to(device, non_blocking=pin_memory)
                
                # 模型前向传播
                # ✅ 推理阶段也传入真实通信特征，收集的注意力才与LR信息一致
                edge_attr_cc_input = edge_attr_cc[:, :2]
                spot_repr, cell_repr, combined, spot_proj, cc_attention, _, _, node_recon_pred, node_mask = model(
                    expr_raw=expr_raw,
                    cell_expr_raw=cell_expr_raw,
                    edge_index_like=edge_index_like,
                    edge_attr_like=edge_attr_like,
                    edge_index_cc=edge_index_cc,
                    edge_attr_cc=edge_attr_cc_input,
                    return_attention=True,
                    edge_mask_ratio=0.0,
                    node_mask_ratio=0.0
                )
                
                if cc_attention is not None:
                    all_cc_attention_scores.append(cc_attention.detach().cpu())
                    all_edge_index_cc.append(edge_index_cc.detach().cpu())

                    edge_attr_extended = edge_attr_cc.detach().cpu() 
                    all_edge_attr_cc.append(edge_attr_extended)
                    
                    center_spot_idx = batch['center_spot_idx'][b]
                    spot_indices = torch.full((edge_index_cc.size(1),), center_spot_idx, dtype=torch.long)
                    all_spot_indices.append(spot_indices)
                    
                    spot_cell_mapping = batch['spot_cell_mapping'][b]
                    cell_node_to_cell_type = {}
                    for (spot_local_idx, cell_type_id), cell_node_local_idx in spot_cell_mapping.items():
                        cell_node_to_cell_type[cell_node_local_idx] = cell_type_id
                    all_cell_node_mappings.append(cell_node_to_cell_type)
                    
                    batch_indices = torch.full((edge_index_cc.size(1),), batch_idx, dtype=torch.long)
                    all_batch_indices.append(batch_indices)
                    
                    n_spots_sub = batch['n_spots_sub'][b]
                    n_spots_sub_tensor = torch.full((edge_index_cc.size(1),), n_spots_sub, dtype=torch.long)
                    all_n_spots_sub.append(n_spots_sub_tensor)
                    
                    # 新增：收集发送方和接收方barcode
                    subgraph_spot_indices = batch['spot_indices'][b]  # 子图中所有spot的全局索引
                    cell_node_to_spot_cell = {cell_node: (spot_local, cell_type) for (spot_local, cell_type), cell_node in spot_cell_mapping.items()}
                    
                    src_barcodes = []
                    dst_barcodes = []
                    
                    for e in range(edge_index_cc.size(1)):
                        src_global = edge_index_cc[0, e].item()
                        dst_global = edge_index_cc[1, e].item()
                        
                        # 转换全局节点索引到局部细胞节点索引
                        src_local = src_global - n_spots_sub
                        dst_local = dst_global - n_spots_sub
                        
                        src_spot_local, _ = cell_node_to_spot_cell[src_local]
                        dst_spot_local, _ = cell_node_to_spot_cell[dst_local]
                        
                        src_barcode = spot_names[subgraph_spot_indices[src_spot_local]]
                        dst_barcode = spot_names[subgraph_spot_indices[dst_spot_local]]
                        
                        src_barcodes.append(src_barcode)
                        dst_barcodes.append(dst_barcode)
                    
                    all_src_barcodes.append(src_barcodes)  # 列表 of strings
                    all_dst_barcodes.append(dst_barcodes)  # 列表 of strings
            processed_eval_batches += 1
    
    print(f"Eval batches used:   {processed_eval_batches}")

    total_edges_collected = sum(scores.shape[0] for scores in all_cc_attention_scores)
    if len(all_cc_attention_scores) > 0:
        print(f"Edges collected:     {total_edges_collected} (avg/spot={total_edges_collected/len(all_cc_attention_scores):.1f})")
    else:
        print("Edges collected:     0")
    
    # 计算degree-scaled attention（反归一化注意力得分）
    print("Post-process:        degree-scaled attention")
    for i in range(len(all_cc_attention_scores)):
        edge_index = all_edge_index_cc[i]
        attention = all_cc_attention_scores[i]
        targets = edge_index[1]  # 目标节点索引
        
        # 计算每个目标节点的入度
        unique_targets, counts = torch.unique(targets, return_counts=True)
        degree_dict = {target.item(): count.item() for target, count in zip(unique_targets, counts)}
        
        # 计算scaled attention: attention_score * degree
        scaled_attention = torch.zeros_like(attention)
        for e in range(attention.shape[0]):
            target = targets[e].item()
            deg = degree_dict.get(target, 0)
            scaled_attention[e] = attention[e] * deg
        
        all_cc_attention_scores[i] = scaled_attention
    
    
    evaluate_cell_communication(
        all_cc_attention_scores=all_cc_attention_scores,
        all_edge_index_cc=all_edge_index_cc,
        all_edge_attr_cc=all_edge_attr_cc,
        all_spot_indices=all_spot_indices,
        all_n_spots_sub=all_n_spots_sub,
        all_cell_names=all_cell_names,
        all_cell_node_mappings=all_cell_node_mappings,
        output_dir=args.output_dir,
        n_spots=n_spots,
        n_cells=n_cells,
        spot_names=spot_names,
        all_src_barcodes=all_src_barcodes,
        all_dst_barcodes=all_dst_barcodes,
        export_unified=getattr(args, 'export_unified_csv', False),
        export_filtered=getattr(args, 'export_filtered_csv', True),
        attention_threshold=args.attention_threshold,
        lr_support_by_edge=graph_data.get("lr_support_by_edge"),
    )
    
    # 绘制损失曲线
    plot_training_loss(train_losses, val_losses, args.output_dir, args.epochs)

    print(f"\n{'='*60}\nDone\n{'='*60}")

if __name__ == '__main__':
    main()
