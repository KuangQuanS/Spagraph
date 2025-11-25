import argparse
import logging
import os
import torch
import glob
import pandas as pd
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# VAE已移除：使用MLPEncoder作为编码器，不再引用DualDecoderVAE
# 已经将DGI集成到HeteroSTModel (compute_dgi_loss)，不再需要独立的 DGIPretrainModel
from hetero_model import HeteroSTModel
from hetero_graph_builder import STHeteroSubgraphDataset, hetero_subgraph_collate_fn, setup_logging, set_seed
from evaluate import evaluate_cell_communication, plot_training_loss
from calculate_lr_scores import calculate_lr_scores

def _scalar(x):
    """Convert tensor/number to Python float for logging/accumulation."""
    if isinstance(x, (float, int)):
        return float(x)
    try:
        return x.item()
    except Exception:
        return float(x)

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
                       help='激活基因选择和LR通讯的平均表达阈值 (normalize_total 1e4后，default: 5.0)')
    parser.add_argument('--min_comm_edges', type=int, default=1, 
                       help='最小通讯边数阈值，少于此值的spot将被过滤 (default: 1)')
    parser.add_argument('--spot_cell_expr_csv', type=str, default=None,
                       help='预计算的spot-cell全基因表达CSV文件路径，如果提供则跳过构建步骤')

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
    parser.add_argument('--edge_mask_ratio', type=float, default=0.2, help='mask通讯边的比例 (默认20%%)')
    parser.add_argument('--node_mask_ratio', type=float, default=0.15, help='mask节点特征比例 (默认15%%)')
    parser.add_argument('--mask_seed', type=int, default=1234, help='验证阶段mask的固定随机种子')
    parser.add_argument('--lr_id_emb_dim', type=int, default=8, help='LR id 嵌入维度，用于通讯边特征')
    

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小 (已支持真正的批处理)')
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
def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, 'training.log'))
    logging.info("="*80)
    logging.info("="*80)
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')
    
    # ========== 阶段1：加载数据和构建激活基因特征 ==========
    logging.info("="*80)
    logging.info("阶段1: 加载数据和构建激活基因特征")
    logging.info("="*80)
    
    # ✅ 使用 deconv_dir 构建所有路径
    deconv_dir = args.deconv_dir
    vae_weight_path = os.path.join(deconv_dir, 'final_vae.pth')
    vae_npz_path = os.path.join(deconv_dir, 'final_vae_cluster_data.npz')
    
    # 1. 尝试从 stage1 的 npz 文件加载 cluster 信息（用于细胞类型映射）
    cluster_to_celltype = {}
    
    if os.path.exists(vae_npz_path):
        cluster_data = np.load(vae_npz_path, allow_pickle=True)
        
        cluster_ids = cluster_data['cluster_ids']
        celltype_mapping_array = cluster_data['cluster_to_celltype']
        cluster_to_celltype = {str(row['cluster_id']): str(row['celltype']) 
                                for row in celltype_mapping_array}
        
        # 加载全基因表达用于构建spot-cell表达
        expressions_full_array = cluster_data['cluster_expressions_full']  # all genes
        checkpoint_temp = torch.load(vae_weight_path, map_location='cpu', weights_only=False)
        all_genes = checkpoint_temp.get('all_genes', None)
        
        if all_genes is None:
            raise ValueError("VAE checkpoint 中找不到 all_genes 列表")
        
        cluster_index_names = [f"Cluster_{cid}" for cid in cluster_ids]
        cluster_full_expr = pd.DataFrame(
            expressions_full_array,
            index=cluster_index_names,
            columns=all_genes
        )
        logging.info(f"已从 NPZ 加载 cluster 全基因表达: {cluster_full_expr.shape}")
        
    else:
        # NPZ 文件不存在，无法继续
        raise FileNotFoundError(f"找不到 Stage 1 NPZ 文件: {vae_npz_path}\n")

    # 2. 构建激活基因特征（用于MLP输入）
    logging.info("构建激活基因特征...")
    # 激活基因将在基于重构表达数据后计算
    
    # 4. CellChat数据库（用于LR通讯计算）
    logging.info("加载CellChat LR数据库用于通讯计算...")
    cellchat_file = 'cellchat_human.csv'
    lr_db = pd.read_csv(cellchat_file)
    lr_pairs = []
    for _, row in lr_db.iterrows():
        lig = str(row['ligand']).strip()
        rec = str(row['receptor']).strip()
        lr_pairs.append((lig, rec))
    logging.info(f"已加载LR对: {len(lr_pairs)}个")

    # 5. 加载ST数据
    adata = sc.read_h5ad(args.st_h5ad)
    logging.info(f"已加载ST数据: {adata.shape}")
    
    # 6. 加载spot坐标
    spot_coords = adata.obsm['spatial'] if 'spatial' in adata.obsm else None
    if spot_coords is not None:
        logging.info(f"已加载spot坐标: {spot_coords.shape}")
        # 全局记录spot数，后续统计使用
        n_spots = len(spot_coords)
        logging.info(f"spot数量: {n_spots}")
    else:
        raise ValueError("找不到spot坐标")

    # 7. 加载spot-cluster反卷积比例矩阵
    cluster_composition_file = os.path.join(deconv_dir, '*_cluster_composition.csv')
    cluster_composition_files = glob.glob(cluster_composition_file)
    cluster_composition = pd.read_csv(cluster_composition_files[0], index_col=0)

    # 记录cluster数量
    n_clusters = cluster_composition.shape[1]
    logging.info(f"cluster数量: {n_clusters}")
    
    # ========== 阶段1：加载VAE编码器（仅用于获取细胞类型映射） ==========
    logging.info("="*80)
    logging.info("阶段1: 加载 deconv checkpoint（仅用于获取基因列表与cluster映射）")
    logging.info("="*80)
    
    # 仅加载VAE checkpoint以获取基因列表和细胞类型映射
    if os.path.exists(vae_weight_path):
        checkpoint = torch.load(vae_weight_path, map_location=device, weights_only=False)
        all_genes = checkpoint.get('all_genes', None)
        if all_genes is None:
            raise ValueError("VAE checkpoint 中找不到 all_genes 列表")
    else:
        raise FileNotFoundError(f"找不到VAE权重文件: {vae_weight_path}")
  
    # ========== 阶段2：构建spot-cell全基因表达 ==========
    logging.info("="*80)
    logging.info("阶段2: 构建spot-cell全基因表达")
    logging.info("="*80)
    
    spot_names = adata.obs_names.tolist()
    spot_total_counts = np.array(adata.X.sum(axis=1)).flatten()
    logging.info(f"Spot总数: {len(spot_names)}, 平均counts={spot_total_counts.mean():.1f}")
    
    # 统一cluster名称格式（在if-else之前定义，因为后面需要）
    cluster_names = [f"Cluster_{c}" if not c.startswith('Cluster_') else c 
                     for c in cluster_composition.columns]
    
    # 检查是否提供了预计算的CSV文件
    if args.spot_cell_expr_csv is not None and os.path.exists(args.spot_cell_expr_csv):
        logging.info(f"检测到预计算的spot-cell表达文件: {args.spot_cell_expr_csv}")
        logging.info("跳过构建步骤，直接加载...")
        
        # 直接加载预计算的CSV文件
        spot_cell_expr_df = pd.read_csv(args.spot_cell_expr_csv, index_col=0)
        logging.info(f"已加载预计算的spot-cell全基因表达: {spot_cell_expr_df.shape}")
        
        # 验证数据格式
        if spot_cell_expr_df.index.name != 'spot_cell':
            logging.warning("警告: CSV文件index名称不是'spot_cell'，可能存在格式问题")
        
    else:
        # 正常构建流程
        # 构建每个spot-cell的全基因表达
        spot_cell_full_expr = {}
        for spot_idx, spot_name in enumerate(spot_names):
            spot_total_count = spot_total_counts[spot_idx]
            
            for cell_name, celltype in cluster_to_celltype.items():
                cluster_key = f"Cluster_{cell_name}"
                if cluster_key not in cluster_full_expr.index:
                    continue
                
                # 获取cluster在该spot的权重
                if cluster_key in cluster_names:
                    cluster_idx = cluster_names.index(cluster_key)
                    cluster_weight = cluster_composition.iloc[spot_idx, cluster_idx]
                else:
                    cluster_weight = 0.0
                
                if cluster_weight < 1e-6:
                    continue
                
                # 计算: (cluster_full_expr / 1e4) × cluster_weight × spot_total_count
                cluster_expr_normalized = cluster_full_expr.loc[cluster_key].values / 1e4
                spot_cell_expr = cluster_expr_normalized * cluster_weight * spot_total_count
                
                # 累加到该spot的celltype表达
                key = f"{spot_name}_{celltype}"
                if key in spot_cell_full_expr:
                    spot_cell_full_expr[key] += spot_cell_expr
                else:
                    spot_cell_full_expr[key] = spot_cell_expr.copy()
        
        # 转为DataFrame
        spot_cell_expr_df = pd.DataFrame.from_dict(
            spot_cell_full_expr, orient='index', columns=cluster_full_expr.columns
        )
        spot_cell_expr_df.index.name = 'spot_cell'
        
        # 删除全为0的spot-cell（直接过滤，不需要额外的稀有细胞类型过滤）
        row_sums = spot_cell_expr_df.sum(axis=1)
        spot_cell_expr_df = spot_cell_expr_df[row_sums > 0]

        csv_path = os.path.join(args.output_dir, 'spot_cell_full_expr.csv')
        spot_cell_expr_df.to_csv(csv_path)
        logging.info(f"已保存spot-cell全基因表达: {csv_path}, 形状={spot_cell_expr_df.shape}")
    
    # ========== 基于重构表达数据计算激活基因 ==========
    logging.info("基于重构表达数据计算激活基因...")
    
    # 使用阈值方法选择激活基因：每个cell中表达值 > threshold 的基因
    mean_expr_threshold = args.mean_expr_threshold
    logging.info(f"激活基因选择阈值: normalize_total(1e4) > {mean_expr_threshold}")
    
    activated_genes_set = set()
    for spot_cell_name in spot_cell_expr_df.index:
        expr = spot_cell_expr_df.loc[spot_cell_name].values
        
        # 对单个cell进行normalize_total
        total_count = expr.sum()
        if total_count > 0:
            expr_normalized = expr / total_count * 1e4  # [n_genes]
        else:
            expr_normalized = expr  # 表达全为0的情况
        
        # 筛选表达值 > threshold 的基因
        active_mask = expr_normalized > mean_expr_threshold
        active_gene_names = spot_cell_expr_df.columns[active_mask].tolist()
        activated_genes_set.update(active_gene_names)
    
    activated_genes = sorted(list(activated_genes_set))
    logging.info(f"激活基因数量: {len(activated_genes)} (基于重构表达数据从{len(spot_cell_expr_df)}个cell中选择)")
    
    # 检查激活基因是否在cluster表达中可用
    available_activated_genes = [g for g in activated_genes if g in cluster_full_expr.columns]
    logging.info(f"激活基因在cluster表达中的可用性: {len(available_activated_genes)}/{len(activated_genes)}")
    
    # 为MLP创建输入特征：使用激活基因的cluster表达（无需log1p，这里保持原值）
    activated_cluster_expr = cluster_full_expr[available_activated_genes].copy()

    # 为Dataset接口创建占位符（实际不会使用）
    cluster_expr = pd.DataFrame(
        np.zeros((len(cluster_to_celltype), len(available_activated_genes)), dtype=np.float32),
        index=[f"Cluster_{cid}" for cid in cluster_to_celltype.keys()],
        columns=available_activated_genes
    )

    cluster_to_cell = {}
    for cluster_id, celltype_name in cluster_to_celltype.items():
        cluster_name = f"Cluster_{cluster_id}"
        cluster_to_cell[cluster_name] = celltype_name
    logging.info(f"已构建cluster-cell映射: {len(cluster_to_cell)} 个cluster映射到 {len(set(cluster_to_cell.values()))} 个cell类型")
    
    # 提取实际存在的celltype（从spot_cell_expr_df的index中解析）
    cell_names = sorted(set([idx.split('_', 1)[1] for idx in spot_cell_expr_df.index]))
    logging.info(f"实际存在的细胞类型数: {len(cell_names)}")
    # 使用spot-cell全表达对每个cell类型做均值，避免全部为0的占位符
    cell_expr_mean = spot_cell_expr_df.copy()
    cell_expr_mean['celltype'] = [idx.split('_', 1)[1] for idx in spot_cell_expr_df.index]
    cell_expr = cell_expr_mean.groupby('celltype')[available_activated_genes].mean().reindex(cell_names).fillna(0.0)
    
    cell_full_expr = cell_expr_mean.groupby('celltype')[cluster_full_expr.columns].mean().reindex(cell_names).fillna(0.0)

    composition = pd.DataFrame(0.0, index=cluster_composition.index, columns=cell_names)
    for spot_idx in range(len(cluster_composition)):
        for cluster_idx, cluster_name in enumerate(cluster_names):
            celltype = cluster_to_cell.get(cluster_name)
            if celltype and celltype in cell_names:  # 只处理实际存在的细胞类型
                composition.iloc[spot_idx, composition.columns.get_loc(celltype)] += cluster_composition.iloc[spot_idx, cluster_idx]
    
    logging.info(f"Composition矩阵形状: {composition.shape}")
    
    # ========== 阶段3.5：预计算KNN邻域和LR通讯得分 ==========
    logging.info("="*80)
    logging.info("阶段3.5: 预计算KNN邻域和LR通讯得分")
    logging.info("="*80)
    
    knn_mask, csv_path, graph_data = calculate_lr_scores(
        spot_coords=spot_coords,
        composition=composition,
        args=args,
        lr_pairs=lr_pairs,
        adata=adata,
        cell_full_expr=spot_cell_expr_df,  
        output_dir=args.output_dir,
        n_neighbors=args.n_spot_neighbors
    )
    
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
        device=device
    )
    
    # ========== 数据集划分：训练集和验证集 ==========
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    # 使用固定种子确保可重复性
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logging.info(f"数据集划分完成:")
    logging.info(f"   - 训练集: {train_size} 个spots")
    logging.info(f"   - 验证集: {val_size} 个spots")
    logging.info(f"   - 验证集比例: {args.val_split:.1%}")
    
    # ========== 保存LR对映射文件 ==========
    logging.info("保存LR对映射文件...")
    lr_mapping_path = os.path.join(args.output_dir, "lr_pair_mapping.txt")
    with open(lr_mapping_path, 'w') as f:
        f.write("lr_id\tligand\treceptor\n")  # 表头
        for lr_id, (ligand, receptor) in dataset.lr_id_to_pair.items():
            f.write(f"{lr_id}\t{ligand}\t{receptor}\n")
    logging.info(f"LR对映射文件已保存: {lr_mapping_path} (共{len(dataset.lr_id_to_pair)}个LR对)")
    
    # 创建训练和验证数据加载器
    if args.sample_rate < 1.0:
        num_samples = int(len(train_dataset) * args.sample_rate)
        sampler = RandomSampler(train_dataset, num_samples=num_samples, replacement=False)
        logging.info(f"⚡ 采样训练模式: 每个epoch采样 {args.sample_rate*100:.1f}% 训练数据 ({num_samples}/{len(train_dataset)} spots)")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=hetero_subgraph_collate_fn
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=hetero_subgraph_collate_fn
        )
    
    # 验证集数据加载器（不打乱顺序）
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=hetero_subgraph_collate_fn
    )
    
    logging.info("="*80)
    logging.info("阶段3: 构建HeteroGAT模型")
    logging.info("="*80)
    gat_hidden_dims = [int(x) for x in args.gat_hidden_dims.split(',')]
    
    n_genes = cluster_expr.shape[1]
    n_cells = cell_expr.shape[0]
    n_lr_pairs = len(dataset.lr_id_to_pair)
    
    model = HeteroSTModel(
        n_genes=len(available_activated_genes),  # 使用激活基因数量
        mlp_latent_dim=args.mlp_latent_dim,
        mlp_hidden_dims=[int(x) for x in args.mlp_hidden_dims.split(',')],
        gat_hidden_dims=gat_hidden_dims,
        gat_heads=args.gat_heads,
        gat_dropout=args.gat_dropout,
        output_dim=args.output_dim,
        n_celltypes=n_cells,
        n_lr_pairs=n_lr_pairs,
        lr_id_emb_dim=args.lr_id_emb_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    logging.info(f"模型构建完成")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"   - 总参数数: {total_params:,}")
    logging.info(f"   - 可训练参数数: {trainable_params:,}")
    
    # 当使用 DGI loss 作为主训练目标时，确认 encoder/edge attention 已准备好并共享
    logging.info("在主训练中将使用 DGI loss，空间边与通讯边保持独立注意力参数")
    
    # 训练循环
    train_losses = []
    val_losses = []
    train_mask_losses = []
    val_mask_losses = []
    train_node_losses = []
    val_node_losses = []
    
    # ========== 阶段4：训练循环 ==========
    logging.info("="*80)
    logging.info("阶段4: 开始训练（DGI作为主要训练目标）")
    logging.info("="*80)
    logging.info(f"训练配置:")
    logging.info(f"  - 边mask重构: ratio={args.edge_mask_ratio}, lambda={args.lambda_mask_recon}")
    logging.info(f"  - 节点mask重构: ratio={args.node_mask_ratio}, lambda={args.lambda_node_recon}")
    logging.info(f"  - 验证mask种子: {args.mask_seed}")
    logging.info("="*80)

    # 使用外层tqdm跟踪epoch进度
    # 移除position参数避免Jupyter中重复显示，添加leave=True保持最终状态
    epoch_pbar = tqdm(range(args.epochs), desc="Training", leave=True, dynamic_ncols=True)
    
    # 主训练早停参数
    best_val_metric = float('inf')
    patience_counter = 0
    early_stop_patience = args.early_stop_patience
    early_stop_min_delta = args.early_stop_min_delta
    
    if early_stop_patience > 0:
        logging.info(f"主训练早停已启用: patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    else:
        logging.info("主训练早停已禁用")
    
    # 评估 dataloader（对整个数据集进行评估）
    eval_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
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

            # 累积批次损失
            batch_loss = 0.0
            batch_mask = 0.0
            batch_node = 0.0

            # 处理每个subgraph
            for b in range(batch_size):
                # 提取第b个subgraph的数据
                expr_raw = batch['expr_raw'][b].to(device)  # [k+1, n_genes]
                cell_expr_raw = batch['cell_expr_raw'][b].to(device)  # [(k+1)*n_cells, n_marker_genes]

                edge_index_like = batch['edge_index_like'][b].to(device)  # [2, E_like]
                edge_attr_like = batch['edge_attr_like'][b].to(device)    # [E_like]
                edge_index_cc = batch['edge_index_cc'][b].to(device)      # [2, E_cc]
                edge_attr_cc = batch['edge_attr_cc'][b]
                # 可能为空边，保证形状为 [E_cc, 2]
                if edge_attr_cc.dim() == 1:
                    edge_attr_cc = edge_attr_cc.view(-1, 2) if edge_attr_cc.numel() > 0 else edge_attr_cc.new_zeros((0, 2))
                edge_attr_cc = edge_attr_cc.to(device)        # [E_cc, 2] = [lr_score, lr_id]

                # ✅ 使用真实的通信特征 [lr_score, lr_id] 以学习/输出有意义的注意力
                edge_attr_cc_input = edge_attr_cc[:, :2]  # [E_cc, 2]

                # ========== 前向传播 ==========
                spot_repr, cell_repr, combined, spot_proj, cc_attention, predicted_masked_edges, edge_mask, node_recon_pred, node_mask = model(
                    expr_raw=expr_raw,
                    cell_expr_raw=cell_expr_raw,
                    edge_index_like=edge_index_like,
                    edge_attr_like=edge_attr_like,
                    edge_index_cc=edge_index_cc,
                    edge_attr_cc=edge_attr_cc_input,  # 不输入真实强度，避免信息泄露
                    return_attention=True,
                    edge_mask_ratio=args.edge_mask_ratio,
                    node_mask_ratio=args.node_mask_ratio,
                    mask_generator=None
                )

                # ========== DGI训练：使用对比学习损失 ==========
                mask_recon_loss = 0.0
                if edge_mask is not None and edge_mask.any() and predicted_masked_edges is not None:
                    target_scores = edge_attr_cc[:, 0][edge_mask]
                    if target_scores.numel() > 0:
                        mask_recon_loss = torch.nn.functional.mse_loss(predicted_masked_edges, target_scores)

                node_recon_loss = 0.0
                if node_recon_pred is not None and node_mask is not None and node_mask.any():
                    node_target = torch.cat([expr_raw, cell_expr_raw], dim=0)
                    node_recon_loss = torch.nn.functional.mse_loss(
                        node_recon_pred[node_mask], node_target[node_mask]
                    )

                total_loss = (
                    args.lambda_mask_recon * mask_recon_loss
                    + args.lambda_node_recon * node_recon_loss
                )

                # 累积批次损失
                loss = total_loss
                batch_loss += loss
                batch_mask += mask_recon_loss
                batch_node += node_recon_loss

            # 对整个batch求平均损失，然后反向传播
            avg_batch_loss = batch_loss / batch_size
            avg_batch_mask = batch_mask / batch_size
            avg_batch_node = batch_node / batch_size

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
                batch_loss = 0.0
                batch_mask = 0.0
                batch_node = 0.0

                for b in range(batch_size):
                    expr_raw = batch['expr_raw'][b].to(device)
                    cell_expr_raw = batch['cell_expr_raw'][b].to(device)
                    edge_index_like = batch['edge_index_like'][b].to(device)
                    edge_attr_like = batch['edge_attr_like'][b].to(device)
                    edge_index_cc = batch['edge_index_cc'][b].to(device)
                    edge_attr_cc = batch['edge_attr_cc'][b]
                    if edge_attr_cc.dim() == 1:
                        edge_attr_cc = edge_attr_cc.view(-1, 2) if edge_attr_cc.numel() > 0 else edge_attr_cc.new_zeros((0, 2))
                    edge_attr_cc = edge_attr_cc.to(device)  # [E_cc, 2] = [lr_score, lr_id]

                    # ✅ 使用真实通信特征，确保验证注意力与LR信息对应
                    edge_attr_cc_input = edge_attr_cc[:, :2]  # [E_cc, 2]

                    # 前向传播（验证阶段不进行边mask）
                    spot_repr, cell_repr, combined, spot_proj, cc_attention, predicted_masked_edges, edge_mask, node_recon_pred, node_mask = model(
                        expr_raw=expr_raw,
                        cell_expr_raw=cell_expr_raw,
                        edge_index_like=edge_index_like,
                        edge_attr_like=edge_attr_like,
                        edge_index_cc=edge_index_cc,
                        edge_attr_cc=edge_attr_cc_input,  # 不输入真实强度
                        return_attention=True,
                        edge_mask_ratio=args.edge_mask_ratio,  # 验证阶段与训练一致
                        node_mask_ratio=args.node_mask_ratio,
                        mask_generator=val_mask_gen
                    )

                    # 边mask重构（只对被mask的边）
                    mask_recon_loss = 0.0
                    if edge_mask is not None and edge_mask.any() and predicted_masked_edges is not None:
                        target_scores = edge_attr_cc[:, 0][edge_mask]
                        if target_scores.numel() > 0:
                            mask_recon_loss = torch.nn.functional.mse_loss(predicted_masked_edges, target_scores)

                    node_recon_loss = 0.0
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
                logging.info(f"早停触发: 验证loss在{early_stop_patience}个epoch内未改善 (best={best_val_metric:.6f}, current={val_metric:.6f})")
                break
        
        # 更新epoch进度条（只在epoch结束时更新一次）
        epoch_pbar.set_postfix({
            'Train': f'{avg_train_loss:.4f}',
            'Val': f'{avg_val_loss:.4f}',
            'Train_Mask': f'{avg_train_mask:.4f}',
            'Train_Node': f'{avg_train_node:.4f}',
        })
    logging.info(f"   - 评估数据集大小: {len(dataset)} spots")
    logging.info(f"   - 评估batch数量: {len(eval_dataloader)} batches（可能包含被过滤的空子图）")
    
    # 在训练结束后，用训练好的模型对完整数据集进行一次评估，收集注意力得分
    logging.info("使用训练好的模型对完整数据集进行评估，收集cell-cell注意力得分...")
    
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
                expr_raw = batch['expr_raw'][b].to(device)
                cell_expr_raw = batch['cell_expr_raw'][b].to(device)
                edge_index_like = batch['edge_index_like'][b].to(device)
                edge_attr_like = batch['edge_attr_like'][b].to(device)
                edge_index_cc = batch['edge_index_cc'][b].to(device)
                edge_attr_cc = batch['edge_attr_cc'][b]
                if edge_attr_cc.dim() == 1:
                    edge_attr_cc = edge_attr_cc.view(-1, 2) if edge_attr_cc.numel() > 0 else edge_attr_cc.new_zeros((0, 2))
                edge_attr_cc = edge_attr_cc.to(device)
                
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
                    spot_barcodes_full = batch['spot_barcodes'][b]  # 子图中所有spot的barcode
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
                        
                        src_barcode = spot_barcodes_full[src_spot_local]
                        dst_barcode = spot_barcodes_full[dst_spot_local]
                        
                        src_barcodes.append(src_barcode)
                        dst_barcodes.append(dst_barcode)
                    
                    all_src_barcodes.append(src_barcodes)  # 列表 of strings
                    all_dst_barcodes.append(dst_barcodes)  # 列表 of strings
            processed_eval_batches += 1
    
    logging.info(f"评估阶段实际处理的batch数: {processed_eval_batches}")
    
    logging.info(f"评估完成，收集到 {len(all_cc_attention_scores)} 个spots的注意力得分")
    total_edges_collected = sum(scores.shape[0] for scores in all_cc_attention_scores)
    logging.info(f"   - 子图内边数（模型实际使用）: {total_edges_collected} 条")
    if len(all_cc_attention_scores) > 0:
        logging.info(f"   - 平均每个spot子图: {total_edges_collected/len(all_cc_attention_scores):.1f} 条边")
    else:
        logging.info(f"   - 平均每个spot子图: N/A (无注意力得分)")
    logging.info(f"   - 注意：这是子图内的边数，不包含spot间的跨子图通讯")
    
    # 计算degree-scaled attention（反归一化注意力得分）
    logging.info("计算degree-scaled attention（注意力得分 × 目标节点入度）...")
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
    
    logging.info("degree-scaled attention计算完成")
    
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
        attention_threshold=args.attention_threshold
    )
    
    # 绘制损失曲线
    plot_training_loss(train_losses, val_losses, args.output_dir, args.epochs)
    
    logging.info("\n" + "="*80)
    logging.info("训练完成！")
    logging.info("="*80)

if __name__ == '__main__':
    main()
