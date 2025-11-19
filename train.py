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
from evaluate import evaluate_cell_communication, plot_training_loss, plot_dgi_loss
from calculate_lr_scores import calculate_lr_scores

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
    parser.add_argument('--n_spot_neighbors', type=int, default=10, help='Spot邻近数')
    
    # LR通讯参数
    parser.add_argument('--lr_distance_sigma', type=float, default=50.0, help='LR通讯距离衰减参数sigma')
    parser.add_argument('--mean_expr_threshold', type=float, default=5.0, 
                       help='激活基因选择和LR通讯的平均表达阈值 (normalize_total 1e4后，default: 5.0)')
    parser.add_argument('--min_comm_edges', type=int, default=1, 
                       help='最小通讯边数阈值，少于此值的spot将被过滤 (default: 1)')
    parser.add_argument('--spot_cell_expr_csv', type=str, default=None,
                       help='预计算的spot-cell全基因表达CSV文件路径，如果提供则跳过构建步骤')

    # GAT参数
    parser.add_argument('--gat_layers', type=int, default=6, help='GAT层数')
    parser.add_argument('--gat_hidden_dims', type=str, default='512,256,128', help='GAT隐层维度')
    parser.add_argument('--gat_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--gat_dropout', type=float, default=0.3, help='Dropout概率')
    parser.add_argument('--temperature', type=float, default=2.0, help='注意力温度系数，用于控制注意力分布的尖锐程度 (default: 1.0)')
    
    # 模型参数
    parser.add_argument('--output_dim', type=int, default=120, help='输出维度')
    
    # DGI 模块参数 (用于主训练的自监督目标)
    parser.add_argument('--freeze_encoder', type=str, default='false',
                       choices=['true', 'false'],
                       help='是否冻结编码器（true=只训练预测头，false=微调整个模型）')
    
    # 混合训练参数（监督 + 自监督）
    parser.add_argument('--supervised_weight', type=float, default=0.5,
                        help='监督学习权重alpha（0~1），自监督权重为1-alpha，默认0.5')
    parser.add_argument('--negative_sample_ratio', type=float, default=1.0,
                        help='负采样比例（相对于正边数量），默认1.0')
    
    # 双头边过滤参数
    parser.add_argument('--lambda_exist', type=float, default=0.5,
                        help='边存在性损失权重 (default: 0.5)')
    parser.add_argument('--lambda_rate', type=float, default=0.4,
                        help='边强度回归损失权重 (default: 0.4)')
    parser.add_argument('--edge_topk', type=int, default=5,
                        help='推理时每个源节点保留的最大边数 (default: 5)')
    parser.add_argument('--edge_exist_threshold', type=float, default=0.5,
                        help='边存在性概率阈值 (default: 0.5)')
    
    # DGI训练参数，用于在主训练阶段的DGI损失计算（不再用于独立预训练）
    parser.add_argument('--dgi_epochs', type=int, default=30, help='DGI训练的epoch数（仅作为监控使用）')
    parser.add_argument('--dgi_lr', type=float, default=1e-4, help='DGI子模块中可能使用的学习率（不作为主训练的单独优化器）')
    parser.add_argument('--corruption_mode', type=str, default='feature_mask', 
                       choices=['feature_mask', 'gaussian_noise', 'shuffle'],
                       help='DGI特征破坏模式')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='特征mask比例 (建议0.4-0.6，增强对比学习难度)')
    parser.add_argument('--noise_std', type=float, default=0.2, help='高斯噪声标准差 (建议0.1-0.3)')
    parser.add_argument('--edge_drop_rate', type=float, default=0.3, help='边丢弃比例 (建议0.2-0.4，增强图结构扰动)')
    parser.add_argument('--edge_drop_mode', type=str, default='edge_drop_random', choices=['edge_drop_random','edge_drop_weighted','edge_drop_high','edge_drop_low','edge_drop_anneal'], help='边丢弃策略')
    parser.add_argument('--use_dgi_as_main', action='store_true', help='在主训练中使用DGI损失作为主要训练信号')
    parser.add_argument('--lambda_dgi', type=float, default=1.0, help='主训练中DGI loss的权重 (default: 1.0)')
    parser.add_argument('--readout_mode', type=str, default='mean', 
                       choices=['mean', 'sum', 'gated'],
                       help='图readout模式')
    # 早停参数用于主训练（保留作为通用参数）
    parser.add_argument('--dgi_early_stop_patience', type=int, default=5, help='早停patience，0表示不使用早停 (default: 5)')
    parser.add_argument('--dgi_early_stop_min_delta', type=float, default=1e-2, help='早停最小改善阈值 (default: 1e-2)')
    

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小 (已支持真正的批处理)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='检查点间隔')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='每个epoch采样比例 (default: 1.0, 即全部数据; 0.3表示采样30%%)')
    parser.add_argument('--val_split', type=float, default=0.1, 
                       help='验证集比例 (default: 0.1，即10%%作为验证集)')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='早停patience，0表示不使用早停 (default: 0)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0001, help='早停最小改善阈值 (default: 0.0001)')
    
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
    
    # 使用MLP编码器，不再需要VAE
    logging.info("使用MLP编码器替代VAE，直接使用激活基因特征")

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
    
    # 为MLP创建输入特征：使用激活基因的cluster表达
    activated_cluster_expr = cluster_full_expr[available_activated_genes].copy()
    logging.info(f"MLP输入特征形状: {activated_cluster_expr.shape} (clusters x activated genes)")
    
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
    
    cell_expr = pd.DataFrame(
        np.zeros((len(cell_names), len(available_activated_genes)), dtype=np.float32),
        index=cell_names,
        columns=available_activated_genes
    )
    
    cell_full_expr = pd.DataFrame(
        np.zeros((len(cell_names), cluster_full_expr.shape[1]), dtype=np.float32),
        index=cell_names,
        columns=cluster_full_expr.columns
    )

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
    
    model = HeteroSTModel(
        n_genes=len(available_activated_genes),  # 使用激活基因数量
        mlp_latent_dim=args.mlp_latent_dim,
        mlp_hidden_dims=[int(x) for x in args.mlp_hidden_dims.split(',')],
        gat_layers=args.gat_layers,
        gat_hidden_dims=gat_hidden_dims,
        gat_heads=args.gat_heads,
        gat_dropout=args.gat_dropout,
        output_dim=args.output_dim,
        n_celltypes=n_cells,
        temperature=args.temperature
    ).to(device)
    
    logging.info("从头开始训练模型（MLP编码器直接训练，无预训练）")
    
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
    logging.info("在主训练中将使用 DGI loss，edge_attn_comm 已与 edge_attn_spatial 共享（单一 DGI encoder 实例）")
    model.edge_attn_comm = model.edge_attn_spatial
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    # ========== 阶段4：训练循环 ==========
    logging.info("="*80)
    logging.info("阶段4: 开始训练（DGI作为主要训练目标）")
    logging.info("="*80)
    logging.info(f"训练配置:")
    logging.info(f"  - 使用DGI loss作为主要训练信号 (lambda_dgi={args.lambda_dgi})")
    logging.info(f"  - 边腐败模式: {args.corruption_mode}, mask_ratio={args.mask_ratio}")
    logging.info(f"  - 边丢弃模式: {args.edge_drop_mode}, drop_rate={args.edge_drop_rate}")
    logging.info("="*80)

    # 使用外层tqdm跟踪epoch进度
    # 移除position参数避免Jupyter中重复显示，添加leave=True保持最终状态
    epoch_pbar = tqdm(range(args.epochs), desc="Training", leave=True, dynamic_ncols=True)
    
    # 主训练早停参数
    best_val_loss = float('inf')
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
        # ========== 训练阶段 ==========
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader, 1):
            batch_size = batch['batch_size']
            n_spots_sub_list = batch['n_spots_sub']  # ✅ 现在是列表
            n_cells_list = batch['n_cells']  # ✅ 现在是列表

            # === 打印图结构信息 (仅前1个batch) ===
            if batch_idx < 1 and epoch == 0:
                pass  # 移除调试信息

            # 累积批次损失
            batch_loss = 0.0

            # 处理每个subgraph
            for b in range(batch_size):
                # 提取第b个subgraph的数据
                expr_raw = batch['expr_raw'][b].to(device)  # [k+1, n_genes]
                cell_expr_raw = batch['cell_expr_raw'][b].to(device)  # [(k+1)*n_cells, n_marker_genes]

                edge_index_like = batch['edge_index_like'][b].to(device)  # [2, E_like]
                edge_attr_like = batch['edge_attr_like'][b].to(device)    # [E_like]
                edge_index_cc = batch['edge_index_cc'][b].to(device)      # [2, E_cc]
                edge_attr_cc = batch['edge_attr_cc'][b].to(device)        # [E_cc, 3] = [lr_score, lr_id, is_important]

                # ✅ 模型只需要前2列 [lr_score, lr_id]，保留完整的edge_attr_cc用于损失计算
                edge_attr_cc_input = torch.zeros(edge_attr_cc.size(0), 2, device=device)  # [E_cc, 2]

                # ========== 前向传播 ==========
                spot_repr, cell_repr, combined, spot_proj, cc_attention, predicted_masked_edges, edge_mask, exist_logits, rate_pred = model(
                    expr_raw=expr_raw,
                    cell_expr_raw=cell_expr_raw,
                    edge_index_like=edge_index_like,
                    edge_attr_like=edge_attr_like,
                    edge_index_cc=edge_index_cc,
                    edge_attr_cc=edge_attr_cc_input,  # 不输入真实强度，避免信息泄露
                    return_attention=True,
                    edge_mask_ratio=0.0  # 禁用边mask，使用双头监督
                )

                # ========== DGI训练：使用对比学习损失 ==========
                # DGI需要完整的图边和edge_attr (前2列)
                if edge_attr_like.dim() == 1:
                    edge_attr_like_ext = torch.cat([edge_attr_like.unsqueeze(-1), torch.full_like(edge_attr_like.unsqueeze(-1), -1)], dim=-1)
                else:
                    edge_attr_like_ext = edge_attr_like
                edge_attr_cc_dgi = edge_attr_cc[:, :2] if edge_attr_cc.dim() > 1 and edge_attr_cc.size(1) > 2 else edge_attr_cc
                edge_index_combined = torch.cat([edge_index_like, edge_index_cc], dim=1)
                edge_attr_combined = torch.cat([edge_attr_like_ext, edge_attr_cc_dgi], dim=0)
                # 调用模型内置的 compute_dgi_loss（encoder 已与主模型共享），计算DGI损失
                dgi_loss_val = model.compute_dgi_loss(
                    expr_raw,
                    cell_expr_raw,
                    edge_index_combined,
                    edge_attr_combined,
                    corruption_mode=args.corruption_mode,
                    mask_ratio=args.mask_ratio,
                    edge_drop_mode=args.edge_drop_mode,
                    edge_drop_rate=args.edge_drop_rate,
                    epoch=epoch,
                    total_epochs=args.epochs
                )
                total_loss = args.lambda_dgi * dgi_loss_val
                
                # 累积批次损失
                loss = total_loss
                batch_loss += loss

            # 对整个batch求平均损失，然后反向传播
            avg_batch_loss = batch_loss / batch_size

            # 反向传播
            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += avg_batch_loss.item()
        
        # ========== 验证阶段 ==========
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader, 1):
                batch_size = batch['batch_size']
                batch_loss = 0.0

                for b in range(batch_size):
                    expr_raw = batch['expr_raw'][b].to(device)
                    cell_expr_raw = batch['cell_expr_raw'][b].to(device)
                    edge_index_like = batch['edge_index_like'][b].to(device)
                    edge_attr_like = batch['edge_attr_like'][b].to(device)
                    edge_index_cc = batch['edge_index_cc'][b].to(device)
                    edge_attr_cc = batch['edge_attr_cc'][b].to(device)  # [E_cc, 3] = [lr_score, lr_id, is_important]

                    # ✅ 模型只需要前2列 [lr_score, lr_id]
                    edge_attr_cc_input = torch.zeros(edge_attr_cc.size(0), 2, device=device)  # [E_cc, 2]

                    # 前向传播（验证阶段不进行边mask）
                    spot_repr, cell_repr, combined, spot_proj, cc_attention, _, _, exist_logits, rate_pred = model(
                        expr_raw=expr_raw,
                        cell_expr_raw=cell_expr_raw,
                        edge_index_like=edge_index_like,
                        edge_attr_like=edge_attr_like,
                        edge_index_cc=edge_index_cc,
                        edge_attr_cc=edge_attr_cc_input,  # 不输入真实强度
                        return_attention=True,
                        edge_mask_ratio=0.0  # 验证阶段不mask
                    )

                    # 计算验证损失（DGI损失）
                    if edge_attr_like.dim() == 1:
                        edge_attr_like_ext = torch.cat([edge_attr_like.unsqueeze(-1), torch.full_like(edge_attr_like.unsqueeze(-1), -1)], dim=-1)
                    else:
                        edge_attr_like_ext = edge_attr_like
                    edge_attr_cc_dgi = edge_attr_cc[:, :2] if edge_attr_cc.dim() > 1 and edge_attr_cc.size(1) > 2 else edge_attr_cc
                    edge_index_combined = torch.cat([edge_index_like, edge_index_cc], dim=1)
                    edge_attr_combined = torch.cat([edge_attr_like_ext, edge_attr_cc_dgi], dim=0)
                    val_loss = model.compute_dgi_loss(
                        expr_raw,
                        cell_expr_raw,
                        edge_index_combined,
                        edge_attr_combined,
                        corruption_mode=args.corruption_mode,
                        mask_ratio=args.mask_ratio,
                        edge_drop_mode=args.edge_drop_mode,
                        edge_drop_rate=args.edge_drop_rate,
                        epoch=epoch,
                        total_epochs=args.epochs
                    ) * args.lambda_dgi
                    
                    batch_loss += val_loss
                
                avg_batch_loss = batch_loss / batch_size
                total_val_loss += avg_batch_loss.item()
        
        # 计算epoch平均损失
        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # ========== 早停检查 ==========
        if early_stop_patience > 0:
            if avg_val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                #logging.info(f"验证损失改善: {best_val_loss:.6f} (patience重置为0)")
            else:
                patience_counter += 1
                #logging.info(f"验证损失未改善: {avg_val_loss:.6f} vs {best_val_loss:.6f} (patience: {patience_counter}/{early_stop_patience})")
                
            if patience_counter >= early_stop_patience:
                logging.info(f"早停触发: 验证损失在{early_stop_patience}个epoch内未改善")
                break
        
        # 更新epoch进度条（只在epoch结束时更新一次）
        epoch_pbar.set_postfix({
            'Train': f'{avg_train_loss:.4f}',
            'Val': f'{avg_val_loss:.4f}',
            'DGI': f'{avg_train_loss:.4f}',  # DGI损失
            'λ_DGI': f'{args.lambda_dgi:.1f}'  # DGI权重
        })
    logging.info(f"   - 评估数据集大小: {len(dataset)} spots")
    logging.info(f"   - 评估batch数量: {len(eval_dataloader)} batches")
    
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
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", leave=True)):
            batch_size = batch['batch_size']
            
            for b in range(batch_size):
                expr_raw = batch['expr_raw'][b].to(device)
                cell_expr_raw = batch['cell_expr_raw'][b].to(device)
                edge_index_like = batch['edge_index_like'][b].to(device)
                edge_attr_like = batch['edge_attr_like'][b].to(device)
                edge_index_cc = batch['edge_index_cc'][b].to(device)
                edge_attr_cc = batch['edge_attr_cc'][b].to(device)
                
                # 模型前向传播
                edge_attr_cc_input = torch.zeros(edge_attr_cc.size(0), 2, device=device)
                spot_repr, cell_repr, combined, spot_proj, cc_attention, _, _, exist_logits, rate_pred = model(
                    expr_raw=expr_raw,
                    cell_expr_raw=cell_expr_raw,
                    edge_index_like=edge_index_like,
                    edge_attr_like=edge_attr_like,
                    edge_index_cc=edge_index_cc,
                    edge_attr_cc=edge_attr_cc_input,
                    return_attention=True,
                    edge_mask_ratio=0.0
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
    
    logging.info(f"评估完成，收集到 {len(all_cc_attention_scores)} 个spots的注意力得分")
    total_edges_collected = sum(scores.shape[0] for scores in all_cc_attention_scores)
    logging.info(f"   - 子图内边数（模型实际使用）: {total_edges_collected} 条")
    logging.info(f"   - 平均每个spot子图: {total_edges_collected/len(all_cc_attention_scores):.1f} 条边")
    logging.info(f"   - 注意：这是子图内的边数，不包含spot间的跨子图通讯")
    
    evaluate_cell_communication(
        all_cc_attention_scores=all_cc_attention_scores,
        all_edge_index_cc=all_edge_index_cc,
        all_edge_attr_cc=all_edge_attr_cc,
        all_spot_indices=all_spot_indices,
        all_n_spots_sub=all_n_spots_sub,
        all_cell_names=all_cell_names,
        output_dir=args.output_dir,
        n_spots=n_spots,
        n_cells=n_cells,
        spot_names=spot_names,
        all_src_barcodes=all_src_barcodes,
        all_dst_barcodes=all_dst_barcodes
    )
    
    # 绘制损失曲线
    plot_training_loss(train_losses, val_losses, args.output_dir, args.epochs)
    
    logging.info("\n" + "="*80)
    logging.info("训练完成！")
    logging.info("="*80)

if __name__ == '__main__':
    main()